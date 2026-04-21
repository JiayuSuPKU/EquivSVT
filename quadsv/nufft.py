"""
Non-uniform FFT (NUFFT) spectra, kernel and spatial tests for irregular data

When data sit on a regular grid (e.g., a rasterized Visium slide),
:func:`quadsv.power_spectrum_2d` computes :math:`|\\hat{x}(k)|^2` with a plain
2D FFT. For data whose spatial coordinates are **irregular** — e.g.,
imaging-based in situ platforms, Slide-seq, or a Visium slide read straight
from ``adata.obsm['spatial']`` without rasterization — :func:`power_spectrum_2d_nufft`
evaluates the type-1 NUFFT

.. math::

   \\hat c(k_y, k_x) = \\sum_{j=1}^{n} c_j \\,
       \\exp\\!\\bigl[-i(k_y\\,y_j + k_x\\,x_j)\\bigr]

on the same uniform ``(ny, nx)`` k-space grid that :func:`power_spectrum_2d`
would produce for a rasterized input of the same physical extent, and returns
:math:`|\\hat c|^2` in the scipy FFT layout (DC at ``[0, 0]``). Anything
downstream — :func:`quadsv.multisample.radial_bin_spectrum`,
:class:`quadsv.ComparatorIrregular` — works identically.

Notation (shared across this module)
------------------------------------

Dimensions:

- ``n``: number of spots (on the irregular grid).
- ``(ny, nx)``: internal uniform k-grid dimensions; ``n' = ny · nx``.
- ``(dy, dx)``: **physical** spacing per k-grid cell, same unit as the spatial coordinates
  after multiplying ``unit_scale``.
- ``unit_scale``: multiplier that converts the input coordinates ``S`` to the same unit as
  ``(dy, dx)`` (e.g., 0.35 if ``S`` are in pixels at 0.35 μm/pixel). Samples from different
  slides and platforms may ship coordinates in different units; this parameter harmonizes them
  onto the same **physical** unit for the internal k-grid and all downstream spectra and tests.

Vectors and matrices:

- ``S``: the ``n × 2`` spatial coordinate matrix of the irregular points, ordered as ``(y, x)``.
- ``K``: the ``n × n`` translation-invariant kernel at the irregular points.
- ``K'``: the ``n' × n'`` grid kernel with FFT eigenvalues
  ``λ(k) = F(K')(k)``.
- ``U``: the ``n × n'`` type-2 NUFFT evaluation matrix; the band-limited
  approximation is ``K ≈ (1/n') · U · diag(λ) · Uᴴ``.
- ``x̂ = Uᴴ x``: type-1 NUFFT of a length-``N`` signal onto the k-grid ``(ny, nx)`` (vectorized).

"""

from __future__ import annotations

import logging
import warnings

import finufft
import numpy as np
import scipy.fft
import scipy.sparse as sp
from scipy.stats import chi2, norm

from quadsv.fft import FFTKernel
from quadsv.kernels import Kernel
from quadsv.statistics import liu_sf

__all__ = [
    "power_spectrum_2d_nufft",
    "NUFFTKernel",
]

logger = logging.getLogger(__name__)


def _infer_grid_from_coords(
    coords: np.ndarray,
    unit_scale: float = 1.0,
    oversample: float = 2.0,
    padding: float = 1.05,
    min_side: int = 32,
    max_side: int = 1024,
) -> tuple[tuple[int, int], tuple[float, float]]:
    """Pick ``(grid_shape, spacing)`` from coords alone, with no kernel input.

    The k-grid only needs to resolve the signal's sampling Nyquist, which is
    set by the median nearest-neighbor spacing of the coordinates. Finer than
    that is wasted work (aliasing kicks in anyway). Coarser misses kernel
    spectral content. ``oversample=2.0`` is a safe default.

    Returns ``(grid_shape, spacing)`` rounded to FFT-friendly sizes (multiples
    of 8).
    """
    from scipy.spatial import cKDTree

    scaled = np.asarray(coords, dtype=np.float64) * float(unit_scale)
    if scaled.ndim != 2 or scaled.shape[1] != 2:
        raise ValueError(f"coords must be (n, 2), got {scaled.shape}.")
    L_y = float(scaled[:, 0].max() - scaled[:, 0].min()) * padding
    L_x = float(scaled[:, 1].max() - scaled[:, 1].min()) * padding
    if L_y <= 0 or L_x <= 0:
        raise ValueError("coords have zero extent along one or both axes.")
    # Median 1-NN distance — robust proxy for the sampling scale.
    nn = cKDTree(scaled).query(scaled, k=2)[0][:, 1]
    d_nn = float(np.median(nn[nn > 0]))
    spacing_target = d_nn / oversample

    def _round_up(n: float) -> int:
        return int(min(max_side, max(min_side, 8 * int(np.ceil(n / 8)))))

    ny = _round_up(L_y / spacing_target)
    nx = _round_up(L_x / spacing_target)
    return (ny, nx), (L_y / ny, L_x / nx)


def power_spectrum_2d_nufft(
    coords: np.ndarray,
    values: np.ndarray,
    grid_shape: tuple[int, int],
    spacing: tuple[float, float],
    unit_scale: float = 1.0,
    eps: float = 1e-6,
    center_coords: bool = True,
) -> np.ndarray:
    """
    Compute the 2D power spectrum via type-1 NUFFT

    This function computes the power spectrum :math:`P(k) = |\\hat{c}(k)|^2` of
    one or more non-uniform spatial signals via a type-1 NUFFT.
    The output has the same ``(ny, nx)`` layout as
    :func:`quadsv.fft.power_spectrum_2d` with ``fft_solver='fft2'``: DC at
    ``[0, 0]``, Nyquist at ``[ny/2, nx/2]`` (when dimensions are even).

    Parameters
    ----------
    coords : np.ndarray
        Non-uniform spatial coordinates, shape ``(n, 2)`` in the order
        ``(y, x)``. Values outside the physical domain implied by
        ``grid_shape`` and ``spacing`` are folded into ``[-π, π)`` by finufft.
    values : np.ndarray
        Signal strengths at each coordinate. Shape ``(n,)`` for a single
        feature, or ``(n, M)`` for ``M`` stacked features (e.g., genes) on the
        same coordinates. Real-valued; promoted to complex internally.
    grid_shape : tuple[int, int]
        ``(ny, nx)`` of the target uniform k-space grid. Match whatever grid
        you use for rasterized samples so the two paths produce comparable
        spectra.
    spacing : tuple[float, float]
        ``(dy, dx)`` physical spacing per cell of the target grid, in the same
        unit as ``unit_scale * coords``. Together with ``grid_shape`` this
        defines the physical domain extent ``(ny · dy, nx · dx)``.
    unit_scale : float, default 1.0
        Multiplier applied to ``coords`` before scaling into ``[-π, π)``. Use
        this to convert per-sample coordinate units into the common unit of
        ``spacing`` (e.g., 0.35 if ``coords`` are in Visium full-res pixels at
        0.35 μm/pixel and ``spacing`` is in μm).
    eps : float, default 1e-6
        NUFFT tolerance forwarded to finufft.
    center_coords : bool, default True
        If True, subtract the mean of ``coords`` before scaling — avoids
        wrapping artefacts when coordinates are stored with an arbitrary origin
        offset (e.g., Visium pixel coordinates start at a few thousand). Power
        spectra are translation-invariant so recentering does not change the
        result.

    Returns
    -------
    np.ndarray
        Power spectrum. Shape ``(ny, nx)`` for 1D ``values`` or
        ``(ny, nx, M)`` for 2D ``values``, with DC at index ``[0, 0]``.

    Raises
    ------
    ImportError
        If :mod:`finufft` is not installed.
    ValueError
        If input shapes are inconsistent.

    Examples
    --------
    >>> import numpy as np
    >>> coords = np.random.default_rng(0).uniform(0, 100, size=(500, 2))
    >>> vals = np.random.default_rng(1).standard_normal(500)
    >>> P = power_spectrum_2d_nufft(coords, vals, grid_shape=(32, 32), spacing=(4.0, 4.0))
    >>> P.shape
    (32, 32)
    """
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must have shape (n, 2), got {coords.shape}.")
    if values.shape[0] != coords.shape[0]:
        raise ValueError(
            f"values first dim {values.shape[0]} must match coords n={coords.shape[0]}."
        )

    ny, nx = grid_shape
    dy, dx = spacing
    if ny <= 0 or nx <= 0 or dy <= 0 or dx <= 0:
        raise ValueError(f"grid_shape and spacing must be positive, got {grid_shape}, {spacing}.")

    y = coords[:, 0].astype(np.float64) * unit_scale
    x = coords[:, 1].astype(np.float64) * unit_scale
    if center_coords:
        y = y - y.mean()
        x = x - x.mean()

    # Physical domain extents implied by the target uniform grid.
    Ly = ny * dy
    Lx = nx * dx

    # Scale into finufft's [-π, π) window so that mode index k (centred
    # at zero, range [-n/2, (n-1)/2]) corresponds to physical frequency
    # k / L cycles per unit length — matching np.fft.fftfreq(n, d).
    y_scaled = y * (2.0 * np.pi / Ly)
    x_scaled = x * (2.0 * np.pi / Lx)

    # Batched transforms: finufft accepts shape (n_tr, M) for c.
    squeeze = values.ndim == 1
    if squeeze:
        c = values.astype(np.complex128, copy=False)
    else:
        # finufft expects (n_tr, N_points).
        c = np.ascontiguousarray(values.T.astype(np.complex128, copy=False))

    # type-1 NUFFT: nonuniform points -> uniform k-space grid.
    # Output shape: (ny, nx) or (n_tr, ny, nx). DC at CENTRE ([ny//2, nx//2]).
    f_hat = finufft.nufft2d1(y_scaled, x_scaled, c, n_modes=(ny, nx), eps=eps, isign=-1)

    # Power spectrum.
    power = (f_hat.real**2 + f_hat.imag**2).astype(np.float64)

    # Move DC from the centre to [0, 0] so the layout matches scipy.fft.fft2.
    power = np.fft.ifftshift(power, axes=(-2, -1))

    if squeeze:
        return power
    # Put the feature axis back at the end to match power_spectrum_2d(x=(ny, nx, M)).
    return np.moveaxis(power, 0, -1)


# ---------------------------------------------------------------------------
# NUFFTKernel: translation-invariant kernel on irregular spatial points
# ---------------------------------------------------------------------------


class NUFFTKernel(Kernel):
    """
    Spatial kernel over **irregular** 2D coordinates evaluated via NUFFTs.

    Parallels :class:`quadsv.fft.FFTKernel` (which requires a regular grid) and
    implements the :class:`~quadsv.Kernel` interface so it plugs into
    :func:`quadsv.statistics.spatial_q_test` /
    :func:`quadsv.statistics.spatial_r_test` the same way.

    The band-limited approximation of the ``n × n`` irregular-point operator is
    ``K ≈ (1/n') · U · diag(λ) · Uᴴ``, where ``U`` is the ``n × n'`` type-2
    NUFFT matrix and ``λ = F(K')`` is the grid kernel's spectrum. Under this
    approximation, Parseval's identity gives the fast quadratic form
    ``xᵀ K x = (1/n') Σ_k λ(k) |x̂(k)|²`` with ``x̂ = Uᴴ x`` (a single type-1 NUFFT).
    The matrix-vector primitive :meth:`Kx` uses the companion two-shot NUFFT
    ``K z = (1/n') · U · (λ ⊙ Uᴴ z)`` and backs the KPM spectral-density
    estimator and the bipartite R-test cross matrix.

    :func:`quadsv.spatial_q_test` always uses the k-space Parseval path
    (:meth:`xtKx`); :func:`quadsv.spatial_r_test` dispatches on shape —
    paired diagonal for ``M_x == M_y`` via :meth:`xtKy`, full ``(M_x, M_y)``
    cross matrix via :meth:`Kx` otherwise. The matmul counterparts
    (:meth:`xtKx_matmul` / :meth:`xtKy_matmul`) are exposed for callers that
    want to compute the same form directly at the ``n`` irregular points —
    they agree with the spectral path to NUFFT precision (``eps``) on both
    regular and irregular coordinates.

    Parameters
    ----------
    coords : np.ndarray
        Spot coordinates of shape ``(n, 2)`` in order ``(y, x)``.
    grid_shape : tuple[int, int], optional
        ``(ny, nx)`` of the internal uniform k-grid. If ``None`` (default),
        auto-inferred from ``coords``: the grid is sized to cover the bounding
        box and to resolve the sampling Nyquist set by the median
        nearest-neighbor distance (fully coordinate-driven, kernel-agnostic).
        Override only when you know you need a finer or coarser grid.
    spacing : tuple[float, float], optional
        ``(dy, dx)`` physical spacing per k-grid cell (same unit as ``coords``
        after ``unit_scale``). If ``None`` (default), auto-inferred alongside
        ``grid_shape``. When both are supplied, users are responsible for
        ensuring ``ny · dy``, ``nx · dx`` covers the coordinate extent.
    method : str, default ``'matern'``
        Kernel method forwarded to :class:`FFTKernel`. One of ``'gaussian'``,
        ``'matern'``, ``'moran'``, ``'graph_laplacian'``, ``'car'``.
    unit_scale : float, default 1.0
        Multiplier applied to ``coords`` so they share the same unit as
        ``spacing`` (e.g. ``0.35`` if coords are in pixels at 0.35 μm/pixel).
    oversample : float, default 2.0
        Auto-grid oversampling factor above the sampling Nyquist. Used only
        when ``grid_shape`` / ``spacing`` are auto-derived. Larger values give
        a finer k-grid (more accurate, slower); 2.0 is safe for all tested
        kernels.
    eps : float, default 1e-6
        NUFFT tolerance forwarded to finufft.
    workers : int, optional
        Forwarded to :mod:`scipy.fft` (used by :meth:`Kx_grid`) and reserved
        for future finufft parallelism. ``None`` uses the SciPy default.
    **kwargs
        Method-specific kernel hyperparameters (``bandwidth``, ``nu``,
        ``rho``, ``neighbor_degree``) forwarded to the internal
        :class:`FFTKernel`.

    Attributes
    ----------
    coords : np.ndarray
        Original ``(n, 2)`` coordinates.
    n : int
        Number of spots ``n``.
    grid_shape : tuple[int, int]
        Internal k-grid shape ``(ny, nx)``.
    spacing : tuple[float, float]
        Physical spacing per k-grid cell ``(dy, dx)``.
    method : str
        Kernel method name.
    params : dict
        Resolved kernel hyperparameters (snapshot of the internal FFT kernel).
    workers : int or None
        scipy.fft worker count used by :meth:`Kx_grid`.
    stores_precision : bool
        Always ``False`` — NUFFTKernel never holds an ``n × n`` matrix.
    """

    _available_kernels = ["gaussian", "matern", "moran", "graph_laplacian", "car"]

    def __init__(
        self,
        coords: np.ndarray,
        grid_shape: tuple[int, int] | None = None,
        spacing: tuple[float, float] | None = None,
        method: str = "matern",
        unit_scale: float = 1.0,
        oversample: float = 2.0,
        eps: float = 1e-6,
        workers: int | None = None,
        *,
        centering: bool = True,
        **kwargs,
    ) -> None:
        """Construct a translation-invariant kernel over irregular 2D coordinates.

        See the class docstring for a full parameter / attribute reference;
        in brief:

        Parameters
        ----------
        coords : np.ndarray
            ``(n, 2)`` spot coordinates in ``(y, x)`` order.
        grid_shape, spacing : tuple or None
            Internal k-grid shape and per-cell spacing. Both optional;
            auto-inferred from ``coords`` when either is missing.
        method : str, default ``'matern'``
            Kernel method forwarded to the internal :class:`FFTKernel`.
        unit_scale : float, default 1.0
            Coord → physical-unit multiplier so ``coords * unit_scale`` is in
            the same unit as ``spacing``.
        oversample : float, default 2.0
            Auto-grid oversampling above the sampling Nyquist.
        eps : float, default 1e-6
            NUFFT tolerance.
        workers : int, optional
            scipy.fft worker count used by :meth:`Kx_grid`.
        **kwargs
            Method-specific kernel hyperparameters (``bandwidth``, ``nu``,
            ``rho``, ``neighbor_degree``).

        Raises
        ------
        ValueError
            If ``coords`` has the wrong shape, ``method`` is unknown, or
            ``grid_shape`` / ``spacing`` are invalid.
        """
        coords = np.asarray(coords, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"coords must be shape (n, 2), got {coords.shape}.")
        if method not in self._available_kernels:
            raise ValueError(f"method must be one of {self._available_kernels}, got '{method}'.")

        # Auto-derive grid_shape / spacing from coords alone when either is missing.
        # Coordinate-driven: picks a k-grid that resolves the sampling Nyquist.
        if grid_shape is None or spacing is None:
            auto_gs, auto_sp = _infer_grid_from_coords(
                coords, unit_scale=unit_scale, oversample=oversample
            )
            if grid_shape is None:
                grid_shape = auto_gs
            if spacing is None:
                spacing = auto_sp
            logger.info(
                "NUFFTKernel auto-inferred grid_shape=%s spacing=%s from %d coords.",
                grid_shape,
                spacing,
                coords.shape[0],
            )

        ny, nx = int(grid_shape[0]), int(grid_shape[1])
        if ny < 4 or nx < 4:
            raise ValueError(f"grid_shape must be at least (4, 4), got ({ny}, {nx}).")
        dy, dx = float(spacing[0]), float(spacing[1])
        if dy <= 0 or dx <= 0:
            raise ValueError(f"spacing must be positive, got ({dy}, {dx}).")

        super().__init__(centering=centering)
        self.coords: np.ndarray = coords
        self.n: int = coords.shape[0]
        self.grid_shape: tuple[int, int] = (ny, nx)
        self.spacing: tuple[float, float] = (dy, dx)
        self.method: str = method
        self._unit_scale: float = float(unit_scale)
        self._eps: float = float(eps)
        self.workers: int | None = workers
        self.stores_precision: bool = False

        # Internal FFTKernel holds the eigenvalue spectrum on the k-grid. We
        # use fft2 (full spectrum) so the ifftshift trick aligns NUFFT output
        # with the scipy FFT layout (DC at [0, 0]).
        # Internal FFTKernel is always kept on the *raw* spectrum — we own
        # centering at the NUFFT level (the constant mode of K on scattered
        # coords isn't the FFT grid's DC mode in general).
        self._fft_kernel = FFTKernel(
            shape=(ny, nx),
            spacing=(dy, dx),
            topology="square",
            method=method,
            fft_solver="fft2",
            workers=workers,
            centering=False,
            **kwargs,
        )
        self.params: dict = dict(self._fft_kernel.params)

        # Pre-scale coords into finufft's [-π, π) window. Centered so we avoid
        # origin-offset phase artefacts.
        y = coords[:, 0] * self._unit_scale
        x = coords[:, 1] * self._unit_scale
        self._y_mean = float(y.mean())
        self._x_mean = float(x.mean())
        self._y_scaled = (y - self._y_mean) * (2.0 * np.pi / (ny * dy))
        self._x_scaled = (x - self._x_mean) * (2.0 * np.pi / (nx * dx))

    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<NUFFTKernel method={self.method} n={self.n} "
            f"grid={self.grid_shape} spacing={self.spacing} "
            f"params={self.params}>"
        )

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"NUFFTKernel\n"
            f"- Method: {self.method}\n"
            f"- Number of spots: {self.n}\n"
            f"- k-grid: {self.grid_shape} at spacing {self.spacing}\n"
            f"- Params: {self.params}"
        )

    # ------------------------------------------------------------------
    _TOEPLITZ_R_THRESHOLD: int = 2000
    _TOEPLITZ_LAM_TOL: float = 1e-5
    _KPM_DEGREE: int = 60
    _KPM_N_PROBES: int = 30

    def _coord_phi(self) -> np.ndarray:
        """Coord-density function ``φ(k)`` on k-grid (DC at [0, 0]).

        ``φ(Δ) = (1/n) · Σ_i exp(-iΔ·y_i)`` — one type-1 NUFFT of the
        all-ones vector. Symmetrized so ``φ[k] = conj(φ[-k mod N])``
        holds exactly (fixes Nyquist self-conjugate bins that NUFFT
        samples as complex on even-length grids; needed for strict
        Hermiticity of the induced ``G_{k,k'} = n·φ(k'-k)``).
        Cached on the instance (coord-only).
        """
        cache = getattr(self, "_phi_cache", None)
        if cache is not None:
            return cache
        ones = np.ones(self.n, dtype=float)
        phi_hat_centered = self._nufft_type1(ones).squeeze()  # finufft DC-centered
        phi = np.fft.ifftshift(phi_hat_centered) / self.n  # DC at [0, 0]
        ny, nx = phi.shape
        ii = np.arange(ny)[:, None]
        jj = np.arange(nx)[None, :]
        phi = 0.5 * (phi + np.conj(phi[(-ii) % ny, (-jj) % nx]))
        self._phi_cache = phi
        return phi

    def _eigvals_toeplitz_M(self, lam_tol: float | None = None) -> np.ndarray | None:
        """Full-spectrum eigenvalues via the reduced Toeplitz-``G`` matrix.

        Uses the identity (for PSD ``Λ``) that non-zero eigvals of
        ``K_n = (1/n') U Λ Uᴴ`` equal non-zero eigvals of
        ``M = (1/n') Λ^{1/2} G Λ^{1/2}`` with ``G_{k,k'} = n·φ(k'-k)``
        (Toeplitz in k-space, built once from ``_coord_phi``). Truncate
        ``Λ`` to its support — pick the ``r`` modes with
        ``|λ(k)| > lam_tol · max|λ|`` — form the dense ``r × r`` ``M_r``
        by indexed ``φ`` lookups, ``eigvalsh``.

        For ``centering=True`` we apply the rank-1 shift
        ``G → G_H = G − (1/n) · (Uᴴ𝟏)(Uᴴ𝟏)ᴴ = G − n · conj(φ) φᵀ``,
        which accounts for the y-space ``H = I − 𝟏𝟏ᵀ/n`` projection
        ((HU)ᴴ(HU) = UᴴHU = G − (1/n)(Uᴴ𝟏)(Uᴴ𝟏)ᴴ).

        Returns ``None`` when the formula doesn't apply cleanly:

        - Indefinite ``Λ`` (``moran``, anti-phase eigenmodes) — the
          signed square root makes ``M`` non-Hermitian.
        - Broad spectrum with ``r > _TOEPLITZ_R_THRESHOLD`` — dense
          ``r × r`` eigvalsh would be slower than the KPM fallback.

        Returns
        -------
        np.ndarray or None
            Eigenvalues (descending), or ``None`` if the KPM fallback
            should be used instead.
        """
        lam = self._fft_kernel.spectrum.reshape(self.grid_shape)
        if lam.min() < -1e-3 * float(np.abs(lam).max()):
            return None  # indefinite

        lam_flat = lam.ravel()
        lam_max = float(np.abs(lam_flat).max())
        if lam_max == 0:
            return np.zeros(self.n)

        tol_rel = self._TOEPLITZ_LAM_TOL if lam_tol is None else float(lam_tol)
        mask_flat = np.abs(lam_flat) > tol_rel * lam_max
        r = int(mask_flat.sum())
        if r > self._TOEPLITZ_R_THRESHOLD:
            return None  # too broad — KPM is cheaper

        ny, nx = self.grid_shape
        nprime = ny * nx
        mask = mask_flat.reshape(ny, nx)
        iy, ix = np.where(mask)
        lam_r_sqrt = np.sqrt(np.maximum(lam_flat[mask_flat], 0.0))

        phi = self._coord_phi()
        dy = (iy[None, :] - iy[:, None]) % ny
        dx = (ix[None, :] - ix[:, None]) % nx
        G_r = self.n * phi[dy, dx]
        if self.centering:
            # (HU)ᴴ(HU) = G − (1/n)(Uᴴ𝟏)(Uᴴ𝟏)ᴴ = G − n · conj(φ_r) φ_rᵀ.
            # Rank-1 outer product on the restricted mode set.
            phi_r = phi[iy, ix]
            G_r = G_r - self.n * np.outer(np.conj(phi_r), phi_r)
        M_r = (lam_r_sqrt[:, None] * G_r * lam_r_sqrt[None, :]) / nprime
        M_r = 0.5 * (M_r + M_r.conj().T)

        vals = np.linalg.eigvalsh(M_r)
        vals = np.real(vals)
        # For PSD λ the operator ``K_n = (1/n') U Λ Uᴴ`` and ``HKH`` are
        # both PSD — any negative eigenvalue from ``eigvalsh`` is pure
        # finite-precision noise on the near-null subspace. Clip to 0.
        vals = np.maximum(vals, 0.0)
        return np.sort(vals)[::-1]

    def _eigvals_kpm(  # noqa: C901
        self,
        n_samples: int | None = None,
        degree: int | None = None,
        n_probes: int | None = None,
        rng_seed: int = 0,
    ) -> np.ndarray:
        """Approximate full spectrum via the Kernel Polynomial Method (KPM).

        Uses only :meth:`Kx` matvecs — no dense ``n × n`` work. Flow:

        1. Estimate spectral bounds ``[a, b]`` by ``eigsh(which='LA')`` /
           ``eigsh(which='SA')`` on the :meth:`Kx` LinearOperator (one
           call each, a few matvecs apiece).
        2. Rescale the operator to ``T = (K − (a+b)/2 · I) · 2/(b−a)`` so
           its spectrum lives in ``(−1, 1)``.
        3. Chebyshev moments ``μ_p = (1/n) · trace(T_p(T))`` for
           ``p = 0..D`` via stochastic trace estimation with Rademacher
           probes and the three-term recurrence
           ``T_{p+1}(T)v = 2T·T_p(T)v − T_{p-1}(T)v`` — ``D`` degree,
           ``S`` probes, total ``O(D · S · matvec)``.
        4. Apply Jackson kernel damping to suppress the Gibbs ringing that
           polynomial truncation introduces.
        5. Reconstruct the density
           ``ρ(x) = (1/(π √(1−x²))) · (μ_0 + 2 Σ_{p≥1} g_p μ_p T_p(x))``
           on a fine ``x``-grid, clip negatives, renormalize.
        6. Quantile-sample ``n_samples`` values from ``ρ`` and rescale
           back to the original spectrum range.

        Sets ``self._last_spectrum_from_kpm = True`` so downstream
        consumers (Liu's method) can emit a tail-accuracy warning.

        Parameters
        ----------
        n_samples : int, optional
            Number of eigenvalue samples to return (default ``self.n``).
        degree : int, optional
            Polynomial degree ``D`` (default ``_KPM_DEGREE`` = 60).
        n_probes : int, optional
            Hutchinson probe count (default ``_KPM_N_PROBES`` = 30).
        rng_seed : int, default 0
            Seed for reproducibility.

        Returns
        -------
        np.ndarray
            ``(n_samples,)`` descending, with ``mean × n_samples ≈ trace``.
        """
        from scipy.sparse.linalg import LinearOperator, eigsh

        n = self.n
        if n_samples is None:
            n_samples = n
        if degree is None:
            degree = self._KPM_DEGREE
        if n_probes is None:
            n_probes = self._KPM_N_PROBES

        def _matvec(v: np.ndarray) -> np.ndarray:
            return np.asarray(self.Kx(np.asarray(v, dtype=float)))

        op = LinearOperator(shape=(n, n), matvec=_matvec, dtype=float)
        # Spectral bound via Lanczos ``which='LM'`` (robust even when
        # eigenvalues pile up at zero, which happens for ``HKH`` and
        # rank-deficient PSD kernels). Fallback to the Frobenius bound
        # ``|λ|_max ≤ √trace(K²)`` if Lanczos fails.
        try:
            lm = float(eigsh(op, k=1, which="LM", return_eigenvectors=False, maxiter=500)[0])
            lam_bound = abs(lm)
        except Exception:
            lam_bound = float(np.sqrt(max(self.square_trace(method="analytic"), 0.0)))
        if lam_bound <= 1e-12:
            self._last_spectrum_from_kpm = True
            return np.zeros(n_samples)
        # PSD detection — if the kernel method is known-PSD by construction
        # (gaussian/matern/car) OR the FFT spectrum is numerically PSD
        # (graph_laplacian with rho stabilizer), both ``K_n`` and ``HKH``
        # are PSD, so eigenvalues live in ``[0, lam_bound]``. Using
        # ``a = 0`` doubles the KPM k-space resolution compared to a
        # symmetric ``[-b, +b]`` bound and prevents spurious negative
        # samples from Jackson leakage. For indefinite kernels (Moran),
        # eigenvalues span both signs — use the symmetric bound.
        psd_methods = ("gaussian", "matern", "car", "graph_laplacian")
        grid_lam = self._fft_kernel.spectrum
        lam_range = float(np.abs(grid_lam).max())
        is_psd = self.method in psd_methods and (
            float(grid_lam.min()) >= -1e-3 * max(lam_range, 1.0)
        )
        pad = 0.02 * lam_bound
        if is_psd:
            a, b = -pad, lam_bound + pad
        else:
            a, b = -lam_bound - pad, lam_bound + pad
        shift = 0.5 * (a + b)
        scale = 2.0 / (b - a)

        def apply_T(v: np.ndarray) -> np.ndarray:
            """Scaled operator T = (K − shift·I) · scale."""
            return scale * (_matvec(v) - shift * v)

        # Chebyshev moments via Hutchinson.
        rng = np.random.default_rng(rng_seed)
        mu = np.zeros(degree + 1)
        for _ in range(n_probes):
            r = rng.choice([-1.0, 1.0], size=n).astype(float)
            t_prev = r  # T_0(T)·r = r
            t_curr = apply_T(r)  # T_1(T)·r
            mu[0] += np.dot(r, t_prev)
            mu[1] += np.dot(r, t_curr)
            for p in range(2, degree + 1):
                t_next = 2.0 * apply_T(t_curr) - t_prev
                mu[p] += np.dot(r, t_next)
                t_prev, t_curr = t_curr, t_next
        mu /= n * n_probes

        # Jackson kernel damping.
        x_J = np.pi / (degree + 1)
        p_arr = np.arange(degree + 1)
        g = ((degree - p_arr + 1) * np.cos(p_arr * x_J) + np.sin(p_arr * x_J) / np.tan(x_J)) / (
            degree + 1
        )
        mu_damped = mu * g

        # Reconstruct density on fine grid (avoid the ±1 singularities).
        n_grid = 4 * max(n_samples, 512)
        xs = np.linspace(-1 + 1e-4, 1 - 1e-4, n_grid)
        T_vals = np.empty((degree + 1, n_grid))
        T_vals[0] = 1.0
        if degree >= 1:
            T_vals[1] = xs
        for p in range(2, degree + 1):
            T_vals[p] = 2 * xs * T_vals[p - 1] - T_vals[p - 2]
        density = mu_damped[0] * T_vals[0]
        if degree >= 1:
            density = density + 2.0 * np.sum(mu_damped[1:, None] * T_vals[1:], axis=0)
        density = density / (np.pi * np.sqrt(1.0 - xs**2))
        density = np.maximum(density, 0.0)
        # np.trapezoid (numpy ≥ 2); falls back for older numpy.
        _trapz = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]
        mass = _trapz(density, xs)
        if mass <= 0:
            return np.zeros(n_samples)
        density /= mass

        # Quantile sample. CDF via cumulative trapezoid.
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (density[:-1] + density[1:]) * np.diff(xs))])
        cdf /= cdf[-1]
        q = (np.arange(n_samples) + 0.5) / n_samples
        x_samples = np.interp(q, cdf, xs)
        eigvals = x_samples / scale + shift
        if is_psd:
            # Clip Jackson-leakage residual negatives to 0. Magnitude is
            # O(Jackson-window width × max|λ|) — a few percent of lam_bound.
            eigvals = np.maximum(eigvals, 0.0)

        self._last_spectrum_from_kpm = True
        return np.sort(eigvals)[::-1]

    def eigenvalues(self, k: int | None = None, return_full_layout: bool = False) -> np.ndarray:
        """Eigenvalues of the ``n × n`` irregular-point operator.

        Returns the spectrum of the realized operator ``K_n`` (raw) or
        ``H K_n H`` (centered) at the irregular coordinates — **not** the
        internal FFT-grid spectrum. Purely matrix-free: no dense
        ``n × n`` construction.

        Three paths:

        - **``k`` given — Lanczos top-k.** Wrap :meth:`Kx` as a
          :class:`~scipy.sparse.linalg.LinearOperator` and call
          :func:`~scipy.sparse.linalg.eigsh(which='LM')` for the top
          ``k`` eigenvalues by magnitude. Cost: ``O(k × nufft)``. Not
          cached (could be called with different ``k``).
        - **``k=None`` — Toeplitz-M (analytic, preferred).** See
          :meth:`_eigvals_toeplitz_M`. Applies when ``Λ`` is PSD,
          ``centering=False``, and the support size
          ``r = #{k : |λ(k)| > 10⁻⁵ · max|λ|}`` is below
          ``_TOEPLITZ_R_THRESHOLD`` (default 2000). Exact to NUFFT
          ``eps``; cost dominated by ``O(r³)`` eigvalsh on a dense
          ``r × r`` reduced matrix.
        - **``k=None`` — KPM fallback.** Kernel Polynomial Method with
          stochastic trace estimation (see :meth:`_eigvals_kpm`).
          ``O(D · S · nufft)`` total, ``D ≈ 60``, ``S ≈ 30``. Returns
          a quantile-sampled approximation of the density; sets
          ``self._last_spectrum_from_kpm = True`` so downstream Liu
          consumers can warn on tail accuracy.

        Cached per ``centering`` mode for the full-spectrum paths.

        Parameters
        ----------
        k : int, optional
            Top-``k`` eigenvalues by magnitude (Lanczos). ``None``
            returns the full spectrum (Toeplitz-M / KPM).
        return_full_layout : bool, default False
            Kept for API compatibility with :meth:`FFTKernel.eigenvalues`;
            ignored.

        Returns
        -------
        np.ndarray
            Eigenvalues in descending order. Length ``k`` when ``k`` is
            given, ``n`` for the full-spectrum paths.
        """
        del return_full_layout  # kept only for API compatibility
        cache_key = "_spectrum_centered" if self.centering else "_spectrum_raw"
        n = self.n

        # Top-k: Lanczos. Not cached.
        if k is not None:
            from scipy.sparse.linalg import LinearOperator, eigsh

            def _matvec(v: np.ndarray) -> np.ndarray:
                return np.asarray(self.Kx(np.asarray(v, dtype=float)))

            op = LinearOperator(shape=(n, n), matvec=_matvec, dtype=float)
            k_req = min(int(k), n - 2)
            vals, _ = eigsh(op, k=k_req, which="LM")
            return np.sort(np.real(vals))[::-1]

        # Full spectrum — cached.
        cached = getattr(self, cache_key, None)
        if cached is not None and len(cached) == n:
            return cached

        toep = self._eigvals_toeplitz_M()
        if toep is not None:
            pad = max(0, n - len(toep))
            spectrum = np.concatenate([toep, np.zeros(pad)]) if pad else toep[:n]
        else:
            spectrum = self._eigvals_kpm(n_samples=n)

        setattr(self, cache_key, spectrum)
        return spectrum

    # ------------------------------------------------------------------
    # Path A primitives — k-space Parseval (default for xtKx / xtKy)
    # ------------------------------------------------------------------
    def _nufft_type1(self, x: np.ndarray) -> np.ndarray:
        """Type-1 NUFFT of ``x`` onto the k-grid at the cached scaled coords.

        Computes ``x̂(k) = Σ_j x_j exp(-i k · r̃_j)`` where ``r̃_j`` are the
        mean-centered, ``2π/(n·d)``-scaled versions of the input coordinates.
        Shared primitive for :meth:`xtKx` (takes ``|·|²``), :meth:`xtKy`
        (complex inner product), :meth:`Kx`, and :meth:`Kx_grid`.

        Parameters
        ----------
        x : np.ndarray
            ``(n,)`` or ``(n, M)``.

        Returns
        -------
        np.ndarray
            Complex ``(M, ny, nx)`` (always 3-D; ``M=1`` for 1-D input). DC
            at the array centre (finufft convention); callers that want the
            scipy-FFT layout must ``ifftshift`` along the last two axes.
        """
        if x.shape[0] != self.n:
            raise ValueError(f"x first dim {x.shape[0]} does not match n={self.n}.")
        ny, nx = self.grid_shape
        x_in = x[:, None] if x.ndim == 1 else x
        c = np.ascontiguousarray(x_in.T.astype(np.complex128))  # (M, N)
        return finufft.nufft2d1(
            self._y_scaled,
            self._x_scaled,
            c,
            n_modes=(ny, nx),
            eps=self._eps,
            isign=-1,
        )  # (M, ny, nx)

    def xtKx(self, x: np.ndarray) -> float | np.ndarray:
        """Quadratic form ``xᵀ K x`` via **k-space Parseval**.

        Implements the default path:

        .. math::

           x^T K x \\;=\\; \\frac{1}{n'} \\sum_k \\lambda(k) \\, |\\hat x(k)|^{2}

        using one type-1 NUFFT of ``x`` and an elementwise Parseval sum.
        Only the real power spectrum ``|x̂|²`` of shape ``(ny, nx)`` is
        materialized — no ``ifft2``, no spatial-grid copy of ``x``.

        Parameters
        ----------
        x : np.ndarray
            ``(n,)`` for one feature or ``(n, M)`` for ``M`` features.

        Returns
        -------
        float or np.ndarray
            Scalar if ``x`` is 1-D, shape ``(M,)`` otherwise.

        See Also
        --------
        xtKx_matmul : Compute ``xᵀ · Kx`` via the length-``n`` matrix product.
        """
        ny, nx = self.grid_shape
        # HKH quadratic form = raw K on H x. Center per-feature (column mean).
        if self.centering:
            if x.ndim == 1:
                x = x - x.mean()
            else:
                x = x - x.mean(axis=0, keepdims=True)
        x_hat_centered = self._nufft_type1(x)  # (M, ny, nx)
        power = x_hat_centered.real**2 + x_hat_centered.imag**2  # (M, ny, nx)
        # Spectrum is stored in scipy FFT layout (DC at [0,0]); fftshift → centered
        # to match the NUFFT output before multiplying.
        lam = np.fft.fftshift(self._fft_kernel.spectrum.reshape(ny, nx))
        Q = np.sum(lam[None, :, :] * power, axis=(1, 2)) / (ny * nx)
        if x.ndim == 1:
            return float(Q[0])
        return Q.astype(np.float64)

    def xtKy(self, x: np.ndarray, y: np.ndarray) -> float | np.ndarray:
        """Bilinear form ``xᵀ K y`` via **cross Parseval**.

        Implements the default path:

        .. math::

           x^T K y \\;=\\; \\frac{1}{n'} \\sum_k \\lambda(k) \\,
               \\overline{\\hat x(k)}\\, \\hat y(k).

        Paired same-``M`` convention — returns the diagonal of ``Xᵀ K Y``
        (shape ``(M,)``) for batched inputs, scalar for 1-D inputs. For the
        bipartite ``(M_x, M_y)`` cross matrix use :meth:`xtKy_matmul` (or
        build it explicitly via ``X.T @ self.Kx(Y)``).

        Parameters
        ----------
        x, y : np.ndarray
            ``(n,)`` or ``(n, M)`` — must share shape.

        Returns
        -------
        float or np.ndarray
            Scalar for 1-D inputs; ``(M,)`` for batched.
        """
        if x.shape[0] != self.n or y.shape[0] != self.n:
            raise ValueError(
                f"x, y first dim must equal n={self.n}; got {x.shape[0]}, {y.shape[0]}."
            )
        if x.shape != y.shape:
            raise ValueError(f"x and y must share shape; got {x.shape} vs {y.shape}.")
        ny, nx = self.grid_shape
        if self.centering:
            # x^T HKH y = (Hx)^T K (Hy) — subtract per-feature means on both sides.
            if x.ndim == 1:
                x = x - x.mean()
                y = y - y.mean()
            else:
                x = x - x.mean(axis=0, keepdims=True)
                y = y - y.mean(axis=0, keepdims=True)
        x_hat = self._nufft_type1(x)  # (M, ny, nx) complex
        y_hat = self._nufft_type1(y)
        lam = np.fft.fftshift(self._fft_kernel.spectrum.reshape(ny, nx))
        cross = np.real(np.conj(x_hat) * y_hat) * lam[None, :, :]
        R = np.sum(cross, axis=(1, 2)) / (ny * nx)
        if x.ndim == 1:
            return float(R[0])
        return R.astype(np.float64)

    # ------------------------------------------------------------------
    # Path B primitives — n-point vector via NUFFT round-trip
    # ------------------------------------------------------------------
    def Kx(self, z: np.ndarray) -> np.ndarray:
        """Matrix–vector product ``K z`` at the ``n`` irregular coordinates.

        Implements the band-limited apply

        .. math::

           K z \\;\\approx\\; \\tfrac{1}{n'} \\, U \\bigl(\\lambda \\odot U^{\\mathsf H} z\\bigr),

        evaluated as type-1 NUFFT → elementwise multiply by ``λ(k) / n'`` →
        type-2 NUFFT. Output length ``n``, same shape as ``z``. Base primitive
        for :meth:`xtKx_matmul`, :meth:`xtKy_matmul`, the KPM spectral-density
        estimator, and the bipartite R-test in :class:`DetectorIrregular`.

        Parameters
        ----------
        z : np.ndarray
            ``(n,)`` or ``(n, M)``.

        Returns
        -------
        np.ndarray
            Same shape as ``z``.
        """
        if z.shape[0] != self.n:
            raise ValueError(f"z first dim {z.shape[0]} does not match n={self.n}.")
        ny, nx = self.grid_shape
        lam_centred = np.fft.fftshift(self._fft_kernel.spectrum.reshape(ny, nx))
        squeeze = z.ndim == 1
        # HKH z = H · K · H z: center input pre-NUFFT, center output post.
        if self.centering:
            if squeeze:
                z = z - z.mean()
            else:
                z = z - z.mean(axis=0, keepdims=True)
        z_hat = self._nufft_type1(z)  # (M, ny, nx) complex, DC centred
        out_k = np.ascontiguousarray(
            (lam_centred[None, :, :] * z_hat / (ny * nx)).astype(np.complex128)
        )
        Kz = finufft.nufft2d2(
            self._y_scaled,
            self._x_scaled,
            out_k,
            eps=self._eps,
            isign=+1,
        )  # (M, n)
        Kz = np.real(Kz).T  # (n, M)
        if self.centering:
            if squeeze:
                Kz = Kz - Kz.mean(axis=0, keepdims=True)
            else:
                Kz = Kz - Kz.mean(axis=0, keepdims=True)
        return Kz[:, 0] if squeeze else Kz

    def xtKx_matmul(self, x: np.ndarray) -> float | np.ndarray:
        """Quadratic form ``xᵀ K x`` via **direct matmul**.

        Computes ``Q_B = xᵀ · self.Kx(x)`` end-to-end at the ``n`` irregular
        points. Sparse-aware on ``x`` (``x.multiply(Kx).sum``). ~2× the NUFFT
        work of :meth:`xtKx` per feature; agrees with it to NUFFT precision
        on regular grids and to the torus-BC band (~1–2 %) on irregular ones.

        Parameters
        ----------
        x : np.ndarray or scipy.sparse matrix
            ``(n,)`` or ``(n, M)``.

        Returns
        -------
        float or np.ndarray
            Scalar for 1-D input; ``(M,)`` for batched.
        """
        if sp.issparse(x):
            if x.ndim == 1 or (x.shape[1] == 1 and x.shape[0] == self.n):
                x_sp = x.reshape(-1, 1)
                squeeze = True
            else:
                x_sp = x
                squeeze = False
            Kx_dense = self.Kx(x_sp.toarray())
            result = np.asarray(x_sp.multiply(Kx_dense).sum(axis=0)).ravel()
            return float(result[0]) if squeeze else result
        arr = np.asarray(x, dtype=float)
        Kx_dense = self.Kx(arr)
        if arr.ndim == 1:
            return float(np.dot(arr, Kx_dense))
        return np.sum(arr * Kx_dense, axis=0).astype(np.float64)

    def xtKy_matmul(self, x: np.ndarray | sp.spmatrix, y: np.ndarray) -> float | np.ndarray:
        """Bilinear form ``xᵀ K y`` via **direct matmul**.

        Returns the paired ``(M,)`` diagonal of ``Xᵀ K Y`` (sparse-aware on
        ``x``). For the full ``(M_x, M_y)`` bipartite cross matrix build it
        explicitly as ``X.T @ self.Kx(Y)`` — that's what
        :class:`DetectorIrregular` does for ``compute_rstat`` when the two
        feature blocks have different widths.

        Parameters
        ----------
        x, y : np.ndarray or scipy.sparse matrix
            ``(n,)`` or ``(n, M)``.

        Returns
        -------
        float or np.ndarray
            Scalar for 1-D inputs; ``(M,)`` for batched.
        """
        Ky = self.Kx(y)
        if sp.issparse(x):
            x_in = x.reshape(-1, 1) if x.ndim == 1 else x
            Ky_2d = Ky[:, None] if Ky.ndim == 1 else Ky
            result = np.asarray(x_in.multiply(Ky_2d).sum(axis=0)).ravel()
            return float(result[0]) if result.size == 1 else result
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1 and Ky.ndim == 1:
            return float(np.dot(x_arr, Ky))
        x_mat = x_arr.reshape(-1, 1) if x_arr.ndim == 1 else x_arr
        Ky_mat = Ky.reshape(-1, 1) if Ky.ndim == 1 else Ky
        return np.sum(x_mat * Ky_mat, axis=0).astype(np.float64)

    def Kx_grid(self, x: np.ndarray) -> np.ndarray:
        """Grid-domain companion of :meth:`Kx` — ``(ny, nx)`` spatial output.

        Whereas :meth:`Kx` returns the length-``n`` apply at the irregular
        coordinates, :meth:`Kx_grid` returns the apply evaluated on the
        internal uniform grid. Pipeline: type-1 NUFFT → undo the
        coordinate-centering phase (needed here because we keep complex
        coefficients; the square-magnitude and adjoint-round-trip paths of
        :meth:`xtKx` / :meth:`Kx` absorb it automatically) → multiply by
        ``λ(k)`` → ``ifftshift`` → ``ifft2`` → real.

        Parameters
        ----------
        x : np.ndarray
            ``(n,)`` or ``(n, M)``.

        Returns
        -------
        np.ndarray
            Real ``(ny, nx)`` or ``(ny, nx, M)`` in the scipy FFT layout (DC
            at ``[0, 0]``).
        """
        ny, nx = self.grid_shape
        dy, dx = self.spacing
        squeeze = x.ndim == 1
        x_hat_centered = self._nufft_type1(x)  # (M, ny, nx), modes ∈ [-n/2, n/2-1]
        m_y = np.arange(ny) - ny // 2
        m_x = np.arange(nx) - nx // 2
        phase = (
            np.exp(-1j * m_y * self._y_mean * 2.0 * np.pi / (ny * dy))[:, None]
            * np.exp(-1j * m_x * self._x_mean * 2.0 * np.pi / (nx * dx))[None, :]
        )
        # Apply spectrum + undo centering phase in one pass.
        lam_centred = np.fft.fftshift(self._fft_kernel.spectrum.reshape(ny, nx))
        weighted = x_hat_centered * phase[None, :, :] * lam_centred[None, :, :]
        # Shift DC to [0, 0] and inverse-FFT to spatial grid.
        weighted_shifted = np.fft.ifftshift(weighted, axes=(-2, -1))
        Kx_grid = np.real(
            scipy.fft.ifft2(weighted_shifted, axes=(-2, -1), workers=self.workers)
        )  # (M, ny, nx)
        out = np.moveaxis(Kx_grid, 0, -1)  # (ny, nx, M)
        return out[..., 0] if squeeze else out

    # ------------------------------------------------------------------
    # Null-moment estimators — analytic Toeplitz formula and z-scored-probe
    # empirical estimator (for Dirichlet-ratio-sensitive kernels).
    # ------------------------------------------------------------------
    def _coord_power_spectrum(self) -> np.ndarray:
        """``|φ(k)|²`` on the k-grid in scipy-FFT (DC-at-[0,0]) layout.

        Uses the cached (symmetrized) ``φ`` from :meth:`_coord_phi`. The
        magnitude-square is coord-only, independent of the kernel method
        and spectrum. Feeds the exact analytic ``trace(K²)`` formula
        ``trace(K_n²) = (n²/n'²) · Σ_{k,k'} λ(k) λ(k') |φ(k'-k)|²``,
        which reduces to ``(n/n')² · Σ_k λ(k)²`` when coords lie on the
        k-grid (then ``|φ|² = δ``).

        Returns
        -------
        np.ndarray
            ``(ny, nx)`` real, non-negative, with ``|φ(0)|² = 1`` at
            ``[0, 0]`` (DC-at-origin layout).
        """
        cache = getattr(self, "_phi2_cache", None)
        if cache is not None:
            return cache
        phi2 = np.abs(self._coord_phi()) ** 2
        self._phi2_cache = phi2
        return phi2

    def _default_moment_method(self) -> str:
        """Default moment estimator.

        ``'analytic'`` for raw-moment queries — ``trace(K) = (n/n')·Σλ``
        (exact) and ``trace(K²) = (n²/n'²)·λᵀΨλ`` (Toeplitz FFT
        convolution, exact to NUFFT band-limit precision).

        **Exceptions** — ``car`` / ``graph_laplacian`` / ``moran`` under
        ``centering=True`` default to ``'empirical'``. Two distinct
        failure modes:

        - ``car`` / ``graph_laplacian`` have **broad spectral support**
          (polynomial decay, but far slower than Matern/Gaussian — e.g.
          CAR ρ=0.9 spans factor 20 from peak to tail, Matern bw=3
          spans factor 10⁶). Broad support means a larger fraction of
          the trace lives in the tail, so the spectral IPR
          ``r_eff = (Σλ)²/Σλ²`` is much larger than for smooth kernels,
          and ``mean²/(m·sq) → 1`` as ``r_eff → n``. The downstream
          Dirichlet(1/2) ratio ``Var[Q] = 2·(m·sq − mean²)/(m+2)``
          amplifies any relative error in ``sq`` by a factor
          ``1/(1 − mean²/(m·sq))`` — about 13× for CAR and 23× for
          graph_laplacian at practical settings. The analytic ``sq``
          itself is accurate to ~1–4 % (NUFFT band-limit residual), but
          that bias becomes a 20–80 % ``Var[Q]`` bias after amplification.
          Empirical probes measure ``Var[Q]`` directly from the sample
          variance of ``Q`` on z-scored probes and invert through the
          same ratio, so the amplification cancels by construction.
        - ``moran`` has indefinite ``Λ`` (mixed signs on the k-grid,
          ``Σλ ≈ 0`` by construction). The Toeplitz quadratic form
          ``λᵀΨλ`` becomes a delicate cancellation between sign-opposite
          pairs; FFT convolution loses enough precision that the
          analytic result can flip sign (observed values ≈ 0 at
          ``n = 1024`` vs an empirical-ground-truth of ``∼ 16``).
          Empirical bypasses the cancellation.

        Callers who want the analytic path anyway can pass
        ``method='analytic'`` explicitly, and optionally
        ``compute_null_params(..., dirichlet_correction=False)`` to drop
        the ratio correction (a conservative upper bound on ``Var[Q]``).
        """
        if self.centering and self.method in ("car", "graph_laplacian", "moran"):
            return "empirical"
        return "analytic"

    def _get_centered_probe_cache(self, n_probes: int = 100) -> dict:
        """Cached ``Q`` values on z-scored standard-normal probes.

        Estimator of ``trace(HKH)`` and ``trace((HKH)²)`` without ever
        forming ``H`` or ``HKH``. We draw ``V_i ~ N(0, I)`` and standardize
        each column to get ``Z_i = (V_i − V̄_i·𝟏) / σ_{V_i}`` — the same
        transformation :func:`spatial_q_test` applies to real data — then
        compute ``Q_i = Z_iᵀ K Z_i``. Because
        ``Z_iᵀ K Z_i = V_iᵀ (H K H) V_i / σ²`` and ``V_i`` is Gaussian with
        identity covariance, the sample moments of ``{Q_i}`` converge to
        ``trace(HKH)`` (mean) and ``2·trace((HKH)²)`` (variance), giving
        honest null moments for the realized quadratic form without any
        explicit ``H``-projection work.
        """
        cache = getattr(self, "_centered_probe_cache", None)
        if cache is not None and cache["n_probes"] == n_probes:
            return cache
        rng = np.random.default_rng(0)
        V = rng.standard_normal((self.n, n_probes))
        Vz = _standardize_features(V)
        Q_probes = np.atleast_1d(self.xtKx(Vz)).astype(float).ravel()
        self._centered_probe_cache = {"n_probes": n_probes, "Q": Q_probes}
        return self._centered_probe_cache

    def trace(self, method: str | None = None, n_probes: int | None = None) -> float:
        """``trace(K)`` (raw) or ``trace(HKH)`` (centered).

        Parameters
        ----------
        method : {``'analytic'``, ``'empirical'``, ``None``}, optional
            - ``None`` (default): see :meth:`_default_moment_method`.
            - ``'analytic'``: ``(n/n') · Σ_k λ(k)``. Exact —
              ``G_{kk} = n`` regardless of coord arrangement. Adjusts by
              ``-s₁/n`` when ``centering=True`` (``s₁ = 𝟏ᵀ K 𝟏``).
            - ``'empirical'``: sample mean of ``Q = zᵀ K z`` over
              ``n_probes`` z-scored standard-normal probes. A direct
              Monte-Carlo estimate of ``trace(HKH)``; requires
              ``centering=True``.
        n_probes : int, optional
            Probe count for ``'empirical'`` (default 100). Ignored by
            ``'analytic'``.

        Returns
        -------
        float
        """
        if method is None:
            method = self._default_moment_method()
        if method == "empirical":
            if not self.centering:
                raise ValueError("'empirical' requires centering=True.")
            cache = self._get_centered_probe_cache(n_probes if n_probes else 100)
            return float(cache["Q"].mean())
        if method != "analytic":
            raise ValueError(f"method must be 'analytic' or 'empirical', got {method!r}.")
        # trace(K_n) = (n/n') · trace(K_grid) — exact because the diagonal
        # of G = UᴴU is n for any coord arrangement.
        ny, nx = self.grid_shape
        raw = float(self._fft_kernel.trace() * self.n / (ny * nx))
        if not self.centering:
            return raw
        s1, _ = self._ones_stats()
        return raw - s1 / self.n

    def square_trace(self, method: str | None = None, n_probes: int | None = None) -> float:
        """``trace(K²)`` (raw) or ``trace((HKH)²)`` (centered).

        Parameters
        ----------
        method : {``'analytic'``, ``'empirical'``, ``None``}, optional
            - ``None`` (default): see :meth:`_default_moment_method`.
            - ``'analytic'``: ``(n²/n'²) · λᵀ Ψ λ`` with Toeplitz
              ``Ψ_{k,k'} = |φ(k'-k)|²``, evaluated by circular FFT
              convolution of ``|φ|²`` with ``λ`` in ``O(n' log n')``.
              ``φ(k) = (1/n) Σ_i exp(-ik·y_i)`` comes from one type-1
              NUFFT of the ones vector. On a regular grid where coords
              coincide with the k-grid, ``|φ|² = δ`` and the formula
              collapses to ``(n/n')² · Σ_k λ(k)²``. Adjusts by
              ``-2·s₂/n + s₁²/n²`` when ``centering=True``.
            - ``'empirical'``: inverts the Dirichlet(1/2) ratio
              ``Var[Q] = 2·(m·trace((HKH)²) − trace(HKH)²)/(m+2)`` on
              the same probe cache as :meth:`trace` to recover
              ``trace((HKH)²)`` from the probe variance of ``Q``.
              Requires ``centering=True``.
        n_probes : int, optional
            Probe count for ``'empirical'`` (default 100).

        Returns
        -------
        float
        """
        if method is None:
            method = self._default_moment_method()
        if method == "empirical":
            if not self.centering:
                raise ValueError("'empirical' requires centering=True.")
            cache = self._get_centered_probe_cache(n_probes if n_probes else 100)
            Q = cache["Q"]
            tr_HKH = float(Q.mean())
            var_Q_emp = float(Q.var(ddof=1))
            m = max(int(self.n) - 1, 1)
            tr_HKH_sq = (var_Q_emp * (m + 2) / 2.0 + tr_HKH**2) / m
            return max(tr_HKH_sq, 0.0)
        if method != "analytic":
            raise ValueError(f"method must be 'analytic' or 'empirical', got {method!r}.")
        ny, nx = self.grid_shape
        nprime = ny * nx
        lam = self._fft_kernel.spectrum.reshape(ny, nx)
        phi2 = self._coord_power_spectrum()
        lam_f = np.fft.fft2(lam)
        phi2_f = np.fft.fft2(phi2)
        conv = np.fft.ifft2(lam_f * phi2_f).real  # (|φ|² ⋆ λ)(k)
        raw = (self.n**2 / nprime**2) * float(np.sum(lam * conv))
        # ``trace(K²) = Σᵢ μᵢ² ≥ 0`` for any real symmetric ``K``. Negative
        # values here are FFT-cancellation noise on indefinite ``Λ``
        # (e.g. Moran), not a valid result — clip to zero.
        raw = max(raw, 0.0)
        if not self.centering:
            return raw
        s1, s2 = self._ones_stats()
        return max(raw - 2.0 * s2 / self.n + s1**2 / (self.n**2), 0.0)


# ---------------------------------------------------------------------------
# Q-test and R-test on irregular spatial coordinates via NUFFT
# ---------------------------------------------------------------------------


def _standardize_features(X: np.ndarray) -> np.ndarray:
    """Z-score each column (ddof=1), leaving constant columns as zeros.

    Matches :func:`quadsv.statistics.spatial_q_test`'s convention. Used by the
    NUFFT dispatch to standardize at the ``n`` irregular points before the
    type-1 NUFFT.
    """
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True, ddof=1)
    out = np.zeros_like(X, dtype=float)
    valid = sd > 1e-12
    np.divide(X - mu, sd, out=out, where=valid)
    return out


def _q_test_nufft(  # noqa: C901
    Xn: np.ndarray,
    kernel: NUFFTKernel,
    null_params: dict | None = None,
    return_pval: bool = True,
    is_standardized: bool = False,
) -> float | np.ndarray | tuple[float, float] | tuple[np.ndarray, np.ndarray]:
    """
    Spatial Q-test on irregular 2D coordinates.

    Computes ``Q = xᵀ K x`` via the k-space Parseval identity
    ``Q = (1/n') Σ_k λ(k) · |ẑ(k)|²`` (one type-1 NUFFT of ``z`` + a
    Parseval sum; see :meth:`NUFFTKernel.xtKx`). The length-``n`` matmul
    counterpart :meth:`NUFFTKernel.xtKx_matmul` computes the same form to
    NUFFT precision and is exposed for callers that prefer the direct
    round-trip; :func:`spatial_q_test` always uses the spectral path.

    Null moments route through :func:`quadsv.statistics.compute_null_params`,
    which on graph kernels defaults to the empirical moment estimator over
    ``HKH``-centered probes (see :meth:`NUFFTKernel.trace` /
    :meth:`NUFFTKernel.square_trace`).

    Standardization at the ``n`` irregular points is applied internally
    unless ``is_standardized=True``.

    Parameters
    ----------
    Xn : np.ndarray
        ``(n,)`` or ``(n, M)``.
    kernel : NUFFTKernel
    null_params : dict, optional
        Pre-built moments (see :func:`quadsv.compute_null_params`). Read
        keys depend on the kernel method: ``'mean_Q'`` / ``'var_Q'`` for
        Moran (CLT path); ``'eigenvalues'`` for everything else (Liu's
        method).
    return_pval : bool, default True
    is_standardized : bool, default False

    Returns
    -------
    Q : float or np.ndarray
    pval : float or np.ndarray, optional
    """
    Xn = np.asarray(Xn, dtype=float)
    batched = Xn.ndim == 2
    X_in = Xn if batched else Xn[:, None]
    if X_in.shape[0] != kernel.n:
        raise ValueError(f"Xn first dim {X_in.shape[0]} does not match kernel.n={kernel.n}.")

    z = X_in if is_standardized else _standardize_features(X_in)

    # Spectral Parseval path — agrees with xtKx_matmul to NUFFT precision
    # (verified on both regular and irregular grids). Keeping only one path
    # here avoids a stateful switch on the kernel object.
    Q_arr = np.atleast_1d(kernel.xtKx(z)).ravel()

    if not return_pval:
        return Q_arr if batched else float(Q_arr[0])

    # Dispatch on the user-selected null approximation. This mirrors the
    # MatrixKernel path in `spatial_q_test`: the caller picks one of
    # {'clt', 'welch', 'liu'} via ``null_params['method']``; defaults keep
    # backward-compatible behavior — CLT for Moran (its kernel is
    # indefinite so Welch/Liu are degenerate) and Liu for the PSD kernels.
    if null_params is not None and "method" in null_params:
        null_approx = str(null_params["method"])
    else:
        null_approx = "clt" if kernel.method == "moran" else "liu"
    if kernel.method == "moran" and null_approx != "clt":
        raise ValueError(
            f"Moran's I kernel is indefinite; only null_method='clt' is "
            f"supported for the Q-test. Got method={null_approx!r}."
        )

    def _get_mean_var() -> tuple[float, float]:
        """Mean/var of Q under H0 — from user-supplied params or recompute."""
        if null_params is not None and "mean_Q" in null_params and "var_Q" in null_params:
            return float(null_params["mean_Q"]), float(null_params["var_Q"])
        # Route through compute_null_params to pick up the H-centering +
        # finite-n ratio correction; for NUFFT graph kernels the
        # ``'empirical'`` default on trace()/square_trace() ensures the
        # corrections capture the spreading-kernel smoothing too.
        from quadsv.statistics import compute_null_params

        p = compute_null_params(kernel, method="clt")
        return float(p["mean_Q"]), float(p["var_Q"])

    if null_approx == "clt":
        mean_Q, var_Q = _get_mean_var()
        sigma = float(np.sqrt(var_Q))
        if sigma <= 1e-12:
            pvals = np.ones_like(Q_arr)
        else:
            z_scores = (Q_arr - mean_Q) / sigma
            pvals = chi2.sf(z_scores**2, df=1)

    elif null_approx == "welch":
        # Welch-Satterthwaite: Q ~ g · χ²(df=h) with g = var / (2·mean),
        # h = 2·mean² / var. Requires mean > 0 (PSD kernel).
        if null_params is not None and "scale_g" in null_params and "df_h" in null_params:
            g = float(null_params["scale_g"])
            h = float(null_params["df_h"])
        else:
            mean_Q, var_Q = _get_mean_var()
            if mean_Q <= 0 or var_Q <= 0:
                pvals = np.ones_like(Q_arr)
                g = h = None
            else:
                g = var_Q / (2.0 * mean_Q)
                h = 2.0 * mean_Q**2 / var_Q
        if g is not None:
            pvals = chi2.sf(Q_arr / g, df=h)

    elif null_approx == "liu":
        if null_params is not None and "eigenvalues" in null_params:
            sig_evals = np.asarray(null_params["eigenvalues"], dtype=float)
        else:
            evals = kernel.eigenvalues(return_full_layout=True)
            if getattr(kernel, "_last_spectrum_from_kpm", False):
                warnings.warn(
                    "Liu's method p-value uses a KPM-reconstructed spectrum "
                    "for this NUFFTKernel (Toeplitz-M fallback not available "
                    "for centered/indefinite or broad-spectrum cases). The "
                    "density is Jackson-damped with finite polynomial degree, "
                    "so p-values in the extreme tail (p ≲ 1e-3) may carry a "
                    "few-percent systematic error from the moment-matching. "
                    "Use 'welch' for a moment-only alternative, or increase "
                    "NUFFTKernel._KPM_DEGREE / _KPM_N_PROBES for more accuracy.",
                    stacklevel=2,
                )
            if evals.min() < -0.1:
                raise ValueError(
                    "Kernel has significant negative eigenvalues; Liu's method may be invalid."
                )
            sig_evals = evals[evals > 1e-9]
        pvals = np.array([liu_sf(float(q), sig_evals) for q in Q_arr])

    else:
        raise ValueError(f"Unknown null approximation method: {null_approx!r}")

    if batched:
        return Q_arr, pvals
    return float(Q_arr[0]), float(pvals[0])


def _r_test_nufft(
    Xn: np.ndarray,
    Yn: np.ndarray,
    kernel: NUFFTKernel,
    null_params: dict | None = None,
    return_pval: bool = True,
    is_standardized: bool = False,
) -> float | np.ndarray | tuple[float, float] | tuple[np.ndarray, np.ndarray]:
    """
    Spatial R-test on irregular 2D coordinates.

    Dispatches purely on the input shape:

    - Paired (``M_x == M_y``) — returns the ``(M,)`` diagonal of
      ``Xᵀ K Y`` via cross Parseval
      ``R = (1/n') Σ_k λ(k) · conj(x̂(k)) · ŷ(k)`` (one pair of
      type-1 NUFFTs; see :meth:`NUFFTKernel.xtKy`).
    - Bipartite (``M_x != M_y``) — returns the full ``(M_x, M_y)`` cross
      matrix via ``Xᵀ · self.Kx(Y)``.

    In either case ``var_R = kernel.square_trace()`` (the default
    ``centering=True`` returns ``trace((HKH)²)``, which is exactly
    ``Var[Xᵀ K Y]`` on z-scored inputs).

    Parameters
    ----------
    Xn, Yn : np.ndarray
        ``(n,)`` or ``(n, M)``.
    kernel : NUFFTKernel
    null_params : dict, optional
        ``{'var_R': ...}`` in the n-point-operator units.
    return_pval : bool, default True
    is_standardized : bool, default False
    """
    Xn = np.asarray(Xn, dtype=float)
    Yn = np.asarray(Yn, dtype=float)
    if Xn.ndim == 1:
        Xn = Xn[:, None]
    if Yn.ndim == 1:
        Yn = Yn[:, None]
    if Xn.shape[0] != kernel.n or Yn.shape[0] != kernel.n:
        raise ValueError(
            f"Xn, Yn first dim must equal kernel.n={kernel.n}; "
            f"got {Xn.shape[0]}, {Yn.shape[0]}."
        )

    Xz = Xn if is_standardized else _standardize_features(Xn)
    Yz = Yn if is_standardized else _standardize_features(Yn)

    if Xn.shape[1] == Yn.shape[1]:
        # Paired diagonal via cross Parseval — one type-1 NUFFT per column.
        R = np.atleast_1d(kernel.xtKy(Xz, Yz))
    else:
        # Bipartite (M_x, M_y) cross matrix via NUFFT round-trip on Y.
        KY = kernel.Kx(Yz)  # (n, M_y)
        R = Xz.T @ KY  # (M_x, M_y)

    if not return_pval:
        return R.squeeze() if R.size > 1 else float(R)

    if null_params is not None and "var_R" in null_params:
        var_R = float(null_params["var_R"])
    else:
        # kernel.square_trace() returns trace((HKH)²) by default (centering=True),
        # which is exactly Var[R] for Zₓᵀ K Zᵧ with both X, Y z-scored.
        var_R = float(kernel.square_trace())
    sigma = float(np.sqrt(max(var_R, 1e-30)))
    z_scores = R / sigma
    pvals = 2.0 * norm.sf(np.abs(z_scores))
    return R.squeeze(), pvals.squeeze()
