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
    ``K z = (1/n') · U · (λ ⊙ Uᴴ z)`` and serves as
    the base for the Hutchinson null estimators and the bipartite R-test cross
    matrix.

    :attr:`compute_method` selects which path :func:`quadsv.spatial_q_test` /
    :func:`quadsv.spatial_r_test` take against this kernel:

    - ``'spectral'`` (default): k-space Parseval via :meth:`xtKx` /
      :meth:`xtKy`; null moments from the analytic n-point-scaled FFT
      spectrum.
    - ``'matmul'``: length-``n`` matrix product via :meth:`xtKx_matmul` /
      :meth:`Kx`; null moments from Hutchinson probes through ``K``.

    Both paths agree to NUFFT precision (``eps``) on a regular grid
    and to ~1 – 2 % on irregular points.

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
    compute_method : {``'spectral'``, ``'matmul'``}, default ``'spectral'``
        Which of the two paths :func:`quadsv.spatial_q_test` /
        :func:`quadsv.spatial_r_test` use. Stored on the instance as a mutable
        attribute so callers can flip it between calls without rebuilding.
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
    compute_method : str
        Currently selected NUFFT path — mutable. One of:

        - ``'spectral'`` (default, fast): k-space Parseval via
          :meth:`xtKx` / :meth:`xtKy`; null moments from the analytic
          n-point-scaled FFT spectrum.
        - ``'matmul'``: full matrix product via
          :meth:`xtKx_matmul` / :meth:`Kx`; null moments estimated by
          Hutchinson probes.

        Flip at any time between calls to swap paths without rebuilding.
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
        compute_method: str = "spectral",
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
        compute_method : {``'spectral'``, ``'matmul'``}, default ``'spectral'``
            Initial value for the mutable :attr:`compute_method` attribute
            that selects the NUFFT path used by
            :func:`quadsv.spatial_q_test` / :func:`quadsv.spatial_r_test`.
        **kwargs
            Method-specific kernel hyperparameters (``bandwidth``, ``nu``,
            ``rho``, ``neighbor_degree``).

        Raises
        ------
        ValueError
            If ``coords`` has the wrong shape, ``method`` / ``compute_method``
            is unknown, or ``grid_shape`` / ``spacing`` are invalid.
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

        self.coords: np.ndarray = coords
        self.n: int = coords.shape[0]
        self.grid_shape: tuple[int, int] = (ny, nx)
        self.spacing: tuple[float, float] = (dy, dx)
        self.method: str = method
        self._unit_scale: float = float(unit_scale)
        self._eps: float = float(eps)
        self.workers: int | None = workers
        self.stores_precision: bool = False
        if compute_method not in ("spectral", "matmul"):
            raise ValueError(
                f"compute_method must be 'spectral' or 'matmul', got {compute_method!r}."
            )
        # Public mutable attribute — description lives in the class
        # docstring's Attributes section so Sphinx AutoAPI picks it up
        # cleanly. Mutable: flip between 'spectral' / 'matmul' at any time
        # to swap NUFFT paths without rebuilding the kernel.
        self.compute_method: str = compute_method

        # Internal FFTKernel holds the eigenvalue spectrum on the k-grid. We
        # use fft2 (full spectrum) so the ifftshift trick aligns NUFFT output
        # with the scipy FFT layout (DC at [0, 0]).
        self._fft_kernel = FFTKernel(
            shape=(ny, nx),
            spacing=(dy, dx),
            topology="square",
            method=method,
            fft_solver="fft2",
            workers=workers,
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
            f"compute_method={self.compute_method!r} params={self.params}>"
        )

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"NUFFTKernel\n"
            f"- Method: {self.method}\n"
            f"- Number of spots: {self.n}\n"
            f"- k-grid: {self.grid_shape} at spacing {self.spacing}\n"
            f"- Compute method: {self.compute_method}\n"
            f"- Params: {self.params}"
        )

    # ------------------------------------------------------------------
    # Cached scaling between the n'-point grid operator K' and the n-point
    # irregular-point operator K. Under the band-limited approximation
    # ``K ≈ (1/n') · U · diag(λ) · Uᴴ`` and H₀ with x iid N(0, 1) at n points,
    # the moments of K scale with ``n/n'`` relative to K'.
    # ------------------------------------------------------------------
    @property
    def _n_over_nprime(self) -> float:
        ny, nx = self.grid_shape
        return self.n / (ny * nx)

    # ------------------------------------------------------------------
    def eigenvalues(self, k: int | None = None, return_full_layout: bool = False) -> np.ndarray:
        """Eigenvalues of the ``n × n`` irregular-point operator ``K`` (analytic).

        Returns ``fft_kernel.eigenvalues(...) · n / n'``, i.e. the internal
        FFT-kernel spectrum rescaled to the n-point operator — the analytic
        Liu spectrum used by both compute paths.

        Parameters
        ----------
        k : int, optional
            Return the top-``k`` sorted descending. ``None`` returns all.
        return_full_layout : bool, default False
            If ``True``, return the complete unsorted spectrum in the scipy
            FFT mode layout (length ``n'``). Useful when the caller wants to
            weight modes by grid position rather than by magnitude ranking.
        """
        raw = self._fft_kernel.eigenvalues(k=k, return_full_layout=return_full_layout)
        return raw * self._n_over_nprime

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
        for :meth:`xtKx_matmul`, :meth:`xtKy_matmul`, the Hutchinson null
        estimators, and the bipartite R-test in :class:`DetectorIrregular`.

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
        :class:`DetectorIrregular` does for ``compute_rstat`` after setting
        ``kernel.compute_method = 'matmul'``.

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
    # Null-moment estimators — analytic (Path A) and Hutchinson (Path B)
    # ------------------------------------------------------------------
    def _get_rvs_trace_cache(self, n_probes: int = 15) -> dict:
        """Cache Hutchinson probes ``v ∈ {±1}^n`` and their ``K v`` images.

        Used by :meth:`trace` / :meth:`square_trace` with
        ``method='hutchinson'``. The probes are drawn from a seeded RNG
        (``default_rng(0)``) so repeated calls return the same values for a
        given kernel instance, mirroring
        :meth:`MatrixKernelBase._get_rvs_trace_cache`.
        """
        cache = getattr(self, "_trace_rvs_cache", None)
        if cache is not None and cache["n_probes"] == n_probes:
            return cache
        rng = np.random.default_rng(0)
        rvs = rng.choice([-1.0, 1.0], size=(self.n, n_probes)).astype(np.float64)
        Krvs = self.Kx(rvs)  # (n, n_probes)
        self._trace_rvs_cache = {"n_probes": n_probes, "rvs": rvs, "Krvs": Krvs}
        return self._trace_rvs_cache

    def trace(self, method: str = "analytic", n_probes: int = 15) -> float:
        """``trace(K)`` of the ``n × n`` irregular-point operator.

        Parameters
        ----------
        method : {``'analytic'``, ``'hutchinson'``}, default ``'analytic'``
            - ``'analytic'``: ``(n/n') · fft_kernel.trace()``.
              Exact under the band-limited approximation; zero NUFFT work,
              no randomness.
            - ``'hutchinson'``: ``(1/m) Σ_i vᵢᵀ (K vᵢ)`` over
              ``m = n_probes`` cached ``±1`` probes. Exact in expectation
              for the NUFFT-applied operator; deterministic per instance
              because probes are drawn from a seeded RNG.
        n_probes : int, default 15
            Number of Hutchinson probes (ignored when ``method='analytic'``).
        """
        if method == "analytic":
            return float(self._fft_kernel.trace() * self._n_over_nprime)
        if method == "hutchinson":
            cache = self._get_rvs_trace_cache(n_probes)
            return float(np.sum(cache["rvs"] * cache["Krvs"]) / cache["n_probes"])
        raise ValueError(f"method must be 'analytic' or 'hutchinson', got {method!r}.")

    def square_trace(self, method: str = "analytic", n_probes: int = 15) -> float:
        """``trace(K²)`` of the ``n × n`` irregular-point operator.

        Same parameter semantics as :meth:`trace`.

        - ``'analytic'``: ``(n/n')² · fft_kernel.square_trace()``.
        - ``'hutchinson'``: ``(1/m) Σ_i ‖K vᵢ‖²`` over the cached probes.
        """
        if method == "analytic":
            return float(self._fft_kernel.square_trace() * self._n_over_nprime**2)
        if method == "hutchinson":
            cache = self._get_rvs_trace_cache(n_probes)
            return float(np.sum(cache["Krvs"] ** 2) / cache["n_probes"])
        raise ValueError(f"method must be 'analytic' or 'hutchinson', got {method!r}.")


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

    Two computational paths approximate the same ``xᵀ K x`` on the
    ``n × n`` irregular-point operator ``K``; which one runs is controlled
    by :attr:`NUFFTKernel.compute_method` on the kernel instance (mutable):

    - ``'spectral'`` (default) — fast. Computes
      ``Q = (1/n') Σ_k λ(k) · |ẑ(k)|²`` via one type-1 NUFFT of ``z`` and a
      Parseval sum (:meth:`NUFFTKernel.xtKx`). Null moments come from the
      analytic ``n/n'``-scaled FFT spectrum, aligning with the n-point
      operator's null distribution.
    - ``'matmul'`` — slower, end-to-end at the ``n`` irregular
      points. Computes ``Q = zᵀ · self.Kx(z)`` via two NUFFTs + a
      sparse-aware contraction (:meth:`NUFFTKernel.xtKx_matmul`). Null
      moments are Hutchinson estimates over cached ``±1`` probes through ``K``.

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
    compute_method = kernel.compute_method
    Xn = np.asarray(Xn, dtype=float)
    batched = Xn.ndim == 2
    X_in = Xn if batched else Xn[:, None]
    if X_in.shape[0] != kernel.n:
        raise ValueError(f"Xn first dim {X_in.shape[0]} does not match kernel.n={kernel.n}.")

    z = X_in if is_standardized else _standardize_features(X_in)

    # Compute Q via the requested path.
    if compute_method == "spectral":
        Q_arr = np.atleast_1d(kernel.xtKx(z)).ravel()
    else:
        Q_arr = np.atleast_1d(kernel.xtKx_matmul(z)).ravel()

    if not return_pval:
        return Q_arr if batched else float(Q_arr[0])

    # Null moments — analytic (spectral) vs. Hutchinson (matmul) per user spec.
    moment_method = "analytic" if compute_method == "spectral" else "hutchinson"
    if kernel.method == "moran":
        if null_params is not None and "mean_Q" in null_params and "var_Q" in null_params:
            mean_Q = float(null_params["mean_Q"])
            var_Q = float(null_params["var_Q"])
        else:
            mean_Q = kernel.trace(method=moment_method)
            var_Q = 2.0 * kernel.square_trace(method=moment_method)
        sigma = float(np.sqrt(var_Q))
        if sigma <= 1e-12:
            pvals = np.ones_like(Q_arr)
        else:
            z_scores = (Q_arr - mean_Q) / sigma
            pvals = chi2.sf(z_scores**2, df=1)
    else:
        # Liu's mixture uses the analytic n-point-scaled spectrum in both paths
        # (the Hutchinson path would need a stochastic spectrum estimator —
        # out of scope).
        if null_params is not None and "eigenvalues" in null_params:
            sig_evals = np.asarray(null_params["eigenvalues"], dtype=float)
        else:
            evals = kernel.eigenvalues(return_full_layout=True)
            if evals.min() < -0.1:
                raise ValueError(
                    "Kernel has significant negative eigenvalues; Liu's method may be invalid."
                )
            sig_evals = evals[evals > 1e-9]
        pvals = np.array([liu_sf(float(q), sig_evals) for q in Q_arr])

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

    The path is chosen by :attr:`NUFFTKernel.compute_method` on the kernel:

    - ``'spectral'`` (default) — cross Parseval
      ``R = (1/n') Σ_k λ(k) · conj(x̂(k)) · ŷ(k)`` via
      :meth:`NUFFTKernel.xtKy`; analytic
      ``var_R = (n/n')² · fft_kernel.square_trace()``.
    - ``'matmul'`` — ``R = Xᵀ · self.Kx(Y)`` (full ``(M_x, M_y)`` cross
      matrix) with a sparse-aware contraction; Hutchinson
      ``var_R = kernel.square_trace(method='hutchinson')``.

    Paired same-``M`` inputs get the ``(M,)`` diagonal in ``'spectral'``
    mode; ``'matmul'`` always returns the full cross matrix (needed by
    :class:`DetectorIrregular`).

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
    compute_method = kernel.compute_method
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

    if compute_method == "spectral" and Xn.shape[1] == Yn.shape[1]:
        # Path A paired diagonal via cross Parseval.
        R = np.atleast_1d(kernel.xtKy(Xz, Yz))
    else:
        # Full (M_x, M_y) cross matrix via NUFFT round-trip on Y.
        KY = kernel.Kx(Yz)  # (n, M_y)
        R = Xz.T @ KY  # (M_x, M_y)

    if not return_pval:
        return R.squeeze() if R.size > 1 else float(R)

    moment_method = "analytic" if compute_method == "spectral" else "hutchinson"
    if null_params is not None and "var_R" in null_params:
        var_R = float(null_params["var_R"])
    else:
        var_R = kernel.square_trace(method=moment_method)
    sigma = float(np.sqrt(max(var_R, 1e-30)))
    z_scores = R / sigma
    pvals = 2.0 * norm.sf(np.abs(z_scores))
    return R.squeeze(), pvals.squeeze()
