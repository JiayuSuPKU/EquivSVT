"""
Non-uniform FFT (NUFFT) power spectra for spatial omics on irregular grids.

When data sit on a regular grid (e.g., a rasterized Visium slide),
:func:`quadsv.power_spectrum_2d` computes :math:`|\\hat{x}(k)|^2` with a plain
2D FFT. For data whose spatial coordinates are **irregular** — e.g., Slide-seq,
Stereo-seq, or a Visium slide read straight from
``adata.obsm['spatial']`` without rasterization — we need a non-uniform FFT
instead. :func:`power_spectrum_2d_nufft` evaluates

.. math::

   \\hat c(k_y, k_x) = \\sum_{j=1}^N c_j \\,
       \\exp\\!\\bigl[-i(k_y\\,y_j + k_x\\,x_j)\\bigr]

at the same uniform ``(ny, nx)`` k-space grid that :func:`power_spectrum_2d`
would produce for a rasterized input of the same physical extent, then returns
:math:`|\\hat c|^2` with the standard scipy FFT layout (DC at ``[0, 0]``).
Anything downstream — :func:`quadsv.spectral_compare.radial_bin_spectrum`,
:class:`quadsv.PatternComparatorNUFFT` — works
identically.

Per-sample unit handling
------------------------

Samples in different studies may ship coordinates in different units (μm,
Visium full-resolution pixels, etc.). :func:`power_spectrum_2d_nufft` accepts a
``unit_scale`` multiplier that converts each sample's coordinate axis to the
*common* physical unit chosen for ``spacing``. The common frequency grid is
``1 / (ny·dy)`` by ``1 / (nx·dx)`` cycles per unit length, so spectra from
slides with different shapes and pixel conventions become directly comparable.
"""

from __future__ import annotations

import logging

import finufft
import numpy as np
from scipy.stats import chi2, norm

from quadsv.fft import FFTKernel
from quadsv.statistics import liu_sf

__all__ = [
    "power_spectrum_2d_nufft",
    "NUFFTKernel",
    "spatial_q_test_nufft",
    "spatial_r_test_nufft",
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
        raise ValueError(f"coords must be (N, 2), got {scaled.shape}.")
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
    Compute the 2D power spectrum :math:`|\\hat{c}(k)|^2` of one or more non-uniform
    spatial signals via a type-1 NUFFT.

    The output has the same ``(ny, n_kx)`` layout as
    :func:`quadsv.power_spectrum_2d` with ``fft_solver='fft2'`` — DC at
    ``[0, 0]``, Nyquist at ``[ny/2, nx/2]`` (when dimensions are even) — so the
    result is a drop-in substitute for the rasterized spectrum in the rest of
    the pipeline.

    Parameters
    ----------
    coords : np.ndarray
        Non-uniform spatial coordinates, shape ``(N, 2)`` in the order
        ``(y, x)``. Values outside the physical domain implied by
        ``grid_shape`` and ``spacing`` are folded into ``[-π, π)`` by finufft.
    values : np.ndarray
        Signal strengths at each coordinate. Shape ``(N,)`` for a single
        feature, or ``(N, M)`` for ``M`` stacked features (e.g., genes) on the
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
        raise ValueError(f"coords must have shape (N, 2), got {coords.shape}.")
    if values.shape[0] != coords.shape[0]:
        raise ValueError(
            f"values first dim {values.shape[0]} must match coords N={coords.shape[0]}."
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


class NUFFTKernel:
    """
    Translation-invariant spatial kernel over **irregular** 2D coordinates,
    evaluated via a type-1 NUFFT and the FFTKernel eigenvalue spectrum.

    Parallels :class:`quadsv.FFTKernel`, which requires a regular grid.
    ``NUFFTKernel`` lets ``xtKx``-style quadratic forms (the Q-test primitive)
    and matrix-vector products ``Kz`` run in ``O(N log N + K log K)`` on
    arbitrary point sets — a 1,000× speed-up over the dense ``SpatialKernel``
    at N ≳ 10⁴.

    The kernel is diagonalized on a common uniform k-grid shared with an
    internal :class:`FFTKernel`. Quadratic forms use the Parseval identity

    .. math::
       x^T K x = \\frac{1}{L_y L_x} \\sum_k \\lambda(k)\\,|\\hat x_\\mathrm{NUFFT}(k)|^2

    where :math:`\\lambda(k)` is the eigenvalue spectrum precomputed by the
    internal ``FFTKernel`` and :math:`\\hat x_\\mathrm{NUFFT}` is the type-1
    non-uniform FFT of the signal at the user-supplied coordinates. The result
    is exact (up to NUFFT precision ``eps``) on a regular grid and matches the
    dense Euclidean quadratic form to within the usual torus-boundary-condition
    band (~2%) on irregular points.

    Parameters
    ----------
    coords : np.ndarray
        Spot coordinates of shape ``(N, 2)`` in order ``(y, x)``.
    grid_shape : tuple[int, int], optional
        ``(ny, nx)`` of the internal uniform k-grid. If ``None`` (default),
        auto-inferred from ``coords``: the grid is sized to cover the bounding
        box and to resolve the sampling Nyquist set by the median nearest-
        neighbor distance (fully coordinate-driven, kernel-agnostic). Override
        only when you know you need a finer or coarser grid.
    spacing : tuple[float, float], optional
        ``(dy, dx)`` physical spacing per k-grid cell (same unit as ``coords``
        after ``unit_scale``). If ``None`` (default), auto-inferred alongside
        ``grid_shape``. When both are supplied, users are responsible for
        ensuring ``grid_shape * spacing`` covers the coordinate extent.
    method : str, default 'matern'
        Kernel method, forwarded to :class:`FFTKernel`.
    unit_scale : float, default 1.0
        Multiplier applied to ``coords`` so they share the same unit as
        ``spacing`` (e.g., 0.35 if coords are in pixels at 0.35 μm/pixel).
    oversample : float, default 2.0
        Auto-grid oversampling factor above the sampling Nyquist. Only used
        when ``grid_shape`` / ``spacing`` are auto-derived. Larger values give
        a finer k-grid (more accurate, slower); 2.0 is safe for all tested
        kernels.
    eps : float, default 1e-6
        NUFFT tolerance.
    workers : int, optional
        Not currently used (reserved for future finufft parallelism).
    **kwargs
        Kernel-method-specific parameters (``bandwidth``, ``nu``, ``rho``,
        ``neighbor_degree``) forwarded to :class:`FFTKernel`.

    Attributes
    ----------
    coords : np.ndarray
        Original coordinates.
    n : int
        Number of spots.
    grid_shape : tuple[int, int]
        Internal k-grid shape.
    spacing : tuple[float, float]
        Physical spacing per k-grid cell.
    method : str
        Kernel method.
    params : dict
        Resolved kernel parameters.
    is_implicit : bool
        Always ``False`` — NUFFTKernel never holds an N×N matrix.
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
        **kwargs,
    ) -> None:
        coords = np.asarray(coords, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"coords must be shape (N, 2), got {coords.shape}.")
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
        self.is_implicit: bool = False

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
            f"grid={self.grid_shape} spacing={self.spacing} params={self.params}>"
        )

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"NUFFTKernel\n"
            f"- Method: {self.method}\n"
            f"- n_spots: {self.n}\n"
            f"- k-grid: {self.grid_shape} at spacing {self.spacing}\n"
            f"- Params: {self.params}"
        )

    # ------------------------------------------------------------------
    def eigenvalues(self, k: int | None = None, return_full: bool = False) -> np.ndarray:
        """Return the ``k`` largest eigenvalues of the underlying spectrum.

        Forwards to the internal :class:`FFTKernel`'s ``eigenvalues``. With
        ``return_full=True``, returns the complete unsorted spectrum in the
        FFT mode layout (useful for Liu's null approximation).
        """
        return self._fft_kernel.eigenvalues(k=k, return_full=return_full)

    # ------------------------------------------------------------------
    def xtKx(self, x: np.ndarray) -> float | np.ndarray:
        """Quadratic form ``x^T K x`` at the kernel's irregular coordinates.

        Parameters
        ----------
        x : np.ndarray
            Signal of shape ``(n,)`` for one feature or ``(n, M)`` for ``M``
            features sharing the same coordinates.

        Returns
        -------
        float or np.ndarray
            Scalar if ``x`` is 1D, shape ``(M,)`` if ``x`` is 2D.

        Raises
        ------
        ValueError
            If ``x`` has the wrong leading dimension.
        """
        if x.shape[0] != self.n:
            raise ValueError(f"x first dim {x.shape[0]} does not match n={self.n}.")
        ny, nx = self.grid_shape
        dy, dx = self.spacing

        power = power_spectrum_2d_nufft(
            coords=self.coords,
            values=x,
            grid_shape=(ny, nx),
            spacing=(dy, dx),
            unit_scale=self._unit_scale,
            eps=self._eps,
            center_coords=True,
        )
        lam = self._fft_kernel.spectrum.reshape(ny, nx)

        if x.ndim == 1:
            return float(np.sum(lam * power) / (ny * nx))
        # Batched: power shape (ny, nx, M), reduce along first two axes.
        weighted = lam[:, :, None] * power
        return (np.sum(weighted, axis=(0, 1)) / (ny * nx)).astype(np.float64)

    # ------------------------------------------------------------------
    def Kx(self, z: np.ndarray) -> np.ndarray:
        """Matrix-vector product ``K x`` at the kernel's irregular coordinates.

        Implemented as type-1 NUFFT → multiply by ``λ(k)`` → type-2 NUFFT.
        Complexity: ``O(N log N + K log K)`` per feature.

        Parameters
        ----------
        z : np.ndarray
            Input shape ``(n,)`` or ``(n, M)``.

        Returns
        -------
        np.ndarray
            ``K z`` of matching shape.
        """
        if z.shape[0] != self.n:
            raise ValueError(f"z first dim {z.shape[0]} does not match n={self.n}.")
        ny, nx = self.grid_shape
        lam_centred = np.fft.fftshift(self._fft_kernel.spectrum.reshape(ny, nx))

        squeeze = z.ndim == 1
        z_in = z[:, None] if squeeze else z
        z_complex = np.ascontiguousarray(z_in.T.astype(np.complex128))  # (M, n)

        z_hat = finufft.nufft2d1(
            self._y_scaled,
            self._x_scaled,
            z_complex,
            n_modes=(ny, nx),
            eps=self._eps,
            isign=-1,
        )  # (M, ny, nx), DC centred
        # Apply spectrum.
        if z_hat.ndim == 2:
            out_k = lam_centred * z_hat / (ny * nx)
        else:
            out_k = lam_centred[None, :, :] * z_hat / (ny * nx)
        out_k = np.ascontiguousarray(out_k.astype(np.complex128))

        Kz = finufft.nufft2d2(
            self._y_scaled,
            self._x_scaled,
            out_k,
            eps=self._eps,
            isign=+1,
        )  # (M, n)
        Kz = np.real(Kz).T  # back to (n, M)
        if squeeze:
            return Kz[:, 0]
        return Kz

    # ------------------------------------------------------------------
    def trace(self) -> float:
        """``trace(K)`` of the effective torus-BC kernel.

        Computed deterministically from the eigenvalue spectrum held by the
        internal :class:`FFTKernel` (no Hutchinson estimation needed).
        """
        return float(self._fft_kernel.trace())

    def square_trace(self) -> float:
        """``trace(K²)`` — deterministic, from the eigenvalue spectrum."""
        return float(self._fft_kernel.square_trace())


# ---------------------------------------------------------------------------
# Q-test and R-test on irregular spatial coordinates via NUFFT
# ---------------------------------------------------------------------------


def _standardize_features(X: np.ndarray) -> np.ndarray:
    """Z-score each column (ddof=1), leaving constant columns as zeros.

    Matches the convention of :func:`quadsv.spatial_q_test_fft`.
    """
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True, ddof=1)
    out = np.zeros_like(X, dtype=float)
    valid = sd > 1e-12
    np.divide(X - mu, sd, out=out, where=valid)
    return out


def spatial_q_test_nufft(  # noqa: C901
    Xn: np.ndarray,
    kernel: NUFFTKernel,
    null_params: dict | None = None,
    return_pval: bool = True,
    is_standardized: bool = False,
) -> float | np.ndarray | tuple[float, float] | tuple[np.ndarray, np.ndarray]:
    """
    Spatial Q-test on irregular 2D coordinates — NUFFT analogue of
    :func:`quadsv.spatial_q_test_fft`.

    Uses the kernel's full eigenvalue spectrum (from the internal
    :class:`FFTKernel`) and Liu's chi-squared-mixture approximation to the
    null distribution of ``Q = z^T K z`` where ``z`` is standardized. For
    Moran's I (which has negative eigenvalues) a normal approximation based on
    ``trace(K)`` and ``trace(K²)`` is used, matching the FFT path exactly.

    Parameters
    ----------
    Xn : np.ndarray
        Signal of shape ``(n,)`` or ``(n, M)``. Each column is Z-score
        standardized internally unless ``is_standardized=True``.
    kernel : NUFFTKernel
        Pre-constructed NUFFT kernel.
    null_params : dict, optional
        Pre-computed null parameters from
        :func:`quadsv.statistics.compute_null_params`. When supplied, the
        cached ``eigenvalues`` / ``mean_Q`` / ``var_Q`` entries are reused
        so the spectrum does not need to be re-fetched per feature. Note
        that the cached entries are assumed to have **already been
        rescaled** to the N-point operator (i.e., multiplied by
        ``N / (ny * nx)``) — the on-the-fly path below does this
        rescaling internally for callers that pass ``None``.
    return_pval : bool, default True
        If True, return ``(Q, pval)`` tuple; else just ``Q``.
    is_standardized : bool, default False
        If True, skip the internal standardization.

    Returns
    -------
    Q : float or np.ndarray
        Test statistic. Scalar if input was 1D, shape ``(M,)`` otherwise.
    pval : float or np.ndarray, optional
        Tail probability under H₀ — Liu's method for most kernels, normal
        approximation for Moran's I. Returned only when ``return_pval=True``.

    Raises
    ------
    ValueError
        If ``Xn``'s first dimension does not match ``kernel.n``.

    Examples
    --------
    >>> import numpy as np
    >>> from quadsv import NUFFTKernel, spatial_q_test_nufft
    >>> rng = np.random.default_rng(0)
    >>> coords = rng.uniform(0, 20, size=(400, 2))
    >>> kernel = NUFFTKernel(coords, method="matern", bandwidth=2.0, nu=1.5)
    >>> z = rng.standard_normal(400)
    >>> Q, pval = spatial_q_test_nufft(z, kernel)
    >>> 0.0 <= pval <= 1.0
    True
    """
    Xn = np.asarray(Xn, dtype=float)
    if Xn.ndim == 1:
        batched = False
        X_in = Xn[:, None]
    elif Xn.ndim == 2:
        batched = True
        X_in = Xn
    else:
        raise ValueError(f"Xn must be 1D or 2D, got shape {Xn.shape}.")
    if X_in.shape[0] != kernel.n:
        raise ValueError(f"Xn first dim {X_in.shape[0]} does not match kernel.n={kernel.n}.")

    z = X_in if is_standardized else _standardize_features(X_in)
    Q_arr = np.atleast_1d(kernel.xtKx(z)).ravel()

    if not return_pval:
        return float(Q_arr[0]) if not batched else Q_arr

    # The internal FFTKernel's spectrum sums to ny*nx (its own grid trace); the
    # NUFFT Q on N irregular points targets an effective N×N operator whose
    # trace is N * k(0) ≈ N. Rescale eigenvalues by N/(ny*nx) so both Liu's
    # mixture and the Moran normal approximation see the right moments.
    scale = kernel.n / (kernel.grid_shape[0] * kernel.grid_shape[1])

    if kernel.method == "moran":
        if null_params is not None and "mean_Q" in null_params and "var_Q" in null_params:
            mean_Q = float(null_params["mean_Q"])
            var_Q = float(null_params["var_Q"])
        else:
            mean_Q = kernel.trace() * scale
            var_Q = 2.0 * kernel.square_trace() * (scale**2)
        sigma = float(np.sqrt(var_Q))
        if sigma <= 1e-12:
            pvals = np.ones_like(Q_arr)
        else:
            z_scores = (Q_arr - mean_Q) / sigma
            pvals = chi2.sf(z_scores**2, df=1)
    else:
        if null_params is not None and "eigenvalues" in null_params:
            sig_evals = np.asarray(null_params["eigenvalues"], dtype=float)
        else:
            evals = kernel.eigenvalues(return_full=True)
            if evals.min() < -0.1:
                raise ValueError(
                    "Kernel has significant negative eigenvalues; Liu's method may be invalid."
                )
            sig_evals = evals[evals > 1e-9] * scale
        pvals = np.array([liu_sf(float(q), sig_evals) for q in Q_arr])

    if batched:
        return Q_arr, pvals
    return float(Q_arr[0]), float(pvals[0])


def spatial_r_test_nufft(
    Xn: np.ndarray,
    Yn: np.ndarray,
    kernel: NUFFTKernel,
    null_params: dict | None = None,
    return_pval: bool = True,
    is_standardized: bool = False,
) -> float | np.ndarray | tuple[float, float] | tuple[np.ndarray, np.ndarray]:
    """
    Spatial R-test on irregular 2D coordinates — NUFFT analogue of
    :func:`quadsv.spatial_r_test_fft`.

    Computes ``R = x^T K y`` and returns a two-sided normal-approximation
    p-value with null variance ``var_R = trace(K²)``.

    Parameters
    ----------
    Xn, Yn : np.ndarray
        Signals of shape ``(n,)`` or ``(n, M)``. When both are 2D they must
        share the same ``M`` (paired columns); the output then has shape
        ``(M,)``. When shapes differ, returns an ``(M_x, M_y)`` pair matrix.
    kernel : NUFFTKernel
        Pre-constructed NUFFT kernel.
    null_params : dict, optional
        Pre-computed null parameters from
        :func:`quadsv.statistics.compute_null_params`. Only the
        ``var_R`` entry is consumed here; it is expected to already be
        rescaled to the N-point operator (``trace(K²) * (N/(ny*nx))²``).
        If None, this rescaling is done internally from
        ``kernel.square_trace()``.
    return_pval : bool, default True
        If True, return ``(R, pval)``; else just ``R``.
    is_standardized : bool, default False
        If True, skip the internal standardization.

    Returns
    -------
    R : float or np.ndarray
    pval : float or np.ndarray, optional
        Two-sided p-values under the normal approximation.

    Raises
    ------
    ValueError
        If the leading dimensions of ``Xn`` or ``Yn`` don't match ``kernel.n``.

    Examples
    --------
    >>> import numpy as np
    >>> from quadsv import NUFFTKernel, spatial_r_test_nufft
    >>> rng = np.random.default_rng(0)
    >>> coords = rng.uniform(0, 20, size=(400, 2))
    >>> kernel = NUFFTKernel(coords, method="matern", bandwidth=2.0, nu=1.5)
    >>> x = rng.standard_normal(400)
    >>> y = rng.standard_normal(400)
    >>> R, pval = spatial_r_test_nufft(x, y, kernel)
    >>> 0.0 <= pval <= 1.0
    True
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

    KY = kernel.Kx(Yz)  # (n, M_y)
    R = Xz.T @ KY  # (M_x, M_y)

    if not return_pval:
        return R.squeeze() if R.size > 1 else float(R)

    # Rescale trace(K²) to the N-point effective operator, matching the
    # eigenvalue rescaling used by spatial_q_test_nufft.
    if null_params is not None and "var_R" in null_params:
        var_R = float(null_params["var_R"])
    else:
        scale = kernel.n / (kernel.grid_shape[0] * kernel.grid_shape[1])
        var_R = float(kernel.square_trace()) * (scale**2)
    sigma = float(np.sqrt(max(var_R, 1e-30)))
    z_scores = R / sigma
    pvals = 2.0 * norm.sf(np.abs(z_scores))
    return R.squeeze(), pvals.squeeze()
