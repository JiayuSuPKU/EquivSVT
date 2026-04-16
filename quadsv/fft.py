from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import scipy.fft
import scipy.sparse as sp
from scipy.special import gamma, kv
from scipy.stats import chi2, norm

from quadsv.statistics import liu_sf

__all__ = ["FFTKernel", "power_spectrum_2d", "spatial_q_test_fft", "spatial_r_test_fft"]


def power_spectrum_2d(
    x: np.ndarray,
    fft_solver: str = "fft2",
    workers: int | None = None,
) -> np.ndarray:
    """
    Compute the 2D power spectrum :math:`|\\hat{x}(k)|^2` of one or more grid signals.

    The result is *translation-invariant*: shifting the input image leaves the power
    spectrum unchanged. This makes the spectrum a natural alignment-free representation
    of a spatial pattern. Use :func:`quadsv.spectral_compare.radial_bin_spectrum` to
    further reduce the 2D spectrum to a 1D radial-binned vector that is also
    rotation-invariant.

    Parameters
    ----------
    x : np.ndarray
        Grid signal of shape ``(ny, nx)`` for a single feature, or ``(ny, nx, M)``
        for ``M`` stacked features sharing the grid.
    fft_solver : {'fft2', 'rfft2'}, default 'fft2'
        FFT routine. ``'rfft2'`` returns the half-spectrum of shape
        ``(ny, nx // 2 + 1)`` and roughly halves memory.
    workers : int, optional
        Number of parallel workers forwarded to :mod:`scipy.fft`. ``None`` uses the
        SciPy default.

    Returns
    -------
    np.ndarray
        Power spectrum. Shape ``(ny, n_kx)`` if input was 2D, or ``(ny, n_kx, M)``
        if input was 3D, where ``n_kx = nx`` for ``fft2`` and ``nx // 2 + 1`` for
        ``rfft2``. Layout matches the corresponding :mod:`scipy.fft` routine
        (zero-frequency bin at ``[0, 0]``, no fftshift applied).

    Raises
    ------
    ValueError
        If ``fft_solver`` is not one of ``'fft2'`` or ``'rfft2'``.

    Examples
    --------
    >>> img = np.random.randn(32, 32)
    >>> P = power_spectrum_2d(img, fft_solver='rfft2')
    >>> P.shape
    (32, 17)
    """
    if fft_solver not in ("fft2", "rfft2"):
        raise ValueError(f"fft_solver must be 'fft2' or 'rfft2', got '{fft_solver}'")

    squeeze = x.ndim == 2
    if squeeze:
        x = x[..., np.newaxis]

    if fft_solver == "fft2":
        x_hat = scipy.fft.fft2(x, axes=(0, 1), workers=workers)
    else:
        x_hat = scipy.fft.rfft2(x, axes=(0, 1), workers=workers)

    power = np.abs(x_hat) ** 2

    if squeeze:
        power = power[..., 0]
    return power


class FFTKernel:
    """
    FFT-accelerated spatial kernel for dense grid data.

    Operates on evenly-spaced grid data (raster data) with spectral decomposition
    via FFT under periodic (torus) boundary conditions.

    Attributes
    ----------
    ny, nx : int
        Grid dimensions (number of rows and columns).
    n_grid : int
        Total number of grid points (``ny * nx``).
    topology : {'square', 'hex'}
        Grid topology. ``'hex'`` mirrors 10x Visium hexagonal layouts.
    method : str
        Kernel method (``'gaussian'``, ``'matern'``, ``'moran'``, ``'graph_laplacian'``,
        ``'car'``).
    params : dict
        Resolved kernel parameters (e.g. ``bandwidth``, ``nu``, ``neighbor_degree``,
        ``rho``) after defaults are merged with user overrides.
    fft_solver : {'fft2', 'rfft2'}
        FFT routine in use. ``'rfft2'`` stores roughly half the spectrum.
    n_rfft : int
        Length of the flattened spectrum: ``ny * nx`` for ``fft2`` and
        ``ny * (nx // 2 + 1)`` for ``rfft2``.
    workers : int or None
        Number of parallel workers forwarded to :mod:`scipy.fft`.
    spectrum : np.ndarray
        Flattened (row-major) eigenvalues of the kernel matrix, shape ``(n_rfft,)``.
        Eagerly computed in ``__init__``. See :meth:`eigenvalues` for a sorted /
        full-FFT-layout accessor.
    """

    _available_kernels = ["gaussian", "matern", "moran", "graph_laplacian", "car"]

    def __init__(
        self,
        shape: tuple[int, int],
        spacing: tuple[float, float] = (1.0, 1.0),
        topology: str = "square",
        method: str = "matern",
        workers: int | None = None,
        fft_solver: str = "fft2",
        **kwargs,
    ) -> None:
        """
        Initialize FFT-accelerated spatial kernel for grid data.

        Parameters
        ----------
        shape : tuple of int
            Grid dimensions (ny, nx).
        spacing : tuple of float, default (1.0, 1.0)
            Physical distance between pixels (dy, dx).
        topology : {'square', 'hex'}, default 'square'
            Grid topology. 'hex' is for Visium-like hexagonal layouts.
        method : str, default 'matern'
            Kernel method: 'gaussian', 'matern', 'moran', 'graph_laplacian', 'car'.
        workers : Optional[int], default None
            Number of parallel workers for fft computations.
        fft_solver : {'fft2', 'rfft2'}, default 'fft2'
            FFT solver to use. 'fft2' (full FFT) or 'rfft2' (real FFT, ~50% memory).
            Default is 'fft2' for better compatibility and robustness on most architectures.
        **kwargs : dict
            Kernel parameters (bandwidth, nu, neighbor_degree, rho).

        Examples
        --------
        >>> kernel = FFTKernel((64, 64), method='gaussian', bandwidth=2.0)
        >>> kernel = FFTKernel((64, 64), topology='hex', method='matern')
        """
        ny, nx = shape
        if ny < 2 or nx < 2:
            raise ValueError(f"Grid dimensions must be >= 2, got ({ny}, {nx})")
        self.ny: int = ny
        """Number of grid rows."""
        self.nx: int = nx
        """Number of grid columns."""
        self._dy, self._dx = spacing
        self.n_grid: int = self.ny * self.nx
        """Total number of grid points (``ny * nx``)."""

        # FFT solver selection
        if fft_solver not in ("fft2", "rfft2"):
            raise ValueError(f"fft_solver must be 'fft2' or 'rfft2', got '{fft_solver}'")
        self.fft_solver: str = fft_solver
        """FFT routine in use (``'fft2'`` or ``'rfft2'``)."""
        self.n_rfft: int = (
            self.ny * self.nx if fft_solver == "fft2" else self.ny * (self.nx // 2 + 1)
        )
        """Length of the flattened spectrum buffer (``ny*nx`` for ``fft2``, ``ny*(nx//2+1)`` for ``rfft2``)."""

        # Sanity Checks
        if topology not in ("square", "hex"):
            raise ValueError(f"topology must be 'square' or 'hex', got '{topology}'")
        if method not in self._available_kernels:
            raise ValueError(f"method must be one of {self._available_kernels}, got '{method}'")

        self.topology: str = topology
        """Grid topology (``'square'`` or ``'hex'``)."""
        self.method: str = method
        """Kernel method name."""

        # Update kernel parameters from defaults
        params = self._get_default_params(method).copy()
        if kwargs:
            for key, value in kwargs.items():
                if key in params:
                    params[key] = value
                else:
                    raise ValueError(f"Unknown parameter '{key}' for method '{method}'")

        self.params: dict = params
        """Resolved kernel parameters after defaults are merged with user overrides."""
        self.workers: int | None = workers
        """Number of parallel workers forwarded to :mod:`scipy.fft`, or ``None`` for the library default."""

        # 1. Precompute Distances
        # For Periodic: Distances wrap around (min(d, L-d)).
        if self.topology == "hex":
            self._min_dist_sq = self._precompute_hex_torus()
        else:
            self._min_dist_sq = self._precompute_square_dists()

        # 2. Precompute Kernel spectrum
        self.spectrum: np.ndarray = self._compute_eigenvalues()
        """Flattened (row-major) eigenvalues of the kernel matrix, shape ``(n_rfft,)``."""

    def _format_params(self) -> str:
        """Format kernel params safely without dumping large arrays/matrices."""
        if not self.params:
            return "None"
        parts = []
        for k, v in self.params.items():
            try:
                if isinstance(v, np.ndarray):
                    parts.append(f"{k}=array(shape={v.shape}, dtype={v.dtype})")
                elif sp.issparse(v):
                    parts.append(f"{k}=sparse(shape={v.shape}, nnz={v.nnz})")
                else:
                    parts.append(f"{k}={v}")
            except Exception:
                parts.append(f"{k}=?")
        return ", ".join(parts)

    def __repr__(self) -> str:
        """
        Return a detailed, machine-readable representation of the FFTKernel.

        Returns
        -------
        str
            String representation in angle-bracket format.
        """
        spectrum_info = (
            f"spectrum shape={self.spectrum.shape}"
            if self.spectrum is not None
            else "spectrum=None"
        )
        return (
            f"<FFTKernel method={self.method} shape=({self.ny}, {self.nx}) topology={self.topology} "
            f"fft_solver={self.fft_solver} {spectrum_info} params={{ {self._format_params()} }}>"
        )

    def __str__(self) -> str:
        """
        Return a human-friendly, multi-line representation of the FFTKernel.

        Returns
        -------
        str
            Multi-line string summary.
        """
        lines = [
            "FFTKernel",
            f"- Method: {self.method}",
            f"- Grid shape: ({self.ny}, {self.nx})",
            f"- Topology: {self.topology}",
            f"- Spacing: ({self._dy}, {self._dx})",
            f"- FFT Solver: {self.fft_solver}",
        ]

        if self.spectrum is not None:
            lines.append(
                f"- Spectrum: shape={self.spectrum.shape}, min={np.min(self.spectrum):.4g}, max={np.max(self.spectrum):.4g}"
            )
        else:
            lines.append("- Spectrum: None")

        lines.append(f"- Params: {self._format_params()}")
        return "\n".join(lines)

    def _get_default_params(self, method: str) -> dict[str, Any]:
        """
        Returns default parameters for specific kernel methods.

        Parameters
        ----------
        method : str
            Kernel method name. Should be one of _available_kernels.

        Returns
        -------
        dict[str, Any]
            Method defaults: bandwidth (gaussian/matern), nu (matern), neighbor_degree (moran/graph_laplacian/car), rho (car).
        """
        method_defaults = {
            "gaussian": {"bandwidth": 2.0},
            "matern": {"nu": 1.5, "bandwidth": 2.0},
            "moran": {"neighbor_degree": 1},
            "graph_laplacian": {"neighbor_degree": 1},
            "car": {"rho": 0.9, "neighbor_degree": 1},
        }
        return method_defaults.get(method, {})

    def _precompute_square_dists(self):
        """Computes wrap-around torus distances from (0,0) to (y,x)."""
        y = np.arange(self.ny) * self._dy
        x = np.arange(self.nx) * self._dx

        # Wrap-around distance for periodic boundaries
        y = np.minimum(y, (self.ny * self._dy) - y)
        x = np.minimum(x, (self.nx * self._dx) - x)

        yy, xx = np.meshgrid(y, x, indexing="ij")
        return yy**2 + xx**2

    def _precompute_hex_torus(self):
        """Squared torus distances on a hexagonal grid (Visium convention).

        Spot ``(r, c)`` lies at physical ``(y, x) = (r * sqrt(3)/2, c + 0.5 * (r%2))``
        in units of the center-to-center horizontal step — i.e., odd rows are shifted
        half a step in +x, matching the 10x Visium ``array_row`` / ``array_col``
        layout.

        Returns an ``(ny, nx)`` array consistent with the ``(ny, nx)`` signal shape
        expected by :meth:`xtKx`. (The previous implementation returned ``(nx, ny)``
        and scrambled the spectrum for non-square grids, silently breaking anisotropic
        signals on any real Visium slide — Visium is never square. Tests covered only
        square hex grids so the bug was invisible. Fixed.)

        Periodicity in the y direction is well-defined only when ``ny`` is even (so
        the row-parity shift is preserved under wrap-around); callers feeding odd
        ``ny`` will get a near-periodic but slightly off torus and a warning is
        emitted.
        """
        if self.ny % 2 != 0:
            warnings.warn(
                f"Hex topology expects an even number of rows (ny); got ny={self.ny}. "
                "Periodic boundary conditions are approximate for odd ny.",
                UserWarning,
                stacklevel=2,
            )
        r = np.arange(self.ny)  # row index, first (ny) axis
        c = np.arange(self.nx)  # col index, second (nx) axis
        rr, cc = np.meshgrid(r, c, indexing="ij")  # both shape (ny, nx)

        y_phys = rr * (np.sqrt(3) / 2.0)
        x_phys = cc + 0.5 * (rr % 2)
        coords_grid = np.stack([y_phys, x_phys], axis=-1)  # (ny, nx, 2)

        # Torus periods: width in x is nx, height in y is ny * sqrt(3)/2.
        P_y = np.array([self.ny * (np.sqrt(3) / 2.0), 0.0])
        P_x = np.array([0.0, float(self.nx)])

        min_d2 = np.full((self.ny, self.nx), np.inf)
        for k in (-1, 0, 1):
            for m in (-1, 0, 1):
                shift = k * P_y + m * P_x
                shifted = coords_grid + shift.reshape(1, 1, 2)
                d2 = np.sum(shifted**2, axis=-1)
                min_d2 = np.minimum(min_d2, d2)
        return min_d2

    def _compute_eigenvalues(self):  # noqa: C901
        """Spectral decomposition of the kernel using fft2 or rfft2.

        Returns eigenvalues in the selected FFT layout.
        """

        # --- Continuous Kernels ---
        if self.method == "gaussian":
            bw = self.params["bandwidth"]
            K_img = np.exp(-0.5 * (self._min_dist_sq / bw**2))
            if self.fft_solver == "fft2":
                spectrum_2d = scipy.fft.fft2(K_img, workers=self.workers)
            else:
                spectrum_2d = scipy.fft.rfft2(K_img, workers=self.workers)
            return np.real(spectrum_2d).ravel()

        elif self.method == "matern":
            bw = self.params["bandwidth"]
            nu = self.params["nu"]
            d = np.sqrt(self._min_dist_sq)
            mask_zero = d == 0
            d[mask_zero] = 1.0  # dummy value, overwritten below
            factor = (np.sqrt(2 * nu) * d) / bw
            K_img = (2 ** (1 - nu) / gamma(nu)) * (factor**nu) * kv(nu, factor)
            K_img[mask_zero] = 1.0  # correct limit: K(x, x) = 1
            if self.fft_solver == "fft2":
                spectrum_2d = scipy.fft.fft2(K_img, workers=self.workers)
            else:
                spectrum_2d = scipy.fft.rfft2(K_img, workers=self.workers)
            return np.real(spectrum_2d).ravel()

        # --- Graph-based Kernels (Moran / Graph Laplacian / CAR) ---
        elif self.method in ["moran", "graph_laplacian", "car"]:
            degree_order = self.params["neighbor_degree"]

            unique_dists = np.unique(self._min_dist_sq)

            if degree_order < len(unique_dists):
                cutoff_sq = unique_dists[degree_order]
            else:
                cutoff_sq = unique_dists[-1]

            # Construct Adjacency Image
            W_img = (self._min_dist_sq <= cutoff_sq).astype(float)
            W_img[0, 0] = 0.0

            # Row-Normalization Factor
            # For Periodic: Exact constant degree.
            degree = np.sum(W_img)

            if degree == 0:
                return np.ones(self.n_grid)

            # Compute Spectrum of Normalized W
            if self.fft_solver == "fft2":
                spectrum_2d = scipy.fft.fft2(W_img, workers=self.workers)
            else:
                spectrum_2d = scipy.fft.rfft2(W_img, workers=self.workers)
            lam_W = np.real(spectrum_2d).ravel() / degree

            if self.method == "moran":
                return lam_W
            elif self.method == "graph_laplacian":
                return 1.0 - lam_W
            elif self.method == "car":
                rho = self.params["rho"]
                # Cap rho to prevent singularity if rho is too close to 1
                if rho >= 1.0:
                    warnings.warn(
                        f"rho={rho} >= 1.0 causes singularity in CAR kernel; clamping to 0.99",
                        UserWarning,
                        stacklevel=2,
                    )
                    rho = 0.99
                return 1.0 / (1.0 - rho * lam_W)

        else:
            raise ValueError("Unknown method")

    def xtKx(self, x: np.ndarray) -> float | np.ndarray:
        """
        Compute the quadratic form x^T K x efficiently using FFT.

        Uses Parseval's theorem to compute the result in frequency domain
        for O(N log N) complexity instead of O(N²).

        Parameters
        ----------
        x : np.ndarray
            Input data tensor. Shape (ny, nx) for single feature or (ny, nx, M) for M features.

        Returns
        -------
        float or np.ndarray
            Quadratic form value(s). Scalar if input was 2D, shape (M,) if input was 3D.
        """
        if x.ndim == 2:
            x = x[..., np.newaxis]

        ny, nx, M = x.shape

        if ny != self.ny or nx != self.nx:
            raise ValueError(
                f"Data shape ({ny}, {nx}) does not match kernel ({self.ny}, {self.nx})"
            )

        # Transform using selected FFT solver via the shared power-spectrum helper.
        x_power = power_spectrum_2d(x, fft_solver=self.fft_solver, workers=self.workers)

        if self.fft_solver == "fft2":
            # Reshape spectrum for full fft2: (ny, nx, 1)
            lam = self.spectrum.reshape(self.ny, self.nx, 1)

            # Weighted Sum (Parseval's Theorem)
            weighted_power = np.sum(x_power * lam, axis=(0, 1))

        else:
            # Reshape spectrum for rfft2: (ny, nx//2+1, 1)
            lam = self.spectrum.reshape(self.ny, self.nx // 2 + 1, 1)

            # Weighted Sum (Parseval's Theorem) with correction for rfft2
            weighted = x_power * lam
            weighted_power = 2.0 * np.sum(weighted, axis=(0, 1))

            # Correction: Subtract the first column (fx=0) once
            # because we added it twice in the line above, but it only exists once.
            weighted_power -= np.sum(weighted[:, 0, :], axis=0)

            # Correction: If width is even, the last column is Nyquist (fx=N/2).
            # It is also unique (real-valued in full spectrum), so subtract it once.
            if nx % 2 == 0:
                weighted_power -= np.sum(weighted[:, -1, :], axis=0)

        # FFT is unnormalized: Parseval requires 1/N normalization
        Q = weighted_power / (ny * nx)

        # Unwrap if M=1
        return Q.item() if M == 1 else Q.ravel()

    def eigenvalues(self, k: int | None = None, return_full: bool = False) -> np.ndarray:
        """
        Get the eigenvalues of the kernel matrix.

        Parameters
        ----------
        k : int, optional
            Number of largest eigenvalues to return. If None, returns all.
        return_full : bool, default False
            Only for fft_solver='rfft2'.
            If True, returns eigenvalues in full FFT layout (ny, nx) flattened.

        Returns
        -------
        np.ndarray
            Eigenvalues. If return_full=True, shape is (ny * nx,).
                If k specified, returns top-k in descending order.

        Notes
        -----
        If fft_solver='rfft2', spectrum is stored in rfft2 format, not full FFT format.
        To convert to full FFT format, use return_full=True.
        """
        if self.spectrum is None:
            self.spectrum = self._compute_eigenvalues()

        # Simple logic for fft2 or no full return
        if (self.fft_solver == "fft2") or (not return_full):
            if k is None:
                return self.spectrum
            else:
                idx = np.argsort(-self.spectrum)[:k]
                return self.spectrum[idx]

        else:  # fft_solver == 'rfft2' and return_full=True
            # Convert rfft2 layout to full FFT layout
            full_fft = np.zeros((self.ny, self.nx), dtype=self.spectrum.dtype)
            rfft_size = self.nx // 2 + 1
            full_fft[:, :rfft_size] = self.spectrum.reshape(self.ny, rfft_size)
            # Fill in negative frequencies using Hermitian symmetry
            for i in range(self.ny):
                for j in range(1, rfft_size - 1):
                    full_fft[i, self.nx - j] = full_fft[i, j].conj()

            if k is None:
                return full_fft.ravel()
            else:
                # Return top-k largest eigenvalues
                idx = np.argsort(-full_fft.ravel())[:k]
                return full_fft.ravel()[idx]

    def trace(self) -> float:
        """
        Compute the trace of the kernel matrix.

        Returns
        -------
        float
            Trace of K (sum of eigenvalues).
        """
        return np.sum(self.eigenvalues(return_full=True))

    def square_trace(self) -> float:
        """
        Compute the trace of the squared kernel matrix.

        Returns
        -------
        float
            Trace of K² (sum of squared eigenvalues).
        """
        return np.sum(self.eigenvalues(return_full=True) ** 2)


def spatial_q_test_fft(
    Xn: np.ndarray, kernel: FFTKernel, return_pval: bool = True, is_standardized: bool = False
) -> float | np.ndarray | tuple[float, float] | tuple[np.ndarray, np.ndarray]:
    """
    FFT-accelerated spatial Q-test for grid data.

    Tests whether a spatial variable exhibits significant clustering or dispersion
    using FFT-based spectral decomposition. This function provides fast approximation
    via Parseval's theorem compared to dense kernel methods.

    Parameters
    ----------
    Xn : np.ndarray
        Input data tensor. Shape (ny, nx) for single feature or (ny, nx, M) for M features.
        Order follows kernel dimensions. Will be automatically reshaped to 3D if 2D.
    kernel : FFTKernel
        Pre-constructed FFT kernel object for grid data.
    return_pval : bool, default True
        If True, returns (Q, pval) tuple; if False, returns Q only.
    is_standardized : bool, default False
        If True, skips Z-score standardization internally. Otherwise standardizes
        per-feature (mean 0, std 1) across spatial dimensions.

    Returns
    -------
    Q : float or np.ndarray
        Test statistic. Scalar if input was 2D; array of shape (M,) if 3D.
    pval : float or np.ndarray, optional
        Tail probability under null hypothesis. Only returned if return_pval=True.
        Uses Liu's method for most kernels; Normal approximation for Moran's I.

    Raises
    ------
    AssertionError
        If Xn spatial dimensions don't match kernel shape (ny, nx).

    Notes
    -----
    Under H₀: data is spatially independent.
    Under H₁: mean-shift present.

    Computationally: Q = z^T K z where z is standardized data.
    Uses FFT via Parseval's theorem to compute :math:`Q = \\sum_{i,j} \\lambda_{i,j} Z^2_{i,j}`
    in O(N log N) time instead of O(N³) dense methods.

    For Moran's I kernel (which has negative eigenvalues), uses Normal approximation
    based on asymptotic theory. For other kernels, uses Liu's chi-squared mixture approximation.

    Examples
    --------
    >>> ny, nx = 32, 32
    >>> kernel = FFTKernel((ny, nx), method='gaussian', bandwidth=1.0)
    >>> data = np.random.randn(ny, nx)
    >>> Q, pval = spatial_q_test_fft(data, kernel)
    """
    Xn = np.asarray(Xn).astype(float)
    if Xn.ndim == 2:
        Xn = Xn[..., np.newaxis]

    ny, nx, M = Xn.shape
    if ny != kernel.ny or nx != kernel.nx:
        raise ValueError(
            f"Data shape ({ny}, {nx}) does not match kernel ({kernel.ny}, {kernel.nx})"
        )

    # 1. Standardization (Z-score across spatial dimensions)
    if is_standardized:
        z = Xn
    else:
        # Mean/Std per feature slice
        means = np.mean(Xn, axis=(0, 1), keepdims=True)
        stds = np.std(Xn, axis=(0, 1), keepdims=True, ddof=1)

        # Handle constant features (std=0)
        # Create result array
        z = np.zeros_like(Xn)

        # Mask where std > 0 (shape 1,1,M broadcastable)
        valid = stds > 1e-12

        # Safe division
        np.divide(Xn - means, stds, out=z, where=valid)

    # 2. Compute Q statistic: z^T K z
    # Helper returns (M,) array or scalar if input was 2D
    Q = kernel.xtKx(z)

    if not return_pval:
        return Q

    # 3. P-value approximation

    # Use clt approximation (Normal) for Moran's I since it has negative eigenvalues
    if kernel.method in ["moran"]:
        # Under null, Q ~ N(mean_Q, var_Q)
        mean_Q = kernel.trace()
        var_Q = 2.0 * kernel.square_trace()

        if np.ndim(Q) == 0:
            sigma = np.sqrt(var_Q)
            z_score = (Q - mean_Q) / sigma if sigma > 1e-12 else 0.0
            pval = chi2.sf(z_score**2, df=1)
        else:
            sigma = np.sqrt(var_Q)
            z_scores = (Q - mean_Q) / sigma if sigma > 1e-12 else np.zeros_like(Q)
            pval = chi2.sf(z_scores**2, df=1)

        return Q, pval

    # For other kernels, use Liu's method
    evals = kernel.eigenvalues(return_full=True)
    if evals.min() < -0.1:
        raise ValueError(
            "Kernel has significant negative eigenvalues; Liu's method may be invalid."
        )

    # Filter numerical noise
    sig_evals = evals[evals > 1e-9]

    if np.ndim(Q) == 0:
        pval = liu_sf(Q, sig_evals)
    else:
        pval = np.array([liu_sf(q, sig_evals) for q in Q])

    return Q, pval


def _standardize_grid(X: np.ndarray) -> np.ndarray:
    """Z-score standardize a grid tensor along spatial dims (0, 1)."""
    m = np.mean(X, axis=(0, 1), keepdims=True)
    s = np.std(X, axis=(0, 1), keepdims=True, ddof=1)
    Z = np.zeros_like(X)
    np.divide(X - m, s, out=Z, where=(s > 1e-12))
    return Z


def _spectral_cross_product(
    Zx: np.ndarray, Zy: np.ndarray, kernel: FFTKernel, ny: int, nx: int
) -> np.ndarray:
    """Compute sum of conj(Zx_hat) * lambda * Zy_hat in frequency domain."""
    if kernel.fft_solver == "fft2":
        Zx_hat = scipy.fft.fft2(Zx, axes=(0, 1), workers=kernel.workers)
        Zy_hat = scipy.fft.fft2(Zy, axes=(0, 1), workers=kernel.workers)
        lam = kernel.eigenvalues().reshape(ny, nx, 1)
        spectral_prod = np.real(np.conj(Zx_hat) * lam * Zy_hat)
        return np.sum(spectral_prod, axis=(0, 1))

    # rfft2 case with symmetry correction
    Zx_hat = scipy.fft.rfft2(Zx, axes=(0, 1), workers=kernel.workers)
    Zy_hat = scipy.fft.rfft2(Zy, axes=(0, 1), workers=kernel.workers)
    lam = kernel.eigenvalues().reshape(ny, nx // 2 + 1, 1)
    spectral_prod = np.real(np.conj(Zx_hat) * lam * Zy_hat)
    R_sum = 2.0 * np.sum(spectral_prod, axis=(0, 1))
    R_sum -= np.sum(spectral_prod[:, 0, :], axis=0)
    if nx % 2 == 0:
        R_sum -= np.sum(spectral_prod[:, -1, :], axis=0)
    return R_sum


def spatial_r_test_fft(
    Xn: np.ndarray,
    Yn: np.ndarray,
    kernel: FFTKernel,
    return_pval: bool = True,
    is_standardized: bool = False,
) -> float | np.ndarray | tuple[float, float] | tuple[np.ndarray, np.ndarray]:
    """
    FFT-accelerated spatial R-test (bivariate) for grid data.

    Tests for spatial co-variation between two variables using the specified kernel.
    Computes the cross-variance statistic R = x^T K y via FFT-based spectral methods.

    Parameters
    ----------
    Xn : np.ndarray
        First input tensor. Shape (ny, nx) for single feature or (ny, nx, M) for M features.
        Will be automatically reshaped to 3D if 2D.
    Yn : np.ndarray
        Second input tensor. Must have the same shape as Xn.
    kernel : FFTKernel
        Pre-constructed FFT kernel object for grid data.
    return_pval : bool, default True
        If True, returns (R, pval) tuple; if False, returns R only.
    is_standardized : bool, default False
        If True, skips standardization. Otherwise standardizes each variable
        independently (mean 0, std 1) across spatial dimensions.

    Returns
    -------
    R : float or np.ndarray
        Test statistic (cross-variance). Scalar if input was 2D; array of shape (M,) if 3D.
    pval : float or np.ndarray, optional
        Two-tailed p-value under null hypothesis (no spatial co-variation).
        Based on Normal approximation: :math:`z = R / \\sqrt{\\text{Trace}(K^2)}`.
        Only returned if return_pval=True.

    Raises
    ------
    AssertionError
        If Xn and Yn shapes don't match, or spatial dimensions don't match kernel.

    Notes
    -----
    Under H₀: x and y are spatially independent.
    Under H₁: spatial co-clustering or co-dispersion present.

    Computationally: R = z_x^T K z_y where z_x, z_y are standardized data.
    Uses FFT via Parseval's theorem: :math:`R = \\frac{1}{N} \\sum_{i,j} \\overline{Z_{x}}_{i,j} \\lambda_{i,j} Z_{y_{i,j}}`

    P-value calculation assumes asymptotic Normality with variance estimated from
    kernel trace: :math:`\\text{Var}(R) \\approx \\text{Trace}(K^2) / N^2`.
    Returns two-tailed probability: :math:`p = 2 P(|Z| > |\\text{z-score}|)`.
    Examples
    --------
    >>> ny, nx = 32, 32
    >>> kernel = FFTKernel((ny, nx), method='gaussian', bandwidth=1.0)
    >>> x_data = np.random.randn(ny, nx)
    >>> y_data = np.random.randn(ny, nx)
    >>> R, pval = spatial_r_test_fft(x_data, y_data, kernel)
    """
    Xn = np.asarray(Xn).astype(float)
    Yn = np.asarray(Yn).astype(float)

    if Xn.ndim == 2:
        Xn = Xn[..., np.newaxis]
    if Yn.ndim == 2:
        Yn = Yn[..., np.newaxis]

    ny, nx, M = Xn.shape
    if Xn.shape != Yn.shape:
        raise ValueError(f"Xn and Yn shapes must match, got {Xn.shape} and {Yn.shape}")
    if ny != kernel.ny or nx != kernel.nx:
        raise ValueError(
            f"Data shape ({ny}, {nx}) does not match kernel ({kernel.ny}, {kernel.nx})"
        )

    # 1. Standardization
    if is_standardized:
        Zx, Zy = Xn, Yn
    else:
        Zx = _standardize_grid(Xn)
        Zy = _standardize_grid(Yn)

    # 2. Compute R = Zx^T K Zy via FFT (Parseval's theorem)
    R_sum = _spectral_cross_product(Zx, Zy, kernel, ny, nx)

    # Apply Parseval's 1/N normalization
    n_pixels = ny * nx
    R = R_sum / n_pixels

    # Unwrap if M=1
    if M == 1 and R.size == 1:
        R = R.item()

    if not return_pval:
        return R

    # 3. P-values (Normal Approximation)
    # Assume x, y are effectively N(0, I) white noise.
    # Variance of R is Trace(K^2) = sum(lambda^2)
    sigma = np.sqrt(kernel.square_trace())

    if sigma > 1e-12:
        z_scores = R / sigma
        pval = 2 * norm.sf(np.abs(z_scores))
    else:
        pval = np.ones_like(R) if isinstance(R, np.ndarray) else 1.0

    return R, pval
