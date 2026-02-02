from typing import Any

import numpy as np
import scipy.fft
import scipy.sparse as sp
from scipy.special import gamma, kv
from scipy.stats import chi2, norm

from quadsv.statistics import liu_sf


class FFTKernel:
    """
    FFT-accelerated spatial kernel for dense grid data.

    Operates on evenly-spaced grid data (Raster data) with spectral decomposition
    via FFT with periodic (Torus) boundary conditions.

    Attributes
    ----------
    ny, nx : int
        Grid dimensions.
    topology : {'square', 'hex'}
        Grid topology.
    method : str
        Kernel method.
    params : dict
        Kernel parameters (bandwidth, nu, neighbor_degree, rho).
    spectrum : np.ndarray
        Precomputed eigenvalues of the kernel matrix. Shape (n_rfft,) where n_rfft = ny * (nx//2 + 1).
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
        self.ny, self.nx = shape
        self.dy, self.dx = spacing
        self.n_grid = self.ny * self.nx

        # FFT solver selection
        assert fft_solver in ["fft2", "rfft2"], "fft_solver must be 'fft2' or 'rfft2'."
        self.fft_solver = fft_solver
        if fft_solver == "fft2":
            self.n_rfft = self.ny * self.nx  # Full spectrum
        else:
            self.n_rfft = self.ny * (self.nx // 2 + 1)  # rfft2 spectrum size

        # Sanity Checks
        assert topology in ["square", "hex"], "topology must be 'square' or 'hex'."
        assert (
            method in self._available_kernels
        ), f"method must be one of {self._available_kernels}."

        self.topology = topology
        self.method = method

        # Update kernel parameters from defaults
        params = self._get_default_params(method).copy()
        if kwargs:
            for key, value in kwargs.items():
                if key in params:
                    params[key] = value
                else:
                    raise ValueError(f"Unknown parameter '{key}' for method '{method}'")

        self.params = params
        self.workers = workers

        # 1. Precompute Distances
        # For Periodic: Distances wrap around (min(d, L-d)).
        if self.topology == "hex":
            self.min_dist_sq = self._precompute_hex_torus()
        else:
            self.min_dist_sq = self._precompute_square_dists()

        # 2. Precompute Kernel spectrum
        self.spectrum = self._compute_eigenvalues()

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
            f"- Spacing: ({self.dy}, {self.dx})",
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
        y = np.arange(self.ny) * self.dy
        x = np.arange(self.nx) * self.dx

        # Wrap-around distance for periodic boundaries
        y = np.minimum(y, (self.ny * self.dy) - y)
        x = np.minimum(x, (self.nx * self.dx) - x)

        yy, xx = np.meshgrid(y, x, indexing="ij")
        return yy**2 + xx**2

    def _precompute_hex_torus(self):
        """Calculates torus distances on a hexagonal grid (Visium Standard)."""
        # (Hex implementation remains the same, strictly periodic)
        i = np.arange(self.nx)
        j = np.arange(self.ny)
        ii, jj = np.meshgrid(i, j, indexing="ij")

        y_phys = ii * (np.sqrt(3) / 2.0)
        x_phys = jj + 0.5 * (ii % 2)

        coords_grid = np.stack([x_phys, y_phys], axis=-1)

        width_phys = self.ny
        height_phys = self.nx * (np.sqrt(3) / 2.0)

        P_x = np.array([width_phys, 0])
        P_y = np.array([0, height_phys])

        min_d2 = np.full((self.nx, self.ny), np.inf)

        for k in [-1, 0, 1]:
            for m in [-1, 0, 1]:
                shift = k * P_x + m * P_y
                shifted_coords = coords_grid + shift.reshape(1, 1, 2)
                d2 = np.sum(shifted_coords**2, axis=-1)
                min_d2 = np.minimum(min_d2, d2)

        return min_d2

    def _compute_eigenvalues(self):  # noqa: C901
        """Spectral decomposition of the kernel using fft2 or rfft2.

        Returns eigenvalues in the selected FFT layout.
        """

        # --- Continuous Kernels ---
        if self.method == "gaussian":
            bw = self.params["bandwidth"]
            K_img = np.exp(-0.5 * (self.min_dist_sq / bw**2))
            if self.fft_solver == "fft2":
                spectrum_2d = scipy.fft.fft2(K_img, workers=self.workers)
            else:
                spectrum_2d = scipy.fft.rfft2(K_img, workers=self.workers)
            return np.real(spectrum_2d).ravel()

        elif self.method == "matern":
            bw = self.params["bandwidth"]
            nu = self.params["nu"]
            d = np.sqrt(self.min_dist_sq)
            d[d == 0] = 1e-15
            factor = (np.sqrt(2 * nu) * d) / bw
            K_img = (2 ** (1 - nu) / gamma(nu)) * (factor**nu) * kv(nu, factor)
            K_img[0, 0] = 1.0
            if self.fft_solver == "fft2":
                spectrum_2d = scipy.fft.fft2(K_img, workers=self.workers)
            else:
                spectrum_2d = scipy.fft.rfft2(K_img, workers=self.workers)
            return np.real(spectrum_2d).ravel()

        # --- Graph-based Kernels (Moran / Graph Laplacian / CAR) ---
        elif self.method in ["moran", "graph_laplacian", "car"]:
            degree_order = self.params["neighbor_degree"]

            unique_dists = np.unique(self.min_dist_sq)

            if degree_order < len(unique_dists):
                cutoff_sq = unique_dists[degree_order]
            else:
                cutoff_sq = unique_dists[-1]

            # Construct Adjacency Image
            W_img = (self.min_dist_sq <= cutoff_sq).astype(float)
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

        assert ny == self.ny and nx == self.nx

        # Transform using selected FFT solver
        if self.fft_solver == "fft2":
            x_hat = scipy.fft.fft2(x, axes=(0, 1), workers=self.workers)
            x_power = np.abs(x_hat) ** 2
            # Reshape spectrum for full fft2: (ny, nx, 1)
            lam = self.spectrum.reshape(self.ny, self.nx, 1)

            # Weighted Sum (Parseval's Theorem)
            weighted_power = np.sum(x_power * lam, axis=(0, 1))

        else:
            x_hat = scipy.fft.rfft2(x, axes=(0, 1), workers=self.workers)
            # Reshape spectrum for rfft2: (ny, nx//2+1, 1)
            lam = self.spectrum.reshape(self.ny, self.nx // 2 + 1, 1)

            # Weighted Sum (Parseval's Theorem) with correction for rfft2
            power_spectrum = np.abs(x_hat) ** 2 * lam
            weighted_power = 2.0 * np.sum(power_spectrum, axis=(0, 1))

            # Correction: Subtract the first column (fx=0) once
            # because we added it twice in the line above, but it only exists once.
            weighted_power -= np.sum(power_spectrum[:, 0, :], axis=0)

            # Correction: If width is even, the last column is Nyquist (fx=N/2).
            # It is also unique (real-valued in full spectrum), so subtract it once.
            if nx % 2 == 0:
                weighted_power -= np.sum(power_spectrum[:, -1, :], axis=0)

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
        return_full : bool, default False.
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
        """Returns the trace of the squared kernel matrix."""
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
    Uses FFT via Parseval's theorem to compute $Q = \\sum_{i,j} \\lambda_{i,j} Z^2_{i,j}$
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
    assert (
        ny == kernel.ny and nx == kernel.nx
    ), f"Data shape ({ny}, {nx}) does not match kernel ({kernel.ny}, {kernel.nx})"

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

    # For otehr kernels, use Liu's method
    evals = kernel.eigenvalues(return_full=True)
    assert (
        evals.min() >= -0.1
    ), "Kernel has significant negative eigenvalues; Liu's method may be invalid."

    # Filter numerical noise
    sig_evals = evals[evals > 1e-9]

    if np.ndim(Q) == 0:
        pval = liu_sf(Q, sig_evals)
    else:
        pval = np.array([liu_sf(q, sig_evals) for q in Q])

    return Q, pval


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
        Based on Normal approximation: $z = R / \\sqrt{\\text{Trace}(K^2)}$.
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
    Uses FFT via Parseval's theorem: $R = \\frac{1}{N} \\sum_{i,j} \\overline{Z_x}_{i,j} \\lambda_{i,j} Z_y_{i,j}$.

    P-value calculation assumes asymptotic Normality with variance estimated from
    kernel trace: $\\text{Var}(R) \\approx \\text{Trace}(K^2) / N^2$.
    Returns two-tailed probability: $p = 2 P(|Z| > |z\\text{-score}|)$.

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
    assert Xn.shape == Yn.shape, "Xn and Yn shapes must match"
    assert ny == kernel.ny and nx == kernel.nx

    # 1. Standardization
    if is_standardized:
        Zx, Zy = Xn, Yn
    else:
        # X
        mx = np.mean(Xn, axis=(0, 1), keepdims=True)
        sx = np.std(Xn, axis=(0, 1), keepdims=True, ddof=1)
        Zx = np.zeros_like(Xn)
        np.divide(Xn - mx, sx, out=Zx, where=(sx > 1e-12))

        # Y
        my = np.mean(Yn, axis=(0, 1), keepdims=True)
        sy = np.std(Yn, axis=(0, 1), keepdims=True, ddof=1)
        Zy = np.zeros_like(Yn)
        np.divide(Yn - my, sy, out=Zy, where=(sy > 1e-12))

    # 2. Compute R = Zx^T K Zy via FFT
    # R = (1/N) * sum( conj(Zx_hat) * Spectrum * Zy_hat )
    if kernel.fft_solver == "fft2":
        Zx_hat = scipy.fft.fft2(Zx, axes=(0, 1), workers=kernel.workers)
        Zy_hat = scipy.fft.fft2(Zy, axes=(0, 1), workers=kernel.workers)
        lam = kernel.eigenvalues().reshape(ny, nx, 1)

        # Compute spectral product and sum
        spectral_prod = np.real(np.conj(Zx_hat) * lam * Zy_hat)
        R_sum = np.sum(spectral_prod, axis=(0, 1))
    else:
        # rfft2 case with symmetry correction
        Zx_hat = scipy.fft.rfft2(Zx, axes=(0, 1), workers=kernel.workers)
        Zy_hat = scipy.fft.rfft2(Zy, axes=(0, 1), workers=kernel.workers)
        lam = kernel.eigenvalues().reshape(ny, nx // 2 + 1, 1)

        spectral_prod = np.real(np.conj(Zx_hat) * lam * Zy_hat)

        # --- Symmetry Correction for rfft2 ---
        # Sum over frequency axes (0, 1)
        R_sum = 2.0 * np.sum(spectral_prod, axis=(0, 1))

        # Subtract DC component (freq index 0) once
        R_sum -= np.sum(spectral_prod[:, 0, :], axis=0)

        # If width is even, subtract Nyquist component (last freq index) once
        if nx % 2 == 0:
            R_sum -= np.sum(spectral_prod[:, -1, :], axis=0)

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
