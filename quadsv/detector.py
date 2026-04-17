from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from scipy.stats import norm
from tqdm import tqdm

from quadsv.kernels import Kernel, SpatialKernel
from quadsv.nufft import NUFFTKernel, spatial_q_test_nufft, spatial_r_test_nufft
from quadsv.statistics import compute_null_params, spatial_q_test
from quadsv.utils import _apply_bh_correction

__all__ = ["PatternDetector"]

logger = logging.getLogger(__name__)


# helper function for parallel Q-stat computation
def _qstat_worker(
    X_csc: sp.csc_matrix,
    feature_indices: np.ndarray,
    kernel_obj: Kernel,
    null_params: dict[str, float | np.ndarray],
    means: np.ndarray,
    stds: np.ndarray,
    names: list[str],
    return_pval: bool,
    chunk_size: int = 64,
) -> list[dict[str, str | float]]:
    """
    Worker function for parallel Q-statistic computation.

    Processes a batch of features using pre-computed null distribution parameters.
    Handles zero-variance features gracefully.

    Parameters
    ----------
    X_csc : scipy.sparse.csc_matrix
        Sparse feature matrix (N_samples, N_features_batch). Column-compressed format.
    feature_indices : np.ndarray
        Global indices of features in this batch (for looking up means/stds/names).
    kernel_obj : Kernel
        Pre-constructed kernel object (shared across workers).
    null_params : dict
        Pre-computed null distribution parameters from compute_null_params().
        Keys: 'mean_Q', 'var_Q'.
    means : np.ndarray
        Global feature means (shape: N_total_features).
    stds : np.ndarray
        Global feature standard deviations (shape: N_total_features).
    names : List[str]
        Global feature names (length: N_total_features).
    return_pval : bool
        Whether to compute p-values for each feature.
    chunk_size : int, default 64
        Number of features to process in each internal batch for vectorization.

    Returns
    -------
    list[dict[str, str or float]]
        Each dict contains: {'Feature': str, 'Q': float, 'P_value': float, 'Z_score': float}
    """
    results = []

    # Process in chunks (e.g., 64 features at a time)
    BATCH_SIZE = chunk_size
    # n_features is the number of features in THIS worker's chunk
    n_features = len(feature_indices)

    mu = null_params["mean_Q"]
    sigma = np.sqrt(null_params["var_Q"])

    for start in range(0, n_features, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_features)

        # 1. Determine Indices
        # 'local_slice' is for indexing X_csc (which is already a subset/chunk)
        local_slice = slice(start, end)

        # 'batch_global_indices' is for looking up global metadata (means, stds, names)
        batch_global_indices = feature_indices[start:end]

        # 2. Extract Batch: Convert sparse subset to dense (N, batch_size)
        # We index X_csc using LOCAL indices (0..chunk_size), not global IDs
        X_batch = X_csc[:, local_slice].toarray()

        # 3. Standardize Batch using GLOBAL statistics
        b_means = means[batch_global_indices]
        b_stds = stds[batch_global_indices]

        # Prevent division by zero
        valid_mask = b_stds > 1e-9

        Z_batch = np.zeros_like(X_batch)
        # Broadcasting: (N, M) - (M,) / (M,)
        if np.any(valid_mask):
            Z_batch[:, valid_mask] = (X_batch[:, valid_mask] - b_means[valid_mask]) / b_stds[
                valid_mask
            ]

        # 4. Vectorized Q-test
        if return_pval:
            Q_batch, P_batch = spatial_q_test(
                Z_batch, kernel_obj, null_params, return_pval=True, is_standardized=True
            )
        else:
            Q_batch = spatial_q_test(
                Z_batch, kernel_obj, null_params, return_pval=False, is_standardized=True
            )
            P_batch = np.full(Q_batch.shape, np.nan)

        # Ensure outputs are arrays even if batch size is 1
        Q_batch = np.atleast_1d(Q_batch)
        P_batch = np.atleast_1d(P_batch)

        # 5. Compute Z-scores relative to null distribution
        Z_scores = (Q_batch - mu) / sigma if sigma > 0 else np.zeros_like(Q_batch)

        # 6. Pack results
        for k, global_idx in enumerate(batch_global_indices):
            # If the feature had 0 variance, force null results
            if not valid_mask[k]:
                res = {"Feature": names[global_idx], "Q": 0.0, "P_value": 1.0, "Z_score": 0.0}
            else:
                res = {
                    "Feature": names[global_idx],
                    "Q": Q_batch[k],
                    "P_value": P_batch[k],
                    "Z_score": Z_scores[k],
                }
            results.append(res)

    return results


# optimized R-stat worker with pre-computed K@Y
def _rstat_worker_chunked(
    X_csc: sp.csc_matrix,
    y_chunk_indices: np.ndarray,
    x_indices: np.ndarray,
    kernel_obj: Kernel,
    null_params: dict[str, float],
    means: np.ndarray,
    stds: np.ndarray,
    names: list[str],
    return_pval: bool,
) -> list[dict[str, str | float]]:
    """
    Optimized worker for R-statistic computation with pre-computed K@Y.

    Pre-computes K@Y_chunk once, then reuses for all X features paired with Y chunk.
    This avoids redundant K matrix multiplications.

    Parameters
    ----------
    X_csc : scipy.sparse.csc_matrix
        Sparse feature matrix (N_samples, N_features).
    y_chunk_indices : np.ndarray
        Indices of Y features to process in this chunk.
    x_indices : np.ndarray
        Indices of all X features to pair with Y chunk.
    kernel_obj : Kernel
        Pre-constructed kernel object.
    null_params : dict
        Pre-computed null parameters: 'mean_R', 'var_R'.
    means : np.ndarray
        Feature means for standardization.
    stds : np.ndarray
        Feature standard deviations.
    names : List[str]
        Feature names.
    return_pval : bool
        Whether to compute p-values.

    Returns
    -------
    list[dict[str, str or float]]
        Results for all (x, y) pairs in this chunk.
    """
    results = []
    sigma = np.sqrt(null_params["var_R"])

    # 1. Extract and standardize Y chunk.
    # spatial_r_test standardizes both X and Y (subtracts means, divides by
    # stds), which destroys sparsity, so each per-pair-chunk slice has to be
    # dense. This is still only `chunk_size` columns at a time — a small
    # (N, chunk_size) block — not the full AnnData.
    Y_chunk = X_csc[:, y_chunk_indices].toarray()  # (N, n_y)
    y_means = means[y_chunk_indices]
    y_stds = stds[y_chunk_indices]
    y_valid = y_stds > 1e-9

    Zy_chunk = np.zeros_like(Y_chunk)
    if np.any(y_valid):
        Zy_chunk[:, y_valid] = (Y_chunk[:, y_valid] - y_means[y_valid]) / y_stds[y_valid]

    # 2. Pre-compute K @ Zy_chunk (ONCE for entire chunk)
    # Handle implicit vs explicit kernels
    if hasattr(kernel_obj, "is_implicit") and kernel_obj.is_implicit:
        # Implicit solve: M^-1 @ Zy_chunk
        if sp.issparse(kernel_obj._K):
            from scipy.sparse.linalg import splu

            if kernel_obj._lu is None:
                kernel_obj._lu = splu(kernel_obj._K.tocsc())
            KZy_chunk = kernel_obj._lu.solve(Zy_chunk)
        else:
            from scipy.linalg import lu_factor, lu_solve

            if kernel_obj._lu is None:
                kernel_obj._lu = lu_factor(kernel_obj._K)
            KZy_chunk = lu_solve(kernel_obj._lu, Zy_chunk)
    else:
        # Explicit kernel
        KZy_chunk = kernel_obj._K.dot(Zy_chunk)  # (N, n_y)

    # 3. Process all X features against this Y chunk
    # n_x = len(x_indices)
    # n_y = len(y_chunk_indices)

    for x_idx in x_indices:
        # Extract X feature
        x_col = X_csc[:, x_idx].toarray().flatten()  # (N,)

        # Standardize X
        x_mean = means[x_idx]
        x_std = stds[x_idx]

        if x_std > 1e-9:
            zx = (x_col - x_mean) / x_std
        else:
            zx = np.zeros_like(x_col)

        # Compute R for all pairs (x, y_i) in chunk
        # R_i = zx^T @ (K @ zy_i) = sum(zx * (K@zy_i))
        # Broadcast: (N,) * (N, n_y) -> (N, n_y), then sum(axis=0) -> (n_y,)
        R_batch = np.sum(zx[:, np.newaxis] * KZy_chunk, axis=0)  # (n_y,)

        # P-values (Normal approximation)
        if return_pval:
            Z_scores = R_batch / sigma if sigma > 0 else np.zeros_like(R_batch)
            P_batch = 2 * norm.sf(np.abs(Z_scores))
        else:
            Z_scores = np.full_like(R_batch, np.nan)
            P_batch = np.full_like(R_batch, np.nan)

        # Pack results for this X feature
        for y_idx, y_global_idx in enumerate(y_chunk_indices):
            name_x = names[x_idx]
            name_y = names[y_global_idx]

            res = {
                "Feature_1": name_x,
                "Feature_2": name_y,
                "R": R_batch[y_idx],
                "Z_score": Z_scores[y_idx],
                "P_value": P_batch[y_idx],
            }
            results.append(res)

    return results


class PatternDetector:
    """
    Detect spatial or graph-based feature variability using the quadratic-form statistic.

    This class implements univariate (Q-test) and bivariate (R-test) spatial statistics
    for feature screening using kernel-based methods. Supports multiple kernel types
    (Gaussian, Matérn, Moran's I, graph Laplacian, CAR) and integrates with scanpy/AnnData.

    The core test statistic is:

    - Univariate:  :math:`Q = \\mathbf{x}^T \\mathbf{K} \\mathbf{x}` for standardized feature vector :math:`\\mathbf{x}` and kernel :math:`\\mathbf{K}`.
    - Bivariate: :math:`R = \\mathbf{x}^T \\mathbf{K} \\mathbf{y}` for cross-spatial correlation.

    Parameters
    ----------
    adata : scanpy.AnnData
        Annotated data matrix containing gene expression or other features.
    min_cells : int, default 1
        Minimum number of non-zero observations required for a feature to be tested.
    min_cells_frac : float, optional
        If provided, overrides min_cells to be this fraction of total cells.

    Attributes
    ----------
    adata : scanpy.AnnData
        Reference to the input data object.
    n : int
        Number of samples (cells/observations).
    kernel_ : Kernel or None
        Active kernel object (set by build_kernel_* methods).
    kernel_method_ : str or None
        Method used to construct the current kernel.
    kernel_params_ : dict or None
        Parameters used to construct the current kernel.

    See Also
    --------
    :meth:`~quadsv.detector.PatternDetector.compute_qstat`
        Compute univariate spatial statistics (Q-test).
    :meth:`~quadsv.detector.PatternDetector.compute_rstat`
        Compute bivariate spatial statistics (R-test).
    :meth:`~quadsv.detector.PatternDetector.build_kernel_from_coordinates`
        Construct kernel from spatial coordinates.
    :meth:`~quadsv.detector.PatternDetector.build_kernel_from_obsp`
        Construct kernel from precomputed distance/adjacency matrix.

    Examples
    --------
    >>> import anndata
    >>> adata = anndata.read_h5ad('data.h5ad')
    >>> detector = PatternDetector(adata, min_cells=10)
    >>> detector.build_kernel_from_coordinates(adata.obsm['spatial'], method='gaussian', bandwidth=2.0)
    >>> q_results = detector.compute_qstat(source='var', n_jobs=4)
    """

    _available_kernels = ["gaussian", "matern", "moran", "graph_laplacian", "car"]

    def __init__(self, adata: Any, min_cells: int = 1, min_cells_frac: float | None = None) -> None:
        """
        Initialize the PatternDetector with an AnnData object.

        Parameters
        ----------
        adata : scanpy.AnnData
            Annotated data matrix.
        min_cells : int, default 1
            Minimum number of cells with non-zero expression for feature inclusion.
            Clamped to ``[1, n_cells]``: values larger than the number of samples in
            ``adata`` are silently capped at ``n_cells``.
        min_cells_frac : float, optional
            If provided, overrides ``min_cells`` with
            ``max(1, int(min_cells_frac * n_cells))``.
        """
        self.adata = adata
        """Reference to the input :class:`anndata.AnnData` object."""
        self.n: int = adata.shape[0]
        """Number of samples (cells/observations)."""
        if min_cells_frac is not None:
            self.min_cells: int = max(1, int(min_cells_frac * self.n))
        else:
            self.min_cells = min(min_cells, self.n)
        """Minimum number of non-zero observations a feature must have to be tested."""
        self.kernel_: Any | None = None
        """Active kernel object (``SpatialKernel`` for ``backend='matrix'`` or
        ``NUFFTKernel`` for ``backend='nufft'``), set by ``build_kernel*`` methods."""
        self.kernel_method_: str | None = None
        """Name of the kernel method used to build :attr:`kernel_`, or ``None`` until built."""
        self.kernel_params_: dict | None = None
        """Parameters used to build :attr:`kernel_`, or ``None`` until built."""
        self.backend_: str | None = None
        """Which backend built the kernel — ``'matrix'`` (SpatialKernel, dense or
        sparse-implicit) or ``'nufft'`` (NUFFTKernel on irregular points). ``None``
        until :meth:`build_kernel` / :meth:`build_kernel_from_coordinates` /
        :meth:`build_kernel_from_obsp` has been called."""

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
            Method defaults: bandwidth (gaussian/matern), nu (matern), k_neighbors (moran/graph_laplacian/car), rho (car).
        """
        method_defaults = {
            "gaussian": {"bandwidth": 2.0},
            "matern": {"bandwidth": 2.0, "nu": 1.5},
            "moran": {"k_neighbors": 4},
            "graph_laplacian": {"k_neighbors": 4},
            "car": {"rho": 0.9, "k_neighbors": 4, "standardize": False},
        }
        return method_defaults.get(method, {})

    def build_kernel(
        self,
        backend: str = "matrix",
        method: str = "matern",
        coordinates_key: str = "spatial",
        **kernel_params: Any,
    ) -> PatternDetector:
        """
        Build a kernel over ``adata.obsm[coordinates_key]``.

        This is the unified builder — a single entry point that picks the
        right kernel representation for the data. Irregular spots on a
        regular grid work with either backend; irregular spots on an
        arbitrary layout essentially require ``backend='nufft'`` at
        large ``N``.

        Parameters
        ----------
        backend : {'matrix', 'nufft'}, default 'matrix'
            - ``'matrix'`` — builds a :class:`~quadsv.kernels.SpatialKernel`.
              The kernel class decides internally whether to materialize
              the dense ``(N, N)`` kernel or keep the sparse precision
              matrix and solve linear systems on demand (switch at
              ``N > 5000``). The caller does not choose representation;
              the switch is memory-driven.
            - ``'nufft'`` — builds a :class:`~quadsv.nufft.NUFFTKernel`
              that evaluates ``x^T K x`` and ``K x`` via type-1/type-2
              non-uniform FFTs in ``O(N log N + K log K)``. Needs
              ``finufft`` installed. Ideal for ≥ 10⁴ irregular spots.
        method : str, default 'matern'
            Kernel method. One of ``'gaussian'``, ``'matern'``, ``'moran'``,
            ``'graph_laplacian'``, ``'car'``.
        coordinates_key : str, default 'spatial'
            Key in ``adata.obsm`` containing the ``(n_obs, 2)`` spatial
            coordinates.
        **kernel_params
            Kernel parameters forwarded to the underlying kernel class.
            Matrix backend: ``bandwidth``, ``nu``, ``rho``, ``k_neighbors``.
            NUFFT backend: ``bandwidth``, ``nu``, ``rho``, plus
            ``grid_shape``, ``spacing``, ``unit_scale``, ``oversample``,
            ``eps`` (see :class:`NUFFTKernel`). For NUFFT, ``grid_shape``
            and ``spacing`` are auto-inferred from the coordinates alone
            when not supplied.

        Returns
        -------
        PatternDetector
            ``self`` for chaining.

        Raises
        ------
        ValueError
            If ``backend`` is not recognized, if ``method`` is not one of
            the available kernels, or if the backend requires an optional
            dependency that is not installed.
        KeyError
            If ``coordinates_key`` is missing from ``adata.obsm``.
        """
        if backend not in ("matrix", "nufft"):
            raise ValueError(f"backend must be 'matrix' or 'nufft', got '{backend}'.")
        if method not in self._available_kernels:
            raise ValueError(
                f"Method '{method}' not recognized. Must be one of {self._available_kernels}."
            )
        if coordinates_key not in self.adata.obsm:
            raise KeyError(
                f"adata.obsm has no key '{coordinates_key}'; "
                f"available: {list(self.adata.obsm.keys())}."
            )
        coords = np.asarray(self.adata.obsm[coordinates_key], dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(
                f"adata.obsm['{coordinates_key}'] must be (n_obs, 2), got {coords.shape}."
            )
        if coords.shape[0] != self.n:
            raise ValueError(
                f"coords shape {coords.shape} inconsistent with adata.shape=({self.n}, ...)."
            )

        if backend == "matrix":
            self.build_kernel_from_coordinates(coords, method=method, **kernel_params)
        else:
            logger.info("Building %s NUFFTKernel over %d spots...", method, self.n)
            self.kernel_ = NUFFTKernel(coords=coords, method=method, **kernel_params)
            self.kernel_method_ = method
            self.kernel_params_ = dict(self.kernel_.params)
            self.backend_ = "nufft"
        return self

    def build_kernel_from_coordinates(
        self, coords: np.ndarray, method: str = "car", **kernel_params: Any
    ) -> None:
        """
        Builds a spatial kernel from an array of coordinates.

        Constructs distance-based kernel matrix from sample coordinates using
        specified method. Updates ``self.kernel_`` and ``self.kernel_params_``.

        Parameters
        ----------
        coords : np.ndarray
            Coordinate array of shape (n_samples, n_dims). Typically 2D spatial coordinates.
        method : str, default 'car'
            Kernel construction method. Must be one of: 'gaussian', 'matern', 'moran',
            'graph_laplacian', 'car'.
        kernel_params : dict, optional
            Additional kernel-specific parameters (e.g., bandwidth, rho, k_neighbors, nu).

        Raises
        ------
        ValueError
            If method not in available kernels or coordinate shape mismatch.

        Notes
        -----
        Common parameters by method:

        - gaussian: bandwidth (float, default 1.0)
        - matern: bandwidth (float, default 1.0), nu (float, default 1.5)
        - moran: k_neighbors (int, default 10)
        - graph_laplacian: k_neighbors (int, default 10)
        - car: rho (float in [0, 1), default 0.9), k_neighbors (int, default 10)

        Examples
        --------
        >>> coords = np.random.randn(1000, 2)
        >>> detector.build_kernel_from_coordinates(coords, method='gaussian', bandwidth=1.5)
        """
        coords = np.asarray(coords)
        if method not in self._available_kernels:
            raise ValueError(
                f"Method '{method}' not recognized. Must be one of {self._available_kernels}."
            )

        if coords.shape[0] != self.n:
            raise ValueError(
                f"Coordinate shape {coords.shape} does not match adata shape ({self.n},)."
            )

        logger.info("Building %s kernel from coordinates (n_samples=%d)...", method, self.n)

        if not kernel_params:
            # Use default parameters if none provided
            kernel_params_ = self._get_default_params(method).copy()
        else:
            # Update provided params with defaults for missing keys
            kernel_params_ = self._get_default_params(method).copy()
            for key, value in kernel_params.items():
                if key in kernel_params_:
                    kernel_params_[key] = value
                else:
                    raise ValueError(f"Unknown parameter '{key}' for method '{method}'")

        self.kernel_ = SpatialKernel.from_coordinates(coords, method=method, **kernel_params_)
        self.kernel_method_ = method
        self.kernel_params_ = kernel_params_
        self.backend_ = "matrix"

    def build_kernel_from_obsp(  # noqa: C901
        self, key: str, is_distance: bool = False, method: str = "car", **kernel_params: Any
    ) -> None:
        """
        Builds a graph kernel from a precomputed matrix in adata.obsp.

        Performs necessary transformations (e.g., distance to Gaussian weights,
        adjacency to graph Laplacian) before initializing the kernel object.
        Isolated nodes (zero degree) are automatically removed.

        Parameters
        ----------
        key : str
            Key in adata.obsp containing the precomputed matrix.
        is_distance : bool, default False
            If True, matrix is treated as distances. Otherwise treated as adjacency/connectivity.
        method : str, default 'car'
            Kernel construction method. Must be one of: 'gaussian', 'matern', 'moran',
            'graph_laplacian', 'car', or 'precomputed'.
            - 'precomputed': Uses obsp[key] directly as kernel matrix K (no transformation).
            - Other methods: Apply distance/adjacency transformations.
        kernel_params : dict, optional
            Additional kernel-specific parameters (e.g., bandwidth, rho, nu).

        Raises
        ------
        KeyError
            If key not found in adata.obsp.
        ValueError
            If method not recognized or invalid for matrix type (e.g., Gaussian on adjacency).

        Notes
        -----
        For distance matrices (is_distance=True):

        - gaussian: :math:`K = \\exp(-d^2 / (2 \\sigma^2))` where :math:`\\sigma` is bandwidth.
        - matern: Matérn kernel with nu and bandwidth parameters.

        For adjacency matrices (is_distance=False):

        - Symmetric normalization: :math:`W_{norm} = D^{-1/2} W D^{-1/2}`.
        - moran: Uses :math:`W_{norm}` directly as kernel.
        - graph_laplacian: :math:`K = I - W_{norm}`.
        - car: :math:`K = (I - \\rho W_{norm})^{-1}` (implicit kernel).

        Isolated nodes (degree=0) are automatically removed from the data.

        Examples
        --------
        >>> detector.build_kernel_from_obsp('distances', is_distance=True, method='gaussian', bandwidth=5.0)
        """
        if key not in self.adata.obsp:
            raise KeyError(f"Matrix key '{key}' not found in adata.obsp")

        matrix = self.adata.obsp[key]

        if method not in self._available_kernels + ["precomputed"]:
            raise ValueError(
                f"Method '{method}' not recognized. Must be one of {self._available_kernels} or 'precomputed'."
            )

        # If the user says 'precomputed', they imply the matrix IS the kernel K
        if method == "precomputed":
            logger.info("Using obsp['%s'] directly as kernel matrix (n_samples=%d)...", key, self.n)
            self.kernel_ = SpatialKernel.from_matrix(matrix, is_inverse=False)
            self.kernel_params_ = kernel_params
            self.kernel_method_ = method
            self.backend_ = "matrix"
            return

        logger.info(
            "Building %s kernel from obsp['%s'] (is_distance=%s, n_samples=%d)...",
            method,
            key,
            is_distance,
            self.n,
        )

        # Set kernel configuration parameters
        self.kernel_method_ = method
        if not kernel_params:
            # Use default parameters if none provided
            kernel_params_ = self._get_default_params(method).copy()
        else:
            # Update provided params with defaults for missing keys
            kernel_params_ = self._get_default_params(method).copy()
            for key, value in kernel_params.items():
                if key in kernel_params_:
                    kernel_params_[key] = value
                else:
                    raise ValueError(f"Unknown parameter '{key}' for method '{method}'")

        # --- Distance Based Transformations ---
        if is_distance:
            if method == "gaussian":
                bw = kernel_params_["bandwidth"]
                # K = exp(-d^2 / 2bw^2)
                if sp.issparse(matrix):
                    matrix = matrix.toarray()  # Gaussian usually requires dense
                K = np.exp(-(matrix**2) / (2 * bw**2))
                self.kernel_ = SpatialKernel.from_matrix(K, is_inverse=False)
                self.kernel_params_ = {"bandwidth": bw}
            elif method == "matern":
                from scipy.special import gamma, kv

                bw = kernel_params_["bandwidth"]
                nu = kernel_params_["nu"]
                if sp.issparse(matrix):
                    matrix = matrix.toarray()

                dists = matrix.copy()
                dists[dists == 0] = 1e-15
                factor = (np.sqrt(2 * nu) * dists) / bw
                K = (2 ** (1 - nu) / gamma(nu)) * (factor**nu) * kv(nu, factor)
                np.fill_diagonal(K, 1.0)
                self.kernel_ = SpatialKernel.from_matrix(K, is_inverse=False)
                self.kernel_params_ = {"bandwidth": bw, "nu": nu}
            else:
                raise ValueError(f"Method {method} not supported for distance matrices.")

        # --- Connectivity Based Transformations ---
        else:
            # Assume the input matrix is W (adjacency/connectivity)
            W = matrix
            if not sp.issparse(W):
                W = sp.csr_matrix(W)

            # Remove isolated cells (zero-degree nodes)
            row_sums_raw = np.array(W.sum(axis=1)).flatten()
            keep_mask = row_sums_raw > 0
            if not np.all(keep_mask):
                removed = int((~keep_mask).sum())
                logger.info(
                    "Removing %d isolated samples with zero degree from adjacency matrix...",
                    removed,
                )
                W = W[keep_mask][:, keep_mask]
                self.adata = self.adata[keep_mask].copy()
                self.n = W.shape[0]
                self.min_cells = min(self.min_cells, self.n)

            # Symmetrize and symmetric normalization
            W = W.astype(float)
            W_sym = 0.5 * (W + W.T)
            row_sums = np.array(W_sym.sum(axis=1)).flatten()
            row_sums[row_sums == 0] = 1.0
            inv_D_sqrt = sp.diags(1.0 / np.sqrt(row_sums))
            W_norm = inv_D_sqrt @ W_sym @ inv_D_sqrt

            if method == "moran":
                # Already symmetric and normalized
                K = W_norm
                self.kernel_ = SpatialKernel.from_matrix(K, is_inverse=False)
                self.kernel_params_ = {}

            elif method == "graph_laplacian":
                # K = I - W_norm
                I = sp.eye(self.n, format="csr")
                K = I - W_norm
                self.kernel_ = SpatialKernel.from_matrix(K, is_inverse=False)
                self.kernel_params_ = {}

            elif method == "car":
                # This is the "Implicit" case.
                # The kernel K = (I - rho*W)^-1.
                # We construct M = (I - rho*W) and tell SpatialKernel it is the INVERSE.
                rho = kernel_params_["rho"]
                standardize = kernel_params_["standardize"]
                I = sp.eye(self.n, format="csc")
                M = I - rho * W_norm  # M is the inverse of K

                # We pass M and set is_inverse=True. SpatialKernel will handle
                # standardizing the precision to ensure diag(K)=1 if requested.
                self.kernel_ = SpatialKernel.from_matrix(
                    M,
                    is_inverse=True,
                    method="car",
                    rho=rho,
                    standardize=standardize,
                )
                self.kernel_params_ = {"rho": rho, "standardize": standardize}

            else:
                raise ValueError(f"Method {method} not supported for connectivity matrices.")

        self.backend_ = "matrix"

    def _prepare_data(
        self, source: str, keys: list[str] | None, min_cells: int, layer: str | None = None
    ) -> tuple[sp.csr_matrix, list[str], np.ndarray, np.ndarray]:
        """
        Prepare anndata feature data for testing (standardization, filtering).

        Extracts features from .obs or .var, applies quality filters (min non-zero count),
        handles categorical features (one-hot encoding), and computes per-feature statistics.

        Parameters
        ----------
        source : str
            Feature source: 'obs' or 'var'.
        keys : Optional[List[str]]
            Feature names to extract. If None, uses all features in source.
        min_cells : int
            Minimum number of non-zero values required per feature.
        layer : Optional[str]
            If source='var', which layer to use. If None, uses .X.

        Returns
        -------
        X_csc : scipy.sparse.csc_matrix
            Sparse feature matrix (n_samples, n_features), column-compressed.
        names : List[str]
            Feature names in order matching X_csc columns.
        means : np.ndarray
            Per-feature means (shape: n_features).
        stds : np.ndarray
            Per-feature standard deviations (shape: n_features).

        Notes
        -----
        - Categorical features in .obs are automatically one-hot encoded.
        - Features with zero variance or fewer than min_cells non-zeros are filtered.
        - Sparse matrices are preserved for memory efficiency.
        """

        if source == "obs":
            # check if keys are in obs
            if keys is not None:
                valid = [k for k in keys if k in self.adata.obs.columns]
                if len(valid) == 0:
                    raise ValueError("None of the specified keys found in adata.obs.")
                adata_tmp = self.adata.obs[valid].copy()
            else:
                raise ValueError("Keys must be provided when feature source is 'obs'.")

            # check if .obs[keys] are char/categorical
            if any(adata_tmp.dtypes == "object") or any(adata_tmp.dtypes == "category"):
                logger.info(
                    "Categorical features detected in .obs[keys]; performing one-hot encoding..."
                )
                # one-hot encode categorical variables while keeping others unchanged
                dummies = pd.get_dummies(adata_tmp)

                names = dummies.columns.tolist()
                X_dense = dummies.values.astype(np.float32)
                means = X_dense.mean(axis=0)
                stds = X_dense.std(axis=0, ddof=1)
                X_csc = sp.csc_matrix(X_dense)
            else:
                # All numeric - no encoding needed
                names = adata_tmp.columns.tolist()
                X_dense = adata_tmp.values.astype(np.float32)
                means = X_dense.mean(axis=0)
                stds = X_dense.std(axis=0, ddof=1)
                X_csc = sp.csc_matrix(X_dense)

        elif source == "var":
            # check if keys are in var
            if keys is not None:
                valid = [g for g in keys if g in self.adata.var_names]
                adata_tmp = self.adata[:, valid].copy()
            else:
                adata_tmp = self.adata.copy()

            # extract feature matrix
            if layer is not None:
                X = adata_tmp.layers[layer]
            else:
                X = adata_tmp.X

            names = adata_tmp.var_names.tolist()

            # compute means and stds (ddof=1 for sample std, consistent with statistics.py)
            n_obs = X.shape[0]
            if sp.issparse(X):
                means = np.array(X.mean(axis=0)).flatten()
                X2 = X.copy()
                X2.data **= 2
                means2 = np.array(X2.mean(axis=0)).flatten()
                var = means2 - (means**2)
                var[var < 0] = 0
                # Bessel correction: population var -> sample var
                if n_obs > 1:
                    var = var * n_obs / (n_obs - 1)
                stds = np.sqrt(var)
                X_csc = X.tocsc()
            else:
                means = np.mean(X, axis=0)
                stds = np.std(X, axis=0, ddof=1)
                X_csc = sp.csc_matrix(X)
        else:
            raise ValueError("Source must be either 'obs' or 'var'.")

        # Filter constant features and those with too few non-zeros
        to_keep = (stds > 0) & (X_csc.getnnz(axis=0) >= min_cells)
        X_csc = X_csc[:, to_keep]
        names = [names[i] for i in range(len(names)) if to_keep[i]]
        means = means[to_keep]
        stds = stds[to_keep]

        return X_csc, names, means, stds

    def compute_qstat(
        self,
        source: str = "var",
        features: list[str] | None = None,
        n_jobs: int = -1,
        layer: str | None = None,
        return_pval: bool = True,
        chunk_size: int = 64,
    ) -> pd.DataFrame:
        """
        Compute univariate spatial Q-statistic for selected features.

        Tests each feature for significant spatial clustering or dispersion using the
        pre-built kernel. Parallelizes across features and applies Benjamini-Hochberg
        multiple testing correction.

        Parameters
        ----------
        source : str, default 'var'
            Feature source: 'var' (genes) or 'obs' (metadata columns).
        features : Optional[List[str]]
            Feature names to test. If None, tests all features in source.
        n_jobs : int, default -1
            Number of parallel jobs. -1 uses all available cores; 1 for sequential.
        layer : Optional[str]
            If source='var', which layer to use (e.g., 'raw', 'log1p'). If None, uses .X.
        return_pval : bool, default True
            If True, returns p-values and BH-corrected p-values. If False, returns Q only.
        chunk_size : int, default 64
            Number of features to process in each internal batch for vectorization.
            If n_jobs > 1, each worker processes chunk_size features at a time.
            May be adjusted for memory/speed tradeoff.

        Returns
        -------
        df : pd.DataFrame
            Results sorted by Q (descending). Columns:
            - Feature: feature name
            - Q: test statistic (univariate spatial variability)
            - Z_score: standardized Q by null mean/std
            - P_value: tail probability under null (if return_pval=True)
            - P_adj: Benjamini-Hochberg adjusted p-value (if return_pval=True)

        Raises
        ------
        ValueError
            If kernel not initialized, or source is invalid.

        Notes
        -----
        Under H₀: feature has no spatial structure.
        Under H₁: significant spatial signal (clustering or dispersion).

        Zero-variance features are assigned Q=0, P_value=1.0.

        The null-distribution approximation is auto-selected from
        ``self.kernel_method_`` (``'clt'`` for Moran's I, ``'welch'`` for all other
        kernels) and cannot be overridden through this method. For full control
        over the null method (including ``'liu'``), call
        :func:`quadsv.statistics.spatial_q_test` directly.

        Examples
        --------
        >>> detector.build_kernel_from_coordinates(adata.obsm['spatial'])
        >>> results = detector.compute_qstat(source='var', features=['Gene1', 'Gene2'], n_jobs=-1)
        >>> top_genes = results.iloc[:10]
        """

        # 1. Ensure Kernel Exists
        if self.kernel_ is None:
            raise ValueError(
                "Kernel not initialized. Call .build_kernel(), "
                ".build_kernel_from_coordinates(), or .build_kernel_from_obsp() first."
            )

        # NUFFT backend takes a different code path (no dense K, different
        # null rescaling). Dispatch early and delegate.
        if self.backend_ == "nufft":
            return self._compute_qstat_nufft(
                source=source,
                features=features,
                n_jobs=n_jobs,
                layer=layer,
                return_pval=return_pval,
                chunk_size=chunk_size,
            )

        # 2. Compute Null Distribution
        null_method = "clt" if self.kernel_method_ in ["moran"] else "welch"
        logger.info("Computing null distribution approximation (method=%s)...", null_method)
        null_params = compute_null_params(self.kernel_, method=null_method)

        # 3. Prepare Data
        X_csc, names, means, stds = self._prepare_data(
            source=source, keys=features, min_cells=self.min_cells, layer=layer
        )

        # 4. Parallel Execution
        n_feats = len(names)
        if n_jobs == -1:
            import os

            n_jobs = os.cpu_count() or 1

        indices = np.arange(n_feats)
        chunks = np.array_split(indices, max(n_jobs * 4, 1))

        logger.info("Testing %d features using %d cores...", n_feats, n_jobs)

        results_list = Parallel(n_jobs=n_jobs)(
            delayed(_qstat_worker)(
                X_csc[:, chunk_idxs],
                chunk_idxs,
                self.kernel_,
                null_params,
                means,
                stds,
                names,
                return_pval,
                chunk_size,
            )
            for chunk_idxs in tqdm(
                chunks,
                desc=f"Q ({self.kernel_method_})",
                bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}",
            )
        )

        # 5. Aggregate Results
        flat_results = [item for sublist in results_list for item in sublist]
        df = pd.DataFrame(flat_results).set_index("Feature")
        if not return_pval:
            df = df.drop(columns=["P_value"])

        # 6. Multiple testing correction (Benjamini-Hochberg)
        if return_pval:
            _apply_bh_correction(df)

        return df.sort_values(by="Q", ascending=False)

    def compute_rstat(  # noqa: C901
        self,
        features_x: list[str] | None = None,
        features_y: list[str] | None = None,
        source: str = "var",
        n_jobs: int = -1,
        layer: str | None = None,
        return_pval: bool = True,
        chunk_size: int = 64,
    ) -> pd.DataFrame:
        """
        Compute bivariate spatial R-statistic (cross-spatial correlation) for feature pairs.

        Tests for significant spatial co-variation between pairs of features using
        the pre-built kernel. Supports symmetric (all pairs within one set) or bipartite
        (all X vs Y pairs) modes. Parallelizes computation and applies multiple testing correction.

        Parameters
        ----------
        features_x : Optional[List[str]]
            Feature names for the first set. If None and features_y is None, uses all features (symmetric mode).
        features_y : Optional[List[str]]
            Feature names for the second set. If None, computes all pairwise within features_x.
            If provided, computes all X vs Y pairs (bipartite mode).
        source : str, default 'var'
            Feature source: 'var' (genes) or 'obs' (metadata columns).
        n_jobs : int, default -1
            Number of parallel jobs. -1 uses all available cores; 1 for sequential.
        layer : Optional[str]
            If source='var', which layer to use (e.g., 'raw', 'log1p'). If None, uses .X.
        return_pval : bool, default True
            If True, returns p-values and BH-corrected p-values. If False, returns R only.
        chunk_size : int, default 64
            Number of Y features to process together, pre-computing K@Y_chunk.
            Larger values use more memory but fewer K matrix multiplications.

        Returns
        -------
        df : pd.DataFrame
            Results sorted by absolute Z_score (descending). Columns:

            - Feature_1: name of first feature
            - Feature_2: name of second feature
            - R: test statistic (bivariate spatial correlation, range approximately [-1, 1])
            - Z_score: standardized R by null mean/std
            - P_value: two-tailed p-value under null (if return_pval=True)
            - P_adj: Benjamini-Hochberg adjusted p-value (if return_pval=True)

        Raises
        ------
        ValueError
            If kernel not initialized, features_x is None when features_y is provided, or no valid pairs generated.

        Notes
        -----
        Under H₀: features are spatially independent.
        Under H₁: significant spatial co-clustering or co-dispersion.

        Different than quadsv.statistics.spatial_r_test, this function always returns R-statistics of
        all requested feature pairs in the symmetric mode (features_y = None). Thus, for features_x = [A, B, C],
        the output will contain (A, A), (A, B), (A, C), (B, A), (B, B), (B, C), (C, A), (C, B), (C, C).

        P-value calculation uses a normal approximation based on Tr(K²) and is not
        configurable through this method. For finer control over the null model,
        call :func:`quadsv.statistics.spatial_r_test` directly.

        Zero-variance features are handled gracefully (assigned R=0, P=1).

        Examples
        --------
        >>> detector.build_kernel_from_coordinates(adata.obsm['spatial'])
        >>> # All pairwise correlations within gene set
        >>> results = detector.compute_rstat(features_x=['Gene1', 'Gene2', 'Gene3'], n_jobs=-1)
        >>> # Cross-correlation between two gene sets
        >>> results = detector.compute_rstat(
        ...     features_x=['Gene1', 'Gene2'],
        ...     features_y=['Gene3', 'Gene4'],
        ...     n_jobs=-1
        ... )
        """
        if self.kernel_ is None:
            raise ValueError("Kernel not initialized.")

        if self.backend_ == "nufft":
            return self._compute_rstat_nufft(
                features_x=features_x,
                features_y=features_y,
                source=source,
                n_jobs=n_jobs,
                layer=layer,
                return_pval=return_pval,
                chunk_size=chunk_size,
            )

        # 1. Compute Null Params for R
        # We need Trace(KK^T) which is Trace(K^2) for symmetric K
        # compute_null_params already computes var_Q = 2*Tr(K^2).
        # var_R = Tr(K^2) = var_Q / 2.
        logger.info("Computing null distribution for R statistic...")
        q_null = compute_null_params(self.kernel_, method="clt")
        null_params = {
            "mean_R": 0.0,
            "var_R": q_null["var_Q"] / 2.0,  # Derive from existing trace calculation
        }

        # 2. Prepare Data
        # We load all unique features needed
        if features_y is None:
            unique_feats = features_x
            mode = "symmetric"
        else:
            if features_x is None:
                raise ValueError("features_x cannot be None.")
            unique_feats = list(set(features_x) | set(features_y))
            mode = "bipartite"

        X_csc, names, means, stds = self._prepare_data(
            source=source, keys=unique_feats, min_cells=self.min_cells, layer=layer
        )

        # Map names to indices in the prepared matrix
        name_to_idx = {n: i for i, n in enumerate(names)}

        # 3. Generate Y chunks (for pre-computing K@Y)
        if mode == "symmetric":
            valid_feats = [f for f in features_x if f in name_to_idx]
            valid_y_indices = [name_to_idx[f] for f in valid_feats]
        else:
            valid_y = [f for f in features_y if f in name_to_idx]
            valid_y_indices = [name_to_idx[f] for f in valid_y]

        # Chunk Y indices for K@Y pre-computation
        y_chunks = []
        for start in range(0, len(valid_y_indices), chunk_size):
            end = min(start + chunk_size, len(valid_y_indices))
            y_chunks.append(valid_y_indices[start:end])

        # 4. Generate X features list
        if mode == "symmetric":
            valid_x_indices = [name_to_idx[f] for f in valid_feats]
        else:
            valid_x = [f for f in features_x if f in name_to_idx]
            valid_x_indices = [name_to_idx[f] for f in valid_x]

        if len(valid_x_indices) == 0 or len(valid_y_indices) == 0:
            raise ValueError("No valid features found after filtering.")

        # 5. Parallel Execution: Process each Y chunk
        if n_jobs == -1:
            import os

            n_jobs = os.cpu_count() or 1

        logger.info(
            "Testing %d x %d pairs using %d cores with chunk_size=%d...",
            len(valid_x_indices),
            len(valid_y_indices),
            n_jobs,
            chunk_size,
        )

        results_list = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_rstat_worker_chunked)(
                X_csc,
                y_chunk,
                valid_x_indices,
                self.kernel_,
                null_params,
                means,
                stds,
                names,
                return_pval,
            )
            for y_chunk in tqdm(
                y_chunks,
                desc=f"R ({self.kernel_method_})",
                bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}",
            )
        )

        # 6. Aggregate
        flat_results = [item for sublist in results_list for item in sublist]
        df = pd.DataFrame(flat_results)

        # 7. Multiple testing correction (Benjamini-Hochberg)
        if return_pval:
            _apply_bh_correction(df)

        return df.sort_values(by="Z_score", key=abs, ascending=False)

    # ------------------------------------------------------------------
    # NUFFT backend paths
    # ------------------------------------------------------------------

    def _prepare_features_nufft(
        self, source: str, features: list[str] | None, layer: str | None
    ) -> tuple[sp.csc_matrix, list[str], np.ndarray, np.ndarray]:
        """Pull a sparse ``(n_spots, n_features)`` CSC + names + per-feature
        mean/std. NUFFT-friendly: computes variance from sparse moments and
        never materializes a full dense ``X``."""
        if source == "var":
            X = self.adata.X if layer is None else self.adata.layers[layer]
            X_csc = X.tocsc() if sp.issparse(X) else sp.csc_matrix(X)
            names = list(self.adata.var_names)
            if features is not None:
                selected = [g for g in features if g in names]
                missing = set(features) - set(selected)
                if missing:
                    logger.warning(
                        "Requested features not in adata.var_names: %s", sorted(missing)[:5]
                    )
                idx = [names.index(g) for g in selected]
                X_csc = X_csc[:, idx]
                names = selected
        elif source == "obs":
            if features is None:
                raise ValueError("source='obs' requires `features` (obs column names).")
            cols = [features] if isinstance(features, str) else list(features)
            missing = [c for c in cols if c not in self.adata.obs.columns]
            if missing:
                raise KeyError(f"obs columns missing: {missing}")
            X_csc = sp.csc_matrix(self.adata.obs[cols].to_numpy(dtype=np.float64))
            names = list(cols)
        else:
            raise ValueError(f"source must be 'var' or 'obs', got '{source}'.")

        nnz_per = np.asarray((X_csc != 0).sum(axis=0)).ravel()
        means = np.asarray(X_csc.mean(axis=0)).ravel()
        sq = X_csc.multiply(X_csc)
        sq_mean = np.asarray(sq.mean(axis=0)).ravel()
        var = np.maximum(sq_mean - means**2, 0.0)
        stds = np.sqrt(var)
        keep = (stds > 0) & (nnz_per >= self.min_cells)

        X_kept = X_csc[:, keep]
        names_kept = [names[i] for i, k in enumerate(keep) if k]
        return X_kept, names_kept, means[keep], stds[keep]

    def _compute_qstat_nufft(  # noqa: C901
        self,
        source: str,
        features: list[str] | None,
        n_jobs: int,
        layer: str | None,
        return_pval: bool,
        chunk_size: int,
    ) -> pd.DataFrame:
        """NUFFT dispatch for :meth:`compute_qstat`. Builds null params once and
        delegates per-feature work to :func:`spatial_q_test_nufft`."""
        kernel = self.kernel_
        scale = kernel.n / (kernel.grid_shape[0] * kernel.grid_shape[1])
        null_params: dict[str, float | np.ndarray] | None = None
        if return_pval:
            if kernel.method == "moran":
                null_params = {
                    "method": "clt",
                    "mean_Q": kernel.trace() * scale,
                    "var_Q": 2.0 * kernel.square_trace() * (scale**2),
                }
            else:
                full = kernel.eigenvalues(return_full=True)
                if full.min() < -0.1:
                    raise ValueError(
                        "Kernel has significant negative eigenvalues; "
                        "Liu's method may be invalid."
                    )
                null_params = {
                    "method": "liu",
                    "eigenvalues": full[full > 1e-9] * scale,
                }

        logger.info("Preparing %s features (layer=%s)...", source, layer)
        X_kept, names_kept, means, stds = self._prepare_features_nufft(source, features, layer)
        n_feats = len(names_kept)
        if n_feats == 0:
            cols = ["Feature", "Q", "Z_score", "P_value", "P_adj"]
            return pd.DataFrame(columns=cols)

        logger.info("Testing %d features via NUFFT (n_jobs=%s)...", n_feats, n_jobs)

        def _batch(batch_idx: np.ndarray) -> list[dict[str, Any]]:
            # Densify one small block at a time; never materialize full X.
            block = np.asarray(X_kept[:, batch_idx].todense(), dtype=np.float64)
            block_std = (block - means[batch_idx][None, :]) / np.where(
                stds[batch_idx] > 0, stds[batch_idx], 1.0
            )[None, :]
            if return_pval:
                Q_arr, P_arr = spatial_q_test_nufft(
                    block_std, kernel, null_params=null_params, is_standardized=True
                )
                Q_arr = np.atleast_1d(Q_arr)
                P_arr = np.atleast_1d(P_arr)
            else:
                Q_arr = np.atleast_1d(
                    spatial_q_test_nufft(block_std, kernel, return_pval=False, is_standardized=True)
                )
                P_arr = np.full_like(Q_arr, np.nan)
            # Reference trace and trace²-based Z for reporting.
            if null_params is not None:
                if "eigenvalues" in null_params:
                    evals = np.asarray(null_params["eigenvalues"], dtype=float)
                    trK = float(evals.sum())
                    varQ = 2.0 * float((evals**2).sum())
                else:
                    trK = float(null_params["mean_Q"])
                    varQ = float(null_params["var_Q"])
                sigma = float(np.sqrt(varQ)) if varQ > 0 else 0.0
                Z_arr = (Q_arr - trK) / sigma if sigma > 0 else np.zeros_like(Q_arr)
            else:
                Z_arr = np.full_like(Q_arr, np.nan)

            out = []
            for local_i, gi in enumerate(batch_idx):
                out.append(
                    {
                        "Feature": names_kept[gi],
                        "Q": float(Q_arr[local_i]),
                        "Z_score": float(Z_arr[local_i]),
                        "P_value": float(P_arr[local_i]),
                    }
                )
            return out

        idx_all = np.arange(n_feats)
        batches = [idx_all[i : i + chunk_size] for i in range(0, n_feats, chunk_size)]
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_batch)(batch)
            for batch in tqdm(batches, desc=f"Q (NUFFT, {self.kernel_method_})")
        )
        flat = [row for chunk in results for row in chunk]
        df = pd.DataFrame(flat)
        if not return_pval:
            df = df.drop(columns=["P_value", "Z_score"], errors="ignore")
        elif return_pval:
            _apply_bh_correction(df)
        return df.sort_values("Q", ascending=False).reset_index(drop=True)

    def _compute_rstat_nufft(
        self,
        features_x: list[str] | None,
        features_y: list[str] | None,
        source: str,
        n_jobs: int,
        layer: str | None,
        return_pval: bool,
        chunk_size: int,
    ) -> pd.DataFrame:
        """NUFFT dispatch for :meth:`compute_rstat`."""
        kernel = self.kernel_
        scale = kernel.n / (kernel.grid_shape[0] * kernel.grid_shape[1])
        var_R = float(kernel.square_trace()) * (scale**2) if return_pval else 0.0
        null_params = {"var_R": var_R}

        if features_x is None and features_y is not None:
            raise ValueError("Provide features_x when features_y is specified.")

        X_kept, names_kept, means_x, stds_x = self._prepare_features_nufft(
            source, features_x, layer
        )
        if features_y is None:
            X_y, names_y, means_y, stds_y = X_kept, names_kept, means_x, stds_x
        else:
            X_y, names_y, means_y, stds_y = self._prepare_features_nufft(source, features_y, layer)

        if len(names_kept) == 0 or len(names_y) == 0:
            return pd.DataFrame(
                columns=["Feature_1", "Feature_2", "R", "Z_score", "P_value", "P_adj"]
            )

        logger.info("Testing %d x %d feature pairs via NUFFT...", len(names_kept), len(names_y))

        # Densify the X block once (kept dense in memory for standardization).
        X_block = np.asarray(X_kept.todense())
        X_std = (X_block - means_x[None, :]) / np.where(stds_x > 0, stds_x, 1.0)[None, :]

        results: list[dict[str, Any]] = []
        y_chunks = [slice(i, i + chunk_size) for i in range(0, len(names_y), chunk_size)]
        for ysl in tqdm(y_chunks, desc=f"R (NUFFT, {self.kernel_method_})"):
            Y_block = np.asarray(X_y[:, ysl].todense())
            Y_std = (Y_block - means_y[ysl][None, :]) / np.where(stds_y[ysl] > 0, stds_y[ysl], 1.0)[
                None, :
            ]
            # Use the R-test on the chunk so null-param handling lives in one place.
            R_chunk = spatial_r_test_nufft(
                X_std,
                Y_std,
                kernel,
                null_params=null_params,
                return_pval=False,
                is_standardized=True,
            )
            R_chunk = np.atleast_2d(R_chunk)
            if R_chunk.shape != (len(names_kept), Y_std.shape[1]):
                R_chunk = R_chunk.reshape(len(names_kept), Y_std.shape[1])
            for i, name_x in enumerate(names_kept):
                for j, name_y in enumerate(names_y[ysl]):
                    r = float(R_chunk[i, j])
                    row = {"Feature_1": name_x, "Feature_2": name_y, "R": r}
                    if return_pval and var_R > 0:
                        z = r / np.sqrt(var_R)
                        row["Z_score"] = z
                        row["P_value"] = float(2.0 * norm.sf(abs(z)))
                    results.append(row)

        df = pd.DataFrame(results)
        if return_pval:
            _apply_bh_correction(df)
        return df.sort_values("R", ascending=False).reset_index(drop=True)
