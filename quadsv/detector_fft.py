import warnings

# Suppress known deprecation warnings from SpatialData dependencies BEFORE importing anything else
warnings.filterwarnings("ignore", category=FutureWarning, message=".*legacy Dask DataFrame.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated.*")

try:
    import spatialdata as sd
except ImportError as e:
    raise ImportError(
        "spatialdata is required for PatternDetectorFFT. Please install it via 'pip install spatialdata'."
    ) from e

from typing import Any

import numpy as np
import pandas as pd
import scipy.fft
from joblib import Parallel, delayed
from scipy.stats import norm
from tqdm import tqdm

from quadsv.fft import FFTKernel, spatial_q_test_fft


def _qstat_worker_fft(
    raster_layer, feature_batch: list[str], kernel: FFTKernel, return_pval: bool
) -> list[dict]:
    """
    Worker function for parallel Q-statistic computation with FFT kernels.

    Parameters
    ----------
    raster_layer : xarray.DataArray
        Rasterized data layer (lazy Dask array).
    feature_batch : List[str]
        Feature names to process in this batch.
    kernel : FFTKernel
        Pre-constructed FFT kernel object.
    return_pval : bool
        Whether to compute p-values.

    Returns
    -------
    List[dict]
        Results for each feature: {'Feature': str, 'Q': float, 'P_value': float, 'Z_score': float}
    """
    results = []

    # Load data to memory for batch: shape (M, ny, nx)
    data_chunk = raster_layer.sel(c=feature_batch).values
    # Transpose to (ny, nx, M) for kernel
    data_chunk_transposed = np.moveaxis(data_chunk, 0, -1)

    # Compute statistics
    if return_pval:
        stats, pvals = spatial_q_test_fft(data_chunk_transposed, kernel, return_pval=True)
    else:
        stats = spatial_q_test_fft(data_chunk_transposed, kernel, return_pval=False)
        pvals = None

    # Ensure array semantics for iteration (handle 0-d arrays)
    stats = np.atleast_1d(np.asarray(stats))
    if pvals is not None:
        pvals = np.atleast_1d(np.asarray(pvals, dtype=object))
    else:
        pvals = np.array([None] * len(feature_batch), dtype=object)

    # Null parameters
    mu = kernel.trace()
    sigma = np.sqrt(2.0 * kernel.square_trace())
    z_scores = (stats - mu) / sigma if sigma > 1e-12 else np.zeros_like(stats)

    # Format batch results
    for j, gene in enumerate(feature_batch):
        results.append(
            {
                "Feature": gene,
                "Q": float(stats[j]),
                "P_value": float(pvals[j]) if pvals[j] is not None else None,
                "Z_score": float(z_scores[j]),
            }
        )

    return results


class PatternDetectorFFT:
    """
    High-level API for detecting spatial patterns in grid data using FFT-accelerated tests.

    This class integrates with spatialdata objects to perform univariate (Q-test) and
    bivariate (R-test) spatial statistics across large feature sets efficiently using
    FFT-based kernel methods and parallel processing.

    Parameters
    ----------
    sdata : sd.SpatialData
        A SpatialData object containing tables with spatial coordinates and rasterized images.
    min_count : int, optional
        Minimum count threshold for features to be included in analysis. If None, no filtering.
    kernel_method : str, default 'car'
        Kernel method to use. Must be one of: 'gaussian', 'matern', 'moran', 'graph_laplacian', 'car'.
    kernel_params : dict, optional
        Additional kernel-specific parameters (e.g., bandwidth, nu, rho, neighbor_degree).

    Attributes
    ----------
    sdata : sd.SpatialData
        Reference to the input spatial data object.
    kernel_ : FFTKernel or None
        Lazily constructed FFT kernel; initialized on first compute_qstat/compute_rstat call.
    kernel_method_ : str
        The selected kernel method.
    kernel_params_ : dict
        Pre-processed kernel parameters with defaults.

    Methods
    -------
    compute_qstat(bins, table_name, col_key, row_key, features=None, n_jobs=-1, return_pval=True, chunk_size=100)
        Compute univariate spatial Q-test statistics for features in parallel.
    compute_rstat(bins, table_name, col_key, row_key, features_x=None, features_y=None, n_jobs=-1, return_pval=True, chunk_size=100)
        Compute bivariate spatial R-test (cross-correlation) statistics in parallel.

    Examples
    --------
    >>> detector = PatternDetectorFFT(sdata, kernel_method='car', rho=0.8)
    >>> q_results = detector.compute_qstat('grid', 'table', 'col_idx', 'row_idx', features=['Gene_1', 'Gene_2'])
    >>> r_results = detector.compute_rstat('grid', 'table', 'col_idx', 'row_idx')
    """

    _available_kernels = ["gaussian", "matern", "moran", "graph_laplacian", "car"]

    def __init__(
        self,
        sdata: sd.SpatialData,
        min_count: int | None = None,
        kernel_method: str = "car",
        **kernel_params: Any,
    ) -> None:
        """
        Initialize the detector with the SpatialData object and grid settings.

        Parameters
        ----------
        sdata : sd.SpatialData
            The spatial data object containing the tables.
        min_count : int or None, optional
            Minimum count threshold for features to be included.
        kernel_method : str, default 'car'
            The kernel method to use. Must be one of: 'gaussian', 'matern', 'moran', 'graph_laplacian', 'car'.
        kernel_params : dict, optional
            Additional parameters for the kernel (e.g., bandwidth, nu, neighbor_degree, rho).
            Defaults are merged with user-provided values.

        Raises
        ------
        AssertionError
            If kernel_method is not in _available_kernels.
        """
        self.sdata = sdata
        self.min_count = min_count

        assert (
            kernel_method in self._available_kernels
        ), f"kernel_method must be one of {self._available_kernels}."
        self.kernel_method_ = kernel_method

        # Merge user params with defaults
        defaults = self._get_default_params(self.kernel_method_).copy()
        if kernel_params:
            for key, value in kernel_params.items():
                if key in defaults:
                    defaults[key] = value
                else:
                    raise ValueError(f"Unknown parameter '{key}' for method '{kernel_method}'")
        self.kernel_params_ = defaults

        # FFTKernel will be initialized once grid shape is known
        self.kernel_ = None

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
            Merged dictionary of grid defaults and method-specific defaults.
            Grid defaults: spacing, topology.
            Method defaults: bandwidth (gaussian/matern), nu (matern), neighbor_degree (moran/graph_laplacian/car), rho (car).
        """
        general_defaults = {
            "spacing": (1.0, 1.0),
            "topology": "square",
            "fft_solver": "fft2",
            "workers": None,
        }
        method_defaults = {
            "gaussian": {"bandwidth": 2.0},
            "matern": {"nu": 1.5, "bandwidth": 2.0},
            "moran": {"neighbor_degree": 1},
            "graph_laplacian": {"neighbor_degree": 1},
            "car": {"rho": 0.9, "neighbor_degree": 1},
        }
        return {**general_defaults, **method_defaults.get(method, {})}

    def _rasterize_bins(
        self,
        bins: str,
        table_name: str,
        col_key: str,
        row_key: str,
        value_key: str | None = None,
        return_region_as_labels: bool = False,
    ) -> str:
        """
        Wrapper for spatialdata.rasterize_bins with format validation.

        Converts sparse table into a rasterized (grid) image. Ensures CSC sparse
        format for efficient processing and stores result in sdata.images.

        Parameters
        ----------
        bins : str
            Name of the SpatialElement (Shape) which defines the grid-like bins.
        table_name : str
            Name of the table annotating the SpatialElement in sdata.tables.
        col_key : str
            Column in sdata[table_name].obs containing column indices (integers) for bins.
        row_key : str
            Column in sdata[table_name].obs containing row indices (integers) for bins.
        value_key : str, optional
            Column in sdata[table_name].obs to use as pixel values. If None, uses counts/presence.
        return_region_as_labels : bool, default False
            If True, returns bin region masks as integer labels. If False, returns aggregated values.

        Returns
        -------
        img_key : str
            Key under which the rasterized image is stored in sdata.images.
            Format: 'rasterized_{table_name}'.

        Notes
        -----
        This method ensures the underlying matrix is in CSC sparse format for efficient
        column-wise operations required by rasterize_bins.
        """
        # Ensure the table exists
        if table_name not in self.sdata.tables:
            raise ValueError(f"Table {table_name} not found in sdata.")

        # Performance optimization: rasterize_bins requires CSC matrix
        # We check and convert if necessary to avoid runtime errors
        if hasattr(self.sdata.tables[table_name], "X"):
            if not isinstance(self.sdata.tables[table_name].X, (np.ndarray)):
                # Assuming scipy sparse; check format
                if self.sdata.tables[table_name].X.format != "csc":
                    self.sdata.tables[table_name].X = self.sdata.tables[table_name].X.tocsc()

        # Construct the output key
        img_key = f"rasterized_{table_name}"

        print(f"Rasterizing {table_name} into {img_key}...")
        rasterized = sd.rasterize_bins(
            self.sdata,
            bins=bins,
            table_name=table_name,
            col_key=col_key,
            row_key=row_key,
            value_key=value_key,
            return_region_as_labels=return_region_as_labels,
        )

        # Store in sdata images
        self.sdata[img_key] = rasterized
        return img_key

    def _filter_features(self, features, table_name):
        """Helper to validate and filter features based on min_count."""
        # Extract all features from table
        all_features = self.sdata.tables[table_name].var_names.to_list()

        if features is None:
            # Use all features
            valid = all_features
        else:
            valid = [f for f in features if f in all_features]
            if not valid:
                raise ValueError("No valid features found.")

        valid = np.array(valid)

        # Apply min_count filter
        if self.min_count is not None:
            counts = np.asarray(self.sdata.tables[table_name][:, valid].X.sum(axis=0)).ravel()
            valid = valid[counts >= self.min_count]
            if len(valid) == 0:
                raise ValueError(f"No features passed min_count={self.min_count}")

        return valid

    def _apply_bh_correction(self, df):
        """Applies Benjamini-Hochberg correction to P_value column in place."""
        pvals = df["P_value"].to_numpy()
        valid_mask = np.isfinite(pvals)
        m = valid_mask.sum()
        if m == 0:
            return

        p_sorted_idx = np.argsort(pvals[valid_mask])
        p_sorted = pvals[valid_mask][p_sorted_idx]
        ranks = np.arange(1, m + 1)

        bh_vals = p_sorted * m / ranks
        bh_adj = np.minimum.accumulate(bh_vals[::-1])[::-1]
        bh_adj = np.clip(bh_adj, 0, 1)

        p_adj = np.full_like(pvals, np.nan)
        p_adj_indices = np.where(valid_mask)[0][p_sorted_idx]
        p_adj[p_adj_indices] = bh_adj

        df["P_adj"] = p_adj

    def compute_qstat(
        self,
        bins: str,
        table_name: str,
        col_key: str,
        row_key: str,
        features: list[str] | None = None,
        n_jobs: int = -1,
        workers: int | None = None,
        return_pval: bool = True,
        chunk_size: int = 256,
    ) -> pd.DataFrame:
        """
        Computes spatial Q-statistic for specified features in parallel.

        Tests each feature for significant spatial clustering or dispersion using
        the selected kernel method. Parallelizes computation across features and
        applies Benjamini-Hochberg multiple testing correction.

        Parameters
        ----------
        bins : str
            Name of the SpatialElement (Shape) which defines the grid-like bins.
        table_name : str
            Name of the table annotating the SpatialElement in sdata.tables.
        col_key : str
            Column in sdata[table_name].obs containing column indices (integers) for bins.
        row_key : str
            Column in sdata[table_name].obs containing row indices (integers) for bins.
        features : Optional[List[str]]
            List of feature names to analyze. If None, all features in table are processed.
        n_jobs : int, default -1
            Number of parallel worker processes for feature chunks. -1 uses all available cores; 1 for sequential.
        workers : int, optional
            Number of threads for scipy.fft parallelization within each worker.
            Will overwrite kernel_params_['workers'] if provided.
        return_pval : bool, default True
            Whether to return p-values and adjusted p-values (Benjamini-Hochberg).
        chunk_size : int, default 256
            Number of features to process in a single batch per worker (for efficiency).
            Larger grids may require smaller chunks to avoid OOM.

        Returns
        -------
        df : pd.DataFrame
            Results table sorted by Q (descending). Columns:
            - Feature: feature name
            - Q: test statistic (univariate spatial statistic)
            - Z_score: standardized Q by null mean/std
            - P_value: tail probability under H₀ (if return_pval=True)
            - P_adj: adjusted p-value via Benjamini-Hochberg (if return_pval=True)

        Raises
        ------
        ValueError
            If table_name not found in sdata, or no valid features pass filters.

        Notes
        -----
        Under H₀: feature is spatially independent (null distribution approximated via Liu or Normal).
        Under H₁: mean-shift present.

        Rasterization is performed lazily via dask for memory efficiency.
        Kernel is auto-initialized if grid shape differs from previous call.

        Examples
        --------
        >>> detector = PatternDetectorFFT(sdata, kernel_method='car', rho=0.8)
        >>> results = detector.compute_qstat('grid_bins', 'counts', 'col_idx', 'row_idx', n_jobs=4, workers=-1)
        >>> top_features = results.head(10)  # Top 10 by spatial signal strength
        """
        # 1. Rasterize (Lazy construction)
        img_key = self._rasterize_bins(
            bins=bins, table_name=table_name, col_key=col_key, row_key=row_key
        )

        # Get the DataArray (Lazy Dask array)
        raster_layer = self.sdata[img_key]

        # 2. Handle Feature Selection
        features = self._filter_features(features, table_name)

        # 3. Build Kernel (Requires Grid Shape)
        _, ny, nx = raster_layer.shape

        # Only rebuild kernel if shape changed or not initialized
        if self.kernel_ is None or (self.kernel_.ny, self.kernel_.nx) != (ny, nx):
            print(f"Building FFTKernel ({self.kernel_method_}) for grid shape ({ny}, {nx})...")
            # Pass workers to kernel for FFT parallelization
            if workers is not None:  # Override if provided
                self.kernel_params_["workers"] = workers
            self.kernel_ = FFTKernel(
                shape=(ny, nx), method=self.kernel_method_, **self.kernel_params_
            )

        # 4. Determine number of parallel jobs
        if n_jobs == -1:
            import os

            n_jobs = os.cpu_count() or 1

        # 5. Split features into batches for parallel processing
        feature_batches = [
            features[i : i + chunk_size] for i in range(0, len(features), chunk_size)
        ]

        print(
            f"Processing {len(features)} features in {len(feature_batches)} batches using {n_jobs} cores..."
        )

        # 6. Parallel execution
        results_list = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_qstat_worker_fft)(raster_layer, batch, self.kernel_, return_pval)
            for batch in tqdm(
                feature_batches,
                desc=f"Q ({self.kernel_method_})",
                bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}",
            )
        )

        # 7. Flatten results
        results = [item for sublist in results_list for item in sublist]

        # 5. Compile results
        df = pd.DataFrame(results).set_index("Feature")
        if not return_pval:
            df = df.drop(columns=["P_value"])

        # 6. Multiple testing correction (Benjamini-Hochberg) in place
        if return_pval:
            self._apply_bh_correction(df)

        return df.sort_values(by="Q", ascending=False)

    def _compute_batch_spectral_embeddings(self, raster_layer, feature_names):
        """
        Helper: Loads data, standardizes, and computes weighted spectral components.
        Returns matrix of shape (n_features, n_spectral_components).
        """
        # 1. Load Data (IO Bound)
        # Shape: (N_features, Y, X)
        data = raster_layer.sel(c=feature_names).values
        n_feats, ny, nx = data.shape

        # 2. Standardize (In-place to save memory)
        # Mean/Std per feature
        means = np.mean(data, axis=(1, 2), keepdims=True)
        stds = np.std(data, axis=(1, 2), keepdims=True, ddof=1)

        # Avoid div by zero
        stds[stds < 1e-12] = 1.0
        data = (data - means) / stds

        # 3. FFT and Spectral Weighting
        # Use selected FFT solver
        if self.kernel_.fft_solver == "fft2":
            freq_data = scipy.fft.fft2(data, axes=(1, 2), workers=-1)
            rfft_spectrum = self.kernel_.eigenvalues().reshape(ny, nx)
        else:
            freq_data = scipy.fft.rfft2(data, axes=(1, 2), workers=-1)
            rfft_spectrum = self.kernel_.eigenvalues().reshape(ny, nx // 2 + 1)

        weights = np.sqrt(np.abs(rfft_spectrum))

        # Broadcast multiply
        weighted_freq = freq_data * weights[None, :, :]

        # 4. Flatten spatial dimensions for matrix multiplication
        # Result: (N_features, n_freq_bins)
        return weighted_freq.reshape(n_feats, -1)

    def compute_rstat(
        self,
        bins: str,
        table_name: str,
        col_key: str,
        row_key: str,
        features_x: list[str] | None = None,
        features_y: list[str] | None = None,
        return_pval: bool = True,
        chunk_size: int = 256,
        workers: int | None = -1,
    ) -> pd.DataFrame:
        """
        Computes spatial R-statistic (bivariate spatial correlation) for feature pairs.

        Tests for significant spatial co-variation between pairs of features using
        kernel-based cross-correlation. Parallelizes computation and applies multiple
        testing correction via Benjamini-Hochberg adjustment.

        Parameters
        ----------
        bins : str
            Name of the SpatialElement (Shape) which defines the grid-like bins.
        table_name : str
            Name of the table annotating the SpatialElement in sdata.tables.
        col_key : str
            Column in sdata[table_name].obs containing column indices (integers) for bins.
        row_key : str
            Column in sdata[table_name].obs containing row indices (integers) for bins.
        features_x : Optional[List[str]]
            List of feature names for the X variable. If None and features_y is None,
            all features are used (symmetric pairwise mode).
        features_y : Optional[List[str]]
            List of feature names for the Y variable. If None, computes all pairwise
            comparisons within features_x (mode='symmetric'). If provided with features_x,
            computes all X vs Y pairs (mode='bipartite').
        workers : int, default -1
            Number of parallel workers for scipt.fft. -1 uses all available cores.
            Will overwrite kernel_params_['workers'] if provided.
        return_pval : bool, default True
            Whether to return p-values and adjusted p-values (Benjamini-Hochberg).
        chunk_size : int, default 256
            Number of genes to process in a batch.
            Recommended: 512 - 2048 depending on RAM. Larger grids may require smaller chunks to avoid OOM.

        Returns
        -------
        df : pd.DataFrame
            Results table sorted by R (descending). Columns:
            - Feature_1: first feature name
            - Feature_2: second feature name
            - R: test statistic (bivariate spatial correlation)
            - Z_score: standardized R by null mean/std
            - P_value: two-tailed p-value under H₀ (if return_pval=True)
            - P_adj: adjusted p-value via Benjamini-Hochberg (if return_pval=True)

        Raises
        ------
        ValueError
            If table_name not found in sdata, or features_x is None when features_y is provided,
            or no valid feature pairs are generated.

        Notes
        -----
        Under H₀: features are spatially independent (Normal approximation).
        Under H₁: significant spatial co-clustering or co-dispersion detected.

        Symmetric mode (features_x only): Returns upper triangular pairs (Feature_1 < Feature_2).
        Bipartite mode (features_x and features_y): Returns all X vs Y pairs.

        Examples
        --------
        >>> detector = PatternDetectorFFT(sdata, kernel_method='car', rho=0.8)
        >>> results = detector.compute_rstat('grid_bins', 'counts', 'col_idx', 'row_idx', workers=4)
        >>> top_pairs = results.head(10)  # Top 10 spatially co-correlated pairs
        """
        import gc  # Garbage collector

        # 1. Rasterize (Lazy)
        img_key = self._rasterize_bins(bins, table_name, col_key, row_key)
        raster_layer = self.sdata[img_key]

        # 2. Resolve Features
        all_features = raster_layer.coords["c"].values

        if features_x is None and features_y is None:
            features_x = all_features
            features_y = None
            mode = "symmetric"
        elif features_x is not None and features_y is None:
            mode = "symmetric"
        else:
            mode = "bipartite"

        features_x = self._filter_features(features_x, table_name)
        if mode == "bipartite":
            features_y = self._filter_features(features_y, table_name)

        # 3. Initialize Kernel
        _, ny, nx = raster_layer.shape

        # Update workers in kernel params if provided
        if workers is not None:
            self.kernel_params_["workers"] = workers

        if self.kernel_ is None or (self.kernel_.ny, self.kernel_.nx) != (ny, nx):
            print(f"Building FFTKernel ({self.kernel_method_}) for grid shape ({ny}, {nx})...")
            self.kernel_ = FFTKernel((ny, nx), method=self.kernel_method_, **self.kernel_params_)

        # 4. Prepare Batches
        chunks_x = np.array_split(features_x, np.ceil(len(features_x) / chunk_size))
        if mode == "bipartite":
            chunks_y = np.array_split(features_y, np.ceil(len(features_y) / chunk_size))
        else:
            chunks_y = chunks_x

        print(
            f"Computing R-stats: {len(features_x)} x {len(features_y) if features_y is not None else len(features_x)} matrix."
        )
        print(f"Processing in {len(chunks_x)} chunks of size ~{chunk_size}...")

        sigma = np.sqrt(self.kernel_.square_trace())
        results_list = []

        # 5. Block Iteration
        for i, batch_x_names in enumerate(tqdm(chunks_x, desc="Processing X chunks")):
            # Load Embeddings X (High Memory Usage)
            embeddings_x = self._compute_batch_spectral_embeddings(raster_layer, batch_x_names)

            start_j = i if mode == "symmetric" else 0

            for j in range(start_j, len(chunks_y)):
                batch_y_names = chunks_y[j]

                # Load Embeddings Y
                if mode == "symmetric" and i == j:
                    embeddings_y = embeddings_x  # Reference, no copy
                else:
                    embeddings_y = self._compute_batch_spectral_embeddings(
                        raster_layer, batch_y_names
                    )

                # --- CROSS-BATCH CORRELATION ---
                # R_block shape: (chunk_size, chunk_size) -> Very small
                # This step reduces millions of pixels down to a simple correlation number
                R_block = np.matmul(embeddings_x, embeddings_y.conj().T).real

                # Normalize by grid size (rfft2 is unnormalized)
                R_block /= nx * ny

                # --- Format Results ---
                # Create meshgrid of indices for this block
                n_x = len(batch_x_names)
                n_y = len(batch_y_names)

                # Create coordinate grids
                # If symmetric and diagonal block, we only want upper triangle
                if mode == "symmetric" and i == j:
                    # Get upper triangle indices
                    r_idx, c_idx = np.triu_indices(n_x)

                    # Extract values
                    r_vals = R_block[r_idx, c_idx]
                    feat_1 = batch_x_names[r_idx]
                    feat_2 = batch_y_names[c_idx]
                else:
                    # Full block
                    # Flatten the block
                    r_vals = R_block.ravel()
                    # Repeat X names for rows, Tile Y names for cols
                    feat_1 = np.repeat(batch_x_names, n_y)
                    feat_2 = np.tile(batch_y_names, n_x)

                # Store block results
                batch_df = pd.DataFrame({"Feature_1": feat_1, "Feature_2": feat_2, "R": r_vals})

                if return_pval:
                    if sigma > 1e-12:
                        z_scores = r_vals / sigma
                        p_vals = 2 * norm.sf(np.abs(z_scores))
                    else:
                        z_scores = np.zeros_like(r_vals)
                        p_vals = np.ones_like(r_vals)

                    batch_df["Z_score"] = z_scores
                    batch_df["P_value"] = p_vals

                results_list.append(batch_df)

                # Explicit cleanup for Y
                if not (mode == "symmetric" and i == j):
                    del embeddings_y

            # Explicit cleanup for X
            del embeddings_x
            gc.collect()  # Force memory release before next big load

        # 6. Finalize
        if not results_list:
            return pd.DataFrame(columns=["Feature_1", "Feature_2", "R", "P_value", "P_adj"])

        final_df = pd.concat(results_list, ignore_index=True)

        if return_pval and not final_df.empty:
            self._apply_bh_correction(final_df)

        return final_df.sort_values(by="R", key=abs, ascending=False)
