"""
High-level PatternDetector for irregular 2D spatial coordinates (NUFFT backend).

Mirrors :class:`quadsv.PatternDetector` but does not require a regular grid and
does not build an N×N kernel matrix. Instead it wraps :class:`NUFFTKernel`,
which evaluates quadratic forms ``x^T K x`` and matrix-vector products ``K z``
via type-1/type-2 non-uniform FFTs in ``O(N log N + K log K)`` per feature —
making gene-by-gene Q and R tests tractable on ≥ 10⁵ spots.

Takes an :class:`anndata.AnnData` directly, reads coordinates from
``adata.obsm[spatial_key]`` by default, and exposes ``compute_qstat`` /
``compute_rstat`` methods with the same return DataFrame schema as
:class:`PatternDetector`.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from scipy.stats import chi2, norm
from tqdm import tqdm

from quadsv.nufft import NUFFTKernel
from quadsv.statistics import liu_sf
from quadsv.utils import _apply_bh_correction

__all__ = ["PatternDetectorNUFFT"]

logger = logging.getLogger(__name__)


def _qstat_worker_nufft(
    X_csc: sp.csc_matrix,
    feature_indices: np.ndarray,
    kernel: NUFFTKernel,
    eigenvalues: np.ndarray | None,
    moran_moments: tuple[float, float] | None,
    return_pval: bool,
    feature_names: list[str],
    means: np.ndarray,
    stds: np.ndarray,
    chunk_size: int = 64,
) -> list[dict]:
    """Worker: compute NUFFT Q-stat for a block of features (columns of X_csc).

    p-values use Liu's method on the full eigenvalue spectrum (most kernels)
    or the normal approximation on ``(trace(K), trace(K²))`` for Moran's I.
    """
    results: list[dict] = []
    for start in range(0, len(feature_indices), chunk_size):
        idx = feature_indices[start : start + chunk_size]
        block = np.asarray(X_csc[:, idx].todense())
        block_std = (block - means[idx][None, :]) / np.where(stds[idx] > 0, stds[idx], 1.0)[None, :]
        Q = np.atleast_1d(kernel.xtKx(block_std)).ravel()

        if not return_pval:
            for local_i, global_i in enumerate(idx):
                results.append({"Feature": feature_names[global_i], "Q": float(Q[local_i])})
            continue

        if moran_moments is not None:
            mean_Q, var_Q = moran_moments
            sigma = float(np.sqrt(var_Q))
            zscores = (Q - mean_Q) / sigma if sigma > 1e-12 else np.zeros_like(Q)
            pvals = chi2.sf(zscores**2, df=1)
        else:
            assert eigenvalues is not None
            # eigenvalues passed in are already scaled by N/(ny*nx) by caller.
            pvals = np.array([liu_sf(float(q), eigenvalues) for q in Q])
            trK = float(eigenvalues.sum())
            varQ = 2.0 * float((eigenvalues**2).sum())
            sigma = float(np.sqrt(varQ))
            zscores = (Q - trK) / sigma if sigma > 1e-12 else np.zeros_like(Q)

        for local_i, global_i in enumerate(idx):
            results.append(
                {
                    "Feature": feature_names[global_i],
                    "Q": float(Q[local_i]),
                    "Z_score": float(zscores[local_i]),
                    "P_value": float(pvals[local_i]),
                }
            )
    return results


class PatternDetectorNUFFT:
    """
    Detect spatially variable features on irregular 2D coordinates via NUFFT.

    Parallel API to :class:`quadsv.PatternDetector` and
    :class:`quadsv.PatternDetectorFFT`, but targets the large-N irregular
    regime: Slide-seq, Stereo-seq at full resolution, any spot layout read
    directly from ``adata.obsm['spatial']``. The core kernel is
    :class:`NUFFTKernel`; both Q-test (univariate) and R-test (bivariate) run
    in ``O(N log N)`` per feature.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix. Must have ``adata.obsm[spatial_key]`` populated
        with ``(n_obs, 2)`` coordinates (y, x).
    grid_shape : tuple[int, int], optional
        Internal k-grid resolution for the NUFFT. If ``None`` (default), the
        k-grid is auto-inferred from the spot coordinates alone: size tracks
        the bounding box, spacing tracks the median nearest-neighbor distance.
        Advanced users can override.
    spacing : tuple[float, float], optional
        ``(dy, dx)`` physical spacing per k-grid cell. Auto-inferred with
        ``grid_shape`` when ``None``.
    spatial_key : str, default 'spatial'
        Key in ``adata.obsm`` holding the 2D coordinates.
    coord_order : {'yx', 'xy'}, default 'yx'
        Order of the two columns in ``obsm[spatial_key]``. Visium and Space
        Ranger outputs use ``(y, x)``; many other tools use ``(x, y)``.
    min_cells : int, default 1
        Minimum non-zero spots per feature to include.
    min_cells_frac : float, optional
        If provided, overrides ``min_cells`` with
        ``max(1, int(min_cells_frac * n_obs))``.
    unit_scale : float, default 1.0
        Multiplier converting ``obsm[spatial_key]`` into the same unit as
        ``spacing`` (e.g., 0.35 for full-res pixel coords at 0.35 μm/pixel).

    Attributes
    ----------
    adata : anndata.AnnData
        Reference to the input data.
    n : int
        Number of spots.
    kernel_ : NUFFTKernel or None
        Built by :meth:`build_kernel`; None until then.
    kernel_method_ : str or None
    kernel_params_ : dict or None
    """

    _available_kernels = ["gaussian", "matern", "moran", "graph_laplacian", "car"]

    def __init__(
        self,
        adata: Any,
        grid_shape: tuple[int, int] | None = None,
        spacing: tuple[float, float] | None = None,
        spatial_key: str = "spatial",
        coord_order: str = "yx",
        min_cells: int = 1,
        min_cells_frac: float | None = None,
        unit_scale: float = 1.0,
        oversample: float = 2.0,
    ) -> None:
        if coord_order not in ("yx", "xy"):
            raise ValueError(f"coord_order must be 'yx' or 'xy', got '{coord_order}'.")
        if spatial_key not in adata.obsm:
            raise KeyError(
                f"adata.obsm has no key '{spatial_key}'; available: {list(adata.obsm.keys())}."
            )
        raw_coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
        if raw_coords.ndim != 2 or raw_coords.shape[1] != 2:
            raise ValueError(
                f"adata.obsm['{spatial_key}'] must be (n_obs, 2), got {raw_coords.shape}."
            )

        self.adata = adata
        self.n: int = int(adata.n_obs)
        self.spatial_key: str = spatial_key
        self.coord_order: str = coord_order
        self.unit_scale: float = float(unit_scale)

        # Canonicalize to (y, x).
        if coord_order == "xy":
            raw_coords = raw_coords[:, [1, 0]]
        self._coords: np.ndarray = raw_coords

        if min_cells_frac is not None:
            self.min_cells: int = max(1, int(min_cells_frac * self.n))
        else:
            self.min_cells = min(int(min_cells), self.n)

        # grid_shape / spacing are auto-derived lazily by NUFFTKernel in
        # build_kernel() if still None here. Stored for reproducibility.
        self.grid_shape: tuple[int, int] | None = (
            None if grid_shape is None else (int(grid_shape[0]), int(grid_shape[1]))
        )
        self.spacing: tuple[float, float] | None = (
            None if spacing is None else (float(spacing[0]), float(spacing[1]))
        )
        self.oversample: float = float(oversample)

        self.kernel_: NUFFTKernel | None = None
        self.kernel_method_: str | None = None
        self.kernel_params_: dict | None = None

        logger.info(
            "PatternDetectorNUFFT initialized: n=%d, grid=%s, spacing=%s, unit_scale=%s",
            self.n,
            self.grid_shape if self.grid_shape else "(auto)",
            self.spacing if self.spacing else "(auto)",
            self.unit_scale,
        )

    # ------------------------------------------------------------------
    def build_kernel(
        self,
        method: str = "matern",
        eps: float = 1e-6,
        **kernel_params: Any,
    ) -> PatternDetectorNUFFT:
        """Build the :class:`NUFFTKernel` from the detector's cached coordinates.

        Parameters
        ----------
        method : str, default 'matern'
            Kernel method (``'gaussian'``, ``'matern'``, ``'moran'``,
            ``'graph_laplacian'``, ``'car'``).
        eps : float, default 1e-6
            NUFFT tolerance.
        **kernel_params
            Kernel-specific parameters (``bandwidth``, ``nu``, ``rho``,
            ``neighbor_degree``).

        Returns
        -------
        PatternDetectorNUFFT
            ``self`` for chaining.

        Raises
        ------
        ValueError
            If ``method`` is not one of the available kernels.
        """
        if method not in self._available_kernels:
            raise ValueError(
                f"Unknown kernel method '{method}'. Options: {self._available_kernels}."
            )
        logger.info("Building %s NUFFTKernel over %d spots...", method, self.n)
        self.kernel_ = NUFFTKernel(
            coords=self._coords,
            grid_shape=self.grid_shape,
            spacing=self.spacing,
            method=method,
            unit_scale=self.unit_scale,
            oversample=self.oversample,
            eps=eps,
            **kernel_params,
        )
        # Snap back the auto-derived grid_shape / spacing for reproducibility.
        self.grid_shape = self.kernel_.grid_shape
        self.spacing = self.kernel_.spacing
        self.kernel_method_ = method
        self.kernel_params_ = dict(self.kernel_.params)
        return self

    # ------------------------------------------------------------------
    def _ensure_kernel(self) -> NUFFTKernel:
        if self.kernel_ is None:
            raise RuntimeError("Call .build_kernel() before computing statistics.")
        return self.kernel_

    # ------------------------------------------------------------------
    def _prepare_features(
        self,
        source: str,
        features: list[str] | None,
        layer: str | None,
    ) -> tuple[sp.csc_matrix, list[str], np.ndarray, np.ndarray]:
        """Pull a (n_spots, n_features) CSC matrix + names + per-feature mean/std."""
        if source == "var":
            if layer is None:
                X = self.adata.X
            else:
                X = self.adata.layers[layer]
            if sp.issparse(X):
                X_csc = X.tocsc()
            else:
                X_csc = sp.csc_matrix(X)
            names = list(self.adata.var_names)
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

        if features is not None and source == "var":
            selected = [g for g in features if g in names]
            miss = set(features) - set(selected)
            if miss:
                logger.warning("Requested genes not in adata.var_names: %s", sorted(miss)[:5])
            idx = [names.index(g) for g in selected]
            X_csc = X_csc[:, idx]
            names = selected

        # Per-feature summary stats + filter low-expression.
        X_ct = X_csc  # keep CSC for column slicing
        nnz_per = np.asarray((X_ct != 0).sum(axis=0)).ravel()
        means = np.asarray(X_ct.mean(axis=0)).ravel()
        # std via E[X²] - E[X]² (keep sparse-friendly)
        sq = X_ct.multiply(X_ct)
        sq_mean = np.asarray(sq.mean(axis=0)).ravel()
        var = np.maximum(sq_mean - means**2, 0.0)
        stds = np.sqrt(var)
        keep = (stds > 0) & (nnz_per >= self.min_cells)

        X_kept = X_ct[:, keep]
        names_kept = [names[i] for i, k in enumerate(keep) if k]
        return X_kept, names_kept, means[keep], stds[keep]

    # ------------------------------------------------------------------
    def compute_qstat(
        self,
        source: str = "var",
        features: list[str] | None = None,
        n_jobs: int = -1,
        layer: str | None = None,
        return_pval: bool = True,
        chunk_size: int = 64,
    ) -> pd.DataFrame:
        """Univariate spatial Q-test for many features in parallel.

        p-values use Liu's chi-squared-mixture approximation on the kernel's
        full eigenvalue spectrum (same convention as
        :func:`quadsv.spatial_q_test_fft`) — deterministic, no RNG.

        Parameters
        ----------
        source : {'var', 'obs'}, default 'var'
            ``'var'`` for gene columns (from ``adata.X`` / ``adata.layers[layer]``);
            ``'obs'`` for columns in ``adata.obs``.
        features : list of str, optional
            Subset of feature names. If None, tests every feature in ``source``.
        n_jobs : int, default -1
            Parallel workers for the per-feature chunked loop.
        layer : str, optional
            Layer key when ``source='var'``. Default ``adata.X``.
        return_pval : bool, default True
            If True, compute p-values via Liu's method (or the normal
            approximation for Moran's I, matching the FFT Q-test).
        chunk_size : int, default 64
            Features per worker batch (the NUFFT is batched over them).

        Returns
        -------
        pd.DataFrame
            Columns ``Feature``, ``Q``, ``Z_score``, ``P_value``, ``P_adj``
            (last two omitted when ``return_pval=False``). Sorted by Q descending.

        Raises
        ------
        RuntimeError
            If :meth:`build_kernel` has not been called.
        """
        kernel = self._ensure_kernel()

        # Pre-compute the null-distribution inputs (deterministic, from the
        # kernel's spectrum). Rescale by N/(ny*nx) to target the effective
        # N-point operator — same convention as spatial_q_test_nufft.
        # Moran → normal approximation; otherwise Liu.
        scale = kernel.n / (kernel.grid_shape[0] * kernel.grid_shape[1])
        eigenvalues = None
        moran_moments: tuple[float, float] | None = None
        if return_pval:
            if kernel.method == "moran":
                moran_moments = (
                    kernel.trace() * scale,
                    2.0 * kernel.square_trace() * (scale**2),
                )
            else:
                full = kernel.eigenvalues(return_full=True)
                if full.min() < -0.1:
                    raise ValueError(
                        "Kernel has significant negative eigenvalues; Liu's method may be invalid."
                    )
                eigenvalues = full[full > 1e-9] * scale

        logger.info("Preparing %s features (layer=%s)...", source, layer)
        X_kept, names_kept, means, stds = self._prepare_features(source, features, layer)
        n_feats = len(names_kept)
        if n_feats == 0:
            return pd.DataFrame(columns=["Feature", "Q", "Z_score", "P_value", "P_adj"])
        logger.info("Testing %d features via NUFFT (n_jobs=%s)...", n_feats, n_jobs)

        idx_all = np.arange(n_feats)
        batches = [idx_all[i : i + chunk_size] for i in range(0, n_feats, chunk_size)]
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_qstat_worker_nufft)(
                X_kept,
                batch,
                kernel,
                eigenvalues,
                moran_moments,
                return_pval,
                names_kept,
                means,
                stds,
                chunk_size,
            )
            for batch in tqdm(batches, desc=f"Q (NUFFT, {self.kernel_method_})")
        )
        flat: list[dict] = [row for chunk in results for row in chunk]
        df = pd.DataFrame(flat)
        if return_pval:
            _apply_bh_correction(df)
        df = df.sort_values("Q", ascending=False).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    def compute_rstat(
        self,
        features_x: list[str] | None = None,
        features_y: list[str] | None = None,
        source: str = "var",
        n_jobs: int = -1,
        layer: str | None = None,
        return_pval: bool = True,
        chunk_size: int = 64,
    ) -> pd.DataFrame:
        """Bivariate spatial R-test ``R = x^T K y`` for feature pairs.

        Null variance ``var_R = trace(K²)`` is derived deterministically from
        the kernel's eigenvalue spectrum (same convention as
        :func:`quadsv.spatial_r_test_fft`).

        Parameters
        ----------
        features_x, features_y : list of str, optional
            Feature names. If both None, uses all features in a symmetric
            (features_x × features_x) layout. If only ``features_y`` is given,
            raises; if only ``features_x``, runs symmetric on those.
        source, n_jobs, layer, return_pval, chunk_size
            Same meaning as :meth:`compute_qstat`.

        Returns
        -------
        pd.DataFrame
            Columns ``Feature_1``, ``Feature_2``, ``R``, ``Z_score``,
            ``P_value``, ``P_adj``.

        Raises
        ------
        RuntimeError
            If :meth:`build_kernel` has not been called.
        """
        kernel = self._ensure_kernel()
        if return_pval:
            scale = kernel.n / (kernel.grid_shape[0] * kernel.grid_shape[1])
            var_R = float(kernel.square_trace()) * (scale**2)
        else:
            var_R = 0.0

        if features_x is None and features_y is not None:
            raise ValueError("Provide features_x when features_y is specified.")

        X_kept, names_kept, means, stds = self._prepare_features(source, features_x, layer)
        if features_y is None:
            X_y, names_y, means_y, stds_y = X_kept, names_kept, means, stds
            symmetric = True
        else:
            X_y, names_y, means_y, stds_y = self._prepare_features(source, features_y, layer)
            symmetric = False

        if len(names_kept) == 0 or len(names_y) == 0:
            return pd.DataFrame(
                columns=["Feature_1", "Feature_2", "R", "Z_score", "P_value", "P_adj"]
            )

        logger.info(
            "Testing %d x %d feature pairs via NUFFT (symmetric=%s)...",
            len(names_kept),
            len(names_y),
            symmetric,
        )

        # Standardize X-block (kept in memory).
        X_block = np.asarray(X_kept.todense())
        X_std = (X_block - means[None, :]) / np.where(stds > 0, stds, 1.0)[None, :]

        results: list[dict] = []
        y_chunks = [slice(i, i + chunk_size) for i in range(0, len(names_y), chunk_size)]
        for ysl in tqdm(y_chunks, desc=f"R (NUFFT, {self.kernel_method_})"):
            Y_block = np.asarray(X_y[:, ysl].todense())
            Y_std = (Y_block - means_y[ysl][None, :]) / np.where(stds_y[ysl] > 0, stds_y[ysl], 1.0)[
                None, :
            ]
            KY = kernel.Kz(Y_std)  # (n, n_y_chunk)
            # R_ij = X_std[:, i]^T KY[:, j]
            R_chunk = X_std.T @ KY  # (n_x, n_y_chunk)
            for i, name_x in enumerate(names_kept):
                for local_j, name_y in enumerate(names_y[ysl]):
                    r = float(R_chunk[i, local_j])
                    row = {"Feature_1": name_x, "Feature_2": name_y, "R": r}
                    if return_pval and var_R > 0:
                        z = r / np.sqrt(var_R)
                        row["Z_score"] = z
                        row["P_value"] = float(2.0 * norm.sf(abs(z)))
                    results.append(row)

        df = pd.DataFrame(results)
        if return_pval:
            _apply_bh_correction(df)
        df = df.sort_values("R", ascending=False).reset_index(drop=True)
        return df
