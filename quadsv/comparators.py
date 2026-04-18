"""
High-level wrapper classes for cross-sample spatial-pattern comparison.

The array-level primitives (spectrum compute, radial binning, rotation
alignment, statistical tests) live in :mod:`quadsv.multisample`. This
module only carries the two user-facing classes that wire those primitives
onto concrete container types:

- :class:`ComparatorIrregular` — a sequence of :class:`anndata.AnnData`,
  irregular spots. Spectra are computed with a batched type-1 NUFFT.
- :class:`ComparatorGrid` — a sequence of
  :class:`spatialdata.SpatialData`, regular rasterized bins. Spectra are
  computed with a single batched 2D FFT per sample after calling
  :func:`spatialdata.rasterize_bins` on user-supplied bin / table /
  col-row / value keys (same rasterization recipe as
  :class:`~quadsv.DetectorGrid`).

Both classes share the same post-fit surface (``normalize_background``,
``shape_normalize``, ``residualize``, ``test_pattern``, ``test_expression``,
``benchmark``) via the private :class:`_ComparatorBase` mixin.
"""

from __future__ import annotations

import logging
import warnings
from abc import abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.fft
import scipy.sparse as sp
from joblib import Parallel, delayed
from tqdm.auto import tqdm

# Suppress known deprecation warnings from SpatialData dependencies BEFORE importing them.
warnings.filterwarnings("ignore", category=FutureWarning, message=".*legacy Dask DataFrame.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated.*")

import anndata as _ad
import spatialdata as sd

from quadsv.multisample import (
    _AVAILABLE_STATISTICS,
    _ZSCORE_CLIP,
    align_spectra_by_rotation,
    benchmark_statistics,
    compare_two_groups,
    compare_two_groups_masked,
    compare_two_groups_scalar,
    compute_sample_spectrum,
    normalize_by_background,
    radial_bin_spectrum,
    residualize_against_covariates,
    shape_normalize,
)

__all__ = ["ComparatorIrregular", "ComparatorGrid"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------


class _ComparatorBase:
    """Shared state + shared methods for NUFFT / FFT pattern comparators.

    Concrete subclasses must populate ``self.samples``, ``self.groups``,
    ``self.gene_names``, ``self._grid_shapes``, ``self.spacings``,
    ``self._spectrum_fft_solver``, ``self.feature_mode``, ``self.center``,
    ``self.n_radial_bins``, ``self.fft_solver``, ``self.workers``,
    ``self.freq_edges``, ``self.presence_threshold``,
    ``self.min_samples_per_group`` in their ``__init__``, then implement
    :meth:`_compute_spectra`.
    """

    # --- attribute stubs populated by subclass __init__ ---------------
    samples: list[Any]
    groups: np.ndarray
    gene_names: list[str]
    feature_mode: str
    center: str | None
    n_radial_bins: int
    fft_solver: str  # user-visible choice; see _spectrum_fft_solver for the effective one
    workers: int | None
    freq_edges: np.ndarray | None
    presence_threshold: float
    min_samples_per_group: int
    _spectrum_fft_solver: str
    _grid_shapes: list[tuple[int, int]]
    spacings: list[tuple[float, float]] | None

    # --- populated by :meth:`fit` -------------------------------------
    spectra_: np.ndarray | None = None
    """``(n_samples, n_genes, K)`` feature matrix after radial binning / 2D flatten."""
    dc_: np.ndarray | None = None
    """``(n_samples, n_genes)`` per-sample per-gene DC scalars (grid means)."""
    presence_: np.ndarray | None = None
    """``(n_samples, n_genes)`` boolean mask — True = gene cleared
    ``presence_threshold`` in that sample."""
    rotation_angles_: np.ndarray | None = None
    """Per-sample rotation angles (degrees), populated when
    ``feature_mode='2d'`` and :meth:`fit` has been called."""

    _raw_2d_spectra: list[np.ndarray] | None = None

    # ------------------------------------------------------------------
    @abstractmethod
    def _compute_spectra(
        self, n_jobs: int, progress: bool
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        """Compute per-sample 2D spectra + DC + presence mask.

        Implemented by each backend. Returns ``(raw_2d, dc, presence)`` where
        ``raw_2d`` is a list of ``(n_genes, ny, n_kx)`` spectra (layout
        determined by :attr:`_spectrum_fft_solver`), ``dc`` is a
        ``(n_samples, n_genes)`` float array, and ``presence`` is a
        ``(n_samples, n_genes)`` boolean mask.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    def fit(
        self,
        n_jobs: int = -1,
        landmark_genes: Sequence[str] | None = None,
        progress: bool = True,
    ) -> _ComparatorBase:
        """
        Compute per-sample power spectra and (if ``feature_mode='2d'``) rotation-align.

        Parameters
        ----------
        n_jobs : int, default -1
            Parallelism over samples for the per-sample spectrum pass. When
            ``progress=True`` the outer loop is sequential (so the tqdm bar is
            accurate); finufft / scipy.fft are multi-threaded internally via
            OpenMP so this rarely loses in practice.
        landmark_genes : sequence of str, optional
            Only used in ``feature_mode='2d'``. Names of genes (matched against
            :attr:`gene_names`) whose spectra define the rotation-alignment
            landmarks. Recovered rotations are still applied to every gene in
            :attr:`gene_names`. If None (default), every gene is used as a
            landmark.
        progress : bool, default True
            Show tqdm progress bars over the three phases (spectrum compute,
            optional rotation alignment, radial binning).

        Returns
        -------
        self
        """
        logger.info(
            "Computing per-sample spectra (n_samples=%d, center=%s)...",
            len(self._grid_shapes),
            self.center,
        )
        self._raw_2d_spectra, self.dc_, self.presence_ = self._compute_spectra(
            n_jobs=n_jobs, progress=progress
        )

        if self.feature_mode == "2d":
            # Pick landmarks (defaults to every gene).
            if landmark_genes is not None:
                name_to_idx = {g: i for i, g in enumerate(self.gene_names)}
                missing = [g for g in landmark_genes if g not in name_to_idx]
                if missing:
                    raise KeyError(f"landmark_genes not in gene_names: {missing}")
                lm_idx = np.asarray([name_to_idx[g] for g in landmark_genes], dtype=int)
                landmark_spectra = [s[lm_idx] for s in self._raw_2d_spectra]
            else:
                landmark_spectra = self._raw_2d_spectra
            aligned, angles = align_spectra_by_rotation(
                landmark_spectra,
                grid_shapes=self._grid_shapes,
                target_spectra=self._raw_2d_spectra,
                fft_solver=self._spectrum_fft_solver,
                progress=progress,
            )
            self._raw_2d_spectra = aligned
            self.rotation_angles_ = angles

        # Build a common bin-edge grid for physical-frequency mode.
        if self.feature_mode == "radial" and self.spacings is not None and self.freq_edges is None:
            nyquists = [1.0 / (2.0 * max(dy, dx)) for (dy, dx) in self.spacings]
            f_max = float(min(nyquists))
            self.freq_edges = np.linspace(0.0, f_max * (1.0 + 1e-9), self.n_radial_bins + 1)
            logger.info(
                "Auto-generated %d radial bins on [0, %.4g] cycles per unit length.",
                self.n_radial_bins,
                f_max,
            )

        # Reduce to per-sample feature matrices of shape (n_genes, K) and stack.
        feats: list[np.ndarray] = []
        iter_samples: Any = enumerate(zip(self._raw_2d_spectra, self._grid_shapes, strict=False))
        if progress:
            iter_samples = tqdm(
                iter_samples, total=len(self._raw_2d_spectra), desc="Radial binning"
            )
        for i, (spec_i, shape) in iter_samples:
            if self.feature_mode == "radial":
                spacing_i = self.spacings[i] if self.spacings is not None else None
                f = radial_bin_spectrum(
                    spec_i,
                    grid_shape=shape,
                    n_bins=self.n_radial_bins,
                    fft_solver=self._spectrum_fft_solver,
                    spacing=spacing_i,
                    edges=self.freq_edges,
                )
            else:
                ny, nx = shape
                k = min(self.n_radial_bins, ny // 2, nx // 2)
                low = spec_i[:, :k, :k] if spec_i.shape[-1] > k else spec_i[:, :k, :]
                f = low.reshape(low.shape[0], -1)
            feats.append(f)
        K = min(f.shape[-1] for f in feats)
        feats = [f[..., :K] for f in feats]
        self.spectra_ = np.stack(feats, axis=0)
        return self

    # ------------------------------------------------------------------
    # Post-fit transforms
    # ------------------------------------------------------------------
    def normalize_background(self) -> _ComparatorBase:
        """Apply per-sample geometric-mean background normalization in place."""
        if self.spectra_ is None:
            raise RuntimeError("Call .fit() before .normalize_background().")
        for i in range(self.spectra_.shape[0]):
            self.spectra_[i] = normalize_by_background(self.spectra_[i])
        return self

    def shape_normalize(self) -> _ComparatorBase:
        """Rescale each ``(sample, gene)`` spectrum to sum-1 along frequency."""
        if self.spectra_ is None:
            raise RuntimeError("Call .fit() before .shape_normalize().")
        self.spectra_ = shape_normalize(self.spectra_, axis=-1)
        return self

    def residualize(self, covariates: Sequence[np.ndarray]) -> _ComparatorBase:
        """Regress out per-sample covariate spectra from :attr:`spectra_`.

        ``covariates[i]`` should be a ``(n_covariates, ny_i, nx_i)`` array
        matching :attr:`_grid_shapes[i]`.
        """
        if self.spectra_ is None:
            raise RuntimeError("Call .fit() before .residualize().")
        if len(covariates) != len(self.samples):
            raise ValueError(
                f"covariates length {len(covariates)} != n_samples {len(self.samples)}."
            )
        for i, cov in enumerate(covariates):
            if cov.ndim != 3:
                raise ValueError(f"covariate sample {i} must be 3D, got {cov.shape}.")
            cov_2d = compute_sample_spectrum(
                cov, fft_solver=self._spectrum_fft_solver, workers=self.workers
            )
            # Use the covariate's own raster shape — for the NUFFT path the
            # sample's internal k-grid (self._grid_shapes[i]) is auto-inferred
            # and may differ from the covariate raster. ``freq_edges`` (shared
            # across samples when ``feature_mode='radial'``) is what aligns the
            # bins, not grid_shape.
            cov_shape = cov.shape[-2:]
            spacing = self.spacings[i] if self.spacings is not None else None
            if self.feature_mode == "radial":
                cov_feat = radial_bin_spectrum(
                    cov_2d,
                    grid_shape=cov_shape,
                    n_bins=self.n_radial_bins,
                    fft_solver=self._spectrum_fft_solver,
                    spacing=spacing,
                    edges=self.freq_edges,
                )
            else:
                ny, nx = cov_shape
                k = min(self.n_radial_bins, ny // 2, nx // 2)
                low = cov_2d[:, :k, :k] if cov_2d.shape[-1] > k else cov_2d[:, :k, :]
                cov_feat = low.reshape(low.shape[0], -1)
            cov_feat = cov_feat[..., : self.spectra_.shape[-1]]
            self.spectra_[i] = residualize_against_covariates(self.spectra_[i], cov_feat)
        return self

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------
    def test_pattern(
        self,
        statistic: str = "log_l2",
        n_perm: int = 1000,
        random_state: int | None = None,
        freq_weights: np.ndarray | None = None,
        n_perm_max: int = 10000,
    ) -> Any:
        """Two-group spectral-pattern test on :attr:`spectra_`.

        Dispatches to :func:`quadsv.multisample.compare_two_groups_masked`
        when any ``(sample, gene)`` pair is marked absent in :attr:`presence_`
        (e.g. when ``presence_threshold > 0``), otherwise to
        :func:`~quadsv.multisample.compare_two_groups`.
        """
        if self.spectra_ is None:
            raise RuntimeError("Call .fit() before .test_pattern().")
        use_masked = self.presence_ is not None and not self.presence_.all()
        if use_masked:
            return compare_two_groups_masked(
                self.spectra_,
                self.groups,
                self.presence_,
                gene_names=self.gene_names,
                statistic=statistic,
                n_perm=n_perm,
                random_state=random_state,
                min_samples_per_group=self.min_samples_per_group,
                freq_weights=freq_weights,
                n_perm_max=n_perm_max,
            )
        return compare_two_groups(
            self.spectra_,
            self.groups,
            gene_names=self.gene_names,
            statistic=statistic,
            n_perm=n_perm,
            random_state=random_state,
            freq_weights=freq_weights,
            n_perm_max=n_perm_max,
        )

    test = test_pattern

    def test_expression(
        self,
        n_perm: int = 1000,
        random_state: int | None = None,
        n_perm_max: int = 10000,
    ) -> Any:
        """Classical DE test on the DC component (per-sample per-gene grid mean)."""
        if self.dc_ is None:
            raise RuntimeError("Call .fit() before .test_expression().")
        return compare_two_groups_scalar(
            self.dc_,
            self.groups,
            gene_names=self.gene_names,
            n_perm=n_perm,
            random_state=random_state,
            n_perm_max=n_perm_max,
        )

    def benchmark(
        self,
        statistics: Sequence[str] = _AVAILABLE_STATISTICS,
        n_perm: int = 1000,
        random_state: int | None = None,
        n_perm_max: int = 10000,
    ) -> dict[str, Any]:
        """Run :func:`benchmark_statistics` on :attr:`spectra_`."""
        if self.spectra_ is None:
            raise RuntimeError("Call .fit() before .benchmark().")
        return benchmark_statistics(
            self.spectra_,
            self.groups,
            gene_names=self.gene_names,
            statistics=statistics,
            n_perm=n_perm,
            random_state=random_state,
            n_perm_max=n_perm_max,
        )


# ---------------------------------------------------------------------------
# Shared input-validation helpers
# ---------------------------------------------------------------------------


def _validate_groups(groups: np.ndarray, n_samples: int) -> np.ndarray:
    groups = np.asarray(groups)
    if groups.shape != (n_samples,):
        raise ValueError(f"groups length {groups.shape} does not match n_samples={n_samples}.")
    if np.unique(groups).size != 2:
        raise ValueError("groups must contain exactly two distinct labels.")
    return groups


def _validate_common(
    center: str | None,
    feature_mode: str,
    fft_solver: str,
    presence_threshold: float,
    min_samples_per_group: int,
) -> str:
    if center not in ("mean", "zscore", None):
        raise ValueError(f"center must be 'mean', 'zscore', or None, got {center!r}.")
    if feature_mode not in ("radial", "2d"):
        raise ValueError(f"feature_mode must be 'radial' or '2d', got '{feature_mode}'.")
    if feature_mode == "2d" and fft_solver != "fft2":
        logger.info("feature_mode='2d' works best with fft_solver='fft2'; switching automatically.")
        fft_solver = "fft2"
    if not 0.0 <= float(presence_threshold) <= 1.0:
        raise ValueError(f"presence_threshold must be in [0, 1], got {presence_threshold}.")
    if int(min_samples_per_group) < 2:
        raise ValueError(f"min_samples_per_group must be >= 2, got {min_samples_per_group}.")
    return fft_solver


# ---------------------------------------------------------------------------
# ComparatorIrregular — AnnData / irregular spots
# ---------------------------------------------------------------------------


class ComparatorIrregular(_ComparatorBase):
    """
    Cross-sample pattern comparison on irregular spots via NUFFT.

    Accepts a list of :class:`anndata.AnnData` (one per sample). For each
    sample, the per-sample ``obsm[coordinates_key]`` supplies the irregular
    ``(y, x)`` coordinates and ``.X`` (or ``.layers[layer]`` when set) is the
    expression matrix. Spectra are evaluated with a batched type-1 NUFFT
    (``finufft.nufft2d1``), densifying at most :attr:`nufft_chunk_size`
    columns of ``.X`` at a time so the full slab is never materialized.

    Parameters
    ----------
    samples : sequence of :class:`anndata.AnnData`
    groups : np.ndarray
        Two-label group vector, length ``len(samples)``.
    gene_names : sequence of str, optional
        If None, inferred from the first sample; every other sample must share
        the same ``var_names``.
    feature_mode : {'radial', '2d'}, default 'radial'
    n_radial_bins : int, default 30
    coordinates_key : str, default 'spatial'
    layer : str, optional
    unit_scales : sequence of float, optional
        Per-sample multiplier applied to coords before NUFFT (e.g. pixels→μm).
    grid_shape, spacing : optional
        When both given, used for every sample. Otherwise each sample's
        k-grid is auto-inferred from coords via
        :func:`quadsv.nufft._infer_grid_from_coords`.
    freq_edges : np.ndarray, optional
    center : {'mean', 'zscore', None}, default 'mean'
    eps : float, default 1e-6
        NUFFT tolerance.
    presence_threshold : float, default 0.0
        Minimum fraction of non-zero spots for a gene to count as "observed"
        in a sample (feeds :attr:`presence_` and, transitively, the masked
        pattern test).
    min_samples_per_group : int, default 2
    nufft_chunk_size : int, default 64
        Number of genes per batched NUFFT call. 32–128 balances finufft's
        per-call overhead against the `(n_spots, chunk)` transient RAM.
    workers : int, optional
        Forwarded to per-sample FFTs used by :meth:`residualize`.
    """

    def __init__(
        self,
        samples: Sequence[Any],
        groups: np.ndarray,
        gene_names: Sequence[str] | None = None,
        *,
        feature_mode: str = "radial",
        n_radial_bins: int = 30,
        coordinates_key: str = "spatial",
        layer: str | None = None,
        unit_scales: Sequence[float] | None = None,
        grid_shape: tuple[int, int] | None = None,
        spacing: tuple[float, float] | None = None,
        freq_edges: np.ndarray | None = None,
        center: str | None = "mean",
        eps: float = 1e-6,
        presence_threshold: float = 0.0,
        min_samples_per_group: int = 2,
        nufft_chunk_size: int = 64,
        workers: int | None = None,
    ) -> None:
        fft_solver = _validate_common(
            center, feature_mode, "fft2", presence_threshold, min_samples_per_group
        )
        samples_list = list(samples)
        if len(samples_list) == 0:
            raise ValueError("samples must be a non-empty list.")
        for i, s in enumerate(samples_list):
            if not isinstance(s, _ad.AnnData):
                raise TypeError(f"sample {i} is {type(s).__name__}, expected anndata.AnnData.")

        groups = _validate_groups(groups, len(samples_list))
        resolved = _resolve_anndata_gene_names(samples_list, gene_names, layer=layer)

        self.samples = samples_list
        self.groups = groups
        self.gene_names = list(resolved)
        self.feature_mode = feature_mode
        self.n_radial_bins = int(n_radial_bins)
        self.fft_solver = fft_solver
        self.center = center
        self.workers = workers
        self.freq_edges = None if freq_edges is None else np.asarray(freq_edges, dtype=float)
        self.presence_threshold = float(presence_threshold)
        self.min_samples_per_group = int(min_samples_per_group)
        self.nufft_chunk_size = max(1, int(nufft_chunk_size))
        # NUFFT always produces full-2D layout (fft2), regardless of user's
        # ``fft_solver`` (which is moot here).
        self._spectrum_fft_solver = "fft2"

        self._layer = layer
        self._coordinates_key = coordinates_key
        self._nufft_eps = float(eps)

        # Per-sample coords / grids.
        from quadsv.nufft import _infer_grid_from_coords

        if unit_scales is None:
            unit_scales = [1.0] * len(samples_list)
        if len(unit_scales) != len(samples_list):
            raise ValueError(
                f"unit_scales length {len(unit_scales)} does not match "
                f"n_samples={len(samples_list)}."
            )
        self._unit_scales: list[float] = [float(s) for s in unit_scales]

        coords_list: list[np.ndarray] = []
        grids: list[tuple[int, int]] = []
        spacings: list[tuple[float, float]] = []
        for i, ad_s in enumerate(samples_list):
            if coordinates_key not in ad_s.obsm:
                raise KeyError(
                    f"sample {i} has no obsm['{coordinates_key}']; "
                    f"available: {list(ad_s.obsm.keys())}."
                )
            c = np.asarray(ad_s.obsm[coordinates_key], dtype=np.float64)
            if c.ndim != 2 or c.shape[1] != 2:
                raise ValueError(
                    f"sample {i} obsm['{coordinates_key}'] must be (N, 2), got {c.shape}."
                )
            coords_list.append(c)
            if grid_shape is not None and spacing is not None:
                gs_i = (int(grid_shape[0]), int(grid_shape[1]))
                sp_i = (float(spacing[0]), float(spacing[1]))
            else:
                gs_i, sp_i = _infer_grid_from_coords(c * self._unit_scales[i], oversample=2.0)
            grids.append(gs_i)
            spacings.append(sp_i)
        self._coords = coords_list
        self._grid_shapes = grids
        self.spacings = spacings

    # ------------------------------------------------------------------
    def _compute_spectra(  # noqa: C901
        self, n_jobs: int, progress: bool
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        from quadsv.nufft import power_spectrum_2d_nufft

        chunk_size = self.nufft_chunk_size
        n_samples_total = len(self.samples)

        def _one(i: int, pbar: tqdm | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            adata = self.samples[i]
            pts = self._coords[i]
            scale = self._unit_scales[i]
            grid_i = self._grid_shapes[i]
            spacing_i = self.spacings[i]

            X_src = adata.X if self._layer is None else adata.layers[self._layer]
            n_genes = len(self.gene_names)
            n_spots = X_src.shape[0]

            if sp.issparse(X_src):
                dc = np.asarray(X_src.mean(axis=0)).ravel()
                nnz_per = np.asarray((X_src != 0).sum(axis=0)).ravel()
                if self.center == "zscore":
                    sq = X_src.multiply(X_src)
                    sq_mean = np.asarray(sq.mean(axis=0)).ravel()
                    sd_arr = np.sqrt(np.maximum(sq_mean - dc**2, 0.0))
                else:
                    sd_arr = None
                X_csc = X_src.tocsc()
                X_dense = None
            else:
                X_dense = np.asarray(X_src, dtype=np.float64)
                dc = X_dense.mean(axis=0)
                nnz_per = (X_dense != 0).sum(axis=0)
                sd_arr = X_dense.std(axis=0) if self.center == "zscore" else None
                X_csc = None

            presence_i = (nnz_per / max(n_spots, 1)) >= self.presence_threshold

            if self.center == "zscore":
                positive = sd_arr[sd_arr > 0] if sd_arr is not None else np.empty(0)
                sd_floor = float(np.median(positive)) * 0.1 if positive.size else 1.0
                sd_safe = np.maximum(sd_arr, sd_floor) if sd_arr is not None else None
            else:
                sd_safe = None

            ny, nx = grid_i
            spec_stack = np.empty((n_genes, ny, nx), dtype=np.float64)

            for start in range(0, n_genes, chunk_size):
                stop = min(start + chunk_size, n_genes)
                cols = slice(start, stop)
                if X_csc is not None:
                    block = np.asarray(X_csc[:, cols].toarray(), dtype=np.float64)
                else:
                    block = X_dense[:, cols].astype(np.float64, copy=True)

                if self.center == "mean":
                    block -= dc[None, cols]
                elif self.center == "zscore":
                    block = (block - dc[None, cols]) / sd_safe[None, cols]
                    np.clip(block, -_ZSCORE_CLIP, _ZSCORE_CLIP, out=block)

                p_chunk = power_spectrum_2d_nufft(
                    pts,
                    block,
                    grid_shape=grid_i,
                    spacing=spacing_i,
                    unit_scale=scale,
                    eps=self._nufft_eps,
                    center_coords=True,
                )
                spec_stack[start:stop] = np.moveaxis(p_chunk, -1, 0)
                if pbar is not None:
                    pbar.update(1)

            return spec_stack, dc, presence_i

        return _run_per_sample(
            _one,
            n_samples_total,
            n_chunks_per_sample=int(np.ceil(len(self.gene_names) / chunk_size)),
            desc="NUFFT spectra (per-gene chunks)",
            n_jobs=n_jobs,
            progress=progress,
        )


# ---------------------------------------------------------------------------
# ComparatorGrid — SpatialData / regular bins via rasterize_bins
# ---------------------------------------------------------------------------


class ComparatorGrid(_ComparatorBase):
    """
    Cross-sample pattern comparison on regular bins via FFT + SpatialData.

    Accepts a list of :class:`spatialdata.SpatialData` (one per sample). For
    each sample, :func:`spatialdata.rasterize_bins` turns the designated bin
    shape + table into a dense ``(n_genes, ny, nx)`` image, which is then fed
    to the batched 2D FFT. All samples are expected to share the same
    rasterization schema (``bins`` / ``table_name`` / ``col_key`` / ``row_key``
    / ``value_key``) — this mirrors :class:`~quadsv.DetectorGrid`.

    Parameters
    ----------
    samples : sequence of :class:`spatialdata.SpatialData`
    groups : np.ndarray
    bins : str
        SpatialElement key for the bin shapes in each ``sdata``.
    table_name : str
        Table key in each ``sdata.tables``.
    col_key, row_key : str
        Column / row-index columns in the table's ``.obs``.
    value_key : str, optional
        Expression column in ``.obs``; defaults to ``None`` (rasterizes counts
        / presence directly off ``.X``).
    gene_names : sequence of str, optional
        If None, inferred from the first sample's table. All samples must
        share ``var_names``.
    feature_mode, n_radial_bins, fft_solver, workers, freq_edges, center,
    presence_threshold, min_samples_per_group : see parent.
    fft_chunk_size : int, default 256
        Genes per batched ``scipy.fft`` call on the rasterized block. Keeps
        transient memory bounded at ``O(ny · nx · chunk · 8 B)``. The raster
        itself is still built once per sample (full ``(n_genes, ny, nx)``
        footprint is unavoidable on SpatialData).
    """

    def __init__(
        self,
        samples: Sequence[Any],
        groups: np.ndarray,
        *,
        bins: str,
        table_name: str,
        col_key: str,
        row_key: str,
        value_key: str | None = None,
        gene_names: Sequence[str] | None = None,
        feature_mode: str = "radial",
        n_radial_bins: int = 30,
        fft_solver: str = "rfft2",
        workers: int | None = None,
        spacing: tuple[float, float] | None = None,
        freq_edges: np.ndarray | None = None,
        center: str | None = "mean",
        presence_threshold: float = 0.0,
        min_samples_per_group: int = 2,
        fft_chunk_size: int = 256,
    ) -> None:
        fft_solver = _validate_common(
            center, feature_mode, fft_solver, presence_threshold, min_samples_per_group
        )
        samples_list = list(samples)
        if len(samples_list) == 0:
            raise ValueError("samples must be a non-empty list.")
        for i, s in enumerate(samples_list):
            if not isinstance(s, sd.SpatialData):
                raise TypeError(
                    f"sample {i} is {type(s).__name__}, expected spatialdata.SpatialData."
                )

        groups = _validate_groups(groups, len(samples_list))
        resolved = _resolve_spatialdata_gene_names(samples_list, gene_names, table_name)

        self.samples = samples_list
        self.groups = groups
        self.gene_names = list(resolved)
        self.feature_mode = feature_mode
        self.n_radial_bins = int(n_radial_bins)
        self.fft_solver = fft_solver
        self.center = center
        self.workers = workers
        self.freq_edges = None if freq_edges is None else np.asarray(freq_edges, dtype=float)
        self.presence_threshold = float(presence_threshold)
        self.min_samples_per_group = int(min_samples_per_group)
        self.fft_chunk_size = max(1, int(fft_chunk_size))
        self._spectrum_fft_solver = fft_solver

        self._bins = bins
        self._table_name = table_name
        self._col_key = col_key
        self._row_key = row_key
        self._value_key = value_key

        # Grid shape is determined per-sample at fit time by rasterize_bins
        # (the raster's .shape[-2:] carries it). Record the placeholder now
        # and fill in during _compute_spectra; spacing is always (1.0, 1.0)
        # because rasterize_bins outputs one pixel per bin — users can
        # override via the spacing kwarg when the bins encode a physical
        # pitch.
        self._spacing_override = None if spacing is None else (float(spacing[0]), float(spacing[1]))
        self._grid_shapes = []  # populated by _compute_spectra
        self.spacings = None  # populated alongside

    # ------------------------------------------------------------------
    def _rasterize_one(self, sdata: Any) -> np.ndarray:
        """Wrap :func:`spatialdata.rasterize_bins`. Returns a
        ``(n_genes, ny, nx)`` float array in :attr:`gene_names` order.
        """
        from quadsv._rasterize import rasterize_table

        table = sdata.tables[self._table_name]
        img = rasterize_table(
            sdata,
            bins=self._bins,
            table_name=self._table_name,
            col_key=self._col_key,
            row_key=self._row_key,
            value_key=self._value_key,
            return_region_as_labels=False,
        )
        arr = np.asarray(img.data if hasattr(img, "data") else img, dtype=np.float64)
        # Expected shape: (n_genes_in_table, ny, nx). Reindex gene axis to
        # self.gene_names (which was validated to match table var_names).
        if arr.ndim != 3:
            raise ValueError(
                f"rasterize_bins returned shape {arr.shape}, expected (n_genes, ny, nx)."
            )
        table_names = list(table.var_names)
        if table_names == list(self.gene_names):
            return arr
        idx = np.asarray([table_names.index(g) for g in self.gene_names], dtype=int)
        return arr[idx]

    def _compute_spectra(
        self, n_jobs: int, progress: bool
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        chunk = self.fft_chunk_size

        def _one(i: int, pbar: tqdm | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            raster = self._rasterize_one(self.samples[i])
            n_genes, ny, nx = raster.shape
            # Lock grid shape + spacing from the first-seen rasterization.
            self._grid_shapes_local_i = (ny, nx)

            frac_nonzero = (raster != 0).reshape(n_genes, -1).mean(axis=1)
            presence_i = frac_nonzero >= self.presence_threshold
            dc = raster.mean(axis=(1, 2))

            expected_kx = nx if self._spectrum_fft_solver == "fft2" else nx // 2 + 1
            spec_stack = np.empty((n_genes, ny, expected_kx), dtype=np.float64)
            for start in range(0, n_genes, chunk):
                stop = min(start + chunk, n_genes)
                # Reuse the shared helper; it applies the sparse-gene zscore
                # guard identically to the NUFFT path.
                spec_chunk = compute_sample_spectrum(
                    raster[start:stop],
                    fft_solver=self._spectrum_fft_solver,
                    workers=self.workers,
                    center=self.center,
                    return_dc=False,
                )
                spec_stack[start:stop] = spec_chunk
                if pbar is not None:
                    pbar.update(1)

            return spec_stack, dc, presence_i

        # Run the per-sample loop and collect the grid shapes (only known
        # after rasterize_bins returns).
        n_samples_total = len(self.samples)
        raw_2d: list[np.ndarray | None] = [None] * n_samples_total
        dc_list: list[np.ndarray | None] = [None] * n_samples_total
        pres_list: list[np.ndarray | None] = [None] * n_samples_total
        grids: list[tuple[int, int]] = []

        run_sequential = progress or n_jobs == 1
        if run_sequential:
            n_chunks_total = sum(
                int(np.ceil(len(self.gene_names) / chunk)) for _ in range(n_samples_total)
            )
            pbar: tqdm | None = (
                tqdm(total=n_chunks_total, desc="FFT spectra (per-gene chunks)")
                if progress
                else None
            )
            for i in range(n_samples_total):
                if pbar is not None:
                    pbar.set_postfix_str(f"sample {i + 1}/{n_samples_total}")
                r0, r1, r2 = _one(i, pbar=pbar)
                raw_2d[i] = r0
                dc_list[i] = r1
                pres_list[i] = r2
                grids.append(self._grid_shapes_local_i)
            if pbar is not None:
                pbar.close()
        else:
            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_one)(i) for i in range(n_samples_total)
            )
            for i, r in enumerate(results):
                raw_2d[i], dc_list[i], pres_list[i] = r
                # When running via joblib the `self._grid_shapes_local_i`
                # side-channel isn't reliable — infer from the returned spec.
                grids.append(raw_2d[i].shape[-2:])

        # Record per-sample grids + spacings for downstream radial binning.
        self._grid_shapes = grids
        if self._spacing_override is not None:
            self.spacings = [self._spacing_override] * len(grids)
        else:
            self.spacings = [(1.0, 1.0)] * len(grids)
        del self._grid_shapes_local_i  # tidy attr soup

        dc = np.stack([np.asarray(x) for x in dc_list], axis=0)
        presence = np.stack([np.asarray(x) for x in pres_list], axis=0)
        return [np.asarray(x) for x in raw_2d], dc, presence


# ---------------------------------------------------------------------------
# shared per-sample runner
# ---------------------------------------------------------------------------


def _run_per_sample(
    worker: Any,
    n_samples_total: int,
    *,
    n_chunks_per_sample: int,
    desc: str,
    n_jobs: int,
    progress: bool,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Invoke ``worker(i, pbar)`` for each sample with a shared tqdm bar.

    Used by :class:`ComparatorIrregular` where each sample is split into
    multiple per-gene-chunk tqdm ticks.
    """
    raw_2d: list[np.ndarray | None] = [None] * n_samples_total
    dc_list: list[np.ndarray | None] = [None] * n_samples_total
    pres_list: list[np.ndarray | None] = [None] * n_samples_total

    run_sequential = progress or n_jobs == 1
    if run_sequential:
        n_total = n_samples_total * n_chunks_per_sample
        pbar: tqdm | None = tqdm(total=n_total, desc=desc) if progress else None
        for i in range(n_samples_total):
            if pbar is not None:
                pbar.set_postfix_str(f"sample {i + 1}/{n_samples_total}")
            r0, r1, r2 = worker(i, pbar)
            raw_2d[i] = r0
            dc_list[i] = r1
            pres_list[i] = r2
        if pbar is not None:
            pbar.close()
    else:
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(worker)(i, None) for i in range(n_samples_total)
        )
        for i, r in enumerate(results):
            raw_2d[i], dc_list[i], pres_list[i] = r

    dc = np.stack([np.asarray(x) for x in dc_list], axis=0)
    presence = np.stack([np.asarray(x) for x in pres_list], axis=0)
    return [np.asarray(x) for x in raw_2d], dc, presence


# ---------------------------------------------------------------------------
# Gene-name resolution helpers
# ---------------------------------------------------------------------------


def _resolve_anndata_gene_names(
    samples: list[Any],
    gene_names: Sequence[str] | None,
    *,
    layer: str | None,
) -> list[str]:
    first = samples[0]
    if gene_names is None:
        gene_names = list(first.var_names)
    for i, s in enumerate(samples):
        if list(s.var_names) != list(gene_names):
            raise ValueError(
                f"sample {i} has var_names that do not match the reference "
                "(all AnnData samples must share the same gene axis)."
            )
        if layer is not None and layer not in s.layers:
            raise KeyError(f"sample {i} is missing layer '{layer}'.")
    return list(gene_names)


def _resolve_spatialdata_gene_names(
    samples: list[Any],
    gene_names: Sequence[str] | None,
    table_name: str,
) -> list[str]:
    first = samples[0]
    if table_name not in first.tables:
        raise KeyError(
            f"sample 0 has no table '{table_name}'; available: {list(first.tables.keys())}."
        )
    if gene_names is None:
        gene_names = list(first.tables[table_name].var_names)
    for i, s in enumerate(samples):
        if table_name not in s.tables:
            raise KeyError(f"sample {i} has no table '{table_name}'.")
        tbl_names = list(s.tables[table_name].var_names)
        if tbl_names != list(gene_names):
            raise ValueError(f"sample {i}'s table has var_names that do not match the reference.")
    return list(gene_names)


# `scipy.fft` import keeps ``compute_sample_spectrum`` fast on SpatialData path.
_ = scipy.fft  # quiet flake8 about unused import
