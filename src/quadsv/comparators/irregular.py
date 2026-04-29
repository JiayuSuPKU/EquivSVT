"""
:class:`ComparatorIrregular` — cross-sample pattern comparison on a
list of :class:`anndata.AnnData` (irregular spots, NUFFT backend).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import anndata as _ad
import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from quadsv.comparators.base import (
    _ComparatorBase,
    _validate_common,
    _validate_groups_or_design,
)

__all__ = ["ComparatorIrregular"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ComparatorIrregular — AnnData / irregular spots
# ---------------------------------------------------------------------------


class ComparatorIrregular(_ComparatorBase):
    """
    Cross-sample pattern comparison on irregular spots via NUFFT.

    Accepts a list of :class:`anndata.AnnData` (one per sample). For each
    sample, the per-sample ``obsm[obsm_key]`` supplies the irregular
    ``(y, x)`` coordinates and ``.X`` (or ``.layers[layer]`` when set) is the
    expression matrix. Spectra are evaluated with a batched type-1 NUFFT
    (``finufft.nufft2d1``), densifying at most :attr:`nufft_chunk_size`
    columns of ``.X`` at a time so the full slab is never materialized.

    Parameters
    ----------
    samples : sequence of :class:`anndata.AnnData`
    groups : np.ndarray, optional
        Two-label group vector, length ``len(samples)``. Pass exactly one
        of ``groups`` or ``design``.
    design : pd.DataFrame or np.ndarray, optional
        Sample-level design matrix for the GLM Wald test path. A DataFrame
        is auto-encoded via patsy (``~ <every column>`` formula, with
        treatment-coded categoricals + intercept); a numpy array of shape
        ``(n_samples, p)`` is used as-is. Pair with
        ``test_pattern(contrast=..., null="wald")`` to test a 1-DOF linear
        contrast. The two-group ``groups=`` API is preserved for
        back-compat and uses the same Wald math under the hood when
        called with ``null="wald"``.
    gene_names : sequence of str, optional
        If None, inferred from the first sample; every other sample must share
        the same ``var_names``.
    feature_mode : {'radial', '2d'}, default 'radial'
    n_radial_bins : int, default 30
    obsm_key : str, default 'spatial'
    layer : str, optional
    unit_scales : sequence of float, optional
        Per-sample multiplier applied to coords before NUFFT (e.g. pixels→μm).
    grid_shape, spacing : optional
        When both given, used for every sample. Otherwise each sample's
        k-grid is auto-inferred from coords via
        :func:`quadsv.kernels.nufft._infer_grid_from_coords`.
    freq_edges : np.ndarray, optional
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
        groups: np.ndarray | None = None,
        gene_names: Sequence[str] | None = None,
        *,
        design: "Any | None" = None,
        feature_mode: str = "radial",
        n_radial_bins: int = 30,
        obsm_key: str = "spatial",
        layer: str | None = None,
        unit_scales: Sequence[float] | None = None,
        grid_shape: tuple[int, int] | None = None,
        spacing: tuple[float, float] | None = None,
        freq_edges: np.ndarray | None = None,
        eps: float = 1e-6,
        presence_threshold: float = 0.0,
        min_samples_per_group: int = 2,
        nufft_chunk_size: int = 64,
        workers: int | None = None,
    ) -> None:
        fft_solver = _validate_common(
            feature_mode, "fft2", presence_threshold, min_samples_per_group
        )
        samples_list = list(samples)
        if len(samples_list) == 0:
            raise ValueError("samples must be a non-empty list.")
        for i, s in enumerate(samples_list):
            if not isinstance(s, _ad.AnnData):
                raise TypeError(f"sample {i} is {type(s).__name__}, expected anndata.AnnData.")

        groups, design = _validate_groups_or_design(groups, design, len(samples_list))
        resolved = _resolve_anndata_gene_names(samples_list, gene_names, layer=layer)

        self.samples = samples_list
        self.groups = groups
        self.design = design
        self.gene_names = list(resolved)
        self.feature_mode = feature_mode
        self.n_radial_bins = int(n_radial_bins)
        self.fft_solver = fft_solver
        self.workers = workers
        self.freq_edges = None if freq_edges is None else np.asarray(freq_edges, dtype=float)
        self.presence_threshold = float(presence_threshold)
        self.min_samples_per_group = int(min_samples_per_group)
        self.nufft_chunk_size = max(1, int(nufft_chunk_size))
        # NUFFT always produces full-2D layout (fft2), regardless of user's
        # ``fft_solver`` (which is moot here).
        self._spectrum_fft_solver = "fft2"

        self._layer = layer
        self._obsm_key = obsm_key
        self._nufft_eps = float(eps)

        # Per-sample coords / grids.
        from quadsv.kernels.nufft import _infer_grid_from_coords

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
            if obsm_key not in ad_s.obsm:
                raise KeyError(
                    f"sample {i} has no obsm['{obsm_key}']; "
                    f"available: {list(ad_s.obsm.keys())}."
                )
            c = np.asarray(ad_s.obsm[obsm_key], dtype=np.float64)
            if c.ndim != 2 or c.shape[1] != 2:
                raise ValueError(f"sample {i} obsm['{obsm_key}'] must be (n, 2), got {c.shape}.")
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
        from quadsv.kernels.nufft import power_spectrum_2d_nufft

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
                X_csc = X_src.tocsc()
                X_dense = None
            else:
                X_dense = np.asarray(X_src, dtype=np.float64)
                dc = X_dense.mean(axis=0)
                nnz_per = (X_dense != 0).sum(axis=0)
                X_csc = None

            presence_i = (nnz_per / max(n_spots, 1)) >= self.presence_threshold

            ny, nx = grid_i
            spec_stack = np.empty((n_genes, ny, nx), dtype=np.float64)

            for start in range(0, n_genes, chunk_size):
                stop = min(start + chunk_size, n_genes)
                cols = slice(start, stop)
                if X_csc is not None:
                    block = np.asarray(X_csc[:, cols].toarray(), dtype=np.float64)
                else:
                    block = X_dense[:, cols].astype(np.float64, copy=True)

                # Per-gene mean centering: removes the DC bin and prevents
                # per-sample mean-shift leakage into low-frequency bins. The
                # raw DC scalars are preserved on ``self.dc_`` for the
                # complementary :meth:`test_expression` path.
                block -= dc[None, cols]

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
