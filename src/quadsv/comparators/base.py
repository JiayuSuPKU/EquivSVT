"""
Shared mixin and input-validation helpers for the comparator layer.

This module hosts the private :class:`_ComparatorBase` mixin that
:class:`~quadsv.ComparatorIrregular` and :class:`~quadsv.ComparatorGrid`
inherit from for their post-fit surface (``normalize_background``,
``shape_normalize``, ``residualize``, ``test_pattern``,
``test_expression``, ``benchmark``), plus the two ``_validate_*``
helpers used by both classes' constructors.

Concrete classes live in sibling modules:
:mod:`quadsv.comparators.irregular` and
:mod:`quadsv.comparators.grid`.
"""

from __future__ import annotations

import logging
import warnings
from abc import abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np
from tqdm.auto import tqdm

# Suppress known deprecation warnings from SpatialData dependencies BEFORE importing them.
warnings.filterwarnings("ignore", category=FutureWarning, message=".*legacy Dask DataFrame.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated.*")

from quadsv.comparators.multisample import (
    _AVAILABLE_STATISTICS,
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

__all__: list[str] = []

logger = logging.getLogger(__name__)


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
        null: str = "permutation",
        n_perm: int = 1000,
        random_state: int | None = None,
        freq_weights: np.ndarray | None = None,
        n_perm_max: int = 10000,
    ) -> Any:
        """Two-group spectral-pattern test on :attr:`spectra_`.

        Dispatches to
        :func:`quadsv.comparators.multisample.compare_two_groups_masked`
        when any ``(sample, gene)`` pair is marked absent in :attr:`presence_`
        (e.g. when ``presence_threshold > 0``), otherwise to
        :func:`~quadsv.comparators.multisample.compare_two_groups`.

        ``null`` selects the null-distribution method:
        ``'permutation'`` (default, back-compat) or ``'wald'`` (alias
        ``'liu'``) for the analytic Wald-type test. Only ``log_l2`` accepts
        ``null='wald'``; the masked path does not yet support it.
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
                null=null,
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
            null=null,
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
        null: str = "permutation",
        n_perm: int = 1000,
        random_state: int | None = None,
        n_perm_max: int = 10000,
    ) -> dict[str, Any]:
        """Run :func:`benchmark_statistics` on :attr:`spectra_`.

        ``null`` is forwarded; only ``log_l2`` honours ``null='wald'``,
        other statistics ignore it.
        """
        if self.spectra_ is None:
            raise RuntimeError("Call .fit() before .benchmark().")
        return benchmark_statistics(
            self.spectra_,
            self.groups,
            gene_names=self.gene_names,
            statistics=statistics,
            null=null,
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
