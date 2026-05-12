"""
Shared mixin and input-validation helpers for the comparator layer.

This module hosts the private :class:`_ComparatorBase` mixin that
:class:`~quadsv.ComparatorIrregular` and :class:`~quadsv.ComparatorGrid`
inherit from for their post-fit surface (``normalize_background``,
``normalize_covariates``, ``normalize_shape``, ``test_pattern``,
``test_expression``), plus the two ``_validate_*`` helpers used by both
classes' constructors.

Each chainable instance method (``.normalize_background``,
``.normalize_covariates``, ``.normalize_shape``) is a thin wrapper
around the same-named standalone function in
:mod:`quadsv.comparators.multisample`.

Concrete classes live in sibling modules:
:mod:`quadsv.comparators.irregular` and
:mod:`quadsv.comparators.grid`.
"""

from __future__ import annotations

import logging
import warnings
from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm.auto import tqdm

if TYPE_CHECKING:
    import pandas as pd

# Suppress known deprecation warnings from SpatialData dependencies BEFORE importing them.
warnings.filterwarnings("ignore", category=FutureWarning, message=".*legacy Dask DataFrame.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated.*")

from quadsv.comparators.multisample import (
    align_spectra_by_rotation,
    compare_designs,
    compare_two_groups,
    compare_two_groups_masked,
    compare_two_groups_scalar,
    compute_sample_spectrum,
    radial_bin_spectrum,
)
from quadsv.comparators.multisample import (
    # Aliased to leading-underscore names to avoid shadowing the
    # like-named instance methods on the comparator class below.
    normalize_background as _normalize_background,
)
from quadsv.comparators.multisample import (
    normalize_covariates as _normalize_covariates,
)
from quadsv.comparators.multisample import (
    normalize_shape as _normalize_shape,
)
from quadsv.statistics import (
    gene_pattern_diversity as _gene_pattern_diversity,
)
from quadsv.statistics import (
    within_group_pattern_diversity as _within_group_pattern_diversity,
)

__all__: list[str] = []

logger = logging.getLogger(__name__)


class _ComparatorBase:
    """Shared state + shared methods for NUFFT / FFT pattern comparators.

    Concrete subclasses must populate ``self.samples``, ``self.groups``,
    ``self.gene_names``, ``self._grid_shapes``, ``self.spacings``,
    ``self._spectrum_fft_solver``, ``self.feature_mode``,
    ``self.n_radial_bins``, ``self.fft_solver``, ``self.workers``,
    ``self.freq_edges``, ``self.presence_threshold``,
    ``self.min_samples_per_group`` in their ``__init__``, then implement
    :meth:`_compute_spectra`.
    """

    # --- attribute stubs populated by subclass __init__ ---------------
    samples: list[Any]
    groups: np.ndarray | None
    """Binary group labels of length ``n_samples`` when constructed with
    ``groups=`` (back-compat path); ``None`` when constructed with
    ``design=``."""
    design: Any | None
    """Original ``design`` argument (DataFrame or ndarray); ``None`` when
    constructed with ``groups=``."""
    gene_names: list[str]
    feature_mode: str
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
            "Computing per-sample spectra (n_samples=%d, mean-centered)...",
            len(self._grid_shapes),
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
            self.spectra_[i] = _normalize_background(self.spectra_[i])
        return self

    def normalize_shape(self) -> _ComparatorBase:
        """Rescale each ``(sample, gene)`` spectrum to sum-1 along frequency.

        Thin chainable wrapper around the standalone
        :func:`~quadsv.comparators.multisample.normalize_shape`.
        """
        if self.spectra_ is None:
            raise RuntimeError("Call .fit() before .normalize_shape().")
        self.spectra_ = _normalize_shape(self.spectra_, axis=-1)
        return self

    def normalize_covariates(self, covariates: Sequence[np.ndarray]) -> _ComparatorBase:
        """Regress out per-sample covariate spectra from :attr:`spectra_`.

        Thin chainable wrapper around the standalone
        :func:`~quadsv.comparators.multisample.normalize_covariates`.

        ``covariates[i]`` should be a ``(n_covariates, ny_i, nx_i)`` array
        matching :attr:`_grid_shapes[i]`.
        """
        if self.spectra_ is None:
            raise RuntimeError("Call .fit() before .normalize_covariates().")
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
            self.spectra_[i] = _normalize_covariates(self.spectra_[i], cov_feat)
        return self

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------
    def test_pattern(
        self,
        statistic: str = "log_l2",
        null: str = "permutation",
        contrast: str | dict[str, float] | np.ndarray | None = None,
        n_perm: int = 1000,
        random_state: int | None = None,
        freq_weights: np.ndarray | None = None,
        n_perm_max: int = 10000,
    ) -> Any:
        """Spectral-pattern test on :attr:`spectra_`.

        Dispatches between three execution paths:

        - **Binary, permutation null** (`groups=` constructor path,
          ``null="permutation"``, ``contrast=None``): the back-compat
          two-group permutation test via
          :func:`~quadsv.comparators.multisample.compare_two_groups`
          (or its masked variant when any presence flag is False).
        - **Binary, Wald null** (`groups=` constructor path,
          ``null="wald"``, ``contrast=None``): the analytic Wald test for
          the same binary indicator.
        - **GLM Wald** (`design=` constructor path **or** explicit
          ``contrast=``): the generalized analytic Wald test via
          :func:`~quadsv.comparators.multisample.compare_designs`.

        ``contrast`` is required when the comparator was constructed with
        ``design=``. When ``contrast`` is supplied alongside the binary
        ``groups=`` path the design is constructed on the fly from
        ``groups`` (single-column DataFrame) so the same contrast resolution
        rules apply.
        """
        if self.spectra_ is None:
            raise RuntimeError("Call .fit() before .test_pattern().")

        use_glm = (contrast is not None) or (self.groups is None)
        if use_glm:
            null_canon = null if null in ("wald", "liu") else null
            if null_canon not in ("wald", "liu"):
                raise NotImplementedError(
                    "Only null='wald' (alias 'liu') is supported when "
                    "contrast= is provided or the comparator was constructed "
                    "with design=. Pass null='wald' or use the binary "
                    "groups= path with null='permutation'."
                )
            if contrast is None:
                raise ValueError(
                    "test_pattern() requires `contrast=` when the comparator "
                    "was constructed with `design=`."
                )
            design = self.design if self.design is not None else _groups_to_design(self.groups)
            return compare_designs(
                self.spectra_,
                design,
                contrast,
                gene_names=self.gene_names,
                statistic=statistic,
                null=null,
                freq_weights=freq_weights,
            )

        # Binary path (groups was set, contrast is None).
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
        """Classical DE test on the DC component (per-sample per-gene grid mean).

        Currently the binary-groups path only — call only when the
        comparator was constructed with ``groups=``.
        """
        if self.dc_ is None:
            raise RuntimeError("Call .fit() before .test_expression().")
        if self.groups is None:
            raise NotImplementedError(
                "test_expression() currently requires the binary `groups=` "
                "constructor path. The DC-component DE test does not yet "
                "support a general design matrix; use a downstream tool "
                "(e.g., scanpy.tl.rank_genes_groups) on the per-sample DC "
                "values for now."
            )
        return compare_two_groups_scalar(
            self.dc_,
            self.groups,
            gene_names=self.gene_names,
            n_perm=n_perm,
            random_state=random_state,
            n_perm_max=n_perm_max,
        )

    def effective_rank(
        self,
        level: str = "within_group",
        weights: np.ndarray | None = None,
    ) -> float | np.ndarray:
        """Effective rank ``K_eff`` of the spectrum covariance.

        Quantifies how concentrated the spatial-frequency content is along
        the eigen-directions of the relevant covariance matrix.
        ``K_eff = (Σλ)² / Σλ²`` — bounded by 1 (rank-1, all power on a
        single direction → Wald test reduces to a 1-DoF test) and ``K``
        (uniformly spread, Liu's CLT smoothing is most accurate).

        Parameters
        ----------
        level : {'within_group', 'per_sample'}, default 'within_group'
            ``'within_group'``: returns a single ``K_eff`` for the pooled
            within-group covariance (the same Σ used by ``log_l2 +
            null='wald'``). Useful for diagnosing whether the analytic
            null should be trusted on this cohort. Requires
            ``self.groups``.

            ``'per_sample'``: returns an ``(n_samples,)`` array — the
            effective rank of each sample's gene-wise spectrum
            covariance. High variability across samples means
            sample-to-sample heterogeneity in spatial-pattern structure,
            which is a separate concern from cross-condition difference.

        weights : np.ndarray, optional
            Per-bin weights (same semantics as ``freq_weights``). When
            given, returns the effective rank of
            ``W^{1/2} Σ W^{1/2}`` — useful for analysing how a
            frequency-weighted L2 statistic redistributes its power.

        Returns
        -------
        float (when ``level='within_group'``) or np.ndarray of shape
        ``(n_samples,)`` (when ``level='per_sample'``).
        """
        if self.spectra_ is None:
            raise RuntimeError("Call .fit() before .effective_rank().")
        if level == "within_group":
            if self.groups is None:
                raise ValueError(
                    "level='within_group' requires the binary `groups=` "
                    "constructor path. Use level='per_sample' for the "
                    "design= constructor path."
                )
            return _within_group_pattern_diversity(self.spectra_, self.groups, weights=weights)
        if level == "per_sample":
            n_samples = self.spectra_.shape[0]
            return np.array(
                [
                    _gene_pattern_diversity(self.spectra_[i], weights=weights)
                    for i in range(n_samples)
                ]
            )
        raise ValueError(f"level must be 'within_group' or 'per_sample', got {level!r}.")


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


def _groups_to_design(groups: np.ndarray) -> Any:
    """Wrap a 1-D groups array in a single-column pandas DataFrame.

    The resulting DataFrame is consumed by patsy in
    :func:`compare_designs` to build a design matrix
    ``[intercept, group[T.<level>]]``. The contrast string ``"group"``
    then auto-resolves to the indicator column.

    pandas is only imported here (rather than at module import time) so
    that callers using the array-only constructor signatures don't pay
    the import cost.
    """
    import pandas as pd  # local import: optional dep for the DataFrame path

    return pd.DataFrame({"group": np.asarray(groups)})


def _validate_groups_or_design(
    groups: np.ndarray | None,
    design: pd.DataFrame | np.ndarray | None,
    n_samples: int,
) -> tuple[np.ndarray | None, Any | None]:
    """Validate that exactly one of ``groups`` or ``design`` is supplied.

    Returns ``(groups_or_None, design_or_None)`` — leaves the original
    object intact for downstream use; subclasses store both. The caller
    can build the actual numeric design matrix via
    :func:`quadsv.comparators.multisample._build_design_matrix` when needed.
    """
    if groups is None and design is None:
        raise ValueError("Exactly one of `groups` or `design` must be supplied at construction.")
    if groups is not None and design is not None:
        raise ValueError("Pass either `groups` or `design`, not both.")
    if groups is not None:
        return _validate_groups(groups, n_samples), None
    # design is not None
    # Light validation only; the heavy lifting (patsy / column resolution)
    # happens lazily at test_pattern() time so we don't load patsy unless
    # the GLM path is actually used.
    try:
        import pandas as _pd  # noqa: F401
    except ImportError:
        _pd = None
    if _pd is not None and isinstance(design, _pd.DataFrame):
        if len(design) != n_samples:
            raise ValueError(f"design DataFrame length {len(design)} != n_samples={n_samples}.")
    elif isinstance(design, np.ndarray):
        if design.ndim != 2 or design.shape[0] != n_samples:
            raise ValueError(
                f"design ndarray must be (n_samples, p) = ({n_samples}, p), " f"got {design.shape}."
            )
    else:
        raise TypeError(
            f"design must be a pandas DataFrame or numpy ndarray, " f"got {type(design).__name__}."
        )
    return None, design


def _validate_common(
    feature_mode: str,
    fft_solver: str,
    presence_threshold: float,
    min_samples_per_group: int,
) -> str:
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
