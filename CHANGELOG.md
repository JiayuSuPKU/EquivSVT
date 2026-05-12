# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **GLM design API for cross-sample pattern comparison.** New public
  `compare_designs(spectra, design, contrast, …)` generalises the
  two-group test to arbitrary OLS designs (binary, continuous,
  multi-factor) with an analytic Wald null. The two-group case is
  recovered exactly. `ComparatorIrregular` / `ComparatorGrid`
  constructors accept a `design=` keyword (mutually exclusive with
  `groups=`); DataFrames are patsy-encoded, ndarrays are used as-is.
  `Comparator.test_pattern(...)` gains a `contrast=` argument
  (column name, dict, or contrast vector).
- **Analytic Wald null for `log_l2`** (`null="wald"`, alias
  `"liu"`) on `compare_two_groups`, `compare_two_groups_masked`, and
  `compare_designs`. Per-gene statistic is integrated via Liu's
  approximation against a pooled-across-genes **full** within-group
  Σ (a single 30×30 eigendecomposition before each Liu integration);
  bypasses the small-n permutation BH-floor while keeping mean
  within-group null FPR at ~0.012 across the three benchmark panels.
  Emits a `UserWarning` at residual df < 3. The masked variant uses
  a mask-aware pooled estimator with per-gene noncentrality scaling
  so genes with different observed cohorts get correctly-scaled
  eigenvalues.
- **Analytic Welch t null for `compare_two_groups_scalar`**
  (`null="welch"`, now the default). Computes per-gene two-sided
  p-values from the Welch-Satterthwaite t-distribution; lets the DE
  companion bypass the permutation `1/(n_perm+1)` raw-p floor on
  small cohorts. The previous permutation null is preserved as
  `null="permutation"`.
- **`normalize_shape: bool = False` keyword** on every spectrum-input
  comparison test (`compare_two_groups`, `compare_two_groups_masked`,
  `compare_designs`). When True, divides each per-(sample, gene)
  spectrum by its sum along the frequency axis before the statistic
  is computed, so the test fires only on shape-only redistribution
  of power across radial frequencies. Statistic-agnostic; default
  False preserves prior behaviour.
- **Effective-rank diagnostics** for the within-group covariance
  used by the Wald null: `quadsv.effective_rank(cov, weights=None)`
  primitive (`K_eff = (Σλ)² / Σλ²`),
  `quadsv.gene_pattern_diversity(spectra)` for per-sample
  heterogeneity,
  `quadsv.within_group_pattern_diversity(spectra, groups)` for
  cohort-level, and a chainable
  `Comparator.effective_rank(level=…)` accessor.
- **Top-level convenience exports**:
  `quadsv.Detector(data, …)` and `quadsv.Comparator(data_list, …)`
  factories that dispatch on `AnnData` vs `SpatialData`;
  `quadsv.compute_null_params`, `quadsv.auto_chunk_size`,
  `quadsv.liu_sf` promoted to top level (canonical
  `quadsv.statistics` paths still work).
- **Public-API freeze test** (`tests/test_public_api.py`) snapshots
  `__all__`, docstring presence, canonical-path identity, and
  asserts removed legacy paths raise `ModuleNotFoundError`.

### Changed
- **Breaking: package layout migrated to `src/quadsv/`** with the
  four conceptual layers as physical subpackages —
  `quadsv.kernels.{fft,nufft}`,
  `quadsv.detectors.{base,irregular,grid}`,
  `quadsv.comparators.{__init__,multisample}`. `import quadsv` and
  `from quadsv import …` keep working; editable installs must be
  reissued (`pip install -e ".[dev]"`). Lint / format commands now
  target `src/ tests/`.
- **Breaking: unified `normalize_*` surface API in
  `quadsv.comparators.multisample`** (no aliases):
    * `normalize_by_background` → `normalize_background`
    * `residualize_against_covariates` → `normalize_covariates`
    * `shape_normalize` → `normalize_shape`
  Consistent first-arg `spectra`, keyword-only after, NumPy-style
  docstrings with LaTeX math.
  `normalize_covariates`'s first positional arg is renamed
  `gene_spectra` → `spectra` and the `eps` kwarg is dropped (the
  closed-form pseudoinverse is numerically robust on the typical
  low-`n_cov` designs used here). The chainable comparator
  instance methods follow suit:
    * `.shape_normalize()` → `.normalize_shape()`
    * `.residualize()` → `.normalize_covariates()`

### Removed
- **Breaking: `center` argument retired** across the comparator API.
  `ComparatorIrregular`, `ComparatorGrid`, and
  `compute_sample_spectrum` no longer accept `center`. Per-gene
  mean centring (the previous default) is now the only spectrum
  normalisation path. The `_ZSCORE_CLIP` constant, the
  `zscore_clip` parameter, and the per-bin clamp in the NUFFT loop
  are deleted (~50 LOC).
- **Breaking: `benchmark_statistics` function and the matching
  `Comparator.benchmark()` method retired.** Invoke
  `compare_two_groups` directly with each `statistic=` value to A/B
  compare on the same fitted spectra (~95 LOC).
- **Breaking: `statistic="hotelling_lw"` and `statistic="mmd_rbf"`
  paths retired** from every comparison function. Both were
  impractically slow and consistently dominated on sensitivity by
  `log_l2 + null='wald'` or `cauchy_welch`. `_AVAILABLE_STATISTICS`
  now reads `("log_l2", "cauchy_welch")`.
- **Breaking: six legacy-path shim modules removed** —
  `quadsv.fft`, `quadsv.nufft`, `quadsv.detector`,
  `quadsv.detector_grid`, `quadsv._detector_base`,
  `quadsv.multisample`. Use the canonical subpackage paths.
- **Breaking: backend ABCs `Kernel` and `MatrixKernelBase` no
  longer re-exported from top-level `quadsv`**. They live at
  `quadsv.kernels` and are intended for backend authors.

### Fixed
- CI workflow install step referenced non-existent extras
  (`[dev,test,spatial]` and `[docs,spatial]`); narrowed to the
  actual `[dev]` / `[docs]` extras in `pyproject.toml`.

## Release Process

- [ ] Run full test suite: `pytest tests/ --cov=quadsv`
- [ ] Check documentation builds: `sphinx-build -b html docs/ docs/_build/`
- [ ] Update version in `pyproject.toml`
- [ ] Update this CHANGELOG
- [ ] Create git tag: `git tag -a v0.1.0 -m "Release v0.1.0"`
- [ ] Build package: `python -m build`
- [ ] Upload to PyPI: `python -m twine upload dist/*`

## [0.1.0] - 2026-02-02

### Added
- Initial public release
- Q-test framework for univariate spatial pattern detection
- R-test framework for bivariate spatial co-expression
- Core kernel methods: Gaussian, Matérn, CAR, Graph Laplacian, Moran's I
- Implicit mode for scalable large-N computation (N > 5000)
- FFT acceleration for regular grid data (Visium HD)
- PatternDetector for AnnData integration (genome-wide SVG detection)
- PatternDetectorFFT for large-scale Visium HD analysis
- Null approximation methods: CLT, Welch/Satterthwaite, Liu
- Comprehensive test suite (unit + integration tests)
- Tutorial test cases demonstrating all major workflows
- Complete documentation with quickstart and theory sections
- Support for Python 3.10, 3.11, 3.12

## [0.1.1]

### Fixed
- Fix type hinting issues in `quadsv.kernels` module
