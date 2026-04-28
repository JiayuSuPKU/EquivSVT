# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Analytic Wald-type null for `log_l2`: pass `null="wald"` (alias
  `null="liu"`) to `compare_two_groups`, `compare_two_groups_masked`,
  `benchmark_statistics`, or `Comparator.test_pattern`/`.benchmark`.
  Under H₀ the per-gene statistic `T² = D'WD` is a quadratic form in a
  Gaussian difference-of-means; the tail of the resulting weighted-χ²
  mixture is integrated via Liu's approximation
  (`quadsv.statistics.liu_sf`). The within-group variance is estimated
  diagonally per frequency bin and pooled across genes. Default null
  remains `"permutation"` for back-compat. Currently only valid with
  `statistic="log_l2"`; `compare_two_groups_masked` will raise
  `NotImplementedError` (the per-gene presence subsetting breaks the
  pooled-σ² estimator). The Wald path bypasses the small-n
  exact-permutation BH-floor at the cost of mild anti-conservatism on
  contrasts with very small residual df (e.g. 1-vs-2 splits) — see
  the new `TestLogL2WaldNull` calibration tests.
- Package layout migrated to `src/quadsv/` (no behavioural change for
  consumers; existing `import quadsv` and `from quadsv import ...`
  keep working). Editable installs must be reissued
  (`pip install -e ".[dev]"`).
- Three power-user helpers promoted to top-level:
  `quadsv.compute_null_params`, `quadsv.auto_chunk_size`,
  `quadsv.liu_sf`. The canonical `quadsv.statistics` paths still
  work.
- New type-dispatch factory entry points:
  `quadsv.Detector(data, **kw)` and
  `quadsv.Comparator(data_list, **kw)`. They dispatch on
  `isinstance(data, AnnData)` to `DetectorIrregular` /
  `ComparatorIrregular` vs `isinstance(data, SpatialData)` to
  `DetectorGrid` / `ComparatorGrid`. Mixed-type lists raise
  `TypeError`.
- New `tests/test_public_api.py` freezes the public surface:
  `__all__` snapshot, docstring presence, canonical-path identity,
  the negative assertion that the removed legacy module paths
  raise `ModuleNotFoundError`, and a check that backend ABCs do
  not leak into the top-level namespace.

### Fixed
- CI workflow's install step referenced non-existent extras
  (`[dev,test,spatial]` and `[docs,spatial]`); narrowed to the
  actual `[dev]` / `[docs]` extras defined in `pyproject.toml`.

### Changed
- The four conceptual layers are now physical subpackages:
  - `quadsv.kernels.{fft,nufft}` (was `quadsv.fft`, `quadsv.nufft`)
  - `quadsv.detectors.{base,irregular,grid}` (was
    `quadsv._detector_base`, `quadsv.detector`,
    `quadsv.detector_grid`)
  - `quadsv.comparators.{__init__,multisample}` (was
    `quadsv.comparators` flat module + `quadsv.multisample`)
  Lint / format commands now target `src/ tests/` (was
  `quadsv/ tests/`).

### Removed
- The six legacy-path shim modules `quadsv.fft`, `quadsv.nufft`,
  `quadsv.detector`, `quadsv.detector_grid`,
  `quadsv._detector_base`, and `quadsv.multisample`. Use the
  canonical subpackage paths above.
- The backend ABCs `Kernel` and `MatrixKernelBase` are no longer
  re-exported from the top-level `quadsv` namespace. They live at
  `quadsv.kernels` and are intended for backend authors. Subclass
  `quadsv.kernels.Kernel` for a custom kernel and
  `quadsv.kernels.MatrixKernelBase` for a custom matrix backend.

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
