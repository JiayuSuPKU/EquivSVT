"""
``quadsv.comparators`` — cross-sample spatial-pattern comparison.

Subpackage grouping the layer-4 public classes:

- :class:`ComparatorIrregular` — wraps a sequence of
  :class:`anndata.AnnData` (irregular spots, NUFFT backend).
- :class:`ComparatorGrid` — wraps a sequence of
  :class:`spatialdata.SpatialData` (regular rasterized bins, FFT
  backend).

Both classes share the same post-fit surface
(``normalize_background``, ``shape_normalize``, ``residualize``,
``test_pattern``, ``test_expression``, ``benchmark``) through the
private :class:`~quadsv.comparators.base._ComparatorBase` mixin.

The array-level primitives (spectrum compute, radial binning,
rotation alignment, statistical tests) live in
:mod:`quadsv.comparators.multisample`.
"""

from quadsv.comparators.grid import ComparatorGrid
from quadsv.comparators.irregular import ComparatorIrregular

__all__ = ["ComparatorIrregular", "ComparatorGrid"]
