"""
quadsv: kernel-based spatial variability / co-expression tests for spatial omics.

The top-level namespace exposes exactly four conceptual layers, nothing else:

1. **Kernels** — :class:`MatrixKernel` (dense / sparse), :class:`FFTKernel`
   (regular grid), :class:`NUFFTKernel` (irregular 2D coordinates). All three
   are registered subclasses of the :class:`Kernel` ABC.
2. **Statistical tests** — :func:`spatial_q_test` and :func:`spatial_r_test`.
   A single entry point per test dispatches on the kernel type (matrix, FFT,
   or NUFFT). Signature: ``(x, kernel, null_params=None, return_pval=True,
   is_standardized=False)``.
3. **Detectors** — :class:`DetectorIrregular` consumes :class:`anndata.AnnData`
   (irregular grids, matrix/NUFFT backends); :class:`DetectorGrid` consumes
   :class:`spatialdata.SpatialData` (regular grids, FFT backend).
4. **Comparators** — cross-sample pattern comparison:
   :class:`ComparatorIrregular` on a list of AnnData (NUFFT backend);
   :class:`ComparatorGrid` on a list of SpatialData (FFT backend).

Utilities, constants, and internal primitives (Liu's SF, Visium I/O,
radial binning, etc.) remain accessible under their submodules
(``quadsv.io_visium``, ``quadsv.statistics``, ``quadsv.fft``, ``quadsv.nufft``,
``quadsv.multisample``) but are not re-exported at the top level.
"""

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("quadsv")
except (ImportError, PackageNotFoundError):
    __version__ = "0.1.0"

# Layer 1: Kernels
from quadsv.comparators import ComparatorGrid, ComparatorIrregular
from quadsv.detector import DetectorIrregular
from quadsv.detector_grid import DetectorGrid
from quadsv.fft import FFTKernel
from quadsv.kernels import Kernel, MatrixKernel
from quadsv.nufft import NUFFTKernel

# Layer 2: Statistical tests (single dispatching entry point per test)
from quadsv.statistics import spatial_q_test, spatial_r_test

__all__ = [
    # Kernels
    "Kernel",
    "MatrixKernel",
    "FFTKernel",
    "NUFFTKernel",
    # Statistical tests
    "spatial_q_test",
    "spatial_r_test",
    # Detectors
    "DetectorIrregular",
    "DetectorGrid",
    # Cross-sample
    "ComparatorIrregular",
    "ComparatorGrid",
]
