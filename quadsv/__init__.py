"""
quadsv: kernel-based spatial variability / co-expression tests for spatial omics.

The top-level namespace exposes exactly four conceptual layers, nothing else:

1. **Kernels** — :class:`SpatialKernel` (dense / sparse), :class:`FFTKernel`
   (regular grid), :class:`NUFFTKernel` (irregular 2D coordinates).
2. **Statistical tests** — ``spatial_q_test`` / ``spatial_r_test`` (dense),
   plus the FFT and NUFFT variants. All share the same signature shape:
   ``(x, kernel, null_params=None, return_pval=True, is_standardized=False)``.
3. **PatternDetector** — :class:`PatternDetector` consumes
   :class:`anndata.AnnData` (irregular grids, matrix/NUFFT backends),
   :class:`PatternDetectorFFT` consumes :class:`spatialdata.SpatialData`
   (regular grids, FFT backend).
4. **SpectralComparator** — cross-sample pattern comparison on a list of
   AnnData (NUFFT backend) or SpatialData (FFT backend).

Utilities, constants, and internal primitives (Liu's SF, Visium I/O,
radial binning, etc.) remain accessible under their submodules
(``quadsv.io_visium``, ``quadsv.statistics``, ``quadsv.fft``, ``quadsv.nufft``,
``quadsv.spectral_compare``) but are not re-exported at the top level.
"""

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("quadsv")
except (ImportError, PackageNotFoundError):
    __version__ = "0.1.0"

# Layer 1: Kernels
# Layer 3: PatternDetector
from quadsv.detector import PatternDetector
from quadsv.detector_fft import PatternDetectorFFT

# Layer 2: Statistical tests
from quadsv.fft import FFTKernel, spatial_q_test_fft, spatial_r_test_fft
from quadsv.kernels import SpatialKernel
from quadsv.nufft import NUFFTKernel, spatial_q_test_nufft, spatial_r_test_nufft

# Layer 4: SpectralComparator (cross-sample)
from quadsv.spectral_compare import SpectralComparator
from quadsv.statistics import spatial_q_test, spatial_r_test

__all__ = [
    # Kernels
    "SpatialKernel",
    "FFTKernel",
    "NUFFTKernel",
    # Statistical tests
    "spatial_q_test",
    "spatial_r_test",
    "spatial_q_test_fft",
    "spatial_r_test_fft",
    "spatial_q_test_nufft",
    "spatial_r_test_nufft",
    # Detectors
    "PatternDetector",
    "PatternDetectorFFT",
    # Cross-sample
    "SpectralComparator",
]
