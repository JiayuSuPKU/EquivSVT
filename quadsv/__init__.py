"""
quadsv: Spatial statistics library for kernel-based hypothesis testing

This package implements kernel-based hypothesis tests for detecting
spatial variability and equivalence testing using different kernel methods
(Moran, Gaussian RBF, Matérn, Laplacian).
"""

import logging

# Library-style logging: attach a NullHandler so no records are emitted unless the
# consumer configures a handler. User code can do
# ``logging.getLogger('quadsv').setLevel(logging.INFO)`` to see progress messages.
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Automatically read version from installed package or fallback to hardcoded
try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("quadsv")
except (ImportError, PackageNotFoundError):
    # Fallback for development environments where package is not installed
    __version__ = "0.1.0"

# Import core classes and functions
# Application classes for pattern detection
from quadsv.detector import PatternDetector
from quadsv.fft import FFTKernel, power_spectrum_2d, spatial_q_test_fft, spatial_r_test_fft
from quadsv.io_visium import (
    VISIUM_V1_SPOT_SPACING_UM,
    load_visium_sample,
    visium_hex_spacing_um,
    visium_to_grid,
)
from quadsv.kernels import SpatialKernel
from quadsv.spectral_compare import (
    SpectralComparator,
    benchmark_statistics,
    compare_two_groups,
    compare_two_groups_scalar,
    shape_normalize,
)
from quadsv.statistics import spatial_q_test, spatial_r_test

# NUFFT is an optional extra; only surface it at the top level when finufft is
# actually installed, so `import quadsv` stays light for rasterized workflows.
try:
    from quadsv.nufft import power_spectrum_2d_nufft  # noqa: F401

    _HAS_NUFFT = True
except ImportError:  # pragma: no cover
    _HAS_NUFFT = False

# Define public API
__all__ = [
    # Core classes
    "SpatialKernel",
    "FFTKernel",
    # Statistical functions
    "spatial_q_test",
    "spatial_r_test",
    "spatial_q_test_fft",
    "spatial_r_test_fft",
    "power_spectrum_2d",
    # Detector classes
    "PatternDetector",
    # Cross-sample spectral comparison
    "SpectralComparator",
    "compare_two_groups",
    "compare_two_groups_scalar",
    "benchmark_statistics",
    "shape_normalize",
    # Visium I/O
    "load_visium_sample",
    "visium_to_grid",
    "visium_hex_spacing_um",
    "VISIUM_V1_SPOT_SPACING_UM",
]

# PatternDetectorFFT requires optional spatialdata dependency
try:
    from quadsv.detector_fft import PatternDetectorFFT

    __all__.append("PatternDetectorFFT")
except ImportError:
    pass

# Append NUFFT symbol only if the optional finufft dep is present.
if _HAS_NUFFT:
    __all__.append("power_spectrum_2d_nufft")
