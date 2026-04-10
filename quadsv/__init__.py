"""
quadsv: Spatial statistics library for kernel-based hypothesis testing

This package implements kernel-based hypothesis tests for detecting
spatial variability and equivalence testing using different kernel methods
(Moran, Gaussian RBF, Matérn, Laplacian).
"""

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
from quadsv.fft import FFTKernel, spatial_q_test_fft, spatial_r_test_fft
from quadsv.kernels import SpatialKernel
from quadsv.statistics import spatial_q_test, spatial_r_test

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
    # Detector classes
    "PatternDetector",
]

# PatternDetectorFFT requires optional spatialdata dependency
try:
    from quadsv.detector_fft import PatternDetectorFFT

    __all__.append("PatternDetectorFFT")
except ImportError:
    pass
