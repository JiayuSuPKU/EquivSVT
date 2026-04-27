"""
quadsv: kernel-based spatial pattern detection and comparison for spatial omics.

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
"""

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

# Version resolution order: prefer the file written by ``setuptools-scm`` at
# build time (``src/quadsv/_version.py`` — see ``[tool.setuptools_scm]`` in
# ``pyproject.toml``), fall back to installed-package metadata, then to a
# last-known release string for unbuilt / shallow-clone checkouts.
try:
    from quadsv._version import version as __version__  # type: ignore[assignment]
except ImportError:  # _version.py absent — source checkout without a build step
    try:
        from importlib.metadata import PackageNotFoundError, version

        __version__ = version("quadsv")
    except (ImportError, PackageNotFoundError):
        __version__ = "0.0.0+unknown"

from quadsv.comparators import ComparatorGrid, ComparatorIrregular
from quadsv.detectors.grid import DetectorGrid
from quadsv.detectors.irregular import DetectorIrregular
from quadsv.kernels import Kernel, MatrixKernel, MatrixKernelBase
from quadsv.kernels.fft import FFTKernel
from quadsv.kernels.nufft import NUFFTKernel
from quadsv.statistics import (
    auto_chunk_size,
    compute_null_params,
    liu_sf,
    spatial_q_test,
    spatial_r_test,
)

__all__ = [
    # Kernels
    "MatrixKernelBase",
    "MatrixKernel",
    "FFTKernel",
    "NUFFTKernel",
    # Statistical tests
    "spatial_q_test",
    "spatial_r_test",
    # Statistical-test power-user helpers (precompute-once, reuse-many-times)
    "compute_null_params",
    "auto_chunk_size",
    "liu_sf",
    # Detectors
    "DetectorIrregular",
    "DetectorGrid",
    # Cross-sample
    "ComparatorIrregular",
    "ComparatorGrid",
]
