"""Legacy-path shim. The canonical module is :mod:`quadsv.kernels.nufft`.

This module remains importable for one release so existing
``from quadsv.nufft import NUFFTKernel`` (and similar) lines keep
working. New code should import from :mod:`quadsv.kernels.nufft`
instead.
"""

from quadsv.kernels.nufft import *  # noqa: F401,F403
from quadsv.kernels.nufft import __all__  # noqa: F401
