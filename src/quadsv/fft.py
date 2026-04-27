"""Legacy-path shim. The canonical module is :mod:`quadsv.kernels.fft`.

This module remains importable for one release so existing
``from quadsv.fft import FFTKernel`` (and similar) lines keep working.
New code should import from :mod:`quadsv.kernels.fft` instead.
"""

from quadsv.kernels.fft import *  # noqa: F401,F403
from quadsv.kernels.fft import __all__  # noqa: F401
