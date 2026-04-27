"""Legacy-path shim. The canonical module is :mod:`quadsv.detectors.base`.

This module remains importable for one release so existing
``from quadsv._detector_base import Detector`` lines keep working.
New code should import :class:`~quadsv.Detector` (or
:class:`quadsv.detectors.Detector`) instead.
"""

from quadsv.detectors.base import *  # noqa: F401,F403
from quadsv.detectors.base import __all__  # noqa: F401
