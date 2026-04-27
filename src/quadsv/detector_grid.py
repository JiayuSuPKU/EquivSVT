"""Legacy-path shim. The canonical module is :mod:`quadsv.detectors.grid`.

This module remains importable for one release so existing
``from quadsv.detector_grid import DetectorGrid`` (and similar) lines
keep working. New code should import from :mod:`quadsv.detectors.grid`
(or simply ``from quadsv import DetectorGrid``) instead.
"""

from quadsv.detectors.grid import *  # noqa: F401,F403
from quadsv.detectors.grid import __all__  # noqa: F401
