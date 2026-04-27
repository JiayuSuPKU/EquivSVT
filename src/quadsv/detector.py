"""Legacy-path shim. The canonical module is :mod:`quadsv.detectors.irregular`.

This module remains importable for one release so existing
``from quadsv.detector import DetectorIrregular`` (and similar) lines
keep working. New code should import from
:mod:`quadsv.detectors.irregular` (or simply ``from quadsv import
DetectorIrregular``) instead.
"""

from quadsv.detectors.irregular import *  # noqa: F401,F403
from quadsv.detectors.irregular import __all__  # noqa: F401
