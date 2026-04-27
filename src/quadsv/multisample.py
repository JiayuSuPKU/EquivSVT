"""Legacy-path shim. The canonical module is :mod:`quadsv.comparators.multisample`.

This module remains importable for one release so existing
``from quadsv.multisample import …`` lines keep working. New code
should import from :mod:`quadsv.comparators.multisample` instead.
"""

from quadsv.comparators.multisample import *  # noqa: F401,F403
from quadsv.comparators.multisample import __all__  # noqa: F401
