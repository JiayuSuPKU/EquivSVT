"""Public-API freeze test — long-term guardrail for the four-layer
``quadsv`` surface.

What this test enforces:

1. **Snapshot of ``quadsv.__all__``.** Any addition or removal to the
   top-level exports forces a deliberate edit to ``EXPECTED_ALL``
   below, which surfaces in code review.
2. **Every public name imports + has a docstring.** Catches typos in
   ``__all__`` and missing documentation on new public symbols.
3. **Canonical-path identity.** Top-level re-exports resolve to the
   *same* object as their canonical path (e.g.
   ``quadsv.compute_null_params is quadsv.statistics.compute_null_params``).
   Guards against accidental re-export breakage during refactors.
4. **Legacy-shim back-compat.** Each pre-Stage-2 import path
   (``quadsv.fft``, ``quadsv.nufft``, ``quadsv.detector``,
   ``quadsv.detector_grid``, ``quadsv._detector_base``,
   ``quadsv.multisample``) still resolves to the canonical class /
   function under ``quadsv.kernels.*``, ``quadsv.detectors.*``, or
   ``quadsv.comparators.multisample``. Locks in the one-release
   back-compat contract.
"""

from __future__ import annotations

import importlib

import quadsv

# ---------------------------------------------------------------------------
# Snapshot of the four-layer public surface. Any drift from this list — adds
# or removes — must be a deliberate edit reviewed alongside the change.
# Group order mirrors the package docstring (Kernels → Statistics →
# Detectors → Comparators → Factories).
# ---------------------------------------------------------------------------
EXPECTED_ALL: list[str] = [
    # Kernels
    "Kernel",
    "MatrixKernel",
    "FFTKernel",
    "NUFFTKernel",
    # Statistical tests
    "spatial_q_test",
    "spatial_r_test",
    # Statistical-test power-user helpers
    "compute_null_params",
    "auto_chunk_size",
    "liu_sf",
    # Detectors
    "DetectorIrregular",
    "DetectorGrid",
    # Cross-sample
    "ComparatorIrregular",
    "ComparatorGrid",
    # Factories
    "Detector",
    "Comparator",
]


def test_top_level_all_matches_snapshot():
    """``quadsv.__all__`` matches ``EXPECTED_ALL`` (set comparison).

    Order doesn't matter; the *set* is the contract. Edit
    ``EXPECTED_ALL`` deliberately when the public surface changes.
    """
    assert set(quadsv.__all__) == set(EXPECTED_ALL), (
        "quadsv.__all__ drifted from the expected snapshot.\n"
        f"  added:   {sorted(set(quadsv.__all__) - set(EXPECTED_ALL))}\n"
        f"  removed: {sorted(set(EXPECTED_ALL) - set(quadsv.__all__))}"
    )


def test_every_public_name_resolves_and_documented():
    """Every name in ``quadsv.__all__`` must resolve and carry a
    non-empty docstring."""
    for name in quadsv.__all__:
        obj = getattr(quadsv, name, None)
        assert obj is not None, f"{name} listed in __all__ but unresolved"
        doc = getattr(obj, "__doc__", None)
        assert doc and doc.strip(), f"{name} has no docstring"


# ---------------------------------------------------------------------------
# Canonical-path identity contract.
#
# Each top-level re-export must point at the same object as the
# canonical submodule path. If the re-export drifts (e.g. somebody
# accidentally rebinds the name in ``quadsv.__init__``), tests still
# import the canonical class but the user-facing shortcut becomes
# stale; this test fails loudly.
# ---------------------------------------------------------------------------
_CANONICAL_PATHS: dict[str, tuple[str, str]] = {
    # name on quadsv: (submodule, attribute on submodule)
    "Kernel": ("quadsv.kernels", "Kernel"),
    "MatrixKernel": ("quadsv.kernels", "MatrixKernel"),
    "FFTKernel": ("quadsv.kernels.fft", "FFTKernel"),
    "NUFFTKernel": ("quadsv.kernels.nufft", "NUFFTKernel"),
    "spatial_q_test": ("quadsv.statistics", "spatial_q_test"),
    "spatial_r_test": ("quadsv.statistics", "spatial_r_test"),
    "compute_null_params": ("quadsv.statistics", "compute_null_params"),
    "auto_chunk_size": ("quadsv.statistics", "auto_chunk_size"),
    "liu_sf": ("quadsv.statistics", "liu_sf"),
    "DetectorIrregular": ("quadsv.detectors.irregular", "DetectorIrregular"),
    "DetectorGrid": ("quadsv.detectors.grid", "DetectorGrid"),
    "ComparatorIrregular": ("quadsv.comparators", "ComparatorIrregular"),
    "ComparatorGrid": ("quadsv.comparators", "ComparatorGrid"),
    "Detector": ("quadsv.api", "Detector"),
    "Comparator": ("quadsv.api", "Comparator"),
}


def test_top_level_objects_identity_match_canonical_paths():
    """Every top-level re-export points at the same object as the
    canonical submodule path.
    """
    for name, (modpath, attr) in _CANONICAL_PATHS.items():
        top = getattr(quadsv, name)
        canonical = getattr(importlib.import_module(modpath), attr)
        assert top is canonical, f"quadsv.{name} drifted from {modpath}.{attr}"


# ---------------------------------------------------------------------------
# Legacy-path shim contract (one-release back-compat from Stage 2).
#
# Each pre-Stage-2 module path must still resolve. The shim modules
# do ``from quadsv.<canonical> import *`` plus ``__all__``, so checking
# every public name on each canonical module round-trips through the
# legacy path is the strongest assertion.
# ---------------------------------------------------------------------------
_LEGACY_SHIMS: dict[str, str] = {
    "quadsv.fft": "quadsv.kernels.fft",
    "quadsv.nufft": "quadsv.kernels.nufft",
    "quadsv.detector": "quadsv.detectors.irregular",
    "quadsv.detector_grid": "quadsv.detectors.grid",
    "quadsv._detector_base": "quadsv.detectors.base",
    "quadsv.multisample": "quadsv.comparators.multisample",
}


def test_legacy_shim_paths_resolve_to_canonical():
    """Each legacy module path is importable and re-exports every
    name listed in the canonical module's ``__all__`` with object
    identity.
    """
    for legacy_path, canonical_path in _LEGACY_SHIMS.items():
        legacy = importlib.import_module(legacy_path)
        canonical = importlib.import_module(canonical_path)
        for name in canonical.__all__:
            assert hasattr(legacy, name), f"{legacy_path}.{name} missing"
            assert getattr(legacy, name) is getattr(
                canonical, name
            ), f"{legacy_path}.{name} is not {canonical_path}.{name}"
