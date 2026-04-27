"""Tests for soft-deprecated public-API symbols.

When a previously-exported top-level name is being retired, we keep it
resolvable via the package's ``__getattr__`` shim and emit a single
``DeprecationWarning`` pointing at the canonical path. These tests
codify the contract.
"""

from __future__ import annotations

import unittest
import warnings

import quadsv


class TestMatrixKernelBaseDeprecation(unittest.TestCase):
    def test_top_level_emits_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            obj = quadsv.MatrixKernelBase  # noqa: F841 — accessed for side effect
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        self.assertTrue(dep, "MatrixKernelBase access did not emit DeprecationWarning")
        msg = str(dep[0].message)
        self.assertIn("MatrixKernelBase", msg)
        self.assertIn("quadsv.kernels", msg)

    def test_canonical_path_silent(self):
        """``from quadsv.kernels import MatrixKernelBase`` is the
        canonical entry point and must not warn."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from quadsv.kernels import MatrixKernelBase  # noqa: F401
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        self.assertFalse(dep, f"Canonical import emitted DeprecationWarning(s): {dep}")

    def test_top_level_not_in_all(self):
        """The deprecated name must not appear in ``quadsv.__all__`` —
        ``from quadsv import *`` should not pull it in."""
        self.assertNotIn("MatrixKernelBase", quadsv.__all__)

    def test_top_level_resolves_to_canonical_object(self):
        """The lazy resolver must return the *same* class object as the
        canonical path — patching one must affect the other."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from_top = quadsv.MatrixKernelBase
        from quadsv.kernels import MatrixKernelBase as canonical

        self.assertIs(from_top, canonical)

    def test_unknown_top_level_raises_attribute_error(self):
        """The ``__getattr__`` shim must not swallow real typos."""
        with self.assertRaises(AttributeError):
            quadsv.NonexistentSymbolThatShouldNeverExist  # noqa: B018


if __name__ == "__main__":
    unittest.main()
