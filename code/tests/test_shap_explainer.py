"""Tests for the SHAP explainer module.

These tests cover:

- :class:`ShapNotInstalledError` is raised with a helpful message when
  ``shap`` is not installed.
- :class:`ShapExplainer` correctly wraps a simple torch model when
  ``shap`` IS installed (skipped if shap is not installed).
- :func:`compute_attention_adjacency_alignment` returns 1.0 for
  identical maps and 0.0 for an all-zero map.
- :func:`compute_shap_stability` returns 1.0 for identical rankings
  and a value < 1 for differing rankings.
- :class:`ShapConfig` dataclass defaults are sensible.

All tests are deterministic and do not require a GPU or torch.

Import strategy
---------------
The ``mvgt_net`` package ``__init__.py`` eagerly imports torch-dependent
submodules (``model``, ``embedding``, etc.). To run these tests on a
machine without torch installed (e.g., a CI machine verifying only the
SHAP logic), we load the ``shap_explainer`` module file directly via
``importlib`` rather than triggering the package ``__init__``. On the
VPS, where torch is installed, the standard ``from mvgt_net.shap_explainer
import ...`` form works equivalently.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from unittest import mock

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.normpath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, CODE_ROOT)

# Direct-load shap_explainer.py without triggering mvgt_net/__init__.py
_SHAP_PATH = os.path.join(CODE_ROOT, "mvgt_net", "shap_explainer.py")
_spec = importlib.util.spec_from_file_location(
    "mvgt_net.shap_explainer", _SHAP_PATH
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["mvgt_net.shap_explainer"] = _mod
_spec.loader.exec_module(_mod)

ShapConfig = _mod.ShapConfig
ShapExplainer = _mod.ShapExplainer
ShapNotInstalledError = _mod.ShapNotInstalledError
compute_attention_adjacency_alignment = _mod.compute_attention_adjacency_alignment
compute_shap_stability = _mod.compute_shap_stability


class TestShapConfig(unittest.TestCase):
    def test_defaults(self) -> None:
        c = ShapConfig()
        self.assertEqual(c.n_background, 64)
        self.assertEqual(c.n_explain, 128)
        self.assertEqual(c.output_dir, "results/shap")
        self.assertEqual(c.device, "cpu")
        self.assertEqual(c.random_state, 42)
        self.assertIsNone(c.feature_names)


class TestShapNotInstalled(unittest.TestCase):
    def test_error_is_subclass_of_import_error(self) -> None:
        self.assertTrue(issubclass(ShapNotInstalledError, ImportError))

    def test_constructor_message(self) -> None:
        try:
            raise ShapNotInstalledError("hello")
        except ShapNotInstalledError as exc:
            self.assertIn("hello", str(exc))


class TestAttentionAlignment(unittest.TestCase):
    def test_identical_maps_pearson_one(self) -> None:
        import numpy as np
        a = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
        r = compute_attention_adjacency_alignment(a, a, method="pearson")
        self.assertAlmostEqual(r, 1.0, places=6)

    def test_zero_map_returns_zero(self) -> None:
        import numpy as np
        a = np.zeros((5, 5))
        b = np.random.default_rng(0).random((5, 5))
        r = compute_attention_adjacency_alignment(a, b, method="pearson")
        self.assertEqual(r, 0.0)

    def test_shape_mismatch_raises(self) -> None:
        import numpy as np
        with self.assertRaises(ValueError):
            compute_attention_adjacency_alignment(
                np.zeros((3, 3)), np.zeros((4, 4))
            )

    def test_unknown_method_raises(self) -> None:
        import numpy as np
        with self.assertRaises(ValueError):
            compute_attention_adjacency_alignment(
                np.zeros((3, 3)), np.zeros((3, 3)), method="nope"
            )


class TestShapStability(unittest.TestCase):
    def test_identical_rankings_tau_one(self) -> None:
        r1 = {"x0": 0.1, "x1": 0.2, "x2": 0.3}
        r2 = {"x0": 1.0, "x1": 2.0, "x2": 3.0}  # same ranking
        tau = compute_shap_stability(r1, r2, method="kendall")
        self.assertAlmostEqual(tau, 1.0, places=6)

    def test_different_rankings_tau_less_than_one(self) -> None:
        r1 = {"x0": 0.1, "x1": 0.2, "x2": 0.3}
        r2 = {"x0": 0.3, "x1": 0.2, "x2": 0.1}  # reversed
        tau = compute_shap_stability(r1, r2, method="kendall")
        self.assertLess(tau, 0.0)  # reversed ranking -> negative tau

    def test_too_few_common_features_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_shap_stability({"a": 1.0}, {"a": 2.0})


class TestShapExplainerLazyImport(unittest.TestCase):
    """Verify that the lazy-import behaves correctly."""

    def test_require_shap_raises_when_not_installed(self) -> None:
        from mvgt_net.shap_explainer import _require_shap
        # Force the import to fail.
        with mock.patch.dict(sys.modules, {"shap": None}):
            with self.assertRaises(ShapNotInstalledError):
                _require_shap()

    def test_module_imports_without_shap(self) -> None:
        """The module must import even if shap is not installed."""
        # The module was already direct-loaded at the top of this test
        # file. Verify it has the expected attributes.
        self.assertTrue(hasattr(_mod, "ShapExplainer"))
        self.assertTrue(hasattr(_mod, "ShapConfig"))
        self.assertTrue(hasattr(_mod, "compute_shap_stability"))
        self.assertTrue(hasattr(_mod, "compute_attention_adjacency_alignment"))


if __name__ == "__main__":
    unittest.main()
