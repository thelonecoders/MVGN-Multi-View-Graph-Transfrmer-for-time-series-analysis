"""Tests for the baselines package.

These tests exercise the parts of the package that have a deterministic,
zero-hallucination behavior:

- The 12 stubs import cleanly.
- The factory builds all 12 stubs and raises on unknown names.
- The leaderboard loader finds the JSON, lists the 12 names, and
  raises :class:`LeaderboardNotPopulatedError` when scores are null.
- The base class raises :class:`NotImplementedError` on forward/step.

Tests that would require the leaderboard to be populated (e.g.,
"compare_with_leaderboard --mode cite succeeds") are deliberately
absent here -- they live in ``test_compare_with_leaderboard.py`` and
skip when the leaderboard is not yet fetched.
"""

from __future__ import annotations

import json
import os
import sys
import unittest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.normpath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, CODE_ROOT)

from baselines import (  # noqa: E402  -- intentional late import
    BaseBaseline,
    LEADERBOARD_FILE,
    LeaderboardNotPopulatedError,
    build_baseline,
    list_available_baselines,
    load_leaderboard,
)


class TestBaselinesPackage(unittest.TestCase):
    def test_twelve_stubs_present(self) -> None:
        names = list_available_baselines()
        self.assertEqual(len(names), 12)
        for n in names:
            self.assertIsInstance(n, str)

    def test_factory_builds_each(self) -> None:
        for n in list_available_baselines():
            m = build_baseline(n)
            self.assertIsInstance(m, BaseBaseline)
            self.assertEqual(m.NAME, n)

    def test_factory_unknown_raises(self) -> None:
        with self.assertRaises(KeyError):
            build_baseline("DoesNotExist")

    def test_stubs_raise_not_implemented(self) -> None:
        # Stub forward/train_step MUST raise NotImplementedError until
        # the user replaces the body with the upstream authors' code.
        m = build_baseline("iTransformer")
        with self.assertRaises(NotImplementedError):
            m.forward(None)
        with self.assertRaises(NotImplementedError):
            m.train_step(None, None)

    def test_to_dict_round_trip(self) -> None:
        m = build_baseline("PatchTST")
        d = m.to_dict()
        self.assertEqual(d["name"], "PatchTST")
        self.assertIn("config", d)
        self.assertIn("lookback", d["config"])


class TestLeaderboardLoader(unittest.TestCase):
    def test_load_leaderboard_returns_dict(self) -> None:
        data = load_leaderboard()
        self.assertIsInstance(data, dict)
        self.assertIn("baselines", data)
        self.assertIn("_meta", data)

    def test_leaderboard_has_twelve_baselines(self) -> None:
        data = load_leaderboard()
        self.assertEqual(len(data["baselines"]), 12)

    def test_leaderboard_initially_unpopulated(self) -> None:
        # The shipped JSON has scores=null. This test will fail
        # (intentionally) once fetch_leaderboard.py has populated the
        # file -- at which point the test should be updated to assert
        # is_populated() == True.
        data = load_leaderboard()
        all_null = all(
            entry.get("scores") is None
            for entry in data["baselines"].values()
        )
        # Either all null (shipped) OR all populated (post-fetch) --
        # never a partial state, which would be a hallucination risk.
        all_populated = all(
            entry.get("scores") is not None
            for entry in data["baselines"].values()
        )
        self.assertTrue(
            all_null or all_populated,
            "Leaderboard JSON is in a partial-populated state. "
            "Either fully populate it (fetch_leaderboard.py) or "
            "reset all scores to null.",
        )

    def test_get_baseline_scores_raises_when_unpopulated(self) -> None:
        data = load_leaderboard()
        all_null = all(
            entry.get("scores") is None
            for entry in data["baselines"].values()
        )
        if not all_null:
            self.skipTest("leaderboard is populated; skip unpopulated test")
        with self.assertRaises(LeaderboardNotPopulatedError):
            from baselines import get_baseline_scores
            get_baseline_scores("iTransformer", "Solar")

    def test_unknown_baseline_raises_keyerror(self) -> None:
        with self.assertRaises(KeyError):
            from baselines import get_baseline_scores
            get_baseline_scores("NoSuchModel", "Solar")


if __name__ == "__main__":
    unittest.main()
