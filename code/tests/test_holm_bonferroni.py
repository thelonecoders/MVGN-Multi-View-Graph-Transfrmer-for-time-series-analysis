"""Unit tests for Holm-Bonferroni correction in run_all_experiments.py.

Reference: Holm (1979) "A simple sequentially rejective multiple test
procedure", Scandinavian Journal of Statistics 6(2): 65-70.
"""
import os
import sys

import pytest

# Make the scripts package importable
CODE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CODE_ROOT)

from scripts.run_all_experiments import holm_bonferroni


class TestHolmBonferroni:
    def test_textbook_example(self):
        """Wikipedia Holm-Bonferroni worked example."""
        ps = [0.005, 0.01, 0.02, 0.04, 0.5]
        result = holm_bonferroni(ps, alpha=0.05)
        adjs = [r[0] for r in result]
        rejects = [r[1] for r in result]
        # Expected adjusted (with monotonicity): 0.025, 0.04, 0.06, 0.08, 0.5
        assert adjs == pytest.approx([0.025, 0.04, 0.06, 0.08, 0.5], abs=1e-9)
        assert rejects == [True, True, False, False, False]

    def test_empty(self):
        assert holm_bonferroni([]) == []

    def test_single_significant(self):
        assert holm_bonferroni([0.03], alpha=0.05) == [(0.03, True)]

    def test_single_not_significant(self):
        assert holm_bonferroni([0.06], alpha=0.05) == [(0.06, False)]

    def test_monotonicity_enforced(self):
        """When raw adj would decrease, monotonicity must hold."""
        ps = [0.001] * 10 + [0.0001]
        result = holm_bonferroni(ps, alpha=0.05)
        adjs = sorted(r[0] for r in result)
        prev = 0.0
        for a in adjs:
            assert a >= prev - 1e-12, f"monotonicity violated: {a} < {prev}"
            prev = max(prev, a)

    def test_identical_p_values(self):
        """Identical raw p-values must give identical adjusted p-values."""
        ps = [0.01] * 5
        result = holm_bonferroni(ps, alpha=0.05)
        adjs = set(r[0] for r in result)
        assert len(adjs) == 1, f"identical p-values should give identical adj, got {adjs}"

    def test_alpha_threshold(self):
        """p-value exactly at alpha boundary after correction."""
        # With m=2, p1=0.025 -> adj = 0.025 * 2 = 0.05 (boundary)
        ps = [0.025, 0.04]
        result = holm_bonferroni(ps, alpha=0.05)
        # Sorted: 0.025 (rank 0, adj = 0.05), 0.04 (rank 1, adj = 0.04)
        # But monotonicity: max(0.04, 0.05) = 0.05 for second
        adjs = [r[0] for r in result]
        assert adjs[0] == pytest.approx(0.05, abs=1e-9)
        assert adjs[1] == pytest.approx(0.05, abs=1e-9)
        # Both reject at alpha=0.05 (<=)
        assert all(r[1] for r in result)

    def test_capped_at_one(self):
        """Adjusted p-values cannot exceed 1.0."""
        ps = [0.5, 0.6, 0.7, 0.8]
        result = holm_bonferroni(ps, alpha=0.05)
        for adj, _ in result:
            assert adj <= 1.0, f"adjusted p > 1: {adj}"

    def test_preserves_input_order(self):
        """Output order matches input order, not sorted order."""
        ps = [0.04, 0.005, 0.5, 0.01, 0.02]  # shuffled
        result = holm_bonferroni(ps, alpha=0.05)
        # The smallest (0.005, idx 1) should get adj = 0.005*5 = 0.025
        # The largest (0.5, idx 2) should get adj = 0.5*1 = 0.5
        assert result[1][0] == pytest.approx(0.025, abs=1e-9)
        assert result[2][0] == pytest.approx(0.5, abs=1e-9)
