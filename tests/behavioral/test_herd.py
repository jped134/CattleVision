"""Tests for HerdAnalyzer."""

import numpy as np
import pytest

from cattlevision.behavioral.herd import HerdAnalyzer


class TestHerdAnalyzer:
    def test_singleton_returns_default(self):
        analyzer = HerdAnalyzer(min_herd_size=2)
        scores = analyzer.update({"cow_001": np.array([100.0, 100.0])})
        assert scores["cow_001"] == pytest.approx(1.0)

    def test_cohesive_herd_scores_near_one(self):
        analyzer = HerdAnalyzer(rolling_window=5)
        positions = {
            "cow_001": np.array([100.0, 100.0]),
            "cow_002": np.array([110.0, 100.0]),
            "cow_003": np.array([120.0, 100.0]),
        }
        # Feed the same positions repeatedly to build a stable baseline
        for _ in range(10):
            scores = analyzer.update(positions)
        for s in scores.values():
            assert 0.5 <= s <= 2.0

    def test_isolated_cow_scores_high(self):
        analyzer = HerdAnalyzer(rolling_window=5, min_herd_size=2)
        base = {
            "cow_001": np.array([100.0, 100.0]),
            "cow_002": np.array([110.0, 100.0]),
            "cow_003": np.array([120.0, 100.0]),
        }
        # Establish baseline
        for _ in range(10):
            analyzer.update(base)
        # Isolate cow_001
        isolated = {
            "cow_001": np.array([2000.0, 2000.0]),
            "cow_002": np.array([110.0, 100.0]),
            "cow_003": np.array([120.0, 100.0]),
        }
        scores = analyzer.update(isolated)
        assert scores["cow_001"] > scores["cow_002"]

    def test_reset_clears_history(self):
        analyzer = HerdAnalyzer()
        positions = {"a": np.array([0.0, 0.0]), "b": np.array([10.0, 0.0])}
        analyzer.update(positions)
        analyzer.reset()
        assert len(analyzer._min_dist_history) == 0
