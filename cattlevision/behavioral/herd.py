"""Herd-level social behavior analysis.

For each frame, computes a per-cow isolation score based on how far that cow
is from its nearest peer relative to the rolling historical baseline of that
distance.  A score >> 1 means the cow is unusually far from the herd — a
clinically significant indicator of illness or injury.

isolation_score_c = min_dist_to_any_peer_c / (rolling_mean_min_dist_c + ε)
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict

import numpy as np


class HerdAnalyzer:
    """Compute per-cow isolation scores from concurrent cow positions.

    Args:
        rolling_window: Number of frames to average the minimum-distance baseline over.
        min_herd_size: Minimum number of cows required before scoring is meaningful.
                       Singletons always return 1.0 (no peer comparison possible).
    """

    def __init__(self, rolling_window: int = 150, min_herd_size: int = 2) -> None:
        self.rolling_window = rolling_window
        self.min_herd_size = min_herd_size
        # cow_id → deque of recent min-distances (used to build baseline)
        self._min_dist_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=rolling_window)
        )

    def update(self, positions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Compute isolation scores for all cows in this frame.

        Args:
            positions: {cow_id: np.array([cx, cy])} for all confirmed cows.

        Returns:
            {cow_id: isolation_score} — 1.0 when herd is too small to score.
        """
        cow_ids = list(positions.keys())
        scores: Dict[str, float] = {}

        if len(cow_ids) < self.min_herd_size:
            for cid in cow_ids:
                scores[cid] = 1.0
            return scores

        pos_matrix = np.stack([positions[cid] for cid in cow_ids])  # (N, 2)

        # Pairwise Euclidean distance matrix
        diff = pos_matrix[:, None, :] - pos_matrix[None, :, :]      # (N, N, 2)
        dist_mat = np.linalg.norm(diff, axis=-1)                     # (N, N)
        np.fill_diagonal(dist_mat, np.inf)

        for i, cid in enumerate(cow_ids):
            min_dist = float(dist_mat[i].min())
            self._min_dist_history[cid].append(min_dist)

            history = self._min_dist_history[cid]
            baseline = float(np.mean(history)) if history else min_dist
            scores[cid] = float(min_dist / (baseline + 1e-8))

        return scores

    def reset(self) -> None:
        self._min_dist_history.clear()
