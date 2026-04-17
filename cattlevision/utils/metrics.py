"""Evaluation metrics for re-identification.

Implements:
  * Cumulative Matching Characteristic (CMC) – rank-k accuracy
  * mean Average Precision (mAP)

Both are standard benchmarks from the person re-ID literature, adapted
here for cattle identification.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def compute_cmc_map(
    embeddings: np.ndarray,
    labels: List[int],
    max_rank: int = 10,
) -> Dict[str, float]:
    """Compute CMC curve and mAP using leave-one-out evaluation.

    Each sample is used as a query once; the remaining samples form the
    gallery.  This is appropriate for small datasets where no explicit
    query/gallery split exists.

    Args:
        embeddings: (N, D) float32 array of L2-normalised embeddings.
        labels: Length-N list of integer identity labels.
        max_rank: Maximum rank to report in the CMC curve.

    Returns:
        Dictionary with keys ``"rank1"``, ``"rank5"``, ``"rank10"``,
        ``"mAP"``, and ``"cmc"`` (full CMC array).
    """
    n = len(labels)
    labels_arr = np.array(labels)

    # Cosine similarity matrix (equivalent to dot product on normalised vecs)
    sim = embeddings @ embeddings.T  # (N, N)

    cmc_counts = np.zeros(max_rank, dtype=np.float64)
    ap_sum = 0.0

    for q_idx in range(n):
        q_label = labels_arr[q_idx]

        # Remove the query itself from the ranking
        scores = sim[q_idx].copy()
        scores[q_idx] = -np.inf

        sorted_idx = np.argsort(-scores)
        gallery_labels = labels_arr[sorted_idx]
        matches = (gallery_labels == q_label)

        # CMC: does any correct match appear within rank k?
        for k in range(min(max_rank, n - 1)):
            if matches[:k + 1].any():
                cmc_counts[k] += 1

        # Average Precision
        num_positives = matches.sum()
        if num_positives == 0:
            continue
        precision_at_k = np.cumsum(matches) / (np.arange(len(matches)) + 1)
        ap = (precision_at_k * matches).sum() / num_positives
        ap_sum += ap

    cmc = cmc_counts / n
    return {
        "rank1":  float(cmc[0]) if max_rank >= 1 else 0.0,
        "rank5":  float(cmc[4]) if max_rank >= 5 else 0.0,
        "rank10": float(cmc[9]) if max_rank >= 10 else 0.0,
        "mAP":    float(ap_sum / n),
        "cmc":    cmc.tolist(),
    }
