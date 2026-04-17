"""Metric learning loss functions for cow re-identification.

Two implementations are provided:

  * TripletLoss – operates on (anchor, positive, negative) triplets.
    Straightforward and easy to debug; use with CowReIDDataset(mode="triplet").

  * BatchHardTripletLoss – mines the hardest valid triplet within each
    mini-batch online (Hermans et al., 2017: "In Defense of the Triplet
    Loss for Person Re-Identification").  More efficient and typically
    converges to better optima; use with CowReIDDataset(mode="single")
    and a sampler that guarantees multiple samples per identity per batch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """Triplet loss with optional L2 normalisation.

    Loss = max(0, ||a - p||² − ||a - n||² + margin)

    Args:
        margin: Minimum desired gap between positive and negative distances.
        normalize: If True, L2-normalise embeddings before computing distances.
                   Should match whether the embedder already normalises output.
    """

    def __init__(self, margin: float = 0.3, normalize: bool = False):
        super().__init__()
        self.margin = margin
        self.normalize = normalize
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2, reduction="mean")

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        if self.normalize:
            anchor   = F.normalize(anchor,   p=2, dim=1)
            positive = F.normalize(positive, p=2, dim=1)
            negative = F.normalize(negative, p=2, dim=1)
        return self.loss_fn(anchor, positive, negative)


class BatchHardTripletLoss(nn.Module):
    """Online batch-hard triplet mining.

    Within a mini-batch, for each anchor selects:
      * hardest positive  – same-label embedding with largest distance
      * hardest negative  – different-label embedding with smallest distance

    Args:
        margin: Triplet margin.  Use ``"soft"`` for softplus margin.
        normalize: L2-normalise before computing pairwise distances.
    """

    def __init__(self, margin: float | str = 0.3, normalize: bool = False):
        super().__init__()
        self.margin = margin
        self.normalize = normalize

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        # Pairwise squared Euclidean distances
        dist_mat = self._pairwise_distances(embeddings)

        # --- hard positive (max distance among same-label pairs) ---
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)   # (B, B)
        # Exclude the diagonal (self-pairs) from positives
        labels_eq.fill_diagonal_(False)
        # Set non-positive entries to 0 so max picks the hardest positive
        pos_dist = (dist_mat * labels_eq.float()).max(dim=1).values

        # --- hard negative (min distance among different-label pairs) ---
        labels_ne = ~(labels.unsqueeze(0) == labels.unsqueeze(1))
        # Set non-negative entries to large value so min picks hardest negative
        neg_dist = dist_mat.clone()
        neg_dist[~labels_ne] = dist_mat.max().item() + 1.0
        neg_dist = neg_dist.min(dim=1).values

        if self.margin == "soft":
            loss = F.softplus(pos_dist - neg_dist).mean()
        else:
            loss = F.relu(pos_dist - neg_dist + self.margin).mean()
        return loss

    @staticmethod
    def _pairwise_distances(embeddings: torch.Tensor) -> torch.Tensor:
        """Compute pairwise squared Euclidean distance matrix."""
        dot = torch.mm(embeddings, embeddings.t())
        sq_norm = torch.diag(dot)
        distances = sq_norm.unsqueeze(0) - 2.0 * dot + sq_norm.unsqueeze(1)
        return distances.clamp(min=0.0)
