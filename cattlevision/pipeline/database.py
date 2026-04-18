"""Identity database for known cattle.

Stores a gallery of L2-normalised embeddings per cow identity and
supports nearest-neighbour lookup via cosine similarity.

The database can be persisted to / loaded from a NumPy .npz file so that
registered cattle survive process restarts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class IdentityDatabase:
    """Gallery-based identity store.

    Each identity is represented by one or more embedding vectors (one per
    enrolled image).  At query time the similarity is computed against all
    enrolled embeddings and the best match is returned.

    Args:
        similarity_threshold: Cosine similarity below which a query is
                              classified as ``"unknown"`` rather than
                              matched to a known individual.
        aggregation: How to combine multiple embeddings per identity.
                     ``"mean"`` averages them (a single prototype per ID);
                     ``"max"``  takes the maximum similarity over all enrolled
                     embeddings (more robust to appearance variation).
    """

    def __init__(
        self,
        similarity_threshold: float = 0.6,
        aggregation: str = "max",
    ):
        assert aggregation in ("mean", "max")
        self.similarity_threshold = similarity_threshold
        self.aggregation = aggregation

        # cow_id → list of (embedding_dim,) numpy arrays
        self._gallery: Dict[str, List[np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Enrollment
    # ------------------------------------------------------------------

    def register(self, cow_id: str, embedding: np.ndarray) -> None:
        """Add an embedding to the gallery for cow_id."""
        emb = _normalise(embedding)
        self._gallery.setdefault(cow_id, []).append(emb)

    def register_batch(self, cow_ids: List[str], embeddings: np.ndarray) -> None:
        """Register multiple embeddings at once."""
        for cid, emb in zip(cow_ids, embeddings):
            self.register(cid, emb)

    def remove(self, cow_id: str) -> None:
        """Remove all gallery entries for a given identity."""
        self._gallery.pop(cow_id, None)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, embedding: np.ndarray) -> Tuple[str, float]:
        """Find the best-matching identity.

        Returns:
            (cow_id, similarity) — cow_id is ``"unknown"`` when the best
            similarity falls below the threshold.
        """
        if not self._gallery:
            return "unknown", 0.0

        emb = _normalise(embedding)
        best_id, best_sim = "unknown", -1.0

        for cow_id, gallery_embs in self._gallery.items():
            sim = self._similarity(emb, gallery_embs)
            if sim > best_sim:
                best_sim = sim
                best_id = cow_id

        if best_sim < self.similarity_threshold:
            return "unknown", float(best_sim)
        return best_id, float(best_sim)

    def query_batch(self, embeddings: np.ndarray) -> List[Tuple[str, float]]:
        return [self.query(e) for e in embeddings]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the gallery to a .npz file."""
        arrays: Dict[str, np.ndarray] = {}
        for cow_id, embs in self._gallery.items():
            arrays[cow_id] = np.stack(embs)
        np.savez(str(path), **arrays)

    @classmethod
    def load(
        cls,
        path: str | Path,
        similarity_threshold: float = 0.6,
        aggregation: str = "max",
    ) -> "IdentityDatabase":
        db = cls(similarity_threshold=similarity_threshold, aggregation=aggregation)
        data = np.load(str(path))
        for cow_id in data.files:
            embs = data[cow_id]  # (k, D)
            for emb in embs:
                db._gallery.setdefault(cow_id, []).append(emb)
        return db

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def num_identities(self) -> int:
        return len(self._gallery)

    @property
    def identity_names(self) -> List[str]:
        return list(self._gallery.keys())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _similarity(self, query_emb: np.ndarray, gallery_embs: List[np.ndarray]) -> float:
        gallery = np.stack(gallery_embs)  # (k, D)
        sims = gallery @ query_emb        # (k,)
        return float(sims.max() if self.aggregation == "max" else sims.mean())


def _normalise(embedding: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(embedding)
    return embedding / (norm + 1e-8)
