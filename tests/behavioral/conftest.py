"""Shared fixtures and synthetic data generators for behavioral tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import pytest


RNG_SEED = 42
EMBEDDING_DIM = 64   # small dim for fast tests
FRAME_WH = (1280, 720)


def make_embedding(rng: np.random.Generator, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Return a random L2-normalised embedding."""
    v = rng.standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def make_kalman_state(
    cx: float = 640.0,
    cy: float = 360.0,
    scale: float = 10000.0,
    aspect: float = 1.5,
    d_cx: float = 0.0,
    d_cy: float = 0.0,
    d_scale: float = 0.0,
) -> np.ndarray:
    return np.array([cx, cy, scale, aspect, d_cx, d_cy, d_scale], dtype=np.float32)


def make_track_dict(
    track_id: int = 1,
    cow_id: str = "cow_001",
    similarity: float = 0.85,
    bbox: Optional[List[int]] = None,
    confirmed: bool = True,
    kalman_state: Optional[np.ndarray] = None,
    embedding: Optional[np.ndarray] = None,
) -> dict:
    rng = np.random.default_rng(track_id)
    return {
        "track_id":    track_id,
        "cow_id":      cow_id,
        "similarity":  similarity,
        "confidence":  0.9,
        "bbox":        np.array(bbox or [580, 300, 700, 420]),
        "confirmed":   confirmed,
        "kalman_state": (kalman_state if kalman_state is not None else make_kalman_state()).tolist(),
        "embedding":   embedding if embedding is not None else make_embedding(rng),
    }


class SyntheticScenario:
    """Generate sequences of tracker_results dicts for programmatic test scenarios."""

    def __init__(self, seed: int = RNG_SEED, embedding_dim: int = EMBEDDING_DIM):
        self.rng = np.random.default_rng(seed)
        self.embedding_dim = embedding_dim

    def normal_sequence(
        self,
        n_frames: int = 350,
        n_cows: int = 2,
    ) -> List[List[dict]]:
        """Smooth Brownian-motion cow positions, stable scale, stable similarity."""
        cow_ids = [f"cow_{i+1:03d}" for i in range(n_cows)]
        positions = {cid: np.array([200.0 + i * 300, 360.0]) for i, cid in enumerate(cow_ids)}
        embeddings = {cid: make_embedding(self.rng, self.embedding_dim) for cid in cow_ids}

        frames = []
        for _ in range(n_frames):
            results = []
            for tid, cid in enumerate(cow_ids, 1):
                positions[cid] += self.rng.standard_normal(2) * 2.0
                cx, cy = positions[cid]
                ks = make_kalman_state(cx=cx, cy=cy, d_cx=self.rng.normal(0, 0.5),
                                       d_cy=self.rng.normal(0, 0.5))
                results.append(make_track_dict(
                    track_id=tid,
                    cow_id=cid,
                    kalman_state=ks,
                    embedding=embeddings[cid],
                    bbox=[int(cx-60), int(cy-60), int(cx+60), int(cy+60)],
                ))
            frames.append(results)
        return frames

    def inject_speed_spike(
        self,
        frames: List[List[dict]],
        cow_id: str,
        at_frame: int,
        magnitude: float = 30.0,
    ) -> List[List[dict]]:
        """Replace one cow's d_cx/d_cy with a huge velocity at a specific frame."""
        result = [list(f) for f in frames]
        frame = result[at_frame]
        for i, tr in enumerate(frame):
            if tr["cow_id"] == cow_id:
                ks = list(tr["kalman_state"])
                ks[4] = magnitude   # d_cx
                ks[5] = magnitude   # d_cy
                tr = dict(tr)
                tr["kalman_state"] = ks
                frame[i] = tr
        return result

    def inject_embedding_shift(
        self,
        frames: List[List[dict]],
        cow_id: str,
        at_frame: int,
    ) -> List[List[dict]]:
        """Replace one cow's embedding with an orthogonal vector from at_frame onward."""
        result = [list(f) for f in frames]
        new_emb = make_embedding(np.random.default_rng(999), self.embedding_dim)
        for fi in range(at_frame, len(result)):
            for i, tr in enumerate(result[fi]):
                if tr["cow_id"] == cow_id:
                    tr = dict(tr)
                    tr["embedding"] = new_emb
                    result[fi][i] = tr
        return result

    def inject_isolation(
        self,
        frames: List[List[dict]],
        cow_id: str,
        at_frame: int,
        offset: float = 800.0,
    ) -> List[List[dict]]:
        """Move one cow far from the herd from at_frame onward."""
        result = [list(f) for f in frames]
        for fi in range(at_frame, len(result)):
            for i, tr in enumerate(result[fi]):
                if tr["cow_id"] == cow_id:
                    tr = dict(tr)
                    ks = list(tr["kalman_state"])
                    ks[0] += offset   # cx
                    ks[1] += offset   # cy
                    tr["kalman_state"] = ks
                    tr["bbox"] = [int(ks[0]-60), int(ks[1]-60), int(ks[0]+60), int(ks[1]+60)]
                    result[fi][i] = tr
        return result
