"""Feature extraction from per-frame tracker state.

Produces a 12-dimensional feature vector for each confirmed cow per frame.
All values are floats; velocities come from the Kalman state directly so
they are already smoothed and do not require finite-difference estimation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


FEATURE_NAMES = [
    "speed",              # 0  sqrt(d_cx² + d_cy²)
    "d_cx",               # 1  horizontal velocity (px/frame)
    "d_cy",               # 2  vertical velocity (px/frame)
    "scale",              # 3  sqrt(w * h)  — proxy for size/distance
    "d_scale",            # 4  rate of area change
    "aspect_ratio",       # 5  w / h — posture proxy (lying ↔ standing)
    "cx_norm",            # 6  cx / frame_width
    "cy_norm",            # 7  cy / frame_height
    "reid_similarity",    # 8  re-ID cosine similarity from IdentityDatabase
    "det_confidence",     # 9  detector confidence score
    "optical_flow_mag",   # 10 mean LK optical flow magnitude within bbox
    "isolation_score",    # 11 min_peer_dist / rolling_mean_min_peer_dist
]

FEATURE_DIM = len(FEATURE_NAMES)  # 12


@dataclass
class FrameObservation:
    """All raw signals for one confirmed track in one frame."""

    cow_id: str
    track_id: int
    frame_number: int
    timestamp: float
    bbox: np.ndarray          # [x1, y1, x2, y2]
    kalman_state: np.ndarray  # shape (7,): [cx, cy, s, r, d_cx, d_cy, d_s]
    embedding: np.ndarray     # shape (embedding_dim,), L2-normalised
    reid_similarity: float
    det_confidence: float
    frame_wh: Tuple[int, int]  # (width, height) for normalisation
    optical_flow_mag: float = 0.0
    isolation_score: float = 1.0


class FeatureExtractor:
    """Build FrameObservation and the FEATURE_DIM-dimensional feature vector."""

    def extract(
        self,
        track_dict: dict,
        kalman_state: np.ndarray,
        embedding: np.ndarray,
        frame_number: int,
        timestamp: float,
        frame_wh: Tuple[int, int],
        optical_flow_mag: float = 0.0,
        isolation_score: float = 1.0,
    ) -> FrameObservation:
        return FrameObservation(
            cow_id=track_dict["cow_id"],
            track_id=track_dict["track_id"],
            frame_number=frame_number,
            timestamp=timestamp,
            bbox=np.asarray(track_dict["bbox"]),
            kalman_state=kalman_state,
            embedding=embedding,
            reid_similarity=float(track_dict.get("similarity", 0.0)),
            det_confidence=float(track_dict.get("confidence", 1.0)),
            frame_wh=frame_wh,
            optical_flow_mag=optical_flow_mag,
            isolation_score=isolation_score,
        )

    def to_feature_vector(self, obs: FrameObservation) -> np.ndarray:
        """Return shape-(FEATURE_DIM,) float32 feature vector."""
        ks = obs.kalman_state  # [cx, cy, s, r, d_cx, d_cy, d_s]
        cx, cy   = float(ks[0]), float(ks[1])
        s        = float(ks[2])   # scale = w*h
        r        = float(ks[3])   # aspect ratio w/h
        d_cx     = float(ks[4])
        d_cy     = float(ks[5])
        d_s      = float(ks[6])

        fw, fh = obs.frame_wh

        vec = np.array([
            math.sqrt(d_cx ** 2 + d_cy ** 2),       # speed
            d_cx,                                     # d_cx
            d_cy,                                     # d_cy
            math.sqrt(max(s, 0.0)),                   # scale
            d_s,                                      # d_scale
            max(r, 1e-4),                             # aspect_ratio
            cx / (fw + 1e-8),                         # cx_norm
            cy / (fh + 1e-8),                         # cy_norm
            obs.reid_similarity,                      # reid_similarity
            obs.det_confidence,                       # det_confidence
            obs.optical_flow_mag,                     # optical_flow_mag
            obs.isolation_score,                      # isolation_score
        ], dtype=np.float32)

        return vec
