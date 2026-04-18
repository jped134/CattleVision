"""Multi-frame IoU + appearance tracker for cattle.

Implements a lightweight variant of SORT (Simple Online and Realtime
Tracking) extended with appearance re-ID to handle occlusion and
identity recovery across frames:

  1. Predict new positions with a constant-velocity Kalman filter.
  2. Match detections to existing tracks using a cost matrix that blends
     IoU overlap and cosine embedding distance.
  3. Promote unmatched detections to new tentative tracks.
  4. Expire tracks that go unmatched for ``max_age`` consecutive frames.
  5. Once a track surpasses ``min_hits`` confirmed frames its identity
     is propagated back to the calling pipeline.

The tracker operates on CowIdentifier outputs so it requires no
additional model — it reuses the embeddings already produced during
identification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Kalman filter (constant velocity, state = [cx, cy, s, r, dcx, dcy, ds])
# ---------------------------------------------------------------------------

class KalmanBoxTracker:
    """Predict and update a single bounding-box track using a Kalman filter.

    State vector: [cx, cy, scale, aspect_ratio, d_cx, d_cy, d_scale]
    Observation:  [cx, cy, scale, aspect_ratio]

    'scale' = w*h (area), 'aspect_ratio' = w/h (assumed constant).
    """

    count = 0

    def __init__(self, bbox: np.ndarray):
        from scipy.linalg import block_diag

        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.time_since_update = 0

        # State transition matrix (constant velocity)
        dt = 1
        self._F = np.eye(7)
        self._F[0, 4] = dt
        self._F[1, 5] = dt
        self._F[2, 6] = dt

        # Observation matrix
        self._H = np.zeros((4, 7))
        self._H[:4, :4] = np.eye(4)

        # Uncertainty matrices
        self._P = np.diag([10., 10., 10., 10., 1e4, 1e4, 1e4])
        self._Q = np.diag([1., 1., 1., 1., 0.01, 0.01, 0.0001])
        self._R = np.diag([1., 1., 10., 10.])

        self._x = np.zeros((7, 1))
        obs = _bbox_to_z(bbox)
        self._x[:4] = obs

    def predict(self) -> np.ndarray:
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.age += 1

        if self._x[6] + self._x[2] <= 0:
            self._x[6] *= 0.0

        self._P = self._F @ self._P @ self._F.T + self._Q
        self._x = self._F @ self._x
        return _z_to_bbox(self._x)

    def update(self, bbox: np.ndarray) -> None:
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        z = _bbox_to_z(bbox)
        y = z - self._H @ self._x
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)
        self._x = self._x + K @ y
        self._P = (np.eye(7) - K @ self._H) @ self._P

    @property
    def bbox(self) -> np.ndarray:
        return _z_to_bbox(self._x)


# ---------------------------------------------------------------------------
# Track object
# ---------------------------------------------------------------------------

@dataclass
class Track:
    kalman: KalmanBoxTracker
    cow_id: str
    similarity: float
    embedding: np.ndarray = field(repr=False)
    confirmed: bool = False


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class CowTracker:
    """Associate detections across frames to maintain persistent identities.

    Args:
        max_age: Frames a track can go unmatched before deletion.
        min_hits: Confirmed matches required before a track is reported.
        iou_threshold: Minimum IoU for an assignment to be valid.
        appearance_weight: Blend weight for appearance vs IoU in cost matrix
                           (0 = pure IoU, 1 = pure appearance distance).
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        appearance_weight: float = 0.4,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.appearance_weight = appearance_weight
        self._tracks: List[Track] = []
        KalmanBoxTracker.count = 0

    def update(self, results) -> List[dict]:
        """Process one frame of identification results.

        Args:
            results: List of IdentificationResult from CowIdentifier.

        Returns:
            List of dicts with keys: ``track_id``, ``cow_id``,
            ``similarity``, ``bbox``, ``confirmed``.
        """
        # Predict positions for existing tracks
        predicted_bboxes = []
        tracks_to_keep = []
        for track in self._tracks:
            pred = track.kalman.predict()
            if np.any(np.isnan(pred)):
                continue
            tracks_to_keep.append(track)
            predicted_bboxes.append(pred)
        self._tracks = tracks_to_keep

        if not results:
            self._expire_old_tracks()
            return []

        det_bboxes    = np.array([r.bbox for r in results], dtype=float)
        det_embeddings = np.stack([r.embedding for r in results])

        if self._tracks:
            pred_bboxes = np.array(predicted_bboxes, dtype=float)
            cost = self._cost_matrix(pred_bboxes, det_bboxes, det_embeddings)
            row_ind, col_ind = linear_sum_assignment(cost)

            matched_tracks  = set()
            matched_dets    = set()
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] > (1 - self.iou_threshold):
                    continue
                self._tracks[r].kalman.update(det_bboxes[c])
                self._tracks[r].cow_id    = results[c].cow_id
                self._tracks[r].similarity = results[c].similarity
                self._tracks[r].embedding  = results[c].embedding
                self._tracks[r].confirmed  = (
                    self._tracks[r].kalman.hit_streak >= self.min_hits
                )
                matched_tracks.add(r)
                matched_dets.add(c)
        else:
            matched_tracks, matched_dets = set(), set()

        # Spawn new tracks for unmatched detections
        for c, res in enumerate(results):
            if c not in matched_dets:
                kalman = KalmanBoxTracker(res.bbox)
                self._tracks.append(Track(
                    kalman=kalman,
                    cow_id=res.cow_id,
                    similarity=res.similarity,
                    embedding=res.embedding,
                    confirmed=False,
                ))

        self._expire_old_tracks()

        output = []
        for track in self._tracks:
            if track.kalman.time_since_update <= 1:
                output.append({
                    "track_id":  track.kalman.id,
                    "cow_id":    track.cow_id,
                    "similarity": track.similarity,
                    "bbox":      track.kalman.bbox.astype(int),
                    "confirmed": track.confirmed,
                })
        return output

    def reset(self) -> None:
        self._tracks.clear()
        KalmanBoxTracker.count = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _expire_old_tracks(self) -> None:
        self._tracks = [
            t for t in self._tracks
            if t.kalman.time_since_update <= self.max_age
        ]

    def _cost_matrix(
        self,
        pred_bboxes: np.ndarray,
        det_bboxes: np.ndarray,
        det_embeddings: np.ndarray,
    ) -> np.ndarray:
        n_tracks = len(pred_bboxes)
        n_dets   = len(det_bboxes)
        cost = np.ones((n_tracks, n_dets))

        iou_mat = _iou_matrix(pred_bboxes, det_bboxes)
        iou_cost = 1 - iou_mat

        # Appearance cost: 1 - cosine_similarity
        track_embs = np.stack([t.embedding for t in self._tracks
                               if t.kalman.time_since_update <= self.max_age])
        if track_embs.shape[0] == n_tracks and det_embeddings.shape[0] == n_dets:
            sim_mat = track_embs @ det_embeddings.T
            app_cost = 1 - np.clip(sim_mat, -1, 1)
            cost = (1 - self.appearance_weight) * iou_cost + self.appearance_weight * app_cost
        else:
            cost = iou_cost
        return cost


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """Convert [x1,y1,x2,y2] to [cx, cy, s, r]."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2
    cy = bbox[1] + h / 2
    s = w * h
    r = w / (h + 1e-8)
    return np.array([[cx], [cy], [s], [r]], dtype=float)


def _z_to_bbox(z: np.ndarray) -> np.ndarray:
    """Convert [cx, cy, s, r, ...] back to [x1, y1, x2, y2]."""
    cx, cy, s, r = float(z[0]), float(z[1]), float(z[2]), float(z[3])
    w = np.sqrt(max(s * r, 0))
    h = s / (w + 1e-8)
    return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])


def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of [x1,y1,x2,y2] boxes."""
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    ix1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    iy1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    ix2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    iy2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-8)
