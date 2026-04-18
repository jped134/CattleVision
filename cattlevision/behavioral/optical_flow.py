"""Sparse Lucas-Kanade optical flow estimator, bounded to detection bboxes.

Maintains per-track state (previous frame + previous feature points) so that
flow is estimated incrementally between consecutive frames.  On the first
call for a new track, or when no corners are found, returns 0.0.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np


# LK pyramid parameters
_LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)


class OpticalFlowEstimator:
    """Sparse optical flow within bounding boxes, per track_id.

    Args:
        max_corners: Maximum Shi-Tomasi corners to detect per bbox.
        quality_level: Shi-Tomasi quality threshold (0–1).
        min_distance: Minimum pixel distance between detected corners.
    """

    def __init__(
        self,
        max_corners: int = 20,
        quality_level: float = 0.3,
        min_distance: float = 7.0,
    ) -> None:
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance

        # track_id → (prev_gray_roi, prev_pts, bbox_offset)
        self._state: Dict[int, Tuple[np.ndarray, np.ndarray, Tuple[int, int]]] = {}

    def estimate(
        self,
        frame_gray: np.ndarray,
        prev_frame_gray: Optional[np.ndarray],
        bbox: np.ndarray,
        track_id: int,
    ) -> float:
        """Return mean optical flow magnitude within bbox (0.0 on first call).

        Args:
            frame_gray: Current grayscale frame (H×W uint8).
            prev_frame_gray: Previous grayscale frame; None on first frame.
            bbox: [x1, y1, x2, y2] pixel coordinates.
            track_id: Unique integer track identifier.

        Returns:
            Mean Euclidean magnitude of tracked feature point displacements,
            in pixels per frame.
        """
        if prev_frame_gray is None:
            self._seed_corners(frame_gray, bbox, track_id)
            return 0.0

        state = self._state.get(track_id)
        if state is None:
            self._seed_corners(frame_gray, bbox, track_id)
            return 0.0

        prev_gray_roi, prev_pts, offset = state
        if prev_pts is None or len(prev_pts) == 0:
            self._seed_corners(frame_gray, bbox, track_id)
            return 0.0

        # Extract current ROI
        x1, y1, x2, y2 = self._clip_bbox(bbox, frame_gray.shape)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        curr_gray_roi = frame_gray[y1:y2, x1:x2]

        # Compute sparse LK flow
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray_roi, curr_gray_roi, prev_pts, None, **_LK_PARAMS
        )

        if curr_pts is None or status is None:
            self._seed_corners(frame_gray, bbox, track_id)
            return 0.0

        good_prev = prev_pts[status.ravel() == 1]
        good_curr = curr_pts[status.ravel() == 1]

        if len(good_prev) == 0:
            self._seed_corners(frame_gray, bbox, track_id)
            return 0.0

        magnitudes = np.linalg.norm(good_curr - good_prev, axis=1)
        mag = float(magnitudes.mean())

        # Re-seed corners for next frame
        self._seed_corners(frame_gray, bbox, track_id)
        return mag

    def reset_track(self, track_id: int) -> None:
        """Remove cached state for a track that has expired."""
        self._state.pop(track_id, None)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _seed_corners(self, gray: np.ndarray, bbox: np.ndarray, track_id: int) -> None:
        x1, y1, x2, y2 = self._clip_bbox(bbox, gray.shape)
        if x2 <= x1 or y2 <= y1:
            self._state[track_id] = (gray[0:1, 0:1], np.empty((0, 1, 2), dtype=np.float32), (x1, y1))
            return
        roi = gray[y1:y2, x1:x2]
        pts = cv2.goodFeaturesToTrack(
            roi,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
        )
        if pts is None:
            pts = np.empty((0, 1, 2), dtype=np.float32)
        self._state[track_id] = (roi.copy(), pts, (x1, y1))

    @staticmethod
    def _clip_bbox(bbox: np.ndarray, shape: tuple) -> Tuple[int, int, int, int]:
        H, W = shape[:2]
        x1 = int(max(0, bbox[0]))
        y1 = int(max(0, bbox[1]))
        x2 = int(min(W, bbox[2]))
        y2 = int(min(H, bbox[3]))
        return x1, y1, x2, y2
