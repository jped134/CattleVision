"""Tests for OpticalFlowEstimator."""

import numpy as np
import pytest
import cv2

from cattlevision.behavioral.optical_flow import OpticalFlowEstimator


def _gray(shape=(200, 200), value: int = 128) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint8)


def _checkerboard(shape=(200, 200), tile=20) -> np.ndarray:
    img = np.zeros(shape, dtype=np.uint8)
    for y in range(0, shape[0], tile):
        for x in range(0, shape[1], tile):
            if (y // tile + x // tile) % 2 == 0:
                img[y:y+tile, x:x+tile] = 255
    return img


BBOX = np.array([10, 10, 190, 190])


class TestOpticalFlowEstimator:
    def test_first_frame_returns_zero(self):
        est = OpticalFlowEstimator()
        gray = _checkerboard()
        mag = est.estimate(gray, None, BBOX, track_id=1)
        assert mag == pytest.approx(0.0)

    def test_static_frame_near_zero(self):
        est = OpticalFlowEstimator()
        gray = _checkerboard()
        est.estimate(gray, None, BBOX, track_id=1)
        mag = est.estimate(gray, gray, BBOX, track_id=1)
        assert mag < 1.0

    def test_shifted_frame_nonzero(self):
        est = OpticalFlowEstimator()
        gray1 = _checkerboard()
        # Shift by 5 pixels
        M = np.float32([[1, 0, 5], [0, 1, 5]])
        gray2 = cv2.warpAffine(gray1, M, (gray1.shape[1], gray1.shape[0]))
        est.estimate(gray1, None, BBOX, track_id=1)
        mag = est.estimate(gray2, gray1, BBOX, track_id=1)
        assert mag > 0.0

    def test_reset_track_clears_state(self):
        est = OpticalFlowEstimator()
        gray = _checkerboard()
        est.estimate(gray, None, BBOX, track_id=5)
        assert 5 in est._state
        est.reset_track(5)
        assert 5 not in est._state

    def test_missing_track_returns_zero(self):
        est = OpticalFlowEstimator()
        gray = _checkerboard()
        # Call with prev_frame but no prior seeding
        mag = est.estimate(gray, gray, BBOX, track_id=99)
        assert mag == pytest.approx(0.0)
