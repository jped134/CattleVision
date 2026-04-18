"""Tests for FeatureExtractor and FrameObservation."""

import math
import numpy as np
import pytest

from cattlevision.behavioral.features import (
    FeatureExtractor, FEATURE_DIM, FEATURE_NAMES,
)
from tests.behavioral.conftest import make_kalman_state, make_embedding, make_track_dict


class TestFeatureExtractor:
    def _extractor(self):
        return FeatureExtractor()

    def test_feature_vector_shape(self):
        ext = self._extractor()
        tr = make_track_dict()
        obs = ext.extract(tr, np.array(tr["kalman_state"]), np.array(tr["embedding"]),
                          frame_number=1, timestamp=0.0, frame_wh=(1280, 720))
        vec = ext.to_feature_vector(obs)
        assert vec.shape == (FEATURE_DIM,)
        assert vec.dtype == np.float32

    def test_speed_from_kalman_state(self):
        ext = self._extractor()
        ks = make_kalman_state(d_cx=3.0, d_cy=4.0)
        tr = make_track_dict(kalman_state=ks)
        obs = ext.extract(tr, ks, np.array(tr["embedding"]),
                          frame_number=1, timestamp=0.0, frame_wh=(1280, 720))
        vec = ext.to_feature_vector(obs)
        speed_idx = FEATURE_NAMES.index("speed")
        assert abs(vec[speed_idx] - 5.0) < 1e-4

    def test_cx_norm_bounded(self):
        ext = self._extractor()
        ks = make_kalman_state(cx=640.0, cy=360.0)
        tr = make_track_dict(kalman_state=ks)
        obs = ext.extract(tr, ks, np.array(tr["embedding"]),
                          frame_number=1, timestamp=0.0, frame_wh=(1280, 720))
        vec = ext.to_feature_vector(obs)
        cx_idx = FEATURE_NAMES.index("cx_norm")
        assert 0.0 <= vec[cx_idx] <= 1.0

    def test_optical_flow_passthrough(self):
        ext = self._extractor()
        tr = make_track_dict()
        obs = ext.extract(tr, np.array(tr["kalman_state"]), np.array(tr["embedding"]),
                          frame_number=1, timestamp=0.0, frame_wh=(1280, 720),
                          optical_flow_mag=7.5)
        vec = ext.to_feature_vector(obs)
        of_idx = FEATURE_NAMES.index("optical_flow_mag")
        assert abs(vec[of_idx] - 7.5) < 1e-5

    def test_feature_names_length_matches_dim(self):
        assert len(FEATURE_NAMES) == FEATURE_DIM
