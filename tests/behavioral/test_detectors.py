"""Tests for individual anomaly detectors."""

import numpy as np
import pytest

from cattlevision.behavioral.detectors import (
    ZScoreMotionDetector,
    IsolationForestDetector,
    AppearanceDriftDetector,
    SocialIsolationDetector,
)
from cattlevision.behavioral.features import FeatureExtractor, FEATURE_NAMES
from cattlevision.behavioral.profile import CowBehaviorProfile
from tests.behavioral.conftest import (
    make_kalman_state, make_embedding, make_track_dict, EMBEDDING_DIM,
)


def _ready_profile(n: int = 20, d_cx: float = 0.0, use_if: bool = False) -> CowBehaviorProfile:
    ext = FeatureExtractor()
    rng = np.random.default_rng(0)
    profile = CowBehaviorProfile(
        "cow_001", burn_in_frames=n, use_isolation_forest=use_if,
        n_estimators=10, refit_interval=500,
    )
    emb = make_embedding(rng, EMBEDDING_DIM)
    for _ in range(n):
        ks = make_kalman_state(d_cx=float(rng.normal(d_cx, 0.1)))
        tr = make_track_dict(kalman_state=ks, embedding=emb)
        obs = ext.extract(tr, ks, emb, 1, 0.0, (1280, 720))
        profile.update(obs, ext.to_feature_vector(obs))
    return profile


def _obs_vec(d_cx: float = 0.0, embedding=None, isolation: float = 1.0):
    rng = np.random.default_rng(1)
    ext = FeatureExtractor()
    emb = embedding if embedding is not None else make_embedding(rng, EMBEDDING_DIM)
    ks = make_kalman_state(d_cx=d_cx)
    tr = make_track_dict(kalman_state=ks, embedding=emb)
    obs = ext.extract(tr, ks, emb, 1, 0.0, (1280, 720), isolation_score=isolation)
    return obs, ext.to_feature_vector(obs)


class TestZScoreMotionDetector:
    def test_returns_none_before_ready(self):
        det = ZScoreMotionDetector()
        profile = CowBehaviorProfile("c", burn_in_frames=100)
        obs, vec = _obs_vec()
        assert det.score(obs, vec, profile) is None

    def test_silent_on_normal_motion(self):
        profile = _ready_profile(n=50, d_cx=0.0)
        det = ZScoreMotionDetector(threshold=3.0, features=["d_cx", "speed"])
        obs, vec = _obs_vec(d_cx=0.0)
        score = det.score(obs, vec, profile)
        assert score is not None
        assert score < 0.5

    def test_fires_on_speed_spike(self):
        profile = _ready_profile(n=50, d_cx=0.0)
        det = ZScoreMotionDetector(threshold=3.0, features=["d_cx", "speed"])
        obs, vec = _obs_vec(d_cx=100.0)
        score = det.score(obs, vec, profile)
        assert score is not None
        assert score > 0.8


class TestIsolationForestDetector:
    def test_returns_none_without_fitted_model(self):
        try:
            import sklearn  # noqa
        except ImportError:
            pytest.skip("sklearn not installed")
        det = IsolationForestDetector()
        profile = CowBehaviorProfile("c", burn_in_frames=100, use_isolation_forest=True)
        obs, vec = _obs_vec()
        assert det.score(obs, vec, profile) is None

    def test_score_in_range_when_fitted(self):
        try:
            import sklearn  # noqa
        except ImportError:
            pytest.skip("sklearn not installed")
        profile = _ready_profile(n=20, use_if=True)
        det = IsolationForestDetector()
        obs, vec = _obs_vec()
        score = det.score(obs, vec, profile)
        assert score is not None
        assert 0.0 <= score <= 1.0


class TestAppearanceDriftDetector:
    def test_returns_none_before_ready(self):
        det = AppearanceDriftDetector()
        profile = CowBehaviorProfile("c", burn_in_frames=100)
        obs, vec = _obs_vec()
        assert det.score(obs, vec, profile) is None

    def test_low_score_for_stable_embedding(self):
        rng = np.random.default_rng(0)
        emb = make_embedding(rng, EMBEDDING_DIM)
        profile = _ready_profile(n=20)
        # Override embedding centroid
        profile._embedding_centroid = emb.copy()
        det = AppearanceDriftDetector(threshold=0.3)
        obs, vec = _obs_vec(embedding=emb)
        score = det.score(obs, vec, profile)
        assert score is not None
        assert score < 0.05

    def test_high_score_for_drifted_embedding(self):
        rng = np.random.default_rng(0)
        emb_a = make_embedding(rng, EMBEDDING_DIM)
        emb_b = make_embedding(rng, EMBEDDING_DIM)
        profile = _ready_profile(n=20)
        profile._embedding_centroid = emb_a.copy()
        det = AppearanceDriftDetector(threshold=0.3)
        obs, vec = _obs_vec(embedding=emb_b)
        score = det.score(obs, vec, profile)
        assert score is not None
        assert score > 0.0


class TestSocialIsolationDetector:
    def test_returns_none_before_ready(self):
        det = SocialIsolationDetector()
        profile = CowBehaviorProfile("c", burn_in_frames=100)
        obs, vec = _obs_vec()
        assert det.score(obs, vec, profile) is None

    def test_zero_score_for_normal_isolation(self):
        profile = _ready_profile(n=20)
        det = SocialIsolationDetector(threshold=2.5)
        obs, vec = _obs_vec(isolation=1.0)
        score = det.score(obs, vec, profile)
        assert score == 0.0

    def test_positive_score_when_isolated(self):
        profile = _ready_profile(n=20)
        det = SocialIsolationDetector(threshold=2.5)
        obs, vec = _obs_vec(isolation=10.0)
        score = det.score(obs, vec, profile)
        assert score is not None
        assert score > 0.0
