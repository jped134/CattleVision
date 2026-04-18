"""Tests for WelfordEstimator and CowBehaviorProfile."""

import math
import numpy as np
import pytest

from cattlevision.behavioral.profile import WelfordEstimator, CowBehaviorProfile
from cattlevision.behavioral.features import FeatureExtractor, FEATURE_DIM
from tests.behavioral.conftest import (
    make_kalman_state, make_embedding, make_track_dict, EMBEDDING_DIM, RNG_SEED,
)


# ---------------------------------------------------------------------------
# WelfordEstimator
# ---------------------------------------------------------------------------

class TestWelfordEstimator:
    def test_mean_converges(self):
        w = WelfordEstimator()
        for _ in range(1000):
            w.update(5.0)
        assert abs(w.mean - 5.0) < 1e-6

    def test_std_converges(self):
        rng = np.random.default_rng(0)
        w = WelfordEstimator()
        for x in rng.normal(0, 2.0, 2000):
            w.update(float(x))
        assert abs(w.std - 2.0) < 0.1

    def test_n_increments(self):
        w = WelfordEstimator()
        for i in range(10):
            w.update(float(i))
        assert w.n == 10

    def test_initial_std_safe(self):
        w = WelfordEstimator()
        w.update(3.0)
        assert w.std >= 0.0  # no error on n=1


# ---------------------------------------------------------------------------
# CowBehaviorProfile
# ---------------------------------------------------------------------------

def _make_profile(burn_in: int = 10, use_if: bool = False) -> CowBehaviorProfile:
    return CowBehaviorProfile(
        cow_id="cow_001",
        burn_in_frames=burn_in,
        refit_interval=500,
        history_maxlen=200,
        use_isolation_forest=use_if,
    )


def _make_obs_and_vec(track_id: int = 1, d_cx: float = 0.0, embedding=None):
    ext = FeatureExtractor()
    rng = np.random.default_rng(track_id)
    emb = embedding if embedding is not None else make_embedding(rng, EMBEDDING_DIM)
    ks = make_kalman_state(d_cx=d_cx)
    tr = make_track_dict(track_id=track_id, kalman_state=ks, embedding=emb)
    obs = ext.extract(tr, ks, emb, frame_number=1, timestamp=0.0, frame_wh=(1280, 720))
    vec = ext.to_feature_vector(obs)
    return obs, vec


class TestCowBehaviorProfile:
    def test_not_ready_before_burn_in(self):
        profile = _make_profile(burn_in=10)
        obs, vec = _make_obs_and_vec()
        for _ in range(9):
            profile.update(obs, vec)
        assert not profile.is_ready

    def test_ready_after_burn_in(self):
        profile = _make_profile(burn_in=10)
        obs, vec = _make_obs_and_vec()
        for _ in range(10):
            profile.update(obs, vec)
        assert profile.is_ready

    def test_zscore_zero_for_mean_value(self):
        profile = _make_profile(burn_in=5)
        obs, vec = _make_obs_and_vec(d_cx=2.0)
        for _ in range(5):
            profile.update(obs, vec)
        scores = profile.zscore_anomaly_score(vec)
        # Speed should be near 0 since this IS the mean
        assert scores["d_cx"] < 0.01

    def test_zscore_high_for_outlier(self):
        profile = _make_profile(burn_in=50)
        rng = np.random.default_rng(1)
        ext = FeatureExtractor()
        # Feed stable data
        for _ in range(50):
            ks = make_kalman_state(d_cx=float(rng.normal(0, 0.1)))
            tr = make_track_dict(kalman_state=ks)
            obs = ext.extract(tr, ks, np.array(tr["embedding"]),
                              frame_number=1, timestamp=0.0, frame_wh=(1280, 720))
            profile.update(obs, ext.to_feature_vector(obs))
        # Now query with a 10-sigma outlier
        ks_outlier = make_kalman_state(d_cx=100.0)
        tr_out = make_track_dict(kalman_state=ks_outlier)
        obs_out = ext.extract(tr_out, ks_outlier, np.array(tr_out["embedding"]),
                              frame_number=51, timestamp=0.0, frame_wh=(1280, 720))
        vec_out = ext.to_feature_vector(obs_out)
        scores = profile.zscore_anomaly_score(vec_out)
        assert scores["d_cx"] > 0.9

    def test_appearance_drift_zero_stable(self):
        profile = _make_profile(burn_in=5)
        rng = np.random.default_rng(0)
        emb = make_embedding(rng, EMBEDDING_DIM)
        obs, vec = _make_obs_and_vec(embedding=emb)
        for _ in range(5):
            profile.update(obs, vec)
        drift = profile.appearance_drift_score(emb)
        assert drift < 0.05

    def test_appearance_drift_high_for_orthogonal(self):
        profile = _make_profile(burn_in=50)
        rng = np.random.default_rng(0)
        emb_a = make_embedding(rng, EMBEDDING_DIM)
        obs, vec = _make_obs_and_vec(embedding=emb_a)
        for _ in range(50):
            profile.update(obs, vec)
        # Orthogonal embedding (negate)
        emb_b = make_embedding(rng, EMBEDDING_DIM)
        emb_b -= 2 * np.dot(emb_b, emb_a) * emb_a  # project out a component
        emb_b /= np.linalg.norm(emb_b) + 1e-8
        drift = profile.appearance_drift_score(emb_b)
        assert drift > 0.1

    def test_isolation_forest_fitted_after_burn_in(self):
        try:
            import sklearn  # noqa
        except ImportError:
            pytest.skip("sklearn not installed")
        profile = _make_profile(burn_in=15, use_if=True)
        obs, vec = _make_obs_and_vec()
        for _ in range(15):
            profile.update(obs, vec)
        assert profile._if_model is not None

    def test_isolation_forest_score_in_range(self):
        try:
            import sklearn  # noqa
        except ImportError:
            pytest.skip("sklearn not installed")
        profile = _make_profile(burn_in=15, use_if=True)
        obs, vec = _make_obs_and_vec()
        for _ in range(15):
            profile.update(obs, vec)
        score = profile.isolation_forest_score(vec)
        assert 0.0 <= score <= 1.0

    def test_summary_has_expected_keys(self):
        profile = _make_profile(burn_in=5)
        obs, vec = _make_obs_and_vec()
        for _ in range(5):
            profile.update(obs, vec)
        s = profile.summary()
        for key in ("cow_id", "frames_seen", "is_ready", "if_fitted", "feature_means"):
            assert key in s
