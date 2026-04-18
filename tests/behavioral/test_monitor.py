"""Integration tests for BehavioralMonitor."""

import numpy as np
import pytest

from cattlevision.behavioral.monitor import BehavioralMonitor
from cattlevision.behavioral.config import BehavioralConfig
from tests.behavioral.conftest import (
    SyntheticScenario, FRAME_WH,
)

# ---- helpers ---------------------------------------------------------------

def _make_monitor(burn_in: int = 20, use_if: bool = False) -> BehavioralMonitor:
    cfg = BehavioralConfig(
        burn_in_frames=burn_in,
        refit_interval=500,
        optical_flow_enabled=False,   # no real video in tests
        social_analysis_enabled=True,
        isolation_forest_enabled=use_if,
        isolation_forest_n_estimators=10,
        composite_alert_threshold=0.50,
        alert_cooldown_frames=5,
        zscore_threshold=2.5,
        zscore_motion_features=["speed", "d_cx", "d_cy"],
        appearance_drift_enabled=True,
        appearance_drift_threshold=0.20,
        social_isolation_threshold=2.5,
        detector_weights={
            "zscore_motion": 0.50,
            "isolation_forest": 0.00,
            "appearance_drift": 0.30,
            "social_isolation": 0.20,
        },
    )
    return BehavioralMonitor(cfg)


def _blank_frame(w: int = FRAME_WH[0], h: int = FRAME_WH[1]) -> np.ndarray:
    import cv2
    return np.zeros((h, w, 3), dtype=np.uint8)


class _FakeTrack:
    """Minimal Track-like object for tests that don't need Kalman internals."""
    class _Kalman:
        def __init__(self, tid):
            self.id = tid
            self._x = np.zeros((7, 1), dtype=np.float32)
    def __init__(self, tid, emb):
        self.kalman = self._Kalman(tid)
        self.embedding = emb


def _run_sequence(monitor, frames, fake_tracks_fn=None):
    """Feed synthetic frames through the monitor, collect all alerts."""
    frame = _blank_frame()
    all_alerts = []
    for fi, tracker_results in enumerate(frames):
        if fake_tracks_fn is not None:
            tracks = fake_tracks_fn(tracker_results)
        else:
            tracks = []
        alerts = monitor.process(
            tracker_results=tracker_results,
            tracks=tracks,
            frame=frame,
            frame_number=fi,
            timestamp=float(fi),
            frame_wh=FRAME_WH,
        )
        all_alerts.extend(alerts)
    return all_alerts


# ---- tests -----------------------------------------------------------------

class TestBehavioralMonitor:
    def test_no_alerts_during_burn_in(self):
        monitor = _make_monitor(burn_in=100)
        scenario = SyntheticScenario(seed=0)
        frames = scenario.normal_sequence(n_frames=90, n_cows=2)
        # Before burn-in no alerts should fire
        alerts = _run_sequence(monitor, frames)
        assert len(alerts) == 0

    def test_empty_frame_no_crash(self):
        monitor = _make_monitor()
        alerts = monitor.process([], [], _blank_frame(), frame_number=0, timestamp=0.0)
        assert alerts == []

    def test_reset_clears_profiles(self):
        monitor = _make_monitor(burn_in=5)
        scenario = SyntheticScenario(seed=1)
        frames = scenario.normal_sequence(n_frames=5, n_cows=1)
        _run_sequence(monitor, frames)
        assert len(monitor._profiles) > 0
        monitor.reset()
        assert len(monitor._profiles) == 0

    def test_profiles_summary_keys(self):
        monitor = _make_monitor(burn_in=5)
        scenario = SyntheticScenario(seed=2)
        frames = scenario.normal_sequence(n_frames=5, n_cows=1)
        _run_sequence(monitor, frames)
        s = monitor.profiles_summary()
        for v in s.values():
            assert "cow_id" in v
            assert "frames_seen" in v
            assert "is_ready" in v

    def test_alert_has_correct_fields(self):
        monitor = _make_monitor(burn_in=20)
        scenario = SyntheticScenario(seed=3)
        frames = scenario.normal_sequence(n_frames=22, n_cows=2)
        # Inject a huge speed spike after burn-in
        frames = scenario.inject_speed_spike(frames, "cow_001", at_frame=21, magnitude=200.0)
        alerts = _run_sequence(monitor, frames)
        for alert in alerts:
            assert isinstance(alert.cow_id, str)
            assert isinstance(alert.score, float)
            assert 0.0 <= alert.score <= 1.0
            assert alert.bbox.shape == (4,)
            assert isinstance(alert.anomaly_type, str)

    def test_alert_cooldown_deduplicates(self):
        monitor = _make_monitor(burn_in=10, use_if=False)
        monitor.config.alert_cooldown_frames = 50
        scenario = SyntheticScenario(seed=4)
        frames = scenario.normal_sequence(n_frames=15, n_cows=2)
        # Spike every frame from 10 onward
        for fi in range(10, 15):
            frames = scenario.inject_speed_spike(frames, "cow_001", at_frame=fi, magnitude=300.0)
        alerts = _run_sequence(monitor, frames)
        # With cooldown=50, cow_001 should fire at most once
        cow1_alerts = [a for a in alerts if a.cow_id == "cow_001"]
        assert len(cow1_alerts) <= 1

    def test_get_profile_returns_profile_after_update(self):
        monitor = _make_monitor(burn_in=3)
        scenario = SyntheticScenario(seed=5)
        frames = scenario.normal_sequence(n_frames=3, n_cows=1)
        _run_sequence(monitor, frames)
        profile = monitor.get_profile("cow_001")
        assert profile is not None
        assert profile.cow_id == "cow_001"

    def test_from_config_with_dict(self):
        monitor = BehavioralMonitor.from_config({
            "burn_in_frames": 10,
            "isolation_forest_enabled": False,
        })
        assert monitor.config.burn_in_frames == 10
        assert not monitor.config.isolation_forest_enabled


class TestAnomalyAlert:
    def test_to_dict(self):
        from cattlevision.behavioral.alerts import AnomalyAlert
        alert = AnomalyAlert(
            cow_id="cow_001", track_id=1,
            frame_number=100, timestamp=1234.5,
            anomaly_type="motion", anomaly_subtype="high_speed",
            score=0.8, detector_scores={"zscore_motion": 0.8},
            bbox=np.array([10, 20, 100, 200]),
            feature_snapshot={"speed": 30.0},
            frames_in_profile=200,
        )
        d = alert.to_dict()
        assert d["cow_id"] == "cow_001"
        assert isinstance(d["bbox"], list)
        assert d["score"] == pytest.approx(0.8)

    def test_to_json(self):
        import json
        from cattlevision.behavioral.alerts import AnomalyAlert
        alert = AnomalyAlert(
            cow_id="cow_002", track_id=2,
            frame_number=50, timestamp=0.0,
            anomaly_type="social", anomaly_subtype="isolation",
            score=0.7, detector_scores={},
            bbox=np.array([0, 0, 50, 50]),
        )
        parsed = json.loads(alert.to_json())
        assert parsed["cow_id"] == "cow_002"


class TestAlertBuffer:
    def test_first_alert_emitted(self):
        from cattlevision.behavioral.alerts import AlertBuffer, AnomalyAlert
        buf = AlertBuffer(cooldown_frames=10)
        alert = AnomalyAlert(
            cow_id="c", track_id=1, frame_number=5, timestamp=0.0,
            anomaly_type="motion", anomaly_subtype="high_speed",
            score=0.9, detector_scores={}, bbox=np.array([0, 0, 10, 10]),
        )
        assert buf.maybe_emit(alert) is not None

    def test_cooldown_suppresses_repeat(self):
        from cattlevision.behavioral.alerts import AlertBuffer, AnomalyAlert
        buf = AlertBuffer(cooldown_frames=10)
        def _alert(fn):
            return AnomalyAlert(
                cow_id="c", track_id=1, frame_number=fn, timestamp=0.0,
                anomaly_type="motion", anomaly_subtype="high_speed",
                score=0.9, detector_scores={}, bbox=np.array([0, 0, 10, 10]),
            )
        buf.maybe_emit(_alert(0))
        assert buf.maybe_emit(_alert(5)) is None   # within cooldown
        assert buf.maybe_emit(_alert(10)) is not None  # cooldown expired
