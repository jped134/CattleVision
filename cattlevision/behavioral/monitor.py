"""BehavioralMonitor — main orchestrator for cow behavioral anomaly detection.

Drop-in layer that sits immediately after CowTracker.update():

    tracker_results = tracker.update(identify_results)
    alerts = monitor.process(
        tracker_results=tracker_results,
        tracks=tracker._tracks,
        frame=frame,
        frame_number=frame_idx,
        timestamp=time.time(),
    )

Per-frame processing steps
---------------------------
1. Compute herd isolation scores from confirmed cow centroids.
2. Compute optical flow magnitudes per active track bbox.
3. For each confirmed track:
   a. Extract FrameObservation and 12-dim feature vector.
   b. Update the cow's CowBehaviorProfile (history, Welford, embedding EMA).
   c. If profile.is_ready: run all enabled detectors, collect scores.
   d. Blend detector scores into composite score using configured weights.
   e. Emit AnomalyAlert if composite score > threshold (AlertBuffer gating).
4. Return List[AnomalyAlert] (empty list if no anomalies or still in burn-in).
"""

from __future__ import annotations

import time as time_module
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .alerts import AlertBuffer, AnomalyAlert
from .config import BehavioralConfig
from .detectors import (
    AppearanceDriftDetector,
    IsolationForestDetector,
    SocialIsolationDetector,
    ZScoreMotionDetector,
)
from .features import FeatureExtractor, FrameObservation
from .herd import HerdAnalyzer
from .optical_flow import OpticalFlowEstimator
from .profile import CowBehaviorProfile


class BehavioralMonitor:
    """Stateful per-cow behavioral anomaly detector.

    Args:
        config: BehavioralConfig instance.
    """

    def __init__(self, config: BehavioralConfig) -> None:
        self.config = config

        self._feature_extractor = FeatureExtractor()
        self._herd_analyzer = HerdAnalyzer(
            min_herd_size=config.social_min_herd_size,
        )
        self._optical_flow = OpticalFlowEstimator(
            max_corners=config.optical_flow_max_corners,
        )
        self._alert_buffer = AlertBuffer(cooldown_frames=config.alert_cooldown_frames)
        self._profiles: Dict[str, CowBehaviorProfile] = {}

        # Build active detector list from config
        self._detectors = self._build_detectors()

        # Previous grayscale frame for optical flow
        self._prev_gray: Optional[np.ndarray] = None

    @classmethod
    def from_config(cls, behavioral_cfg: dict) -> "BehavioralMonitor":
        """Construct from a raw dict (e.g. cfg.get('behavioral', {}))."""
        config = BehavioralConfig.from_dict(behavioral_cfg)
        return cls(config)

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def process(
        self,
        tracker_results: List[dict],
        tracks: list,                       # List[Track] from tracker._tracks
        frame: np.ndarray,                  # BGR frame
        frame_number: int,
        timestamp: Optional[float] = None,
        frame_wh: Optional[Tuple[int, int]] = None,
    ) -> List[AnomalyAlert]:
        """Process one frame of tracker output and return any anomaly alerts."""
        if timestamp is None:
            timestamp = time_module.time()
        if frame_wh is None:
            h, w = frame.shape[:2]
            frame_wh = (w, h)

        if not tracker_results:
            self._update_prev_gray(frame)
            return []

        # Build track_id → Track lookup for Kalman state access
        track_by_id: dict = {t.kalman.id: t for t in tracks}

        # ---- 1. Herd-level isolation scores ----------------------------
        positions: Dict[str, np.ndarray] = {}
        for tr in tracker_results:
            if tr.get("confirmed", False):
                ks = self._get_kalman_state(tr, track_by_id)
                if ks is not None:
                    positions[tr["cow_id"]] = ks[:2]  # (cx, cy)

        isolation_scores = (
            self._herd_analyzer.update(positions)
            if self.config.social_analysis_enabled
            else {}
        )

        # ---- 2. Optical flow per track ---------------------------------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow_mags: Dict[int, float] = {}
        if self.config.optical_flow_enabled:
            for tr in tracker_results:
                tid = tr["track_id"]
                bbox = np.asarray(tr["bbox"])
                flow_mags[tid] = self._optical_flow.estimate(
                    gray, self._prev_gray, bbox, tid
                )

        # ---- 3. Per-track scoring --------------------------------------
        alerts: List[AnomalyAlert] = []
        for tr in tracker_results:
            if not tr.get("confirmed", False):
                continue

            tid = tr["track_id"]
            cow_id = tr["cow_id"]

            ks = self._get_kalman_state(tr, track_by_id)
            emb = self._get_embedding(tr, track_by_id)
            if ks is None or emb is None:
                continue

            obs = self._feature_extractor.extract(
                track_dict=tr,
                kalman_state=ks,
                embedding=emb,
                frame_number=frame_number,
                timestamp=timestamp,
                frame_wh=frame_wh,
                optical_flow_mag=flow_mags.get(tid, 0.0),
                isolation_score=isolation_scores.get(cow_id, 1.0),
            )
            fvec = self._feature_extractor.to_feature_vector(obs)

            profile = self._get_or_create_profile(cow_id)
            profile.update(obs, fvec)

            if not profile.is_ready:
                continue

            # ---- score each detector ----------------------------------
            detector_scores: Dict[str, float] = {}
            primary_type = "composite"
            primary_subtype = "composite"

            for det in self._detectors:
                s = det.score(obs, fvec, profile)
                if s is not None:
                    detector_scores[det.name] = s

            if not detector_scores:
                continue

            # ---- composite weighted score ----------------------------
            weights = self.config.detector_weights
            total_w = sum(weights.get(k, 0.0) for k in detector_scores)
            if total_w <= 0:
                composite = float(max(detector_scores.values()))
            else:
                composite = sum(
                    detector_scores[k] * weights.get(k, 0.0)
                    for k in detector_scores
                ) / total_w

            if composite < self.config.composite_alert_threshold:
                continue

            # Determine primary anomaly type from highest-scoring detector
            top_det_name = max(detector_scores, key=detector_scores.get)
            for det in self._detectors:
                if det.name == top_det_name:
                    primary_type = det.anomaly_type
                    primary_subtype = det.subtype
                    break

            alert = AnomalyAlert(
                cow_id=cow_id,
                track_id=tid,
                frame_number=frame_number,
                timestamp=timestamp,
                anomaly_type=primary_type,
                anomaly_subtype=primary_subtype,
                score=float(min(1.0, composite)),
                detector_scores=detector_scores,
                bbox=obs.bbox.copy(),
                feature_snapshot={
                    name: float(fvec[i])
                    for i, name in enumerate(
                        __import__(
                            "cattlevision.behavioral.features",
                            fromlist=["FEATURE_NAMES"],
                        ).FEATURE_NAMES
                    )
                },
                frames_in_profile=profile.frames_seen,
            )

            emitted = self._alert_buffer.maybe_emit(alert)
            if emitted is not None:
                alerts.append(emitted)

        self._update_prev_gray(frame)
        return alerts

    # ------------------------------------------------------------------
    # Profile management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all per-cow profiles and tracker state."""
        self._profiles.clear()
        self._herd_analyzer.reset()
        self._alert_buffer.reset()
        self._prev_gray = None

    def get_profile(self, cow_id: str) -> Optional[CowBehaviorProfile]:
        return self._profiles.get(cow_id)

    def profiles_summary(self) -> Dict[str, dict]:
        return {cid: p.summary() for cid, p in self._profiles.items()}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_or_create_profile(self, cow_id: str) -> CowBehaviorProfile:
        if cow_id not in self._profiles:
            cfg = self.config
            self._profiles[cow_id] = CowBehaviorProfile(
                cow_id=cow_id,
                burn_in_frames=cfg.burn_in_frames,
                refit_interval=cfg.refit_interval,
                history_maxlen=cfg.history_maxlen,
                embedding_ema_alpha=cfg.embedding_ema_alpha,
                use_isolation_forest=cfg.isolation_forest_enabled,
                n_estimators=cfg.isolation_forest_n_estimators,
                contamination=cfg.isolation_forest_contamination,
            )
        return self._profiles[cow_id]

    def _build_detectors(self) -> list:
        cfg = self.config
        detectors = [
            ZScoreMotionDetector(
                threshold=cfg.zscore_threshold,
                features=cfg.zscore_motion_features,
            ),
        ]
        if cfg.isolation_forest_enabled:
            detectors.append(IsolationForestDetector(
                threshold=cfg.isolation_forest_threshold,
            ))
        if cfg.appearance_drift_enabled:
            detectors.append(AppearanceDriftDetector(
                threshold=cfg.appearance_drift_threshold,
            ))
        if cfg.social_analysis_enabled:
            detectors.append(SocialIsolationDetector(
                threshold=cfg.social_isolation_threshold,
            ))
        return detectors

    def _get_kalman_state(self, tr: dict, track_by_id: dict) -> Optional[np.ndarray]:
        # Prefer the pre-added key in the output dict; fall back to direct access
        if "kalman_state" in tr:
            return np.asarray(tr["kalman_state"], dtype=np.float32)
        tid = tr["track_id"]
        track = track_by_id.get(tid)
        if track is None:
            return None
        return track.kalman._x.flatten().astype(np.float32)

    def _get_embedding(self, tr: dict, track_by_id: dict) -> Optional[np.ndarray]:
        if "embedding" in tr and tr["embedding"] is not None:
            return np.asarray(tr["embedding"], dtype=np.float32)
        tid = tr["track_id"]
        track = track_by_id.get(tid)
        if track is None:
            return None
        return track.embedding.astype(np.float32)

    def _update_prev_gray(self, frame: np.ndarray) -> None:
        self._prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
