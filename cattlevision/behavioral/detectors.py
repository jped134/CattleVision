"""Anomaly detector implementations.

Each detector implements a simple protocol:
    score(obs, feature_vec, profile) -> Optional[float]

Returning None means the detector has no opinion for this frame (e.g., the
profile is not yet ready).  Returning a float in [0, 1] provides an anomaly
score that the BehavioralMonitor blends into a composite alert.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .features import FrameObservation, FEATURE_NAMES
from .profile import CowBehaviorProfile


class ZScoreMotionDetector:
    """Flag abnormal motion features via per-feature z-score baselines.

    Watches a configurable subset of features (default: speed, d_scale,
    optical_flow_mag) and returns the maximum anomaly score across those
    features.  The z-score threshold maps to a [0,1] score via:
        score = 1 - exp(-z²/2)
    so a 3-sigma outlier yields score ≈ 0.989.
    """

    name = "zscore_motion"
    anomaly_type = "motion"

    def __init__(
        self,
        threshold: float = 3.0,
        features: Optional[List[str]] = None,
    ) -> None:
        self.threshold = threshold
        self.features = features or ["speed", "d_scale", "optical_flow_mag"]

    def score(
        self,
        obs: FrameObservation,
        feature_vec: np.ndarray,
        profile: CowBehaviorProfile,
    ) -> Optional[float]:
        if not profile.is_ready:
            return None
        per_feature = profile.zscore_anomaly_score(feature_vec)
        scores = [per_feature.get(f, 0.0) for f in self.features]
        return float(max(scores)) if scores else None

    @property
    def subtype(self) -> str:
        return "high_speed" if "speed" in self.features else "motion_anomaly"


class IsolationForestDetector:
    """Flag multi-feature anomalies using a fitted Isolation Forest.

    The IF score from sklearn's decision_function is sigmoid-mapped to [0,1].
    Returns None until the profile's burn-in is complete and the model is
    fitted.
    """

    name = "isolation_forest"
    anomaly_type = "composite"

    def __init__(self, threshold: float = 0.6) -> None:
        self.threshold = threshold

    def score(
        self,
        obs: FrameObservation,
        feature_vec: np.ndarray,
        profile: CowBehaviorProfile,
    ) -> Optional[float]:
        if not profile.is_ready or profile._if_model is None:
            return None
        return profile.isolation_forest_score(feature_vec)

    @property
    def subtype(self) -> str:
        return "multi_feature_anomaly"


class AppearanceDriftDetector:
    """Flag appearance changes via cosine drift from the embedding EMA centroid.

    A cow that has rolled in mud, changed lighting, or whose identity is being
    confused across frames will show a high appearance drift score.
    """

    name = "appearance_drift"
    anomaly_type = "appearance"

    def __init__(self, threshold: float = 0.30) -> None:
        self.threshold = threshold

    def score(
        self,
        obs: FrameObservation,
        feature_vec: np.ndarray,
        profile: CowBehaviorProfile,
    ) -> Optional[float]:
        if not profile.is_ready:
            return None
        return profile.appearance_drift_score(obs.embedding)

    @property
    def subtype(self) -> str:
        return "embedding_drift"


class SocialIsolationDetector:
    """Flag cows that are abnormally far from the herd.

    The isolation_score feature (index 11) encodes the ratio of current
    minimum peer distance to the rolling mean minimum peer distance.  A
    z-score of this ratio above threshold indicates unusual separation.
    """

    name = "social_isolation"
    anomaly_type = "social"

    def __init__(self, threshold: float = 2.5) -> None:
        self.threshold = threshold

    def score(
        self,
        obs: FrameObservation,
        feature_vec: np.ndarray,
        profile: CowBehaviorProfile,
    ) -> Optional[float]:
        if not profile.is_ready:
            return None
        # isolation_score > threshold → anomalous
        iso = obs.isolation_score
        if iso <= self.threshold:
            return 0.0
        # Map linearly: threshold → 0, 2*threshold → 1
        raw = (iso - self.threshold) / (self.threshold + 1e-8)
        return float(min(1.0, raw))

    @property
    def subtype(self) -> str:
        return "isolation"
