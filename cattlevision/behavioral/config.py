"""Configuration dataclass for the behavioral anomaly detection layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class BehavioralConfig:
    enabled: bool = True

    # Profiling
    burn_in_frames: int = 300
    refit_interval: int = 500
    history_maxlen: int = 2000
    embedding_ema_alpha: float = 0.05

    # Optical flow
    optical_flow_enabled: bool = True
    optical_flow_max_corners: int = 20

    # Herd analysis
    social_analysis_enabled: bool = True
    social_min_herd_size: int = 2
    social_isolation_threshold: float = 2.5

    # Z-score motion detector
    zscore_threshold: float = 3.0
    zscore_motion_features: List[str] = field(default_factory=lambda: [
        "speed", "d_scale", "optical_flow_mag"
    ])

    # Isolation Forest
    isolation_forest_enabled: bool = True
    isolation_forest_n_estimators: int = 50
    isolation_forest_contamination: float = 0.05
    isolation_forest_threshold: float = 0.60

    # Appearance drift
    appearance_drift_enabled: bool = True
    appearance_drift_threshold: float = 0.30

    # Alert gating
    composite_alert_threshold: float = 0.65
    alert_cooldown_frames: int = 30

    # Composite score weights (values should sum to 1.0)
    detector_weights: Dict[str, float] = field(default_factory=lambda: {
        "zscore_motion":    0.30,
        "isolation_forest": 0.35,
        "appearance_drift": 0.20,
        "social_isolation": 0.15,
    })

    @classmethod
    def from_dict(cls, d: dict) -> "BehavioralConfig":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in d.items() if k in valid}
        return cls(**kwargs)
