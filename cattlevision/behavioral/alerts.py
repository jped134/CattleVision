"""Alert data structures for behavioral anomaly events."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np


@dataclass
class AnomalyAlert:
    """One anomaly event for a single identified cow."""

    # Identity
    cow_id: str
    track_id: int

    # Timing
    frame_number: int
    timestamp: float

    # Classification
    anomaly_type: str     # "motion" | "appearance" | "social" | "composite"
    anomaly_subtype: str  # "high_speed" | "embedding_drift" | "isolation" | …

    # Scoring
    score: float                       # [0.0, 1.0]
    detector_scores: Dict[str, float]  # {detector_name: score}

    # Spatial
    bbox: np.ndarray  # [x1, y1, x2, y2]

    # Diagnostic context
    feature_snapshot: Dict[str, float] = field(default_factory=dict)
    frames_in_profile: int = 0

    def to_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "bbox"}
        d["bbox"] = self.bbox.tolist() if isinstance(self.bbox, np.ndarray) else self.bbox
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def __str__(self) -> str:
        return (
            f"ALERT  cow={self.cow_id}  track={self.track_id}"
            f"  frame={self.frame_number}  type={self.anomaly_type}/{self.anomaly_subtype}"
            f"  score={self.score:.3f}"
        )


class AlertBuffer:
    """Suppress duplicate alerts for the same cow within a cooldown window.

    When a cow triggers an alert, further alerts for that cow are suppressed for
    ``cooldown_frames`` frames so that a sustained anomaly does not flood logs.
    """

    def __init__(self, cooldown_frames: int = 30):
        self.cooldown_frames = cooldown_frames
        self._last_alert_frame: Dict[str, int] = {}  # cow_id → last frame emitted

    def maybe_emit(self, alert: AnomalyAlert) -> Optional[AnomalyAlert]:
        """Return the alert if it passes the cooldown gate, else None."""
        last = self._last_alert_frame.get(alert.cow_id, -self.cooldown_frames - 1)
        if alert.frame_number - last >= self.cooldown_frames:
            self._last_alert_frame[alert.cow_id] = alert.frame_number
            return alert
        return None

    def reset(self) -> None:
        self._last_alert_frame.clear()
