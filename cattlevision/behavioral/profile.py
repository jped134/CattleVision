"""Per-cow behavioral profile: baseline accumulation and anomaly scoring.

Lifecycle
---------
BURN_IN  (frames < burn_in_frames) — accumulate history, no scoring.
ACTIVE   (frames >= burn_in_frames) — fit IsolationForest once, score every frame.
REFITTING — triggered every refit_interval frames to track gradual behavioral drift.

The profile owns:
  * Welford online estimators (O(1) memory) for per-feature z-scoring.
  * A rolling deque of feature vectors for IsolationForest refitting.
  * An EMA centroid of embedding vectors for appearance drift scoring.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional

import numpy as np

from .features import FrameObservation, FEATURE_NAMES, FEATURE_DIM


class WelfordEstimator:
    """Incremental mean and variance (Welford's online algorithm, O(1) memory)."""

    def __init__(self) -> None:
        self.n = 0
        self._M = 0.0
        self._S = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self._M
        self._M += delta / self.n
        self._S += delta * (x - self._M)

    @property
    def mean(self) -> float:
        return self._M

    @property
    def var(self) -> float:
        return self._S / (self.n - 1) if self.n > 1 else 1.0

    @property
    def std(self) -> float:
        return math.sqrt(self.var)


class CowBehaviorProfile:
    """Stateful per-cow baseline accumulator and anomaly scorer.

    Args:
        cow_id: The cattle identity this profile belongs to.
        burn_in_frames: Minimum observations before anomaly scoring is active.
        refit_interval: Refit IsolationForest every this many post-burn-in frames.
        history_maxlen: Rolling window size for IsolationForest training data.
        embedding_ema_alpha: EMA smoothing factor for embedding centroid.
        use_isolation_forest: Toggle IsolationForest scoring (disable in tests
                              if sklearn is absent or for speed).
    """

    def __init__(
        self,
        cow_id: str,
        burn_in_frames: int = 300,
        refit_interval: int = 500,
        history_maxlen: int = 2000,
        embedding_ema_alpha: float = 0.05,
        use_isolation_forest: bool = True,
        n_estimators: int = 50,
        contamination: float = 0.05,
    ) -> None:
        self.cow_id = cow_id
        self.burn_in_frames = burn_in_frames
        self.refit_interval = refit_interval
        self.embedding_ema_alpha = embedding_ema_alpha
        self.use_isolation_forest = use_isolation_forest
        self.n_estimators = n_estimators
        self.contamination = contamination

        self._frames_seen: int = 0
        self._frames_since_refit: int = 0

        # Welford estimators — one per feature
        self._welford: List[WelfordEstimator] = [WelfordEstimator() for _ in range(FEATURE_DIM)]

        # Rolling history for IsolationForest (re)fitting
        self._history: deque = deque(maxlen=history_maxlen)

        # IsolationForest model (None until first fit)
        self._if_model = None

        # Embedding EMA centroid
        self._embedding_centroid: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def frames_seen(self) -> int:
        return self._frames_seen

    @property
    def is_ready(self) -> bool:
        return self._frames_seen >= self.burn_in_frames

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, obs: FrameObservation, feature_vec: np.ndarray) -> None:
        """Ingest one frame. Triggers baseline fitting when burn-in completes."""
        self._frames_seen += 1

        # Update Welford estimators
        for i, v in enumerate(feature_vec):
            self._welford[i].update(float(v))

        # Accumulate history for IsolationForest
        self._history.append(feature_vec.copy())

        # Update embedding EMA centroid
        alpha = self.embedding_ema_alpha
        if self._embedding_centroid is None:
            self._embedding_centroid = obs.embedding.copy()
        else:
            self._embedding_centroid = (
                (1 - alpha) * self._embedding_centroid + alpha * obs.embedding
            )

        # Fit at burn-in boundary
        if self._frames_seen == self.burn_in_frames:
            self._fit_baseline()
            return

        # Periodic refit after burn-in
        if self.is_ready:
            self._frames_since_refit += 1
            if self._frames_since_refit >= self.refit_interval:
                self._fit_baseline()
                self._frames_since_refit = 0

    def _fit_baseline(self) -> None:
        if not self.use_isolation_forest or len(self._history) < 10:
            return
        try:
            from sklearn.ensemble import IsolationForest
            X = np.array(self._history, dtype=np.float32)
            self._if_model = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=42,
            )
            self._if_model.fit(X)
        except ImportError:
            self.use_isolation_forest = False

    # ------------------------------------------------------------------
    # Anomaly scoring
    # ------------------------------------------------------------------

    def zscore_anomaly_score(self, feature_vec: np.ndarray) -> Dict[str, float]:
        """Return {feature_name: anomaly_score ∈ [0,1]} for each feature.

        score = 1 - exp(-z²/2)  (Gaussian tail transform)
        """
        scores = {}
        for i, (w, name) in enumerate(zip(self._welford, FEATURE_NAMES)):
            if w.n < 2:
                scores[name] = 0.0
                continue
            z = abs(float(feature_vec[i]) - w.mean) / (w.std + 1e-8)
            scores[name] = float(1.0 - math.exp(-0.5 * z ** 2))
        return scores

    def isolation_forest_score(self, feature_vec: np.ndarray) -> float:
        """Return IF anomaly score ∈ [0,1]. 0 if model not yet fitted."""
        if self._if_model is None:
            return 0.0
        # sklearn returns [-1, 1] via decision_function; higher = more normal
        raw = self._if_model.decision_function(feature_vec.reshape(1, -1))[0]
        # Map to [0, 1]: anomalous (negative raw) → high score
        score = float(1.0 / (1.0 + math.exp(5.0 * raw)))
        return max(0.0, min(1.0, score))

    def appearance_drift_score(self, embedding: np.ndarray) -> float:
        """Cosine distance from rolling EMA centroid, ∈ [0, 1]."""
        if self._embedding_centroid is None:
            return 0.0
        centroid = self._embedding_centroid
        norm_c = np.linalg.norm(centroid)
        norm_e = np.linalg.norm(embedding)
        if norm_c < 1e-8 or norm_e < 1e-8:
            return 0.0
        cosine_sim = float(np.dot(centroid, embedding) / (norm_c * norm_e))
        return float(max(0.0, 1.0 - cosine_sim))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        return {
            "cow_id": self.cow_id,
            "frames_seen": self._frames_seen,
            "is_ready": self.is_ready,
            "if_fitted": self._if_model is not None,
            "feature_means": {
                name: w.mean for name, w in zip(FEATURE_NAMES, self._welford)
            },
        }
