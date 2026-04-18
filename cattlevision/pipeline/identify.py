"""High-level cow identification pipeline.

Ties together detection, embedding, and identity lookup into a single
easy-to-use API for both single images and video streams.

Typical usage::

    from cattlevision import CowIdentifier

    identifier = CowIdentifier(
        embedder_path="runs/train/best.pt",
        database_path="gallery.npz",
    )
    results = identifier.identify_image(frame)
    for r in results:
        print(r.cow_id, r.similarity, r.bbox)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, List, Optional, Union

import cv2
import numpy as np

from cattlevision.models.detector import CowDetector, Detection
from cattlevision.models.embedder import CowEmbedder
from cattlevision.pipeline.database import IdentityDatabase
from cattlevision.utils.visualization import draw_identities


@dataclass
class IdentificationResult:
    """Result for one detected cow in a frame."""
    cow_id: str
    similarity: float
    bbox: np.ndarray           # [x1, y1, x2, y2]
    confidence: float          # detector confidence
    embedding: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    track_id: Optional[int] = None


class CowIdentifier:
    """End-to-end pipeline: detect → embed → identify.

    Args:
        detector_path: Path to YOLO .pt weights, or a model name string.
        embedder_path: Path to CowEmbedder checkpoint (.pt).  When None the
                       embedder runs with random weights (useful for testing
                       the pipeline structure without a trained model).
        database_path: Path to a saved IdentityDatabase (.npz).
        similarity_threshold: Cosine similarity cutoff for known/unknown.
        det_conf_threshold: Minimum detector confidence.
        device: Torch device string.
    """

    def __init__(
        self,
        detector_path: Optional[Union[str, Path]] = None,
        embedder_path: Optional[Union[str, Path]] = None,
        database_path: Optional[Union[str, Path]] = None,
        similarity_threshold: float = 0.6,
        det_conf_threshold: float = 0.25,
        device: Optional[str] = None,
    ):
        self.detector = CowDetector(
            model_path=detector_path,
            conf_threshold=det_conf_threshold,
            device=device,
        )
        if embedder_path and Path(embedder_path).exists():
            self.embedder = CowEmbedder.load(embedder_path, device=device or "cpu")
        else:
            self.embedder = CowEmbedder()
            if device:
                self.embedder.to(device)

        self.database = (
            IdentityDatabase.load(database_path, similarity_threshold=similarity_threshold)
            if database_path and Path(database_path).exists()
            else IdentityDatabase(similarity_threshold=similarity_threshold)
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def identify_image(self, image: np.ndarray) -> List[IdentificationResult]:
        """Detect and identify all cows in a single BGR image.

        Args:
            image: HxWx3 BGR numpy array.

        Returns:
            List of IdentificationResult, one per detected cow.
        """
        detections = self.detector.detect(image)
        if not detections:
            return []

        crops = self.detector.crop_detections(image, detections)
        # Filter empty crops (edge case: bbox entirely outside frame)
        valid = [(d, c) for d, c in zip(detections, crops) if c.size > 0]
        if not valid:
            return []
        detections, crops = zip(*valid)

        embeddings = self.embedder.embed_batch(list(crops))
        results = []
        for det, emb in zip(detections, embeddings):
            cow_id, sim = self.database.query(emb)
            results.append(IdentificationResult(
                cow_id=cow_id,
                similarity=sim,
                bbox=det.bbox,
                confidence=det.confidence,
                embedding=emb,
                track_id=det.track_id,
            ))
        return results

    def identify_video(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        show: bool = False,
    ) -> Generator[List[IdentificationResult], None, None]:
        """Process a video file frame-by-frame.

        Yields the identification results for each frame so the caller
        can process them incrementally (e.g. log sightings, update UI).

        Args:
            video_path: Path to input video file.
            output_path: If provided, writes an annotated video here.
            show: If True, display each frame in a window (press q to quit).

        Yields:
            List[IdentificationResult] for each frame.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise OSError(f"Cannot open video: {video_path}")

        writer = None
        if output_path:
            fps  = cap.get(cv2.CAP_PROP_FPS) or 25
            w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                results = self.identify_image(frame)
                if writer or show:
                    annotated = self._annotate(frame, results)
                    if writer:
                        writer.write(annotated)
                    if show:
                        cv2.imshow("CattleVision", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                yield results
        finally:
            cap.release()
            if writer:
                writer.release()
            if show:
                cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Gallery management
    # ------------------------------------------------------------------

    def enroll(self, cow_id: str, image: np.ndarray) -> None:
        """Enroll a cow by registering embeddings from a reference image."""
        detections = self.detector.detect(image)
        if not detections:
            # No cow detected — embed the full image as-is
            emb = self.embedder.embed_image(image)
            self.database.register(cow_id, emb)
            return
        crops = self.detector.crop_detections(image, detections)
        for crop in crops:
            if crop.size > 0:
                emb = self.embedder.embed_image(crop)
                self.database.register(cow_id, emb)

    def save_database(self, path: Union[str, Path]) -> None:
        self.database.save(path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _annotate(self, image: np.ndarray, results: List[IdentificationResult]) -> np.ndarray:
        return draw_identities(
            image,
            bboxes=[r.bbox for r in results],
            cow_ids=[r.cow_id for r in results],
            confidences=[r.confidence for r in results],
            similarity_scores=[r.similarity for r in results],
        )
