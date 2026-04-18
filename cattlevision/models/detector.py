"""Cow detection using YOLOv8.

Wraps Ultralytics YOLO to detect cow bounding boxes in images/video frames.
The COCO class id for 'cow' is 19; only detections of that class are returned.
Falls back to a generic Faster R-CNN when ultralytics is unavailable.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import cv2


COW_COCO_CLASS_ID = 19  # COCO class index for 'cow'


@dataclass
class Detection:
    """Single detection result."""
    bbox: np.ndarray          # [x1, y1, x2, y2] in pixel coordinates
    confidence: float
    class_id: int = COW_COCO_CLASS_ID
    track_id: Optional[int] = None


class CowDetector:
    """Detect cows in images using YOLOv8.

    Args:
        model_path: Path to a .pt weights file, or a YOLOv8 model name
                    (e.g. 'yolov8n.pt').  If None, defaults to 'yolov8n.pt'
                    (downloaded automatically on first use).
        conf_threshold: Minimum confidence score to keep a detection.
        iou_threshold: NMS IoU threshold.
        device: 'cpu', 'cuda', or 'mps'.  None = auto-select.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self._model = self._load_model(model_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray) -> List[Detection]:
        """Return all cow detections in a single BGR image.

        Args:
            image: HxWx3 BGR numpy array (OpenCV format).

        Returns:
            List of Detection objects, sorted by descending confidence.
        """
        raw = self._model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[COW_COCO_CLASS_ID],
            verbose=False,
        )
        detections: List[Detection] = []
        for result in raw:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu())
                cls = int(box.cls[0].cpu())
                tid = int(box.id[0].cpu()) if box.id is not None else None
                detections.append(Detection(bbox=xyxy, confidence=conf, class_id=cls, track_id=tid))
        return sorted(detections, key=lambda d: d.confidence, reverse=True)

    def detect_batch(self, images: List[np.ndarray]) -> List[List[Detection]]:
        """Run detection on a batch of images."""
        return [self.detect(img) for img in images]

    def crop_detections(self, image: np.ndarray, detections: List[Detection]) -> List[np.ndarray]:
        """Return cropped BGR patches for each detection bbox."""
        crops = []
        h, w = image.shape[:2]
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crops.append(image[y1:y2, x1:x2].copy())
        return crops

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self, model_path: Optional[Union[str, Path]]):
        try:
            from ultralytics import YOLO
            path = str(model_path) if model_path else "yolov8n.pt"
            model = YOLO(path)
            if self.device:
                model.to(self.device)
            return model
        except ImportError:
            return _FasterRCNNFallback(
                conf_threshold=self.conf_threshold,
                device=self.device,
            )


# ---------------------------------------------------------------------------
# Fallback detector using torchvision Faster R-CNN
# ---------------------------------------------------------------------------

class _FasterRCNNFallback:
    """Minimal Faster R-CNN wrapper used when ultralytics is not installed."""

    def __init__(self, conf_threshold: float = 0.25, device: Optional[str] = None):
        import torch
        import torchvision
        self.conf_threshold = conf_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights="DEFAULT"
        )
        self.model.eval()
        self.model.to(self.device)

    def __call__(self, image: np.ndarray, **kwargs):
        import torch
        from torchvision.transforms.functional import to_tensor
        from PIL import Image as PILImage

        pil = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        tensor = to_tensor(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
        results = []
        for output in outputs:
            results.append(_FasterRCNNResult(output, self.conf_threshold))
        return results


class _FasterRCNNResult:
    """Mimics the Ultralytics result interface for our detector."""

    def __init__(self, output: dict, conf_threshold: float):
        import torch
        mask = (
            (output["labels"] == COW_COCO_CLASS_ID) &
            (output["scores"] >= conf_threshold)
        )
        self.boxes = _FasterRCNNBoxes(
            xyxy=output["boxes"][mask].cpu(),
            scores=output["scores"][mask].cpu(),
            labels=output["labels"][mask].cpu(),
        ) if mask.any() else None


class _FasterRCNNBoxes:
    def __init__(self, xyxy, scores, labels):
        self._xyxy = xyxy
        self._scores = scores
        self._labels = labels
        self.id = None

    def __iter__(self):
        for i in range(len(self._xyxy)):
            yield _SingleBox(self._xyxy[i], self._scores[i], self._labels[i])

    def __len__(self):
        return len(self._xyxy)


class _SingleBox:
    def __init__(self, xyxy, score, label):
        self.xyxy = [xyxy]
        self.conf = [score]
        self.cls = [label]
        self.id = None
