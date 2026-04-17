"""Visualization helpers for drawing detection and identity results."""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


# Palette: up to 20 visually distinct colours (BGR)
_PALETTE = [
    (230,  25,  75), ( 60, 180,  75), (255, 225,  25), (  0, 130, 200),
    (245, 130,  48), (145,  30, 180), ( 70, 240, 240), (240,  50, 230),
    (210, 245,  60), (250, 190, 212), (  0, 128, 128), (220, 190, 255),
    (170, 110,  40), (255, 250, 200), (128,   0,   0), (170, 255, 195),
    (128, 128,   0), (255, 215, 180), (  0,   0, 128), (128, 128, 128),
]


def _color_for(cow_id: str | int) -> Tuple[int, int, int]:
    return _PALETTE[hash(str(cow_id)) % len(_PALETTE)]


def draw_identities(
    image: np.ndarray,
    bboxes: List[np.ndarray],
    cow_ids: List[str],
    confidences: Optional[List[float]] = None,
    similarity_scores: Optional[List[float]] = None,
    font_scale: float = 0.6,
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes and identity labels on an image.

    Args:
        image: BGR numpy array; will be copied before drawing.
        bboxes: List of [x1, y1, x2, y2] arrays.
        cow_ids: Identity label for each box (e.g. "cow_007" or "unknown").
        confidences: Optional detection confidence per box.
        similarity_scores: Optional re-ID similarity score per box.
        font_scale: OpenCV font scale.
        thickness: Line/text thickness.

    Returns:
        Annotated BGR image copy.
    """
    out = image.copy()
    for i, (bbox, cid) in enumerate(zip(bboxes, cow_ids)):
        x1, y1, x2, y2 = map(int, bbox)
        color = _color_for(cid)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        label = str(cid)
        if confidences is not None:
            label += f"  det:{confidences[i]:.2f}"
        if similarity_scores is not None:
            label += f"  sim:{similarity_scores[i]:.2f}"

        # Background rectangle for readability
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        bg_y1 = max(y1 - th - baseline - 4, 0)
        cv2.rectangle(out, (x1, bg_y1), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            out, label, (x1 + 2, y1 - baseline),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness,
            cv2.LINE_AA,
        )
    return out


def draw_gallery_row(
    query_image: np.ndarray,
    gallery_images: List[np.ndarray],
    gallery_labels: List[str],
    is_match: List[bool],
    target_height: int = 128,
) -> np.ndarray:
    """Visualise a query alongside its top-k gallery matches.

    Matched results have a green border; non-matches have a red border.
    """
    def _resize(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        ratio = target_height / h
        return cv2.resize(img, (int(w * ratio), target_height))

    query_resized = _resize(query_image)
    cv2.putText(
        query_resized, "QUERY", (4, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA,
    )

    gallery_tiles = []
    for img, label, match in zip(gallery_images, gallery_labels, is_match):
        tile = _resize(img)
        border_color = (0, 200, 0) if match else (0, 0, 200)
        tile = cv2.copyMakeBorder(tile, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=border_color)
        cv2.putText(
            tile, label, (4, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )
        gallery_tiles.append(tile)

    # Pad all tiles to the same height before hstacking
    max_h = max(t.shape[0] for t in [query_resized] + gallery_tiles)
    def _pad_h(img):
        pad = max_h - img.shape[0]
        return cv2.copyMakeBorder(img, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)

    row = np.hstack([_pad_h(query_resized)] + [_pad_h(t) for t in gallery_tiles])
    return row
