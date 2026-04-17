"""Dataset classes for cow re-identification.

Expected directory layout (ImageNet-style, one folder per cow):

    data_root/
      cow_001/
        frame_0042.jpg
        frame_0107.jpg
        ...
      cow_002/
        ...

Each subdirectory name is treated as the identity label.  The dataset
supports two sampling modes:

  * ``mode="triplet"`` – returns (anchor, positive, negative) tuples for
    triplet-loss training.
  * ``mode="single"`` – returns (image, label_index) pairs for standard
    cross-entropy / proxy-based losses.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class CowReIDDataset(Dataset):
    """Cow re-identification dataset.

    Args:
        root: Path to dataset root (see layout above).
        transform: Callable applied to each loaded BGR crop.
        mode: ``"triplet"`` or ``"single"``.
        min_images_per_id: Skip identities with fewer images than this
                           (required for valid triplet sampling).
    """

    def __init__(
        self,
        root: str | Path,
        transform: Optional[Callable] = None,
        mode: str = "triplet",
        min_images_per_id: int = 2,
    ):
        assert mode in ("triplet", "single"), f"Unknown mode: {mode}"
        self.root = Path(root)
        self.transform = transform
        self.mode = mode

        self._id_to_paths: Dict[str, List[Path]] = {}
        self._all_paths: List[Path] = []
        self._all_labels: List[int] = []
        self._label_to_idx: Dict[str, int] = {}

        self._build_index(min_images_per_id)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        if self.mode == "triplet":
            return sum(len(v) for v in self._id_to_paths.values())
        return len(self._all_paths)

    def __getitem__(self, idx: int):
        if self.mode == "triplet":
            return self._get_triplet(idx)
        return self._get_single(idx)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_identities(self) -> int:
        return len(self._id_to_paths)

    @property
    def identity_names(self) -> List[str]:
        return list(self._label_to_idx.keys())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_index(self, min_images: int) -> None:
        label_idx = 0
        for id_dir in sorted(self.root.iterdir()):
            if not id_dir.is_dir():
                continue
            paths = [
                p for p in sorted(id_dir.iterdir())
                if p.suffix.lower() in _IMAGE_EXTENSIONS
            ]
            if len(paths) < min_images:
                continue
            cow_id = id_dir.name
            self._id_to_paths[cow_id] = paths
            self._label_to_idx[cow_id] = label_idx
            self._all_paths.extend(paths)
            self._all_labels.extend([label_idx] * len(paths))
            label_idx += 1

    def _load(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path))
        if img is None:
            raise OSError(f"Could not read image: {path}")
        if self.transform is not None:
            # transform expects RGB uint8 numpy array
            img = img[:, :, ::-1].copy()
            img = self.transform(img)
        return img

    def _get_single(self, idx: int) -> Tuple:
        img = self._load(self._all_paths[idx])
        return img, self._all_labels[idx]

    def _get_triplet(self, idx: int) -> Tuple:
        # Map flat index → identity
        cow_ids = list(self._id_to_paths.keys())
        cumulative = 0
        anchor_id = cow_ids[0]
        for cid in cow_ids:
            n = len(self._id_to_paths[cid])
            if idx < cumulative + n:
                anchor_id = cid
                local_idx = idx - cumulative
                break
            cumulative += n
        else:
            local_idx = 0

        anchor_paths = self._id_to_paths[anchor_id]
        anchor_path  = anchor_paths[local_idx]
        positive_path = random.choice([p for p in anchor_paths if p != anchor_path])

        neg_id = random.choice([cid for cid in cow_ids if cid != anchor_id])
        negative_path = random.choice(self._id_to_paths[neg_id])

        anchor   = self._load(anchor_path)
        positive = self._load(positive_path)
        negative = self._load(negative_path)

        anchor_label = self._label_to_idx[anchor_id]
        return anchor, positive, negative, anchor_label
