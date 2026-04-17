"""Cow re-identification via deep metric learning.

Architecture:
  ResNet-50 backbone (pretrained on ImageNet)
  → Global Average Pool
  → FC projection head  → L2-normalised embedding (dim=256 by default)

The embedding space is trained with triplet loss so that images of the
same individual cluster together and images of different individuals are
pushed apart.  At inference, cosine similarity (equivalent to dot product
on L2-normalised vectors) is used to match a query embedding against a
gallery of known identities.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


# Standard ImageNet normalisation used by the pretrained backbone
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# Input resolution fed to the backbone
_INPUT_SIZE = 224


class CowEmbedder(nn.Module):
    """Produce a fixed-size L2-normalised embedding for a cow crop.

    Args:
        embedding_dim: Dimensionality of the output embedding vector.
        backbone: torchvision model name for the feature extractor.
        pretrained: Load ImageNet-pretrained weights for the backbone.
        dropout: Dropout probability before the projection head.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        backbone: str = "resnet50",
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # ---- backbone --------------------------------------------------
        weights = "DEFAULT" if pretrained else None
        base = getattr(models, backbone)(weights=weights)

        # Strip the final FC layer; keep everything up to global avg pool
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])
        backbone_out_dim = base.fc.in_features  # 2048 for ResNet-50

        # ---- projection head ------------------------------------------
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(backbone_out_dim, embedding_dim),
        )

        # ---- inference transform ---------------------------------------
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((_INPUT_SIZE, _INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    # ------------------------------------------------------------------
    # nn.Module interface
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, 3, H, W) float tensor in [0, 1], already normalised.

        Returns:
            (B, embedding_dim) L2-normalised embeddings.
        """
        features = self.feature_extractor(x)
        embeddings = self.head(features)
        return F.normalize(embeddings, p=2, dim=1)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def embed_image(self, image: np.ndarray) -> np.ndarray:
        """Embed a single BGR crop (H×W×3 uint8) → (embedding_dim,) numpy array."""
        device = next(self.parameters()).device
        rgb = image[:, :, ::-1].copy()  # BGR → RGB
        tensor = self.transform(rgb).unsqueeze(0).to(device)
        self.eval()
        return self.forward(tensor).squeeze(0).cpu().numpy()

    @torch.no_grad()
    def embed_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Embed a list of BGR crops → (N, embedding_dim) numpy array."""
        device = next(self.parameters()).device
        tensors = []
        for img in images:
            rgb = img[:, :, ::-1].copy()
            tensors.append(self.transform(rgb))
        batch = torch.stack(tensors).to(device)
        self.eval()
        return self.forward(batch).cpu().numpy()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        torch.save({"state_dict": self.state_dict(), "embedding_dim": self.embedding_dim}, path)

    @classmethod
    def load(cls, path: Union[str, Path], device: str = "cpu") -> "CowEmbedder":
        checkpoint = torch.load(path, map_location=device)
        model = cls(embedding_dim=checkpoint["embedding_dim"], pretrained=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model
