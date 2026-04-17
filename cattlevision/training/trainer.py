"""Training loop for the CowEmbedder model."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from cattlevision.data.dataset import CowReIDDataset
from cattlevision.data.transforms import build_train_transforms, build_val_transforms
from cattlevision.models.embedder import CowEmbedder
from cattlevision.training.losses import BatchHardTripletLoss
from cattlevision.utils.metrics import compute_cmc_map


class Trainer:
    """Orchestrate embedder training.

    Args:
        data_root: Root directory of the re-ID dataset.
        output_dir: Where to save checkpoints and logs.
        embedding_dim: Output embedding size.
        epochs: Total training epochs.
        batch_size: Mini-batch size (should be even multiples of identities).
        lr: Initial learning rate.
        margin: Triplet loss margin.
        val_split: Fraction of identities to hold out for validation.
        device: ``"cpu"``, ``"cuda"``, or ``"mps"``.
    """

    def __init__(
        self,
        data_root: str,
        output_dir: str = "runs/train",
        embedding_dim: int = 256,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 3e-4,
        margin: float = 0.3,
        val_split: float = 0.15,
        device: Optional[str] = None,
    ):
        self.data_root   = Path(data_root)
        self.output_dir  = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.lr          = lr
        self.margin      = margin
        self.val_split   = val_split
        self.device      = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def run(self) -> CowEmbedder:
        """Execute the full training procedure and return the best model."""
        train_loader, val_loader = self._build_loaders()
        model = CowEmbedder(embedding_dim=self.embedding_dim).to(self.device)
        criterion = BatchHardTripletLoss(margin=self.margin, normalize=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        best_map, best_path = 0.0, self.output_dir / "best.pt"

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(model, train_loader, criterion, optimizer)
            scheduler.step()
            metrics = self._validate(model, val_loader)
            mAP = metrics.get("mAP", 0.0)
            rank1 = metrics.get("rank1", 0.0)

            print(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"loss={train_loss:.4f} | mAP={mAP:.3f} | rank-1={rank1:.3f}"
            )

            if mAP > best_map:
                best_map = mAP
                model.save(best_path)

        print(f"\nBest mAP: {best_map:.3f}  (saved to {best_path})")
        return CowEmbedder.load(best_path, device=self.device)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_loaders(self):
        full = CowReIDDataset(
            self.data_root,
            transform=build_train_transforms(),
            mode="single",
        )
        n_val = max(1, int(len(full) * self.val_split))
        n_train = len(full) - n_val
        train_set, val_set = random_split(full, [n_train, n_val])

        # Swap val_set's transform to val transforms (no augmentation)
        val_set.dataset.transform = build_val_transforms()

        train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        return train_loader, val_loader

    def _train_epoch(self, model, loader, criterion, optimizer) -> float:
        model.train()
        total_loss = 0.0
        for images, labels in tqdm(loader, desc="  train", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad()
            embeddings = model(images)
            loss = criterion(embeddings, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(loader), 1)

    @torch.no_grad()
    def _validate(self, model, loader) -> dict:
        model.eval()
        all_embeddings, all_labels = [], []
        for images, labels in loader:
            embeddings = model(images.to(self.device))
            all_embeddings.append(embeddings.cpu())
            all_labels.extend(labels.tolist())
        if not all_embeddings:
            return {}
        embeddings_np = torch.cat(all_embeddings).numpy()
        return compute_cmc_map(embeddings_np, all_labels)
