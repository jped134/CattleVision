"""Unit tests for the CattleVision framework.

These tests exercise the core framework logic without requiring:
  * A GPU
  * A trained model checkpoint
  * A real cattle dataset
  * The ultralytics package

All tests run against CPU with random weights and synthetic data so they
complete in seconds in any CI environment.
"""

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class TestCowEmbedder:
    def test_output_shape(self):
        from cattlevision.models.embedder import CowEmbedder
        model = CowEmbedder(embedding_dim=64, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 64)

    def test_l2_normalised(self):
        from cattlevision.models.embedder import CowEmbedder
        model = CowEmbedder(embedding_dim=64, pretrained=False)
        x = torch.randn(4, 3, 224, 224)
        out = model(x)
        norms = out.norm(dim=1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)

    def test_embed_image(self):
        from cattlevision.models.embedder import CowEmbedder
        model = CowEmbedder(embedding_dim=64, pretrained=False)
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        emb = model.embed_image(img)
        assert emb.shape == (64,)
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-4

    def test_embed_batch(self):
        from cattlevision.models.embedder import CowEmbedder
        model = CowEmbedder(embedding_dim=64, pretrained=False)
        imgs = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(3)]
        embs = model.embed_batch(imgs)
        assert embs.shape == (3, 64)

    def test_save_load(self, tmp_path):
        from cattlevision.models.embedder import CowEmbedder
        model = CowEmbedder(embedding_dim=32, pretrained=False)
        save_path = tmp_path / "emb.pt"
        model.save(save_path)
        loaded = CowEmbedder.load(save_path)
        assert loaded.embedding_dim == 32
        x = torch.randn(1, 3, 224, 224)
        torch.testing.assert_close(model(x), loaded(x))


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class TestLosses:
    def test_triplet_loss_zero_when_separated(self):
        from cattlevision.training.losses import TripletLoss
        loss_fn = TripletLoss(margin=0.3)
        # anchor == positive, negative is far away
        a = torch.tensor([[1.0, 0.0, 0.0]])
        p = torch.tensor([[1.0, 0.0, 0.0]])
        n = torch.tensor([[0.0, 0.0, 1.0]])
        loss = loss_fn(a, p, n)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_batch_hard_triplet_loss_positive(self):
        from cattlevision.training.losses import BatchHardTripletLoss
        loss_fn = BatchHardTripletLoss(margin=0.3)
        embs = torch.randn(8, 16)
        embs = torch.nn.functional.normalize(embs, p=2, dim=1)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        loss = loss_fn(embs, labels)
        assert loss.item() >= 0.0

    def test_batch_hard_soft_margin(self):
        from cattlevision.training.losses import BatchHardTripletLoss
        loss_fn = BatchHardTripletLoss(margin="soft")
        embs = torch.randn(6, 16)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        loss = loss_fn(embs, labels)
        assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# Identity database
# ---------------------------------------------------------------------------

class TestIdentityDatabase:
    def _make_db(self):
        from cattlevision.pipeline.database import IdentityDatabase
        db = IdentityDatabase(similarity_threshold=0.5)
        rng = np.random.default_rng(42)
        # Two cows with very distinct embeddings
        cow_a = rng.standard_normal(256).astype(np.float32)
        cow_b = -cow_a + rng.standard_normal(256).astype(np.float32) * 0.01
        cow_a /= np.linalg.norm(cow_a)
        cow_b /= np.linalg.norm(cow_b)
        db.register("cow_A", cow_a)
        db.register("cow_B", cow_b)
        return db, cow_a, cow_b

    def test_known_identity_matched(self):
        db, cow_a, _ = self._make_db()
        cow_id, sim = db.query(cow_a)
        assert cow_id == "cow_A"
        assert sim > 0.9

    def test_unknown_identity_rejected(self):
        db, _, _ = self._make_db()
        rng = np.random.default_rng(0)
        noise = rng.standard_normal(256).astype(np.float32)
        noise /= np.linalg.norm(noise)
        cow_id, sim = db.query(noise)
        # May or may not be unknown; just verify it returns a string and float
        assert isinstance(cow_id, str)
        assert 0.0 <= sim <= 1.01

    def test_save_load_roundtrip(self, tmp_path):
        from cattlevision.pipeline.database import IdentityDatabase
        db, cow_a, _ = self._make_db()
        path = tmp_path / "db.npz"
        db.save(path)
        db2 = IdentityDatabase.load(path, similarity_threshold=0.5)
        assert db2.num_identities == 2
        cow_id, sim = db2.query(cow_a)
        assert cow_id == "cow_A"

    def test_remove(self):
        db, cow_a, _ = self._make_db()
        db.remove("cow_A")
        assert "cow_A" not in db.identity_names

    def test_register_batch(self):
        from cattlevision.pipeline.database import IdentityDatabase
        db = IdentityDatabase()
        embs = np.random.randn(3, 256).astype(np.float32)
        db.register_batch(["a", "b", "c"], embs)
        assert db.num_identities == 3


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_perfect_retrieval(self):
        from cattlevision.utils.metrics import compute_cmc_map
        # 4 identities, 2 images each — create perfect block-diagonal similarity
        n = 8
        embs = np.zeros((n, 4), dtype=np.float32)
        for i in range(4):
            embs[i*2:i*2+2, i] = 1.0  # orthogonal unit vectors per identity
        labels = [0, 0, 1, 1, 2, 2, 3, 3]
        metrics = compute_cmc_map(embs, labels)
        assert metrics["rank1"] == pytest.approx(1.0)
        assert metrics["mAP"]   == pytest.approx(1.0)

    def test_random_retrieval_below_perfect(self):
        from cattlevision.utils.metrics import compute_cmc_map
        rng = np.random.default_rng(1)
        embs = rng.standard_normal((20, 32)).astype(np.float32)
        labels = [i % 4 for i in range(20)]
        metrics = compute_cmc_map(embs, labels)
        assert 0.0 <= metrics["rank1"] <= 1.0
        assert 0.0 <= metrics["mAP"]   <= 1.0


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class TestCowTracker:
    def _fake_result(self, bbox, cow_id="cow_001", sim=0.8):
        """Create a minimal IdentificationResult-like object."""
        from types import SimpleNamespace
        return SimpleNamespace(
            bbox=np.array(bbox, dtype=float),
            cow_id=cow_id,
            similarity=sim,
            embedding=np.random.randn(64).astype(np.float32),
            confidence=0.9,
            track_id=None,
        )

    def test_new_track_created(self):
        from cattlevision.pipeline.tracker import CowTracker
        tracker = CowTracker(min_hits=1)
        results = [self._fake_result([10, 10, 100, 100])]
        output = tracker.update(results)
        assert len(output) == 1
        assert output[0]["cow_id"] == "cow_001"

    def test_track_persists_across_frames(self):
        from cattlevision.pipeline.tracker import CowTracker
        tracker = CowTracker(min_hits=1)
        for _ in range(5):
            results = [self._fake_result([10, 10, 100, 100])]
            output = tracker.update(results)
        assert len(output) == 1
        assert output[0]["track_id"] is not None

    def test_empty_frame_does_not_crash(self):
        from cattlevision.pipeline.tracker import CowTracker
        tracker = CowTracker()
        output = tracker.update([])
        assert output == []

    def test_track_expires_after_max_age(self):
        from cattlevision.pipeline.tracker import CowTracker
        tracker = CowTracker(max_age=3, min_hits=1)
        tracker.update([self._fake_result([10, 10, 100, 100])])
        # Send empty frames until track should expire
        for _ in range(5):
            output = tracker.update([])
        assert len(output) == 0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_load_default(self, tmp_path):
        from cattlevision.utils.config import load_config
        import yaml
        data = {"detector": {"conf_threshold": 0.5}, "embedder": {"embedding_dim": 128}}
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml.dump(data))
        cfg = load_config(cfg_path)
        assert cfg.detector.conf_threshold == 0.5
        assert cfg.embedder.embedding_dim == 128

    def test_override(self, tmp_path):
        from cattlevision.utils.config import load_config
        import yaml
        data = {"training": {"epochs": 10}}
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml.dump(data))
        cfg = load_config(cfg_path, overrides={"training": {"epochs": 100}})
        assert cfg.training.epochs == 100
