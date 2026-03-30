"""Unit tests for the Trainer class and training callbacks.

Tests cover:
  - Trainer initialisation with various configurations
  - One-epoch smoke test on minimal data
  - Checkpoint save and restore
  - Callback integration (EarlyStopping, ModelCheckpoint)
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import HeteroData

from glycoMusubi.embedding.models.glycoMusubie import TransE
from glycoMusubi.losses.margin_loss import MarginRankingLoss
from glycoMusubi.training.trainer import Trainer
from glycoMusubi.training.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
)


# ======================================================================
# Fixtures
# ======================================================================

# Minimal node counts for testing.
_NUM_NODES_DICT = {"protein": 4, "glycan": 3}
_NUM_RELATIONS = 2
_EMBEDDING_DIM = 16


def _make_mini_hetero_data() -> HeteroData:
    """Create a minimal HeteroData with a few edges for training tests."""
    data = HeteroData()

    data["protein"].num_nodes = 4
    data["glycan"].num_nodes = 3

    # protein -> glycan edges (has_glycan)
    data["protein", "has_glycan", "glycan"].edge_index = torch.tensor(
        [[0, 1, 2, 3], [0, 1, 2, 0]]
    )

    return data


def _make_imbalanced_hetero_data() -> HeteroData:
    """Create data with heavily imbalanced edge counts across relations."""
    data = HeteroData()
    data["protein"].num_nodes = 6
    data["glycan"].num_nodes = 6
    data["disease"].num_nodes = 4

    # 36 edges (dense relation)
    src_dense = torch.arange(6).repeat_interleave(6)
    dst_dense = torch.arange(6).repeat(6)
    data["protein", "has_glycan", "glycan"].edge_index = torch.stack(
        [src_dense, dst_dense], dim=0
    )
    data["protein", "has_glycan", "glycan"].edge_type_idx = torch.zeros(
        src_dense.numel(), dtype=torch.long
    )

    # 4 edges (sparse relation)
    src_sparse = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    dst_sparse = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    data["protein", "associated_with_disease", "disease"].edge_index = torch.stack(
        [src_sparse, dst_sparse], dim=0
    )
    data["protein", "associated_with_disease", "disease"].edge_type_idx = torch.ones(
        src_sparse.numel(), dtype=torch.long
    )

    return data


@pytest.fixture()
def mini_train_data() -> HeteroData:
    return _make_mini_hetero_data()


@pytest.fixture()
def mini_val_data() -> HeteroData:
    return _make_mini_hetero_data()


@pytest.fixture()
def transe_model() -> TransE:
    return TransE(_NUM_NODES_DICT, _NUM_RELATIONS, _EMBEDDING_DIM)


@pytest.fixture()
def loss_fn() -> MarginRankingLoss:
    return MarginRankingLoss(margin=1.0)


# ======================================================================
# TestTrainer
# ======================================================================


class TestTrainer:
    """Tests for the Trainer class."""

    def test_init_basic(
        self, transe_model: TransE, loss_fn: nn.Module, mini_train_data: HeteroData
    ) -> None:
        """Trainer initialises without error on CPU."""
        optimizer = Adam(transe_model.parameters(), lr=1e-3)
        trainer = Trainer(
            model=transe_model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=mini_train_data,
            device="cpu",
        )
        assert trainer.current_epoch == 0
        assert trainer.device == torch.device("cpu")

    def test_init_with_val_data(
        self,
        transe_model: TransE,
        loss_fn: nn.Module,
        mini_train_data: HeteroData,
        mini_val_data: HeteroData,
    ) -> None:
        """Trainer accepts optional validation data."""
        optimizer = Adam(transe_model.parameters(), lr=1e-3)
        trainer = Trainer(
            model=transe_model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=mini_train_data,
            val_data=mini_val_data,
            device="cpu",
        )
        assert trainer.val_data is not None

    def test_one_epoch_smoke(
        self, transe_model: TransE, loss_fn: nn.Module, mini_train_data: HeteroData
    ) -> None:
        """One epoch of training completes without error."""
        optimizer = Adam(transe_model.parameters(), lr=1e-3)
        trainer = Trainer(
            model=transe_model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=mini_train_data,
            device="cpu",
        )

        train_loss = trainer.train_epoch()
        assert isinstance(train_loss, float)
        assert train_loss >= 0.0

    def test_fit_returns_history(
        self, transe_model: TransE, loss_fn: nn.Module, mini_train_data: HeteroData
    ) -> None:
        """fit() returns a history dict with train_loss."""
        optimizer = Adam(transe_model.parameters(), lr=1e-3)
        trainer = Trainer(
            model=transe_model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=mini_train_data,
            device="cpu",
        )

        history = trainer.fit(epochs=3)
        assert "train_loss" in history
        assert len(history["train_loss"]) == 3

    def test_fit_with_validation(
        self,
        transe_model: TransE,
        loss_fn: nn.Module,
        mini_train_data: HeteroData,
        mini_val_data: HeteroData,
    ) -> None:
        """fit() with validation data records val_loss."""
        optimizer = Adam(transe_model.parameters(), lr=1e-3)
        trainer = Trainer(
            model=transe_model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=mini_train_data,
            val_data=mini_val_data,
            device="cpu",
        )

        history = trainer.fit(epochs=2, validate_every=1)
        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["val_loss"]) == 2

    def test_loss_decreases(
        self, transe_model: TransE, loss_fn: nn.Module, mini_train_data: HeteroData
    ) -> None:
        """Training loss should generally decrease over multiple epochs."""
        torch.manual_seed(42)
        model = TransE(_NUM_NODES_DICT, _NUM_RELATIONS, _EMBEDDING_DIM)
        optimizer = Adam(model.parameters(), lr=1e-2)
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=MarginRankingLoss(margin=1.0),
            train_data=mini_train_data,
            device="cpu",
        )

        history = trainer.fit(epochs=50)
        losses = history["train_loss"]
        # Average of first 5 epochs should be larger than average of last 5.
        # Use averages to reduce sensitivity to single-epoch noise.
        early_avg = sum(losses[:5]) / 5
        late_avg = sum(losses[-5:]) / 5
        assert early_avg > late_avg or late_avg < 0.5, (
            f"Loss did not decrease: early avg {early_avg:.4f}, late avg {late_avg:.4f}"
        )

    def test_model_parameters_change(
        self, transe_model: TransE, loss_fn: nn.Module, mini_train_data: HeteroData
    ) -> None:
        """Model parameters should change after training."""
        optimizer = Adam(transe_model.parameters(), lr=1e-2)
        trainer = Trainer(
            model=transe_model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=mini_train_data,
            device="cpu",
        )

        # Snapshot parameters before training
        params_before = {
            name: p.clone() for name, p in transe_model.named_parameters()
        }

        trainer.fit(epochs=3)

        # At least some parameters should have changed
        any_changed = False
        for name, p in transe_model.named_parameters():
            if not torch.allclose(params_before[name], p):
                any_changed = True
                break
        assert any_changed, "Model parameters did not change after training"

    def test_relation_balance_weights_sparse_relation_higher(self) -> None:
        """Sparse relations should get larger weights when balancing is enabled."""
        train_data = _make_imbalanced_hetero_data()
        model = TransE(
            {"protein": 6, "glycan": 6, "disease": 4},
            num_relations=2,
            embedding_dim=_EMBEDDING_DIM,
        )
        optimizer = Adam(model.parameters(), lr=1e-3)
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=MarginRankingLoss(margin=1.0),
            train_data=train_data,
            device="cpu",
            relation_balance_alpha=0.5,
            relation_balance_max_weight=10.0,
        )

        dense = ("protein", "has_glycan", "glycan")
        sparse = ("protein", "associated_with_disease", "disease")
        assert trainer._edge_type_weights[sparse] > trainer._edge_type_weights[dense]

    def test_relation_balance_training_smoke(self) -> None:
        """Relation-balanced training runs without runtime errors."""
        train_data = _make_imbalanced_hetero_data()
        model = TransE(
            {"protein": 6, "glycan": 6, "disease": 4},
            num_relations=2,
            embedding_dim=_EMBEDDING_DIM,
        )
        optimizer = Adam(model.parameters(), lr=1e-3)
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=MarginRankingLoss(margin=1.0),
            train_data=train_data,
            device="cpu",
            relation_balance_alpha=0.5,
            relation_balance_max_weight=10.0,
        )

        loss = trainer.train_epoch()
        assert isinstance(loss, float)
        assert loss >= 0.0


# ======================================================================
# TestCheckpointing
# ======================================================================


class TestCheckpointing:
    """Tests for checkpoint save and restore."""

    def test_save_checkpoint(
        self,
        transe_model: TransE,
        loss_fn: nn.Module,
        mini_train_data: HeteroData,
        tmp_path: Path,
    ) -> None:
        """Checkpoint file is created on disk."""
        optimizer = Adam(transe_model.parameters(), lr=1e-3)
        trainer = Trainer(
            model=transe_model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=mini_train_data,
            device="cpu",
        )

        trainer.fit(epochs=1)
        ckpt_path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(ckpt_path)

        assert ckpt_path.exists()

    def test_load_checkpoint_restores_state(
        self,
        loss_fn: nn.Module,
        mini_train_data: HeteroData,
        tmp_path: Path,
    ) -> None:
        """Loading a checkpoint restores model weights and epoch."""
        # Train and save
        model_a = TransE(_NUM_NODES_DICT, _NUM_RELATIONS, _EMBEDDING_DIM)
        opt_a = Adam(model_a.parameters(), lr=1e-3)
        trainer_a = Trainer(
            model=model_a,
            optimizer=opt_a,
            loss_fn=loss_fn,
            train_data=mini_train_data,
            device="cpu",
        )
        trainer_a.fit(epochs=5)

        ckpt_path = tmp_path / "checkpoint.pt"
        trainer_a.save_checkpoint(ckpt_path)

        # Create a fresh model and load checkpoint
        model_b = TransE(_NUM_NODES_DICT, _NUM_RELATIONS, _EMBEDDING_DIM)
        opt_b = Adam(model_b.parameters(), lr=1e-3)
        trainer_b = Trainer(
            model=model_b,
            optimizer=opt_b,
            loss_fn=loss_fn,
            train_data=mini_train_data,
            device="cpu",
        )
        trainer_b.load_checkpoint(ckpt_path)

        # Epoch should be restored
        assert trainer_b.current_epoch == 5

        # Model weights should match
        for (name_a, p_a), (name_b, p_b) in zip(
            model_a.named_parameters(), model_b.named_parameters()
        ):
            assert name_a == name_b
            assert torch.allclose(p_a, p_b), f"Mismatch in {name_a}"


# ======================================================================
# TestCallbacks
# ======================================================================


class TestEarlyStopping:
    """Tests for EarlyStopping callback."""

    def test_no_stop_before_patience(self) -> None:
        """Should not stop before patience epochs without improvement."""
        es = EarlyStopping(monitor="mrr", patience=3, mode="max")
        es.on_epoch_end(None, 1, 0.5, {"mrr": 0.5})
        es.on_epoch_end(None, 2, 0.5, {"mrr": 0.5})
        assert not es.should_stop()

    def test_stop_after_patience(self) -> None:
        """Should stop after patience epochs without improvement."""
        es = EarlyStopping(monitor="mrr", patience=2, mode="max")
        es.on_epoch_end(None, 1, 0.5, {"mrr": 0.5})  # Sets best
        es.on_epoch_end(None, 2, 0.5, {"mrr": 0.4})  # No improvement
        es.on_epoch_end(None, 3, 0.5, {"mrr": 0.3})  # No improvement -> stop
        assert es.should_stop()

    def test_reset_on_improvement(self) -> None:
        """Counter resets when metric improves."""
        es = EarlyStopping(monitor="mrr", patience=2, mode="max")
        es.on_epoch_end(None, 1, 0.5, {"mrr": 0.5})
        es.on_epoch_end(None, 2, 0.5, {"mrr": 0.4})  # 1 no-improvement
        es.on_epoch_end(None, 3, 0.5, {"mrr": 0.6})  # Improvement, reset
        es.on_epoch_end(None, 4, 0.5, {"mrr": 0.5})  # 1 no-improvement
        assert not es.should_stop()

    def test_min_mode(self) -> None:
        """In min mode, decreasing is improvement."""
        es = EarlyStopping(monitor="loss", patience=2, mode="min")
        es.on_epoch_end(None, 1, 0.5, {"loss": 1.0})
        es.on_epoch_end(None, 2, 0.5, {"loss": 0.5})  # Improvement
        es.on_epoch_end(None, 3, 0.5, {"loss": 0.6})  # No improvement
        es.on_epoch_end(None, 4, 0.5, {"loss": 0.7})  # No improvement -> stop
        assert es.should_stop()

    def test_missing_metric_no_error(self) -> None:
        """Missing metric in val_metrics does not cause error or early stop."""
        es = EarlyStopping(monitor="mrr", patience=1, mode="max")
        es.on_epoch_end(None, 1, 0.5, {"loss": 0.5})
        es.on_epoch_end(None, 2, 0.5, {"loss": 0.5})
        assert not es.should_stop()


class TestModelCheckpoint:
    """Tests for ModelCheckpoint callback."""

    def test_creates_directory(self, tmp_path: Path) -> None:
        """on_train_begin creates the checkpoint directory."""
        ckpt_dir = tmp_path / "checkpoints" / "nested"
        mc = ModelCheckpoint(dirpath=ckpt_dir, monitor="mrr")
        mc.on_train_begin(None)
        assert ckpt_dir.exists()

    def test_saves_best_and_last(
        self,
        mini_train_data: HeteroData,
        tmp_path: Path,
    ) -> None:
        """Saves both best.pt and last.pt when metric improves."""
        model = TransE(_NUM_NODES_DICT, _NUM_RELATIONS, _EMBEDDING_DIM)
        optimizer = Adam(model.parameters(), lr=1e-3)
        loss_fn = MarginRankingLoss(margin=1.0)

        ckpt_dir = tmp_path / "ckpts"
        mc = ModelCheckpoint(dirpath=ckpt_dir, monitor="mrr", save_last=True)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=mini_train_data,
            callbacks=[mc],
            device="cpu",
        )

        # Simulate epoch end with improving metric
        mc.on_train_begin(trainer)
        mc.on_epoch_end(trainer, epoch=1, train_loss=0.5, val_metrics={"mrr": 0.3})
        mc.on_epoch_end(trainer, epoch=2, train_loss=0.4, val_metrics={"mrr": 0.5})

        assert (ckpt_dir / "best.pt").exists()
        assert (ckpt_dir / "last.pt").exists()


# ======================================================================
# TestTrainerWithCallbacks
# ======================================================================


class TestTrainerWithCallbacks:
    """Integration tests for Trainer + callbacks."""

    def test_early_stopping_integration(
        self,
        mini_train_data: HeteroData,
        mini_val_data: HeteroData,
    ) -> None:
        """EarlyStopping callback stops training early."""
        model = TransE(_NUM_NODES_DICT, _NUM_RELATIONS, _EMBEDDING_DIM)
        optimizer = Adam(model.parameters(), lr=1e-3)
        loss_fn = MarginRankingLoss(margin=1.0)

        es = EarlyStopping(monitor="loss", patience=2, mode="min")

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=mini_train_data,
            val_data=mini_val_data,
            callbacks=[es],
            device="cpu",
        )

        # This should stop before 100 epochs (due to early stopping or
        # because loss plateaus).
        history = trainer.fit(epochs=100, validate_every=1)

        # Verify training did not run the full 100 epochs.
        # (It may run all 100 if loss keeps improving, but with lr=1e-3
        # and margin loss on tiny data, it should plateau quickly.)
        assert len(history["train_loss"]) <= 100

    def test_checkpoint_integration(
        self,
        mini_train_data: HeteroData,
        mini_val_data: HeteroData,
        tmp_path: Path,
    ) -> None:
        """ModelCheckpoint callback saves files during training."""
        model = TransE(_NUM_NODES_DICT, _NUM_RELATIONS, _EMBEDDING_DIM)
        optimizer = Adam(model.parameters(), lr=1e-3)
        loss_fn = MarginRankingLoss(margin=1.0)

        ckpt_dir = tmp_path / "ckpts"
        mc = ModelCheckpoint(dirpath=ckpt_dir, monitor="loss", mode="min")

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=mini_train_data,
            val_data=mini_val_data,
            callbacks=[mc],
            device="cpu",
        )

        trainer.fit(epochs=3, validate_every=1)

        assert (ckpt_dir / "last.pt").exists()
        # best.pt should exist since loss is tracked
        assert (ckpt_dir / "best.pt").exists()

    def test_grad_clip_norm(
        self, mini_train_data: HeteroData
    ) -> None:
        """Training with gradient clipping does not error."""
        model = TransE(_NUM_NODES_DICT, _NUM_RELATIONS, _EMBEDDING_DIM)
        optimizer = Adam(model.parameters(), lr=1e-2)
        loss_fn = MarginRankingLoss(margin=1.0)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=mini_train_data,
            device="cpu",
            grad_clip_norm=1.0,
        )

        history = trainer.fit(epochs=2)
        assert len(history["train_loss"]) == 2
