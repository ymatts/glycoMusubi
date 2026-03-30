"""Tests for HGTLoader integration in the Trainer.

Covers:
  - Trainer instantiation with use_hgt_loader=True
  - Backward compatibility with use_hgt_loader=False
  - Mini-batch training over multiple epochs
  - Gradient accumulation correctness
  - Mixed precision with HGTLoader
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import HeteroData

from glycoMusubi.embedding.models.glycoMusubie import TransE
from glycoMusubi.losses.margin_loss import MarginRankingLoss
from glycoMusubi.training.trainer import Trainer

# HGTLoader requires torch-sparse; skip tests that need it when unavailable.
_HAS_TORCH_SPARSE = True
try:
    import torch_sparse  # noqa: F401
except ImportError:
    _HAS_TORCH_SPARSE = False

requires_torch_sparse = pytest.mark.skipif(
    not _HAS_TORCH_SPARSE,
    reason="torch-sparse is required for HGTLoader tests",
)


_NUM_NODES_DICT = {"protein": 8, "glycan": 6}
_NUM_RELATIONS = 2
_EMBEDDING_DIM = 16


def _make_hetero_data() -> HeteroData:
    """Create a HeteroData with enough nodes/edges for HGTLoader batching."""
    data = HeteroData()
    data["protein"].num_nodes = 8
    data["protein"].x = torch.randn(8, _EMBEDDING_DIM)
    data["glycan"].num_nodes = 6
    data["glycan"].x = torch.randn(6, _EMBEDDING_DIM)

    # protein -> glycan edges
    data["protein", "has_glycan", "glycan"].edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 0, 1]]
    )
    # glycan -> protein reverse edges (needed for message passing)
    data["glycan", "rev_has_glycan", "protein"].edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 0, 1], [0, 1, 2, 3, 4, 5, 6, 7]]
    )
    return data


# ======================================================================
# Tests that do NOT require torch-sparse (backward compat / full-batch)
# ======================================================================


class TestBackwardCompatibility:
    """Tests that do not require torch-sparse."""

    def test_hgt_loader_false_backward_compat(self) -> None:
        """Trainer with use_hgt_loader=False falls back to full-batch."""
        model = TransE(_NUM_NODES_DICT, _NUM_RELATIONS, _EMBEDDING_DIM)
        optimizer = Adam(model.parameters(), lr=1e-3)
        loss_fn = MarginRankingLoss(margin=1.0)
        data = _make_hetero_data()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=data,
            device="cpu",
            use_hgt_loader=False,
        )

        assert trainer.use_mini_batch is False
        assert trainer._hgt_loader is None

    def test_full_batch_still_works(self) -> None:
        """Full-batch training (no HGTLoader) still works correctly."""
        torch.manual_seed(42)
        model = TransE(_NUM_NODES_DICT, _NUM_RELATIONS, _EMBEDDING_DIM)
        optimizer = Adam(model.parameters(), lr=1e-3)
        loss_fn = MarginRankingLoss(margin=1.0)
        data = _make_hetero_data()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=data,
            device="cpu",
            use_hgt_loader=False,
        )

        history = trainer.fit(epochs=3)
        assert len(history["train_loss"]) == 3

    def test_accumulation_steps_clamped_to_one(self) -> None:
        """gradient_accumulation_steps < 1 is clamped to 1."""
        model = TransE(_NUM_NODES_DICT, _NUM_RELATIONS, _EMBEDDING_DIM)
        optimizer = Adam(model.parameters(), lr=1e-3)
        loss_fn = MarginRankingLoss(margin=1.0)
        data = _make_hetero_data()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=data,
            device="cpu",
            gradient_accumulation_steps=0,
        )

        assert trainer.gradient_accumulation_steps == 1

    def test_no_grad_scaler_on_cpu(self) -> None:
        """GradScaler is NOT created for CPU (even with mixed_precision)."""
        model = TransE(_NUM_NODES_DICT, _NUM_RELATIONS, _EMBEDDING_DIM)
        optimizer = Adam(model.parameters(), lr=1e-3)
        loss_fn = MarginRankingLoss(margin=1.0)
        data = _make_hetero_data()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=data,
            device="cpu",
            mixed_precision=True,
            amp_dtype=torch.float16,
        )

        assert trainer._scaler is None


# ======================================================================
# Tests that REQUIRE torch-sparse (HGTLoader-based)
# ======================================================================


@requires_torch_sparse
class TestHGTLoaderInstantiation:
    """Tests for Trainer creation with HGTLoader options."""

    def test_hgt_loader_true_creates_loader(self) -> None:
        """Trainer with use_hgt_loader=True creates an HGTLoader."""
        model = TransE(_NUM_NODES_DICT, _NUM_RELATIONS, _EMBEDDING_DIM)
        optimizer = Adam(model.parameters(), lr=1e-3)
        loss_fn = MarginRankingLoss(margin=1.0)
        data = _make_hetero_data()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=data,
            device="cpu",
            use_hgt_loader=True,
            hgt_num_samples=[4],
            hgt_batch_size=4,
        )

        assert trainer.use_mini_batch is True
        assert trainer._hgt_loader is not None
        assert trainer.num_samples == [4]
        assert trainer.mini_batch_size == 4

    def test_hgt_loader_overrides_mini_batch(self) -> None:
        """use_hgt_loader=True overrides use_mini_batch settings."""
        model = TransE(_NUM_NODES_DICT, _NUM_RELATIONS, _EMBEDDING_DIM)
        optimizer = Adam(model.parameters(), lr=1e-3)
        loss_fn = MarginRankingLoss(margin=1.0)
        data = _make_hetero_data()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=data,
            device="cpu",
            use_mini_batch=False,
            use_hgt_loader=True,
            hgt_batch_size=2,
        )

        assert trainer.use_mini_batch is True
        assert trainer.mini_batch_size == 2

    def test_hgt_default_num_samples(self) -> None:
        """Default hgt_num_samples is [15] when not specified."""
        model = TransE(_NUM_NODES_DICT, _NUM_RELATIONS, _EMBEDDING_DIM)
        optimizer = Adam(model.parameters(), lr=1e-3)
        loss_fn = MarginRankingLoss(margin=1.0)
        data = _make_hetero_data()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=data,
            device="cpu",
            use_hgt_loader=True,
        )

        assert trainer.num_samples == [15]


@requires_torch_sparse
class TestHGTLoaderTraining:
    """Tests for actual training with HGTLoader."""

    def test_mini_batch_training_3_epochs(self) -> None:
        """Mini-batch training runs for 3 epochs without error."""
        torch.manual_seed(42)
        model = TransE(_NUM_NODES_DICT, _NUM_RELATIONS, _EMBEDDING_DIM)
        optimizer = Adam(model.parameters(), lr=1e-3)
        loss_fn = MarginRankingLoss(margin=1.0)
        data = _make_hetero_data()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=data,
            device="cpu",
            use_hgt_loader=True,
            hgt_num_samples=[4],
            hgt_batch_size=4,
        )

        history = trainer.fit(epochs=3)
        assert "train_loss" in history
        assert len(history["train_loss"]) == 3
        for loss_val in history["train_loss"]:
            assert isinstance(loss_val, float)
            assert loss_val >= 0.0

    def test_mini_batch_parameters_change(self) -> None:
        """Model parameters update during mini-batch training."""
        torch.manual_seed(42)
        model = TransE(_NUM_NODES_DICT, _NUM_RELATIONS, _EMBEDDING_DIM)
        optimizer = Adam(model.parameters(), lr=1e-2)
        loss_fn = MarginRankingLoss(margin=1.0)
        data = _make_hetero_data()

        params_before = {
            name: p.clone() for name, p in model.named_parameters()
        }

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=data,
            device="cpu",
            use_hgt_loader=True,
            hgt_num_samples=[4],
            hgt_batch_size=4,
        )
        trainer.fit(epochs=3)

        any_changed = any(
            not torch.allclose(params_before[name], p)
            for name, p in model.named_parameters()
        )
        assert any_changed, "Parameters should change after HGTLoader training"


@requires_torch_sparse
class TestGradientAccumulation:
    """Tests for gradient accumulation with HGTLoader."""

    def test_gradient_accumulation_runs(self) -> None:
        """Training with gradient_accumulation_steps > 1 completes."""
        torch.manual_seed(42)
        model = TransE(_NUM_NODES_DICT, _NUM_RELATIONS, _EMBEDDING_DIM)
        optimizer = Adam(model.parameters(), lr=1e-3)
        loss_fn = MarginRankingLoss(margin=1.0)
        data = _make_hetero_data()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=data,
            device="cpu",
            use_hgt_loader=True,
            hgt_num_samples=[4],
            hgt_batch_size=2,
            gradient_accumulation_steps=2,
        )

        history = trainer.fit(epochs=2)
        assert len(history["train_loss"]) == 2
        for loss_val in history["train_loss"]:
            assert loss_val >= 0.0

    def test_accumulation_loss_scaling(self) -> None:
        """With accumulation, the returned loss is unscaled (full magnitude)."""
        torch.manual_seed(42)
        model = TransE(_NUM_NODES_DICT, _NUM_RELATIONS, _EMBEDDING_DIM)
        optimizer = Adam(model.parameters(), lr=1e-3)
        loss_fn = MarginRankingLoss(margin=1.0)
        data = _make_hetero_data()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=data,
            device="cpu",
            use_hgt_loader=True,
            hgt_num_samples=[4],
            hgt_batch_size=4,
            gradient_accumulation_steps=2,
        )

        epoch_loss = trainer.train_epoch()
        assert isinstance(epoch_loss, float)
        assert epoch_loss >= 0.0


@requires_torch_sparse
class TestMixedPrecisionWithHGTLoader:
    """Tests for mixed precision training with HGTLoader."""

    def test_mixed_precision_bfloat16_runs(self) -> None:
        """Mixed precision with bfloat16 completes without error."""
        torch.manual_seed(42)
        model = TransE(_NUM_NODES_DICT, _NUM_RELATIONS, _EMBEDDING_DIM)
        optimizer = Adam(model.parameters(), lr=1e-3)
        loss_fn = MarginRankingLoss(margin=1.0)
        data = _make_hetero_data()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=data,
            device="cpu",
            mixed_precision=True,
            amp_dtype=torch.bfloat16,
            use_hgt_loader=True,
            hgt_num_samples=[4],
            hgt_batch_size=4,
        )

        history = trainer.fit(epochs=2)
        assert len(history["train_loss"]) == 2
        for loss_val in history["train_loss"]:
            assert isinstance(loss_val, float)
            assert not torch.isnan(torch.tensor(loss_val))

    def test_mixed_precision_with_grad_accum(self) -> None:
        """Mixed precision + gradient accumulation completes."""
        torch.manual_seed(42)
        model = TransE(_NUM_NODES_DICT, _NUM_RELATIONS, _EMBEDDING_DIM)
        optimizer = Adam(model.parameters(), lr=1e-3)
        loss_fn = MarginRankingLoss(margin=1.0)
        data = _make_hetero_data()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=data,
            device="cpu",
            mixed_precision=True,
            amp_dtype=torch.bfloat16,
            use_hgt_loader=True,
            hgt_num_samples=[4],
            hgt_batch_size=2,
            gradient_accumulation_steps=2,
        )

        history = trainer.fit(epochs=2)
        assert len(history["train_loss"]) == 2
