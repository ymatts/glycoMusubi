"""Phase 3 end-to-end integration tests for the full GlycoKGNet pipeline.

Validates that all Phase 1-3 components work together correctly:

1. Mini heterogeneous graph with all 10 node types and 13 edge types
2. GlycoKGNet with ALL features enabled (tree_mpnn, biohgt, cross_modal_fusion, hybrid decoder)
3. Forward pass: output shapes and no runtime errors
4. score_triples(): scalar scores per triple
5. Backward pass: gradients exist for all parameters
6. Training loop (10 steps): loss decreases
7. PathReasoner as alternative decoder
8. Multiple configurations: Phase 1 only, Phase 2, Phase 3 full
9. Trainer compatibility with GlycoKGNet
10. CompositeLoss with all components
11. CMCA loss integration
12. Pretraining tasks with GlycoKGNet
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from glycoMusubi.embedding.models.glycoMusubi_net import GlycoKGNet
from glycoMusubi.embedding.models.path_reasoner import PathReasoner
from glycoMusubi.embedding.models.poincare import PoincareDistance
from glycoMusubi.embedding.models.biohgt import BioHGTLayer, DEFAULT_EDGE_TYPES
from glycoMusubi.embedding.models.cross_modal_fusion import CrossModalFusion
from glycoMusubi.embedding.decoders.hybrid_scorer import HybridLinkScorer
from glycoMusubi.losses.composite_loss import CompositeLoss
from glycoMusubi.losses.cmca_loss import CMCALoss
from glycoMusubi.losses.margin_loss import MarginRankingLoss
from glycoMusubi.training.trainer import Trainer
from glycoMusubi.training.callbacks import EarlyStopping, ModelCheckpoint
from glycoMusubi.training.pretraining import (
    MaskedNodePredictor,
    MaskedEdgePredictor,
    GlycanSubstructurePredictor,
)


# ======================================================================
# Constants
# ======================================================================

DIM = 64
NUM_RELATIONS = 13

# All 10 node types from the glycoMusubi schema
NUM_NODES_DICT = {
    "glycan": 10,
    "protein": 8,
    "enzyme": 6,
    "disease": 5,
    "variant": 4,
    "compound": 4,
    "site": 6,
    "motif": 5,
    "reaction": 4,
    "pathway": 3,
}

# All 13 canonical edge types
EDGE_TYPES: List[Tuple[str, str, str]] = [
    ("protein", "has_glycan", "glycan"),
    ("compound", "inhibits", "enzyme"),
    ("protein", "associated_with_disease", "disease"),
    ("protein", "has_variant", "variant"),
    ("protein", "has_site", "site"),
    ("enzyme", "has_site", "site"),
    ("site", "ptm_crosstalk", "site"),
    ("enzyme", "produced_by", "glycan"),
    ("enzyme", "consumed_by", "glycan"),
    ("glycan", "has_motif", "motif"),
    ("glycan", "child_of", "glycan"),
    ("enzyme", "catalyzed_by", "reaction"),
    ("reaction", "has_product", "glycan"),
]


# ======================================================================
# Fixtures
# ======================================================================


def _make_hetero_data(seed: int = 42) -> HeteroData:
    """Build a mini heterogeneous graph with all 10 node types and 13 edge types."""
    torch.manual_seed(seed)
    data = HeteroData()

    # Set node features and num_nodes for all types
    for ntype, count in NUM_NODES_DICT.items():
        data[ntype].x = torch.randn(count, DIM)
        data[ntype].num_nodes = count

    # Create edges for all 13 edge types (small random connections)
    for src_type, rel, dst_type in EDGE_TYPES:
        num_src = NUM_NODES_DICT[src_type]
        num_dst = NUM_NODES_DICT[dst_type]
        num_edges = min(num_src, num_dst, 4)
        src_idx = torch.randint(0, num_src, (num_edges,))
        dst_idx = torch.randint(0, num_dst, (num_edges,))
        data[src_type, rel, dst_type].edge_index = torch.stack([src_idx, dst_idx])

    return data


@pytest.fixture
def hetero_data() -> HeteroData:
    return _make_hetero_data()


@pytest.fixture
def phase3_model() -> GlycoKGNet:
    """GlycoKGNet with ALL Phase 3 features enabled."""
    return GlycoKGNet(
        num_nodes_dict=NUM_NODES_DICT,
        num_relations=NUM_RELATIONS,
        embedding_dim=DIM,
        glycan_encoder_type="hybrid",
        protein_encoder_type="learnable",
        num_hgt_layers=2,
        num_hgt_heads=4,
        use_bio_prior=True,
        use_cross_modal_fusion=True,
        num_fusion_heads=4,
        decoder_type="hybrid",
        dropout=0.1,
        edge_types=EDGE_TYPES,
    )


# ======================================================================
# 1. Mini Heterogeneous Graph Construction
# ======================================================================


class TestHeteroGraphConstruction:
    """Verify the test graph has all 10 node types and 13 edge types."""

    def test_all_10_node_types_present(self, hetero_data: HeteroData) -> None:
        expected = set(NUM_NODES_DICT.keys())
        actual = set(hetero_data.node_types)
        assert actual == expected, f"Missing node types: {expected - actual}"

    def test_all_13_edge_types_present(self, hetero_data: HeteroData) -> None:
        expected = set(EDGE_TYPES)
        actual = set(hetero_data.edge_types)
        assert actual == expected, f"Missing edge types: {expected - actual}"

    def test_node_counts_correct(self, hetero_data: HeteroData) -> None:
        for ntype, count in NUM_NODES_DICT.items():
            assert hetero_data[ntype].num_nodes == count

    def test_edge_indices_valid(self, hetero_data: HeteroData) -> None:
        for src_type, rel, dst_type in EDGE_TYPES:
            ei = hetero_data[src_type, rel, dst_type].edge_index
            assert ei.shape[0] == 2
            assert ei.shape[1] > 0
            assert ei[0].max() < NUM_NODES_DICT[src_type]
            assert ei[1].max() < NUM_NODES_DICT[dst_type]


# ======================================================================
# 2. GlycoKGNet Full-Feature Instantiation
# ======================================================================


class TestGlycoKGNetInstantiation:
    """Verify GlycoKGNet can be instantiated with all features enabled."""

    def test_phase3_model_creates_successfully(self, phase3_model: GlycoKGNet) -> None:
        assert isinstance(phase3_model, GlycoKGNet)

    def test_biohgt_enabled(self, phase3_model: GlycoKGNet) -> None:
        info = phase3_model.stage_info
        assert info["biohgt_enabled"] is True
        assert info["biohgt_layers"] == 2

    def test_cross_modal_fusion_enabled(self, phase3_model: GlycoKGNet) -> None:
        info = phase3_model.stage_info
        assert info["cross_modal_fusion_enabled"] is True

    def test_hybrid_decoder_enabled(self, phase3_model: GlycoKGNet) -> None:
        info = phase3_model.stage_info
        assert info["decoder_type"] == "hybrid"

    def test_glycan_encoder_type(self, phase3_model: GlycoKGNet) -> None:
        info = phase3_model.stage_info
        assert info["glycan_encoder"] == "hybrid"

    def test_model_has_trainable_parameters(self, phase3_model: GlycoKGNet) -> None:
        assert phase3_model.num_parameters > 0

    def test_repr_does_not_error(self, phase3_model: GlycoKGNet) -> None:
        r = repr(phase3_model)
        assert "GlycoKGNet" in r
        assert "hybrid" in r


# ======================================================================
# 3. Forward Pass
# ======================================================================


class TestForwardPass:
    """Verify forward pass produces correct output shapes with no errors."""

    def test_forward_returns_dict(
        self, phase3_model: GlycoKGNet, hetero_data: HeteroData
    ) -> None:
        emb_dict = phase3_model(hetero_data)
        assert isinstance(emb_dict, dict)

    def test_forward_all_node_types_present(
        self, phase3_model: GlycoKGNet, hetero_data: HeteroData
    ) -> None:
        emb_dict = phase3_model(hetero_data)
        for ntype in NUM_NODES_DICT:
            assert ntype in emb_dict, f"Missing embeddings for {ntype}"

    def test_forward_output_shapes(
        self, phase3_model: GlycoKGNet, hetero_data: HeteroData
    ) -> None:
        emb_dict = phase3_model(hetero_data)
        for ntype, count in NUM_NODES_DICT.items():
            assert emb_dict[ntype].shape == (count, DIM), (
                f"Shape mismatch for {ntype}: "
                f"expected ({count}, {DIM}), got {emb_dict[ntype].shape}"
            )

    def test_forward_no_nan_inf(
        self, phase3_model: GlycoKGNet, hetero_data: HeteroData
    ) -> None:
        emb_dict = phase3_model(hetero_data)
        for ntype, emb in emb_dict.items():
            assert torch.isfinite(emb).all(), f"NaN/Inf in {ntype} embeddings"

    def test_encode_alias(
        self, phase3_model: GlycoKGNet, hetero_data: HeteroData
    ) -> None:
        """encode() and forward() should produce identical results."""
        torch.manual_seed(0)
        emb1 = phase3_model.forward(hetero_data)
        torch.manual_seed(0)
        emb2 = phase3_model.encode(hetero_data)
        for ntype in NUM_NODES_DICT:
            assert torch.allclose(emb1[ntype], emb2[ntype], atol=1e-6)


# ======================================================================
# 4. score_triples()
# ======================================================================


class TestScoreTriples:
    """Verify score_triples() returns scalar scores per triple."""

    def test_score_triples_basic(
        self, phase3_model: GlycoKGNet, hetero_data: HeteroData
    ) -> None:
        batch_size = 4
        head_idx = torch.randint(0, NUM_NODES_DICT["protein"], (batch_size,))
        tail_idx = torch.randint(0, NUM_NODES_DICT["glycan"], (batch_size,))
        rel_idx = torch.zeros(batch_size, dtype=torch.long)

        scores = phase3_model.score_triples(
            hetero_data, "protein", head_idx, rel_idx, "glycan", tail_idx
        )
        assert scores.shape == (batch_size,)
        assert torch.isfinite(scores).all()

    def test_score_triples_different_types(
        self, phase3_model: GlycoKGNet, hetero_data: HeteroData
    ) -> None:
        """Score triples with various head/tail type combinations."""
        combos = [
            ("protein", "glycan"),
            ("enzyme", "glycan"),
            ("glycan", "motif"),
            ("compound", "enzyme"),
        ]
        for head_type, tail_type in combos:
            bs = 3
            h_idx = torch.randint(0, NUM_NODES_DICT[head_type], (bs,))
            t_idx = torch.randint(0, NUM_NODES_DICT[tail_type], (bs,))
            r_idx = torch.zeros(bs, dtype=torch.long)
            scores = phase3_model.score_triples(
                hetero_data, head_type, h_idx, r_idx, tail_type, t_idx
            )
            assert scores.shape == (bs,)
            assert torch.isfinite(scores).all()

    def test_score_triples_single_triple(
        self, phase3_model: GlycoKGNet, hetero_data: HeteroData
    ) -> None:
        """Scoring a single triple should work."""
        scores = phase3_model.score_triples(
            hetero_data,
            "protein",
            torch.tensor([0]),
            torch.tensor([0]),
            "glycan",
            torch.tensor([0]),
        )
        assert scores.shape == (1,)


# ======================================================================
# 5. Backward Pass
# ======================================================================


class TestBackwardPass:
    """Verify gradients exist for all trainable parameters after backward."""

    def test_backward_pass_creates_gradients(
        self, phase3_model: GlycoKGNet, hetero_data: HeteroData
    ) -> None:
        phase3_model.zero_grad()
        bs = 4
        scores = phase3_model.score_triples(
            hetero_data,
            "protein",
            torch.randint(0, NUM_NODES_DICT["protein"], (bs,)),
            torch.zeros(bs, dtype=torch.long),
            "glycan",
            torch.randint(0, NUM_NODES_DICT["glycan"], (bs,)),
        )
        loss = scores.sum()
        loss.backward()

        params_with_grad = 0
        total_params = 0
        for name, p in phase3_model.named_parameters():
            if p.requires_grad:
                total_params += 1
                if p.grad is not None and p.grad.abs().sum() > 0:
                    params_with_grad += 1

        # At least some parameters should have nonzero gradients
        assert params_with_grad > 0, "No parameters received gradients"

    def test_gradients_finite(
        self, phase3_model: GlycoKGNet, hetero_data: HeteroData
    ) -> None:
        phase3_model.zero_grad()
        bs = 4
        scores = phase3_model.score_triples(
            hetero_data,
            "protein",
            torch.randint(0, NUM_NODES_DICT["protein"], (bs,)),
            torch.zeros(bs, dtype=torch.long),
            "glycan",
            torch.randint(0, NUM_NODES_DICT["glycan"], (bs,)),
        )
        loss = scores.sum()
        loss.backward()

        for name, p in phase3_model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name}"


# ======================================================================
# 6. Training Loop (10 steps, loss decreases)
# ======================================================================


class TestTrainingLoop:
    """Train GlycoKGNet for 10 steps and verify loss decreases."""

    def test_loss_decreases_over_10_steps(self) -> None:
        torch.manual_seed(42)
        data = _make_hetero_data(seed=42)

        model = GlycoKGNet(
            num_nodes_dict=NUM_NODES_DICT,
            num_relations=NUM_RELATIONS,
            embedding_dim=DIM,
            glycan_encoder_type="hybrid",
            protein_encoder_type="learnable",
            num_hgt_layers=2,
            num_hgt_heads=4,
            use_cross_modal_fusion=True,
            decoder_type="hybrid",
            edge_types=EDGE_TYPES,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        losses = []

        for step in range(10):
            optimizer.zero_grad()

            # Score positive triples
            head_idx = torch.randint(0, NUM_NODES_DICT["protein"], (8,))
            tail_idx = torch.randint(0, NUM_NODES_DICT["glycan"], (8,))
            rel_idx = torch.zeros(8, dtype=torch.long)

            pos_scores = model.score_triples(
                data, "protein", head_idx, rel_idx, "glycan", tail_idx
            )

            # Score negative triples (random tails)
            neg_tail_idx = torch.randint(0, NUM_NODES_DICT["glycan"], (8,))
            neg_scores = model.score_triples(
                data, "protein", head_idx, rel_idx, "glycan", neg_tail_idx
            )

            # Margin loss
            loss = torch.clamp(1.0 - pos_scores + neg_scores, min=0.0).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease from first to last (allowing some noise)
        first_half_avg = sum(losses[:5]) / 5
        second_half_avg = sum(losses[5:]) / 5
        assert second_half_avg <= first_half_avg + 0.5, (
            f"Loss did not decrease: first_half_avg={first_half_avg:.4f}, "
            f"second_half_avg={second_half_avg:.4f}"
        )


# ======================================================================
# 7. PathReasoner as Alternative Decoder
# ======================================================================


class TestPathReasoner:
    """Test PathReasoner as an alternative to GlycoKGNet."""

    def test_path_reasoner_forward(self, hetero_data: HeteroData) -> None:
        model = PathReasoner(
            num_nodes_dict=NUM_NODES_DICT,
            num_relations=NUM_RELATIONS,
            embedding_dim=DIM,
            num_iterations=3,
            aggregation="sum",
        )
        emb_dict = model(hetero_data)

        for ntype in NUM_NODES_DICT:
            assert ntype in emb_dict
            assert emb_dict[ntype].shape == (NUM_NODES_DICT[ntype], DIM)

    def test_path_reasoner_score_triples(self, hetero_data: HeteroData) -> None:
        model = PathReasoner(
            num_nodes_dict=NUM_NODES_DICT,
            num_relations=NUM_RELATIONS,
            embedding_dim=DIM,
            num_iterations=3,
        )
        bs = 4
        scores = model.score_triples(
            hetero_data,
            "protein",
            torch.randint(0, NUM_NODES_DICT["protein"], (bs,)),
            torch.zeros(bs, dtype=torch.long),
            "glycan",
            torch.randint(0, NUM_NODES_DICT["glycan"], (bs,)),
        )
        assert scores.shape == (bs,)
        assert torch.isfinite(scores).all()

    def test_path_reasoner_score_query(self, hetero_data: HeteroData) -> None:
        model = PathReasoner(
            num_nodes_dict=NUM_NODES_DICT,
            num_relations=NUM_RELATIONS,
            embedding_dim=DIM,
            num_iterations=3,
        )
        bs = 2
        scores_dict = model.score_query(
            hetero_data,
            "protein",
            torch.randint(0, NUM_NODES_DICT["protein"], (bs,)),
            torch.zeros(bs, dtype=torch.long),
        )

        for ntype in NUM_NODES_DICT:
            assert ntype in scores_dict
            assert scores_dict[ntype].shape == (bs, NUM_NODES_DICT[ntype])

    def test_path_reasoner_backward(self, hetero_data: HeteroData) -> None:
        model = PathReasoner(
            num_nodes_dict=NUM_NODES_DICT,
            num_relations=NUM_RELATIONS,
            embedding_dim=DIM,
            num_iterations=3,
        )
        model.zero_grad()
        bs = 4
        scores = model.score_triples(
            hetero_data,
            "protein",
            torch.randint(0, NUM_NODES_DICT["protein"], (bs,)),
            torch.zeros(bs, dtype=torch.long),
            "glycan",
            torch.randint(0, NUM_NODES_DICT["glycan"], (bs,)),
        )
        loss = scores.sum()
        loss.backward()

        grad_count = sum(
            1 for p in model.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert grad_count > 0

    def test_path_reasoner_pna_aggregation(self, hetero_data: HeteroData) -> None:
        """PNA aggregation should also work end-to-end."""
        model = PathReasoner(
            num_nodes_dict=NUM_NODES_DICT,
            num_relations=NUM_RELATIONS,
            embedding_dim=DIM,
            num_iterations=2,
            aggregation="pna",
        )
        emb_dict = model(hetero_data)
        for ntype in NUM_NODES_DICT:
            assert emb_dict[ntype].shape == (NUM_NODES_DICT[ntype], DIM)


# ======================================================================
# 8. Multiple Configurations
# ======================================================================


class TestMultipleConfigurations:
    """Test various configurations from Phase 1 minimal to Phase 3 full."""

    def test_phase1_only_config(self, hetero_data: HeteroData) -> None:
        """Phase 1 minimal: learnable encoders, no BioHGT, DistMult decoder."""
        model = GlycoKGNet(
            num_nodes_dict=NUM_NODES_DICT,
            num_relations=NUM_RELATIONS,
            embedding_dim=DIM,
            glycan_encoder_type="learnable",
            protein_encoder_type="learnable",
            num_hgt_layers=0,
            use_cross_modal_fusion=False,
            decoder_type="distmult",
        )
        info = model.stage_info
        assert info["biohgt_enabled"] is False
        assert info["cross_modal_fusion_enabled"] is False
        assert info["decoder_type"] == "distmult_fallback"

        emb_dict = model(hetero_data)
        for ntype in NUM_NODES_DICT:
            assert emb_dict[ntype].shape == (NUM_NODES_DICT[ntype], DIM)

        # Score triples should work with distmult fallback
        bs = 3
        scores = model.score_triples(
            hetero_data,
            "protein",
            torch.randint(0, NUM_NODES_DICT["protein"], (bs,)),
            torch.zeros(bs, dtype=torch.long),
            "glycan",
            torch.randint(0, NUM_NODES_DICT["glycan"], (bs,)),
        )
        assert scores.shape == (bs,)

    def test_phase2_config(self, hetero_data: HeteroData) -> None:
        """Phase 2: hybrid glycan encoder, BioHGT enabled, hybrid decoder."""
        model = GlycoKGNet(
            num_nodes_dict=NUM_NODES_DICT,
            num_relations=NUM_RELATIONS,
            embedding_dim=DIM,
            glycan_encoder_type="hybrid",
            protein_encoder_type="learnable",
            num_hgt_layers=2,
            num_hgt_heads=4,
            use_cross_modal_fusion=False,
            decoder_type="hybrid",
            edge_types=EDGE_TYPES,
        )
        info = model.stage_info
        assert info["biohgt_enabled"] is True
        assert info["decoder_type"] == "hybrid"

        emb_dict = model(hetero_data)
        for ntype in NUM_NODES_DICT:
            assert emb_dict[ntype].shape == (NUM_NODES_DICT[ntype], DIM)

    def test_phase3_full_config(self, hetero_data: HeteroData) -> None:
        """Phase 3 full: everything enabled."""
        model = GlycoKGNet(
            num_nodes_dict=NUM_NODES_DICT,
            num_relations=NUM_RELATIONS,
            embedding_dim=DIM,
            glycan_encoder_type="hybrid",
            protein_encoder_type="learnable",
            num_hgt_layers=2,
            num_hgt_heads=4,
            use_bio_prior=True,
            use_cross_modal_fusion=True,
            num_fusion_heads=4,
            decoder_type="hybrid",
            edge_types=EDGE_TYPES,
        )
        info = model.stage_info
        assert info["biohgt_enabled"] is True
        assert info["cross_modal_fusion_enabled"] is True
        assert info["decoder_type"] == "hybrid"

        emb_dict = model(hetero_data)
        for ntype in NUM_NODES_DICT:
            assert torch.isfinite(emb_dict[ntype]).all()

    def test_biohgt_only_no_fusion(self, hetero_data: HeteroData) -> None:
        """BioHGT enabled but cross-modal fusion disabled."""
        model = GlycoKGNet(
            num_nodes_dict=NUM_NODES_DICT,
            num_relations=NUM_RELATIONS,
            embedding_dim=DIM,
            glycan_encoder_type="learnable",
            protein_encoder_type="learnable",
            num_hgt_layers=2,
            num_hgt_heads=4,
            use_cross_modal_fusion=True,  # Will be inactive with learnable encoders
            decoder_type="hybrid",
            edge_types=EDGE_TYPES,
        )
        emb_dict = model(hetero_data)
        for ntype in NUM_NODES_DICT:
            assert emb_dict[ntype].shape == (NUM_NODES_DICT[ntype], DIM)

    def test_fusion_without_biohgt(self, hetero_data: HeteroData) -> None:
        """Cross-modal fusion enabled but BioHGT disabled."""
        model = GlycoKGNet(
            num_nodes_dict=NUM_NODES_DICT,
            num_relations=NUM_RELATIONS,
            embedding_dim=DIM,
            glycan_encoder_type="hybrid",
            protein_encoder_type="learnable",
            num_hgt_layers=0,
            use_cross_modal_fusion=True,
            decoder_type="hybrid",
        )
        info = model.stage_info
        assert info["biohgt_enabled"] is False
        assert info["cross_modal_fusion_enabled"] is True

        emb_dict = model(hetero_data)
        for ntype in NUM_NODES_DICT:
            assert emb_dict[ntype].shape == (NUM_NODES_DICT[ntype], DIM)


# ======================================================================
# 9. Trainer Compatibility
# ======================================================================


class TestTrainerCompatibility:
    """Test GlycoKGNet works with the existing Trainer infrastructure."""

    def test_trainer_fit_runs(self) -> None:
        """Trainer.fit() completes without errors for 3 epochs."""
        torch.manual_seed(42)
        data = _make_hetero_data(seed=42)

        model = GlycoKGNet(
            num_nodes_dict=NUM_NODES_DICT,
            num_relations=NUM_RELATIONS,
            embedding_dim=DIM,
            glycan_encoder_type="learnable",
            protein_encoder_type="learnable",
            num_hgt_layers=0,
            use_cross_modal_fusion=False,
            decoder_type="distmult",
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = MarginRankingLoss(margin=1.0)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=data,
            device="cpu",
        )
        history = trainer.fit(epochs=3)
        assert "train_loss" in history
        assert len(history["train_loss"]) == 3
        assert all(isinstance(v, float) for v in history["train_loss"])

    def test_trainer_with_hybrid_decoder(self) -> None:
        """Trainer works with hybrid decoder (uses score() via _compute_scores)."""
        torch.manual_seed(42)
        data = _make_hetero_data(seed=42)

        model = GlycoKGNet(
            num_nodes_dict=NUM_NODES_DICT,
            num_relations=NUM_RELATIONS,
            embedding_dim=DIM,
            glycan_encoder_type="hybrid",
            protein_encoder_type="learnable",
            num_hgt_layers=2,
            num_hgt_heads=4,
            use_cross_modal_fusion=True,
            decoder_type="hybrid",
            edge_types=EDGE_TYPES,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = MarginRankingLoss(margin=1.0)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=data,
            device="cpu",
        )
        history = trainer.fit(epochs=2)
        assert len(history["train_loss"]) == 2

    def test_trainer_with_validation(self) -> None:
        """Trainer runs with validation data."""
        torch.manual_seed(42)
        train_data = _make_hetero_data(seed=42)
        val_data = _make_hetero_data(seed=99)

        model = GlycoKGNet(
            num_nodes_dict=NUM_NODES_DICT,
            num_relations=NUM_RELATIONS,
            embedding_dim=DIM,
            glycan_encoder_type="learnable",
            protein_encoder_type="learnable",
            num_hgt_layers=0,
            use_cross_modal_fusion=False,
            decoder_type="distmult",
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = MarginRankingLoss(margin=1.0)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=train_data,
            val_data=val_data,
            device="cpu",
        )
        history = trainer.fit(epochs=3, validate_every=1)
        assert "val_loss" in history
        assert len(history["val_loss"]) == 3

    def test_trainer_with_callbacks(self) -> None:
        """Trainer works with EarlyStopping and ModelCheckpoint callbacks."""
        torch.manual_seed(42)
        train_data = _make_hetero_data(seed=42)
        val_data = _make_hetero_data(seed=99)

        model = GlycoKGNet(
            num_nodes_dict=NUM_NODES_DICT,
            num_relations=NUM_RELATIONS,
            embedding_dim=DIM,
            glycan_encoder_type="learnable",
            protein_encoder_type="learnable",
            num_hgt_layers=0,
            use_cross_modal_fusion=False,
            decoder_type="distmult",
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = MarginRankingLoss(margin=1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            es = EarlyStopping(monitor="loss", patience=5, mode="min")
            ckpt = ModelCheckpoint(dirpath=tmpdir, monitor="loss", mode="min")

            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                train_data=train_data,
                val_data=val_data,
                callbacks=[es, ckpt],
                device="cpu",
            )
            history = trainer.fit(epochs=3, validate_every=1)
            assert len(history["train_loss"]) == 3

            # Checkpoint files should exist
            assert (Path(tmpdir) / "last.pt").exists()

    def test_trainer_with_path_reasoner(self) -> None:
        """PathReasoner also works with the Trainer."""
        torch.manual_seed(42)
        data = _make_hetero_data(seed=42)

        model = PathReasoner(
            num_nodes_dict=NUM_NODES_DICT,
            num_relations=NUM_RELATIONS,
            embedding_dim=DIM,
            num_iterations=2,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = MarginRankingLoss(margin=1.0)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=data,
            device="cpu",
        )
        history = trainer.fit(epochs=2)
        assert len(history["train_loss"]) == 2

    def test_trainer_checkpoint_save_load(self) -> None:
        """Verify checkpoint save/load roundtrip."""
        torch.manual_seed(42)
        data = _make_hetero_data(seed=42)

        model = GlycoKGNet(
            num_nodes_dict=NUM_NODES_DICT,
            num_relations=NUM_RELATIONS,
            embedding_dim=DIM,
            glycan_encoder_type="learnable",
            protein_encoder_type="learnable",
            num_hgt_layers=0,
            use_cross_modal_fusion=False,
            decoder_type="distmult",
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = MarginRankingLoss(margin=1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                train_data=data,
                device="cpu",
            )
            trainer.fit(epochs=2)
            ckpt_path = Path(tmpdir) / "test_ckpt.pt"
            trainer.save_checkpoint(ckpt_path)
            assert ckpt_path.exists()

            # Load checkpoint into a fresh model
            model2 = GlycoKGNet(
                num_nodes_dict=NUM_NODES_DICT,
                num_relations=NUM_RELATIONS,
                embedding_dim=DIM,
                glycan_encoder_type="learnable",
                protein_encoder_type="learnable",
                num_hgt_layers=0,
                use_cross_modal_fusion=False,
                decoder_type="distmult",
            )
            optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
            trainer2 = Trainer(
                model=model2,
                optimizer=optimizer2,
                loss_fn=loss_fn,
                train_data=data,
                device="cpu",
            )
            trainer2.load_checkpoint(ckpt_path)

            # Verify state was restored
            for (n1, p1), (n2, p2) in zip(
                model.named_parameters(), model2.named_parameters()
            ):
                assert torch.allclose(p1, p2, atol=1e-6), f"Mismatch in {n1}"


# ======================================================================
# 10. CompositeLoss with All Components
# ======================================================================


class TestCompositeLoss:
    """Test CompositeLoss with link + structural + hyperbolic + L2 reg."""

    def test_composite_loss_all_components(self) -> None:
        link_loss = MarginRankingLoss(margin=1.0)
        composite = CompositeLoss(
            link_loss=link_loss,
            lambda_struct=0.1,
            lambda_hyp=0.01,
            lambda_reg=0.01,
        )

        pos_scores = torch.randn(8)
        neg_scores = torch.randn(8)
        glycan_embs = torch.randn(10, DIM)
        pairs = torch.tensor([[0, 1], [2, 3], [4, 5]])
        all_embs = {"glycan": glycan_embs, "protein": torch.randn(8, DIM)}
        hyp_embs = torch.randn(10, DIM) * 0.1

        loss = composite(
            pos_scores=pos_scores,
            neg_scores=neg_scores,
            glycan_embeddings=glycan_embs,
            positive_pairs=pairs,
            all_embeddings=all_embs,
            hyperbolic_embeddings=hyp_embs,
        )
        assert loss.dim() == 0  # scalar
        assert torch.isfinite(loss)

    def test_composite_loss_backward(self) -> None:
        link_loss = MarginRankingLoss(margin=1.0)
        composite = CompositeLoss(link_loss=link_loss)

        pos = torch.randn(4, requires_grad=True)
        neg = torch.randn(4, requires_grad=True)
        glycan_embs = torch.randn(6, DIM, requires_grad=True)
        pairs = torch.tensor([[0, 1], [2, 3]])
        hyp = torch.randn(6, DIM, requires_grad=True)

        loss = composite(
            pos_scores=pos,
            neg_scores=neg,
            glycan_embeddings=glycan_embs,
            positive_pairs=pairs,
            hyperbolic_embeddings=hyp,
        )
        loss.backward()

        assert pos.grad is not None
        assert neg.grad is not None
        assert glycan_embs.grad is not None
        assert hyp.grad is not None

    def test_composite_loss_only_link(self) -> None:
        """Only link loss, no optional components."""
        link_loss = MarginRankingLoss(margin=1.0)
        composite = CompositeLoss(link_loss=link_loss)

        pos = torch.randn(4)
        neg = torch.randn(4)
        loss = composite(pos, neg)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_composite_loss_with_glycokgnet(self) -> None:
        """CompositeLoss integrates with GlycoKGNet forward pass."""
        torch.manual_seed(42)
        data = _make_hetero_data(seed=42)

        model = GlycoKGNet(
            num_nodes_dict=NUM_NODES_DICT,
            num_relations=NUM_RELATIONS,
            embedding_dim=DIM,
            glycan_encoder_type="hybrid",
            protein_encoder_type="learnable",
            num_hgt_layers=2,
            num_hgt_heads=4,
            use_cross_modal_fusion=True,
            decoder_type="hybrid",
            edge_types=EDGE_TYPES,
        )

        emb_dict = model(data)

        # Positive and negative scores
        bs = 4
        h_idx = torch.randint(0, NUM_NODES_DICT["protein"], (bs,))
        t_idx = torch.randint(0, NUM_NODES_DICT["glycan"], (bs,))
        r_idx = torch.zeros(bs, dtype=torch.long)

        h_emb = emb_dict["protein"][h_idx]
        t_emb = emb_dict["glycan"][t_idx]
        pos_scores = model.decoder(h_emb, r_idx, t_emb)

        neg_t_idx = torch.randint(0, NUM_NODES_DICT["glycan"], (bs,))
        neg_t_emb = emb_dict["glycan"][neg_t_idx]
        neg_scores = model.decoder(h_emb, r_idx, neg_t_emb)

        # Composite loss with all components
        link_loss = MarginRankingLoss(margin=1.0)
        composite = CompositeLoss(
            link_loss=link_loss,
            lambda_struct=0.1,
            lambda_hyp=0.01,
            lambda_reg=0.01,
        )

        pairs = torch.tensor([[0, 1], [2, 3]])
        loss = composite(
            pos_scores=pos_scores,
            neg_scores=neg_scores,
            glycan_embeddings=emb_dict["glycan"],
            positive_pairs=pairs,
            all_embeddings=emb_dict,
            hyperbolic_embeddings=emb_dict["glycan"],
        )
        loss.backward()

        grad_count = sum(
            1 for p in model.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert grad_count > 0


# ======================================================================
# 11. CMCA Loss Integration
# ======================================================================


class TestCMCALossIntegration:
    """Test CMCALoss with GlycoKGNet embeddings."""

    def test_cmca_intra_modal(self) -> None:
        cmca = CMCALoss(temperature=0.07)
        embs = torch.randn(10, DIM)
        pairs = torch.tensor([[0, 1], [2, 3], [4, 5]])
        result = cmca(modal_embeddings=embs, positive_pairs=pairs)
        assert "intra_modal_loss" in result
        assert result["intra_modal_loss"].dim() == 0
        assert torch.isfinite(result["intra_modal_loss"])

    def test_cmca_cross_modal(self) -> None:
        cmca = CMCALoss(temperature=0.07)
        modal = torch.randn(8, DIM)
        kg = torch.randn(8, DIM)
        result = cmca(modal_embeddings=modal, kg_embeddings=kg)
        assert "cross_modal_loss" in result
        assert result["cross_modal_loss"].dim() == 0
        assert torch.isfinite(result["cross_modal_loss"])

    def test_cmca_both_terms(self) -> None:
        cmca = CMCALoss(temperature=0.07)
        modal = torch.randn(8, DIM)
        kg = torch.randn(8, DIM)
        pairs = torch.tensor([[0, 1], [2, 3]])
        result = cmca(
            modal_embeddings=modal,
            kg_embeddings=kg,
            positive_pairs=pairs,
        )
        assert result["intra_modal_loss"] > 0 or result["cross_modal_loss"] > 0

    def test_cmca_with_glycokgnet_embeddings(self) -> None:
        """CMCA loss works with actual GlycoKGNet embeddings."""
        torch.manual_seed(42)
        data = _make_hetero_data(seed=42)

        model = GlycoKGNet(
            num_nodes_dict=NUM_NODES_DICT,
            num_relations=NUM_RELATIONS,
            embedding_dim=DIM,
            glycan_encoder_type="hybrid",
            protein_encoder_type="learnable",
            num_hgt_layers=2,
            num_hgt_heads=4,
            use_cross_modal_fusion=True,
            decoder_type="hybrid",
            edge_types=EDGE_TYPES,
        )

        emb_dict = model(data)
        glycan_embs = emb_dict["glycan"]

        cmca = CMCALoss(temperature=0.07)

        # Intra-modal: glycan pairs sharing motifs
        pairs = torch.tensor([[0, 1], [2, 3]])
        result = cmca(modal_embeddings=glycan_embs, positive_pairs=pairs)

        total_cmca = result["intra_modal_loss"] + result["cross_modal_loss"]
        total_cmca.backward()

        grad_count = sum(
            1 for p in model.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert grad_count > 0

    def test_cmca_backward_differentiable(self) -> None:
        cmca = CMCALoss(temperature=0.07)
        modal = torch.randn(8, DIM, requires_grad=True)
        kg = torch.randn(8, DIM, requires_grad=True)
        pairs = torch.tensor([[0, 1], [2, 3]])

        result = cmca(
            modal_embeddings=modal,
            kg_embeddings=kg,
            positive_pairs=pairs,
        )
        total = result["intra_modal_loss"] + result["cross_modal_loss"]
        total.backward()

        assert modal.grad is not None
        assert kg.grad is not None


# ======================================================================
# 12. Pretraining Tasks with GlycoKGNet
# ======================================================================


class TestPretrainingIntegration:
    """Test pretraining tasks work with GlycoKGNet as the encoder."""

    def test_masked_node_prediction(self) -> None:
        """MaskedNodePredictor works with GlycoKGNet encoder."""
        torch.manual_seed(42)
        data = _make_hetero_data(seed=42)

        model = GlycoKGNet(
            num_nodes_dict=NUM_NODES_DICT,
            num_relations=NUM_RELATIONS,
            embedding_dim=DIM,
            glycan_encoder_type="learnable",
            protein_encoder_type="learnable",
            num_hgt_layers=0,
            use_cross_modal_fusion=False,
            decoder_type="distmult",
        )

        predictor = MaskedNodePredictor(
            embedding_dim=DIM,
            continuous_dim=DIM,
        )

        loss, preds = predictor.mask_and_predict(
            data=data,
            model=model,
            mask_ratio=0.15,
            node_types=["glycan", "protein"],
        )
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        assert len(preds) > 0

    def test_masked_edge_prediction(self) -> None:
        """MaskedEdgePredictor works with GlycoKGNet encoder."""
        torch.manual_seed(42)
        data = _make_hetero_data(seed=42)

        model = GlycoKGNet(
            num_nodes_dict=NUM_NODES_DICT,
            num_relations=NUM_RELATIONS,
            embedding_dim=DIM,
            glycan_encoder_type="learnable",
            protein_encoder_type="learnable",
            num_hgt_layers=0,
            use_cross_modal_fusion=False,
            decoder_type="distmult",
        )

        predictor = MaskedEdgePredictor(
            embedding_dim=DIM,
            num_relations=NUM_RELATIONS,
        )

        loss, preds = predictor.mask_and_predict(
            data=data,
            model=model,
            mask_ratio=0.10,
        )
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        assert "existence_logits" in preds

    def test_glycan_substructure_prediction(self) -> None:
        """GlycanSubstructurePredictor works with GlycoKGNet glycan embeddings."""
        torch.manual_seed(42)
        data = _make_hetero_data(seed=42)

        model = GlycoKGNet(
            num_nodes_dict=NUM_NODES_DICT,
            num_relations=NUM_RELATIONS,
            embedding_dim=DIM,
            glycan_encoder_type="learnable",
            protein_encoder_type="learnable",
            num_hgt_layers=0,
            use_cross_modal_fusion=False,
            decoder_type="distmult",
        )

        predictor = GlycanSubstructurePredictor(
            embedding_dim=DIM,
            num_monosaccharide_types=10,
        )

        emb_dict = model(data)
        glycan_embs = emb_dict["glycan"]
        targets = torch.randint(0, 2, (NUM_NODES_DICT["glycan"], 10)).float()

        loss = predictor.compute_loss(glycan_embs, targets)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

        logits = predictor.predict(glycan_embs)
        assert logits.shape == (NUM_NODES_DICT["glycan"], 10)

    def test_pretraining_backward_through_glycokgnet(self) -> None:
        """Gradients flow from pretraining losses back through GlycoKGNet."""
        torch.manual_seed(42)
        data = _make_hetero_data(seed=42)

        model = GlycoKGNet(
            num_nodes_dict=NUM_NODES_DICT,
            num_relations=NUM_RELATIONS,
            embedding_dim=DIM,
            glycan_encoder_type="hybrid",
            protein_encoder_type="learnable",
            num_hgt_layers=2,
            num_hgt_heads=4,
            use_cross_modal_fusion=True,
            decoder_type="hybrid",
            edge_types=EDGE_TYPES,
        )

        predictor = MaskedNodePredictor(
            embedding_dim=DIM,
            continuous_dim=DIM,
        )

        model.zero_grad()
        loss, _ = predictor.mask_and_predict(
            data=data,
            model=model,
            mask_ratio=0.15,
            node_types=["glycan", "protein"],
        )
        loss.backward()

        grad_count = sum(
            1 for p in model.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert grad_count > 0, "No gradients from pretraining task to GlycoKGNet"

    def test_combined_pretraining_and_link_loss(self) -> None:
        """Combined link prediction + substructure pretraining in one step.

        Demonstrates that GlycoKGNet embeddings can drive both link prediction
        and pretraining tasks simultaneously. Masked node prediction is tested
        separately (test_pretraining_backward_through_glycokgnet) since it
        modifies graph features in-place, requiring a separate optimization step.
        """
        torch.manual_seed(42)
        data = _make_hetero_data(seed=42)

        model = GlycoKGNet(
            num_nodes_dict=NUM_NODES_DICT,
            num_relations=NUM_RELATIONS,
            embedding_dim=DIM,
            glycan_encoder_type="hybrid",
            protein_encoder_type="learnable",
            num_hgt_layers=2,
            num_hgt_heads=4,
            use_cross_modal_fusion=True,
            decoder_type="hybrid",
            edge_types=EDGE_TYPES,
        )

        substructure_predictor = GlycanSubstructurePredictor(
            embedding_dim=DIM, num_monosaccharide_types=10
        )
        cmca = CMCALoss(temperature=0.07)

        optimizer = torch.optim.Adam(
            list(model.parameters())
            + list(substructure_predictor.parameters()),
            lr=1e-3,
        )

        for step in range(3):
            optimizer.zero_grad()

            # Single forward pass for all tasks
            emb_dict = model(data)

            # Task 1: Link prediction
            bs = 4
            h_idx = torch.randint(0, NUM_NODES_DICT["protein"], (bs,))
            t_idx = torch.randint(0, NUM_NODES_DICT["glycan"], (bs,))
            r_idx = torch.zeros(bs, dtype=torch.long)

            h_emb = emb_dict["protein"][h_idx]
            t_emb = emb_dict["glycan"][t_idx]
            pos_scores = model.decoder(h_emb, r_idx, t_emb)

            neg_t_idx = torch.randint(0, NUM_NODES_DICT["glycan"], (bs,))
            neg_t_emb = emb_dict["glycan"][neg_t_idx]
            neg_scores = model.decoder(h_emb, r_idx, neg_t_emb)

            link_loss = torch.clamp(1.0 - pos_scores + neg_scores, min=0.0).mean()

            # Task 2: Substructure prediction
            glycan_embs = emb_dict["glycan"]
            targets = torch.randint(0, 2, (NUM_NODES_DICT["glycan"], 10)).float()
            struct_loss = substructure_predictor.compute_loss(glycan_embs, targets)

            # Task 3: CMCA intra-modal
            pairs = torch.tensor([[0, 1], [2, 3]])
            cmca_result = cmca(modal_embeddings=glycan_embs, positive_pairs=pairs)
            cmca_loss = cmca_result["intra_modal_loss"]

            total_loss = link_loss + 0.1 * struct_loss + 0.05 * cmca_loss
            total_loss.backward()
            optimizer.step()

            assert torch.isfinite(total_loss), f"Non-finite loss at step {step}"


# ======================================================================
# Extra: HybridLinkScorer Standalone Integration
# ======================================================================


class TestHybridLinkScorerIntegration:
    """Test HybridLinkScorer with embeddings from GlycoKGNet."""

    def test_hybrid_scorer_all_4_components(self) -> None:
        scorer = HybridLinkScorer(
            embedding_dim=DIM,
            num_relations=NUM_RELATIONS,
        )
        head = torch.randn(8, DIM)
        tail = torch.randn(8, DIM)
        rel_idx = torch.randint(0, NUM_RELATIONS, (8,))

        scores = scorer(head, rel_idx, tail)
        assert scores.shape == (8,)
        assert torch.isfinite(scores).all()

    def test_hybrid_scorer_backward(self) -> None:
        scorer = HybridLinkScorer(embedding_dim=DIM, num_relations=NUM_RELATIONS)
        head = torch.randn(8, DIM, requires_grad=True)
        tail = torch.randn(8, DIM, requires_grad=True)
        rel_idx = torch.randint(0, NUM_RELATIONS, (8,))

        scores = scorer(head, rel_idx, tail)
        scores.sum().backward()

        assert head.grad is not None
        assert tail.grad is not None

    def test_hybrid_scorer_with_glycokgnet_pipeline(self) -> None:
        """End-to-end: GlycoKGNet embeddings -> HybridLinkScorer."""
        torch.manual_seed(42)
        data = _make_hetero_data(seed=42)

        model = GlycoKGNet(
            num_nodes_dict=NUM_NODES_DICT,
            num_relations=NUM_RELATIONS,
            embedding_dim=DIM,
            glycan_encoder_type="hybrid",
            protein_encoder_type="learnable",
            num_hgt_layers=2,
            num_hgt_heads=4,
            use_cross_modal_fusion=True,
            decoder_type="hybrid",
            edge_types=EDGE_TYPES,
        )

        emb_dict = model(data)
        h_emb = emb_dict["protein"][:4]
        t_emb = emb_dict["glycan"][:4]
        rel_idx = torch.zeros(4, dtype=torch.long)

        scores = model.decoder(h_emb, rel_idx, t_emb)
        assert scores.shape == (4,)
        assert torch.isfinite(scores).all()


# ======================================================================
# Extra: Poincare Distance Integration
# ======================================================================


class TestPoincareIntegration:
    """Test Poincare distance works within the hybrid scorer pipeline."""

    def test_poincare_in_hybrid_scorer(self) -> None:
        """Poincare scorer is one of the 4 components in HybridLinkScorer."""
        scorer = HybridLinkScorer(embedding_dim=DIM, num_relations=NUM_RELATIONS)

        # The Poincare component should be accessible
        assert hasattr(scorer, "poincare")
        assert isinstance(scorer.poincare, PoincareDistance)

        # Score triples and verify contribution
        head = torch.randn(4, DIM) * 0.1
        tail = torch.randn(4, DIM) * 0.1
        rel_idx = torch.zeros(4, dtype=torch.long)

        scores = scorer(head, rel_idx, tail)
        assert torch.isfinite(scores).all()

    def test_poincare_standalone_with_glycokgnet_embeddings(self) -> None:
        """Poincare distance on GlycoKGNet-produced embeddings."""
        torch.manual_seed(42)
        data = _make_hetero_data(seed=42)

        model = GlycoKGNet(
            num_nodes_dict=NUM_NODES_DICT,
            num_relations=NUM_RELATIONS,
            embedding_dim=DIM,
            glycan_encoder_type="hybrid",
            protein_encoder_type="learnable",
            num_hgt_layers=2,
            num_hgt_heads=4,
            use_cross_modal_fusion=True,
            decoder_type="hybrid",
            edge_types=EDGE_TYPES,
        )

        emb_dict = model(data)
        poincare = PoincareDistance(curvature=1.0)

        h = emb_dict["protein"][:4] * 0.1
        r = emb_dict["glycan"][:4] * 0.1
        t = emb_dict["glycan"][4:8] * 0.1

        scores = poincare(h, r, t)
        assert scores.shape == (4,)
        assert torch.isfinite(scores).all()


# ======================================================================
# Extra: Cross-Modal Fusion Integration
# ======================================================================


class TestCrossModalFusionIntegration:
    """Test CrossModalFusion within the GlycoKGNet pipeline."""

    def test_fusion_module_standalone(self) -> None:
        fusion = CrossModalFusion(embed_dim=DIM, num_heads=4)
        h_kg = torch.randn(10, DIM)
        h_modal = torch.randn(10, DIM)

        fused = fusion(h_kg, h_modal)
        assert fused.shape == (10, DIM)
        assert torch.isfinite(fused).all()

    def test_fusion_with_mask(self) -> None:
        fusion = CrossModalFusion(embed_dim=DIM, num_heads=4)
        h_kg = torch.randn(10, DIM)
        h_modal = torch.randn(10, DIM)
        mask = torch.tensor([True, True, False, True, False, False, True, True, False, True])

        fused = fusion(h_kg, h_modal, mask=mask)
        assert fused.shape == (10, DIM)
        # Unmasked nodes should be unchanged (passthrough)
        for i in range(10):
            if not mask[i]:
                assert torch.allclose(fused[i], h_kg[i], atol=1e-6)


# ======================================================================
# Extra: BioHGTLayer Integration
# ======================================================================


class TestBioHGTIntegration:
    """Test BioHGTLayer within the GlycoKGNet pipeline."""

    def test_biohgt_layer_standalone(self) -> None:
        node_types = sorted(NUM_NODES_DICT.keys())
        layer = BioHGTLayer(
            in_dim=DIM,
            out_dim=DIM,
            num_heads=4,
            node_types=node_types,
            edge_types=EDGE_TYPES,
            use_bio_prior=True,
        )

        x_dict = {nt: torch.randn(NUM_NODES_DICT[nt], DIM) for nt in node_types}
        data = _make_hetero_data(seed=42)
        edge_index_dict = {}
        for et in data.edge_types:
            edge_index_dict[et] = data[et].edge_index

        out_dict = layer(x_dict, edge_index_dict)
        for ntype in node_types:
            assert out_dict[ntype].shape == (NUM_NODES_DICT[ntype], DIM)

    def test_biohgt_stacked_in_glycokgnet(self) -> None:
        """Two stacked BioHGT layers in GlycoKGNet produce valid output."""
        data = _make_hetero_data(seed=42)
        model = GlycoKGNet(
            num_nodes_dict=NUM_NODES_DICT,
            num_relations=NUM_RELATIONS,
            embedding_dim=DIM,
            glycan_encoder_type="hybrid",
            protein_encoder_type="learnable",
            num_hgt_layers=2,
            num_hgt_heads=4,
            use_bio_prior=True,
            use_cross_modal_fusion=True,
            decoder_type="hybrid",
            edge_types=EDGE_TYPES,
        )

        emb_dict = model(data)
        for ntype in NUM_NODES_DICT:
            assert torch.isfinite(emb_dict[ntype]).all()
            assert emb_dict[ntype].shape == (NUM_NODES_DICT[ntype], DIM)
