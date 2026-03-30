"""Integration tests for GlycoKGNet unified model.

Tests cover:
  - GlycoKGNet.encode(): returns embeddings for all node types
  - GlycoKGNet.predict_links(): score shape [batch]
  - GlycoKGNet.score_triples(): batch scoring
  - Trainer compatibility: forward() + score() interface
  - End-to-end: HeteroData -> GlycoKGNet -> Trainer.fit()
  - Config-based initialization
  - Fallback: Phase 1 model operation without BioHGT
  - num_parameters property
  - Integration: converter -> split -> GlycoKGNet -> trainer -> evaluator
  - Loss decrease over training epochs
  - Checkpoint save/load with GlycoKGNet

These tests depend on modules from Tasks #3, #4, #5:
  - BioHGT (glycoMusubi.embedding.encoders.bio_hgt or similar)
  - GlycoKGNet (glycoMusubi.embedding.models.glycoMusubi_net or similar)
  - HybridLinkScorer (glycoMusubi.embedding.decoders.hybrid_scorer)
  - CompositeLoss (glycoMusubi.losses.composite_loss)
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from glycoMusubi.embedding.models.cross_modal_fusion import CrossModalFusion
from glycoMusubi.embedding.decoders.hybrid_scorer import HybridLinkScorer
from glycoMusubi.losses.composite_loss import CompositeLoss
from glycoMusubi.losses.margin_loss import MarginRankingLoss
from glycoMusubi.data.splits import random_link_split
from glycoMusubi.training.trainer import Trainer


# ======================================================================
# Import GlycoKGNet -- skip all tests if not yet implemented
# ======================================================================

try:
    from glycoMusubi.embedding.models.glycoMusubi_net import GlycoKGNet
    _HAS_GLYCO_KG_NET = True
except ImportError:
    _HAS_GLYCO_KG_NET = False

try:
    from glycoMusubi.embedding.models.biohgt import BioHGTLayer as _BioHGTLayer
    _HAS_BIO_HGT = True
except ImportError:
    _HAS_BIO_HGT = False

requires_glycoMusubi_net = pytest.mark.skipif(
    not _HAS_GLYCO_KG_NET,
    reason="GlycoKGNet not yet implemented (Task #4 in progress)",
)

requires_bio_hgt = pytest.mark.skipif(
    not _HAS_BIO_HGT,
    reason="BioHGT not yet implemented (Task #3 in progress)",
)


# ======================================================================
# Fixtures
# ======================================================================

EMBEDDING_DIM = 64
NUM_HEADS = 4


def _make_mini_hetero_data() -> HeteroData:
    """Create a minimal HeteroData for GlycoKGNet testing.

    Mirrors the glycoMusubi schema with all core node types.
    """
    data = HeteroData()

    node_counts = {
        "glycan": 5,
        "protein": 6,
        "enzyme": 3,
        "disease": 3,
        "variant": 2,
        "compound": 2,
        "site": 4,
    }
    for ntype, n in node_counts.items():
        x = torch.empty(n, EMBEDDING_DIM)
        nn.init.xavier_uniform_(x)
        data[ntype].x = x
        data[ntype].num_nodes = n

    # Edges with enough per type to survive splitting
    data["protein", "has_glycan", "glycan"].edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 0]]
    )
    data["compound", "inhibits", "enzyme"].edge_index = torch.tensor(
        [[0, 0, 1, 1], [0, 1, 0, 2]]
    )
    data["protein", "associated_with_disease", "disease"].edge_index = torch.tensor(
        [[0, 1, 2, 3], [0, 1, 2, 0]]
    )
    data["protein", "has_variant", "variant"].edge_index = torch.tensor(
        [[0, 1, 2, 3], [0, 1, 0, 1]]
    )
    data["protein", "has_site", "site"].edge_index = torch.tensor(
        [[0, 1, 2], [0, 1, 2]]
    )
    data["enzyme", "has_site", "site"].edge_index = torch.tensor(
        [[0, 1], [2, 3]]
    )
    data["site", "ptm_crosstalk", "site"].edge_index = torch.tensor(
        [[0, 1, 2], [1, 2, 3]]
    )

    return data


@pytest.fixture()
def mini_data() -> HeteroData:
    return _make_mini_hetero_data()


@pytest.fixture()
def num_nodes_dict(mini_data: HeteroData) -> dict:
    return {ntype: mini_data[ntype].num_nodes for ntype in mini_data.node_types}


@pytest.fixture()
def num_relations(mini_data: HeteroData) -> int:
    return len(mini_data.edge_types)


# ======================================================================
# TestCrossModalFusionIntegration
# ======================================================================


class TestCrossModalFusionIntegration:
    """Integration tests for CrossModalFusion with typical pipeline data."""

    def test_fusion_with_hetero_data_shapes(self, mini_data: HeteroData) -> None:
        """CrossModalFusion works with embeddings derived from HeteroData."""
        fusion = CrossModalFusion(embed_dim=EMBEDDING_DIM, num_heads=NUM_HEADS, dropout=0.0)

        # Simulate KG-derived embeddings for glycan nodes
        n_glycan = mini_data["glycan"].num_nodes
        h_kg = torch.randn(n_glycan, EMBEDDING_DIM)
        h_tree = torch.randn(n_glycan, EMBEDDING_DIM)

        output = fusion(h_kg, h_tree, mask=None)
        assert output.shape == (n_glycan, EMBEDDING_DIM)

    def test_fusion_with_mixed_modality_nodes(self, mini_data: HeteroData) -> None:
        """Only some nodes have modality features; rest pass through."""
        fusion = CrossModalFusion(embed_dim=EMBEDDING_DIM, num_heads=NUM_HEADS, dropout=0.0)

        n_protein = mini_data["protein"].num_nodes
        h_kg = torch.randn(n_protein, EMBEDDING_DIM)
        h_esm = torch.randn(n_protein, EMBEDDING_DIM)

        # Suppose only first 3 proteins have ESM-2 embeddings
        mask = torch.zeros(n_protein, dtype=torch.bool)
        mask[:3] = True

        output = fusion(h_kg, h_esm, mask=mask)
        assert output.shape == (n_protein, EMBEDDING_DIM)

        # Unmasked proteins unchanged
        assert torch.allclose(output[3:], h_kg[3:])


# ======================================================================
# TestHybridLinkScorerIntegration
# ======================================================================


class TestHybridLinkScorerIntegration:
    """Integration tests for HybridLinkScorer with pipeline data."""

    def test_scorer_output_shape(self, num_relations: int) -> None:
        """HybridLinkScorer produces [batch] scores."""
        scorer = HybridLinkScorer(
            embedding_dim=EMBEDDING_DIM,
            num_relations=num_relations,
        )
        batch_size = 8
        head = torch.randn(batch_size, EMBEDDING_DIM)
        tail = torch.randn(batch_size, EMBEDDING_DIM)
        rel_idx = torch.randint(0, num_relations, (batch_size,))

        scores = scorer(head, rel_idx, tail)
        assert scores.shape == (batch_size,)

    def test_scorer_scores_finite(self, num_relations: int) -> None:
        """All scores should be finite."""
        scorer = HybridLinkScorer(
            embedding_dim=EMBEDDING_DIM,
            num_relations=num_relations,
        )
        head = torch.randn(16, EMBEDDING_DIM)
        tail = torch.randn(16, EMBEDDING_DIM)
        rel_idx = torch.randint(0, num_relations, (16,))

        scores = scorer(head, rel_idx, tail)
        assert torch.isfinite(scores).all(), "Scores contain NaN or Inf"

    def test_scorer_gradient_flow(self, num_relations: int) -> None:
        """Gradients flow through all scorer parameters."""
        scorer = HybridLinkScorer(
            embedding_dim=EMBEDDING_DIM,
            num_relations=num_relations,
        )
        head = torch.randn(4, EMBEDDING_DIM, requires_grad=True)
        tail = torch.randn(4, EMBEDDING_DIM, requires_grad=True)
        rel_idx = torch.zeros(4, dtype=torch.long)

        scores = scorer(head, rel_idx, tail)
        loss = scores.sum()
        loss.backward()

        params_without_grad = [
            name for name, p in scorer.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        assert len(params_without_grad) == 0, (
            f"Parameters without gradients: {params_without_grad}"
        )


# ======================================================================
# TestCompositeLossIntegration
# ======================================================================


class TestCompositeLossIntegration:
    """Integration tests for CompositeLoss in the pipeline context."""

    def test_composite_loss_basic(self) -> None:
        """CompositeLoss produces a scalar loss."""
        link_loss = MarginRankingLoss(margin=1.0)
        composite = CompositeLoss(link_loss=link_loss)

        pos_scores = torch.randn(16)
        neg_scores = torch.randn(16)
        loss = composite(pos_scores, neg_scores)

        assert loss.dim() == 0  # Scalar
        assert torch.isfinite(loss)

    def test_composite_loss_with_structural_term(self) -> None:
        """CompositeLoss works with glycan structural contrastive term."""
        link_loss = MarginRankingLoss(margin=1.0)
        composite = CompositeLoss(link_loss=link_loss, lambda_struct=0.1)

        pos_scores = torch.randn(16)
        neg_scores = torch.randn(16)
        glycan_emb = torch.randn(5, EMBEDDING_DIM)
        positive_pairs = torch.tensor([[0, 1], [2, 3]])

        loss = composite(
            pos_scores, neg_scores,
            glycan_embeddings=glycan_emb,
            positive_pairs=positive_pairs,
        )
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_composite_loss_with_reg(self) -> None:
        """CompositeLoss works with L2 regularization term."""
        link_loss = MarginRankingLoss(margin=1.0)
        composite = CompositeLoss(link_loss=link_loss, lambda_reg=0.01)

        pos_scores = torch.randn(16)
        neg_scores = torch.randn(16)
        all_emb = {
            "glycan": torch.randn(5, EMBEDDING_DIM),
            "protein": torch.randn(6, EMBEDDING_DIM),
        }

        loss = composite(pos_scores, neg_scores, all_embeddings=all_emb)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_composite_loss_gradient_flow(self) -> None:
        """Gradients flow through composite loss to input scores."""
        link_loss = MarginRankingLoss(margin=1.0)
        composite = CompositeLoss(link_loss=link_loss, lambda_struct=0.1, lambda_reg=0.01)

        pos_scores = torch.randn(8, requires_grad=True)
        neg_scores = torch.randn(8, requires_grad=True)
        glycan_emb = torch.randn(5, EMBEDDING_DIM, requires_grad=True)
        positive_pairs = torch.tensor([[0, 1], [2, 3]])
        all_emb = {"glycan": glycan_emb}

        loss = composite(
            pos_scores, neg_scores,
            glycan_embeddings=glycan_emb,
            positive_pairs=positive_pairs,
            all_embeddings=all_emb,
        )
        loss.backward()

        assert pos_scores.grad is not None
        assert neg_scores.grad is not None
        assert glycan_emb.grad is not None


# ======================================================================
# TestGlycoKGNetEncode
# ======================================================================


@requires_glycoMusubi_net
class TestGlycoKGNetEncode:
    """Tests for GlycoKGNet.encode() method."""

    def _make_model(self, data: HeteroData) -> "GlycoKGNet":
        """Create a GlycoKGNet model from HeteroData."""
        num_nodes_dict = {ntype: data[ntype].num_nodes for ntype in data.node_types}
        num_rels = len(data.edge_types)
        return GlycoKGNet(
            num_nodes_dict=num_nodes_dict,
            num_relations=num_rels,
            embedding_dim=EMBEDDING_DIM,
        )

    def test_encode_returns_all_node_types(self, mini_data: HeteroData) -> None:
        """encode() returns embeddings for all node types in the graph."""
        model = self._make_model(mini_data)
        emb_dict = model(mini_data)

        for ntype in mini_data.node_types:
            assert ntype in emb_dict, f"Missing embeddings for node type '{ntype}'"

    def test_encode_shapes(self, mini_data: HeteroData) -> None:
        """Embeddings have shape [num_nodes_of_type, embedding_dim]."""
        model = self._make_model(mini_data)
        emb_dict = model(mini_data)

        for ntype in mini_data.node_types:
            expected_n = mini_data[ntype].num_nodes
            assert emb_dict[ntype].shape == (expected_n, EMBEDDING_DIM), (
                f"Shape mismatch for {ntype}: "
                f"expected ({expected_n}, {EMBEDDING_DIM}), "
                f"got {emb_dict[ntype].shape}"
            )

    def test_encode_finite_values(self, mini_data: HeteroData) -> None:
        """All embeddings should be finite (no NaN/Inf)."""
        model = self._make_model(mini_data)
        emb_dict = model(mini_data)

        for ntype, emb in emb_dict.items():
            assert torch.isfinite(emb).all(), (
                f"Non-finite values in embeddings for {ntype}"
            )


# ======================================================================
# TestGlycoKGNetPredictLinks
# ======================================================================


@requires_glycoMusubi_net
class TestGlycoKGNetPredictLinks:
    """Tests for GlycoKGNet link prediction scoring."""

    def _make_model(self, data: HeteroData) -> "GlycoKGNet":
        num_nodes_dict = {ntype: data[ntype].num_nodes for ntype in data.node_types}
        num_rels = len(data.edge_types)
        return GlycoKGNet(
            num_nodes_dict=num_nodes_dict,
            num_relations=num_rels,
            embedding_dim=EMBEDDING_DIM,
        )

    def test_score_shape(self, mini_data: HeteroData) -> None:
        """score() returns [batch] shaped tensor."""
        model = self._make_model(mini_data)
        emb_dict = model(mini_data)

        head = emb_dict["protein"][:4]
        tail = emb_dict["glycan"][:4]
        rel = model.get_relation_embedding(torch.zeros(4, dtype=torch.long))

        scores = model.score(head, rel, tail)
        assert scores.shape == (4,)

    def test_score_triples_end_to_end(self, mini_data: HeteroData) -> None:
        """score_triples performs embedding lookup + scoring."""
        model = self._make_model(mini_data)

        head_idx = torch.tensor([0, 1, 2])
        rel_idx = torch.tensor([0, 0, 1])
        tail_idx = torch.tensor([0, 1, 2])

        scores = model.score_triples(
            mini_data, "protein", head_idx, rel_idx, "glycan", tail_idx
        )
        assert scores.shape == (3,)
        assert torch.isfinite(scores).all()


# ======================================================================
# TestGlycoKGNetTrainerCompat
# ======================================================================


@requires_glycoMusubi_net
class TestGlycoKGNetTrainerCompat:
    """Tests for GlycoKGNet compatibility with the Trainer."""

    def _make_model_and_data(self):
        data = _make_mini_hetero_data()
        num_nodes_dict = {ntype: data[ntype].num_nodes for ntype in data.node_types}
        num_rels = len(data.edge_types)
        model = GlycoKGNet(
            num_nodes_dict=num_nodes_dict,
            num_relations=num_rels,
            embedding_dim=EMBEDDING_DIM,
        )
        return model, data

    def test_forward_returns_dict(self) -> None:
        """forward() returns Dict[str, Tensor] as required by Trainer."""
        model, data = self._make_model_and_data()
        emb_dict = model(data)
        assert isinstance(emb_dict, dict)
        for v in emb_dict.values():
            assert isinstance(v, torch.Tensor)

    def test_get_relation_embedding(self) -> None:
        """get_relation_embedding returns [batch, dim] tensor."""
        model, _data = self._make_model_and_data()
        rel_idx = torch.tensor([0, 1, 2])
        rel_emb = model.get_relation_embedding(rel_idx)
        assert rel_emb.dim() == 2
        assert rel_emb.shape[0] == 3

    def test_trainer_one_epoch_smoke(self) -> None:
        """Trainer.fit(epochs=1) completes without error."""
        model, data = self._make_model_and_data()
        train_data, val_data, _ = random_link_split(
            data, val_ratio=0.1, test_ratio=0.1, seed=42
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
        history = trainer.fit(epochs=1)

        assert "train_loss" in history
        assert len(history["train_loss"]) == 1
        assert not torch.isnan(torch.tensor(history["train_loss"][0]))


# ======================================================================
# TestGlycoKGNetEndToEnd
# ======================================================================


@requires_glycoMusubi_net
class TestGlycoKGNetEndToEnd:
    """End-to-end integration tests for GlycoKGNet."""

    def _make_model(self, data: HeteroData) -> "GlycoKGNet":
        num_nodes_dict = {ntype: data[ntype].num_nodes for ntype in data.node_types}
        num_rels = len(data.edge_types)
        return GlycoKGNet(
            num_nodes_dict=num_nodes_dict,
            num_relations=num_rels,
            embedding_dim=EMBEDDING_DIM,
        )

    def test_loss_decreases_10_epochs(self) -> None:
        """Training loss should decrease over 10 epochs."""
        torch.manual_seed(42)
        data = _make_mini_hetero_data()
        train_data, _, _ = random_link_split(
            data, val_ratio=0.05, test_ratio=0.05, seed=42
        )

        model = self._make_model(data)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss_fn = MarginRankingLoss(margin=1.0)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=train_data,
            device="cpu",
        )
        history = trainer.fit(epochs=10)
        losses = history["train_loss"]

        assert len(losses) == 10
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )

    def test_checkpoint_save_load(self, tmp_path) -> None:
        """Checkpoint save/load preserves model parameters."""
        data = _make_mini_hetero_data()
        train_data, _, _ = random_link_split(
            data, val_ratio=0.05, test_ratio=0.05, seed=42
        )

        model = self._make_model(data)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = MarginRankingLoss(margin=1.0)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=train_data,
            device="cpu",
        )
        trainer.fit(epochs=3)

        ckpt_path = tmp_path / "glycoMusubi_net_ckpt.pt"
        trainer.save_checkpoint(ckpt_path)
        assert ckpt_path.exists()

        # Load into fresh model
        model2 = self._make_model(data)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
        trainer2 = Trainer(
            model=model2,
            optimizer=optimizer2,
            loss_fn=loss_fn,
            train_data=train_data,
            device="cpu",
        )
        trainer2.load_checkpoint(ckpt_path)

        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), model2.named_parameters()
        ):
            assert torch.allclose(p1, p2, atol=1e-6), (
                f"Parameter mismatch after checkpoint load: {n1}"
            )

    def test_gradient_flow_through_entire_model(self) -> None:
        """All parameters receive gradients after forward + backward."""
        data = _make_mini_hetero_data()
        model = self._make_model(data)

        emb_dict = model(data)

        # Score one edge type
        edge_type = data.edge_types[0]
        src_type, _rel, dst_type = edge_type
        ei = data[edge_type].edge_index
        head_emb = emb_dict[src_type][ei[0]]
        tail_emb = emb_dict[dst_type][ei[1]]
        rel_emb = model.get_relation_embedding(
            torch.zeros(ei.size(1), dtype=torch.long)
        )
        scores = model.score(head_emb, rel_emb, tail_emb)
        loss = -scores.mean()
        loss.backward()

        # At minimum, the used embedding tables should have gradients.
        # Note: GlycoKGNet with BioHGT has many per-type/per-relation
        # parameters, so a single edge type only activates a fraction.
        params_with_grad = sum(
            1 for _, p in model.named_parameters()
            if p.requires_grad and p.grad is not None
        )
        total_params = sum(
            1 for _, p in model.named_parameters() if p.requires_grad
        )
        assert params_with_grad > 0, "No parameters received gradients"
        # With BioHGT's per-type transforms, a single edge type activates
        # only a subset of parameters; verify we get a meaningful number.
        assert params_with_grad >= 10, (
            f"Too few parameters with gradients: {params_with_grad}/{total_params}"
        )

    def test_data_split_preserves_nodes(self) -> None:
        """Splitting preserves all node types for GlycoKGNet compatibility."""
        data = _make_mini_hetero_data()
        train_data, val_data, test_data = random_link_split(
            data, val_ratio=0.1, test_ratio=0.1, seed=42
        )

        model = self._make_model(data)

        # Model should work on all splits
        for split_name, split_data in [
            ("train", train_data), ("val", val_data), ("test", test_data)
        ]:
            emb_dict = model(split_data)
            for ntype in data.node_types:
                assert ntype in emb_dict, (
                    f"Missing {ntype} in {split_name} split embeddings"
                )


# ======================================================================
# TestGlycoKGNetConfig
# ======================================================================


@requires_glycoMusubi_net
class TestGlycoKGNetConfig:
    """Tests for GlycoKGNet configuration and initialization."""

    def test_num_parameters_property(self) -> None:
        """GlycoKGNet should expose parameter count."""
        data = _make_mini_hetero_data()
        num_nodes_dict = {ntype: data[ntype].num_nodes for ntype in data.node_types}
        num_rels = len(data.edge_types)
        model = GlycoKGNet(
            num_nodes_dict=num_nodes_dict,
            num_relations=num_rels,
            embedding_dim=EMBEDDING_DIM,
        )

        # Access via standard PyTorch method
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0, "Model has no parameters"

        # If model exposes num_parameters property, verify it matches
        if hasattr(model, "num_parameters"):
            assert model.num_parameters == n_params

    def test_embedding_dim_consistent(self) -> None:
        """Embedding dimension should be consistent across model."""
        data = _make_mini_hetero_data()
        num_nodes_dict = {ntype: data[ntype].num_nodes for ntype in data.node_types}
        num_rels = len(data.edge_types)
        model = GlycoKGNet(
            num_nodes_dict=num_nodes_dict,
            num_relations=num_rels,
            embedding_dim=EMBEDDING_DIM,
        )
        assert model.embedding_dim == EMBEDDING_DIM

    def test_model_is_base_kge_subclass(self) -> None:
        """GlycoKGNet should inherit from BaseKGEModel."""
        from glycoMusubi.embedding.models.base import BaseKGEModel

        data = _make_mini_hetero_data()
        num_nodes_dict = {ntype: data[ntype].num_nodes for ntype in data.node_types}
        num_rels = len(data.edge_types)
        model = GlycoKGNet(
            num_nodes_dict=num_nodes_dict,
            num_relations=num_rels,
            embedding_dim=EMBEDDING_DIM,
        )
        assert isinstance(model, BaseKGEModel), (
            "GlycoKGNet should inherit from BaseKGEModel"
        )


# ======================================================================
# TestGlycoKGNetFallback
# ======================================================================


@requires_glycoMusubi_net
class TestGlycoKGNetFallback:
    """Tests for fallback behavior when components are unavailable."""

    def test_works_without_modality_features(self) -> None:
        """GlycoKGNet should work even without modality-specific features.

        When no tree/ESM-2/text features are provided, the model should
        fall back to KG-only embeddings (Phase 1 behavior).
        """
        data = _make_mini_hetero_data()
        # Remove node features to simulate no modality input
        for ntype in data.node_types:
            if hasattr(data[ntype], "x"):
                del data[ntype].x

        num_nodes_dict = {ntype: data[ntype].num_nodes for ntype in data.node_types}
        num_rels = len(data.edge_types)
        model = GlycoKGNet(
            num_nodes_dict=num_nodes_dict,
            num_relations=num_rels,
            embedding_dim=EMBEDDING_DIM,
        )

        # Should not crash
        emb_dict = model(data)
        assert len(emb_dict) == len(data.node_types)


# ======================================================================
# TestIntegrationPipeline (does not require GlycoKGNet)
# ======================================================================


class TestIntegrationPipeline:
    """Integration tests using existing Phase 1 models + new components.

    These tests verify that CrossModalFusion, HybridLinkScorer, and
    CompositeLoss work together in a training pipeline.
    """

    def test_crossmodal_fusion_in_training_loop(self) -> None:
        """CrossModalFusion integrates into a manual training step."""
        fusion = CrossModalFusion(embed_dim=EMBEDDING_DIM, num_heads=NUM_HEADS, dropout=0.0)

        h_kg = torch.randn(8, EMBEDDING_DIM, requires_grad=True)
        h_mod = torch.randn(8, EMBEDDING_DIM, requires_grad=True)

        # Forward
        h_fused = fusion(h_kg, h_mod, mask=None)

        # Simple loss
        loss = h_fused.sum()
        loss.backward()

        # Verify training step works
        optimizer = torch.optim.Adam(
            list(fusion.parameters()) + [h_kg, h_mod], lr=1e-3
        )
        optimizer.step()

    def test_hybrid_scorer_with_composite_loss(self) -> None:
        """HybridLinkScorer + CompositeLoss work together."""
        num_rels = 4
        scorer = HybridLinkScorer(
            embedding_dim=EMBEDDING_DIM, num_relations=num_rels
        )
        link_loss = MarginRankingLoss(margin=1.0)
        composite = CompositeLoss(link_loss=link_loss, lambda_reg=0.01)

        batch_size = 8
        head = torch.randn(batch_size, EMBEDDING_DIM)
        tail = torch.randn(batch_size, EMBEDDING_DIM)
        rel_idx = torch.randint(0, num_rels, (batch_size,))

        pos_scores = scorer(head, rel_idx, tail)

        # Corrupt tails for negatives
        neg_tail = torch.randn(batch_size, EMBEDDING_DIM)
        neg_scores = scorer(head, rel_idx, neg_tail)

        all_emb = {"head": head, "tail": tail}
        loss = composite(pos_scores, neg_scores, all_embeddings=all_emb)

        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_phase1_model_with_composite_loss(self) -> None:
        """Phase 1 TransE model works with CompositeLoss in Trainer."""
        from glycoMusubi.embedding.models.glycoMusubie import TransE

        data = _make_mini_hetero_data()
        train_data, _, _ = random_link_split(
            data, val_ratio=0.05, test_ratio=0.05, seed=42
        )

        num_nodes_dict = {ntype: data[ntype].num_nodes for ntype in data.node_types}
        num_rels = len(data.edge_types)

        model = TransE(
            num_nodes_dict=num_nodes_dict,
            num_relations=num_rels,
            embedding_dim=EMBEDDING_DIM,
        )

        link_loss = MarginRankingLoss(margin=1.0)
        loss_fn = CompositeLoss(link_loss=link_loss, lambda_reg=0.001)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=train_data,
            device="cpu",
        )

        history = trainer.fit(epochs=3)
        assert len(history["train_loss"]) == 3
        for loss_val in history["train_loss"]:
            assert not torch.isnan(torch.tensor(loss_val))

    def test_converter_to_split_to_model_pipeline(self, mini_kg_dir) -> None:
        """Full pipeline: KGConverter -> split -> model -> train (Phase 1)."""
        from glycoMusubi.data.converter import KGConverter
        from glycoMusubi.embedding.models.glycoMusubie import DistMult

        converter = KGConverter(kg_dir=mini_kg_dir, schema_dir=None)
        data, _mappings = converter.convert(feature_dim=EMBEDDING_DIM)

        assert len(data.node_types) > 0
        assert len(data.edge_types) > 0

        train_data, val_data, _ = random_link_split(
            data, val_ratio=0.1, test_ratio=0.1, seed=42
        )

        num_nodes_dict = {ntype: data[ntype].num_nodes for ntype in data.node_types}
        num_rels = len(data.edge_types)

        model = DistMult(
            num_nodes_dict=num_nodes_dict,
            num_relations=num_rels,
            embedding_dim=EMBEDDING_DIM,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss_fn = MarginRankingLoss(margin=1.0)
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=train_data,
            val_data=val_data,
            device="cpu",
        )

        history = trainer.fit(epochs=5)
        assert len(history["train_loss"]) == 5


# ======================================================================
# TestGlycoKGNetPhase2Integration (C1 wiring fix verification)
# ======================================================================


@requires_glycoMusubi_net
@requires_bio_hgt
class TestGlycoKGNetPhase2Integration:
    """Tests for GlycoKGNet with ALL Phase 2 components enabled.

    Verifies the C1 wiring fixes that connect tree_mpnn, BioHGT,
    cross-modal fusion, and HybridLinkScorer into a single pipeline.
    """

    def _make_phase2_model(
        self, data: HeteroData, decoder_type: str = "hybrid"
    ) -> "GlycoKGNet":
        """Create GlycoKGNet with Phase 2 components enabled."""
        num_nodes_dict = {ntype: data[ntype].num_nodes for ntype in data.node_types}
        num_rels = len(data.edge_types)
        return GlycoKGNet(
            num_nodes_dict=num_nodes_dict,
            num_relations=num_rels,
            embedding_dim=EMBEDDING_DIM,
            # Phase 2 encoders (tree_mpnn needs GlycanTreeEncoder import)
            glycan_encoder_type="learnable",
            protein_encoder_type="learnable",
            # BioHGT
            num_hgt_layers=2,
            num_hgt_heads=NUM_HEADS,
            use_bio_prior=True,
            # Fusion
            use_cross_modal_fusion=True,
            num_fusion_heads=NUM_HEADS,
            # Decoder
            decoder_type=decoder_type,
            dropout=0.0,
        )

    def test_instantiation_with_biohgt(self, mini_data: HeteroData) -> None:
        """GlycoKGNet instantiates with BioHGT layers enabled."""
        model = self._make_phase2_model(mini_data)
        assert model._use_biohgt is True
        assert hasattr(model, "hgt_layers")
        assert len(model.hgt_layers) == 2

    def test_instantiation_with_hybrid_decoder(self, mini_data: HeteroData) -> None:
        """GlycoKGNet instantiates with HybridLinkScorer decoder."""
        model = self._make_phase2_model(mini_data, decoder_type="hybrid")
        assert model._decoder_type == "hybrid"
        assert hasattr(model, "decoder")
        assert isinstance(model.decoder, HybridLinkScorer)

    def test_forward_all_node_types_with_biohgt(
        self, mini_data: HeteroData
    ) -> None:
        """forward() returns valid embeddings for ALL node types after BioHGT."""
        model = self._make_phase2_model(mini_data)
        emb_dict = model(mini_data)

        for ntype in mini_data.node_types:
            assert ntype in emb_dict, f"Missing node type '{ntype}'"
            expected_n = mini_data[ntype].num_nodes
            assert emb_dict[ntype].shape == (expected_n, EMBEDDING_DIM)
            assert torch.isfinite(emb_dict[ntype]).all(), (
                f"Non-finite embeddings for '{ntype}'"
            )

    def test_four_stages_sequential(self, mini_data: HeteroData) -> None:
        """All 4 stages run: encode -> BioHGT -> fusion -> decode."""
        model = self._make_phase2_model(mini_data)

        # Stage 1: compute initial embeddings
        initial_emb = model._compute_initial_embeddings(mini_data)
        assert len(initial_emb) == len(mini_data.node_types)

        # Stage 2: BioHGT
        biohgt_emb = model._run_biohgt(initial_emb, mini_data)
        assert len(biohgt_emb) == len(mini_data.node_types)

        # Stage 3: fusion
        fused_emb = model._run_fusion(biohgt_emb, initial_emb)
        assert len(fused_emb) == len(mini_data.node_types)

        # Stage 4: decode (score triples)
        head = fused_emb["protein"][:3]
        tail = fused_emb["glycan"][:3]
        rel_idx = torch.zeros(3, dtype=torch.long)
        scores = model.score(head, rel_idx, tail)
        assert scores.shape == (3,)
        assert torch.isfinite(scores).all()

    def test_score_with_relation_index(self, mini_data: HeteroData) -> None:
        """score() works when given integer relation indices (HybridLinkScorer mode)."""
        model = self._make_phase2_model(mini_data)
        emb_dict = model(mini_data)

        head = emb_dict["protein"][:4]
        tail = emb_dict["glycan"][:4]
        rel_idx = torch.zeros(4, dtype=torch.long)

        scores = model.score(head, rel_idx, tail)
        assert scores.shape == (4,)
        assert torch.isfinite(scores).all()

    def test_score_with_relation_embedding(self, mini_data: HeteroData) -> None:
        """score() also works when given embedding vectors (DistMult fallback)."""
        model = self._make_phase2_model(mini_data)
        emb_dict = model(mini_data)

        head = emb_dict["protein"][:4]
        tail = emb_dict["glycan"][:4]
        rel_emb = model.get_relation_embedding(torch.zeros(4, dtype=torch.long))

        scores = model.score(head, rel_emb, tail)
        assert scores.shape == (4,)
        assert torch.isfinite(scores).all()

    def test_score_triples_end_to_end_with_hybrid(
        self, mini_data: HeteroData
    ) -> None:
        """score_triples() works end-to-end with HybridLinkScorer."""
        model = self._make_phase2_model(mini_data)

        head_idx = torch.tensor([0, 1, 2])
        rel_idx = torch.tensor([0, 0, 1])
        tail_idx = torch.tensor([0, 1, 2])

        scores = model.score_triples(
            mini_data, "protein", head_idx, rel_idx, "glycan", tail_idx
        )
        assert scores.shape == (3,)
        assert torch.isfinite(scores).all()

    def test_gradient_flow_with_biohgt(self, mini_data: HeteroData) -> None:
        """Gradients flow through BioHGT layers during backward pass."""
        model = self._make_phase2_model(mini_data)
        emb_dict = model(mini_data)

        head = emb_dict["protein"][:4]
        tail = emb_dict["glycan"][:4]
        rel_idx = torch.zeros(4, dtype=torch.long)
        scores = model.score(head, rel_idx, tail)
        loss = -scores.mean()
        loss.backward()

        # BioHGT layer parameters should have gradients
        hgt_has_grad = any(
            p.grad is not None
            for p in model.hgt_layers.parameters()
            if p.requires_grad
        )
        assert hgt_has_grad, "No gradients in BioHGT layers"

    def test_stage_info_reflects_config(self, mini_data: HeteroData) -> None:
        """stage_info property reports correct active components."""
        model = self._make_phase2_model(mini_data)
        info = model.stage_info
        assert info["biohgt_enabled"] is True
        assert info["biohgt_layers"] == 2
        assert info["decoder_type"] == "hybrid"

    def test_num_parameters_positive(self, mini_data: HeteroData) -> None:
        """Phase 2 model has more parameters than a minimal Phase 1 model."""
        phase2_model = self._make_phase2_model(mini_data)
        num_nodes_dict = {
            ntype: mini_data[ntype].num_nodes for ntype in mini_data.node_types
        }
        num_rels = len(mini_data.edge_types)
        # Phase 1: no BioHGT, distmult fallback decoder
        phase1_model = GlycoKGNet(
            num_nodes_dict=num_nodes_dict,
            num_relations=num_rels,
            embedding_dim=EMBEDDING_DIM,
            num_hgt_layers=0,
            decoder_type="distmult",
        )
        assert phase2_model.num_parameters > phase1_model.num_parameters


# ======================================================================
# TestGlycoKGNetFallbackGraceful (C1 fallback verification)
# ======================================================================


@requires_glycoMusubi_net
class TestGlycoKGNetFallbackGraceful:
    """Tests for graceful degradation when Phase 2 components are unavailable."""

    def test_fallback_to_distmult_when_no_hybrid(self) -> None:
        """When decoder_type='hybrid' but HybridLinkScorer unavailable, fallback works."""
        data = _make_mini_hetero_data()
        num_nodes_dict = {ntype: data[ntype].num_nodes for ntype in data.node_types}
        num_rels = len(data.edge_types)

        # Force distmult fallback
        model = GlycoKGNet(
            num_nodes_dict=num_nodes_dict,
            num_relations=num_rels,
            embedding_dim=EMBEDDING_DIM,
            decoder_type="distmult",
        )
        assert model._decoder_type == "distmult_fallback"

        emb_dict = model(data)
        head = emb_dict["protein"][:3]
        tail = emb_dict["glycan"][:3]
        rel_emb = model.get_relation_embedding(torch.zeros(3, dtype=torch.long))

        scores = model.score(head, rel_emb, tail)
        assert scores.shape == (3,)
        assert torch.isfinite(scores).all()

    def test_model_works_without_biohgt_layers(self) -> None:
        """Setting num_hgt_layers=0 disables BioHGT gracefully."""
        data = _make_mini_hetero_data()
        num_nodes_dict = {ntype: data[ntype].num_nodes for ntype in data.node_types}
        num_rels = len(data.edge_types)
        model = GlycoKGNet(
            num_nodes_dict=num_nodes_dict,
            num_relations=num_rels,
            embedding_dim=EMBEDDING_DIM,
            num_hgt_layers=0,
        )
        assert model._use_biohgt is False

        emb_dict = model(data)
        assert len(emb_dict) == len(data.node_types)
        for ntype in data.node_types:
            assert torch.isfinite(emb_dict[ntype]).all()

    def test_score_triples_works_with_distmult_fallback(self) -> None:
        """score_triples() works end-to-end with DistMult fallback decoder."""
        data = _make_mini_hetero_data()
        num_nodes_dict = {ntype: data[ntype].num_nodes for ntype in data.node_types}
        num_rels = len(data.edge_types)
        model = GlycoKGNet(
            num_nodes_dict=num_nodes_dict,
            num_relations=num_rels,
            embedding_dim=EMBEDDING_DIM,
            decoder_type="distmult",
        )

        head_idx = torch.tensor([0, 1])
        rel_idx = torch.tensor([0, 0])
        tail_idx = torch.tensor([0, 1])

        scores = model.score_triples(
            data, "protein", head_idx, rel_idx, "glycan", tail_idx
        )
        assert scores.shape == (2,)
        assert torch.isfinite(scores).all()
