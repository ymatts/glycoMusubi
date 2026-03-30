"""Tests for the MaskedEdgePredictor fix (edge restoration).

Covers:
  - Edge count is preserved after predict + restore
  - edge_type_idx is properly restored
  - Original graph data is not corrupted
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from glycoMusubi.training.pretraining import MaskedEdgePredictor


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


class _SimpleEncoder(nn.Module):
    """Minimal encoder that returns node embeddings from an embedding table."""

    def __init__(self, num_nodes_dict: dict, dim: int = 32) -> None:
        super().__init__()
        self.embeddings = nn.ModuleDict(
            {nt: nn.Embedding(n, dim) for nt, n in num_nodes_dict.items()}
        )

    def forward(self, data: HeteroData) -> dict:
        emb = {}
        for nt, emb_mod in self.embeddings.items():
            n = data[nt].num_nodes
            idx = torch.arange(n, device=emb_mod.weight.device)
            emb[nt] = emb_mod(idx)
        return emb


def _make_test_graph() -> HeteroData:
    """Create a test HeteroData with edge_type_idx attributes."""
    data = HeteroData()

    data["protein"].num_nodes = 5
    data["protein"].x = torch.randn(5, 32)
    data["glycan"].num_nodes = 4
    data["glycan"].x = torch.randn(4, 32)

    # protein -> glycan: 6 edges
    ei1 = torch.tensor([[0, 1, 2, 3, 4, 0], [0, 1, 2, 3, 0, 1]], dtype=torch.long)
    data["protein", "has_glycan", "glycan"].edge_index = ei1
    data["protein", "has_glycan", "glycan"].edge_type_idx = torch.zeros(
        6, dtype=torch.long
    )

    # protein -> protein: 4 edges
    ei2 = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    data["protein", "interacts", "protein"].edge_index = ei2
    data["protein", "interacts", "protein"].edge_type_idx = torch.ones(
        4, dtype=torch.long
    )

    return data


def _count_edges(data: HeteroData) -> dict:
    """Return edge counts per edge type."""
    counts = {}
    for et in data.edge_types:
        ei = data[et].edge_index
        counts[et] = ei.size(1) if ei is not None else 0
    return counts


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestEdgeCountPreservation:
    """Edge count is the same before and after mask_and_predict."""

    def test_edge_count_preserved(self) -> None:
        data = _make_test_graph()
        counts_before = _count_edges(data)

        encoder = _SimpleEncoder({"protein": 5, "glycan": 4}, dim=32)
        predictor = MaskedEdgePredictor(embedding_dim=32, num_relations=2)

        _loss, _preds = predictor.mask_and_predict(data, encoder, mask_ratio=0.3)

        counts_after = _count_edges(data)
        assert counts_before == counts_after

    def test_edge_count_preserved_high_mask_ratio(self) -> None:
        """Even with a high mask ratio, all edges are restored."""
        data = _make_test_graph()
        counts_before = _count_edges(data)

        encoder = _SimpleEncoder({"protein": 5, "glycan": 4}, dim=32)
        predictor = MaskedEdgePredictor(embedding_dim=32, num_relations=2)

        _loss, _preds = predictor.mask_and_predict(data, encoder, mask_ratio=0.9)

        counts_after = _count_edges(data)
        assert counts_before == counts_after

    def test_edge_count_preserved_minimal_mask(self) -> None:
        """With very small mask_ratio, still at least 1 edge is masked/restored."""
        data = _make_test_graph()
        counts_before = _count_edges(data)

        encoder = _SimpleEncoder({"protein": 5, "glycan": 4}, dim=32)
        predictor = MaskedEdgePredictor(embedding_dim=32, num_relations=1)

        _loss, _preds = predictor.mask_and_predict(data, encoder, mask_ratio=0.01)

        counts_after = _count_edges(data)
        assert counts_before == counts_after


class TestEdgeTypeIdxRestoration:
    """edge_type_idx is properly restored after mask_and_predict."""

    def test_edge_type_idx_restored(self) -> None:
        data = _make_test_graph()

        # Save originals
        orig_idx = {}
        for et in data.edge_types:
            if hasattr(data[et], "edge_type_idx"):
                orig_idx[et] = data[et].edge_type_idx.clone()

        encoder = _SimpleEncoder({"protein": 5, "glycan": 4}, dim=32)
        predictor = MaskedEdgePredictor(embedding_dim=32, num_relations=2)

        _loss, _preds = predictor.mask_and_predict(data, encoder, mask_ratio=0.3)

        for et, orig in orig_idx.items():
            assert torch.equal(
                data[et].edge_type_idx, orig
            ), f"edge_type_idx not restored for {et}"

    def test_edge_type_idx_values_unchanged(self) -> None:
        """The actual values in edge_type_idx must be identical after restore."""
        data = _make_test_graph()

        # protein->glycan should be all 0s, protein->protein should be all 1s
        encoder = _SimpleEncoder({"protein": 5, "glycan": 4}, dim=32)
        predictor = MaskedEdgePredictor(embedding_dim=32, num_relations=2)

        _loss, _preds = predictor.mask_and_predict(data, encoder, mask_ratio=0.5)

        pg_idx = data["protein", "has_glycan", "glycan"].edge_type_idx
        pp_idx = data["protein", "interacts", "protein"].edge_type_idx

        assert (pg_idx == 0).all(), f"Expected all 0s, got {pg_idx}"
        assert (pp_idx == 1).all(), f"Expected all 1s, got {pp_idx}"


class TestOriginalGraphIntegrity:
    """Original graph data (edge_index content) is not corrupted."""

    def test_edge_index_content_unchanged(self) -> None:
        data = _make_test_graph()

        # Save original edge indices
        orig_edges = {}
        for et in data.edge_types:
            orig_edges[et] = data[et].edge_index.clone()

        encoder = _SimpleEncoder({"protein": 5, "glycan": 4}, dim=32)
        predictor = MaskedEdgePredictor(embedding_dim=32, num_relations=2)

        _loss, _preds = predictor.mask_and_predict(data, encoder, mask_ratio=0.3)

        for et, orig_ei in orig_edges.items():
            restored_ei = data[et].edge_index
            assert torch.equal(
                restored_ei, orig_ei
            ), f"edge_index content corrupted for {et}"

    def test_node_features_not_corrupted(self) -> None:
        """Node features should remain unchanged (MaskedEdgePredictor doesn't touch them)."""
        data = _make_test_graph()

        orig_protein_x = data["protein"].x.clone()
        orig_glycan_x = data["glycan"].x.clone()

        encoder = _SimpleEncoder({"protein": 5, "glycan": 4}, dim=32)
        predictor = MaskedEdgePredictor(embedding_dim=32, num_relations=2)

        _loss, _preds = predictor.mask_and_predict(data, encoder, mask_ratio=0.3)

        assert torch.equal(data["protein"].x, orig_protein_x)
        assert torch.equal(data["glycan"].x, orig_glycan_x)

    def test_multiple_rounds_no_degradation(self) -> None:
        """Running mask_and_predict multiple times should not degrade the graph."""
        data = _make_test_graph()

        orig_edges = {}
        orig_idx = {}
        for et in data.edge_types:
            orig_edges[et] = data[et].edge_index.clone()
            if hasattr(data[et], "edge_type_idx"):
                orig_idx[et] = data[et].edge_type_idx.clone()

        encoder = _SimpleEncoder({"protein": 5, "glycan": 4}, dim=32)
        predictor = MaskedEdgePredictor(embedding_dim=32, num_relations=2)

        for _ in range(5):
            _loss, _preds = predictor.mask_and_predict(data, encoder, mask_ratio=0.3)

        for et, orig_ei in orig_edges.items():
            assert torch.equal(data[et].edge_index, orig_ei), (
                f"edge_index degraded after multiple rounds for {et}"
            )
        for et, orig in orig_idx.items():
            assert torch.equal(data[et].edge_type_idx, orig), (
                f"edge_type_idx degraded after multiple rounds for {et}"
            )

    def test_loss_is_finite(self) -> None:
        """Loss returned by mask_and_predict should be finite."""
        data = _make_test_graph()
        encoder = _SimpleEncoder({"protein": 5, "glycan": 4}, dim=32)
        predictor = MaskedEdgePredictor(embedding_dim=32, num_relations=2)

        loss, preds = predictor.mask_and_predict(data, encoder, mask_ratio=0.3)

        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        assert loss.item() >= 0.0

    def test_predictions_contain_existence_logits(self) -> None:
        """Predictions should contain existence_logits."""
        data = _make_test_graph()
        encoder = _SimpleEncoder({"protein": 5, "glycan": 4}, dim=32)
        predictor = MaskedEdgePredictor(embedding_dim=32, num_relations=2)

        _loss, preds = predictor.mask_and_predict(data, encoder, mask_ratio=0.3)

        assert "existence_logits" in preds
        assert preds["existence_logits"].dim() == 1
        assert preds["existence_logits"].numel() > 0

    def test_predictions_contain_relation_logits(self) -> None:
        """With num_relations > 1 and edge_type_idx, relation_logits should be present."""
        data = _make_test_graph()
        encoder = _SimpleEncoder({"protein": 5, "glycan": 4}, dim=32)
        predictor = MaskedEdgePredictor(embedding_dim=32, num_relations=2)

        _loss, preds = predictor.mask_and_predict(data, encoder, mask_ratio=0.3)

        assert "relation_logits" in preds
        assert preds["relation_logits"].dim() == 2
        assert preds["relation_logits"].size(1) == 2
