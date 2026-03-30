"""Integration tests for Phase 4 node classification and graph-level prediction.

Tests cover:
  - GlycoKGNet.node_classify() produces correct logits
  - GlycoKGNet.predict_graph() produces correct predictions
  - CompositeLoss with L_node term works end-to-end
  - Backward through node classification path
  - Combined link + node classification training loop (5 steps)
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from glycoMusubi.embedding.decoders.node_classifier import NodeClassifier
from glycoMusubi.embedding.decoders.graph_level_decoder import GraphLevelDecoder
from glycoMusubi.losses.composite_loss import CompositeLoss
from glycoMusubi.losses.margin_loss import MarginRankingLoss

try:
    from glycoMusubi.embedding.models.glycoMusubi_net import GlycoKGNet
    _HAS_GLYCO_KG_NET = True
except ImportError:
    _HAS_GLYCO_KG_NET = False

requires_glycoMusubi_net = pytest.mark.skipif(
    not _HAS_GLYCO_KG_NET,
    reason="GlycoKGNet not available",
)


# ======================================================================
# Constants
# ======================================================================

EMBEDDING_DIM = 64
NUM_CLASSES_TAXONOMY = 5
NUM_CLASSES_FUNCTION = 8
NUM_CLASSES_GRAPH = 3


# ======================================================================
# Fixtures
# ======================================================================


def _make_mini_hetero_data() -> HeteroData:
    """Create a minimal HeteroData for testing."""
    data = HeteroData()

    node_counts = {
        "glycan": 5,
        "protein": 6,
        "disease": 3,
    }
    for ntype, n in node_counts.items():
        x = torch.empty(n, EMBEDDING_DIM)
        nn.init.xavier_uniform_(x)
        data[ntype].x = x
        data[ntype].num_nodes = n

    data["protein", "has_glycan", "glycan"].edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 0]]
    )
    data["protein", "associated_with_disease", "disease"].edge_index = torch.tensor(
        [[0, 1, 2, 3], [0, 1, 2, 0]]
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


def _make_model_with_node_classifier(
    num_nodes_dict: dict, num_relations: int,
) -> "GlycoKGNet":
    """Create GlycoKGNet with a NodeClassifier."""
    nc = NodeClassifier(
        embed_dim=EMBEDDING_DIM,
        task_configs={
            "glycan_taxonomy": NUM_CLASSES_TAXONOMY,
            "protein_function": NUM_CLASSES_FUNCTION,
        },
    )
    return GlycoKGNet(
        num_nodes_dict=num_nodes_dict,
        num_relations=num_relations,
        embedding_dim=EMBEDDING_DIM,
        num_hgt_layers=0,
        decoder_type="distmult",
        node_classifier=nc,
    )


def _make_model_with_graph_decoder(
    num_nodes_dict: dict, num_relations: int,
) -> "GlycoKGNet":
    """Create GlycoKGNet with a GraphLevelDecoder."""
    gd = GraphLevelDecoder(
        embed_dim=EMBEDDING_DIM,
        num_classes=NUM_CLASSES_GRAPH,
    )
    return GlycoKGNet(
        num_nodes_dict=num_nodes_dict,
        num_relations=num_relations,
        embedding_dim=EMBEDDING_DIM,
        num_hgt_layers=0,
        decoder_type="distmult",
        graph_decoder=gd,
    )


# ======================================================================
# TestNodeClassify
# ======================================================================


@requires_glycoMusubi_net
class TestNodeClassify:
    """Tests for GlycoKGNet.node_classify()."""

    def test_node_classify_glycan_output_shape(
        self, mini_data: HeteroData, num_nodes_dict: dict, num_relations: int,
    ) -> None:
        """node_classify() produces (num_glycans, num_classes) logits."""
        model = _make_model_with_node_classifier(num_nodes_dict, num_relations)
        logits = model.node_classify(mini_data, "glycan_taxonomy", "glycan")
        assert logits.shape == (num_nodes_dict["glycan"], NUM_CLASSES_TAXONOMY)

    def test_node_classify_protein_output_shape(
        self, mini_data: HeteroData, num_nodes_dict: dict, num_relations: int,
    ) -> None:
        """node_classify() works for protein nodes too."""
        model = _make_model_with_node_classifier(num_nodes_dict, num_relations)
        logits = model.node_classify(mini_data, "protein_function", "protein")
        assert logits.shape == (num_nodes_dict["protein"], NUM_CLASSES_FUNCTION)

    def test_node_classify_finite(
        self, mini_data: HeteroData, num_nodes_dict: dict, num_relations: int,
    ) -> None:
        """All logits are finite."""
        model = _make_model_with_node_classifier(num_nodes_dict, num_relations)
        logits = model.node_classify(mini_data, "glycan_taxonomy", "glycan")
        assert torch.isfinite(logits).all()

    def test_node_classify_without_classifier_raises(
        self, mini_data: HeteroData, num_nodes_dict: dict, num_relations: int,
    ) -> None:
        """node_classify() raises RuntimeError if no classifier was provided."""
        model = GlycoKGNet(
            num_nodes_dict=num_nodes_dict,
            num_relations=num_relations,
            embedding_dim=EMBEDDING_DIM,
            num_hgt_layers=0,
            decoder_type="distmult",
        )
        with pytest.raises(RuntimeError, match="node_classify.*requires.*NodeClassifier"):
            model.node_classify(mini_data, "glycan_taxonomy", "glycan")


# ======================================================================
# TestPredictGraph
# ======================================================================


@requires_glycoMusubi_net
class TestPredictGraph:
    """Tests for GlycoKGNet.predict_graph()."""

    def test_predict_graph_all_nodes(
        self, mini_data: HeteroData, num_nodes_dict: dict, num_relations: int,
    ) -> None:
        """predict_graph() pools all node types, returns (1, num_classes)."""
        model = _make_model_with_graph_decoder(num_nodes_dict, num_relations)
        pred = model.predict_graph(mini_data)
        assert pred.shape == (1, NUM_CLASSES_GRAPH)

    def test_predict_graph_subgraph(
        self, mini_data: HeteroData, num_nodes_dict: dict, num_relations: int,
    ) -> None:
        """predict_graph() with subgraph_nodes restricts pooling."""
        model = _make_model_with_graph_decoder(num_nodes_dict, num_relations)
        subgraph = {
            "glycan": torch.tensor([0, 1, 2]),
            "protein": torch.tensor([0, 1]),
        }
        pred = model.predict_graph(mini_data, subgraph_nodes=subgraph)
        assert pred.shape == (1, NUM_CLASSES_GRAPH)

    def test_predict_graph_finite(
        self, mini_data: HeteroData, num_nodes_dict: dict, num_relations: int,
    ) -> None:
        """Predictions are finite."""
        model = _make_model_with_graph_decoder(num_nodes_dict, num_relations)
        pred = model.predict_graph(mini_data)
        assert torch.isfinite(pred).all()

    def test_predict_graph_without_decoder_raises(
        self, mini_data: HeteroData, num_nodes_dict: dict, num_relations: int,
    ) -> None:
        """predict_graph() raises RuntimeError if no decoder was provided."""
        model = GlycoKGNet(
            num_nodes_dict=num_nodes_dict,
            num_relations=num_relations,
            embedding_dim=EMBEDDING_DIM,
            num_hgt_layers=0,
            decoder_type="distmult",
        )
        with pytest.raises(RuntimeError, match="predict_graph.*requires.*GraphLevelDecoder"):
            model.predict_graph(mini_data)


# ======================================================================
# TestCompositeLossWithNodeClassification
# ======================================================================


@requires_glycoMusubi_net
class TestCompositeLossWithNodeClassification:
    """Tests for CompositeLoss with the L_node term."""

    def test_l_node_added_when_lambda_positive(self) -> None:
        """L_node is added when lambda_node > 0 and logits+labels provided."""
        link_loss = MarginRankingLoss(margin=1.0)
        composite = CompositeLoss(link_loss=link_loss, lambda_node=0.5)

        pos_scores = torch.randn(8)
        neg_scores = torch.randn(8)
        logits = torch.randn(10, NUM_CLASSES_TAXONOMY)
        labels = torch.randint(0, NUM_CLASSES_TAXONOMY, (10,))

        loss_with_node = composite(
            pos_scores, neg_scores,
            node_logits=logits, node_labels=labels,
        )
        loss_without_node = composite(pos_scores, neg_scores)

        # Adding node classification loss should increase total (almost surely)
        assert loss_with_node.item() != loss_without_node.item()

    def test_l_node_disabled_when_lambda_zero(self) -> None:
        """L_node is skipped when lambda_node = 0."""
        link_loss = MarginRankingLoss(margin=1.0)
        composite = CompositeLoss(link_loss=link_loss, lambda_node=0.0)

        pos_scores = torch.randn(8)
        neg_scores = torch.randn(8)
        logits = torch.randn(10, NUM_CLASSES_TAXONOMY)
        labels = torch.randint(0, NUM_CLASSES_TAXONOMY, (10,))

        loss_with = composite(
            pos_scores, neg_scores,
            node_logits=logits, node_labels=labels,
        )
        loss_without = composite(pos_scores, neg_scores)

        assert torch.allclose(loss_with, loss_without, atol=1e-6)

    def test_l_node_is_scalar(self) -> None:
        """Composite loss with L_node returns a scalar."""
        link_loss = MarginRankingLoss(margin=1.0)
        composite = CompositeLoss(link_loss=link_loss, lambda_node=0.5)

        pos = torch.randn(4)
        neg = torch.randn(4)
        logits = torch.randn(5, 3)
        labels = torch.randint(0, 3, (5,))

        loss = composite(pos, neg, node_logits=logits, node_labels=labels)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_l_node_gradient_flow(self) -> None:
        """Gradients flow through L_node to node logits."""
        link_loss = MarginRankingLoss(margin=1.0)
        composite = CompositeLoss(link_loss=link_loss, lambda_node=0.5)

        pos = torch.tensor([1.0], requires_grad=True)
        neg = torch.tensor([0.0], requires_grad=True)
        logits = torch.randn(5, 3, requires_grad=True)
        labels = torch.randint(0, 3, (5,))

        loss = composite(pos, neg, node_logits=logits, node_labels=labels)
        loss.backward()

        assert logits.grad is not None
        assert pos.grad is not None


# ======================================================================
# TestBackwardThroughNodeClassification
# ======================================================================


@requires_glycoMusubi_net
class TestBackwardThroughNodeClassification:
    """Tests for backward pass through the full node classification path."""

    def test_gradients_flow_from_node_classify(
        self, mini_data: HeteroData, num_nodes_dict: dict, num_relations: int,
    ) -> None:
        """Backward from node_classify() loss reaches model parameters."""
        model = _make_model_with_node_classifier(num_nodes_dict, num_relations)
        logits = model.node_classify(mini_data, "glycan_taxonomy", "glycan")

        labels = torch.randint(0, NUM_CLASSES_TAXONOMY, (num_nodes_dict["glycan"],))
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()

        # Node classifier params should have gradients
        params_with_grad = sum(
            1 for p in model.node_classifier.parameters()
            if p.grad is not None
        )
        assert params_with_grad > 0, "No gradients in node classifier"

        # Embedding parameters should also have gradients
        total_with_grad = sum(
            1 for _, p in model.named_parameters()
            if p.requires_grad and p.grad is not None
        )
        assert total_with_grad > params_with_grad, (
            "Only classifier params have gradients; embedding params should too"
        )

    def test_gradients_flow_from_predict_graph(
        self, mini_data: HeteroData, num_nodes_dict: dict, num_relations: int,
    ) -> None:
        """Backward from predict_graph() loss reaches model parameters."""
        model = _make_model_with_graph_decoder(num_nodes_dict, num_relations)
        pred = model.predict_graph(mini_data)

        labels = torch.randint(0, NUM_CLASSES_GRAPH, (1,))
        loss = torch.nn.functional.cross_entropy(pred, labels)
        loss.backward()

        # Graph decoder params should have gradients
        params_with_grad = sum(
            1 for p in model.graph_decoder.parameters()
            if p.grad is not None
        )
        assert params_with_grad > 0, "No gradients in graph decoder"


# ======================================================================
# TestCombinedTrainingLoop
# ======================================================================


@requires_glycoMusubi_net
class TestCombinedTrainingLoop:
    """Tests for a combined link + node classification training loop."""

    def test_combined_training_5_steps(
        self, mini_data: HeteroData, num_nodes_dict: dict, num_relations: int,
    ) -> None:
        """Combined link prediction + node classification for 5 steps."""
        nc = NodeClassifier(
            embed_dim=EMBEDDING_DIM,
            task_configs={"glycan_taxonomy": NUM_CLASSES_TAXONOMY},
        )
        model = GlycoKGNet(
            num_nodes_dict=num_nodes_dict,
            num_relations=num_relations,
            embedding_dim=EMBEDDING_DIM,
            num_hgt_layers=0,
            decoder_type="distmult",
            node_classifier=nc,
        )

        link_loss_fn = MarginRankingLoss(margin=1.0)
        composite = CompositeLoss(
            link_loss=link_loss_fn,
            lambda_node=0.5,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Generate fake labels for glycan taxonomy
        glycan_labels = torch.randint(
            0, NUM_CLASSES_TAXONOMY, (num_nodes_dict["glycan"],)
        )

        losses = []
        for step in range(5):
            optimizer.zero_grad()

            # Link prediction scores (simplified)
            emb_dict = model(mini_data)
            edge_type = mini_data.edge_types[0]
            src_type, _, dst_type = edge_type
            ei = mini_data[edge_type].edge_index
            head_emb = emb_dict[src_type][ei[0]]
            tail_emb = emb_dict[dst_type][ei[1]]
            rel_emb = model.get_relation_embedding(
                torch.zeros(ei.size(1), dtype=torch.long)
            )
            pos_scores = model.score(head_emb, rel_emb, tail_emb)

            # Negative samples (random tails)
            neg_tail = emb_dict[dst_type][
                torch.randint(0, num_nodes_dict[dst_type], (ei.size(1),))
            ]
            neg_scores = model.score(head_emb, rel_emb, neg_tail)

            # Node classification logits
            node_logits = model.node_classify(
                mini_data, "glycan_taxonomy", "glycan"
            )

            loss = composite(
                pos_scores, neg_scores,
                node_logits=node_logits,
                node_labels=glycan_labels,
            )

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # All losses should be finite
        assert all(torch.isfinite(torch.tensor(l)) for l in losses), (
            f"Non-finite losses: {losses}"
        )
        # Should have completed 5 steps
        assert len(losses) == 5

    def test_both_decoders_together(
        self, mini_data: HeteroData, num_nodes_dict: dict, num_relations: int,
    ) -> None:
        """Both node_classifier and graph_decoder can be used on the same model."""
        nc = NodeClassifier(
            embed_dim=EMBEDDING_DIM,
            task_configs={"glycan_taxonomy": NUM_CLASSES_TAXONOMY},
        )
        gd = GraphLevelDecoder(
            embed_dim=EMBEDDING_DIM,
            num_classes=NUM_CLASSES_GRAPH,
        )
        model = GlycoKGNet(
            num_nodes_dict=num_nodes_dict,
            num_relations=num_relations,
            embedding_dim=EMBEDDING_DIM,
            num_hgt_layers=0,
            decoder_type="distmult",
            node_classifier=nc,
            graph_decoder=gd,
        )

        # Both methods should work
        node_logits = model.node_classify(mini_data, "glycan_taxonomy", "glycan")
        graph_pred = model.predict_graph(mini_data)

        assert node_logits.shape == (num_nodes_dict["glycan"], NUM_CLASSES_TAXONOMY)
        assert graph_pred.shape == (1, NUM_CLASSES_GRAPH)
        assert torch.isfinite(node_logits).all()
        assert torch.isfinite(graph_pred).all()
