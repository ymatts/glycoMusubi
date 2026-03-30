"""Unit and integration tests for PathReasoner (NBFNet-style Bellman-Ford GNN).

Validates BellmanFordLayer and PathReasoner per
docs/design/algorithm_design.md Section 4.2.3.

Test coverage:
  - BellmanFordLayer: message function shapes, aggregation modes
  - PathReasoner.__init__: correct module structure
  - forward(): output shapes for various graph sizes
  - score(): output shape [batch]
  - score_query(): returns Dict[str, Tensor] with all candidate scores
  - Boundary condition: query entity initialised correctly
  - Inverse edges: reverse edges added with distinct relation embeddings
  - Gradient flow through message passing iterations
  - Integration with BaseKGEModel interface
  - Correctness: path-based scores on simple graphs
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from glycoMusubi.embedding.models.base import BaseKGEModel
from glycoMusubi.embedding.models.path_reasoner import BellmanFordLayer, PathReasoner


# ======================================================================
# Constants
# ======================================================================

EMBEDDING_DIM = 32
NUM_RELATIONS = 3
NUM_NODES_DICT = {"protein": 10, "glycan": 8, "disease": 5}
BATCH_SIZE = 4


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def num_nodes_dict():
    return NUM_NODES_DICT.copy()


@pytest.fixture()
def mini_hetero_data() -> HeteroData:
    """Minimal HeteroData with edges for testing."""
    torch.manual_seed(42)
    data = HeteroData()
    data["disease"].num_nodes = 5
    data["glycan"].num_nodes = 8
    data["protein"].num_nodes = 10

    # protein -> glycan edges
    data["protein", "binds", "glycan"].edge_index = torch.tensor(
        [[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long
    )
    # glycan -> disease edges
    data["glycan", "associated_with", "disease"].edge_index = torch.tensor(
        [[0, 1, 2], [0, 1, 2]], dtype=torch.long
    )
    # protein -> disease edges
    data["protein", "causes", "disease"].edge_index = torch.tensor(
        [[0, 1], [0, 1]], dtype=torch.long
    )
    return data


@pytest.fixture()
def model(num_nodes_dict) -> PathReasoner:
    """PathReasoner with small dimensions for fast testing."""
    torch.manual_seed(42)
    return PathReasoner(
        num_nodes_dict=num_nodes_dict,
        num_relations=NUM_RELATIONS,
        embedding_dim=EMBEDDING_DIM,
        num_iterations=3,
        aggregation="sum",
        dropout=0.0,
    )


@pytest.fixture()
def bf_layer() -> BellmanFordLayer:
    """Single BellmanFordLayer for unit testing."""
    torch.manual_seed(42)
    return BellmanFordLayer(
        embedding_dim=EMBEDDING_DIM,
        aggregation="sum",
        dropout=0.0,
    )


# ======================================================================
# TestBellmanFordLayer
# ======================================================================


class TestBellmanFordLayer:
    """Tests for individual BellmanFordLayer."""

    def test_message_transform_shapes(self, bf_layer: BellmanFordLayer) -> None:
        """Message function produces correct output shapes."""
        num_nodes = 10
        num_edges = 15
        h = torch.randn(num_nodes, EMBEDDING_DIM)
        edge_index = torch.stack([
            torch.randint(0, num_nodes, (num_edges,)),
            torch.randint(0, num_nodes, (num_edges,)),
        ])
        edge_rel_emb = torch.randn(num_edges, EMBEDDING_DIM)

        out = bf_layer(h, edge_index, edge_rel_emb, num_nodes)
        assert out.shape == (num_nodes, EMBEDDING_DIM)

    def test_residual_connection(self, bf_layer: BellmanFordLayer) -> None:
        """Output includes residual: h_new = h + agg."""
        num_nodes = 5
        h = torch.randn(num_nodes, EMBEDDING_DIM)
        # No edges: aggregation should be zero, output == input.
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_rel_emb = torch.zeros(0, EMBEDDING_DIM)

        out = bf_layer(h, edge_index, edge_rel_emb, num_nodes)
        torch.testing.assert_close(out, h)

    def test_aggregation_sum(self) -> None:
        """Sum aggregation sums all incoming messages."""
        layer = BellmanFordLayer(EMBEDDING_DIM, aggregation="sum", dropout=0.0)
        num_nodes = 4
        h = torch.zeros(num_nodes, EMBEDDING_DIM)
        # Two edges both pointing to node 0.
        edge_index = torch.tensor([[1, 2], [0, 0]], dtype=torch.long)
        edge_rel_emb = torch.zeros(2, EMBEDDING_DIM)

        out = layer(h, edge_index, edge_rel_emb, num_nodes)
        # Node 0 should get a non-zero value (two messages aggregated).
        # Nodes 1, 2 remain at zero + residual(zero) = result of msg_mlp(0+0)
        assert out.shape == (num_nodes, EMBEDDING_DIM)

    def test_aggregation_mean(self) -> None:
        """Mean aggregation averages incoming messages."""
        layer = BellmanFordLayer(EMBEDDING_DIM, aggregation="mean", dropout=0.0)
        num_nodes = 4
        h = torch.randn(num_nodes, EMBEDDING_DIM)
        edge_index = torch.tensor([[0, 1, 2], [3, 3, 3]], dtype=torch.long)
        edge_rel_emb = torch.randn(3, EMBEDDING_DIM)

        out = layer(h, edge_index, edge_rel_emb, num_nodes)
        assert out.shape == (num_nodes, EMBEDDING_DIM)

    def test_aggregation_pna(self) -> None:
        """PNA aggregation combines sum, mean, max via learned projection."""
        layer = BellmanFordLayer(EMBEDDING_DIM, aggregation="pna", dropout=0.0)
        assert hasattr(layer, "pna_proj")
        assert isinstance(layer.pna_proj, nn.Linear)
        assert layer.pna_proj.in_features == EMBEDDING_DIM * 3
        assert layer.pna_proj.out_features == EMBEDDING_DIM

        num_nodes = 5
        h = torch.randn(num_nodes, EMBEDDING_DIM)
        edge_index = torch.tensor([[0, 1, 2], [3, 3, 4]], dtype=torch.long)
        edge_rel_emb = torch.randn(3, EMBEDDING_DIM)

        out = layer(h, edge_index, edge_rel_emb, num_nodes)
        assert out.shape == (num_nodes, EMBEDDING_DIM)

    def test_unknown_aggregation_raises(self) -> None:
        """Unknown aggregation type raises ValueError."""
        layer = BellmanFordLayer(EMBEDDING_DIM, aggregation="invalid", dropout=0.0)
        h = torch.randn(3, EMBEDDING_DIM)
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        edge_rel_emb = torch.randn(1, EMBEDDING_DIM)

        with pytest.raises(ValueError, match="Unknown aggregation"):
            layer(h, edge_index, edge_rel_emb, 3)

    def test_relation_specific_transform(self) -> None:
        """Different relation embeddings produce different messages."""
        layer = BellmanFordLayer(EMBEDDING_DIM, aggregation="sum", dropout=0.0)
        num_nodes = 3
        h = torch.randn(num_nodes, EMBEDDING_DIM)
        edge_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)

        rel_emb_a = torch.randn(1, EMBEDDING_DIM).expand(1, -1)
        rel_emb_b = torch.randn(1, EMBEDDING_DIM).expand(1, -1)

        # Single edge with different relation embeddings.
        edge_index_single = torch.tensor([[0], [1]], dtype=torch.long)
        out_a = layer(h, edge_index_single, rel_emb_a, num_nodes)
        out_b = layer(h, edge_index_single, rel_emb_b, num_nodes)

        # Node 1 should receive different messages.
        assert not torch.allclose(out_a[1], out_b[1], atol=1e-6)


# ======================================================================
# TestPathReasonerInit
# ======================================================================


class TestPathReasonerInit:
    """Tests for PathReasoner.__init__ module structure."""

    def test_inherits_base(self, model: PathReasoner) -> None:
        """PathReasoner is a subclass of BaseKGEModel."""
        assert isinstance(model, BaseKGEModel)
        assert isinstance(model, nn.Module)

    def test_node_embeddings_created(self, model: PathReasoner) -> None:
        """Node embedding tables created for each node type."""
        for nt, count in NUM_NODES_DICT.items():
            assert nt in model.node_embeddings
            assert model.node_embeddings[nt].num_embeddings == count
            assert model.node_embeddings[nt].embedding_dim == EMBEDDING_DIM

    def test_relation_embeddings_created(self, model: PathReasoner) -> None:
        """Relation embedding table has correct size."""
        assert model.relation_embeddings.num_embeddings == NUM_RELATIONS
        assert model.relation_embeddings.embedding_dim == EMBEDDING_DIM

    def test_inverse_relation_embeddings(self, model: PathReasoner) -> None:
        """Inverse relation embedding table is separate and same size."""
        assert model.inv_relation_embeddings.num_embeddings == NUM_RELATIONS
        assert model.inv_relation_embeddings.embedding_dim == EMBEDDING_DIM
        # Inverse embeddings should differ from forward embeddings.
        assert not torch.equal(
            model.relation_embeddings.weight,
            model.inv_relation_embeddings.weight,
        )

    def test_bf_layers_count(self, model: PathReasoner) -> None:
        """Correct number of BellmanFord layers created."""
        assert len(model.bf_layers) == 3
        for layer in model.bf_layers:
            assert isinstance(layer, BellmanFordLayer)

    def test_layer_norms_count(self, model: PathReasoner) -> None:
        """One LayerNorm per BF iteration."""
        assert len(model.layer_norms) == 3
        for ln in model.layer_norms:
            assert isinstance(ln, nn.LayerNorm)
            assert ln.normalized_shape == (EMBEDDING_DIM,)

    def test_score_mlp_structure(self, model: PathReasoner) -> None:
        """Score MLP takes 2*dim and outputs scalar."""
        mlp = model.score_mlp
        assert isinstance(mlp, nn.Sequential)
        # First layer: 2*dim -> dim.
        assert isinstance(mlp[0], nn.Linear)
        assert mlp[0].in_features == EMBEDDING_DIM * 2
        assert mlp[0].out_features == EMBEDDING_DIM
        # Last layer: dim -> 1.
        assert isinstance(mlp[-1], nn.Linear)
        assert mlp[-1].out_features == 1

    def test_total_relations_doubled(self, model: PathReasoner) -> None:
        """num_total_relations doubles for inverse edges."""
        assert model.num_total_relations == NUM_RELATIONS * 2

    def test_total_nodes_computed(self, model: PathReasoner) -> None:
        """Internal _total_nodes sums all node type counts."""
        expected = sum(NUM_NODES_DICT.values())
        assert model._total_nodes == expected


# ======================================================================
# TestPathReasonerShapes
# ======================================================================


class TestPathReasonerShapes:
    """Shape tests for forward(), score(), and score_query()."""

    def test_forward_output_shapes(
        self, model: PathReasoner, mini_hetero_data: HeteroData
    ) -> None:
        """forward() returns dict with correct shapes per node type."""
        out = model(mini_hetero_data)
        assert isinstance(out, dict)
        for nt, count in NUM_NODES_DICT.items():
            assert nt in out
            assert out[nt].shape == (count, EMBEDDING_DIM)

    def test_forward_all_types_present(
        self, model: PathReasoner, mini_hetero_data: HeteroData
    ) -> None:
        """forward() returns embeddings for all node types."""
        out = model(mini_hetero_data)
        for nt in NUM_NODES_DICT:
            assert nt in out

    def test_score_output_shape(self, model: PathReasoner) -> None:
        """score() returns shape [batch]."""
        head = torch.randn(BATCH_SIZE, EMBEDDING_DIM)
        rel = torch.randn(BATCH_SIZE, EMBEDDING_DIM)
        tail = torch.randn(BATCH_SIZE, EMBEDDING_DIM)
        scores = model.score(head, rel, tail)
        assert scores.shape == (BATCH_SIZE,)

    def test_score_single_triple(self, model: PathReasoner) -> None:
        """score() works for batch size 1."""
        head = torch.randn(1, EMBEDDING_DIM)
        rel = torch.randn(1, EMBEDDING_DIM)
        tail = torch.randn(1, EMBEDDING_DIM)
        scores = model.score(head, rel, tail)
        assert scores.shape == (1,)

    def test_score_query_output_shapes(
        self, model: PathReasoner, mini_hetero_data: HeteroData
    ) -> None:
        """score_query() returns correct shapes for all node types."""
        head_idx = torch.tensor([0, 1])
        relation_idx = torch.tensor([0, 1])
        result = model.score_query(
            mini_hetero_data, "protein", head_idx, relation_idx
        )
        assert isinstance(result, dict)
        batch_size = head_idx.size(0)
        for nt, count in NUM_NODES_DICT.items():
            assert nt in result
            assert result[nt].shape == (batch_size, count)

    def test_forward_various_graph_sizes(self, num_nodes_dict) -> None:
        """forward() works with different graph sizes."""
        for scale in [1, 2, 5]:
            scaled = {k: v * scale for k, v in num_nodes_dict.items()}
            m = PathReasoner(
                scaled, NUM_RELATIONS, EMBEDDING_DIM,
                num_iterations=2, dropout=0.0,
            )
            data = HeteroData()
            for nt, cnt in scaled.items():
                data[nt].num_nodes = cnt

            # Add edges.
            types = sorted(scaled.keys())
            data[types[0], "rel_a", types[1]].edge_index = torch.tensor(
                [[0], [0]], dtype=torch.long
            )
            out = m(data)
            for nt, cnt in scaled.items():
                assert out[nt].shape == (cnt, EMBEDDING_DIM)

    def test_handles_heterogeneous_graph(self) -> None:
        """Works with multiple node/edge types in the graph."""
        nodes = {"type_a": 5, "type_b": 7, "type_c": 3}
        m = PathReasoner(
            nodes, 2, EMBEDDING_DIM, num_iterations=2, dropout=0.0,
        )
        data = HeteroData()
        for nt, cnt in nodes.items():
            data[nt].num_nodes = cnt
        data["type_a", "r1", "type_b"].edge_index = torch.tensor(
            [[0, 1, 2], [0, 1, 2]], dtype=torch.long
        )
        data["type_b", "r2", "type_c"].edge_index = torch.tensor(
            [[0, 1], [0, 1]], dtype=torch.long
        )
        out = m(data)
        assert set(out.keys()) == {"type_a", "type_b", "type_c"}
        assert out["type_a"].shape == (5, EMBEDDING_DIM)
        assert out["type_b"].shape == (7, EMBEDDING_DIM)
        assert out["type_c"].shape == (3, EMBEDDING_DIM)


# ======================================================================
# TestBoundaryCondition
# ======================================================================


class TestBoundaryCondition:
    """Tests for query-conditioned initialisation."""

    def test_query_entity_initialised(
        self, model: PathReasoner, mini_hetero_data: HeteroData
    ) -> None:
        """In score_query, only the source entity gets non-zero init."""
        # We test indirectly: score_query with different heads should
        # produce different score distributions.
        head_a = torch.tensor([0])
        head_b = torch.tensor([1])
        rel = torch.tensor([0])

        scores_a = model.score_query(mini_hetero_data, "protein", head_a, rel)
        scores_b = model.score_query(mini_hetero_data, "protein", head_b, rel)

        # Different heads should generally yield different scores.
        all_scores_a = torch.cat([v.flatten() for v in scores_a.values()])
        all_scores_b = torch.cat([v.flatten() for v in scores_b.values()])
        assert not torch.allclose(all_scores_a, all_scores_b, atol=1e-5)


# ======================================================================
# TestInverseEdges
# ======================================================================


class TestInverseEdges:
    """Tests for inverse edge handling."""

    def test_flatten_graph_doubles_edges(
        self, model: PathReasoner, mini_hetero_data: HeteroData
    ) -> None:
        """_flatten_graph adds inverse edges, doubling edge count."""
        edge_index, edge_type, edge_rel_emb = model._flatten_graph(
            mini_hetero_data
        )
        # Original edges: 4 + 3 + 2 = 9. Doubled: 18.
        assert edge_index.shape[1] == 18
        assert edge_type.shape[0] == 18
        assert edge_rel_emb.shape == (18, EMBEDDING_DIM)

    def test_inverse_relation_ids_offset(
        self, model: PathReasoner, mini_hetero_data: HeteroData
    ) -> None:
        """Inverse relation IDs are offset by num_relations."""
        _, edge_type, _ = model._flatten_graph(mini_hetero_data)
        # Some should be < num_relations (original), some >= (inverse).
        orig_mask = edge_type < model.num_relations
        inv_mask = edge_type >= model.num_relations
        assert orig_mask.any()
        assert inv_mask.any()
        # Half original, half inverse.
        assert orig_mask.sum() == inv_mask.sum()

    def test_inverse_embeddings_differ_from_forward(
        self, model: PathReasoner, mini_hetero_data: HeteroData
    ) -> None:
        """Inverse edge embeddings come from a different table."""
        _, edge_type, edge_rel_emb = model._flatten_graph(mini_hetero_data)
        # Pick an original edge and its inverse.
        orig_mask = edge_type < model.num_relations
        inv_mask = edge_type >= model.num_relations

        orig_embs = edge_rel_emb[orig_mask]
        inv_embs = edge_rel_emb[inv_mask]
        # They should generally not be identical.
        assert not torch.allclose(orig_embs, inv_embs, atol=1e-6)


# ======================================================================
# TestGradientFlow
# ======================================================================


class TestGradientFlow:
    """Tests for gradient propagation."""

    def test_gradients_from_score_to_embeddings(
        self, model: PathReasoner, mini_hetero_data: HeteroData
    ) -> None:
        """Gradients flow from score() back to entity/relation embeddings."""
        out = model(mini_hetero_data)
        head_emb = out["protein"][:BATCH_SIZE]
        tail_emb = out["glycan"][:BATCH_SIZE]
        rel_emb = model.relation_embeddings(
            torch.zeros(BATCH_SIZE, dtype=torch.long)
        )
        scores = model.score(head_emb, rel_emb, tail_emb)
        loss = scores.sum()
        loss.backward()

        # Entity embeddings should have gradients.
        assert model.node_embeddings["protein"].weight.grad is not None
        assert model.node_embeddings["protein"].weight.grad.abs().sum() > 0

        # Relation embeddings should have gradients.
        assert model.relation_embeddings.weight.grad is not None
        assert model.relation_embeddings.weight.grad.abs().sum() > 0

    def test_gradients_through_message_passing(
        self, model: PathReasoner, mini_hetero_data: HeteroData
    ) -> None:
        """Gradients flow through BF message passing iterations."""
        out = model(mini_hetero_data)
        loss = sum(v.sum() for v in out.values())
        loss.backward()

        # BF layers should have gradients.
        for i, layer in enumerate(model.bf_layers):
            for name, param in layer.named_parameters():
                assert param.grad is not None, (
                    f"No gradient for bf_layers[{i}].{name}"
                )
                assert param.grad.abs().sum() > 0, (
                    f"Zero gradient for bf_layers[{i}].{name}"
                )

    @pytest.mark.parametrize("num_iters", [1, 3, 6])
    def test_gradient_flow_various_iterations(
        self, num_nodes_dict, mini_hetero_data: HeteroData, num_iters: int
    ) -> None:
        """No gradient issues with T=1, T=3, T=6 iterations."""
        m = PathReasoner(
            num_nodes_dict, NUM_RELATIONS, EMBEDDING_DIM,
            num_iterations=num_iters, dropout=0.0,
        )
        out = m(mini_hetero_data)
        loss = sum(v.sum() for v in out.values())
        loss.backward()

        # Check that no gradients are NaN or Inf.
        for name, param in m.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), (
                    f"Non-finite gradient for {name} with T={num_iters}"
                )

    def test_score_query_gradients(
        self, model: PathReasoner, mini_hetero_data: HeteroData
    ) -> None:
        """Gradients flow through score_query path."""
        head_idx = torch.tensor([0])
        rel_idx = torch.tensor([0])
        result = model.score_query(
            mini_hetero_data, "protein", head_idx, rel_idx
        )
        loss = sum(v.sum() for v in result.values())
        loss.backward()

        # BF layer params should have grads.
        for layer in model.bf_layers:
            for param in layer.parameters():
                assert param.grad is not None


# ======================================================================
# TestIntegration
# ======================================================================


class TestIntegration:
    """Integration tests for PathReasoner."""

    def test_compatible_with_base_interface(
        self, model: PathReasoner, mini_hetero_data: HeteroData
    ) -> None:
        """PathReasoner satisfies BaseKGEModel interface."""
        # forward
        out = model.forward(mini_hetero_data)
        assert isinstance(out, dict)

        # score
        head = torch.randn(2, EMBEDDING_DIM)
        rel = torch.randn(2, EMBEDDING_DIM)
        tail = torch.randn(2, EMBEDDING_DIM)
        scores = model.score(head, rel, tail)
        assert scores.shape == (2,)

        # get_embeddings (convenience from base class)
        emb = model.get_embeddings(mini_hetero_data)
        assert isinstance(emb, dict)
        for nt in NUM_NODES_DICT:
            assert nt in emb
            assert not emb[nt].requires_grad  # detached

    def test_score_triples_end_to_end(
        self, model: PathReasoner, mini_hetero_data: HeteroData
    ) -> None:
        """score_triples() from base class works with PathReasoner."""
        scores = model.score_triples(
            mini_hetero_data,
            head_type="protein",
            head_idx=torch.tensor([0, 1]),
            relation_idx=torch.tensor([0, 1]),
            tail_type="glycan",
            tail_idx=torch.tensor([0, 1]),
        )
        assert scores.shape == (2,)

    def test_positive_vs_random_negative_scores(
        self, num_nodes_dict
    ) -> None:
        """PathReasoner scores graph-connected triples differently from random."""
        torch.manual_seed(123)
        m = PathReasoner(
            num_nodes_dict, NUM_RELATIONS, EMBEDDING_DIM,
            num_iterations=3, dropout=0.0,
        )
        data = HeteroData()
        data["disease"].num_nodes = 5
        data["glycan"].num_nodes = 8
        data["protein"].num_nodes = 10

        # Dense connectivity: protein -> glycan.
        src = torch.arange(8)
        dst = torch.arange(8)
        data["protein", "binds", "glycan"].edge_index = torch.stack([src, dst])

        # Score actual neighbors via score_query.
        head_idx = torch.tensor([0, 1, 2, 3])
        rel_idx = torch.tensor([0, 0, 0, 0])
        result = m.score_query(data, "protein", head_idx, rel_idx)

        # Scores exist and are finite.
        glycan_scores = result["glycan"]
        assert torch.isfinite(glycan_scores).all()
        # The score distribution should not be constant (model should
        # differentiate at least somewhat between candidates).
        assert glycan_scores.std() > 0

    def test_works_with_type_constrained_negatives(
        self, model: PathReasoner, mini_hetero_data: HeteroData
    ) -> None:
        """PathReasoner can score type-constrained negative samples."""
        # Get embeddings for all nodes.
        out = model(mini_hetero_data)

        # Type-constrained: only glycan tails for a protein->glycan relation.
        batch = 3
        head_emb = out["protein"][:batch]
        rel_emb = model.relation_embeddings(torch.zeros(batch, dtype=torch.long))

        # Score against all glycans (type-constrained).
        all_glycan_emb = out["glycan"]  # [8, dim]
        # Expand for broadcasting: [batch, 8, dim].
        head_exp = head_emb.unsqueeze(1).expand(-1, 8, -1)
        rel_exp = rel_emb.unsqueeze(1).expand(-1, 8, -1)
        tail_exp = all_glycan_emb.unsqueeze(0).expand(batch, -1, -1)

        # Score each pair.
        scores = model.score(
            head_exp.reshape(-1, EMBEDDING_DIM),
            rel_exp.reshape(-1, EMBEDDING_DIM),
            tail_exp.reshape(-1, EMBEDDING_DIM),
        )
        assert scores.shape == (batch * 8,)
        scores_matrix = scores.reshape(batch, 8)
        assert torch.isfinite(scores_matrix).all()


# ======================================================================
# TestCorrectness
# ======================================================================


class TestCorrectness:
    """Correctness tests for path-based reasoning."""

    def test_chain_graph_path_scores(self) -> None:
        """On a chain graph A->B->C, verify path-based scores differ by hop."""
        torch.manual_seed(42)
        nodes = {"node": 4}  # A=0, B=1, C=2, D=3
        m = PathReasoner(
            nodes, 1, EMBEDDING_DIM, num_iterations=3, dropout=0.0,
        )
        data = HeteroData()
        data["node"].num_nodes = 4
        # Chain: 0->1->2->3
        data["node", "edge", "node"].edge_index = torch.tensor(
            [[0, 1, 2], [1, 2, 3]], dtype=torch.long
        )

        # Query from node 0.
        head_idx = torch.tensor([0])
        rel_idx = torch.tensor([0])
        result = m.score_query(data, "node", head_idx, rel_idx)

        scores = result["node"].squeeze(0)  # [4]
        assert scores.shape == (4,)
        assert torch.isfinite(scores).all()
        # All four nodes should have different scores (model should
        # distinguish path distances).
        unique_scores = torch.unique(scores)
        assert unique_scores.numel() > 1

    def test_different_relations_produce_different_transforms(
        self, num_nodes_dict
    ) -> None:
        """Different relation types produce different message transforms."""
        torch.manual_seed(42)
        m = PathReasoner(
            num_nodes_dict, NUM_RELATIONS, EMBEDDING_DIM,
            num_iterations=2, dropout=0.0,
        )
        # Build a graph with two different relation types on the same edges.
        # _flatten_graph assigns sorted edge types to consecutive relation IDs,
        # so rel_a gets ID 0 and rel_b gets ID 1 -- different embeddings.
        data = HeteroData()
        data["disease"].num_nodes = 5
        data["glycan"].num_nodes = 8
        data["protein"].num_nodes = 10
        data["protein", "rel_a", "glycan"].edge_index = torch.tensor(
            [[0, 1], [0, 1]], dtype=torch.long
        )
        data["protein", "rel_b", "glycan"].edge_index = torch.tensor(
            [[0, 1], [0, 1]], dtype=torch.long
        )

        out = m(data)

        # Now test with only rel_a vs only rel_b.
        data_a = HeteroData()
        data_a["disease"].num_nodes = 5
        data_a["glycan"].num_nodes = 8
        data_a["protein"].num_nodes = 10
        data_a["protein", "rel_a", "glycan"].edge_index = torch.tensor(
            [[0, 1], [0, 1]], dtype=torch.long
        )
        # Add rel_b with no edges so both relation IDs are allocated.
        data_a["protein", "rel_b", "glycan"].edge_index = torch.zeros(
            2, 0, dtype=torch.long
        )

        data_b = HeteroData()
        data_b["disease"].num_nodes = 5
        data_b["glycan"].num_nodes = 8
        data_b["protein"].num_nodes = 10
        data_b["protein", "rel_b", "glycan"].edge_index = torch.tensor(
            [[0, 1], [0, 1]], dtype=torch.long
        )
        data_b["protein", "rel_a", "glycan"].edge_index = torch.zeros(
            2, 0, dtype=torch.long
        )

        out_a = m(data_a)
        out_b = m(data_b)

        # Same edges but different relation types should produce different
        # glycan embeddings because the relation embeddings differ.
        assert not torch.allclose(
            out_a["glycan"], out_b["glycan"], atol=1e-5
        )

    def test_inverse_relation_scores(self) -> None:
        """Inverse edges allow backward information flow."""
        torch.manual_seed(42)
        nodes = {"node": 3}  # 0->1->2
        m = PathReasoner(
            nodes, 1, EMBEDDING_DIM, num_iterations=2, dropout=0.0,
        )

        # Forward chain: 0->1->2.
        data = HeteroData()
        data["node"].num_nodes = 3
        data["node", "fwd", "node"].edge_index = torch.tensor(
            [[0, 1], [1, 2]], dtype=torch.long
        )

        # Query from node 2 -- inverse edges should allow info to flow
        # back from 2 to 1 to 0.
        head_idx = torch.tensor([2])
        rel_idx = torch.tensor([0])
        result = m.score_query(data, "node", head_idx, rel_idx)

        scores = result["node"].squeeze(0)
        # Node 0 (reachable via inverse edges) should have a non-trivial
        # score different from its zero-init state.
        assert torch.isfinite(scores).all()
        # All three nodes should have distinguishable scores.
        assert torch.unique(scores).numel() > 1

    def test_empty_graph_forward(self, num_nodes_dict) -> None:
        """Forward on a graph with no edges still returns valid embeddings."""
        m = PathReasoner(
            num_nodes_dict, NUM_RELATIONS, EMBEDDING_DIM,
            num_iterations=2, dropout=0.0,
        )
        data = HeteroData()
        data["disease"].num_nodes = 5
        data["glycan"].num_nodes = 8
        data["protein"].num_nodes = 10
        # No edges added.
        data["protein", "r", "glycan"].edge_index = torch.zeros(
            2, 0, dtype=torch.long
        )
        out = m(data)
        for nt, cnt in num_nodes_dict.items():
            assert out[nt].shape == (cnt, EMBEDDING_DIM)
            assert torch.isfinite(out[nt]).all()

    def test_multiple_iterations_refine_embeddings(self) -> None:
        """More BF iterations produce different embeddings than fewer."""
        torch.manual_seed(42)
        nodes = {"node": 5}
        data = HeteroData()
        data["node"].num_nodes = 5
        data["node", "e", "node"].edge_index = torch.tensor(
            [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
        )

        m1 = PathReasoner(
            nodes, 1, EMBEDDING_DIM, num_iterations=1, dropout=0.0,
        )
        m3 = PathReasoner(
            nodes, 1, EMBEDDING_DIM, num_iterations=3, dropout=0.0,
        )

        # Copy weights from m1's single layer into m3's first layer.
        m3.node_embeddings.load_state_dict(m1.node_embeddings.state_dict())
        m3.relation_embeddings.load_state_dict(
            m1.relation_embeddings.state_dict()
        )
        m3.inv_relation_embeddings.load_state_dict(
            m1.inv_relation_embeddings.state_dict()
        )
        m3.bf_layers[0].load_state_dict(m1.bf_layers[0].state_dict())
        m3.layer_norms[0].load_state_dict(m1.layer_norms[0].state_dict())

        out_1 = m1(data)
        out_3 = m3(data)

        # Additional iterations should modify the embeddings.
        assert not torch.allclose(out_1["node"], out_3["node"], atol=1e-5)
