"""Unit tests for GraphLevelDecoder.

Tests cover:
  - Instantiation
  - AttentiveReadout produces correct graph-level embedding shape
  - Gate values in [0, 1] (sigmoid output)
  - Gradient flow end-to-end
  - Single-graph (batch=None) handling
  - Batched input handling
  - Parameter count check (~0.2M with default dims)
  - Output is finite
"""

from __future__ import annotations

import pytest
import torch

from glycoMusubi.embedding.decoders.graph_level_decoder import GraphLevelDecoder


# ======================================================================
# Constants
# ======================================================================

EMBEDDING_DIM = 256
HIDDEN_DIM = 128
NUM_CLASSES = 10
NUM_NODES = 30


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def decoder() -> GraphLevelDecoder:
    """GraphLevelDecoder with standard dimensions."""
    return GraphLevelDecoder(
        embed_dim=EMBEDDING_DIM,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
    )


@pytest.fixture()
def node_embeddings() -> torch.Tensor:
    """Random node embeddings [NUM_NODES, EMBEDDING_DIM]."""
    torch.manual_seed(42)
    return torch.randn(NUM_NODES, EMBEDDING_DIM)


# ======================================================================
# TestInstantiation
# ======================================================================


class TestInstantiation:
    """Tests for GraphLevelDecoder construction."""

    def test_has_gate_linear(self, decoder: GraphLevelDecoder) -> None:
        """Decoder has gate_linear layer."""
        assert hasattr(decoder, "gate_linear")
        assert decoder.gate_linear.in_features == EMBEDDING_DIM
        assert decoder.gate_linear.out_features == 1

    def test_has_transform_linear(self, decoder: GraphLevelDecoder) -> None:
        """Decoder has transform_linear layer."""
        assert hasattr(decoder, "transform_linear")
        assert decoder.transform_linear.in_features == EMBEDDING_DIM
        assert decoder.transform_linear.out_features == EMBEDDING_DIM

    def test_has_predictor(self, decoder: GraphLevelDecoder) -> None:
        """Decoder has predictor MLP."""
        assert hasattr(decoder, "predictor")


# ======================================================================
# TestOutputShape
# ======================================================================


class TestOutputShape:
    """Tests for forward output shapes."""

    def test_single_graph_output_shape(
        self, decoder: GraphLevelDecoder, node_embeddings: torch.Tensor,
    ) -> None:
        """Single graph (batch=None) produces (1, num_classes) output."""
        output = decoder(node_embeddings)
        assert output.shape == (1, NUM_CLASSES)

    def test_batched_output_shape(
        self, decoder: GraphLevelDecoder,
    ) -> None:
        """Batched input produces (batch_size, num_classes) output."""
        num_graphs = 4
        # Assign nodes to 4 graphs
        batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3])
        node_emb = torch.randn(12, EMBEDDING_DIM)

        output = decoder(node_emb, batch=batch)
        assert output.shape == (num_graphs, NUM_CLASSES)

    def test_single_node_graph(self, decoder: GraphLevelDecoder) -> None:
        """Works with a single-node graph."""
        single_node = torch.randn(1, EMBEDDING_DIM)
        output = decoder(single_node)
        assert output.shape == (1, NUM_CLASSES)

    def test_output_dtype(
        self, decoder: GraphLevelDecoder, node_embeddings: torch.Tensor,
    ) -> None:
        """Output is float32."""
        output = decoder(node_embeddings)
        assert output.dtype == torch.float32

    def test_output_finite(
        self, decoder: GraphLevelDecoder, node_embeddings: torch.Tensor,
    ) -> None:
        """Output values are finite."""
        output = decoder(node_embeddings)
        assert torch.isfinite(output).all()


# ======================================================================
# TestAttentiveReadout
# ======================================================================


class TestAttentiveReadout:
    """Tests for the attentive readout gating mechanism."""

    def test_gate_values_in_zero_one(self, decoder: GraphLevelDecoder) -> None:
        """Gate values are in [0, 1] (sigmoid output)."""
        node_emb = torch.randn(NUM_NODES, EMBEDDING_DIM)
        gate = torch.sigmoid(decoder.gate_linear(node_emb))
        assert (gate >= 0).all()
        assert (gate <= 1).all()

    def test_gate_shape(self, decoder: GraphLevelDecoder) -> None:
        """Gate produces (num_nodes, 1) tensor."""
        node_emb = torch.randn(NUM_NODES, EMBEDDING_DIM)
        gate = torch.sigmoid(decoder.gate_linear(node_emb))
        assert gate.shape == (NUM_NODES, 1)

    def test_different_nodes_get_different_gates(
        self, decoder: GraphLevelDecoder,
    ) -> None:
        """Different node embeddings produce different gate values."""
        torch.manual_seed(42)
        node_emb = torch.randn(10, EMBEDDING_DIM)
        gate = torch.sigmoid(decoder.gate_linear(node_emb))
        # Not all gate values should be identical
        assert not torch.allclose(gate, gate[0].expand_as(gate), atol=1e-4)


# ======================================================================
# TestGradientFlow
# ======================================================================


class TestGradientFlow:
    """Tests for gradient flow through the decoder."""

    def test_gradients_flow_to_input(self, decoder: GraphLevelDecoder) -> None:
        """Gradients flow back to input node embeddings."""
        node_emb = torch.randn(NUM_NODES, EMBEDDING_DIM, requires_grad=True)
        output = decoder(node_emb)
        loss = output.sum()
        loss.backward()
        assert node_emb.grad is not None

    def test_gradients_flow_to_all_params(self, decoder: GraphLevelDecoder) -> None:
        """All parameters receive gradients."""
        node_emb = torch.randn(NUM_NODES, EMBEDDING_DIM)
        output = decoder(node_emb)
        loss = output.sum()
        loss.backward()

        for name, param in decoder.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_gradients_flow_with_batch(self, decoder: GraphLevelDecoder) -> None:
        """Gradients flow correctly in batched mode."""
        node_emb = torch.randn(12, EMBEDDING_DIM, requires_grad=True)
        batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3])

        output = decoder(node_emb, batch=batch)
        loss = output.sum()
        loss.backward()

        assert node_emb.grad is not None
        for name, param in decoder.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


# ======================================================================
# TestBatchedInput
# ======================================================================


class TestBatchedInput:
    """Tests for batched subgraph input handling."""

    def test_two_graphs_independent(self, decoder: GraphLevelDecoder) -> None:
        """Two identical subgraphs produce identical predictions in eval mode."""
        decoder.eval()
        torch.manual_seed(42)
        node_emb_a = torch.randn(5, EMBEDDING_DIM)
        node_emb_b = node_emb_a.clone()

        # Two graphs: nodes 0-4 = graph 0, nodes 5-9 = graph 1
        all_nodes = torch.cat([node_emb_a, node_emb_b], dim=0)
        batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        output = decoder(all_nodes, batch=batch)
        assert output.shape == (2, NUM_CLASSES)
        assert torch.allclose(output[0], output[1], atol=1e-5)

    def test_single_graph_batch_same_as_none(
        self, decoder: GraphLevelDecoder,
    ) -> None:
        """batch=torch.zeros(N) produces same result as batch=None in eval mode."""
        decoder.eval()
        torch.manual_seed(42)
        node_emb = torch.randn(NUM_NODES, EMBEDDING_DIM)

        out_none = decoder(node_emb, batch=None)
        out_batch = decoder(node_emb, batch=torch.zeros(NUM_NODES, dtype=torch.long))

        assert torch.allclose(out_none, out_batch, atol=1e-5)

    def test_varying_graph_sizes(self, decoder: GraphLevelDecoder) -> None:
        """Handles graphs with different numbers of nodes."""
        torch.manual_seed(42)
        # Graph 0: 2 nodes, Graph 1: 5 nodes, Graph 2: 1 node
        batch = torch.tensor([0, 0, 1, 1, 1, 1, 1, 2])
        node_emb = torch.randn(8, EMBEDDING_DIM)

        output = decoder(node_emb, batch=batch)
        assert output.shape == (3, NUM_CLASSES)
        assert torch.isfinite(output).all()


# ======================================================================
# TestParameterCount
# ======================================================================


class TestParameterCount:
    """Tests for parameter count estimates."""

    def test_param_count_in_range(self) -> None:
        """Total params should be around 0.1-0.2M for default dimensions.

        gate_linear:     256*1 + 1         =     257
        transform_linear: 256*256 + 256    =  65,792
        predictor:
          Linear(256, 128): 256*128 + 128  =  32,896
          Linear(128, 10):  128*10 + 10    =   1,290
        Total:                              ~100,235
        """
        decoder = GraphLevelDecoder(
            embed_dim=EMBEDDING_DIM,
            num_classes=NUM_CLASSES,
            hidden_dim=HIDDEN_DIM,
        )
        n_params = sum(p.numel() for p in decoder.parameters())
        assert 80_000 < n_params < 250_000, f"Expected ~0.1-0.2M params, got {n_params:,}"

    def test_larger_classes_more_params(self) -> None:
        """More output classes increases parameter count."""
        small = GraphLevelDecoder(embed_dim=EMBEDDING_DIM, num_classes=5)
        large = GraphLevelDecoder(embed_dim=EMBEDDING_DIM, num_classes=100)
        n_small = sum(p.numel() for p in small.parameters())
        n_large = sum(p.numel() for p in large.parameters())
        assert n_large > n_small


# ======================================================================
# TestNumericalStability
# ======================================================================


class TestNumericalStability:
    """Tests for numerical stability with edge cases."""

    def test_large_embeddings(self, decoder: GraphLevelDecoder) -> None:
        """Handles large-magnitude embeddings without overflow."""
        large_emb = torch.randn(10, EMBEDDING_DIM) * 100
        output = decoder(large_emb)
        assert torch.isfinite(output).all()

    def test_zero_embeddings(self, decoder: GraphLevelDecoder) -> None:
        """Handles zero embeddings."""
        zero_emb = torch.zeros(10, EMBEDDING_DIM)
        output = decoder(zero_emb)
        assert torch.isfinite(output).all()

    def test_many_nodes(self, decoder: GraphLevelDecoder) -> None:
        """Handles a large number of nodes."""
        many_nodes = torch.randn(500, EMBEDDING_DIM)
        output = decoder(many_nodes)
        assert output.shape == (1, NUM_CLASSES)
        assert torch.isfinite(output).all()
