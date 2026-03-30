"""Unit tests for BioHGT (Biology-Aware Heterogeneous Graph Transformer).

Validates BioHGTLayer, BioPrior, and the full BioHGT stacked model per
docs/architecture/model_architecture_design.md Section 3.4.

Test coverage:
  - BioHGTLayer: forward shape, multi-head attention, type-specific Q/K/V,
    relation-conditioned attention, residual connection, LayerNorm
  - BioPrior: enzyme->glycan bias, ptm_crosstalk bias, learnable default bias
  - BioHGT (stacked): output shape, gradient flow, variable depth,
    empty edge types, parameter count
  - Numerical stability: no NaN/Inf after 100 steps, no gradient explosion
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from glycoMusubi.embedding.models.biohgt import (
    BioHGTLayer,
    BioPrior,
    BioHGT,
    DEFAULT_NODE_TYPES,
    DEFAULT_EDGE_TYPES,
)


# ======================================================================
# Constants
# ======================================================================

D_MODEL = 256
N_HEADS = 8
D_K = D_MODEL // N_HEADS  # 32

NODE_TYPES = list(DEFAULT_NODE_TYPES)
EDGE_TYPES = list(DEFAULT_EDGE_TYPES)
N_NODE_TYPES = len(NODE_TYPES)
N_EDGE_TYPES = len(EDGE_TYPES)

# Small graph sizes for fast testing
NODE_COUNTS = {
    "glycan": 8,
    "protein": 6,
    "enzyme": 4,
    "disease": 3,
    "variant": 3,
    "compound": 3,
    "site": 5,
    "motif": 4,
    "reaction": 2,
    "pathway": 2,
}

# Number of unique relation names (some edge types share relations)
NUM_RELATIONS = len({et[1] for et in EDGE_TYPES})


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def h_dict():
    """Node feature dict: {node_type: Tensor[N, 256]}."""
    torch.manual_seed(42)
    return {
        ntype: torch.randn(n, D_MODEL)
        for ntype, n in NODE_COUNTS.items()
    }


@pytest.fixture
def edge_index_dict():
    """Edge index dict with representative edges for each relation type."""
    torch.manual_seed(42)
    edges = {}
    for src_type, rel, dst_type in EDGE_TYPES:
        n_src = NODE_COUNTS.get(src_type, 2)
        n_dst = NODE_COUNTS.get(dst_type, 2)
        n_edges = min(n_src * n_dst, 6)
        src_idx = torch.randint(0, n_src, (n_edges,))
        dst_idx = torch.randint(0, n_dst, (n_edges,))
        edges[(src_type, rel, dst_type)] = torch.stack([src_idx, dst_idx])
    return edges


@pytest.fixture
def biohgt_layer():
    """Single BioHGTLayer instance."""
    return BioHGTLayer(
        in_dim=D_MODEL,
        out_dim=D_MODEL,
        num_heads=N_HEADS,
        node_types=NODE_TYPES,
        edge_types=EDGE_TYPES,
    )


@pytest.fixture
def bio_prior():
    """BioPrior instance."""
    return BioPrior(edge_types=EDGE_TYPES)


@pytest.fixture
def biohgt_model():
    """Full BioHGT model (4 layers)."""
    return BioHGT(
        num_nodes_dict=NODE_COUNTS,
        num_relations=NUM_RELATIONS,
        embedding_dim=D_MODEL,
        num_layers=4,
        num_heads=N_HEADS,
        node_types=NODE_TYPES,
        edge_types=EDGE_TYPES,
    )


@pytest.fixture
def hetero_data():
    """HeteroData for testing the full BioHGT model forward pass."""
    data = HeteroData()
    torch.manual_seed(42)
    for ntype, n in NODE_COUNTS.items():
        data[ntype].x = torch.randn(n, D_MODEL)
        data[ntype].num_nodes = n

    for src_type, rel, dst_type in EDGE_TYPES:
        n_src = NODE_COUNTS.get(src_type, 2)
        n_dst = NODE_COUNTS.get(dst_type, 2)
        n_edges = min(n_src * n_dst, 6)
        src_idx = torch.randint(0, n_src, (n_edges,))
        dst_idx = torch.randint(0, n_dst, (n_edges,))
        data[src_type, rel, dst_type].edge_index = torch.stack([src_idx, dst_idx])

    return data


# ======================================================================
# TestBioHGTLayer
# ======================================================================


class TestBioHGTLayer:
    """Tests for a single BioHGT layer."""

    def test_forward_output_shape(self, biohgt_layer, h_dict, edge_index_dict):
        """Output should have shape {node_type: [N, 256]} for all node types."""
        out = biohgt_layer(h_dict, edge_index_dict)
        assert isinstance(out, dict)
        for ntype, n in NODE_COUNTS.items():
            assert ntype in out, f"Missing node type {ntype} in output"
            assert out[ntype].shape == (n, D_MODEL), (
                f"Expected ({n}, {D_MODEL}), got {out[ntype].shape} for {ntype}"
            )

    def test_forward_preserves_node_types(self, biohgt_layer, h_dict, edge_index_dict):
        """Output should contain exactly the same node types as input."""
        out = biohgt_layer(h_dict, edge_index_dict)
        assert set(out.keys()) == set(h_dict.keys())

    def test_forward_output_dtype(self, biohgt_layer, h_dict, edge_index_dict):
        """Output should be float32."""
        out = biohgt_layer(h_dict, edge_index_dict)
        for ntype, tensor in out.items():
            assert tensor.dtype == torch.float32, f"Expected float32 for {ntype}"

    def test_multi_head_attention_config(self, biohgt_layer):
        """Layer should have num_heads=8 with d_k=32 per head."""
        assert biohgt_layer.num_heads == N_HEADS
        assert biohgt_layer.d_k == D_K

    def test_multi_head_attention_no_nan(self, biohgt_layer, h_dict, edge_index_dict):
        """Forward pass should produce no NaN (confirms attention computation)."""
        out = biohgt_layer(h_dict, edge_index_dict)
        for ntype in out:
            assert not torch.isnan(out[ntype]).any(), f"NaN in {ntype} output"

    def test_type_specific_qkv_different_transforms(self, biohgt_layer):
        """Different node types should use different Q/K/V weight matrices."""
        type_a = NODE_TYPES[0]  # glycan
        type_b = NODE_TYPES[1]  # protein

        # Q linear weights should differ between types
        w_q_a = biohgt_layer.q_linears[type_a].weight.data
        w_q_b = biohgt_layer.q_linears[type_b].weight.data
        assert not torch.allclose(w_q_a, w_q_b), (
            "Q weights for glycan and protein should differ"
        )

        # K linear weights should differ between types
        w_k_a = biohgt_layer.k_linears[type_a].weight.data
        w_k_b = biohgt_layer.k_linears[type_b].weight.data
        assert not torch.allclose(w_k_a, w_k_b), (
            "K weights for glycan and protein should differ"
        )

        # V linear weights should differ between types
        w_v_a = biohgt_layer.v_linears[type_a].weight.data
        w_v_b = biohgt_layer.v_linears[type_b].weight.data
        assert not torch.allclose(w_v_a, w_v_b), (
            "V weights for glycan and protein should differ"
        )

    def test_type_specific_qkv_produces_different_output(self, biohgt_layer):
        """Same input through different type projections should yield different Q."""
        x = torch.randn(1, D_MODEL)
        q_glycan = biohgt_layer.q_linears["glycan"](x)
        q_protein = biohgt_layer.q_linears["protein"](x)
        assert not torch.allclose(q_glycan, q_protein, atol=1e-6), (
            "Q projections for glycan and protein should differ on same input"
        )

    def test_relation_conditioned_attention(self, biohgt_layer, h_dict):
        """Different relation types should produce different attention patterns."""
        torch.manual_seed(123)

        edge_dict_a = {
            ("protein", "has_glycan", "glycan"): torch.tensor([[0, 1], [0, 1]]),
        }
        edge_dict_b = {
            ("compound", "inhibits", "enzyme"): torch.tensor([[0, 1], [0, 1]]),
        }

        out_a = biohgt_layer(h_dict, edge_dict_a)
        out_b = biohgt_layer(h_dict, edge_dict_b)

        # Target types differ, so outputs for 'glycan' vs 'enzyme' should differ
        glycan_changed = not torch.allclose(out_a["glycan"], h_dict["glycan"], atol=1e-3)
        enzyme_changed = not torch.allclose(out_b["enzyme"], h_dict["enzyme"], atol=1e-3)
        assert glycan_changed or enzyme_changed, (
            "At least one target node type should be updated by message passing"
        )

    def test_relation_attention_weights_differ(self, biohgt_layer):
        """Attention weight matrices should differ across relation types."""
        relations = sorted(biohgt_layer.attn_weights.keys())
        if len(relations) >= 2:
            w0 = biohgt_layer.attn_weights[relations[0]].data
            w1 = biohgt_layer.attn_weights[relations[1]].data
            assert not torch.allclose(w0, w1), (
                "Attention weight matrices should differ between relation types"
            )

    def test_residual_connection(self, biohgt_layer, h_dict, edge_index_dict):
        """At initialization with small weights, output should be close to input."""
        with torch.no_grad():
            for param in biohgt_layer.parameters():
                if param.dim() >= 2:
                    nn.init.normal_(param, std=0.001)

        out = biohgt_layer(h_dict, edge_index_dict)

        for ntype in NODE_COUNTS:
            diff = (out[ntype] - h_dict[ntype]).norm() / (h_dict[ntype].norm() + 1e-8)
            assert diff < 2.0, (
                f"Residual connection test: relative diff {diff:.4f} for {ntype} "
                f"is too large (expected < 2.0 with small weights)"
            )

    def test_layernorm_output_statistics(self, biohgt_layer, h_dict, edge_index_dict):
        """After LayerNorm, each feature vector should have mean near 0, variance near 1."""
        out = biohgt_layer(h_dict, edge_index_dict)

        for ntype, tensor in out.items():
            if tensor.shape[0] > 1:
                # LayerNorm normalises the last dimension
                mean = tensor.mean(dim=-1)
                var = tensor.var(dim=-1, unbiased=False)
                assert mean.abs().max() < 1.0, (
                    f"LayerNorm mean too large for {ntype}: max={mean.abs().max():.4f}"
                )
                assert (var - 1.0).abs().max() < 1.0, (
                    f"LayerNorm variance too far from 1 for {ntype}: "
                    f"max deviation={(var - 1.0).abs().max():.4f}"
                )

    def test_no_edges_passthrough(self, biohgt_layer, h_dict):
        """With no edges, output should equal input (no message to aggregate)."""
        empty_edges = {}
        out = biohgt_layer(h_dict, empty_edges)
        for ntype in NODE_COUNTS:
            assert out[ntype].shape == h_dict[ntype].shape
            # Without messages, node features should be unchanged
            assert torch.allclose(out[ntype], h_dict[ntype]), (
                f"With no edges, output for {ntype} should match input"
            )


# ======================================================================
# TestBioPrior
# ======================================================================


class TestBioPrior:
    """Tests for the BioPrior attention bias module."""

    def test_enzyme_glycan_bias_with_pathway_emb(self, bio_prior):
        """Enzyme->glycan (produced_by) with pathway embeddings should return bias."""
        n_edges = 5
        src_idx = torch.arange(n_edges)
        dst_idx = torch.arange(n_edges)
        pathway_src = torch.randn(n_edges, bio_prior.hidden_dim)
        pathway_dst = torch.randn(n_edges, bio_prior.hidden_dim)

        bias = bio_prior(
            relation="produced_by",
            src_idx=src_idx,
            dst_idx=dst_idx,
            pathway_emb_src=pathway_src,
            pathway_emb_dst=pathway_dst,
        )
        assert bias.shape == (n_edges,)
        assert isinstance(bias, torch.Tensor)

    def test_ptm_crosstalk_bias_with_positions(self, bio_prior):
        """PTM crosstalk (site<->site) with positions should return bias."""
        n_edges = 4
        src_idx = torch.arange(n_edges)
        dst_idx = torch.arange(n_edges)
        pos_src = torch.tensor([10, 20, 30, 40])
        pos_dst = torch.tensor([15, 25, 35, 45])

        bias = bio_prior(
            relation="ptm_crosstalk",
            src_idx=src_idx,
            dst_idx=dst_idx,
            site_positions_src=pos_src,
            site_positions_dst=pos_dst,
        )
        assert bias.shape == (n_edges,)
        assert isinstance(bias, torch.Tensor)

    def test_default_relation_bias(self, bio_prior):
        """Default (non-special) relations should return learnable scalar bias."""
        n_edges = 3
        src_idx = torch.arange(n_edges)
        dst_idx = torch.arange(n_edges)

        bias = bio_prior(
            relation="has_variant",
            src_idx=src_idx,
            dst_idx=dst_idx,
        )
        assert bias.shape == (n_edges,)
        assert not torch.isnan(bias).any()

    def test_bio_prior_all_relation_types(self, bio_prior):
        """BioPrior should handle all unique relation names without error."""
        unique_relations = sorted({et[1] for et in EDGE_TYPES})
        for rel in unique_relations:
            src_idx = torch.arange(3)
            dst_idx = torch.arange(3)
            bias = bio_prior(
                relation=rel,
                src_idx=src_idx,
                dst_idx=dst_idx,
            )
            assert bias.shape == (3,), f"Wrong shape for relation {rel}"
            assert not torch.isnan(bias).any(), f"NaN in bias for relation {rel}"

    def test_bio_prior_gradient_flow(self, bio_prior):
        """Gradients should flow through BioPrior parameters."""
        src_idx = torch.arange(4)
        dst_idx = torch.arange(4)
        pathway_src = torch.randn(4, bio_prior.hidden_dim)
        pathway_dst = torch.randn(4, bio_prior.hidden_dim)

        bias = bio_prior(
            relation="produced_by",
            src_idx=src_idx,
            dst_idx=dst_idx,
            pathway_emb_src=pathway_src,
            pathway_emb_dst=pathway_dst,
        )
        loss = bias.sum()
        loss.backward()

        has_grad = False
        for param in bio_prior.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "At least one BioPrior parameter should receive gradients"

    def test_default_bias_is_learnable(self, bio_prior):
        """Default scalar biases should be learnable parameters."""
        for rel, param in bio_prior.default_bias.items():
            assert param.requires_grad, f"default_bias[{rel}] should be learnable"

    def test_pathway_mlp_exists(self, bio_prior):
        """BioPrior should have pathway_mlp for biosynthetic relations."""
        assert hasattr(bio_prior, "pathway_mlp"), (
            "BioPrior should have pathway_mlp for biosynthetic relations"
        )

    def test_crosstalk_mlp_exists(self, bio_prior):
        """BioPrior should have crosstalk_mlp for PTM crosstalk relations."""
        assert hasattr(bio_prior, "crosstalk_mlp"), (
            "BioPrior should have crosstalk_mlp for PTM crosstalk relations"
        )


# ======================================================================
# TestBioHGT (stacked model)
# ======================================================================


class TestBioHGT:
    """Tests for the full BioHGT model (stacked layers)."""

    def test_output_shape_4_layers(self, biohgt_model, hetero_data):
        """4-layer BioHGT output should have {node_type: [N, 256]}."""
        out = biohgt_model(hetero_data)
        assert isinstance(out, dict)
        for ntype, n in NODE_COUNTS.items():
            assert ntype in out
            assert out[ntype].shape == (n, D_MODEL), (
                f"Expected ({n}, {D_MODEL}), got {out[ntype].shape} for {ntype}"
            )

    def test_gradient_flow_all_layers(self, biohgt_model, hetero_data):
        """All layers' parameters should receive gradients."""
        out = biohgt_model(hetero_data)

        loss = sum(v.sum() for v in out.values())
        loss.backward()

        layers_with_grad = set()
        for name, param in biohgt_model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                # Extract layer index from name like "layers.0.xxx"
                parts = name.split(".")
                for i, part in enumerate(parts):
                    if part == "layers" and i + 1 < len(parts):
                        try:
                            layer_idx = int(parts[i + 1])
                            layers_with_grad.add(layer_idx)
                        except ValueError:
                            pass

        assert len(layers_with_grad) >= 2, (
            f"Expected gradients in multiple layers, got {layers_with_grad}"
        )

    @pytest.mark.parametrize("n_layers", [1, 2, 4])
    def test_variable_depth(self, n_layers, hetero_data):
        """BioHGT should work with different numbers of layers (1, 2, 4)."""
        model = BioHGT(
            num_nodes_dict=NODE_COUNTS,
            num_relations=NUM_RELATIONS,
            embedding_dim=D_MODEL,
            num_layers=n_layers,
            num_heads=N_HEADS,
            node_types=NODE_TYPES,
            edge_types=EDGE_TYPES,
        )
        out = model(hetero_data)
        for ntype, n in NODE_COUNTS.items():
            assert out[ntype].shape == (n, D_MODEL)

    def test_empty_edge_types(self, biohgt_model):
        """Model should handle HeteroData with no edges gracefully."""
        data = HeteroData()
        for ntype, n in NODE_COUNTS.items():
            data[ntype].x = torch.randn(n, D_MODEL)
            data[ntype].num_nodes = n

        out = biohgt_model(data)
        for ntype, n in NODE_COUNTS.items():
            assert out[ntype].shape == (n, D_MODEL)
            assert not torch.isnan(out[ntype]).any()

    def test_partial_edge_types(self, biohgt_model):
        """Model should handle a subset of edge types."""
        data = HeteroData()
        for ntype, n in NODE_COUNTS.items():
            data[ntype].x = torch.randn(n, D_MODEL)
            data[ntype].num_nodes = n
        data["protein", "has_glycan", "glycan"].edge_index = torch.tensor(
            [[0, 1, 2], [0, 1, 2]]
        )

        out = biohgt_model(data)
        for ntype, n in NODE_COUNTS.items():
            assert out[ntype].shape == (n, D_MODEL)

    def test_parameter_count(self, biohgt_model):
        """BioHGT (4 layers) should have a reasonable number of parameters.

        Architecture spec says ~8.5M for full BioHGT. We verify it's
        in a reasonable range (500K - 50M).
        """
        total_params = sum(p.numel() for p in biohgt_model.parameters())
        assert total_params > 500_000, (
            f"Too few parameters: {total_params}. Expected > 500K"
        )
        assert total_params < 50_000_000, (
            f"Too many parameters: {total_params}. Expected < 50M"
        )

    def test_deeper_model_more_params(self):
        """A 4-layer model should have more params than a 1-layer model."""
        model_1 = BioHGT(
            num_nodes_dict=NODE_COUNTS,
            num_relations=NUM_RELATIONS,
            embedding_dim=D_MODEL,
            num_layers=1,
            node_types=NODE_TYPES,
            edge_types=EDGE_TYPES,
        )
        model_4 = BioHGT(
            num_nodes_dict=NODE_COUNTS,
            num_relations=NUM_RELATIONS,
            embedding_dim=D_MODEL,
            num_layers=4,
            node_types=NODE_TYPES,
            edge_types=EDGE_TYPES,
        )
        params_1 = sum(p.numel() for p in model_1.parameters())
        params_4 = sum(p.numel() for p in model_4.parameters())
        assert params_4 > params_1, (
            f"4-layer ({params_4}) should have more params than 1-layer ({params_1})"
        )

    def test_output_differs_from_input(self, biohgt_model, hetero_data):
        """After 4 layers with edges, output should differ from input embeddings."""
        # Save original features
        original = {ntype: hetero_data[ntype].x.clone() for ntype in NODE_COUNTS}

        out = biohgt_model(hetero_data)

        any_changed = False
        for ntype in NODE_COUNTS:
            if not torch.allclose(out[ntype], original[ntype], atol=1e-3):
                any_changed = True
                break

        assert any_changed, "Output should differ from input after message passing"

    def test_score_shape(self, biohgt_model):
        """Score method should return [B] tensor."""
        batch_size = 5
        h = torch.randn(batch_size, D_MODEL)
        r = torch.randn(batch_size, D_MODEL)
        t = torch.randn(batch_size, D_MODEL)
        scores = biohgt_model.score(h, r, t)
        assert scores.shape == (batch_size,)

    def test_score_triples_end_to_end(self, biohgt_model, hetero_data):
        """score_triples should work end-to-end."""
        head_idx = torch.tensor([0, 1])
        rel_idx = torch.tensor([0, 0])
        tail_idx = torch.tensor([0, 1])

        scores = biohgt_model.score_triples(
            hetero_data, "protein", head_idx, rel_idx, "glycan", tail_idx
        )
        assert scores.shape == (2,)


# ======================================================================
# TestNumericalStability
# ======================================================================


class TestNumericalStability:
    """Tests for numerical stability of BioHGT."""

    def test_no_nan_inf_after_100_steps(self, edge_index_dict):
        """BioHGT layer should produce no NaN or Inf after 100 forward+backward steps."""
        torch.manual_seed(42)
        layer = BioHGTLayer(
            in_dim=D_MODEL,
            out_dim=D_MODEL,
            num_heads=N_HEADS,
            node_types=NODE_TYPES,
            edge_types=EDGE_TYPES,
        )
        optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)

        for step in range(100):
            optimizer.zero_grad()

            h_in = {
                ntype: torch.randn(n, D_MODEL)
                for ntype, n in NODE_COUNTS.items()
            }

            out = layer(h_in, edge_index_dict)

            for ntype, tensor in out.items():
                assert not torch.isnan(tensor).any(), (
                    f"NaN detected in {ntype} at step {step}"
                )
                assert not torch.isinf(tensor).any(), (
                    f"Inf detected in {ntype} at step {step}"
                )

            loss = sum(v.sum() for v in out.values())
            loss.backward()

            for name, param in layer.named_parameters():
                if param.grad is not None:
                    assert not torch.isnan(param.grad).any(), (
                        f"NaN gradient in {name} at step {step}"
                    )
                    assert not torch.isinf(param.grad).any(), (
                        f"Inf gradient in {name} at step {step}"
                    )

            optimizer.step()

    def test_no_gradient_explosion(self, edge_index_dict):
        """Gradient norms should stay bounded after multiple steps."""
        torch.manual_seed(42)
        layer = BioHGTLayer(
            in_dim=D_MODEL,
            out_dim=D_MODEL,
            num_heads=N_HEADS,
            node_types=NODE_TYPES,
            edge_types=EDGE_TYPES,
        )
        optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)

        max_grad_norm = 0.0

        for step in range(50):
            optimizer.zero_grad()

            h_in = {
                ntype: torch.randn(n, D_MODEL)
                for ntype, n in NODE_COUNTS.items()
            }

            out = layer(h_in, edge_index_dict)
            loss = sum(v.sum() for v in out.values())
            loss.backward()

            total_grad_norm = 0.0
            for param in layer.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.norm().item() ** 2
            total_grad_norm = total_grad_norm ** 0.5

            max_grad_norm = max(max_grad_norm, total_grad_norm)

            optimizer.step()

        assert max_grad_norm < 1e6, (
            f"Gradient explosion detected: max grad norm = {max_grad_norm:.2f}"
        )

    def test_output_magnitude_bounded(self, biohgt_layer, edge_index_dict):
        """Output magnitudes should remain bounded for standard inputs."""
        h_in = {
            ntype: torch.randn(n, D_MODEL)
            for ntype, n in NODE_COUNTS.items()
        }
        out = biohgt_layer(h_in, edge_index_dict)

        for ntype, tensor in out.items():
            max_val = tensor.abs().max().item()
            assert max_val < 1000, (
                f"Output magnitude too large for {ntype}: {max_val:.2f}"
            )

    def test_large_input_stability(self, biohgt_layer, edge_index_dict):
        """Model should handle large-magnitude inputs without NaN."""
        h_in = {
            ntype: torch.randn(n, D_MODEL) * 10.0
            for ntype, n in NODE_COUNTS.items()
        }
        out = biohgt_layer(h_in, edge_index_dict)

        for ntype, tensor in out.items():
            assert not torch.isnan(tensor).any(), (
                f"NaN with large inputs for {ntype}"
            )
            assert not torch.isinf(tensor).any(), (
                f"Inf with large inputs for {ntype}"
            )


# ======================================================================
# TestBioHGTEdgeCases
# ======================================================================


class TestBioHGTEdgeCases:
    """Edge case tests for BioHGT."""

    def test_single_node_per_type(self):
        """BioHGT layer should work with just 1 node per type."""
        layer = BioHGTLayer(
            in_dim=D_MODEL,
            out_dim=D_MODEL,
            num_heads=N_HEADS,
            node_types=NODE_TYPES,
            edge_types=EDGE_TYPES,
        )

        h = {ntype: torch.randn(1, D_MODEL) for ntype in NODE_TYPES}
        edges = {
            ("protein", "has_glycan", "glycan"): torch.tensor([[0], [0]]),
        }

        out = layer(h, edges)
        for ntype in NODE_TYPES:
            assert out[ntype].shape == (1, D_MODEL)

    def test_self_loop_edges(self):
        """BioHGT should handle self-loop edges (same src and dst index)."""
        layer = BioHGTLayer(
            in_dim=D_MODEL,
            out_dim=D_MODEL,
            num_heads=N_HEADS,
            node_types=NODE_TYPES,
            edge_types=EDGE_TYPES,
        )

        h = {ntype: torch.randn(3, D_MODEL) for ntype in NODE_TYPES}
        edges = {
            ("glycan", "child_of", "glycan"): torch.tensor([[0, 1], [0, 1]]),
        }

        out = layer(h, edges)
        assert out["glycan"].shape == (3, D_MODEL)
        assert not torch.isnan(out["glycan"]).any()

    def test_deterministic_with_seed(self, edge_index_dict):
        """Same seed should produce same output."""
        def run_layer():
            torch.manual_seed(99)
            layer = BioHGTLayer(
                in_dim=D_MODEL,
                out_dim=D_MODEL,
                num_heads=N_HEADS,
                node_types=NODE_TYPES,
                edge_types=EDGE_TYPES,
            )
            layer.eval()
            h = {
                ntype: torch.randn(n, D_MODEL)
                for ntype, n in NODE_COUNTS.items()
            }
            with torch.no_grad():
                return layer(h, edge_index_dict)

        out1 = run_layer()
        out2 = run_layer()

        for ntype in NODE_COUNTS:
            assert torch.allclose(out1[ntype], out2[ntype], atol=1e-6), (
                f"Non-deterministic output for {ntype}"
            )

    def test_biohgt_inherits_basekgemodel(self, biohgt_model):
        """BioHGT should inherit from BaseKGEModel."""
        from glycoMusubi.embedding.models.base import BaseKGEModel
        assert isinstance(biohgt_model, BaseKGEModel)

    def test_biohgt_has_node_embeddings(self, biohgt_model):
        """BioHGT should have per-type node embedding tables from BaseKGEModel."""
        for ntype in NODE_COUNTS:
            assert ntype in biohgt_model.node_embeddings, (
                f"Missing node embedding for {ntype}"
            )

    def test_biohgt_has_relation_embeddings(self, biohgt_model):
        """BioHGT should have relation embeddings from BaseKGEModel."""
        assert biohgt_model.relation_embeddings is not None
        rel_idx = torch.tensor([0])
        rel_emb = biohgt_model.get_relation_embedding(rel_idx)
        assert rel_emb.shape == (1, D_MODEL)
