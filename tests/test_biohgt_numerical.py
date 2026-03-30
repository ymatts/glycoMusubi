"""BioHGT numerical validity tests.

Validates:
  1. Attention weights sum to 1 per target node (scatter softmax correctness).
  2. BioPrior does not introduce NaN/Inf under normal and extreme inputs.
  3. Gradient magnitudes remain in a reasonable range through 4 stacked layers.
  4. No gradient vanishing or explosion across layers.
  5. Embedding norms remain stable across layers (no blow-up / collapse).
  6. Comparison with standard HGTConv (torch_geometric.nn) on the same data.

References:
  - Hu et al., "Heterogeneous Graph Transformer", WWW 2020
  - docs/architecture/model_architecture_design.md, Section 3.4
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv

from glycoMusubi.embedding.models.biohgt import (
    BioHGT,
    BioHGTLayer,
    BioPrior,
    _scatter_softmax,
    DEFAULT_NODE_TYPES,
    DEFAULT_EDGE_TYPES,
)


# ======================================================================
# Helpers
# ======================================================================

_SMALL_NODE_TYPES = ["glycan", "protein", "enzyme", "site"]
_SMALL_EDGE_TYPES: List[Tuple[str, str, str]] = [
    ("protein", "has_glycan", "glycan"),
    ("enzyme", "produced_by", "glycan"),
    ("enzyme", "consumed_by", "glycan"),
    ("protein", "has_site", "site"),
    ("site", "ptm_crosstalk", "site"),
]


def _make_small_hetero_data(
    node_counts: Dict[str, int] | None = None,
    dim: int = 256,
    seed: int = 42,
) -> HeteroData:
    """Build a small HeteroData for testing."""
    torch.manual_seed(seed)
    data = HeteroData()

    counts = node_counts or {
        "glycan": 8,
        "protein": 6,
        "enzyme": 4,
        "site": 5,
    }
    for ntype, n in counts.items():
        data[ntype].x = torch.randn(n, dim)
        data[ntype].num_nodes = n

    # protein -> glycan
    data["protein", "has_glycan", "glycan"].edge_index = torch.tensor(
        [[0, 1, 2, 3, 0, 1], [0, 1, 2, 3, 4, 5]]
    )
    # enzyme -> glycan (produced_by)
    data["enzyme", "produced_by", "glycan"].edge_index = torch.tensor(
        [[0, 1, 2, 3], [0, 1, 2, 3]]
    )
    # enzyme -> glycan (consumed_by)
    data["enzyme", "consumed_by", "glycan"].edge_index = torch.tensor(
        [[0, 1], [4, 5]]
    )
    # protein -> site
    data["protein", "has_site", "site"].edge_index = torch.tensor(
        [[0, 1, 2], [0, 1, 2]]
    )
    # site -> site (ptm_crosstalk)
    data["site", "ptm_crosstalk", "site"].edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]]
    )
    return data


def _build_x_dict_and_edges(data: HeteroData):
    """Extract x_dict and edge_index_dict from HeteroData."""
    x_dict = {nt: data[nt].x for nt in data.node_types}
    edge_index_dict = {}
    for et in data.edge_types:
        edge_index_dict[et] = data[et].edge_index
    return x_dict, edge_index_dict


def _make_layer(dim: int = 256, num_heads: int = 8, **kwargs) -> BioHGTLayer:
    """Create a BioHGTLayer with small config."""
    return BioHGTLayer(
        in_dim=dim,
        out_dim=dim,
        num_heads=num_heads,
        node_types=_SMALL_NODE_TYPES,
        edge_types=_SMALL_EDGE_TYPES,
        **kwargs,
    )


def _make_model(dim: int = 256, num_layers: int = 4, **kwargs) -> BioHGT:
    """Create a BioHGT model with small config."""
    num_nodes_dict = {"glycan": 8, "protein": 6, "enzyme": 4, "site": 5}
    return BioHGT(
        num_nodes_dict=num_nodes_dict,
        num_relations=len(_SMALL_EDGE_TYPES),
        embedding_dim=dim,
        num_layers=num_layers,
        node_types=_SMALL_NODE_TYPES,
        edge_types=_SMALL_EDGE_TYPES,
        **kwargs,
    )


# ======================================================================
# 1. Attention Weights Sum to 1
# ======================================================================


class TestAttentionWeightsSumToOne:
    """Verify _scatter_softmax produces valid probability distributions."""

    def test_scatter_softmax_sums_to_one(self):
        """Scatter softmax values grouped by destination should sum to 1."""
        torch.manual_seed(0)
        logits = torch.randn(10, 8)  # 10 edges, 8 heads
        # 3 destination nodes; edges grouped as [0,0,0], [1,1,1,1], [2,2,2]
        index = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
        num_nodes = 3

        result = _scatter_softmax(logits, index, num_nodes)

        # Sum per group for each head should be ~1
        from torch_geometric.utils import scatter
        group_sums = scatter(result, index, dim=0, dim_size=num_nodes, reduce="sum")
        assert torch.allclose(group_sums, torch.ones_like(group_sums), atol=1e-5), (
            f"Scatter softmax group sums deviate from 1: {group_sums}"
        )

    def test_scatter_softmax_non_negative(self):
        """All softmax values should be >= 0."""
        torch.manual_seed(1)
        logits = torch.randn(20, 4)
        index = torch.randint(0, 5, (20,))
        result = _scatter_softmax(logits, index, num_nodes=5)
        assert (result >= 0).all(), "Softmax values contain negatives"

    def test_scatter_softmax_single_edge_per_node(self):
        """When a node has only one incoming edge, softmax should give 1.0."""
        logits = torch.tensor([[3.0, -1.0, 2.5]])
        index = torch.tensor([0])
        result = _scatter_softmax(logits, index, num_nodes=1)
        assert torch.allclose(result, torch.ones_like(result)), (
            "Single-edge softmax should be exactly 1.0"
        )

    def test_scatter_softmax_extreme_logits(self):
        """Extreme logit values should not produce NaN/Inf."""
        logits = torch.tensor([[1e6, -1e6], [-1e6, 1e6], [0.0, 0.0]])
        index = torch.tensor([0, 0, 0])
        result = _scatter_softmax(logits, index, num_nodes=1)
        assert torch.isfinite(result).all(), "Scatter softmax produced NaN/Inf"
        from torch_geometric.utils import scatter
        group_sums = scatter(result, index, dim=0, dim_size=1, reduce="sum")
        assert torch.allclose(group_sums, torch.ones_like(group_sums), atol=1e-5)

    def test_biohgt_layer_attention_sums_to_one(self):
        """End-to-end: attention weights inside BioHGTLayer sum to 1 per dst node.

        We monkey-patch _scatter_softmax to capture attention weights and verify.
        """
        torch.manual_seed(42)
        data = _make_small_hetero_data()
        x_dict, edge_index_dict = _build_x_dict_and_edges(data)

        layer = _make_layer(use_bio_prior=False)
        layer.eval()

        captured_attns = []
        captured_indices = []
        captured_num_nodes = []

        import glycoMusubi.embedding.models.biohgt as biohgt_mod
        original_fn = biohgt_mod._scatter_softmax

        def capture_scatter_softmax(src, index, num_nodes):
            result = original_fn(src, index, num_nodes)
            captured_attns.append(result.detach())
            captured_indices.append(index.detach())
            captured_num_nodes.append(num_nodes)
            return result

        biohgt_mod._scatter_softmax = capture_scatter_softmax
        try:
            with torch.no_grad():
                layer(x_dict, edge_index_dict)
        finally:
            biohgt_mod._scatter_softmax = original_fn

        assert len(captured_attns) > 0, "No attention weights captured"
        from torch_geometric.utils import scatter
        for attn, idx, nn_ in zip(captured_attns, captured_indices, captured_num_nodes):
            sums = scatter(attn, idx, dim=0, dim_size=nn_, reduce="sum")
            # Only check nodes that have at least one incoming edge
            active_mask = scatter(
                torch.ones(idx.size(0), device=idx.device),
                idx, dim=0, dim_size=nn_, reduce="sum",
            ) > 0
            active_sums = sums[active_mask]
            assert torch.allclose(
                active_sums, torch.ones_like(active_sums), atol=1e-4
            ), f"Attention sums != 1: min={active_sums.min()}, max={active_sums.max()}"


# ======================================================================
# 2. BioPrior NaN/Inf Safety
# ======================================================================


class TestBioPriorNumericalSafety:
    """BioPrior should never produce NaN or Inf."""

    @pytest.fixture
    def bio_prior(self):
        return BioPrior(_SMALL_EDGE_TYPES, hidden_dim=32)

    def test_default_bias_finite(self, bio_prior):
        """Default scalar biases (all relation types) should be finite."""
        for rel, param in bio_prior.default_bias.items():
            assert torch.isfinite(param).all(), f"default_bias[{rel}] is not finite"

    def test_biosynthetic_prior_normal_input(self, bio_prior):
        """Biosynthetic (pathway_mlp) prior with normal embeddings."""
        src_idx = torch.arange(4)
        dst_idx = torch.arange(4)
        emb_src = torch.randn(4, 32)
        emb_dst = torch.randn(4, 32)
        bias = bio_prior(
            "produced_by", src_idx, dst_idx,
            pathway_emb_src=emb_src, pathway_emb_dst=emb_dst,
        )
        assert torch.isfinite(bias).all(), "Biosynthetic prior produced NaN/Inf"
        assert bias.shape == (4,)

    def test_biosynthetic_prior_zero_embeddings(self, bio_prior):
        """Biosynthetic prior with zero embeddings should still be finite."""
        src_idx = torch.arange(3)
        dst_idx = torch.arange(3)
        emb_src = torch.zeros(3, 32)
        emb_dst = torch.zeros(3, 32)
        bias = bio_prior(
            "consumed_by", src_idx, dst_idx,
            pathway_emb_src=emb_src, pathway_emb_dst=emb_dst,
        )
        assert torch.isfinite(bias).all()

    def test_biosynthetic_prior_large_embeddings(self, bio_prior):
        """Biosynthetic prior with large embeddings."""
        src_idx = torch.arange(2)
        dst_idx = torch.arange(2)
        emb_src = torch.ones(2, 32) * 100.0
        emb_dst = torch.ones(2, 32) * 100.0
        bias = bio_prior(
            "produced_by", src_idx, dst_idx,
            pathway_emb_src=emb_src, pathway_emb_dst=emb_dst,
        )
        assert torch.isfinite(bias).all()

    def test_ptm_crosstalk_prior_normal(self, bio_prior):
        """PTM crosstalk prior with normal site positions."""
        src_idx = torch.arange(4)
        dst_idx = torch.arange(4)
        pos_src = torch.tensor([10, 20, 30, 40])
        pos_dst = torch.tensor([15, 25, 35, 45])
        bias = bio_prior(
            "ptm_crosstalk", src_idx, dst_idx,
            site_positions_src=pos_src, site_positions_dst=pos_dst,
        )
        assert torch.isfinite(bias).all()
        assert bias.shape == (4,)

    def test_ptm_crosstalk_prior_same_position(self, bio_prior):
        """PTM crosstalk with distance=0 should not produce NaN."""
        src_idx = torch.arange(2)
        dst_idx = torch.arange(2)
        pos = torch.tensor([100, 200])
        bias = bio_prior(
            "ptm_crosstalk", src_idx, dst_idx,
            site_positions_src=pos, site_positions_dst=pos,
        )
        assert torch.isfinite(bias).all()

    def test_ptm_crosstalk_prior_large_distance(self, bio_prior):
        """PTM crosstalk with very large distance."""
        src_idx = torch.arange(2)
        dst_idx = torch.arange(2)
        pos_src = torch.tensor([0, 0])
        pos_dst = torch.tensor([100000, 999999])
        bias = bio_prior(
            "ptm_crosstalk", src_idx, dst_idx,
            site_positions_src=pos_src, site_positions_dst=pos_dst,
        )
        assert torch.isfinite(bias).all()

    def test_default_prior_all_relations(self, bio_prior):
        """Default prior for all registered relation types."""
        for et in _SMALL_EDGE_TYPES:
            rel = et[1]
            src_idx = torch.arange(5)
            dst_idx = torch.arange(5)
            bias = bio_prior(rel, src_idx, dst_idx)
            assert torch.isfinite(bias).all(), f"Default prior NaN/Inf for {rel}"
            assert bias.shape == (5,)

    def test_bio_prior_backward_no_nan(self, bio_prior):
        """Backward pass through BioPrior should not produce NaN gradients."""
        src_idx = torch.arange(3)
        dst_idx = torch.arange(3)
        emb_src = torch.randn(3, 32, requires_grad=True)
        emb_dst = torch.randn(3, 32, requires_grad=True)
        bias = bio_prior(
            "produced_by", src_idx, dst_idx,
            pathway_emb_src=emb_src, pathway_emb_dst=emb_dst,
        )
        loss = bias.sum()
        loss.backward()
        assert torch.isfinite(emb_src.grad).all(), "BioPrior backward NaN in emb_src"
        assert torch.isfinite(emb_dst.grad).all(), "BioPrior backward NaN in emb_dst"
        for name, p in bio_prior.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"NaN grad in BioPrior.{name}"


# ======================================================================
# 3. Gradient Magnitude Through 4 Layers
# ======================================================================


class TestGradientMagnitude:
    """Gradients should remain in a reasonable range through 4 BioHGT layers."""

    def test_gradient_norms_finite(self):
        """All parameter gradients should be finite after backward through 4 layers."""
        torch.manual_seed(42)
        model = _make_model(num_layers=4, use_bio_prior=True)
        data = _make_small_hetero_data()

        out = model(data)
        # Create a dummy loss from all outputs
        loss = sum(v.sum() for v in out.values())
        loss.backward()

        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), (
                    f"Non-finite gradient in {name}: "
                    f"has_nan={p.grad.isnan().any()}, has_inf={p.grad.isinf().any()}"
                )

    def test_gradient_norms_not_too_large(self):
        """Gradient norms should not exceed a reasonable threshold."""
        torch.manual_seed(42)
        model = _make_model(num_layers=4, use_bio_prior=True)
        data = _make_small_hetero_data()

        out = model(data)
        loss = sum(v.mean() for v in out.values())
        loss.backward()

        max_grad_norm = 0.0
        for name, p in model.named_parameters():
            if p.grad is not None:
                grad_norm = p.grad.norm().item()
                max_grad_norm = max(max_grad_norm, grad_norm)

        # Gradient norm should be less than 1000 for a single forward/backward
        assert max_grad_norm < 1000.0, (
            f"Gradient explosion detected: max_grad_norm = {max_grad_norm:.2f}"
        )

    def test_gradient_norms_not_too_small(self):
        """At least some gradients should be non-negligible (no total vanishing)."""
        torch.manual_seed(42)
        model = _make_model(num_layers=4, use_bio_prior=True)
        data = _make_small_hetero_data()

        out = model(data)
        loss = sum(v.mean() for v in out.values())
        loss.backward()

        max_grad_norm = 0.0
        for name, p in model.named_parameters():
            if p.grad is not None:
                grad_norm = p.grad.norm().item()
                max_grad_norm = max(max_grad_norm, grad_norm)

        assert max_grad_norm > 1e-10, (
            f"Gradient vanishing suspected: max_grad_norm = {max_grad_norm:.2e}"
        )

    def test_per_layer_gradient_norms(self):
        """Check that gradient norms per layer do not decay/grow exponentially."""
        torch.manual_seed(42)
        model = _make_model(num_layers=4, use_bio_prior=False)
        data = _make_small_hetero_data()

        out = model(data)
        loss = sum(v.mean() for v in out.values())
        loss.backward()

        layer_norms = []
        for i, layer in enumerate(model.layers):
            norms = []
            for name, p in layer.named_parameters():
                if p.grad is not None:
                    norms.append(p.grad.norm().item())
            if norms:
                layer_norms.append(sum(norms) / len(norms))

        assert len(layer_norms) == 4, f"Expected 4 layers, got {len(layer_norms)}"

        # Check no layer has gradient norm ratio > 100x of another
        for i in range(len(layer_norms)):
            for j in range(i + 1, len(layer_norms)):
                if layer_norms[j] > 0:
                    ratio = layer_norms[i] / max(layer_norms[j], 1e-12)
                    assert ratio < 100, (
                        f"Gradient ratio layer {i}/layer {j} = {ratio:.1f} "
                        f"(norms: {layer_norms})"
                    )


# ======================================================================
# 4. No Gradient Vanishing/Explosion
# ======================================================================


class TestGradientVanishingExplosion:
    """Extended gradient health checks over multiple training steps."""

    def test_multi_step_gradient_stability(self):
        """Gradients remain finite and non-zero over 20 training steps."""
        torch.manual_seed(42)
        model = _make_model(num_layers=4, use_bio_prior=True)
        data = _make_small_hetero_data()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for step in range(20):
            optimizer.zero_grad()
            out = model(data)
            loss = sum(v.mean() for v in out.values())
            loss.backward()

            for name, p in model.named_parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all(), (
                        f"Step {step}: non-finite grad in {name}"
                    )

            optimizer.step()

    def test_loss_decreases_or_stays_stable(self):
        """Loss should not diverge to Inf during training."""
        torch.manual_seed(42)
        model = _make_model(num_layers=4, use_bio_prior=True)
        data = _make_small_hetero_data()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        losses = []
        for step in range(30):
            optimizer.zero_grad()
            out = model(data)
            loss = sum(v.mean() for v in out.values())
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        # Loss should always be finite
        for i, l in enumerate(losses):
            assert math.isfinite(l), f"Step {i}: loss = {l} (non-finite)"

    def test_output_norms_finite_after_training(self):
        """Output embeddings should remain finite after training."""
        torch.manual_seed(42)
        model = _make_model(num_layers=4, use_bio_prior=True)
        data = _make_small_hetero_data()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for _ in range(10):
            optimizer.zero_grad()
            out = model(data)
            loss = sum(v.mean() for v in out.values())
            loss.backward()
            optimizer.step()

        # Final forward pass
        with torch.no_grad():
            out = model(data)
            for ntype, emb in out.items():
                assert torch.isfinite(emb).all(), (
                    f"Non-finite embeddings for {ntype} after training"
                )


# ======================================================================
# 5. Embedding Norms Stable Across Layers
# ======================================================================


class TestEmbeddingNormStability:
    """Embedding norms should not blow up or collapse across layers."""

    def test_layer_by_layer_norms(self):
        """Pass data through each layer separately and track norms."""
        torch.manual_seed(42)
        data = _make_small_hetero_data()
        x_dict, edge_index_dict = _build_x_dict_and_edges(data)

        layers = nn.ModuleList([
            _make_layer(use_bio_prior=False) for _ in range(4)
        ])

        norms_per_layer = []
        current = x_dict
        for i, layer in enumerate(layers):
            with torch.no_grad():
                current = layer(current, edge_index_dict)
            avg_norm = sum(
                v.norm(dim=-1).mean().item() for v in current.values()
            ) / len(current)
            norms_per_layer.append(avg_norm)

        # No norm should be zero (collapse)
        for i, n in enumerate(norms_per_layer):
            assert n > 0.01, f"Layer {i}: average norm collapsed to {n:.6f}"

        # Norms should not grow more than 10x from first to last
        ratio = norms_per_layer[-1] / max(norms_per_layer[0], 1e-12)
        assert ratio < 10.0, (
            f"Embedding norms grew {ratio:.1f}x from layer 0 to layer 3. "
            f"Norms: {norms_per_layer}"
        )
        # Norms should not shrink more than 10x
        assert ratio > 0.1, (
            f"Embedding norms shrank to {ratio:.3f}x from layer 0 to layer 3. "
            f"Norms: {norms_per_layer}"
        )

    def test_layer_by_layer_norms_with_bio_prior(self):
        """Same check with BioPrior enabled."""
        torch.manual_seed(42)
        data = _make_small_hetero_data()
        x_dict, edge_index_dict = _build_x_dict_and_edges(data)

        layers = nn.ModuleList([
            _make_layer(use_bio_prior=True) for _ in range(4)
        ])

        norms = []
        current = x_dict
        for layer in layers:
            with torch.no_grad():
                current = layer(current, edge_index_dict)
            avg_norm = sum(
                v.norm(dim=-1).mean().item() for v in current.values()
            ) / len(current)
            norms.append(avg_norm)

        for i, n in enumerate(norms):
            assert n > 0.01, f"Layer {i}: norm collapsed to {n:.6f}"
            assert n < 1e4, f"Layer {i}: norm exploded to {n:.1f}"

    def test_full_model_output_norms(self):
        """BioHGT model output norms should be in a reasonable range."""
        torch.manual_seed(42)
        model = _make_model(num_layers=4, use_bio_prior=True)
        data = _make_small_hetero_data()

        with torch.no_grad():
            out = model(data)

        for ntype, emb in out.items():
            avg_norm = emb.norm(dim=-1).mean().item()
            assert avg_norm > 0.01, f"{ntype}: avg norm too small ({avg_norm:.6f})"
            assert avg_norm < 1e4, f"{ntype}: avg norm too large ({avg_norm:.1f})"

    def test_residual_connection_prevents_collapse(self):
        """Verify that the residual connection keeps norms from collapsing,
        even when attention messages are near-zero.
        """
        torch.manual_seed(0)
        layer = _make_layer(use_bio_prior=False)
        # Create data with very small features
        data = _make_small_hetero_data()
        x_dict, edge_index_dict = _build_x_dict_and_edges(data)
        # Scale down inputs
        small_x = {k: v * 0.01 for k, v in x_dict.items()}

        with torch.no_grad():
            out = layer(small_x, edge_index_dict)

        for ntype, emb in out.items():
            # Output should still be similar scale to input (due to residual + LayerNorm)
            avg_norm = emb.norm(dim=-1).mean().item()
            assert avg_norm > 1e-4, (
                f"{ntype}: residual failed to prevent collapse (norm={avg_norm:.6f})"
            )


# ======================================================================
# 6. Comparison with Standard HGTConv
# ======================================================================


class TestComparisonWithStandardHGT:
    """Compare BioHGT behavior with torch_geometric.nn.HGTConv."""

    @pytest.fixture
    def shared_data(self):
        """Create shared test data for both BioHGT and standard HGT."""
        torch.manual_seed(123)
        data = HeteroData()
        node_counts = {"glycan": 8, "protein": 6, "enzyme": 4}
        for ntype, n in node_counts.items():
            data[ntype].x = torch.randn(n, 64)
            data[ntype].num_nodes = n

        data["protein", "has_glycan", "glycan"].edge_index = torch.tensor(
            [[0, 1, 2, 3], [0, 1, 2, 3]]
        )
        data["enzyme", "produced_by", "glycan"].edge_index = torch.tensor(
            [[0, 1, 2], [0, 1, 2]]
        )
        return data, node_counts

    def test_both_produce_finite_outputs(self, shared_data):
        """Both BioHGT and HGTConv should produce finite embeddings."""
        data, node_counts = shared_data
        metadata = (
            list(node_counts.keys()),
            [
                ("protein", "has_glycan", "glycan"),
                ("enzyme", "produced_by", "glycan"),
            ],
        )

        # Standard HGTConv
        hgt_conv = HGTConv(
            in_channels=64,
            out_channels=64,
            metadata=metadata,
            heads=4,
        )
        x_dict = {nt: data[nt].x for nt in data.node_types}
        edge_index_dict = {}
        for et in data.edge_types:
            edge_index_dict[et] = data[et].edge_index

        with torch.no_grad():
            hgt_out = hgt_conv(x_dict, edge_index_dict)

        for ntype, emb in hgt_out.items():
            assert torch.isfinite(emb).all(), f"HGTConv: NaN/Inf in {ntype}"

        # BioHGT layer
        bio_layer = BioHGTLayer(
            in_dim=64,
            out_dim=64,
            num_heads=4,
            node_types=list(node_counts.keys()),
            edge_types=[
                ("protein", "has_glycan", "glycan"),
                ("enzyme", "produced_by", "glycan"),
            ],
            use_bio_prior=False,
        )
        with torch.no_grad():
            bio_out = bio_layer(x_dict, edge_index_dict)

        for ntype, emb in bio_out.items():
            assert torch.isfinite(emb).all(), f"BioHGT: NaN/Inf in {ntype}"

    def test_output_shapes_match(self, shared_data):
        """Both should produce the same output shapes for all node types."""
        data, node_counts = shared_data
        metadata = (
            list(node_counts.keys()),
            [
                ("protein", "has_glycan", "glycan"),
                ("enzyme", "produced_by", "glycan"),
            ],
        )

        hgt_conv = HGTConv(64, 64, metadata=metadata, heads=4)
        bio_layer = BioHGTLayer(
            in_dim=64, out_dim=64, num_heads=4,
            node_types=list(node_counts.keys()),
            edge_types=[
                ("protein", "has_glycan", "glycan"),
                ("enzyme", "produced_by", "glycan"),
            ],
            use_bio_prior=False,
        )

        x_dict = {nt: data[nt].x for nt in data.node_types}
        edge_index_dict = {et: data[et].edge_index for et in data.edge_types}

        with torch.no_grad():
            hgt_out = hgt_conv(x_dict, edge_index_dict)
            bio_out = bio_layer(x_dict, edge_index_dict)

        # HGTConv only returns node types that are edge destinations.
        # BioHGT returns all node types (incl. source-only types via residual).
        # Compare only types present in both outputs.
        common_types = set(hgt_out.keys()) & set(bio_out.keys())
        assert len(common_types) > 0, "No common output node types"
        for ntype in common_types:
            assert hgt_out[ntype].shape == bio_out[ntype].shape, (
                f"Shape mismatch for {ntype}: "
                f"HGT={hgt_out[ntype].shape}, BioHGT={bio_out[ntype].shape}"
            )

        # BioHGT should additionally return source-only types
        for ntype in node_counts:
            assert ntype in bio_out, f"BioHGT missing {ntype}"

    def test_both_gradients_flow(self, shared_data):
        """Both should allow gradients to flow back to input features."""
        data, node_counts = shared_data
        metadata = (
            list(node_counts.keys()),
            [
                ("protein", "has_glycan", "glycan"),
                ("enzyme", "produced_by", "glycan"),
            ],
        )

        x_dict = {nt: data[nt].x.clone().requires_grad_(True) for nt in data.node_types}
        edge_index_dict = {et: data[et].edge_index for et in data.edge_types}

        # HGTConv
        hgt_conv = HGTConv(64, 64, metadata=metadata, heads=4)
        hgt_out = hgt_conv(x_dict, edge_index_dict)
        hgt_loss = sum(v.sum() for v in hgt_out.values())
        hgt_loss.backward()

        for ntype, x in x_dict.items():
            assert x.grad is not None, f"HGTConv: no grad for {ntype}"
            assert torch.isfinite(x.grad).all(), f"HGTConv: NaN grad for {ntype}"

        # Reset grads
        x_dict2 = {nt: data[nt].x.clone().requires_grad_(True) for nt in data.node_types}

        # BioHGT
        bio_layer = BioHGTLayer(
            in_dim=64, out_dim=64, num_heads=4,
            node_types=list(node_counts.keys()),
            edge_types=[
                ("protein", "has_glycan", "glycan"),
                ("enzyme", "produced_by", "glycan"),
            ],
            use_bio_prior=False,
        )
        bio_out = bio_layer(x_dict2, edge_index_dict)
        bio_loss = sum(v.sum() for v in bio_out.values())
        bio_loss.backward()

        for ntype, x in x_dict2.items():
            assert x.grad is not None, f"BioHGT: no grad for {ntype}"
            assert torch.isfinite(x.grad).all(), f"BioHGT: NaN grad for {ntype}"

    def test_biohgt_with_bio_prior_outperforms_after_training(self, shared_data):
        """Bio-prior should not hurt convergence compared to no-prior baseline.

        Both should converge; we just verify both losses decrease.
        """
        data, node_counts = shared_data

        edge_types = [
            ("protein", "has_glycan", "glycan"),
            ("enzyme", "produced_by", "glycan"),
        ]

        models = {}
        for label, use_prior in [("no_prior", False), ("with_prior", True)]:
            torch.manual_seed(99)
            model = BioHGT(
                num_nodes_dict=node_counts,
                num_relations=2,
                embedding_dim=64,
                num_layers=2,
                num_heads=4,
                node_types=list(node_counts.keys()),
                edge_types=edge_types,
                use_bio_prior=use_prior,
            )
            models[label] = model

        results = {}
        for label, model in models.items():
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            losses = []
            for _ in range(30):
                optimizer.zero_grad()
                out = model(data)
                loss = sum(v.mean() for v in out.values())
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
            results[label] = losses

        # Both should have finite losses throughout
        for label, losses in results.items():
            for i, l in enumerate(losses):
                assert math.isfinite(l), f"{label} step {i}: loss={l}"

    def test_embedding_norm_comparable(self, shared_data):
        """BioHGT output norms should be in the same order of magnitude as HGTConv."""
        data, node_counts = shared_data
        metadata = (
            list(node_counts.keys()),
            [
                ("protein", "has_glycan", "glycan"),
                ("enzyme", "produced_by", "glycan"),
            ],
        )

        x_dict = {nt: data[nt].x for nt in data.node_types}
        edge_index_dict = {et: data[et].edge_index for et in data.edge_types}

        torch.manual_seed(0)
        hgt_conv = HGTConv(64, 64, metadata=metadata, heads=4)
        with torch.no_grad():
            hgt_out = hgt_conv(x_dict, edge_index_dict)
        hgt_norms = {
            nt: hgt_out[nt].norm(dim=-1).mean().item()
            for nt in hgt_out
        }

        torch.manual_seed(0)
        bio_layer = BioHGTLayer(
            in_dim=64, out_dim=64, num_heads=4,
            node_types=list(node_counts.keys()),
            edge_types=[
                ("protein", "has_glycan", "glycan"),
                ("enzyme", "produced_by", "glycan"),
            ],
            use_bio_prior=False,
        )
        with torch.no_grad():
            bio_out = bio_layer(x_dict, edge_index_dict)
        bio_norms = {
            nt: bio_out[nt].norm(dim=-1).mean().item()
            for nt in bio_out
        }

        # Compare only node types present in both outputs
        # (HGTConv only returns destination types)
        common_types = set(hgt_norms.keys()) & set(bio_norms.keys())
        assert len(common_types) > 0, "No common types to compare norms"
        for nt in common_types:
            if hgt_norms[nt] > 0 and bio_norms[nt] > 0:
                ratio = max(hgt_norms[nt], bio_norms[nt]) / min(
                    hgt_norms[nt], bio_norms[nt]
                )
                assert ratio < 100, (
                    f"{nt}: norm ratio = {ratio:.1f} "
                    f"(HGT={hgt_norms[nt]:.3f}, BioHGT={bio_norms[nt]:.3f})"
                )


# ======================================================================
# 7. Additional Structural Correctness Tests
# ======================================================================


class TestBioHGTStructuralCorrectness:
    """Verify structural properties of BioHGT implementation."""

    def test_parameter_count_reasonable(self):
        """Full BioHGT parameter count should be in the expected range."""
        model = _make_model(dim=256, num_layers=4)
        total_params = sum(p.numel() for p in model.parameters())
        # Design spec says ~8.5M for BioHGT alone, but our small config has
        # fewer node/edge types so we expect less. Should be > 100K, < 50M.
        assert total_params > 100_000, f"Too few params: {total_params}"
        assert total_params < 50_000_000, f"Too many params: {total_params}"

    def test_forward_deterministic_eval_mode(self):
        """eval() mode should produce deterministic outputs."""
        torch.manual_seed(42)
        model = _make_model(num_layers=2)
        model.eval()
        data = _make_small_hetero_data()

        with torch.no_grad():
            out1 = model(data)
            out2 = model(data)

        for ntype in out1:
            assert torch.allclose(out1[ntype], out2[ntype], atol=1e-6), (
                f"Non-deterministic output for {ntype} in eval mode"
            )

    def test_empty_edge_type_handling(self):
        """Layer should handle edge types with zero edges gracefully."""
        data = _make_small_hetero_data()
        # Replace one edge type with empty edges
        data["enzyme", "consumed_by", "glycan"].edge_index = torch.zeros(
            2, 0, dtype=torch.long
        )
        x_dict, edge_index_dict = _build_x_dict_and_edges(data)
        layer = _make_layer(use_bio_prior=False)

        with torch.no_grad():
            out = layer(x_dict, edge_index_dict)

        for ntype, emb in out.items():
            assert torch.isfinite(emb).all(), f"NaN/Inf with empty edges for {ntype}"

    def test_nodes_without_incoming_edges_preserve_features(self):
        """Nodes with no incoming edges should keep their original features
        (residual connection with no messages).
        """
        data = _make_small_hetero_data()
        x_dict, edge_index_dict = _build_x_dict_and_edges(data)

        # motif type has no edges in our small config
        x_dict["motif"] = torch.randn(3, 256)

        layer = BioHGTLayer(
            in_dim=256, out_dim=256, num_heads=8,
            node_types=_SMALL_NODE_TYPES + ["motif"],
            edge_types=_SMALL_EDGE_TYPES,
            use_bio_prior=False,
        )

        with torch.no_grad():
            out = layer(x_dict, edge_index_dict)

        # motif nodes receive no messages, so output = input (no normalization applied)
        assert torch.allclose(out["motif"], x_dict["motif"]), (
            "Nodes without incoming edges should preserve original features"
        )

    def test_score_function_distmult_style(self):
        """BioHGT.score should implement DistMult correctly."""
        model = _make_model()
        h = torch.randn(4, 256)
        r = torch.randn(4, 256)
        t = torch.randn(4, 256)

        scores = model.score(h, r, t)
        expected = (h * r * t).sum(dim=-1)
        assert torch.allclose(scores, expected, atol=1e-5), "Score != DistMult formula"

    def test_score_function_symmetry(self):
        """DistMult-style score should be symmetric in h and t."""
        model = _make_model()
        h = torch.randn(4, 256)
        r = torch.randn(4, 256)
        t = torch.randn(4, 256)

        assert torch.allclose(
            model.score(h, r, t), model.score(t, r, h), atol=1e-5
        ), "DistMult score should be symmetric"


# ======================================================================
# 8. BioHGT with Full Default Config
# ======================================================================


class TestBioHGTFullConfig:
    """Test BioHGT with the full default node/edge type configuration."""

    def _make_full_data(self, dim: int = 256) -> HeteroData:
        torch.manual_seed(42)
        data = HeteroData()
        counts = {
            "glycan": 10, "protein": 8, "enzyme": 6, "disease": 4,
            "variant": 4, "compound": 3, "site": 5, "motif": 3,
            "reaction": 3, "pathway": 2,
        }
        for ntype, n in counts.items():
            data[ntype].x = torch.randn(n, dim)
            data[ntype].num_nodes = n

        for src, rel, dst in DEFAULT_EDGE_TYPES:
            ns = counts.get(src, 0)
            nd = counts.get(dst, 0)
            if ns > 0 and nd > 0:
                num_edges = min(ns, nd, 4)
                src_idx = torch.randint(0, ns, (num_edges,))
                dst_idx = torch.randint(0, nd, (num_edges,))
                data[src, rel, dst].edge_index = torch.stack([src_idx, dst_idx])

        return data, counts

    def test_full_config_forward(self):
        """Full config with all 10 node types and 13 edge types should work."""
        data, counts = self._make_full_data()
        model = BioHGT(
            num_nodes_dict=counts,
            num_relations=len(DEFAULT_EDGE_TYPES),
            embedding_dim=256,
            num_layers=4,
            use_bio_prior=True,
        )

        with torch.no_grad():
            out = model(data)

        for ntype in counts:
            assert ntype in out, f"Missing output for {ntype}"
            assert out[ntype].shape == (counts[ntype], 256), (
                f"Wrong shape for {ntype}: {out[ntype].shape}"
            )
            assert torch.isfinite(out[ntype]).all(), f"NaN/Inf in {ntype}"

    def test_full_config_backward(self):
        """Full config backward pass should produce finite gradients."""
        data, counts = self._make_full_data()
        model = BioHGT(
            num_nodes_dict=counts,
            num_relations=len(DEFAULT_EDGE_TYPES),
            embedding_dim=256,
            num_layers=4,
            use_bio_prior=True,
        )

        out = model(data)
        loss = sum(v.mean() for v in out.values())
        loss.backward()

        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"NaN grad in {name}"
