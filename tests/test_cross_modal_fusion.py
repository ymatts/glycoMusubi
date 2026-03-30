"""Unit tests for CrossModalFusion gated cross-attention module.

Tests cover:
  - Forward output shape preservation [N, d] -> [N, d]
  - Gate values constrained to [0, 1]
  - mask=None fuses all nodes
  - Partial mask: nodes with modality features are fused, others passthrough
  - Empty mask (all False): pure passthrough
  - All-True mask: all nodes fused
  - Gradient flow through all parameters
  - Numerical stability (no NaN/Inf)
  - Determinism with fixed seed
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from glycoMusubi.embedding.models.cross_modal_fusion import CrossModalFusion


# ======================================================================
# Fixtures
# ======================================================================

EMBED_DIM = 64
NUM_HEADS = 4
DROPOUT = 0.0  # Disable dropout for deterministic tests
N_NODES = 16


@pytest.fixture()
def fusion_module() -> CrossModalFusion:
    """CrossModalFusion with deterministic settings."""
    return CrossModalFusion(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, dropout=DROPOUT)


@pytest.fixture()
def h_kg() -> torch.Tensor:
    """Random KG embeddings [N, d]."""
    torch.manual_seed(42)
    return torch.randn(N_NODES, EMBED_DIM)


@pytest.fixture()
def h_modality() -> torch.Tensor:
    """Random modality-specific embeddings [N, d]."""
    torch.manual_seed(99)
    return torch.randn(N_NODES, EMBED_DIM)


# ======================================================================
# TestCrossModalFusionShape
# ======================================================================


class TestCrossModalFusionShape:
    """Shape-related tests for CrossModalFusion."""

    def test_forward_shape_no_mask(
        self, fusion_module: CrossModalFusion, h_kg: torch.Tensor, h_modality: torch.Tensor
    ) -> None:
        """Output shape matches input shape [N, d] when mask is None."""
        output = fusion_module(h_kg, h_modality, mask=None)
        assert output.shape == (N_NODES, EMBED_DIM)

    def test_forward_shape_with_mask(
        self, fusion_module: CrossModalFusion, h_kg: torch.Tensor, h_modality: torch.Tensor
    ) -> None:
        """Output shape [N, d] preserved with partial mask."""
        mask = torch.zeros(N_NODES, dtype=torch.bool)
        mask[:8] = True  # First 8 nodes have modality features
        output = fusion_module(h_kg, h_modality, mask=mask)
        assert output.shape == (N_NODES, EMBED_DIM)

    def test_forward_shape_single_node(self) -> None:
        """Works with a single node."""
        fusion = CrossModalFusion(embed_dim=32, num_heads=2, dropout=0.0)
        h_kg = torch.randn(1, 32)
        h_mod = torch.randn(1, 32)
        output = fusion(h_kg, h_mod, mask=None)
        assert output.shape == (1, 32)

    def test_forward_shape_large_batch(self) -> None:
        """Works with a large number of nodes."""
        fusion = CrossModalFusion(embed_dim=32, num_heads=2, dropout=0.0)
        h_kg = torch.randn(512, 32)
        h_mod = torch.randn(512, 32)
        output = fusion(h_kg, h_mod, mask=None)
        assert output.shape == (512, 32)


# ======================================================================
# TestCrossModalFusionGate
# ======================================================================


class TestCrossModalFusionGate:
    """Tests for gate mechanism properties."""

    def test_gate_values_in_range(
        self, fusion_module: CrossModalFusion, h_kg: torch.Tensor, h_modality: torch.Tensor
    ) -> None:
        """Gate values must be in [0, 1] due to sigmoid activation."""
        fusion_module.eval()

        # Hook to capture gate values
        gate_values = []

        original_forward = fusion_module.gate_mlp.forward

        def hook_gate(x: torch.Tensor) -> torch.Tensor:
            result = original_forward(x)
            gate_values.append(result.detach())
            return result

        fusion_module.gate_mlp.forward = hook_gate
        try:
            fusion_module(h_kg, h_modality, mask=None)
        finally:
            fusion_module.gate_mlp.forward = original_forward

        assert len(gate_values) > 0
        gate = gate_values[0]
        assert (gate >= 0.0).all(), f"Gate has values below 0: min={gate.min().item()}"
        assert (gate <= 1.0).all(), f"Gate has values above 1: max={gate.max().item()}"


# ======================================================================
# TestCrossModalFusionMask
# ======================================================================


class TestCrossModalFusionMask:
    """Tests for mask handling behavior."""

    def test_mask_none_fuses_all(
        self, fusion_module: CrossModalFusion, h_kg: torch.Tensor, h_modality: torch.Tensor
    ) -> None:
        """When mask=None, all nodes are fused (output differs from input)."""
        fusion_module.eval()
        output = fusion_module(h_kg, h_modality, mask=None)
        # Output should differ from the original KG embeddings
        # (unless the gate is exactly 1.0 for all nodes, which is unlikely)
        assert not torch.allclose(output, h_kg, atol=1e-5), (
            "Fused output should differ from raw KG embeddings"
        )

    def test_empty_mask_passthrough(
        self, fusion_module: CrossModalFusion, h_kg: torch.Tensor, h_modality: torch.Tensor
    ) -> None:
        """When mask is all False, output equals h_kg (passthrough)."""
        mask = torch.zeros(N_NODES, dtype=torch.bool)
        output = fusion_module(h_kg, h_modality, mask=mask)
        assert torch.allclose(output, h_kg), (
            "All-False mask should produce passthrough (output == h_kg)"
        )

    def test_partial_mask_passthrough_for_unmasked(
        self, fusion_module: CrossModalFusion, h_kg: torch.Tensor, h_modality: torch.Tensor
    ) -> None:
        """Nodes where mask=False should pass through unchanged."""
        fusion_module.eval()
        mask = torch.zeros(N_NODES, dtype=torch.bool)
        mask[:4] = True  # Only first 4 nodes have modality features

        output = fusion_module(h_kg, h_modality, mask=mask)

        # Unmasked nodes (4:) should be unchanged
        assert torch.allclose(output[4:], h_kg[4:]), (
            "Nodes without modality features should pass through unchanged"
        )

    def test_partial_mask_fused_nodes_differ(
        self, fusion_module: CrossModalFusion, h_kg: torch.Tensor, h_modality: torch.Tensor
    ) -> None:
        """Nodes where mask=True should be fused (differ from input)."""
        fusion_module.eval()
        mask = torch.zeros(N_NODES, dtype=torch.bool)
        mask[:4] = True

        output = fusion_module(h_kg, h_modality, mask=mask)

        # Fused nodes should differ (with high probability)
        assert not torch.allclose(output[:4], h_kg[:4], atol=1e-5), (
            "Fused nodes should differ from raw KG embeddings"
        )

    def test_all_true_mask_equals_no_mask(
        self, fusion_module: CrossModalFusion, h_kg: torch.Tensor, h_modality: torch.Tensor
    ) -> None:
        """All-True mask should give the same result as mask=None."""
        fusion_module.eval()
        mask_all = torch.ones(N_NODES, dtype=torch.bool)

        output_masked = fusion_module(h_kg, h_modality, mask=mask_all)
        output_no_mask = fusion_module(h_kg, h_modality, mask=None)

        assert torch.allclose(output_masked, output_no_mask, atol=1e-6), (
            "All-True mask should produce the same result as mask=None"
        )

    def test_full_modality_all_nodes_fused(
        self, fusion_module: CrossModalFusion, h_kg: torch.Tensor, h_modality: torch.Tensor
    ) -> None:
        """When all nodes have modality features (mask=all True), all are fused."""
        fusion_module.eval()
        mask = torch.ones(N_NODES, dtype=torch.bool)
        output = fusion_module(h_kg, h_modality, mask=mask)

        # All nodes should be fused
        assert not torch.allclose(output, h_kg, atol=1e-5), (
            "All nodes should be fused when mask is all True"
        )


# ======================================================================
# TestCrossModalFusionGradient
# ======================================================================


class TestCrossModalFusionGradient:
    """Tests for gradient flow through the fusion module."""

    def test_gradient_flow_all_parameters(self) -> None:
        """All parameters should receive gradients after backward."""
        fusion = CrossModalFusion(embed_dim=32, num_heads=2, dropout=0.0)
        h_kg = torch.randn(8, 32, requires_grad=True)
        h_mod = torch.randn(8, 32, requires_grad=True)

        output = fusion(h_kg, h_mod, mask=None)
        loss = output.sum()
        loss.backward()

        # All module parameters should have gradients
        params_without_grad = []
        for name, param in fusion.named_parameters():
            if param.requires_grad and param.grad is None:
                params_without_grad.append(name)

        assert len(params_without_grad) == 0, (
            f"Parameters without gradients: {params_without_grad}"
        )

    def test_gradient_flow_with_mask(self) -> None:
        """Gradients flow through masked nodes only."""
        fusion = CrossModalFusion(embed_dim=32, num_heads=2, dropout=0.0)
        h_kg = torch.randn(8, 32, requires_grad=True)
        h_mod = torch.randn(8, 32, requires_grad=True)
        mask = torch.tensor([True, True, True, True, False, False, False, False])

        output = fusion(h_kg, h_mod, mask=mask)
        loss = output.sum()
        loss.backward()

        # h_kg should have gradients (both fused and passthrough contribute)
        assert h_kg.grad is not None
        # h_mod should have gradients for masked nodes
        assert h_mod.grad is not None

    def test_input_gradient_exists(self) -> None:
        """Input tensors should receive gradients."""
        torch.manual_seed(123)
        fusion = CrossModalFusion(embed_dim=32, num_heads=2, dropout=0.0)
        h_kg = torch.randn(8, 32, requires_grad=True)
        h_mod = torch.randn(8, 32, requires_grad=True)

        output = fusion(h_kg, h_mod, mask=None)
        # Use a weighted sum to avoid symmetry-induced gradient cancellation
        weights = torch.randn(8, 32)
        loss = (output * weights).sum()
        loss.backward()

        assert h_kg.grad is not None, "h_kg should receive gradients"
        assert h_mod.grad is not None, "h_modality should receive gradients"
        assert not torch.all(h_mod.grad == 0), "h_modality gradients should not be all zero"


# ======================================================================
# TestCrossModalFusionNumerical
# ======================================================================


class TestCrossModalFusionNumerical:
    """Numerical stability and consistency tests."""

    def test_no_nan_output(
        self, fusion_module: CrossModalFusion, h_kg: torch.Tensor, h_modality: torch.Tensor
    ) -> None:
        """Output should not contain NaN values."""
        output = fusion_module(h_kg, h_modality, mask=None)
        assert torch.isfinite(output).all(), "Output contains NaN or Inf values"

    def test_no_nan_with_zero_input(self) -> None:
        """Zero input should not produce NaN."""
        fusion = CrossModalFusion(embed_dim=32, num_heads=2, dropout=0.0)
        h_kg = torch.zeros(4, 32)
        h_mod = torch.zeros(4, 32)
        output = fusion(h_kg, h_mod, mask=None)
        assert torch.isfinite(output).all(), "Zero input produced NaN/Inf output"

    def test_deterministic_eval(self) -> None:
        """Module produces identical output on identical input in eval mode."""
        fusion = CrossModalFusion(embed_dim=32, num_heads=2, dropout=0.1)
        fusion.eval()

        torch.manual_seed(42)
        h_kg = torch.randn(8, 32)
        h_mod = torch.randn(8, 32)

        out1 = fusion(h_kg, h_mod, mask=None)
        out2 = fusion(h_kg, h_mod, mask=None)

        assert torch.allclose(out1, out2), (
            "Eval mode should be deterministic for identical inputs"
        )

    def test_output_dtype_matches_input(
        self, fusion_module: CrossModalFusion, h_kg: torch.Tensor, h_modality: torch.Tensor
    ) -> None:
        """Output dtype should match input dtype."""
        output = fusion_module(h_kg, h_modality, mask=None)
        assert output.dtype == h_kg.dtype


# ======================================================================
# TestCrossModalFusionInit
# ======================================================================


class TestCrossModalFusionInit:
    """Tests for module initialization and configuration."""

    def test_embed_dim_stored(self) -> None:
        fusion = CrossModalFusion(embed_dim=128, num_heads=8)
        assert fusion.embed_dim == 128

    def test_num_heads_stored(self) -> None:
        fusion = CrossModalFusion(embed_dim=64, num_heads=4)
        assert fusion.num_heads == 4

    def test_submodules_exist(self) -> None:
        """Module should contain cross_attn, gate_mlp, layer_norm."""
        fusion = CrossModalFusion(embed_dim=64, num_heads=4)
        assert hasattr(fusion, "cross_attn")
        assert hasattr(fusion, "gate_mlp")
        assert hasattr(fusion, "layer_norm")
        assert isinstance(fusion.cross_attn, nn.MultiheadAttention)
        assert isinstance(fusion.layer_norm, nn.LayerNorm)

    def test_parameter_count_reasonable(self) -> None:
        """Verify parameter count is in expected range for d=256."""
        fusion = CrossModalFusion(embed_dim=256, num_heads=4)
        n_params = sum(p.numel() for p in fusion.parameters())
        # Cross-attention (Q,K,V projections + output): ~4*d*d = 262144
        # Gate MLP: 2d*d + d + d*1 + 1 = ~131000
        # LayerNorm: 2*d = 512
        # Total should be in the range of ~400K-600K
        assert 200_000 < n_params < 800_000, f"Unexpected parameter count: {n_params}"
