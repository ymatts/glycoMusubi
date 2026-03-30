"""Tests for the shared scatter_softmax utility (glycoMusubi.utils.scatter).

Validates the DRY refactoring from C1 that extracted scatter_softmax into
a shared module used by BioHGTLayer, TreeMPNNLayer, and BranchingAwarePooling.

Test coverage:
  - Correctness: result matches manual softmax per group
  - Numerical stability: large logit values do not produce NaN/Inf
  - Empty groups: nodes with no incoming messages get zero
  - Multi-dimensional input: works with [E, H] shaped logits
  - Output sums to 1 per group
  - Gradient flow through scatter_softmax
"""

from __future__ import annotations

import pytest
import torch

from glycoMusubi.utils.scatter import scatter_softmax


# ======================================================================
# Correctness
# ======================================================================


class TestScatterSoftmaxCorrectness:
    """Verify scatter_softmax matches manual per-group softmax."""

    def test_single_group(self):
        """All elements in one group: result should equal standard softmax."""
        src = torch.tensor([1.0, 2.0, 3.0])
        index = torch.tensor([0, 0, 0])
        result = scatter_softmax(src, index, num_nodes=1)
        expected = torch.softmax(src, dim=0)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_two_groups(self):
        """Two groups: softmax computed independently per group."""
        src = torch.tensor([1.0, 2.0, 3.0, 4.0])
        index = torch.tensor([0, 0, 1, 1])
        result = scatter_softmax(src, index, num_nodes=2)

        # Group 0: softmax([1.0, 2.0])
        expected_g0 = torch.softmax(torch.tensor([1.0, 2.0]), dim=0)
        # Group 1: softmax([3.0, 4.0])
        expected_g1 = torch.softmax(torch.tensor([3.0, 4.0]), dim=0)

        assert torch.allclose(result[0], expected_g0[0], atol=1e-6)
        assert torch.allclose(result[1], expected_g0[1], atol=1e-6)
        assert torch.allclose(result[2], expected_g1[0], atol=1e-6)
        assert torch.allclose(result[3], expected_g1[1], atol=1e-6)

    def test_singleton_groups(self):
        """Each element in its own group: all results should be 1.0."""
        src = torch.tensor([5.0, -3.0, 0.0])
        index = torch.tensor([0, 1, 2])
        result = scatter_softmax(src, index, num_nodes=3)
        expected = torch.ones(3)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_output_sums_to_one_per_group(self):
        """Within each group, the scatter_softmax values should sum to 1."""
        torch.manual_seed(42)
        src = torch.randn(20)
        index = torch.randint(0, 5, (20,))
        result = scatter_softmax(src, index, num_nodes=5)

        for g in range(5):
            mask = index == g
            if mask.any():
                group_sum = result[mask].sum()
                assert group_sum.item() == pytest.approx(1.0, abs=1e-5), (
                    f"Group {g} sum={group_sum.item()}, expected 1.0"
                )

    def test_output_non_negative(self):
        """All scatter_softmax outputs should be non-negative."""
        torch.manual_seed(0)
        src = torch.randn(50)
        index = torch.randint(0, 10, (50,))
        result = scatter_softmax(src, index, num_nodes=10)
        assert (result >= 0).all()


# ======================================================================
# Numerical stability
# ======================================================================


class TestScatterSoftmaxStability:
    """Verify numerical stability with extreme input values."""

    def test_large_positive_logits(self):
        """Large positive values should not produce NaN or Inf."""
        src = torch.tensor([1000.0, 1001.0, 999.0])
        index = torch.tensor([0, 0, 0])
        result = scatter_softmax(src, index, num_nodes=1)
        assert torch.isfinite(result).all(), "NaN/Inf with large positive logits"
        assert result.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_large_negative_logits(self):
        """Large negative values should not produce NaN or Inf."""
        src = torch.tensor([-1000.0, -1001.0, -999.0])
        index = torch.tensor([0, 0, 0])
        result = scatter_softmax(src, index, num_nodes=1)
        assert torch.isfinite(result).all(), "NaN/Inf with large negative logits"
        assert result.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_mixed_extreme_logits(self):
        """Mix of very large and very small values should be stable."""
        src = torch.tensor([1e6, -1e6, 0.0])
        index = torch.tensor([0, 0, 0])
        result = scatter_softmax(src, index, num_nodes=1)
        assert torch.isfinite(result).all()
        # The largest value should dominate
        assert result[0].item() == pytest.approx(1.0, abs=1e-5)

    def test_identical_logits(self):
        """Identical logits should produce uniform distribution."""
        src = torch.full((4,), 42.0)
        index = torch.tensor([0, 0, 0, 0])
        result = scatter_softmax(src, index, num_nodes=1)
        expected = torch.full((4,), 0.25)
        assert torch.allclose(result, expected, atol=1e-5)


# ======================================================================
# Empty groups
# ======================================================================


class TestScatterSoftmaxEmptyGroups:
    """Verify behavior when some groups have no elements."""

    def test_empty_group_not_referenced(self):
        """Groups with no elements should not cause errors."""
        src = torch.tensor([1.0, 2.0])
        index = torch.tensor([0, 0])
        # num_nodes=3 means groups 1 and 2 are empty
        result = scatter_softmax(src, index, num_nodes=3)
        assert result.shape == (2,)
        assert torch.isfinite(result).all()

    def test_sparse_groups(self):
        """Non-contiguous group assignment should work."""
        src = torch.tensor([1.0, 2.0, 3.0])
        index = torch.tensor([0, 5, 5])
        result = scatter_softmax(src, index, num_nodes=10)
        assert result.shape == (3,)
        assert torch.isfinite(result).all()
        # Group 0 has one element -> softmax = 1.0
        assert result[0].item() == pytest.approx(1.0, abs=1e-6)
        # Group 5 softmax([2, 3])
        expected_g5 = torch.softmax(torch.tensor([2.0, 3.0]), dim=0)
        assert torch.allclose(result[1:], expected_g5, atol=1e-6)


# ======================================================================
# Multi-dimensional input
# ======================================================================


class TestScatterSoftmaxMultiDim:
    """Verify scatter_softmax works with multi-dimensional inputs."""

    def test_2d_input(self):
        """[E, H] shaped input should apply softmax per group per head."""
        num_heads = 4
        src = torch.randn(6, num_heads)
        index = torch.tensor([0, 0, 0, 1, 1, 1])
        result = scatter_softmax(src, index, num_nodes=2)
        assert result.shape == (6, num_heads)
        assert torch.isfinite(result).all()

        # Check per-head sums for each group
        for h in range(num_heads):
            for g in range(2):
                mask = index == g
                group_sum = result[mask, h].sum()
                assert group_sum.item() == pytest.approx(1.0, abs=1e-5), (
                    f"Group {g}, head {h}: sum={group_sum.item()}"
                )

    def test_3d_input_fails_or_works(self):
        """3D input: scatter_softmax should handle it based on dim=0."""
        src = torch.randn(4, 2, 3)
        index = torch.tensor([0, 0, 1, 1])
        result = scatter_softmax(src, index, num_nodes=2)
        assert result.shape == (4, 2, 3)
        assert torch.isfinite(result).all()


# ======================================================================
# Gradient flow
# ======================================================================


class TestScatterSoftmaxGradient:
    """Verify gradient flows through scatter_softmax."""

    def test_gradient_flows(self):
        """Backward pass should produce gradients for input."""
        src = torch.randn(6, requires_grad=True)
        index = torch.tensor([0, 0, 1, 1, 2, 2])
        result = scatter_softmax(src, index, num_nodes=3)
        loss = result.sum()
        loss.backward()
        assert src.grad is not None
        assert torch.isfinite(src.grad).all()

    def test_gradient_correct_direction(self):
        """Increasing a logit should increase its softmax probability."""
        src = torch.tensor([1.0, 2.0], requires_grad=True)
        index = torch.tensor([0, 0])
        result = scatter_softmax(src, index, num_nodes=1)
        # Gradient of result[1] w.r.t. src[1] should be positive
        result[1].backward()
        assert src.grad[1] > 0, "Increasing logit should increase its probability"


# ======================================================================
# Consumers still use shared implementation
# ======================================================================


class TestScatterSoftmaxUsage:
    """Verify that the shared scatter_softmax is imported by consumers."""

    def test_biohgt_imports_shared_scatter(self):
        """BioHGTLayer should import scatter_softmax from glycoMusubi.utils.scatter."""
        import glycoMusubi.embedding.models.biohgt as biohgt_mod
        # The module imports it as _scatter_softmax
        assert hasattr(biohgt_mod, "_scatter_softmax")

    def test_tree_encoder_imports_shared_scatter(self):
        """GlycanTreeEncoder should import scatter_softmax from glycoMusubi.utils.scatter."""
        import glycoMusubi.embedding.encoders.glycan_tree_encoder as tree_mod
        assert hasattr(tree_mod, "scatter_softmax")
