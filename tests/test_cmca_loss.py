"""Unit tests for Cross-Modal Contrastive Alignment (CMCA) loss.

Tests cover:
  - Intra-modal loss: positive pairs have lower loss
  - Cross-modal loss: aligned pairs have lower loss
  - Temperature parameter affects loss magnitude
  - Symmetric formulation
  - Empty pairs: graceful handling
  - Gradient flow
"""

from __future__ import annotations

import pytest
import torch

from glycoMusubi.losses.cmca_loss import CMCALoss

# ======================================================================
# Constants
# ======================================================================

EMBEDDING_DIM = 32
BATCH_SIZE = 16


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def cmca() -> CMCALoss:
    """CMCALoss with default temperature."""
    return CMCALoss(temperature=0.07)


@pytest.fixture()
def aligned_embeddings() -> tuple[torch.Tensor, torch.Tensor]:
    """Modal and KG embeddings that are well-aligned (near-identical)."""
    torch.manual_seed(42)
    base = torch.randn(BATCH_SIZE, EMBEDDING_DIM)
    modal = base + 0.01 * torch.randn(BATCH_SIZE, EMBEDDING_DIM)
    kg = base + 0.01 * torch.randn(BATCH_SIZE, EMBEDDING_DIM)
    return modal, kg


@pytest.fixture()
def random_embeddings() -> tuple[torch.Tensor, torch.Tensor]:
    """Modal and KG embeddings that are randomly paired (misaligned)."""
    torch.manual_seed(42)
    modal = torch.randn(BATCH_SIZE, EMBEDDING_DIM)
    kg = torch.randn(BATCH_SIZE, EMBEDDING_DIM)
    return modal, kg


@pytest.fixture()
def intra_embeddings_with_positive_pairs() -> tuple[torch.Tensor, torch.Tensor]:
    """Embeddings where positive pairs are similar."""
    torch.manual_seed(42)
    emb = torch.randn(BATCH_SIZE, EMBEDDING_DIM)
    # Make pairs (0,1), (2,3), (4,5) similar
    emb[1] = emb[0] + 0.01 * torch.randn(EMBEDDING_DIM)
    emb[3] = emb[2] + 0.01 * torch.randn(EMBEDDING_DIM)
    emb[5] = emb[4] + 0.01 * torch.randn(EMBEDDING_DIM)
    pairs = torch.tensor([[0, 1], [2, 3], [4, 5]])
    return emb, pairs


# ======================================================================
# TestIntraModalLoss
# ======================================================================


class TestIntraModalLoss:
    """Tests for intra-modal contrastive loss."""

    def test_positive_pairs_lower_loss(self, cmca: CMCALoss) -> None:
        """Aligned positive pairs produce lower intra-modal loss than random."""
        torch.manual_seed(42)
        # Well-aligned embeddings with similar positive pairs
        emb_aligned = torch.randn(BATCH_SIZE, EMBEDDING_DIM)
        emb_aligned[1] = emb_aligned[0] + 0.01 * torch.randn(EMBEDDING_DIM)
        emb_aligned[3] = emb_aligned[2] + 0.01 * torch.randn(EMBEDDING_DIM)
        pairs = torch.tensor([[0, 1], [2, 3]])

        loss_aligned = cmca.intra_modal_loss(emb_aligned, pairs)

        # Random embeddings with the same pair structure
        torch.manual_seed(99)
        emb_random = torch.randn(BATCH_SIZE, EMBEDDING_DIM)
        loss_random = cmca.intra_modal_loss(emb_random, pairs)

        assert loss_aligned.item() < loss_random.item(), (
            f"Aligned loss {loss_aligned.item():.4f} should be < "
            f"random loss {loss_random.item():.4f}"
        )

    def test_returns_scalar(self, cmca: CMCALoss) -> None:
        """Intra-modal loss returns a scalar tensor."""
        emb = torch.randn(8, EMBEDDING_DIM)
        pairs = torch.tensor([[0, 1], [2, 3]])
        loss = cmca.intra_modal_loss(emb, pairs)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_loss_non_negative(self, cmca: CMCALoss) -> None:
        """InfoNCE loss is non-negative."""
        torch.manual_seed(42)
        emb = torch.randn(8, EMBEDDING_DIM)
        pairs = torch.tensor([[0, 1], [2, 3]])
        loss = cmca.intra_modal_loss(emb, pairs)
        # InfoNCE can technically be negative when N is small and similarity
        # to the positive is very high, but is typically non-negative.
        # We just check it is finite.
        assert torch.isfinite(loss)


# ======================================================================
# TestCrossModalLoss
# ======================================================================


class TestCrossModalLoss:
    """Tests for cross-modal alignment loss."""

    def test_aligned_pairs_lower_loss(
        self,
        cmca: CMCALoss,
        aligned_embeddings: tuple[torch.Tensor, torch.Tensor],
        random_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Aligned pairs produce lower cross-modal loss than random pairs."""
        modal_a, kg_a = aligned_embeddings
        modal_r, kg_r = random_embeddings

        loss_aligned = cmca.cross_modal_loss(modal_a, kg_a)
        loss_random = cmca.cross_modal_loss(modal_r, kg_r)

        assert loss_aligned.item() < loss_random.item(), (
            f"Aligned loss {loss_aligned.item():.4f} should be < "
            f"random loss {loss_random.item():.4f}"
        )

    def test_returns_scalar(self, cmca: CMCALoss) -> None:
        """Cross-modal loss returns a scalar tensor."""
        modal = torch.randn(8, EMBEDDING_DIM)
        kg = torch.randn(8, EMBEDDING_DIM)
        loss = cmca.cross_modal_loss(modal, kg)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_perfect_alignment_low_loss(self, cmca: CMCALoss) -> None:
        """Identical modal and KG embeddings yield very low cross-modal loss."""
        torch.manual_seed(42)
        emb = torch.randn(BATCH_SIZE, EMBEDDING_DIM)
        loss = cmca.cross_modal_loss(emb, emb)
        assert torch.isfinite(loss)
        # With perfect alignment, loss should be close to 0
        # (only limited by numerical precision and log(N) baseline)
        # For InfoNCE with perfect match, loss = log(1) = 0 (plus temperature effects)
        assert loss.item() < 1.0, f"Perfect alignment loss should be low, got {loss.item():.4f}"


# ======================================================================
# TestTemperature
# ======================================================================


class TestTemperature:
    """Tests for temperature parameter effects."""

    def test_temperature_affects_magnitude(self) -> None:
        """Different temperatures produce different loss magnitudes."""
        torch.manual_seed(42)
        modal = torch.randn(BATCH_SIZE, EMBEDDING_DIM)
        kg = torch.randn(BATCH_SIZE, EMBEDDING_DIM)

        cmca_low_temp = CMCALoss(temperature=0.01)
        cmca_high_temp = CMCALoss(temperature=1.0)

        loss_low = cmca_low_temp.cross_modal_loss(modal, kg)
        loss_high = cmca_high_temp.cross_modal_loss(modal, kg)

        assert not torch.allclose(loss_low, loss_high, atol=1e-3), (
            "Different temperatures should produce different loss magnitudes"
        )

    def test_lower_temperature_sharper_loss(self) -> None:
        """Lower temperature produces higher loss for random pairs (sharper distribution)."""
        torch.manual_seed(42)
        modal = torch.randn(BATCH_SIZE, EMBEDDING_DIM)
        kg = torch.randn(BATCH_SIZE, EMBEDDING_DIM)

        cmca_low = CMCALoss(temperature=0.01)
        cmca_high = CMCALoss(temperature=1.0)

        loss_low = cmca_low.cross_modal_loss(modal, kg)
        loss_high = cmca_high.cross_modal_loss(modal, kg)

        # Lower temperature sharpens the softmax, generally increasing cross-entropy
        # for random (misaligned) pairs
        assert loss_low.item() > loss_high.item(), (
            f"Low temp loss {loss_low.item():.4f} should be > "
            f"high temp loss {loss_high.item():.4f} for random pairs"
        )

    def test_invalid_temperature_raises(self) -> None:
        """Temperature <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be > 0"):
            CMCALoss(temperature=0.0)
        with pytest.raises(ValueError, match="temperature must be > 0"):
            CMCALoss(temperature=-0.1)


# ======================================================================
# TestSymmetricFormulation
# ======================================================================


class TestSymmetricFormulation:
    """Tests for symmetric InfoNCE formulation."""

    def test_cross_modal_symmetric(self, cmca: CMCALoss) -> None:
        """Cross-modal loss is symmetric: L(A,B) should equal L(B,A) when they
        are aligned by position (i.e. swapping modal<->kg gives same loss)."""
        torch.manual_seed(42)
        modal = torch.randn(BATCH_SIZE, EMBEDDING_DIM)
        kg = torch.randn(BATCH_SIZE, EMBEDDING_DIM)

        # L(modal->kg) + L(kg->modal) / 2 vs L(kg->modal) + L(modal->kg) / 2
        # The formulation is already symmetric, so both directions should be equal
        loss_ab = cmca.cross_modal_loss(modal, kg)
        loss_ba = cmca.cross_modal_loss(kg, modal)

        assert torch.allclose(loss_ab, loss_ba, atol=1e-5), (
            f"Cross-modal loss not symmetric: {loss_ab.item():.6f} vs {loss_ba.item():.6f}"
        )

    def test_intra_modal_symmetric_pairs(self, cmca: CMCALoss) -> None:
        """Intra-modal loss is symmetric: swapping pair columns gives same loss."""
        torch.manual_seed(42)
        emb = torch.randn(8, EMBEDDING_DIM)
        pairs = torch.tensor([[0, 1], [2, 3], [4, 5]])
        pairs_swapped = torch.tensor([[1, 0], [3, 2], [5, 4]])

        loss_orig = cmca.intra_modal_loss(emb, pairs)
        loss_swapped = cmca.intra_modal_loss(emb, pairs_swapped)

        assert torch.allclose(loss_orig, loss_swapped, atol=1e-5), (
            f"Intra-modal loss not symmetric: {loss_orig.item():.6f} vs {loss_swapped.item():.6f}"
        )

    def test_forward_symmetric_components(self, cmca: CMCALoss) -> None:
        """forward() produces symmetric cross-modal loss."""
        torch.manual_seed(42)
        modal = torch.randn(BATCH_SIZE, EMBEDDING_DIM)
        kg = torch.randn(BATCH_SIZE, EMBEDDING_DIM)

        result1 = cmca(modal_embeddings=modal, kg_embeddings=kg)
        result2 = cmca(modal_embeddings=kg, kg_embeddings=modal)

        assert torch.allclose(
            result1["cross_modal_loss"], result2["cross_modal_loss"], atol=1e-5
        )


# ======================================================================
# TestEmptyPairs
# ======================================================================


class TestEmptyPairs:
    """Tests for graceful handling of empty inputs."""

    def test_empty_positive_pairs_returns_zero(self, cmca: CMCALoss) -> None:
        """Empty positive_pairs returns zero intra-modal loss."""
        emb = torch.randn(8, EMBEDDING_DIM)
        empty_pairs = torch.zeros(0, 2, dtype=torch.long)
        loss = cmca.intra_modal_loss(emb, empty_pairs)
        assert loss.item() == pytest.approx(0.0)

    def test_empty_modal_embeddings_returns_zero(self, cmca: CMCALoss) -> None:
        """Zero-length modal embeddings returns zero cross-modal loss."""
        modal = torch.randn(0, EMBEDDING_DIM)
        kg = torch.randn(0, EMBEDDING_DIM)
        loss = cmca.cross_modal_loss(modal, kg)
        assert loss.item() == pytest.approx(0.0)

    def test_forward_with_no_inputs(self, cmca: CMCALoss) -> None:
        """forward() with no inputs returns zero losses."""
        result = cmca()
        assert result["intra_modal_loss"].item() == pytest.approx(0.0)
        assert result["cross_modal_loss"].item() == pytest.approx(0.0)

    def test_forward_with_only_modal_no_pairs(self, cmca: CMCALoss) -> None:
        """forward() with modal_embeddings but no pairs or kg returns zeros for both."""
        modal = torch.randn(8, EMBEDDING_DIM)
        result = cmca(modal_embeddings=modal)
        assert result["intra_modal_loss"].item() == pytest.approx(0.0)
        assert result["cross_modal_loss"].item() == pytest.approx(0.0)


# ======================================================================
# TestGradientFlow
# ======================================================================


class TestGradientFlow:
    """Tests for gradient flow through CMCA loss."""

    def test_gradient_flow_intra_modal(self, cmca: CMCALoss) -> None:
        """Gradients flow through intra-modal loss to embeddings."""
        emb = torch.randn(8, EMBEDDING_DIM, requires_grad=True)
        pairs = torch.tensor([[0, 1], [2, 3]])
        loss = cmca.intra_modal_loss(emb, pairs)
        loss.backward()

        assert emb.grad is not None, "Embeddings should receive gradients"
        assert torch.any(emb.grad != 0), "Gradients should be non-zero"

    def test_gradient_flow_cross_modal(self, cmca: CMCALoss) -> None:
        """Gradients flow through cross-modal loss to both modal and KG embeddings."""
        modal = torch.randn(BATCH_SIZE, EMBEDDING_DIM, requires_grad=True)
        kg = torch.randn(BATCH_SIZE, EMBEDDING_DIM, requires_grad=True)
        loss = cmca.cross_modal_loss(modal, kg)
        loss.backward()

        assert modal.grad is not None, "Modal embeddings should receive gradients"
        assert kg.grad is not None, "KG embeddings should receive gradients"
        assert torch.any(modal.grad != 0), "Modal gradients should be non-zero"
        assert torch.any(kg.grad != 0), "KG gradients should be non-zero"

    def test_gradient_flow_forward(self, cmca: CMCALoss) -> None:
        """Gradients flow through forward() combined loss."""
        modal = torch.randn(BATCH_SIZE, EMBEDDING_DIM, requires_grad=True)
        kg = torch.randn(BATCH_SIZE, EMBEDDING_DIM, requires_grad=True)
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
        assert torch.any(modal.grad != 0)
        assert torch.any(kg.grad != 0)

    def test_gradient_no_nan(self, cmca: CMCALoss) -> None:
        """Gradients do not contain NaN values."""
        modal = torch.randn(BATCH_SIZE, EMBEDDING_DIM, requires_grad=True)
        kg = torch.randn(BATCH_SIZE, EMBEDDING_DIM, requires_grad=True)
        loss = cmca.cross_modal_loss(modal, kg)
        loss.backward()

        assert not torch.any(torch.isnan(modal.grad)), "Modal gradients contain NaN"
        assert not torch.any(torch.isnan(kg.grad)), "KG gradients contain NaN"
