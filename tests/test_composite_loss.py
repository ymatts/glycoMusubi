"""Unit tests for CompositeLoss.

Tests cover:
  - L_link only (struct=None, reg=None)
  - L_link + L_struct
  - L_link + L_reg
  - L_link + L_struct + L_reg
  - structural_contrastive_loss: positive pair similarity > negative
  - lambda values effect (lambda=0 disables the corresponding term)
  - backward pass: gradients for all parameters
  - numerical stability with extreme scores
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from glycoMusubi.losses.margin_loss import MarginRankingLoss
from glycoMusubi.losses.bce_loss import BCEWithLogitsKGELoss
from glycoMusubi.losses.composite_loss import CompositeLoss


# ======================================================================
# Constants
# ======================================================================

EMBEDDING_DIM = 32
N_GLYCANS = 10
BATCH_SIZE = 8


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def margin_loss() -> MarginRankingLoss:
    """MarginRankingLoss for use as link_loss."""
    return MarginRankingLoss(margin=1.0)


@pytest.fixture()
def bce_loss() -> BCEWithLogitsKGELoss:
    """BCEWithLogitsKGELoss for use as link_loss."""
    return BCEWithLogitsKGELoss()


@pytest.fixture()
def composite_default(margin_loss: MarginRankingLoss) -> CompositeLoss:
    """CompositeLoss with default lambda values."""
    return CompositeLoss(
        link_loss=margin_loss,
        lambda_struct=0.1,
        lambda_reg=0.01,
    )


@pytest.fixture()
def glycan_embeddings() -> torch.Tensor:
    """Random glycan embeddings [N_glycans, d]."""
    torch.manual_seed(42)
    return torch.randn(N_GLYCANS, EMBEDDING_DIM)


@pytest.fixture()
def positive_pairs() -> torch.Tensor:
    """Positive glycan pairs [P, 2]."""
    return torch.tensor([[0, 1], [2, 3], [4, 5]])


@pytest.fixture()
def all_embeddings() -> dict:
    """Dictionary of named embedding tensors for L2 regularization."""
    torch.manual_seed(42)
    return {
        "entity_embed": torch.randn(20, EMBEDDING_DIM),
        "relation_embed": torch.randn(6, EMBEDDING_DIM),
    }


@pytest.fixture()
def pos_scores() -> torch.Tensor:
    """Positive triple scores."""
    return torch.tensor([3.0, 2.0, 4.0, 1.0])


@pytest.fixture()
def neg_scores() -> torch.Tensor:
    """Negative triple scores."""
    return torch.tensor([-1.0, -2.0, 0.0, -0.5])


# ======================================================================
# TestLinkLossOnly
# ======================================================================


class TestLinkLossOnly:
    """Tests for L_link only (no structural or regularization terms)."""

    def test_link_loss_only_margin(
        self, margin_loss: MarginRankingLoss, pos_scores, neg_scores
    ) -> None:
        """CompositeLoss with only link_loss equals standalone MarginRankingLoss."""
        composite = CompositeLoss(link_loss=margin_loss)
        composite_result = composite(pos_scores, neg_scores)
        standalone_result = margin_loss(pos_scores, neg_scores)
        assert torch.allclose(composite_result, standalone_result, atol=1e-6)

    def test_link_loss_only_bce(
        self, bce_loss: BCEWithLogitsKGELoss, pos_scores, neg_scores
    ) -> None:
        """CompositeLoss with only link_loss equals standalone BCEWithLogitsKGELoss."""
        composite = CompositeLoss(link_loss=bce_loss)
        composite_result = composite(pos_scores, neg_scores)
        standalone_result = bce_loss(pos_scores, neg_scores)
        assert torch.allclose(composite_result, standalone_result, atol=1e-6)

    def test_link_loss_only_returns_scalar(
        self, composite_default: CompositeLoss, pos_scores, neg_scores
    ) -> None:
        """Forward returns a scalar tensor when only link loss is active."""
        loss = composite_default(pos_scores, neg_scores)
        assert loss.dim() == 0


# ======================================================================
# TestLinkPlusStruct
# ======================================================================


class TestLinkPlusStruct:
    """Tests for L_link + L_struct."""

    def test_struct_increases_loss(
        self,
        margin_loss: MarginRankingLoss,
        pos_scores,
        neg_scores,
        glycan_embeddings,
        positive_pairs,
    ) -> None:
        """Adding structural contrastive term increases total loss."""
        composite = CompositeLoss(link_loss=margin_loss, lambda_struct=0.1)
        loss_link_only = composite(pos_scores, neg_scores)
        loss_with_struct = composite(
            pos_scores,
            neg_scores,
            glycan_embeddings=glycan_embeddings,
            positive_pairs=positive_pairs,
        )
        # Structural contrastive loss is non-negative, so total should increase
        assert loss_with_struct.item() >= loss_link_only.item() - 1e-6

    def test_struct_is_added_with_lambda(
        self,
        margin_loss: MarginRankingLoss,
        pos_scores,
        neg_scores,
        glycan_embeddings,
        positive_pairs,
    ) -> None:
        """Structural term is scaled by lambda_struct."""
        lambda_val = 0.5
        composite = CompositeLoss(link_loss=margin_loss, lambda_struct=lambda_val)

        loss_total = composite(
            pos_scores, neg_scores,
            glycan_embeddings=glycan_embeddings,
            positive_pairs=positive_pairs,
        )
        loss_link = margin_loss(pos_scores, neg_scores)
        struct_loss = composite.structural_contrastive_loss(
            glycan_embeddings, positive_pairs
        )

        expected = loss_link + lambda_val * struct_loss
        assert torch.allclose(loss_total, expected, atol=1e-5)


# ======================================================================
# TestLinkPlusReg
# ======================================================================


class TestLinkPlusReg:
    """Tests for L_link + L_reg."""

    def test_reg_increases_loss(
        self,
        margin_loss: MarginRankingLoss,
        pos_scores,
        neg_scores,
        all_embeddings,
    ) -> None:
        """Adding L2 regularization increases total loss."""
        composite = CompositeLoss(link_loss=margin_loss, lambda_reg=0.01)
        loss_link_only = composite(pos_scores, neg_scores)
        loss_with_reg = composite(
            pos_scores, neg_scores,
            all_embeddings=all_embeddings,
        )
        assert loss_with_reg.item() > loss_link_only.item()

    def test_reg_is_added_with_lambda(
        self,
        margin_loss: MarginRankingLoss,
        pos_scores,
        neg_scores,
        all_embeddings,
    ) -> None:
        """Regularization term is scaled by lambda_reg."""
        lambda_val = 0.05
        composite = CompositeLoss(link_loss=margin_loss, lambda_reg=lambda_val)

        loss_total = composite(
            pos_scores, neg_scores,
            all_embeddings=all_embeddings,
        )
        loss_link = margin_loss(pos_scores, neg_scores)
        reg_loss = CompositeLoss._l2_regularization(all_embeddings)

        expected = loss_link + lambda_val * reg_loss
        assert torch.allclose(loss_total, expected, atol=1e-5)


# ======================================================================
# TestLinkPlusStructPlusReg
# ======================================================================


class TestLinkPlusStructPlusReg:
    """Tests for L_link + L_struct + L_reg (all three terms)."""

    def test_all_three_terms(
        self,
        margin_loss: MarginRankingLoss,
        pos_scores,
        neg_scores,
        glycan_embeddings,
        positive_pairs,
        all_embeddings,
    ) -> None:
        """Full composite loss equals sum of all three weighted terms."""
        lambda_s = 0.2
        lambda_r = 0.03
        composite = CompositeLoss(
            link_loss=margin_loss,
            lambda_struct=lambda_s,
            lambda_reg=lambda_r,
        )

        loss_total = composite(
            pos_scores, neg_scores,
            glycan_embeddings=glycan_embeddings,
            positive_pairs=positive_pairs,
            all_embeddings=all_embeddings,
        )

        loss_link = margin_loss(pos_scores, neg_scores)
        loss_struct = composite.structural_contrastive_loss(
            glycan_embeddings, positive_pairs
        )
        loss_reg = CompositeLoss._l2_regularization(all_embeddings)

        expected = loss_link + lambda_s * loss_struct + lambda_r * loss_reg
        assert torch.allclose(loss_total, expected, atol=1e-5)

    def test_returns_scalar(
        self,
        composite_default: CompositeLoss,
        pos_scores,
        neg_scores,
        glycan_embeddings,
        positive_pairs,
        all_embeddings,
    ) -> None:
        """Full composite loss is a scalar."""
        loss = composite_default(
            pos_scores, neg_scores,
            glycan_embeddings=glycan_embeddings,
            positive_pairs=positive_pairs,
            all_embeddings=all_embeddings,
        )
        assert loss.dim() == 0
        assert torch.isfinite(loss)


# ======================================================================
# TestStructuralContrastiveLoss
# ======================================================================


class TestStructuralContrastiveLoss:
    """Tests for structural_contrastive_loss method."""

    def test_positive_pair_similarity_higher(self) -> None:
        """Positive pairs should have higher similarity than random pairs
        when embeddings are set up so positive pairs are close."""
        composite = CompositeLoss(
            link_loss=MarginRankingLoss(),
            struct_temperature=0.07,
        )

        # Create embeddings where pairs (0,1) and (2,3) are similar
        torch.manual_seed(42)
        embeddings = torch.randn(6, EMBEDDING_DIM)
        # Make pair (0,1) very similar
        embeddings[1] = embeddings[0] + 0.01 * torch.randn(EMBEDDING_DIM)
        # Make pair (2,3) very similar
        embeddings[3] = embeddings[2] + 0.01 * torch.randn(EMBEDDING_DIM)

        positive_pairs = torch.tensor([[0, 1], [2, 3]])

        # Normalize to check cosine similarity
        z = torch.nn.functional.normalize(embeddings, dim=-1)
        pos_sim = (z[0] * z[1]).sum().item()
        # Average similarity with non-partner nodes
        neg_sims = []
        for j in range(2, 6):
            neg_sims.append((z[0] * z[j]).sum().item())
        avg_neg_sim = sum(neg_sims) / len(neg_sims)

        assert pos_sim > avg_neg_sim

    def test_empty_positive_pairs(self) -> None:
        """Empty positive_pairs returns zero loss."""
        composite = CompositeLoss(link_loss=MarginRankingLoss())
        embeddings = torch.randn(5, EMBEDDING_DIM)
        empty_pairs = torch.zeros(0, 2, dtype=torch.long)

        loss = composite.structural_contrastive_loss(embeddings, empty_pairs)
        assert loss.item() == pytest.approx(0.0)

    def test_symmetric_loss(self) -> None:
        """Loss is symmetric: swapping columns in positive_pairs gives same loss."""
        composite = CompositeLoss(link_loss=MarginRankingLoss())
        torch.manual_seed(42)
        embeddings = torch.randn(6, EMBEDDING_DIM)
        pairs = torch.tensor([[0, 1], [2, 3]])
        pairs_swapped = torch.tensor([[1, 0], [3, 2]])

        loss_orig = composite.structural_contrastive_loss(embeddings, pairs)
        loss_swapped = composite.structural_contrastive_loss(embeddings, pairs_swapped)

        assert torch.allclose(loss_orig, loss_swapped, atol=1e-5)

    def test_loss_decreases_with_aligned_embeddings(self) -> None:
        """Loss should be lower when positive pairs have similar embeddings."""
        composite = CompositeLoss(link_loss=MarginRankingLoss())

        # Random embeddings (high contrastive loss)
        torch.manual_seed(42)
        embeddings_random = torch.randn(4, EMBEDDING_DIM)

        # Aligned embeddings (positive pairs are identical -> low contrastive loss)
        embeddings_aligned = torch.randn(4, EMBEDDING_DIM)
        embeddings_aligned[1] = embeddings_aligned[0]
        embeddings_aligned[3] = embeddings_aligned[2]

        pairs = torch.tensor([[0, 1], [2, 3]])

        loss_random = composite.structural_contrastive_loss(
            embeddings_random, pairs
        )
        loss_aligned = composite.structural_contrastive_loss(
            embeddings_aligned, pairs
        )

        assert loss_aligned.item() < loss_random.item()


# ======================================================================
# TestLambdaEffects
# ======================================================================


class TestLambdaEffects:
    """Tests for lambda value effects."""

    def test_lambda_struct_zero_disables_struct(
        self,
        pos_scores,
        neg_scores,
        glycan_embeddings,
        positive_pairs,
    ) -> None:
        """lambda_struct=0 makes structural term vanish."""
        margin = MarginRankingLoss(margin=1.0)
        composite = CompositeLoss(link_loss=margin, lambda_struct=0.0)

        loss_with_struct = composite(
            pos_scores, neg_scores,
            glycan_embeddings=glycan_embeddings,
            positive_pairs=positive_pairs,
        )
        loss_link_only = margin(pos_scores, neg_scores)

        assert torch.allclose(loss_with_struct, loss_link_only, atol=1e-6)

    def test_lambda_reg_zero_disables_reg(
        self,
        pos_scores,
        neg_scores,
        all_embeddings,
    ) -> None:
        """lambda_reg=0 makes regularization term vanish."""
        margin = MarginRankingLoss(margin=1.0)
        composite = CompositeLoss(link_loss=margin, lambda_reg=0.0)

        loss_with_reg = composite(
            pos_scores, neg_scores,
            all_embeddings=all_embeddings,
        )
        loss_link_only = margin(pos_scores, neg_scores)

        assert torch.allclose(loss_with_reg, loss_link_only, atol=1e-6)

    def test_larger_lambda_struct_increases_struct_contribution(
        self,
        margin_loss: MarginRankingLoss,
        pos_scores,
        neg_scores,
        glycan_embeddings,
        positive_pairs,
    ) -> None:
        """Increasing lambda_struct increases the structural term contribution."""
        composite_small = CompositeLoss(link_loss=margin_loss, lambda_struct=0.01)
        composite_large = CompositeLoss(link_loss=margin_loss, lambda_struct=1.0)

        loss_small = composite_small(
            pos_scores, neg_scores,
            glycan_embeddings=glycan_embeddings,
            positive_pairs=positive_pairs,
        )
        loss_large = composite_large(
            pos_scores, neg_scores,
            glycan_embeddings=glycan_embeddings,
            positive_pairs=positive_pairs,
        )

        link_loss = margin_loss(pos_scores, neg_scores)
        # Both losses subtract the link loss to get the struct contribution
        struct_contrib_small = loss_small.item() - link_loss.item()
        struct_contrib_large = loss_large.item() - link_loss.item()

        assert struct_contrib_large > struct_contrib_small

    def test_larger_lambda_reg_increases_reg_contribution(
        self,
        margin_loss: MarginRankingLoss,
        pos_scores,
        neg_scores,
        all_embeddings,
    ) -> None:
        """Increasing lambda_reg increases the regularization contribution."""
        composite_small = CompositeLoss(link_loss=margin_loss, lambda_reg=0.001)
        composite_large = CompositeLoss(link_loss=margin_loss, lambda_reg=1.0)

        loss_small = composite_small(
            pos_scores, neg_scores,
            all_embeddings=all_embeddings,
        )
        loss_large = composite_large(
            pos_scores, neg_scores,
            all_embeddings=all_embeddings,
        )

        assert loss_large.item() > loss_small.item()


# ======================================================================
# TestBackwardPass
# ======================================================================


class TestBackwardPass:
    """Tests for backward pass and gradient flow."""

    def test_gradients_flow_through_link_loss(self) -> None:
        """Gradients flow through the link loss term."""
        margin = MarginRankingLoss(margin=1.0)
        composite = CompositeLoss(link_loss=margin)

        pos = torch.tensor([0.5], requires_grad=True)
        neg = torch.tensor([1.0], requires_grad=True)
        loss = composite(pos, neg)
        loss.backward()

        assert pos.grad is not None
        assert neg.grad is not None

    def test_gradients_flow_through_struct_loss(self) -> None:
        """Gradients flow through the structural contrastive term."""
        margin = MarginRankingLoss(margin=1.0)
        composite = CompositeLoss(link_loss=margin, lambda_struct=0.1)

        pos = torch.tensor([0.5], requires_grad=True)
        neg = torch.tensor([1.0], requires_grad=True)
        glycan_emb = torch.randn(4, EMBEDDING_DIM, requires_grad=True)
        pairs = torch.tensor([[0, 1], [2, 3]])

        loss = composite(
            pos, neg,
            glycan_embeddings=glycan_emb,
            positive_pairs=pairs,
        )
        loss.backward()

        assert pos.grad is not None
        assert neg.grad is not None
        assert glycan_emb.grad is not None

    def test_gradients_flow_through_reg_loss(self) -> None:
        """Gradients flow through the L2 regularization term."""
        margin = MarginRankingLoss(margin=1.0)
        composite = CompositeLoss(link_loss=margin, lambda_reg=0.01)

        pos = torch.tensor([0.5], requires_grad=True)
        neg = torch.tensor([1.0], requires_grad=True)
        emb = torch.randn(10, EMBEDDING_DIM, requires_grad=True)
        all_emb = {"entity": emb}

        loss = composite(
            pos, neg,
            all_embeddings=all_emb,
        )
        loss.backward()

        assert pos.grad is not None
        assert emb.grad is not None

    def test_gradients_all_components(self) -> None:
        """Gradients flow through all three terms simultaneously."""
        margin = MarginRankingLoss(margin=1.0)
        composite = CompositeLoss(
            link_loss=margin, lambda_struct=0.1, lambda_reg=0.01,
        )

        pos = torch.tensor([0.5], requires_grad=True)
        neg = torch.tensor([1.0], requires_grad=True)
        glycan_emb = torch.randn(4, EMBEDDING_DIM, requires_grad=True)
        pairs = torch.tensor([[0, 1], [2, 3]])
        entity_emb = torch.randn(10, EMBEDDING_DIM, requires_grad=True)

        loss = composite(
            pos, neg,
            glycan_embeddings=glycan_emb,
            positive_pairs=pairs,
            all_embeddings={"entity": entity_emb},
        )
        loss.backward()

        assert pos.grad is not None
        assert neg.grad is not None
        assert glycan_emb.grad is not None
        assert entity_emb.grad is not None


# ======================================================================
# TestNumericalStability
# ======================================================================


class TestNumericalStability:
    """Tests for numerical stability with extreme inputs."""

    def test_extreme_positive_scores(self) -> None:
        """CompositeLoss handles very large positive scores."""
        margin = MarginRankingLoss(margin=1.0)
        composite = CompositeLoss(link_loss=margin)

        pos = torch.tensor([1e6, 1e6])
        neg = torch.tensor([0.0, 0.0])
        loss = composite(pos, neg)
        assert torch.isfinite(loss)

    def test_extreme_negative_scores(self) -> None:
        """CompositeLoss handles very large negative scores."""
        margin = MarginRankingLoss(margin=1.0)
        composite = CompositeLoss(link_loss=margin)

        pos = torch.tensor([0.0, 0.0])
        neg = torch.tensor([1e6, 1e6])
        loss = composite(pos, neg)
        assert torch.isfinite(loss)

    def test_extreme_embeddings_struct(self) -> None:
        """Structural contrastive loss handles large embedding magnitudes.

        L2 normalization inside the method should prevent overflow.
        """
        composite = CompositeLoss(link_loss=MarginRankingLoss())
        embeddings = torch.randn(4, EMBEDDING_DIM) * 1e4
        pairs = torch.tensor([[0, 1], [2, 3]])

        loss = composite.structural_contrastive_loss(embeddings, pairs)
        assert torch.isfinite(loss)

    def test_extreme_embeddings_reg(self) -> None:
        """L2 regularization handles large embedding magnitudes."""
        large_emb = {"embed": torch.randn(10, EMBEDDING_DIM) * 1e3}
        reg = CompositeLoss._l2_regularization(large_emb)
        assert torch.isfinite(reg)
        assert reg.item() > 0

    def test_zero_embeddings_reg(self) -> None:
        """L2 regularization returns 0 for zero embeddings."""
        zero_emb = {"embed": torch.zeros(10, EMBEDDING_DIM)}
        reg = CompositeLoss._l2_regularization(zero_emb)
        assert reg.item() == pytest.approx(0.0)

    def test_empty_embeddings_reg(self) -> None:
        """L2 regularization returns 0 for empty dict."""
        reg = CompositeLoss._l2_regularization({})
        assert reg.item() == pytest.approx(0.0)

    def test_bce_extreme_scores(self) -> None:
        """CompositeLoss with BCE handles extreme scores (tests sigmoid stability)."""
        bce = BCEWithLogitsKGELoss()
        composite = CompositeLoss(link_loss=bce)

        pos = torch.tensor([100.0, -100.0])
        neg = torch.tensor([-100.0, 100.0])
        loss = composite(pos, neg)
        assert torch.isfinite(loss)

    def test_identical_embeddings_struct(self) -> None:
        """Structural loss with identical embeddings (degenerate case)."""
        composite = CompositeLoss(link_loss=MarginRankingLoss())
        # All embeddings identical -> all similarities equal -> cross-entropy
        # should still be finite (log(1/N) * N terms)
        base = torch.randn(1, EMBEDDING_DIM)
        embeddings = base.expand(4, -1).contiguous()
        pairs = torch.tensor([[0, 1]])

        loss = composite.structural_contrastive_loss(embeddings, pairs)
        assert torch.isfinite(loss)


# ======================================================================
# TestHyperbolicRegularization
# ======================================================================


class TestHyperbolicRegularization:
    """Tests for hyperbolic boundary regularization L_hyp."""

    def test_hyp_reg_computed_correctly(self) -> None:
        """L_hyp = mean(lambda_x^2) where lambda_x = 2 / (1 - c * ||x||^2)."""
        composite = CompositeLoss(
            link_loss=MarginRankingLoss(), curvature=1.0,
        )
        torch.manual_seed(42)
        embeddings = torch.randn(10, EMBEDDING_DIM) * 0.3

        result = composite.hyperbolic_regularization(embeddings)

        # Manual computation
        c = 1.0
        eps = 1e-5
        x_sqnorm = (embeddings * embeddings).sum(dim=-1)
        x_sqnorm = x_sqnorm.clamp(max=1.0 / c - eps)
        lam = 2.0 / (1.0 - c * x_sqnorm).clamp(min=eps)
        expected = (lam ** 2).mean()

        assert torch.allclose(result, expected, atol=1e-5)

    def test_hyp_reg_non_negative(self) -> None:
        """L_hyp is always non-negative (squared conformal factor)."""
        composite = CompositeLoss(link_loss=MarginRankingLoss())
        torch.manual_seed(42)
        embeddings = torch.randn(10, EMBEDDING_DIM) * 0.5
        loss = composite.hyperbolic_regularization(embeddings)
        assert loss.item() >= 0

    def test_hyp_reg_at_origin(self) -> None:
        """At the origin (||x||=0), lambda=2, so L_hyp = 4."""
        composite = CompositeLoss(link_loss=MarginRankingLoss(), curvature=1.0)
        embeddings = torch.zeros(5, EMBEDDING_DIM)
        loss = composite.hyperbolic_regularization(embeddings)
        # lambda_0 = 2 / (1 - 0) = 2, lambda^2 = 4
        assert loss.item() == pytest.approx(4.0, abs=1e-4)

    def test_hyp_reg_increases_near_boundary(self) -> None:
        """L_hyp is larger for points near the boundary than at origin."""
        composite = CompositeLoss(link_loss=MarginRankingLoss(), curvature=1.0)

        origin = torch.zeros(5, EMBEDDING_DIM)
        # Near-boundary: ||x|| close to 1
        direction = torch.randn(5, EMBEDDING_DIM)
        direction = direction / direction.norm(dim=-1, keepdim=True)
        near_boundary = direction * 0.9

        loss_origin = composite.hyperbolic_regularization(origin)
        loss_boundary = composite.hyperbolic_regularization(near_boundary)

        assert loss_boundary.item() > loss_origin.item()

    def test_hyp_reg_added_to_total_loss(self) -> None:
        """L_hyp is added to total loss when hyperbolic_embeddings is provided."""
        margin = MarginRankingLoss(margin=1.0)
        lambda_hyp = 0.05
        composite = CompositeLoss(
            link_loss=margin, lambda_hyp=lambda_hyp,
        )

        pos = torch.tensor([2.0, 3.0])
        neg = torch.tensor([-1.0, -2.0])

        torch.manual_seed(42)
        hyp_emb = torch.randn(10, EMBEDDING_DIM) * 0.3

        loss_with_hyp = composite(pos, neg, hyperbolic_embeddings=hyp_emb)
        loss_link_only = composite(pos, neg)

        # loss_with_hyp = L_link + lambda_hyp * L_hyp
        hyp_loss = composite.hyperbolic_regularization(hyp_emb)
        expected = loss_link_only + lambda_hyp * hyp_loss

        assert torch.allclose(loss_with_hyp, expected, atol=1e-5)

    def test_hyp_reg_gradient_flow(self) -> None:
        """Gradients flow through L_hyp to embeddings."""
        composite = CompositeLoss(
            link_loss=MarginRankingLoss(), lambda_hyp=0.1,
        )
        pos = torch.tensor([1.0], requires_grad=True)
        neg = torch.tensor([0.0], requires_grad=True)
        hyp_emb = torch.randn(5, EMBEDDING_DIM, requires_grad=True)

        loss = composite(pos, neg, hyperbolic_embeddings=hyp_emb)
        loss.backward()

        assert hyp_emb.grad is not None
        assert torch.isfinite(hyp_emb.grad).all()

    def test_lambda_hyp_zero_disables_hyp(self) -> None:
        """lambda_hyp=0 makes hyperbolic term vanish."""
        margin = MarginRankingLoss(margin=1.0)
        composite = CompositeLoss(link_loss=margin, lambda_hyp=0.0)

        pos = torch.tensor([2.0, 3.0])
        neg = torch.tensor([-1.0, -2.0])
        hyp_emb = torch.randn(10, EMBEDDING_DIM) * 0.5

        loss_with_hyp = composite(pos, neg, hyperbolic_embeddings=hyp_emb)
        loss_link_only = margin(pos, neg)

        assert torch.allclose(loss_with_hyp, loss_link_only, atol=1e-6)

    def test_hyp_reg_finite_near_boundary(self) -> None:
        """L_hyp is finite for near-boundary points (clamping works)."""
        composite = CompositeLoss(link_loss=MarginRankingLoss(), curvature=1.0)
        direction = torch.randn(10, EMBEDDING_DIM)
        direction = direction / direction.norm(dim=-1, keepdim=True)
        near_boundary = direction * 0.999
        loss = composite.hyperbolic_regularization(near_boundary)
        assert torch.isfinite(loss)
