"""Unit tests for KGE loss functions.

Tests cover:
  - MarginRankingLoss: margin behaviour, broadcasting, reductions
  - BCEWithLogitsKGELoss: basic BCE, adversarial weighting, label smoothing
"""

from __future__ import annotations

import pytest
import torch

from glycoMusubi.losses.margin_loss import MarginRankingLoss
from glycoMusubi.losses.bce_loss import BCEWithLogitsKGELoss


# ======================================================================
# TestMarginRankingLoss
# ======================================================================


class TestMarginRankingLoss:
    """Tests for MarginRankingLoss."""

    def test_zero_loss_when_positive_exceeds_margin(self) -> None:
        """When pos_score - neg_score >= margin, loss should be 0."""
        loss_fn = MarginRankingLoss(margin=1.0)
        pos_scores = torch.tensor([10.0, 5.0, 3.0])
        neg_scores = torch.tensor([0.0, 0.0, 0.0])
        loss = loss_fn(pos_scores, neg_scores)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_loss_when_margin_violated(self) -> None:
        """When neg_score > pos_score, loss > 0."""
        loss_fn = MarginRankingLoss(margin=1.0)
        pos_scores = torch.tensor([0.0])
        neg_scores = torch.tensor([2.0])
        loss = loss_fn(pos_scores, neg_scores)
        # loss = clamp(1.0 - 0.0 + 2.0, min=0) = 3.0
        assert loss.item() == pytest.approx(3.0)

    def test_margin_value(self) -> None:
        """Loss equals margin when pos_score == neg_score."""
        margin = 5.0
        loss_fn = MarginRankingLoss(margin=margin)
        pos_scores = torch.tensor([1.0])
        neg_scores = torch.tensor([1.0])
        loss = loss_fn(pos_scores, neg_scores)
        assert loss.item() == pytest.approx(margin)

    def test_broadcast_multiple_negatives(self) -> None:
        """Supports [batch] pos_scores and [batch, K] neg_scores."""
        loss_fn = MarginRankingLoss(margin=1.0)
        pos_scores = torch.tensor([10.0, 10.0])  # [2]
        neg_scores = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )  # [2, 3]
        loss = loss_fn(pos_scores, neg_scores)
        # margin - 10 + 0 = -9, clamped to 0 -> loss = 0
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_reduction_sum(self) -> None:
        """Sum reduction gives total loss, not mean."""
        loss_fn = MarginRankingLoss(margin=1.0, reduction="sum")
        pos_scores = torch.tensor([0.0, 0.0])
        neg_scores = torch.tensor([0.0, 0.0])
        loss = loss_fn(pos_scores, neg_scores)
        # Each element: clamp(1.0, min=0) = 1.0; sum = 2.0
        assert loss.item() == pytest.approx(2.0)

    def test_reduction_none(self) -> None:
        """No reduction returns element-wise loss."""
        loss_fn = MarginRankingLoss(margin=1.0, reduction="none")
        pos_scores = torch.tensor([5.0, 0.0])
        neg_scores = torch.tensor([0.0, 0.0])
        loss = loss_fn(pos_scores, neg_scores)
        assert loss.shape == (2,)
        assert loss[0].item() == pytest.approx(0.0, abs=1e-6)
        assert loss[1].item() == pytest.approx(1.0)

    def test_gradient_flows(self) -> None:
        """Gradients flow through the loss."""
        loss_fn = MarginRankingLoss(margin=1.0)
        pos_scores = torch.tensor([0.5], requires_grad=True)
        neg_scores = torch.tensor([1.0], requires_grad=True)
        loss = loss_fn(pos_scores, neg_scores)
        loss.backward()
        assert pos_scores.grad is not None
        assert neg_scores.grad is not None


# ======================================================================
# TestBCEWithLogitsKGELoss
# ======================================================================


class TestBCEWithLogitsKGELoss:
    """Tests for BCEWithLogitsKGELoss."""

    def test_basic_loss_positive(self) -> None:
        """Loss is small when positive scores are high and negatives are low."""
        loss_fn = BCEWithLogitsKGELoss()
        pos_scores = torch.tensor([5.0, 5.0])  # high -> sigmoid ~ 1
        neg_scores = torch.tensor([-5.0, -5.0])  # low -> sigmoid ~ 0
        loss = loss_fn(pos_scores, neg_scores)
        assert loss.item() < 0.1

    def test_basic_loss_negative(self) -> None:
        """Loss is high when scores are inverted."""
        loss_fn = BCEWithLogitsKGELoss()
        pos_scores = torch.tensor([-5.0, -5.0])  # low
        neg_scores = torch.tensor([5.0, 5.0])  # high
        loss = loss_fn(pos_scores, neg_scores)
        assert loss.item() > 5.0

    def test_adversarial_temperature(self) -> None:
        """Adversarial temperature reweights negative samples (requires K>1)."""
        loss_fn_no_adv = BCEWithLogitsKGELoss(adversarial_temperature=None)
        loss_fn_adv = BCEWithLogitsKGELoss(adversarial_temperature=1.0)

        pos_scores = torch.tensor([1.0, 1.0])
        # K=4 negatives per positive → shape [2, 4]
        neg_scores = torch.tensor([[0.5, -0.5, 0.1, -0.3], [0.8, -0.2, 0.3, -0.6]])

        loss_no_adv = loss_fn_no_adv(pos_scores, neg_scores)
        loss_adv = loss_fn_adv(pos_scores, neg_scores)

        # Both should produce finite loss, but values differ.
        assert torch.isfinite(loss_no_adv)
        assert torch.isfinite(loss_adv)
        # With adversarial weighting, harder negatives get more weight.
        assert not torch.allclose(loss_no_adv, loss_adv)

    def test_adversarial_skipped_for_1d(self) -> None:
        """Adversarial weighting is skipped for 1D neg_scores (K=1)."""
        loss_fn_no_adv = BCEWithLogitsKGELoss(adversarial_temperature=None)
        loss_fn_adv = BCEWithLogitsKGELoss(adversarial_temperature=1.0)

        pos_scores = torch.tensor([1.0, 1.0])
        neg_scores = torch.tensor([0.5, -0.5])

        loss_no_adv = loss_fn_no_adv(pos_scores, neg_scores)
        loss_adv = loss_fn_adv(pos_scores, neg_scores)

        # With 1D neg_scores (K=1), adversarial has no effect.
        assert torch.allclose(loss_no_adv, loss_adv)

    def test_label_smoothing(self) -> None:
        """Label smoothing changes the loss value."""
        loss_fn_no_smooth = BCEWithLogitsKGELoss(label_smoothing=0.0)
        loss_fn_smooth = BCEWithLogitsKGELoss(label_smoothing=0.1)

        pos_scores = torch.tensor([3.0])
        neg_scores = torch.tensor([-3.0])

        loss_no = loss_fn_no_smooth(pos_scores, neg_scores)
        loss_sm = loss_fn_smooth(pos_scores, neg_scores)

        # Label smoothing should change the loss value.
        assert not torch.allclose(loss_no, loss_sm)

    def test_reduction_sum(self) -> None:
        """Sum reduction."""
        loss_fn = BCEWithLogitsKGELoss(reduction="sum")
        pos_scores = torch.tensor([0.0])
        neg_scores = torch.tensor([0.0])
        loss = loss_fn(pos_scores, neg_scores)
        assert torch.isfinite(loss)

    def test_reduction_none(self) -> None:
        """No reduction returns tuple of (pos_loss, neg_loss)."""
        loss_fn = BCEWithLogitsKGELoss(reduction="none")
        pos_scores = torch.tensor([2.0, -2.0])
        neg_scores = torch.tensor([1.0, -1.0])
        result = loss_fn(pos_scores, neg_scores)
        assert isinstance(result, tuple)
        pos_loss, neg_loss = result
        assert pos_loss.shape == (2,)
        assert neg_loss.shape == (2,)

    def test_multiple_negatives(self) -> None:
        """Supports [batch, K] negative scores."""
        loss_fn = BCEWithLogitsKGELoss()
        pos_scores = torch.tensor([3.0, 3.0])
        neg_scores = torch.tensor([[-1.0, -2.0, -3.0], [-1.0, -2.0, -3.0]])
        loss = loss_fn(pos_scores, neg_scores)
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_gradient_flows(self) -> None:
        """Gradients flow through the loss."""
        loss_fn = BCEWithLogitsKGELoss()
        pos_scores = torch.tensor([1.0], requires_grad=True)
        neg_scores = torch.tensor([0.0], requires_grad=True)
        loss = loss_fn(pos_scores, neg_scores)
        loss.backward()
        assert pos_scores.grad is not None
        assert neg_scores.grad is not None
