"""Binary cross-entropy loss for KGE training."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEWithLogitsKGELoss(nn.Module):
    """Binary cross-entropy with logits loss for knowledge graph embedding.

    Positive triples are labelled 1, negative triples 0.  Supports optional
    self-adversarial negative weighting (Sun et al., RotatE, ICLR 2019).

    Parameters
    ----------
    adversarial_temperature : float or None
        If provided, negative samples are reweighted by
        ``softmax(score * temperature)`` (self-adversarial scheme).
    label_smoothing : float
        Label smoothing factor (default 0.0).
    reduction : str
        ``'mean'`` or ``'sum'`` (default ``'mean'``).
    """

    def __init__(
        self,
        adversarial_temperature: Optional[float] = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.adversarial_temperature = adversarial_temperature
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Compute BCE loss over positive and negative triples.

        Parameters
        ----------
        pos_scores : torch.Tensor
            Scores for positive triples ``[batch]``.
        neg_scores : torch.Tensor
            Scores for negative triples ``[batch]`` or ``[batch, K]``.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        # Positive target = 1, negative target = 0
        pos_target = torch.ones_like(pos_scores)
        neg_target = torch.zeros_like(neg_scores)

        # Optional label smoothing
        if self.label_smoothing > 0.0:
            pos_target = pos_target * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
            neg_target = neg_target * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, pos_target, reduction="none"
        )

        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores, neg_target, reduction="none"
        )

        # Self-adversarial negative weighting (Sun et al., RotatE, ICLR 2019).
        # Only meaningful when K > 1 negatives per positive (dim >= 2).
        # With K=1, softmax over a single element is always 1.0 — no effect.
        if (
            self.adversarial_temperature is not None
            and neg_scores.dim() >= 2
        ):
            with torch.no_grad():
                neg_weights = F.softmax(
                    neg_scores * self.adversarial_temperature,
                    dim=-1,  # over K negatives per positive
                )
            # Weighted sum over K negatives → [batch]
            neg_loss = (neg_loss * neg_weights).sum(dim=-1)

        if self.reduction == "mean":
            return pos_loss.mean() + neg_loss.mean()
        if self.reduction == "sum":
            return pos_loss.sum() + neg_loss.sum()
        return pos_loss, neg_loss
