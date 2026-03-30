"""Margin-based ranking loss for KGE training."""

from __future__ import annotations

import torch
import torch.nn as nn


class MarginRankingLoss(nn.Module):
    """Pairwise margin ranking loss for knowledge graph embedding.

    For each (positive, negative) pair the loss penalises cases where the
    positive score does not exceed the negative score by at least ``margin``:

    ``L = max(0, margin - pos_score + neg_score)``

    Parameters
    ----------
    margin : float
        Required score gap between positive and negative triples (default 9.0).
    reduction : str
        ``'mean'`` or ``'sum'`` (default ``'mean'``).
    """

    def __init__(self, margin: float = 9.0, reduction: str = "mean") -> None:
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the margin ranking loss.

        Parameters
        ----------
        pos_scores : torch.Tensor
            Scores for positive triples ``[batch]`` or ``[batch, 1]``.
        neg_scores : torch.Tensor
            Scores for negative triples.  May be:
            * ``[batch]``      -- one negative per positive.
            * ``[batch, K]``   -- K negatives per positive (broadcast).

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        # Ensure compatible shapes for broadcasting
        if pos_scores.dim() == 1 and neg_scores.dim() == 2:
            pos_scores = pos_scores.unsqueeze(-1)

        loss = torch.clamp(self.margin - pos_scores + neg_scores, min=0.0)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
