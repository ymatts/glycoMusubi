"""DistMult decoder for link prediction."""

from __future__ import annotations

import torch
import torch.nn as nn


class DistMultDecoder(nn.Module):
    """DistMult bilinear diagonal decoder.

    ``score(h, r, t) = <h, r, t>``  (element-wise product then sum).
    """

    def forward(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """Score a batch of triples.

        Parameters
        ----------
        head : torch.Tensor
            ``[batch, dim]``
        relation : torch.Tensor
            ``[batch, dim]``
        tail : torch.Tensor
            ``[batch, dim]``

        Returns
        -------
        torch.Tensor
            Scores ``[batch]``.
        """
        return (head * relation * tail).sum(dim=-1)
