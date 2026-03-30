"""TransE decoder for link prediction."""

from __future__ import annotations

import torch
import torch.nn as nn


class TransEDecoder(nn.Module):
    """TransE-style translational distance decoder.

    ``score(h, r, t) = -||h + r - t||_p``

    Parameters
    ----------
    p_norm : int
        L-p norm (default 2).
    """

    def __init__(self, p_norm: int = 2) -> None:
        super().__init__()
        self.p_norm = p_norm

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
        return -torch.norm(head + relation - tail, p=self.p_norm, dim=-1)
