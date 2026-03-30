"""RotatE decoder for link prediction."""

from __future__ import annotations

import torch
import torch.nn as nn


class RotatEDecoder(nn.Module):
    """RotatE rotation-in-complex-space decoder.

    ``score(h, r, t) = -||h . r - t||``

    Entity embeddings are real tensors of dimension ``d`` that are reshaped
    to ``[B, d/2, 2]`` and reinterpreted as complex numbers.  Relation
    embeddings are phase angles of dimension ``d/2``.
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
            Real entity embeddings ``[batch, dim]`` (dim must be even).
        relation : torch.Tensor
            Phase angles ``[batch, dim // 2]``.
        tail : torch.Tensor
            Real entity embeddings ``[batch, dim]``.

        Returns
        -------
        torch.Tensor
            Scores ``[batch]``.
        """
        complex_dim = relation.size(-1)

        head_c = torch.view_as_complex(head.view(-1, complex_dim, 2))
        tail_c = torch.view_as_complex(tail.view(-1, complex_dim, 2))

        rel_c = torch.polar(torch.ones_like(relation), relation)

        diff = head_c * rel_c - tail_c
        return -diff.abs().sum(dim=-1)
