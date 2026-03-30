"""Compositional relation embedding module inspired by CompGCN.

Composes relation embeddings from (source node type, edge type, target node
type) using one of three composition operators: subtraction, multiplication,
or circular correlation.

References
----------
- Section 3.4 of ``docs/architecture/model_architecture_design.md``
- Vashishth et al., "Composition-Based Multi-Relational Graph Convolutional
  Networks", ICLR 2020
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


class CompositionalRelationEmbedding(nn.Module):
    """Compose relation embeddings from source type, edge type, and target type.

    ``RelEmb(r) = Compose(NodeTypeEmb(T_src), EdgeTypeEmb(edge_type), NodeTypeEmb(T_dst))``

    Three composition modes are supported:

    * ``"subtraction"``: ``e_src - e_edge + e_dst``
    * ``"multiplication"``: ``e_src * e_edge * e_dst``
    * ``"circular_correlation"``: ``IFFT(conj(FFT(e_src * e_edge)) * FFT(e_dst))``

    Parameters
    ----------
    num_node_types : int
        Number of distinct node types in the heterogeneous graph.
    num_edge_types : int
        Number of distinct edge (relation) types.
    embedding_dim : int
        Dimensionality of the composed relation embedding.
    compose_mode : str
        Composition operator, one of ``"subtraction"``, ``"multiplication"``,
        or ``"circular_correlation"`` (default ``"subtraction"``).
    """

    VALID_MODES = ("subtraction", "multiplication", "circular_correlation")

    def __init__(
        self,
        num_node_types: int,
        num_edge_types: int,
        embedding_dim: int,
        compose_mode: Literal[
            "subtraction", "multiplication", "circular_correlation"
        ] = "subtraction",
    ) -> None:
        super().__init__()
        if compose_mode not in self.VALID_MODES:
            raise ValueError(
                f"compose_mode must be one of {self.VALID_MODES}, "
                f"got '{compose_mode}'"
            )
        self.compose_mode = compose_mode
        self.embedding_dim = embedding_dim

        self.node_type_embed = nn.Embedding(num_node_types, embedding_dim)
        self.edge_type_embed = nn.Embedding(num_edge_types, embedding_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Xavier uniform initialisation."""
        nn.init.xavier_uniform_(self.node_type_embed.weight)
        nn.init.xavier_uniform_(self.edge_type_embed.weight)

    # ------------------------------------------------------------------
    # Composition operators
    # ------------------------------------------------------------------

    @staticmethod
    def _circular_correlation(
        a: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        """Circular correlation via FFT.

        ``corr(a, b) = IFFT(conj(FFT(a)) * FFT(b))``

        Parameters
        ----------
        a, b : torch.Tensor
            Real tensors ``[..., d]``.

        Returns
        -------
        torch.Tensor
            Real-valued circular correlation ``[..., d]``.
        """
        fa = torch.fft.rfft(a, dim=-1)
        fb = torch.fft.rfft(b, dim=-1)
        return torch.fft.irfft(fa.conj() * fb, n=a.size(-1), dim=-1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        src_type_idx: torch.Tensor,
        edge_type_idx: torch.Tensor,
        dst_type_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Compose a relation embedding from type indices.

        Parameters
        ----------
        src_type_idx : torch.Tensor
            Source node type indices ``[batch]`` or scalar.
        edge_type_idx : torch.Tensor
            Edge type indices ``[batch]`` or scalar.
        dst_type_idx : torch.Tensor
            Destination node type indices ``[batch]`` or scalar.

        Returns
        -------
        torch.Tensor
            Composed relation embeddings ``[batch, embedding_dim]`` (or
            ``[embedding_dim]`` for scalar inputs).
        """
        e_src = self.node_type_embed(src_type_idx)
        e_edge = self.edge_type_embed(edge_type_idx)
        e_dst = self.node_type_embed(dst_type_idx)

        if self.compose_mode == "subtraction":
            return e_src - e_edge + e_dst
        elif self.compose_mode == "multiplication":
            return e_src * e_edge * e_dst
        else:  # circular_correlation
            return self._circular_correlation(e_src * e_edge, e_dst)
