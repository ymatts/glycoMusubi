"""Gated cross-attention fusion for multi-modal KG embeddings.

Integrates modality-specific features (glycan tree structure, ESM-2 protein
embeddings, text embeddings) with KG-derived features using a gated
cross-attention mechanism.

For nodes with external modality features:
  h_fused = gate * h_KG + (1 - gate) * CrossAttn(h_KG, h_modality)

For nodes without external features:
  h_fused = h_KG  (passthrough)

Reference: Section 3.5 of docs/architecture/model_architecture_design.md
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class CrossModalFusion(nn.Module):
    """Gated cross-attention fusion for multi-modal embeddings.

    Fuses KG-derived node embeddings with modality-specific embeddings using
    cross-attention (Q from KG, K/V from modality) with a learned gating
    mechanism.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of both KG and modality embeddings.
    num_heads : int
        Number of attention heads in the cross-attention layer.
    dropout : float
        Dropout probability applied in the cross-attention and gate MLP.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Cross-attention: Q from KG embeddings, K/V from modality embeddings
        self.cross_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Gating mechanism: decides how much modality info to incorporate
        # Input: concatenation of h_KG and h_cross (cross-attention output)
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )

        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        h_kg: torch.Tensor,
        h_modality: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse KG embeddings with modality-specific embeddings.

        Parameters
        ----------
        h_kg : torch.Tensor
            KG node embeddings, shape ``[N, embed_dim]``.
        h_modality : torch.Tensor
            Modality-specific embeddings (tree/ESM-2/text), shape
            ``[N, embed_dim]``.
        mask : torch.Tensor or None
            Boolean mask of shape ``[N]``. ``True`` for nodes that have
            modality features. Nodes where mask is ``False`` receive
            passthrough (h_kg unchanged). When ``None``, all nodes are fused.

        Returns
        -------
        torch.Tensor
            Fused embeddings, shape ``[N, embed_dim]``.
        """
        if mask is not None and not mask.any():
            # No nodes to fuse -- early return
            return h_kg

        # Determine which nodes to process
        if mask is not None:
            h_kg_active = h_kg[mask]
            h_mod_active = h_modality[mask]
        else:
            h_kg_active = h_kg
            h_mod_active = h_modality

        # Cross-attention: Q from KG, K/V from modality
        # MultiheadAttention expects [B, seq_len, embed_dim]
        # We treat each node as a length-1 sequence
        q = h_kg_active.unsqueeze(1)       # [M, 1, d]
        kv = h_mod_active.unsqueeze(1)     # [M, 1, d]

        h_cross, _ = self.cross_attn(q, kv, kv)  # [M, 1, d]
        h_cross = h_cross.squeeze(1)               # [M, d]

        # Gated fusion
        gate = self.gate_mlp(torch.cat([h_kg_active, h_cross], dim=-1))  # [M, 1]
        h_fused = gate * h_kg_active + (1.0 - gate) * h_cross           # [M, d]
        h_fused = self.layer_norm(h_fused)

        # Scatter fused embeddings back into full output
        if mask is not None:
            output = h_kg.clone()
            output[mask] = h_fused
            return output
        else:
            return h_fused
