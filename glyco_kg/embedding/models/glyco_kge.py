"""TransE, DistMult, and RotatE implementations for glycoMusubi.

All three models inherit from :class:`BaseKGEModel` and operate on
PyG :class:`HeteroData` objects.  Per-type embedding tables are used to
handle the heterogeneous node types (enzyme, protein, glycan, disease,
variant, compound, site), and each type is projected into a shared
embedding space before scoring.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from glycoMusubi.embedding.models.base import BaseKGEModel


# ======================================================================
# TransE
# ======================================================================


class TransE(BaseKGEModel):
    """TransE: translational distance model.

    ``score(h, r, t) = -||h + r - t||``

    Parameters
    ----------
    num_nodes_dict : Dict[str, int]
        Node-type -> count mapping.
    num_relations : int
        Number of relation types.
    embedding_dim : int
        Embedding dimensionality (default 256).
    p_norm : int
        L-p norm to use (1 or 2, default 2).
    """

    def __init__(
        self,
        num_nodes_dict: Dict[str, int],
        num_relations: int,
        embedding_dim: int = 256,
        p_norm: int = 2,
    ) -> None:
        super().__init__(num_nodes_dict, num_relations, embedding_dim)
        self.p_norm = p_norm

    # ----- forward / score -----

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """Return per-type node embeddings (lookup only for TransE)."""
        emb_dict: Dict[str, torch.Tensor] = {}
        for node_type, emb_module in self.node_embeddings.items():
            num_nodes = self.num_nodes_dict[node_type]
            idx = torch.arange(num_nodes, device=emb_module.weight.device)
            emb_dict[node_type] = emb_module(idx)
        return emb_dict

    def score(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """``-||h + r - t||_p``"""
        return -torch.norm(head + relation - tail, p=self.p_norm, dim=-1)


# ======================================================================
# DistMult
# ======================================================================


class DistMult(BaseKGEModel):
    """DistMult: bilinear diagonal model.

    ``score(h, r, t) = <h, r, t>``  (element-wise product, then sum).

    Parameters
    ----------
    num_nodes_dict : Dict[str, int]
        Node-type -> count mapping.
    num_relations : int
        Number of relation types.
    embedding_dim : int
        Embedding dimensionality (default 256).
    """

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """Return per-type node embeddings (lookup only for DistMult)."""
        emb_dict: Dict[str, torch.Tensor] = {}
        for node_type, emb_module in self.node_embeddings.items():
            num_nodes = self.num_nodes_dict[node_type]
            idx = torch.arange(num_nodes, device=emb_module.weight.device)
            emb_dict[node_type] = emb_module(idx)
        return emb_dict

    def score(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """``<h, r, t>`` -- element-wise product then sum."""
        return (head * relation * tail).sum(dim=-1)


# ======================================================================
# RotatE
# ======================================================================


class RotatE(BaseKGEModel):
    """RotatE: rotation in complex space.

    ``score(h, r, t) = -||h . r - t||``

    Head and tail embeddings live in ``C^{d/2}`` (interpreted as pairs of
    real numbers).  The relation is parameterised as a phase angle so that
    ``r = exp(i * theta)``, representing a rotation on the unit circle.

    Parameters
    ----------
    num_nodes_dict : Dict[str, int]
        Node-type -> count mapping.
    num_relations : int
        Number of relation types.
    embedding_dim : int
        *Real* embedding dimensionality.  Must be even so that the complex
        dimension is ``embedding_dim // 2``.
    gamma : float
        Fixed margin used to initialise the phase range (default 9.0).
    """

    def __init__(
        self,
        num_nodes_dict: Dict[str, int],
        num_relations: int,
        embedding_dim: int = 256,
        gamma: float = 9.0,
    ) -> None:
        if embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be even for RotatE")

        # Relation embeddings will store *phase angles* in R^{d/2}.
        super().__init__(num_nodes_dict, num_relations, embedding_dim)
        self.complex_dim = embedding_dim // 2
        self.gamma = gamma

        # Override relation embedding to have half the dimension (phase only)
        self.relation_embeddings = nn.Embedding(num_relations, self.complex_dim)
        self._init_rotate_relations()

    def _init_rotate_relations(self) -> None:
        """Initialise relation phases uniformly in ``[-pi, pi]``."""
        phase_range = (self.gamma + 2.0) / self.complex_dim
        nn.init.uniform_(
            self.relation_embeddings.weight,
            -phase_range,
            phase_range,
        )

    # ----- forward / score -----

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """Return per-type node embeddings (lookup only for RotatE)."""
        emb_dict: Dict[str, torch.Tensor] = {}
        for node_type, emb_module in self.node_embeddings.items():
            num_nodes = self.num_nodes_dict[node_type]
            idx = torch.arange(num_nodes, device=emb_module.weight.device)
            emb_dict[node_type] = emb_module(idx)
        return emb_dict

    def score(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """``-||h . r - t||`` in complex space.

        ``head``/``tail`` are ``[B, d]`` real tensors reshaped to ``[B, d/2, 2]``
        to form complex numbers.  ``relation`` is ``[B, d/2]`` phase angles.
        """
        # Reshape entity embeddings to complex: [B, d/2, 2]
        head_c = torch.view_as_complex(head.view(-1, self.complex_dim, 2))
        tail_c = torch.view_as_complex(tail.view(-1, self.complex_dim, 2))

        # Relation as unit-modulus complex number: exp(i * phase)
        rel_c = torch.polar(
            torch.ones_like(relation),
            relation,
        )

        # Score: -||h * r - t||
        diff = head_c * rel_c - tail_c
        return -diff.abs().sum(dim=-1)

    def get_relation_embedding(self, relation_idx: torch.Tensor) -> torch.Tensor:
        """Return relation phase angles ``[batch, d/2]``."""
        return self.relation_embeddings(relation_idx)
