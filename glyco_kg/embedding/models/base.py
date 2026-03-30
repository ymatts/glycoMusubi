"""Abstract base class for Knowledge Graph Embedding models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData


class BaseKGEModel(ABC, nn.Module):
    """Base class for all KGE models operating on heterogeneous graphs.

    Subclasses must implement :meth:`forward` (node embedding computation)
    and :meth:`score` (triple scoring).

    Parameters
    ----------
    num_nodes_dict : Dict[str, int]
        Mapping from node type to the number of nodes of that type.
    num_relations : int
        Total number of distinct relation (edge) types.
    embedding_dim : int
        Dimensionality of entity and relation embeddings.
    """

    def __init__(
        self,
        num_nodes_dict: Dict[str, int],
        num_relations: int,
        embedding_dim: int = 256,
    ) -> None:
        super().__init__()
        self.num_nodes_dict = num_nodes_dict
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim

        # Per-type entity embedding tables
        self.node_embeddings = nn.ModuleDict(
            {
                node_type: nn.Embedding(num_nodes, embedding_dim)
                for node_type, num_nodes in num_nodes_dict.items()
            }
        )

        # Relation embedding table
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        self._init_embeddings()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_embeddings(self) -> None:
        """Xavier-uniform initialisation for all embedding tables."""
        for emb in self.node_embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """Compute node embeddings from a heterogeneous graph.

        Parameters
        ----------
        data : HeteroData
            PyG heterogeneous graph batch.

        Returns
        -------
        Dict[str, torch.Tensor]
            ``{node_type: Tensor[num_nodes, embedding_dim]}``
        """

    @abstractmethod
    def score(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """Score a batch of (head, relation, tail) triples.

        Parameters
        ----------
        head : torch.Tensor
            Head entity embeddings ``[batch, dim]``.
        relation : torch.Tensor
            Relation embeddings ``[batch, dim]``.
        tail : torch.Tensor
            Tail entity embeddings ``[batch, dim]``.

        Returns
        -------
        torch.Tensor
            Scalar scores ``[batch]``.
        """

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def get_embeddings(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """Return detached node embeddings (evaluation/export helper).

        Parameters
        ----------
        data : HeteroData
            PyG heterogeneous graph.

        Returns
        -------
        Dict[str, torch.Tensor]
            Detached embeddings per node type.
        """
        with torch.no_grad():
            return {k: v.detach() for k, v in self.forward(data).items()}

    def get_relation_embedding(self, relation_idx: torch.Tensor) -> torch.Tensor:
        """Look up relation embeddings by index.

        Parameters
        ----------
        relation_idx : torch.Tensor
            Integer relation indices ``[batch]``.

        Returns
        -------
        torch.Tensor
            Relation embeddings ``[batch, dim]``.
        """
        return self.relation_embeddings(relation_idx)

    def score_triples(
        self,
        data: HeteroData,
        head_type: str,
        head_idx: torch.Tensor,
        relation_idx: torch.Tensor,
        tail_type: str,
        tail_idx: torch.Tensor,
    ) -> torch.Tensor:
        """End-to-end scoring: embed nodes then score triples.

        Parameters
        ----------
        data : HeteroData
            Input heterogeneous graph.
        head_type : str
            Node type of head entities.
        head_idx : torch.Tensor
            Indices of head entities within their type ``[batch]``.
        relation_idx : torch.Tensor
            Relation type indices ``[batch]``.
        tail_type : str
            Node type of tail entities.
        tail_idx : torch.Tensor
            Indices of tail entities within their type ``[batch]``.

        Returns
        -------
        torch.Tensor
            Triple scores ``[batch]``.
        """
        emb_dict = self.forward(data)
        head_emb = emb_dict[head_type][head_idx]
        tail_emb = emb_dict[tail_type][tail_idx]
        rel_emb = self.relation_embeddings(relation_idx)
        return self.score(head_emb, rel_emb, tail_emb)
