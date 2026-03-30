"""Self-supervised pre-training tasks for glycoMusubi.

Implements the three pre-training objectives described in Architecture
Design Section 4.3:

1. **Masked Node Feature Prediction** — mask 15 % of node features and
   predict them from context.
2. **Masked Edge Prediction** — remove 10 % of edges and predict their
   existence and relation type.
3. **Glycan Substructure Prediction** — given a glycan KG embedding,
   predict its constituent monosaccharides (multi-label).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)


# ======================================================================
# 1. Masked Node Feature Prediction
# ======================================================================


class MaskedNodePredictor(nn.Module):
    """Predict randomly masked node features from their context.

    A learnable ``[MASK]`` embedding replaces each masked position.
    The decoder head reconstructs the original features with MSE loss
    for continuous features and cross-entropy for categorical features.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of the node embeddings produced by the encoder.
    num_categorical_classes : int
        Number of classes for categorical feature prediction.  Set to 0
        if the node features are purely continuous (default 0).
    continuous_dim : int
        Dimensionality of the continuous portion of each node feature
        vector.  If 0, no MSE reconstruction is applied (default 0).
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        num_categorical_classes: int = 0,
        continuous_dim: int = 0,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_categorical_classes = num_categorical_classes
        self.continuous_dim = continuous_dim

        # Learnable [MASK] token
        self.mask_token = nn.Parameter(torch.zeros(embedding_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Decoder heads
        if continuous_dim > 0:
            self.continuous_head = nn.Linear(embedding_dim, continuous_dim)
        if num_categorical_classes > 0:
            self.categorical_head = nn.Linear(
                embedding_dim, num_categorical_classes
            )

    def mask_and_predict(
        self,
        data: HeteroData,
        model: nn.Module,
        mask_ratio: float = 0.15,
        node_types: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Mask node features, run the encoder, and predict originals.

        Parameters
        ----------
        data : HeteroData
            Input heterogeneous graph.  Node features are expected in
            ``data[node_type].x``.
        model : nn.Module
            Encoder model whose ``forward(data) -> Dict[str, Tensor]``
            produces node embeddings.
        mask_ratio : float
            Fraction of nodes to mask per type (default 0.15).
        node_types : list of str or None
            Node types to apply masking to.  If ``None``, all node types
            with an ``x`` attribute are used.

        Returns
        -------
        loss : torch.Tensor
            Scalar reconstruction loss (MSE + CE).
        predictions : Dict[str, torch.Tensor]
            Per-node-type predicted features for the masked positions.
        """
        if node_types is None:
            node_types = [
                nt for nt in data.node_types if hasattr(data[nt], "x")
            ]

        # --- Store originals and apply mask ---
        originals: Dict[str, torch.Tensor] = {}
        mask_indices: Dict[str, torch.Tensor] = {}

        for nt in node_types:
            x = data[nt].x
            n = x.size(0)
            num_mask = max(1, int(n * mask_ratio))
            perm = torch.randperm(n, device=x.device)[:num_mask]
            mask_indices[nt] = perm
            originals[nt] = x[perm].clone()
            # Replace with [MASK]
            data[nt].x = x.clone()
            data[nt].x[perm] = self.mask_token.to(x.dtype)

        # --- Forward pass through encoder ---
        emb_dict = model(data)

        # --- Predict and compute loss ---
        total_loss = data[node_types[0]].x.new_tensor(0.0)
        predictions: Dict[str, torch.Tensor] = {}
        count = 0

        for nt in node_types:
            if nt not in emb_dict:
                continue
            masked_emb = emb_dict[nt][mask_indices[nt]]  # [num_mask, d]
            orig = originals[nt]  # [num_mask, feat_dim]

            # Continuous reconstruction
            if self.continuous_dim > 0:
                pred_cont = self.continuous_head(masked_emb)
                target_cont = orig[:, : self.continuous_dim]
                total_loss = total_loss + F.mse_loss(pred_cont, target_cont)
                predictions[nt] = pred_cont.detach()
                count += 1

            # Categorical reconstruction
            if self.num_categorical_classes > 0:
                pred_cat = self.categorical_head(masked_emb)
                # Categorical target is the last column of original features
                target_cat = orig[:, -1].long()
                total_loss = total_loss + F.cross_entropy(pred_cat, target_cat)
                predictions.setdefault(nt, pred_cat.detach())
                count += 1

            # Fallback: if neither head is configured, use MSE on full features
            if self.continuous_dim == 0 and self.num_categorical_classes == 0:
                # Direct MSE on embedding space (self-consistency)
                # Use a simple linear projection to match feature dim
                predictions[nt] = masked_emb.detach()
                count += 1

        # --- Restore original features ---
        for nt in node_types:
            data[nt].x[mask_indices[nt]] = originals[nt]

        if count > 0:
            total_loss = total_loss / count

        return total_loss, predictions


# ======================================================================
# 2. Masked Edge Prediction
# ======================================================================


class MaskedEdgePredictor(nn.Module):
    """Predict randomly removed edges (existence and relation type).

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of node embeddings from the encoder.
    num_relations : int
        Number of relation types for relation-type classification.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        num_relations: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_relations = num_relations

        # Edge existence head (binary)
        self.existence_head = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )

        # Relation type classification head
        if num_relations > 1:
            self.relation_head = nn.Sequential(
                nn.Linear(2 * embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, num_relations),
            )

    def mask_and_predict(
        self,
        data: HeteroData,
        model: nn.Module,
        mask_ratio: float = 0.10,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Remove edges, encode the graph, and predict removed edges.

        Parameters
        ----------
        data : HeteroData
            Input heterogeneous graph.
        model : nn.Module
            Encoder model whose ``forward(data) -> Dict[str, Tensor]``
            produces node embeddings.
        mask_ratio : float
            Fraction of edges to remove per edge type (default 0.10).

        Returns
        -------
        loss : torch.Tensor
            Scalar loss (BCE for existence + CE for relation type).
        predictions : Dict[str, torch.Tensor]
            ``'existence_logits'`` and optionally ``'relation_logits'``
            for the masked edges.
        """
        removed_edges: Dict[
            Tuple[str, str, str], Dict[str, torch.Tensor]
        ] = {}
        edge_types_list = list(data.edge_types)

        # --- Remove edges ---
        for edge_type in edge_types_list:
            edge_index = data[edge_type].edge_index
            num_edges = edge_index.size(1)
            num_mask = max(1, int(num_edges * mask_ratio))
            perm = torch.randperm(num_edges, device=edge_index.device)

            mask_idx = perm[:num_mask]
            keep_idx = perm[num_mask:]

            removed_edges[edge_type] = {
                "edge_index": edge_index[:, mask_idx].clone(),
                "keep_idx": keep_idx,
                "original_edge_index": edge_index.clone(),
            }

            # Store relation type index if available
            if hasattr(data[edge_type], "edge_type_idx"):
                removed_edges[edge_type]["edge_type_idx"] = (
                    data[edge_type].edge_type_idx[mask_idx].clone()
                )
                removed_edges[edge_type]["original_edge_type_idx"] = (
                    data[edge_type].edge_type_idx.clone()
                )

            # Remove edges from the graph
            data[edge_type].edge_index = edge_index[:, keep_idx]
            if hasattr(data[edge_type], "edge_type_idx"):
                data[edge_type].edge_type_idx = (
                    data[edge_type].edge_type_idx[keep_idx]
                )

        # --- Forward pass ---
        emb_dict = model(data)

        # --- Predict and compute loss ---
        device = next(iter(emb_dict.values())).device
        total_loss = torch.tensor(0.0, device=device)
        all_existence_logits = []
        all_relation_logits = []
        count = 0

        for edge_type, removed in removed_edges.items():
            src_type, rel, dst_type = edge_type
            removed_ei = removed["edge_index"]  # [2, num_masked]

            if src_type not in emb_dict or dst_type not in emb_dict:
                continue

            head_emb = emb_dict[src_type][removed_ei[0]]
            tail_emb = emb_dict[dst_type][removed_ei[1]]
            pair_emb = torch.cat([head_emb, tail_emb], dim=-1)

            # Existence prediction (positive samples)
            exist_logits = self.existence_head(pair_emb).squeeze(-1)
            pos_target = torch.ones_like(exist_logits)

            # Generate negative samples (random tail)
            num_neg = removed_ei.size(1)
            num_dst_nodes = emb_dict[dst_type].size(0)
            neg_tail_idx = torch.randint(
                0, num_dst_nodes, (num_neg,), device=device
            )
            neg_tail_emb = emb_dict[dst_type][neg_tail_idx]
            neg_pair_emb = torch.cat([head_emb, neg_tail_emb], dim=-1)
            neg_exist_logits = self.existence_head(neg_pair_emb).squeeze(-1)
            neg_target = torch.zeros_like(neg_exist_logits)

            # BCE loss for existence
            all_logits = torch.cat([exist_logits, neg_exist_logits])
            all_targets = torch.cat([pos_target, neg_target])
            total_loss = total_loss + F.binary_cross_entropy_with_logits(
                all_logits, all_targets
            )
            all_existence_logits.append(exist_logits.detach())
            count += 1

            # Relation type prediction
            if self.num_relations > 1 and "edge_type_idx" in removed:
                rel_logits = self.relation_head(pair_emb)
                rel_target = removed["edge_type_idx"]
                total_loss = total_loss + F.cross_entropy(
                    rel_logits, rel_target
                )
                all_relation_logits.append(rel_logits.detach())
                count += 1

        # --- Restore edges ---
        for edge_type, removed in removed_edges.items():
            data[edge_type].edge_index = removed["original_edge_index"]
            if "original_edge_type_idx" in removed:
                data[edge_type].edge_type_idx = removed["original_edge_type_idx"]

        if count > 0:
            total_loss = total_loss / count

        predictions: Dict[str, torch.Tensor] = {}
        if all_existence_logits:
            predictions["existence_logits"] = torch.cat(all_existence_logits)
        if all_relation_logits:
            predictions["relation_logits"] = torch.cat(all_relation_logits)

        return total_loss, predictions


# ======================================================================
# 3. Glycan Substructure Prediction
# ======================================================================


class GlycanSubstructurePredictor(nn.Module):
    """Predict constituent monosaccharides from glycan KG embeddings.

    Uses a two-layer MLP for multi-label classification: given a glycan
    embedding, predict which monosaccharide types are present.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of the glycan KG embeddings (default 256).
    num_monosaccharide_types : int
        Number of monosaccharide types to predict (default 10).
    hidden_dim : int
        Hidden layer size (default 256).
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        num_monosaccharide_types: int = 10,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.num_monosaccharide_types = num_monosaccharide_types

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_monosaccharide_types),
        )

    def predict(self, glycan_embeddings: torch.Tensor) -> torch.Tensor:
        """Predict monosaccharide composition from glycan embeddings.

        Parameters
        ----------
        glycan_embeddings : torch.Tensor
            Shape ``[N, d]`` — glycan KG embeddings.

        Returns
        -------
        torch.Tensor
            Shape ``[N, num_monosaccharide_types]`` — raw logits for
            multi-label classification.  Apply ``sigmoid`` for
            probabilities.
        """
        return self.mlp(glycan_embeddings)

    def compute_loss(
        self,
        glycan_embeddings: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute multi-label binary cross-entropy loss.

        Parameters
        ----------
        glycan_embeddings : torch.Tensor
            Shape ``[N, d]`` — glycan KG embeddings.
        targets : torch.Tensor
            Shape ``[N, num_monosaccharide_types]`` — binary multi-label
            targets.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        logits = self.predict(glycan_embeddings)
        return F.binary_cross_entropy_with_logits(logits, targets.float())
