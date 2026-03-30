"""Inductive scoring adapter for zero-shot link prediction.

Wraps a trained GlycoKGNet model to score triples involving unseen
entities.  For known (train) entities, uses the learned embeddings.
For held-out (inductive) entities, computes embeddings on-the-fly
from biological features (WURCS, ESM-2, etc.) via the model's
feature encoders.

Implements the ``ScorableModel`` protocol from
``glycoMusubi.evaluation.link_prediction`` so it can be used with the
standard ``LinkPredictionEvaluator``.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)


class InductiveAdapter:
    """Adapter that provides inductive scoring for held-out entities.

    Parameters
    ----------
    model : nn.Module
        A trained GlycoKGNet model (or any BaseKGEModel).
    train_data : HeteroData
        The training graph (used to compute train entity embeddings).
    holdout_entity_ids : dict[str, set[int]]
        Per-type sets of held-out entity indices.
    edge_type_to_idx : dict
        Mapping from ``(src_type, rel, dst_type)`` to relation index.
    node_type_order : list[str]
        Ordered list of node types (determines global offset).
    device : str
        Device for computation.
    """

    def __init__(
        self,
        model: nn.Module,
        train_data: HeteroData,
        holdout_entity_ids: Dict[str, Set[int]],
        edge_type_to_idx: Dict[Tuple[str, str, str], int],
        node_type_order: List[str],
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.train_data = train_data
        self.holdout_entity_ids = holdout_entity_ids
        self.edge_type_to_idx = edge_type_to_idx
        self.node_type_order = node_type_order
        self.device = device

        # Compute global node type offsets
        self.node_type_offsets: Dict[str, int] = {}
        self.num_nodes_per_type: Dict[str, int] = {}
        offset = 0
        for ntype in node_type_order:
            self.node_type_offsets[ntype] = offset
            num = train_data[ntype].num_nodes if ntype in train_data.node_types else 0
            self.num_nodes_per_type[ntype] = num
            offset += num
        self.total_entities = offset

        # Precompute all entity embeddings (train + inductive)
        self._emb_dict: Optional[Dict[str, torch.Tensor]] = None
        self._global_embeddings: Optional[torch.Tensor] = None

        # Build relation type info for type-restricted scoring
        self._relation_src_type: Dict[int, str] = {}
        self._relation_dst_type: Dict[int, str] = {}
        for (src, rel, dst), idx in edge_type_to_idx.items():
            self._relation_src_type[idx] = src
            self._relation_dst_type[idx] = dst

    def _ensure_embeddings(self) -> None:
        """Compute and cache all entity embeddings."""
        if self._global_embeddings is not None:
            return

        self.model.eval()
        with torch.no_grad():
            self._emb_dict = self.model(
                self.train_data.to(self.device)
            )

        # Build global embedding matrix [total_entities, dim]
        dim = next(iter(self._emb_dict.values())).shape[1]
        self._global_embeddings = torch.zeros(
            self.total_entities, dim, device=self.device
        )

        for ntype in self.node_type_order:
            if ntype not in self._emb_dict:
                continue
            offset = self.node_type_offsets[ntype]
            num = self._emb_dict[ntype].shape[0]
            self._global_embeddings[offset : offset + num] = self._emb_dict[
                ntype
            ]

        # Re-compute holdout entity embeddings using feature-only paths
        # to avoid noise from untrained learnable embeddings.
        self._recompute_holdout_embeddings()

        logger.info(
            "InductiveAdapter: cached embeddings for %d entities (%d dim)",
            self.total_entities,
            dim,
        )

    def _recompute_holdout_embeddings(self) -> None:
        """Replace holdout entity embeddings with feature-only versions.

        During training, learnable embedding weights for holdout entities
        receive no meaningful gradient signal (their edges are removed).
        For hybrid-mode glycan encoders the untrained learnable part
        injects noise that drowns out the WURCS features.  This method
        overwrites holdout embeddings in ``_global_embeddings`` with
        feature-only variants computed by the model's encoders.

        - **tree_mpnn glycan**: already fully feature-based → skip.
        - **hybrid glycan**: zero out learnable part, keep WURCS projection.
        - **wurcs_features glycan**: already feature-based → skip.
        - **learnable-only glycan**: no features → zero embedding.
        - **esm2 / esm2_site_aware protein**: re-run encoder; proteins
          whose ESM-2 cache is missing fall back to zero embedding.
        - **learnable-only protein**: no features → zero embedding.
        """
        assert self._global_embeddings is not None
        model = self.model
        device = self.device
        n_replaced = 0

        for ntype, holdout_ids in self.holdout_entity_ids.items():
            if not holdout_ids or ntype not in self.node_type_offsets:
                continue

            offset = self.node_type_offsets[ntype]
            holdout_list = sorted(holdout_ids)
            idx_tensor = torch.tensor(
                holdout_list, dtype=torch.long, device=device
            )

            if ntype == "glycan":
                n_replaced += self._recompute_glycan_holdout(
                    model, idx_tensor, holdout_list, offset
                )
            elif ntype == "protein":
                n_replaced += self._recompute_protein_holdout(
                    model, idx_tensor, holdout_list, offset
                )
            else:
                # Non-featured entity type: zero out holdout embeddings
                for local_idx in holdout_list:
                    self._global_embeddings[offset + local_idx] = 0.0
                n_replaced += len(holdout_list)

        if n_replaced > 0:
            logger.info(
                "InductiveAdapter: re-computed %d holdout entity embeddings "
                "using feature-only encoders",
                n_replaced,
            )

    def _recompute_glycan_holdout(
        self,
        model: nn.Module,
        idx_tensor: torch.Tensor,
        holdout_list: List[int],
        offset: int,
    ) -> int:
        """Re-compute holdout glycan embeddings. Returns count replaced."""
        enc_type = getattr(model, "_glycan_encoder_type", "learnable")

        if enc_type == "tree_mpnn":
            # Tree-MPNN is purely feature-based (no learnable shortcut).
            # The forward-pass embeddings are already correct.
            return 0

        if enc_type == "wurcs_features" and hasattr(model, "glycan_encoder"):
            # Pure WURCS projection — already feature-only.
            return 0

        if enc_type == "hybrid" and hasattr(model, "glycan_encoder"):
            encoder = model.glycan_encoder
            feats = encoder._get_wurcs_features(idx_tensor)
            wurcs_emb = encoder.wurcs_proj(feats)
            zero_learnable = torch.zeros_like(wurcs_emb)
            fused = torch.cat([zero_learnable, wurcs_emb], dim=-1)
            holdout_emb = encoder.fusion_mlp(fused)
            for i, local_idx in enumerate(holdout_list):
                self._global_embeddings[offset + local_idx] = holdout_emb[i]
            return len(holdout_list)

        # learnable-only: no biological features available → zero
        for local_idx in holdout_list:
            self._global_embeddings[offset + local_idx] = 0.0
        return len(holdout_list)

    def _recompute_protein_holdout(
        self,
        model: nn.Module,
        idx_tensor: torch.Tensor,
        holdout_list: List[int],
        offset: int,
    ) -> int:
        """Re-compute holdout protein embeddings. Returns count replaced."""
        enc_type = getattr(model, "_protein_encoder_type", "learnable")

        if enc_type in ("esm2", "esm2_site_aware") and hasattr(
            model, "protein_encoder"
        ):
            encoder = model.protein_encoder
            # Re-run the encoder for holdout proteins.  The encoder
            # returns ESM-2-projected embeddings for proteins with cache
            # and learnable embeddings for those without.  We accept the
            # ESM-2-based ones and zero out the learnable-only ones.
            holdout_emb = encoder(idx_tensor)  # [N, dim]

            n_replaced = 0
            for i, local_idx in enumerate(holdout_list):
                global_idx = offset + local_idx
                has_features = local_idx not in getattr(
                    encoder, "_missing_indices", set()
                )
                if has_features:
                    self._global_embeddings[global_idx] = holdout_emb[i]
                else:
                    self._global_embeddings[global_idx] = 0.0
                n_replaced += 1
            return n_replaced

        # learnable-only: no features → zero
        for local_idx in holdout_list:
            self._global_embeddings[offset + local_idx] = 0.0
        return len(holdout_list)

    def score_t(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        num_entities: int,
    ) -> torch.Tensor:
        """Score all tail candidates for ``(head, relation, ?)``.

        Uses **type-restricted scoring**: for each relation, only candidates
        of the valid destination type receive real scores.  All other
        positions are filled with ``-1e9`` so they never rank above valid
        candidates.  This matches the training negative-sampling
        distribution and the transductive evaluator behaviour.

        Parameters
        ----------
        head : torch.Tensor
            Global head indices ``[batch]``.
        relation : torch.Tensor
            Relation indices ``[batch]``.
        num_entities : int
            Total number of candidate entities.

        Returns
        -------
        torch.Tensor
            Scores ``[batch, num_entities]``.
        """
        self._ensure_embeddings()
        assert self._global_embeddings is not None

        head = head.to(self.device)
        relation = relation.to(self.device)
        batch = head.size(0)

        out = torch.full((batch, num_entities), -1e9, device=self.device)

        # Group batch items by relation index
        rel_groups: Dict[int, List[int]] = {}
        for i in range(batch):
            ridx = int(relation[i].item())
            rel_groups.setdefault(ridx, []).append(i)

        for ridx, indices in rel_groups.items():
            dst_type = self._relation_dst_type[ridx]
            dst_offset = self.node_type_offsets[dst_type]
            dst_count = self.num_nodes_per_type[dst_type]
            dst_emb = self._global_embeddings[dst_offset:dst_offset + dst_count]

            idx_t = torch.tensor(indices, dtype=torch.long, device=self.device)
            group_head_emb = self._global_embeddings[head[idx_t]]  # [G, dim]
            group_rel = relation[idx_t]  # [G]

            group_scores = self._batch_score(
                group_head_emb, group_rel, dst_emb
            )  # [G, dst_count]
            out[idx_t, dst_offset:dst_offset + dst_count] = group_scores

        return out

    def score_h(
        self,
        tail: torch.Tensor,
        relation: torch.Tensor,
        num_entities: int,
    ) -> torch.Tensor:
        """Score all head candidates for ``(?, relation, tail)``.

        Uses **type-restricted scoring**: for each relation, only candidates
        of the valid source type receive real scores.  All other positions
        are filled with ``-1e9``.

        Parameters
        ----------
        tail : torch.Tensor
            Global tail indices ``[batch]``.
        relation : torch.Tensor
            Relation indices ``[batch]``.
        num_entities : int
            Total number of candidate entities.

        Returns
        -------
        torch.Tensor
            Scores ``[batch, num_entities]``.
        """
        self._ensure_embeddings()
        assert self._global_embeddings is not None

        tail = tail.to(self.device)
        relation = relation.to(self.device)
        batch = tail.size(0)

        out = torch.full((batch, num_entities), -1e9, device=self.device)

        # Group batch items by relation index
        rel_groups: Dict[int, List[int]] = {}
        for i in range(batch):
            ridx = int(relation[i].item())
            rel_groups.setdefault(ridx, []).append(i)

        for ridx, indices in rel_groups.items():
            src_type = self._relation_src_type[ridx]
            src_offset = self.node_type_offsets[src_type]
            src_count = self.num_nodes_per_type[src_type]
            src_emb = self._global_embeddings[src_offset:src_offset + src_count]

            idx_t = torch.tensor(indices, dtype=torch.long, device=self.device)
            group_tail_emb = self._global_embeddings[tail[idx_t]]  # [G, dim]
            group_rel = relation[idx_t]  # [G]

            group_scores = self._batch_score_h(
                group_tail_emb, group_rel, src_emb
            )  # [G, src_count]
            out[idx_t, src_offset:src_offset + src_count] = group_scores

        return out

    def _batch_score(
        self,
        head_emb: torch.Tensor,
        relation: torch.Tensor,
        all_tail_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Score head against all tails.

        Parameters
        ----------
        head_emb : torch.Tensor
            ``[B, dim]``
        relation : torch.Tensor
            ``[B]`` relation indices
        all_tail_emb : torch.Tensor
            ``[E, dim]``

        Returns
        -------
        torch.Tensor
            ``[B, E]``
        """
        batch_size = head_emb.shape[0]
        num_entities = all_tail_emb.shape[0]

        # Check if model has hybrid decoder
        has_hybrid = hasattr(self.model, '_decoder_type') and self.model._decoder_type == "hybrid"

        if has_hybrid and hasattr(self.model, 'decoder'):
            # Use HybridLinkScorer's batch scoring
            scores = torch.zeros(
                batch_size, num_entities, device=self.device
            )
            # Score in chunks to manage memory
            chunk_size = min(4096, num_entities)
            for start in range(0, num_entities, chunk_size):
                end = min(start + chunk_size, num_entities)
                tail_chunk = all_tail_emb[start:end]  # [C, dim]
                # Expand head for broadcasting
                h_exp = head_emb.unsqueeze(1).expand(
                    -1, tail_chunk.shape[0], -1
                )  # [B, C, dim]
                t_exp = tail_chunk.unsqueeze(0).expand(
                    batch_size, -1, -1
                )  # [B, C, dim]
                r_exp = relation.unsqueeze(1).expand(
                    -1, tail_chunk.shape[0]
                )  # [B, C]
                # Flatten for scoring
                h_flat = h_exp.reshape(-1, h_exp.shape[-1])
                t_flat = t_exp.reshape(-1, t_exp.shape[-1])
                r_flat = r_exp.reshape(-1)
                s = self.model.score(h_flat, r_flat, t_flat)
                scores[:, start:end] = s.view(batch_size, -1)
            return scores
        else:
            # DistMult fallback: <h, r, t>
            rel_emb = self.model.get_relation_embedding(relation)  # [B, dim]
            hr = head_emb * rel_emb  # [B, dim]
            return torch.mm(hr, all_tail_emb.t())  # [B, E]

    def _batch_score_h(
        self,
        tail_emb: torch.Tensor,
        relation: torch.Tensor,
        all_head_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Score all heads against tail.

        Parameters
        ----------
        tail_emb : torch.Tensor
            ``[B, dim]``
        relation : torch.Tensor
            ``[B]`` relation indices
        all_head_emb : torch.Tensor
            ``[E, dim]``

        Returns
        -------
        torch.Tensor
            ``[B, E]``
        """
        batch_size = tail_emb.shape[0]
        num_entities = all_head_emb.shape[0]

        has_hybrid = hasattr(self.model, '_decoder_type') and self.model._decoder_type == "hybrid"

        if has_hybrid and hasattr(self.model, 'decoder'):
            scores = torch.zeros(
                batch_size, num_entities, device=self.device
            )
            chunk_size = min(4096, num_entities)
            for start in range(0, num_entities, chunk_size):
                end = min(start + chunk_size, num_entities)
                head_chunk = all_head_emb[start:end]
                h_exp = head_chunk.unsqueeze(0).expand(
                    batch_size, -1, -1
                )
                t_exp = tail_emb.unsqueeze(1).expand(
                    -1, head_chunk.shape[0], -1
                )
                r_exp = relation.unsqueeze(1).expand(
                    -1, head_chunk.shape[0]
                )
                h_flat = h_exp.reshape(-1, h_exp.shape[-1])
                t_flat = t_exp.reshape(-1, t_exp.shape[-1])
                r_flat = r_exp.reshape(-1)
                s = self.model.score(h_flat, r_flat, t_flat)
                scores[:, start:end] = s.view(batch_size, -1)
            return scores
        else:
            rel_emb = self.model.get_relation_embedding(relation)
            rt = rel_emb * tail_emb
            return torch.mm(rt, all_head_emb.t())
