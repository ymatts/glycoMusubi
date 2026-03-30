"""NBFNet-style Path Reasoner for glycoMusubi link prediction.

Implements a generalized Bellman-Ford GNN that performs iterative message
passing on the heterogeneous knowledge graph to compute path-based
representations for link prediction.  Based on the NBFNet / BioPathNet
architecture described in ``docs/design/algorithm_design.md`` Section 4.2.3.

Key ideas:

1. **Query-conditioned initialisation** -- for a query ``(h, r, ?)``, only
   the source node *h* receives a non-zero initial representation.
2. **Relation-specific message functions** -- each relation type conditions
   the message via additive relation embeddings:
   ``MSG(h_u, r) = Linear(h_u + e_r)``.
3. **Iterative aggregation** -- *T* rounds of Bellman-Ford-style message
   passing propagate information along paths of length <= *T*.
4. **Inverse edges** -- every original edge ``(u, r, v)`` is augmented with
   an inverse edge ``(v, r_inv, u)`` so that information flows in both
   directions.
5. **MLP scoring head** -- final tail representation is concatenated with
   the query relation embedding and scored through an MLP.

Each BF iteration uses a separate message layer with additive relation
conditioning, keeping the parameter count within the 2-4M budget while
allowing distinct transforms at each hop depth.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.utils import scatter

from glycoMusubi.embedding.models.base import BaseKGEModel


class BellmanFordLayer(nn.Module):
    """Single iteration of generalised Bellman-Ford message passing.

    For every edge ``(u, r_uv, v)`` in the flattened (homogeneous) graph,
    computes a relation-conditioned message and scatters it to the
    destination node via a configurable aggregation.

    The message function is ``MSG(h_u, r) = Linear(h_u + e_r)`` where
    ``e_r`` is a learnable relation embedding looked up from an external
    table.  This is more parameter-efficient than a full per-relation
    weight matrix while still providing relation-specific transforms.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of node / relation representations.
    aggregation : str
        Aggregation method: ``"sum"``, ``"mean"``, or ``"pna"``
        (sum + mean + max combined via a learned linear layer).
    dropout : float
        Dropout rate applied after the message transform.
    """

    def __init__(
        self,
        embedding_dim: int,
        aggregation: str = "sum",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.aggregation = aggregation

        # Shared message MLP: h_u + e_r -> message.
        self.msg_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.dropout = nn.Dropout(dropout)

        # PNA aggregation combines sum, mean, max (3x dim) -> dim.
        if aggregation == "pna":
            self.pna_proj = nn.Linear(embedding_dim * 3, embedding_dim)

    # ------------------------------------------------------------------

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_rel_emb: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Run one Bellman-Ford iteration.

        Parameters
        ----------
        h : Tensor[N, dim]
            Current node representations.
        edge_index : Tensor[2, E]
            Source and destination indices for all edges.
        edge_rel_emb : Tensor[E, dim]
            Pre-looked-up relation embedding for each edge.
        num_nodes : int
            Total number of nodes in the flattened graph.

        Returns
        -------
        Tensor[N, dim]
            Updated node representations.
        """
        src, dst = edge_index  # [E]

        # --- Relation-conditioned message: MLP(h_u + e_r) ---
        h_src = h[src]  # [E, dim]
        msg = self.msg_mlp(h_src + edge_rel_emb)  # [E, dim]
        msg = self.dropout(msg)

        # --- Aggregation ---
        if self.aggregation == "sum":
            agg = scatter(msg, dst, dim=0, dim_size=num_nodes, reduce="sum")
        elif self.aggregation == "mean":
            agg = scatter(msg, dst, dim=0, dim_size=num_nodes, reduce="mean")
        elif self.aggregation == "pna":
            agg_sum = scatter(msg, dst, dim=0, dim_size=num_nodes, reduce="sum")
            agg_mean = scatter(msg, dst, dim=0, dim_size=num_nodes, reduce="mean")
            agg_max = scatter(msg, dst, dim=0, dim_size=num_nodes, reduce="max")
            agg = self.pna_proj(torch.cat([agg_sum, agg_mean, agg_max], dim=-1))
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        # Residual connection.
        return h + agg


class PathReasoner(BaseKGEModel):
    """NBFNet-style Bellman-Ford GNN for path-based link prediction.

    For a query ``(h, r, ?)``, the model initialises the source node *h*
    with its entity embedding and propagates representations along all
    paths of length <= ``num_iterations`` through the heterogeneous KG.
    The final representation of each candidate tail node is scored via
    an MLP conditioned on the query relation.

    The input heterogeneous graph is **flattened** internally: node types
    are mapped to contiguous global indices and edge types are mapped to
    integer relation IDs.  Inverse edges are added automatically.

    Each BF iteration uses a **separate** message layer with per-iteration
    layer normalisation.  The additive relation-conditioning
    (``MSG = MLP(h_u + e_r)``) keeps per-layer parameter count low while
    still providing distinct transforms at each hop.

    Parameters
    ----------
    num_nodes_dict : Dict[str, int]
        ``{node_type: count}``.
    num_relations : int
        Number of *original* (non-inverse) relation types.
    embedding_dim : int
        Dimensionality of entity / relation embeddings.
    num_iterations : int
        Number of Bellman-Ford iterations (message-passing rounds).
    aggregation : str
        Aggregation used in each BF layer (``"sum"`` | ``"mean"`` | ``"pna"``).
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        num_nodes_dict: Dict[str, int],
        num_relations: int,
        embedding_dim: int = 256,
        num_iterations: int = 6,
        aggregation: str = "sum",
        dropout: float = 0.1,
    ) -> None:
        super().__init__(num_nodes_dict, num_relations, embedding_dim)

        self.num_iterations = num_iterations
        self.aggregation = aggregation

        # Total relation count doubles because we add inverse relations.
        self.num_total_relations = num_relations * 2

        # Sorted node type order for deterministic flattening.
        self._node_type_order: List[str] = sorted(num_nodes_dict.keys())
        self._total_nodes = sum(num_nodes_dict[nt] for nt in self._node_type_order)

        # Inverse-relation embedding table (separate from original).
        # Original relations: indices [0, num_relations)
        # Inverse relations:  indices [num_relations, 2*num_relations)
        self.inv_relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        nn.init.xavier_uniform_(self.inv_relation_embeddings.weight)

        # Separate BF message layer per iteration for expressiveness.
        self.bf_layers = nn.ModuleList(
            [
                BellmanFordLayer(
                    embedding_dim=embedding_dim,
                    aggregation=aggregation,
                    dropout=dropout,
                )
                for _ in range(num_iterations)
            ]
        )

        # Per-iteration layer normalisation.
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(embedding_dim) for _ in range(num_iterations)]
        )

        # Scoring MLP: [h_t^(T) || e_r] -> scalar.
        self.score_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
        )

    # ------------------------------------------------------------------
    # Flattening helpers
    # ------------------------------------------------------------------

    def _node_type_offset(self, node_type: str) -> int:
        """Return the global offset for *node_type* in the flattened graph."""
        offset = 0
        for nt in self._node_type_order:
            if nt == node_type:
                return offset
            offset += self.num_nodes_dict[nt]
        raise ValueError(f"Unknown node type: {node_type}")

    def _flatten_graph(
        self, data: HeteroData
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert a HeteroData object to flattened edge_index + edge_rel_emb.

        Inverse edges are appended automatically.  Original relation *i*
        uses ``relation_embeddings(i)``; its inverse uses
        ``inv_relation_embeddings(i)``.

        Returns
        -------
        edge_index : Tensor[2, E_total]
            Global source and destination node indices.
        edge_type : Tensor[E_total]
            Integer relation type for each edge (used only for caching /
            debugging; the actual conditioning is via *edge_rel_emb*).
        edge_rel_emb : Tensor[E_total, dim]
            Pre-looked-up relation embedding for each edge.
        """
        # Build a mapping from canonical edge-type triple to an integer
        # relation id.  The mapping is deterministic (sorted) but only
        # covers edge types present in *data*.
        seen_rels: Dict[Tuple[str, str, str], int] = {}
        rel_counter = 0
        for et in sorted(data.edge_types):
            if et not in seen_rels:
                seen_rels[et] = rel_counter
                rel_counter += 1

        all_src: List[torch.Tensor] = []
        all_dst: List[torch.Tensor] = []
        all_type: List[torch.Tensor] = []

        for et in sorted(data.edge_types):
            src_type, _rel, dst_type = et
            ei = data[et].edge_index  # [2, E_rel]
            if ei.numel() == 0:
                continue

            src_offset = self._node_type_offset(src_type)
            dst_offset = self._node_type_offset(dst_type)
            rel_id = seen_rels[et]

            src_global = ei[0] + src_offset
            dst_global = ei[1] + dst_offset

            # Original edges.
            all_src.append(src_global)
            all_dst.append(dst_global)
            all_type.append(torch.full_like(ei[0], rel_id))

            # Inverse edges: swap src <-> dst, shift relation id.
            all_src.append(dst_global)
            all_dst.append(src_global)
            all_type.append(torch.full_like(ei[0], rel_id + self.num_relations))

        device = next(self.parameters()).device
        if len(all_src) == 0:
            return (
                torch.zeros(2, 0, dtype=torch.long, device=device),
                torch.zeros(0, dtype=torch.long, device=device),
                torch.zeros(0, self.embedding_dim, device=device),
            )

        edge_index = torch.stack(
            [torch.cat(all_src), torch.cat(all_dst)], dim=0
        )
        edge_type = torch.cat(all_type)

        # Look up relation embeddings: original from self.relation_embeddings,
        # inverse from self.inv_relation_embeddings.
        is_inverse = edge_type >= self.num_relations
        orig_ids = edge_type.clone()
        orig_ids[is_inverse] -= self.num_relations

        orig_emb = self.relation_embeddings(orig_ids)        # [E, dim]
        inv_emb = self.inv_relation_embeddings(orig_ids)      # [E, dim]
        edge_rel_emb = torch.where(
            is_inverse.unsqueeze(-1), inv_emb, orig_emb
        )  # [E, dim]

        return edge_index, edge_type, edge_rel_emb

    def _build_initial_embeddings(self, data: HeteroData) -> torch.Tensor:
        """Compute initial node embeddings for the flattened graph.

        Uses learned embeddings from the base class tables, respecting
        the deterministic ordering in ``_node_type_order``.

        Returns
        -------
        Tensor[N_total, embedding_dim]
        """
        device = next(self.parameters()).device
        parts: List[torch.Tensor] = []
        for nt in self._node_type_order:
            num_n = self.num_nodes_dict[nt]
            if num_n == 0:
                continue
            idx = torch.arange(num_n, device=device)
            parts.append(self.node_embeddings[nt](idx))
        return torch.cat(parts, dim=0)  # [N_total, dim]

    # ------------------------------------------------------------------
    # BaseKGEModel interface
    # ------------------------------------------------------------------

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """Compute node embeddings via Bellman-Ford message passing.

        Runs the full BF propagation using **all** nodes as implicit
        sources (i.e. each node's initial embedding acts as its own
        boundary condition).  This produces general-purpose node
        embeddings analogous to standard GNN-based KGE models.

        For query-conditioned inference, use :meth:`score_query` instead.

        Parameters
        ----------
        data : HeteroData
            PyG heterogeneous graph.

        Returns
        -------
        Dict[str, Tensor]
            ``{node_type: Tensor[N_type, embedding_dim]}``
        """
        edge_index, _edge_type, edge_rel_emb = self._flatten_graph(data)
        h = self._build_initial_embeddings(data)

        for t in range(self.num_iterations):
            h = self.bf_layers[t](h, edge_index, edge_rel_emb, self._total_nodes)
            h = self.layer_norms[t](h)

        # Split back into per-type embeddings.
        out: Dict[str, torch.Tensor] = {}
        offset = 0
        for nt in self._node_type_order:
            num_n = self.num_nodes_dict[nt]
            if num_n > 0:
                out[nt] = h[offset : offset + num_n]
                offset += num_n
        return out

    def score(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """Score a batch of ``(head, relation, tail)`` triples.

        Uses the MLP scoring head: ``MLP([h_tail || e_relation])``.

        Parameters
        ----------
        head : Tensor[B, dim]
            Head entity embeddings (unused in path-based scoring when
            representations are already query-conditioned, but accepted
            for API compatibility).
        relation : Tensor[B, dim]
            Relation embeddings.
        tail : Tensor[B, dim]
            Tail entity embeddings (query-conditioned BF representations
            when called from :meth:`score_query`).

        Returns
        -------
        Tensor[B]
            Scalar scores.
        """
        combined = torch.cat([tail, relation], dim=-1)  # [B, 2*dim]
        return self.score_mlp(combined).squeeze(-1)      # [B]

    def score_query(
        self,
        data: HeteroData,
        head_type: str,
        head_idx: torch.Tensor,
        relation_idx: torch.Tensor,
        max_parallel: int = 64,
    ) -> Dict[str, torch.Tensor]:
        """Score all candidate tails for a batch of queries.

        For each query ``(h, r, ?)``, initialises node *h* with its
        entity embedding, runs *T* iterations of Bellman-Ford, and
        returns scores for every node in the graph (grouped by type).

        Queries are batched by replicating the graph structure across
        the batch dimension, enabling parallel BF propagation.  When
        ``batch_size * N_total`` exceeds memory limits, queries are
        processed in chunks of ``max_parallel``.

        Parameters
        ----------
        data : HeteroData
            PyG heterogeneous graph.
        head_type : str
            Node type of the query head entities.
        head_idx : Tensor[B]
            Indices of head entities within their type.
        relation_idx : Tensor[B]
            Relation type indices for the queries.
        max_parallel : int
            Maximum number of queries to process simultaneously
            (controls memory vs speed trade-off).  Default 64.

        Returns
        -------
        Dict[str, Tensor]
            ``{node_type: Tensor[B, N_type]}`` -- scores for every
            candidate tail node of each type.
        """
        device = next(self.parameters()).device
        edge_index, _edge_type, edge_rel_emb = self._flatten_graph(data)
        batch_size = head_idx.size(0)
        N = self._total_nodes
        E = edge_index.size(1)

        # Look up relation embeddings (original relations only).
        rel_emb = self.relation_embeddings(relation_idx)  # [B, dim]

        # Full entity embedding table.
        all_emb = self._build_initial_embeddings(data)  # [N_total, dim]

        # Global head indices.
        head_offset = self._node_type_offset(head_type)
        head_global = head_idx + head_offset  # [B]

        # Process queries in chunks to balance parallelism and memory.
        all_scores_flat: List[torch.Tensor] = []

        for chunk_start in range(0, batch_size, max_parallel):
            chunk_end = min(chunk_start + max_parallel, batch_size)
            chunk_size = chunk_end - chunk_start

            # Replicate edge_index for each query in the chunk.
            offsets = torch.arange(
                chunk_size, device=device
            ).unsqueeze(1) * N  # [C, 1]
            batched_src = (edge_index[0].unsqueeze(0) + offsets).reshape(-1)
            batched_dst = (edge_index[1].unsqueeze(0) + offsets).reshape(-1)
            batched_edge_index = torch.stack(
                [batched_src, batched_dst]
            )  # [2, C*E]
            batched_edge_rel_emb = edge_rel_emb.repeat(chunk_size, 1)  # [C*E, dim]

            # Initialise: h[b*N + head_global[b]] = all_emb[head_global[b]].
            h = torch.zeros(
                chunk_size * N, self.embedding_dim, device=device
            )
            chunk_heads = head_global[chunk_start:chunk_end]
            batch_head_indices = (
                torch.arange(chunk_size, device=device) * N + chunk_heads
            )
            h[batch_head_indices] = all_emb[chunk_heads]

            # Bellman-Ford iterations on the batched super-graph.
            for t in range(self.num_iterations):
                h = self.bf_layers[t](
                    h, batched_edge_index, batched_edge_rel_emb, chunk_size * N
                )
                h = self.layer_norms[t](h)

            # Score all candidates: reshape to [C, N, dim].
            h = h.view(chunk_size, N, self.embedding_dim)
            r_chunk = rel_emb[chunk_start:chunk_end]  # [C, dim]
            r_expanded = r_chunk.unsqueeze(1).expand(-1, N, -1)  # [C, N, dim]

            combined = torch.cat([h, r_expanded], dim=-1)  # [C, N, 2*dim]
            chunk_scores = self.score_mlp(combined).squeeze(-1)  # [C, N]
            all_scores_flat.append(chunk_scores)

        # Concatenate chunks -> [B, N_total]
        scores_flat = torch.cat(all_scores_flat, dim=0)  # [B, N_total]

        # Split by node type.
        result: Dict[str, torch.Tensor] = {}
        offset = 0
        for nt in self._node_type_order:
            num_n = self.num_nodes_dict[nt]
            if num_n > 0:
                result[nt] = scores_flat[:, offset : offset + num_n]
                offset += num_n
        return result
