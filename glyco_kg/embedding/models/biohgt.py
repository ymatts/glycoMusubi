"""Biology-Aware Heterogeneous Graph Transformer (BioHGT) for glycoMusubi.

Implements a heterogeneous graph transformer with biology-informed attention
priors tailored to the glycoMusubi schema.  Key innovations over standard HGT:

1. **Type-specific Q/K/V transforms** -- separate linear projections per node type.
2. **Relation-conditioned attention** -- per-edge-type attention and message
   weight matrices.
3. **BioPrior attention bias** -- domain-specific inductive biases for
   enzyme->glycan (biosynthetic pathway order) and site<->site (PTM crosstalk)
   edges, plus learnable scalar biases for all other relation types.

The module integrates with PyG's ``HeteroData`` interface and inherits from
:class:`BaseKGEModel` for compatibility with the glycoMusubi training pipeline.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.utils import scatter

from glycoMusubi.embedding.models.base import BaseKGEModel
from glycoMusubi.utils.scatter import scatter_softmax as _scatter_softmax

# Default node and edge types for glycoMusubi (Phase 2 extended schema).
DEFAULT_NODE_TYPES: List[str] = [
    "glycan",
    "protein",
    "enzyme",
    "disease",
    "variant",
    "compound",
    "site",
    "motif",
    "reaction",
    "pathway",
]

DEFAULT_EDGE_TYPES: List[Tuple[str, str, str]] = [
    ("protein", "has_glycan", "glycan"),
    ("compound", "inhibits", "enzyme"),
    ("protein", "associated_with_disease", "disease"),
    ("protein", "has_variant", "variant"),
    ("protein", "has_site", "site"),
    ("enzyme", "has_site", "site"),
    ("site", "ptm_crosstalk", "site"),
    ("enzyme", "produced_by", "glycan"),
    ("enzyme", "consumed_by", "glycan"),
    ("glycan", "has_motif", "motif"),
    ("glycan", "child_of", "glycan"),
    ("enzyme", "catalyzed_by", "reaction"),
    ("reaction", "has_product", "glycan"),
]


# ======================================================================
# BioPrior -- biology-aware attention bias
# ======================================================================


class BioPrior(nn.Module):
    """Biology-aware attention bias for BioHGT.

    Produces a scalar attention bias for each edge in a mini-batch, depending
    on the relation type:

    * **enzyme -> glycan** (``produced_by`` / ``consumed_by``): a learnable
      embedding-based bias that captures biosynthetic pathway order similarity.
    * **site <-> site** (``ptm_crosstalk``): a learnable distance-based bias
      parameterised as an MLP over absolute position difference features.
    * **All other relations**: a single learnable scalar bias per relation type.

    Parameters
    ----------
    edge_types : List[Tuple[str, str, str]]
        Canonical edge types ``(src_type, relation, dst_type)``.
    hidden_dim : int
        Hidden dimension for the pathway / crosstalk sub-networks.
    """

    # Relations that receive specialised priors.
    _BIOSYNTHETIC_RELATIONS = {"produced_by", "consumed_by"}
    _CROSSTALK_RELATIONS = {"ptm_crosstalk"}

    def __init__(
        self,
        edge_types: List[Tuple[str, str, str]],
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Collect unique relation names for quick lookup.
        self._relation_set = {et[1] for et in edge_types}

        # --- biosynthetic pathway prior ---
        # A small MLP that converts a pair of learnable "pathway-order"
        # embeddings into a scalar bias.  The embeddings themselves are
        # stored externally (on BioHGTLayer) because they are shared
        # across layers, but the scoring MLP lives here.
        if self._BIOSYNTHETIC_RELATIONS & self._relation_set:
            self.pathway_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )

        # --- PTM crosstalk prior ---
        # An MLP that converts positional-difference features into a bias.
        if self._CROSSTALK_RELATIONS & self._relation_set:
            # Input: |pos_src - pos_dst| encoded as a small feature vector.
            self.crosstalk_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            # Learnable positional encoding for absolute distance.
            self.distance_proj = nn.Linear(1, hidden_dim)

        # --- default scalar biases (one per relation name) ---
        unique_relations = sorted({et[1] for et in edge_types})
        self.default_bias = nn.ParameterDict(
            {rel: nn.Parameter(torch.zeros(1)) for rel in unique_relations}
        )

    # ------------------------------------------------------------------

    def forward(
        self,
        relation: str,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        *,
        pathway_emb_src: Optional[torch.Tensor] = None,
        pathway_emb_dst: Optional[torch.Tensor] = None,
        site_positions_src: Optional[torch.Tensor] = None,
        site_positions_dst: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute attention bias for a set of edges of a single relation type.

        Parameters
        ----------
        relation : str
            The relation name (e.g. ``"produced_by"``).
        src_idx, dst_idx : Tensor[E]
            Source / destination node indices within the edge set.
        pathway_emb_src, pathway_emb_dst : Tensor[E, hidden_dim], optional
            Pre-looked-up pathway-order embeddings for biosynthetic relations.
        site_positions_src, site_positions_dst : Tensor[E], optional
            Integer residue positions for PTM crosstalk edges.

        Returns
        -------
        Tensor[E]
            Scalar attention bias per edge.
        """
        num_edges = src_idx.size(0)
        device = src_idx.device

        # Biosynthetic pathway prior
        if (
            relation in self._BIOSYNTHETIC_RELATIONS
            and pathway_emb_src is not None
            and pathway_emb_dst is not None
        ):
            pair_feat = torch.cat([pathway_emb_src, pathway_emb_dst], dim=-1)
            bias = self.pathway_mlp(pair_feat).squeeze(-1)  # [E]
            return bias

        # PTM crosstalk prior
        if (
            relation in self._CROSSTALK_RELATIONS
            and site_positions_src is not None
            and site_positions_dst is not None
        ):
            dist = (site_positions_src.float() - site_positions_dst.float()).abs()
            dist_feat = self.distance_proj(dist.unsqueeze(-1))  # [E, hidden_dim]
            bias = self.crosstalk_mlp(dist_feat).squeeze(-1)  # [E]
            return bias

        # Default: learnable scalar broadcast to all edges.
        return self.default_bias[relation].expand(num_edges)


# ======================================================================
# BioHGTLayer
# ======================================================================


class BioHGTLayer(nn.Module):
    """Single BioHGT layer with biology-aware attention.

    For target node *i* of type ``T_i``, aggregates messages from neighbours
    *j* of type ``T_j`` connected by relation ``r = (T_src, edge_type, T_dst)``:

    .. math::

        Q_i &= W_Q^{T_i} h_i + b_Q^{T_i} \\\\
        K_j &= W_K^{T_j} h_j + b_K^{T_j} \\\\
        V_j &= W_V^{T_j} h_j + b_V^{T_j} \\\\
        \\alpha_{ij}^r &= \\mathrm{softmax}\\bigl(
            (Q_i W_{\\mathrm{attn}}^r K_j^\\top) / \\sqrt{d_k}
            + \\mathrm{BioPrior}(i,j,r)\\bigr) \\\\
        m_i^r &= \\sum_j \\alpha_{ij}^r W_{\\mathrm{msg}}^r V_j \\\\
        h_i^{l+1} &= \\mathrm{LayerNorm}\\bigl(h_i^l
            + \\mathrm{FFN}(\\mathrm{AGG}_r(m_i^r))\\bigr)

    Parameters
    ----------
    in_dim : int
        Input feature dimension per node (default 256).
    out_dim : int
        Output feature dimension per node (default 256).
    num_heads : int
        Number of attention heads (default 8).
    node_types : List[str]
        All node type names in the KG.
    edge_types : List[Tuple[str, str, str]]
        All canonical ``(src, rel, dst)`` edge types.
    use_bio_prior : bool
        Whether to apply biology-aware attention biases.
    dropout : float
        Dropout rate for attention weights and FFN.
    """

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 256,
        num_heads: int = 8,
        node_types: Optional[List[str]] = None,
        edge_types: Optional[List[Tuple[str, str, str]]] = None,
        use_bio_prior: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        node_types = node_types or DEFAULT_NODE_TYPES
        edge_types = edge_types or DEFAULT_EDGE_TYPES

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.d_k = out_dim // num_heads
        self.node_types = node_types
        self.edge_types = edge_types
        self.use_bio_prior = use_bio_prior

        # -- Type-specific Q / K / V linear transforms --
        self.q_linears = nn.ModuleDict(
            {nt: nn.Linear(in_dim, out_dim) for nt in node_types}
        )
        self.k_linears = nn.ModuleDict(
            {nt: nn.Linear(in_dim, out_dim) for nt in node_types}
        )
        self.v_linears = nn.ModuleDict(
            {nt: nn.Linear(in_dim, out_dim) for nt in node_types}
        )

        # -- Relation-conditioned attention and message weight matrices --
        # Keyed by *relation name* (not full triple) so that the same
        # relation used with different source types (e.g. protein-has_site-site
        # vs enzyme-has_site-site) shares weights.
        unique_relations = sorted({et[1] for et in edge_types})
        self.attn_weights = nn.ParameterDict(
            {
                rel: nn.Parameter(
                    torch.empty(num_heads, self.d_k, self.d_k)
                )
                for rel in unique_relations
            }
        )
        self.msg_weights = nn.ParameterDict(
            {
                rel: nn.Parameter(
                    torch.empty(num_heads, self.d_k, self.d_k)
                )
                for rel in unique_relations
            }
        )

        # Initialise relation matrices.
        for p in list(self.attn_weights.values()) + list(self.msg_weights.values()):
            nn.init.xavier_uniform_(p, gain=1.0 / math.sqrt(2.0))

        # -- BioPrior module --
        self.bio_prior: Optional[BioPrior] = None
        if use_bio_prior:
            self.bio_prior = BioPrior(edge_types, hidden_dim=32)

        # -- Feed-forward network (per node type) --
        self.ffn = nn.ModuleDict(
            {
                nt: nn.Sequential(
                    nn.Linear(out_dim, out_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(out_dim * 4, out_dim),
                )
                for nt in node_types
            }
        )

        # -- LayerNorm (per node type) --
        self.norm_attn = nn.ModuleDict(
            {nt: nn.LayerNorm(out_dim) for nt in node_types}
        )
        self.norm_ffn = nn.ModuleDict(
            {nt: nn.LayerNorm(out_dim) for nt in node_types}
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        *,
        bio_prior_kwargs: Optional[
            Dict[Tuple[str, str, str], dict]
        ] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run one BioHGT layer.

        Parameters
        ----------
        x_dict : Dict[str, Tensor]
            ``{node_type: Tensor[N_type, in_dim]}``.
        edge_index_dict : Dict[Tuple, Tensor]
            ``{(src_type, rel, dst_type): Tensor[2, E_rel]}``.
        bio_prior_kwargs : dict, optional
            Per-edge-type keyword arguments forwarded to
            :meth:`BioPrior.forward` (e.g. pathway embeddings).

        Returns
        -------
        Dict[str, Tensor]
            Updated node features ``{node_type: Tensor[N_type, out_dim]}``.
        """
        bio_prior_kwargs = bio_prior_kwargs or {}

        # Accumulate messages per destination node type.
        # Each entry is a tensor [N_dst, out_dim] accumulated via scatter.
        msg_accum: Dict[str, torch.Tensor] = {}

        for (src_type, rel, dst_type), edge_index in edge_index_dict.items():
            if src_type not in x_dict or dst_type not in x_dict:
                continue
            if edge_index.numel() == 0:
                continue

            src_x = x_dict[src_type]   # [N_src, in_dim]
            dst_x = x_dict[dst_type]   # [N_dst, in_dim]
            src_idx = edge_index[0]     # [E]
            dst_idx = edge_index[1]     # [E]

            # --- Q / K / V ---
            Q = self.q_linears[dst_type](dst_x[dst_idx])  # [E, out_dim]
            K = self.k_linears[src_type](src_x[src_idx])   # [E, out_dim]
            V = self.v_linears[src_type](src_x[src_idx])   # [E, out_dim]

            # Reshape to multi-head: [E, H, d_k]
            Q = Q.view(-1, self.num_heads, self.d_k)
            K = K.view(-1, self.num_heads, self.d_k)
            V = V.view(-1, self.num_heads, self.d_k)

            # --- Relation-conditioned attention ---
            # W_attn^rel: [H, d_k, d_k]
            W_attn = self.attn_weights[rel]
            # K_rel = K @ W_attn^T  => [E, H, d_k]
            K_rel = torch.einsum("ehd,hdf->ehf", K, W_attn)
            # Dot-product attention: [E, H]
            attn_logits = (Q * K_rel).sum(dim=-1) / math.sqrt(self.d_k)

            # --- Biology-aware prior ---
            if self.bio_prior is not None:
                extra = bio_prior_kwargs.get((src_type, rel, dst_type), {})
                bio_bias = self.bio_prior(
                    relation=rel,
                    src_idx=src_idx,
                    dst_idx=dst_idx,
                    **extra,
                )  # [E]
                attn_logits = attn_logits + bio_bias.unsqueeze(-1)  # broadcast H

            # --- Softmax per destination node ---
            attn = _scatter_softmax(attn_logits, dst_idx, num_nodes=dst_x.size(0))
            attn = self.attn_dropout(attn)  # [E, H]

            # --- Message: V transformed by relation message weights ---
            W_msg = self.msg_weights[rel]  # [H, d_k, d_k]
            V_rel = torch.einsum("ehd,hdf->ehf", V, W_msg)  # [E, H, d_k]
            msg = attn.unsqueeze(-1) * V_rel  # [E, H, d_k]
            msg = msg.view(-1, self.out_dim)  # [E, out_dim]

            # --- Scatter-add to destination ---
            agg = scatter(msg, dst_idx, dim=0, dim_size=dst_x.size(0), reduce="sum")

            if dst_type in msg_accum:
                msg_accum[dst_type] = msg_accum[dst_type] + agg
            else:
                msg_accum[dst_type] = agg

        # --- Residual + LayerNorm + FFN ---
        out_dict: Dict[str, torch.Tensor] = {}
        for ntype, h in x_dict.items():
            if ntype in msg_accum:
                h = self.norm_attn[ntype](h + self.ffn_dropout(msg_accum[ntype]))
                h = self.norm_ffn[ntype](h + self.ffn_dropout(self.ffn[ntype](h)))
            out_dict[ntype] = h

        return out_dict


# ======================================================================
# BioHGT -- full stacked model
# ======================================================================


class BioHGT(BaseKGEModel):
    """Biology-Aware Heterogeneous Graph Transformer.

    Stacks multiple :class:`BioHGTLayer` modules for message passing over a
    heterogeneous knowledge graph and provides a DistMult-style scoring head
    for link prediction.

    Parameters
    ----------
    num_nodes_dict : Dict[str, int]
        ``{node_type: count}`` -- inherited from :class:`BaseKGEModel`.
    num_relations : int
        Total number of distinct relation types.
    embedding_dim : int
        Unified embedding dimensionality (default 256).
    num_layers : int
        Number of stacked BioHGT layers (default 4).
    num_heads : int
        Attention heads per layer (default 8).
    node_types : List[str] | None
        Explicit list of node type names.  If *None*, derived from
        ``num_nodes_dict`` keys.
    edge_types : List[Tuple[str, str, str]] | None
        Explicit canonical edge types.  If *None*, uses
        :data:`DEFAULT_EDGE_TYPES`.
    use_bio_prior : bool
        Enable biology-aware attention biases (default True).
    dropout : float
        Dropout rate (default 0.1).
    """

    def __init__(
        self,
        num_nodes_dict: Dict[str, int],
        num_relations: int,
        embedding_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        node_types: Optional[List[str]] = None,
        edge_types: Optional[List[Tuple[str, str, str]]] = None,
        use_bio_prior: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(num_nodes_dict, num_relations, embedding_dim)

        resolved_node_types = node_types or sorted(num_nodes_dict.keys())
        resolved_edge_types = edge_types or DEFAULT_EDGE_TYPES

        self.num_layers = num_layers
        self.resolved_node_types = resolved_node_types
        self.resolved_edge_types = resolved_edge_types

        # Input projection: from embedding_dim to embedding_dim (identity
        # compatible, but allows gradient isolation from the embedding table).
        self.input_proj = nn.ModuleDict(
            {nt: nn.Linear(embedding_dim, embedding_dim) for nt in resolved_node_types}
        )

        # Stacked BioHGT layers.
        self.layers = nn.ModuleList(
            [
                BioHGTLayer(
                    in_dim=embedding_dim,
                    out_dim=embedding_dim,
                    num_heads=num_heads,
                    node_types=resolved_node_types,
                    edge_types=resolved_edge_types,
                    use_bio_prior=use_bio_prior,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    # ------------------------------------------------------------------
    # BaseKGEModel interface
    # ------------------------------------------------------------------

    def forward(
        self,
        data: HeteroData,
        *,
        bio_prior_kwargs: Optional[
            Dict[Tuple[str, str, str], dict]
        ] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute node embeddings via BioHGT message passing.

        Parameters
        ----------
        data : HeteroData
            PyG heterogeneous graph containing node features
            (``data[node_type].x``) and edge indices
            (``data[src, rel, dst].edge_index``).
        bio_prior_kwargs : dict, optional
            Forwarded to each :class:`BioHGTLayer`.

        Returns
        -------
        Dict[str, Tensor]
            ``{node_type: Tensor[N_type, embedding_dim]}``
        """
        # --- Build x_dict ---
        x_dict: Dict[str, torch.Tensor] = {}
        for ntype in self.resolved_node_types:
            if ntype not in data.node_types:
                continue
            store = data[ntype]
            if hasattr(store, "x") and store.x is not None:
                # External features provided (e.g. from GlycanTreeEncoder).
                x_dict[ntype] = self.input_proj[ntype](store.x)
            else:
                # Fall back to learnable embeddings.
                num_n = self.num_nodes_dict.get(ntype, 0)
                if num_n == 0:
                    continue
                device = next(self.parameters()).device
                idx = torch.arange(num_n, device=device)
                x_dict[ntype] = self.input_proj[ntype](
                    self.node_embeddings[ntype](idx)
                )

        # --- Build edge_index_dict ---
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
        for et in data.edge_types:
            edge_index_dict[et] = data[et].edge_index

        # --- Message passing ---
        for layer in self.layers:
            x_dict = layer(
                x_dict, edge_index_dict, bio_prior_kwargs=bio_prior_kwargs
            )

        return x_dict

    def score(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """DistMult-style scoring: ``<h, r, t>``.

        Parameters
        ----------
        head : Tensor[B, dim]
        relation : Tensor[B, dim]
        tail : Tensor[B, dim]

        Returns
        -------
        Tensor[B]
        """
        return (head * relation * tail).sum(dim=-1)


