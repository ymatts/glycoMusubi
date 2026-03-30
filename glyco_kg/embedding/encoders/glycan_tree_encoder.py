"""Tree-MPNN based glycan encoder.

Encodes glycan branching tree structures into fixed-dimensional embeddings
using a hierarchical message-passing neural network that respects parent-child
and sibling relationships.

Architecture (from model_architecture_design.md Section 3.1)::

    1. Initial node features: mono_type(d=32) + anomeric(d=4) + ring(d=4) + mods(d=16) = d=56
    2. 3 layers bottom-up Tree-MPNN:
       - Children aggregation with attention
       - Sibling aggregation
       - GRU update
    3. 1 layer top-down refinement
    4. Branching-Aware Attention Pooling (4 heads)
    5. Output: d=256

Approximate parameter count: ~1.2M
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from glycoMusubi.embedding.encoders.wurcs_tree_parser import (
    NUM_MODIFICATIONS,
    NUM_MONO_TYPES,
    GlycanTree,
    glycan_tree_to_tensors,
    parse_wurcs_to_tree,
)
from glycoMusubi.utils.scatter import scatter_softmax

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Edge feature encoder
# -----------------------------------------------------------------------

class LinkageEncoder(nn.Module):
    """Encode glycosidic bond features into a fixed-dimensional vector.

    Edge features: parent_carbon (one-hot 7) + child_carbon (one-hot 7)
                 + bond_type (one-hot 3) = 17 -> projected to ``d_edge``.
    """

    _NUM_CARBONS: int = 7  # 0-6 (0 = unknown/ambiguous)
    _NUM_BOND_TYPES: int = 3  # alpha, beta, unknown

    def __init__(self, d_edge: int = 24) -> None:
        super().__init__()
        raw_dim = self._NUM_CARBONS * 2 + self._NUM_BOND_TYPES  # 17
        self.proj = nn.Linear(raw_dim, d_edge)

    def forward(
        self,
        linkage_parent_carbon: torch.Tensor,
        linkage_child_carbon: torch.Tensor,
        bond_type: torch.Tensor,
    ) -> torch.Tensor:
        """Encode edge features.

        Parameters
        ----------
        linkage_parent_carbon, linkage_child_carbon:
            Long tensors of shape ``[E]`` with carbon positions (0-6).
        bond_type:
            Long tensor of shape ``[E]`` with bond type indices (0-2).

        Returns
        -------
        Tensor of shape ``[E, d_edge]``.
        """
        pc = F.one_hot(
            linkage_parent_carbon.clamp(0, self._NUM_CARBONS - 1),
            self._NUM_CARBONS,
        ).float()
        cc = F.one_hot(
            linkage_child_carbon.clamp(0, self._NUM_CARBONS - 1),
            self._NUM_CARBONS,
        ).float()
        bt = F.one_hot(
            bond_type.clamp(0, self._NUM_BOND_TYPES - 1),
            self._NUM_BOND_TYPES,
        ).float()
        raw = torch.cat([pc, cc, bt], dim=-1)
        return self.proj(raw)


# -----------------------------------------------------------------------
# Tree-MPNN Layer (bottom-up)
# -----------------------------------------------------------------------

class TreeMPNNLayer(nn.Module):
    """A single layer of bottom-up Tree Message Passing.

    For each node *v*:

    1. **Child aggregation** with attention::

        MSG(h_c, h_v, e_vc) = MLP_child([h_c || e_vc])
        alpha_c = softmax_over_children( MLP_attn([h_c || h_v]) )
        m_v = sum_c alpha_c * MSG(h_c, h_v, e_vc)

    2. **Sibling aggregation**::

        s_v = MLP_sibling( mean({h_s : s in siblings(v)}) )

    3. **GRU update**::

        h_v^(l+1) = GRU(h_v^l, [m_v || s_v])
    """

    def __init__(self, d_model: int, d_edge: int = 24) -> None:
        super().__init__()
        self.d_model = d_model

        # Child message MLP: takes child hidden + edge features
        self.child_msg = nn.Sequential(
            nn.Linear(d_model + d_edge, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Attention scorer for children
        self.child_attn = nn.Sequential(
            nn.Linear(2 * d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1),
        )

        # Sibling MLP
        self.sibling_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # GRU update: input is [child_agg || sibling_agg] (2*d_model)
        self.gru = nn.GRUCell(2 * d_model, d_model)

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        parent_map: Dict[int, int],
        children_map: Dict[int, List[int]],
        topo_order_bu: List[int],
    ) -> torch.Tensor:
        """Run one bottom-up Tree-MPNN layer.

        Parameters
        ----------
        h:
            Node hidden states ``[N, d_model]``.
        edge_index:
            ``[2, E]`` directed edges (parent -> child).
        edge_attr:
            ``[E, d_edge]`` edge features.
        parent_map:
            Mapping from child index -> parent index.
        children_map:
            Mapping from parent index -> list of (child_idx, edge_idx) tuples.
        topo_order_bu:
            Node indices in bottom-up order (leaves first).

        Returns
        -------
        Updated hidden states ``[N, d_model]``.
        """
        n = h.size(0)
        device = h.device

        # Pre-compute all child messages and attention scores
        # child_msg_all[j] = MLP_child([h_child || edge_attr]) for edge j
        if edge_index.size(1) > 0:
            child_indices = edge_index[1]  # child nodes
            child_h = h[child_indices]  # [E, d_model]
            parent_indices = edge_index[0]
            parent_h = h[parent_indices]

            msgs = self.child_msg(torch.cat([child_h, edge_attr], dim=-1))  # [E, d_model]
            attn_input = torch.cat([child_h, parent_h], dim=-1)  # [E, 2*d_model]
            attn_scores = self.child_attn(attn_input)  # [E, 1]
        else:
            msgs = torch.zeros(0, self.d_model, device=device)
            attn_scores = torch.zeros(0, 1, device=device)

        # Aggregate messages per parent node using scatter
        # For each parent, compute attention-weighted sum over children
        child_agg = torch.zeros(n, self.d_model, device=device)
        sibling_agg = torch.zeros(n, self.d_model, device=device)

        if edge_index.size(1) > 0:
            parent_ids = edge_index[0]  # [E]

            # Softmax attention per parent using shared utility
            attn_weights = scatter_softmax(
                attn_scores.squeeze(-1), parent_ids, num_nodes=n,
            ).unsqueeze(-1)  # [E, 1]

            weighted_msgs = attn_weights * msgs  # [E, d_model]
            child_agg.scatter_add_(
                0,
                parent_ids.unsqueeze(1).expand(-1, self.d_model),
                weighted_msgs,
            )

            # Sibling aggregation: for each node, compute mean of siblings' h
            # siblings(v) = children(parent(v)) \ {v}
            # Efficient approach: for each node with a parent, sum children h
            # at parent level, then for node v: sibling_sum = children_sum[parent(v)] - h[v]
            children_sum = torch.zeros(n, self.d_model, device=device)
            children_count = torch.zeros(n, 1, device=device)
            child_ids = edge_index[1]
            children_sum.scatter_add_(
                0,
                parent_ids.unsqueeze(1).expand(-1, self.d_model),
                h[child_ids],
            )
            children_count.scatter_add_(
                0,
                parent_ids.unsqueeze(1),
                torch.ones(parent_ids.size(0), 1, device=device),
            )

            # For each child node, sibling features = (children_sum[parent] - h[child]) / (count - 1)
            for edge_j in range(edge_index.size(1)):
                p = parent_ids[edge_j].item()
                c = child_ids[edge_j].item()
                cnt = children_count[p, 0].item()
                if cnt > 1:
                    sib_mean = (children_sum[p] - h[c]) / (cnt - 1)
                    sibling_agg[c] = sib_mean

        sibling_feat = self.sibling_mlp(sibling_agg)  # [N, d_model]

        # GRU update
        gru_input = torch.cat([child_agg, sibling_feat], dim=-1)  # [N, 2*d_model]
        h_new = self.gru(gru_input, h)  # [N, d_model]

        return self.norm(h_new)


# -----------------------------------------------------------------------
# Top-down refinement
# -----------------------------------------------------------------------

class TopDownRefinement(nn.Module):
    """Single top-down refinement pass (root -> leaves).

    For each node v::

        h_v^refined = MLP([h_v^bu || h_parent^refined])

    The root uses a zero vector for the parent embedding.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        h: torch.Tensor,
        topo_order_td: List[int],
        parent_map: Dict[int, int],
    ) -> torch.Tensor:
        """Apply top-down refinement.

        Parameters
        ----------
        h:
            Bottom-up hidden states ``[N, d_model]``.
        topo_order_td:
            Node indices in top-down order (root first).
        parent_map:
            Mapping from child index -> parent index.

        Returns
        -------
        Refined hidden states ``[N, d_model]``.
        """
        device = h.device
        d_model = h.size(1)
        h_refined = torch.zeros_like(h)

        for v in topo_order_td:
            parent = parent_map.get(v)
            if parent is None:
                # Root node: use zero vector as parent context
                h_parent = torch.zeros(d_model, device=device)
            else:
                h_parent = h_refined[parent]

            h_refined[v] = self.mlp(torch.cat([h[v], h_parent], dim=-1))

        return self.norm(h_refined)


# -----------------------------------------------------------------------
# Branching-Aware Attention Pooling
# -----------------------------------------------------------------------

class BranchingAwarePooling(nn.Module):
    """Branching-Aware Attention Pooling with multiple heads.

    Computes a graph-level embedding from node embeddings by:

    1. Multi-head attention pooling over all nodes.
    2. Mean pooling over branching nodes.
    3. Depth encoding from max tree depth.
    4. Fusing all components.
    """

    def __init__(
        self,
        d_model: int = 256,
        output_dim: int = 256,
        num_heads: int = 4,
        max_depth: int = 32,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Multi-head attention
        self.attn_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, self.d_head),
                    nn.Tanh(),
                    nn.Linear(self.d_head, 1),
                )
                for _ in range(num_heads)
            ]
        )
        self.head_proj = nn.Linear(num_heads * d_model, d_model)

        # Depth encoding
        self.depth_embed = nn.Embedding(max_depth, 8)
        self.max_depth = max_depth

        # Branch fuse: [h_global || h_branch || depth_enc]
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 2 + 8, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        h: torch.Tensor,
        batch: torch.Tensor,
        is_branch: torch.Tensor,
        depth: torch.Tensor,
    ) -> torch.Tensor:
        """Pool node embeddings into graph-level embeddings.

        Parameters
        ----------
        h:
            Node hidden states ``[total_nodes, d_model]``.
        batch:
            Batch assignment ``[total_nodes]`` (which graph each node belongs to).
        is_branch:
            Boolean mask ``[total_nodes]`` for branching nodes.
        depth:
            Depth of each node ``[total_nodes]``.

        Returns
        -------
        Graph-level embeddings ``[B, output_dim]``.
        """
        device = h.device
        num_graphs = batch.max().item() + 1 if batch.numel() > 0 else 1

        # --- Multi-head attention pooling ---
        head_outputs = []
        for head in self.attn_heads:
            scores = head(h)  # [total_nodes, 1]

            # Scatter-based softmax per graph using shared utility
            attn = scatter_softmax(
                scores.squeeze(-1), batch, num_nodes=num_graphs,
            ).unsqueeze(-1)

            weighted = attn * h  # [total_nodes, d_model]
            pooled = torch.zeros(num_graphs, h.size(1), device=device)
            pooled.scatter_add_(
                0,
                batch.unsqueeze(1).expand(-1, h.size(1)),
                weighted,
            )
            head_outputs.append(pooled)

        h_global = self.head_proj(torch.cat(head_outputs, dim=-1))  # [B, d_model]

        # --- Branch point features ---
        h_branch = torch.zeros(num_graphs, h.size(1), device=device)
        branch_count = torch.zeros(num_graphs, 1, device=device)

        if is_branch.any():
            branch_mask = is_branch.bool()
            branch_h = h[branch_mask]
            branch_batch = batch[branch_mask]
            h_branch.scatter_add_(
                0,
                branch_batch.unsqueeze(1).expand(-1, h.size(1)),
                branch_h,
            )
            branch_count.scatter_add_(
                0,
                branch_batch.unsqueeze(1),
                torch.ones(branch_batch.size(0), 1, device=device),
            )
            h_branch = h_branch / (branch_count + 1e-8)

        # --- Depth encoding ---
        # Use max depth per graph
        max_depths = torch.zeros(num_graphs, dtype=torch.long, device=device)
        max_depths.scatter_reduce_(
            0,
            batch,
            depth,
            reduce="amax",
            include_self=True,
        )
        max_depths = max_depths.clamp(0, self.max_depth - 1)
        depth_enc = self.depth_embed(max_depths)  # [B, 8]

        # --- Fuse ---
        fused = torch.cat([h_global, h_branch, depth_enc], dim=-1)
        return self.norm(self.fuse(fused))


# -----------------------------------------------------------------------
# GlycanTreeEncoder
# -----------------------------------------------------------------------

class GlycanTreeEncoder(nn.Module):
    """Tree-MPNN based glycan encoder.

    Encodes the branching tree structure of glycans from WURCS strings
    into fixed-dimensional embeddings that capture both local monosaccharide
    context and global topology.

    Parameters
    ----------
    output_dim:
        Dimensionality of the output glycan embedding.
    hidden_dim:
        Internal hidden dimension for message passing.
    num_mono_types:
        Size of the monosaccharide type vocabulary.
    num_bottom_up_layers:
        Number of bottom-up Tree-MPNN layers.
    num_attention_heads:
        Number of attention heads in the pooling layer.
    dropout:
        Dropout probability.
    """

    def __init__(
        self,
        output_dim: int = 256,
        hidden_dim: int = 256,
        num_mono_types: int = NUM_MONO_TYPES,
        num_bottom_up_layers: int = 3,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # --- Node feature embeddings ---
        # mono_type(d=32) + anomeric(d=4) + ring(d=4) + mods(d=16) = d=56
        self.mono_embed = nn.Embedding(num_mono_types, 32)
        self.anomeric_embed = nn.Embedding(3, 4)  # alpha, beta, unknown
        self.ring_embed = nn.Embedding(4, 4)  # pyranose, furanose, open, unknown
        self.mod_proj = nn.Linear(NUM_MODIFICATIONS, 16)

        # Input projection: 56 -> hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(56, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # --- Edge encoder ---
        self.d_edge = 24
        self.edge_encoder = LinkageEncoder(d_edge=self.d_edge)

        # --- Bottom-up Tree-MPNN layers ---
        self.bu_layers = nn.ModuleList(
            [TreeMPNNLayer(hidden_dim, d_edge=self.d_edge) for _ in range(num_bottom_up_layers)]
        )

        # --- Top-down refinement ---
        self.td_layer = TopDownRefinement(hidden_dim)

        # --- Branching-Aware Attention Pooling ---
        self.pooling = BranchingAwarePooling(
            d_model=hidden_dim,
            output_dim=output_dim,
            num_heads=num_attention_heads,
        )

        # Initialise weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    # Batch preparation
    # ------------------------------------------------------------------

    def _prepare_batch(
        self, trees: List[GlycanTree]
    ) -> Dict[str, torch.Tensor]:
        """Convert a list of GlycanTree into a batched tensor dict.

        Concatenates all node/edge tensors and creates a ``batch`` vector
        for graph-level operations (similar to PyG batching).
        """
        all_tensors: List[Dict[str, torch.Tensor]] = []
        for tree in trees:
            all_tensors.append(glycan_tree_to_tensors(tree))

        if not all_tensors:
            device = next(self.parameters()).device
            return {
                "mono_type": torch.zeros(0, dtype=torch.long, device=device),
                "anomeric": torch.zeros(0, dtype=torch.long, device=device),
                "ring_form": torch.zeros(0, dtype=torch.long, device=device),
                "modifications": torch.zeros(0, NUM_MODIFICATIONS, device=device),
                "edge_index": torch.zeros(2, 0, dtype=torch.long, device=device),
                "linkage_parent_carbon": torch.zeros(0, dtype=torch.long, device=device),
                "linkage_child_carbon": torch.zeros(0, dtype=torch.long, device=device),
                "bond_type": torch.zeros(0, dtype=torch.long, device=device),
                "depth": torch.zeros(0, dtype=torch.long, device=device),
                "is_branch": torch.zeros(0, dtype=torch.bool, device=device),
                "batch": torch.zeros(0, dtype=torch.long, device=device),
            }

        # Concatenate with node-index offsets
        node_offset = 0
        mono_types, anomerics, ring_forms, modifications_list = [], [], [], []
        edge_indices, parent_carbons, child_carbons, bond_types = [], [], [], []
        depths, is_branches, batches = [], [], []

        for graph_idx, t in enumerate(all_tensors):
            n = t["num_nodes"]
            mono_types.append(t["mono_type"])
            anomerics.append(t["anomeric"])
            ring_forms.append(t["ring_form"])
            modifications_list.append(t["modifications"])
            depths.append(t["depth"])
            is_branches.append(t["is_branch"])
            batches.append(torch.full((n,), graph_idx, dtype=torch.long))

            # Offset edge indices
            ei = t["edge_index"]
            if ei.size(1) > 0:
                edge_indices.append(ei + node_offset)
                parent_carbons.append(t["linkage_parent_carbon"])
                child_carbons.append(t["linkage_child_carbon"])
                bond_types.append(t["bond_type"])

            node_offset += n

        device = next(self.parameters()).device

        result = {
            "mono_type": torch.cat(mono_types).to(device),
            "anomeric": torch.cat(anomerics).to(device),
            "ring_form": torch.cat(ring_forms).to(device),
            "modifications": torch.cat(modifications_list).to(device),
            "depth": torch.cat(depths).to(device),
            "is_branch": torch.cat(is_branches).to(device),
            "batch": torch.cat(batches).to(device),
        }

        if edge_indices:
            result["edge_index"] = torch.cat(edge_indices, dim=1).to(device)
            result["linkage_parent_carbon"] = torch.cat(parent_carbons).to(device)
            result["linkage_child_carbon"] = torch.cat(child_carbons).to(device)
            result["bond_type"] = torch.cat(bond_types).to(device)
        else:
            result["edge_index"] = torch.zeros(2, 0, dtype=torch.long, device=device)
            result["linkage_parent_carbon"] = torch.zeros(0, dtype=torch.long, device=device)
            result["linkage_child_carbon"] = torch.zeros(0, dtype=torch.long, device=device)
            result["bond_type"] = torch.zeros(0, dtype=torch.long, device=device)

        return result

    @staticmethod
    def _build_tree_maps(
        edge_index: torch.Tensor,
    ) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
        """Build parent_map and children_map from edge_index.

        Returns
        -------
        parent_map:
            {child_idx: parent_idx}
        children_map:
            {parent_idx: [child_idx, ...]}
        """
        parent_map: Dict[int, int] = {}
        children_map: Dict[int, List[int]] = {}

        for j in range(edge_index.size(1)):
            p = edge_index[0, j].item()
            c = edge_index[1, j].item()
            parent_map[c] = p
            children_map.setdefault(p, []).append(c)

        return parent_map, children_map

    @staticmethod
    def _compute_topo_order_bu(
        num_nodes: int,
        children_map: Dict[int, List[int]],
        batch: torch.Tensor,
        root_indices: List[int],
    ) -> List[int]:
        """Compute bottom-up topological order across the whole batch."""
        order: List[int] = []
        visited: set = set()

        def _dfs(idx: int) -> None:
            visited.add(idx)
            for child in children_map.get(idx, []):
                if child not in visited:
                    _dfs(child)
            order.append(idx)

        for root in root_indices:
            if root not in visited:
                _dfs(root)

        # Include any orphan nodes not reachable from roots
        for i in range(num_nodes):
            if i not in visited:
                order.append(i)

        return order

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, trees: List[GlycanTree]) -> torch.Tensor:
        """Encode a batch of glycan trees.

        Parameters
        ----------
        trees:
            List of :class:`GlycanTree` objects.

        Returns
        -------
        Tensor of shape ``[batch_size, output_dim]``.  Returns a zero
        tensor if the input list is empty.
        """
        device = next(self.parameters()).device

        if not trees:
            return torch.zeros(0, self.output_dim, device=device)

        # Filter out empty trees and track which are valid
        valid_indices: List[int] = []
        valid_trees: List[GlycanTree] = []
        for i, tree in enumerate(trees):
            if tree.num_nodes > 0:
                valid_indices.append(i)
                valid_trees.append(tree)

        batch_size = len(trees)

        if not valid_trees:
            return torch.zeros(batch_size, self.output_dim, device=device)

        # Prepare batched tensors
        data = self._prepare_batch(valid_trees)

        # --- Encode node features ---
        h_mono = self.mono_embed(data["mono_type"])  # [N, 32]
        h_anom = self.anomeric_embed(data["anomeric"])  # [N, 4]
        h_ring = self.ring_embed(data["ring_form"])  # [N, 4]
        h_mod = self.mod_proj(data["modifications"])  # [N, 16]

        h = self.input_proj(
            torch.cat([h_mono, h_anom, h_ring, h_mod], dim=-1)
        )  # [N, hidden_dim]

        # --- Encode edge features ---
        edge_index = data["edge_index"]
        if edge_index.size(1) > 0:
            edge_attr = self.edge_encoder(
                data["linkage_parent_carbon"],
                data["linkage_child_carbon"],
                data["bond_type"],
            )  # [E, d_edge]
        else:
            edge_attr = torch.zeros(0, self.d_edge, device=device)

        # Build tree structure maps
        parent_map, children_map = self._build_tree_maps(edge_index)

        # Compute root indices per graph in the batch
        # Roots are nodes that have no parent
        all_nodes = set(range(h.size(0)))
        child_nodes = set(parent_map.keys())
        root_indices = sorted(all_nodes - child_nodes)

        # Compute topological order
        topo_order_bu = self._compute_topo_order_bu(
            h.size(0), children_map, data["batch"], root_indices
        )
        topo_order_td = list(reversed(topo_order_bu))

        # --- Bottom-up Tree-MPNN ---
        for layer in self.bu_layers:
            h = layer(h, edge_index, edge_attr, parent_map, children_map, topo_order_bu)

        # --- Top-down refinement ---
        h = self.td_layer(h, topo_order_td, parent_map)

        # --- Branching-Aware Attention Pooling ---
        h_valid = self.pooling(
            h, data["batch"], data["is_branch"], data["depth"]
        )  # [num_valid, output_dim]

        # Reconstruct full batch (zero vectors for invalid trees)
        if len(valid_indices) == batch_size:
            return h_valid

        output = torch.zeros(batch_size, self.output_dim, device=device)
        for out_idx, orig_idx in enumerate(valid_indices):
            output[orig_idx] = h_valid[out_idx]
        return output

    def encode_wurcs(self, wurcs_list: List[str]) -> torch.Tensor:
        """Convenience method: encode a list of WURCS strings directly.

        Parses each WURCS string into a tree and then encodes the batch.
        Invalid WURCS strings produce zero-vector embeddings.

        Parameters
        ----------
        wurcs_list:
            List of WURCS 2.0 strings.

        Returns
        -------
        Tensor of shape ``[len(wurcs_list), output_dim]``.
        """
        trees: List[GlycanTree] = []
        for wurcs in wurcs_list:
            try:
                tree = parse_wurcs_to_tree(wurcs)
                trees.append(tree)
            except (ValueError, IndexError) as exc:
                logger.debug("Failed to parse WURCS: %s -- %s", wurcs[:60], exc)
                # Create a minimal single-node tree as fallback
                from glycoMusubi.embedding.encoders.wurcs_tree_parser import (
                    MonosaccharideNode,
                )
                fallback = GlycanTree(
                    nodes=[
                        MonosaccharideNode(
                            index=0,
                            wurcs_residue="",
                            mono_type="Unknown",
                            mono_type_idx=0,
                            anomeric="unknown",
                            anomeric_idx=2,
                            ring_form="unknown",
                            ring_form_idx=3,
                            modifications=[],
                        )
                    ],
                    edges=[],
                    root_idx=0,
                )
                trees.append(fallback)

        return self.forward(trees)
