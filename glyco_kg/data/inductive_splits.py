"""Inductive entity-level splits for zero-shot link prediction.

Splits entities (not edges) into train and hold-out sets, enabling
evaluation of a model's ability to embed unseen entities using only
their features (WURCS structures, ESM-2 embeddings, etc.).

Hold-out entities are selected from node types that have biological
features (glycan, protein) and are stratified to ensure participation
in target relations (e.g., has_glycan).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)


@dataclass
class InductiveSplit:
    """Container for inductive entity split information.

    Attributes
    ----------
    train_entity_ids : dict[str, set[int]]
        Entity indices kept for training, per node type.
    holdout_entity_ids : dict[str, set[int]]
        Entity indices held out for inductive evaluation, per node type.
    train_data : HeteroData
        Graph with only training entities and their edges.
    inductive_triples : list[tuple[str, int, str, int, str]]
        Triples involving held-out entities:
        ``(src_type, src_idx, relation, dst_idx, dst_type)``
    stats : dict[str, int]
        Summary statistics.
    """

    train_entity_ids: Dict[str, Set[int]] = field(default_factory=dict)
    holdout_entity_ids: Dict[str, Set[int]] = field(default_factory=dict)
    train_data: Optional[HeteroData] = None
    inductive_triples: List[Tuple[str, int, str, int, str]] = field(
        default_factory=list
    )
    stats: Dict[str, int] = field(default_factory=dict)


def _compute_degree(
    data: HeteroData, node_type: str
) -> Dict[int, int]:
    """Compute degree (number of edges) for each node of a given type."""
    degree: Dict[int, int] = defaultdict(int)
    for etype in data.edge_types:
        src_type, rel, dst_type = etype
        ei = data[etype].edge_index
        if src_type == node_type:
            for i in range(ei.size(1)):
                degree[ei[0, i].item()] += 1
        if dst_type == node_type:
            for i in range(ei.size(1)):
                degree[ei[1, i].item()] += 1
    return dict(degree)


def _get_relation_participants(
    data: HeteroData,
    node_type: str,
    target_relations: Optional[List[str]] = None,
) -> Set[int]:
    """Get entity indices that participate in target relations."""
    participants: Set[int] = set()
    for etype in data.edge_types:
        src_type, rel, dst_type = etype
        if target_relations and rel not in target_relations:
            continue
        ei = data[etype].edge_index
        if src_type == node_type:
            participants.update(ei[0].tolist())
        if dst_type == node_type:
            participants.update(ei[1].tolist())
    return participants


def create_inductive_split(
    data: HeteroData,
    holdout_ratio: float = 0.15,
    min_degree: int = 2,
    holdout_node_types: Optional[List[str]] = None,
    target_relations: Optional[List[str]] = None,
    seed: int = 42,
) -> InductiveSplit:
    """Create an entity-level inductive split.

    Parameters
    ----------
    data : HeteroData
        The full heterogeneous graph.
    holdout_ratio : float
        Fraction of eligible entities to hold out per node type.
    min_degree : int
        Minimum degree for an entity to be eligible for hold-out.
    holdout_node_types : list of str, optional
        Node types to hold out (default: ``["glycan", "protein"]``).
    target_relations : list of str, optional
        If provided, only entities participating in these relations
        are eligible for hold-out (ensures evaluation coverage).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    InductiveSplit
        The split with train graph and inductive triples.
    """
    if holdout_node_types is None:
        holdout_node_types = ["glycan", "protein"]

    generator = torch.Generator().manual_seed(seed)

    split = InductiveSplit()
    total_holdout = 0
    total_train_entities = 0

    # 1. Select hold-out entities
    for ntype in data.node_types:
        num_nodes = data[ntype].num_nodes
        if ntype not in holdout_node_types:
            split.train_entity_ids[ntype] = set(range(num_nodes))
            split.holdout_entity_ids[ntype] = set()
            total_train_entities += num_nodes
            continue

        # Compute degree for each entity
        degree = _compute_degree(data, ntype)

        # Get entities participating in target relations
        if target_relations:
            participants = _get_relation_participants(
                data, ntype, target_relations
            )
        else:
            participants = set(range(num_nodes))

        # Filter: degree >= min_degree AND participates in target relations
        eligible = [
            idx
            for idx in range(num_nodes)
            if degree.get(idx, 0) >= min_degree and idx in participants
        ]

        num_holdout = max(1, int(len(eligible) * holdout_ratio))
        # Don't hold out more than half the eligible nodes
        num_holdout = min(num_holdout, len(eligible) // 2)

        if num_holdout == 0 or len(eligible) == 0:
            logger.warning(
                "No eligible entities for hold-out in type '%s' "
                "(eligible=%d, min_degree=%d)",
                ntype,
                len(eligible),
                min_degree,
            )
            split.train_entity_ids[ntype] = set(range(num_nodes))
            split.holdout_entity_ids[ntype] = set()
            total_train_entities += num_nodes
            continue

        # Random selection
        perm = torch.randperm(len(eligible), generator=generator)
        holdout_indices = {eligible[perm[i].item()] for i in range(num_holdout)}
        train_indices = set(range(num_nodes)) - holdout_indices

        split.train_entity_ids[ntype] = train_indices
        split.holdout_entity_ids[ntype] = holdout_indices
        total_holdout += len(holdout_indices)
        total_train_entities += len(train_indices)

        logger.info(
            "Inductive split for '%s': %d train, %d holdout (from %d eligible, %d total)",
            ntype,
            len(train_indices),
            len(holdout_indices),
            len(eligible),
            num_nodes,
        )

    # 2. Separate edges into train edges and inductive triples
    train_data = HeteroData()

    # Copy node stores
    for ntype in data.node_types:
        for key, val in data[ntype].items():
            train_data[ntype][key] = val

    total_train_edges = 0
    total_inductive_edges = 0

    for etype in data.edge_types:
        src_type, rel, dst_type = etype
        ei = data[etype].edge_index
        holdout_src = split.holdout_entity_ids.get(src_type, set())
        holdout_dst = split.holdout_entity_ids.get(dst_type, set())

        train_mask = torch.ones(ei.size(1), dtype=torch.bool)

        for col in range(ei.size(1)):
            src = ei[0, col].item()
            dst = ei[1, col].item()
            if src in holdout_src or dst in holdout_dst:
                train_mask[col] = False
                split.inductive_triples.append(
                    (src_type, src, rel, dst, dst_type)
                )

        train_data[etype].edge_index = ei[:, train_mask]
        total_train_edges += train_mask.sum().item()
        total_inductive_edges += (~train_mask).sum().item()

    split.train_data = train_data
    split.stats = {
        "total_train_entities": total_train_entities,
        "total_holdout_entities": total_holdout,
        "total_train_edges": total_train_edges,
        "total_inductive_triples": total_inductive_edges,
        "holdout_node_types": len(holdout_node_types),
    }

    logger.info(
        "Inductive split complete: %d train entities, %d holdout entities, "
        "%d train edges, %d inductive triples",
        total_train_entities,
        total_holdout,
        total_train_edges,
        total_inductive_edges,
    )

    return split


def inductive_triples_to_tensor(
    split: InductiveSplit,
    edge_type_to_idx: Dict[Tuple[str, str, str], int],
    node_type_offsets: Dict[str, int],
) -> torch.Tensor:
    """Convert inductive triples to a global [N, 3] tensor.

    Maps heterogeneous (typed) triples to a global entity space using
    the same node_type_offsets used during training.

    Parameters
    ----------
    split : InductiveSplit
        The inductive split containing typed triples.
    edge_type_to_idx : dict
        Mapping from ``(src_type, rel, dst_type)`` to relation index.
    node_type_offsets : dict
        Mapping from node type to global offset.

    Returns
    -------
    torch.Tensor
        Shape ``[N, 3]`` with columns ``(head_global, relation_idx, tail_global)``.
    """
    rows = []
    skipped = 0
    for src_type, src_idx, rel, dst_idx, dst_type in split.inductive_triples:
        etype = (src_type, rel, dst_type)
        if etype not in edge_type_to_idx:
            skipped += 1
            continue
        rel_idx = edge_type_to_idx[etype]
        h_global = node_type_offsets[src_type] + src_idx
        t_global = node_type_offsets[dst_type] + dst_idx
        rows.append([h_global, rel_idx, t_global])

    if skipped > 0:
        logger.warning(
            "Skipped %d inductive triples with unknown edge types", skipped
        )

    if not rows:
        return torch.zeros(0, 3, dtype=torch.long)
    return torch.tensor(rows, dtype=torch.long)
