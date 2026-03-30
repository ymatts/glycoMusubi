"""Train / validation / test link-split utilities for HeteroData.

Includes inverse-relation leak prevention: when an ``inverse_relation_map``
is provided, triples whose inverse appears in the training set are moved
from validation / test back to training to avoid information leakage.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _split_edge_index(
    edge_index: torch.Tensor,
    val_ratio: float,
    test_ratio: float,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomly partition ``edge_index`` columns into train / val / test.

    Returns three edge_index tensors.
    """
    num_edges = edge_index.size(1)
    perm = torch.randperm(num_edges, generator=generator)

    num_test = int(num_edges * test_ratio)
    num_val = int(num_edges * val_ratio)

    test_idx = perm[:num_test]
    val_idx = perm[num_test : num_test + num_val]
    train_idx = perm[num_test + num_val :]

    return (
        edge_index[:, train_idx],
        edge_index[:, val_idx],
        edge_index[:, test_idx],
    )


def _build_triple_set(
    data: HeteroData,
) -> Set[Tuple[str, int, str, int, str]]:
    """Build a set of ``(src_type, src_id, rel, dst_id, dst_type)`` tuples."""
    triples: Set[Tuple[str, int, str, int, str]] = set()
    for etype in data.edge_types:
        src_type, rel, dst_type = etype
        ei = data[etype].edge_index
        for col in range(ei.size(1)):
            h = ei[0, col].item()
            t = ei[1, col].item()
            triples.add((src_type, h, rel, t, dst_type))
    return triples


def _remove_inverse_leaks(
    train_data: HeteroData,
    eval_data: HeteroData,
    inverse_relation_map: Dict[str, str],
    split_name: str,
) -> HeteroData:
    """Move leaked inverse triples from *eval_data* back to *train_data*.

    A triple ``(t, r_inv, h)`` in the eval set is a leak if
    ``(h, r, t)`` is in the training set and ``r`` <-> ``r_inv`` are
    inverses according to *inverse_relation_map*.

    Iterates until convergence because moving an eval triple to train
    may introduce new leaks for other eval triples.

    Parameters
    ----------
    train_data : HeteroData
        Training graph (triples are added here when moved).
    eval_data : HeteroData
        Evaluation graph (val or test) to check for leaks.
    inverse_relation_map : dict
        Bidirectional mapping ``{relation: inverse_relation, ...}``.
    split_name : str
        Label for logging (e.g. ``"val"`` or ``"test"``).

    Returns
    -------
    HeteroData
        The eval_data with leaked edges removed (moved to train_data).
    """
    total_moved = 0
    max_iterations = 100  # safety bound

    for iteration in range(max_iterations):
        # Rebuild training triple set each iteration (it grows as we move edges)
        train_triples = _build_triple_set(train_data)
        moved_this_round = 0

        for etype in list(eval_data.edge_types):
            src_type, rel, dst_type = etype
            inv_rel = inverse_relation_map.get(rel)
            if inv_rel is None:
                continue

            ei = eval_data[etype].edge_index
            if ei.size(1) == 0:
                continue

            keep_mask = torch.ones(ei.size(1), dtype=torch.bool)

            for col in range(ei.size(1)):
                h = ei[0, col].item()
                t = ei[1, col].item()
                # Check if inverse (t, inv_rel, h) with swapped node types is in train
                if (dst_type, t, inv_rel, h, src_type) in train_triples:
                    keep_mask[col] = False

            num_leaked = int((~keep_mask).sum().item())
            if num_leaked == 0:
                continue

            moved_this_round += num_leaked

            # Move leaked edges to train (same edge type)
            leaked_ei = ei[:, ~keep_mask]
            if etype in train_data.edge_types and train_data[etype].edge_index.size(1) > 0:
                train_data[etype].edge_index = torch.cat(
                    [train_data[etype].edge_index, leaked_ei], dim=1
                )
            else:
                train_data[etype].edge_index = leaked_ei

            # Keep non-leaked in eval
            eval_data[etype].edge_index = ei[:, keep_mask]

            logger.debug(
                "  [iter %d] Moved %d inverse-leaked edges from %s to train (etype=%s, inv=%s)",
                iteration,
                num_leaked,
                split_name,
                etype,
                inv_rel,
            )

        total_moved += moved_this_round
        if moved_this_round == 0:
            break

    if total_moved > 0:
        logger.info(
            "Inverse leak prevention: moved %d edges from %s to train.",
            total_moved,
            split_name,
        )

    return eval_data


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def check_inverse_leak(
    train_data: HeteroData,
    val_data: HeteroData,
    test_data: HeteroData,
    inverse_relation_map: Dict[str, str],
) -> Dict[str, int]:
    """Check for inverse-relation leaks between train and val/test sets.

    A leak exists when ``(h, r, t)`` is in train and ``(t, r_inv, h)``
    appears in val or test.

    Parameters
    ----------
    train_data : HeteroData
        Training graph.
    val_data : HeteroData
        Validation graph.
    test_data : HeteroData
        Test graph.
    inverse_relation_map : dict
        Bidirectional mapping ``{relation: inverse_relation, ...}``.

    Returns
    -------
    dict
        ``{"val_leaks": <int>, "test_leaks": <int>}`` with counts of
        leaked triples in each split.
    """
    train_triples = _build_triple_set(train_data)
    result: Dict[str, int] = {"val_leaks": 0, "test_leaks": 0}

    for split_name, split_data in [("val", val_data), ("test", test_data)]:
        leak_count = 0
        for etype in split_data.edge_types:
            src_type, rel, dst_type = etype
            inv_rel = inverse_relation_map.get(rel)
            if inv_rel is None:
                continue

            ei = split_data[etype].edge_index
            for col in range(ei.size(1)):
                h = ei[0, col].item()
                t = ei[1, col].item()
                if (dst_type, t, inv_rel, h, src_type) in train_triples:
                    leak_count += 1

        result[f"{split_name}_leaks"] = leak_count

    if result["val_leaks"] > 0 or result["test_leaks"] > 0:
        logger.warning(
            "Inverse relation leaks detected: val=%d, test=%d",
            result["val_leaks"],
            result["test_leaks"],
        )
    else:
        logger.info("No inverse relation leaks detected.")

    return result


def random_link_split(
    data: HeteroData,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
    inverse_relation_map: Optional[Dict[str, str]] = None,
) -> Tuple[HeteroData, HeteroData, HeteroData]:
    """Split every edge type in *data* into train / val / test sets.

    Node features and ``num_nodes`` are shared across all three returned
    :class:`HeteroData` objects.  Only the ``edge_index`` tensors differ.

    Parameters
    ----------
    data : HeteroData
        The full heterogeneous graph.
    val_ratio : float
        Fraction of edges reserved for validation.
    test_ratio : float
        Fraction of edges reserved for test.
    seed : int
        Random seed for reproducibility.
    inverse_relation_map : dict, optional
        Bidirectional mapping ``{relation: inverse_relation, ...}``.
        When provided, triples whose inverse is in the training set are
        moved from val/test back to train to prevent information leakage.

    Returns
    -------
    train_data, val_data, test_data : HeteroData
    """
    if not (0 < val_ratio + test_ratio < 1):
        raise ValueError("val_ratio + test_ratio must be in (0, 1)")

    generator = torch.Generator().manual_seed(seed)

    train_data = HeteroData()
    val_data = HeteroData()
    test_data = HeteroData()

    # Copy node stores (shared, not duplicated in memory)
    for ntype in data.node_types:
        for key, val in data[ntype].items():
            train_data[ntype][key] = val
            val_data[ntype][key] = val
            test_data[ntype][key] = val

    total_train = 0
    total_val = 0
    total_test = 0

    for etype in data.edge_types:
        ei = data[etype].edge_index
        train_ei, val_ei, test_ei = _split_edge_index(
            ei, val_ratio, test_ratio, generator
        )
        train_data[etype].edge_index = train_ei
        val_data[etype].edge_index = val_ei
        test_data[etype].edge_index = test_ei

        total_train += train_ei.size(1)
        total_val += val_ei.size(1)
        total_test += test_ei.size(1)

    # Inverse-relation leak prevention
    if inverse_relation_map:
        val_data = _remove_inverse_leaks(
            train_data, val_data, inverse_relation_map, "val"
        )
        test_data = _remove_inverse_leaks(
            train_data, test_data, inverse_relation_map, "test"
        )

    logger.info(
        "random_link_split: train=%d, val=%d, test=%d (seed=%d)",
        sum(train_data[e].edge_index.size(1) for e in train_data.edge_types),
        sum(val_data[e].edge_index.size(1) for e in val_data.edge_types),
        sum(test_data[e].edge_index.size(1) for e in test_data.edge_types),
        seed,
    )

    return train_data, val_data, test_data


def relation_stratified_split(
    data: HeteroData,
    val_ratio: float = 0.05,
    test_ratio: float = 0.10,
    seed: int = 42,
    inverse_relation_map: Optional[Dict[str, str]] = None,
) -> Tuple[HeteroData, HeteroData, HeteroData]:
    """Per-relation stratified split ensuring every relation has proportional representation.

    This is identical to :func:`random_link_split` in behaviour because
    each edge type is already split independently.  The function is provided
    as a distinct entry point for clarity and potential future extension
    (e.g., ensuring minimum counts per relation).

    Edge types with fewer than ``ceil(1 / min(val_ratio, test_ratio))``
    edges will have at least one edge in each split to avoid empty sets.

    Parameters
    ----------
    data : HeteroData
    val_ratio, test_ratio, seed
        Same semantics as :func:`random_link_split`.
    inverse_relation_map : dict, optional
        Same semantics as :func:`random_link_split`.

    Returns
    -------
    train_data, val_data, test_data : HeteroData
    """
    if not (0 < val_ratio + test_ratio < 1):
        raise ValueError("val_ratio + test_ratio must be in (0, 1)")

    generator = torch.Generator().manual_seed(seed)

    train_data = HeteroData()
    val_data = HeteroData()
    test_data = HeteroData()

    # Copy node stores
    for ntype in data.node_types:
        for key, val in data[ntype].items():
            train_data[ntype][key] = val
            val_data[ntype][key] = val
            test_data[ntype][key] = val

    for etype in data.edge_types:
        ei = data[etype].edge_index
        num_edges = ei.size(1)

        if num_edges == 0:
            train_data[etype].edge_index = ei
            val_data[etype].edge_index = ei
            test_data[etype].edge_index = ei
            continue

        # Ensure at least 1 edge in each split when there are >= 3 edges
        num_test = max(1, int(num_edges * test_ratio)) if num_edges >= 3 else 0
        num_val = max(1, int(num_edges * val_ratio)) if num_edges >= 3 else 0

        # Clamp so train always gets at least 1 edge
        if num_test + num_val >= num_edges:
            num_test = max(1, num_edges // 3)
            num_val = max(1, num_edges // 3)
            if num_test + num_val >= num_edges:
                num_val = 0
                num_test = 0

        perm = torch.randperm(num_edges, generator=generator)
        test_idx = perm[:num_test]
        val_idx = perm[num_test : num_test + num_val]
        train_idx = perm[num_test + num_val :]

        train_data[etype].edge_index = ei[:, train_idx]
        val_data[etype].edge_index = ei[:, val_idx]
        test_data[etype].edge_index = ei[:, test_idx]

        logger.debug(
            "  etype=%s: train=%d val=%d test=%d",
            etype,
            train_idx.numel(),
            val_idx.numel(),
            test_idx.numel(),
        )

    # Inverse-relation leak prevention
    if inverse_relation_map:
        val_data = _remove_inverse_leaks(
            train_data, val_data, inverse_relation_map, "val"
        )
        test_data = _remove_inverse_leaks(
            train_data, test_data, inverse_relation_map, "test"
        )

    return train_data, val_data, test_data
