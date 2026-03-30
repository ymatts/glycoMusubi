"""Rank-based evaluation metrics for knowledge graph embeddings.

Implements standard KGE evaluation metrics following the filtered ranking
protocol from Bordes et al. (2013). All rank computations use 1-indexed
ranks and float64 arithmetic for numerical stability.

References
----------
Bordes, A., et al. "Translating Embeddings for Modeling Multi-relational
Data." NeurIPS, 2013.
Ali, M., et al. "Bringing Light Into the Dark: A Large-scale Evaluation
of Knowledge Graph Embedding Models." TPAMI, 2021.
"""

from __future__ import annotations

import torch
from typing import Optional


def compute_ranks(
    scores: torch.Tensor,
    target_idx: torch.Tensor,
    filtered_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the rank of each target entity among all candidates.

    For each query in the batch, the rank of the target entity is defined
    as 1 + the number of candidate entities with a **strictly higher**
    score than the target.  This corresponds to the *optimistic* rank
    convention, which is standard in KGE evaluation.

    Parameters
    ----------
    scores : torch.Tensor
        Shape ``[batch, num_entities]``.  Raw scores for every candidate
        entity, where higher scores indicate greater plausibility.
    target_idx : torch.Tensor
        Shape ``[batch]``.  Integer index of the correct (target) entity
        for each query.
    filtered_mask : torch.Tensor or None
        Shape ``[batch, num_entities]``.  Boolean tensor where ``True``
        marks positions that should be **excluded** from the ranking
        (i.e. other known-true triples that are not the current test
        triple).  The target position itself must **not** be masked.
        If ``None``, unfiltered (raw) ranking is performed.

    Returns
    -------
    torch.Tensor
        Shape ``[batch]``, dtype ``torch.long``.  1-indexed rank of the
        target entity for each query.  Rank 1 is the best.

    Raises
    ------
    ValueError
        If ``scores`` is not 2-D, ``target_idx`` is not 1-D, or their
        batch dimensions disagree.
    """
    if scores.ndim != 2:
        raise ValueError(
            f"scores must be 2-D [batch, num_entities], got shape {scores.shape}"
        )
    if target_idx.ndim != 1:
        raise ValueError(
            f"target_idx must be 1-D [batch], got shape {target_idx.shape}"
        )
    if scores.shape[0] != target_idx.shape[0]:
        raise ValueError(
            f"Batch size mismatch: scores {scores.shape[0]} vs "
            f"target_idx {target_idx.shape[0]}"
        )

    batch_size = scores.shape[0]

    # Gather target scores: [batch]
    target_scores = scores[torch.arange(batch_size, device=scores.device), target_idx]

    # Apply filtered mask: set excluded positions to -inf so they never
    # rank above the target.
    if filtered_mask is not None:
        if filtered_mask.shape != scores.shape:
            raise ValueError(
                f"filtered_mask shape {filtered_mask.shape} must match "
                f"scores shape {scores.shape}"
            )
        scores = scores.clone()
        scores[filtered_mask] = float("-inf")

    # Rank = 1 + number of entities with strictly higher score.
    # Using float64 for the comparison avoids precision issues when many
    # scores are close together.
    higher_count = (
        scores.to(torch.float64) > target_scores.unsqueeze(1).to(torch.float64)
    ).sum(dim=1)

    ranks = (higher_count + 1).long()
    return ranks


def compute_mrr(ranks: torch.Tensor) -> float:
    """Compute Mean Reciprocal Rank (MRR).

    Parameters
    ----------
    ranks : torch.Tensor
        1-D tensor of 1-indexed ranks.

    Returns
    -------
    float
        MRR value in [0, 1].  Returns 0.0 for an empty tensor.
    """
    if ranks.numel() == 0:
        return 0.0
    return (1.0 / ranks.to(torch.float64)).mean().item()


def compute_hits_at_k(ranks: torch.Tensor, k: int) -> float:
    """Compute Hits@K — the fraction of queries ranked at or above *k*.

    Parameters
    ----------
    ranks : torch.Tensor
        1-D tensor of 1-indexed ranks.
    k : int
        Cutoff threshold.  Must be >= 1.

    Returns
    -------
    float
        Fraction of ranks <= k, in [0, 1].  Returns 0.0 for an empty
        tensor.

    Raises
    ------
    ValueError
        If *k* < 1.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if ranks.numel() == 0:
        return 0.0
    return (ranks <= k).to(torch.float64).mean().item()


def compute_mr(ranks: torch.Tensor) -> float:
    """Compute Mean Rank (MR).

    Parameters
    ----------
    ranks : torch.Tensor
        1-D tensor of 1-indexed ranks.

    Returns
    -------
    float
        Mean rank (lower is better).  Returns 0.0 for an empty tensor.
    """
    if ranks.numel() == 0:
        return 0.0
    return ranks.to(torch.float64).mean().item()


def compute_amr(ranks: torch.Tensor, num_candidates: int) -> float:
    """Compute Adjusted Mean Rank (AMR).

    AMR normalizes the mean rank by the expected mean rank of a random
    predictor, yielding a scale-independent metric.

        AMR = MR / E[MR_random] = MR / ((num_candidates + 1) / 2)

    A perfect predictor yields AMR close to 0; a random predictor yields
    AMR close to 1.

    Parameters
    ----------
    ranks : torch.Tensor
        1-D tensor of 1-indexed ranks.
    num_candidates : int
        Total number of candidate entities.  Must be >= 1.

    Returns
    -------
    float
        AMR value.  Returns 0.0 for an empty tensor.

    Raises
    ------
    ValueError
        If *num_candidates* < 1.
    """
    if num_candidates < 1:
        raise ValueError(f"num_candidates must be >= 1, got {num_candidates}")
    if ranks.numel() == 0:
        return 0.0
    mr = compute_mr(ranks)
    expected_random_mr = (num_candidates + 1) / 2.0
    return mr / expected_random_mr
