"""glycoMusubi specific evaluation metrics.

Domain-aware metrics that measure how well the learned embeddings
capture glycan structural relationships, cross-modal alignment with
proteins, and hierarchical taxonomic consistency.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor


def glycan_structure_recovery(
    structural_similarities: Tensor,
    embedding_distances: Tensor,
) -> float:
    """Glycan Structure Recovery (GSR).

    Spearman rank correlation between pairwise structural similarity
    scores (e.g. Tanimoto on glycan fingerprints) and the corresponding
    embedding-space distances.  A high GSR indicates that the embedding
    faithfully preserves structural relationships.

    Parameters
    ----------
    structural_similarities : Tensor
        Shape ``[N]``.  Pairwise structural similarity values.
    embedding_distances : Tensor
        Shape ``[N]``.  Corresponding pairwise distances in embedding
        space (e.g. cosine or Euclidean).

    Returns
    -------
    float
        Spearman rank correlation coefficient in ``[-1, 1]``.
    """
    if structural_similarities.numel() < 2:
        return 0.0

    ranks_sim = _rank(structural_similarities)
    ranks_dist = _rank(embedding_distances)

    n = ranks_sim.numel()
    d = ranks_sim - ranks_dist
    rho = 1.0 - (6.0 * (d * d).sum().item()) / (n * (n * n - 1))
    return float(rho)


def cross_modal_alignment_score(
    glycan_emb: Tensor,
    protein_emb: Tensor,
    known_pairs: Tensor,
) -> float:
    """Cross-modal Alignment Score (CAS).

    For each known glycan-protein binding pair, compute the rank of the
    true protein partner among all proteins by cosine similarity.
    Return the average reciprocal rank.

    Parameters
    ----------
    glycan_emb : Tensor
        Shape ``[G, d]``.  Glycan embeddings.
    protein_emb : Tensor
        Shape ``[P, d]``.  Protein embeddings.
    known_pairs : Tensor
        Shape ``[K, 2]`` with columns ``(glycan_idx, protein_idx)``.

    Returns
    -------
    float
        Mean reciprocal rank of the true binding partner.
    """
    if known_pairs.numel() == 0:
        return 0.0

    # Normalise for cosine similarity
    glycan_norm = torch.nn.functional.normalize(glycan_emb, dim=-1)
    protein_norm = torch.nn.functional.normalize(protein_emb, dim=-1)

    # Similarity matrix: [G, P]
    sim_matrix = glycan_norm @ protein_norm.t()

    reciprocal_ranks = []
    for pair_idx in range(known_pairs.size(0)):
        g_idx = known_pairs[pair_idx, 0].long()
        p_idx = known_pairs[pair_idx, 1].long()
        sims = sim_matrix[g_idx]  # [P]
        # Rank: 1 + number of proteins with higher similarity
        rank = (sims > sims[p_idx]).sum().item() + 1
        reciprocal_ranks.append(1.0 / rank)

    return float(sum(reciprocal_ranks) / len(reciprocal_ranks))


def taxonomy_hierarchical_consistency(
    predictions: Dict[str, Tensor],
    labels: Dict[str, Tensor],
) -> float:
    """Taxonomy Hierarchical Consistency (THC).

    Measures the fraction of instances where, given a correct
    parent-level prediction, the child-level prediction is also correct.

    Parameters
    ----------
    predictions : dict of {level_name: Tensor}
        Predicted class indices for each taxonomy level.  Keys must be
        ordered from coarsest to finest (e.g.
        ``{"kingdom": ..., "phylum": ..., "class": ...}``).
    labels : dict of {level_name: Tensor}
        Ground-truth class indices, same structure as ``predictions``.

    Returns
    -------
    float
        Hierarchical consistency score in ``[0, 1]``.  Returns 1.0 when
        every correct parent prediction has a correct child prediction.
    """
    levels = list(predictions.keys())
    if len(levels) < 2:
        return 1.0

    consistent = 0
    total = 0

    for i in range(len(levels) - 1):
        parent_level = levels[i]
        child_level = levels[i + 1]

        parent_correct = predictions[parent_level] == labels[parent_level]
        child_correct = predictions[child_level] == labels[child_level]

        # Among instances with correct parent prediction, count correct children
        parent_correct_mask = parent_correct.bool()
        n_parent_correct = parent_correct_mask.sum().item()

        if n_parent_correct > 0:
            n_child_also_correct = (
                child_correct[parent_correct_mask].sum().item()
            )
            consistent += n_child_also_correct
            total += n_parent_correct

    return float(consistent / total) if total > 0 else 1.0


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _rank(x: Tensor) -> Tensor:
    """Return 1-indexed ranks for a 1-D tensor (average tie-breaking)."""
    sorted_indices = x.argsort()
    ranks = torch.empty_like(x, dtype=torch.float64)
    ranks[sorted_indices] = torch.arange(
        1, x.numel() + 1, dtype=torch.float64, device=x.device
    )
    return ranks
