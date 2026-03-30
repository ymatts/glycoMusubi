"""Filtered link-prediction evaluator for heterogeneous KG embeddings.

Implements the standard filtered evaluation protocol: for each test triple
``(h, r, t)``, all head and tail candidates are scored, and known-true
triples (excluding the test triple itself) are removed from the ranking
before computing metrics.

The evaluator is model-agnostic — any model that exposes a ``score_t``
and ``score_h`` interface can be evaluated.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import torch

from glycoMusubi.evaluation.metrics import (
    compute_amr,
    compute_hits_at_k,
    compute_mr,
    compute_mrr,
    compute_ranks,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class ScorableModel(Protocol):
    """Protocol that a KGE model must satisfy for link-prediction evaluation.

    Both scoring methods receive long tensors on the model's device and must
    return a float tensor of shape ``[batch, num_candidates]`` where higher
    values indicate greater plausibility.
    """

    def score_t(
        self, head: torch.Tensor, relation: torch.Tensor, num_entities: int
    ) -> torch.Tensor:
        """Score all tail candidates for given ``(head, relation, ?)`` queries.

        Parameters
        ----------
        head : torch.Tensor
            Shape ``[batch]`` — head entity indices.
        relation : torch.Tensor
            Shape ``[batch]`` — relation indices.
        num_entities : int
            Total number of candidate tail entities.

        Returns
        -------
        torch.Tensor
            Shape ``[batch, num_entities]``.
        """
        ...

    def score_h(
        self, tail: torch.Tensor, relation: torch.Tensor, num_entities: int
    ) -> torch.Tensor:
        """Score all head candidates for given ``(?, relation, tail)`` queries.

        Parameters
        ----------
        tail : torch.Tensor
            Shape ``[batch]`` — tail entity indices.
        relation : torch.Tensor
            Shape ``[batch]`` — relation indices.
        num_entities : int
            Total number of candidate head entities.

        Returns
        -------
        torch.Tensor
            Shape ``[batch, num_entities]``.
        """
        ...


# ---------------------------------------------------------------------------
# Evaluation result container
# ---------------------------------------------------------------------------

@dataclass
class LinkPredictionResult:
    """Container for link-prediction evaluation results.

    Attributes
    ----------
    metrics : dict[str, float]
        Overall (averaged) metric values, e.g. ``{"mrr": 0.35, ...}``.
    head_metrics : dict[str, float]
        Metrics computed only on head-prediction queries ``(?, r, t)``.
    tail_metrics : dict[str, float]
        Metrics computed only on tail-prediction queries ``(h, r, ?)``.
    per_relation : dict[str, dict[str, float]]
        Per-relation metric breakdown, keyed by relation name/index.
    num_triples : int
        Number of test triples evaluated.
    """

    metrics: Dict[str, float] = field(default_factory=dict)
    head_metrics: Dict[str, float] = field(default_factory=dict)
    tail_metrics: Dict[str, float] = field(default_factory=dict)
    per_relation: Dict[str, Dict[str, float]] = field(default_factory=dict)
    num_triples: int = 0


# ---------------------------------------------------------------------------
# Triple-set helper for building filtered masks
# ---------------------------------------------------------------------------

class _TripleIndex:
    """Fast lookup of known triples for filtered ranking.

    Internally stores two dictionaries for O(1) lookup:
    * ``(relation, tail) -> set of heads``   (for head prediction)
    * ``(head, relation) -> set of tails``   (for tail prediction)
    """

    def __init__(self, triples: torch.Tensor) -> None:
        """Build index from a ``[N, 3]`` long tensor of ``(h, r, t)`` rows."""
        self._rt_to_heads: Dict[Tuple[int, int], set] = defaultdict(set)
        self._hr_to_tails: Dict[Tuple[int, int], set] = defaultdict(set)

        # Materialise once on CPU for indexing speed.
        triples_cpu = triples.cpu()
        for i in range(triples_cpu.shape[0]):
            h = triples_cpu[i, 0].item()
            r = triples_cpu[i, 1].item()
            t = triples_cpu[i, 2].item()
            self._rt_to_heads[(r, t)].add(h)
            self._hr_to_tails[(h, r)].add(t)

    def heads_for(self, r: int, t: int) -> set:
        """Return the set of known heads for ``(?, r, t)``."""
        return self._rt_to_heads.get((r, t), set())

    def tails_for(self, h: int, r: int) -> set:
        """Return the set of known tails for ``(h, r, ?)``."""
        return self._hr_to_tails.get((h, r), set())


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

_DEFAULT_METRICS: Tuple[str, ...] = ("mrr", "hits@1", "hits@3", "hits@10", "mr")


class LinkPredictionEvaluator:
    """Filtered link-prediction evaluator.

    Parameters
    ----------
    metrics : sequence of str
        Metric names to compute.  Supported: ``"mrr"``, ``"mr"``,
        ``"amr"``, ``"hits@K"`` for any positive integer K.
    filtered : bool
        If ``True`` (default), perform *filtered* ranking — known-true
        triples are excluded from the ranking for each test query.
    batch_size : int
        Number of test triples to score per forward pass.

    Examples
    --------
    >>> evaluator = LinkPredictionEvaluator()
    >>> result = evaluator.evaluate(
    ...     model=my_model,
    ...     test_triples=test_hrt,
    ...     all_triples=all_hrt,
    ...     num_entities=num_entities,
    ... )
    >>> print(result.metrics)
    {'mrr': 0.354, 'hits@1': 0.261, ...}
    """

    def __init__(
        self,
        metrics: Sequence[str] = _DEFAULT_METRICS,
        filtered: bool = True,
        batch_size: int = 256,
    ) -> None:
        self.metric_names: Tuple[str, ...] = tuple(metrics)
        self.filtered: bool = filtered
        self.batch_size: int = batch_size

        # Pre-parse hits@K values for validation.
        self._hits_k_values: List[int] = []
        for name in self.metric_names:
            if name.startswith("hits@"):
                k = int(name.split("@")[1])
                if k < 1:
                    raise ValueError(f"hits@K requires K >= 1, got {name}")
                self._hits_k_values.append(k)

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self,
        model: ScorableModel,
        test_triples: torch.Tensor,
        all_triples: torch.Tensor,
        num_entities: int,
        relation_names: Optional[Dict[int, str]] = None,
        per_relation: bool = True,
    ) -> LinkPredictionResult:
        """Run filtered link-prediction evaluation.

        Parameters
        ----------
        model : ScorableModel
            A model implementing ``score_t`` and ``score_h``.
        test_triples : torch.Tensor
            Shape ``[N, 3]`` — test triples as ``(head, relation, tail)``
            long tensor.
        all_triples : torch.Tensor
            Shape ``[M, 3]`` — **all** known triples (train + valid + test)
            used to build the filter.  Must be a superset of
            *test_triples*.
        num_entities : int
            Total number of entities in the KG.
        relation_names : dict[int, str] or None
            Optional mapping from relation index to human-readable name.
            Used as keys in the per-relation result.
        per_relation : bool
            Whether to compute per-relation metrics.

        Returns
        -------
        LinkPredictionResult
            Aggregated and per-relation metrics.
        """
        if test_triples.ndim != 2 or test_triples.shape[1] != 3:
            raise ValueError(
                f"test_triples must be [N, 3], got {test_triples.shape}"
            )
        if all_triples.ndim != 2 or all_triples.shape[1] != 3:
            raise ValueError(
                f"all_triples must be [M, 3], got {all_triples.shape}"
            )

        n_test = test_triples.shape[0]
        if n_test == 0:
            logger.warning("Empty test set — returning zero metrics.")
            return LinkPredictionResult()

        # Build triple index for filtering.
        triple_index = _TripleIndex(all_triples) if self.filtered else None

        # Accumulate ranks.
        all_head_ranks: List[torch.Tensor] = []
        all_tail_ranks: List[torch.Tensor] = []

        # Per-relation accumulation: rel_idx -> list of rank tensors.
        rel_head_ranks: Dict[int, List[torch.Tensor]] = defaultdict(list)
        rel_tail_ranks: Dict[int, List[torch.Tensor]] = defaultdict(list)

        for start in range(0, n_test, self.batch_size):
            end = min(start + self.batch_size, n_test)
            batch = test_triples[start:end]  # [B, 3]
            heads = batch[:, 0]
            rels = batch[:, 1]
            tails = batch[:, 2]

            # --- Tail prediction: (h, r, ?) ---
            tail_scores = model.score_t(heads, rels, num_entities)  # [B, E]
            tail_mask = self._build_tail_filter(
                heads, rels, tails, num_entities, triple_index
            )
            tail_ranks = compute_ranks(tail_scores, tails, tail_mask)
            all_tail_ranks.append(tail_ranks)

            # --- Head prediction: (?, r, t) ---
            head_scores = model.score_h(tails, rels, num_entities)  # [B, E]
            head_mask = self._build_head_filter(
                heads, rels, tails, num_entities, triple_index
            )
            head_ranks = compute_ranks(head_scores, heads, head_mask)
            all_head_ranks.append(head_ranks)

            # Per-relation bookkeeping.
            if per_relation:
                for i in range(batch.shape[0]):
                    rel_idx = rels[i].item()
                    rel_tail_ranks[rel_idx].append(tail_ranks[i : i + 1])
                    rel_head_ranks[rel_idx].append(head_ranks[i : i + 1])

        head_ranks_cat = torch.cat(all_head_ranks)
        tail_ranks_cat = torch.cat(all_tail_ranks)
        both_ranks = torch.cat([head_ranks_cat, tail_ranks_cat])

        result = LinkPredictionResult(
            metrics=self._compute_metric_dict(both_ranks, num_entities),
            head_metrics=self._compute_metric_dict(head_ranks_cat, num_entities),
            tail_metrics=self._compute_metric_dict(tail_ranks_cat, num_entities),
            num_triples=n_test,
        )

        # Per-relation aggregation.
        if per_relation:
            for rel_idx in sorted(set(rel_head_ranks) | set(rel_tail_ranks)):
                rh = torch.cat(rel_head_ranks[rel_idx]) if rel_idx in rel_head_ranks else torch.tensor([], dtype=torch.long)
                rt = torch.cat(rel_tail_ranks[rel_idx]) if rel_idx in rel_tail_ranks else torch.tensor([], dtype=torch.long)
                combined = torch.cat([rh, rt]) if rh.numel() + rt.numel() > 0 else torch.tensor([], dtype=torch.long)
                key = (
                    relation_names[rel_idx]
                    if relation_names and rel_idx in relation_names
                    else str(rel_idx)
                )
                result.per_relation[key] = self._compute_metric_dict(
                    combined, num_entities
                )

        logger.info(
            "Link-prediction evaluation complete: %d triples, MRR=%.4f, "
            "Hits@1=%.4f, Hits@10=%.4f",
            n_test,
            result.metrics.get("mrr", 0.0),
            result.metrics.get("hits@1", 0.0),
            result.metrics.get("hits@10", 0.0),
        )

        return result

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _compute_metric_dict(
        self, ranks: torch.Tensor, num_entities: int
    ) -> Dict[str, float]:
        """Compute all requested metrics from a rank tensor."""
        if ranks.numel() == 0:
            return {name: 0.0 for name in self.metric_names}

        result: Dict[str, float] = {}
        for name in self.metric_names:
            if name == "mrr":
                result[name] = compute_mrr(ranks)
            elif name == "mr":
                result[name] = compute_mr(ranks)
            elif name == "amr":
                result[name] = compute_amr(ranks, num_entities)
            elif name.startswith("hits@"):
                k = int(name.split("@")[1])
                result[name] = compute_hits_at_k(ranks, k)
            else:
                logger.warning("Unknown metric '%s' — skipping.", name)
        return result

    def _build_tail_filter(
        self,
        heads: torch.Tensor,
        rels: torch.Tensor,
        tails: torch.Tensor,
        num_entities: int,
        triple_index: Optional[_TripleIndex],
    ) -> Optional[torch.Tensor]:
        """Build a boolean mask for tail prediction filtering.

        For query ``(h, r, ?)``, mask out all tails ``t'`` such that
        ``(h, r, t')`` is a known triple **and** ``t' != t`` (the true
        test target).

        Returns ``None`` when ``triple_index`` is ``None`` (unfiltered).
        """
        if triple_index is None:
            return None

        batch_size = heads.shape[0]
        mask = torch.zeros(batch_size, num_entities, dtype=torch.bool,
                           device=heads.device)

        heads_cpu = heads.cpu()
        rels_cpu = rels.cpu()
        tails_cpu = tails.cpu()

        for i in range(batch_size):
            h = heads_cpu[i].item()
            r = rels_cpu[i].item()
            t = tails_cpu[i].item()
            known_tails = triple_index.tails_for(h, r)
            for kt in known_tails:
                if kt != t:
                    mask[i, kt] = True

        return mask

    def _build_head_filter(
        self,
        heads: torch.Tensor,
        rels: torch.Tensor,
        tails: torch.Tensor,
        num_entities: int,
        triple_index: Optional[_TripleIndex],
    ) -> Optional[torch.Tensor]:
        """Build a boolean mask for head prediction filtering.

        For query ``(?, r, t)``, mask out all heads ``h'`` such that
        ``(h', r, t)`` is a known triple **and** ``h' != h`` (the true
        test target).

        Returns ``None`` when ``triple_index`` is ``None`` (unfiltered).
        """
        if triple_index is None:
            return None

        batch_size = heads.shape[0]
        mask = torch.zeros(batch_size, num_entities, dtype=torch.bool,
                           device=heads.device)

        heads_cpu = heads.cpu()
        rels_cpu = rels.cpu()
        tails_cpu = tails.cpu()

        for i in range(batch_size):
            h = heads_cpu[i].item()
            r = rels_cpu[i].item()
            t = tails_cpu[i].item()
            known_heads = triple_index.heads_for(r, t)
            for kh in known_heads:
                if kh != h:
                    mask[i, kh] = True

        return mask
