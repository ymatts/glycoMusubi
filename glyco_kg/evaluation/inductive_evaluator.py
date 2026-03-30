"""Per-relation inductive link prediction evaluator.

Extends the standard ``LinkPredictionEvaluator`` with:
- Relation-specific filtering (evaluate only biological relations)
- Type-restricted candidate sets (only score valid entity types)
- Separate transductive vs inductive MRR reporting
- Function-aware candidate filtering for has_glycan evaluation
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch

from glycoMusubi.evaluation.link_prediction import (
    LinkPredictionEvaluator,
    LinkPredictionResult,
    ScorableModel,
)
from glycoMusubi.evaluation.metrics import (
    compute_hits_at_k,
    compute_mr,
    compute_mrr,
    compute_ranks,
)

logger = logging.getLogger(__name__)


@dataclass
class FunctionFilter:
    """Filter evaluation candidates by glycan function class.

    For ``protein -> has_glycan -> glycan`` tail prediction, restricts the
    candidate set to glycans that share a function class with the query
    protein's site residue type (N→N-linked, S/T→O-linked).

    Parameters
    ----------
    glycan_function_indices : dict[str, set[int]]
        ``{function_term: {global_glycan_indices}}``.
    protein_function_type : dict[int, str]
        ``{global_protein_idx: function_type}`` inferred from site data.
    has_glycan_rel_indices : set[int]
        Relation indices corresponding to ``has_glycan``.
    all_glycan_global_indices : set[int]
        All glycan global indices (for fallback when protein type is unknown).
    """

    glycan_function_indices: Dict[str, Set[int]]
    protein_function_type: Dict[int, str]
    has_glycan_rel_indices: Set[int]
    all_glycan_global_indices: Set[int] = field(default_factory=set)

    def get_candidate_mask(
        self,
        head_idx: int,
        rel_idx: int,
        num_entities: int,
        device: torch.device,
        mode: str = "tail",
    ) -> Optional[torch.Tensor]:
        """Return a boolean mask of valid candidates, or None if no restriction.

        Parameters
        ----------
        head_idx : int
            Global head entity index.
        rel_idx : int
            Relation index.
        num_entities : int
            Total number of entities.
        device : torch.device
            Target device.
        mode : str
            ``"tail"`` for tail prediction, ``"head"`` for head prediction.

        Returns
        -------
        Optional[torch.Tensor]
            Boolean tensor ``[num_entities]`` where ``True`` = valid candidate.
            Returns ``None`` if no filtering applies.
        """
        if mode != "tail":
            return None
        if rel_idx not in self.has_glycan_rel_indices:
            return None

        func_type = self.protein_function_type.get(head_idx)
        if func_type and func_type in self.glycan_function_indices:
            valid_set = self.glycan_function_indices[func_type]
        elif self.all_glycan_global_indices:
            # Fallback: allow any function-bearing glycan
            valid_set = self.all_glycan_global_indices
        else:
            return None

        mask = torch.zeros(num_entities, dtype=torch.bool, device=device)
        for idx in valid_set:
            if idx < num_entities:
                mask[idx] = True
        return mask


@dataclass
class InductiveEvalResult:
    """Container for inductive evaluation results.

    Attributes
    ----------
    transductive : LinkPredictionResult
        Metrics on triples where both entities were seen during training.
    inductive : LinkPredictionResult
        Metrics on triples involving at least one held-out entity.
    per_relation_inductive : dict[str, dict[str, float]]
        Per-relation inductive metrics.
    holdout_stats : dict[str, int]
        Statistics about the hold-out set.
    """

    transductive: LinkPredictionResult = field(
        default_factory=LinkPredictionResult
    )
    inductive: LinkPredictionResult = field(
        default_factory=LinkPredictionResult
    )
    per_relation_inductive: Dict[str, Dict[str, float]] = field(
        default_factory=dict
    )
    holdout_stats: Dict[str, int] = field(default_factory=dict)


class InductiveEvaluator:
    """Evaluator for inductive (zero-shot) link prediction.

    Parameters
    ----------
    metrics : tuple of str
        Metric names to compute.
    filtered : bool
        Whether to use filtered ranking.
    batch_size : int
        Batch size for scoring.
    target_relations : list of str, optional
        If specified, only evaluate these relation names.
    """

    def __init__(
        self,
        metrics: Tuple[str, ...] = ("mrr", "hits@1", "hits@3", "hits@10", "mr"),
        filtered: bool = True,
        batch_size: int = 128,
        target_relations: Optional[List[str]] = None,
        function_filter: Optional[FunctionFilter] = None,
    ) -> None:
        self.metric_names = metrics
        self.filtered = filtered
        self.batch_size = batch_size
        self.target_relations = set(target_relations) if target_relations else None
        self.function_filter = function_filter

        self._hits_k_values = []
        for name in self.metric_names:
            if name.startswith("hits@"):
                self._hits_k_values.append(int(name.split("@")[1]))

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
            elif name.startswith("hits@"):
                k = int(name.split("@")[1])
                result[name] = compute_hits_at_k(ranks, k)
        return result

    @torch.no_grad()
    def evaluate(
        self,
        model: ScorableModel,
        test_triples: torch.Tensor,
        all_triples: torch.Tensor,
        num_entities: int,
        holdout_global_ids: Optional[Set[int]] = None,
        relation_names: Optional[Dict[int, str]] = None,
    ) -> InductiveEvalResult:
        """Run inductive evaluation with transductive/inductive split.

        Parameters
        ----------
        model : ScorableModel
            Model implementing ``score_t`` and ``score_h``.
        test_triples : torch.Tensor
            Shape ``[N, 3]`` test triples ``(head, relation, tail)``.
        all_triples : torch.Tensor
            Shape ``[M, 3]`` all known triples for filtering.
        num_entities : int
            Total number of entities.
        holdout_global_ids : set of int, optional
            Global entity indices of held-out entities.
        relation_names : dict, optional
            Relation index to name mapping.

        Returns
        -------
        InductiveEvalResult
        """
        if test_triples.ndim != 2 or test_triples.shape[1] != 3:
            raise ValueError(f"test_triples must be [N, 3], got {test_triples.shape}")

        n_test = test_triples.shape[0]
        if n_test == 0:
            return InductiveEvalResult()

        holdout = holdout_global_ids or set()

        # Filter to target relations if specified
        if self.target_relations and relation_names:
            rel_name_to_idx = {v: k for k, v in relation_names.items()}
            target_rel_indices = {
                rel_name_to_idx[r]
                for r in self.target_relations
                if r in rel_name_to_idx
            }
            if target_rel_indices:
                mask = torch.tensor(
                    [test_triples[i, 1].item() in target_rel_indices for i in range(n_test)],
                    dtype=torch.bool,
                )
                test_triples = test_triples[mask]
                n_test = test_triples.shape[0]
                logger.info(
                    "Filtered to %d target relations -> %d test triples",
                    len(target_rel_indices),
                    n_test,
                )

        if n_test == 0:
            return InductiveEvalResult()

        # Build filter index
        from glycoMusubi.evaluation.link_prediction import _TripleIndex

        triple_index = _TripleIndex(all_triples) if self.filtered else None

        # Accumulate ranks separately for transductive and inductive
        trans_head_ranks: List[torch.Tensor] = []
        trans_tail_ranks: List[torch.Tensor] = []
        induct_head_ranks: List[torch.Tensor] = []
        induct_tail_ranks: List[torch.Tensor] = []

        # Per-relation inductive ranks
        rel_induct_ranks: Dict[int, List[torch.Tensor]] = defaultdict(list)

        for start in range(0, n_test, self.batch_size):
            end = min(start + self.batch_size, n_test)
            batch = test_triples[start:end]
            heads = batch[:, 0]
            rels = batch[:, 1]
            tails = batch[:, 2]

            # Tail prediction
            tail_scores = model.score_t(heads, rels, num_entities)
            # Apply function filter to tail scores (restricts has_glycan candidates)
            if self.function_filter is not None:
                for i in range(batch.shape[0]):
                    cand_mask = self.function_filter.get_candidate_mask(
                        heads[i].item(), rels[i].item(), num_entities,
                        tail_scores.device, mode="tail",
                    )
                    if cand_mask is not None:
                        tail_scores[i, ~cand_mask] = -1e9
            tail_filter = self._build_filter(
                heads, rels, tails, num_entities, triple_index, mode="tail"
            )
            tail_ranks = compute_ranks(tail_scores, tails, tail_filter)

            # Head prediction
            head_scores = model.score_h(tails, rels, num_entities)
            head_filter = self._build_filter(
                heads, rels, tails, num_entities, triple_index, mode="head"
            )
            head_ranks = compute_ranks(head_scores, heads, head_filter)

            # Split into transductive and inductive
            for i in range(batch.shape[0]):
                h = heads[i].item()
                t = tails[i].item()
                r = rels[i].item()
                is_inductive = h in holdout or t in holdout

                if is_inductive:
                    induct_tail_ranks.append(tail_ranks[i:i + 1])
                    induct_head_ranks.append(head_ranks[i:i + 1])
                    rel_induct_ranks[r].append(tail_ranks[i:i + 1])
                    rel_induct_ranks[r].append(head_ranks[i:i + 1])
                else:
                    trans_tail_ranks.append(tail_ranks[i:i + 1])
                    trans_head_ranks.append(head_ranks[i:i + 1])

        result = InductiveEvalResult()

        # Transductive metrics
        if trans_head_ranks:
            trans_h = torch.cat(trans_head_ranks)
            trans_t = torch.cat(trans_tail_ranks)
            trans_all = torch.cat([trans_h, trans_t])
            result.transductive = LinkPredictionResult(
                metrics=self._compute_metric_dict(trans_all, num_entities),
                head_metrics=self._compute_metric_dict(trans_h, num_entities),
                tail_metrics=self._compute_metric_dict(trans_t, num_entities),
                num_triples=len(trans_head_ranks),
            )

        # Inductive metrics
        if induct_head_ranks:
            ind_h = torch.cat(induct_head_ranks)
            ind_t = torch.cat(induct_tail_ranks)
            ind_all = torch.cat([ind_h, ind_t])
            result.inductive = LinkPredictionResult(
                metrics=self._compute_metric_dict(ind_all, num_entities),
                head_metrics=self._compute_metric_dict(ind_h, num_entities),
                tail_metrics=self._compute_metric_dict(ind_t, num_entities),
                num_triples=len(induct_head_ranks),
            )

        # Per-relation inductive metrics
        for rel_idx, ranks_list in rel_induct_ranks.items():
            combined = torch.cat(ranks_list)
            key = (
                relation_names[rel_idx]
                if relation_names and rel_idx in relation_names
                else str(rel_idx)
            )
            result.per_relation_inductive[key] = self._compute_metric_dict(
                combined, num_entities
            )

        result.holdout_stats = {
            "num_holdout_entities": len(holdout),
            "num_inductive_triples": len(induct_head_ranks),
            "num_transductive_triples": len(trans_head_ranks),
        }

        # Log summary
        ind_mrr = result.inductive.metrics.get("mrr", 0.0)
        trans_mrr = result.transductive.metrics.get("mrr", 0.0)
        logger.info(
            "Inductive evaluation: trans_MRR=%.4f (%d triples), "
            "inductive_MRR=%.4f (%d triples)",
            trans_mrr,
            len(trans_head_ranks),
            ind_mrr,
            len(induct_head_ranks),
        )
        for rel_name, metrics in result.per_relation_inductive.items():
            logger.info(
                "  [inductive] %s: MRR=%.4f, Hits@1=%.4f, Hits@10=%.4f",
                rel_name,
                metrics.get("mrr", 0.0),
                metrics.get("hits@1", 0.0),
                metrics.get("hits@10", 0.0),
            )

        return result

    def _build_filter(
        self,
        heads: torch.Tensor,
        rels: torch.Tensor,
        tails: torch.Tensor,
        num_entities: int,
        triple_index: Optional[object],
        mode: str,
    ) -> Optional[torch.Tensor]:
        """Build filter mask for known triples."""
        if triple_index is None:
            return None

        batch_size = heads.shape[0]
        mask = torch.zeros(
            batch_size, num_entities, dtype=torch.bool, device=heads.device
        )

        heads_cpu = heads.cpu()
        rels_cpu = rels.cpu()
        tails_cpu = tails.cpu()

        for i in range(batch_size):
            h = heads_cpu[i].item()
            r = rels_cpu[i].item()
            t = tails_cpu[i].item()

            if mode == "tail":
                known = triple_index.tails_for(h, r)
                for k in known:
                    if k != t:
                        mask[i, k] = True
            else:
                known = triple_index.heads_for(r, t)
                for k in known:
                    if k != h:
                        mask[i, k] = True

        return mask
