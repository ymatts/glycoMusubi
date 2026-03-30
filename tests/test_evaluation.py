"""Unit tests for evaluation metrics and LinkPredictionEvaluator.

Tests cover:
  - compute_ranks (raw and filtered)
  - compute_mrr (perfect and worst-case)
  - compute_hits_at_k
  - compute_mr / compute_amr
  - LinkPredictionEvaluator with a mock scorable model
"""

from __future__ import annotations

import pytest
import torch

from glycoMusubi.evaluation.metrics import (
    compute_amr,
    compute_hits_at_k,
    compute_mr,
    compute_mrr,
    compute_ranks,
)
from glycoMusubi.evaluation.link_prediction import (
    LinkPredictionEvaluator,
    LinkPredictionResult,
)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def perfect_ranks() -> torch.Tensor:
    """All ranks are 1 (perfect prediction)."""
    return torch.ones(10, dtype=torch.long)


@pytest.fixture()
def varied_ranks() -> torch.Tensor:
    """Ranks: 1, 2, 3, 5, 10."""
    return torch.tensor([1, 2, 3, 5, 10], dtype=torch.long)


# ======================================================================
# TestMetrics
# ======================================================================


class TestMetrics:
    """Tests for individual metric functions."""

    def test_mrr_perfect(self, perfect_ranks: torch.Tensor) -> None:
        """All ranks = 1 implies MRR = 1.0."""
        mrr = compute_mrr(perfect_ranks)
        assert mrr == pytest.approx(1.0)

    def test_mrr_worst(self) -> None:
        """Worst case: all ranks equal num_entities.

        For 100 entities all ranked at position 100, MRR = 0.01.
        """
        ranks = torch.full((5,), 100, dtype=torch.long)
        mrr = compute_mrr(ranks)
        assert mrr == pytest.approx(1.0 / 100.0)

    def test_mrr_empty(self) -> None:
        """Empty ranks tensor returns 0.0."""
        mrr = compute_mrr(torch.tensor([], dtype=torch.long))
        assert mrr == 0.0

    def test_mrr_varied(self, varied_ranks: torch.Tensor) -> None:
        """MRR of [1, 2, 3, 5, 10] = mean(1, 0.5, 1/3, 0.2, 0.1)."""
        expected = (1.0 + 0.5 + 1.0 / 3.0 + 0.2 + 0.1) / 5.0
        mrr = compute_mrr(varied_ranks)
        assert mrr == pytest.approx(expected, rel=1e-6)

    def test_hits_at_k_all_hit(self, perfect_ranks: torch.Tensor) -> None:
        """All ranks = 1, so Hits@1 = Hits@10 = 1.0."""
        assert compute_hits_at_k(perfect_ranks, k=1) == pytest.approx(1.0)
        assert compute_hits_at_k(perfect_ranks, k=10) == pytest.approx(1.0)

    def test_hits_at_k_partial(self, varied_ranks: torch.Tensor) -> None:
        """Hits@3 for ranks [1,2,3,5,10] = 3/5 = 0.6."""
        h3 = compute_hits_at_k(varied_ranks, k=3)
        assert h3 == pytest.approx(3.0 / 5.0)

    def test_hits_at_k_none(self) -> None:
        """All ranks above k yield Hits@k = 0."""
        ranks = torch.tensor([5, 10, 20], dtype=torch.long)
        assert compute_hits_at_k(ranks, k=1) == pytest.approx(0.0)

    def test_hits_at_k_invalid(self) -> None:
        """k < 1 raises ValueError."""
        with pytest.raises(ValueError, match="k must be >= 1"):
            compute_hits_at_k(torch.tensor([1]), k=0)

    def test_hits_at_k_empty(self) -> None:
        """Empty tensor returns 0.0."""
        assert compute_hits_at_k(torch.tensor([], dtype=torch.long), k=5) == 0.0

    def test_mr_basic(self, varied_ranks: torch.Tensor) -> None:
        """Mean rank of [1,2,3,5,10] = 21/5 = 4.2."""
        mr = compute_mr(varied_ranks)
        assert mr == pytest.approx(4.2)

    def test_mr_empty(self) -> None:
        assert compute_mr(torch.tensor([], dtype=torch.long)) == 0.0

    def test_amr_perfect(self, perfect_ranks: torch.Tensor) -> None:
        """AMR for perfect ranks is close to 0."""
        amr = compute_amr(perfect_ranks, num_candidates=100)
        expected = 1.0 / ((100 + 1) / 2.0)
        assert amr == pytest.approx(expected, rel=1e-6)

    def test_amr_random(self) -> None:
        """AMR for a random predictor should be close to 1.0.

        Random predictor has expected MR = (N+1)/2, so AMR ~ 1.0.
        """
        num_candidates = 1000
        expected_mr = (num_candidates + 1) / 2.0
        ranks = torch.full((100,), int(expected_mr), dtype=torch.long)
        amr = compute_amr(ranks, num_candidates=num_candidates)
        assert amr == pytest.approx(1.0, rel=0.01)

    def test_amr_invalid(self) -> None:
        with pytest.raises(ValueError, match="num_candidates must be >= 1"):
            compute_amr(torch.tensor([1]), num_candidates=0)


# ======================================================================
# TestComputeRanks
# ======================================================================


class TestComputeRanks:
    """Tests for the compute_ranks function."""

    def test_perfect_scores(self) -> None:
        """Target entity has the highest score => rank 1."""
        scores = torch.tensor(
            [
                [0.1, 0.9, 0.2, 0.3],  # target idx 1 has highest
                [0.8, 0.1, 0.3, 0.2],  # target idx 0 has highest
            ]
        )
        target_idx = torch.tensor([1, 0])
        ranks = compute_ranks(scores, target_idx)
        assert ranks.tolist() == [1, 1]

    def test_worst_scores(self) -> None:
        """Target entity has the lowest score => rank = num_entities."""
        scores = torch.tensor(
            [
                [0.9, 0.1, 0.8, 0.7],  # target idx 1 has lowest
            ]
        )
        target_idx = torch.tensor([1])
        ranks = compute_ranks(scores, target_idx)
        assert ranks.item() == 4

    def test_tied_scores(self) -> None:
        """With tied scores, rank = 1 + count of strictly higher."""
        scores = torch.tensor(
            [
                [0.5, 0.5, 0.5, 0.5],
            ]
        )
        target_idx = torch.tensor([2])
        ranks = compute_ranks(scores, target_idx)
        # No entity has strictly higher score, so rank = 1.
        assert ranks.item() == 1

    def test_filtered_ranking(self) -> None:
        """Filtered mask excludes known-true triples from ranking."""
        scores = torch.tensor(
            [
                [0.9, 0.5, 0.3, 0.1],  # target idx 1
            ]
        )
        target_idx = torch.tensor([1])

        # Without filter: entity 0 has higher score, so rank = 2.
        ranks_raw = compute_ranks(scores, target_idx)
        assert ranks_raw.item() == 2

        # Mask out entity 0 (pretend it's another known-true triple).
        mask = torch.tensor([[True, False, False, False]])
        ranks_filtered = compute_ranks(scores, target_idx, filtered_mask=mask)
        # Entity 0 is masked to -inf, so target is now rank 1.
        assert ranks_filtered.item() == 1

    def test_validation_errors(self) -> None:
        """Input shape validation."""
        with pytest.raises(ValueError, match="must be 2-D"):
            compute_ranks(torch.randn(5), torch.tensor([0]))

        with pytest.raises(ValueError, match="must be 1-D"):
            compute_ranks(torch.randn(2, 5), torch.tensor([[0, 1]]))

        with pytest.raises(ValueError, match="Batch size mismatch"):
            compute_ranks(torch.randn(2, 5), torch.tensor([0]))

    def test_filtered_mask_shape_mismatch(self) -> None:
        """Filtered mask must match scores shape."""
        with pytest.raises(ValueError, match="filtered_mask shape"):
            compute_ranks(
                torch.randn(2, 5),
                torch.tensor([0, 1]),
                filtered_mask=torch.zeros(3, 5, dtype=torch.bool),
            )


# ======================================================================
# TestLinkPredictionEvaluator
# ======================================================================


class _MockScorableModel:
    """Mock model that returns predetermined scores for evaluation testing.

    For tail prediction: score_t returns scores where the correct tail
    gets a high score.
    For head prediction: score_h returns scores where the correct head
    gets a high score.
    """

    def __init__(self, num_entities: int, test_triples: torch.Tensor) -> None:
        self.num_entities = num_entities
        # Build lookup for correct answers
        self._hr_to_t = {}
        self._rt_to_h = {}
        for i in range(test_triples.shape[0]):
            h, r, t = test_triples[i].tolist()
            self._hr_to_t[(h, r)] = t
            self._rt_to_h[(r, t)] = h

    def score_t(
        self, head: torch.Tensor, relation: torch.Tensor, num_entities: int
    ) -> torch.Tensor:
        batch_size = head.shape[0]
        scores = torch.zeros(batch_size, num_entities)
        for i in range(batch_size):
            h, r = head[i].item(), relation[i].item()
            correct_t = self._hr_to_t.get((h, r))
            if correct_t is not None:
                scores[i, correct_t] = 10.0  # High score for correct answer
        return scores

    def score_h(
        self, tail: torch.Tensor, relation: torch.Tensor, num_entities: int
    ) -> torch.Tensor:
        batch_size = tail.shape[0]
        scores = torch.zeros(batch_size, num_entities)
        for i in range(batch_size):
            t, r = tail[i].item(), relation[i].item()
            correct_h = self._rt_to_h.get((r, t))
            if correct_h is not None:
                scores[i, correct_h] = 10.0
        return scores


class TestLinkPredictionEvaluator:
    """Tests for the full evaluator pipeline."""

    @pytest.fixture()
    def simple_triples(self) -> torch.Tensor:
        """Small set of test triples: (head, relation, tail)."""
        return torch.tensor(
            [
                [0, 0, 1],
                [1, 1, 2],
                [2, 0, 3],
            ],
            dtype=torch.long,
        )

    def test_evaluate_returns_all_metrics(
        self, simple_triples: torch.Tensor
    ) -> None:
        """Evaluator returns all requested metrics."""
        num_entities = 5
        model = _MockScorableModel(num_entities, simple_triples)
        evaluator = LinkPredictionEvaluator(
            metrics=("mrr", "hits@1", "hits@3", "hits@10", "mr"),
            filtered=True,
            batch_size=4,
        )

        result = evaluator.evaluate(
            model=model,
            test_triples=simple_triples,
            all_triples=simple_triples,
            num_entities=num_entities,
        )

        assert isinstance(result, LinkPredictionResult)
        assert result.num_triples == 3

        # Check that all metric keys are present.
        for metric_name in ("mrr", "hits@1", "hits@3", "hits@10", "mr"):
            assert metric_name in result.metrics
            assert metric_name in result.head_metrics
            assert metric_name in result.tail_metrics

    def test_perfect_model_metrics(self, simple_triples: torch.Tensor) -> None:
        """A perfect model that always ranks correctly should get MRR=1."""
        num_entities = 5
        model = _MockScorableModel(num_entities, simple_triples)
        evaluator = LinkPredictionEvaluator(
            metrics=("mrr", "hits@1", "hits@10"),
            filtered=True,
            batch_size=4,
        )

        result = evaluator.evaluate(
            model=model,
            test_triples=simple_triples,
            all_triples=simple_triples,
            num_entities=num_entities,
        )

        # The mock model gives the correct answer score=10 and others 0,
        # so all ranks should be 1.
        assert result.metrics["mrr"] == pytest.approx(1.0)
        assert result.metrics["hits@1"] == pytest.approx(1.0)
        assert result.metrics["hits@10"] == pytest.approx(1.0)

    def test_per_relation_metrics(self, simple_triples: torch.Tensor) -> None:
        """Per-relation metrics are computed when requested."""
        num_entities = 5
        model = _MockScorableModel(num_entities, simple_triples)
        evaluator = LinkPredictionEvaluator(
            metrics=("mrr", "hits@1"),
            filtered=True,
            batch_size=4,
        )

        result = evaluator.evaluate(
            model=model,
            test_triples=simple_triples,
            all_triples=simple_triples,
            num_entities=num_entities,
            per_relation=True,
            relation_names={0: "has_glycan", 1: "inhibits"},
        )

        # We should have entries for both relations.
        assert "has_glycan" in result.per_relation
        assert "inhibits" in result.per_relation

        # Check that per-relation metrics have the correct keys.
        for rel_name in ("has_glycan", "inhibits"):
            assert "mrr" in result.per_relation[rel_name]
            assert "hits@1" in result.per_relation[rel_name]

    def test_per_relation_disabled(self, simple_triples: torch.Tensor) -> None:
        """Per-relation metrics empty when disabled."""
        num_entities = 5
        model = _MockScorableModel(num_entities, simple_triples)
        evaluator = LinkPredictionEvaluator(
            metrics=("mrr",),
            filtered=False,
            batch_size=4,
        )

        result = evaluator.evaluate(
            model=model,
            test_triples=simple_triples,
            all_triples=simple_triples,
            num_entities=num_entities,
            per_relation=False,
        )

        assert result.per_relation == {}

    def test_unfiltered_evaluation(self, simple_triples: torch.Tensor) -> None:
        """Unfiltered evaluation runs without error."""
        num_entities = 5
        model = _MockScorableModel(num_entities, simple_triples)
        evaluator = LinkPredictionEvaluator(
            metrics=("mrr",),
            filtered=False,
        )

        result = evaluator.evaluate(
            model=model,
            test_triples=simple_triples,
            all_triples=simple_triples,
            num_entities=num_entities,
        )

        assert "mrr" in result.metrics
        assert result.num_triples == 3

    def test_empty_test_set(self) -> None:
        """Empty test triples return zero metrics."""
        empty = torch.zeros(0, 3, dtype=torch.long)
        all_triples = torch.tensor([[0, 0, 1]], dtype=torch.long)

        model = _MockScorableModel(2, empty)
        evaluator = LinkPredictionEvaluator(metrics=("mrr",))

        result = evaluator.evaluate(
            model=model,
            test_triples=empty,
            all_triples=all_triples,
            num_entities=2,
        )

        assert result.num_triples == 0

    def test_invalid_triples_shape(self) -> None:
        """Invalid triple shapes raise ValueError."""
        model = _MockScorableModel(2, torch.tensor([[0, 0, 1]]))
        evaluator = LinkPredictionEvaluator()

        with pytest.raises(ValueError, match="test_triples must be"):
            evaluator.evaluate(
                model=model,
                test_triples=torch.randn(3, 2),
                all_triples=torch.tensor([[0, 0, 1]], dtype=torch.long),
                num_entities=2,
            )

    def test_batch_size_smaller_than_test(self) -> None:
        """Evaluator handles batch_size < num test triples correctly."""
        triples = torch.tensor(
            [[0, 0, 1], [1, 0, 2], [2, 0, 3], [3, 0, 4]],
            dtype=torch.long,
        )
        num_entities = 5
        model = _MockScorableModel(num_entities, triples)
        evaluator = LinkPredictionEvaluator(
            metrics=("mrr",),
            filtered=True,
            batch_size=2,  # Smaller than 4 test triples
        )

        result = evaluator.evaluate(
            model=model,
            test_triples=triples,
            all_triples=triples,
            num_entities=num_entities,
        )

        assert result.num_triples == 4
        assert result.metrics["mrr"] == pytest.approx(1.0)
