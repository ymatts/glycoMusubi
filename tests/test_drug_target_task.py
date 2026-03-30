"""Unit tests for DrugTargetTask downstream evaluation.

Tests cover:
  - Instantiation and default parameters
  - prepare_data with mock compound/enzyme embeddings
  - Time-split data preparation
  - AUC-ROC computation
  - Hit@K for novel targets
  - Enrichment Factor @1% computation and edge cases
  - Full evaluate() pipeline with synthetic data
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from glycoMusubi.evaluation.tasks.drug_target import DrugTargetTask


# ======================================================================
# Constants
# ======================================================================

EMB_DIM = 32
NUM_COMPOUNDS = 50
NUM_ENZYMES = 30
NUM_EDGES = 40


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def mock_hetero_data() -> HeteroData:
    """Create minimal HeteroData with compound-inhibits-enzyme edges."""
    data = HeteroData()
    data["compound"].num_nodes = NUM_COMPOUNDS
    data["enzyme"].num_nodes = NUM_ENZYMES

    rng = np.random.RandomState(0)
    src = torch.from_numpy(rng.randint(0, NUM_COMPOUNDS, size=NUM_EDGES))
    dst = torch.from_numpy(rng.randint(0, NUM_ENZYMES, size=NUM_EDGES))
    data["compound", "inhibits", "enzyme"].edge_index = torch.stack([src, dst])
    return data


@pytest.fixture()
def mock_hetero_data_with_time() -> HeteroData:
    """HeteroData with timestamp attribute on inhibits edges."""
    data = HeteroData()
    data["compound"].num_nodes = NUM_COMPOUNDS
    data["enzyme"].num_nodes = NUM_ENZYMES

    rng = np.random.RandomState(0)
    src = torch.from_numpy(rng.randint(0, NUM_COMPOUNDS, size=NUM_EDGES))
    dst = torch.from_numpy(rng.randint(0, NUM_ENZYMES, size=NUM_EDGES))
    data["compound", "inhibits", "enzyme"].edge_index = torch.stack([src, dst])

    # Assign timestamps: years 2015-2024
    timestamps = torch.tensor(
        rng.uniform(2015, 2024, size=NUM_EDGES), dtype=torch.float32
    )
    data["compound", "inhibits", "enzyme"].timestamp = timestamps
    return data


@pytest.fixture()
def mock_embeddings() -> dict[str, torch.Tensor]:
    """Synthetic embeddings for compound and enzyme nodes."""
    torch.manual_seed(42)
    return {
        "compound": torch.randn(NUM_COMPOUNDS, EMB_DIM),
        "enzyme": torch.randn(NUM_ENZYMES, EMB_DIM),
    }


# ======================================================================
# TestDrugTargetInstantiation
# ======================================================================


class TestDrugTargetInstantiation:
    """Tests for DrugTargetTask construction."""

    def test_default_params(self) -> None:
        task = DrugTargetTask()
        assert task.name == "drug_target_identification"
        assert task.k_values == [10, 20, 50]
        assert task.classifier_hidden == 128
        assert task.neg_ratio == 5
        assert task.test_fraction == 0.2

    def test_custom_params(self) -> None:
        task = DrugTargetTask(
            k_values=[5, 15],
            classifier_hidden=64,
            neg_ratio=3,
            test_fraction=0.3,
        )
        assert task.k_values == [5, 15]
        assert task.classifier_hidden == 64
        assert task.neg_ratio == 3
        assert task.test_fraction == 0.3


# ======================================================================
# TestPrepareData
# ======================================================================


class TestPrepareData:
    """Tests for prepare_data method."""

    def test_basic_prepare(
        self,
        mock_embeddings: dict[str, torch.Tensor],
        mock_hetero_data: HeteroData,
    ) -> None:
        """prepare_data returns correctly shaped tensors."""
        task = DrugTargetTask(neg_ratio=2, seed=0)
        X_train, y_train, X_test, y_test, novel_mask = task.prepare_data(
            mock_embeddings, mock_hetero_data
        )

        # Feature dim = compound_dim + enzyme_dim
        assert X_train.shape[1] == EMB_DIM * 2
        assert X_test.shape[1] == EMB_DIM * 2

        # Labels are binary
        assert set(y_train.unique().tolist()).issubset({0.0, 1.0})
        assert set(y_test.unique().tolist()).issubset({0.0, 1.0})

        # Total samples = positives + negatives
        total = X_train.shape[0] + X_test.shape[0]
        expected = NUM_EDGES + NUM_EDGES * 2  # neg_ratio=2
        assert total == expected

        # Novel mask length matches test set
        assert len(novel_mask) == X_test.shape[0]

    def test_missing_compound_embeddings(
        self, mock_hetero_data: HeteroData
    ) -> None:
        """Raises ValueError when compound embeddings are missing."""
        task = DrugTargetTask()
        embeddings = {"enzyme": torch.randn(NUM_ENZYMES, EMB_DIM)}
        with pytest.raises(ValueError, match="compound"):
            task.prepare_data(embeddings, mock_hetero_data)

    def test_missing_enzyme_embeddings(
        self, mock_hetero_data: HeteroData
    ) -> None:
        """Raises ValueError when enzyme embeddings are missing."""
        task = DrugTargetTask()
        embeddings = {"compound": torch.randn(NUM_COMPOUNDS, EMB_DIM)}
        with pytest.raises(ValueError, match="enzyme"):
            task.prepare_data(embeddings, mock_hetero_data)

    def test_missing_edge_type(
        self, mock_embeddings: dict[str, torch.Tensor]
    ) -> None:
        """Raises ValueError when inhibits edge is absent."""
        data = HeteroData()
        data["compound"].num_nodes = NUM_COMPOUNDS
        data["enzyme"].num_nodes = NUM_ENZYMES
        task = DrugTargetTask()
        with pytest.raises(ValueError, match="not found"):
            task.prepare_data(mock_embeddings, data)


# ======================================================================
# TestTimeSplit
# ======================================================================


class TestTimeSplit:
    """Tests for time-based data splitting."""

    def test_time_split_uses_timestamps(
        self,
        mock_embeddings: dict[str, torch.Tensor],
        mock_hetero_data_with_time: HeteroData,
    ) -> None:
        """When timestamps exist, time-based split is used."""
        task = DrugTargetTask(neg_ratio=2, test_fraction=0.3, seed=0)
        X_train, y_train, X_test, y_test, novel_mask = task.prepare_data(
            mock_embeddings, mock_hetero_data_with_time
        )

        # Should produce non-empty train and test sets
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert X_train.shape[0] + X_test.shape[0] > 0

    def test_random_split_without_timestamps(
        self,
        mock_embeddings: dict[str, torch.Tensor],
        mock_hetero_data: HeteroData,
    ) -> None:
        """Without timestamps, random split is used."""
        task = DrugTargetTask(neg_ratio=2, test_fraction=0.2, seed=0)
        X_train, _, X_test, _, _ = task.prepare_data(
            mock_embeddings, mock_hetero_data
        )
        total = X_train.shape[0] + X_test.shape[0]
        # ~20% should be test
        test_frac = X_test.shape[0] / total
        assert 0.1 < test_frac < 0.4


# ======================================================================
# TestEnrichmentFactor
# ======================================================================


class TestEnrichmentFactor:
    """Tests for _enrichment_factor static method."""

    def test_perfect_enrichment(self) -> None:
        """All positives at top yields high EF."""
        y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        y_scores = np.array([0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        ef = DrugTargetTask._enrichment_factor(y_true, y_scores, fraction=0.2)
        # Top 20% (2 samples) = both positives; 2/2 = 1.0
        # Expected rate = 2/10 = 0.2; EF = 1.0 / 0.2 = 5.0
        assert ef == pytest.approx(5.0)

    def test_no_positives(self) -> None:
        """Returns 0.0 when no positives exist."""
        y_true = np.zeros(10, dtype=float)
        y_scores = np.random.rand(10)
        ef = DrugTargetTask._enrichment_factor(y_true, y_scores, fraction=0.1)
        assert ef == 0.0

    def test_empty_input(self) -> None:
        """Returns 0.0 for empty arrays."""
        ef = DrugTargetTask._enrichment_factor(
            np.array([], dtype=float), np.array([]), fraction=0.1
        )
        assert ef == 0.0

    def test_all_positives_in_bottom(self) -> None:
        """EF is 0 when positives are at the bottom."""
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=float)
        y_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
        # Top 10% (1 sample) index 0 => label 0
        ef = DrugTargetTask._enrichment_factor(y_true, y_scores, fraction=0.1)
        assert ef == 0.0

    def test_fraction_yields_at_least_one(self) -> None:
        """Even with very small fraction, at least 1 sample is checked."""
        y_true = np.array([1, 0, 0], dtype=float)
        y_scores = np.array([0.9, 0.1, 0.1])
        ef = DrugTargetTask._enrichment_factor(y_true, y_scores, fraction=0.001)
        # n_top = max(1, int(3 * 0.001)) = 1; hit=1; obs_rate=1.0;
        # exp_rate=1/3; EF=3.0
        assert ef == pytest.approx(3.0)


# ======================================================================
# TestHitAtKNovel
# ======================================================================


class TestHitAtKNovel:
    """Tests for _hit_at_k_novel static method."""

    def test_all_novel_in_top_k(self) -> None:
        """All novel positives appear in top-K."""
        y_true = np.array([1, 1, 0, 0, 0], dtype=float)
        y_scores = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
        novel_mask = np.array([True, True, False, False, False])
        hit = DrugTargetTask._hit_at_k_novel(y_true, y_scores, novel_mask, k=2)
        assert hit == pytest.approx(1.0)

    def test_no_novel_positives(self) -> None:
        """Returns 0.0 when there are no novel positives."""
        y_true = np.array([1, 0, 0], dtype=float)
        y_scores = np.array([0.9, 0.5, 0.1])
        novel_mask = np.array([False, False, False])
        hit = DrugTargetTask._hit_at_k_novel(y_true, y_scores, novel_mask, k=3)
        assert hit == 0.0

    def test_partial_hits(self) -> None:
        """Only some novel positives in top-K."""
        y_true = np.array([1, 1, 1, 0, 0], dtype=float)
        y_scores = np.array([0.9, 0.1, 0.8, 0.7, 0.6])
        novel_mask = np.array([True, True, False, False, False])
        # Top-2 indices: 0 and 2; novel positives at 0 and 1
        # Only index 0 is in top-2 => 1/2 = 0.5
        hit = DrugTargetTask._hit_at_k_novel(y_true, y_scores, novel_mask, k=2)
        assert hit == pytest.approx(0.5)


# ======================================================================
# TestEvaluatePipeline
# ======================================================================


class TestEvaluatePipeline:
    """End-to-end tests for evaluate()."""

    def test_evaluate_returns_expected_keys(
        self,
        mock_embeddings: dict[str, torch.Tensor],
        mock_hetero_data: HeteroData,
    ) -> None:
        """evaluate() returns all expected metric keys."""
        task = DrugTargetTask(k_values=[5, 10], neg_ratio=2, seed=0)
        results = task.evaluate(mock_embeddings, mock_hetero_data)

        assert "auc_roc" in results
        assert "hit@5_novel" in results
        assert "hit@10_novel" in results
        assert "enrichment_factor@1%" in results

    def test_evaluate_auc_in_range(
        self,
        mock_embeddings: dict[str, torch.Tensor],
        mock_hetero_data: HeteroData,
    ) -> None:
        """AUC-ROC must be between 0 and 1."""
        task = DrugTargetTask(neg_ratio=2, seed=0)
        results = task.evaluate(mock_embeddings, mock_hetero_data)
        assert 0.0 <= results["auc_roc"] <= 1.0

    def test_evaluate_hit_at_k_in_range(
        self,
        mock_embeddings: dict[str, torch.Tensor],
        mock_hetero_data: HeteroData,
    ) -> None:
        """Hit@K values must be between 0 and 1."""
        task = DrugTargetTask(k_values=[10], neg_ratio=2, seed=0)
        results = task.evaluate(mock_embeddings, mock_hetero_data)
        assert 0.0 <= results["hit@10_novel"] <= 1.0

    def test_evaluate_ef_non_negative(
        self,
        mock_embeddings: dict[str, torch.Tensor],
        mock_hetero_data: HeteroData,
    ) -> None:
        """Enrichment factor must be non-negative."""
        task = DrugTargetTask(neg_ratio=2, seed=0)
        results = task.evaluate(mock_embeddings, mock_hetero_data)
        assert results["enrichment_factor@1%"] >= 0.0

    def test_evaluate_with_time_split(
        self,
        mock_embeddings: dict[str, torch.Tensor],
        mock_hetero_data_with_time: HeteroData,
    ) -> None:
        """evaluate() works with time-split data."""
        task = DrugTargetTask(
            k_values=[5], neg_ratio=2, test_fraction=0.3, seed=0
        )
        results = task.evaluate(mock_embeddings, mock_hetero_data_with_time)
        assert "auc_roc" in results
        assert 0.0 <= results["auc_roc"] <= 1.0

    def test_reproducibility(
        self,
        mock_embeddings: dict[str, torch.Tensor],
        mock_hetero_data: HeteroData,
    ) -> None:
        """Same seed produces same results when torch seed is also set."""
        torch.manual_seed(0)
        task = DrugTargetTask(k_values=[5], neg_ratio=2, seed=99)
        r1 = task.evaluate(mock_embeddings, mock_hetero_data)

        torch.manual_seed(0)
        task2 = DrugTargetTask(k_values=[5], neg_ratio=2, seed=99)
        r2 = task2.evaluate(mock_embeddings, mock_hetero_data)

        assert r1["auc_roc"] == pytest.approx(r2["auc_roc"], abs=1e-4)
