"""Unit tests for DiseaseAssociationTask downstream evaluation.

Tests cover:
  - Task instantiation and name property
  - prepare_data extracts associations from HeteroData edges
  - Recall@K computation with known rankings
  - NDCG@K computation with known relevance
  - AUC-ROC computation
  - Cosine similarity scoring
  - Edge direction handling (forward and reverse)
  - Handling of rare diseases (few positive examples)
  - Empty / missing association cases
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from glycoMusubi.evaluation.downstream import BaseDownstreamTask
from glycoMusubi.evaluation.tasks.disease_association import DiseaseAssociationTask


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def task() -> DiseaseAssociationTask:
    """Default DiseaseAssociationTask."""
    return DiseaseAssociationTask(k_values=[5, 10, 20])


@pytest.fixture()
def synthetic_data() -> tuple[dict[str, torch.Tensor], HeteroData]:
    """Synthetic data with glycan-disease associations."""
    n_entities = 50
    n_diseases = 5
    dim = 16

    data = HeteroData()
    data["glycan"].x = torch.randn(n_entities, dim)
    data["glycan"].num_nodes = n_entities
    data["disease"].x = torch.randn(n_diseases, dim)
    data["disease"].num_nodes = n_diseases

    # Create associations: disease 0 has entities {0,1,2}, disease 1 has {5,10}, etc.
    src = torch.tensor([0, 1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45])
    dst = torch.tensor([0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4])
    data["glycan", "associated_with_disease", "disease"].edge_index = torch.stack(
        [src, dst]
    )

    embeddings = {
        "glycan": torch.randn(n_entities, dim),
        "disease": torch.randn(n_diseases, dim),
    }

    return embeddings, data


# ======================================================================
# TestInstantiation
# ======================================================================


class TestInstantiation:
    """Tests for DiseaseAssociationTask construction."""

    def test_is_base_downstream_task(self) -> None:
        """DiseaseAssociationTask is a subclass of BaseDownstreamTask."""
        assert issubclass(DiseaseAssociationTask, BaseDownstreamTask)

    def test_name_property(self, task: DiseaseAssociationTask) -> None:
        """Task name is 'disease_association_prediction'."""
        assert task.name == "disease_association_prediction"

    def test_default_parameters(self) -> None:
        """Default parameters are set correctly."""
        t = DiseaseAssociationTask()
        assert t.k_values == [10, 20, 50]
        assert t.relation_type == "associated_with_disease"
        assert t.source_node_type == "glycan"
        assert t.disease_node_type == "disease"

    def test_custom_parameters(self) -> None:
        """Custom parameters are applied."""
        t = DiseaseAssociationTask(
            k_values=[5, 25],
            relation_type="linked_to",
            source_node_type="protein",
            disease_node_type="condition",
        )
        assert t.k_values == [5, 25]
        assert t.relation_type == "linked_to"
        assert t.source_node_type == "protein"
        assert t.disease_node_type == "condition"


# ======================================================================
# TestPrepareData
# ======================================================================


class TestPrepareData:
    """Tests for the prepare_data method."""

    def test_extracts_associations(
        self,
        task: DiseaseAssociationTask,
        synthetic_data: tuple,
    ) -> None:
        """prepare_data returns correct shapes and associations."""
        embeddings, data = synthetic_data
        entity_emb, disease_emb, d2e = task.prepare_data(embeddings, data)

        assert entity_emb.shape == (50, 16)
        assert disease_emb.shape == (5, 16)
        assert len(d2e) == 5
        assert d2e[0] == {0, 1, 2}
        assert d2e[1] == {5, 10}

    def test_missing_entity_key_raises(self, task: DiseaseAssociationTask) -> None:
        """Missing source_node_type key raises ValueError."""
        data = HeteroData()
        with pytest.raises(ValueError, match="glycan"):
            task.prepare_data({"disease": torch.randn(3, 8)}, data)

    def test_missing_disease_key_raises(self, task: DiseaseAssociationTask) -> None:
        """Missing disease_node_type key raises ValueError."""
        data = HeteroData()
        with pytest.raises(ValueError, match="disease"):
            task.prepare_data({"glycan": torch.randn(10, 8)}, data)

    def test_reverse_edge_direction(self) -> None:
        """Associations extracted correctly with reversed edge direction."""
        task = DiseaseAssociationTask()
        data = HeteroData()
        data["glycan"].x = torch.randn(10, 8)
        data["glycan"].num_nodes = 10
        data["disease"].x = torch.randn(3, 8)
        data["disease"].num_nodes = 3

        # Reversed: (disease, associated_with_disease, glycan)
        src = torch.tensor([0, 0, 1, 2])
        dst = torch.tensor([1, 3, 5, 7])
        data["disease", "associated_with_disease", "glycan"].edge_index = (
            torch.stack([src, dst])
        )

        embeddings = {
            "glycan": torch.randn(10, 8),
            "disease": torch.randn(3, 8),
        }
        _, _, d2e = task.prepare_data(embeddings, data)

        assert d2e[0] == {1, 3}
        assert d2e[1] == {5}
        assert d2e[2] == {7}

    def test_no_matching_edges(self) -> None:
        """Returns empty associations when no matching edge type exists."""
        task = DiseaseAssociationTask(relation_type="nonexistent_relation")
        data = HeteroData()
        data["glycan"].x = torch.randn(10, 8)
        data["glycan"].num_nodes = 10
        data["disease"].x = torch.randn(3, 8)
        data["disease"].num_nodes = 3

        embeddings = {
            "glycan": torch.randn(10, 8),
            "disease": torch.randn(3, 8),
        }
        _, _, d2e = task.prepare_data(embeddings, data)
        assert len(d2e) == 0


# ======================================================================
# TestMetricHelpers
# ======================================================================


class TestMetricHelpers:
    """Tests for individual metric computation methods."""

    def test_recall_at_k_perfect(self) -> None:
        """All true items in top-K gives recall = 1.0."""
        ranked = np.array([0, 1, 2, 3, 4])
        true_idx = np.array([0, 1])
        recall = DiseaseAssociationTask._compute_recall_at_k(ranked, true_idx, k=5)
        assert recall == pytest.approx(1.0)

    def test_recall_at_k_partial(self) -> None:
        """Only some true items in top-K."""
        ranked = np.array([0, 1, 2, 3, 4])
        true_idx = np.array([0, 1, 4])
        recall = DiseaseAssociationTask._compute_recall_at_k(ranked, true_idx, k=3)
        assert recall == pytest.approx(2.0 / 3.0)

    def test_recall_at_k_none(self) -> None:
        """No true items in top-K gives recall = 0.0."""
        ranked = np.array([0, 1, 2, 3, 4])
        true_idx = np.array([3, 4])
        recall = DiseaseAssociationTask._compute_recall_at_k(ranked, true_idx, k=2)
        assert recall == pytest.approx(0.0)

    def test_recall_at_k_empty_true(self) -> None:
        """Empty true indices gives recall = 0.0."""
        ranked = np.array([0, 1, 2])
        recall = DiseaseAssociationTask._compute_recall_at_k(
            ranked, np.array([]), k=3
        )
        assert recall == 0.0

    def test_ndcg_perfect_ranking(self) -> None:
        """Perfect ranking gives NDCG = 1.0."""
        scores = np.array([0.9, 0.8, 0.1, 0.05])
        true_idx = np.array([0, 1])
        ndcg = DiseaseAssociationTask._compute_ndcg(scores, true_idx, k=4)
        assert ndcg == pytest.approx(1.0)

    def test_ndcg_worst_ranking(self) -> None:
        """All relevant items at bottom gives NDCG < 1."""
        scores = np.array([0.01, 0.02, 0.9, 0.8])
        true_idx = np.array([0, 1])
        ndcg = DiseaseAssociationTask._compute_ndcg(scores, true_idx, k=4)
        assert ndcg < 1.0

    def test_ndcg_empty_true(self) -> None:
        """Empty true indices gives NDCG = 0.0."""
        scores = np.array([0.5, 0.3, 0.1])
        ndcg = DiseaseAssociationTask._compute_ndcg(scores, np.array([]), k=3)
        assert ndcg == 0.0

    def test_ndcg_single_relevant(self) -> None:
        """Single relevant item at rank 1 gives NDCG = 1.0."""
        scores = np.array([0.9, 0.1, 0.05])
        true_idx = np.array([0])
        ndcg = DiseaseAssociationTask._compute_ndcg(scores, true_idx, k=3)
        assert ndcg == pytest.approx(1.0)

    def test_ndcg_known_value(self) -> None:
        """NDCG with known value: relevant item at position 2 (0-indexed 1)."""
        # Item at index 2 is relevant, but has the 2nd highest score
        scores = np.array([0.9, 0.1, 0.8, 0.05])
        true_idx = np.array([2])
        ndcg = DiseaseAssociationTask._compute_ndcg(scores, true_idx, k=4)
        # Item 2 ends up at rank 2 (0.8 is second highest)
        # DCG = 1/log2(3) = 0.6309
        # IDCG = 1/log2(2) = 1.0
        expected = (1.0 / np.log2(3)) / (1.0 / np.log2(2))
        assert ndcg == pytest.approx(expected, rel=1e-4)

    def test_cosine_scores_orthogonal(self) -> None:
        """Orthogonal vectors have zero cosine similarity."""
        entity_emb = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        disease_vec = np.array([0.0, 1.0])
        scores = DiseaseAssociationTask._cosine_scores(entity_emb, disease_vec)
        assert scores[0] == pytest.approx(0.0, abs=1e-10)
        assert scores[1] == pytest.approx(1.0, abs=1e-10)
        assert scores[2] == pytest.approx(0.0, abs=1e-10)

    def test_cosine_scores_parallel(self) -> None:
        """Parallel vectors have cosine similarity = 1.0."""
        entity_emb = np.array([[2.0, 3.0]])
        disease_vec = np.array([4.0, 6.0])
        scores = DiseaseAssociationTask._cosine_scores(entity_emb, disease_vec)
        assert scores[0] == pytest.approx(1.0, abs=1e-10)

    def test_cosine_scores_zero_disease_vec(self) -> None:
        """Zero disease vector returns all-zero scores."""
        entity_emb = np.array([[1.0, 2.0], [3.0, 4.0]])
        disease_vec = np.array([0.0, 0.0])
        scores = DiseaseAssociationTask._cosine_scores(entity_emb, disease_vec)
        assert np.allclose(scores, 0.0)

    def test_auc_roc_perfect(self) -> None:
        """Perfect separation gives AUC-ROC = 1.0."""
        y_true = np.array([1.0, 1.0, 0.0, 0.0])
        y_scores = np.array([0.9, 0.8, 0.2, 0.1])
        auc = DiseaseAssociationTask._compute_auc_roc(y_true, y_scores)
        assert auc == pytest.approx(1.0)

    def test_auc_roc_random(self) -> None:
        """Random predictions give AUC-ROC close to 0.5 on large sample."""
        rng = np.random.RandomState(42)
        n = 1000
        y_true = np.zeros(n)
        y_true[:100] = 1.0
        y_scores = rng.rand(n)
        auc = DiseaseAssociationTask._compute_auc_roc(y_true, y_scores)
        assert 0.3 < auc < 0.7


# ======================================================================
# TestEvaluate
# ======================================================================


class TestEvaluate:
    """Tests for the evaluate method."""

    def test_returns_all_metric_keys(
        self,
        task: DiseaseAssociationTask,
        synthetic_data: tuple,
    ) -> None:
        """evaluate returns auc_roc, recall@K, ndcg@K for all K."""
        embeddings, data = synthetic_data
        results = task.evaluate(embeddings, data)

        assert "auc_roc" in results
        for k in [5, 10, 20]:
            assert f"recall@{k}" in results
            assert f"ndcg@{k}" in results
        assert "num_diseases_evaluated" in results

    def test_num_diseases_evaluated(
        self,
        task: DiseaseAssociationTask,
        synthetic_data: tuple,
    ) -> None:
        """All diseases with associations should be evaluated."""
        embeddings, data = synthetic_data
        results = task.evaluate(embeddings, data)

        assert results["num_diseases_evaluated"] == 5.0

    def test_metrics_in_valid_range(
        self,
        task: DiseaseAssociationTask,
        synthetic_data: tuple,
    ) -> None:
        """All metric values are in valid ranges."""
        embeddings, data = synthetic_data
        results = task.evaluate(embeddings, data)

        assert 0.0 <= results["auc_roc"] <= 1.0
        for k in [5, 10, 20]:
            assert 0.0 <= results[f"recall@{k}"] <= 1.0
            assert 0.0 <= results[f"ndcg@{k}"] <= 1.0

    def test_no_associations_returns_zeros(self) -> None:
        """No disease associations returns all-zero metrics."""
        task = DiseaseAssociationTask(
            k_values=[5, 10],
            relation_type="nonexistent",
        )
        data = HeteroData()
        data["glycan"].x = torch.randn(10, 8)
        data["glycan"].num_nodes = 10
        data["disease"].x = torch.randn(3, 8)
        data["disease"].num_nodes = 3

        embeddings = {
            "glycan": torch.randn(10, 8),
            "disease": torch.randn(3, 8),
        }
        results = task.evaluate(embeddings, data)

        assert results["auc_roc"] == 0.0
        assert results["recall@5"] == 0.0
        assert results["ndcg@10"] == 0.0
        assert results["num_diseases_evaluated"] == 0.0

    def test_rare_disease_single_association(self) -> None:
        """Handles rare diseases with only one known association."""
        task = DiseaseAssociationTask(k_values=[3, 5])
        data = HeteroData()
        n_entities = 20
        n_diseases = 2
        data["glycan"].x = torch.randn(n_entities, 8)
        data["glycan"].num_nodes = n_entities
        data["disease"].x = torch.randn(n_diseases, 8)
        data["disease"].num_nodes = n_diseases

        # Each disease has only 1 associated entity
        src = torch.tensor([0, 5])
        dst = torch.tensor([0, 1])
        data["glycan", "associated_with_disease", "disease"].edge_index = (
            torch.stack([src, dst])
        )

        embeddings = {
            "glycan": torch.randn(n_entities, 8),
            "disease": torch.randn(n_diseases, 8),
        }
        results = task.evaluate(embeddings, data)

        assert results["num_diseases_evaluated"] == 2.0
        assert 0.0 <= results["auc_roc"] <= 1.0
        for k in [3, 5]:
            assert 0.0 <= results[f"recall@{k}"] <= 1.0

    def test_perfect_scores_high_metrics(self) -> None:
        """When disease embedding equals its only associated entity,
        that entity should rank first, yielding perfect metrics."""
        task = DiseaseAssociationTask(k_values=[1, 5])
        n_entities = 10
        n_diseases = 1
        dim = 8

        data = HeteroData()
        data["glycan"].x = torch.randn(n_entities, dim)
        data["glycan"].num_nodes = n_entities
        data["disease"].x = torch.randn(n_diseases, dim)
        data["disease"].num_nodes = n_diseases

        src = torch.tensor([3])
        dst = torch.tensor([0])
        data["glycan", "associated_with_disease", "disease"].edge_index = (
            torch.stack([src, dst])
        )

        # Make entity 3's embedding identical to disease 0
        entity_emb = torch.randn(n_entities, dim)
        disease_emb = entity_emb[3].unsqueeze(0).clone()

        embeddings = {"glycan": entity_emb, "disease": disease_emb}
        results = task.evaluate(embeddings, data)

        assert results["recall@1"] == pytest.approx(1.0)
        assert results["ndcg@1"] == pytest.approx(1.0)
        assert results["auc_roc"] == pytest.approx(1.0)

    def test_custom_node_types(self) -> None:
        """Task works with non-default node type names."""
        task = DiseaseAssociationTask(
            k_values=[3],
            relation_type="linked_to",
            source_node_type="protein",
            disease_node_type="condition",
        )

        data = HeteroData()
        data["protein"].x = torch.randn(10, 8)
        data["protein"].num_nodes = 10
        data["condition"].x = torch.randn(2, 8)
        data["condition"].num_nodes = 2

        src = torch.tensor([0, 1, 3])
        dst = torch.tensor([0, 0, 1])
        data["protein", "linked_to", "condition"].edge_index = (
            torch.stack([src, dst])
        )

        embeddings = {
            "protein": torch.randn(10, 8),
            "condition": torch.randn(2, 8),
        }
        results = task.evaluate(embeddings, data)

        assert results["num_diseases_evaluated"] == 2.0
        assert "recall@3" in results
