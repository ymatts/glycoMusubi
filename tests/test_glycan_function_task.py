"""Unit tests for GlycanFunctionTask downstream evaluation.

Tests cover:
  - Task instantiation and name property
  - prepare_data with multiple label conventions
  - Cross-validated MLP classification per taxonomy level
  - Accuracy, macro F1, and MCC metric computation
  - Aggregate metrics across levels
  - Handling of missing labels, single-class, and insufficient samples
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from glycoMusubi.evaluation.downstream import BaseDownstreamTask
from glycoMusubi.evaluation.tasks.glycan_function import GlycanFunctionTask


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def task() -> GlycanFunctionTask:
    """Default GlycanFunctionTask with reduced folds/iterations for speed."""
    return GlycanFunctionTask(
        classifier_hidden=32,
        n_folds=3,
        max_iter=50,
        random_state=42,
        min_samples_per_level=6,
    )


@pytest.fixture()
def synthetic_data() -> tuple[dict[str, torch.Tensor], HeteroData]:
    """Synthetic glycan embeddings and HeteroData with taxonomy labels.

    Creates separable clusters so the classifier can learn them.
    """
    rng = np.random.RandomState(42)
    n_glycans = 60
    dim = 16

    # Generate embeddings with 3 clusters for domain-level classification
    labels_domain = np.array([0] * 20 + [1] * 20 + [2] * 20)
    X = np.zeros((n_glycans, dim))
    for i in range(n_glycans):
        cluster = labels_domain[i]
        X[i] = rng.randn(dim) * 0.3 + cluster * 2.0  # well-separated

    embeddings = {"glycan": torch.tensor(X, dtype=torch.float32)}

    data = HeteroData()
    data["glycan"].x = torch.randn(n_glycans, dim)
    data["glycan"].num_nodes = n_glycans
    data["glycan"].domain = torch.tensor(labels_domain, dtype=torch.long)
    data["glycan"].kingdom = torch.tensor(
        rng.randint(0, 4, size=n_glycans), dtype=torch.long
    )

    return embeddings, data


# ======================================================================
# TestInstantiation
# ======================================================================


class TestInstantiation:
    """Tests for GlycanFunctionTask construction."""

    def test_is_base_downstream_task(self) -> None:
        """GlycanFunctionTask is a subclass of BaseDownstreamTask."""
        assert issubclass(GlycanFunctionTask, BaseDownstreamTask)

    def test_name_property(self, task: GlycanFunctionTask) -> None:
        """Task name is 'glycan_function_prediction'."""
        assert task.name == "glycan_function_prediction"

    def test_taxonomy_levels(self) -> None:
        """All 8 taxonomy levels are defined."""
        assert len(GlycanFunctionTask.TAXONOMY_LEVELS) == 8
        assert GlycanFunctionTask.TAXONOMY_LEVELS[0] == "domain"
        assert GlycanFunctionTask.TAXONOMY_LEVELS[-1] == "species"

    def test_default_parameters(self) -> None:
        """Default parameters are set correctly."""
        t = GlycanFunctionTask()
        assert t.classifier_hidden == 128
        assert t.n_folds == 5
        assert t.max_iter == 300
        assert t.random_state == 42
        assert t.min_samples_per_level == 10

    def test_custom_parameters(self) -> None:
        """Custom parameters are applied."""
        t = GlycanFunctionTask(
            classifier_hidden=64,
            n_folds=3,
            max_iter=100,
            random_state=123,
            min_samples_per_level=5,
        )
        assert t.classifier_hidden == 64
        assert t.n_folds == 3
        assert t.max_iter == 100
        assert t.random_state == 123
        assert t.min_samples_per_level == 5


# ======================================================================
# TestPrepareData
# ======================================================================


class TestPrepareData:
    """Tests for the prepare_data method."""

    def test_extracts_embeddings_and_labels(
        self,
        task: GlycanFunctionTask,
        synthetic_data: tuple,
    ) -> None:
        """prepare_data returns numpy arrays with correct shapes."""
        embeddings, data = synthetic_data
        X, labels = task.prepare_data(embeddings, data)

        assert isinstance(X, np.ndarray)
        assert X.shape == (60, 16)
        assert "domain" in labels
        assert "kingdom" in labels
        assert labels["domain"].shape == (60,)

    def test_missing_glycan_key_raises(self, task: GlycanFunctionTask) -> None:
        """Missing 'glycan' key in embeddings raises ValueError."""
        data = HeteroData()
        with pytest.raises(ValueError, match="glycan"):
            task.prepare_data({"protein": torch.randn(10, 8)}, data)

    def test_taxonomy_dict_convention(self, task: GlycanFunctionTask) -> None:
        """Labels extracted via data['glycan'].taxonomy dict."""
        data = HeteroData()
        data["glycan"].x = torch.randn(10, 8)
        data["glycan"].num_nodes = 10
        data["glycan"].taxonomy = {
            "domain": torch.randint(0, 2, (10,)),
            "phylum": torch.randint(0, 3, (10,)),
        }
        embeddings = {"glycan": torch.randn(10, 8)}
        X, labels = task.prepare_data(embeddings, data)

        assert "domain" in labels
        assert "phylum" in labels
        assert len(labels) == 2

    def test_taxonomy_prefix_convention(self, task: GlycanFunctionTask) -> None:
        """Labels extracted via data['glycan'].taxonomy_{level} attributes."""
        data = HeteroData()
        data["glycan"].x = torch.randn(10, 8)
        data["glycan"].num_nodes = 10
        data["glycan"].taxonomy_order = torch.randint(0, 5, (10,))
        embeddings = {"glycan": torch.randn(10, 8)}
        X, labels = task.prepare_data(embeddings, data)

        assert "order" in labels

    def test_no_labels_found(self, task: GlycanFunctionTask) -> None:
        """prepare_data returns empty labels when no taxonomy data exists."""
        data = HeteroData()
        data["glycan"].x = torch.randn(10, 8)
        data["glycan"].num_nodes = 10
        embeddings = {"glycan": torch.randn(10, 8)}
        X, labels = task.prepare_data(embeddings, data)

        assert len(labels) == 0

    def test_no_glycan_node_type(self, task: GlycanFunctionTask) -> None:
        """prepare_data returns empty labels when 'glycan' node type is missing from data."""
        data = HeteroData()
        data["protein"].x = torch.randn(10, 8)
        data["protein"].num_nodes = 10
        embeddings = {"glycan": torch.randn(10, 8)}
        X, labels = task.prepare_data(embeddings, data)

        assert len(labels) == 0


# ======================================================================
# TestEvaluate
# ======================================================================


class TestEvaluate:
    """Tests for the evaluate method."""

    def test_returns_per_level_metrics(
        self,
        task: GlycanFunctionTask,
        synthetic_data: tuple,
    ) -> None:
        """evaluate returns accuracy/f1/mcc for each level with data."""
        embeddings, data = synthetic_data
        results = task.evaluate(embeddings, data)

        assert "domain_accuracy" in results
        assert "domain_f1" in results
        assert "domain_mcc" in results
        assert "kingdom_accuracy" in results

    def test_returns_aggregate_metrics(
        self,
        task: GlycanFunctionTask,
        synthetic_data: tuple,
    ) -> None:
        """evaluate returns mean_accuracy, mean_f1, mean_mcc."""
        embeddings, data = synthetic_data
        results = task.evaluate(embeddings, data)

        assert "mean_accuracy" in results
        assert "mean_f1" in results
        assert "mean_mcc" in results
        assert "num_levels_evaluated" in results
        assert results["num_levels_evaluated"] >= 1

    def test_separable_data_high_accuracy(self) -> None:
        """Well-separated clusters should yield high accuracy."""
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=300,
            n_features=16,
            n_informative=8,
            n_classes=3,
            n_clusters_per_class=1,
            class_sep=3.0,
            random_state=42,
        )

        embeddings = {"glycan": torch.tensor(X, dtype=torch.float32)}
        data = HeteroData()
        n = len(y)
        data["glycan"].x = torch.randn(n, 16)
        data["glycan"].num_nodes = n
        data["glycan"].domain = torch.tensor(y, dtype=torch.long)

        task = GlycanFunctionTask(
            classifier_hidden=32,
            n_folds=3,
            max_iter=500,
            min_samples_per_level=6,
        )
        results = task.evaluate(embeddings, data)

        assert results["domain_accuracy"] > 0.85
        assert results["domain_f1"] > 0.80

    def test_metrics_in_valid_range(
        self,
        task: GlycanFunctionTask,
        synthetic_data: tuple,
    ) -> None:
        """All metric values are in valid ranges."""
        embeddings, data = synthetic_data
        results = task.evaluate(embeddings, data)

        for key, value in results.items():
            if key == "num_levels_evaluated":
                assert value >= 0
            elif "accuracy" in key or "f1" in key:
                assert 0.0 <= value <= 1.0, f"{key}={value} out of [0,1]"
            elif "mcc" in key:
                assert -1.0 <= value <= 1.0, f"{key}={value} out of [-1,1]"

    def test_skips_insufficient_samples(self) -> None:
        """Levels with too few samples are skipped."""
        n = 6
        embeddings = {"glycan": torch.randn(n, 8)}
        data = HeteroData()
        data["glycan"].x = torch.randn(n, 8)
        data["glycan"].num_nodes = n
        data["glycan"].domain = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

        task = GlycanFunctionTask(
            n_folds=2,
            max_iter=10,
            min_samples_per_level=100,  # require 100 samples
        )
        results = task.evaluate(embeddings, data)

        assert results["num_levels_evaluated"] == 0
        assert results["mean_accuracy"] == 0.0

    def test_skips_single_class(self) -> None:
        """Levels where all samples have the same class are skipped."""
        n = 20
        embeddings = {"glycan": torch.randn(n, 8)}
        data = HeteroData()
        data["glycan"].x = torch.randn(n, 8)
        data["glycan"].num_nodes = n
        data["glycan"].domain = torch.zeros(n, dtype=torch.long)  # all same class

        task = GlycanFunctionTask(n_folds=2, max_iter=10, min_samples_per_level=5)
        results = task.evaluate(embeddings, data)

        assert "domain_accuracy" not in results
        assert results["num_levels_evaluated"] == 0

    def test_no_labels_returns_zero_metrics(self, task: GlycanFunctionTask) -> None:
        """No taxonomy labels yields zero aggregate metrics."""
        embeddings = {"glycan": torch.randn(20, 8)}
        data = HeteroData()
        data["glycan"].x = torch.randn(20, 8)
        data["glycan"].num_nodes = 20

        results = task.evaluate(embeddings, data)

        assert results["mean_accuracy"] == 0.0
        assert results["mean_f1"] == 0.0
        assert results["mean_mcc"] == 0.0
        assert results["num_levels_evaluated"] == 0.0

    def test_multiple_levels(self) -> None:
        """Evaluate works across multiple taxonomy levels simultaneously."""
        rng = np.random.RandomState(42)
        n = 60
        dim = 8

        embeddings = {"glycan": torch.randn(n, dim)}
        data = HeteroData()
        data["glycan"].x = torch.randn(n, dim)
        data["glycan"].num_nodes = n
        data["glycan"].domain = torch.tensor(
            rng.randint(0, 2, size=n), dtype=torch.long
        )
        data["glycan"].kingdom = torch.tensor(
            rng.randint(0, 3, size=n), dtype=torch.long
        )
        data["glycan"].phylum = torch.tensor(
            rng.randint(0, 4, size=n), dtype=torch.long
        )

        task = GlycanFunctionTask(
            n_folds=2, max_iter=20, min_samples_per_level=6
        )
        results = task.evaluate(embeddings, data)

        assert results["num_levels_evaluated"] >= 2
        # Aggregate should be average of per-level
        level_accs = [
            v for k, v in results.items()
            if k.endswith("_accuracy") and k != "mean_accuracy"
        ]
        if level_accs:
            assert results["mean_accuracy"] == pytest.approx(
                np.mean(level_accs), rel=1e-6
            )
