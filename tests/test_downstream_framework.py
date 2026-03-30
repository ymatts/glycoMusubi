"""Tests for the downstream evaluation framework.

Tests cover:
  - BaseDownstreamTask is abstract (cannot be instantiated directly)
  - Concrete subclasses must implement prepare_data, evaluate, name
  - DownstreamEvaluator instantiation with multiple tasks
  - DownstreamEvaluator.evaluate_all runs all tasks and returns correct structure
  - Results are dict[str, dict[str, float]]
  - Embedding extraction helper works with mock BaseKGEModel
"""

from __future__ import annotations

from typing import Dict
from unittest.mock import MagicMock

import pytest
import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from glycoMusubi.evaluation.downstream import BaseDownstreamTask, DownstreamEvaluator


# ======================================================================
# Helpers
# ======================================================================


class _ConcreteTask(BaseDownstreamTask):
    """Minimal concrete implementation for testing."""

    def __init__(self, task_name: str = "dummy_task", metrics: dict | None = None):
        self._name = task_name
        self._metrics = metrics or {"accuracy": 0.9, "f1": 0.85}

    @property
    def name(self) -> str:
        return self._name

    def prepare_data(self, embeddings: Dict[str, Tensor], data: HeteroData) -> tuple:
        return (embeddings, data)

    def evaluate(
        self, embeddings: Dict[str, Tensor], data: HeteroData, **kwargs
    ) -> Dict[str, float]:
        return dict(self._metrics)


class _FailingTask(BaseDownstreamTask):
    """Task that always raises during evaluate."""

    @property
    def name(self) -> str:
        return "failing_task"

    def prepare_data(self, embeddings: Dict[str, Tensor], data: HeteroData) -> tuple:
        return ()

    def evaluate(
        self, embeddings: Dict[str, Tensor], data: HeteroData, **kwargs
    ) -> Dict[str, float]:
        raise RuntimeError("Intentional failure for testing")


class _PartialTask(BaseDownstreamTask):
    """Only implements name but not the other abstract methods."""

    @property
    def name(self) -> str:
        return "partial"


@pytest.fixture()
def tiny_hetero_data() -> HeteroData:
    """Minimal HeteroData with glycan and protein nodes + edges."""
    data = HeteroData()
    data["glycan"].x = torch.randn(5, 16)
    data["glycan"].num_nodes = 5
    data["protein"].x = torch.randn(4, 16)
    data["protein"].num_nodes = 4
    data["glycan", "binds", "protein"].edge_index = torch.tensor(
        [[0, 1, 2], [0, 1, 2]]
    )
    return data


@pytest.fixture()
def tiny_embeddings() -> Dict[str, Tensor]:
    """Embeddings matching tiny_hetero_data."""
    return {
        "glycan": torch.randn(5, 16),
        "protein": torch.randn(4, 16),
    }


# ======================================================================
# TestBaseDownstreamTask
# ======================================================================


class TestBaseDownstreamTask:
    """Tests for the abstract base class."""

    def test_cannot_instantiate_directly(self) -> None:
        """BaseDownstreamTask is abstract and cannot be instantiated."""
        with pytest.raises(TypeError, match="abstract method"):
            BaseDownstreamTask()

    def test_partial_implementation_cannot_instantiate(self) -> None:
        """A subclass missing abstract methods cannot be instantiated."""
        with pytest.raises(TypeError, match="abstract method"):
            _PartialTask()

    def test_concrete_subclass_instantiates(self) -> None:
        """A fully concrete subclass can be created."""
        task = _ConcreteTask()
        assert task.name == "dummy_task"

    def test_concrete_subclass_has_required_methods(self) -> None:
        """Concrete subclass has prepare_data, evaluate, and name."""
        task = _ConcreteTask()
        assert callable(task.prepare_data)
        assert callable(task.evaluate)
        assert isinstance(task.name, str)

    def test_prepare_data_returns_tuple(
        self, tiny_embeddings: Dict[str, Tensor], tiny_hetero_data: HeteroData
    ) -> None:
        """prepare_data returns a tuple."""
        task = _ConcreteTask()
        result = task.prepare_data(tiny_embeddings, tiny_hetero_data)
        assert isinstance(result, tuple)

    def test_evaluate_returns_dict(
        self, tiny_embeddings: Dict[str, Tensor], tiny_hetero_data: HeteroData
    ) -> None:
        """evaluate returns a dict of metric name to float value."""
        task = _ConcreteTask()
        result = task.evaluate(tiny_embeddings, tiny_hetero_data)
        assert isinstance(result, dict)
        for k, v in result.items():
            assert isinstance(k, str)
            assert isinstance(v, float)


# ======================================================================
# TestDownstreamEvaluator
# ======================================================================


class TestDownstreamEvaluator:
    """Tests for DownstreamEvaluator."""

    def test_instantiation_single_task(self) -> None:
        """Can create evaluator with a single task."""
        evaluator = DownstreamEvaluator(tasks=[_ConcreteTask()])
        assert len(evaluator.tasks) == 1

    def test_instantiation_multiple_tasks(self) -> None:
        """Can create evaluator with multiple tasks."""
        tasks = [
            _ConcreteTask("task_a", {"m1": 0.8}),
            _ConcreteTask("task_b", {"m2": 0.7}),
            _ConcreteTask("task_c", {"m3": 0.6}),
        ]
        evaluator = DownstreamEvaluator(tasks=tasks)
        assert len(evaluator.tasks) == 3

    def test_evaluate_all_runs_all_tasks(
        self, tiny_embeddings: Dict[str, Tensor], tiny_hetero_data: HeteroData
    ) -> None:
        """evaluate_all runs every registered task and returns results for each."""
        tasks = [
            _ConcreteTask("alpha", {"score": 0.9}),
            _ConcreteTask("beta", {"score": 0.8}),
        ]
        evaluator = DownstreamEvaluator(tasks=tasks)
        results = evaluator.evaluate_all(tiny_embeddings, tiny_hetero_data)

        assert "alpha" in results
        assert "beta" in results
        assert results["alpha"]["score"] == pytest.approx(0.9)
        assert results["beta"]["score"] == pytest.approx(0.8)

    def test_evaluate_all_result_structure(
        self, tiny_embeddings: Dict[str, Tensor], tiny_hetero_data: HeteroData
    ) -> None:
        """Results are dict[str, dict[str, float]]."""
        evaluator = DownstreamEvaluator(
            tasks=[_ConcreteTask("t", {"a": 0.5, "b": 0.6})]
        )
        results = evaluator.evaluate_all(tiny_embeddings, tiny_hetero_data)

        assert isinstance(results, dict)
        for task_name, metrics in results.items():
            assert isinstance(task_name, str)
            assert isinstance(metrics, dict)
            for metric_name, value in metrics.items():
                assert isinstance(metric_name, str)
                assert isinstance(value, float)

    def test_evaluate_all_handles_failing_task(
        self, tiny_embeddings: Dict[str, Tensor], tiny_hetero_data: HeteroData
    ) -> None:
        """A failing task does not crash evaluate_all; it gets empty metrics."""
        tasks = [
            _ConcreteTask("good_task", {"m": 0.9}),
            _FailingTask(),
        ]
        evaluator = DownstreamEvaluator(tasks=tasks)
        results = evaluator.evaluate_all(tiny_embeddings, tiny_hetero_data)

        assert results["good_task"]["m"] == pytest.approx(0.9)
        assert results["failing_task"] == {}

    def test_evaluate_all_empty_task_list(
        self, tiny_embeddings: Dict[str, Tensor], tiny_hetero_data: HeteroData
    ) -> None:
        """Evaluator with no tasks returns empty results dict."""
        evaluator = DownstreamEvaluator(tasks=[])
        results = evaluator.evaluate_all(tiny_embeddings, tiny_hetero_data)
        assert results == {}


# ======================================================================
# TestEmbeddingExtractionWithMockModel
# ======================================================================


class TestEmbeddingExtractionWithMockModel:
    """Test that embedding extraction works with a mock BaseKGEModel."""

    def test_get_embeddings_returns_dict(self, tiny_hetero_data: HeteroData) -> None:
        """A mock model's get_embeddings returns detached embeddings dict."""
        mock_model = MagicMock()
        mock_model.forward.return_value = {
            "glycan": torch.randn(5, 16),
            "protein": torch.randn(4, 16),
        }
        mock_model.get_embeddings = MagicMock(
            side_effect=lambda data: {
                k: v.detach() for k, v in mock_model.forward(data).items()
            }
        )

        embeddings = mock_model.get_embeddings(tiny_hetero_data)
        assert isinstance(embeddings, dict)
        assert "glycan" in embeddings
        assert "protein" in embeddings
        assert embeddings["glycan"].shape == (5, 16)
        assert embeddings["protein"].shape == (4, 16)

    def test_embeddings_fed_to_evaluator(self, tiny_hetero_data: HeteroData) -> None:
        """Embeddings from a mock model can be consumed by DownstreamEvaluator."""
        mock_model = MagicMock()
        emb = {
            "glycan": torch.randn(5, 16),
            "protein": torch.randn(4, 16),
        }
        mock_model.get_embeddings.return_value = emb

        evaluator = DownstreamEvaluator(
            tasks=[_ConcreteTask("task1", {"auc": 0.95})]
        )
        embeddings = mock_model.get_embeddings(tiny_hetero_data)
        results = evaluator.evaluate_all(embeddings, tiny_hetero_data)

        assert results["task1"]["auc"] == pytest.approx(0.95)


# ======================================================================
# TestEvaluateMultiSeed
# ======================================================================


class TestEvaluateMultiSeed:
    """Tests for DownstreamEvaluator.evaluate_multi_seed."""

    def test_multi_seed_returns_mean_std(self, tiny_hetero_data: HeteroData) -> None:
        """evaluate_multi_seed aggregates per-seed results into mean/std."""
        task = _ConcreteTask("mt", {"accuracy": 0.9})
        evaluator = DownstreamEvaluator(tasks=[task])

        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model.get_embeddings = MagicMock(
            return_value={
                "glycan": torch.randn(5, 16),
                "protein": torch.randn(4, 16),
            }
        )

        result = evaluator.evaluate_multi_seed(
            model_factory=lambda: mock_model,
            data=tiny_hetero_data,
            seeds=[42, 123],
        )

        assert "mt" in result
        assert "accuracy" in result["mt"]
        assert "mean" in result["mt"]["accuracy"]
        assert "std" in result["mt"]["accuracy"]
        assert result["mt"]["accuracy"]["mean"] == pytest.approx(0.9)
