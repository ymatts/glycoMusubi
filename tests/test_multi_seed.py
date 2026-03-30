"""Unit tests for multi-seed evaluation.

Tests cover:
  - Runs with multiple seeds
  - Returns mean and std for metrics
  - Reproducible with same seeds
  - Different seeds produce different results
  - Handles model_factory correctly
"""

from __future__ import annotations

from typing import Dict

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from glycoMusubi.evaluation.multi_seed import multi_seed_evaluation

# ======================================================================
# Constants
# ======================================================================

EMBEDDING_DIM = 16
NUM_NODES = 10


# ======================================================================
# Helpers
# ======================================================================


class TinyModel(nn.Module):
    """Minimal model whose output depends on random init and seed."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _make_data() -> HeteroData:
    """Create a minimal HeteroData for testing."""
    data = HeteroData()
    data["node"].x = torch.randn(NUM_NODES, EMBEDDING_DIM)
    data["node"].num_nodes = NUM_NODES
    data["node", "edge", "node"].edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]]
    )
    return data


def _dummy_eval_fn(model: nn.Module, data: HeteroData) -> Dict[str, float]:
    """Evaluate using model output norm as a proxy metric."""
    model.eval()
    with torch.no_grad():
        out = model(data["node"].x)
        mrr = out.mean().item()
        hits1 = out.std().item()
    return {
        "MRR": mrr,
        "Hits@1": hits1,
    }


def _dummy_train_fn(model: nn.Module, data: HeteroData, seed: int) -> None:
    """Minimal training step that modifies model weights."""
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model.train()
    out = model(data["node"].x)
    loss = out.sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def data() -> HeteroData:
    """Minimal HeteroData."""
    torch.manual_seed(42)
    return _make_data()


# ======================================================================
# TestMultiSeedRuns
# ======================================================================


class TestMultiSeedRuns:
    """Tests that multi_seed_evaluation runs correctly with multiple seeds."""

    def test_runs_with_multiple_seeds(self, data: HeteroData) -> None:
        """Function completes with 5 default seeds."""
        result = multi_seed_evaluation(
            model_factory=TinyModel,
            data=data,
            eval_fn=_dummy_eval_fn,
        )
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_runs_with_custom_seeds(self, data: HeteroData) -> None:
        """Function completes with custom seed list."""
        seeds = [1, 2, 3]
        result = multi_seed_evaluation(
            model_factory=TinyModel,
            data=data,
            seeds=seeds,
            eval_fn=_dummy_eval_fn,
        )
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_runs_with_train_fn(self, data: HeteroData) -> None:
        """Function completes when train_fn is provided."""
        result = multi_seed_evaluation(
            model_factory=TinyModel,
            data=data,
            seeds=[42, 123],
            train_fn=_dummy_train_fn,
            eval_fn=_dummy_eval_fn,
        )
        assert isinstance(result, dict)
        assert "MRR" in result

    def test_eval_fn_required(self, data: HeteroData) -> None:
        """Raises ValueError when eval_fn is None."""
        with pytest.raises(ValueError, match="eval_fn is required"):
            multi_seed_evaluation(
                model_factory=TinyModel,
                data=data,
            )


# ======================================================================
# TestMeanAndStd
# ======================================================================


class TestMeanAndStd:
    """Tests that mean and std are returned for all metrics."""

    def test_returns_mean_and_std(self, data: HeteroData) -> None:
        """Result contains mean and std for each metric."""
        result = multi_seed_evaluation(
            model_factory=TinyModel,
            data=data,
            seeds=[42, 123, 456],
            eval_fn=_dummy_eval_fn,
        )

        for metric_name, stats in result.items():
            assert "mean" in stats, f"Missing 'mean' for {metric_name}"
            assert "std" in stats, f"Missing 'std' for {metric_name}"
            assert isinstance(stats["mean"], float)
            assert isinstance(stats["std"], float)

    def test_returns_all_expected_metrics(self, data: HeteroData) -> None:
        """All metrics from eval_fn appear in result."""
        result = multi_seed_evaluation(
            model_factory=TinyModel,
            data=data,
            seeds=[42, 123],
            eval_fn=_dummy_eval_fn,
        )
        assert "MRR" in result
        assert "Hits@1" in result

    def test_std_zero_for_single_seed(self, data: HeteroData) -> None:
        """std is 0 when only one seed is used."""
        result = multi_seed_evaluation(
            model_factory=TinyModel,
            data=data,
            seeds=[42],
            eval_fn=_dummy_eval_fn,
        )
        for metric_name, stats in result.items():
            assert stats["std"] == pytest.approx(0.0), (
                f"std should be 0 for single seed, got {stats['std']} for {metric_name}"
            )

    def test_mean_is_correct(self, data: HeteroData) -> None:
        """Mean value is the average of individual seed results."""
        seeds = [42, 123, 456]
        individual_results = []

        for seed in seeds:
            torch.manual_seed(seed)
            model = TinyModel()
            metrics = _dummy_eval_fn(model, data)
            individual_results.append(metrics)

        result = multi_seed_evaluation(
            model_factory=TinyModel,
            data=data,
            seeds=seeds,
            eval_fn=_dummy_eval_fn,
        )

        for metric_name in result:
            values = [r[metric_name] for r in individual_results]
            expected_mean = sum(values) / len(values)
            assert result[metric_name]["mean"] == pytest.approx(
                expected_mean, abs=1e-4
            ), f"Mean mismatch for {metric_name}"


# ======================================================================
# TestReproducibility
# ======================================================================


class TestReproducibility:
    """Tests that results are reproducible with the same seeds."""

    def test_reproducible_with_same_seeds(self, data: HeteroData) -> None:
        """Two runs with the same seeds produce identical results."""
        seeds = [42, 123, 456]

        result1 = multi_seed_evaluation(
            model_factory=TinyModel,
            data=data,
            seeds=seeds,
            eval_fn=_dummy_eval_fn,
        )

        result2 = multi_seed_evaluation(
            model_factory=TinyModel,
            data=data,
            seeds=seeds,
            eval_fn=_dummy_eval_fn,
        )

        for metric_name in result1:
            assert result1[metric_name]["mean"] == pytest.approx(
                result2[metric_name]["mean"], abs=1e-6
            ), f"Mean not reproducible for {metric_name}"
            assert result1[metric_name]["std"] == pytest.approx(
                result2[metric_name]["std"], abs=1e-6
            ), f"Std not reproducible for {metric_name}"

    def test_reproducible_with_train_fn(self, data: HeteroData) -> None:
        """Reproducibility holds even with training."""
        seeds = [42, 123]

        result1 = multi_seed_evaluation(
            model_factory=TinyModel,
            data=data,
            seeds=seeds,
            train_fn=_dummy_train_fn,
            eval_fn=_dummy_eval_fn,
        )

        result2 = multi_seed_evaluation(
            model_factory=TinyModel,
            data=data,
            seeds=seeds,
            train_fn=_dummy_train_fn,
            eval_fn=_dummy_eval_fn,
        )

        for metric_name in result1:
            assert result1[metric_name]["mean"] == pytest.approx(
                result2[metric_name]["mean"], abs=1e-6
            )

    def test_different_seeds_different_results(self, data: HeteroData) -> None:
        """Different seeds produce different individual model outputs."""
        seeds_a = [42, 123, 456]
        seeds_b = [7, 8, 9]

        result_a = multi_seed_evaluation(
            model_factory=TinyModel,
            data=data,
            seeds=seeds_a,
            eval_fn=_dummy_eval_fn,
        )

        result_b = multi_seed_evaluation(
            model_factory=TinyModel,
            data=data,
            seeds=seeds_b,
            eval_fn=_dummy_eval_fn,
        )

        # At least one metric mean should differ between the two seed sets
        any_different = False
        for metric_name in result_a:
            if metric_name in result_b:
                if abs(result_a[metric_name]["mean"] - result_b[metric_name]["mean"]) > 1e-6:
                    any_different = True
                    break

        assert any_different, "Different seeds should produce at least some different results"

    def test_non_trivial_variance(self, data: HeteroData) -> None:
        """Multiple seeds produce non-zero variance (model init differs)."""
        result = multi_seed_evaluation(
            model_factory=TinyModel,
            data=data,
            seeds=[42, 123, 456, 789, 1024],
            eval_fn=_dummy_eval_fn,
        )

        # At least one metric should have non-trivial std
        any_nonzero_std = False
        for metric_name, stats in result.items():
            if stats["std"] > 1e-8:
                any_nonzero_std = True
                break

        assert any_nonzero_std, "Multiple seeds should produce non-trivial variance"


# ======================================================================
# TestModelFactory
# ======================================================================


class TestModelFactory:
    """Tests that model_factory is called correctly per seed."""

    def test_model_factory_called_per_seed(self, data: HeteroData) -> None:
        """model_factory is called once per seed (fresh model each time)."""
        call_count = 0

        def counting_factory() -> TinyModel:
            nonlocal call_count
            call_count += 1
            return TinyModel()

        seeds = [42, 123, 456]
        multi_seed_evaluation(
            model_factory=counting_factory,
            data=data,
            seeds=seeds,
            eval_fn=_dummy_eval_fn,
        )
        assert call_count == len(seeds), (
            f"model_factory called {call_count} times, expected {len(seeds)}"
        )

    def test_fresh_model_each_seed(self, data: HeteroData) -> None:
        """Each seed gets a freshly initialized model (not shared state)."""
        weight_checksums = []

        def tracking_eval(model: nn.Module, data: HeteroData) -> Dict[str, float]:
            # Capture initial weight checksum
            w = list(model.parameters())[0].data.clone()
            weight_checksums.append(w.sum().item())
            return {"metric": w.mean().item()}

        multi_seed_evaluation(
            model_factory=TinyModel,
            data=data,
            seeds=[42, 123, 456],
            eval_fn=tracking_eval,
        )

        # Different seeds => different random init => different weight checksums
        assert len(set(f"{x:.6f}" for x in weight_checksums)) > 1, (
            "Models should have different initial weights across seeds"
        )
