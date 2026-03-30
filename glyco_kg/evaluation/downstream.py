"""Base framework for downstream evaluation tasks on KG embeddings.

Provides an abstract base class for downstream tasks and an evaluator
that orchestrates running multiple tasks with optional multi-seed
robustness.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from glycoMusubi.evaluation.multi_seed import _set_seed

logger = logging.getLogger(__name__)

_DEFAULT_SEEDS: List[int] = [42, 123, 456, 789, 1024]


class BaseDownstreamTask(ABC):
    """Abstract base class for downstream evaluation tasks.

    Subclasses must implement :meth:`prepare_data`, :meth:`evaluate`,
    and the :attr:`name` property.
    """

    @abstractmethod
    def prepare_data(
        self,
        embeddings: Dict[str, Tensor],
        data: HeteroData,
    ) -> tuple:
        """Prepare train/test data from KG embeddings.

        Parameters
        ----------
        embeddings : dict[str, Tensor]
            Node embeddings per type, ``{node_type: Tensor[N, dim]}``.
        data : HeteroData
            The heterogeneous graph.

        Returns
        -------
        tuple
            Task-specific prepared data (features, labels, splits, etc.).
        """

    @abstractmethod
    def evaluate(
        self,
        embeddings: Dict[str, Tensor],
        data: HeteroData,
        **kwargs,
    ) -> Dict[str, float]:
        """Run evaluation and return a metrics dictionary.

        Parameters
        ----------
        embeddings : dict[str, Tensor]
            Node embeddings per type.
        data : HeteroData
            The heterogeneous graph.

        Returns
        -------
        dict[str, float]
            Metric name to value mapping.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable task name."""


class DownstreamEvaluator:
    """Run multiple downstream tasks and aggregate results.

    Parameters
    ----------
    tasks : list of BaseDownstreamTask
        Downstream tasks to evaluate.
    """

    def __init__(self, tasks: List[BaseDownstreamTask]) -> None:
        self.tasks = tasks

    def evaluate_all(
        self,
        embeddings: Dict[str, Tensor],
        data: HeteroData,
        task_kwargs: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Run all tasks and return aggregated metrics.

        Parameters
        ----------
        embeddings : dict[str, Tensor]
            Node embeddings per type.
        data : HeteroData
            The heterogeneous graph.

        Returns
        -------
        dict[str, dict[str, float]]
            ``{task_name: {metric: value}}``.
        """
        results: Dict[str, Dict[str, float]] = {}
        task_kwargs = task_kwargs or {}
        for task in self.tasks:
            logger.info("Running downstream task: %s", task.name)
            try:
                kwargs = task_kwargs.get(task.name, {})
                metrics = task.evaluate(embeddings, data, **kwargs)
                results[task.name] = metrics
                logger.info(
                    "  %s: %s",
                    task.name,
                    ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
                )
            except Exception:
                logger.exception("Task '%s' failed.", task.name)
                results[task.name] = {}
        return results

    def evaluate_multi_seed(
        self,
        model_factory: Callable[[], torch.nn.Module],
        data: HeteroData,
        seeds: Optional[List[int]] = None,
        train_fn: Optional[Callable] = None,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Run multi-seed evaluation for all tasks.

        For each seed, a fresh model is created via *model_factory*,
        optionally trained with *train_fn*, and all downstream tasks
        are evaluated on the resulting embeddings.

        Parameters
        ----------
        model_factory : callable
            Zero-argument callable returning a fresh model.
        data : HeteroData
            The heterogeneous graph.
        seeds : list of int or None
            Random seeds (default ``[42, 123, 456, 789, 1024]``).
        train_fn : callable or None
            ``(model, data, seed) -> None`` — trains model in-place.

        Returns
        -------
        dict[str, dict[str, dict[str, float]]]
            ``{task_name: {metric: {"mean": float, "std": float}}}``.
        """
        if seeds is None:
            seeds = list(_DEFAULT_SEEDS)

        # task_name -> metric_name -> list of values across seeds
        all_metrics: Dict[str, Dict[str, List[float]]] = {
            task.name: {} for task in self.tasks
        }

        for seed in seeds:
            logger.info("Multi-seed downstream evaluation: seed=%d", seed)
            _set_seed(seed)

            model = model_factory()
            if train_fn is not None:
                train_fn(model, data, seed)

            model.eval()
            with torch.no_grad():
                embeddings = model.get_embeddings(data)

            for task in self.tasks:
                try:
                    metrics = task.evaluate(embeddings, data)
                except Exception:
                    logger.exception(
                        "Task '%s' failed for seed=%d", task.name, seed
                    )
                    continue
                for metric_name, value in metrics.items():
                    all_metrics[task.name].setdefault(metric_name, []).append(
                        value
                    )

        # Aggregate mean/std
        result: Dict[str, Dict[str, Dict[str, float]]] = {}
        for task_name, metric_dict in all_metrics.items():
            result[task_name] = {}
            for metric_name, values in metric_dict.items():
                t = torch.tensor(values, dtype=torch.float64)
                result[task_name][metric_name] = {
                    "mean": t.mean().item(),
                    "std": t.std().item() if len(values) > 1 else 0.0,
                }
            if result[task_name]:
                logger.info(
                    "  %s: %s",
                    task_name,
                    ", ".join(
                        f"{k}={v['mean']:.4f}+/-{v['std']:.4f}"
                        for k, v in result[task_name].items()
                    ),
                )

        return result
