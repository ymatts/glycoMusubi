"""Multi-seed evaluation for robust performance estimation.

Runs training and evaluation with multiple random seeds, then reports
the mean and standard deviation for each metric.  This eliminates
seed-dependent variance from reported numbers and provides confidence
intervals.

References
----------
- Section 5 (Evaluation Protocol) of the glycoMusubi design documents.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

import torch
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)

_DEFAULT_SEEDS: List[int] = [42, 123, 456, 789, 1024]


def _set_seed(seed: int) -> None:
    """Set random seed for reproducibility across PyTorch and Python."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def multi_seed_evaluation(
    model_factory: Callable[[], torch.nn.Module],
    data: HeteroData,
    seeds: Optional[List[int]] = None,
    train_fn: Optional[Callable[[torch.nn.Module, HeteroData, int], None]] = None,
    eval_fn: Optional[
        Callable[[torch.nn.Module, HeteroData], Dict[str, float]]
    ] = None,
    train_kwargs: Optional[Dict] = None,
    eval_kwargs: Optional[Dict] = None,
) -> Dict[str, Dict[str, float]]:
    """Run training and evaluation with multiple seeds.

    Parameters
    ----------
    model_factory : callable
        Zero-argument callable that returns a freshly initialised model.
        Called once per seed to ensure independent runs.
    data : HeteroData
        The heterogeneous graph data.  The same data object is reused
        across seeds (only model weights and RNG differ).
    seeds : list of int or None
        Random seeds to use (default ``[42, 123, 456, 789, 1024]``).
    train_fn : callable or None
        ``(model, data, seed) -> None`` — trains the model in-place.
        If ``None``, the model is evaluated without training.
    eval_fn : callable or None
        ``(model, data) -> {metric_name: float}`` — evaluates the model
        and returns a metrics dictionary.  Required.
    train_kwargs : dict or None
        Extra keyword arguments forwarded to ``train_fn`` (reserved for
        future use; not currently forwarded).
    eval_kwargs : dict or None
        Extra keyword arguments forwarded to ``eval_fn`` (reserved for
        future use; not currently forwarded).

    Returns
    -------
    Dict[str, Dict[str, float]]
        ``{metric_name: {"mean": float, "std": float}}`` aggregated
        across all seeds.

    Raises
    ------
    ValueError
        If ``eval_fn`` is ``None``.
    """
    if eval_fn is None:
        raise ValueError("eval_fn is required for multi-seed evaluation")

    if seeds is None:
        seeds = list(_DEFAULT_SEEDS)

    all_metrics: Dict[str, List[float]] = {}

    for seed in seeds:
        logger.info("Multi-seed evaluation: seed=%d", seed)
        _set_seed(seed)

        model = model_factory()

        if train_fn is not None:
            train_fn(model, data, seed)

        metrics = eval_fn(model, data)

        for metric_name, value in metrics.items():
            all_metrics.setdefault(metric_name, []).append(value)

    # Aggregate
    result: Dict[str, Dict[str, float]] = {}
    for metric_name, values in all_metrics.items():
        t = torch.tensor(values, dtype=torch.float64)
        result[metric_name] = {
            "mean": t.mean().item(),
            "std": t.std().item() if len(values) > 1 else 0.0,
        }
        logger.info(
            "  %s: %.4f +/- %.4f",
            metric_name,
            result[metric_name]["mean"],
            result[metric_name]["std"],
        )

    return result
