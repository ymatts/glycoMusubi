"""Reproducibility utilities: seed fixing and deterministic mode."""

from __future__ import annotations

import os
import random
from typing import Optional


def set_seed(seed: int) -> None:
    """Fix random seeds for Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed:
        Integer seed value.
    """
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def set_deterministic(enabled: bool = True) -> None:
    """Enable or disable deterministic mode for PyTorch.

    When enabled, sets ``torch.backends.cudnn.deterministic = True`` and
    ``torch.backends.cudnn.benchmark = False`` for reproducible results
    (at a potential performance cost).

    Parameters
    ----------
    enabled:
        ``True`` to enable deterministic mode.
    """
    try:
        import torch
        torch.backends.cudnn.deterministic = enabled
        torch.backends.cudnn.benchmark = not enabled
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(enabled)
    except ImportError:
        pass

    if enabled:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def seed_worker(worker_id: int) -> None:  # noqa: ARG001
    """Seed function for ``torch.utils.data.DataLoader`` workers.

    Pass as ``worker_init_fn`` so each DataLoader worker gets a
    deterministic, unique seed derived from the global state.

    Parameters
    ----------
    worker_id:
        Worker index (provided by DataLoader).
    """
    try:
        import torch
        import numpy as np
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    except ImportError:
        pass
