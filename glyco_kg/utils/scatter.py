"""Shared scatter-softmax utility for glycoMusubi.

Used by BioHGTLayer, TreeMPNNLayer, and BranchingAwarePooling.
"""

from __future__ import annotations

import torch
from torch_geometric.utils import scatter


def scatter_softmax(
    src: torch.Tensor,
    index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Numerically stable softmax grouped by *index*.

    Parameters
    ----------
    src : Tensor[E, ...]
        Logits.
    index : Tensor[E]
        Group indices (destination node indices).
    num_nodes : int
        Total number of destination nodes (for scatter dim_size).

    Returns
    -------
    Tensor[E, ...]
        Softmax-normalised values within each group.
    """
    # Max per group for numerical stability.
    src_max = scatter(src, index, dim=0, dim_size=num_nodes, reduce="max")
    src = src - src_max[index]

    exp_src = src.exp()
    exp_sum = scatter(exp_src, index, dim=0, dim_size=num_nodes, reduce="sum")
    # Avoid division by zero for nodes that receive no messages.
    exp_sum = exp_sum.clamp(min=1e-12)
    return exp_src / exp_sum[index]
