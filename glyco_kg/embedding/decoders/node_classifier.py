"""Node classification decoder with per-task MLP heads.

Provides type-specific classification heads for node-level prediction tasks
such as glycan taxonomy, protein function, and disease category.

References
----------
- Section 3.6.2 of ``docs/architecture/model_architecture_design.md``
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class NodeClassifier(nn.Module):
    """Type-specific MLP heads for node classification tasks.

    For each task the decoder applies a two-layer MLP::

        logits = Linear(128, num_classes)(
            Dropout(0.1)(
                GELU(
                    Linear(embed_dim, 128)(h_node)
                )
            )
        )

    Parameters
    ----------
    embed_dim : int
        Input embedding dimension (e.g. 256).
    task_configs : dict[str, int]
        ``{task_name: num_classes}`` mapping.  One MLP head is created
        per task.
    hidden_dim : int
        Hidden layer dimension (default 128).
    dropout : float
        Dropout probability (default 0.1).
    """

    def __init__(
        self,
        embed_dim: int,
        task_configs: Dict[str, int],
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.task_configs = dict(task_configs)
        self.heads = nn.ModuleDict()
        for task_name, num_classes in task_configs.items():
            self.heads[task_name] = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

    def forward(self, embeddings: torch.Tensor, task: str) -> torch.Tensor:
        """Classify nodes for a specific task.

        Parameters
        ----------
        embeddings : torch.Tensor
            Node embeddings ``[num_nodes, embed_dim]``.
        task : str
            Task name (must be a key in ``task_configs``).

        Returns
        -------
        torch.Tensor
            Logits of shape ``[num_nodes, num_classes]``.

        Raises
        ------
        KeyError
            If *task* is not a registered classification head.
        """
        if task not in self.heads:
            raise KeyError(
                f"Unknown classification task '{task}'. "
                f"Available tasks: {sorted(self.heads.keys())}"
            )
        return self.heads[task](embeddings)
