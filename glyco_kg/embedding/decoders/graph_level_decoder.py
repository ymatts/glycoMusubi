"""Graph-level prediction decoder with attentive readout pooling.

Uses a learnable gating mechanism (AttentiveReadout) to aggregate node
embeddings into a graph-level representation, followed by an MLP for
prediction.

References
----------
- Section 3.6.3 of ``docs/architecture/model_architecture_design.md``
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.utils import scatter


class GraphLevelDecoder(nn.Module):
    """AttentiveReadout + MLP for graph-level predictions.

    **AttentiveReadout**::

        gate_i = sigmoid(Linear(h_i))
        h_graph = scatter_add(gate_i * Linear(h_i), batch)

    Then: ``prediction = MLP(h_graph)``

    Parameters
    ----------
    embed_dim : int
        Input node embedding dimension.
    num_classes : int
        Number of output classes / targets.
    hidden_dim : int
        Hidden dimension for the MLP (default 128).
    dropout : float
        Dropout probability (default 0.1).
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Attentive readout components
        self.gate_linear = nn.Linear(embed_dim, 1)
        self.transform_linear = nn.Linear(embed_dim, embed_dim)

        # Prediction MLP
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Aggregate node embeddings and predict graph-level properties.

        Parameters
        ----------
        node_embeddings : torch.Tensor
            ``(num_nodes, embed_dim)`` node feature matrix.
        batch : torch.Tensor, optional
            ``(num_nodes,)`` batch assignment vector mapping each node to
            its graph index.  If ``None``, all nodes belong to a single
            graph.

        Returns
        -------
        torch.Tensor
            ``(batch_size, num_classes)`` prediction logits.
        """
        # Gate: scalar attention weight per node
        gate = torch.sigmoid(self.gate_linear(node_embeddings))  # [N, 1]

        # Transform node embeddings
        h = self.transform_linear(node_embeddings)  # [N, embed_dim]

        # Gated node features
        gated = gate * h  # [N, embed_dim]

        # Pool to graph level
        if batch is None:
            # Single graph: sum all nodes
            h_graph = gated.sum(dim=0, keepdim=True)  # [1, embed_dim]
        else:
            num_graphs = int(batch.max().item()) + 1
            h_graph = scatter(gated, batch, dim=0, dim_size=num_graphs, reduce="sum")

        return self.predictor(h_graph)
