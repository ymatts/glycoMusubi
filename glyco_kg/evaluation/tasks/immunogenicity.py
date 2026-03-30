"""Glycan immunogenicity classification task.

Binary classification of whether a glycan is immunogenic, using
KG embeddings as features with bootstrap confidence intervals.

Metrics
-------
- AUC-ROC (target > 0.80)
- Sensitivity (true positive rate)
- Specificity (true negative rate)
- Bootstrap 95% confidence intervals for AUC-ROC
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch import Tensor
from torch_geometric.data import HeteroData

from glycoMusubi.evaluation.downstream import BaseDownstreamTask
from glycoMusubi.evaluation.statistical_tests import bootstrap_ci

logger = logging.getLogger(__name__)


class _ImmunogenicityMLP(nn.Module):
    """Simple MLP for immunogenicity classification.

    Architecture: Linear(in, hidden) -> ReLU -> Dropout -> Linear(hidden, 1).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _find_glycan_type(data: HeteroData) -> Optional[str]:
    """Find the glycan node type in HeteroData."""
    for node_type in data.node_types:
        if "glycan" in node_type.lower():
            return node_type
    return None


class ImmunogenicityTask(BaseDownstreamTask):
    """Binary classification of glycan immunogenicity.

    Uses glycan KG embeddings as features.  If the HeteroData object
    has a ``y`` attribute on the glycan node store, it is used as the
    label.  Otherwise, the task looks for ``immunogenic``,
    ``immunogenicity``, or ``label`` attributes.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension of the MLP classifier.
    lr : float
        Learning rate for Adam optimizer.
    epochs : int
        Number of training epochs.
    test_ratio : float
        Fraction of data held out for testing.
    n_bootstrap : int
        Number of bootstrap resamples for confidence intervals.
    dropout : float
        Dropout rate in the MLP.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        epochs: int = 100,
        test_ratio: float = 0.2,
        n_bootstrap: int = 10000,
        dropout: float = 0.3,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.test_ratio = test_ratio
        self.n_bootstrap = n_bootstrap
        self.dropout = dropout

    @property
    def name(self) -> str:
        return "immunogenicity"

    def prepare_data(
        self,
        embeddings: Dict[str, Tensor],
        data: HeteroData,
    ) -> Tuple[Tensor, Tensor]:
        """Extract glycan embeddings and immunogenicity labels.

        Parameters
        ----------
        embeddings : dict[str, Tensor]
            Node embeddings per type.
        data : HeteroData
            The heterogeneous graph.

        Returns
        -------
        tuple of (Tensor, Tensor)
            Features ``[N, dim]`` and binary labels ``[N]``.
        """
        glycan_type = _find_glycan_type(data)
        if glycan_type is None:
            raise ValueError(
                "No glycan node type found in data. "
                f"Available node types: {data.node_types}"
            )

        glycan_store = data[glycan_type]
        features = embeddings[glycan_type]

        # Try multiple attribute names for labels
        labels = None
        for attr_name in ("y", "immunogenic", "immunogenicity", "label"):
            if hasattr(glycan_store, attr_name):
                labels = getattr(glycan_store, attr_name)
                break

        if labels is None:
            raise ValueError(
                f"No immunogenicity labels found on '{glycan_type}' node store. "
                "Expected one of: y, immunogenic, immunogenicity, label."
            )

        labels = labels.float()
        return features, labels

    def evaluate(
        self,
        embeddings: Dict[str, Tensor],
        data: HeteroData,
        **kwargs,
    ) -> Dict[str, float]:
        """Train classifier and compute metrics with bootstrap CIs.

        Parameters
        ----------
        embeddings : dict[str, Tensor]
            Node embeddings per type.
        data : HeteroData
            The heterogeneous graph.

        Returns
        -------
        dict[str, float]
            Metrics including ``auc_roc``, ``sensitivity``,
            ``specificity``, and bootstrap CI bounds.
        """
        try:
            features, labels = self.prepare_data(embeddings, data)
        except ValueError as e:
            logger.warning("Immunogenicity skipped: %s", e)
            return {}
        features = features.detach()
        labels = labels.detach()

        n_samples = features.shape[0]
        input_dim = features.shape[1]
        device = features.device

        # Train/test split
        indices = np.arange(n_samples)
        rng = np.random.RandomState(42)
        rng.shuffle(indices)
        n_test = max(1, int(n_samples * self.test_ratio))
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        x_train = features[train_idx]
        y_train = labels[train_idx]
        x_test = features[test_idx]
        y_test = labels[test_idx]

        # Train MLP
        model = _ImmunogenicityMLP(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        model.train()
        for _epoch in range(self.epochs):
            optimizer.zero_grad()
            logits = model(x_train).squeeze(-1)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            test_logits = model(x_test).squeeze(-1)
            test_probs = torch.sigmoid(test_logits).cpu().numpy()
            y_true = y_test.cpu().numpy()

        # Predictions at 0.5 threshold
        y_pred = (test_probs >= 0.5).astype(np.int64)

        # AUC-ROC
        n_classes = len(np.unique(y_true))
        if n_classes < 2:
            auc_roc = 0.0
            logger.warning(
                "Only one class in test set; AUC-ROC is undefined."
            )
        else:
            auc_roc = float(roc_auc_score(y_true, test_probs))

        # Sensitivity (recall for positive class) and specificity
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))

        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        metrics: Dict[str, float] = {
            "auc_roc": auc_roc,
            "sensitivity": sensitivity,
            "specificity": specificity,
        }

        # Bootstrap CIs for AUC-ROC (only if we have both classes)
        if n_classes >= 2 and len(y_true) >= 10:
            try:
                combined = np.column_stack([y_true, test_probs])

                def _auc_from_combined(arr: np.ndarray) -> float:
                    yt = arr[:, 0]
                    yp = arr[:, 1]
                    if len(np.unique(yt)) < 2:
                        return 0.5
                    return roc_auc_score(yt, yp)

                ci_low, ci_high = bootstrap_ci(
                    _auc_from_combined,
                    combined,
                    n_bootstrap=self.n_bootstrap,
                )
                metrics["auc_roc_ci_lower"] = ci_low
                metrics["auc_roc_ci_upper"] = ci_high
            except Exception:
                logger.warning("Bootstrap CI computation failed.", exc_info=True)

        return metrics
