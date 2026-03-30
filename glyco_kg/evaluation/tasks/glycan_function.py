"""Multi-label glycan function classification across taxonomy levels.

Evaluates the quality of glycan embeddings by training per-taxonomy-level
classifiers and measuring accuracy, macro F1, and MCC.  This follows the
glycoMusubi Evaluation Framework 1.2 specification.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from torch_geometric.data import HeteroData

from glycoMusubi.evaluation.downstream import BaseDownstreamTask

logger = logging.getLogger(__name__)


class GlycanFunctionTask(BaseDownstreamTask):
    """Multi-label glycan function classification across taxonomy levels.

    For each of the 8 taxonomy levels (domain through species), a small MLP
    classifier is trained on glycan embeddings in a stratified cross-validation
    setting.  Three metrics are computed per level:

    * **Accuracy** (weighted) -- target > 0.75
    * **Macro F1** -- target > 0.65
    * **MCC** (Matthews Correlation Coefficient) -- target > 0.60

    Aggregate metrics (mean across levels with sufficient data) are also
    reported.

    Parameters
    ----------
    classifier_hidden : int
        Hidden-layer size for the MLP classifier.
    n_folds : int
        Number of cross-validation folds.
    max_iter : int
        Maximum training iterations for the MLP.
    random_state : int
        Seed for reproducibility.
    min_samples_per_level : int
        Minimum number of labelled samples required at a taxonomy level
        for evaluation to proceed on that level.
    """

    TAXONOMY_LEVELS: List[str] = [
        "domain",
        "kingdom",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
    ]

    def __init__(
        self,
        classifier_hidden: int = 128,
        n_folds: int = 5,
        max_iter: int = 300,
        random_state: int = 42,
        min_samples_per_level: int = 10,
    ) -> None:
        self.classifier_hidden = classifier_hidden
        self.n_folds = n_folds
        self.max_iter = max_iter
        self.random_state = random_state
        self.min_samples_per_level = min_samples_per_level

    @property
    def name(self) -> str:
        return "glycan_function_prediction"

    # ------------------------------------------------------------------
    # BaseDownstreamTask interface
    # ------------------------------------------------------------------

    def prepare_data(
        self,
        embeddings: dict[str, torch.Tensor],
        data: HeteroData,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Extract glycan embeddings and per-level taxonomy labels.

        The method looks for glycan node embeddings and taxonomy annotations.
        Taxonomy labels are expected in ``data["glycan"].taxonomy``, a dict
        mapping level names to integer-encoded label arrays, **or** as
        individual attributes (``data["glycan"].taxonomy_domain``, etc.).

        Parameters
        ----------
        embeddings : dict[str, Tensor]
            Node-type -> embedding matrix.  Must contain ``"glycan"``.
        data : HeteroData
            The heterogeneous graph, possibly with taxonomy annotations.

        Returns
        -------
        X : np.ndarray
            Glycan embeddings of shape ``[N, dim]``.
        labels : dict[str, np.ndarray]
            Mapping from taxonomy level to integer labels of shape ``[N]``.
            Only levels with sufficient labelled samples are included.
        """
        if "glycan" not in embeddings:
            raise ValueError(
                "embeddings dict must contain 'glycan' key; "
                f"available keys: {list(embeddings.keys())}"
            )

        X = embeddings["glycan"]
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()

        labels: Dict[str, np.ndarray] = {}

        for level in self.TAXONOMY_LEVELS:
            level_labels = self._extract_level_labels(data, level, X.shape[0])
            if level_labels is None:
                continue
            labels[level] = level_labels

        return X, labels

    def evaluate(
        self,
        embeddings: dict[str, torch.Tensor],
        data: HeteroData,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Run cross-validated classification for each taxonomy level.

        Returns
        -------
        dict[str, float]
            Keys follow the pattern ``{level}_{metric}`` plus aggregated
            ``mean_accuracy``, ``mean_f1``, ``mean_mcc``.
        """
        X, labels = self.prepare_data(embeddings, data)
        results: Dict[str, float] = {}
        level_accuracies: List[float] = []
        level_f1s: List[float] = []
        level_mccs: List[float] = []

        for level, y in labels.items():
            # Filter to samples that have a valid label (>= 0)
            valid_mask = y >= 0
            X_level = X[valid_mask]
            y_level = y[valid_mask]

            if len(y_level) < self.min_samples_per_level:
                logger.info(
                    "Taxonomy level '%s': only %d samples, skipping (need %d).",
                    level,
                    len(y_level),
                    self.min_samples_per_level,
                )
                continue

            n_classes = len(np.unique(y_level))
            if n_classes < 2:
                logger.info(
                    "Taxonomy level '%s': only %d class(es), skipping.",
                    level,
                    n_classes,
                )
                continue

            acc, f1, mcc = self._cross_validate(X_level, y_level)
            results[f"{level}_accuracy"] = acc
            results[f"{level}_f1"] = f1
            results[f"{level}_mcc"] = mcc
            level_accuracies.append(acc)
            level_f1s.append(f1)
            level_mccs.append(mcc)

            logger.info(
                "Level '%s': accuracy=%.4f, f1=%.4f, mcc=%.4f",
                level,
                acc,
                f1,
                mcc,
            )

        # Aggregate across evaluated levels
        if level_accuracies:
            results["mean_accuracy"] = float(np.mean(level_accuracies))
            results["mean_f1"] = float(np.mean(level_f1s))
            results["mean_mcc"] = float(np.mean(level_mccs))
        else:
            results["mean_accuracy"] = 0.0
            results["mean_f1"] = 0.0
            results["mean_mcc"] = 0.0

        results["num_levels_evaluated"] = float(len(level_accuracies))
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_level_labels(
        self,
        data: HeteroData,
        level: str,
        n_glycans: int,
    ) -> Optional[np.ndarray]:
        """Try to extract taxonomy labels for a single level.

        Checks several attribute naming conventions on ``data["glycan"]``:
        1. ``data["glycan"].taxonomy`` dict with level key
        2. ``data["glycan"].taxonomy_{level}`` attribute
        3. ``data["glycan"].{level}`` attribute

        Returns ``None`` if the level is not found.
        """
        glycan_store = data["glycan"] if "glycan" in data.node_types else None
        if glycan_store is None:
            return None

        # Convention 1: taxonomy dict
        taxonomy = getattr(glycan_store, "taxonomy", None)
        if isinstance(taxonomy, dict) and level in taxonomy:
            arr = taxonomy[level]
            if isinstance(arr, torch.Tensor):
                arr = arr.cpu().numpy()
            return np.asarray(arr)

        # Convention 2: taxonomy_{level}
        attr_name = f"taxonomy_{level}"
        val = getattr(glycan_store, attr_name, None)
        if val is not None:
            if isinstance(val, torch.Tensor):
                val = val.cpu().numpy()
            return np.asarray(val)

        # Convention 3: direct attribute
        val = getattr(glycan_store, level, None)
        if val is not None:
            if isinstance(val, torch.Tensor):
                val = val.cpu().numpy()
            return np.asarray(val)

        return None

    def _cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Run stratified k-fold CV and return mean (accuracy, f1, mcc)."""
        n_folds = min(self.n_folds, len(np.unique(y)))
        # Ensure each fold has at least 1 sample per class
        n_folds = min(n_folds, min(np.bincount(y)))
        n_folds = max(n_folds, 2)

        skf = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        fold_accs: List[float] = []
        fold_f1s: List[float] = []
        fold_mccs: List[float] = []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf = MLPClassifier(
                hidden_layer_sizes=(self.classifier_hidden,),
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            fold_accs.append(accuracy_score(y_test, y_pred))
            fold_f1s.append(f1_score(y_test, y_pred, average="macro", zero_division=0))
            fold_mccs.append(matthews_corrcoef(y_test, y_pred))

        return (
            float(np.mean(fold_accs)),
            float(np.mean(fold_f1s)),
            float(np.mean(fold_mccs)),
        )
