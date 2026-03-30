"""Glycan-protein interaction prediction task.

Binary classification: predict whether a glycan-protein interaction
exists using concatenated KG embeddings as features.

Evaluation uses 5-fold cross-validation with negative sampling
(1:5 positive-to-negative ratio).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from torch import Tensor
from torch_geometric.data import HeteroData

from glycoMusubi.evaluation.downstream import BaseDownstreamTask

logger = logging.getLogger(__name__)


class _InteractionMLP(nn.Module):
    """Simple MLP classifier for interaction prediction.

    Architecture: Linear(in, hidden) -> ReLU -> Dropout -> Linear(hidden, 1).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Return logits of shape ``[batch, 1]``."""
        return self.net(x)


def _find_glycan_protein_edges(data: HeteroData) -> Optional[Tuple[str, str, str]]:
    """Find the glycan-protein edge type in the HeteroData object.

    Looks for edge types matching common naming patterns:
    ``(glycan, *, protein|enzyme)`` or ``(protein|enzyme, *, glycan)``.

    Returns the edge type tuple, or None if not found.
    """
    for edge_type in data.edge_types:
        src, rel, dst = edge_type
        src_l, dst_l = src.lower(), dst.lower()
        if ("glycan" in src_l and ("protein" in dst_l or "enzyme" in dst_l)) or (
            ("protein" in src_l or "enzyme" in src_l) and "glycan" in dst_l
        ):
            return edge_type
    return None


class GlycanProteinInteractionTask(BaseDownstreamTask):
    """Predict glycan-protein interactions from KG embeddings.

    Uses concatenated glycan + protein embeddings as features for
    binary classification (interaction exists = 1, not = 0).

    Metrics
    -------
    - AUC-ROC (target > 0.85)
    - AUC-PR (target > 0.60)
    - F1@optimal threshold (target > 0.70)

    Parameters
    ----------
    neg_ratio : int
        Negative-to-positive sampling ratio (default 5).
    n_folds : int
        Number of cross-validation folds (default 5).
    classifier_hidden : int
        Hidden dimension of the MLP classifier.
    lr : float
        Learning rate for Adam optimizer.
    epochs : int
        Number of training epochs per fold.
    dropout : float
        Dropout rate in the MLP classifier.
    """

    def __init__(
        self,
        neg_ratio: int = 5,
        n_folds: int = 5,
        classifier_hidden: int = 128,
        lr: float = 1e-3,
        epochs: int = 100,
        dropout: float = 0.3,
    ) -> None:
        self.neg_ratio = neg_ratio
        self.n_folds = n_folds
        self.classifier_hidden = classifier_hidden
        self.lr = lr
        self.epochs = epochs
        self.dropout = dropout

    @property
    def name(self) -> str:
        return "glycan_protein_interaction"

    def prepare_data(
        self,
        embeddings: Dict[str, Tensor],
        data: HeteroData,
    ) -> Tuple[Tensor, Tensor]:
        """Prepare feature matrix and labels.

        Extracts glycan-protein edges from *data*, generates type-
        constrained negative samples, and concatenates embeddings
        for each pair.

        Parameters
        ----------
        embeddings : dict[str, Tensor]
            Node embeddings per type.
        data : HeteroData
            The heterogeneous graph.

        Returns
        -------
        tuple of (Tensor, Tensor)
            Features ``[N, 2*dim]`` and binary labels ``[N]``.
        """
        edge_type = _find_glycan_protein_edges(data)
        if edge_type is None:
            raise ValueError(
                "No glycan-protein edge type found in data. "
                f"Available edge types: {data.edge_types}"
            )

        src_type, _, dst_type = edge_type
        edge_index = data[edge_type].edge_index  # [2, num_edges]

        src_emb = embeddings[src_type]  # [N_src, dim]
        dst_emb = embeddings[dst_type]  # [N_dst, dim]
        num_src = src_emb.shape[0]
        num_dst = dst_emb.shape[0]

        # Positive samples
        pos_src_idx = edge_index[0]
        pos_dst_idx = edge_index[1]
        num_pos = pos_src_idx.shape[0]

        pos_features = torch.cat(
            [src_emb[pos_src_idx], dst_emb[pos_dst_idx]], dim=1
        )
        pos_labels = torch.ones(num_pos, dtype=torch.float32)

        # Negative samples (type-constrained)
        num_neg = num_pos * self.neg_ratio
        existing_edges = set()
        for i in range(num_pos):
            existing_edges.add((pos_src_idx[i].item(), pos_dst_idx[i].item()))

        neg_src_list: List[int] = []
        neg_dst_list: List[int] = []
        rng = np.random.RandomState(42)
        attempts = 0
        max_attempts = num_neg * 10
        while len(neg_src_list) < num_neg and attempts < max_attempts:
            s = rng.randint(0, num_src)
            d = rng.randint(0, num_dst)
            if (s, d) not in existing_edges:
                neg_src_list.append(s)
                neg_dst_list.append(d)
                existing_edges.add((s, d))
            attempts += 1

        neg_src_idx = torch.tensor(neg_src_list, dtype=torch.long)
        neg_dst_idx = torch.tensor(neg_dst_list, dtype=torch.long)

        neg_features = torch.cat(
            [src_emb[neg_src_idx], dst_emb[neg_dst_idx]], dim=1
        )
        neg_labels = torch.zeros(len(neg_src_list), dtype=torch.float32)

        features = torch.cat([pos_features, neg_features], dim=0)
        labels = torch.cat([pos_labels, neg_labels], dim=0)

        return features, labels

    def evaluate(
        self,
        embeddings: Dict[str, Tensor],
        data: HeteroData,
        **kwargs,
    ) -> Dict[str, float]:
        """Run 5-fold CV evaluation.

        Parameters
        ----------
        embeddings : dict[str, Tensor]
            Node embeddings per type.
        data : HeteroData
            The heterogeneous graph.

        Returns
        -------
        dict[str, float]
            ``{"auc_roc": ..., "auc_pr": ..., "f1_optimal": ...}``
            averaged over folds.
        """
        train_data = kwargs.get("train_data")
        test_data = kwargs.get("test_data")
        if train_data is not None and test_data is not None:
            return self._evaluate_with_external_split(
                embeddings=embeddings,
                full_data=data,
                train_data=train_data,
                test_data=test_data,
                split_seed=kwargs.get("split_seed"),
            )

        features, labels = self.prepare_data(embeddings, data)
        features = features.detach()
        labels = labels.detach()

        n_samples = features.shape[0]
        input_dim = features.shape[1]

        # Create fold indices
        indices = np.arange(n_samples)
        rng = np.random.RandomState(42)
        rng.shuffle(indices)
        folds = np.array_split(indices, self.n_folds)

        fold_auc_roc: List[float] = []
        fold_auc_pr: List[float] = []
        fold_f1: List[float] = []

        device = features.device

        for fold_idx in range(self.n_folds):
            test_idx = folds[fold_idx]
            train_idx = np.concatenate(
                [folds[j] for j in range(self.n_folds) if j != fold_idx]
            )

            x_train = features[train_idx]
            y_train = labels[train_idx]
            x_test = features[test_idx]
            y_test = labels[test_idx]

            # Train MLP
            model = _InteractionMLP(
                input_dim=input_dim,
                hidden_dim=self.classifier_hidden,
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

            # AUC-ROC
            if len(np.unique(y_true)) < 2:
                logger.warning(
                    "Fold %d has only one class; skipping metrics.", fold_idx
                )
                continue

            auc_roc = roc_auc_score(y_true, test_probs)
            auc_pr = average_precision_score(y_true, test_probs)

            # F1 at optimal threshold
            precision_arr, recall_arr, thresholds = precision_recall_curve(
                y_true, test_probs
            )
            denom = precision_arr + recall_arr
            denom = np.where(denom > 0, denom, 1.0)  # avoid division warning
            f1_scores = 2 * precision_arr * recall_arr / denom
            f1_scores[precision_arr + recall_arr == 0] = 0.0
            best_f1 = float(np.max(f1_scores))

            fold_auc_roc.append(auc_roc)
            fold_auc_pr.append(auc_pr)
            fold_f1.append(best_f1)

            logger.debug(
                "  Fold %d: AUC-ROC=%.4f, AUC-PR=%.4f, F1=%.4f",
                fold_idx,
                auc_roc,
                auc_pr,
                best_f1,
            )

        if not fold_auc_roc:
            logger.warning("No valid folds for glycan-protein interaction task.")
            return {"auc_roc": 0.0, "auc_pr": 0.0, "f1_optimal": 0.0}

        return {
            "auc_roc": float(np.mean(fold_auc_roc)),
            "auc_pr": float(np.mean(fold_auc_pr)),
            "f1_optimal": float(np.mean(fold_f1)),
        }

    def _evaluate_with_external_split(
        self,
        embeddings: Dict[str, Tensor],
        full_data: HeteroData,
        train_data: HeteroData,
        test_data: HeteroData,
        split_seed: Optional[int] = None,
    ) -> Dict[str, float]:
        edge_type = _find_glycan_protein_edges(full_data)
        if edge_type is None:
            raise ValueError(
                "No glycan-protein edge type found in data. "
                f"Available edge types: {full_data.edge_types}"
            )
        if edge_type not in train_data.edge_types or edge_type not in test_data.edge_types:
            raise ValueError(f"Edge type {edge_type} missing in split data.")

        src_type, _, dst_type = edge_type
        src_emb = embeddings[src_type]
        dst_emb = embeddings[dst_type]
        device = src_emb.device
        num_src = src_emb.shape[0]
        num_dst = dst_emb.shape[0]

        full_pos = full_data[edge_type].edge_index
        full_pos_set = {(int(full_pos[0, i]), int(full_pos[1, i])) for i in range(full_pos.size(1))}

        train_pos = train_data[edge_type].edge_index.to(device)
        test_pos = test_data[edge_type].edge_index.to(device)

        if train_pos.size(1) == 0 or test_pos.size(1) == 0:
            raise ValueError("Train/test split has no positive glycan-protein edges.")

        def _make_neg(num_needed: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
            rng = np.random.RandomState(seed)
            s_list: List[int] = []
            d_list: List[int] = []
            used = set()
            attempts = 0
            max_attempts = max(10 * num_needed, 1000)
            while len(s_list) < num_needed and attempts < max_attempts:
                s = int(rng.randint(0, num_src))
                d = int(rng.randint(0, num_dst))
                pair = (s, d)
                if pair not in full_pos_set and pair not in used:
                    used.add(pair)
                    s_list.append(s)
                    d_list.append(d)
                attempts += 1
            if len(s_list) == 0:
                raise ValueError("Failed to generate negative samples.")
            return (
                torch.tensor(s_list, dtype=torch.long, device=device),
                torch.tensor(d_list, dtype=torch.long, device=device),
            )

        tr_neg_s, tr_neg_d = _make_neg(train_pos.size(1) * self.neg_ratio, seed=42)
        te_neg_s, te_neg_d = _make_neg(test_pos.size(1) * self.neg_ratio, seed=43)

        x_train = torch.cat(
            [
                torch.cat([src_emb[train_pos[0]], dst_emb[train_pos[1]]], dim=1),
                torch.cat([src_emb[tr_neg_s], dst_emb[tr_neg_d]], dim=1),
            ],
            dim=0,
        )
        y_train = torch.cat(
            [
                torch.ones(train_pos.size(1), dtype=torch.float32, device=device),
                torch.zeros(tr_neg_s.size(0), dtype=torch.float32, device=device),
            ],
            dim=0,
        )
        x_test = torch.cat(
            [
                torch.cat([src_emb[test_pos[0]], dst_emb[test_pos[1]]], dim=1),
                torch.cat([src_emb[te_neg_s], dst_emb[te_neg_d]], dim=1),
            ],
            dim=0,
        )
        y_test = torch.cat(
            [
                torch.ones(test_pos.size(1), dtype=torch.float32, device=device),
                torch.zeros(te_neg_s.size(0), dtype=torch.float32, device=device),
            ],
            dim=0,
        )

        model = _InteractionMLP(
            input_dim=x_train.shape[1],
            hidden_dim=self.classifier_hidden,
            dropout=self.dropout,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            logits = model(x_train).squeeze(-1)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(x_test).squeeze(-1)).cpu().numpy()
            y_true = y_test.cpu().numpy()

        if len(np.unique(y_true)) < 2:
            return {"auc_roc": 0.0, "auc_pr": 0.0, "f1_optimal": 0.0}

        auc_roc = float(roc_auc_score(y_true, probs))
        auc_pr = float(average_precision_score(y_true, probs))
        precision_arr, recall_arr, _ = precision_recall_curve(y_true, probs)
        denom = precision_arr + recall_arr
        denom = np.where(denom > 0, denom, 1.0)
        f1_scores = 2 * precision_arr * recall_arr / denom
        f1_scores[precision_arr + recall_arr == 0] = 0.0
        best_f1 = float(np.max(f1_scores))
        return {
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "f1_optimal": best_f1,
            "n_train_pos": float(train_pos.size(1)),
            "n_train_neg": float(tr_neg_s.size(0)),
            "n_test_pos": float(test_pos.size(1)),
            "n_test_neg": float(te_neg_s.size(0)),
            "split_seed": float(split_seed) if split_seed is not None else -1.0,
        }
