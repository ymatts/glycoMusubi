"""Drug target identification task using KG embeddings.

Scores compound-enzyme pairs by training an MLP classifier on concatenated
embeddings.  Uses time-split evaluation when timestamps are available,
otherwise falls back to a random train/test split.

Metrics
-------
- AUC-ROC (target > 0.80)
- Hit@K for novel targets (target > 0.30 @K=10)
- Enrichment Factor @1% (target > 10x)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from glycoMusubi.evaluation.downstream import BaseDownstreamTask

logger = logging.getLogger(__name__)


class DrugTargetTask(BaseDownstreamTask):
    """Identify drug targets using KG embeddings.

    Positive examples are existing ``(compound, inhibits, enzyme)`` edges.
    Negative examples are randomly sampled compound-enzyme pairs that do
    **not** appear in the graph.

    Parameters
    ----------
    k_values : list of int
        K values for Hit@K on novel targets.
    classifier_hidden : int
        Hidden dimension for the two-layer MLP classifier.
    neg_ratio : int
        Number of negative samples per positive.
    test_fraction : float
        Fraction of data reserved for testing when no time-split is used.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        k_values: Optional[List[int]] = None,
        classifier_hidden: int = 128,
        neg_ratio: int = 5,
        test_fraction: float = 0.2,
        seed: int = 42,
    ) -> None:
        self.k_values = k_values or [10, 20, 50]
        self.classifier_hidden = classifier_hidden
        self.neg_ratio = neg_ratio
        self.test_fraction = test_fraction
        self.seed = seed

    @property
    def name(self) -> str:
        return "drug_target_identification"

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_data(
        self,
        embeddings: Dict[str, Tensor],
        data: HeteroData,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, np.ndarray]:
        """Extract compound-enzyme edges and build train/test features.

        Returns
        -------
        tuple of (X_train, y_train, X_test, y_test, test_entity_mask)
            ``test_entity_mask`` is a boolean array indicating which test
            positives involve entities unseen during training (novel targets).
        """
        edge_key = ("compound", "inhibits", "enzyme")

        if "compound" not in embeddings or "enzyme" not in embeddings:
            raise ValueError(
                "Embeddings must contain both 'compound' and 'enzyme' types."
            )

        comp_emb = embeddings["compound"]
        enz_emb = embeddings["enzyme"]

        # Retrieve positive edges
        if edge_key in data.edge_types:
            edge_index = data[edge_key].edge_index  # [2, num_edges]
        else:
            raise ValueError(
                f"Edge type {edge_key} not found in HeteroData. "
                f"Available edge types: {data.edge_types}"
            )

        src = edge_index[0]  # compound indices
        dst = edge_index[1]  # enzyme indices
        num_pos = src.shape[0]

        if num_pos == 0:
            raise ValueError("No positive compound-enzyme edges found.")

        # Build positive set for fast lookup during negative sampling
        pos_set = set(zip(src.tolist(), dst.tolist()))

        # Generate negative samples
        rng = np.random.RandomState(self.seed)
        num_neg = num_pos * self.neg_ratio
        num_compounds = comp_emb.shape[0]
        num_enzymes = enz_emb.shape[0]

        neg_src_list: List[int] = []
        neg_dst_list: List[int] = []
        while len(neg_src_list) < num_neg:
            batch_size = min(num_neg * 2, num_neg - len(neg_src_list) + num_neg)
            c = rng.randint(0, num_compounds, size=batch_size)
            e = rng.randint(0, num_enzymes, size=batch_size)
            for ci, ei in zip(c, e):
                if (ci, ei) not in pos_set:
                    neg_src_list.append(ci)
                    neg_dst_list.append(ei)
                    if len(neg_src_list) >= num_neg:
                        break

        neg_src = torch.tensor(neg_src_list, dtype=torch.long)
        neg_dst = torch.tensor(neg_dst_list, dtype=torch.long)

        # Concatenate embeddings: [compound_emb || enzyme_emb]
        pos_features = torch.cat(
            [comp_emb[src], enz_emb[dst]], dim=1
        )
        neg_features = torch.cat(
            [comp_emb[neg_src], enz_emb[neg_dst]], dim=1
        )

        all_features = torch.cat([pos_features, neg_features], dim=0)
        all_labels = torch.cat([
            torch.ones(num_pos, dtype=torch.float32),
            torch.zeros(num_neg, dtype=torch.float32),
        ])

        # Check for time-split attribute on edges
        has_time = hasattr(data[edge_key], "timestamp") or hasattr(
            data[edge_key], "time"
        )

        n_total = all_features.shape[0]
        indices = np.arange(n_total)

        if has_time:
            # Use time-based split on positives
            ts = getattr(data[edge_key], "timestamp", None)
            if ts is None:
                ts = data[edge_key].time
            ts_np = ts.cpu().numpy()
            cutoff = np.percentile(ts_np, (1.0 - self.test_fraction) * 100)
            pos_train_mask = ts_np <= cutoff
            pos_test_mask = ~pos_train_mask

            # Split negatives proportionally
            n_neg_test = int(self.neg_ratio * pos_test_mask.sum())
            neg_indices = np.arange(num_pos, n_total)
            rng.shuffle(neg_indices)
            neg_test_idx = neg_indices[:n_neg_test]
            neg_train_idx = neg_indices[n_neg_test:]

            train_idx = np.concatenate([
                np.where(pos_train_mask)[0], neg_train_idx
            ])
            test_idx = np.concatenate([
                np.where(pos_test_mask)[0] + 0,  # positives already at front
                neg_test_idx,
            ])
        else:
            # Random split
            rng.shuffle(indices)
            split = int(n_total * (1 - self.test_fraction))
            train_idx = indices[:split]
            test_idx = indices[split:]

        X_train = all_features[train_idx]
        y_train = all_labels[train_idx]
        X_test = all_features[test_idx]
        y_test = all_labels[test_idx]

        # Identify novel targets: enzymes in test positives that were
        # not seen in training positives.
        train_pos_idx = train_idx[train_idx < num_pos]
        test_pos_idx = test_idx[test_idx < num_pos]
        train_enzymes = set(dst[train_pos_idx].tolist()) if len(train_pos_idx) > 0 else set()
        novel_mask = np.zeros(len(test_idx), dtype=bool)
        for i, idx in enumerate(test_pos_idx):
            enz_id = dst[idx].item()
            pos_in_test = np.where(test_idx == idx)[0]
            if len(pos_in_test) > 0 and enz_id not in train_enzymes:
                novel_mask[pos_in_test[0]] = True

        return X_train, y_train, X_test, y_test, novel_mask

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        embeddings: Dict[str, Tensor],
        data: HeteroData,
        **kwargs,
    ) -> Dict[str, float]:
        """Train MLP classifier and compute evaluation metrics.

        Returns
        -------
        dict[str, float]
            Keys: ``auc_roc``, ``hit@{k}_novel``, ``enrichment_factor@1%``.
        """
        from sklearn.metrics import roc_auc_score

        train_data = kwargs.get("train_data")
        test_data = kwargs.get("test_data")
        if train_data is not None and test_data is not None:
            X_train, y_train, X_test, y_test, novel_mask = self._prepare_data_from_external_split(
                embeddings=embeddings,
                full_data=data,
                train_data=train_data,
                test_data=test_data,
                split_seed=kwargs.get("split_seed"),
            )
        else:
            X_train, y_train, X_test, y_test, novel_mask = self.prepare_data(
                embeddings, data
            )

        # Train a simple MLP
        model = self._build_classifier(X_train.shape[1])
        self._train_classifier(model, X_train, y_train)

        # Predict
        model.eval()
        with torch.no_grad():
            logits = model(X_test).squeeze(-1)
            y_scores = torch.sigmoid(logits).cpu().numpy()

        y_true = y_test.cpu().numpy()

        results: Dict[str, float] = {}

        # AUC-ROC
        if len(np.unique(y_true)) > 1:
            results["auc_roc"] = float(roc_auc_score(y_true, y_scores))
        else:
            results["auc_roc"] = 0.0
            logger.warning("Only one class in test set; AUC-ROC set to 0.0.")

        # Hit@K for novel targets
        for k in self.k_values:
            results[f"hit@{k}_novel"] = self._hit_at_k_novel(
                y_true, y_scores, novel_mask, k
            )

        # Enrichment Factor @1%
        results["enrichment_factor@1%"] = self._enrichment_factor(
            y_true, y_scores, fraction=0.01
        )
        for k, v in getattr(self, "_external_split_counts", {}).items():
            results[k] = float(v)

        logger.info(
            "DrugTarget: AUC-ROC=%.4f, EF@1%%=%.2f",
            results["auc_roc"],
            results["enrichment_factor@1%"],
        )

        return results

    def _prepare_data_from_external_split(
        self,
        embeddings: Dict[str, Tensor],
        full_data: HeteroData,
        train_data: HeteroData,
        test_data: HeteroData,
        split_seed: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, np.ndarray]:
        edge_key = ("compound", "inhibits", "enzyme")
        if edge_key not in full_data.edge_types:
            raise ValueError(f"Edge type {edge_key} not found in full data.")
        if edge_key not in train_data.edge_types or edge_key not in test_data.edge_types:
            raise ValueError(f"Edge type {edge_key} missing in split data.")

        comp_emb = embeddings["compound"]
        enz_emb = embeddings["enzyme"]
        device = comp_emb.device

        full_ei = full_data[edge_key].edge_index.to(device)
        train_ei = train_data[edge_key].edge_index.to(device)
        test_ei = test_data[edge_key].edge_index.to(device)

        if train_ei.size(1) == 0 or test_ei.size(1) == 0:
            raise ValueError("Train/test split has no positive compound-enzyme edges.")

        full_pos_set = {(int(full_ei[0, i]), int(full_ei[1, i])) for i in range(full_ei.size(1))}
        num_compounds = comp_emb.shape[0]
        num_enzymes = enz_emb.shape[0]

        def _make_neg(num_needed: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
            rng = np.random.RandomState(seed)
            s_list: List[int] = []
            d_list: List[int] = []
            used = set()
            attempts = 0
            max_attempts = max(10 * num_needed, 1000)
            while len(s_list) < num_needed and attempts < max_attempts:
                c = int(rng.randint(0, num_compounds))
                e = int(rng.randint(0, num_enzymes))
                pair = (c, e)
                if pair not in full_pos_set and pair not in used:
                    used.add(pair)
                    s_list.append(c)
                    d_list.append(e)
                attempts += 1
            if len(s_list) == 0:
                raise ValueError("Failed to generate negative samples.")
            return (
                torch.tensor(s_list, dtype=torch.long, device=device),
                torch.tensor(d_list, dtype=torch.long, device=device),
            )

        tr_neg_src, tr_neg_dst = _make_neg(train_ei.size(1) * self.neg_ratio, self.seed)
        te_neg_src, te_neg_dst = _make_neg(test_ei.size(1) * self.neg_ratio, self.seed + 1)

        X_train = torch.cat(
            [
                torch.cat([comp_emb[train_ei[0]], enz_emb[train_ei[1]]], dim=1),
                torch.cat([comp_emb[tr_neg_src], enz_emb[tr_neg_dst]], dim=1),
            ],
            dim=0,
        )
        y_train = torch.cat(
            [
                torch.ones(train_ei.size(1), dtype=torch.float32, device=device),
                torch.zeros(tr_neg_src.size(0), dtype=torch.float32, device=device),
            ],
            dim=0,
        )
        X_test = torch.cat(
            [
                torch.cat([comp_emb[test_ei[0]], enz_emb[test_ei[1]]], dim=1),
                torch.cat([comp_emb[te_neg_src], enz_emb[te_neg_dst]], dim=1),
            ],
            dim=0,
        )
        y_test = torch.cat(
            [
                torch.ones(test_ei.size(1), dtype=torch.float32, device=device),
                torch.zeros(te_neg_src.size(0), dtype=torch.float32, device=device),
            ],
            dim=0,
        )

        train_enzymes = set(train_ei[1].tolist())
        test_pos_len = int(test_ei.size(1))
        novel_mask = np.zeros(test_pos_len + int(te_neg_src.size(0)), dtype=bool)
        for i in range(test_pos_len):
            enz_id = int(test_ei[1, i].item())
            if enz_id not in train_enzymes:
                novel_mask[i] = True

        self._external_split_counts = {
            "n_train_pos": float(train_ei.size(1)),
            "n_train_neg": float(tr_neg_src.size(0)),
            "n_test_pos": float(test_ei.size(1)),
            "n_test_neg": float(te_neg_src.size(0)),
            "split_seed": float(split_seed) if split_seed is not None else -1.0,
        }
        return X_train, y_train, X_test, y_test, novel_mask

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_classifier(self, input_dim: int) -> torch.nn.Module:
        """Build a two-layer MLP binary classifier."""
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, self.classifier_hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(self.classifier_hidden, 1),
        )

    def _train_classifier(
        self,
        model: torch.nn.Module,
        X: Tensor,
        y: Tensor,
        epochs: int = 100,
        lr: float = 1e-3,
    ) -> None:
        """Train the MLP with BCE loss."""
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            logits = model(X).squeeze(-1)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

    @staticmethod
    def _enrichment_factor(
        y_true: np.ndarray,
        y_scores: np.ndarray,
        fraction: float = 0.01,
    ) -> float:
        """Enrichment Factor at a given fraction.

        EF@x% = (hits in top x%) / (expected hits in x%)
               = (hits_in_top / n_top) / (total_positives / total)

        Returns 0.0 when there are no positives or the top fraction
        contains zero samples.
        """
        n_total = len(y_true)
        n_top = max(1, int(n_total * fraction))
        total_positives = y_true.sum()

        if total_positives == 0 or n_total == 0:
            return 0.0

        # Indices of top-scoring samples
        top_indices = np.argsort(y_scores)[::-1][:n_top]
        hits_in_top = y_true[top_indices].sum()

        observed_rate = hits_in_top / n_top
        expected_rate = total_positives / n_total
        return float(observed_rate / expected_rate) if expected_rate > 0 else 0.0

    @staticmethod
    def _hit_at_k_novel(
        y_true: np.ndarray,
        y_scores: np.ndarray,
        novel_mask: np.ndarray,
        k: int,
    ) -> float:
        """Hit@K restricted to novel targets (entities unseen in training).

        Among the top-K scored samples, compute the fraction of novel
        positives that are recovered.

        Returns 0.0 when there are no novel positives.
        """
        novel_positive_mask = novel_mask & (y_true == 1)
        n_novel_pos = novel_positive_mask.sum()
        if n_novel_pos == 0:
            return 0.0

        top_k_indices = set(np.argsort(y_scores)[::-1][:k].tolist())
        novel_pos_indices = set(np.where(novel_positive_mask)[0].tolist())
        hits = len(top_k_indices & novel_pos_indices)
        return float(hits / n_novel_pos)
