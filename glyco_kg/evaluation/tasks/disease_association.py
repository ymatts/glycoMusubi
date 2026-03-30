"""Disease association prediction for glycan/protein entities.

Evaluates embeddings by ranking candidate entities for each disease using
cosine similarity on learned embeddings.  Implements leave-one-out evaluation
with AUC-ROC, Recall@K, and NDCG@K metrics per the glycoMusubi Evaluation
Framework 1.2.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch_geometric.data import HeteroData

from glycoMusubi.evaluation.downstream import BaseDownstreamTask

logger = logging.getLogger(__name__)


class DiseaseAssociationTask(BaseDownstreamTask):
    """Predict glycan/protein--disease associations via embedding similarity.

    For each disease, all candidate entities (glycans or proteins) are scored
    by cosine similarity between their embedding and the disease embedding.
    Known positive associations serve as ground truth, and ranking metrics
    are computed.

    Metrics:
    * **AUC-ROC** -- target > 0.80
    * **Recall@K** for K in ``k_values`` -- target > 0.50 at K=50
    * **NDCG@K** -- target > 0.40 at K=20

    Parameters
    ----------
    k_values : list[int]
        Cutoff values for Recall@K and NDCG@K.
    relation_type : str
        Name of the disease-association relation in the KG edge types.
    source_node_type : str
        Entity type on the non-disease side of the association edge
        (e.g. ``"glycan"`` or ``"protein"``).
    disease_node_type : str
        Entity type for disease nodes.
    """

    def __init__(
        self,
        k_values: Optional[List[int]] = None,
        relation_type: str = "associated_with_disease",
        source_node_type: str = "glycan",
        disease_node_type: str = "disease",
    ) -> None:
        self.k_values: List[int] = k_values if k_values is not None else [10, 20, 50]
        self.relation_type = relation_type
        self.source_node_type = source_node_type
        self.disease_node_type = disease_node_type

    @property
    def name(self) -> str:
        return "disease_association_prediction"

    # ------------------------------------------------------------------
    # BaseDownstreamTask interface
    # ------------------------------------------------------------------

    def prepare_data(
        self,
        embeddings: dict[str, torch.Tensor],
        data: HeteroData,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, Set[int]]]:
        """Extract entity/disease embeddings and positive associations.

        Parameters
        ----------
        embeddings : dict[str, Tensor]
            Must contain keys for both ``source_node_type`` and
            ``disease_node_type``.
        data : HeteroData
            Must contain edges of the configured relation type.

        Returns
        -------
        entity_emb : np.ndarray
            Shape ``[N_entities, dim]``.
        disease_emb : np.ndarray
            Shape ``[N_diseases, dim]``.
        disease_to_entities : dict[int, set[int]]
            Mapping from disease index to the set of positively associated
            entity indices.
        """
        for key in (self.source_node_type, self.disease_node_type):
            if key not in embeddings:
                raise ValueError(
                    f"embeddings dict must contain '{key}'; "
                    f"available keys: {list(embeddings.keys())}"
                )

        entity_emb = embeddings[self.source_node_type]
        disease_emb = embeddings[self.disease_node_type]
        if isinstance(entity_emb, torch.Tensor):
            entity_emb = entity_emb.detach().cpu().numpy()
        if isinstance(disease_emb, torch.Tensor):
            disease_emb = disease_emb.detach().cpu().numpy()

        disease_to_entities = self._extract_associations(data)

        return entity_emb, disease_emb, disease_to_entities

    def evaluate(
        self,
        embeddings: dict[str, torch.Tensor],
        data: HeteroData,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute disease association ranking metrics.

        Returns
        -------
        dict[str, float]
            Contains ``auc_roc``, ``recall@K``, ``ndcg@K`` for each K,
            and ``num_diseases_evaluated``.
        """
        entity_emb, disease_emb, disease_to_entities = self.prepare_data(
            embeddings, data
        )

        if not disease_to_entities:
            logger.warning("No disease associations found; returning zero metrics.")
            return self._zero_metrics()

        n_entities = entity_emb.shape[0]
        all_auc: List[float] = []
        all_recall: Dict[int, List[float]] = {k: [] for k in self.k_values}
        all_ndcg: Dict[int, List[float]] = {k: [] for k in self.k_values}

        for disease_idx, true_entity_set in disease_to_entities.items():
            if disease_idx >= disease_emb.shape[0]:
                continue
            if not true_entity_set:
                continue

            # Score all candidate entities by cosine similarity
            d_vec = disease_emb[disease_idx]  # [dim]
            scores = self._cosine_scores(entity_emb, d_vec)  # [N_entities]

            # Binary labels for AUC-ROC
            y_true = np.zeros(n_entities, dtype=np.float64)
            true_indices = np.array(sorted(true_entity_set), dtype=np.int64)
            valid_true = true_indices[true_indices < n_entities]
            if len(valid_true) == 0:
                continue
            y_true[valid_true] = 1.0

            # AUC-ROC (needs both positive and negative samples)
            if y_true.sum() > 0 and y_true.sum() < n_entities:
                auc = self._compute_auc_roc(y_true, scores)
                all_auc.append(auc)

            # Ranking: sort by descending score
            ranked_indices = np.argsort(-scores)

            for k in self.k_values:
                recall = self._compute_recall_at_k(ranked_indices, valid_true, k)
                ndcg = self._compute_ndcg(scores, valid_true, k)
                all_recall[k].append(recall)
                all_ndcg[k].append(ndcg)

        results: Dict[str, float] = {}

        results["auc_roc"] = float(np.mean(all_auc)) if all_auc else 0.0
        for k in self.k_values:
            results[f"recall@{k}"] = (
                float(np.mean(all_recall[k])) if all_recall[k] else 0.0
            )
            results[f"ndcg@{k}"] = (
                float(np.mean(all_ndcg[k])) if all_ndcg[k] else 0.0
            )
        results["num_diseases_evaluated"] = float(
            max(len(all_auc), max((len(v) for v in all_recall.values()), default=0))
        )

        logger.info(
            "Disease association evaluation: %d diseases, AUC=%.4f, "
            "Recall@%d=%.4f, NDCG@%d=%.4f",
            int(results["num_diseases_evaluated"]),
            results["auc_roc"],
            self.k_values[-1],
            results.get(f"recall@{self.k_values[-1]}", 0.0),
            self.k_values[1] if len(self.k_values) > 1 else self.k_values[0],
            results.get(
                f"ndcg@{self.k_values[1] if len(self.k_values) > 1 else self.k_values[0]}",
                0.0,
            ),
        )

        return results

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_scores(entity_emb: np.ndarray, disease_vec: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between each entity and a disease vector.

        Parameters
        ----------
        entity_emb : np.ndarray
            Shape ``[N, dim]``.
        disease_vec : np.ndarray
            Shape ``[dim]``.

        Returns
        -------
        np.ndarray
            Shape ``[N]`` of cosine similarities.
        """
        d_norm = np.linalg.norm(disease_vec)
        if d_norm < 1e-12:
            return np.zeros(entity_emb.shape[0], dtype=np.float64)
        e_norms = np.linalg.norm(entity_emb, axis=1)
        e_norms = np.maximum(e_norms, 1e-12)
        return (entity_emb @ disease_vec) / (e_norms * d_norm)

    @staticmethod
    def _compute_auc_roc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Compute AUC-ROC from binary labels and continuous scores."""
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, y_scores))

    @staticmethod
    def _compute_recall_at_k(
        ranked_indices: np.ndarray,
        true_indices: np.ndarray,
        k: int,
    ) -> float:
        """Recall at K: fraction of true positives in the top-K ranked list.

        Parameters
        ----------
        ranked_indices : np.ndarray
            All entity indices sorted by descending score.
        true_indices : np.ndarray
            Indices of the true positive entities.
        k : int
            Cutoff.

        Returns
        -------
        float
            Recall@K in [0, 1].
        """
        if len(true_indices) == 0:
            return 0.0
        top_k = set(ranked_indices[:k].tolist())
        hits = sum(1 for idx in true_indices if idx in top_k)
        return hits / len(true_indices)

    @staticmethod
    def _compute_ndcg(
        scores: np.ndarray,
        true_indices: np.ndarray,
        k: int,
    ) -> float:
        """Normalized Discounted Cumulative Gain at K with binary relevance.

        DCG@K  = sum_{i=1}^{K} rel_i / log2(i + 1)
        NDCG@K = DCG@K / ideal_DCG@K

        Parameters
        ----------
        scores : np.ndarray
            Continuous scores for all entities (shape ``[N]``).
        true_indices : np.ndarray
            Indices of the relevant (positive) entities.
        k : int
            Cutoff.

        Returns
        -------
        float
            NDCG@K in [0, 1].
        """
        if len(true_indices) == 0:
            return 0.0

        true_set = set(true_indices.tolist())
        ranked = np.argsort(-scores)[:k]

        # DCG@K
        dcg = 0.0
        for i, idx in enumerate(ranked):
            if idx in true_set:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because i is 0-indexed

        # Ideal DCG@K: all true items ranked first
        n_relevant = min(len(true_indices), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant))

        if idcg < 1e-12:
            return 0.0
        return dcg / idcg

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _extract_associations(
        self,
        data: HeteroData,
    ) -> Dict[int, Set[int]]:
        """Extract disease-entity associations from the HeteroData edge index.

        Looks for edges of type
        ``(source_node_type, relation_type, disease_node_type)`` or the
        reverse direction
        ``(disease_node_type, relation_type, source_node_type)``.
        """
        disease_to_entities: Dict[int, Set[int]] = {}

        # Try canonical direction: (entity, relation, disease)
        forward_key = (self.source_node_type, self.relation_type, self.disease_node_type)
        # Try reverse direction: (disease, relation, entity)
        reverse_key = (self.disease_node_type, self.relation_type, self.source_node_type)

        edge_index = None
        is_reversed = False

        for edge_type in data.edge_types:
            if edge_type == forward_key:
                edge_index = data[edge_type].edge_index
                break
            if edge_type == reverse_key:
                edge_index = data[edge_type].edge_index
                is_reversed = True
                break

        if edge_index is None:
            logger.warning(
                "No edges found for relation '%s' between '%s' and '%s'. "
                "Available edge types: %s",
                self.relation_type,
                self.source_node_type,
                self.disease_node_type,
                list(data.edge_types),
            )
            return disease_to_entities

        edge_index_cpu = edge_index.cpu()

        if is_reversed:
            # edge_index[0] = disease, edge_index[1] = entity
            disease_indices = edge_index_cpu[0].numpy()
            entity_indices = edge_index_cpu[1].numpy()
        else:
            # edge_index[0] = entity, edge_index[1] = disease
            entity_indices = edge_index_cpu[0].numpy()
            disease_indices = edge_index_cpu[1].numpy()

        for d_idx, e_idx in zip(disease_indices, entity_indices):
            disease_to_entities.setdefault(int(d_idx), set()).add(int(e_idx))

        logger.info(
            "Extracted %d disease-entity associations across %d diseases.",
            sum(len(v) for v in disease_to_entities.values()),
            len(disease_to_entities),
        )

        return disease_to_entities

    def _zero_metrics(self) -> Dict[str, float]:
        """Return a result dict with all metrics set to zero."""
        results: Dict[str, float] = {"auc_roc": 0.0}
        for k in self.k_values:
            results[f"recall@{k}"] = 0.0
            results[f"ndcg@{k}"] = 0.0
        results["num_diseases_evaluated"] = 0.0
        return results
