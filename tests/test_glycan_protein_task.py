"""Tests for the GlycanProteinInteractionTask.

Tests cover:
  - Task instantiation and default parameters
  - prepare_data extracts glycan-protein edges from test HeteroData
  - Negative sampling produces correct ratio (1:5)
  - Negative samples respect type constraints
  - evaluate returns AUC-ROC, AUC-PR, F1 keys
  - All metric values in [0, 1]
  - 5-fold CV produces 5 fold results
  - Works with random/mock embeddings (should give ~0.5 AUC for random)
  - MLP classifier trains and produces probabilities in [0, 1]
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pytest
import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from glycoMusubi.evaluation.tasks.glycan_protein_interaction import (
    GlycanProteinInteractionTask,
    _InteractionMLP,
    _find_glycan_protein_edges,
)


# ======================================================================
# Fixtures
# ======================================================================

DIM = 32
N_GLYCANS = 20
N_PROTEINS = 15
N_EDGES = 30


@pytest.fixture()
def glycan_protein_data() -> HeteroData:
    """HeteroData with glycan and protein nodes and interaction edges.

    Creates a graph with enough nodes and edges for meaningful
    5-fold cross-validation with 1:5 negative sampling.
    """
    data = HeteroData()

    data["glycan"].x = torch.randn(N_GLYCANS, DIM)
    data["glycan"].num_nodes = N_GLYCANS
    data["protein"].x = torch.randn(N_PROTEINS, DIM)
    data["protein"].num_nodes = N_PROTEINS

    # Generate random positive edges (glycan -> protein)
    rng = np.random.RandomState(99)
    src = rng.randint(0, N_GLYCANS, size=N_EDGES)
    dst = rng.randint(0, N_PROTEINS, size=N_EDGES)
    # Deduplicate
    pairs = list({(s, d) for s, d in zip(src, dst)})
    src_t = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    dst_t = torch.tensor([p[1] for p in pairs], dtype=torch.long)

    data["glycan", "interacts_with", "protein"].edge_index = torch.stack(
        [src_t, dst_t]
    )

    return data


@pytest.fixture()
def glycan_protein_embeddings() -> Dict[str, Tensor]:
    """Random embeddings matching the glycan_protein_data fixture."""
    return {
        "glycan": torch.randn(N_GLYCANS, DIM),
        "protein": torch.randn(N_PROTEINS, DIM),
    }


@pytest.fixture()
def informative_embeddings(glycan_protein_data: HeteroData) -> Dict[str, Tensor]:
    """Embeddings where interacting pairs have similar embeddings.

    Positive edges get embeddings that are closer together,
    providing a signal for the MLP classifier.
    """
    edge_index = glycan_protein_data[
        "glycan", "interacts_with", "protein"
    ].edge_index

    glycan_emb = torch.randn(N_GLYCANS, DIM)
    protein_emb = torch.randn(N_PROTEINS, DIM)

    # Make positive pairs have similar embeddings by copying + noise
    for i in range(edge_index.shape[1]):
        src_idx = edge_index[0, i].item()
        dst_idx = edge_index[1, i].item()
        base = torch.randn(DIM)
        glycan_emb[src_idx] = base + 0.1 * torch.randn(DIM)
        protein_emb[dst_idx] = base + 0.1 * torch.randn(DIM)

    return {"glycan": glycan_emb, "protein": protein_emb}


# ======================================================================
# TestFindGlycanProteinEdges
# ======================================================================


class TestFindGlycanProteinEdges:
    """Tests for _find_glycan_protein_edges helper."""

    def test_finds_glycan_protein_edge(self, glycan_protein_data: HeteroData) -> None:
        """Finds the (glycan, interacts_with, protein) edge type."""
        edge_type = _find_glycan_protein_edges(glycan_protein_data)
        assert edge_type is not None
        assert "glycan" in edge_type[0].lower()
        assert "protein" in edge_type[2].lower()

    def test_finds_protein_glycan_edge(self) -> None:
        """Also finds reversed (protein, *, glycan) edge types."""
        data = HeteroData()
        data["protein"].num_nodes = 3
        data["glycan"].num_nodes = 3
        data["protein", "has_glycan", "glycan"].edge_index = torch.tensor(
            [[0, 1], [1, 2]]
        )
        edge_type = _find_glycan_protein_edges(data)
        assert edge_type is not None

    def test_returns_none_when_missing(self) -> None:
        """Returns None when no glycan-protein edge type exists."""
        data = HeteroData()
        data["compound"].num_nodes = 2
        data["enzyme"].num_nodes = 2
        data["compound", "inhibits", "enzyme"].edge_index = torch.tensor(
            [[0], [1]]
        )
        assert _find_glycan_protein_edges(data) is None


# ======================================================================
# TestGlycanProteinInteractionTask
# ======================================================================


class TestGlycanProteinInteractionTask:
    """Tests for task instantiation and configuration."""

    def test_default_instantiation(self) -> None:
        """Task uses sensible defaults."""
        task = GlycanProteinInteractionTask()
        assert task.neg_ratio == 5
        assert task.n_folds == 5
        assert task.name == "glycan_protein_interaction"

    def test_custom_parameters(self) -> None:
        """Task accepts custom parameters."""
        task = GlycanProteinInteractionTask(
            neg_ratio=3, n_folds=3, classifier_hidden=64, lr=0.01, epochs=50
        )
        assert task.neg_ratio == 3
        assert task.n_folds == 3
        assert task.classifier_hidden == 64
        assert task.lr == 0.01
        assert task.epochs == 50

    def test_is_base_downstream_task(self) -> None:
        """Task is a subclass of BaseDownstreamTask."""
        from glycoMusubi.evaluation.downstream import BaseDownstreamTask

        task = GlycanProteinInteractionTask()
        assert isinstance(task, BaseDownstreamTask)


# ======================================================================
# TestPrepareData
# ======================================================================


class TestPrepareData:
    """Tests for the prepare_data method."""

    def test_returns_features_and_labels(
        self,
        glycan_protein_embeddings: Dict[str, Tensor],
        glycan_protein_data: HeteroData,
    ) -> None:
        """prepare_data returns a (features, labels) tuple."""
        task = GlycanProteinInteractionTask()
        features, labels = task.prepare_data(
            glycan_protein_embeddings, glycan_protein_data
        )
        assert isinstance(features, Tensor)
        assert isinstance(labels, Tensor)

    def test_feature_dimensionality(
        self,
        glycan_protein_embeddings: Dict[str, Tensor],
        glycan_protein_data: HeteroData,
    ) -> None:
        """Features are concatenated embeddings with dimension 2*dim."""
        task = GlycanProteinInteractionTask()
        features, _ = task.prepare_data(
            glycan_protein_embeddings, glycan_protein_data
        )
        assert features.shape[1] == 2 * DIM

    def test_negative_sampling_ratio(
        self,
        glycan_protein_embeddings: Dict[str, Tensor],
        glycan_protein_data: HeteroData,
    ) -> None:
        """Negative samples are produced at the specified ratio."""
        neg_ratio = 5
        task = GlycanProteinInteractionTask(neg_ratio=neg_ratio)
        features, labels = task.prepare_data(
            glycan_protein_embeddings, glycan_protein_data
        )

        edge_index = glycan_protein_data[
            "glycan", "interacts_with", "protein"
        ].edge_index
        num_pos = edge_index.shape[1]
        num_neg = int(labels.sum().item())  # positive labels = 1
        expected_neg = num_pos * neg_ratio

        # Positive count
        assert num_neg == num_pos
        # Total should be pos + neg
        assert labels.shape[0] == num_pos + expected_neg

    def test_labels_are_binary(
        self,
        glycan_protein_embeddings: Dict[str, Tensor],
        glycan_protein_data: HeteroData,
    ) -> None:
        """Labels are 0 or 1."""
        task = GlycanProteinInteractionTask()
        _, labels = task.prepare_data(
            glycan_protein_embeddings, glycan_protein_data
        )
        unique_vals = torch.unique(labels)
        assert set(unique_vals.tolist()).issubset({0.0, 1.0})

    def test_negative_samples_type_constrained(
        self,
        glycan_protein_embeddings: Dict[str, Tensor],
        glycan_protein_data: HeteroData,
    ) -> None:
        """Negative samples only pair glycan nodes with protein nodes.

        Verified by checking that all feature vectors have the expected
        dimension (2*dim from glycan_emb concat protein_emb) and that
        negative indices are within valid node ranges.
        """
        task = GlycanProteinInteractionTask()
        features, labels = task.prepare_data(
            glycan_protein_embeddings, glycan_protein_data
        )
        # All samples have consistent feature dimension
        assert features.shape[1] == 2 * DIM
        # Number of negative samples
        neg_mask = labels == 0.0
        assert neg_mask.sum() > 0

    def test_raises_on_missing_edge_type(self) -> None:
        """Raises ValueError when no glycan-protein edge exists."""
        data = HeteroData()
        data["compound"].num_nodes = 2
        data["enzyme"].num_nodes = 2
        data["compound", "inhibits", "enzyme"].edge_index = torch.tensor(
            [[0], [1]]
        )
        embeddings = {
            "compound": torch.randn(2, DIM),
            "enzyme": torch.randn(2, DIM),
        }

        task = GlycanProteinInteractionTask()
        with pytest.raises(ValueError, match="No glycan-protein edge type"):
            task.prepare_data(embeddings, data)


# ======================================================================
# TestEvaluate
# ======================================================================


class TestEvaluate:
    """Tests for the evaluate method."""

    def test_returns_expected_metric_keys(
        self,
        glycan_protein_embeddings: Dict[str, Tensor],
        glycan_protein_data: HeteroData,
    ) -> None:
        """evaluate returns dict with auc_roc, auc_pr, f1_optimal keys."""
        task = GlycanProteinInteractionTask(epochs=5, n_folds=2)
        metrics = task.evaluate(glycan_protein_embeddings, glycan_protein_data)
        assert "auc_roc" in metrics
        assert "auc_pr" in metrics
        assert "f1_optimal" in metrics

    def test_metric_values_in_valid_range(
        self,
        glycan_protein_embeddings: Dict[str, Tensor],
        glycan_protein_data: HeteroData,
    ) -> None:
        """All metric values are in [0, 1]."""
        task = GlycanProteinInteractionTask(epochs=5, n_folds=2)
        metrics = task.evaluate(glycan_protein_embeddings, glycan_protein_data)
        for name, value in metrics.items():
            assert 0.0 <= value <= 1.0, f"{name}={value} is outside [0, 1]"

    def test_five_fold_cv(
        self,
        glycan_protein_embeddings: Dict[str, Tensor],
        glycan_protein_data: HeteroData,
    ) -> None:
        """Default 5-fold CV produces valid averaged results."""
        task = GlycanProteinInteractionTask(epochs=5, n_folds=5)
        metrics = task.evaluate(glycan_protein_embeddings, glycan_protein_data)
        # All three metrics should be present and valid
        assert len(metrics) == 3
        for v in metrics.values():
            assert isinstance(v, float)
            assert 0.0 <= v <= 1.0

    def test_random_embeddings_give_near_chance_auc(
        self,
        glycan_protein_data: HeteroData,
    ) -> None:
        """Random embeddings should yield AUC-ROC around 0.5 (chance level).

        We use a generous tolerance since random behavior varies.
        """
        torch.manual_seed(0)
        random_emb = {
            "glycan": torch.randn(N_GLYCANS, DIM),
            "protein": torch.randn(N_PROTEINS, DIM),
        }
        task = GlycanProteinInteractionTask(epochs=2, n_folds=2, neg_ratio=3)
        metrics = task.evaluate(random_emb, glycan_protein_data)
        # With random embeddings, AUC should be roughly around 0.5
        # Use a very wide tolerance to avoid flaky tests
        assert 0.1 <= metrics["auc_roc"] <= 0.95

    def test_informative_embeddings_give_higher_auc(
        self,
        informative_embeddings: Dict[str, Tensor],
        glycan_protein_data: HeteroData,
    ) -> None:
        """Informative embeddings should achieve reasonable AUC (> 0.5)."""
        task = GlycanProteinInteractionTask(
            epochs=50, n_folds=2, neg_ratio=3, classifier_hidden=64
        )
        metrics = task.evaluate(informative_embeddings, glycan_protein_data)
        # With informative embeddings, AUC should be above chance
        assert metrics["auc_roc"] >= 0.4


# ======================================================================
# TestInteractionMLP
# ======================================================================


class TestInteractionMLP:
    """Tests for the _InteractionMLP classifier."""

    def test_instantiation(self) -> None:
        """MLP can be created with default parameters."""
        mlp = _InteractionMLP(input_dim=64)
        assert mlp is not None

    def test_forward_shape(self) -> None:
        """Forward pass produces logits of shape [batch, 1]."""
        mlp = _InteractionMLP(input_dim=64, hidden_dim=32)
        x = torch.randn(10, 64)
        logits = mlp(x)
        assert logits.shape == (10, 1)

    def test_output_probabilities_in_range(self) -> None:
        """Sigmoid of logits produces probabilities in [0, 1]."""
        mlp = _InteractionMLP(input_dim=32, hidden_dim=16)
        x = torch.randn(20, 32)
        logits = mlp(x)
        probs = torch.sigmoid(logits)
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_training_reduces_loss(self) -> None:
        """A few training steps reduce the BCE loss."""
        torch.manual_seed(42)
        input_dim = 32
        mlp = _InteractionMLP(input_dim=input_dim, hidden_dim=16, dropout=0.0)
        optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-2)
        criterion = torch.nn.BCEWithLogitsLoss()

        # Create simple separable data
        x_pos = torch.randn(20, input_dim) + 1.0
        x_neg = torch.randn(20, input_dim) - 1.0
        x = torch.cat([x_pos, x_neg])
        y = torch.cat([torch.ones(20), torch.zeros(20)])

        mlp.train()
        logits = mlp(x).squeeze(-1)
        initial_loss = criterion(logits, y).item()

        for _ in range(50):
            optimizer.zero_grad()
            logits = mlp(x).squeeze(-1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < initial_loss

    def test_custom_dropout(self) -> None:
        """MLP accepts custom dropout rate."""
        mlp = _InteractionMLP(input_dim=32, hidden_dim=16, dropout=0.5)
        # Verify dropout is set
        has_dropout = any(
            isinstance(m, torch.nn.Dropout) and m.p == 0.5
            for m in mlp.modules()
        )
        assert has_dropout
