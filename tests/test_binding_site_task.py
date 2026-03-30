"""Unit tests for BindingSiteTask downstream evaluation.

Tests cover:
  - Instantiation and default parameters
  - prepare_data with mock site/protein embeddings
  - Residue-level AUC computation
  - Site-level F1 computation
  - N-linked vs O-linked distinction
  - evaluate() pipeline with synthetic data
  - Edge cases (missing embeddings, no site type info)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from glycoMusubi.evaluation.tasks.binding_site import BindingSiteTask


# ======================================================================
# Constants
# ======================================================================

EMB_DIM = 32
NUM_SITES = 40
NUM_PROTEINS = 20


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def mock_hetero_data() -> HeteroData:
    """Create minimal HeteroData with site and protein nodes."""
    data = HeteroData()
    data["site"].num_nodes = NUM_SITES
    data["protein"].num_nodes = NUM_PROTEINS

    # has_site edges: protein -> site
    rng = np.random.RandomState(0)
    src = torch.from_numpy(rng.randint(0, NUM_PROTEINS, size=NUM_SITES))
    dst = torch.arange(NUM_SITES)
    data["protein", "has_site", "site"].edge_index = torch.stack([src, dst])
    return data


@pytest.fixture()
def mock_hetero_data_with_types() -> HeteroData:
    """HeteroData with glyco_type metadata on site nodes."""
    data = HeteroData()
    data["site"].num_nodes = NUM_SITES
    data["protein"].num_nodes = NUM_PROTEINS

    rng = np.random.RandomState(0)
    src = torch.from_numpy(rng.randint(0, NUM_PROTEINS, size=NUM_SITES))
    dst = torch.arange(NUM_SITES)
    data["protein", "has_site", "site"].edge_index = torch.stack([src, dst])

    # Assign glyco_type: half N-linked, half O-linked
    types = []
    for i in range(NUM_SITES):
        if i % 2 == 0:
            types.append("N-linked")
        else:
            types.append("O-linked")
    data["site"].glyco_type = types
    return data


@pytest.fixture()
def mock_embeddings() -> dict[str, torch.Tensor]:
    """Synthetic embeddings for site and protein nodes."""
    torch.manual_seed(42)
    return {
        "site": torch.randn(NUM_SITES, EMB_DIM),
        "protein": torch.randn(NUM_PROTEINS, EMB_DIM),
    }


@pytest.fixture()
def mock_embeddings_site_only() -> dict[str, torch.Tensor]:
    """Embeddings with only site type (no protein)."""
    torch.manual_seed(42)
    return {
        "site": torch.randn(NUM_SITES, EMB_DIM),
    }


# ======================================================================
# TestBindingSiteInstantiation
# ======================================================================


class TestBindingSiteInstantiation:
    """Tests for BindingSiteTask construction."""

    def test_default_params(self) -> None:
        task = BindingSiteTask()
        assert task.name == "binding_site_prediction"
        assert task.window_size == 15
        assert task.classifier_hidden == 128
        assert task.neg_ratio == 3
        assert task.test_fraction == 0.2

    def test_custom_params(self) -> None:
        task = BindingSiteTask(
            window_size=21,
            classifier_hidden=64,
            neg_ratio=5,
            test_fraction=0.3,
        )
        assert task.window_size == 21
        assert task.classifier_hidden == 64
        assert task.neg_ratio == 5
        assert task.test_fraction == 0.3


# ======================================================================
# TestPrepareData
# ======================================================================


class TestPrepareData:
    """Tests for prepare_data method."""

    def test_basic_prepare(
        self,
        mock_embeddings: dict[str, torch.Tensor],
        mock_hetero_data: HeteroData,
    ) -> None:
        """prepare_data returns correctly shaped tensors."""
        task = BindingSiteTask(neg_ratio=2, seed=0)
        X_train, y_train, X_test, y_test, site_types = task.prepare_data(
            mock_embeddings, mock_hetero_data
        )

        assert X_train.shape[1] == EMB_DIM
        assert X_test.shape[1] == EMB_DIM

        # Labels are binary
        assert set(y_train.unique().tolist()).issubset({0.0, 1.0})
        assert set(y_test.unique().tolist()).issubset({0.0, 1.0})

        # Total samples = sites + negatives
        total = X_train.shape[0] + X_test.shape[0]
        expected = NUM_SITES + NUM_SITES * 2  # neg_ratio=2
        assert total == expected

    def test_site_types_none_without_metadata(
        self,
        mock_embeddings: dict[str, torch.Tensor],
        mock_hetero_data: HeteroData,
    ) -> None:
        """site_types is None when no glyco_type metadata exists."""
        task = BindingSiteTask(seed=0)
        _, _, _, _, site_types = task.prepare_data(
            mock_embeddings, mock_hetero_data
        )
        assert site_types is None

    def test_site_types_present_with_metadata(
        self,
        mock_embeddings: dict[str, torch.Tensor],
        mock_hetero_data_with_types: HeteroData,
    ) -> None:
        """site_types array is returned when glyco_type metadata exists."""
        task = BindingSiteTask(seed=0)
        _, _, _, _, site_types = task.prepare_data(
            mock_embeddings, mock_hetero_data_with_types
        )
        assert site_types is not None
        # Should contain N-linked, O-linked, or unknown values
        unique_types = set(site_types)
        assert unique_types.issubset({"N-linked", "O-linked", "unknown"})

    def test_missing_site_embeddings(
        self, mock_hetero_data: HeteroData
    ) -> None:
        """Raises ValueError when site embeddings are missing."""
        task = BindingSiteTask()
        embeddings = {"protein": torch.randn(NUM_PROTEINS, EMB_DIM)}
        with pytest.raises(ValueError, match="site"):
            task.prepare_data(embeddings, mock_hetero_data)

    def test_fallback_negatives_without_protein(
        self,
        mock_embeddings_site_only: dict[str, torch.Tensor],
        mock_hetero_data: HeteroData,
    ) -> None:
        """Falls back to random noise negatives when protein embs are absent."""
        task = BindingSiteTask(neg_ratio=2, seed=0)
        X_train, y_train, X_test, y_test, _ = task.prepare_data(
            mock_embeddings_site_only, mock_hetero_data
        )
        total = X_train.shape[0] + X_test.shape[0]
        expected = NUM_SITES + NUM_SITES * 2
        assert total == expected


# ======================================================================
# TestEvaluatePipeline
# ======================================================================


class TestEvaluatePipeline:
    """End-to-end tests for evaluate()."""

    def test_evaluate_returns_expected_keys(
        self,
        mock_embeddings: dict[str, torch.Tensor],
        mock_hetero_data: HeteroData,
    ) -> None:
        """evaluate() returns residue_auc and site_f1."""
        task = BindingSiteTask(neg_ratio=2, seed=0)
        results = task.evaluate(mock_embeddings, mock_hetero_data)

        assert "residue_auc" in results
        assert "site_f1" in results

    def test_residue_auc_in_range(
        self,
        mock_embeddings: dict[str, torch.Tensor],
        mock_hetero_data: HeteroData,
    ) -> None:
        """Residue-level AUC must be between 0 and 1."""
        task = BindingSiteTask(neg_ratio=2, seed=0)
        results = task.evaluate(mock_embeddings, mock_hetero_data)
        assert 0.0 <= results["residue_auc"] <= 1.0

    def test_site_f1_in_range(
        self,
        mock_embeddings: dict[str, torch.Tensor],
        mock_hetero_data: HeteroData,
    ) -> None:
        """Site-level F1 must be between 0 and 1."""
        task = BindingSiteTask(neg_ratio=2, seed=0)
        results = task.evaluate(mock_embeddings, mock_hetero_data)
        assert 0.0 <= results["site_f1"] <= 1.0

    def test_evaluate_with_type_info(
        self,
        mock_embeddings: dict[str, torch.Tensor],
        mock_hetero_data_with_types: HeteroData,
    ) -> None:
        """evaluate() reports per-type AUC when glyco_type metadata exists."""
        task = BindingSiteTask(neg_ratio=2, seed=0)
        results = task.evaluate(mock_embeddings, mock_hetero_data_with_types)

        assert "residue_auc" in results
        assert "site_f1" in results
        # Per-type AUC may or may not be present depending on test split
        # but evaluate should not error

    def test_evaluate_without_protein_embeddings(
        self,
        mock_embeddings_site_only: dict[str, torch.Tensor],
        mock_hetero_data: HeteroData,
    ) -> None:
        """evaluate() works with only site embeddings (no protein)."""
        task = BindingSiteTask(neg_ratio=2, seed=0)
        results = task.evaluate(mock_embeddings_site_only, mock_hetero_data)
        assert "residue_auc" in results
        assert "site_f1" in results

    def test_reproducibility(
        self,
        mock_embeddings: dict[str, torch.Tensor],
        mock_hetero_data: HeteroData,
    ) -> None:
        """Same seed produces same results when torch seed is also set."""
        torch.manual_seed(0)
        task = BindingSiteTask(neg_ratio=2, seed=99)
        r1 = task.evaluate(mock_embeddings, mock_hetero_data)

        torch.manual_seed(0)
        task2 = BindingSiteTask(neg_ratio=2, seed=99)
        r2 = task2.evaluate(mock_embeddings, mock_hetero_data)

        assert r1["residue_auc"] == pytest.approx(r2["residue_auc"], abs=1e-4)
        assert r1["site_f1"] == pytest.approx(r2["site_f1"], abs=1e-4)


# ======================================================================
# TestExtractSiteTypes
# ======================================================================


class TestExtractSiteTypes:
    """Tests for _extract_site_types static method."""

    def test_returns_none_without_metadata(self) -> None:
        """Returns None when data has no glyco_type or site_type."""
        data = HeteroData()
        data["site"].num_nodes = 5
        result = BindingSiteTask._extract_site_types(
            data, np.array([0, 1, 2]), num_pos=5
        )
        assert result is None

    def test_extracts_n_linked(self) -> None:
        """Correctly identifies N-linked sites."""
        data = HeteroData()
        data["site"].num_nodes = 3
        data["site"].glyco_type = ["N-linked", "O-linked", "N-linked"]
        result = BindingSiteTask._extract_site_types(
            data, np.array([0, 1, 2]), num_pos=3
        )
        assert result is not None
        assert result[0] == "N-linked"
        assert result[1] == "O-linked"
        assert result[2] == "N-linked"

    def test_negatives_are_unknown(self) -> None:
        """Negative samples (idx >= num_pos) are marked unknown."""
        data = HeteroData()
        data["site"].num_nodes = 2
        data["site"].glyco_type = ["N-linked", "O-linked"]
        result = BindingSiteTask._extract_site_types(
            data, np.array([0, 1, 5, 6]), num_pos=2
        )
        assert result is not None
        assert result[0] == "N-linked"
        assert result[1] == "O-linked"
        assert result[2] == "unknown"
        assert result[3] == "unknown"

    def test_site_type_attr_fallback(self) -> None:
        """Falls back to site_type when glyco_type is absent."""
        data = HeteroData()
        data["site"].num_nodes = 2
        data["site"].site_type = ["N_linked", "O_linked"]
        result = BindingSiteTask._extract_site_types(
            data, np.array([0, 1]), num_pos=2
        )
        assert result is not None
        assert result[0] == "N-linked"
        assert result[1] == "O-linked"
