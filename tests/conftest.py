"""Shared pytest fixtures for glycoMusubi test suite."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

@pytest.fixture
def test_data_dir():
    """Path to the test_data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def mini_nodes_path(test_data_dir):
    """Path to mini_nodes.tsv."""
    return test_data_dir / "mini_nodes.tsv"


@pytest.fixture
def mini_edges_path(test_data_dir):
    """Path to mini_edges.tsv."""
    return test_data_dir / "mini_edges.tsv"


@pytest.fixture
def mini_relation_config_path(test_data_dir):
    """Path to mini_relation_config.yaml."""
    return test_data_dir / "mini_relation_config.yaml"


# ---------------------------------------------------------------------------
# HeteroData fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mini_hetero_data():
    """Minimal HeteroData fixture matching mini_nodes/mini_edges structure.

    Node types and counts mirror the mini TSV files:
      protein: 2, enzyme: 2, glycan: 3, disease: 2,
      variant: 2, compound: 2, site: 3
    """
    from torch_geometric.data import HeteroData

    data = HeteroData()

    # -- Node features (Xavier-uniform, dim=256) --
    node_counts = {
        "protein": 2,
        "enzyme": 2,
        "glycan": 3,
        "disease": 2,
        "variant": 2,
        "compound": 2,
        "site": 3,
    }
    for ntype, n in node_counts.items():
        x = torch.empty(n, 256)
        torch.nn.init.xavier_uniform_(x)
        data[ntype].x = x
        data[ntype].num_nodes = n

    # -- Edges (match mini_edges.tsv) --
    # inhibits: compound -> enzyme (3 edges)
    data["compound", "inhibits", "enzyme"].edge_index = torch.tensor(
        [[0, 0, 1], [0, 1, 0]]
    )
    # has_glycan: protein -> glycan (4 edges)
    data["protein", "has_glycan", "glycan"].edge_index = torch.tensor(
        [[0, 0, 1, 1], [0, 1, 0, 2]]
    )
    # associated_with_disease: protein -> disease (3 edges)
    data["protein", "associated_with_disease", "disease"].edge_index = torch.tensor(
        [[0, 1, 1], [0, 0, 1]]
    )
    # has_variant: protein -> variant (3 edges)
    data["protein", "has_variant", "variant"].edge_index = torch.tensor(
        [[0, 0, 1], [0, 1, 0]]
    )
    # has_site: protein -> site (2 edges)
    data["protein", "has_site", "site"].edge_index = torch.tensor(
        [[0, 0], [0, 1]]
    )
    # has_site: enzyme -> site (1 edge)
    data["enzyme", "has_site", "site"].edge_index = torch.tensor(
        [[0], [2]]
    )
    # ptm_crosstalk: site -> site (2 edges)
    data["site", "ptm_crosstalk", "site"].edge_index = torch.tensor(
        [[0, 1], [1, 2]]
    )

    return data


# ---------------------------------------------------------------------------
# Triples / tensors
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_triples():
    """Test triples as (head_idx, relation_idx, tail_idx) tensor.

    Relation indices:
      0 = has_glycan, 1 = associated_with_disease, 2 = inhibits
    """
    return torch.tensor([
        [0, 0, 0],  # protein0 -has_glycan-> glycan0
        [0, 0, 1],  # protein0 -has_glycan-> glycan1
        [1, 0, 0],  # protein1 -has_glycan-> glycan0
        [0, 1, 0],  # protein0 -associated_with_disease-> disease0
        [1, 1, 1],  # protein1 -associated_with_disease-> disease1
        [0, 2, 0],  # compound0 -inhibits-> enzyme0
    ])


@pytest.fixture
def device():
    """Best available torch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# KGConverter convenience
# ---------------------------------------------------------------------------

@pytest.fixture
def mini_kg_dir(test_data_dir, tmp_path):
    """Prepare a temporary KG directory with mini nodes/edges for KGConverter.

    Copies mini_nodes.tsv -> nodes.tsv and mini_edges.tsv -> edges.tsv
    into a fresh tmp_path so KGConverter can load them by default names.
    """
    import shutil

    src_nodes = test_data_dir / "mini_nodes.tsv"
    src_edges = test_data_dir / "mini_edges.tsv"

    shutil.copy(src_nodes, tmp_path / "nodes.tsv")
    shutil.copy(src_edges, tmp_path / "edges.tsv")

    return tmp_path
