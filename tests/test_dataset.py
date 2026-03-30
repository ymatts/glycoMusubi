"""Unit tests for glycoMusubi.data.dataset.GlycoKGDataset."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch

from glycoMusubi.data.dataset import GlycoKGDataset


@pytest.fixture
def dataset_root(tmp_path, test_data_dir):
    """Set up a temp directory structure for GlycoKGDataset.

    Copies mini_nodes.tsv -> nodes.tsv, mini_edges.tsv -> edges.tsv
    into a kg/ subdirectory.
    """
    kg_dir = tmp_path / "kg"
    kg_dir.mkdir()
    shutil.copy(test_data_dir / "mini_nodes.tsv", kg_dir / "nodes.tsv")
    shutil.copy(test_data_dir / "mini_edges.tsv", kg_dir / "edges.tsv")
    return tmp_path, kg_dir


class TestGlycoKGDataset:
    """Tests for GlycoKGDataset (InMemoryDataset wrapper)."""

    def test_dataset_len(self, dataset_root):
        """Dataset contains exactly one HeteroData graph."""
        root, kg_dir = dataset_root
        schema_dir = Path(__file__).resolve().parents[1] / "schemas"
        ds = GlycoKGDataset(
            root=root / "dataset",
            kg_dir=kg_dir,
            schema_dir=schema_dir,
            feature_dim=32,
        )
        assert len(ds) == 1

    def test_dataset_get(self, dataset_root):
        """ds[0] returns a HeteroData with node types and edge types."""
        root, kg_dir = dataset_root
        schema_dir = Path(__file__).resolve().parents[1] / "schemas"
        ds = GlycoKGDataset(
            root=root / "dataset",
            kg_dir=kg_dir,
            schema_dir=schema_dir,
            feature_dim=32,
        )
        data = ds[0]
        assert hasattr(data, "node_types")
        assert hasattr(data, "edge_types")
        assert len(data.node_types) > 0
        assert len(data.edge_types) > 0

    def test_hetero_data_property(self, dataset_root):
        """hetero_data property returns the same object as ds[0]."""
        root, kg_dir = dataset_root
        schema_dir = Path(__file__).resolve().parents[1] / "schemas"
        ds = GlycoKGDataset(
            root=root / "dataset",
            kg_dir=kg_dir,
            schema_dir=schema_dir,
            feature_dim=32,
        )
        hd = ds.hetero_data
        assert hd is not None
        assert len(hd.node_types) == len(ds[0].node_types)

    def test_processed_cache(self, dataset_root):
        """Second instantiation loads from cache without re-processing."""
        root, kg_dir = dataset_root
        schema_dir = Path(__file__).resolve().parents[1] / "schemas"
        ds_dir = root / "dataset_cache"

        # First: triggers process()
        ds1 = GlycoKGDataset(
            root=ds_dir, kg_dir=kg_dir, schema_dir=schema_dir, feature_dim=64
        )
        processed_file = Path(ds_dir) / "processed" / "hetero_data.pt"
        assert processed_file.exists()

        # Second: loads from cache
        ds2 = GlycoKGDataset(
            root=ds_dir, kg_dir=kg_dir, schema_dir=schema_dir, feature_dim=64
        )
        assert len(ds2) == 1

    def test_processed_file_names(self, dataset_root):
        """processed_file_names returns the expected list."""
        root, kg_dir = dataset_root
        schema_dir = Path(__file__).resolve().parents[1] / "schemas"
        ds = GlycoKGDataset(
            root=root / "dataset",
            kg_dir=kg_dir,
            schema_dir=schema_dir,
            feature_dim=32,
        )
        assert ds.processed_file_names == ["hetero_data.pt"]
