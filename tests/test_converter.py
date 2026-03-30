"""Unit tests for glycoMusubi.data.converter.KGConverter."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from glycoMusubi.data.converter import KGConverter

# ---------------------------------------------------------------------------
# Expected counts derived from tests/test_data/mini_nodes.tsv & mini_edges.tsv
# ---------------------------------------------------------------------------

EXPECTED_NODE_TYPES = {"protein", "enzyme", "glycan", "disease", "variant", "compound", "site", "reaction", "motif"}

# Counts matching tests/test_data/mini_nodes.tsv (maintained by verify-system)
EXPECTED_NODE_COUNTS = {
    "protein": 4,
    "enzyme": 2,
    "glycan": 3,
    "disease": 2,
    "variant": 1,
    "compound": 1,
    "site": 2,
    "reaction": 1,
    "motif": 1,
}

TOTAL_NODES = sum(EXPECTED_NODE_COUNTS.values())  # 17
TOTAL_EDGES = 23  # from mini_edges.tsv (15 original + 8 new relation types)

EXPECTED_EDGE_RELATIONS = {
    "inhibits",
    "has_glycan",
    "associated_with_disease",
    "has_variant",
    "has_site",
    "ptm_crosstalk",
    "parent_of",
    "child_of",
    "subsumes",
    "subsumed_by",
    "has_product",
    "produced_by",
    "has_motif",
    "catalyzed_by",
}


class TestKGConverter:
    """Tests for KGConverter against mini test data."""

    @pytest.fixture(autouse=True)
    def setup_converter(self, mini_kg_dir):
        """Create a KGConverter pointed at the mini KG directory."""
        schema_dir = mini_kg_dir.parent / "test_data"
        # Use the project schemas directory (which has both relation_config and edge_schema)
        from pathlib import Path
        project_schemas = Path(__file__).resolve().parents[1] / "schemas"
        self.converter = KGConverter(kg_dir=mini_kg_dir, schema_dir=project_schemas)

    def test_load_dataframes(self, mini_kg_dir):
        """load_dataframes returns non-empty DataFrames with expected columns."""
        nodes_df, edges_df = self.converter.load_dataframes()

        assert isinstance(nodes_df, pd.DataFrame)
        assert isinstance(edges_df, pd.DataFrame)

        assert "node_id" in nodes_df.columns
        assert "node_type" in nodes_df.columns
        assert "source_id" in edges_df.columns
        assert "target_id" in edges_df.columns
        assert "relation" in edges_df.columns

        assert len(nodes_df) == TOTAL_NODES
        assert len(edges_df) == TOTAL_EDGES

    def test_build_node_mappings(self):
        """build_node_mappings creates a mapping for every node type."""
        nodes_df, _ = self.converter.load_dataframes()
        mappings = self.converter.build_node_mappings(nodes_df)

        # All 7 node types must have mappings
        assert set(mappings.keys()) == EXPECTED_NODE_TYPES

        # Check per-type counts
        for ntype, expected_count in EXPECTED_NODE_COUNTS.items():
            assert len(mappings[ntype]) == expected_count, (
                f"Node type '{ntype}' expected {expected_count} nodes, "
                f"got {len(mappings[ntype])}"
            )

        # Mapping values should be contiguous 0-based integers
        for ntype, mapping in mappings.items():
            indices = sorted(mapping.values())
            assert indices == list(range(len(indices)))

    def test_build_hetero_graph_structure(self):
        """build_hetero_graph produces HeteroData with correct edge_index shapes."""
        nodes_df, edges_df = self.converter.load_dataframes()
        mappings = self.converter.build_node_mappings(nodes_df)
        data = self.converter.build_hetero_graph(nodes_df, edges_df, mappings, feature_dim=32)

        # edge_index must be [2, num_edges] for each edge type
        for etype in data.edge_types:
            ei = data[etype].edge_index
            assert ei.dim() == 2
            assert ei.size(0) == 2
            assert ei.size(1) > 0  # No empty edge types in our mini data

        # Total edges across all types should equal TOTAL_EDGES
        total = sum(data[et].edge_index.size(1) for et in data.edge_types)
        assert total == TOTAL_EDGES

    def test_all_node_types_present(self):
        """All 7 node types from the schema are present in the HeteroData."""
        nodes_df, edges_df = self.converter.load_dataframes()
        mappings = self.converter.build_node_mappings(nodes_df)
        data = self.converter.build_hetero_graph(nodes_df, edges_df, mappings)

        assert set(data.node_types) == EXPECTED_NODE_TYPES

    def test_all_edge_types_present(self):
        """All 6 relation types from edge_schema are present."""
        nodes_df, edges_df = self.converter.load_dataframes()
        mappings = self.converter.build_node_mappings(nodes_df)
        data = self.converter.build_hetero_graph(nodes_df, edges_df, mappings)

        relations_in_data = {et[1] for et in data.edge_types}
        assert EXPECTED_EDGE_RELATIONS.issubset(relations_in_data)

    def test_node_features_shape(self):
        """Node features have shape [num_nodes, feature_dim] for each type."""
        feature_dim = 64
        nodes_df, edges_df = self.converter.load_dataframes()
        mappings = self.converter.build_node_mappings(nodes_df)
        data = self.converter.build_hetero_graph(
            nodes_df, edges_df, mappings, feature_dim=feature_dim
        )

        for ntype in data.node_types:
            x = data[ntype].x
            expected_n = EXPECTED_NODE_COUNTS[ntype]
            assert x.shape == (expected_n, feature_dim), (
                f"Node type '{ntype}': expected shape ({expected_n}, {feature_dim}), "
                f"got {tuple(x.shape)}"
            )

    def test_edge_index_valid_range(self):
        """All edge indices fall within [0, num_nodes) for their respective types."""
        nodes_df, edges_df = self.converter.load_dataframes()
        mappings = self.converter.build_node_mappings(nodes_df)
        data = self.converter.build_hetero_graph(nodes_df, edges_df, mappings)

        for etype in data.edge_types:
            src_type, rel, dst_type = etype
            ei = data[etype].edge_index
            num_src = data[src_type].num_nodes
            num_dst = data[dst_type].num_nodes

            assert ei[0].max().item() < num_src, (
                f"Edge type {etype}: source index {ei[0].max().item()} >= num_src {num_src}"
            )
            assert ei[1].max().item() < num_dst, (
                f"Edge type {etype}: target index {ei[1].max().item()} >= num_dst {num_dst}"
            )
            assert ei.min().item() >= 0, f"Edge type {etype}: negative index found"

    def test_convert_end_to_end(self):
        """convert() returns a valid HeteroData and node_mappings."""
        data, mappings = self.converter.convert(feature_dim=128)

        assert len(data.node_types) == len(EXPECTED_NODE_TYPES)
        assert len(mappings) == len(EXPECTED_NODE_TYPES)
        for ntype in data.node_types:
            assert data[ntype].x.shape[1] == 128

    def test_extract_node_metadata(self):
        """extract_node_metadata parses JSON metadata correctly."""
        nodes_df, _ = self.converter.load_dataframes()
        metadata = self.converter.extract_node_metadata(nodes_df)

        # Enzyme O43451-1 should have source "GlyGen"
        assert "enzyme" in metadata
        assert "O43451-1" in metadata["enzyme"]
        assert metadata["enzyme"]["O43451-1"].get("source") == "GlyGen"

        # Glycan G00002BB has empty metadata
        assert metadata["glycan"]["G00002BB"] == {}

    def test_missing_nodes_file_raises(self, tmp_path):
        """KGConverter raises FileNotFoundError for missing nodes file."""
        from pathlib import Path
        empty_dir = tmp_path / "empty_kg"
        empty_dir.mkdir()
        project_schemas = Path(__file__).resolve().parents[1] / "schemas"
        converter = KGConverter(kg_dir=empty_dir, schema_dir=project_schemas)
        with pytest.raises(FileNotFoundError, match="nodes"):
            converter.load_dataframes()
