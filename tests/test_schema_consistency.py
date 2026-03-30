"""Schema consistency and data-conversion accuracy tests for glycoMusubi.

This module validates that converter.py faithfully reflects the YAML schema
definitions and that TSV-to-HeteroData conversion introduces no data loss
or structural errors.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from glycoMusubi.data.converter import KGConverter

# ---------------------------------------------------------------------------
# Paths to schema files
# ---------------------------------------------------------------------------

_SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas"
_NODE_SCHEMA = _SCHEMA_DIR / "node_schema.yaml"
_EDGE_SCHEMA = _SCHEMA_DIR / "edge_schema.yaml"
_RELATION_CFG = _SCHEMA_DIR / "relation_config.yaml"
_ENTITY_CFG = _SCHEMA_DIR / "entity_config.yaml"


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture(scope="module")
def node_schema():
    return _load_yaml(_NODE_SCHEMA)


@pytest.fixture(scope="module")
def edge_schema():
    return _load_yaml(_EDGE_SCHEMA)


@pytest.fixture(scope="module")
def relation_config():
    return _load_yaml(_RELATION_CFG)


@pytest.fixture(scope="module")
def entity_config():
    return _load_yaml(_ENTITY_CFG)


@pytest.fixture(scope="module")
def converter():
    """KGConverter with default schema_dir (project schemas/)."""
    return KGConverter(kg_dir="kg", schema_dir=_SCHEMA_DIR)


@pytest.fixture(scope="module")
def mini_converter(test_data_dir_module):
    """KGConverter configured for mini test data."""
    return KGConverter(kg_dir=test_data_dir_module, schema_dir=_SCHEMA_DIR)


@pytest.fixture(scope="module")
def test_data_dir_module(tmp_path_factory):
    """Prepare a temporary KG directory with mini nodes/edges for KGConverter."""
    import shutil
    src_dir = Path(__file__).parent / "test_data"
    tmp = tmp_path_factory.mktemp("mini_kg")
    shutil.copy(src_dir / "mini_nodes.tsv", tmp / "nodes.tsv")
    shutil.copy(src_dir / "mini_edges.tsv", tmp / "edges.tsv")
    return tmp


@pytest.fixture(scope="module")
def mini_nodes_df(mini_converter):
    nodes_df, _ = mini_converter.load_dataframes()
    return nodes_df


@pytest.fixture(scope="module")
def mini_edges_df(mini_converter):
    _, edges_df = mini_converter.load_dataframes()
    return edges_df


@pytest.fixture(scope="module")
def mini_node_mappings(mini_converter, mini_nodes_df):
    return mini_converter.build_node_mappings(mini_nodes_df)


@pytest.fixture(scope="module")
def mini_hetero(mini_converter, mini_nodes_df, mini_edges_df, mini_node_mappings):
    return mini_converter.build_hetero_graph(
        mini_nodes_df, mini_edges_df, mini_node_mappings, feature_dim=32
    )


# ===================================================================
# 1. Schema Consistency: node types
# ===================================================================

class TestNodeTypeConsistency:
    """Verify that converter handles all node types defined in schemas."""

    def test_all_node_types_in_converter(self, node_schema, converter):
        """node_schema.yaml node types must all be represented in the converter's relation map.

        Every node type that appears as a source or target in any relation
        should be reachable via the converter's _relation_type_map.
        """
        schema_node_types = set(node_schema["node_types"].keys())

        # Collect all node types referenced from converter relation map
        converter_node_types: set[str] = set()
        for pairs in converter._relation_type_map.values():
            for src, tgt in pairs:
                converter_node_types.add(src)
                converter_node_types.add(tgt)

        missing = schema_node_types - converter_node_types
        assert missing == set(), (
            f"Node types defined in node_schema.yaml but unreachable in "
            f"converter relation map: {missing}"
        )

    def test_node_schema_matches_entity_config(self, node_schema, entity_config):
        """node_schema.yaml node types should be a superset of entity_config types."""
        schema_types = set(node_schema["node_types"].keys())
        entity_types = set(entity_config["entity_types"].keys())

        missing_from_schema = entity_types - schema_types
        assert missing_from_schema == set(), (
            f"entity_config has types not in node_schema: {missing_from_schema}"
        )

        # entity_config may omit types like 'site' that are derived, but log it
        extra_in_schema = schema_types - entity_types
        if extra_in_schema:
            # This is acceptable (e.g. 'site' is defined in node_schema but
            # generated from edge data rather than a standalone source table)
            pass

    def test_mini_data_has_all_node_types(self, node_schema, mini_nodes_df):
        """The mini test dataset should cover all 7 node types."""
        schema_types = set(node_schema["node_types"].keys())
        data_types = set(mini_nodes_df["node_type"].unique())
        missing = schema_types - data_types
        assert missing == set(), (
            f"Mini test data is missing node types: {missing}"
        )


# ===================================================================
# 2. Schema Consistency: edge / relation types
# ===================================================================

class TestEdgeTypeConsistency:
    """Verify that converter handles all edge types from schemas."""

    def test_all_edge_types_in_converter(self, edge_schema, converter):
        """edge_schema.yaml edge types must all appear in converter._relation_type_map."""
        schema_edge_types = set(edge_schema["edge_types"].keys())
        converter_edge_types = set(converter._relation_type_map.keys())

        missing = schema_edge_types - converter_edge_types
        assert missing == set(), (
            f"Edge types defined in edge_schema.yaml but missing from "
            f"converter: {missing}"
        )

    def test_relation_config_subset_of_edge_schema(self, edge_schema, relation_config):
        """relation_config.yaml types should be a subset of edge_schema types."""
        edge_types = set(edge_schema["edge_types"].keys())
        rel_types = set(relation_config["relation_types"].keys())

        extra = rel_types - edge_types
        assert extra == set(), (
            f"relation_config has types not in edge_schema: {extra}"
        )

    def test_mini_data_has_all_edge_types(self, edge_schema, mini_edges_df):
        """The mini test dataset should cover all 6 edge types."""
        schema_types = set(edge_schema["edge_types"].keys())
        data_types = set(mini_edges_df["relation"].unique())
        missing = schema_types - data_types
        assert missing == set(), (
            f"Mini test data is missing edge types: {missing}"
        )


# ===================================================================
# 3. Source/target type matching
# ===================================================================

class TestRelationSourceTargetMatch:
    """Verify source_type/target_type in configs match converter edge tuples."""

    def test_relation_config_source_target_match(self, relation_config, converter):
        """relation_config source/target types must match converter edge tuples."""
        for rel_name, spec in relation_config["relation_types"].items():
            src = spec["source_type"]
            tgt = spec["target_type"]

            # Normalise to lists
            src_list = src if isinstance(src, list) else [src]
            tgt_list = tgt if isinstance(tgt, list) else [tgt]

            expected_pairs = {(s, t) for s in src_list for t in tgt_list}
            actual_pairs = set(converter._relation_type_map.get(rel_name, []))

            assert actual_pairs == expected_pairs, (
                f"Relation '{rel_name}': expected pairs {expected_pairs}, "
                f"got {actual_pairs}"
            )

    def test_edge_schema_source_target_match(self, edge_schema, converter):
        """edge_schema source/target types must match converter edge tuples."""
        for rel_name, spec in edge_schema["edge_types"].items():
            src = spec["source_type"]
            tgt = spec["target_type"]

            src_list = src if isinstance(src, list) else [src]
            tgt_list = tgt if isinstance(tgt, list) else [tgt]

            expected_pairs = {(s, t) for s in src_list for t in tgt_list}
            actual_pairs = set(converter._relation_type_map.get(rel_name, []))

            assert actual_pairs == expected_pairs, (
                f"Relation '{rel_name}': expected pairs {expected_pairs}, "
                f"got {actual_pairs}"
            )

    def test_edge_type_endpoints_use_valid_node_types(self, node_schema, converter):
        """All source/target types in the converter must be valid node types."""
        valid_types = set(node_schema["node_types"].keys())

        for rel, pairs in converter._relation_type_map.items():
            for src, tgt in pairs:
                assert src in valid_types, (
                    f"Relation '{rel}' uses invalid source type '{src}'"
                )
                assert tgt in valid_types, (
                    f"Relation '{rel}' uses invalid target type '{tgt}'"
                )


# ===================================================================
# 4. Node ID pattern preservation
# ===================================================================

class TestNodeIdPatterns:
    """Verify that node IDs follow expected patterns from schema."""

    def test_glycan_id_pattern(self, node_schema, mini_nodes_df):
        """Glycan node IDs must match GlyTouCan pattern."""
        import re
        pattern = node_schema["node_types"]["glycan"]["id_pattern"]
        glycan_ids = mini_nodes_df[
            mini_nodes_df["node_type"] == "glycan"
        ]["node_id"].values

        for gid in glycan_ids:
            assert re.match(pattern, gid), (
                f"Glycan ID '{gid}' does not match pattern '{pattern}'"
            )

    def test_compound_id_pattern(self, node_schema, mini_nodes_df):
        """Compound node IDs must match ChEMBL pattern."""
        import re
        pattern = node_schema["node_types"]["compound"]["id_pattern"]
        compound_ids = mini_nodes_df[
            mini_nodes_df["node_type"] == "compound"
        ]["node_id"].values

        for cid in compound_ids:
            assert re.fullmatch(pattern, cid), (
                f"Compound ID '{cid}' does not match pattern '{pattern}'"
            )

    def test_site_id_pattern(self, node_schema, mini_nodes_df):
        """Site node IDs must match the SITE:: pattern."""
        import re
        pattern = node_schema["node_types"]["site"]["id_pattern"]
        site_ids = mini_nodes_df[
            mini_nodes_df["node_type"] == "site"
        ]["node_id"].values

        for sid in site_ids:
            assert re.match(pattern, sid), (
                f"Site ID '{sid}' does not match pattern '{pattern}'"
            )

    def test_node_ids_preserved_in_mappings(self, mini_nodes_df, mini_node_mappings):
        """Every node_id from the DataFrame must appear in node_mappings."""
        for _, row in mini_nodes_df.iterrows():
            nid = str(row["node_id"])
            ntype = str(row["node_type"])
            assert ntype in mini_node_mappings, (
                f"Node type '{ntype}' missing from mappings"
            )
            assert nid in mini_node_mappings[ntype], (
                f"Node ID '{nid}' missing from mappings['{ntype}']"
            )


# ===================================================================
# 5. Metadata extraction accuracy
# ===================================================================

class TestMetadataExtraction:
    """Verify that metadata fields (WURCS, gene_symbol, etc.) are correctly parsed."""

    def test_extract_node_metadata_roundtrip(self, mini_converter, mini_nodes_df):
        """extract_node_metadata should parse JSON metadata without loss."""
        metadata = mini_converter.extract_node_metadata(mini_nodes_df)

        # Every node type present in the DataFrame must appear
        expected_types = set(mini_nodes_df["node_type"].unique())
        actual_types = set(metadata.keys())
        assert expected_types == actual_types

        # Every node_id must appear under its type
        for _, row in mini_nodes_df.iterrows():
            ntype = str(row["node_type"])
            nid = str(row["node_id"])
            assert nid in metadata[ntype], (
                f"Node '{nid}' missing from metadata['{ntype}']"
            )

    def test_enzyme_metadata_has_source(self, mini_converter, mini_nodes_df):
        """Enzyme nodes should retain source in metadata."""
        metadata = mini_converter.extract_node_metadata(mini_nodes_df)
        assert "enzyme" in metadata
        assert "O43451-1" in metadata["enzyme"]
        assert metadata["enzyme"]["O43451-1"]["source"] == "GlyGen"

    def test_protein_metadata_has_source(self, mini_converter, mini_nodes_df):
        """Protein nodes should retain source in metadata."""
        metadata = mini_converter.extract_node_metadata(mini_nodes_df)
        p_meta = metadata["protein"]["P12345"]
        assert p_meta.get("source") == "UniProt"

    def test_site_metadata_has_position_and_residue(self, mini_converter, mini_nodes_df):
        """Site nodes should retain position and residue in metadata."""
        metadata = mini_converter.extract_node_metadata(mini_nodes_df)
        site_meta = metadata["site"]["SITE::P12345::42::N"]
        assert site_meta["position"] == 42
        assert site_meta["residue"] == "N"
        assert site_meta["source"] == "UniProt"

    def test_disease_metadata_has_source(self, mini_converter, mini_nodes_df):
        """Disease nodes should retain the source field."""
        metadata = mini_converter.extract_node_metadata(mini_nodes_df)
        d_meta = metadata["disease"]["Diabetes mellitus"]
        assert "source" in d_meta
        assert d_meta["source"] == "UniProt"

    def test_compound_metadata_has_source(self, mini_converter, mini_nodes_df):
        """Compound nodes should retain the source field."""
        metadata = mini_converter.extract_node_metadata(mini_nodes_df)
        c_meta = metadata["compound"]["CHEMBL10001"]
        assert c_meta["source"] == "ChEMBL"

    def test_metadata_json_parse_preserves_types(self, mini_converter, mini_nodes_df):
        """JSON parsing should preserve numeric types (e.g. position as int)."""
        metadata = mini_converter.extract_node_metadata(mini_nodes_df)
        site_meta = metadata["site"]["SITE::P12345::42::N"]
        assert isinstance(site_meta["position"], int)


# ===================================================================
# 6. Data conversion: no data loss
# ===================================================================

class TestNoDataLoss:
    """Verify TSV-to-HeteroData conversion preserves all data."""

    def test_total_node_count(self, mini_nodes_df, mini_hetero, mini_node_mappings):
        """Total number of nodes must match original DataFrame."""
        expected = len(mini_nodes_df)
        actual = sum(
            mini_hetero[ntype].num_nodes
            for ntype in mini_node_mappings
        )
        assert actual == expected, (
            f"Node count mismatch: DataFrame has {expected}, "
            f"HeteroData has {actual}"
        )

    def test_per_type_node_count(self, mini_nodes_df, mini_hetero, mini_node_mappings):
        """Per-type node counts must match DataFrame groupby counts."""
        expected_counts = mini_nodes_df.groupby("node_type").size().to_dict()

        for ntype, exp_count in expected_counts.items():
            actual_count = mini_hetero[ntype].num_nodes
            assert actual_count == exp_count, (
                f"Node type '{ntype}': expected {exp_count}, got {actual_count}"
            )

    def test_total_edge_count(self, mini_edges_df, mini_hetero):
        """Total edge count in HeteroData must match DataFrame."""
        expected = len(mini_edges_df)

        actual = 0
        for edge_type in mini_hetero.edge_types:
            actual += mini_hetero[edge_type].edge_index.size(1)

        assert actual == expected, (
            f"Edge count mismatch: DataFrame has {expected}, "
            f"HeteroData has {actual}"
        )

    def test_per_relation_edge_count(self, mini_edges_df, mini_hetero):
        """Per-relation edge counts must match DataFrame groupby counts."""
        # Count edges per relation in DataFrame
        expected_per_rel = mini_edges_df.groupby("relation").size().to_dict()

        # Count edges per relation in HeteroData
        actual_per_rel: dict[str, int] = {}
        for src_type, rel, dst_type in mini_hetero.edge_types:
            actual_per_rel[rel] = (
                actual_per_rel.get(rel, 0)
                + mini_hetero[src_type, rel, dst_type].edge_index.size(1)
            )

        for rel, exp_count in expected_per_rel.items():
            act_count = actual_per_rel.get(rel, 0)
            assert act_count == exp_count, (
                f"Relation '{rel}': expected {exp_count} edges, got {act_count}"
            )

    def test_all_node_types_present_in_hetero(self, mini_nodes_df, mini_hetero):
        """All node types from DataFrame must exist in HeteroData."""
        expected_types = set(mini_nodes_df["node_type"].unique())
        actual_types = set(mini_hetero.node_types)
        missing = expected_types - actual_types
        assert missing == set(), (
            f"Node types missing from HeteroData: {missing}"
        )

    def test_all_edge_types_present_in_hetero(self, mini_edges_df, mini_hetero):
        """All relation types from DataFrame must exist in HeteroData."""
        expected_rels = set(mini_edges_df["relation"].unique())
        actual_rels = {et[1] for et in mini_hetero.edge_types}
        missing = expected_rels - actual_rels
        assert missing == set(), (
            f"Relation types missing from HeteroData: {missing}"
        )


# ===================================================================
# 7. Edge index validity
# ===================================================================

class TestEdgeIndexValidity:
    """Verify edge_index tensors are structurally valid."""

    def test_edge_index_in_bounds(self, mini_hetero, mini_node_mappings):
        """All edge indices must be within [0, num_nodes) for their type."""
        for src_type, rel, dst_type in mini_hetero.edge_types:
            ei = mini_hetero[src_type, rel, dst_type].edge_index
            src_max = len(mini_node_mappings[src_type])
            dst_max = len(mini_node_mappings[dst_type])

            assert ei[0].max().item() < src_max, (
                f"({src_type},{rel},{dst_type}): source index {ei[0].max().item()} "
                f">= num_nodes {src_max}"
            )
            assert ei[1].max().item() < dst_max, (
                f"({src_type},{rel},{dst_type}): target index {ei[1].max().item()} "
                f">= num_nodes {dst_max}"
            )
            assert ei[0].min().item() >= 0, "Negative source index"
            assert ei[1].min().item() >= 0, "Negative target index"

    def test_edge_index_shape(self, mini_hetero):
        """edge_index must have shape [2, num_edges]."""
        for edge_type in mini_hetero.edge_types:
            ei = mini_hetero[edge_type].edge_index
            assert ei.dim() == 2, f"{edge_type}: edge_index is not 2D"
            assert ei.size(0) == 2, (
                f"{edge_type}: edge_index first dim is {ei.size(0)}, expected 2"
            )

    def test_edge_index_dtype(self, mini_hetero):
        """edge_index must be torch.long."""
        for edge_type in mini_hetero.edge_types:
            ei = mini_hetero[edge_type].edge_index
            assert ei.dtype == torch.long, (
                f"{edge_type}: edge_index dtype is {ei.dtype}, expected torch.long"
            )

    def test_no_self_loops_in_asymmetric_relations(self, mini_hetero):
        """Asymmetric relations (different src/dst type) cannot have self-loops."""
        for src_type, rel, dst_type in mini_hetero.edge_types:
            if src_type == dst_type:
                continue  # Self-loops are only possible when types match
            ei = mini_hetero[src_type, rel, dst_type].edge_index
            # Different type spaces: same index != same node, so no check needed.
            # This test documents the structural impossibility.

    def test_ptm_crosstalk_no_self_loops(self, mini_hetero):
        """ptm_crosstalk (site->site) should not have self-loops."""
        key = ("site", "ptm_crosstalk", "site")
        if key in mini_hetero.edge_types:
            ei = mini_hetero[key].edge_index
            self_loops = (ei[0] == ei[1]).sum().item()
            assert self_loops == 0, (
                f"ptm_crosstalk has {self_loops} self-loop(s)"
            )


# ===================================================================
# 8. Node feature initialisation
# ===================================================================

class TestNodeFeatures:
    """Verify initial node features are correctly set up."""

    def test_feature_dim_consistent(self, mini_hetero):
        """All node types should have the same feature dimensionality."""
        dims = set()
        for ntype in mini_hetero.node_types:
            dims.add(mini_hetero[ntype].x.size(1))
        assert len(dims) == 1, f"Inconsistent feature dims: {dims}"

    def test_feature_values_finite(self, mini_hetero):
        """Xavier-initialised features must be finite (no NaN/Inf)."""
        for ntype in mini_hetero.node_types:
            x = mini_hetero[ntype].x
            assert torch.isfinite(x).all(), (
                f"Node type '{ntype}' has non-finite feature values"
            )

    def test_num_nodes_matches_feature_rows(self, mini_hetero):
        """x.size(0) must equal num_nodes for every type."""
        for ntype in mini_hetero.node_types:
            assert mini_hetero[ntype].x.size(0) == mini_hetero[ntype].num_nodes, (
                f"Node type '{ntype}': x rows != num_nodes"
            )


# ===================================================================
# 9. Duplicate edge detection
# ===================================================================

class TestDuplicateEdges:
    """Check for duplicate edges in the converted graph."""

    def test_no_duplicate_edges(self, mini_hetero):
        """Each edge type should not contain duplicate (src, dst) pairs."""
        for edge_type in mini_hetero.edge_types:
            ei = mini_hetero[edge_type].edge_index
            pairs = set()
            num_edges = ei.size(1)
            for i in range(num_edges):
                pair = (ei[0, i].item(), ei[1, i].item())
                assert pair not in pairs, (
                    f"{edge_type}: duplicate edge {pair}"
                )
                pairs.add(pair)


# ===================================================================
# 10. Converter robustness: missing / dangling references
# ===================================================================

class TestConverterRobustness:
    """Test converter behaviour with edge cases."""

    def test_edges_with_missing_nodes_are_skipped(self, tmp_path):
        """Edges referencing non-existent nodes should be skipped, not crash."""
        nodes_tsv = tmp_path / "nodes.tsv"
        edges_tsv = tmp_path / "edges.tsv"

        nodes_tsv.write_text(
            "node_id\tnode_type\tlabel\tmetadata\n"
            "P00001\tprotein\tTestProt\t{}\n"
            "G00001AA\tglycan\tTestGlycan\t{}\n"
        )
        edges_tsv.write_text(
            "source_id\ttarget_id\trelation\tmetadata\n"
            "P00001\tG00001AA\thas_glycan\t{}\n"
            "P00001\tMISSING_NODE\thas_glycan\t{}\n"
        )

        conv = KGConverter(kg_dir=tmp_path, schema_dir=_SCHEMA_DIR)
        nodes_df, edges_df = conv.load_dataframes()
        mappings = conv.build_node_mappings(nodes_df)
        data = conv.build_hetero_graph(nodes_df, edges_df, mappings)

        # Only the valid edge should be present
        total_edges = sum(
            data[et].edge_index.size(1) for et in data.edge_types
        )
        assert total_edges == 1

    def test_unknown_relation_handled_gracefully(self, tmp_path):
        """Edges with unknown relation types should use actual node types."""
        nodes_tsv = tmp_path / "nodes.tsv"
        edges_tsv = tmp_path / "edges.tsv"

        nodes_tsv.write_text(
            "node_id\tnode_type\tlabel\tmetadata\n"
            "P00001\tprotein\tTestProt\t{}\n"
            "G00001AA\tglycan\tTestGlycan\t{}\n"
        )
        edges_tsv.write_text(
            "source_id\ttarget_id\trelation\tmetadata\n"
            "P00001\tG00001AA\tnew_relation\t{}\n"
        )

        conv = KGConverter(kg_dir=tmp_path, schema_dir=_SCHEMA_DIR)
        nodes_df, edges_df = conv.load_dataframes()
        mappings = conv.build_node_mappings(nodes_df)
        data = conv.build_hetero_graph(nodes_df, edges_df, mappings)

        # The edge should exist with the actual node types
        assert ("protein", "new_relation", "glycan") in data.edge_types
