#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests for build_kg module - node/edge builder logic.
"""

import pytest
import sys
import os
import json
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from build_kg import (
    merge_metadata,
    NodeAccumulator,
    EdgeAccumulator,
)


class TestMergeMetadata:
    """Tests for merge_metadata function."""
    
    def test_merge_empty_dicts(self):
        """Test merging two empty dictionaries."""
        result = merge_metadata({}, {})
        assert result == {}
    
    def test_merge_with_empty_existing(self):
        """Test merging when existing is empty."""
        result = merge_metadata({}, {"key": "value"})
        assert result == {"key": "value"}
    
    def test_merge_with_empty_new(self):
        """Test merging when new is empty."""
        result = merge_metadata({"key": "value"}, {})
        assert result == {"key": "value"}
    
    def test_merge_non_overlapping_keys(self):
        """Test merging dictionaries with different keys."""
        existing = {"a": 1, "b": 2}
        new = {"c": 3, "d": 4}
        result = merge_metadata(existing, new)
        assert result == {"a": 1, "b": 2, "c": 3, "d": 4}
    
    def test_merge_same_values(self):
        """Test merging when values are identical."""
        existing = {"key": "value"}
        new = {"key": "value"}
        result = merge_metadata(existing, new)
        assert result == {"key": "value"}
    
    def test_merge_different_values_creates_list(self):
        """Test that different values for same key creates a list."""
        existing = {"key": "value1"}
        new = {"key": "value2"}
        result = merge_metadata(existing, new)
        assert result["key"] == ["value1", "value2"]
    
    def test_merge_into_existing_list(self):
        """Test merging into an existing list."""
        existing = {"key": ["value1", "value2"]}
        new = {"key": "value3"}
        result = merge_metadata(existing, new)
        assert result["key"] == ["value1", "value2", "value3"]
    
    def test_merge_duplicate_in_list(self):
        """Test that duplicates are not added to list."""
        existing = {"key": ["value1", "value2"]}
        new = {"key": "value1"}
        result = merge_metadata(existing, new)
        assert result["key"] == ["value1", "value2"]
    
    def test_merge_complex_metadata(self):
        """Test merging complex metadata structures."""
        existing = {
            "source": "GlyGen",
            "confidence": 0.9,
            "tags": ["enzyme", "human"]
        }
        new = {
            "source": "UniProt",
            "confidence": 0.95,
            "organism": "Homo sapiens"
        }
        result = merge_metadata(existing, new)
        assert result["source"] == ["GlyGen", "UniProt"]
        assert result["confidence"] == [0.9, 0.95]
        assert result["tags"] == ["enzyme", "human"]
        assert result["organism"] == "Homo sapiens"


class TestNodeAccumulator:
    """Tests for NodeAccumulator class."""
    
    def test_add_single_node(self):
        """Test adding a single node."""
        acc = NodeAccumulator()
        acc.add("N001", "enzyme", "Test Enzyme", {"source": "test"})
        
        df = acc.to_dataframe()
        assert len(df) == 1
        assert df.iloc[0]["node_id"] == "N001"
        assert df.iloc[0]["node_type"] == "enzyme"
        assert df.iloc[0]["label"] == "Test Enzyme"
    
    def test_add_multiple_unique_nodes(self):
        """Test adding multiple unique nodes."""
        acc = NodeAccumulator()
        acc.add("N001", "enzyme", "Enzyme 1", {})
        acc.add("N002", "protein", "Protein 1", {})
        acc.add("N003", "glycan", "Glycan 1", {})
        
        df = acc.to_dataframe()
        assert len(df) == 3
    
    def test_duplicate_node_merges_metadata(self):
        """Test that duplicate nodes merge metadata."""
        acc = NodeAccumulator()
        acc.add("N001", "enzyme", "Enzyme 1", {"source": "GlyGen"})
        acc.add("N001", "enzyme", "Enzyme 1", {"source": "UniProt"})
        
        df = acc.to_dataframe()
        assert len(df) == 1
        
        metadata = json.loads(df.iloc[0]["metadata"])
        assert metadata["source"] == ["GlyGen", "UniProt"]
    
    def test_duplicate_count_tracking(self):
        """Test that duplicate count is tracked."""
        acc = NodeAccumulator()
        acc.add("N001", "enzyme", "Enzyme 1", {})
        acc.add("N001", "enzyme", "Enzyme 1", {})
        acc.add("N001", "enzyme", "Enzyme 1", {})
        
        assert acc.duplicate_count == 2
    
    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        acc = NodeAccumulator()
        acc.add("N001", "enzyme", "Enzyme 1", {"source": "test"})
        acc.add("N002", "protein", "Protein 1", {})
        
        df = acc.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["node_id", "node_type", "label", "metadata"]
    
    def test_empty_accumulator(self):
        """Test empty accumulator."""
        acc = NodeAccumulator()
        df = acc.to_dataframe()
        assert len(df) == 0
        assert acc.duplicate_count == 0
    
    def test_metadata_serialization(self):
        """Test that metadata is properly serialized to JSON."""
        acc = NodeAccumulator()
        acc.add("N001", "enzyme", "Enzyme 1", {"key": "value", "num": 42})
        
        df = acc.to_dataframe()
        metadata = json.loads(df.iloc[0]["metadata"])
        assert metadata["key"] == "value"
        assert metadata["num"] == 42


class TestEdgeAccumulator:
    """Tests for EdgeAccumulator class."""
    
    def test_add_single_edge(self):
        """Test adding a single edge."""
        acc = EdgeAccumulator()
        acc.add("N001", "N002", "inhibits", {"source": "test"})
        
        df = acc.to_dataframe()
        assert len(df) == 1
        assert df.iloc[0]["source_id"] == "N001"
        assert df.iloc[0]["target_id"] == "N002"
        assert df.iloc[0]["relation"] == "inhibits"
    
    def test_add_multiple_unique_edges(self):
        """Test adding multiple unique edges."""
        acc = EdgeAccumulator()
        acc.add("N001", "N002", "inhibits", {})
        acc.add("N002", "N003", "has_glycan", {})
        acc.add("N003", "N004", "associated_with_disease", {})
        
        df = acc.to_dataframe()
        assert len(df) == 3
    
    def test_duplicate_edge_merges_metadata(self):
        """Test that duplicate edges merge metadata."""
        acc = EdgeAccumulator()
        acc.add("N001", "N002", "inhibits", {"IC50": "10nM"})
        acc.add("N001", "N002", "inhibits", {"Ki": "5nM"})
        
        df = acc.to_dataframe()
        assert len(df) == 1
        
        metadata = json.loads(df.iloc[0]["metadata"])
        assert metadata["IC50"] == "10nM"
        assert metadata["Ki"] == "5nM"
    
    def test_same_nodes_different_relation(self):
        """Test that same nodes with different relations are separate edges."""
        acc = EdgeAccumulator()
        acc.add("N001", "N002", "inhibits", {})
        acc.add("N001", "N002", "has_glycan", {})
        
        df = acc.to_dataframe()
        assert len(df) == 2
    
    def test_duplicate_count_tracking(self):
        """Test that duplicate count is tracked."""
        acc = EdgeAccumulator()
        acc.add("N001", "N002", "inhibits", {})
        acc.add("N001", "N002", "inhibits", {})
        acc.add("N001", "N002", "inhibits", {})
        
        assert acc.duplicate_count == 2
    
    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        acc = EdgeAccumulator()
        acc.add("N001", "N002", "inhibits", {"source": "test"})
        acc.add("N002", "N003", "has_glycan", {})
        
        df = acc.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["source_id", "target_id", "relation", "metadata"]
    
    def test_empty_accumulator(self):
        """Test empty accumulator."""
        acc = EdgeAccumulator()
        df = acc.to_dataframe()
        assert len(df) == 0
        assert acc.duplicate_count == 0
    
    def test_edge_key_generation(self):
        """Test that edge keys are generated correctly."""
        acc = EdgeAccumulator()
        
        acc.add("A", "B", "rel1", {})
        acc.add("B", "A", "rel1", {})
        
        df = acc.to_dataframe()
        assert len(df) == 2


class TestNodeEdgeIntegration:
    """Integration tests for node and edge accumulators."""
    
    def test_build_simple_graph(self):
        """Test building a simple graph with nodes and edges."""
        nodes = NodeAccumulator()
        edges = EdgeAccumulator()
        
        nodes.add("E001", "enzyme", "Enzyme A", {"ec": "2.4.1.1"})
        nodes.add("C001", "compound", "Compound X", {"chembl_id": "CHEMBL123"})
        nodes.add("G001", "glycan", "Glycan Y", {"glytoucan_ac": "G12345AB"})
        
        edges.add("C001", "E001", "inhibits", {"IC50": "10nM"})
        edges.add("E001", "G001", "has_glycan", {})
        
        nodes_df = nodes.to_dataframe()
        edges_df = edges.to_dataframe()
        
        assert len(nodes_df) == 3
        assert len(edges_df) == 2
        
        node_ids = set(nodes_df["node_id"])
        assert all(edges_df["source_id"].isin(node_ids))
        assert all(edges_df["target_id"].isin(node_ids))
    
    def test_large_graph_deduplication(self):
        """Test deduplication with many duplicate entries."""
        nodes = NodeAccumulator()
        edges = EdgeAccumulator()
        
        for i in range(100):
            nodes.add("N001", "enzyme", "Enzyme 1", {"iteration": i})
            edges.add("N001", "N002", "inhibits", {"iteration": i})
        
        nodes.add("N002", "compound", "Compound 1", {})
        
        assert len(nodes.to_dataframe()) == 2
        assert len(edges.to_dataframe()) == 1
        assert nodes.duplicate_count == 99
        assert edges.duplicate_count == 99


class TestLocationNodesAndEdges:
    """Tests for cellular_location node/edge handling."""

    def test_add_location_node(self):
        """Test adding a cellular_location node."""
        acc = NodeAccumulator()
        acc.add(
            "LOC::golgi_apparatus",
            "cellular_location",
            "Golgi apparatus",
            {"source": "UniProt"},
        )

        df = acc.to_dataframe()
        assert len(df) == 1
        assert df.iloc[0]["node_id"] == "LOC::golgi_apparatus"
        assert df.iloc[0]["node_type"] == "cellular_location"

    def test_add_localized_in_edge(self):
        """Test adding a localized_in edge."""
        acc = EdgeAccumulator()
        acc.add("P12345", "LOC::golgi_apparatus", "localized_in", {"source": "UniProt"})

        df = acc.to_dataframe()
        assert len(df) == 1
        assert df.iloc[0]["relation"] == "localized_in"
        assert df.iloc[0]["source_id"] == "P12345"
        assert df.iloc[0]["target_id"] == "LOC::golgi_apparatus"

    def test_location_deduplication(self):
        """Test that duplicate location nodes are merged."""
        acc = NodeAccumulator()
        acc.add("LOC::golgi_apparatus", "cellular_location", "Golgi apparatus", {"source": "UniProt"})
        acc.add("LOC::golgi_apparatus", "cellular_location", "Golgi apparatus", {"source": "GlyGen"})

        df = acc.to_dataframe()
        assert len(df) == 1
        meta = json.loads(df.iloc[0]["metadata"])
        assert meta["source"] == ["UniProt", "GlyGen"]

    def test_location_edge_deduplication(self):
        """Test that duplicate localized_in edges are merged."""
        acc = EdgeAccumulator()
        acc.add("P12345", "LOC::golgi_apparatus", "localized_in", {"source": "UniProt"})
        acc.add("P12345", "LOC::golgi_apparatus", "localized_in", {"source": "GlyGen"})

        df = acc.to_dataframe()
        assert len(df) == 1

    def test_graph_with_location_nodes(self):
        """Test building a graph that includes location nodes and edges."""
        nodes = NodeAccumulator()
        edges = EdgeAccumulator()

        nodes.add("P12345", "protein", "Protein A", {"source": "UniProt"})
        nodes.add("LOC::golgi_apparatus", "cellular_location", "Golgi apparatus", {"source": "UniProt"})
        nodes.add("LOC::endoplasmic_reticulum", "cellular_location", "Endoplasmic reticulum", {"source": "UniProt"})

        edges.add("P12345", "LOC::golgi_apparatus", "localized_in", {"source": "UniProt"})
        edges.add("P12345", "LOC::endoplasmic_reticulum", "localized_in", {"source": "UniProt"})

        nodes_df = nodes.to_dataframe()
        edges_df = edges.to_dataframe()

        assert len(nodes_df) == 3
        assert len(edges_df) == 2

        node_ids = set(nodes_df["node_id"])
        assert all(edges_df["source_id"].isin(node_ids))
        assert all(edges_df["target_id"].isin(node_ids))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
