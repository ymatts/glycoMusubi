"""Tests for KG quality metrics.

Covers:
  - compute_kg_quality returns all expected metric keys
  - Graph density is correct for known graphs
  - Average degree is correct
  - Connected components: 1 for connected, >1 for disconnected
  - Clustering coefficient in [0, 1]
  - Relation type entropy (uniform = max entropy)
  - Per-type coverage sums to 1.0
  - Works with multiple node/edge types
"""

from __future__ import annotations

import math

import pytest
import torch
from torch_geometric.data import HeteroData

from glycoMusubi.evaluation.kg_quality import compute_kg_quality


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_complete_graph() -> HeteroData:
    """Complete undirected graph K4: 4 nodes, 6 directed edges (= 6 undirected).

    For density computation: 2*6 / (4*3) = 1.0
    """
    data = HeteroData()
    data["node"].num_nodes = 4
    data["node"].x = torch.randn(4, 16)

    # All 6 undirected edges as directed pairs
    src = [0, 0, 0, 1, 1, 2]
    dst = [1, 2, 3, 2, 3, 3]
    data["node", "connects", "node"].edge_index = torch.tensor(
        [src, dst], dtype=torch.long
    )
    return data


def _make_disconnected_graph() -> HeteroData:
    """Graph with 2 connected components: {0,1} and {2,3}."""
    data = HeteroData()
    data["node"].num_nodes = 4
    data["node"].x = torch.randn(4, 16)

    data["node", "connects", "node"].edge_index = torch.tensor(
        [[0, 2], [1, 3]], dtype=torch.long
    )
    return data


def _make_heterogeneous_graph() -> HeteroData:
    """Heterogeneous graph with protein, glycan, and disease types."""
    data = HeteroData()
    data["protein"].num_nodes = 3
    data["protein"].x = torch.randn(3, 16)
    data["glycan"].num_nodes = 2
    data["glycan"].x = torch.randn(2, 16)
    data["disease"].num_nodes = 2
    data["disease"].x = torch.randn(2, 16)

    data["protein", "has_glycan", "glycan"].edge_index = torch.tensor(
        [[0, 1, 2], [0, 1, 0]], dtype=torch.long
    )
    data["protein", "associated_with", "disease"].edge_index = torch.tensor(
        [[0, 1], [0, 1]], dtype=torch.long
    )
    return data


def _make_uniform_relation_graph() -> HeteroData:
    """Graph with 2 relation types, each having exactly the same edge count."""
    data = HeteroData()
    data["a"].num_nodes = 4
    data["a"].x = torch.randn(4, 8)
    data["b"].num_nodes = 4
    data["b"].x = torch.randn(4, 8)

    data["a", "rel1", "b"].edge_index = torch.tensor(
        [[0, 1], [0, 1]], dtype=torch.long
    )
    data["a", "rel2", "b"].edge_index = torch.tensor(
        [[2, 3], [2, 3]], dtype=torch.long
    )
    return data


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestComputeKGQualityKeys:
    """Tests that compute_kg_quality returns all expected keys."""

    def test_all_expected_keys_present(self) -> None:
        data = _make_complete_graph()
        metrics = compute_kg_quality(data)

        expected_keys = {
            "num_nodes",
            "num_edges",
            "num_node_types",
            "num_edge_types",
            "graph_density",
            "avg_degree",
            "num_connected_components",
            "clustering_coefficient",
            "per_type_coverage",
            "relation_entropy",
        }
        assert expected_keys == set(metrics.keys())

    def test_values_are_numeric(self) -> None:
        data = _make_complete_graph()
        metrics = compute_kg_quality(data)

        for key, val in metrics.items():
            if key == "per_type_coverage":
                assert isinstance(val, dict)
                for v in val.values():
                    assert isinstance(v, float)
            else:
                assert isinstance(val, float), f"{key} is not float: {type(val)}"


class TestGraphDensity:
    """Tests for density computation."""

    def test_complete_graph_density(self) -> None:
        """K4 with 6 edges: density = 2*6 / (4*3) = 1.0."""
        data = _make_complete_graph()
        metrics = compute_kg_quality(data)
        assert pytest.approx(metrics["graph_density"], abs=1e-6) == 1.0

    def test_sparse_graph_density(self) -> None:
        """2 edges among 4 nodes: density = 2*2 / (4*3) = 1/3."""
        data = _make_disconnected_graph()
        metrics = compute_kg_quality(data)
        assert pytest.approx(metrics["graph_density"], abs=1e-6) == 1.0 / 3.0

    def test_single_node_density_zero(self) -> None:
        """Graph with 1 node has density 0."""
        data = HeteroData()
        data["node"].num_nodes = 1
        data["node"].x = torch.randn(1, 8)
        metrics = compute_kg_quality(data)
        assert metrics["graph_density"] == 0.0


class TestAvgDegree:
    """Tests for average degree."""

    def test_complete_graph_avg_degree(self) -> None:
        """K4: avg_degree = 2*6/4 = 3.0."""
        data = _make_complete_graph()
        metrics = compute_kg_quality(data)
        assert pytest.approx(metrics["avg_degree"], abs=1e-6) == 3.0

    def test_heterogeneous_avg_degree(self) -> None:
        """3+2+2=7 nodes, 3+2=5 edges: avg_degree = 2*5/7."""
        data = _make_heterogeneous_graph()
        metrics = compute_kg_quality(data)
        expected = 2.0 * 5.0 / 7.0
        assert pytest.approx(metrics["avg_degree"], abs=1e-6) == expected


class TestConnectedComponents:
    """Tests for connected components counting."""

    def test_connected_graph_one_component(self) -> None:
        """Complete graph has 1 connected component."""
        data = _make_complete_graph()
        metrics = compute_kg_quality(data)
        assert metrics["num_connected_components"] == 1.0

    def test_disconnected_graph_two_components(self) -> None:
        """Graph with 2 components: {0,1} and {2,3}."""
        data = _make_disconnected_graph()
        metrics = compute_kg_quality(data)
        assert metrics["num_connected_components"] == 2.0


class TestClusteringCoefficient:
    """Tests for clustering coefficient."""

    def test_clustering_in_valid_range(self) -> None:
        data = _make_complete_graph()
        metrics = compute_kg_quality(data)
        cc = metrics["clustering_coefficient"]
        assert 0.0 <= cc <= 1.0

    def test_complete_graph_clustering(self) -> None:
        """Complete graph K4 should have clustering coefficient 1.0."""
        data = _make_complete_graph()
        metrics = compute_kg_quality(data)
        assert pytest.approx(metrics["clustering_coefficient"], abs=1e-6) == 1.0

    def test_disconnected_clustering(self) -> None:
        """Disconnected graph with isolated edges has clustering 0."""
        data = _make_disconnected_graph()
        metrics = compute_kg_quality(data)
        assert pytest.approx(metrics["clustering_coefficient"], abs=1e-6) == 0.0


class TestRelationEntropy:
    """Tests for relation type entropy."""

    def test_uniform_distribution_max_entropy(self) -> None:
        """Uniform distribution over 2 types: entropy = ln(2)."""
        data = _make_uniform_relation_graph()
        metrics = compute_kg_quality(data)
        expected_entropy = math.log(2)
        assert pytest.approx(metrics["relation_entropy"], abs=1e-6) == expected_entropy

    def test_single_relation_zero_entropy(self) -> None:
        """Single relation type: entropy = -1*log(1) = 0."""
        data = _make_complete_graph()
        metrics = compute_kg_quality(data)
        assert pytest.approx(metrics["relation_entropy"], abs=1e-6) == 0.0

    def test_entropy_nonnegative(self) -> None:
        data = _make_heterogeneous_graph()
        metrics = compute_kg_quality(data)
        assert metrics["relation_entropy"] >= 0.0


class TestPerTypeCoverage:
    """Tests for per-type node coverage."""

    def test_single_type_coverage_is_one(self) -> None:
        """Single node type: coverage = 1.0."""
        data = _make_complete_graph()
        metrics = compute_kg_quality(data)
        coverage = metrics["per_type_coverage"]
        assert len(coverage) == 1
        assert pytest.approx(coverage["node"], abs=1e-6) == 1.0

    def test_multi_type_coverage_sums_to_one(self) -> None:
        """Multiple node types: coverages sum to 1.0."""
        data = _make_heterogeneous_graph()
        metrics = compute_kg_quality(data)
        coverage = metrics["per_type_coverage"]
        assert pytest.approx(sum(coverage.values()), abs=1e-6) == 1.0

    def test_coverage_values_match_fractions(self) -> None:
        """Verify individual coverage fractions: 3/7, 2/7, 2/7."""
        data = _make_heterogeneous_graph()
        metrics = compute_kg_quality(data)
        coverage = metrics["per_type_coverage"]
        assert pytest.approx(coverage["protein"], abs=1e-6) == 3.0 / 7.0
        assert pytest.approx(coverage["glycan"], abs=1e-6) == 2.0 / 7.0
        assert pytest.approx(coverage["disease"], abs=1e-6) == 2.0 / 7.0


class TestHeterogeneousGraph:
    """Tests that compute_kg_quality works with heterogeneous graphs."""

    def test_hetero_node_edge_counts(self) -> None:
        data = _make_heterogeneous_graph()
        metrics = compute_kg_quality(data)
        assert metrics["num_nodes"] == 7.0
        assert metrics["num_edges"] == 5.0
        assert metrics["num_node_types"] == 3.0
        assert metrics["num_edge_types"] == 2.0

    def test_hetero_metrics_all_finite(self) -> None:
        data = _make_heterogeneous_graph()
        metrics = compute_kg_quality(data)
        for key, val in metrics.items():
            if key == "per_type_coverage":
                for v in val.values():
                    assert math.isfinite(v)
            else:
                assert math.isfinite(val), f"{key} is not finite: {val}"
