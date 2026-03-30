"""Knowledge graph structural quality metrics.

Computes graph-theoretic properties of a heterogeneous knowledge graph
stored as a PyG ``HeteroData`` object.  Useful for sanity-checking data
ingestion and monitoring graph evolution across pipeline runs.
"""

from __future__ import annotations

import logging
import math
from typing import Dict

import torch
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)


def compute_kg_quality(data: HeteroData) -> Dict[str, float]:
    """Compute knowledge graph quality metrics.

    Parameters
    ----------
    data : HeteroData
        A PyG heterogeneous graph.

    Returns
    -------
    dict with keys:
        - ``num_nodes``: total node count
        - ``num_edges``: total edge count
        - ``num_node_types``: number of distinct node types
        - ``num_edge_types``: number of distinct edge types
        - ``graph_density``: 2|E| / (|V|(|V|-1)), 0 if |V| < 2
        - ``avg_degree``: 2|E| / |V|, 0 if |V| == 0
        - ``num_connected_components``: weakly connected components
        - ``clustering_coefficient``: average local clustering coefficient
        - ``per_type_coverage``: dict of {node_type: fraction_of_total}
        - ``relation_entropy``: Shannon entropy of relation type distribution
    """
    # --- Node / edge counts ---
    num_nodes = _total_nodes(data)
    num_edges = _total_edges(data)
    num_node_types = len(data.node_types)
    num_edge_types = len(data.edge_types)

    # --- Density & degree ---
    if num_nodes >= 2:
        graph_density = (2.0 * num_edges) / (num_nodes * (num_nodes - 1))
    else:
        graph_density = 0.0

    avg_degree = (2.0 * num_edges) / num_nodes if num_nodes > 0 else 0.0

    # --- Per-type coverage ---
    per_type_coverage: Dict[str, float] = {}
    if num_nodes > 0:
        for nt in data.node_types:
            n = data[nt].num_nodes or 0
            per_type_coverage[nt] = n / num_nodes
    else:
        for nt in data.node_types:
            per_type_coverage[nt] = 0.0

    # --- Relation entropy ---
    relation_entropy = _relation_entropy(data, num_edges)

    # --- Connected components & clustering (via networkx) ---
    num_cc, clustering = _graph_topology_metrics(data)

    return {
        "num_nodes": float(num_nodes),
        "num_edges": float(num_edges),
        "num_node_types": float(num_node_types),
        "num_edge_types": float(num_edge_types),
        "graph_density": graph_density,
        "avg_degree": avg_degree,
        "num_connected_components": float(num_cc),
        "clustering_coefficient": clustering,
        "per_type_coverage": per_type_coverage,
        "relation_entropy": relation_entropy,
    }


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _total_nodes(data: HeteroData) -> int:
    total = 0
    for nt in data.node_types:
        n = data[nt].num_nodes
        if n is not None:
            total += n
    return total


def _total_edges(data: HeteroData) -> int:
    total = 0
    for et in data.edge_types:
        ei = data[et].edge_index
        if ei is not None:
            total += ei.size(1)
    return total


def _relation_entropy(data: HeteroData, num_edges: int) -> float:
    """Shannon entropy of the relation-type distribution: -sum(p log p)."""
    if num_edges == 0:
        return 0.0

    entropy = 0.0
    for et in data.edge_types:
        ei = data[et].edge_index
        if ei is None:
            continue
        count = ei.size(1)
        if count > 0:
            p = count / num_edges
            entropy -= p * math.log(p)
    return entropy


def _graph_topology_metrics(data: HeteroData) -> tuple[int, float]:
    """Compute connected components and clustering coefficient via networkx.

    Builds a simple undirected graph by mapping all heterogeneous node
    IDs into a single flat ID space.  This is O(V + E) in memory.

    Returns
    -------
    num_connected_components : int
    clustering_coefficient : float
    """
    try:
        import networkx as nx
    except ImportError:
        logger.warning(
            "networkx not installed; connected components and clustering "
            "coefficient will be reported as 0."
        )
        return 0, 0.0

    G = nx.Graph()

    # Build a flat node-ID mapping: (node_type, local_id) -> global_id
    offset = 0
    offsets: Dict[str, int] = {}
    for nt in data.node_types:
        offsets[nt] = offset
        n = data[nt].num_nodes or 0
        G.add_nodes_from(range(offset, offset + n))
        offset += n

    # Add edges
    for et in data.edge_types:
        src_type, _, dst_type = et
        ei = data[et].edge_index
        if ei is None:
            continue
        src_offset = offsets[src_type]
        dst_offset = offsets[dst_type]
        src_ids = ei[0].cpu().numpy() + src_offset
        dst_ids = ei[1].cpu().numpy() + dst_offset
        G.add_edges_from(zip(src_ids, dst_ids))

    num_cc = nx.number_connected_components(G)
    clustering = nx.average_clustering(G) if G.number_of_nodes() > 0 else 0.0

    return num_cc, clustering
