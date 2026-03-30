#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
validate_kg.py

Validates knowledge graph integrity with comprehensive checks including:
- Orphan node detection
- Duplicate edge detection
- Cycle detection for directed relations
- Degree distribution analysis
- Connected components analysis
- Metadata completeness check
- Auto-fix functionality for common issues
"""

import os
import sys
import re
import json
import pandas as pd
import logging
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(__file__))

from utils.config_loader import load_config
from utils.sequence_tools import parse_site_id, validate_site_id, is_canonical_isoform

config = load_config()

KG_DIR = os.path.join(os.path.dirname(__file__), "..", config.directories.kg_output)
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", config.directories.logs)

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "validate_kg.log"))
    ]
)
logger = logging.getLogger(__name__)

GLYTOUCAN_ID_PATTERN = re.compile(r'^G\d{5}[A-Z]{2}$')
UNIPROT_AC_PATTERN = re.compile(r'^[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}(-\d+)?$')
CHEMBL_ID_PATTERN = re.compile(r'^CHEMBL\d+$')
SITE_ID_PATTERN = re.compile(r'^SITE::[A-Z0-9]+-?\d*::\d+::[A-Z]$')
LOCATION_ID_PATTERN = re.compile(r'^LOC::[a-z_]+$')

ACYCLIC_RELATIONS = {'inhibits', 'has_variant'}

SITE_VALIDATION_LOG = "validation_sites.txt"


@dataclass
class ValidationReport:
    """Container for validation results."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    fixes_applied: List[str] = field(default_factory=list)
    
    def add_error(self, message: str):
        self.errors.append(message)
        logger.error(message)
    
    def add_warning(self, message: str):
        self.warnings.append(message)
        logger.warning(message)
    
    def add_info(self, message: str):
        self.info.append(message)
        logger.info(message)
    
    def add_fix(self, message: str):
        self.fixes_applied.append(message)
        logger.info(f"FIX: {message}")
    
    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0
    
    def to_text(self) -> str:
        lines = [
            "=" * 60,
            "KNOWLEDGE GRAPH VALIDATION REPORT",
            f"Timestamp: {self.timestamp}",
            f"Status: {'PASSED' if self.is_valid else 'FAILED'}",
            "=" * 60,
            "",
        ]
        
        if self.errors:
            lines.append(f"ERRORS ({len(self.errors)}):")
            lines.append("-" * 40)
            for e in self.errors:
                lines.append(f"  [ERROR] {e}")
            lines.append("")
        
        if self.warnings:
            lines.append(f"WARNINGS ({len(self.warnings)}):")
            lines.append("-" * 40)
            for w in self.warnings:
                lines.append(f"  [WARN] {w}")
            lines.append("")
        
        if self.info:
            lines.append(f"INFO ({len(self.info)}):")
            lines.append("-" * 40)
            for i in self.info:
                lines.append(f"  [INFO] {i}")
            lines.append("")
        
        if self.statistics:
            lines.append("STATISTICS:")
            lines.append("-" * 40)
            for k, v in self.statistics.items():
                if isinstance(v, dict):
                    lines.append(f"  {k}:")
                    for sk, sv in v.items():
                        lines.append(f"    {sk}: {sv}")
                else:
                    lines.append(f"  {k}: {v}")
            lines.append("")
        
        if self.fixes_applied:
            lines.append(f"FIXES APPLIED ({len(self.fixes_applied)}):")
            lines.append("-" * 40)
            for f in self.fixes_applied:
                lines.append(f"  [FIX] {f}")
            lines.append("")
        
        return "\n".join(lines)


class KGValidator:
    """Comprehensive knowledge graph validator."""
    
    def __init__(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame):
        self.nodes_df = nodes_df
        self.edges_df = edges_df
        self.report = ValidationReport()
        
        self.node_ids = set(nodes_df["node_id"].astype(str)) if not nodes_df.empty else set()
        self.node_types = {}
        if not nodes_df.empty:
            for _, row in nodes_df.iterrows():
                self.node_types[str(row["node_id"])] = row.get("node_type", "unknown")
        
        self._build_adjacency()
    
    def _build_adjacency(self):
        """Build adjacency lists for graph analysis."""
        self.adjacency = defaultdict(list)
        self.reverse_adjacency = defaultdict(list)
        self.edges_by_relation = defaultdict(list)
        
        if self.edges_df.empty:
            return
        
        for _, row in self.edges_df.iterrows():
            source = str(row["source_id"])
            target = str(row["target_id"])
            relation = str(row.get("relation", "unknown"))
            
            self.adjacency[source].append((target, relation))
            self.reverse_adjacency[target].append((source, relation))
            self.edges_by_relation[relation].append((source, target))
    
    def check_orphan_nodes(self) -> Set[str]:
        """Find nodes not connected to any edges."""
        linked_ids = set()
        if not self.edges_df.empty:
            linked_ids = set(self.edges_df["source_id"].astype(str)).union(
                set(self.edges_df["target_id"].astype(str))
            )
        
        orphan_nodes = self.node_ids - linked_ids
        
        if orphan_nodes:
            self.report.add_warning(f"Found {len(orphan_nodes)} orphan nodes")
            self.report.statistics["orphan_nodes"] = {
                "count": len(orphan_nodes),
                "sample": list(orphan_nodes)[:10]
            }
        else:
            self.report.add_info("No orphan nodes found")
        
        return orphan_nodes
    
    def check_duplicate_edges(self) -> pd.DataFrame:
        """Find duplicate edges."""
        if self.edges_df.empty:
            return pd.DataFrame()
        
        duplicate_mask = self.edges_df.duplicated(subset=["source_id", "target_id", "relation"], keep=False)
        duplicates = self.edges_df[duplicate_mask]
        
        if len(duplicates) > 0:
            unique_dup_count = len(self.edges_df[self.edges_df.duplicated(subset=["source_id", "target_id", "relation"], keep='first')])
            self.report.add_warning(f"Found {unique_dup_count} duplicate edges")
            self.report.statistics["duplicate_edges"] = unique_dup_count
        else:
            self.report.add_info("No duplicate edges found")
        
        return duplicates
    
    def check_invalid_references(self) -> Tuple[Set[str], Set[str]]:
        """Find edges referencing non-existent nodes."""
        invalid_sources = set()
        invalid_targets = set()
        
        if self.edges_df.empty:
            return invalid_sources, invalid_targets
        
        source_ids = set(self.edges_df["source_id"].astype(str))
        target_ids = set(self.edges_df["target_id"].astype(str))
        
        invalid_sources = source_ids - self.node_ids
        invalid_targets = target_ids - self.node_ids
        
        if invalid_sources:
            self.report.add_error(f"Found {len(invalid_sources)} edges with invalid source IDs")
            self.report.statistics["invalid_source_ids"] = list(invalid_sources)[:10]
        
        if invalid_targets:
            self.report.add_error(f"Found {len(invalid_targets)} edges with invalid target IDs")
            self.report.statistics["invalid_target_ids"] = list(invalid_targets)[:10]
        
        if not invalid_sources and not invalid_targets:
            self.report.add_info("All edge references are valid")
        
        return invalid_sources, invalid_targets
    
    def check_self_loops(self) -> List[Tuple[str, str]]:
        """Find edges where source equals target."""
        self_loops = []
        
        if self.edges_df.empty:
            return self_loops
        
        for _, row in self.edges_df.iterrows():
            if str(row["source_id"]) == str(row["target_id"]):
                self_loops.append((str(row["source_id"]), str(row.get("relation", "unknown"))))
        
        if self_loops:
            self.report.add_warning(f"Found {len(self_loops)} self-loop edges")
            self.report.statistics["self_loops"] = self_loops[:10]
        else:
            self.report.add_info("No self-loop edges found")
        
        return self_loops
    
    def detect_cycles(self, relation: str) -> List[List[str]]:
        """Detect cycles in edges of a specific relation type using DFS."""
        cycles = []
        edges = self.edges_by_relation.get(relation, [])
        
        if not edges:
            return cycles
        
        adj = defaultdict(list)
        for source, target in edges:
            adj[source].append(target)
        
        visited = set()
        rec_stack = set()
        path = []
        
        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in adj[node]:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:] + [neighbor])
                    return True
            
            path.pop()
            rec_stack.remove(node)
            return False
        
        all_nodes = set()
        for source, target in edges:
            all_nodes.add(source)
            all_nodes.add(target)
        
        for node in all_nodes:
            if node not in visited:
                dfs(node)
        
        return cycles
    
    def check_acyclic_relations(self):
        """Check that acyclic relations don't contain cycles."""
        for relation in ACYCLIC_RELATIONS:
            cycles = self.detect_cycles(relation)
            if cycles:
                self.report.add_error(f"Found {len(cycles)} cycles in '{relation}' relation (should be acyclic)")
                self.report.statistics[f"cycles_{relation}"] = [c[:5] for c in cycles[:5]]
            else:
                self.report.add_info(f"No cycles found in '{relation}' relation")
    
    def compute_degree_distribution(self) -> Dict[str, Dict[str, int]]:
        """Compute in-degree and out-degree distribution."""
        in_degree = defaultdict(int)
        out_degree = defaultdict(int)
        
        if self.edges_df.empty:
            return {"in_degree": {}, "out_degree": {}}
        
        for _, row in self.edges_df.iterrows():
            source = str(row["source_id"])
            target = str(row["target_id"])
            out_degree[source] += 1
            in_degree[target] += 1
        
        in_values = list(in_degree.values()) if in_degree else [0]
        out_values = list(out_degree.values()) if out_degree else [0]
        
        stats = {
            "in_degree": {
                "min": min(in_values),
                "max": max(in_values),
                "avg": sum(in_values) / len(in_values) if in_values else 0,
                "nodes_with_zero": len(self.node_ids - set(in_degree.keys()))
            },
            "out_degree": {
                "min": min(out_values),
                "max": max(out_values),
                "avg": sum(out_values) / len(out_values) if out_values else 0,
                "nodes_with_zero": len(self.node_ids - set(out_degree.keys()))
            }
        }
        
        self.report.statistics["degree_distribution"] = stats
        self.report.add_info(f"Degree distribution computed: max_in={stats['in_degree']['max']}, max_out={stats['out_degree']['max']}")
        
        return stats
    
    def find_connected_components(self) -> List[Set[str]]:
        """Find weakly connected components using union-find."""
        parent = {node: node for node in self.node_ids}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        if not self.edges_df.empty:
            for _, row in self.edges_df.iterrows():
                source = str(row["source_id"])
                target = str(row["target_id"])
                if source in parent and target in parent:
                    union(source, target)
        
        components = defaultdict(set)
        for node in self.node_ids:
            components[find(node)].add(node)
        
        component_list = list(components.values())
        component_sizes = sorted([len(c) for c in component_list], reverse=True)
        
        self.report.statistics["connected_components"] = {
            "count": len(component_list),
            "largest_size": component_sizes[0] if component_sizes else 0,
            "sizes": component_sizes[:10]
        }
        
        if len(component_list) > 1:
            self.report.add_warning(f"Graph has {len(component_list)} disconnected components")
        else:
            self.report.add_info("Graph is fully connected (1 component)")
        
        return component_list
    
    def check_metadata_completeness(self):
        """Check metadata field completeness."""
        if self.nodes_df.empty:
            return
        
        nodes_with_metadata = 0
        nodes_with_empty_metadata = 0
        
        for _, row in self.nodes_df.iterrows():
            metadata = row.get("metadata", "")
            if pd.notna(metadata) and metadata and metadata != "{}":
                nodes_with_metadata += 1
            else:
                nodes_with_empty_metadata += 1
        
        completeness = nodes_with_metadata / len(self.nodes_df) * 100 if len(self.nodes_df) > 0 else 0
        
        self.report.statistics["metadata_completeness"] = {
            "nodes_with_metadata": nodes_with_metadata,
            "nodes_without_metadata": nodes_with_empty_metadata,
            "completeness_percent": round(completeness, 2)
        }
        
        if completeness < 50:
            self.report.add_warning(f"Low metadata completeness: {completeness:.1f}%")
        else:
            self.report.add_info(f"Metadata completeness: {completeness:.1f}%")
    
    def check_id_formats(self):
        """Validate ID formats for different node types."""
        if self.nodes_df.empty:
            return
        
        invalid_ids = {"glycan": [], "enzyme": [], "protein": [], "compound": [], "cellular_location": []}

        for _, row in self.nodes_df.iterrows():
            node_id = str(row["node_id"])
            node_type = row.get("node_type", "")

            if node_type == "glycan":
                if not GLYTOUCAN_ID_PATTERN.match(node_id):
                    invalid_ids["glycan"].append(node_id)
            elif node_type in ("enzyme", "protein"):
                if not UNIPROT_AC_PATTERN.match(node_id):
                    invalid_ids[node_type].append(node_id)
            elif node_type == "compound":
                if not CHEMBL_ID_PATTERN.match(node_id):
                    invalid_ids["compound"].append(node_id)
            elif node_type == "cellular_location":
                if not LOCATION_ID_PATTERN.match(node_id):
                    invalid_ids["cellular_location"].append(node_id)
        
        for node_type, ids in invalid_ids.items():
            if ids:
                self.report.add_warning(f"Found {len(ids)} invalid {node_type} IDs")
                self.report.statistics[f"invalid_{node_type}_ids"] = ids[:10]
    
    def compute_statistics(self):
        """Compute general graph statistics."""
        self.report.statistics["total_nodes"] = len(self.nodes_df)
        self.report.statistics["total_edges"] = len(self.edges_df)
        
        if not self.nodes_df.empty:
            node_type_counts = self.nodes_df["node_type"].value_counts().to_dict()
            self.report.statistics["nodes_by_type"] = node_type_counts
        
        if not self.edges_df.empty:
            relation_counts = self.edges_df["relation"].value_counts().to_dict()
            self.report.statistics["edges_by_relation"] = relation_counts
        
        if len(self.nodes_df) > 0 and len(self.edges_df) > 0:
            density = len(self.edges_df) / (len(self.nodes_df) * (len(self.nodes_df) - 1))
            self.report.statistics["graph_density"] = round(density, 6)
    
    def validate_site_nodes(self) -> List[str]:
        """
        Validate site nodes for proper format and consistency.
        
        Checks:
        - Site ID format (SITE::<UniProtID>::<Position>::<Residue>)
        - Residue/position consistency
        - Canonical isoform usage
        
        Returns:
            List of validation warnings for site nodes
        """
        site_warnings = []
        
        if self.nodes_df.empty:
            return site_warnings
        
        site_nodes = self.nodes_df[self.nodes_df["node_type"] == "site"]
        
        if site_nodes.empty:
            self.report.add_info("No site nodes found in KG")
            return site_warnings
        
        invalid_format = []
        non_canonical = []
        invalid_residue = []
        
        for _, row in site_nodes.iterrows():
            node_id = str(row["node_id"])
            
            if not validate_site_id(node_id):
                invalid_format.append(node_id)
                continue
            
            parsed = parse_site_id(node_id)
            if parsed is None:
                invalid_format.append(node_id)
                continue
            
            uniprot_id, position, residue = parsed
            
            if not is_canonical_isoform(uniprot_id):
                non_canonical.append(node_id)
            
            if residue not in "ACDEFGHIKLMNPQRSTVWY":
                invalid_residue.append(node_id)
        
        if invalid_format:
            msg = f"Found {len(invalid_format)} site nodes with invalid ID format"
            site_warnings.append(msg)
            self.report.add_warning(msg)
            self.report.statistics["invalid_site_format"] = invalid_format[:10]
        
        if non_canonical:
            msg = f"Found {len(non_canonical)} site nodes with non-canonical isoform IDs"
            site_warnings.append(msg)
            self.report.add_warning(msg)
            self.report.statistics["non_canonical_sites"] = non_canonical[:10]
        
        if invalid_residue:
            msg = f"Found {len(invalid_residue)} site nodes with invalid residue codes"
            site_warnings.append(msg)
            self.report.add_warning(msg)
            self.report.statistics["invalid_residue_sites"] = invalid_residue[:10]
        
        if not site_warnings:
            self.report.add_info(f"All {len(site_nodes)} site nodes have valid format")
        
        self.report.statistics["total_site_nodes"] = len(site_nodes)
        
        return site_warnings
    
    def validate_site_protein_mapping(self) -> List[str]:
        """
        Validate that all site nodes map to existing proteins.
        
        Checks:
        - Each site's UniProt ID exists as a protein/enzyme node
        - has_site edges connect valid proteins to valid sites
        
        Returns:
            List of validation warnings for site-protein mapping
        """
        mapping_warnings = []
        
        if self.nodes_df.empty or self.edges_df.empty:
            return mapping_warnings
        
        site_nodes = self.nodes_df[self.nodes_df["node_type"] == "site"]
        
        if site_nodes.empty:
            return mapping_warnings
        
        protein_ids = set()
        for _, row in self.nodes_df.iterrows():
            node_type = row.get("node_type", "")
            if node_type in ("protein", "enzyme"):
                protein_ids.add(str(row["node_id"]))
        
        unmapped_sites = []
        
        for _, row in site_nodes.iterrows():
            node_id = str(row["node_id"])
            parsed = parse_site_id(node_id)
            
            if parsed is None:
                continue
            
            uniprot_id, _, _ = parsed
            base_id = uniprot_id.split('-')[0] if '-' in uniprot_id else uniprot_id
            
            found = (
                uniprot_id in protein_ids or
                base_id in protein_ids or
                f"{base_id}-1" in protein_ids
            )
            
            if not found:
                unmapped_sites.append(node_id)
        
        if unmapped_sites:
            msg = f"Found {len(unmapped_sites)} site nodes without matching protein nodes"
            mapping_warnings.append(msg)
            self.report.add_warning(msg)
            self.report.statistics["unmapped_sites"] = unmapped_sites[:10]
        else:
            self.report.add_info("All site nodes map to existing proteins")
        
        return mapping_warnings
    
    def validate_ptm_crosstalk_edges(self) -> List[str]:
        """
        Validate PTM crosstalk edges.
        
        Checks:
        - Both ends of ptm_crosstalk edges exist as site nodes
        - Edge metadata contains required fields
        
        Returns:
            List of validation warnings for PTM crosstalk edges
        """
        edge_warnings = []
        
        if self.edges_df.empty:
            return edge_warnings
        
        crosstalk_edges = self.edges_df[self.edges_df["relation"] == "ptm_crosstalk"]
        
        if crosstalk_edges.empty:
            self.report.add_info("No PTM crosstalk edges found in KG")
            return edge_warnings
        
        site_ids = set()
        if not self.nodes_df.empty:
            site_nodes = self.nodes_df[self.nodes_df["node_type"] == "site"]
            site_ids = set(site_nodes["node_id"].astype(str))
        
        invalid_source = []
        invalid_target = []
        
        for _, row in crosstalk_edges.iterrows():
            source_id = str(row["source_id"])
            target_id = str(row["target_id"])
            
            if source_id not in site_ids:
                invalid_source.append(source_id)
            
            if target_id not in site_ids:
                invalid_target.append(target_id)
        
        if invalid_source:
            msg = f"Found {len(invalid_source)} ptm_crosstalk edges with invalid source site"
            edge_warnings.append(msg)
            self.report.add_warning(msg)
            self.report.statistics["invalid_crosstalk_sources"] = invalid_source[:10]
        
        if invalid_target:
            msg = f"Found {len(invalid_target)} ptm_crosstalk edges with invalid target site"
            edge_warnings.append(msg)
            self.report.add_warning(msg)
            self.report.statistics["invalid_crosstalk_targets"] = invalid_target[:10]
        
        if not edge_warnings:
            self.report.add_info(f"All {len(crosstalk_edges)} PTM crosstalk edges are valid")
        
        self.report.statistics["total_ptm_crosstalk_edges"] = len(crosstalk_edges)
        
        return edge_warnings
    
    def validate_has_site_edges(self) -> List[str]:
        """
        Validate has_site edges (protein -> site).
        
        Checks:
        - Source is a protein/enzyme node
        - Target is a site node
        
        Returns:
            List of validation warnings for has_site edges
        """
        edge_warnings = []
        
        if self.edges_df.empty:
            return edge_warnings
        
        has_site_edges = self.edges_df[self.edges_df["relation"] == "has_site"]
        
        if has_site_edges.empty:
            self.report.add_info("No has_site edges found in KG")
            return edge_warnings
        
        protein_ids = set()
        site_ids = set()
        
        if not self.nodes_df.empty:
            for _, row in self.nodes_df.iterrows():
                node_type = row.get("node_type", "")
                node_id = str(row["node_id"])
                
                if node_type in ("protein", "enzyme"):
                    protein_ids.add(node_id)
                elif node_type == "site":
                    site_ids.add(node_id)
        
        invalid_source = []
        invalid_target = []
        
        for _, row in has_site_edges.iterrows():
            source_id = str(row["source_id"])
            target_id = str(row["target_id"])
            
            if source_id not in protein_ids:
                invalid_source.append(source_id)
            
            if target_id not in site_ids:
                invalid_target.append(target_id)
        
        if invalid_source:
            msg = f"Found {len(invalid_source)} has_site edges with non-protein source"
            edge_warnings.append(msg)
            self.report.add_warning(msg)
            self.report.statistics["invalid_has_site_sources"] = invalid_source[:10]
        
        if invalid_target:
            msg = f"Found {len(invalid_target)} has_site edges with non-site target"
            edge_warnings.append(msg)
            self.report.add_warning(msg)
            self.report.statistics["invalid_has_site_targets"] = invalid_target[:10]
        
        if not edge_warnings:
            self.report.add_info(f"All {len(has_site_edges)} has_site edges are valid")
        
        self.report.statistics["total_has_site_edges"] = len(has_site_edges)
        
        return edge_warnings
    
    def validate_localized_in_edges(self) -> List[str]:
        """
        Validate localized_in edges (protein/enzyme -> cellular_location).

        Checks:
        - Source is a protein or enzyme node
        - Target is a cellular_location node with valid LOC:: ID

        Returns:
            List of validation warnings for localized_in edges
        """
        edge_warnings = []

        if self.edges_df.empty:
            return edge_warnings

        loc_edges = self.edges_df[self.edges_df["relation"] == "localized_in"]

        if loc_edges.empty:
            self.report.add_info("No localized_in edges found in KG")
            return edge_warnings

        protein_ids = set()
        location_ids = set()

        if not self.nodes_df.empty:
            for _, row in self.nodes_df.iterrows():
                node_type = row.get("node_type", "")
                node_id = str(row["node_id"])

                if node_type in ("protein", "enzyme"):
                    protein_ids.add(node_id)
                elif node_type == "cellular_location":
                    location_ids.add(node_id)

        invalid_source = []
        invalid_target = []

        for _, row in loc_edges.iterrows():
            source_id = str(row["source_id"])
            target_id = str(row["target_id"])

            if source_id not in protein_ids:
                invalid_source.append(source_id)

            if target_id not in location_ids:
                invalid_target.append(target_id)

        if invalid_source:
            msg = f"Found {len(invalid_source)} localized_in edges with non-protein/enzyme source"
            edge_warnings.append(msg)
            self.report.add_warning(msg)
            self.report.statistics["invalid_localized_in_sources"] = invalid_source[:10]

        if invalid_target:
            msg = f"Found {len(invalid_target)} localized_in edges with non-location target"
            edge_warnings.append(msg)
            self.report.add_warning(msg)
            self.report.statistics["invalid_localized_in_targets"] = invalid_target[:10]

        if not edge_warnings:
            self.report.add_info(f"All {len(loc_edges)} localized_in edges are valid")

        self.report.statistics["total_localized_in_edges"] = len(loc_edges)

        return edge_warnings

    def run_site_validation(self) -> List[str]:
        """
        Run all site-specific validation checks.
        
        Outputs warnings to logs/validation_sites.txt
        
        Returns:
            List of all site validation warnings
        """
        all_warnings = []
        
        logger.info("Running site-specific validation...")
        
        all_warnings.extend(self.validate_site_nodes())
        all_warnings.extend(self.validate_site_protein_mapping())
        all_warnings.extend(self.validate_has_site_edges())
        all_warnings.extend(self.validate_ptm_crosstalk_edges())
        
        site_log_path = os.path.join(LOG_DIR, SITE_VALIDATION_LOG)
        with open(site_log_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("SITE VALIDATION REPORT\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write("=" * 60 + "\n\n")
            
            if all_warnings:
                f.write(f"WARNINGS ({len(all_warnings)}):\n")
                f.write("-" * 40 + "\n")
                for w in all_warnings:
                    f.write(f"  [WARN] {w}\n")
            else:
                f.write("No site validation warnings.\n")
            
            f.write("\n")
            f.write("STATISTICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Total site nodes: {self.report.statistics.get('total_site_nodes', 0)}\n")
            f.write(f"  Total has_site edges: {self.report.statistics.get('total_has_site_edges', 0)}\n")
            f.write(f"  Total ptm_crosstalk edges: {self.report.statistics.get('total_ptm_crosstalk_edges', 0)}\n")
        
        logger.info(f"Site validation report written to {site_log_path}")
        
        return all_warnings
    
    def run_all_checks(self):
        """Run all validation checks."""
        logger.info("Starting comprehensive KG validation...")
        
        self.compute_statistics()
        self.check_orphan_nodes()
        self.check_duplicate_edges()
        self.check_invalid_references()
        self.check_self_loops()
        self.check_acyclic_relations()
        self.compute_degree_distribution()
        self.find_connected_components()
        self.check_metadata_completeness()
        self.check_id_formats()
        self.validate_localized_in_edges()
        self.run_site_validation()

        logger.info("Validation complete")
        return self.report


class KGAutoFixer:
    """Auto-fix common KG issues."""
    
    def __init__(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame, report: ValidationReport):
        self.nodes_df = nodes_df.copy()
        self.edges_df = edges_df.copy()
        self.report = report
    
    def remove_orphan_nodes(self) -> int:
        """Remove nodes not connected to any edges."""
        if self.edges_df.empty:
            return 0
        
        linked_ids = set(self.edges_df["source_id"].astype(str)).union(
            set(self.edges_df["target_id"].astype(str))
        )
        
        original_count = len(self.nodes_df)
        self.nodes_df = self.nodes_df[self.nodes_df["node_id"].astype(str).isin(linked_ids)]
        removed = original_count - len(self.nodes_df)
        
        if removed > 0:
            self.report.add_fix(f"Removed {removed} orphan nodes")
        
        return removed
    
    def remove_duplicate_edges(self) -> int:
        """Remove duplicate edges, keeping the first occurrence."""
        original_count = len(self.edges_df)
        self.edges_df = self.edges_df.drop_duplicates(subset=["source_id", "target_id", "relation"], keep='first')
        removed = original_count - len(self.edges_df)
        
        if removed > 0:
            self.report.add_fix(f"Removed {removed} duplicate edges")
        
        return removed
    
    def remove_self_loops(self) -> int:
        """Remove edges where source equals target."""
        original_count = len(self.edges_df)
        self.edges_df = self.edges_df[self.edges_df["source_id"].astype(str) != self.edges_df["target_id"].astype(str)]
        removed = original_count - len(self.edges_df)
        
        if removed > 0:
            self.report.add_fix(f"Removed {removed} self-loop edges")
        
        return removed
    
    def remove_invalid_references(self) -> int:
        """Remove edges with invalid node references."""
        if self.edges_df.empty:
            return 0
        
        node_ids = set(self.nodes_df["node_id"].astype(str))
        original_count = len(self.edges_df)
        
        valid_mask = (
            self.edges_df["source_id"].astype(str).isin(node_ids) &
            self.edges_df["target_id"].astype(str).isin(node_ids)
        )
        self.edges_df = self.edges_df[valid_mask]
        removed = original_count - len(self.edges_df)
        
        if removed > 0:
            self.report.add_fix(f"Removed {removed} edges with invalid references")
        
        return removed
    
    def apply_all_fixes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply all auto-fixes and return cleaned DataFrames."""
        logger.info("Applying auto-fixes...")
        
        self.remove_duplicate_edges()
        self.remove_self_loops()
        self.remove_invalid_references()
        self.remove_orphan_nodes()
        
        logger.info(f"Auto-fix complete: {len(self.nodes_df)} nodes, {len(self.edges_df)} edges remaining")
        return self.nodes_df, self.edges_df


def main(auto_fix: bool = False):
    """Main entry point for KG validation."""
    start_time = datetime.now()
    logger.info(f"Starting KG validation at {start_time}")
    
    nodes_path = os.path.join(KG_DIR, "nodes.tsv")
    edges_path = os.path.join(KG_DIR, "edges.tsv")
    report_path = os.path.join(LOG_DIR, "kg_validation_report.txt")

    if not os.path.exists(nodes_path):
        logger.error(f"Nodes file not found: {nodes_path}")
        print("KG files not found. Build the KG before validation.")
        return
    
    if not os.path.exists(edges_path):
        logger.error(f"Edges file not found: {edges_path}")
        print("KG files not found. Build the KG before validation.")
        return

    nodes_df = pd.read_csv(nodes_path, sep="\t")
    edges_df = pd.read_csv(edges_path, sep="\t")
    
    logger.info(f"Loaded {len(nodes_df)} nodes and {len(edges_df)} edges")

    validator = KGValidator(nodes_df, edges_df)
    report = validator.run_all_checks()
    
    if auto_fix:
        fixer = KGAutoFixer(nodes_df, edges_df, report)
        fixed_nodes, fixed_edges = fixer.apply_all_fixes()
        
        fixed_nodes_path = os.path.join(KG_DIR, "nodes_fixed.tsv")
        fixed_edges_path = os.path.join(KG_DIR, "edges_fixed.tsv")
        
        fixed_nodes.to_csv(fixed_nodes_path, sep="\t", index=False)
        fixed_edges.to_csv(fixed_edges_path, sep="\t", index=False)
        
        logger.info(f"Fixed KG saved to {fixed_nodes_path} and {fixed_edges_path}")

    with open(report_path, "w") as f:
        f.write(report.to_text())

    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Validation completed at {end_time} (duration: {duration})")
    
    print(f"\nValidation report written to {report_path}")
    print(f"Status: {'PASSED' if report.is_valid else 'FAILED'}")
    print(f"Errors: {len(report.errors)}, Warnings: {len(report.warnings)}")
    
    if report.statistics:
        print(f"\nStatistics:")
        print(f"  Total nodes: {report.statistics.get('total_nodes', 0)}")
        print(f"  Total edges: {report.statistics.get('total_edges', 0)}")
        if 'connected_components' in report.statistics:
            print(f"  Connected components: {report.statistics['connected_components']['count']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate glycoMusubi knowledge graph")
    parser.add_argument("--auto-fix", action="store_true", help="Apply automatic fixes for common issues")
    args = parser.parse_args()
    
    main(auto_fix=args.auto_fix)
