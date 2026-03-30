#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
kg_network_pyvis.py

Interactive network visualization for glycoMusubi using PyVis.

Usage:
    python viz/kg_network_pyvis.py                    # Visualize full KG
    python viz/kg_network_pyvis.py --max-nodes 500   # Limit nodes
    python viz/kg_network_pyvis.py --node-type enzyme glycan  # Filter by type
    python viz/kg_network_pyvis.py --output my_graph.html     # Custom output
"""

import os
import sys
import json
import argparse
import pandas as pd
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False
    print("Warning: pyvis not installed. Install with: pip install pyvis")

KG_DIR = os.path.join(os.path.dirname(__file__), "..", "kg")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")

NODE_TYPE_COLORS = {
    'enzyme': '#FF6B6B',
    'protein': '#4ECDC4',
    'glycan': '#45B7D1',
    'disease': '#96CEB4',
    'variant': '#FFEAA7',
    'compound': '#DDA0DD',
    'unknown': '#808080',
}

NODE_TYPE_SHAPES = {
    'enzyme': 'diamond',
    'protein': 'dot',
    'glycan': 'triangle',
    'disease': 'square',
    'variant': 'star',
    'compound': 'hexagon',
    'unknown': 'dot',
}

RELATION_COLORS = {
    'inhibits': '#E74C3C',
    'has_glycan': '#3498DB',
    'associated_with_disease': '#2ECC71',
    'has_variant': '#F39C12',
    'unknown': '#95A5A6',
}


@dataclass
class VisualizationConfig:
    """Configuration for KG visualization."""
    width: str = "100%"
    height: str = "800px"
    bgcolor: str = "#ffffff"
    font_color: str = "#000000"
    directed: bool = True
    physics_enabled: bool = True
    node_size_base: int = 20
    edge_width: int = 2
    show_labels: bool = True
    max_label_length: int = 30


class KGVisualizer:
    """Interactive knowledge graph visualizer using PyVis."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        if not PYVIS_AVAILABLE:
            raise ImportError("pyvis is required. Install with: pip install pyvis")
        
        self.config = config or VisualizationConfig()
        self.nodes_df: Optional[pd.DataFrame] = None
        self.edges_df: Optional[pd.DataFrame] = None
        self.network: Optional[Network] = None
    
    def load_kg(self, nodes_path: Optional[str] = None, edges_path: Optional[str] = None):
        """Load knowledge graph from TSV files."""
        nodes_path = nodes_path or os.path.join(KG_DIR, "nodes.tsv")
        edges_path = edges_path or os.path.join(KG_DIR, "edges.tsv")
        
        if not os.path.exists(nodes_path):
            raise FileNotFoundError(f"Nodes file not found: {nodes_path}")
        if not os.path.exists(edges_path):
            raise FileNotFoundError(f"Edges file not found: {edges_path}")
        
        self.nodes_df = pd.read_csv(nodes_path, sep="\t")
        self.edges_df = pd.read_csv(edges_path, sep="\t")
        
        print(f"Loaded {len(self.nodes_df)} nodes and {len(self.edges_df)} edges")
    
    def filter_by_node_types(self, node_types: List[str]):
        """Filter graph to only include specified node types."""
        if self.nodes_df is None:
            raise ValueError("Load KG first with load_kg()")
        
        self.nodes_df = self.nodes_df[self.nodes_df['node_type'].isin(node_types)]
        
        node_ids = set(self.nodes_df['node_id'].astype(str))
        self.edges_df = self.edges_df[
            (self.edges_df['source_id'].astype(str).isin(node_ids)) &
            (self.edges_df['target_id'].astype(str).isin(node_ids))
        ]
        
        print(f"Filtered to {len(self.nodes_df)} nodes and {len(self.edges_df)} edges")
    
    def filter_by_relations(self, relations: List[str]):
        """Filter graph to only include specified relation types."""
        if self.edges_df is None:
            raise ValueError("Load KG first with load_kg()")
        
        self.edges_df = self.edges_df[self.edges_df['relation'].isin(relations)]
        
        edge_node_ids = set(self.edges_df['source_id'].astype(str)).union(
            set(self.edges_df['target_id'].astype(str))
        )
        self.nodes_df = self.nodes_df[self.nodes_df['node_id'].astype(str).isin(edge_node_ids)]
        
        print(f"Filtered to {len(self.nodes_df)} nodes and {len(self.edges_df)} edges")
    
    def sample_nodes(self, max_nodes: int, seed: int = 42):
        """Sample a subset of nodes for visualization."""
        if self.nodes_df is None:
            raise ValueError("Load KG first with load_kg()")
        
        if len(self.nodes_df) <= max_nodes:
            return
        
        self.nodes_df = self.nodes_df.sample(n=max_nodes, random_state=seed)
        
        node_ids = set(self.nodes_df['node_id'].astype(str))
        self.edges_df = self.edges_df[
            (self.edges_df['source_id'].astype(str).isin(node_ids)) &
            (self.edges_df['target_id'].astype(str).isin(node_ids))
        ]
        
        print(f"Sampled to {len(self.nodes_df)} nodes and {len(self.edges_df)} edges")
    
    def _truncate_label(self, label: str) -> str:
        """Truncate label to max length."""
        if len(label) > self.config.max_label_length:
            return label[:self.config.max_label_length - 3] + "..."
        return label
    
    def _get_node_title(self, row: pd.Series) -> str:
        """Generate hover tooltip for a node."""
        lines = [
            f"<b>ID:</b> {row['node_id']}",
            f"<b>Type:</b> {row.get('node_type', 'unknown')}",
            f"<b>Name:</b> {row.get('label', row.get('node_name', 'N/A'))}",
        ]
        
        metadata = row.get('metadata', '{}')
        if pd.notna(metadata) and metadata and metadata != '{}':
            try:
                meta_dict = json.loads(metadata) if isinstance(metadata, str) else metadata
                for k, v in list(meta_dict.items())[:5]:
                    lines.append(f"<b>{k}:</b> {v}")
            except (json.JSONDecodeError, TypeError):
                pass
        
        return "<br>".join(lines)
    
    def _get_edge_title(self, row: pd.Series) -> str:
        """Generate hover tooltip for an edge."""
        lines = [
            f"<b>Source:</b> {row['source_id']}",
            f"<b>Target:</b> {row['target_id']}",
            f"<b>Relation:</b> {row.get('relation', 'unknown')}",
        ]
        
        metadata = row.get('metadata', '{}')
        if pd.notna(metadata) and metadata and metadata != '{}':
            try:
                meta_dict = json.loads(metadata) if isinstance(metadata, str) else metadata
                for k, v in list(meta_dict.items())[:5]:
                    lines.append(f"<b>{k}:</b> {v}")
            except (json.JSONDecodeError, TypeError):
                pass
        
        return "<br>".join(lines)
    
    def build_network(self):
        """Build PyVis network from loaded data."""
        if self.nodes_df is None or self.edges_df is None:
            raise ValueError("Load KG first with load_kg()")
        
        self.network = Network(
            height=self.config.height,
            width=self.config.width,
            bgcolor=self.config.bgcolor,
            font_color=self.config.font_color,
            directed=self.config.directed,
        )
        
        for _, row in self.nodes_df.iterrows():
            node_id = str(row['node_id'])
            node_type = row.get('node_type', 'unknown')
            label = str(row.get('label', row.get('node_name', node_id)))
            
            self.network.add_node(
                node_id,
                label=self._truncate_label(label) if self.config.show_labels else "",
                title=self._get_node_title(row),
                color=NODE_TYPE_COLORS.get(node_type, NODE_TYPE_COLORS['unknown']),
                shape=NODE_TYPE_SHAPES.get(node_type, NODE_TYPE_SHAPES['unknown']),
                size=self.config.node_size_base,
                group=node_type,
            )
        
        for _, row in self.edges_df.iterrows():
            source = str(row['source_id'])
            target = str(row['target_id'])
            relation = row.get('relation', 'unknown')
            
            if source in [n['id'] for n in self.network.nodes] and \
               target in [n['id'] for n in self.network.nodes]:
                self.network.add_edge(
                    source,
                    target,
                    title=self._get_edge_title(row),
                    color=RELATION_COLORS.get(relation, RELATION_COLORS['unknown']),
                    width=self.config.edge_width,
                    label=relation if self.config.show_labels else "",
                )
        
        if self.config.physics_enabled:
            self.network.set_options("""
            var options = {
                "physics": {
                    "forceAtlas2Based": {
                        "gravitationalConstant": -50,
                        "centralGravity": 0.01,
                        "springLength": 100,
                        "springConstant": 0.08
                    },
                    "maxVelocity": 50,
                    "solver": "forceAtlas2Based",
                    "timestep": 0.35,
                    "stabilization": {"iterations": 150}
                },
                "nodes": {
                    "font": {"size": 12}
                },
                "edges": {
                    "smooth": {"type": "continuous"}
                }
            }
            """)
        
        print(f"Built network with {len(self.network.nodes)} nodes and {len(self.network.edges)} edges")
    
    def save(self, output_path: Optional[str] = None):
        """Save network visualization to HTML file."""
        if self.network is None:
            raise ValueError("Build network first with build_network()")
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = output_path or os.path.join(OUTPUT_DIR, "kg_visualization.html")
        
        self.network.save_graph(output_path)
        print(f"Visualization saved to: {output_path}")
        return output_path


def visualize_kg(
    nodes_path: Optional[str] = None,
    edges_path: Optional[str] = None,
    output_path: Optional[str] = None,
    max_nodes: Optional[int] = None,
    node_types: Optional[List[str]] = None,
    relations: Optional[List[str]] = None,
    config: Optional[VisualizationConfig] = None,
) -> str:
    """
    Convenience function to visualize a knowledge graph.
    
    Args:
        nodes_path: Path to nodes TSV file
        edges_path: Path to edges TSV file
        output_path: Path for output HTML file
        max_nodes: Maximum number of nodes to visualize
        node_types: Filter to specific node types
        relations: Filter to specific relation types
        config: Visualization configuration
    
    Returns:
        Path to the generated HTML file
    """
    visualizer = KGVisualizer(config)
    visualizer.load_kg(nodes_path, edges_path)
    
    if node_types:
        visualizer.filter_by_node_types(node_types)
    
    if relations:
        visualizer.filter_by_relations(relations)
    
    if max_nodes:
        visualizer.sample_nodes(max_nodes)
    
    visualizer.build_network()
    return visualizer.save(output_path)


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Visualize glycoMusubi knowledge graph using PyVis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--nodes', '-n',
        help="Path to nodes TSV file (default: kg/nodes.tsv)"
    )
    
    parser.add_argument(
        '--edges', '-e',
        help="Path to edges TSV file (default: kg/edges.tsv)"
    )
    
    parser.add_argument(
        '--output', '-o',
        help="Output HTML file path (default: output/kg_visualization.html)"
    )
    
    parser.add_argument(
        '--max-nodes', '-m',
        type=int,
        help="Maximum number of nodes to visualize"
    )
    
    parser.add_argument(
        '--node-type', '-t',
        nargs='+',
        choices=['enzyme', 'protein', 'glycan', 'disease', 'variant', 'compound'],
        help="Filter to specific node types"
    )
    
    parser.add_argument(
        '--relation', '-r',
        nargs='+',
        choices=['inhibits', 'has_glycan', 'associated_with_disease', 'has_variant'],
        help="Filter to specific relation types"
    )
    
    parser.add_argument(
        '--no-physics',
        action='store_true',
        help="Disable physics simulation"
    )
    
    parser.add_argument(
        '--no-labels',
        action='store_true',
        help="Hide node and edge labels"
    )
    
    args = parser.parse_args()
    
    if not PYVIS_AVAILABLE:
        print("Error: pyvis is required. Install with: pip install pyvis")
        sys.exit(1)
    
    config = VisualizationConfig(
        physics_enabled=not args.no_physics,
        show_labels=not args.no_labels,
    )
    
    try:
        output_path = visualize_kg(
            nodes_path=args.nodes,
            edges_path=args.edges,
            output_path=args.output,
            max_nodes=args.max_nodes,
            node_types=args.node_type,
            relations=args.relation,
            config=config,
        )
        print(f"\nVisualization complete: {output_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to run the pipeline first to generate the KG files.")
        sys.exit(1)


if __name__ == "__main__":
    main()
