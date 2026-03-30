#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
kg_network_d3.py

D3.js-compatible JSON export for glycoMusubi knowledge graph visualization.

Generates JSON files that can be used with D3.js force-directed graph layouts.

Usage:
    python viz/kg_network_d3.py                    # Export full KG
    python viz/kg_network_d3.py --max-nodes 500   # Limit nodes
    python viz/kg_network_d3.py --output my_graph.json  # Custom output
    python viz/kg_network_d3.py --html             # Also generate HTML viewer
"""

import os
import sys
import json
import argparse
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

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

RELATION_COLORS = {
    'inhibits': '#E74C3C',
    'has_glycan': '#3498DB',
    'associated_with_disease': '#2ECC71',
    'has_variant': '#F39C12',
    'unknown': '#95A5A6',
}


@dataclass
class D3Node:
    """D3.js node representation."""
    id: str
    label: str
    group: str
    color: str
    size: int = 10
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class D3Link:
    """D3.js link representation."""
    source: str
    target: str
    relation: str
    color: str
    value: int = 1
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class D3Graph:
    """D3.js graph representation."""
    nodes: List[D3Node]
    links: List[D3Link]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'nodes': [asdict(n) for n in self.nodes],
            'links': [asdict(l) for l in self.links],
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class D3Exporter:
    """Export knowledge graph to D3.js-compatible JSON format."""
    
    def __init__(self):
        self.nodes_df: Optional[pd.DataFrame] = None
        self.edges_df: Optional[pd.DataFrame] = None
        self.graph: Optional[D3Graph] = None
    
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
    
    def _parse_metadata(self, metadata_str: Any) -> Optional[Dict[str, Any]]:
        """Parse metadata JSON string."""
        if pd.isna(metadata_str) or not metadata_str or metadata_str == '{}':
            return None
        
        try:
            if isinstance(metadata_str, str):
                return json.loads(metadata_str)
            return metadata_str
        except (json.JSONDecodeError, TypeError):
            return None
    
    def build_graph(self):
        """Build D3.js graph from loaded data."""
        if self.nodes_df is None or self.edges_df is None:
            raise ValueError("Load KG first with load_kg()")
        
        d3_nodes = []
        node_id_set = set()
        
        for _, row in self.nodes_df.iterrows():
            node_id = str(row['node_id'])
            node_type = row.get('node_type', 'unknown')
            label = str(row.get('label', row.get('node_name', node_id)))
            
            d3_node = D3Node(
                id=node_id,
                label=label,
                group=node_type,
                color=NODE_TYPE_COLORS.get(node_type, NODE_TYPE_COLORS['unknown']),
                size=10,
                metadata=self._parse_metadata(row.get('metadata')),
            )
            d3_nodes.append(d3_node)
            node_id_set.add(node_id)
        
        d3_links = []
        for _, row in self.edges_df.iterrows():
            source = str(row['source_id'])
            target = str(row['target_id'])
            relation = row.get('relation', 'unknown')
            
            if source in node_id_set and target in node_id_set:
                d3_link = D3Link(
                    source=source,
                    target=target,
                    relation=relation,
                    color=RELATION_COLORS.get(relation, RELATION_COLORS['unknown']),
                    value=1,
                    metadata=self._parse_metadata(row.get('metadata')),
                )
                d3_links.append(d3_link)
        
        self.graph = D3Graph(nodes=d3_nodes, links=d3_links)
        print(f"Built D3 graph with {len(d3_nodes)} nodes and {len(d3_links)} links")
    
    def save_json(self, output_path: Optional[str] = None) -> str:
        """Save graph to JSON file."""
        if self.graph is None:
            raise ValueError("Build graph first with build_graph()")
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = output_path or os.path.join(OUTPUT_DIR, "kg_d3_graph.json")
        
        with open(output_path, 'w') as f:
            f.write(self.graph.to_json())
        
        print(f"D3 JSON saved to: {output_path}")
        return output_path
    
    def save_html_viewer(self, output_path: Optional[str] = None, json_path: Optional[str] = None) -> str:
        """Generate an HTML file with embedded D3.js viewer."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = output_path or os.path.join(OUTPUT_DIR, "kg_d3_viewer.html")
        
        if json_path is None:
            json_data = self.graph.to_json() if self.graph else '{}'
            data_source = f"const graphData = {json_data};"
        else:
            rel_path = os.path.relpath(json_path, os.path.dirname(output_path))
            data_source = f"""
            let graphData = null;
            fetch('{rel_path}')
                .then(response => response.json())
                .then(data => {{
                    graphData = data;
                    initGraph();
                }});
            """
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>glycoMusubi Knowledge Graph Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
        }}
        #container {{
            width: 100vw;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        #header {{
            background-color: #333;
            color: white;
            padding: 10px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        #header h1 {{
            margin: 0;
            font-size: 1.5em;
        }}
        #controls {{
            display: flex;
            gap: 10px;
        }}
        #controls select, #controls button {{
            padding: 5px 10px;
            border-radius: 4px;
            border: none;
            cursor: pointer;
        }}
        #graph {{
            flex: 1;
            background-color: white;
        }}
        #tooltip {{
            position: absolute;
            background-color: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 10px;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
            max-width: 300px;
            display: none;
        }}
        #legend {{
            position: absolute;
            bottom: 20px;
            left: 20px;
            background-color: rgba(255, 255, 255, 0.9);
            padding: 10px;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 10px;
        }}
        .node {{
            cursor: pointer;
        }}
        .link {{
            stroke-opacity: 0.6;
        }}
    </style>
</head>
<body>
    <div id="container">
        <div id="header">
            <h1>glycoMusubi Knowledge Graph</h1>
            <div id="controls">
                <select id="nodeTypeFilter">
                    <option value="all">All Node Types</option>
                    <option value="enzyme">Enzymes</option>
                    <option value="protein">Proteins</option>
                    <option value="glycan">Glycans</option>
                    <option value="disease">Diseases</option>
                    <option value="variant">Variants</option>
                    <option value="compound">Compounds</option>
                </select>
                <button onclick="resetZoom()">Reset Zoom</button>
            </div>
        </div>
        <div id="graph"></div>
    </div>
    <div id="tooltip"></div>
    <div id="legend">
        <h4 style="margin: 0 0 10px 0;">Node Types</h4>
        <div class="legend-item"><div class="legend-color" style="background-color: #FF6B6B;"></div>Enzyme</div>
        <div class="legend-item"><div class="legend-color" style="background-color: #4ECDC4;"></div>Protein</div>
        <div class="legend-item"><div class="legend-color" style="background-color: #45B7D1;"></div>Glycan</div>
        <div class="legend-item"><div class="legend-color" style="background-color: #96CEB4;"></div>Disease</div>
        <div class="legend-item"><div class="legend-color" style="background-color: #FFEAA7;"></div>Variant</div>
        <div class="legend-item"><div class="legend-color" style="background-color: #DDA0DD;"></div>Compound</div>
    </div>
    
    <script>
        {data_source}
        
        let svg, simulation, node, link, zoom;
        
        function initGraph() {{
            if (!graphData) return;
            
            const container = document.getElementById('graph');
            const width = container.clientWidth;
            const height = container.clientHeight;
            
            svg = d3.select('#graph')
                .append('svg')
                .attr('width', width)
                .attr('height', height);
            
            const g = svg.append('g');
            
            zoom = d3.zoom()
                .scaleExtent([0.1, 10])
                .on('zoom', (event) => g.attr('transform', event.transform));
            
            svg.call(zoom);
            
            simulation = d3.forceSimulation(graphData.nodes)
                .force('link', d3.forceLink(graphData.links).id(d => d.id).distance(100))
                .force('charge', d3.forceManyBody().strength(-300))
                .force('center', d3.forceCenter(width / 2, height / 2))
                .force('collision', d3.forceCollide().radius(30));
            
            link = g.append('g')
                .selectAll('line')
                .data(graphData.links)
                .enter()
                .append('line')
                .attr('class', 'link')
                .attr('stroke', d => d.color)
                .attr('stroke-width', 2);
            
            node = g.append('g')
                .selectAll('circle')
                .data(graphData.nodes)
                .enter()
                .append('circle')
                .attr('class', 'node')
                .attr('r', d => d.size)
                .attr('fill', d => d.color)
                .call(d3.drag()
                    .on('start', dragstarted)
                    .on('drag', dragged)
                    .on('end', dragended))
                .on('mouseover', showTooltip)
                .on('mouseout', hideTooltip);
            
            const labels = g.append('g')
                .selectAll('text')
                .data(graphData.nodes)
                .enter()
                .append('text')
                .text(d => d.label.length > 15 ? d.label.substring(0, 15) + '...' : d.label)
                .attr('font-size', 10)
                .attr('dx', 12)
                .attr('dy', 4);
            
            simulation.on('tick', () => {{
                link
                    .attr('x1', d => d.source.x)
                    .attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x)
                    .attr('y2', d => d.target.y);
                
                node
                    .attr('cx', d => d.x)
                    .attr('cy', d => d.y);
                
                labels
                    .attr('x', d => d.x)
                    .attr('y', d => d.y);
            }});
        }}
        
        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}
        
        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}
        
        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}
        
        function showTooltip(event, d) {{
            const tooltip = document.getElementById('tooltip');
            let content = `<strong>${{d.label}}</strong><br>`;
            content += `Type: ${{d.group}}<br>`;
            content += `ID: ${{d.id}}<br>`;
            if (d.metadata) {{
                for (const [key, value] of Object.entries(d.metadata).slice(0, 5)) {{
                    content += `${{key}}: ${{value}}<br>`;
                }}
            }}
            tooltip.innerHTML = content;
            tooltip.style.left = (event.pageX + 10) + 'px';
            tooltip.style.top = (event.pageY + 10) + 'px';
            tooltip.style.display = 'block';
        }}
        
        function hideTooltip() {{
            document.getElementById('tooltip').style.display = 'none';
        }}
        
        function resetZoom() {{
            svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity);
        }}
        
        document.getElementById('nodeTypeFilter').addEventListener('change', function() {{
            const filterType = this.value;
            node.style('opacity', d => filterType === 'all' || d.group === filterType ? 1 : 0.1);
            link.style('opacity', d => {{
                if (filterType === 'all') return 0.6;
                const sourceMatch = graphData.nodes.find(n => n.id === d.source.id || n.id === d.source)?.group === filterType;
                const targetMatch = graphData.nodes.find(n => n.id === d.target.id || n.id === d.target)?.group === filterType;
                return sourceMatch || targetMatch ? 0.6 : 0.1;
            }});
        }});
        
        // Initialize if data is embedded
        if (graphData) {{
            initGraph();
        }}
    </script>
</body>
</html>
"""
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"D3 HTML viewer saved to: {output_path}")
        return output_path


def export_to_d3(
    nodes_path: Optional[str] = None,
    edges_path: Optional[str] = None,
    output_path: Optional[str] = None,
    max_nodes: Optional[int] = None,
    node_types: Optional[List[str]] = None,
    generate_html: bool = False,
) -> str:
    """
    Convenience function to export knowledge graph to D3.js format.
    
    Args:
        nodes_path: Path to nodes TSV file
        edges_path: Path to edges TSV file
        output_path: Path for output JSON file
        max_nodes: Maximum number of nodes to export
        node_types: Filter to specific node types
        generate_html: Also generate HTML viewer
    
    Returns:
        Path to the generated JSON file
    """
    exporter = D3Exporter()
    exporter.load_kg(nodes_path, edges_path)
    
    if node_types:
        exporter.filter_by_node_types(node_types)
    
    if max_nodes:
        exporter.sample_nodes(max_nodes)
    
    exporter.build_graph()
    json_path = exporter.save_json(output_path)
    
    if generate_html:
        html_path = output_path.replace('.json', '.html') if output_path else None
        exporter.save_html_viewer(html_path, json_path)
    
    return json_path


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Export glycoMusubi to D3.js-compatible JSON format",
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
        help="Output JSON file path (default: output/kg_d3_graph.json)"
    )
    
    parser.add_argument(
        '--max-nodes', '-m',
        type=int,
        help="Maximum number of nodes to export"
    )
    
    parser.add_argument(
        '--node-type', '-t',
        nargs='+',
        choices=['enzyme', 'protein', 'glycan', 'disease', 'variant', 'compound'],
        help="Filter to specific node types"
    )
    
    parser.add_argument(
        '--html',
        action='store_true',
        help="Also generate HTML viewer"
    )
    
    args = parser.parse_args()
    
    try:
        json_path = export_to_d3(
            nodes_path=args.nodes,
            edges_path=args.edges,
            output_path=args.output,
            max_nodes=args.max_nodes,
            node_types=args.node_type,
            generate_html=args.html,
        )
        print(f"\nExport complete: {json_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to run the pipeline first to generate the KG files.")
        sys.exit(1)


if __name__ == "__main__":
    main()
