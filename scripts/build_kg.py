#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_kg.py

Builds knowledge graph from cleaned data tables.
Includes proper node deduplication, metadata merging, and dtype-aware Parquet export.
"""

import os
import sys
import pandas as pd
import json
import pyarrow.parquet as pq
import pyarrow as pa
import yaml
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from utils.config_loader import load_config
from utils.parallel import get_worker_count, chunked_by_count, seed_worker
from utils.sequence_tools import create_site_id, parse_site_id

config = load_config()


def get_parallel_settings():
    """Get parallel processing settings from config and environment."""
    parallel_env = os.environ.get('GLYCO_KG_PARALLEL', '').lower()
    if parallel_env == 'true':
        parallel_enabled = True
    elif parallel_env == 'false':
        parallel_enabled = False
    else:
        parallel_enabled = config.parallel.enabled
    
    workers_env = os.environ.get('GLYCO_KG_WORKERS')
    if workers_env:
        workers = int(workers_env)
    else:
        workers = get_worker_count(None, config.parallel.workers)
    
    seed_env = os.environ.get('GLYCO_KG_PARALLEL_SEED')
    if seed_env:
        seed = int(seed_env)
    else:
        seed = config.parallel.seed
    
    return parallel_enabled, workers, seed

DATA_CLEAN_DIR = os.path.join(os.path.dirname(__file__), "..", config.directories.clean_data)
KG_DIR = os.path.join(os.path.dirname(__file__), "..", config.directories.kg_output)
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", config.directories.logs)
SCHEMA_DIR = os.path.join(os.path.dirname(__file__), "..", "schemas")

os.makedirs(KG_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "build_kg.log"))
    ]
)
logger = logging.getLogger(__name__)


def load_schema(schema_name: str) -> Dict[str, Any]:
    """Load a schema file from the schemas directory."""
    schema_path = os.path.join(SCHEMA_DIR, schema_name)
    if os.path.exists(schema_path):
        with open(schema_path, 'r') as f:
            return yaml.safe_load(f)
    logger.warning(f"Schema file not found: {schema_path}")
    return {}


def merge_metadata(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two metadata dictionaries, combining values for duplicate keys.
    
    Args:
        existing: Existing metadata dictionary.
        new: New metadata to merge.
    
    Returns:
        Merged metadata dictionary.
    """
    result = existing.copy()
    
    for key, value in new.items():
        if key not in result:
            result[key] = value
        elif result[key] == value:
            continue
        elif isinstance(result[key], list):
            if value not in result[key]:
                result[key].append(value)
        else:
            if result[key] != value:
                result[key] = [result[key], value]
    
    return result


class NodeAccumulator:
    """Accumulates nodes with deduplication and metadata merging."""
    
    def __init__(self):
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._duplicate_count = 0
    
    def add(
        self,
        node_id: str,
        node_type: str,
        label: str,
        metadata: Dict[str, Any],
        sequence: Optional[str] = None
    ):
        """Add a node, merging metadata if it already exists."""
        if node_id in self._nodes:
            existing = self._nodes[node_id]
            existing_meta = json.loads(existing.get('metadata', '{}'))
            merged_meta = merge_metadata(existing_meta, metadata)
            existing['metadata'] = json.dumps(merged_meta)
            if sequence:
                existing_seq = existing.get("sequence", "")
                if not existing_seq:
                    existing["sequence"] = sequence
            
            if existing.get('node_type') != node_type:
                logger.warning(f"Node {node_id} has conflicting types: {existing['node_type']} vs {node_type}")
            
            self._duplicate_count += 1
        else:
            self._nodes[node_id] = {
                'node_id': node_id,
                'node_type': node_type,
                'label': label,
                'metadata': json.dumps(metadata)
            }
            if sequence:
                self._nodes[node_id]["sequence"] = sequence
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert accumulated nodes to DataFrame."""
        if not self._nodes:
            return pd.DataFrame(columns=['node_id', 'node_type', 'label', 'metadata', 'sequence'])
        
        df = pd.DataFrame(list(self._nodes.values()))
        logger.info(f"Built {len(df)} unique nodes ({self._duplicate_count} duplicates merged)")
        return df
    
    @property
    def duplicate_count(self) -> int:
        return self._duplicate_count
    
    def get_raw_data(self) -> Dict[str, Dict[str, Any]]:
        """Get raw node data for merging from workers."""
        return self._nodes.copy()
    
    def merge_from_raw(self, raw_nodes: Dict[str, Dict[str, Any]]):
        """Merge raw node data from another accumulator."""
        for node_id, node_data in raw_nodes.items():
            if node_id in self._nodes:
                existing = self._nodes[node_id]
                existing_meta = json.loads(existing.get('metadata', '{}'))
                new_meta = json.loads(node_data.get('metadata', '{}'))
                merged_meta = merge_metadata(existing_meta, new_meta)
                existing['metadata'] = json.dumps(merged_meta)
                existing_seq = existing.get("sequence", "")
                new_seq = node_data.get("sequence", "")
                if new_seq and not existing_seq:
                    existing["sequence"] = new_seq
                self._duplicate_count += 1
            else:
                self._nodes[node_id] = node_data


class EdgeAccumulator:
    """Accumulates edges with deduplication."""
    
    def __init__(self):
        self._edges: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self._duplicate_count = 0
    
    def add(self, source_id: str, target_id: str, relation: str, metadata: Dict[str, Any]):
        """Add an edge, merging metadata if it already exists."""
        key = (source_id, target_id, relation)
        
        if key in self._edges:
            existing = self._edges[key]
            existing_meta = json.loads(existing.get('metadata', '{}'))
            merged_meta = merge_metadata(existing_meta, metadata)
            existing['metadata'] = json.dumps(merged_meta)
            self._duplicate_count += 1
        else:
            self._edges[key] = {
                'source_id': source_id,
                'target_id': target_id,
                'relation': relation,
                'metadata': json.dumps(metadata)
            }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert accumulated edges to DataFrame."""
        if not self._edges:
            return pd.DataFrame(columns=['source_id', 'target_id', 'relation', 'metadata'])
        
        df = pd.DataFrame(list(self._edges.values()))
        logger.info(f"Built {len(df)} unique edges ({self._duplicate_count} duplicates merged)")
        return df
    
    @property
    def duplicate_count(self) -> int:
        return self._duplicate_count
    
    def get_raw_data(self) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
        """Get raw edge data for merging from workers."""
        return self._edges.copy()
    
    def merge_from_raw(self, raw_edges: Dict[Tuple[str, str, str], Dict[str, Any]]):
        """Merge raw edge data from another accumulator."""
        for key, edge_data in raw_edges.items():
            if key in self._edges:
                existing = self._edges[key]
                existing_meta = json.loads(existing.get('metadata', '{}'))
                new_meta = json.loads(edge_data.get('metadata', '{}'))
                merged_meta = merge_metadata(existing_meta, new_meta)
                existing['metadata'] = json.dumps(merged_meta)
                self._duplicate_count += 1
            else:
                self._edges[key] = edge_data


# Worker functions for multiprocessing (must be top-level for pickling)

def _process_node_table_partition(args):
    """
    Worker function to process a partition of a node table.
    
    Args:
        args: Tuple of (table_name, rows_data, worker_id, seed)
    
    Returns:
        Dict with nodes data.
    """
    table_name, rows_data, worker_id, seed = args
    seed_worker(worker_id, seed)
    
    nodes = {}
    
    for row in rows_data:
        if table_name == "enzymes_clean":
            node_id = str(row["enzyme_id"])
            nodes[node_id] = {
                'node_id': node_id,
                'node_type': "enzyme",
                'label': str(row.get("gene_symbol", row["enzyme_id"])),
                'metadata': json.dumps({"source": "GlyGen"})
            }
        elif table_name == "proteins_clean":
            node_id = str(row["protein_id"])
            sequence = str(row.get("sequence", "") or "").strip()
            nodes[node_id] = {
                'node_id': node_id,
                'node_type': "protein",
                'label': str(row.get("gene_symbol", row["protein_id"])),
                'metadata': json.dumps({"source": "UniProt"}),
                'sequence': sequence
            }
        elif table_name == "glycans_clean":
            node_id = str(row["glycan_id"])
            meta = {}
            if row.get("structure") and pd.notna(row.get("structure")):
                meta["WURCS"] = str(row["structure"])
            nodes[node_id] = {
                'node_id': node_id,
                'node_type': "glycan",
                'label': str(row["glycan_id"]),
                'metadata': json.dumps(meta)
            }
        elif table_name == "diseases_clean":
            node_id = str(row["disease_id"])
            nodes[node_id] = {
                'node_id': node_id,
                'node_type': "disease",
                'label': str(row.get("name", row["disease_id"])),
                'metadata': json.dumps({"source": "UniProt"})
            }
        elif table_name == "variants_clean":
            node_id = str(row["variant_id"])
            desc = str(row.get("description", "")) if row.get("description") and pd.notna(row.get("description")) else ""
            nodes[node_id] = {
                'node_id': node_id,
                'node_type': "variant",
                'label': str(row["variant_id"]),
                'metadata': json.dumps({"description": desc})
            }
        elif table_name == "compounds_clean":
            node_id = str(row["compound_id"])
            nodes[node_id] = {
                'node_id': node_id,
                'node_type': "compound",
                'label': str(row.get("name", row["compound_id"])),
                'metadata': json.dumps({"source": "ChEMBL"})
            }
        elif table_name == "locations_clean":
            node_id = str(row["location_id"])
            nodes[node_id] = {
                'node_id': node_id,
                'node_type': "cellular_location",
                'label': str(row.get("name", node_id)),
                'metadata': json.dumps({"source": "UniProt"})
            }

    return {'nodes': nodes, 'worker_id': worker_id}


def _process_edge_table_partition(args):
    """
    Worker function to process a partition of an edge table.
    
    Args:
        args: Tuple of (table_name, rows_data, worker_id, seed)
    
    Returns:
        Dict with edges data.
    """
    table_name, rows_data, worker_id, seed = args
    seed_worker(worker_id, seed)
    
    edges = {}
    
    for row in rows_data:
        if table_name == "edges_enzyme_compound":
            meta = {"source": "ChEMBL"}
            if row.get("type") and pd.notna(row.get("type")):
                meta["type"] = str(row["type"])
            if row.get("value") and pd.notna(row.get("value")):
                meta["value"] = float(row["value"]) if row["value"] else None
            if row.get("units") and pd.notna(row.get("units")):
                meta["units"] = str(row["units"])
            
            key = (str(row["compound_id"]), str(row["enzyme_id"]), "inhibits")
            edges[key] = {
                'source_id': str(row["compound_id"]),
                'target_id': str(row["enzyme_id"]),
                'relation': "inhibits",
                'metadata': json.dumps(meta)
            }
        elif table_name == "edges_glycan_protein":
            key = (str(row["protein_id"]), str(row["glycan_id"]), "has_glycan")
            edges[key] = {
                'source_id': str(row["protein_id"]),
                'target_id': str(row["glycan_id"]),
                'relation': "has_glycan",
                'metadata': json.dumps({"source": "GlyGen"})
            }
        elif table_name == "edges_protein_disease":
            key = (str(row["protein_id"]), str(row["disease_id"]), "associated_with_disease")
            edges[key] = {
                'source_id': str(row["protein_id"]),
                'target_id': str(row["disease_id"]),
                'relation': "associated_with_disease",
                'metadata': json.dumps({"source": "UniProt"})
            }
        elif table_name == "edges_protein_variant":
            key = (str(row["protein_id"]), str(row["variant_id"]), "has_variant")
            edges[key] = {
                'source_id': str(row["protein_id"]),
                'target_id': str(row["variant_id"]),
                'relation': "has_variant",
                'metadata': json.dumps({"source": "UniProt"})
            }
        elif table_name == "edges_protein_location":
            key = (str(row["protein_id"]), str(row["location_id"]), "localized_in")
            edges[key] = {
                'source_id': str(row["protein_id"]),
                'target_id': str(row["location_id"]),
                'relation': "localized_in",
                'metadata': json.dumps({"source": "UniProt"})
            }
        elif table_name == "edges_enzyme_location":
            key = (str(row["enzyme_id"]), str(row["location_id"]), "localized_in")
            edges[key] = {
                'source_id': str(row["enzyme_id"]),
                'target_id': str(row["location_id"]),
                'relation': "localized_in",
                'metadata': json.dumps({"source": "UniProt"})
            }

    return {'edges': edges, 'worker_id': worker_id}


def load_clean_tables() -> Dict[str, pd.DataFrame]:
    """Load all cleaned data tables from the data_clean directory."""
    tables = {}
    
    if not os.path.exists(DATA_CLEAN_DIR):
        logger.warning(f"Clean data directory not found: {DATA_CLEAN_DIR}")
        return tables
    
    for fname in os.listdir(DATA_CLEAN_DIR):
        if fname.endswith(".tsv"):
            key = fname.replace(".tsv", "")
            try:
                tables[key] = pd.read_csv(os.path.join(DATA_CLEAN_DIR, fname), sep="\t")
                logger.info(f"Loaded {key}: {len(tables[key])} rows")
            except Exception as e:
                logger.error(f"Failed to load {fname}: {e}")
    
    return tables


def load_glycan_immunogenicity_labels() -> Dict[str, int]:
    """
    Load optional glycan immunogenicity labels from data_clean.

    Supported file: data_clean/glycan_immunogenicity.tsv
    Supported label columns: label, immunogenicity, y
    Returns mapping: {glycan_id: 0|1}
    """
    path = os.path.join(DATA_CLEAN_DIR, "glycan_immunogenicity.tsv")
    if not os.path.exists(path):
        return {}

    try:
        df = pd.read_csv(path, sep="\t")
    except Exception as e:
        logger.warning(f"Failed to load glycan immunogenicity labels: {e}")
        return {}

    if "glycan_id" not in df.columns:
        logger.warning("glycan_immunogenicity.tsv missing 'glycan_id' column")
        return {}

    label_col = None
    for c in ("label", "immunogenicity", "y"):
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        logger.warning("glycan_immunogenicity.tsv missing label column (label/immunogenicity/y)")
        return {}

    labels: Dict[str, int] = {}
    skipped = 0
    for _, row in df.iterrows():
        gid = str(row.get("glycan_id", "")).strip()
        if not gid:
            skipped += 1
            continue
        raw = row.get(label_col)
        try:
            val = int(float(raw))
        except Exception:
            skipped += 1
            continue
        if val not in (0, 1):
            skipped += 1
            continue
        labels[gid] = val

    logger.info(
        "Loaded glycan immunogenicity labels: %d (skipped=%d) from %s",
        len(labels),
        skipped,
        path,
    )
    return labels


def load_glycan_function_labels() -> Dict[str, str]:
    """
    Load optional glycan function labels from data_clean.

    Supported file: data_clean/glycan_function_labels.tsv
    Supported term columns: function_term, label, class
    Returns mapping: {glycan_id: function_term}
    If multiple labels exist for one glycan, the lexicographically-first
    label is used for deterministic single-label evaluation.
    """
    path = os.path.join(DATA_CLEAN_DIR, "glycan_function_labels.tsv")
    if not os.path.exists(path):
        return {}

    try:
        df = pd.read_csv(path, sep="\t")
    except Exception as e:
        logger.warning(f"Failed to load glycan function labels: {e}")
        return {}

    if "glycan_id" not in df.columns:
        logger.warning("glycan_function_labels.tsv missing 'glycan_id' column")
        return {}

    term_col = None
    for c in ("function_term", "label", "class"):
        if c in df.columns:
            term_col = c
            break
    if term_col is None:
        logger.warning("glycan_function_labels.tsv missing term column (function_term/label/class)")
        return {}

    grouped: Dict[str, set] = defaultdict(set)
    skipped = 0
    for _, row in df.iterrows():
        gid = str(row.get("glycan_id", "")).strip()
        if not gid:
            skipped += 1
            continue
        term = str(row.get(term_col, "")).strip()
        if not term or term.lower() in {"nan", "none", "null"}:
            skipped += 1
            continue
        grouped[gid].add(term)

    labels = {gid: sorted(list(terms))[0] for gid, terms in grouped.items() if terms}
    logger.info(
        "Loaded glycan function labels: %d glycans (unique_terms=%d, skipped=%d) from %s",
        len(labels),
        len({v for v in labels.values()}),
        skipped,
        path,
    )
    return labels


def load_uniprot_sites() -> pd.DataFrame:
    """
    Load UniProt glycosylation sites from data_clean/uniprot_sites.tsv.
    
    Returns:
        DataFrame with UniProt site data or empty DataFrame if not found.
    """
    sites_path = os.path.join(DATA_CLEAN_DIR, "uniprot_sites.tsv")
    
    if not os.path.exists(sites_path):
        logger.info("UniProt sites file not found, skipping site loading")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(sites_path, sep="\t")
        logger.info(f"Loaded {len(df)} UniProt glycosylation sites")
        return df
    except Exception as e:
        logger.error(f"Failed to load UniProt sites: {e}")
        return pd.DataFrame()


def load_ptmcode_sites() -> pd.DataFrame:
    """
    Load PTMCode sites from data_clean/ptmcode_sites.tsv.
    
    Returns:
        DataFrame with PTMCode site data or empty DataFrame if not found.
    """
    sites_path = os.path.join(DATA_CLEAN_DIR, "ptmcode_sites.tsv")
    
    if not os.path.exists(sites_path):
        logger.info("PTMCode sites file not found, skipping PTMCode site loading")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(sites_path, sep="\t")
        logger.info(f"Loaded {len(df)} PTMCode sites")
        return df
    except Exception as e:
        logger.error(f"Failed to load PTMCode sites: {e}")
        return pd.DataFrame()


def load_ptmcode_edges() -> pd.DataFrame:
    """
    Load PTMCode crosstalk edges from data_clean/ptmcode_edges.tsv.
    
    Returns:
        DataFrame with PTMCode edge data or empty DataFrame if not found.
    """
    edges_path = os.path.join(DATA_CLEAN_DIR, "ptmcode_edges.tsv")
    
    if not os.path.exists(edges_path):
        logger.info("PTMCode edges file not found, skipping PTMCode edge loading")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(edges_path, sep="\t")
        logger.info(f"Loaded {len(df)} PTMCode crosstalk edges")
        return df
    except Exception as e:
        logger.error(f"Failed to load PTMCode edges: {e}")
        return pd.DataFrame()


def build_site_nodes(
    nodes: 'NodeAccumulator',
    uniprot_sites_df: pd.DataFrame,
    ptmcode_sites_df: pd.DataFrame
) -> int:
    """
    Build site nodes from UniProt and PTMCode site data.
    
    Site ID format: SITE::<UniProtID>::<Position>::<Residue>
    
    Args:
        nodes: NodeAccumulator to add nodes to
        uniprot_sites_df: DataFrame with UniProt glycosylation sites
        ptmcode_sites_df: DataFrame with PTMCode PTM sites
    
    Returns:
        Number of site nodes added
    """
    site_count = 0
    
    if not uniprot_sites_df.empty:
        for _, row in uniprot_sites_df.iterrows():
            uniprot_id = str(row.get('uniprot_id', ''))
            position = row.get('site_position')
            residue = str(row.get('site_residue', ''))
            
            if not uniprot_id or pd.isna(position) or not residue:
                continue
            
            position = int(position)
            site_id = create_site_id(uniprot_id, position, residue)
            
            meta = {
                "source": "UniProt",
                "position": position,
                "residue": residue,
                "site_type": str(row.get('site_type', 'glycosylation')),
            }
            
            if pd.notna(row.get('evidence_code')):
                meta["evidence_code"] = str(row['evidence_code'])
            if pd.notna(row.get('evidence_type')):
                meta["evidence_type"] = str(row['evidence_type'])
            if pd.notna(row.get('description')):
                meta["description"] = str(row['description'])
            
            nodes.add(
                node_id=site_id,
                node_type="site",
                label=f"{uniprot_id}:{residue}{position}",
                metadata=meta
            )
            site_count += 1
    
    if not ptmcode_sites_df.empty:
        for _, row in ptmcode_sites_df.iterrows():
            uniprot_id = str(row.get('uniprot_id', ''))
            position = row.get('site_position')
            residue = str(row.get('site_residue', ''))
            
            if not uniprot_id or pd.isna(position) or not residue:
                continue
            
            position = int(position)
            site_id = create_site_id(uniprot_id, position, residue)
            
            meta = {
                "source": "PTMCode",
                "position": position,
                "residue": residue,
                "ptm_type": str(row.get('ptm_type', 'unknown')),
            }
            
            nodes.add(
                node_id=site_id,
                node_type="site",
                label=f"{uniprot_id}:{residue}{position}",
                metadata=meta
            )
            site_count += 1
    
    logger.info(f"Built {site_count} site nodes")
    return site_count


def build_protein_to_site_edges(
    edges: 'EdgeAccumulator',
    uniprot_sites_df: pd.DataFrame,
    ptmcode_sites_df: pd.DataFrame,
    existing_protein_ids: set
) -> int:
    """
    Build protein -> has_site -> site edges.
    
    Args:
        edges: EdgeAccumulator to add edges to
        uniprot_sites_df: DataFrame with UniProt glycosylation sites
        ptmcode_sites_df: DataFrame with PTMCode PTM sites
        existing_protein_ids: Set of existing protein node IDs
    
    Returns:
        Number of edges added
    """
    edge_count = 0
    
    if not uniprot_sites_df.empty:
        for _, row in uniprot_sites_df.iterrows():
            uniprot_id = str(row.get('uniprot_id', ''))
            position = row.get('site_position')
            residue = str(row.get('site_residue', ''))
            
            if not uniprot_id or pd.isna(position) or not residue:
                continue
            
            position = int(position)
            site_id = create_site_id(uniprot_id, position, residue)
            
            base_protein_id = uniprot_id.split('-')[0] if '-' in uniprot_id else uniprot_id
            
            protein_id = None
            if uniprot_id in existing_protein_ids:
                protein_id = uniprot_id
            elif base_protein_id in existing_protein_ids:
                protein_id = base_protein_id
            elif f"{base_protein_id}-1" in existing_protein_ids:
                protein_id = f"{base_protein_id}-1"
            
            if protein_id:
                edges.add(
                    source_id=protein_id,
                    target_id=site_id,
                    relation="has_site",
                    metadata={"source": "UniProt", "site_type": str(row.get('site_type', 'glycosylation'))}
                )
                edge_count += 1
    
    if not ptmcode_sites_df.empty:
        for _, row in ptmcode_sites_df.iterrows():
            uniprot_id = str(row.get('uniprot_id', ''))
            position = row.get('site_position')
            residue = str(row.get('site_residue', ''))
            
            if not uniprot_id or pd.isna(position) or not residue:
                continue
            
            position = int(position)
            site_id = create_site_id(uniprot_id, position, residue)
            
            base_protein_id = uniprot_id.split('-')[0] if '-' in uniprot_id else uniprot_id
            
            protein_id = None
            if uniprot_id in existing_protein_ids:
                protein_id = uniprot_id
            elif base_protein_id in existing_protein_ids:
                protein_id = base_protein_id
            elif f"{base_protein_id}-1" in existing_protein_ids:
                protein_id = f"{base_protein_id}-1"
            
            if protein_id:
                edges.add(
                    source_id=protein_id,
                    target_id=site_id,
                    relation="has_site",
                    metadata={"source": "PTMCode", "ptm_type": str(row.get('ptm_type', 'unknown'))}
                )
                edge_count += 1
    
    logger.info(f"Built {edge_count} protein-to-site edges")
    return edge_count


def build_ptmcode_edges(
    edges: 'EdgeAccumulator',
    ptmcode_edges_df: pd.DataFrame
) -> int:
    """
    Build site -> ptm_crosstalk -> site edges from PTMCode data.
    
    Args:
        edges: EdgeAccumulator to add edges to
        ptmcode_edges_df: DataFrame with PTMCode crosstalk edges
    
    Returns:
        Number of edges added
    """
    edge_count = 0
    
    if ptmcode_edges_df.empty:
        return edge_count
    
    for _, row in ptmcode_edges_df.iterrows():
        site1_id = row.get('site1_id')
        site2_id = row.get('site2_id')
        
        if pd.isna(site1_id) or pd.isna(site2_id):
            site1_uniprot = str(row.get('site1_uniprot', ''))
            site1_position = row.get('site1_position')
            site1_residue = str(row.get('site1_residue', ''))
            
            site2_uniprot = str(row.get('site2_uniprot', ''))
            site2_position = row.get('site2_position')
            site2_residue = str(row.get('site2_residue', ''))
            
            if not site1_uniprot or pd.isna(site1_position) or not site1_residue:
                continue
            if not site2_uniprot or pd.isna(site2_position) or not site2_residue:
                continue
            
            site1_id = create_site_id(site1_uniprot, int(site1_position), site1_residue)
            site2_id = create_site_id(site2_uniprot, int(site2_position), site2_residue)
        
        score = row.get('score', 0.0)
        if pd.isna(score):
            score = 0.0
        
        meta = {
            "source": "PTMCode",
            "score": float(score),
        }
        
        if pd.notna(row.get('site1_ptm_type')):
            meta["site1_ptm_type"] = str(row['site1_ptm_type'])
        if pd.notna(row.get('site2_ptm_type')):
            meta["site2_ptm_type"] = str(row['site2_ptm_type'])
        
        edges.add(
            source_id=str(site1_id),
            target_id=str(site2_id),
            relation="ptm_crosstalk",
            metadata=meta
        )
        edge_count += 1
    
    logger.info(f"Built {edge_count} PTM crosstalk edges")
    return edge_count


def build_site_glycan_edges(
    edges: 'EdgeAccumulator',
    uniprot_sites_df: pd.DataFrame,
    glycan_protein_df: pd.DataFrame,
) -> int:
    """
    Build site -> glycosylated_at -> glycan edges by type-matching.

    Join logic:
    1. For each glycosylation site, find the matching protein in the KG
    2. Find all glycans associated with that protein (from edges_glycan_protein)
    3. Check if site_type matches glycan function_term (N-linked→N-linked, etc.)
    4. Create site→glycan edge if type matches
    """
    TYPE_MAP = {
        "N-linked": "N-linked",
        "O-linked": "O-linked",
        "C-mannosylation": "C-linked",
    }

    # Build protein→glycan lookup (base protein ID → set of glycan_ids)
    protein_glycans: Dict[str, Set[str]] = defaultdict(set)
    for _, row in glycan_protein_df.iterrows():
        pid = str(row["protein_id"])
        base = pid.split("-")[0]
        gid = str(row["glycan_id"])
        protein_glycans[pid].add(gid)
        protein_glycans[base].add(gid)

    # Build glycan→function_terms lookup
    glycan_functions: Dict[str, Set[str]] = defaultdict(set)
    func_labels_path = os.path.join(DATA_CLEAN_DIR, "glycan_function_labels.tsv")
    if os.path.exists(func_labels_path):
        func_df = pd.read_csv(func_labels_path, sep="\t")
        for _, row in func_df.iterrows():
            gid = str(row.get("glycan_id", ""))
            term = str(row.get("function_term", ""))
            if gid and term:
                glycan_functions[gid].add(term)

    edge_count = 0
    for _, row in uniprot_sites_df.iterrows():
        uniprot_id = str(row.get("uniprot_id", ""))
        position = row.get("site_position")
        residue = str(row.get("site_residue", ""))
        site_type = str(row.get("site_type", ""))

        if not uniprot_id or pd.isna(position) or not residue:
            continue

        position = int(position)
        site_id = create_site_id(uniprot_id, position, residue)

        base = uniprot_id.split("-")[0] if "-" in uniprot_id else uniprot_id

        # Find glycans for this protein
        glycans = protein_glycans.get(uniprot_id, set()) | \
                  protein_glycans.get(base, set()) | \
                  protein_glycans.get(f"{base}-1", set())

        if not glycans:
            continue

        # Get target function term for type matching
        target_term = TYPE_MAP.get(site_type)

        for gid in glycans:
            if target_term:
                gfuncs = glycan_functions.get(gid, set())
                if gfuncs and target_term not in gfuncs:
                    continue  # Skip type mismatch
            elif site_type == "glycosylation":
                pass  # Generic type → accept all glycans
            else:
                continue  # Unknown site_type

            edges.add(
                source_id=site_id,
                target_id=gid,
                relation="glycosylated_at",
                metadata={"source": "inferred", "site_type": site_type}
            )
            edge_count += 1

    logger.info(f"Built {edge_count} site-glycan (glycosylated_at) edges")
    return edge_count


def construct_nodes_and_edges(tables: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construct nodes and edges from cleaned data tables.
    Uses accumulators for proper deduplication and metadata merging.
    """
    nodes = NodeAccumulator()
    edges = EdgeAccumulator()
    glycan_labels = load_glycan_immunogenicity_labels()
    glycan_function_labels = load_glycan_function_labels()

    if "enzymes_clean" in tables:
        for _, row in tables["enzymes_clean"].iterrows():
            nodes.add(
                node_id=str(row["enzyme_id"]),
                node_type="enzyme",
                label=str(row.get("gene_symbol", row["enzyme_id"])),
                metadata={"source": "GlyGen"}
            )

    if "proteins_clean" in tables:
        for _, row in tables["proteins_clean"].iterrows():
            sequence = str(row.get("sequence", "") or "").strip()
            nodes.add(
                node_id=str(row["protein_id"]),
                node_type="protein",
                label=str(row.get("gene_symbol", row["protein_id"])),
                metadata={"source": "UniProt"},
                sequence=sequence
            )

    if "glycans_clean" in tables:
        for _, row in tables["glycans_clean"].iterrows():
            meta = {}
            if pd.notna(row.get("structure")):
                meta["WURCS"] = str(row["structure"])
            gid = str(row["glycan_id"])
            if gid in glycan_labels:
                meta["immunogenicity"] = int(glycan_labels[gid])
            if gid in glycan_function_labels:
                meta["glycan_function"] = str(glycan_function_labels[gid])
            nodes.add(
                node_id=gid,
                node_type="glycan",
                label=gid,
                metadata=meta
            )

    if "diseases_clean" in tables:
        for _, row in tables["diseases_clean"].iterrows():
            nodes.add(
                node_id=str(row["disease_id"]),
                node_type="disease",
                label=str(row.get("name", row["disease_id"])),
                metadata={"source": "UniProt"}
            )

    if "variants_clean" in tables:
        for _, row in tables["variants_clean"].iterrows():
            desc = str(row.get("description", "")) if pd.notna(row.get("description")) else ""
            nodes.add(
                node_id=str(row["variant_id"]),
                node_type="variant",
                label=str(row["variant_id"]),
                metadata={"description": desc}
            )

    if "compounds_clean" in tables:
        for _, row in tables["compounds_clean"].iterrows():
            nodes.add(
                node_id=str(row["compound_id"]),
                node_type="compound",
                label=str(row.get("name", row["compound_id"])),
                metadata={"source": "ChEMBL"}
            )

    if "locations_clean" in tables:
        for _, row in tables["locations_clean"].iterrows():
            nodes.add(
                node_id=str(row["location_id"]),
                node_type="cellular_location",
                label=str(row.get("name", row["location_id"])),
                metadata={"source": "UniProt"}
            )
        logger.info(f"Added {len(tables['locations_clean'])} cellular_location nodes")

    if "edges_enzyme_compound" in tables:
        for _, row in tables["edges_enzyme_compound"].iterrows():
            meta = {"source": "ChEMBL"}
            if pd.notna(row.get("type")):
                meta["type"] = str(row["type"])
            if pd.notna(row.get("value")):
                meta["value"] = float(row["value"]) if row["value"] else None
            if pd.notna(row.get("units")):
                meta["units"] = str(row["units"])
            
            edges.add(
                source_id=str(row["compound_id"]),
                target_id=str(row["enzyme_id"]),
                relation="inhibits",
                metadata=meta
            )

    if "edges_glycan_protein" in tables:
        for _, row in tables["edges_glycan_protein"].iterrows():
            edges.add(
                source_id=str(row["protein_id"]),
                target_id=str(row["glycan_id"]),
                relation="has_glycan",
                metadata={"source": "GlyGen"}
            )

    if "edges_protein_disease" in tables:
        for _, row in tables["edges_protein_disease"].iterrows():
            edges.add(
                source_id=str(row["protein_id"]),
                target_id=str(row["disease_id"]),
                relation="associated_with_disease",
                metadata={"source": "UniProt"}
            )

    if "edges_protein_variant" in tables:
        for _, row in tables["edges_protein_variant"].iterrows():
            edges.add(
                source_id=str(row["protein_id"]),
                target_id=str(row["variant_id"]),
                relation="has_variant",
                metadata={"source": "UniProt"}
            )

    # Phase 2: New node types
    
    # Motif nodes
    if "motifs_clean" in tables:
        for _, row in tables["motifs_clean"].iterrows():
            nodes.add(
                node_id=str(row["motif_id"]),
                node_type="motif",
                label=str(row.get("name", row["motif_id"])),
                metadata={"source": "GlyGen"}
            )
        logger.info(f"Added {len(tables['motifs_clean'])} motif nodes")

    # Reaction nodes
    if "reactions_clean" in tables:
        for _, row in tables["reactions_clean"].iterrows():
            meta = {"source": "GlyGen"}
            if pd.notna(row.get("residue_name")):
                meta["residue_name"] = str(row["residue_name"])
            if pd.notna(row.get("glycan_id")):
                meta["glycan_id"] = str(row["glycan_id"])
            nodes.add(
                node_id=str(row["reaction_id"]),
                node_type="reaction",
                label=str(row["reaction_id"]),
                metadata=meta
            )
        logger.info(f"Added {len(tables['reactions_clean'])} reaction nodes")

    # Pathway nodes
    if "pathways_clean" in tables:
        for _, row in tables["pathways_clean"].iterrows():
            meta = {"source": "GlyGen"}
            if pd.notna(row.get("type")):
                meta["pathway_type"] = str(row["type"])
            nodes.add(
                node_id=str(row["pathway_id"]),
                node_type="pathway",
                label=str(row.get("name", row["pathway_id"])),
                metadata=meta
            )
        logger.info(f"Added {len(tables['pathways_clean'])} pathway nodes")

    # Phase 2: New edge types
    
    # Glycan -> Enzyme edges (produced_by, consumed_by)
    if "edges_glycan_enzyme" in tables:
        for _, row in tables["edges_glycan_enzyme"].iterrows():
            meta = {"source": "GlyGen"}
            if pd.notna(row.get("gene")):
                meta["gene"] = str(row["gene"])
            if pd.notna(row.get("protein_name")):
                meta["protein_name"] = str(row["protein_name"])
            
            relation = str(row.get("relation", "produced_by"))
            edges.add(
                source_id=str(row["glycan_id"]),
                target_id=str(row["enzyme_id"]),
                relation=relation,
                metadata=meta
            )
        logger.info(f"Added {len(tables['edges_glycan_enzyme'])} glycan-enzyme edges")

    # Glycan -> Glycan subsumption edges (parent_of, child_of, subsumes, subsumed_by)
    if "edges_glycan_subsumption" in tables:
        for _, row in tables["edges_glycan_subsumption"].iterrows():
            relation = str(row.get("relation", "child_of"))
            edges.add(
                source_id=str(row["glycan_id"]),
                target_id=str(row["related_glycan_id"]),
                relation=relation,
                metadata={"source": "GlyGen"}
            )
        logger.info(f"Added {len(tables['edges_glycan_subsumption'])} glycan-glycan subsumption edges")

    # Glycan -> Motif edges (has_motif)
    if "edges_glycan_motif" in tables:
        for _, row in tables["edges_glycan_motif"].iterrows():
            relation = str(row.get("relation", "has_motif"))
            edges.add(
                source_id=str(row["glycan_id"]),
                target_id=str(row["motif_id"]),
                relation=relation,
                metadata={"source": "GlyGen"}
            )
        logger.info(f"Added {len(tables['edges_glycan_motif'])} glycan-motif edges")

    # Reaction edges (catalyzed_by, has_product)
    if "edges_reaction" in tables:
        for _, row in tables["edges_reaction"].iterrows():
            relation = str(row.get("relation", "catalyzed_by"))
            meta = {"source": "GlyGen"}
            if pd.notna(row.get("target_type")):
                meta["target_type"] = str(row["target_type"])
            edges.add(
                source_id=str(row["reaction_id"]),
                target_id=str(row["target_id"]),
                relation=relation,
                metadata=meta
            )
        logger.info(f"Added {len(tables['edges_reaction'])} reaction edges")

    # Enzyme -> Disease edges (associated_with_disease)
    if "edges_enzyme_disease" in tables:
        for _, row in tables["edges_enzyme_disease"].iterrows():
            edges.add(
                source_id=str(row["enzyme_id"]),
                target_id=str(row["disease_id"]),
                relation="associated_with_disease",
                metadata={"source": "UniProt"}
            )
        logger.info(f"Added {len(tables['edges_enzyme_disease'])} enzyme-disease edges")

    # Protein -> Location edges (localized_in)
    if "edges_protein_location" in tables:
        for _, row in tables["edges_protein_location"].iterrows():
            edges.add(
                source_id=str(row["protein_id"]),
                target_id=str(row["location_id"]),
                relation="localized_in",
                metadata={"source": "UniProt"}
            )
        logger.info(f"Added {len(tables['edges_protein_location'])} protein-location edges")

    # Enzyme -> Location edges (localized_in)
    if "edges_enzyme_location" in tables:
        for _, row in tables["edges_enzyme_location"].iterrows():
            edges.add(
                source_id=str(row["enzyme_id"]),
                target_id=str(row["location_id"]),
                relation="localized_in",
                metadata={"source": "UniProt"}
            )
        logger.info(f"Added {len(tables['edges_enzyme_location'])} enzyme-location edges")

    # Site-level PTM data integration
    # Load site data if enabled in config
    if config.site_data.enable_uniprot_sites or config.site_data.enable_ptmcode:
        logger.info("Loading site-level PTM data...")
        
        uniprot_sites_df = pd.DataFrame()
        ptmcode_sites_df = pd.DataFrame()
        ptmcode_edges_df = pd.DataFrame()
        
        if config.site_data.enable_uniprot_sites:
            uniprot_sites_df = load_uniprot_sites()
        
        if config.site_data.enable_ptmcode:
            ptmcode_sites_df = load_ptmcode_sites()
            ptmcode_edges_df = load_ptmcode_edges()
        
        # Build site nodes
        if not uniprot_sites_df.empty or not ptmcode_sites_df.empty:
            build_site_nodes(nodes, uniprot_sites_df, ptmcode_sites_df)
            
            # Get existing protein/enzyme IDs for edge building
            existing_protein_ids = set()
            if "proteins_clean" in tables:
                existing_protein_ids.update(tables["proteins_clean"]["protein_id"].astype(str))
            if "enzymes_clean" in tables:
                existing_protein_ids.update(tables["enzymes_clean"]["enzyme_id"].astype(str))
            
            # Build protein -> site edges
            build_protein_to_site_edges(edges, uniprot_sites_df, ptmcode_sites_df, existing_protein_ids)
        
        # Build PTMCode crosstalk edges
        if not ptmcode_edges_df.empty:
            build_ptmcode_edges(edges, ptmcode_edges_df)

        # Build site -> glycan edges (glycosylated_at)
        if not uniprot_sites_df.empty and "edges_glycan_protein" in tables:
            build_site_glycan_edges(
                edges, uniprot_sites_df,
                tables["edges_glycan_protein"],
            )

    return nodes.to_dataframe(), edges.to_dataframe()


def create_arrow_schema_for_nodes() -> pa.Schema:
    """Create Arrow schema for nodes based on schema file."""
    node_schema = load_schema("node_schema.yaml")
    
    arrow_fields = [
        pa.field("node_id", pa.string(), nullable=False),
        pa.field("node_type", pa.string(), nullable=False),
        pa.field("label", pa.string(), nullable=True),
        pa.field("metadata", pa.string(), nullable=True),
        pa.field("sequence", pa.string(), nullable=True),
    ]
    
    return pa.schema(arrow_fields)


def create_arrow_schema_for_edges() -> pa.Schema:
    """Create Arrow schema for edges based on schema file."""
    edge_schema = load_schema("edge_schema.yaml")
    
    arrow_fields = [
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("target_id", pa.string(), nullable=False),
        pa.field("relation", pa.string(), nullable=False),
        pa.field("metadata", pa.string(), nullable=True),
    ]
    
    return pa.schema(arrow_fields)


def export_kg(nodes_df: pd.DataFrame, edges_df: pd.DataFrame):
    """
    Export knowledge graph to TSV and Parquet formats.
    Uses proper Arrow schemas for type preservation.
    """
    os.makedirs(KG_DIR, exist_ok=True)

    nodes_df.to_csv(os.path.join(KG_DIR, "nodes.tsv"), sep="\t", index=False)
    edges_df.to_csv(os.path.join(KG_DIR, "edges.tsv"), sep="\t", index=False)
    logger.info(f"Exported TSV files to {KG_DIR}")

    if not nodes_df.empty:
        node_schema = create_arrow_schema_for_nodes()
        
        nodes_for_parquet = nodes_df.copy()
        nodes_for_parquet['node_id'] = nodes_for_parquet['node_id'].astype(str)
        nodes_for_parquet['node_type'] = nodes_for_parquet['node_type'].astype(str)
        nodes_for_parquet['label'] = nodes_for_parquet['label'].astype(str)
        nodes_for_parquet['metadata'] = nodes_for_parquet['metadata'].astype(str)
        if 'sequence' not in nodes_for_parquet.columns:
            nodes_for_parquet['sequence'] = None
        else:
            nodes_for_parquet['sequence'] = (
                nodes_for_parquet['sequence']
                .where(nodes_for_parquet['sequence'].notna(), None)
                .astype(object)
            )
        
        table = pa.Table.from_pandas(nodes_for_parquet, schema=node_schema, preserve_index=False)
        pq.write_table(table, os.path.join(KG_DIR, "nodes.parquet"))
        logger.info(f"Exported {len(nodes_df)} nodes to nodes.parquet")

    if not edges_df.empty:
        edge_schema = create_arrow_schema_for_edges()
        
        edges_for_parquet = edges_df.copy()
        edges_for_parquet['source_id'] = edges_for_parquet['source_id'].astype(str)
        edges_for_parquet['target_id'] = edges_for_parquet['target_id'].astype(str)
        edges_for_parquet['relation'] = edges_for_parquet['relation'].astype(str)
        edges_for_parquet['metadata'] = edges_for_parquet['metadata'].astype(str)
        
        table = pa.Table.from_pandas(edges_for_parquet, schema=edge_schema, preserve_index=False)
        pq.write_table(table, os.path.join(KG_DIR, "edges.parquet"))
        logger.info(f"Exported {len(edges_df)} edges to edges.parquet")

    logger.info(f"Knowledge graph export complete: {len(nodes_df)} nodes, {len(edges_df)} edges")


def main():
    """Main entry point for knowledge graph construction."""
    start_time = datetime.now()
    logger.info(f"Starting KG build at {start_time}")
    
    tables = load_clean_tables()
    
    if not tables:
        logger.warning("No clean data tables found. Run clean_data.py first.")
        return
    
    nodes_df, edges_df = construct_nodes_and_edges(tables)
    export_kg(nodes_df, edges_df)
    
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"KG build completed at {end_time} (duration: {duration})")


if __name__ == "__main__":
    main()
