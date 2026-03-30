"""KGConverter: Transform KG TSV/Parquet files into PyG HeteroData."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
import yaml
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)

# Default schema directory relative to project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SCHEMA_DIR = _PROJECT_ROOT / "schemas"


def _load_relation_config(schema_dir: Path) -> dict:
    """Load relation_config.yaml and return the relation_types mapping."""
    path = schema_dir / "relation_config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"relation_config.yaml not found at {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("relation_types", {})


def _load_edge_schema(schema_dir: Path) -> dict:
    """Load edge_schema.yaml for edge types not in relation_config."""
    path = schema_dir / "edge_schema.yaml"
    if not path.exists():
        return {}
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("edge_types", {})


class KGConverter:
    """Convert KG output (nodes.tsv/parquet + edges.tsv/parquet) to PyG HeteroData.

    The converter reads the node and edge tables produced by ``build_kg.py``
    and constructs a :class:`torch_geometric.data.HeteroData` instance with
    proper heterogeneous edge types inferred from ``relation_config.yaml``
    and ``edge_schema.yaml``.

    Parameters
    ----------
    kg_dir : str or Path
        Directory containing ``nodes.tsv`` / ``nodes.parquet`` and
        ``edges.tsv`` / ``edges.parquet``.
    schema_dir : str or Path or None
        Directory containing schema YAML files.  Defaults to ``<project_root>/schemas``.
    """

    def __init__(
        self,
        kg_dir: Union[str, Path] = "kg",
        schema_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.kg_dir = Path(kg_dir)
        self.schema_dir = Path(schema_dir) if schema_dir else _DEFAULT_SCHEMA_DIR

        # Build relation -> (source_type, target_type) from both configs
        self._relation_type_map: Dict[str, List[Tuple[str, str]]] = {}
        self._build_relation_type_map()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_relation_type_map(self) -> None:
        """Populate ``_relation_type_map`` from relation_config + edge_schema."""
        try:
            rel_cfg = _load_relation_config(self.schema_dir)
        except FileNotFoundError:
            rel_cfg = {}

        edge_cfg = _load_edge_schema(self.schema_dir)

        # Merge both configs (relation_config takes precedence)
        all_relations: dict = {}
        all_relations.update(edge_cfg)
        all_relations.update(rel_cfg)

        for rel_name, spec in all_relations.items():
            src = spec.get("source_type")
            tgt = spec.get("target_type")
            if src is None or tgt is None:
                continue

            # source_type can be a list (e.g. has_site: [protein, enzyme])
            src_list = src if isinstance(src, list) else [src]
            tgt_list = tgt if isinstance(tgt, list) else [tgt]

            pairs: List[Tuple[str, str]] = []
            for s in src_list:
                for t in tgt_list:
                    pairs.append((s, t))
            self._relation_type_map[rel_name] = pairs

    def _resolve_edge_type(
        self,
        relation: str,
        src_node_type: Optional[str],
        dst_node_type: Optional[str],
    ) -> Tuple[str, str, str]:
        """Determine the canonical (src_type, relation, dst_type) triplet.

        Uses the schema when possible, falling back to the actual node types
        of the endpoints when multiple source types are allowed (e.g. has_site).
        """
        pairs = self._relation_type_map.get(relation)
        if pairs is None:
            # Unknown relation — use actual node types
            s = src_node_type or "unknown"
            t = dst_node_type or "unknown"
            return (s, relation, t)

        if len(pairs) == 1:
            return (pairs[0][0], relation, pairs[0][1])

        # Multiple valid (src, dst) pairs — pick the one matching actual types
        for s, t in pairs:
            if s == src_node_type and t == dst_node_type:
                return (s, relation, t)

        # Fallback: use first pair
        return (pairs[0][0], relation, pairs[0][1])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load nodes and edges DataFrames from KG directory.

        Prefers Parquet files when available; falls back to TSV.

        Returns
        -------
        nodes_df : pd.DataFrame
            Columns: ``node_id, node_type, label, metadata``
        edges_df : pd.DataFrame
            Columns: ``source_id, target_id, relation, metadata``
        """
        # Nodes
        nodes_pq = self.kg_dir / "nodes.parquet"
        nodes_tsv = self.kg_dir / "nodes.tsv"
        if nodes_pq.exists():
            nodes_df = pd.read_parquet(nodes_pq)
            logger.info("Loaded nodes from %s (%d rows)", nodes_pq, len(nodes_df))
        elif nodes_tsv.exists():
            nodes_df = pd.read_csv(nodes_tsv, sep="\t")
            logger.info("Loaded nodes from %s (%d rows)", nodes_tsv, len(nodes_df))
        else:
            raise FileNotFoundError(
                f"Neither nodes.parquet nor nodes.tsv found in {self.kg_dir}"
            )

        # Edges
        edges_pq = self.kg_dir / "edges.parquet"
        edges_tsv = self.kg_dir / "edges.tsv"
        if edges_pq.exists():
            edges_df = pd.read_parquet(edges_pq)
            logger.info("Loaded edges from %s (%d rows)", edges_pq, len(edges_df))
        elif edges_tsv.exists():
            edges_df = pd.read_csv(edges_tsv, sep="\t")
            logger.info("Loaded edges from %s (%d rows)", edges_tsv, len(edges_df))
        else:
            raise FileNotFoundError(
                f"Neither edges.parquet nor edges.tsv found in {self.kg_dir}"
            )

        return nodes_df, edges_df

    def build_node_mappings(
        self, nodes_df: pd.DataFrame
    ) -> Dict[str, Dict[str, int]]:
        """Create per-type node_id -> integer index mappings.

        Parameters
        ----------
        nodes_df : pd.DataFrame
            Must contain ``node_id`` and ``node_type`` columns.

        Returns
        -------
        mappings : dict[str, dict[str, int]]
            ``{node_type: {node_id: local_index, ...}, ...}``
        """
        mappings: Dict[str, Dict[str, int]] = {}
        for ntype, group in nodes_df.groupby("node_type", sort=False):
            ids = group["node_id"].values
            mappings[str(ntype)] = {str(nid): idx for idx, nid in enumerate(ids)}
        for ntype, m in mappings.items():
            logger.info("Node type '%s': %d nodes", ntype, len(m))
        return mappings

    def build_hetero_graph(
        self,
        nodes_df: pd.DataFrame,
        edges_df: pd.DataFrame,
        node_mappings: Dict[str, Dict[str, int]],
        feature_dim: int = 256,
    ) -> HeteroData:
        """Build a :class:`HeteroData` graph from DataFrames and mappings.

        Each node type gets a random-initialised ``x`` tensor of shape
        ``(num_nodes, feature_dim)`` (Xavier uniform) so that downstream
        models can train from scratch or replace them with pre-computed
        features.

        Parameters
        ----------
        nodes_df : pd.DataFrame
        edges_df : pd.DataFrame
        node_mappings : dict
            Output of :meth:`build_node_mappings`.
        feature_dim : int
            Dimensionality of initial node feature vectors.

        Returns
        -------
        data : HeteroData
        """
        data = HeteroData()

        # -- Node features (Xavier-uniform initialised) --
        for ntype, mapping in node_mappings.items():
            num_nodes = len(mapping)
            x = torch.empty(num_nodes, feature_dim)
            torch.nn.init.xavier_uniform_(x)
            data[ntype].x = x
            data[ntype].num_nodes = num_nodes

        # Build a global node_id -> (node_type, local_idx) lookup
        global_lookup: Dict[str, Tuple[str, int]] = {}
        for ntype, mapping in node_mappings.items():
            for nid, idx in mapping.items():
                global_lookup[nid] = (ntype, idx)

        # -- Edges --
        # Accumulate edges per canonical edge type
        edge_collector: Dict[Tuple[str, str, str], List[List[int]]] = {}

        src_ids = edges_df["source_id"].astype(str).values
        tgt_ids = edges_df["target_id"].astype(str).values
        relations = edges_df["relation"].astype(str).values

        skipped = 0
        for i in range(len(edges_df)):
            sid = src_ids[i]
            tid = tgt_ids[i]
            rel = relations[i]

            src_info = global_lookup.get(sid)
            tgt_info = global_lookup.get(tid)

            if src_info is None or tgt_info is None:
                skipped += 1
                continue

            src_type, src_idx = src_info
            tgt_type, tgt_idx = tgt_info

            canon = self._resolve_edge_type(rel, src_type, tgt_type)

            if canon not in edge_collector:
                edge_collector[canon] = [[], []]
            edge_collector[canon][0].append(src_idx)
            edge_collector[canon][1].append(tgt_idx)

        if skipped > 0:
            logger.warning(
                "Skipped %d edges with missing source/target nodes", skipped
            )

        for (s, r, t), (src_list, tgt_list) in edge_collector.items():
            ei = torch.tensor([src_list, tgt_list], dtype=torch.long)
            data[s, r, t].edge_index = ei
            logger.info(
                "Edge type (%s, %s, %s): %d edges", s, r, t, ei.size(1)
            )

        # Optional supervised labels from node metadata (downstream tasks).
        self._attach_optional_node_labels(data, nodes_df, node_mappings)

        return data

    def _attach_optional_node_labels(
        self,
        data: HeteroData,
        nodes_df: pd.DataFrame,
        node_mappings: Dict[str, Dict[str, int]],
    ) -> None:
        """Attach optional label tensors from metadata when available.

        Currently supported:
        - glycan.y from metadata key 'immunogenicity' (0/1)
        - glycan.taxonomy_domain from metadata key 'glycan_function' (categorical)
        """
        if "glycan" not in node_mappings:
            return

        glycan_map = node_mappings["glycan"]
        y = torch.full((len(glycan_map),), -1, dtype=torch.long)
        labeled = 0
        function_term_per_idx: Dict[int, str] = {}

        glycan_rows = nodes_df[nodes_df["node_type"].astype(str) == "glycan"]
        for _, row in glycan_rows.iterrows():
            gid = str(row["node_id"])
            idx = glycan_map.get(gid)
            if idx is None:
                continue
            raw = row.get("metadata", "{}")
            try:
                meta = json.loads(str(raw)) if pd.notna(raw) and str(raw) else {}
            except (json.JSONDecodeError, TypeError):
                meta = {}

            if "immunogenicity" not in meta:
                pass
            else:
                try:
                    val = int(meta["immunogenicity"])
                except Exception:
                    val = None
                if val in (0, 1):
                    y[idx] = val
                    labeled += 1

            term = str(meta.get("glycan_function", "")).strip()
            if term:
                function_term_per_idx[idx] = term

        if labeled > 0:
            data["glycan"].y = y
            logger.info("Attached glycan.y labels from metadata: %d/%d", labeled, len(glycan_map))

        if function_term_per_idx:
            terms = sorted(set(function_term_per_idx.values()))
            term_to_idx = {t: i for i, t in enumerate(terms)}
            domain = torch.full((len(glycan_map),), -1, dtype=torch.long)
            for idx, term in function_term_per_idx.items():
                domain[idx] = term_to_idx[term]
            data["glycan"].taxonomy_domain = domain
            logger.info(
                "Attached glycan.taxonomy_domain labels from metadata: %d/%d (classes=%d)",
                len(function_term_per_idx),
                len(glycan_map),
                len(terms),
            )

    def extract_node_metadata(
        self, nodes_df: pd.DataFrame
    ) -> Dict[str, Dict[str, dict]]:
        """Parse JSON metadata for every node, grouped by type.

        Returns
        -------
        metadata : dict[str, dict[str, dict]]
            ``{node_type: {node_id: parsed_metadata, ...}, ...}``
        """
        result: Dict[str, Dict[str, dict]] = {}

        for _, row in nodes_df.iterrows():
            ntype = str(row["node_type"])
            nid = str(row["node_id"])
            raw = row.get("metadata", "{}")

            if pd.isna(raw) or raw == "":
                meta = {}
            else:
                try:
                    meta = json.loads(str(raw))
                except (json.JSONDecodeError, TypeError):
                    meta = {}

            result.setdefault(ntype, {})[nid] = meta

        return result

    # ------------------------------------------------------------------
    # Convenience: one-shot conversion
    # ------------------------------------------------------------------

    def convert(self, feature_dim: int = 256) -> Tuple[HeteroData, Dict[str, Dict[str, int]]]:
        """End-to-end conversion: load files, build mappings, build graph.

        Returns
        -------
        data : HeteroData
        node_mappings : dict
        """
        nodes_df, edges_df = self.load_dataframes()
        node_mappings = self.build_node_mappings(nodes_df)
        data = self.build_hetero_graph(nodes_df, edges_df, node_mappings, feature_dim)
        return data, node_mappings
