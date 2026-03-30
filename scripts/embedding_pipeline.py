#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Embedding pipeline entry-point for glycoMusubi.

Orchestrates three stages:
    1. featurize -- convert KG TSV/Parquet into HeteroData
    2. train     -- train KGE model and save checkpoint
    3. evaluate  -- filtered link-prediction evaluation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import pandas as pd

# Ensure project root import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from glycoMusubi.utils.config import ExperimentConfig, load_experiment_config
from glycoMusubi.utils.logging_setup import setup_logging
from glycoMusubi.utils.reproducibility import set_deterministic, set_seed

logger = logging.getLogger(__name__)

STAGES = ["featurize", "train", "evaluate", "evaluate_inductive", "downstream"]


def _dataset_payload_path(cfg: ExperimentConfig) -> Path:
    return cfg.run_dir / "dataset.pt"


def _load_dataset_payload(path: Path) -> Dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "data" in payload:
        return payload
    # Backward compatibility: plain HeteroData
    return {"data": payload, "node_mappings": {}}


def _build_edge_type_mapping(edge_types: List[Tuple[str, str, str]]) -> Dict[Tuple[str, str, str], int]:
    ordered = sorted(edge_types)
    return {etype: i for i, etype in enumerate(ordered)}


def _prune_invalid_edge_types(data) -> List[Tuple[str, str, str]]:
    """
    Remove edge types whose source/target node types are not present in data.node_types.
    Returns the list of removed edge types.
    """
    node_types = set(data.node_types)
    removed: List[Tuple[str, str, str]] = []
    for etype in list(data.edge_types):
        src_type, _, dst_type = etype
        if src_type not in node_types or dst_type not in node_types:
            del data[etype]
            removed.append(etype)
    return removed


def _filter_edge_types(data, allowed_edge_types: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    """Keep only edge types listed in allowed_edge_types."""
    allowed = set(allowed_edge_types)
    removed: List[Tuple[str, str, str]] = []
    for etype in list(data.edge_types):
        if etype not in allowed:
            del data[etype]
            removed.append(etype)
    return removed


def _attach_edge_type_indices(data, edge_type_to_idx: Dict[Tuple[str, str, str], int]) -> None:
    for etype in data.edge_types:
        ridx = edge_type_to_idx[etype]
        num_edges = int(data[etype].edge_index.size(1))
        data[etype].edge_type_idx = torch.full((num_edges,), ridx, dtype=torch.long)


def _build_site_id(uniprot_id: str, position: int, residue: str) -> str:
    return f"SITE::{uniprot_id}::{int(position)}::{str(residue)}"


def _load_positive_site_indices_from_uniprot(node_mappings: Dict[str, Dict[str, int]]) -> List[int]:
    site_map = node_mappings.get("site", {})
    if not site_map:
        return []
    path = Path("data_clean/uniprot_sites.tsv")
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path, sep="\t", low_memory=False)
    except Exception:
        return []

    site_ids: List[str] = []
    if "site_id" in df.columns:
        site_ids = [str(x).strip() for x in df["site_id"].dropna().astype(str)]
    else:
        required = {"uniprot_id", "site_position", "site_residue"}
        if not required.issubset(set(df.columns)):
            return []
        for _, row in df.iterrows():
            try:
                sid = _build_site_id(
                    str(row["uniprot_id"]).strip(),
                    int(row["site_position"]),
                    str(row["site_residue"]).strip(),
                )
                site_ids.append(sid)
            except Exception:
                continue

    out: List[int] = []
    for sid in site_ids:
        idx = site_map.get(sid)
        if idx is not None:
            out.append(int(idx))
    return sorted(set(out))


def _build_model(
    cfg: ExperimentConfig,
    num_nodes_dict: Dict[str, int],
    num_relations: int,
    *,
    wurcs_map: Optional[Dict[int, str]] = None,
    edge_types: Optional[List[Tuple[str, str, str]]] = None,
    site_positions_map: Optional[Dict[int, List]] = None,
    function_feature_map: Optional[Dict[int, torch.Tensor]] = None,
):
    from glycoMusubi.embedding.models import DistMult, RotatE, TransE

    name = str(cfg.model.name).lower()
    emb_dim = int(getattr(cfg.model, "embedding_dim", cfg.embedding_dim))

    if name == "transe":
        p_norm = int(getattr(cfg.model, "p_norm", 2))
        return TransE(
            num_nodes_dict=num_nodes_dict,
            num_relations=num_relations,
            embedding_dim=emb_dim,
            p_norm=p_norm,
        )
    if name == "distmult":
        return DistMult(
            num_nodes_dict=num_nodes_dict,
            num_relations=num_relations,
            embedding_dim=emb_dim,
        )
    if name == "rotate":
        return RotatE(
            num_nodes_dict=num_nodes_dict,
            num_relations=num_relations,
            embedding_dim=emb_dim,
        )
    if name == "glycokgnet":
        from glycoMusubi.embedding.models import GlycoKGNet

        text_node_types = getattr(cfg.model, "text_node_types", None)
        esm2_cache = getattr(cfg.model, "esm2_cache_path", None)

        return GlycoKGNet(
            num_nodes_dict=num_nodes_dict,
            num_relations=num_relations,
            embedding_dim=emb_dim,
            glycan_encoder_type=str(getattr(cfg.model, "glycan_encoder_type", "learnable")),
            protein_encoder_type=str(getattr(cfg.model, "protein_encoder_type", "learnable")),
            num_hgt_layers=int(getattr(cfg.model, "num_hgt_layers", 4)),
            num_hgt_heads=int(getattr(cfg.model, "num_hgt_heads", 8)),
            hgt_max_edges_per_type=int(getattr(cfg.model, "hgt_max_edges_per_type", 0)),
            use_bio_prior=bool(getattr(cfg.model, "use_bio_prior", True)),
            use_cross_modal_fusion=bool(getattr(cfg.model, "use_cross_modal_fusion", True)),
            num_fusion_heads=int(getattr(cfg.model, "num_fusion_heads", 4)),
            decoder_type=str(getattr(cfg.model, "decoder_type", "hybrid")),
            dropout=float(getattr(cfg.model, "dropout", 0.1)),
            esm2_cache_path=esm2_cache,
            wurcs_map=wurcs_map,
            text_node_types=text_node_types,
            edge_types=edge_types,
            site_positions_map=site_positions_map,
            function_feature_map=function_feature_map,
        )

    raise ValueError(
        f"Unsupported model.name={cfg.model.name!r} in this pipeline. "
        "Use TransE / DistMult / RotatE / GlycoKGNet."
    )


def _build_optimizer(cfg: ExperimentConfig, model: torch.nn.Module):
    name = str(cfg.training.optimizer).lower()
    lr = float(cfg.training.lr)
    wd = float(cfg.training.weight_decay)

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adagrad":
        return torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)

    raise ValueError(f"Unsupported training.optimizer={cfg.training.optimizer!r}")


def _hetero_to_global_triples(data, edge_type_to_idx: Dict[Tuple[str, str, str], int]):
    offsets: Dict[str, Tuple[int, int]] = {}
    start = 0
    for ntype in sorted(data.node_types):
        num_nodes = int(getattr(data[ntype], "num_nodes", 0) or data[ntype].x.size(0))
        offsets[ntype] = (start, num_nodes)
        start += num_nodes

    triples = []
    for etype in data.edge_types:
        s_type, _, t_type = etype
        ridx = edge_type_to_idx[etype]
        ei = data[etype].edge_index
        s_off, _ = offsets[s_type]
        t_off, _ = offsets[t_type]
        src = ei[0].cpu().long() + s_off
        dst = ei[1].cpu().long() + t_off
        rel = torch.full((ei.size(1),), ridx, dtype=torch.long)
        triples.append(torch.stack([src, rel, dst], dim=1))

    if triples:
        return torch.cat(triples, dim=0), offsets
    return torch.empty((0, 3), dtype=torch.long), offsets


class _HeteroScoringAdapter:
    """Adapter exposing score_t/score_h over global entity IDs.

    For models with a hybrid decoder (e.g., GlycoKGNet with HybridLinkScorer),
    relation *indices* are passed instead of embedding vectors so the full
    hybrid scoring path (DistMult + RotatE + Neural + Poincaré) is used.
    """

    def __init__(self, model, data, edge_type_to_idx: Dict[Tuple[str, str, str], int], device: str):
        self.model = model.to(device)
        self.data = data.to(device)
        self.device = torch.device(device)
        self.edge_type_to_idx = edge_type_to_idx

        self.idx_to_edge_type = {v: k for k, v in edge_type_to_idx.items()}
        self.offsets: Dict[str, Tuple[int, int]] = {}
        start = 0
        for ntype in sorted(self.data.node_types):
            num_nodes = int(getattr(self.data[ntype], "num_nodes", 0) or self.data[ntype].x.size(0))
            self.offsets[ntype] = (start, num_nodes)
            start += num_nodes
        self.num_entities = start

        # Detect hybrid decoder for index-based scoring
        self._use_hybrid = hasattr(model, "_decoder_type") and model._decoder_type == "hybrid"

        self.model.eval()
        with torch.no_grad():
            self.emb_dict = self.model(self.data)

    def _global_to_local(self, node_type: str, global_idx: torch.Tensor) -> torch.Tensor:
        start, size = self.offsets[node_type]
        local = global_idx - start
        return local.clamp(min=0, max=max(size - 1, 0))

    @torch.no_grad()
    def score_t(self, head: torch.Tensor, relation: torch.Tensor, num_entities: int) -> torch.Tensor:
        """Score all tail candidates. Groups triples by relation for batched scoring."""
        batch = head.size(0)
        out = torch.full((batch, num_entities), -1e9, device=self.device)

        head_dev = head.to(self.device)
        rel_dev = relation.to(self.device)

        # Group batch indices by relation for efficient batched processing
        rel_groups: Dict[int, List[int]] = {}
        for i in range(batch):
            ridx = int(rel_dev[i].item())
            rel_groups.setdefault(ridx, []).append(i)

        for ridx, indices in rel_groups.items():
            s_type, _, t_type = self.idx_to_edge_type[ridx]
            idx_t = torch.tensor(indices, dtype=torch.long, device=self.device)
            group_heads = head_dev[idx_t]

            h_local = self._global_to_local(s_type, group_heads)
            h_embs = self.emb_dict[s_type][h_local]  # [G, dim]

            tails = self.emb_dict[t_type]  # [T, dim]
            num_tails = tails.size(0)
            t_start, _ = self.offsets[t_type]

            # Score each head in the group against all tails
            for j, bi in enumerate(indices):
                h_emb = h_embs[j:j+1].expand(num_tails, -1)
                if self._use_hybrid:
                    rep_r = torch.full((num_tails,), ridx, dtype=torch.long, device=self.device)
                else:
                    rel_idx_t = torch.tensor([ridx], dtype=torch.long, device=self.device)
                    rep_r = self.model.get_relation_embedding(rel_idx_t).expand(num_tails, -1)
                scores = self.model.score(h_emb, rep_r, tails)
                out[bi, t_start:t_start + num_tails] = scores

        return out

    @torch.no_grad()
    def score_h(self, tail: torch.Tensor, relation: torch.Tensor, num_entities: int) -> torch.Tensor:
        """Score all head candidates. Groups triples by relation for batched scoring."""
        batch = tail.size(0)
        out = torch.full((batch, num_entities), -1e9, device=self.device)

        tail_dev = tail.to(self.device)
        rel_dev = relation.to(self.device)

        rel_groups: Dict[int, List[int]] = {}
        for i in range(batch):
            ridx = int(rel_dev[i].item())
            rel_groups.setdefault(ridx, []).append(i)

        for ridx, indices in rel_groups.items():
            s_type, _, t_type = self.idx_to_edge_type[ridx]
            idx_t = torch.tensor(indices, dtype=torch.long, device=self.device)
            group_tails = tail_dev[idx_t]

            t_local = self._global_to_local(t_type, group_tails)
            t_embs = self.emb_dict[t_type][t_local]  # [G, dim]

            heads = self.emb_dict[s_type]  # [H, dim]
            num_heads = heads.size(0)
            s_start, _ = self.offsets[s_type]

            for j, bi in enumerate(indices):
                t_emb = t_embs[j:j+1].expand(num_heads, -1)
                if self._use_hybrid:
                    rep_r = torch.full((num_heads,), ridx, dtype=torch.long, device=self.device)
                else:
                    rel_idx_t = torch.tensor([ridx], dtype=torch.long, device=self.device)
                    rep_r = self.model.get_relation_embedding(rel_idx_t).expand(num_heads, -1)
                scores = self.model.score(heads, rep_r, t_emb)
                out[bi, s_start:s_start + num_heads] = scores

        return out


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def run_featurize(cfg: ExperimentConfig) -> None:
    """Convert KG files into HeteroData and persist as dataset payload."""
    from glycoMusubi.data import KGConverter

    kg_dir = Path(cfg.data.kg_dir)
    logger.info("Featurize: loading KG from %s", kg_dir)

    converter = KGConverter(kg_dir=str(kg_dir))
    data, node_mappings = converter.convert(feature_dim=cfg.embedding_dim)

    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "data": data,
        "node_mappings": node_mappings,
    }
    out_path = _dataset_payload_path(cfg)
    torch.save(payload, out_path)

    num_edges = sum(int(data[etype].edge_index.size(1)) for etype in data.edge_types)
    logger.info(
        "Featurize: saved %s (node_types=%d, edge_types=%d, edges=%d)",
        out_path,
        len(data.node_types),
        len(data.edge_types),
        num_edges,
    )


def _build_inverse_relation_map() -> Dict[str, str]:
    """Load inverse relation pairs from edge schema."""
    import yaml

    schema_path = Path("schemas/edge_schema.yaml")
    if not schema_path.exists():
        return {}
    try:
        with open(schema_path) as f:
            schema = yaml.safe_load(f)
    except Exception:
        return {}

    inv_map: Dict[str, str] = {}
    relations = schema.get("edge_types", schema.get("relations", schema.get("edges", {})))
    if isinstance(relations, dict):
        for rel_name, rel_def in relations.items():
            if isinstance(rel_def, dict) and "inverse" in rel_def:
                inv = rel_def["inverse"]
                inv_map[rel_name] = inv
                inv_map[inv] = rel_name
    return inv_map


def _resolve_checkpoint(run_dir: Path) -> Path:
    """Return best.pt if available, otherwise last.pt."""
    best = run_dir / "best.pt"
    if best.exists():
        return best
    last = run_dir / "last.pt"
    if last.exists():
        return last
    raise FileNotFoundError(f"No checkpoint found in {run_dir}")


def _build_wurcs_map(node_mappings: Dict[str, Dict[str, int]]) -> Optional[Dict[int, str]]:
    """Build glycan index → WURCS string map from nodes.tsv metadata."""
    glycan_map = node_mappings.get("glycan", {})
    if not glycan_map:
        return None

    nodes_path = Path("kg/nodes.tsv")
    if not nodes_path.exists():
        return None

    try:
        df = pd.read_csv(nodes_path, sep="\t", low_memory=False)
    except Exception:
        return None

    id_col = "node_id" if "node_id" in df.columns else df.columns[0]
    glycan_rows = df[df.get("node_type", df.get("type", "")) == "glycan"]
    if glycan_rows.empty:
        glycan_rows = df[df[id_col].isin(glycan_map.keys())]

    # Find WURCS: try dedicated column first, then parse from metadata JSON
    wurcs_col = None
    for col in ("wurcs", "WURCS", "structure"):
        if col in df.columns:
            wurcs_col = col
            break

    wurcs_map: Dict[int, str] = {}
    for _, row in glycan_rows.iterrows():
        node_id = str(row[id_col]).strip()
        if node_id not in glycan_map:
            continue

        wurcs = ""
        if wurcs_col is not None:
            wurcs = str(row[wurcs_col]).strip() if pd.notna(row[wurcs_col]) else ""

        # Fallback: extract WURCS from metadata JSON
        if (not wurcs or wurcs == "nan") and "metadata" in df.columns and pd.notna(row.get("metadata")):
            try:
                meta = json.loads(str(row["metadata"]))
                wurcs = str(meta.get("WURCS", meta.get("wurcs", meta.get("structure", "")))).strip()
            except (json.JSONDecodeError, TypeError):
                pass

        if wurcs and wurcs != "nan":
            wurcs_map[glycan_map[node_id]] = wurcs

    return wurcs_map if wurcs_map else None


def _build_site_positions_map(
    node_mappings: Dict[str, Dict[str, int]],
    site_map_path: Optional[str] = None,
) -> Optional[Dict[int, List[int]]]:
    """Build protein_index -> list_of_site_positions from site_positions_map.pt.

    Converts UniProt ID-based site map to integer-indexed map matching
    the ProteinEncoder's expected format.
    """
    if site_map_path is None:
        return None
    path = Path(site_map_path)
    if not path.exists():
        logger.warning("Site map file not found: %s", path)
        return None

    protein_map = node_mappings.get("protein", {})
    if not protein_map:
        return None

    raw = torch.load(path, map_location="cpu", weights_only=False)
    protein_to_sites = raw.get("protein_to_sites", {})

    result: Dict[int, List[int]] = {}
    for uniprot_id, sites_info in protein_to_sites.items():
        idx = protein_map.get(uniprot_id)
        if idx is None:
            base = uniprot_id.split("-")[0]
            idx = protein_map.get(base)
        if idx is None:
            continue

        # Convert to 0-indexed positions for ESM2 embedding indexing
        # Store enriched tuples: (0-indexed_pos, residue, site_type)
        sites = [(int(pos) - 1, res, stype) for pos, res, stype in sites_info]
        if sites:
            result[idx] = sites

    return result


def _build_glycan_function_index(
    node_mappings: Dict[str, Dict[str, int]],
    function_labels_path: str = "data_clean/glycan_function_labels.tsv",
) -> Optional[Dict[str, List[int]]]:
    """Build ``{function_term: [glycan_local_indices]}`` from TSV.

    Reads glycan function labels and maps glycan IDs to their local
    integer indices in the KG node embedding table.
    """
    glycan_map = node_mappings.get("glycan", {})
    if not glycan_map:
        logger.warning("No glycan node mappings found; skipping function index.")
        return None

    path = Path(function_labels_path)
    if not path.exists():
        logger.warning("Function labels file not found: %s", path)
        return None

    try:
        df = pd.read_csv(path, sep="\t", low_memory=False)
    except Exception:
        logger.warning("Failed to read function labels: %s", path, exc_info=True)
        return None

    if "glycan_id" not in df.columns or "function_term" not in df.columns:
        logger.warning("Function labels TSV missing glycan_id or function_term column")
        return None

    result: Dict[str, List[int]] = {}
    for _, row in df.iterrows():
        gid = str(row["glycan_id"]).strip()
        func = str(row["function_term"]).strip()
        idx = glycan_map.get(gid)
        if idx is not None and func:
            result.setdefault(func, []).append(idx)

    if result:
        total = len({i for indices in result.values() for i in indices})
        logger.info(
            "Built glycan function index: %d terms, %d unique glycans",
            len(result), total,
        )
    return result if result else None


def _build_glycan_function_feature_map(
    node_mappings: Dict[str, Dict[str, int]],
    function_labels_path: str = "data_clean/glycan_function_labels.tsv",
) -> Optional[Dict[int, torch.Tensor]]:
    """Build ``{glycan_idx: multi_hot_tensor[8]}`` for function feature encoding."""
    from glycoMusubi.embedding.encoders.glycan_encoder import GlycanEncoder

    categories = GlycanEncoder.FUNCTION_CATEGORIES
    cat_to_idx = {c: i for i, c in enumerate(categories)}

    glycan_map = node_mappings.get("glycan", {})
    if not glycan_map:
        return None

    path = Path(function_labels_path)
    if not path.exists():
        return None

    try:
        df = pd.read_csv(path, sep="\t", low_memory=False)
    except Exception:
        return None

    if "glycan_id" not in df.columns or "function_term" not in df.columns:
        return None

    result: Dict[int, torch.Tensor] = {}
    for _, row in df.iterrows():
        gid = str(row["glycan_id"]).strip()
        func = str(row["function_term"]).strip()
        idx = glycan_map.get(gid)
        cat_idx = cat_to_idx.get(func)
        if idx is not None and cat_idx is not None:
            if idx not in result:
                result[idx] = torch.zeros(len(categories))
            result[idx][cat_idx] = 1.0

    if result:
        logger.info("Built glycan function feature map: %d glycans", len(result))
    return result if result else None


def _build_protein_function_type_map(
    node_mappings: Dict[str, Dict[str, int]],
    site_data_path: str = "data_clean/uniprot_sites.tsv",
) -> Optional[Dict[int, str]]:
    """Build ``{protein_local_index: "N-linked"|"O-linked"|"unknown"}`` from site type column.

    Infers the dominant glycosylation type per protein from UniProt site data.
    """
    protein_map = node_mappings.get("protein", {})
    if not protein_map:
        return None

    path = Path(site_data_path)
    if not path.exists():
        return None

    try:
        df = pd.read_csv(path, sep="\t", low_memory=False)
    except Exception:
        return None

    required = {"uniprot_id", "site_type"}
    if not required.issubset(set(df.columns)):
        return None

    # Build base_id -> idx reverse lookup (protein_map keys are isoform IDs like P01588-1)
    base_to_idx: Dict[str, int] = {}
    for key, idx in protein_map.items():
        base = key.split("-")[0]
        base_to_idx[base] = idx
        base_to_idx[key] = idx

    # Collect site types per protein
    from collections import Counter
    protein_types: Dict[int, Counter] = {}
    for _, row in df.iterrows():
        uid = str(row["uniprot_id"]).strip()
        stype = str(row["site_type"]).strip()
        idx = base_to_idx.get(uid)
        if idx is not None and stype:
            protein_types.setdefault(idx, Counter())[stype] += 1

    # Pick dominant type per protein
    result: Dict[int, str] = {}
    for idx, counter in protein_types.items():
        dominant = counter.most_common(1)[0][0]
        if dominant in ("N-linked", "O-linked", "C-linked"):
            result[idx] = dominant
        else:
            result[idx] = "unknown"

    if result:
        logger.info("Built protein function type map: %d proteins", len(result))
    return result if result else None


def _build_loss_fn(cfg: ExperimentConfig):
    """Build loss function based on config."""
    from glycoMusubi.losses import MarginRankingLoss, BCEWithLogitsKGELoss, CompositeLoss

    loss_name = str(getattr(cfg.model, "loss_fn", "margin")).lower()

    if loss_name == "margin":
        return MarginRankingLoss(margin=float(getattr(cfg.model, "margin", 1.0)))

    if loss_name == "bce":
        adv_temp = getattr(cfg.model, "adversarial_temperature", None)
        label_smooth = float(getattr(cfg.model, "label_smoothing", 0.0))
        return BCEWithLogitsKGELoss(
            adversarial_temperature=float(adv_temp) if adv_temp is not None else None,
            label_smoothing=label_smooth,
        )

    if loss_name == "composite":
        adv_temp = getattr(cfg.model, "adversarial_temperature", None)
        label_smooth = float(getattr(cfg.model, "label_smoothing", 0.0))
        link_loss = BCEWithLogitsKGELoss(
            adversarial_temperature=float(adv_temp) if adv_temp is not None else None,
            label_smoothing=label_smooth,
        )
        return CompositeLoss(
            link_loss=link_loss,
            lambda_struct=float(getattr(cfg.model, "lambda_struct", 0.1)),
            lambda_hyp=float(getattr(cfg.model, "lambda_hyp", 0.01)),
            lambda_reg=float(getattr(cfg.model, "lambda_reg", 0.01)),
        )

    raise ValueError(f"Unsupported model.loss_fn={loss_name!r}. Use margin / bce / composite.")


def _build_scheduler(cfg: ExperimentConfig, optimizer):
    """Build LR scheduler based on config."""
    sched_name = str(cfg.training.scheduler).lower()
    if sched_name == "none":
        return None
    if sched_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(cfg.training.epochs),
        )
    if sched_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(cfg.training.scheduler_step_size),
            gamma=float(cfg.training.scheduler_gamma),
        )
    return None


def run_train(cfg: ExperimentConfig) -> None:
    """Train a KGE model and save checkpoint."""
    from glycoMusubi.data import random_link_split
    from glycoMusubi.training.trainer import Trainer
    from glycoMusubi.training.callbacks import EarlyStopping, MetricsLogger, ModelCheckpoint

    run_dir = cfg.run_dir
    payload_path = _dataset_payload_path(cfg)
    logger.info("Train: loading dataset payload from %s", payload_path)
    payload = _load_dataset_payload(payload_path)
    data = payload["data"]
    node_mappings = payload.get("node_mappings", {})

    removed_full = _prune_invalid_edge_types(data)
    if removed_full:
        logger.warning("Train: pruned edge types with missing node stores: %s", sorted(set(removed_full)))

    # Build inverse relation map for leak prevention
    inverse_relation_map = _build_inverse_relation_map()
    if inverse_relation_map:
        logger.info("Train: loaded %d inverse relation pairs for leak prevention", len(inverse_relation_map) // 2)

    split_seed = cfg.data.split_seed if cfg.data.split_seed is not None else cfg.seed
    train_data, val_data, test_data = random_link_split(
        data,
        val_ratio=cfg.data.validation_ratio,
        test_ratio=cfg.data.test_ratio,
        seed=split_seed,
        inverse_relation_map=inverse_relation_map if inverse_relation_map else None,
    )
    removed_train = _prune_invalid_edge_types(train_data)
    removed_val = _prune_invalid_edge_types(val_data)
    removed_test = _prune_invalid_edge_types(test_data)
    if removed_train or removed_val or removed_test:
        removed_all = sorted(set(removed_train + removed_val + removed_test))
        logger.warning("Pruned edge types with missing node stores: %s", removed_all)

    # Keep relation indices stable across train/eval/downstream by defining
    # them from the full pruned graph (not from split-specific edge subsets).
    edge_type_to_idx = _build_edge_type_mapping(list(data.edge_types))
    _attach_edge_type_indices(train_data, edge_type_to_idx)
    _attach_edge_type_indices(val_data, edge_type_to_idx)
    _attach_edge_type_indices(test_data, edge_type_to_idx)

    num_nodes_dict = {
        ntype: int(getattr(train_data[ntype], "num_nodes", 0) or train_data[ntype].x.size(0))
        for ntype in train_data.node_types
    }

    # Build WURCS map for glycan encoder if requested
    wurcs_map = None
    model_name = str(cfg.model.name).lower()
    glycan_enc = str(getattr(cfg.model, "glycan_encoder_type", "learnable")).lower()
    if model_name == "glycokgnet" and glycan_enc in ("wurcs_features", "hybrid"):
        wurcs_map = _build_wurcs_map(node_mappings)
        if wurcs_map:
            logger.info("Train: built WURCS map with %d glycan entries", len(wurcs_map))
        else:
            logger.warning("Train: WURCS map requested but no WURCS data found, falling back to learnable")

    # Build site positions map for protein encoder if esm2_site_aware
    site_positions_map = None
    protein_enc = str(getattr(cfg.model, "protein_encoder_type", "learnable")).lower()
    if model_name == "glycokgnet" and protein_enc == "esm2_site_aware":
        site_map_path = getattr(cfg.model, "site_map_path", None)
        site_positions_map = _build_site_positions_map(node_mappings, site_map_path)
        if site_positions_map:
            logger.info("Train: built site positions map with %d proteins", len(site_positions_map))

    # Build glycan function feature map if requested
    function_feature_map = None
    if model_name == "glycokgnet" and bool(getattr(cfg.model, "use_glycan_function_features", False)):
        function_feature_map = _build_glycan_function_feature_map(node_mappings)

    model = _build_model(
        cfg,
        num_nodes_dict=num_nodes_dict,
        num_relations=len(edge_type_to_idx),
        wurcs_map=wurcs_map,
        edge_types=list(edge_type_to_idx.keys()),
        site_positions_map=site_positions_map,
        function_feature_map=function_feature_map,
    )

    optimizer = _build_optimizer(cfg, model)
    loss_fn = _build_loss_fn(cfg)
    scheduler = _build_scheduler(cfg, optimizer)

    grad_clip = float(cfg.training.grad_clip)
    use_hgt = bool(getattr(cfg.training, "use_hgt_loader", False))
    hgt_batch_size = int(getattr(cfg.training, "hgt_batch_size", 1024))
    hgt_num_samples = getattr(cfg.training, "hgt_num_samples", None)
    accum_steps = int(getattr(cfg.training, "gradient_accumulation_steps", 1))

    # Callbacks: logging, checkpointing, early stopping
    early_patience = int(getattr(cfg.training, "early_stopping_patience", 0))
    callbacks = [
        MetricsLogger(log_file=run_dir / "train_metrics.jsonl"),
        ModelCheckpoint(dirpath=run_dir, monitor="loss", mode="min"),
    ]
    if early_patience > 0 and val_data is not None:
        callbacks.append(
            EarlyStopping(monitor="loss", patience=early_patience, mode="min")
        )

    # Enable mixed precision when configured (bfloat16 preferred on modern GPUs)
    use_mixed = bool(getattr(cfg.training, "mixed_precision", False))
    amp_dtype_name = str(getattr(cfg.training, "amp_dtype", "bfloat16"))
    amp_dtype = torch.bfloat16 if amp_dtype_name == "bfloat16" else torch.float16

    # Function-aware negative pool restriction for has_glycan
    neg_pool_restrictor = None
    if bool(getattr(cfg.training, "function_aware_negatives", False)):
        # Extract glycan indices that actually appear as tails in has_glycan edges
        has_glycan_tail_indices: set[int] = set()
        for etype in train_data.edge_types:
            _, rel_name, dst_type = etype
            if rel_name == "has_glycan" and dst_type == "glycan":
                tail_idx = train_data[etype].edge_index[1].tolist()
                has_glycan_tail_indices.update(tail_idx)
        if has_glycan_tail_indices:
            from glycoMusubi.data.sampler import FunctionPoolRestrictor
            func_map = {"has_glycan_tails": sorted(has_glycan_tail_indices)}
            neg_pool_restrictor = FunctionPoolRestrictor(func_map)
            logger.info(
                "Train: function-aware negatives enabled (%d glycans from has_glycan tails)",
                len(has_glycan_tail_indices),
            )
        else:
            logger.warning("Train: function_aware_negatives=True but no has_glycan tails found")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_data=train_data,
        val_data=val_data,
        device=cfg.resolve_device(),
        scheduler=scheduler,
        callbacks=callbacks,
        grad_clip_norm=(grad_clip if grad_clip > 0 else None),
        use_mini_batch=use_hgt,
        use_hgt_loader=use_hgt,
        hgt_batch_size=hgt_batch_size,
        hgt_num_samples=hgt_num_samples if hgt_num_samples else [512, 256],
        gradient_accumulation_steps=accum_steps,
        relation_balance_alpha=float(getattr(cfg.training, "relation_balance_alpha", 0.0)),
        relation_balance_max_weight=float(getattr(cfg.training, "relation_balance_max_weight", 5.0)),
        max_edges_per_type=int(getattr(cfg.training, "max_edges_per_type", 0)),
        num_negatives=int(getattr(cfg.training, "num_negatives", 1)),
        mixed_precision=use_mixed,
        amp_dtype=amp_dtype,
        neg_pool_restrictor=neg_pool_restrictor,
    )

    history = trainer.fit(
        epochs=int(cfg.training.epochs),
        validate_every=max(1, int(cfg.training.eval_every)),
    )

    # Ensure final checkpoint is saved (ModelCheckpoint callback saves
    # during training, but this guarantees a last.pt exists).
    ckpt_path = run_dir / "last.pt"
    if not ckpt_path.exists():
        trainer.save_checkpoint(ckpt_path)

    eval_payload = {
        "edge_type_to_idx": edge_type_to_idx,
        "split_seed": split_seed,
        "train_edge_counts": {
            f"{etype[0]}::{etype[1]}::{etype[2]}": int(train_data[etype].edge_index.size(1))
            for etype in train_data.edge_types
        },
    }
    torch.save(eval_payload, run_dir / "eval_payload.pt")

    with open(run_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info("Train: complete. checkpoint=%s", ckpt_path)


def run_evaluate(cfg: ExperimentConfig) -> None:
    """Evaluate a trained model using filtered link prediction."""
    from glycoMusubi.data import random_link_split
    from glycoMusubi.evaluation.link_prediction import LinkPredictionEvaluator

    run_dir = cfg.run_dir
    payload = _load_dataset_payload(_dataset_payload_path(cfg))
    data = payload["data"]
    removed_full = _prune_invalid_edge_types(data)
    if removed_full:
        logger.warning("Evaluate: pruned edge types with missing node stores: %s", sorted(set(removed_full)))

    trained_edge_type_to_idx = None
    eval_payload_path = run_dir / "eval_payload.pt"
    if eval_payload_path.exists():
        eval_payload = torch.load(eval_payload_path, map_location="cpu", weights_only=False)
        split_seed = int(eval_payload.get("split_seed", cfg.seed))
        trained_edge_type_to_idx = eval_payload.get("edge_type_to_idx")
    else:
        split_seed = cfg.data.split_seed if cfg.data.split_seed is not None else cfg.seed

    if trained_edge_type_to_idx:
        allowed = list(trained_edge_type_to_idx.keys())
        removed_unseen = _filter_edge_types(data, allowed)
        if removed_unseen:
            logger.warning(
                "Evaluate: removed edge types not seen in training: %s",
                sorted(set(removed_unseen)),
            )
        edge_type_to_idx = trained_edge_type_to_idx
    else:
        edge_type_to_idx = _build_edge_type_mapping(list(data.edge_types))

    # Build inverse relation map for leak prevention
    inverse_relation_map = _build_inverse_relation_map()
    if inverse_relation_map:
        logger.info("Evaluate: loaded %d inverse relation pairs for leak prevention", len(inverse_relation_map) // 2)

    train_data, val_data, test_data = random_link_split(
        data,
        val_ratio=cfg.data.validation_ratio,
        test_ratio=cfg.data.test_ratio,
        seed=split_seed,
        inverse_relation_map=inverse_relation_map if inverse_relation_map else None,
    )

    removed_train = _prune_invalid_edge_types(train_data)
    removed_val = _prune_invalid_edge_types(val_data)
    removed_test = _prune_invalid_edge_types(test_data)
    if removed_train or removed_val or removed_test:
        removed_all = sorted(set(removed_train + removed_val + removed_test))
        logger.warning("Pruned edge types with missing node stores: %s", removed_all)
    _attach_edge_type_indices(train_data, edge_type_to_idx)
    _attach_edge_type_indices(val_data, edge_type_to_idx)
    _attach_edge_type_indices(test_data, edge_type_to_idx)

    num_nodes_dict = {
        ntype: int(getattr(train_data[ntype], "num_nodes", 0) or train_data[ntype].x.size(0))
        for ntype in train_data.node_types
    }
    node_mappings = payload.get("node_mappings", {})
    wurcs_map = None
    model_name = str(cfg.model.name).lower()
    glycan_enc = str(getattr(cfg.model, "glycan_encoder_type", "learnable")).lower()
    if model_name == "glycokgnet" and glycan_enc in ("wurcs_features", "hybrid"):
        wurcs_map = _build_wurcs_map(node_mappings)

    # Build site positions map for protein encoder if esm2_site_aware
    site_positions_map = None
    protein_enc = str(getattr(cfg.model, "protein_encoder_type", "learnable")).lower()
    if model_name == "glycokgnet" and protein_enc == "esm2_site_aware":
        site_map_path = getattr(cfg.model, "site_map_path", None)
        site_positions_map = _build_site_positions_map(node_mappings, site_map_path)
        if site_positions_map:
            logger.info("Evaluate: built site positions map with %d proteins", len(site_positions_map))

    # Build glycan function feature map if requested
    function_feature_map = None
    if model_name == "glycokgnet" and bool(getattr(cfg.model, "use_glycan_function_features", False)):
        function_feature_map = _build_glycan_function_feature_map(node_mappings)

    model = _build_model(
        cfg,
        num_nodes_dict=num_nodes_dict,
        num_relations=len(edge_type_to_idx),
        wurcs_map=wurcs_map,
        edge_types=list(edge_type_to_idx.keys()),
        site_positions_map=site_positions_map,
        function_feature_map=function_feature_map,
    )

    ckpt_path = _resolve_checkpoint(run_dir)
    logger.info("Evaluate: loading checkpoint from %s", ckpt_path.name)
    state = torch.load(ckpt_path, map_location=cfg.resolve_device(), weights_only=False)
    missing, unexpected = model.load_state_dict(state["model_state_dict"], strict=False)
    if missing:
        logger.warning("Evaluate: checkpoint missing %d keys (using defaults): %s", len(missing), missing)
    if unexpected:
        logger.warning("Evaluate: checkpoint has %d unexpected keys: %s", len(unexpected), unexpected)

    all_triples, _ = _hetero_to_global_triples(data, edge_type_to_idx)
    test_triples, _ = _hetero_to_global_triples(test_data, edge_type_to_idx)

    # Optionally subsample test triples for faster evaluation
    max_eval_triples = int(getattr(cfg.data, "max_eval_triples", 0))
    if max_eval_triples > 0 and test_triples.size(0) > max_eval_triples:
        logger.info(
            "Evaluate: subsampling test triples %d -> %d",
            test_triples.size(0), max_eval_triples,
        )
        perm = torch.randperm(test_triples.size(0))[:max_eval_triples]
        test_triples = test_triples[perm]

    adapter = _HeteroScoringAdapter(
        model=model,
        data=data,
        edge_type_to_idx=edge_type_to_idx,
        device=cfg.resolve_device(),
    )

    relation_names = {
        idx: f"{etype[0]}::{etype[1]}::{etype[2]}"
        for etype, idx in edge_type_to_idx.items()
    }

    logger.info(
        "Evaluate: scoring %d test triples against %d entities",
        test_triples.size(0), adapter.num_entities,
    )

    evaluator = LinkPredictionEvaluator(batch_size=max(1, int(cfg.training.batch_size)))
    result = evaluator.evaluate(
        model=adapter,
        test_triples=test_triples,
        all_triples=all_triples,
        num_entities=adapter.num_entities,
        relation_names=relation_names,
        per_relation=True,
    )

    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "metrics": result.metrics,
            "head_metrics": result.head_metrics,
            "tail_metrics": result.tail_metrics,
            "per_relation": result.per_relation,
            "num_triples": result.num_triples,
        }, f, indent=2)

    logger.info(
        "Evaluate: MRR=%.4f Hits@1=%.4f Hits@10=%.4f (triples=%d)",
        result.metrics.get("mrr", 0.0),
        result.metrics.get("hits@1", 0.0),
        result.metrics.get("hits@10", 0.0),
        result.num_triples,
    )
    logger.info("Evaluate: metrics saved to %s", metrics_path)


def run_downstream(cfg: ExperimentConfig) -> None:
    """Run downstream task suite using trained embeddings."""
    from glycoMusubi.data import random_link_split
    from glycoMusubi.evaluation import DownstreamEvaluator
    from glycoMusubi.evaluation.tasks import (
        BindingSiteTask,
        DiseaseAssociationTask,
        DrugTargetTask,
        GlycanFunctionTask,
        GlycanProteinInteractionTask,
        ImmunogenicityTask,
    )

    run_dir = cfg.run_dir
    payload = _load_dataset_payload(_dataset_payload_path(cfg))
    data = payload["data"]
    node_mappings = payload.get("node_mappings", {})

    # Match relation space used during training when available.
    eval_payload_path = run_dir / "eval_payload.pt"
    trained_edge_type_to_idx = None
    if eval_payload_path.exists():
        eval_payload = torch.load(eval_payload_path, map_location="cpu", weights_only=False)
        trained_edge_type_to_idx = eval_payload.get("edge_type_to_idx")

    removed_invalid = _prune_invalid_edge_types(data)
    if removed_invalid:
        logger.warning("Downstream: pruned invalid edge types: %s", sorted(set(removed_invalid)))

    if trained_edge_type_to_idx:
        allowed = list(trained_edge_type_to_idx.keys())
        removed_unseen = _filter_edge_types(data, allowed)
        if removed_unseen:
            logger.warning(
                "Downstream: removed edge types not seen in training: %s",
                sorted(set(removed_unseen)),
            )
        edge_type_to_idx = trained_edge_type_to_idx
    else:
        edge_type_to_idx = _build_edge_type_mapping(list(data.edge_types))

    eval_payload = {}
    if eval_payload_path.exists():
        eval_payload = torch.load(eval_payload_path, map_location="cpu", weights_only=False)
    split_seed = int(eval_payload.get("split_seed", cfg.seed))

    # Build inverse relation map for leak prevention
    inverse_relation_map = _build_inverse_relation_map()
    if inverse_relation_map:
        logger.info("Downstream: loaded %d inverse relation pairs for leak prevention", len(inverse_relation_map) // 2)

    train_split, _, test_split = random_link_split(
        data,
        val_ratio=cfg.data.validation_ratio,
        test_ratio=cfg.data.test_ratio,
        seed=split_seed,
        inverse_relation_map=inverse_relation_map if inverse_relation_map else None,
    )
    _prune_invalid_edge_types(train_split)
    _prune_invalid_edge_types(test_split)

    num_nodes_dict = {
        ntype: int(getattr(data[ntype], "num_nodes", 0) or data[ntype].x.size(0))
        for ntype in data.node_types
    }

    wurcs_map = None
    model_name_lower = str(cfg.model.name).lower()
    glycan_enc = str(getattr(cfg.model, "glycan_encoder_type", "learnable")).lower()
    if model_name_lower == "glycokgnet" and glycan_enc in ("wurcs_features", "hybrid"):
        wurcs_map = _build_wurcs_map(node_mappings)

    # Build site positions map for protein encoder if esm2_site_aware
    site_positions_map = None
    protein_enc = str(getattr(cfg.model, "protein_encoder_type", "learnable")).lower()
    if model_name_lower == "glycokgnet" and protein_enc == "esm2_site_aware":
        site_map_path = getattr(cfg.model, "site_map_path", None)
        site_positions_map = _build_site_positions_map(node_mappings, site_map_path)
        if site_positions_map:
            logger.info("Downstream: built site positions map with %d proteins", len(site_positions_map))

    # Build glycan function feature map if requested
    function_feature_map = None
    if model_name_lower == "glycokgnet" and bool(getattr(cfg.model, "use_glycan_function_features", False)):
        function_feature_map = _build_glycan_function_feature_map(node_mappings)

    model = _build_model(
        cfg,
        num_nodes_dict=num_nodes_dict,
        num_relations=len(edge_type_to_idx),
        wurcs_map=wurcs_map,
        edge_types=list(edge_type_to_idx.keys()),
        site_positions_map=site_positions_map,
        function_feature_map=function_feature_map,
    )

    ckpt_path = _resolve_checkpoint(run_dir)
    logger.info("Evaluate inductive: loading checkpoint from %s", ckpt_path.name)
    state = torch.load(ckpt_path, map_location=cfg.resolve_device(), weights_only=False)
    missing, unexpected = model.load_state_dict(state["model_state_dict"], strict=False)
    if missing:
        logger.warning("Evaluate inductive: checkpoint missing %d keys (using defaults): %s", len(missing), missing)
    if unexpected:
        logger.warning("Evaluate inductive: checkpoint has %d unexpected keys: %s", len(unexpected), unexpected)
    model = model.to(cfg.resolve_device()).eval()

    with torch.no_grad():
        # Avoid leakage in message-passing models: derive embeddings from train split only.
        embeddings = model.get_embeddings(train_split.to(cfg.resolve_device()))
        embeddings = {k: v.detach().cpu() for k, v in embeddings.items()}
    train_split = train_split.cpu()
    test_split = test_split.cpu()
    data = data.cpu()

    tasks = [
        GlycanProteinInteractionTask(),
        DiseaseAssociationTask(source_node_type="enzyme"),
        DrugTargetTask(),
        BindingSiteTask(),
        GlycanFunctionTask(),
        ImmunogenicityTask(n_bootstrap=1000),
    ]
    evaluator = DownstreamEvaluator(tasks=tasks)
    split_kwargs = {
        "glycan_protein_interaction": {
            "train_data": train_split,
            "test_data": test_split,
            "split_seed": split_seed,
        },
        "drug_target_identification": {
            "train_data": train_split,
            "test_data": test_split,
            "split_seed": split_seed,
        },
        "binding_site_prediction": {
            "train_data": train_split,
            "test_data": test_split,
            "split_seed": split_seed,
            "positive_site_indices": _load_positive_site_indices_from_uniprot(node_mappings),
        },
    }
    results = evaluator.evaluate_all(embeddings=embeddings, data=data, task_kwargs=split_kwargs)

    out_path = run_dir / "downstream_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Downstream: results saved to %s", out_path)


def run_evaluate_inductive(cfg: ExperimentConfig) -> None:
    """Run inductive (zero-shot) link prediction evaluation."""
    from glycoMusubi.data.converter import KGConverter
    from glycoMusubi.data.inductive_splits import (
        create_inductive_split,
        inductive_triples_to_tensor,
    )
    from glycoMusubi.evaluation.inductive_adapter import InductiveAdapter
    from glycoMusubi.evaluation.inductive_evaluator import InductiveEvaluator

    run_dir = cfg.run_dir
    device = cfg.resolve_device()

    # Load KG data
    payload_path = _dataset_payload_path(cfg)
    payload = _load_dataset_payload(payload_path)
    data = payload["data"]
    node_mappings = payload.get("node_mappings", {})

    _prune_invalid_edge_types(data)

    # Load trained edge type mapping
    eval_payload_path = run_dir / "eval_payload.pt"
    if eval_payload_path.exists():
        eval_payload = torch.load(eval_payload_path, map_location="cpu", weights_only=False)
        edge_type_to_idx = eval_payload.get("edge_type_to_idx", _build_edge_type_mapping(list(data.edge_types)))
    else:
        edge_type_to_idx = _build_edge_type_mapping(list(data.edge_types))

    # Filter to trained edge types
    allowed = list(edge_type_to_idx.keys())
    _filter_edge_types(data, allowed)

    node_type_order = sorted(data.node_types)
    num_nodes_dict = {
        ntype: int(getattr(data[ntype], "num_nodes", 0) or data[ntype].x.size(0))
        for ntype in node_type_order
    }

    # Create inductive split
    holdout_ratio = float(getattr(cfg.data, "holdout_ratio", 0.15))
    min_degree = int(getattr(cfg.data, "inductive_min_degree", 2))

    logger.info(
        "Inductive eval: creating entity split (holdout=%.2f, min_degree=%d)",
        holdout_ratio,
        min_degree,
    )
    split = create_inductive_split(
        data,
        holdout_ratio=holdout_ratio,
        min_degree=min_degree,
        holdout_node_types=["glycan", "protein"],
        target_relations=["has_glycan", "has_site"],
        seed=cfg.seed,
    )

    # Build model
    wurcs_map = None
    model_name = str(cfg.model.name).lower()
    glycan_enc = str(getattr(cfg.model, "glycan_encoder_type", "learnable")).lower()
    if model_name == "glycokgnet" and glycan_enc in ("wurcs_features", "hybrid"):
        wurcs_map = _build_wurcs_map(node_mappings)

    # Build site positions map for protein encoder if esm2_site_aware
    site_positions_map = None
    protein_enc = str(getattr(cfg.model, "protein_encoder_type", "learnable")).lower()
    if model_name == "glycokgnet" and protein_enc == "esm2_site_aware":
        site_map_path = getattr(cfg.model, "site_map_path", None)
        site_positions_map = _build_site_positions_map(node_mappings, site_map_path)
        if site_positions_map:
            logger.info("Inductive eval: built site positions map with %d proteins", len(site_positions_map))

    # Build glycan function feature map if requested
    function_feature_map = None
    if model_name == "glycokgnet" and bool(getattr(cfg.model, "use_glycan_function_features", False)):
        function_feature_map = _build_glycan_function_feature_map(node_mappings)

    model = _build_model(
        cfg,
        num_nodes_dict=num_nodes_dict,
        num_relations=len(edge_type_to_idx),
        wurcs_map=wurcs_map,
        edge_types=list(edge_type_to_idx.keys()),
        site_positions_map=site_positions_map,
        function_feature_map=function_feature_map,
    )

    # Load checkpoint (prefer best.pt over last.pt)
    try:
        ckpt_path = _resolve_checkpoint(run_dir)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(state["model_state_dict"], strict=False)
        if missing:
            logger.warning("Downstream: checkpoint missing %d keys (using defaults): %s", len(missing), missing)
        if unexpected:
            logger.warning("Downstream: checkpoint has %d unexpected keys: %s", len(unexpected), unexpected)
        logger.info("Loaded checkpoint from %s", ckpt_path)
    except FileNotFoundError:
        logger.warning("No checkpoint found in %s, evaluating with random weights", run_dir)

    model = model.to(device)

    # Attach edge type indices to train data
    train_data = split.train_data
    for etype in train_data.edge_types:
        if etype in edge_type_to_idx:
            num_edges = train_data[etype].edge_index.size(1)
            train_data[etype].edge_type_idx = torch.full(
                (num_edges,), edge_type_to_idx[etype], dtype=torch.long
            )

    # Build global offsets
    offsets: Dict[str, int] = {}
    offset = 0
    for ntype in node_type_order:
        offsets[ntype] = offset
        offset += num_nodes_dict[ntype]
    total_entities = offset

    # Build holdout global IDs
    holdout_global = set()
    for ntype, ids in split.holdout_entity_ids.items():
        for idx in ids:
            holdout_global.add(offsets[ntype] + idx)

    # Build test triples from inductive split
    test_triples = inductive_triples_to_tensor(split, edge_type_to_idx, offsets)
    all_triples, _ = _hetero_to_global_triples(data, edge_type_to_idx)

    logger.info(
        "Inductive eval: %d test triples, %d holdout entities, %d total entities",
        test_triples.shape[0],
        len(holdout_global),
        total_entities,
    )

    if test_triples.shape[0] == 0:
        logger.error("No inductive test triples found — skipping evaluation")
        return

    # Create adapter and evaluator
    adapter = InductiveAdapter(
        model=model,
        train_data=train_data,
        holdout_entity_ids=split.holdout_entity_ids,
        edge_type_to_idx=edge_type_to_idx,
        node_type_order=node_type_order,
        device=device,
    )

    relation_names = {
        idx: f"{etype[0]}::{etype[1]}::{etype[2]}"
        for etype, idx in edge_type_to_idx.items()
    }

    # --- Full-ranking evaluation (standard) ---
    evaluator = InductiveEvaluator(batch_size=128)
    result = evaluator.evaluate(
        model=adapter,
        test_triples=test_triples,
        all_triples=all_triples,
        num_entities=total_entities,
        holdout_global_ids=holdout_global,
        relation_names=relation_names,
    )

    # Save results
    output = {
        "transductive": result.transductive.metrics,
        "inductive": result.inductive.metrics,
        "per_relation_inductive": result.per_relation_inductive,
        "holdout_stats": result.holdout_stats,
    }

    # --- Function-filtered evaluation (dual reporting) ---
    func_map = _build_glycan_function_index(node_mappings)
    prot_func_map = _build_protein_function_type_map(node_mappings)
    if func_map and prot_func_map:
        from glycoMusubi.evaluation.inductive_evaluator import FunctionFilter

        # Convert local glycan indices to global
        glycan_offset = offsets.get("glycan", 0)
        glycan_func_global: Dict[str, set] = {}
        all_glycan_global: set = set()
        for func_term, local_indices in func_map.items():
            global_set = {glycan_offset + i for i in local_indices}
            glycan_func_global[func_term] = global_set
            all_glycan_global.update(global_set)

        # Convert local protein indices to global
        protein_offset = offsets.get("protein", 0)
        prot_func_global: Dict[int, str] = {
            protein_offset + k: v for k, v in prot_func_map.items()
        }

        # Find has_glycan relation indices
        has_glycan_rel_indices: set = set()
        for etype, ridx in edge_type_to_idx.items():
            if etype[1] == "has_glycan":
                has_glycan_rel_indices.add(ridx)

        if has_glycan_rel_indices:
            func_filter = FunctionFilter(
                glycan_function_indices=glycan_func_global,
                protein_function_type=prot_func_global,
                has_glycan_rel_indices=has_glycan_rel_indices,
                all_glycan_global_indices=all_glycan_global,
            )

            evaluator_filtered = InductiveEvaluator(
                batch_size=128,
                function_filter=func_filter,
            )
            result_filtered = evaluator_filtered.evaluate(
                model=adapter,
                test_triples=test_triples,
                all_triples=all_triples,
                num_entities=total_entities,
                holdout_global_ids=holdout_global,
                relation_names=relation_names,
            )

            output["function_filtered"] = {
                "transductive": result_filtered.transductive.metrics,
                "inductive": result_filtered.inductive.metrics,
                "per_relation_inductive": result_filtered.per_relation_inductive,
            }

            filt_ind_mrr = result_filtered.inductive.metrics.get("mrr", 0.0)
            logger.info(
                "Inductive eval (function-filtered): inductive_MRR=%.4f",
                filt_ind_mrr,
            )

    out_path = run_dir / "inductive_metrics.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    ind_mrr = result.inductive.metrics.get("mrr", 0.0)
    trans_mrr = result.transductive.metrics.get("mrr", 0.0)
    logger.info(
        "Inductive eval: transductive_MRR=%.4f, inductive_MRR=%.4f, saved to %s",
        trans_mrr,
        ind_mrr,
        out_path,
    )


STAGE_RUNNERS = {
    "featurize": run_featurize,
    "train": run_train,
    "evaluate": run_evaluate,
    "evaluate_inductive": run_evaluate_inductive,
    "downstream": run_downstream,
}


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def run_pipeline(
    cfg: ExperimentConfig,
    stages: Optional[List[str]] = None,
    dry_run: bool = False,
) -> bool:
    """Execute the embedding pipeline.

    Returns True if all stages succeed.
    """
    if stages is None:
        stages = STAGES

    for s in stages:
        if s not in STAGES:
            raise ValueError(f"Unknown stage '{s}'. Valid: {STAGES}")

    logger.info("=" * 60)
    logger.info("GLYCO-KG EMBEDDING PIPELINE")
    logger.info("Experiment : %s", cfg.name)
    logger.info("Stages     : %s", stages)
    logger.info("Device     : %s (resolved: %s)", cfg.device, cfg.resolve_device())
    logger.info("Seed       : %d", cfg.seed)
    logger.info("Output     : %s", cfg.run_dir)
    logger.info("=" * 60)

    if dry_run:
        from omegaconf import OmegaConf

        logger.info("DRY RUN -- resolved configuration:")
        logger.info("\n%s", OmegaConf.to_yaml(OmegaConf.structured(cfg)))
        return True

    set_seed(cfg.seed)
    if cfg.deterministic:
        set_deterministic(True)

    cfg.run_dir.mkdir(parents=True, exist_ok=True)

    all_ok = True
    for stage in stages:
        logger.info("-" * 40)
        logger.info("STAGE: %s", stage.upper())
        logger.info("-" * 40)
        t0 = time.time()
        try:
            STAGE_RUNNERS[stage](cfg)
            elapsed = time.time() - t0
            logger.info("Stage %s completed in %.2fs", stage, elapsed)
        except Exception:
            elapsed = time.time() - t0
            logger.exception("Stage %s FAILED after %.2fs", stage, elapsed)
            all_ok = False
            break

    return all_ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="glycoMusubi Embedding Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config-dir",
        default="configs",
        help="Root directory for YAML configs (default: configs)",
    )
    parser.add_argument(
        "--experiment", "-e",
        default=None,
        help="Experiment config name (without .yaml)",
    )
    parser.add_argument(
        "--stage", "-s",
        nargs="+",
        choices=STAGES,
        default=None,
        help=f"Stages to run (default: all). Choices: {STAGES}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved config and exit",
    )

    args, unknown = parser.parse_known_args(argv)
    args.overrides = unknown
    return args


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    cfg = load_experiment_config(
        config_dir=args.config_dir,
        experiment=args.experiment,
        overrides=args.overrides if args.overrides else None,
    )

    log_file = str(cfg.run_dir / "embedding_pipeline.log") if not args.dry_run else None
    setup_logging(level=cfg.log_level, log_file=log_file)

    ok = run_pipeline(cfg, stages=args.stage, dry_run=args.dry_run)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
