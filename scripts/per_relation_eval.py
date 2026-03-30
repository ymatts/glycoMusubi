#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-relation link prediction evaluation for TransE vs GlycoKGNet R4.

Computes MRR, Hits@1, Hits@10 broken down by relation type for two models,
using type-restricted candidate sets for speed and biological validity.

Usage:
    python3 scripts/per_relation_eval.py
    python3 scripts/per_relation_eval.py --max-per-rel 500  # limit test triples per relation
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from glycoMusubi.data import random_link_split
from glycoMusubi.evaluation.metrics import compute_hits_at_k, compute_mr, compute_mrr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRANSE_DIR = PROJECT_ROOT / "experiments" / "baseline_transe_400ep"
GLYCOKGNET_DIR = PROJECT_ROOT / "experiments" / "glycokgnet_r4"
DATASET_PATH = TRANSE_DIR / "dataset.pt"  # shared dataset


# ---------------------------------------------------------------------------
# Relation categories for analysis
# ---------------------------------------------------------------------------

BIOLOGICAL_RELATIONS = {
    "inhibits", "has_glycan", "has_site", "associated_with_disease",
    "produced_by", "has_motif", "catalyzed_by", "has_product",
}
STRUCTURAL_RELATIONS = {
    "parent_of", "child_of", "subsumes", "subsumed_by",
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_dataset_payload(path: Path) -> Dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "data" in payload:
        return payload
    return {"data": payload, "node_mappings": {}}


def load_eval_payload(run_dir: Path) -> Dict:
    path = run_dir / "eval_payload.pt"
    return torch.load(path, map_location="cpu", weights_only=False)


def build_edge_type_mapping(edge_types: List[Tuple[str, str, str]]) -> Dict[Tuple[str, str, str], int]:
    ordered = sorted(edge_types)
    return {etype: i for i, etype in enumerate(ordered)}


def attach_edge_type_indices(data, edge_type_to_idx: Dict[Tuple[str, str, str], int]) -> None:
    for etype in data.edge_types:
        ridx = edge_type_to_idx[etype]
        num_edges = int(data[etype].edge_index.size(1))
        data[etype].edge_type_idx = torch.full((num_edges,), ridx, dtype=torch.long)


def prune_invalid_edge_types(data) -> List[Tuple[str, str, str]]:
    node_types = set(data.node_types)
    removed = []
    for etype in list(data.edge_types):
        src_type, _, dst_type = etype
        if src_type not in node_types or dst_type not in node_types:
            del data[etype]
            removed.append(etype)
    return removed


def filter_edge_types(data, allowed_edge_types: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    allowed = set(allowed_edge_types)
    removed = []
    for etype in list(data.edge_types):
        if etype not in allowed:
            del data[etype]
            removed.append(etype)
    return removed


def _build_inverse_relation_map() -> Dict[str, str]:
    """Load inverse relation pairs from edge schema."""
    import yaml

    schema_path = PROJECT_ROOT / "schemas" / "edge_schema.yaml"
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


def hetero_to_global_triples(data, edge_type_to_idx):
    """Convert HeteroData to global (h, r, t) triples tensor."""
    offsets = {}
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


# ---------------------------------------------------------------------------
# Type-restricted triple index (for filtered eval)
# ---------------------------------------------------------------------------

class TripleIndex:
    """O(1) lookup of known triples for filtered ranking."""

    def __init__(self, triples: torch.Tensor):
        self._rt_to_heads: Dict[Tuple[int, int], set] = defaultdict(set)
        self._hr_to_tails: Dict[Tuple[int, int], set] = defaultdict(set)

        tc = triples.cpu()
        for i in range(tc.shape[0]):
            h, r, t = tc[i, 0].item(), tc[i, 1].item(), tc[i, 2].item()
            self._rt_to_heads[(r, t)].add(h)
            self._hr_to_tails[(h, r)].add(t)

    def heads_for(self, r: int, t: int) -> set:
        return self._rt_to_heads.get((r, t), set())

    def tails_for(self, h: int, r: int) -> set:
        return self._hr_to_tails.get((h, r), set())


# ---------------------------------------------------------------------------
# Type-restricted scoring adapter
# ---------------------------------------------------------------------------

class TypeRestrictedScoringAdapter:
    """Score candidates restricted to the correct entity type for each relation.

    Instead of scoring against all ~77K entities, we score against entities
    of the correct head/tail type. This is faster and more meaningful.
    """

    def __init__(self, model, data, edge_type_to_idx, device):
        self.model = model.to(device)
        self.data = data.to(device)
        self.device = torch.device(device)
        self.edge_type_to_idx = edge_type_to_idx
        self.idx_to_edge_type = {v: k for k, v in edge_type_to_idx.items()}

        # Build global entity offsets
        self.offsets: Dict[str, Tuple[int, int]] = {}
        start = 0
        for ntype in sorted(self.data.node_types):
            num_nodes = int(
                getattr(self.data[ntype], "num_nodes", 0)
                or self.data[ntype].x.size(0)
            )
            self.offsets[ntype] = (start, num_nodes)
            start += num_nodes
        self.num_entities = start

        # Detect hybrid decoder for index-based scoring
        self._use_hybrid = (
            hasattr(model, "_decoder_type") and model._decoder_type == "hybrid"
        )

        self.model.eval()
        with torch.no_grad():
            self.emb_dict = self.model(self.data)

    def _global_to_local(self, node_type: str, global_idx: torch.Tensor) -> torch.Tensor:
        s, n = self.offsets[node_type]
        return (global_idx - s).clamp(min=0, max=max(n - 1, 0))

    @torch.no_grad()
    def score_tails_for_relation(
        self, head_global: torch.Tensor, ridx: int
    ) -> torch.Tensor:
        """Score all tail candidates of the correct type for a given relation.

        Returns [batch, num_tail_candidates] scores.
        """
        s_type, _, t_type = self.idx_to_edge_type[ridx]
        h_local = self._global_to_local(s_type, head_global.to(self.device))
        h_embs = self.emb_dict[s_type][h_local]  # [B, dim]
        tails = self.emb_dict[t_type]  # [T, dim]
        num_tails = tails.size(0)
        batch_size = h_embs.size(0)

        # Score each head against all tails using batch matrix operations
        # h_embs: [B, dim], tails: [T, dim]
        if self._use_hybrid:
            # For hybrid: score(h, relation_idx, t) one-at-a-time or batch
            scores = torch.empty(batch_size, num_tails, device=self.device)
            rel_idx_batch = torch.full(
                (num_tails,), ridx, dtype=torch.long, device=self.device
            )
            for i in range(batch_size):
                h_exp = h_embs[i : i + 1].expand(num_tails, -1)
                scores[i] = self.model.score(h_exp, rel_idx_batch, tails)
        else:
            # TransE / DistMult / RotatE: use embedding vectors
            rel_idx_t = torch.tensor([ridx], dtype=torch.long, device=self.device)
            rel_emb = self.model.get_relation_embedding(rel_idx_t)  # [1, dim]
            # TransE: -||h + r - t||
            # Use vectorized computation: expand h and r
            h_exp = h_embs.unsqueeze(1).expand(-1, num_tails, -1)  # [B, T, dim]
            r_exp = rel_emb.expand(batch_size, num_tails, -1)  # [B, T, dim]
            t_exp = tails.unsqueeze(0).expand(batch_size, -1, -1)  # [B, T, dim]
            # Reshape for model.score
            B, T, D = h_exp.shape
            scores_flat = self.model.score(
                h_exp.reshape(B * T, D),
                r_exp.reshape(B * T, -1),
                t_exp.reshape(B * T, D),
            )
            scores = scores_flat.reshape(B, T)

        return scores

    @torch.no_grad()
    def score_heads_for_relation(
        self, tail_global: torch.Tensor, ridx: int
    ) -> torch.Tensor:
        """Score all head candidates of the correct type for a given relation.

        Returns [batch, num_head_candidates] scores.
        """
        s_type, _, t_type = self.idx_to_edge_type[ridx]
        t_local = self._global_to_local(t_type, tail_global.to(self.device))
        t_embs = self.emb_dict[t_type][t_local]  # [B, dim]
        heads = self.emb_dict[s_type]  # [H, dim]
        num_heads = heads.size(0)
        batch_size = t_embs.size(0)

        if self._use_hybrid:
            scores = torch.empty(batch_size, num_heads, device=self.device)
            rel_idx_batch = torch.full(
                (num_heads,), ridx, dtype=torch.long, device=self.device
            )
            for i in range(batch_size):
                t_exp = t_embs[i : i + 1].expand(num_heads, -1)
                scores[i] = self.model.score(heads, rel_idx_batch, t_exp)
        else:
            rel_idx_t = torch.tensor([ridx], dtype=torch.long, device=self.device)
            rel_emb = self.model.get_relation_embedding(rel_idx_t)
            h_exp = heads.unsqueeze(0).expand(batch_size, -1, -1)  # [B, H, dim]
            r_exp = rel_emb.expand(batch_size, num_heads, -1)  # [B, H, dim]
            t_exp = t_embs.unsqueeze(1).expand(-1, num_heads, -1)  # [B, H, dim]
            B, H, D = h_exp.shape
            scores_flat = self.model.score(
                h_exp.reshape(B * H, D),
                r_exp.reshape(B * H, -1),
                t_exp.reshape(B * H, D),
            )
            scores = scores_flat.reshape(B, H)

        return scores

    def get_tail_offset(self, ridx: int) -> int:
        """Return the global offset of the tail entity type for this relation."""
        _, _, t_type = self.idx_to_edge_type[ridx]
        return self.offsets[t_type][0]

    def get_head_offset(self, ridx: int) -> int:
        """Return the global offset of the head entity type for this relation."""
        s_type, _, _ = self.idx_to_edge_type[ridx]
        return self.offsets[s_type][0]

    def get_num_tail_candidates(self, ridx: int) -> int:
        _, _, t_type = self.idx_to_edge_type[ridx]
        return self.offsets[t_type][1]

    def get_num_head_candidates(self, ridx: int) -> int:
        s_type, _, _ = self.idx_to_edge_type[ridx]
        return self.offsets[s_type][1]


# ---------------------------------------------------------------------------
# Per-relation evaluation
# ---------------------------------------------------------------------------

def evaluate_per_relation(
    adapter: TypeRestrictedScoringAdapter,
    test_triples: torch.Tensor,
    all_triples: torch.Tensor,
    edge_type_to_idx: Dict[Tuple[str, str, str], int],
    max_per_rel: int = 0,
    batch_size: int = 64,
) -> Dict[str, Dict[str, float]]:
    """Compute per-relation metrics using type-restricted filtered ranking.

    For each test triple (h, r, t), we score against all entities of the
    correct head/tail type (not all entities globally). This is faster and
    biologically more meaningful.
    """
    triple_index = TripleIndex(all_triples)
    idx_to_etype = {v: k for k, v in edge_type_to_idx.items()}

    # Group test triples by relation
    rel_groups: Dict[int, torch.Tensor] = {}
    for ridx in range(len(edge_type_to_idx)):
        mask = test_triples[:, 1] == ridx
        if mask.any():
            group = test_triples[mask]
            if max_per_rel > 0 and group.size(0) > max_per_rel:
                perm = torch.randperm(group.size(0))[:max_per_rel]
                group = group[perm]
            rel_groups[ridx] = group

    results: Dict[str, Dict[str, float]] = {}

    for ridx, triples in sorted(rel_groups.items()):
        etype = idx_to_etype[ridx]
        rel_name = f"{etype[0]}::{etype[1]}::{etype[2]}"
        n_triples = triples.size(0)

        logger.info(
            "  Evaluating relation %d: %s (%d triples, %d head cands, %d tail cands)",
            ridx,
            rel_name,
            n_triples,
            adapter.get_num_head_candidates(ridx),
            adapter.get_num_tail_candidates(ridx),
        )

        tail_ranks_list = []
        head_ranks_list = []

        t_offset = adapter.get_tail_offset(ridx)
        h_offset = adapter.get_head_offset(ridx)
        n_tail_cands = adapter.get_num_tail_candidates(ridx)
        n_head_cands = adapter.get_num_head_candidates(ridx)

        for start in range(0, n_triples, batch_size):
            end = min(start + batch_size, n_triples)
            batch = triples[start:end]
            heads = batch[:, 0]
            rels = batch[:, 1]
            tails = batch[:, 2]
            B = heads.size(0)

            # --- Tail prediction (h, r, ?) ---
            tail_scores = adapter.score_tails_for_relation(heads, ridx)  # [B, T]

            # Build filter mask for tail prediction
            tail_mask = torch.zeros(B, n_tail_cands, dtype=torch.bool)
            for i in range(B):
                h = heads[i].item()
                r = rels[i].item()
                t = tails[i].item()
                known_tails = triple_index.tails_for(h, r)
                for kt in known_tails:
                    local_kt = kt - t_offset
                    if 0 <= local_kt < n_tail_cands and kt != t:
                        tail_mask[i, local_kt] = True

            # Target index in type-local space
            tail_targets = (tails - t_offset).clamp(0, n_tail_cands - 1)
            tail_scores_cpu = tail_scores.cpu()

            # Apply filter
            tail_scores_cpu[tail_mask] = float("-inf")

            # Compute ranks
            target_scores = tail_scores_cpu[
                torch.arange(B), tail_targets
            ]
            higher = (
                tail_scores_cpu.to(torch.float64)
                > target_scores.unsqueeze(1).to(torch.float64)
            ).sum(dim=1)
            tail_ranks = (higher + 1).long()
            tail_ranks_list.append(tail_ranks)

            # --- Head prediction (?, r, t) ---
            head_scores = adapter.score_heads_for_relation(tails, ridx)  # [B, H]

            # Build filter mask for head prediction
            head_mask = torch.zeros(B, n_head_cands, dtype=torch.bool)
            for i in range(B):
                h = heads[i].item()
                r = rels[i].item()
                t = tails[i].item()
                known_heads = triple_index.heads_for(r, t)
                for kh in known_heads:
                    local_kh = kh - h_offset
                    if 0 <= local_kh < n_head_cands and kh != h:
                        head_mask[i, local_kh] = True

            head_targets = (heads - h_offset).clamp(0, n_head_cands - 1)
            head_scores_cpu = head_scores.cpu()
            head_scores_cpu[head_mask] = float("-inf")

            target_scores_h = head_scores_cpu[
                torch.arange(B), head_targets
            ]
            higher_h = (
                head_scores_cpu.to(torch.float64)
                > target_scores_h.unsqueeze(1).to(torch.float64)
            ).sum(dim=1)
            head_ranks = (higher_h + 1).long()
            head_ranks_list.append(head_ranks)

        all_tail_ranks = torch.cat(tail_ranks_list)
        all_head_ranks = torch.cat(head_ranks_list)
        both_ranks = torch.cat([all_tail_ranks, all_head_ranks])

        results[rel_name] = {
            "mrr": compute_mrr(both_ranks),
            "hits@1": compute_hits_at_k(both_ranks, 1),
            "hits@3": compute_hits_at_k(both_ranks, 3),
            "hits@10": compute_hits_at_k(both_ranks, 10),
            "mr": compute_mr(both_ranks),
            "tail_mrr": compute_mrr(all_tail_ranks),
            "head_mrr": compute_mrr(all_head_ranks),
            "n_triples": n_triples,
            "n_head_cands": n_head_cands,
            "n_tail_cands": n_tail_cands,
        }

        logger.info(
            "    MRR=%.4f  H@1=%.4f  H@10=%.4f  MR=%.1f  (%d triples)",
            results[rel_name]["mrr"],
            results[rel_name]["hits@1"],
            results[rel_name]["hits@10"],
            results[rel_name]["mr"],
            n_triples,
        )

    return results


# ---------------------------------------------------------------------------
# Model building (simplified from embedding_pipeline.py)
# ---------------------------------------------------------------------------

def build_model(model_name: str, num_nodes_dict: Dict[str, int], num_relations: int, cfg_path: Path):
    """Build a model matching the saved checkpoint."""
    from glycoMusubi.embedding.models import DistMult, RotatE, TransE

    name = model_name.lower()
    embedding_dim = 256  # Both experiments use 256

    if name == "transe":
        return TransE(
            num_nodes_dict=num_nodes_dict,
            num_relations=num_relations,
            embedding_dim=embedding_dim,
            p_norm=2,
        )
    if name == "distmult":
        return DistMult(
            num_nodes_dict=num_nodes_dict,
            num_relations=num_relations,
            embedding_dim=embedding_dim,
        )
    if name == "glycokgnet":
        from glycoMusubi.embedding.models import GlycoKGNet

        return GlycoKGNet(
            num_nodes_dict=num_nodes_dict,
            num_relations=num_relations,
            embedding_dim=embedding_dim,
            glycan_encoder_type="learnable",
            protein_encoder_type="learnable",
            num_hgt_layers=0,
            num_hgt_heads=8,
            use_bio_prior=False,
            use_cross_modal_fusion=True,
            num_fusion_heads=4,
            decoder_type="hybrid",
            dropout=0.1,
            edge_types=None,  # Not needed for eval with 0 HGT layers
        )

    raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_comparison_table(
    transe_results: Dict[str, Dict[str, float]],
    glycokgnet_results: Dict[str, Dict[str, float]],
    edge_type_to_idx: Dict[Tuple[str, str, str], int],
):
    """Print a formatted comparison table."""
    idx_to_etype = {v: k for k, v in edge_type_to_idx.items()}

    # Collect all relation names in sorted order
    all_rels = sorted(
        set(transe_results.keys()) | set(glycokgnet_results.keys())
    )

    # Extract short relation name from full etype string
    def short_name(rel_str):
        parts = rel_str.split("::")
        if len(parts) == 3:
            return f"{parts[0]}::{parts[1]}::{parts[2]}"
        return rel_str

    def category(rel_str):
        parts = rel_str.split("::")
        rel = parts[1] if len(parts) == 3 else rel_str
        if rel in STRUCTURAL_RELATIONS:
            return "structural"
        return "biological"

    print("\n" + "=" * 120)
    print("PER-RELATION LINK PREDICTION COMPARISON: TransE-400ep vs GlycoKGNet-R4")
    print("=" * 120)

    header = (
        f"{'Relation':<42s} {'Cat':^6s} {'N':>5s} "
        f"{'|':^1s} {'TransE MRR':>10s} {'H@1':>6s} {'H@10':>6s} "
        f"{'|':^1s} {'GNet MRR':>10s} {'H@1':>6s} {'H@10':>6s} "
        f"{'|':^1s} {'Delta MRR':>10s}"
    )
    print(header)
    print("-" * 120)

    # Accumulators for category summaries
    cat_ranks = {"biological": {"transe": [], "glycokgnet": []},
                 "structural": {"transe": [], "glycokgnet": []}}

    for rel in all_rels:
        cat = category(rel)
        cat_short = "BIO" if cat == "biological" else "STRUCT"

        t_res = transe_results.get(rel, {})
        g_res = glycokgnet_results.get(rel, {})

        n_test = int(t_res.get("n_triples", g_res.get("n_triples", 0)))

        t_mrr = t_res.get("mrr", 0.0)
        t_h1 = t_res.get("hits@1", 0.0)
        t_h10 = t_res.get("hits@10", 0.0)

        g_mrr = g_res.get("mrr", 0.0)
        g_h1 = g_res.get("hits@1", 0.0)
        g_h10 = g_res.get("hits@10", 0.0)

        delta = g_mrr - t_mrr
        delta_str = f"{delta:+.4f}"

        # Track for category summary
        if t_res:
            cat_ranks[cat]["transe"].append((t_mrr, n_test))
        if g_res:
            cat_ranks[cat]["glycokgnet"].append((g_mrr, n_test))

        print(
            f"{rel:<42s} {cat_short:^6s} {n_test:>5d} "
            f"{'|':^1s} {t_mrr:>10.4f} {t_h1:>6.3f} {t_h10:>6.3f} "
            f"{'|':^1s} {g_mrr:>10.4f} {g_h1:>6.3f} {g_h10:>6.3f} "
            f"{'|':^1s} {delta_str:>10s}"
        )

    print("-" * 120)

    # Category summaries (weighted by number of triples)
    print("\nCATEGORY SUMMARY (weighted by triple count):")
    print("-" * 80)
    print(f"{'Category':<15s} {'TransE MRR':>12s} {'GlycoKGNet MRR':>15s} {'Delta':>10s}")
    print("-" * 80)

    for cat_name in ["biological", "structural"]:
        for model_name, model_key in [("TransE", "transe"), ("GlycoKGNet", "glycokgnet")]:
            pass  # computed below

        def weighted_mrr(entries):
            if not entries:
                return 0.0
            total_w = sum(n for _, n in entries)
            if total_w == 0:
                return 0.0
            return sum(m * n for m, n in entries) / total_w

        t_wmrr = weighted_mrr(cat_ranks[cat_name]["transe"])
        g_wmrr = weighted_mrr(cat_ranks[cat_name]["glycokgnet"])
        delta = g_wmrr - t_wmrr

        label = cat_name.upper()
        print(f"{label:<15s} {t_wmrr:>12.4f} {g_wmrr:>15.4f} {delta:>+10.4f}")

    # Overall weighted MRR
    all_t = cat_ranks["biological"]["transe"] + cat_ranks["structural"]["transe"]
    all_g = cat_ranks["biological"]["glycokgnet"] + cat_ranks["structural"]["glycokgnet"]

    def weighted_mrr(entries):
        if not entries:
            return 0.0
        total_w = sum(n for _, n in entries)
        if total_w == 0:
            return 0.0
        return sum(m * n for m, n in entries) / total_w

    t_overall = weighted_mrr(all_t)
    g_overall = weighted_mrr(all_g)
    print(f"{'OVERALL':<15s} {t_overall:>12.4f} {g_overall:>15.4f} {g_overall - t_overall:>+10.4f}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Per-relation link prediction eval")
    parser.add_argument(
        "--max-per-rel",
        type=int,
        default=1000,
        help="Max test triples per relation (0 = all). Default: 1000",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for scoring. Default: 64",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto|cuda|cpu",
    )
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # ---------------------------------------------------------------
    # 1. Load shared dataset and build test split
    # ---------------------------------------------------------------
    logger.info("Loading dataset from %s", DATASET_PATH)
    payload = load_dataset_payload(DATASET_PATH)
    data = payload["data"]
    prune_invalid_edge_types(data)

    eval_info = load_eval_payload(TRANSE_DIR)
    edge_type_to_idx = eval_info["edge_type_to_idx"]
    split_seed = int(eval_info.get("split_seed", 42))

    # Filter to only edge types seen during training
    filter_edge_types(data, list(edge_type_to_idx.keys()))

    # Build inverse relation map for leak prevention
    inverse_relation_map = _build_inverse_relation_map()
    if inverse_relation_map:
        logger.info("Loaded %d inverse relation pairs for leak prevention", len(inverse_relation_map) // 2)

    logger.info("Building train/val/test split with seed=%d", split_seed)
    train_data, val_data, test_data = random_link_split(
        data,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=split_seed,
        inverse_relation_map=inverse_relation_map if inverse_relation_map else None,
    )
    prune_invalid_edge_types(train_data)
    prune_invalid_edge_types(val_data)
    prune_invalid_edge_types(test_data)
    attach_edge_type_indices(train_data, edge_type_to_idx)
    attach_edge_type_indices(val_data, edge_type_to_idx)
    attach_edge_type_indices(test_data, edge_type_to_idx)

    all_triples, offsets = hetero_to_global_triples(data, edge_type_to_idx)
    test_triples, _ = hetero_to_global_triples(test_data, edge_type_to_idx)

    num_nodes_dict = {
        ntype: int(
            getattr(train_data[ntype], "num_nodes", 0)
            or train_data[ntype].x.size(0)
        )
        for ntype in train_data.node_types
    }
    num_relations = len(edge_type_to_idx)

    logger.info(
        "Data: %d test triples, %d total triples, %d entity types, %d relations",
        test_triples.size(0),
        all_triples.size(0),
        len(num_nodes_dict),
        num_relations,
    )

    # Print relation distribution in test set
    idx_to_etype = {v: k for k, v in edge_type_to_idx.items()}
    for ridx in range(num_relations):
        count = (test_triples[:, 1] == ridx).sum().item()
        etype = idx_to_etype[ridx]
        logger.info("  rel %2d: %-50s  %d test triples", ridx, f"{etype[0]}::{etype[1]}::{etype[2]}", count)

    # ---------------------------------------------------------------
    # 2. Load and evaluate TransE
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Loading TransE model from %s", TRANSE_DIR / "last.pt")
    transe_model = build_model("TransE", num_nodes_dict, num_relations, TRANSE_DIR)
    ckpt = torch.load(TRANSE_DIR / "last.pt", map_location=device, weights_only=False)
    transe_model.load_state_dict(ckpt["model_state_dict"])

    logger.info("Building TransE scoring adapter...")
    transe_adapter = TypeRestrictedScoringAdapter(
        transe_model, data, edge_type_to_idx, device
    )

    logger.info("Evaluating TransE per-relation...")
    t0 = time.time()
    transe_results = evaluate_per_relation(
        transe_adapter,
        test_triples,
        all_triples,
        edge_type_to_idx,
        max_per_rel=args.max_per_rel,
        batch_size=args.batch_size,
    )
    t_elapsed = time.time() - t0
    logger.info("TransE evaluation took %.1f seconds", t_elapsed)

    # Free TransE adapter memory
    del transe_adapter, transe_model
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # 3. Load and evaluate GlycoKGNet R4
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Loading GlycoKGNet R4 model from %s", GLYCOKGNET_DIR / "last.pt")
    glycokgnet_model = build_model("GlycoKGNet", num_nodes_dict, num_relations, GLYCOKGNET_DIR)
    ckpt = torch.load(GLYCOKGNET_DIR / "last.pt", map_location=device, weights_only=False)
    glycokgnet_model.load_state_dict(ckpt["model_state_dict"])

    logger.info("Building GlycoKGNet scoring adapter...")
    glycokgnet_adapter = TypeRestrictedScoringAdapter(
        glycokgnet_model, data, edge_type_to_idx, device
    )

    logger.info("Evaluating GlycoKGNet R4 per-relation...")
    t0 = time.time()
    glycokgnet_results = evaluate_per_relation(
        glycokgnet_adapter,
        test_triples,
        all_triples,
        edge_type_to_idx,
        max_per_rel=args.max_per_rel,
        batch_size=args.batch_size,
    )
    g_elapsed = time.time() - t0
    logger.info("GlycoKGNet evaluation took %.1f seconds", g_elapsed)

    # ---------------------------------------------------------------
    # 4. Print comparison
    # ---------------------------------------------------------------
    print_comparison_table(transe_results, glycokgnet_results, edge_type_to_idx)

    # Save results to JSON
    import json

    out_path = PROJECT_ROOT / "logs" / "per_relation_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "transe_400ep": transe_results,
                "glycokgnet_r4": glycokgnet_results,
                "max_per_rel": args.max_per_rel,
                "split_seed": split_seed,
            },
            f,
            indent=2,
        )
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
