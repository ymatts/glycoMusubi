#!/usr/bin/env python3
"""Enzyme-pathway glycan prediction — combines biosynthetic pathway knowledge
with learned contrastive retrieval to predict protein->glycan associations.

Key insight: protein->glycan = "glycan is ON protein" (glycosylation substrate).
enzyme->glycan = "enzyme PRODUCES glycan" (biosynthesis). These are different
relationships. The bridge is: if protein P carries glycan G, and enzyme E produces G,
then E must have been active in the pathway that glycosylated P.

Approach:
1. Build "enzyme activity profiles" for training proteins: which enzymes must have
   been active based on the glycans found on those proteins (from enzyme->glycan edges)
2. Expand glycan->enzyme coverage using glycan subsumption hierarchy
3. For test proteins, predict glycans via:
   a. KG embedding similarity to find similar training proteins
   b. Aggregate their enzyme profiles (weighted by similarity)
   c. Score candidate glycans by the inferred enzyme profile
4. Combine with v3 contrastive retrieval scores using rank-based fusion

Usage:
    python3 scripts/enzyme_pathway_scorer.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F_func

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data_clean"
R8_DIR = BASE_DIR / "experiments_v2" / "glycokgnet_inductive_r8"
V3_DIR = BASE_DIR / "experiments_v2" / "glycan_retrieval_v3"
OUTPUT_DIR = BASE_DIR / "experiments_v2" / "enzyme_pathway_scorer"
ESM2_DIR = DATA_DIR / "esm2_cache"

FUNC_CLASSES = [
    "N-linked", "O-linked", "Glycosphingolipid",
    "Human Milk Oligosaccharide", "GPI anchor", "C-linked", "GAG", "Other",
]
FUNC_TO_IDX = {f: i for i, f in enumerate(FUNC_CLASSES)}
N_FUNC = len(FUNC_CLASSES)


# ── HybridRetrievalModel (copied from v3 for loading) ───────────────────

class HybridRetrievalModel(nn.Module):
    """Two-tower + learned popularity bias + function match score."""

    def __init__(self, protein_dim=1280, glycan_dim=256, embed_dim=256,
                 hidden=768, n_glycans=3345, n_func_classes=8, dropout=0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.protein_proj = nn.Sequential(
            nn.Linear(protein_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
        )
        self.glycan_proj = nn.Sequential(
            nn.Linear(glycan_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
        )
        self.glycan_bias = nn.Embedding(n_glycans, 1)
        self.protein_func_proj = nn.Linear(n_func_classes, embed_dim // 4)
        self.glycan_func_proj = nn.Linear(n_func_classes, embed_dim // 4)
        self.temp = nn.Parameter(torch.tensor(0.07))
        self.score_gate = nn.Sequential(nn.Linear(3, 3), nn.Softmax(dim=-1))

    def encode_protein(self, x):
        return F_func.normalize(self.protein_proj(x), dim=-1)

    def encode_glycan(self, x):
        return F_func.normalize(self.glycan_proj(x), dim=-1)


# ── Data Loading ─────────────────────────────────────────────────────────

def _resolve_esm2(pid: str, id_to_idx: dict) -> Optional[int]:
    """Resolve protein ID to ESM2 cache index."""
    if pid in id_to_idx:
        return id_to_idx[pid]
    base = pid.split("-")[0]
    if f"{base}-1" in id_to_idx:
        return id_to_idx[f"{base}-1"]
    if base in id_to_idx:
        return id_to_idx[base]
    return None


def load_all_data():
    """Load all necessary data files."""
    logger.info("Loading data...")

    # 1. Edges
    gp_edges = pd.read_csv(DATA_DIR / "edges_glycan_protein.tsv", sep="\t")
    gp_edges = gp_edges[["glycan_id", "protein_id"]].drop_duplicates()
    logger.info("Protein-glycan edges: %d", len(gp_edges))

    ge_edges = pd.read_csv(DATA_DIR / "edges_glycan_enzyme.tsv", sep="\t")
    ge_edges = ge_edges[["glycan_id", "enzyme_id"]].drop_duplicates()
    logger.info("Enzyme-glycan edges: %d", len(ge_edges))

    # 2. Function labels
    func_df = pd.read_csv(DATA_DIR / "glycan_function_labels.tsv", sep="\t")
    func_df = func_df[["glycan_id", "function_term"]].drop_duplicates()

    # 3. ESM2 index
    with open(ESM2_DIR / "id_to_idx.json") as f:
        esm_id2idx = json.load(f)

    # 4. R8 KG embeddings
    ds = torch.load(R8_DIR / "dataset.pt", map_location="cpu", weights_only=False)
    node_mappings = ds["node_mappings"]
    ckpt = torch.load(R8_DIR / "best.pt", map_location="cpu", weights_only=False)
    model_state = ckpt["model_state_dict"]

    # 5. Subsumption hierarchy
    sub_df = pd.read_csv(DATA_DIR / "edges_glycan_subsumption.tsv", sep="\t")

    # 6. Location edges
    enz_loc = pd.read_csv(DATA_DIR / "edges_enzyme_location.tsv", sep="\t")
    prot_loc = pd.read_csv(DATA_DIR / "edges_protein_location.tsv", sep="\t")

    return {
        "gp_edges": gp_edges,
        "ge_edges": ge_edges,
        "func_df": func_df,
        "esm_id2idx": esm_id2idx,
        "node_mappings": node_mappings,
        "model_state": model_state,
        "sub_df": sub_df,
        "enz_loc": enz_loc,
        "prot_loc": prot_loc,
    }


# ── Enzyme Profile Building ─────────────────────────────────────────────

def build_glycan_to_enzymes(
    ge_edges: pd.DataFrame,
    sub_df: pd.DataFrame,
) -> Dict[str, Set[str]]:
    """Build glycan_id -> set(enzyme_ids) with subsumption expansion.

    Direct: glycan G -> enzymes that produce G
    Expanded: if G has no direct enzyme annotations, propagate from
    parent/child glycans in the subsumption hierarchy.
    """
    # Direct mapping
    direct: Dict[str, Set[str]] = defaultdict(set)
    for _, row in ge_edges.iterrows():
        direct[row["glycan_id"]].add(row["enzyme_id"])

    # Build subsumption hierarchy
    child_to_parents: Dict[str, Set[str]] = defaultdict(set)
    parent_to_children: Dict[str, Set[str]] = defaultdict(set)
    for _, row in sub_df.iterrows():
        if row["relation"] == "child_of":
            child_to_parents[row["glycan_id"]].add(row["related_glycan_id"])
            parent_to_children[row["related_glycan_id"]].add(row["glycan_id"])

    # Expand: for glycans without direct enzyme annotations,
    # inherit from parents (more general glycans)
    expanded: Dict[str, Set[str]] = defaultdict(set)
    for gid in direct:
        expanded[gid] = set(direct[gid])

    n_expanded = 0
    all_glycans = set(direct.keys()) | set(child_to_parents.keys()) | set(parent_to_children.keys())
    for gid in all_glycans:
        if gid in expanded and expanded[gid]:
            continue
        # Try parents first (inherit from more general glycans)
        inherited = set()
        for parent in child_to_parents.get(gid, set()):
            if parent in direct:
                inherited.update(direct[parent])
        # Try children (inherit from more specific glycans)
        if not inherited:
            for child in parent_to_children.get(gid, set()):
                if child in direct:
                    inherited.update(direct[child])
        if inherited:
            expanded[gid] = inherited
            n_expanded += 1

    logger.info("Glycan->enzyme map: %d direct, %d expanded via subsumption, %d total",
                len(direct), n_expanded, len(expanded))
    return expanded


def compute_enzyme_idf(
    ge_edges: pd.DataFrame,
    all_enzyme_ids: List[str],
) -> np.ndarray:
    """Compute IDF (Inverse Document Frequency) weights for enzymes.

    Enzymes that cover fewer glycans are more informative/discriminative.
    IDF(enzyme) = log(total_glycans / glycans_covered_by_enzyme)
    """
    total_glycans = ge_edges["glycan_id"].nunique()
    enz_coverage = ge_edges.groupby("enzyme_id")["glycan_id"].nunique().to_dict()

    idf = np.ones(len(all_enzyme_ids), dtype=np.float32)
    for i, eid in enumerate(all_enzyme_ids):
        count = enz_coverage.get(eid, 1)
        idf[i] = np.log(total_glycans / max(count, 1))

    logger.info("Enzyme IDF weights: min=%.3f, max=%.3f, mean=%.3f",
                idf.min(), idf.max(), idf.mean())
    return idf


def build_enzyme_profiles(
    train_edges: List[Tuple[str, str]],
    glycan_to_enzymes: Dict[str, Set[str]],
    all_enzyme_ids: List[str],
    idf_weights: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Build TF-IDF weighted enzyme activity profile for each training protein.

    For protein P with known glycans {G1, G2, ...}:
    profile[P][e] = TF(e, P) * IDF(e)
    where TF(e, P) = count of P's glycans produced by enzyme e

    IDF weighting makes specific enzymes (few glycans) more discriminative
    than ubiquitous enzymes (many glycans).

    Returns (protein_profiles, enzyme_idx_map).
    """
    enz_to_idx = {e: i for i, e in enumerate(all_enzyme_ids)}
    n_enz = len(all_enzyme_ids)

    # Accumulate enzyme counts per protein from training glycans
    protein_glycans: Dict[str, Set[str]] = defaultdict(set)
    for pid, gid in train_edges:
        protein_glycans[pid].add(gid)

    profiles: Dict[str, np.ndarray] = {}
    n_with_profile = 0

    for pid, glycans in protein_glycans.items():
        tf = np.zeros(n_enz, dtype=np.float32)
        for gid in glycans:
            for eid in glycan_to_enzymes.get(gid, set()):
                if eid in enz_to_idx:
                    tf[enz_to_idx[eid]] += 1.0
        if tf.sum() > 0:
            # TF-IDF: multiply raw counts by IDF weights
            profile = tf * idf_weights
            # L2 normalize
            norm = np.linalg.norm(profile)
            if norm > 0:
                profile = profile / norm
            profiles[pid] = profile
            n_with_profile += 1

    logger.info("Enzyme profiles (TF-IDF): %d / %d training proteins have non-zero profiles",
                n_with_profile, len(protein_glycans))
    return profiles, enz_to_idx


def build_glycan_enzyme_signatures(
    glycan_to_enzymes: Dict[str, Set[str]],
    enz_to_idx: Dict[str, int],
    idf_weights: np.ndarray,
    valid_gids: List[str],
    gid_to_local: Dict[str, int],
) -> np.ndarray:
    """Build IDF-weighted enzyme signature matrix for all candidate glycans.

    Returns [n_glycans, n_enzymes] matrix where entry (g, e) = IDF(e) if enzyme e
    produces glycan g, else 0. This makes specific enzymes contribute more to scoring.
    """
    n_glycans = len(valid_gids)
    n_enz = len(enz_to_idx)
    signatures = np.zeros((n_glycans, n_enz), dtype=np.float32)

    n_covered = 0
    for gid in valid_gids:
        g_local = gid_to_local[gid]
        enzymes = glycan_to_enzymes.get(gid, set())
        for eid in enzymes:
            if eid in enz_to_idx:
                signatures[g_local, enz_to_idx[eid]] = idf_weights[enz_to_idx[eid]]
        if enzymes:
            n_covered += 1

    # L2 normalize each glycan's signature for consistent scoring
    norms = np.linalg.norm(signatures, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    signatures = signatures / norms

    logger.info("Glycan enzyme signatures (IDF-weighted): %d / %d glycans have enzyme annotations",
                n_covered, n_glycans)
    return signatures


# ── Enzyme Pathway Scoring ───────────────────────────────────────────────

def compute_enzyme_pathway_scores(
    protein_feats_esm: Dict[str, torch.Tensor],
    protein_kg_embs: torch.Tensor,
    protein_kg_map: Dict[str, int],
    train_profiles: Dict[str, np.ndarray],
    glycan_signatures: np.ndarray,
    train_proteins: Set[str],
    enz_loc: pd.DataFrame,
    prot_loc: pd.DataFrame,
    top_k_neighbors: int = 30,
) -> Dict[str, torch.Tensor]:
    """For each test protein, predict glycans using enzyme pathway reasoning.

    Steps:
    1. Find top-K most similar training proteins (by KG embedding cosine sim)
    2. Weight their enzyme profiles by similarity
    3. Score candidate glycans by dot product with inferred enzyme profile
    """
    # Build location maps for location-weighted similarity
    enz_locs: Dict[str, Set[str]] = defaultdict(set)
    for _, row in enz_loc.iterrows():
        enz_locs[row["enzyme_id"]].add(row["location_id"])
    prot_locs: Dict[str, Set[str]] = defaultdict(set)
    for _, row in prot_loc.iterrows():
        prot_locs[row["protein_id"]].add(row["location_id"])

    # Collect training proteins with both KG embeddings and enzyme profiles
    train_pids_with_data = []
    train_emb_list = []
    train_profile_list = []
    for pid in sorted(train_profiles.keys()):
        if pid in protein_kg_map:
            train_pids_with_data.append(pid)
            train_emb_list.append(protein_kg_embs[protein_kg_map[pid]])
            train_profile_list.append(train_profiles[pid])

    if not train_emb_list:
        logger.warning("No training proteins with both KG embeddings and enzyme profiles!")
        return {}

    train_emb_matrix = torch.stack(train_emb_list)  # [N_train, 256]
    train_emb_norm = F_func.normalize(train_emb_matrix, dim=-1)
    train_profile_matrix = np.stack(train_profile_list)  # [N_train, N_enz]
    logger.info("Training proteins with enzyme profiles and KG embeddings: %d", len(train_pids_with_data))

    # Also build ESM2-based similarity matrix for proteins without KG embeddings
    train_esm_list = []
    train_esm_pids = []
    for pid in train_pids_with_data:
        if pid in protein_feats_esm:
            train_esm_list.append(protein_feats_esm[pid])
            train_esm_pids.append(pid)
    if train_esm_list:
        train_esm_matrix = torch.stack(train_esm_list)
        train_esm_norm = F_func.normalize(train_esm_matrix, dim=-1)
        # Map from esm pids to profile indices
        esm_pid_to_profile_idx = {pid: i for i, pid in enumerate(train_pids_with_data)}
    else:
        train_esm_norm = None

    n_glycans = glycan_signatures.shape[0]
    result: Dict[str, torch.Tensor] = {}
    n_kg_sim = 0
    n_esm_sim = 0

    all_pids = set(protein_feats_esm.keys())
    for pid in all_pids:
        if pid in train_proteins:
            continue  # Only score test/val proteins

        inferred_profile = None

        # Try KG embedding similarity first
        if pid in protein_kg_map:
            p_emb = F_func.normalize(
                protein_kg_embs[protein_kg_map[pid]].unsqueeze(0), dim=-1
            )
            cos_sims = torch.matmul(p_emb, train_emb_norm.T).squeeze(0)  # [N_train]

            # Location bonus: boost similarity for proteins in same compartment
            p_loc = prot_locs.get(pid, set())
            if p_loc:
                for j, tr_pid in enumerate(train_pids_with_data):
                    tr_loc = prot_locs.get(tr_pid, set())
                    if tr_loc:
                        overlap = len(p_loc & tr_loc)
                        union = len(p_loc | tr_loc)
                        if overlap > 0:
                            cos_sims[j] += 0.2 * (overlap / union)

            # Top-K neighbors
            top_k = min(top_k_neighbors, len(cos_sims))
            top_vals, top_idx = cos_sims.topk(top_k)

            # Softmax weighting over top-K similarities
            weights = torch.softmax(top_vals * 5.0, dim=0).numpy()  # Temperature=5 for sharper weighting

            inferred_profile = np.zeros(train_profile_matrix.shape[1], dtype=np.float32)
            for w, idx in zip(weights, top_idx.numpy()):
                inferred_profile += w * train_profile_matrix[idx]
            n_kg_sim += 1

        # Fallback: ESM2 similarity
        elif train_esm_norm is not None and pid in protein_feats_esm:
            p_esm = F_func.normalize(protein_feats_esm[pid].unsqueeze(0), dim=-1)
            cos_sims = torch.matmul(p_esm, train_esm_norm.T).squeeze(0)

            top_k = min(top_k_neighbors, len(cos_sims))
            top_vals, top_idx = cos_sims.topk(top_k)
            weights = torch.softmax(top_vals * 3.0, dim=0).numpy()

            inferred_profile = np.zeros(train_profile_matrix.shape[1], dtype=np.float32)
            for w, idx in zip(weights, top_idx.numpy()):
                # Map ESM index to profile index
                tr_pid = train_esm_pids[idx]
                profile_idx = list(train_pids_with_data).index(tr_pid)
                inferred_profile += w * train_profile_matrix[profile_idx]
            n_esm_sim += 1

        if inferred_profile is not None and inferred_profile.sum() > 0:
            # Score each candidate glycan by dot product with inferred enzyme profile
            # glycan_signatures: [N_glycans, N_enz], inferred_profile: [N_enz]
            glycan_scores = glycan_signatures @ inferred_profile  # [N_glycans]
            result[pid] = torch.from_numpy(glycan_scores)

    logger.info("Enzyme-pathway scores: %d proteins (KG-sim: %d, ESM-sim: %d)",
                len(result), n_kg_sim, n_esm_sim)
    return result


# ── v3 Model Scoring ─────────────────────────────────────────────────────

@torch.no_grad()
def compute_v3_scores(
    protein_feats: Dict[str, torch.Tensor],
    glycan_feats: torch.Tensor,
    glycan_func_vecs: torch.Tensor,
    protein_func_vecs: Dict[str, torch.Tensor],
    v3_state: dict,
    n_glycans: int,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Compute v3 hybrid retrieval scores for all proteins."""
    model = HybridRetrievalModel(
        glycan_dim=glycan_feats.shape[1],
        n_glycans=n_glycans,
        n_func_classes=N_FUNC,
    ).to(device)
    model.load_state_dict(v3_state)
    model.eval()

    glycan_feats_d = glycan_feats.to(device)
    glycan_func_vecs_d = glycan_func_vecs.to(device)
    all_g_emb = model.encode_glycan(glycan_feats_d)
    all_bias = model.glycan_bias.weight.squeeze(-1)
    temp = model.temp.abs().clamp(min=0.01)

    result: Dict[str, torch.Tensor] = {}

    for pid, p_feat in protein_feats.items():
        p_emb = model.encode_protein(p_feat.unsqueeze(0).to(device))

        # Contrastive similarity + popularity bias
        scores = torch.matmul(p_emb, all_g_emb.T).squeeze(0) / temp + all_bias

        # Function match
        p_func = protein_func_vecs.get(pid, torch.zeros(N_FUNC)).unsqueeze(0).to(device)
        pf_proj = model.protein_func_proj(p_func)
        gf_proj = model.glycan_func_proj(glycan_func_vecs_d)
        func_scores = torch.matmul(pf_proj, gf_proj.T).squeeze(0)
        scores = scores + func_scores

        result[pid] = scores.cpu()

    logger.info("v3 scores computed for %d proteins", len(result))
    return result


# ── Evaluation ───────────────────────────────────────────────────────────

def evaluate_scores(
    scores_dict: Dict[str, torch.Tensor],
    test_edges: List[Tuple[str, str]],
    glycan_local_map: Dict[str, int],
    glycan_func_map: Dict[str, Set[str]],
    protein_func_map: Dict[str, Set[str]],
    glycan_func_idx: Dict[str, List[int]],
    n_glycans: int,
    label: str = "",
    restrict_func: bool = True,
) -> dict:
    """Evaluate a scoring function on test edges."""
    ranks_full = []
    ranks_func = []

    for pid, gid in test_edges:
        if pid not in scores_dict:
            continue
        if gid not in glycan_local_map:
            continue

        g_local = glycan_local_map[gid]
        scores = scores_dict[pid]

        # Full ranking
        rank_full = (scores > scores[g_local]).sum().item() + 1
        ranks_full.append(rank_full)

        # Function-restricted ranking
        if restrict_func:
            funcs = protein_func_map.get(pid, set())
            valid = set()
            for func in funcs:
                valid.update(glycan_func_idx.get(func, []))
            if not valid:
                valid = set(range(n_glycans))
            valid.add(g_local)
            valid_list = sorted(valid)
            sub_scores = scores[valid_list]
            target_pos = valid_list.index(g_local)
            rank_func = (sub_scores > sub_scores[target_pos]).sum().item() + 1
            ranks_func.append(rank_func)

    ra_full = np.array(ranks_full)
    ra_func = np.array(ranks_func) if ranks_func else ra_full

    results = {
        "mrr": float((1.0 / ra_func).mean()) if len(ra_func) > 0 else 0,
        "hits@1": float((ra_func <= 1).mean()) if len(ra_func) > 0 else 0,
        "hits@3": float((ra_func <= 3).mean()) if len(ra_func) > 0 else 0,
        "hits@10": float((ra_func <= 10).mean()) if len(ra_func) > 0 else 0,
        "hits@50": float((ra_func <= 50).mean()) if len(ra_func) > 0 else 0,
        "mr": float(ra_func.mean()) if len(ra_func) > 0 else 0,
        "mrr_full": float((1.0 / ra_full).mean()) if len(ra_full) > 0 else 0,
        "hits@10_full": float((ra_full <= 10).mean()) if len(ra_full) > 0 else 0,
        "mr_full": float(ra_full.mean()) if len(ra_full) > 0 else 0,
        "n": len(ra_func),
    }

    if label:
        logger.info("[%s] Func-restricted: MRR=%.4f, H@1=%.4f, H@3=%.4f, H@10=%.4f, H@50=%.4f, MR=%.1f (n=%d)",
                    label, results["mrr"], results["hits@1"], results["hits@3"],
                    results["hits@10"], results["hits@50"], results["mr"], results["n"])
        logger.info("[%s] Full ranking:    MRR=%.4f, H@10=%.4f, MR=%.1f",
                    label, results["mrr_full"], results["hits@10_full"], results["mr_full"])

    return results


# ── Score Combination ────────────────────────────────────────────────────

def _minmax_norm(t: torch.Tensor) -> torch.Tensor:
    """Min-max normalize a 1D tensor to [0, 1]."""
    tmin, tmax = t.min(), t.max()
    if tmax - tmin < 1e-12:
        return torch.zeros_like(t)
    return (t - tmin) / (tmax - tmin)


def combine_scores_weighted(
    v3_scores: Dict[str, torch.Tensor],
    enz_scores: Dict[str, torch.Tensor],
    alpha: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Weighted combination: alpha * norm(v3) + (1-alpha) * norm(enz)."""
    all_pids = set(v3_scores.keys()) | set(enz_scores.keys())
    result: Dict[str, torch.Tensor] = {}

    for pid in all_pids:
        s_v3 = v3_scores.get(pid)
        s_enz = enz_scores.get(pid)

        if s_v3 is not None and s_enz is not None:
            result[pid] = alpha * _minmax_norm(s_v3) + (1 - alpha) * _minmax_norm(s_enz)
        elif s_v3 is not None:
            result[pid] = s_v3
        elif s_enz is not None:
            result[pid] = s_enz

    return result


def combine_scores_rank_fusion(
    v3_scores: Dict[str, torch.Tensor],
    enz_scores: Dict[str, torch.Tensor],
    k: int = 60,
) -> Dict[str, torch.Tensor]:
    """Reciprocal Rank Fusion (RRF): combines by reciprocal rank.

    RRF(g) = 1/(k + rank_v3(g)) + 1/(k + rank_enz(g))

    More robust to score distribution differences.
    """
    all_pids = set(v3_scores.keys()) | set(enz_scores.keys())
    result: Dict[str, torch.Tensor] = {}

    for pid in all_pids:
        s_v3 = v3_scores.get(pid)
        s_enz = enz_scores.get(pid)

        if s_v3 is not None and s_enz is not None:
            n = s_v3.shape[0]
            # Convert scores to ranks (rank 1 = highest score)
            rank_v3 = torch.zeros(n)
            rank_enz = torch.zeros(n)
            rank_v3[s_v3.argsort(descending=True)] = torch.arange(1, n + 1, dtype=torch.float)
            rank_enz[s_enz.argsort(descending=True)] = torch.arange(1, n + 1, dtype=torch.float)
            result[pid] = 1.0 / (k + rank_v3) + 1.0 / (k + rank_enz)
        elif s_v3 is not None:
            result[pid] = s_v3
        elif s_enz is not None:
            result[pid] = s_enz

    return result


def _zscore_norm(t: torch.Tensor) -> torch.Tensor:
    """Z-score normalize a 1D tensor."""
    mu = t.mean()
    std = t.std()
    if std < 1e-12:
        return torch.zeros_like(t)
    return (t - mu) / std


def combine_scores_zscore(
    v3_scores: Dict[str, torch.Tensor],
    enz_scores: Dict[str, torch.Tensor],
    alpha: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Z-score normalized combination: alpha * z(v3) + (1-alpha) * z(enz).

    Z-score normalization preserves relative differences better than min-max
    when score distributions have different shapes.
    """
    all_pids = set(v3_scores.keys()) | set(enz_scores.keys())
    result: Dict[str, torch.Tensor] = {}

    for pid in all_pids:
        s_v3 = v3_scores.get(pid)
        s_enz = enz_scores.get(pid)

        if s_v3 is not None and s_enz is not None:
            result[pid] = alpha * _zscore_norm(s_v3) + (1 - alpha) * _zscore_norm(s_enz)
        elif s_v3 is not None:
            result[pid] = s_v3
        elif s_enz is not None:
            result[pid] = s_enz

    return result


def combine_scores_topk_rerank(
    v3_scores: Dict[str, torch.Tensor],
    enz_scores: Dict[str, torch.Tensor],
    top_k: int = 100,
    beta: float = 0.3,
) -> Dict[str, torch.Tensor]:
    """Re-rank only the top-K v3 candidates using enzyme pathway scores.

    Strategy: keep v3 ranking for bottom candidates, but re-rank top-K by
    mixing v3 score with enzyme pathway score. This preserves v3's strong
    global ranking while allowing enzyme knowledge to break ties at the top.

    combined_topK = v3_norm + beta * enz_norm  (only for top-K from v3)
    """
    all_pids = set(v3_scores.keys()) | set(enz_scores.keys())
    result: Dict[str, torch.Tensor] = {}

    for pid in all_pids:
        s_v3 = v3_scores.get(pid)
        s_enz = enz_scores.get(pid)

        if s_v3 is not None and s_enz is not None:
            n = s_v3.shape[0]
            # Get top-K indices from v3
            _, topk_idx = s_v3.topk(min(top_k, n))
            topk_set = set(topk_idx.tolist())

            # Start with v3 scores (min-max normalized)
            combined = _minmax_norm(s_v3).clone()

            # Add enzyme pathway bonus only to top-K candidates
            enz_norm = _minmax_norm(s_enz)
            for idx in topk_set:
                combined[idx] += beta * enz_norm[idx]

            result[pid] = combined
        elif s_v3 is not None:
            result[pid] = s_v3
        elif s_enz is not None:
            result[pid] = s_enz

    return result


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)
    logger.info("=" * 70)
    logger.info("ENZYME-PATHWAY GLYCAN SCORER (v2: collaborative filtering)")
    logger.info("=" * 70)

    # ── Load data ──
    data = load_all_data()
    gp_edges = data["gp_edges"]
    ge_edges = data["ge_edges"]
    func_df = data["func_df"]
    esm_id2idx = data["esm_id2idx"]
    node_mappings = data["node_mappings"]
    model_state = data["model_state"]

    glycan_map_kg = node_mappings["glycan"]
    protein_map_kg = node_mappings["protein"]

    # ── Build glycan feature matrix from KG embeddings ──
    glycan_node_feat = model_state["node_embeddings.glycan.weight"]
    has_glycan_gids = sorted(gp_edges["glycan_id"].unique())

    valid_gids = []
    glycan_local_indices = []
    for gid in has_glycan_gids:
        if gid in glycan_map_kg:
            valid_gids.append(gid)
            glycan_local_indices.append(glycan_map_kg[gid])

    glycan_feats = glycan_node_feat[glycan_local_indices]
    gid_to_local = {g: i for i, g in enumerate(valid_gids)}
    n_glycans = len(valid_gids)
    logger.info("Glycans in eval set: %d (dim=%d)", n_glycans, glycan_feats.shape[1])

    # ── Glycan function maps ──
    glycan_func: Dict[str, Set[str]] = defaultdict(set)
    for _, row in func_df.iterrows():
        glycan_func[row["glycan_id"]].add(row["function_term"])

    glycan_func_vecs = torch.zeros(n_glycans, N_FUNC)
    for gid in valid_gids:
        g_local = gid_to_local[gid]
        for func in glycan_func.get(gid, set()):
            if func in FUNC_TO_IDX:
                glycan_func_vecs[g_local, FUNC_TO_IDX[func]] = 1.0

    glycan_func_idx: Dict[str, List[int]] = defaultdict(list)
    for gid in valid_gids:
        g_local = gid_to_local[gid]
        for func in glycan_func.get(gid, set()):
            glycan_func_idx[func].append(g_local)

    for func, indices in glycan_func_idx.items():
        logger.info("  Glycan class %s: %d glycans", func, len(indices))

    # ── Load protein ESM2 features ──
    protein_feats: Dict[str, torch.Tensor] = {}
    for pid in sorted(gp_edges["protein_id"].unique()):
        idx = _resolve_esm2(pid, esm_id2idx)
        if idx is None:
            continue
        pt_path = ESM2_DIR / f"{idx}.pt"
        if not pt_path.exists():
            continue
        emb = torch.load(pt_path, map_location="cpu", weights_only=True)
        if emb.dim() == 2:
            emb = emb.mean(dim=0)
        protein_feats[pid] = emb

    logger.info("Protein features loaded: %d", len(protein_feats))

    # ── Build protein function maps and edges ──
    protein_func_map: Dict[str, Set[str]] = defaultdict(set)
    all_edges_list: List[Tuple[str, str]] = []

    for _, row in gp_edges.iterrows():
        pid, gid = row["protein_id"], row["glycan_id"]
        if pid in protein_feats and gid in gid_to_local:
            all_edges_list.append((pid, gid))
            for func in glycan_func.get(gid, set()):
                protein_func_map[pid].add(func)

    all_edges_list = list(set(all_edges_list))
    logger.info("Total unique edges: %d", len(all_edges_list))

    protein_func_vecs: Dict[str, torch.Tensor] = {}
    for pid, funcs in protein_func_map.items():
        vec = torch.zeros(N_FUNC)
        for func in funcs:
            if func in FUNC_TO_IDX:
                vec[FUNC_TO_IDX[func]] = 1.0
        protein_func_vecs[pid] = vec

    # ── Protein-level train/val/test split (same seed as v3) ──
    all_proteins = sorted(set(pid for pid, _ in all_edges_list))
    np.random.seed(42)
    prot_order = list(all_proteins)
    np.random.shuffle(prot_order)

    n_tr = int(len(prot_order) * 0.8)
    n_va = int(len(prot_order) * 0.1)
    train_p = set(prot_order[:n_tr])
    val_p = set(prot_order[n_tr:n_tr + n_va])
    test_p = set(prot_order[n_tr + n_va:])

    train_edges = [(p, g) for p, g in all_edges_list if p in train_p]
    val_edges = [(p, g) for p, g in all_edges_list if p in val_p]
    test_edges = [(p, g) for p, g in all_edges_list if p in test_p]

    logger.info("Split: train=%d proteins (%d edges), val=%d proteins (%d edges), test=%d proteins (%d edges)",
                len(train_p), len(train_edges), len(val_p), len(val_edges), len(test_p), len(test_edges))

    # ── Build enzyme pathway structures ──
    logger.info("\n" + "=" * 70)
    logger.info("BUILDING ENZYME PATHWAY STRUCTURES")
    logger.info("=" * 70)

    # glycan -> enzymes (with subsumption expansion)
    glycan_to_enzymes = build_glycan_to_enzymes(ge_edges, data["sub_df"])

    # All enzyme IDs
    all_enzyme_ids = sorted(ge_edges["enzyme_id"].unique())
    logger.info("Total enzymes: %d", len(all_enzyme_ids))

    # Compute enzyme IDF weights
    idf_weights = compute_enzyme_idf(ge_edges, all_enzyme_ids)

    # Build enzyme profiles for training proteins (TF-IDF weighted)
    train_profiles, enz_to_idx = build_enzyme_profiles(
        train_edges, glycan_to_enzymes, all_enzyme_ids, idf_weights
    )

    # Build glycan enzyme signatures (IDF weighted)
    glycan_signatures = build_glycan_enzyme_signatures(
        glycan_to_enzymes, enz_to_idx, idf_weights, valid_gids, gid_to_local
    )

    # ── Compute enzyme-pathway scores ──
    logger.info("\n" + "=" * 70)
    logger.info("COMPUTING ENZYME-PATHWAY SCORES (collaborative filtering)")
    logger.info("=" * 70)

    protein_kg_embs = model_state["node_embeddings.protein.weight"]

    enz_pathway_scores = compute_enzyme_pathway_scores(
        protein_feats_esm=protein_feats,
        protein_kg_embs=protein_kg_embs,
        protein_kg_map=protein_map_kg,
        train_profiles=train_profiles,
        glycan_signatures=glycan_signatures,
        train_proteins=train_p,
        enz_loc=data["enz_loc"],
        prot_loc=data["prot_loc"],
        top_k_neighbors=30,
    )

    # ── Compute v3 retrieval scores ──
    logger.info("\n" + "=" * 70)
    logger.info("COMPUTING V3 RETRIEVAL SCORES")
    logger.info("=" * 70)

    v3_state = torch.load(V3_DIR / "model.pt", map_location="cpu", weights_only=False)
    v3_scores = compute_v3_scores(
        protein_feats, glycan_feats, glycan_func_vecs, protein_func_vecs,
        v3_state, n_glycans, device,
    )

    # ── Evaluate ──
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION")
    logger.info("=" * 70)

    # (a) Enzyme-pathway only
    logger.info("\n--- (a) Enzyme-Pathway Only ---")
    r_enz_test = evaluate_scores(
        enz_pathway_scores, test_edges, gid_to_local,
        glycan_func, protein_func_map, glycan_func_idx, n_glycans,
        label="EnzPathway-test",
    )
    r_enz_val = evaluate_scores(
        enz_pathway_scores, val_edges, gid_to_local,
        glycan_func, protein_func_map, glycan_func_idx, n_glycans,
        label="EnzPathway-val",
    )

    # (b) V3 retrieval only
    logger.info("\n--- (b) V3 Retrieval Only ---")
    r_v3_test = evaluate_scores(
        v3_scores, test_edges, gid_to_local,
        glycan_func, protein_func_map, glycan_func_idx, n_glycans,
        label="V3Retrieval-test",
    )
    r_v3_val = evaluate_scores(
        v3_scores, val_edges, gid_to_local,
        glycan_func, protein_func_map, glycan_func_idx, n_glycans,
        label="V3Retrieval-val",
    )

    # (c) Grid search: weighted combination on val
    logger.info("\n--- (c) Weighted Combination: Grid Search alpha on val ---")
    best_alpha = 0.5
    best_val_mrr = 0.0
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    alpha_results = []

    for alpha in alphas:
        combined = combine_scores_weighted(v3_scores, enz_pathway_scores, alpha=alpha)
        r_val = evaluate_scores(
            combined, val_edges, gid_to_local,
            glycan_func, protein_func_map, glycan_func_idx, n_glycans,
        )
        logger.info("  alpha=%.1f (v3): val MRR=%.4f, H@1=%.4f, H@10=%.4f (n=%d)",
                    alpha, r_val["mrr"], r_val["hits@1"], r_val["hits@10"], r_val["n"])
        alpha_results.append({"alpha": alpha, "method": "weighted", **r_val})

        if r_val["mrr"] > best_val_mrr:
            best_val_mrr = r_val["mrr"]
            best_alpha = alpha

    logger.info("  Best weighted alpha: %.1f (val MRR=%.4f)", best_alpha, best_val_mrr)

    # (d) Rank fusion: grid search k on val
    logger.info("\n--- (d) Rank Fusion (RRF): Grid Search k on val ---")
    best_k = 60
    best_rrf_mrr = 0.0
    rrf_results = []

    for k in [10, 20, 30, 40, 50, 60, 80, 100, 150, 200]:
        combined = combine_scores_rank_fusion(v3_scores, enz_pathway_scores, k=k)
        r_val = evaluate_scores(
            combined, val_edges, gid_to_local,
            glycan_func, protein_func_map, glycan_func_idx, n_glycans,
        )
        logger.info("  RRF k=%d: val MRR=%.4f, H@1=%.4f, H@10=%.4f (n=%d)",
                    k, r_val["mrr"], r_val["hits@1"], r_val["hits@10"], r_val["n"])
        rrf_results.append({"k": k, "method": "rrf", **r_val})

        if r_val["mrr"] > best_rrf_mrr:
            best_rrf_mrr = r_val["mrr"]
            best_k = k

    logger.info("  Best RRF k: %d (val MRR=%.4f)", best_k, best_rrf_mrr)

    # (e) Final test: best weighted
    logger.info("\n--- (e) Final Test: Weighted (alpha=%.1f) ---", best_alpha)
    combined_weighted = combine_scores_weighted(v3_scores, enz_pathway_scores, alpha=best_alpha)
    r_weighted_test = evaluate_scores(
        combined_weighted, test_edges, gid_to_local,
        glycan_func, protein_func_map, glycan_func_idx, n_glycans,
        label="Weighted-test",
    )

    # (f) Final test: best RRF
    logger.info("\n--- (f) Final Test: RRF (k=%d) ---", best_k)
    combined_rrf = combine_scores_rank_fusion(v3_scores, enz_pathway_scores, k=best_k)
    r_rrf_test = evaluate_scores(
        combined_rrf, test_edges, gid_to_local,
        glycan_func, protein_func_map, glycan_func_idx, n_glycans,
        label="RRF-test",
    )

    # Fine-grained alpha search around best
    logger.info("\n--- (g) Fine-grained alpha search around best=%.1f ---", best_alpha)
    fine_best_alpha = best_alpha
    fine_best_mrr = best_val_mrr
    for alpha in np.arange(max(0, best_alpha - 0.15), min(1.01, best_alpha + 0.16), 0.025):
        combined = combine_scores_weighted(v3_scores, enz_pathway_scores, alpha=alpha)
        r_val = evaluate_scores(
            combined, val_edges, gid_to_local,
            glycan_func, protein_func_map, glycan_func_idx, n_glycans,
        )
        logger.info("  alpha=%.3f: val MRR=%.4f", alpha, r_val["mrr"])
        if r_val["mrr"] > fine_best_mrr:
            fine_best_mrr = r_val["mrr"]
            fine_best_alpha = alpha

    if fine_best_alpha != best_alpha:
        logger.info("  Refined alpha: %.3f (val MRR=%.4f)", fine_best_alpha, fine_best_mrr)
        combined_refined = combine_scores_weighted(v3_scores, enz_pathway_scores, alpha=fine_best_alpha)
        r_refined_test = evaluate_scores(
            combined_refined, test_edges, gid_to_local,
            glycan_func, protein_func_map, glycan_func_idx, n_glycans,
            label=f"Refined(a={fine_best_alpha:.3f})-test",
        )
    else:
        r_refined_test = r_weighted_test

    # (h) Z-score combination
    logger.info("\n--- (h) Z-score Combination: Grid Search alpha on val ---")
    best_z_alpha = 0.5
    best_z_mrr = 0.0
    for alpha in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 1.0]:
        combined = combine_scores_zscore(v3_scores, enz_pathway_scores, alpha=alpha)
        r_val = evaluate_scores(
            combined, val_edges, gid_to_local,
            glycan_func, protein_func_map, glycan_func_idx, n_glycans,
        )
        logger.info("  Z-score alpha=%.3f: val MRR=%.4f, H@1=%.4f, H@10=%.4f (n=%d)",
                    alpha, r_val["mrr"], r_val["hits@1"], r_val["hits@10"], r_val["n"])
        if r_val["mrr"] > best_z_mrr:
            best_z_mrr = r_val["mrr"]
            best_z_alpha = alpha
    logger.info("  Best Z-score alpha: %.3f (val MRR=%.4f)", best_z_alpha, best_z_mrr)

    combined_zscore = combine_scores_zscore(v3_scores, enz_pathway_scores, alpha=best_z_alpha)
    r_zscore_test = evaluate_scores(
        combined_zscore, test_edges, gid_to_local,
        glycan_func, protein_func_map, glycan_func_idx, n_glycans,
        label=f"Z-score(a={best_z_alpha:.3f})-test",
    )

    # (i) Top-K re-ranking
    logger.info("\n--- (i) Top-K Re-ranking: Grid Search (top_k, beta) on val ---")
    best_rerank_params = (100, 0.3)
    best_rerank_mrr = 0.0
    rerank_results = []
    for top_k_val in [50, 100, 200, 500]:
        for beta in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
            combined = combine_scores_topk_rerank(v3_scores, enz_pathway_scores, top_k=top_k_val, beta=beta)
            r_val = evaluate_scores(
                combined, val_edges, gid_to_local,
                glycan_func, protein_func_map, glycan_func_idx, n_glycans,
            )
            rerank_results.append({"top_k": top_k_val, "beta": beta, **r_val})
            if r_val["mrr"] > best_rerank_mrr:
                best_rerank_mrr = r_val["mrr"]
                best_rerank_params = (top_k_val, beta)
    logger.info("  Best re-rank: top_k=%d, beta=%.1f (val MRR=%.4f)",
                best_rerank_params[0], best_rerank_params[1], best_rerank_mrr)

    combined_rerank = combine_scores_topk_rerank(
        v3_scores, enz_pathway_scores,
        top_k=best_rerank_params[0], beta=best_rerank_params[1],
    )
    r_rerank_test = evaluate_scores(
        combined_rerank, test_edges, gid_to_local,
        glycan_func, protein_func_map, glycan_func_idx, n_glycans,
        label=f"TopK-rerank(k={best_rerank_params[0]},b={best_rerank_params[1]:.1f})-test",
    )

    # ── Detailed analysis ──
    logger.info("\n" + "=" * 70)
    logger.info("DETAILED ANALYSIS")
    logger.info("=" * 70)

    # Best combination method
    all_combo_results = [
        ("Weighted", r_weighted_test, combined_weighted),
        ("RRF", r_rrf_test, combined_rrf),
        ("Refined", r_refined_test, combined_refined if fine_best_alpha != best_alpha else combined_weighted),
        ("Z-score", r_zscore_test, combined_zscore),
        ("TopK-rerank", r_rerank_test, combined_rerank),
    ]
    all_combo_results.sort(key=lambda x: x[1]["mrr"], reverse=True)
    best_method, _, best_combined = all_combo_results[0]
    logger.info("Best combination method: %s", best_method)

    # Coverage analysis
    logger.info("\n--- Coverage analysis ---")
    n_test_prots = len(set(p for p, _ in test_edges))
    n_enz_coverage = len(set(p for p, _ in test_edges if p in enz_pathway_scores))
    n_v3_coverage = len(set(p for p, _ in test_edges if p in v3_scores))
    n_both = len(set(p for p, _ in test_edges if p in enz_pathway_scores and p in v3_scores))
    logger.info("Test proteins: %d", n_test_prots)
    logger.info("  EnzPathway coverage: %d (%.1f%%)", n_enz_coverage, 100 * n_enz_coverage / n_test_prots)
    logger.info("  V3 coverage: %d (%.1f%%)", n_v3_coverage, 100 * n_v3_coverage / n_test_prots)
    logger.info("  Both: %d (%.1f%%)", n_both, 100 * n_both / n_test_prots)

    # Proteins with enzyme-covered glycans vs not
    enzyme_covered_glycans = set(glycan_to_enzymes.keys())
    test_covered = [(p, g) for p, g in test_edges if g in enzyme_covered_glycans]
    test_uncovered = [(p, g) for p, g in test_edges if g not in enzyme_covered_glycans]
    logger.info("\n--- Glycan enzyme coverage in test set ---")
    logger.info("Test edges with enzyme-covered glycans: %d", len(test_covered))
    logger.info("Test edges without enzyme-covered glycans: %d", len(test_uncovered))

    if test_covered:
        logger.info("\n  Enzyme-covered glycans:")
        for lbl, sd in [("EnzPathway", enz_pathway_scores), ("V3", v3_scores),
                         (best_method, best_combined)]:
            evaluate_scores(
                sd, test_covered, gid_to_local,
                glycan_func, protein_func_map, glycan_func_idx, n_glycans,
                label=f"{lbl}-enz_covered",
            )

    if test_uncovered:
        logger.info("\n  Non-enzyme-covered glycans:")
        for lbl, sd in [("EnzPathway", enz_pathway_scores), ("V3", v3_scores),
                         (best_method, best_combined)]:
            evaluate_scores(
                sd, test_uncovered, gid_to_local,
                glycan_func, protein_func_map, glycan_func_idx, n_glycans,
                label=f"{lbl}-not_enz_covered",
            )

    # Breakdown by function class
    logger.info("\n--- Breakdown by glycan function class ---")
    for func_name in ["N-linked", "O-linked", "GAG"]:
        func_test = [(p, g) for p, g in test_edges if func_name in glycan_func.get(g, set())]
        if func_test:
            logger.info("\n  %s: %d test edges", func_name, len(func_test))
            for lbl, sd in [("EnzPathway", enz_pathway_scores), ("V3", v3_scores),
                             (best_method, best_combined)]:
                evaluate_scores(
                    sd, func_test, gid_to_local,
                    glycan_func, protein_func_map, glycan_func_idx, n_glycans,
                    label=f"{lbl}-{func_name}",
                )

    # Enzyme pathway statistics
    logger.info("\n--- Enzyme pathway statistics ---")
    enz_glycan_counts = defaultdict(int)
    for _, row in ge_edges.iterrows():
        enz_glycan_counts[row["enzyme_id"]] += 1
    counts = sorted(enz_glycan_counts.values(), reverse=True)
    logger.info("Glycans per enzyme: max=%d, median=%d, mean=%.1f, min=%d",
                max(counts), int(np.median(counts)), np.mean(counts), min(counts))

    # Profile quality analysis
    logger.info("\n--- Enzyme profile analysis ---")
    profile_norms = [np.linalg.norm(p * np.linalg.norm(p))
                     for p in train_profiles.values()]
    # Actually compute the number of non-zero entries (effective enzyme count)
    effective_enz = [np.count_nonzero(p) for p in train_profiles.values()]
    logger.info("Effective enzymes per protein profile: mean=%.1f, median=%d, max=%d",
                np.mean(effective_enz), int(np.median(effective_enz)), max(effective_enz))

    # ── Summary table ──
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info("%-40s  MRR     H@1     H@3     H@10    H@50    MR      n", "Method")
    logger.info("-" * 120)
    summary_rows = [
        ("EnzPathway-only (TF-IDF)", r_enz_test),
        ("V3-only", r_v3_test),
        (f"Weighted(a={best_alpha:.1f})", r_weighted_test),
        (f"RRF(k={best_k})", r_rrf_test),
        (f"Refined(a={fine_best_alpha:.3f})", r_refined_test),
        (f"Z-score(a={best_z_alpha:.3f})", r_zscore_test),
        (f"TopK-rerank(k={best_rerank_params[0]},b={best_rerank_params[1]:.1f})", r_rerank_test),
    ]
    for lbl, r in summary_rows:
        logger.info("%-40s  %.4f  %.4f  %.4f  %.4f  %.4f  %6.1f  %d",
                    lbl, r["mrr"], r["hits@1"], r["hits@3"], r["hits@10"],
                    r["hits@50"], r["mr"], r["n"])

    # Delta analysis: find best method
    best_method_name, best_test_r = max(
        [(lbl, r) for lbl, r in summary_rows if lbl != "V3-only" and lbl.startswith("EnzPathway") is False],
        key=lambda x: x[1]["mrr"],
    )
    logger.info("\n--- Delta vs V3-only (best: %s) ---", best_method_name)
    for metric in ["mrr", "hits@1", "hits@10", "hits@50", "mr"]:
        delta = best_test_r[metric] - r_v3_test[metric]
        pct = 100 * delta / max(abs(r_v3_test[metric]), 1e-9)
        logger.info("  %s: %.4f -> %.4f (delta=%+.4f, %+.1f%%)",
                    metric, r_v3_test[metric], best_test_r[metric], delta, pct)

    # ── Save results ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {
        "enzyme_pathway_test": r_enz_test,
        "enzyme_pathway_val": r_enz_val,
        "v3_retrieval_test": r_v3_test,
        "v3_retrieval_val": r_v3_val,
        "weighted_test": r_weighted_test,
        "rrf_test": r_rrf_test,
        "refined_test": r_refined_test,
        "zscore_test": r_zscore_test,
        "rerank_test": r_rerank_test,
        "best_alpha": float(fine_best_alpha),
        "best_z_alpha": float(best_z_alpha),
        "best_rrf_k": best_k,
        "best_rerank_params": {"top_k": best_rerank_params[0], "beta": best_rerank_params[1]},
        "alpha_sweep": alpha_results,
        "rrf_sweep": rrf_results,
        "rerank_sweep": rerank_results,
        "config": {
            "top_k_neighbors": 30,
            "location_bonus": 0.2,
            "kg_sim_temperature": 5.0,
            "esm_sim_temperature": 3.0,
            "n_glycans": n_glycans,
            "n_test_edges": len(test_edges),
            "n_val_edges": len(val_edges),
            "n_train_edges": len(train_edges),
            "n_test_proteins": len(test_p),
            "n_train_profiles": len(train_profiles),
            "n_enzyme_covered_glycans": int(
                sum(1 for g in valid_gids if g in enzyme_covered_glycans)
            ),
        },
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.floating, float)) else x)

    # Save enzyme pathway data for later use
    torch.save({
        "glycan_signatures": glycan_signatures,
        "enz_to_idx": enz_to_idx,
        "gid_to_local": gid_to_local,
        "valid_gids": valid_gids,
        "best_alpha": float(fine_best_alpha),
        "best_rrf_k": best_k,
    }, OUTPUT_DIR / "enzyme_pathway_data.pt")

    elapsed = time.time() - t_start
    logger.info("\nResults saved to %s", OUTPUT_DIR)
    logger.info("Total time: %.1f seconds", elapsed)


if __name__ == "__main__":
    main()
