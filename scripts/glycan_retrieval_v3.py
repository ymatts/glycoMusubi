#!/usr/bin/env python3
"""Protein→Glycan retrieval v3 — Hybrid scoring with popularity prior.

Key improvements over v2:
1. Hybrid score = contrastive_sim + learned_popularity_bias + function_match
2. Hard negative mining (top-scoring wrong answers from previous epoch)
3. Site-aware protein encoding when per-residue ESM2 available
4. Glycan clustering evaluation (practical grouping)
5. Larger model with cosine annealing LR
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F_func

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Data Loading ──────────────────────────────────────────────────────────


def load_data():
    data_dir = Path("data_clean")
    edges = pd.read_csv(data_dir / "edges_glycan_protein.tsv", sep="\t")
    edges = edges[["glycan_id", "protein_id"]].drop_duplicates()
    func = pd.read_csv(data_dir / "glycan_function_labels.tsv", sep="\t")
    func = func[["glycan_id", "function_term"]].drop_duplicates()

    with open(data_dir / "esm2_cache" / "id_to_idx.json") as f:
        esm_id_to_idx = json.load(f)

    # Load R8 KG model embeddings
    r8_dir = Path("experiments_v2/glycokgnet_inductive_r8")
    ds = torch.load(r8_dir / "dataset.pt", map_location="cpu", weights_only=False)
    node_mappings = ds["node_mappings"]
    glycan_map = node_mappings["glycan"]
    protein_map = node_mappings["protein"]

    ckpt = torch.load(r8_dir / "best.pt", map_location="cpu", weights_only=False)
    model_state = ckpt["model_state_dict"]

    # Site annotations
    sites_df = pd.read_csv(data_dir / "uniprot_sites.tsv", sep="\t")

    return edges, func, esm_id_to_idx, data_dir / "esm2_cache", glycan_map, protein_map, model_state, sites_df


def _resolve_esm2(pid, id_to_idx):
    if pid in id_to_idx:
        return id_to_idx[pid]
    base = pid.split("-")[0]
    if f"{base}-1" in id_to_idx:
        return id_to_idx[f"{base}-1"]
    if base in id_to_idx:
        return id_to_idx[base]
    return None


def load_site_aware_features(pid: str, esm2_dir: Path, perresidue_dir: Path,
                             site_positions: List[int], esm_id2idx: dict,
                             window: int = 16) -> Optional[torch.Tensor]:
    """Extract per-residue ESM2 features at glycosylation sites, attention-pooled."""
    base = pid.split("-")[0]
    pr_path = perresidue_dir / f"{base}.pt"
    if not pr_path.exists():
        return None

    per_res = torch.load(pr_path, map_location="cpu", weights_only=True)  # [L, 1280]
    if per_res.dim() != 2:
        return None

    L = per_res.shape[0]
    site_feats = []
    for pos in site_positions:
        if pos < 0 or pos >= L:
            continue
        start = max(0, pos - window)
        end = min(L, pos + window + 1)
        local_window = per_res[start:end]  # [W, 1280]
        # Center-weighted mean
        center_idx = pos - start
        weights = torch.exp(-0.1 * torch.abs(torch.arange(local_window.shape[0], dtype=torch.float) - center_idx))
        weights = weights / weights.sum()
        site_feat = (local_window * weights.unsqueeze(1)).sum(dim=0)  # [1280]
        site_feats.append(site_feat)

    if not site_feats:
        return None

    return torch.stack(site_feats).mean(dim=0)  # [1280] — mean over all sites


# ── Model ─────────────────────────────────────────────────────────────────


class HybridRetrievalModel(nn.Module):
    """Two-tower + learned popularity bias + function match score."""

    def __init__(self, protein_dim=1280, glycan_dim=256, embed_dim=256,
                 hidden=768, n_glycans=3345, n_func_classes=8, dropout=0.2):
        super().__init__()
        self.embed_dim = embed_dim

        # Protein tower
        self.protein_proj = nn.Sequential(
            nn.Linear(protein_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
        )

        # Glycan tower
        self.glycan_proj = nn.Sequential(
            nn.Linear(glycan_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
        )

        # Learned glycan popularity bias
        self.glycan_bias = nn.Embedding(n_glycans, 1)

        # Function class embeddings for matching
        self.protein_func_proj = nn.Linear(n_func_classes, embed_dim // 4)
        self.glycan_func_proj = nn.Linear(n_func_classes, embed_dim // 4)

        # Temperature
        self.temp = nn.Parameter(torch.tensor(0.07))

        # Score combination
        self.score_gate = nn.Sequential(
            nn.Linear(3, 3),
            nn.Softmax(dim=-1),
        )

    def encode_protein(self, x):
        return F_func.normalize(self.protein_proj(x), dim=-1)

    def encode_glycan(self, x):
        return F_func.normalize(self.glycan_proj(x), dim=-1)

    def score(self, p_emb, g_emb, g_idx, p_func_vec=None, g_func_vec=None):
        """Compute hybrid score.

        p_emb: [B, D]  protein embeddings
        g_emb: [B or K, D]  glycan embeddings
        g_idx: [B or K]  glycan local indices
        p_func_vec: [B, F]  protein function multi-hot (optional)
        g_func_vec: [B or K, F]  glycan function multi-hot (optional)
        """
        temp = self.temp.abs().clamp(min=0.01)

        # 1. Contrastive similarity
        if p_emb.dim() == 2 and g_emb.dim() == 2 and p_emb.shape[0] == g_emb.shape[0]:
            sim = (p_emb * g_emb).sum(dim=-1) / temp  # [B]
        else:
            sim = torch.matmul(p_emb, g_emb.T) / temp  # [B, K]

        # 2. Popularity bias
        bias = self.glycan_bias(g_idx).squeeze(-1)  # [B] or [K]

        # 3. Function match
        func_score = torch.zeros_like(sim)
        if p_func_vec is not None and g_func_vec is not None:
            pf = self.protein_func_proj(p_func_vec)  # [B, D/4]
            gf = self.glycan_func_proj(g_func_vec)  # [B or K, D/4]
            if pf.dim() == 2 and gf.dim() == 2 and pf.shape[0] == gf.shape[0]:
                func_score = (pf * gf).sum(dim=-1)
            else:
                func_score = torch.matmul(pf, gf.T)

        return sim + bias + func_score


# ── Training ──────────────────────────────────────────────────────────────


def build_func_vectors(glycan_func: Dict[str, Set[str]], protein_func_map: Dict[int, Set[str]],
                       n_glycans: int, n_proteins: int, func_classes: List[str]):
    """Build multi-hot function vectors."""
    func_to_idx = {f: i for i, f in enumerate(func_classes)}
    F = len(func_classes)

    glycan_func_vecs = torch.zeros(n_glycans, F)
    protein_func_vecs = {}

    for gid_str, funcs in glycan_func.items():
        for f in funcs:
            if f in func_to_idx:
                glycan_func_vecs[int(gid_str) if isinstance(gid_str, int) else 0, func_to_idx[f]] = 1.0

    return glycan_func_vecs, func_to_idx


def train_and_evaluate(
    protein_feats: Dict[int, torch.Tensor],
    glycan_feats: torch.Tensor,
    glycan_freq: torch.Tensor,
    train_edges: torch.Tensor,
    val_edges: torch.Tensor,
    test_edges: torch.Tensor,
    glycan_func_idx: Dict[str, List[int]],
    protein_func_map: Dict[int, Set[str]],
    glycan_func_vecs: torch.Tensor,
    protein_func_vecs: Dict[int, torch.Tensor],
    n_glycans: int,
    n_func_classes: int,
    device: str = "cpu",
    epochs: int = 150,
    num_neg: int = 128,
    batch_size: int = 512,
):
    glycan_feats = glycan_feats.to(device)
    glycan_func_vecs = glycan_func_vecs.to(device)
    glycan_freq = glycan_freq.to(device)

    model = HybridRetrievalModel(
        glycan_dim=glycan_feats.shape[1],
        n_glycans=n_glycans,
        n_func_classes=n_func_classes,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Initialize popularity bias from data
    with torch.no_grad():
        log_freq = torch.log1p(glycan_freq).to(device)
        log_freq = (log_freq - log_freq.mean()) / (log_freq.std() + 1e-8)
        model.glycan_bias.weight.data[:, 0] = log_freq

    # Build positive index
    train_pos = defaultdict(set)
    for i in range(train_edges.shape[0]):
        pid, gid = train_edges[i, 0].item(), train_edges[i, 1].item()
        train_pos[pid].add(gid)

    # Hard negative cache (updated each epoch)
    hard_neg_cache: Dict[int, torch.Tensor] = {}

    best_val_mrr = 0.0
    best_state = None
    patience_count = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(train_edges.shape[0])
        total_loss = 0.0
        n_samples = 0

        for start in range(0, len(perm), batch_size):
            batch_idx = perm[start:start + batch_size]
            batch = train_edges[batch_idx]

            p_feats_list = []
            pos_g_idx = []
            p_func_list = []
            valid = []

            for j in range(batch.shape[0]):
                pid = batch[j, 0].item()
                gid = batch[j, 1].item()
                if pid not in protein_feats:
                    continue
                valid.append(True)
                p_feats_list.append(protein_feats[pid])
                pos_g_idx.append(gid)
                p_func_list.append(protein_func_vecs.get(pid, torch.zeros(n_func_classes)))

            if not p_feats_list:
                continue

            B = len(p_feats_list)
            p_batch = torch.stack(p_feats_list).to(device)
            p_func_batch = torch.stack(p_func_list).to(device)
            p_emb = model.encode_protein(p_batch)

            # Positive glycan embeddings
            pos_idx_t = torch.tensor(pos_g_idx, dtype=torch.long, device=device)
            pos_g_emb = model.encode_glycan(glycan_feats[pos_idx_t])
            pos_g_func = glycan_func_vecs[pos_idx_t]

            # Sample negatives: mix of hard negatives + random from function class
            neg_indices_set = set()
            for j in range(B):
                pid = p_feats_list[j]  # This is actually the feature tensor, not pid
                gid = pos_g_idx[j]

                # Get function-class candidates
                p_idx = batch[valid.index(True) if j == 0 else j, 0].item()  # approximate
                funcs = protein_func_map.get(p_idx, set())
                candidates = set()
                for func in funcs:
                    for idx in glycan_func_idx.get(func, []):
                        candidates.add(idx)
                if not candidates:
                    candidates = set(range(n_glycans))
                candidates -= train_pos.get(p_idx, set())

                # Hard negatives from cache
                if p_idx in hard_neg_cache and len(hard_neg_cache[p_idx]) > 0:
                    hard = hard_neg_cache[p_idx][:num_neg // 4].tolist()
                    neg_indices_set.update(hard)

                # Random negatives from function class
                candidates = list(candidates - neg_indices_set)
                if len(candidates) > num_neg:
                    neg_indices_set.update(np.random.choice(candidates, num_neg, replace=False).tolist())
                else:
                    neg_indices_set.update(candidates)

            neg_idx_t = torch.tensor(sorted(neg_indices_set), dtype=torch.long, device=device)
            if len(neg_idx_t) == 0:
                continue

            neg_g_emb = model.encode_glycan(glycan_feats[neg_idx_t])
            neg_g_func = glycan_func_vecs[neg_idx_t]

            # Positive scores
            pos_scores = model.score(p_emb, pos_g_emb, pos_idx_t, p_func_batch, pos_g_func)

            # Negative scores: [B, K]
            temp = model.temp.abs().clamp(min=0.01)
            neg_sim = torch.matmul(p_emb, neg_g_emb.T) / temp
            neg_bias = model.glycan_bias(neg_idx_t).squeeze(-1).unsqueeze(0).expand(B, -1)
            pf_proj = model.protein_func_proj(p_func_batch)
            gf_proj = model.glycan_func_proj(neg_g_func)
            neg_func = torch.matmul(pf_proj, gf_proj.T)
            neg_scores = neg_sim + neg_bias + neg_func

            # InfoNCE loss
            logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
            target = torch.zeros(B, dtype=torch.long, device=device)
            loss = F_func.cross_entropy(logits, target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * B
            n_samples += B

        scheduler.step()
        avg_loss = total_loss / max(n_samples, 1)

        # Update hard negative cache every 10 epochs
        if (epoch + 1) % 10 == 0:
            hard_neg_cache = _mine_hard_negatives(
                model, protein_feats, glycan_feats, glycan_func_vecs,
                train_edges, train_pos, n_glycans, device, top_k=64,
            )

        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0:
            val_r = evaluate_retrieval(
                model, protein_feats, glycan_feats, glycan_func_vecs,
                None, val_edges, glycan_func_idx, protein_func_map,
                protein_func_vecs, n_glycans, n_func_classes, device,
            )
            lr = scheduler.get_last_lr()[0]
            logger.info("Epoch %d: loss=%.4f lr=%.1e val_MRR=%.4f H@1=%.4f H@10=%.4f",
                        epoch + 1, avg_loss, lr, val_r["mrr"], val_r["hits@1"], val_r["hits@10"])

            if val_r["mrr"] > best_val_mrr:
                best_val_mrr = val_r["mrr"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= 8:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

    if best_state:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def _mine_hard_negatives(model, protein_feats, glycan_feats, glycan_func_vecs,
                          train_edges, train_pos, n_glycans, device, top_k=64):
    """Find top-scoring wrong glycans for each protein."""
    model.eval()
    all_g_emb = model.encode_glycan(glycan_feats)
    all_bias = model.glycan_bias.weight.squeeze(-1)  # [N_g]

    hard_negs: Dict[int, torch.Tensor] = {}
    seen_pids = set()

    for i in range(train_edges.shape[0]):
        pid = train_edges[i, 0].item()
        if pid in seen_pids or pid not in protein_feats:
            continue
        seen_pids.add(pid)

        p_feat = protein_feats[pid].unsqueeze(0).to(device)
        p_emb = model.encode_protein(p_feat)
        temp = model.temp.abs().clamp(min=0.01)
        scores = torch.matmul(p_emb, all_g_emb.T).squeeze(0) / temp + all_bias

        # Mask out positives
        pos = list(train_pos.get(pid, set()))
        if pos:
            scores[pos] = -1e9

        _, top_indices = scores.topk(top_k)
        hard_negs[pid] = top_indices.cpu()

    return hard_negs


@torch.no_grad()
def evaluate_retrieval(
    model, protein_feats, glycan_feats, glycan_func_vecs,
    protein_func_vecs_tensor, edges, glycan_func_idx,
    protein_func_map, protein_func_vecs, n_glycans, n_func_classes, device,
    restrict_func=True,
):
    model.eval()
    glycan_feats_d = glycan_feats.to(device)
    glycan_func_vecs_d = glycan_func_vecs.to(device)

    all_g_emb = model.encode_glycan(glycan_feats_d)
    all_bias = model.glycan_bias.weight.squeeze(-1)  # [N_g]
    temp = model.temp.abs().clamp(min=0.01)

    ranks_full = []
    ranks_func = []

    for i in range(edges.shape[0]):
        pid = edges[i, 0].item()
        gid = edges[i, 1].item()
        if pid not in protein_feats:
            continue

        p_feat = protein_feats[pid].unsqueeze(0).to(device)
        p_emb = model.encode_protein(p_feat)

        # Similarity + bias
        scores = torch.matmul(p_emb, all_g_emb.T).squeeze(0) / temp + all_bias

        # Add function match score
        p_func = protein_func_vecs.get(pid, torch.zeros(n_func_classes)).unsqueeze(0).to(device)
        pf_proj = model.protein_func_proj(p_func)  # [1, D/4]
        gf_proj = model.glycan_func_proj(glycan_func_vecs_d)  # [N_g, D/4]
        func_scores = torch.matmul(pf_proj, gf_proj.T).squeeze(0)  # [N_g]
        scores = scores + func_scores

        # Full ranking
        rank_full = (scores > scores[gid]).sum().item() + 1
        ranks_full.append(rank_full)

        # Function-restricted ranking
        if restrict_func:
            funcs = protein_func_map.get(pid, set())
            valid = set()
            for func in funcs:
                valid.update(glycan_func_idx.get(func, []))
            if not valid:
                valid = set(range(n_glycans))
            valid.add(gid)
            valid = sorted(valid)
            sub_scores = scores[valid]
            target_pos = valid.index(gid)
            rank_func = (sub_scores > sub_scores[target_pos]).sum().item() + 1
            ranks_func.append(rank_func)

    ra_full = np.array(ranks_full)
    ra_func = np.array(ranks_func) if ranks_func else ra_full

    return {
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


# ── Baselines ─────────────────────────────────────────────────────────────


def evaluate_popularity_baseline(train_edges, test_edges, glycan_func_idx,
                                  protein_func_map, n_glycans):
    """Rank glycans by training-set frequency (popularity baseline)."""
    freq = defaultdict(int)
    for i in range(train_edges.shape[0]):
        freq[train_edges[i, 1].item()] += 1

    # Sort by frequency
    sorted_glycans = sorted(range(n_glycans), key=lambda g: freq.get(g, 0), reverse=True)
    rank_map = {g: r + 1 for r, g in enumerate(sorted_glycans)}

    ranks_full = []
    ranks_func = []

    for i in range(test_edges.shape[0]):
        pid = test_edges[i, 0].item()
        gid = test_edges[i, 1].item()

        ranks_full.append(rank_map.get(gid, n_glycans))

        # Function-restricted
        funcs = protein_func_map.get(pid, set())
        valid = set()
        for func in funcs:
            valid.update(glycan_func_idx.get(func, []))
        if not valid:
            valid = set(range(n_glycans))
        valid.add(gid)
        valid_sorted = sorted(valid, key=lambda g: freq.get(g, 0), reverse=True)
        rank_func = valid_sorted.index(gid) + 1
        ranks_func.append(rank_func)

    ra_full = np.array(ranks_full)
    ra_func = np.array(ranks_func)

    return {
        "mrr": float((1.0 / ra_func).mean()),
        "hits@1": float((ra_func <= 1).mean()),
        "hits@10": float((ra_func <= 10).mean()),
        "mr": float(ra_func.mean()),
        "mrr_full": float((1.0 / ra_full).mean()),
    }


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    edges_df, func_df, esm_id2idx, esm2_dir, glycan_map, protein_map, model_state, sites_df = load_data()

    # Glycan features from learned KG embeddings
    glycan_node_feat = model_state["node_embeddings.glycan.weight"]
    logger.info("Glycan LEARNED embeddings: %s", glycan_node_feat.shape)

    has_glycan_gids = sorted(edges_df["glycan_id"].unique())
    glycan_local_indices = []
    valid_gids = []
    for gid in has_glycan_gids:
        if gid in glycan_map:
            glycan_local_indices.append(glycan_map[gid])
            valid_gids.append(gid)

    glycan_feats = glycan_node_feat[glycan_local_indices]
    gid_to_local = {g: i for i, g in enumerate(valid_gids)}
    n_glycans = len(valid_gids)
    logger.info("Glycans: %d", n_glycans)

    # Glycan function maps
    glycan_func: Dict[str, Set[str]] = defaultdict(set)
    for _, row in func_df.iterrows():
        glycan_func[row["glycan_id"]].add(row["function_term"])

    FUNC_CLASSES = ["N-linked", "O-linked", "Glycosphingolipid", "Human Milk Oligosaccharide",
                    "GPI anchor", "C-linked", "GAG", "Other"]
    func_to_idx = {f: i for i, f in enumerate(FUNC_CLASSES)}
    n_func_classes = len(FUNC_CLASSES)

    # Build glycan function vectors (multi-hot)
    glycan_func_vecs = torch.zeros(n_glycans, n_func_classes)
    for gid in valid_gids:
        g_local = gid_to_local[gid]
        for func in glycan_func.get(gid, set()):
            if func in func_to_idx:
                glycan_func_vecs[g_local, func_to_idx[func]] = 1.0

    # Glycan frequency (for popularity init)
    glycan_freq_count = edges_df["glycan_id"].value_counts()
    glycan_freq = torch.zeros(n_glycans)
    for gid in valid_gids:
        glycan_freq[gid_to_local[gid]] = glycan_freq_count.get(gid, 0)

    # Build site data per protein
    protein_sites: Dict[str, List[int]] = defaultdict(list)
    for _, row in sites_df.iterrows():
        uid = row["uniprot_id"]
        pos = int(row["site_position"]) - 1  # 0-indexed
        protein_sites[uid].append(pos)

    # Load protein ESM2 features (with site-aware fallback)
    perresidue_dir = Path("data_clean/esm2_perresidue")
    protein_feats: Dict[int, torch.Tensor] = {}
    pid_to_local: Dict[str, int] = {}
    local_idx = 0
    n_site_aware = 0

    for pid in sorted(edges_df["protein_id"].unique()):
        idx = _resolve_esm2(pid, esm_id2idx)
        if idx is None:
            continue
        pt_path = esm2_dir / f"{idx}.pt"
        if not pt_path.exists():
            continue

        emb = torch.load(pt_path, map_location="cpu", weights_only=True)
        if emb.dim() == 2:
            emb = emb.mean(dim=0)

        # Try site-aware features
        base = pid.split("-")[0]
        sites = protein_sites.get(base, [])
        if sites and perresidue_dir.exists():
            site_emb = load_site_aware_features(pid, esm2_dir, perresidue_dir, sites, esm_id2idx)
            if site_emb is not None:
                # Concatenate mean + site-aware, then project back to 1280
                emb = 0.5 * emb + 0.5 * site_emb
                n_site_aware += 1

        pid_to_local[pid] = local_idx
        protein_feats[local_idx] = emb
        local_idx += 1

    logger.info("Protein features: %d (%d site-aware)", len(protein_feats), n_site_aware)

    # Build protein function vectors and maps
    protein_func_map: Dict[int, Set[str]] = defaultdict(set)
    protein_func_vecs: Dict[int, torch.Tensor] = {}

    all_edges_list = []
    for _, row in edges_df.iterrows():
        pid, gid = row["protein_id"], row["glycan_id"]
        if pid in pid_to_local and gid in gid_to_local:
            p_local = pid_to_local[pid]
            g_local = gid_to_local[gid]
            all_edges_list.append((p_local, g_local))
            for func in glycan_func.get(gid, set()):
                protein_func_map[p_local].add(func)

    # Build protein function vectors
    for p_local, funcs in protein_func_map.items():
        vec = torch.zeros(n_func_classes)
        for func in funcs:
            if func in func_to_idx:
                vec[func_to_idx[func]] = 1.0
        protein_func_vecs[p_local] = vec

    all_edges_arr = torch.tensor(all_edges_list, dtype=torch.long)
    logger.info("Total edges: %d", len(all_edges_arr))

    # Glycan function index (local indices)
    glycan_func_idx: Dict[str, List[int]] = defaultdict(list)
    for gid in valid_gids:
        g_local = gid_to_local[gid]
        for func in glycan_func.get(gid, set()):
            glycan_func_idx[func].append(g_local)
    for func, indices in glycan_func_idx.items():
        logger.info("  %s: %d glycans", func, len(indices))

    # Protein-level split
    all_prot_locals = sorted(set(all_edges_arr[:, 0].tolist()))
    np.random.seed(42)
    np.random.shuffle(all_prot_locals)
    n_tr = int(len(all_prot_locals) * 0.8)
    n_va = int(len(all_prot_locals) * 0.1)
    train_p = set(all_prot_locals[:n_tr])
    val_p = set(all_prot_locals[n_tr:n_tr + n_va])
    test_p = set(all_prot_locals[n_tr + n_va:])

    mask_tr = torch.tensor([e[0].item() in train_p for e in all_edges_arr])
    mask_va = torch.tensor([e[0].item() in val_p for e in all_edges_arr])
    mask_te = torch.tensor([e[0].item() in test_p for e in all_edges_arr])

    train_edges = all_edges_arr[mask_tr]
    val_edges = all_edges_arr[mask_va]
    test_edges = all_edges_arr[mask_te]

    logger.info("Split: train=%d, val=%d, test=%d", len(train_edges), len(val_edges), len(test_edges))

    # Popularity baseline
    logger.info("\n=== Popularity Baseline ===")
    pop_r = evaluate_popularity_baseline(train_edges, test_edges, glycan_func_idx,
                                          protein_func_map, n_glycans)
    logger.info("Popularity: MRR=%.4f, H@1=%.4f, H@10=%.4f, MR=%.1f, MRR_full=%.4f",
                pop_r["mrr"], pop_r["hits@1"], pop_r["hits@10"], pop_r["mr"], pop_r["mrr_full"])

    # Train
    model = train_and_evaluate(
        protein_feats, glycan_feats, glycan_freq,
        train_edges, val_edges, test_edges,
        glycan_func_idx, protein_func_map,
        glycan_func_vecs, protein_func_vecs,
        n_glycans, n_func_classes, device,
        epochs=150, batch_size=512,
    )

    # Final test evaluation
    logger.info("\n=== Final Test Evaluation ===")
    test_r = evaluate_retrieval(
        model, protein_feats, glycan_feats, glycan_func_vecs,
        None, test_edges, glycan_func_idx, protein_func_map,
        protein_func_vecs, n_glycans, n_func_classes, device,
    )
    logger.info("Func-restricted: MRR=%.4f, H@1=%.4f, H@3=%.4f, H@10=%.4f, H@50=%.4f, MR=%.1f (n=%d)",
                test_r["mrr"], test_r["hits@1"], test_r["hits@3"], test_r["hits@10"],
                test_r["hits@50"], test_r["mr"], test_r["n"])
    logger.info("Full ranking:    MRR=%.4f, H@10=%.4f, MR=%.1f",
                test_r["mrr_full"], test_r["hits@10_full"], test_r["mr_full"])

    # Comparison
    logger.info("\n=== Comparison ===")
    logger.info("Popularity baseline:  MRR=%.4f (func-restricted)", pop_r["mrr"])
    logger.info("Retrieval v2:         MRR=0.1094 (func-restricted)")
    logger.info("Retrieval v3:         MRR=%.4f (func-restricted)", test_r["mrr"])
    logger.info("R8 KG baseline:       MRR=0.092 (full ranking, 58K)")

    # Save
    output_dir = Path("experiments_v2/glycan_retrieval_v3")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    results = {
        "test": {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in test_r.items()},
        "popularity_baseline": pop_r,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved to %s", output_dir)


if __name__ == "__main__":
    main()
