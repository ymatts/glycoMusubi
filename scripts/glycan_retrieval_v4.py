#!/usr/bin/env python3
"""Protein→Glycan retrieval v4 — KG protein embeddings + motif-cluster evaluation.

Key changes from v3:
1. Use KG-learned protein embeddings (256-dim) COMBINED with ESM2 (1280-dim)
2. Motif-based glycan clustering for practical evaluation
3. Subsumption-based glycan similarity for soft ranking
4. Evaluation at both exact-ID and cluster level
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F_func
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Data Loading ──────────────────────────────────────────────────────────


def load_data():
    data_dir = Path("data_clean")
    edges = pd.read_csv(data_dir / "edges_glycan_protein.tsv", sep="\t")
    edges = edges[["glycan_id", "protein_id"]].drop_duplicates()
    func = pd.read_csv(data_dir / "glycan_function_labels.tsv", sep="\t")
    func = func[["glycan_id", "function_term"]].drop_duplicates()
    motifs = pd.read_csv(data_dir / "edges_glycan_motif.tsv", sep="\t")
    motifs = motifs[["glycan_id", "motif_id"]].drop_duplicates()

    with open(data_dir / "esm2_cache" / "id_to_idx.json") as f:
        esm_id_to_idx = json.load(f)

    r8_dir = Path("experiments_v2/glycokgnet_inductive_r8")
    ds = torch.load(r8_dir / "dataset.pt", map_location="cpu", weights_only=False)
    node_mappings = ds["node_mappings"]
    glycan_map = node_mappings["glycan"]
    protein_map = node_mappings["protein"]

    ckpt = torch.load(r8_dir / "best.pt", map_location="cpu", weights_only=False)
    model_state = ckpt["model_state_dict"]

    return edges, func, motifs, esm_id_to_idx, data_dir / "esm2_cache", glycan_map, protein_map, model_state


def _resolve_esm2(pid, id_to_idx):
    if pid in id_to_idx:
        return id_to_idx[pid]
    base = pid.split("-")[0]
    if f"{base}-1" in id_to_idx:
        return id_to_idx[f"{base}-1"]
    if base in id_to_idx:
        return id_to_idx[base]
    return None


# ── Glycan Clustering ─────────────────────────────────────────────────────


def build_motif_clusters(motifs_df: pd.DataFrame, valid_gids: List[str],
                         n_clusters: int = 100) -> Dict[str, int]:
    """Cluster glycans by motif co-occurrence using hierarchical clustering."""
    gid_set = set(valid_gids)
    motifs_filtered = motifs_df[motifs_df["glycan_id"].isin(gid_set)]

    # Build motif multi-hot vectors
    all_motifs = sorted(motifs_filtered["motif_id"].unique())
    motif_to_idx = {m: i for i, m in enumerate(all_motifs)}

    gid_to_local = {g: i for i, g in enumerate(valid_gids)}
    n_g = len(valid_gids)
    n_m = len(all_motifs)

    # Sparse motif matrix
    motif_mat = np.zeros((n_g, n_m), dtype=np.float32)
    for _, row in motifs_filtered.iterrows():
        g = gid_to_local.get(row["glycan_id"])
        m = motif_to_idx.get(row["motif_id"])
        if g is not None and m is not None:
            motif_mat[g, m] = 1.0

    # Glycans without motifs get their own singleton cluster
    has_motif = motif_mat.sum(axis=1) > 0
    n_has_motif = has_motif.sum()
    logger.info("Glycans with motifs: %d / %d", n_has_motif, n_g)

    # Cluster only glycans with motifs
    if n_has_motif > 1:
        motif_sub = motif_mat[has_motif]
        dist = pdist(motif_sub, metric="jaccard")
        Z = linkage(dist, method="ward")
        cluster_labels = fcluster(Z, t=min(n_clusters, n_has_motif), criterion="maxclust")
    else:
        cluster_labels = np.array([1])

    # Assign clusters
    glycan_cluster: Dict[str, int] = {}
    motif_idx = 0
    next_cluster = int(cluster_labels.max()) + 1 if len(cluster_labels) > 0 else 1

    for i, gid in enumerate(valid_gids):
        if has_motif[i]:
            glycan_cluster[gid] = int(cluster_labels[motif_idx])
            motif_idx += 1
        else:
            glycan_cluster[gid] = next_cluster
            next_cluster += 1

    n_actual_clusters = len(set(glycan_cluster.values()))
    logger.info("Glycan clusters: %d (target=%d)", n_actual_clusters, n_clusters)

    # Cluster size distribution
    cluster_sizes = defaultdict(int)
    for c in glycan_cluster.values():
        cluster_sizes[c] += 1
    sizes = sorted(cluster_sizes.values(), reverse=True)
    logger.info("  Top-5 cluster sizes: %s", sizes[:5])
    logger.info("  Singleton clusters: %d", sum(1 for s in sizes if s == 1))

    return glycan_cluster


# ── Model ─────────────────────────────────────────────────────────────────


class DualEncoderModel(nn.Module):
    """Protein (ESM2+KG) → Glycan (KG) retrieval model."""

    def __init__(self, esm2_dim=1280, kg_protein_dim=256, glycan_dim=256,
                 embed_dim=256, hidden=512, n_glycans=3345, dropout=0.2):
        super().__init__()
        protein_dim = esm2_dim + kg_protein_dim  # 1536

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

        self.glycan_proj = nn.Sequential(
            nn.Linear(glycan_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
        )

        # Learned glycan popularity bias
        self.glycan_bias = nn.Embedding(n_glycans, 1)
        self.temp = nn.Parameter(torch.tensor(0.07))

    def encode_protein(self, x):
        return F_func.normalize(self.protein_proj(x), dim=-1)

    def encode_glycan(self, x):
        return F_func.normalize(self.glycan_proj(x), dim=-1)

    def score_batch(self, p_emb, all_g_emb, all_bias):
        """Score all glycans for a batch of proteins. Returns [B, N_g]."""
        temp = self.temp.abs().clamp(min=0.01)
        sim = torch.matmul(p_emb, all_g_emb.T) / temp  # [B, N_g]
        return sim + all_bias.unsqueeze(0)


# ── Training ──────────────────────────────────────────────────────────────


def train_model(
    protein_feats: Dict[int, torch.Tensor],
    glycan_feats: torch.Tensor,
    glycan_freq: torch.Tensor,
    train_edges: torch.Tensor,
    val_edges: torch.Tensor,
    glycan_func_idx: Dict[str, List[int]],
    protein_func_map: Dict[int, Set[str]],
    n_glycans: int,
    device: str,
    epochs: int = 200,
    num_neg: int = 256,
    batch_size: int = 512,
    lr: float = 3e-4,
):
    glycan_feats = glycan_feats.to(device)

    # Detect protein dim
    sample_feat = next(iter(protein_feats.values()))
    protein_dim = sample_feat.shape[0]
    logger.info("Protein feature dim: %d", protein_dim)

    model = DualEncoderModel(
        esm2_dim=protein_dim - 256 if protein_dim > 256 else protein_dim,
        kg_protein_dim=256 if protein_dim > 256 else 0,
        glycan_dim=glycan_feats.shape[1],
        n_glycans=n_glycans,
    ).to(device)

    # Override protein_proj input dim
    model.protein_proj[0] = nn.Linear(protein_dim, 512).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6)

    # Init popularity bias
    with torch.no_grad():
        log_freq = torch.log1p(glycan_freq.to(device))
        log_freq = (log_freq - log_freq.mean()) / (log_freq.std() + 1e-8)
        model.glycan_bias.weight.data[:, 0] = log_freq

    train_pos = defaultdict(set)
    for i in range(train_edges.shape[0]):
        train_pos[train_edges[i, 0].item()].add(train_edges[i, 1].item())

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

            for j in range(batch.shape[0]):
                pid = batch[j, 0].item()
                gid = batch[j, 1].item()
                if pid not in protein_feats:
                    continue
                p_feats_list.append(protein_feats[pid])
                pos_g_idx.append(gid)

            if not p_feats_list:
                continue

            B = len(p_feats_list)
            p_batch = torch.stack(p_feats_list).to(device)
            p_emb = model.encode_protein(p_batch)

            pos_idx_t = torch.tensor(pos_g_idx, dtype=torch.long, device=device)
            pos_g_emb = model.encode_glycan(glycan_feats[pos_idx_t])

            # Sample hard negatives: uniform from all glycans
            neg_idx = torch.randint(0, n_glycans, (num_neg,), device=device)
            neg_g_emb = model.encode_glycan(glycan_feats[neg_idx])

            temp = model.temp.abs().clamp(min=0.01)

            # Positive scores
            pos_scores = (p_emb * pos_g_emb).sum(dim=-1) / temp
            pos_bias = model.glycan_bias(pos_idx_t).squeeze(-1)
            pos_scores = pos_scores + pos_bias

            # Negative scores [B, K]
            neg_sim = torch.matmul(p_emb, neg_g_emb.T) / temp
            neg_bias = model.glycan_bias(neg_idx).squeeze(-1).unsqueeze(0).expand(B, -1)
            neg_scores = neg_sim + neg_bias

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

        if (epoch + 1) % 5 == 0:
            val_r = _quick_eval(model, protein_feats, glycan_feats, val_edges, n_glycans, device)
            lr_now = scheduler.get_last_lr()[0]
            logger.info("Epoch %d: loss=%.4f lr=%.1e val_MRR=%.4f H@1=%.4f H@10=%.4f",
                        epoch + 1, avg_loss, lr_now, val_r["mrr"], val_r["hits@1"], val_r["hits@10"])

            if val_r["mrr"] > best_val_mrr:
                best_val_mrr = val_r["mrr"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= 10:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

    if best_state:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def _quick_eval(model, protein_feats, glycan_feats, edges, n_glycans, device):
    """Quick MRR evaluation (full ranking, no func restriction)."""
    model.eval()
    glycan_feats_d = glycan_feats.to(device)
    all_g_emb = model.encode_glycan(glycan_feats_d)
    all_bias = model.glycan_bias.weight.squeeze(-1)

    ranks = []
    for i in range(edges.shape[0]):
        pid = edges[i, 0].item()
        gid = edges[i, 1].item()
        if pid not in protein_feats:
            continue

        p_feat = protein_feats[pid].unsqueeze(0).to(device)
        p_emb = model.encode_protein(p_feat)
        scores = model.score_batch(p_emb, all_g_emb, all_bias).squeeze(0)
        rank = (scores > scores[gid]).sum().item() + 1
        ranks.append(rank)

    ra = np.array(ranks)
    return {
        "mrr": float((1.0 / ra).mean()) if len(ra) > 0 else 0,
        "hits@1": float((ra <= 1).mean()) if len(ra) > 0 else 0,
        "hits@10": float((ra <= 10).mean()) if len(ra) > 0 else 0,
    }


@torch.no_grad()
def evaluate_full(model, protein_feats, glycan_feats, edges,
                  glycan_func_idx, protein_func_map, glycan_cluster,
                  gid_to_local, valid_gids, n_glycans, device):
    """Full evaluation with exact, func-restricted, and cluster-level metrics."""
    model.eval()
    glycan_feats_d = glycan_feats.to(device)
    all_g_emb = model.encode_glycan(glycan_feats_d)
    all_bias = model.glycan_bias.weight.squeeze(-1)

    ranks_full = []
    ranks_func = []
    cluster_correct = {1: 0, 3: 0, 5: 0, 10: 0}
    n_eval = 0

    # Build local_idx → cluster mapping
    local_to_cluster = {}
    for gid, local_idx in gid_to_local.items():
        local_to_cluster[local_idx] = glycan_cluster.get(gid, -1)

    for i in range(edges.shape[0]):
        pid = edges[i, 0].item()
        gid = edges[i, 1].item()
        if pid not in protein_feats:
            continue

        p_feat = protein_feats[pid].unsqueeze(0).to(device)
        p_emb = model.encode_protein(p_feat)
        scores = model.score_batch(p_emb, all_g_emb, all_bias).squeeze(0)

        # Full ranking
        rank_full = (scores > scores[gid]).sum().item() + 1
        ranks_full.append(rank_full)

        # Function-restricted ranking
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

        # Cluster-level: is any glycan from the correct cluster in top-K?
        target_cluster = local_to_cluster.get(gid, -1)
        if target_cluster != -1:
            _, top_indices = scores.topk(50)
            top_clusters = [local_to_cluster.get(idx.item(), -2) for idx in top_indices]
            for k in cluster_correct:
                if target_cluster in top_clusters[:k]:
                    cluster_correct[k] += 1
            n_eval += 1

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

    # Cluster metrics
    if n_eval > 0:
        for k in cluster_correct:
            results[f"cluster_hits@{k}"] = cluster_correct[k] / n_eval
        results["cluster_n"] = n_eval

    return results


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    edges_df, func_df, motifs_df, esm_id2idx, esm2_dir, glycan_map, protein_map, model_state = load_data()

    # Glycan features
    glycan_node_feat = model_state["node_embeddings.glycan.weight"]
    protein_node_feat = model_state["node_embeddings.protein.weight"]
    logger.info("Glycan embeddings: %s, Protein embeddings: %s",
                glycan_node_feat.shape, protein_node_feat.shape)

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

    # Glycan function maps
    glycan_func: Dict[str, Set[str]] = defaultdict(set)
    for _, row in func_df.iterrows():
        glycan_func[row["glycan_id"]].add(row["function_term"])

    # Build glycan clusters
    glycan_cluster = build_motif_clusters(motifs_df, valid_gids, n_clusters=100)

    # Glycan frequency
    glycan_freq_count = edges_df["glycan_id"].value_counts()
    glycan_freq = torch.zeros(n_glycans)
    for gid in valid_gids:
        glycan_freq[gid_to_local[gid]] = glycan_freq_count.get(gid, 0)

    # Load protein features: ESM2 + KG embeddings
    protein_feats: Dict[int, torch.Tensor] = {}
    pid_to_local: Dict[str, int] = {}
    local_idx = 0

    for pid in sorted(edges_df["protein_id"].unique()):
        idx = _resolve_esm2(pid, esm_id2idx)
        if idx is None:
            continue
        pt_path = esm2_dir / f"{idx}.pt"
        if not pt_path.exists():
            continue

        esm2_emb = torch.load(pt_path, map_location="cpu", weights_only=True)
        if esm2_emb.dim() == 2:
            esm2_emb = esm2_emb.mean(dim=0)  # [1280]

        # KG protein embedding
        if pid in protein_map:
            kg_emb = protein_node_feat[protein_map[pid]]  # [256]
        else:
            kg_emb = torch.zeros(256)

        # Concatenate ESM2 + KG
        combined = torch.cat([esm2_emb, kg_emb])  # [1536]

        pid_to_local[pid] = local_idx
        protein_feats[local_idx] = combined
        local_idx += 1

    logger.info("Protein features: %d (dim=%d)", len(protein_feats), protein_feats[0].shape[0])

    # Build edges and function maps
    protein_func_map: Dict[int, Set[str]] = defaultdict(set)
    all_edges_list = []
    for _, row in edges_df.iterrows():
        pid, gid = row["protein_id"], row["glycan_id"]
        if pid in pid_to_local and gid in gid_to_local:
            p_local = pid_to_local[pid]
            g_local = gid_to_local[gid]
            all_edges_list.append((p_local, g_local))
            for func in glycan_func.get(gid, set()):
                protein_func_map[p_local].add(func)

    all_edges_arr = torch.tensor(all_edges_list, dtype=torch.long)
    logger.info("Total edges: %d", len(all_edges_arr))

    glycan_func_idx: Dict[str, List[int]] = defaultdict(list)
    for gid in valid_gids:
        for func in glycan_func.get(gid, set()):
            glycan_func_idx[func].append(gid_to_local[gid])

    # Split
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

    # === Experiment 1: ESM2 only ===
    logger.info("\n" + "=" * 60)
    logger.info("=== Experiment 1: ESM2 only (1280-dim) ===")
    logger.info("=" * 60)

    esm2_only_feats: Dict[int, torch.Tensor] = {}
    for k, v in protein_feats.items():
        esm2_only_feats[k] = v[:1280]  # ESM2 part only

    model_esm2 = train_model(
        esm2_only_feats, glycan_feats, glycan_freq,
        train_edges, val_edges, glycan_func_idx, protein_func_map,
        n_glycans, device, epochs=200, lr=3e-4,
    )
    test_r_esm2 = evaluate_full(
        model_esm2, esm2_only_feats, glycan_feats, test_edges,
        glycan_func_idx, protein_func_map, glycan_cluster,
        gid_to_local, valid_gids, n_glycans, device,
    )

    # === Experiment 2: ESM2 + KG ===
    logger.info("\n" + "=" * 60)
    logger.info("=== Experiment 2: ESM2 + KG (1536-dim) ===")
    logger.info("=" * 60)

    model_combo = train_model(
        protein_feats, glycan_feats, glycan_freq,
        train_edges, val_edges, glycan_func_idx, protein_func_map,
        n_glycans, device, epochs=200, lr=3e-4,
    )
    test_r_combo = evaluate_full(
        model_combo, protein_feats, glycan_feats, test_edges,
        glycan_func_idx, protein_func_map, glycan_cluster,
        gid_to_local, valid_gids, n_glycans, device,
    )

    # === Experiment 3: KG only ===
    logger.info("\n" + "=" * 60)
    logger.info("=== Experiment 3: KG only (256-dim) ===")
    logger.info("=" * 60)

    kg_only_feats: Dict[int, torch.Tensor] = {}
    for k, v in protein_feats.items():
        kg_only_feats[k] = v[1280:]  # KG part only

    model_kg = train_model(
        kg_only_feats, glycan_feats, glycan_freq,
        train_edges, val_edges, glycan_func_idx, protein_func_map,
        n_glycans, device, epochs=200, lr=3e-4,
    )
    test_r_kg = evaluate_full(
        model_kg, kg_only_feats, glycan_feats, test_edges,
        glycan_func_idx, protein_func_map, glycan_cluster,
        gid_to_local, valid_gids, n_glycans, device,
    )

    # === Summary ===
    logger.info("\n" + "=" * 60)
    logger.info("=== SUMMARY ===")
    logger.info("=" * 60)

    for name, r in [("ESM2 only", test_r_esm2), ("ESM2+KG", test_r_combo), ("KG only", test_r_kg)]:
        logger.info("\n--- %s ---", name)
        logger.info("Exact ID:  MRR=%.4f, H@1=%.4f, H@3=%.4f, H@10=%.4f, H@50=%.4f (n=%d)",
                     r["mrr"], r["hits@1"], r["hits@3"], r["hits@10"], r["hits@50"], r["n"])
        logger.info("Full rank: MRR=%.4f, H@10=%.4f, MR=%.1f",
                     r["mrr_full"], r["hits@10_full"], r["mr_full"])
        if "cluster_hits@1" in r:
            logger.info("Cluster:   H@1=%.4f, H@3=%.4f, H@5=%.4f, H@10=%.4f (n=%d)",
                         r.get("cluster_hits@1", 0), r.get("cluster_hits@3", 0),
                         r.get("cluster_hits@5", 0), r.get("cluster_hits@10", 0),
                         r.get("cluster_n", 0))

    logger.info("\nBaselines:")
    logger.info("Popularity: MRR~0.104 (func-restricted)")
    logger.info("v2 (ESM2): MRR=0.109")
    logger.info("v3 (hybrid): MRR=0.118")
    logger.info("R8 KG: MRR=0.092 (full)")

    # Save
    output_dir = Path("experiments_v2/glycan_retrieval_v4")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "esm2_only": {k: float(v) if isinstance(v, (np.floating, float)) else int(v) for k, v in test_r_esm2.items()},
        "esm2_kg": {k: float(v) if isinstance(v, (np.floating, float)) else int(v) for k, v in test_r_combo.items()},
        "kg_only": {k: float(v) if isinstance(v, (np.floating, float)) else int(v) for k, v in test_r_kg.items()},
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    torch.save(model_combo.state_dict(), output_dir / "model_best.pt")
    logger.info("Saved to %s", output_dir)


if __name__ == "__main__":
    main()
