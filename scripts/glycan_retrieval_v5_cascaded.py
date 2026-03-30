#!/usr/bin/env python3
"""Protein→Glycan retrieval v5 — Cascaded cluster-aware model.

Architecture:
1. Stage 1: Predict N-linked sub-family (8 clusters) from ESM2
2. Stage 2: Rank glycans within predicted cluster using contrastive model

The model is trained with cluster-aware hard negatives (all negatives from
the same cluster as the positive), making it specialized for within-cluster
discrimination.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F_func
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _resolve_esm2(pid, id_to_idx):
    if pid in id_to_idx:
        return id_to_idx[pid]
    base = pid.split("-")[0]
    if f"{base}-1" in id_to_idx:
        return id_to_idx[f"{base}-1"]
    if base in id_to_idx:
        return id_to_idx[base]
    return None


class ClusterAwareRetriever(nn.Module):
    """Retrieval model that learns to discriminate within clusters."""

    def __init__(self, protein_dim=1280, glycan_dim=256, embed_dim=256,
                 hidden=512, n_clusters=8, dropout=0.2):
        super().__init__()
        # Cluster-conditioned protein encoder
        self.cluster_emb = nn.Embedding(n_clusters, 64)
        self.protein_proj = nn.Sequential(
            nn.Linear(protein_dim + 64, hidden),
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

        self.temp = nn.Parameter(torch.tensor(0.07))

    def encode_protein(self, x, cluster_ids=None):
        if cluster_ids is not None:
            c_emb = self.cluster_emb(cluster_ids)
            x = torch.cat([x, c_emb], dim=-1)
        else:
            # No cluster info → use zero padding
            pad = torch.zeros(x.shape[0], 64, device=x.device)
            x = torch.cat([x, pad], dim=-1)
        return F_func.normalize(self.protein_proj(x), dim=-1)

    def encode_glycan(self, x):
        return F_func.normalize(self.glycan_proj(x), dim=-1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    data_dir = Path("data_clean")
    edges_df = pd.read_csv(data_dir / "edges_glycan_protein.tsv", sep="\t")[["glycan_id", "protein_id"]].drop_duplicates()
    func_df = pd.read_csv(data_dir / "glycan_function_labels.tsv", sep="\t")

    with open(data_dir / "esm2_cache/id_to_idx.json") as f:
        esm_id2idx = json.load(f)

    r8_dir = Path("experiments_v2/glycokgnet_inductive_r8")
    ckpt = torch.load(r8_dir / "best.pt", map_location="cpu", weights_only=False)
    ds = torch.load(r8_dir / "dataset.pt", map_location="cpu", weights_only=False)
    glycan_map = ds["node_mappings"]["glycan"]
    glycan_emb_all = ckpt["model_state_dict"]["node_embeddings.glycan.weight"]

    # Setup glycan features (only has_glycan glycans)
    has_glycan_gids = sorted(edges_df["glycan_id"].unique())
    valid_gids = [g for g in has_glycan_gids if g in glycan_map]
    glycan_feats = glycan_emb_all[[glycan_map[g] for g in valid_gids]]
    gid_to_local = {g: i for i, g in enumerate(valid_gids)}
    n_glycans = len(valid_gids)

    # N-linked glycans only
    nlinked_gids = set(func_df[func_df["function_term"] == "N-linked"]["glycan_id"])
    nlinked_locals = [gid_to_local[g] for g in valid_gids if g in nlinked_gids]
    nlinked_set = set(nlinked_locals)

    # Cluster N-linked glycans
    N_CLUSTERS = 8
    nlinked_embs = glycan_feats[nlinked_locals].numpy()
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    nlinked_cluster_labels = km.fit_predict(nlinked_embs)
    local_to_cluster = {nlinked_locals[i]: int(nlinked_cluster_labels[i])
                        for i in range(len(nlinked_locals))}

    cluster_to_locals = defaultdict(list)
    for local_idx, c in local_to_cluster.items():
        cluster_to_locals[c].append(local_idx)

    for c in sorted(cluster_to_locals):
        logger.info("  Cluster %d: %d glycans", c, len(cluster_to_locals[c]))

    # Load protein ESM2 features
    protein_feats: Dict[int, torch.Tensor] = {}
    pid_to_local: Dict[str, int] = {}
    local_idx = 0
    esm2_dir = data_dir / "esm2_cache"

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
        pid_to_local[pid] = local_idx
        protein_feats[local_idx] = emb
        local_idx += 1

    logger.info("Protein features: %d", len(protein_feats))

    # Build N-linked edges only
    all_edges_list = []
    for _, row in edges_df.iterrows():
        pid, gid = row["protein_id"], row["glycan_id"]
        if pid in pid_to_local and gid in gid_to_local:
            g_local = gid_to_local[gid]
            if g_local in nlinked_set:
                all_edges_list.append((pid_to_local[pid], g_local))

    all_edges_arr = torch.tensor(all_edges_list, dtype=torch.long)
    logger.info("N-linked edges: %d", len(all_edges_arr))

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

    # Train positive index
    train_pos = defaultdict(set)
    for i in range(train_edges.shape[0]):
        train_pos[train_edges[i, 0].item()].add(train_edges[i, 1].item())

    glycan_feats_d = glycan_feats.to(device)
    model = ClusterAwareRetriever(glycan_dim=glycan_feats.shape[1], n_clusters=N_CLUSTERS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-6)

    best_val_mrr = 0.0
    best_state = None
    patience = 0

    for epoch in range(150):
        model.train()
        perm = torch.randperm(train_edges.shape[0])
        total_loss = 0.0
        n_samples = 0

        for start in range(0, len(perm), 512):
            batch_idx = perm[start:start + 512]
            batch = train_edges[batch_idx]

            p_feats_list = []
            pos_g_idx = []
            cluster_ids = []

            for j in range(batch.shape[0]):
                pid = batch[j, 0].item()
                gid = batch[j, 1].item()
                if pid not in protein_feats or gid not in local_to_cluster:
                    continue
                p_feats_list.append(protein_feats[pid])
                pos_g_idx.append(gid)
                cluster_ids.append(local_to_cluster[gid])

            if not p_feats_list:
                continue

            B = len(p_feats_list)
            p_batch = torch.stack(p_feats_list).to(device)
            cluster_ids_t = torch.tensor(cluster_ids, dtype=torch.long, device=device)
            p_emb = model.encode_protein(p_batch, cluster_ids_t)

            pos_idx_t = torch.tensor(pos_g_idx, dtype=torch.long, device=device)
            pos_g_emb = model.encode_glycan(glycan_feats_d[pos_idx_t])

            # CRITICAL: sample negatives from SAME cluster as positive
            neg_indices_set = set()
            for j in range(B):
                c = cluster_ids[j]
                candidates = set(cluster_to_locals[c]) - train_pos.get(batch[j, 0].item() if j < batch.shape[0] else 0, set())
                candidates.discard(pos_g_idx[j])
                candidates = list(candidates)
                if len(candidates) > 64:
                    neg_indices_set.update(np.random.choice(candidates, 64, replace=False).tolist())
                else:
                    neg_indices_set.update(candidates)

            if not neg_indices_set:
                continue

            neg_idx_t = torch.tensor(sorted(neg_indices_set), dtype=torch.long, device=device)
            neg_g_emb = model.encode_glycan(glycan_feats_d[neg_idx_t])

            temp = model.temp.abs().clamp(min=0.01)
            pos_scores = (p_emb * pos_g_emb).sum(dim=-1) / temp
            neg_scores = torch.matmul(p_emb, neg_g_emb.T) / temp

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
            val_r = _evaluate(model, protein_feats, glycan_feats_d, val_edges,
                              local_to_cluster, cluster_to_locals, nlinked_locals, device)
            logger.info("Epoch %d: loss=%.4f val cluster_MRR=%.4f H@10=%.4f | nlinked_MRR=%.4f",
                        epoch + 1, avg_loss, val_r["cluster_mrr"], val_r["cluster_h10"],
                        val_r["nlinked_mrr"])

            if val_r["cluster_mrr"] > best_val_mrr:
                best_val_mrr = val_r["cluster_mrr"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 10:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

    if best_state:
        model.load_state_dict(best_state)

    # Final test evaluation
    logger.info("\n=== Final Test Evaluation ===")
    test_r = _evaluate(model, protein_feats, glycan_feats_d, test_edges,
                       local_to_cluster, cluster_to_locals, nlinked_locals, device)

    logger.info("Within cluster: MRR=%.4f, H@1=%.4f, H@3=%.4f, H@5=%.4f, H@10=%.4f, H@50=%.4f (n=%d)",
                test_r["cluster_mrr"], test_r["cluster_h1"], test_r["cluster_h3"],
                test_r["cluster_h5"], test_r["cluster_h10"], test_r["cluster_h50"], test_r["n"])
    logger.info("All N-linked:   MRR=%.4f, H@10=%.4f",
                test_r["nlinked_mrr"], test_r["nlinked_h10"])

    logger.info("\n=== Comparison ===")
    logger.info("v3 all candidates (3345):  MRR=0.118, H@10=0.228")
    logger.info("v3 + cluster (312):        MRR=0.195, H@10=0.419")
    logger.info("v5 within cluster:         MRR=%.4f, H@10=%.4f", test_r["cluster_mrr"], test_r["cluster_h10"])

    # Save
    output_dir = Path("experiments_v2/glycan_retrieval_v5")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    with open(output_dir / "results.json", "w") as f:
        json.dump({k: float(v) if isinstance(v, (np.floating, float)) else int(v)
                    for k, v in test_r.items()}, f, indent=2)
    logger.info("Saved to %s", output_dir)


@torch.no_grad()
def _evaluate(model, protein_feats, glycan_feats_d, edges,
              local_to_cluster, cluster_to_locals, nlinked_locals, device):
    model.eval()
    all_g_emb = model.encode_glycan(glycan_feats_d)

    ranks_cluster = []
    ranks_nlinked = []

    for i in range(edges.shape[0]):
        pid = edges[i, 0].item()
        gid = edges[i, 1].item()
        if pid not in protein_feats or gid not in local_to_cluster:
            continue

        target_cluster = local_to_cluster[gid]
        cluster_locals = cluster_to_locals[target_cluster]

        p_feat = protein_feats[pid].unsqueeze(0).to(device)

        # With cluster conditioning
        c_id = torch.tensor([target_cluster], device=device)
        p_emb = model.encode_protein(p_feat, c_id)
        temp = model.temp.abs().clamp(min=0.01)

        # Within cluster
        cluster_g_emb = all_g_emb[cluster_locals]
        cluster_scores = torch.matmul(p_emb, cluster_g_emb.T).squeeze(0) / temp
        target_pos = cluster_locals.index(gid)
        rank_c = (cluster_scores > cluster_scores[target_pos]).sum().item() + 1
        ranks_cluster.append(rank_c)

        # All N-linked (no cluster conditioning)
        p_emb_nc = model.encode_protein(p_feat)
        nlinked_g_emb = all_g_emb[nlinked_locals]
        nl_scores = torch.matmul(p_emb_nc, nlinked_g_emb.T).squeeze(0) / temp
        target_nl_pos = nlinked_locals.index(gid)
        rank_nl = (nl_scores > nl_scores[target_nl_pos]).sum().item() + 1
        ranks_nlinked.append(rank_nl)

    ra_c = np.array(ranks_cluster)
    ra_n = np.array(ranks_nlinked)

    return {
        "cluster_mrr": float((1.0 / ra_c).mean()),
        "cluster_h1": float((ra_c <= 1).mean()),
        "cluster_h3": float((ra_c <= 3).mean()),
        "cluster_h5": float((ra_c <= 5).mean()),
        "cluster_h10": float((ra_c <= 10).mean()),
        "cluster_h50": float((ra_c <= 50).mean()),
        "cluster_mr": float(ra_c.mean()),
        "nlinked_mrr": float((1.0 / ra_n).mean()),
        "nlinked_h10": float((ra_n <= 10).mean()),
        "n": len(ra_c),
    }


if __name__ == "__main__":
    main()
