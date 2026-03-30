#!/usr/bin/env python3
"""Protein→Glycan retrieval v2 — using KG-learned embeddings.

Uses R8 model's learned glycan embeddings (256-dim) + ESM2 protein embeddings.
Much faster and more informative than WURCS features.

Architecture: learned projection + contrastive learning in shared space.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F_func

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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
    glycan_map = node_mappings["glycan"]  # glycan_id -> local_idx
    protein_map = node_mappings["protein"]

    # Load trained model checkpoint to get learned embeddings
    ckpt = torch.load(r8_dir / "best.pt", map_location="cpu", weights_only=False)
    model_state = ckpt["model_state_dict"]

    return edges, func, esm_id_to_idx, data_dir / "esm2_cache", glycan_map, protein_map, model_state


def _resolve_esm2(pid, id_to_idx):
    if pid in id_to_idx:
        return id_to_idx[pid]
    base = pid.split("-")[0]
    if f"{base}-1" in id_to_idx:
        return id_to_idx[f"{base}-1"]
    if base in id_to_idx:
        return id_to_idx[base]
    return None


class RetrievalModel(nn.Module):
    def __init__(self, protein_dim=1280, glycan_dim=256, embed_dim=256, hidden=512, dropout=0.2):
        super().__init__()
        self.protein_proj = nn.Sequential(
            nn.Linear(protein_dim, hidden),
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

    def encode_protein(self, x):
        return F_func.normalize(self.protein_proj(x), dim=-1)

    def encode_glycan(self, x):
        return F_func.normalize(self.glycan_proj(x), dim=-1)


def train_and_evaluate(
    protein_feats: Dict[int, torch.Tensor],
    glycan_feats: torch.Tensor,
    train_edges: torch.Tensor,
    val_edges: torch.Tensor,
    test_edges: torch.Tensor,
    glycan_func_idx: Dict[str, List[int]],
    protein_func_map: Dict[int, Set[str]],
    n_glycans: int,
    device: str = "cpu",
    epochs: int = 100,
    num_neg: int = 128,
):
    glycan_feats = glycan_feats.to(device)
    model = RetrievalModel(glycan_dim=glycan_feats.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)

    # Build positive index
    train_pos = defaultdict(set)
    for i in range(train_edges.shape[0]):
        pid, gid = train_edges[i, 0].item(), train_edges[i, 1].item()
        train_pos[pid].add(gid)

    # Function class candidate indices
    func_candidates: Dict[str, torch.Tensor] = {}
    for func, indices in glycan_func_idx.items():
        func_candidates[func] = torch.tensor(indices, dtype=torch.long, device=device)

    best_val_mrr = 0.0
    best_state = None
    patience_count = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(train_edges.shape[0])
        total_loss = 0.0
        n_batches = 0

        for start in range(0, len(perm), 256):
            batch_idx = perm[start:start + 256]
            batch = train_edges[batch_idx]

            # Get protein embeddings
            p_feats = []
            pos_g_idx = []
            neg_g_indices = []
            valid_mask = []

            for j in range(batch.shape[0]):
                pid = batch[j, 0].item()
                gid = batch[j, 1].item()
                if pid not in protein_feats:
                    valid_mask.append(False)
                    continue
                valid_mask.append(True)
                p_feats.append(protein_feats[pid])
                pos_g_idx.append(gid)

                # Sample negatives from same function class
                funcs = protein_func_map.get(pid, set())
                candidates = set()
                for func in funcs:
                    for idx in glycan_func_idx.get(func, []):
                        candidates.add(idx)
                if not candidates:
                    candidates = set(range(n_glycans))
                candidates -= train_pos.get(pid, set())
                candidates = list(candidates)
                if len(candidates) > num_neg:
                    neg = np.random.choice(candidates, num_neg, replace=False)
                else:
                    neg = np.array(candidates) if candidates else np.array([0])
                neg_g_indices.append(torch.tensor(neg, dtype=torch.long, device=device))

            if not p_feats:
                continue

            p_batch = torch.stack(p_feats).to(device)
            p_emb = model.encode_protein(p_batch)  # [B, D]

            # Batch InfoNCE
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device)
            B = p_emb.shape[0]

            # Vectorized: all positives + shared negatives
            pos_g = glycan_feats[torch.tensor(pos_g_idx, device=device)]
            pos_emb = model.encode_glycan(pos_g)  # [B, D]

            # Positive scores
            pos_scores = (p_emb * pos_emb).sum(dim=-1) / model.temp.abs().clamp(min=0.01)

            # Negative scores (sample shared negatives for efficiency)
            all_neg = torch.cat(neg_g_indices)
            unique_neg = torch.unique(all_neg)
            if len(unique_neg) > num_neg * 2:
                unique_neg = unique_neg[torch.randperm(len(unique_neg))[:num_neg * 2]]
            neg_emb = model.encode_glycan(glycan_feats[unique_neg])  # [K, D]
            neg_scores = torch.matmul(p_emb, neg_emb.T) / model.temp.abs().clamp(min=0.01)  # [B, K]

            # InfoNCE: positive should be highest
            logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)  # [B, 1+K]
            target = torch.zeros(B, dtype=torch.long, device=device)
            loss = F_func.cross_entropy(logits, target)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * B
            n_batches += 1

        avg_loss = total_loss / max(train_edges.shape[0], 1)

        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0:
            val_r = evaluate_retrieval(model, protein_feats, glycan_feats, val_edges,
                                        glycan_func_idx, protein_func_map, n_glycans, device)
            logger.info("Epoch %d: loss=%.4f, val_MRR=%.4f, H@1=%.4f, H@10=%.4f",
                        epoch + 1, avg_loss, val_r["mrr"], val_r["hits@1"], val_r["hits@10"])

            if val_r["mrr"] > best_val_mrr:
                best_val_mrr = val_r["mrr"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= 6:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate_retrieval(
    model, protein_feats, glycan_feats, edges, glycan_func_idx,
    protein_func_map, n_glycans, device, restrict_func=True,
):
    model.eval()
    glycan_feats = glycan_feats.to(device)
    all_g_emb = model.encode_glycan(glycan_feats)  # [N_g, D]

    ranks_full = []
    ranks_func = []

    for i in range(edges.shape[0]):
        pid = edges[i, 0].item()
        gid = edges[i, 1].item()
        if pid not in protein_feats:
            continue

        p_feat = protein_feats[pid].unsqueeze(0).to(device)
        p_emb = model.encode_protein(p_feat)
        scores = torch.matmul(p_emb, all_g_emb.T).squeeze(0)

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
        "mrr": (1.0 / ra_func).mean() if len(ra_func) > 0 else 0,
        "hits@1": (ra_func <= 1).mean() if len(ra_func) > 0 else 0,
        "hits@3": (ra_func <= 3).mean() if len(ra_func) > 0 else 0,
        "hits@10": (ra_func <= 10).mean() if len(ra_func) > 0 else 0,
        "mr": ra_func.mean() if len(ra_func) > 0 else 0,
        "mrr_full": (1.0 / ra_full).mean() if len(ra_full) > 0 else 0,
        "hits@10_full": (ra_full <= 10).mean() if len(ra_full) > 0 else 0,
        "mr_full": ra_full.mean() if len(ra_full) > 0 else 0,
        "n": len(ra_func),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    edges_df, func_df, esm_id2idx, esm2_dir, glycan_map, protein_map, model_state = load_data()

    # Build glycan features from KG model embeddings
    # Find glycan embeddings in model state
    glycan_embed_key = None
    for key in model_state:
        if "glycan" in key.lower() and "embed" in key.lower():
            logger.info("  candidate key: %s, shape=%s", key, model_state[key].shape)

    # Use LEARNED glycan embeddings from trained model
    glycan_node_feat = model_state["node_embeddings.glycan.weight"]  # [58628, 256]
    logger.info("Glycan LEARNED embeddings: %s", glycan_node_feat.shape)

    # Map has_glycan glycans to their indices
    has_glycan_gids = sorted(edges_df["glycan_id"].unique())
    glycan_local_indices = []
    valid_gids = []
    for gid in has_glycan_gids:
        if gid in glycan_map:
            glycan_local_indices.append(glycan_map[gid])
            valid_gids.append(gid)

    logger.info("has_glycan glycans with KG embedding: %d / %d", len(valid_gids), len(has_glycan_gids))

    glycan_feats = glycan_node_feat[glycan_local_indices]  # [N_g, 256]
    gid_to_local = {g: i for i, g in enumerate(valid_gids)}
    n_glycans = len(valid_gids)

    # Load protein ESM2 features
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
        emb = torch.load(pt_path, map_location="cpu", weights_only=True)
        if emb.dim() == 2:
            emb = emb.mean(dim=0)
        pid_to_local[pid] = local_idx
        protein_feats[local_idx] = emb
        local_idx += 1

    logger.info("Protein features: %d", len(protein_feats))

    # Build function maps
    glycan_func: Dict[str, Set[str]] = defaultdict(set)
    for _, row in func_df.iterrows():
        glycan_func[row["glycan_id"]].add(row["function_term"])

    protein_func_map: Dict[int, Set[str]] = defaultdict(set)

    # Build edge index: (protein_local, glycan_local)
    all_edges_list = []
    for _, row in edges_df.iterrows():
        pid, gid = row["protein_id"], row["glycan_id"]
        if pid in pid_to_local and gid in gid_to_local:
            p_local = pid_to_local[pid]
            g_local = gid_to_local[gid]
            all_edges_list.append((p_local, g_local))
            # Track protein→function
            for func in glycan_func.get(gid, set()):
                protein_func_map[p_local].add(func)

    all_edges_arr = torch.tensor(all_edges_list, dtype=torch.long)
    logger.info("Total edges: %d", len(all_edges_arr))

    # Build glycan function index (local indices)
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

    # Train
    model = train_and_evaluate(
        protein_feats, glycan_feats, train_edges, val_edges, test_edges,
        glycan_func_idx, protein_func_map, n_glycans, device, epochs=100,
    )

    # Final test evaluation
    logger.info("\n=== Final Test Evaluation ===")
    test_r = evaluate_retrieval(
        model, protein_feats, glycan_feats, test_edges,
        glycan_func_idx, protein_func_map, n_glycans, device,
    )
    logger.info("Func-restricted: MRR=%.4f, H@1=%.4f, H@3=%.4f, H@10=%.4f, MR=%.1f (n=%d)",
                test_r["mrr"], test_r["hits@1"], test_r["hits@3"], test_r["hits@10"],
                test_r["mr"], test_r["n"])
    logger.info("Full ranking:    MRR=%.4f, H@10=%.4f, MR=%.1f",
                test_r["mrr_full"], test_r["hits@10_full"], test_r["mr_full"])

    # Compare with R8 baseline
    logger.info("\n=== Comparison with R8 KG baseline ===")
    logger.info("R8 has_glycan MRR:   0.092 (full ranking, 58K candidates)")
    logger.info("Retrieval full MRR:  %.4f (3.3K candidates)", test_r["mrr_full"])
    logger.info("Retrieval func MRR:  %.4f (func-restricted)", test_r["mrr"])

    # Save
    output_dir = Path("experiments_v2/glycan_retrieval_v2")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    with open(output_dir / "results.json", "w") as f:
        json.dump({k: float(v) if isinstance(v, (np.floating, float)) else v
                    for k, v in test_r.items()}, f, indent=2)
    logger.info("Saved to %s", output_dir)


if __name__ == "__main__":
    main()
