#!/usr/bin/env python3
"""N-linked glycosylation site predictor v5 — Ranking-focused.

Key insights:
- ~33% of negative labels are likely mislabeled (unannotated glycosylated sites)
- Binary F1 is unreliable; ranking metrics are more meaningful
- Per-protein ranking is the practical use case

Changes:
1. Ranking loss (pairwise margin) instead of pointwise BCE
2. Key position extraction (7 positions) instead of full window CNN
3. Ranking metrics: per-protein MRR, Recall@k, NDCG
4. Mixed loss: ranking + calibrated BCE
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F_func
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

AA_LIST = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}
KEY_OFFSETS = [-3, -2, -1, 0, 1, 2, 3, 4, 5]  # relative to N position


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_dataset(data_dir: str = "data_clean"):
    data_dir = Path(data_dir)
    perresidue_dir = data_dir / "esm2_perresidue"

    sites = pd.read_csv(data_dir / "uniprot_sites.tsv", sep="\t")
    n_sites = sites[sites["site_type"] == "N-linked"]
    seqs_df = pd.read_csv(data_dir / "protein_sequences.tsv", sep="\t")
    seq_dict = dict(zip(seqs_df["uniprot_id"], seqs_df["sequence"]))

    proteins = n_sites["uniprot_id"].unique()
    n_keys = len(KEY_OFFSETS)

    all_key_embs = []     # [N, n_keys, 1280]
    all_key_onehot = []   # [N, n_keys, 21]
    all_context = []      # [N, feat_dim]
    all_labels = []
    all_pids = []

    for uid in proteins:
        seq = seq_dict.get(f"{uid}-1", seq_dict.get(uid, None))
        if seq is None:
            continue
        pr_path = perresidue_dir / f"{uid}.pt"
        if not pr_path.exists():
            continue
        per_res = torch.load(pr_path, map_location="cpu", weights_only=True).float()

        confirmed = set(n_sites[n_sites["uniprot_id"] == uid]["site_position"].values)
        L = per_res.shape[0]
        seq_len = len(seq)

        for m in re.finditer(r"N[^P][ST]", seq):
            pos = m.start()
            pos_1idx = pos + 1

            # Key position ESM2 embeddings
            key_emb = torch.zeros(n_keys, per_res.shape[1])
            key_oh = torch.zeros(n_keys, 21)
            for k, offset in enumerate(KEY_OFFSETS):
                seq_idx = pos + offset
                if 0 <= seq_idx < L:
                    key_emb[k] = per_res[seq_idx]
                if 0 <= seq_idx < seq_len:
                    aa_idx = AA_TO_IDX.get(seq[seq_idx])
                    if aa_idx is not None:
                        key_oh[k, aa_idx] = 1.0
                    else:
                        key_oh[k, 20] = 1.0
                else:
                    key_oh[k, 20] = 1.0

            all_key_embs.append(key_emb)
            all_key_onehot.append(key_oh)

            # Context features
            third = seq[pos + 2] if pos + 2 < seq_len else "?"
            n_motifs = len(re.findall(r"N[^P][ST]", seq))

            # Window features (±15)
            win = seq[max(0, pos - 15):min(seq_len, pos + 18)]
            def frac(s, chars):
                return sum(1 for c in s if c in chars) / max(len(s), 1)

            ctx = torch.tensor([
                1.0 if third == "S" else 0.0,
                1.0 if third == "T" else 0.0,
                pos / max(seq_len, 1),
                (seq_len - pos) / max(seq_len, 1),
                n_motifs,
                np.log1p(seq_len),
                1.0 if pos < 30 else 0.0,
                frac(win, "DEKRH"),
                frac(win, "STCYNQ"),
                frac(win, "AILMFWV"),
                frac(win, "P"),
                frac(win, "C"),
                # Nearby motif density
                len(re.findall(r"N[^P][ST]", seq[max(0, pos-30):min(seq_len, pos+33)])) / 63,
            ], dtype=torch.float32)
            all_context.append(ctx)

            all_labels.append(1.0 if pos_1idx in confirmed else 0.0)
            all_pids.append(uid)

    X_key = torch.stack(all_key_embs)     # [N, 9, 1280]
    X_oh = torch.stack(all_key_onehot)    # [N, 9, 21]
    X_ctx = torch.stack(all_context)      # [N, 13]
    y = torch.tensor(all_labels, dtype=torch.float32)

    # Standardize context
    c_mean = X_ctx.mean(0)
    c_std = X_ctx.std(0).clamp(min=1e-6)
    X_ctx = (X_ctx - c_mean) / c_std

    logger.info("Dataset: %d motifs, keys=%s, ctx=%s", len(y), X_key.shape, X_ctx.shape)
    logger.info("Positive: %d (%.1f%%)", int(y.sum()), 100 * y.mean())

    return X_key, X_oh, X_ctx, y, all_pids


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SitePredictorV5(nn.Module):
    """Key-position extraction + cross-attention."""

    def __init__(
        self,
        esm_dim=1280, n_keys=len(KEY_OFFSETS), oh_dim=21, ctx_dim=13,
        proj_dim=64, hidden=256, n_heads=4, dropout=0.3,
    ):
        super().__init__()
        # Per-key projection
        self.key_proj = nn.Sequential(
            nn.Linear(esm_dim, proj_dim), nn.LayerNorm(proj_dim), nn.GELU())
        self.oh_proj = nn.Sequential(
            nn.Linear(oh_dim, proj_dim // 2), nn.GELU())

        # Self-attention over key positions
        combined_dim = proj_dim + proj_dim // 2
        self.self_attn = nn.MultiheadAttention(
            combined_dim, n_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(combined_dim)

        # Pool: learned query for the center (N residue)
        self.center_query = nn.Parameter(torch.randn(1, 1, combined_dim))

        # Classifier
        total = combined_dim + ctx_dim
        self.classifier = nn.Sequential(
            nn.Linear(total, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x_key, x_oh, x_ctx):
        B = x_key.shape[0]
        # Project
        key_p = self.key_proj(x_key)     # [B, K, proj]
        oh_p = self.oh_proj(x_oh)        # [B, K, proj//2]
        combined = torch.cat([key_p, oh_p], dim=-1)  # [B, K, proj+proj//2]

        # Self-attention
        attn_out, _ = self.self_attn(combined, combined, combined)
        combined = self.attn_norm(combined + attn_out)

        # Pool with center query
        query = self.center_query.expand(B, -1, -1)
        pooled, _ = self.self_attn(query, combined, combined)
        pooled = pooled.squeeze(1)  # [B, combined_dim]

        # Classify
        features = torch.cat([pooled, x_ctx], dim=-1)
        return self.classifier(features).squeeze(-1)


# ---------------------------------------------------------------------------
# Ranking loss
# ---------------------------------------------------------------------------

class RankingBCELoss(nn.Module):
    """Combined ranking (pairwise margin) + BCE loss."""

    def __init__(self, margin: float = 1.0, bce_weight: float = 0.3, rank_weight: float = 0.7):
        super().__init__()
        self.margin = margin
        self.bce_weight = bce_weight
        self.rank_weight = rank_weight

    def forward(self, logits, labels, protein_ids=None):
        # BCE component
        bce = F_func.binary_cross_entropy_with_logits(logits, labels)

        # Pairwise ranking component (within mini-batch)
        pos_mask = labels > 0.5
        neg_mask = labels < 0.5
        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            pos_logits = logits[pos_mask]
            neg_logits = logits[neg_mask]
            # Sample pairs
            n_pairs = min(len(pos_logits) * len(neg_logits), 1024)
            pos_idx = torch.randint(0, len(pos_logits), (n_pairs,), device=logits.device)
            neg_idx = torch.randint(0, len(neg_logits), (n_pairs,), device=logits.device)
            rank_loss = F_func.margin_ranking_loss(
                pos_logits[pos_idx], neg_logits[neg_idx],
                torch.ones(n_pairs, device=logits.device),
                margin=self.margin,
            )
        else:
            rank_loss = torch.tensor(0.0, device=logits.device)

        return self.bce_weight * bce + self.rank_weight * rank_loss


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def per_protein_ranking_metrics(scores, labels, pids):
    prot_data = defaultdict(lambda: {"scores": [], "labels": []})
    for i, pid in enumerate(pids):
        prot_data[pid]["scores"].append(scores[i])
        prot_data[pid]["labels"].append(labels[i])

    mrrs, aucs = [], []
    recall_at = {1: [], 3: [], 5: []}

    for pid, d in prot_data.items():
        s, l = np.array(d["scores"]), np.array(d["labels"])
        n_pos = int(l.sum())
        if n_pos == 0 or n_pos == len(l):
            continue
        order = np.argsort(-s)
        sorted_l = l[order]
        for rank, val in enumerate(sorted_l, 1):
            if val == 1:
                mrrs.append(1.0 / rank)
                break
        try:
            aucs.append(roc_auc_score(l, s))
        except ValueError:
            pass
        for k in recall_at:
            recall_at[k].append(sorted_l[:k].sum() / n_pos)

    return {
        "pp_mrr": np.mean(mrrs) if mrrs else 0,
        "pp_auc": np.mean(aucs) if aucs else 0,
        "n_prot": len(mrrs),
        **{f"recall@{k}": np.mean(v) if v else 0 for k, v in recall_at.items()},
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_evaluate(X_key, X_oh, X_ctx, y, pids, device="cpu", epochs=300, lr=3e-4, batch_size=512):
    unique_pids = sorted(set(pids))
    np.random.seed(42)
    np.random.shuffle(unique_pids)
    n_tr, n_va = int(len(unique_pids) * 0.8), int(len(unique_pids) * 0.1)
    train_p = set(unique_pids[:n_tr])
    val_p = set(unique_pids[n_tr:n_tr + n_va])
    test_p = set(unique_pids[n_tr + n_va:])

    tr_i = torch.tensor([i for i, p in enumerate(pids) if p in train_p])
    va_i = torch.tensor([i for i, p in enumerate(pids) if p in val_p])
    te_i = torch.tensor([i for i, p in enumerate(pids) if p in test_p])

    logger.info("Split: train=%d, val=%d, test=%d", len(tr_i), len(va_i), len(te_i))

    Xk_tr = X_key[tr_i].to(device)
    Xo_tr = X_oh[tr_i].to(device)
    Xc_tr = X_ctx[tr_i].to(device)
    y_tr = y[tr_i].to(device)
    Xk_va = X_key[va_i].to(device)
    Xo_va = X_oh[va_i].to(device)
    Xc_va = X_ctx[va_i].to(device)
    y_va = y[va_i].to(device)
    Xk_te = X_key[te_i].to(device)
    Xo_te = X_oh[te_i].to(device)
    Xc_te = X_ctx[te_i].to(device)
    y_te = y[te_i].to(device)

    logger.info("Train pos: %.1f%%, Test pos: %.1f%%", 100 * y_tr.mean(), 100 * y_te.mean())

    model = SitePredictorV5(ctx_dim=X_ctx.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = RankingBCELoss(margin=1.0, bce_weight=0.3, rank_weight=0.7)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model params: %d (%.1f K)", n_params, n_params / 1000)

    warmup = 10
    def lr_fn(e):
        if e < warmup:
            return e / warmup
        return 0.5 * (1 + np.cos(np.pi * (e - warmup) / max(epochs - warmup, 1)))
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    best_metric, best_state = 0.0, None
    patience, no_imp = 50, 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(y_tr.shape[0])
        total_loss = 0.0
        for i in range(0, y_tr.shape[0], batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            logits = model(Xk_tr[idx], Xo_tr[idx], Xc_tr[idx])
            loss = criterion(logits, y_tr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(idx)
        sched.step()

        model.eval()
        with torch.no_grad():
            val_s = model(Xk_va, Xo_va, Xc_va).sigmoid().cpu().numpy()
            val_t = y_va.cpu().numpy()
        try:
            val_ap = average_precision_score(val_t, val_s)
        except ValueError:
            val_ap = 0.0

        if (epoch + 1) % 30 == 0:
            va_pids = [pids[j] for j in va_i.tolist()]
            rk = per_protein_ranking_metrics(val_s, val_t, va_pids)
            logger.info("Epoch %d: loss=%.4f, val_AUC_PR=%.4f, pp_MRR=%.4f, R@3=%.4f",
                        epoch + 1, total_loss / len(tr_i), val_ap, rk["pp_mrr"], rk["recall@3"])

        if val_ap > best_metric:
            best_metric = val_ap
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    model.load_state_dict(best_state)
    model.eval()

    # Threshold search
    with torch.no_grad():
        val_s = model(Xk_va, Xo_va, Xc_va).sigmoid().cpu().numpy()
        val_t = y_va.cpu().numpy()
    bt, bf = 0.5, 0.0
    for t in np.arange(0.2, 0.8, 0.01):
        f = f1_score(val_t, (val_s > t).astype(float), zero_division=0)
        if f > bf:
            bf, bt = f, t

    # Test
    with torch.no_grad():
        ts = model(Xk_te, Xo_te, Xc_te).sigmoid().cpu().numpy()
        tt = y_te.cpu().numpy()
    tp = (ts > bt).astype(float)

    results = {
        "f1": f1_score(tt, tp),
        "precision": precision_score(tt, tp),
        "recall": recall_score(tt, tp),
        "accuracy": accuracy_score(tt, tp),
        "auc_roc": roc_auc_score(tt, ts),
        "auc_pr": average_precision_score(tt, ts),
        "mcc": matthews_corrcoef(tt.astype(int), tp.astype(int)),
        "threshold": bt,
    }

    te_pids = [pids[j] for j in te_i.tolist()]
    rk = per_protein_ranking_metrics(ts, tt, te_pids)
    results.update(rk)

    logger.info("\n=== Test Results (v5 Ranking-focused) ===")
    logger.info("F1=%.4f, Prec=%.4f, Rec=%.4f, Acc=%.4f",
                results["f1"], results["precision"], results["recall"], results["accuracy"])
    logger.info("AUC-ROC=%.4f, AUC-PR=%.4f, MCC=%.4f",
                results["auc_roc"], results["auc_pr"], results["mcc"])
    logger.info("Per-protein MRR=%.4f, AUC=%.4f", rk["pp_mrr"], rk["pp_auc"])
    logger.info("Recall@1=%.4f, Recall@3=%.4f, Recall@5=%.4f",
                rk["recall@1"], rk["recall@3"], rk["recall@5"])
    logger.info("Proteins evaluated: %d", rk["n_prot"])
    logger.info("\n%s", classification_report(tt, tp,
                target_names=["Non-glycosylated", "Glycosylated"], zero_division=0))

    return model, results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    X_key, X_oh, X_ctx, y, pids = build_dataset()
    model, results = train_and_evaluate(X_key, X_oh, X_ctx, y, pids, device=device, epochs=300)

    output_dir = Path("experiments_v2/nlinked_site_v5")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved to %s", output_dir)


if __name__ == "__main__":
    main()
