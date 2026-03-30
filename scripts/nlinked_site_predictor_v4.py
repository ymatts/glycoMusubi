#!/usr/bin/env python3
"""N-linked glycosylation site predictor v4 — PU learning + ranking evaluation.

Key improvements:
1. PU learning: treat unconfirmed sites as "unlabeled", not "negative"
   - Label smoothing on negatives (target=0.15 instead of 0.0)
   - Asymmetric loss (higher weight for positive examples)
2. Per-protein ranking evaluation (within-protein MRR, AUC)
3. Larger model (hidden=512, wider window=25)
4. ESM2 attention weight features (contact prediction proxy)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F_func
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    classification_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

WINDOW = 25  # wider window
AA_LIST = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}


# ---------------------------------------------------------------------------
# PU Loss
# ---------------------------------------------------------------------------

class PUBCELoss(nn.Module):
    """Positive-Unlabeled BCE loss with label smoothing on negatives."""

    def __init__(self, neg_label: float = 0.15, pos_weight: float = 1.5):
        super().__init__()
        self.neg_label = neg_label
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Smooth negative labels
        smooth_targets = targets * 1.0 + (1 - targets) * self.neg_label

        # Asymmetric weighting
        weight = targets * self.pos_weight + (1 - targets) * 1.0

        bce = F_func.binary_cross_entropy_with_logits(logits, smooth_targets, reduction="none")
        return (weight * bce).mean()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def extract_global_features(seq: str, pos: int) -> torch.Tensor:
    L = max(len(seq), 1)
    third_aa = seq[pos + 2] if pos + 2 < len(seq) else "?"
    n_motifs = len(re.findall(r"N[^P][ST]", seq))

    # Extended amino acid features around site
    up10 = seq[max(0, pos - 10):pos]
    down10 = seq[pos + 3:min(len(seq), pos + 13)]

    def aa_prop(s, aas):
        return sum(1 for c in s if c in aas) / max(len(s), 1)

    return torch.tensor([
        1.0 if third_aa == "S" else 0.0,
        1.0 if third_aa == "T" else 0.0,
        pos / L,
        (len(seq) - pos) / L,
        n_motifs / L * 100,
        len(seq) / 1000.0,
        1.0 if pos < 30 else 0.0,
        seq.count("N") / L,
        (seq.count("S") + seq.count("T")) / L,
        seq.count("P") / L,
        np.log1p(len(seq)),
        n_motifs,
        # Upstream/downstream composition
        aa_prop(up10, "DEKRH"),
        aa_prop(up10, "STCYNQ"),
        aa_prop(up10, "AILMFWV"),
        aa_prop(down10, "DEKRH"),
        aa_prop(down10, "STCYNQ"),
        aa_prop(down10, "AILMFWV"),
        # Proline context
        aa_prop(seq[max(0, pos - 3):pos + 6], "P"),
        # Cysteine nearby (disulfide bridges affect accessibility)
        aa_prop(seq[max(0, pos - 15):min(len(seq), pos + 18)], "C"),
    ], dtype=torch.float32)


def build_dataset(data_dir: str = "data_clean", window: int = WINDOW):
    data_dir = Path(data_dir)
    perresidue_dir = data_dir / "esm2_perresidue"

    sites = pd.read_csv(data_dir / "uniprot_sites.tsv", sep="\t")
    n_sites = sites[sites["site_type"] == "N-linked"]
    seqs_df = pd.read_csv(data_dir / "protein_sequences.tsv", sep="\t")
    seq_dict = dict(zip(seqs_df["uniprot_id"], seqs_df["sequence"]))

    proteins = n_sites["uniprot_id"].unique()
    total_len = 2 * window + 3

    esm_windows = []
    onehot_windows = []
    globals_ = []
    labels = []
    pids = []
    positions = []

    for uid in proteins:
        seq = seq_dict.get(f"{uid}-1", seq_dict.get(uid, None))
        if seq is None:
            continue

        pr_path = perresidue_dir / f"{uid}.pt"
        if not pr_path.exists():
            continue
        per_res = torch.load(pr_path, map_location="cpu", weights_only=True).float()

        confirmed = set(n_sites[n_sites["uniprot_id"] == uid]["site_position"].values)

        for m in re.finditer(r"N[^P][ST]", seq):
            pos = m.start()
            pos_1idx = pos + 1

            # ESM2 per-residue window
            L = per_res.shape[0]
            esm_win = torch.zeros(total_len, per_res.shape[1])
            for i in range(total_len):
                seq_idx = pos - window + i
                if 0 <= seq_idx < L:
                    esm_win[i] = per_res[seq_idx]
            esm_windows.append(esm_win)

            # One-hot
            onehot = torch.zeros(total_len, 21)
            for i in range(total_len):
                seq_idx = pos - window + i
                if 0 <= seq_idx < len(seq):
                    aa_idx = AA_TO_IDX.get(seq[seq_idx])
                    if aa_idx is not None:
                        onehot[i, aa_idx] = 1.0
                    else:
                        onehot[i, 20] = 1.0
                else:
                    onehot[i, 20] = 1.0
            onehot_windows.append(onehot)

            glob = extract_global_features(seq, pos)
            globals_.append(glob)

            label = 1.0 if pos_1idx in confirmed else 0.0
            labels.append(label)
            pids.append(uid)
            positions.append(pos_1idx)

    X_esm = torch.stack(esm_windows)
    X_oh = torch.stack(onehot_windows)
    X_glob = torch.stack(globals_)
    y = torch.tensor(labels, dtype=torch.float32)

    g_mean = X_glob.mean(dim=0)
    g_std = X_glob.std(dim=0).clamp(min=1e-6)
    X_glob = (X_glob - g_mean) / g_std

    logger.info("Dataset: %d motifs, ESM2=%s, OH=%s, Global=%s",
                len(y), X_esm.shape, X_oh.shape, X_glob.shape)
    logger.info("Positive: %d (%.1f%%)", int(y.sum()), 100 * y.mean())

    return X_esm, X_oh, X_glob, y, pids, positions


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SitePredictorV4(nn.Module):
    def __init__(
        self,
        esm_dim=1280, oh_dim=21, glob_dim=20,
        window_len=2 * WINDOW + 3,
        esm_proj=128, cnn_filters=128,
        kernels=(3, 5, 9, 15),
        hidden=512, dropout=0.3,
    ):
        super().__init__()
        # ESM2 branch with projection + multi-scale CNN
        self.esm_proj = nn.Sequential(
            nn.Linear(esm_dim, esm_proj), nn.LayerNorm(esm_proj), nn.GELU())
        self.esm_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(esm_proj, cnn_filters, k, padding=k // 2),
                nn.BatchNorm1d(cnn_filters), nn.GELU(),
                nn.Conv1d(cnn_filters, cnn_filters, k, padding=k // 2),
                nn.BatchNorm1d(cnn_filters), nn.GELU(),
            ) for k in kernels
        ])
        esm_out = cnn_filters * len(kernels)

        # One-hot branch
        self.oh_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(oh_dim, cnn_filters // 2, k, padding=k // 2),
                nn.BatchNorm1d(cnn_filters // 2), nn.GELU(),
            ) for k in kernels
        ])
        oh_out = (cnn_filters // 2) * len(kernels)

        # Attention pooling
        self.esm_attn = nn.Linear(esm_out, 1)
        self.oh_attn = nn.Linear(oh_out, 1)

        # Center token extraction (direct feature from the N residue)
        self.center_proj = nn.Linear(esm_dim, esm_proj)

        total = esm_out + oh_out + esm_proj + glob_dim
        self.classifier = nn.Sequential(
            nn.Linear(total, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden // 4, 1),
        )

    def _cnn_pool(self, x, convs, attn):
        x_t = x.permute(0, 2, 1)
        h = torch.cat([c(x_t) for c in convs], dim=1)
        h_t = h.permute(0, 2, 1)
        w = torch.softmax(attn(h_t).squeeze(-1), dim=-1)
        return (h_t * w.unsqueeze(-1)).sum(dim=1)

    def forward(self, x_esm, x_oh, x_glob):
        esm_p = self.esm_proj(x_esm)
        esm_pool = self._cnn_pool(esm_p, self.esm_convs, self.esm_attn)
        oh_pool = self._cnn_pool(x_oh, self.oh_convs, self.oh_attn)

        # Center token (the N residue itself)
        center_idx = x_esm.shape[1] // 2
        center = self.center_proj(x_esm[:, center_idx, :])

        combined = torch.cat([esm_pool, oh_pool, center, x_glob], dim=-1)
        return self.classifier(combined).squeeze(-1)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def per_protein_ranking_metrics(scores, labels, pids):
    """Per-protein MRR and per-protein AUC."""
    from collections import defaultdict
    prot_data = defaultdict(lambda: {"scores": [], "labels": []})
    for i, pid in enumerate(pids):
        prot_data[pid]["scores"].append(scores[i])
        prot_data[pid]["labels"].append(labels[i])

    mrrs = []
    aucs = []
    recall_at_k = {1: [], 3: [], 5: []}

    for pid, data in prot_data.items():
        s = np.array(data["scores"])
        l = np.array(data["labels"])
        n_pos = int(l.sum())
        if n_pos == 0 or n_pos == len(l):
            continue

        # Sort by score descending
        order = np.argsort(-s)
        sorted_labels = l[order]

        # MRR: rank of first positive
        for rank, val in enumerate(sorted_labels, 1):
            if val == 1:
                mrrs.append(1.0 / rank)
                break

        # Per-protein AUC
        try:
            aucs.append(roc_auc_score(l, s))
        except ValueError:
            pass

        # Recall@k
        for k in recall_at_k:
            top_k_labels = sorted_labels[:k]
            recall_at_k[k].append(top_k_labels.sum() / n_pos)

    results = {
        "per_protein_mrr": np.mean(mrrs) if mrrs else 0.0,
        "per_protein_auc": np.mean(aucs) if aucs else 0.0,
        "n_proteins_evaluated": len(mrrs),
    }
    for k, vals in recall_at_k.items():
        results[f"recall@{k}"] = np.mean(vals) if vals else 0.0

    return results


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_evaluate(
    X_esm, X_oh, X_glob, y, pids,
    device="cpu", epochs=300, lr=3e-4, batch_size=256,
):
    unique_pids = sorted(set(pids))
    np.random.seed(42)
    np.random.shuffle(unique_pids)
    n_train = int(len(unique_pids) * 0.8)
    n_val = int(len(unique_pids) * 0.1)
    train_pids = set(unique_pids[:n_train])
    val_pids = set(unique_pids[n_train:n_train + n_val])
    test_pids = set(unique_pids[n_train + n_val:])

    train_idx = torch.tensor([i for i, p in enumerate(pids) if p in train_pids])
    val_idx = torch.tensor([i for i, p in enumerate(pids) if p in val_pids])
    test_idx = torch.tensor([i for i, p in enumerate(pids) if p in test_pids])

    logger.info("Split: train=%d, val=%d, test=%d", len(train_idx), len(val_idx), len(test_idx))

    Xe_tr, Xo_tr, Xg_tr = X_esm[train_idx].to(device), X_oh[train_idx].to(device), X_glob[train_idx].to(device)
    y_tr = y[train_idx].to(device)
    Xe_va, Xo_va, Xg_va = X_esm[val_idx].to(device), X_oh[val_idx].to(device), X_glob[val_idx].to(device)
    y_va = y[val_idx].to(device)
    Xe_te, Xo_te, Xg_te = X_esm[test_idx].to(device), X_oh[test_idx].to(device), X_glob[test_idx].to(device)
    y_te = y[test_idx].to(device)

    logger.info("Train pos: %.1f%%, Val pos: %.1f%%, Test pos: %.1f%%",
                100 * y_tr.mean(), 100 * y_va.mean(), 100 * y_te.mean())

    model = SitePredictorV4(glob_dim=X_glob.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = PUBCELoss(neg_label=0.15, pos_weight=1.5)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model params: %d (%.1f K)", n_params, n_params / 1000)

    warmup = 15
    def lr_lambda(epoch):
        if epoch < warmup:
            return epoch / warmup
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup) / max(epochs - warmup, 1)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_metric, best_state = 0.0, None
    patience, no_improve = 50, 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(y_tr.shape[0])
        total_loss = 0.0
        for i in range(0, y_tr.shape[0], batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            logits = model(Xe_tr[idx], Xo_tr[idx], Xg_tr[idx])
            loss = criterion(logits, y_tr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(idx)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_s = model(Xe_va, Xo_va, Xg_va).sigmoid().cpu().numpy()
            val_true = y_va.cpu().numpy()

        # Use AUC-PR as validation metric (better for PU setting)
        try:
            val_ap = average_precision_score(val_true, val_s)
        except ValueError:
            val_ap = 0.0

        if (epoch + 1) % 30 == 0:
            val_pids_list = [pids[i] for i in val_idx.tolist()]
            ranking = per_protein_ranking_metrics(val_s, val_true, val_pids_list)
            logger.info("Epoch %d: loss=%.4f, val_AUC_PR=%.4f, pp_MRR=%.4f",
                         epoch + 1, total_loss / len(train_idx), val_ap, ranking["per_protein_mrr"])

        if val_ap > best_val_metric:
            best_val_metric = val_ap
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    model.load_state_dict(best_state)
    model.eval()

    # Threshold search
    with torch.no_grad():
        val_s = model(Xe_va, Xo_va, Xg_va).sigmoid().cpu().numpy()
        val_true = y_va.cpu().numpy()
    best_t, best_f = 0.5, 0.0
    for t in np.arange(0.2, 0.8, 0.01):
        f = f1_score(val_true, (val_s > t).astype(float), zero_division=0)
        if f > best_f:
            best_f = f
            best_t = t

    # Test
    with torch.no_grad():
        test_s = model(Xe_te, Xo_te, Xg_te).sigmoid().cpu().numpy()
        test_true = y_te.cpu().numpy()
    test_preds = (test_s > best_t).astype(float)

    # Standard metrics
    results = {
        "f1": f1_score(test_true, test_preds),
        "precision": precision_score(test_true, test_preds),
        "recall": recall_score(test_true, test_preds),
        "accuracy": accuracy_score(test_true, test_preds),
        "auc_roc": roc_auc_score(test_true, test_s),
        "auc_pr": average_precision_score(test_true, test_s),
        "mcc": matthews_corrcoef(test_true.astype(int), test_preds.astype(int)),
        "threshold": best_t,
    }

    # Per-protein ranking metrics
    test_pids_list = [pids[i] for i in test_idx.tolist()]
    ranking = per_protein_ranking_metrics(test_s, test_true, test_pids_list)
    results.update(ranking)

    logger.info("\n=== Test Results (v4 PU-learning) ===")
    logger.info("F1=%.4f, Prec=%.4f, Rec=%.4f, Acc=%.4f",
                results["f1"], results["precision"], results["recall"], results["accuracy"])
    logger.info("AUC-ROC=%.4f, AUC-PR=%.4f, MCC=%.4f",
                results["auc_roc"], results["auc_pr"], results["mcc"])
    logger.info("Per-protein MRR=%.4f, AUC=%.4f",
                results["per_protein_mrr"], results["per_protein_auc"])
    logger.info("Recall@1=%.4f, Recall@3=%.4f, Recall@5=%.4f",
                results.get("recall@1", 0), results.get("recall@3", 0), results.get("recall@5", 0))
    logger.info("Threshold=%.2f", best_t)
    logger.info("\n%s", classification_report(test_true, test_preds,
                target_names=["Non-glycosylated", "Glycosylated"], zero_division=0))

    return model, results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    X_esm, X_oh, X_glob, y, pids, positions = build_dataset()

    model, results = train_and_evaluate(
        X_esm, X_oh, X_glob, y, pids, device=device, epochs=300,
    )

    output_dir = Path("experiments_v2/nlinked_site_v4")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved to %s", output_dir)


if __name__ == "__main__":
    main()
