#!/usr/bin/env python3
"""N-linked glycosylation site predictor v2 — CNN on local sequence.

Key change: Drop mean-pooled ESM2 (identical for all sites in a protein).
Use 1D-CNN on one-hot encoded local sequence window.
Optionally use per-residue ESM2 if available.

Architecture: 1D-CNN → attention pooling → MLP → binary classification
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

AA_LIST = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}
WINDOW = 20  # ±20 residues around the N residue


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def encode_local_window(seq: str, pos: int, window: int = WINDOW) -> torch.Tensor:
    """One-hot encode ±window residues around position `pos`.

    Returns tensor [2*window + 3, 21] (21 = 20 AA + 1 padding).
    """
    L = len(seq)
    total_len = 2 * window + 3  # window_left + motif(3) + window_right
    encoding = torch.zeros(total_len, 21)

    for i in range(total_len):
        seq_pos = pos - window + i
        if 0 <= seq_pos < L:
            aa = seq[seq_pos]
            idx = AA_TO_IDX.get(aa)
            if idx is not None:
                encoding[i, idx] = 1.0
            else:
                encoding[i, 20] = 1.0  # unknown
        else:
            encoding[i, 20] = 1.0  # padding

    return encoding


def extract_global_features(seq: str, pos: int) -> torch.Tensor:
    """Extract position and protein-level features.

    Returns tensor [12].
    """
    L = max(len(seq), 1)

    # Sequon type
    third_aa = seq[pos + 2] if pos + 2 < len(seq) else "?"
    is_nxs = 1.0 if third_aa == "S" else 0.0
    is_nxt = 1.0 if third_aa == "T" else 0.0

    # Position features
    rel_pos = pos / L
    dist_c = (len(seq) - pos) / L

    # Motif density
    n_motifs = len(re.findall(r"N[^P][ST]", seq))

    # Protein composition
    st_frac = (seq.count("S") + seq.count("T")) / L
    n_frac = seq.count("N") / L
    p_frac = seq.count("P") / L

    # Signal peptide proxy
    in_signal = 1.0 if pos < 30 else 0.0

    # Log length
    log_len = np.log1p(len(seq))

    # Confirmed sequon rate proxy (number of sequons / length)
    sequon_density = n_motifs / L * 100

    return torch.tensor([
        is_nxs, is_nxt, rel_pos, dist_c,
        n_motifs, st_frac, n_frac, p_frac,
        in_signal, log_len, sequon_density,
        len(seq) / 1000.0,  # normalized length
    ], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def build_site_dataset(data_dir: str = "data_clean"):
    data_dir = Path(data_dir)

    sites = pd.read_csv(data_dir / "uniprot_sites.tsv", sep="\t")
    n_sites = sites[sites["site_type"] == "N-linked"]
    seqs_df = pd.read_csv(data_dir / "protein_sequences.tsv", sep="\t")
    seq_dict = dict(zip(seqs_df["uniprot_id"], seqs_df["sequence"]))

    proteins = n_sites["uniprot_id"].unique()

    windows = []
    globals_ = []
    labels = []
    pids = []

    for uid in proteins:
        seq = seq_dict.get(f"{uid}-1", seq_dict.get(uid, None))
        if seq is None:
            continue

        confirmed = set(n_sites[n_sites["uniprot_id"] == uid]["site_position"].values)

        for m in re.finditer(r"N[^P][ST]", seq):
            pos = m.start()
            pos_1idx = pos + 1

            win = encode_local_window(seq, pos)
            glob = extract_global_features(seq, pos)
            label = 1.0 if pos_1idx in confirmed else 0.0

            windows.append(win)
            globals_.append(glob)
            labels.append(label)
            pids.append(uid)

    X_win = torch.stack(windows)      # [N, 43, 21]
    X_glob = torch.stack(globals_)    # [N, 12]
    y = torch.tensor(labels, dtype=torch.float32)

    # Standardize global features
    g_mean = X_glob.mean(dim=0)
    g_std = X_glob.std(dim=0).clamp(min=1e-6)
    X_glob = (X_glob - g_mean) / g_std

    logger.info("Dataset: %d motifs, window=%s, global=%s",
                len(y), X_win.shape, X_glob.shape)
    logger.info("Positive: %d (%.1f%%)", int(y.sum()), 100 * y.mean())

    return X_win, X_glob, y, pids


# ---------------------------------------------------------------------------
# Model: 1D-CNN + attention
# ---------------------------------------------------------------------------

class SiteCNN(nn.Module):
    """1D-CNN on local sequence window + global features."""

    def __init__(
        self,
        n_aa: int = 21,
        window_len: int = 2 * WINDOW + 3,
        global_dim: int = 12,
        n_filters: int = 128,
        kernel_sizes: Tuple[int, ...] = (3, 5, 7, 11),
        hidden: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Multi-scale 1D convolutions
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(n_aa, n_filters, k, padding=k // 2),
                nn.BatchNorm1d(n_filters),
                nn.GELU(),
            )
            for k in kernel_sizes
        ])

        conv_out_dim = n_filters * len(kernel_sizes)

        # Attention pooling
        self.attn = nn.Sequential(
            nn.Linear(conv_out_dim, 1),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(conv_out_dim + global_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self, x_win: torch.Tensor, x_glob: torch.Tensor
    ) -> torch.Tensor:
        """
        x_win: [B, L, 21] one-hot encoded window
        x_glob: [B, 12] global features
        """
        # [B, 21, L] for conv1d
        x = x_win.permute(0, 2, 1)

        # Multi-scale convolutions
        conv_outs = [conv(x) for conv in self.convs]  # each [B, n_filters, L]
        h = torch.cat(conv_outs, dim=1)  # [B, n_filters * n_kernels, L]

        # Attention pooling
        h_t = h.permute(0, 2, 1)  # [B, L, conv_out_dim]
        attn_weights = self.attn(h_t).squeeze(-1)  # [B, L]
        attn_weights = torch.softmax(attn_weights, dim=-1)
        pooled = (h_t * attn_weights.unsqueeze(-1)).sum(dim=1)  # [B, conv_out_dim]

        # Combine with global features
        combined = torch.cat([pooled, x_glob], dim=-1)
        return self.classifier(combined).squeeze(-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_evaluate(
    X_win: torch.Tensor,
    X_glob: torch.Tensor,
    y: torch.Tensor,
    pids: List[str],
    device: str = "cpu",
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 512,
) -> Tuple[SiteCNN, Dict]:
    """Train with protein-level split."""
    # Protein-level split
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

    Xw_tr, Xg_tr, y_tr = X_win[train_idx].to(device), X_glob[train_idx].to(device), y[train_idx].to(device)
    Xw_va, Xg_va, y_va = X_win[val_idx].to(device), X_glob[val_idx].to(device), y[val_idx].to(device)
    Xw_te, Xg_te, y_te = X_win[test_idx].to(device), X_glob[test_idx].to(device), y[test_idx].to(device)

    logger.info("Train pos rate: %.1f%%, Test pos rate: %.1f%%",
                100 * y_tr.mean(), 100 * y_te.mean())

    pos_w = (len(y_tr) - y_tr.sum()) / y_tr.sum().clamp(min=1)
    model = SiteCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))

    warmup = 10
    def lr_lambda(epoch):
        if epoch < warmup:
            return epoch / warmup
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup) / max(epochs - warmup, 1)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_f1, best_state = 0.0, None
    patience, no_improve = 40, 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(Xw_tr.shape[0])
        total_loss = 0.0
        for i in range(0, Xw_tr.shape[0], batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            logits = model(Xw_tr[idx], Xg_tr[idx])
            loss = criterion(logits, y_tr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(idx)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_s = model(Xw_va, Xg_va).sigmoid()
            val_preds = (val_s > 0.5).cpu().numpy()
            val_true = y_va.cpu().numpy()
            val_f1 = f1_score(val_true, val_preds, zero_division=0)

        if (epoch + 1) % 20 == 0:
            logger.info("Epoch %d: loss=%.4f, val_F1=%.4f", epoch + 1, total_loss / len(train_idx), val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
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
        val_s = model(Xw_va, Xg_va).sigmoid().cpu().numpy()
        val_t = y_va.cpu().numpy()
    best_t, best_f = 0.5, 0.0
    for t in np.arange(0.2, 0.8, 0.01):
        f = f1_score(val_t, (val_s > t).astype(float), zero_division=0)
        if f > best_f:
            best_f = f
            best_t = t

    # Test
    with torch.no_grad():
        test_s = model(Xw_te, Xg_te).sigmoid().cpu().numpy()
        test_true = y_te.cpu().numpy()
    test_preds = (test_s > best_t).astype(float)

    results = {
        "f1": f1_score(test_true, test_preds),
        "precision": precision_score(test_true, test_preds),
        "recall": recall_score(test_true, test_preds),
        "accuracy": accuracy_score(test_true, test_preds),
        "auc_roc": roc_auc_score(test_true, test_s),
        "auc_pr": average_precision_score(test_true, test_s),
        "mcc": matthews_corrcoef(test_true.astype(int), test_preds.astype(int)),
        "threshold": best_t,
        "n_test": len(test_true),
        "n_positive": int(test_true.sum()),
    }

    logger.info("\n=== Test Results (CNN) ===")
    logger.info("F1=%.4f, Prec=%.4f, Rec=%.4f, Acc=%.4f",
                results["f1"], results["precision"], results["recall"], results["accuracy"])
    logger.info("AUC-ROC=%.4f, AUC-PR=%.4f, MCC=%.4f",
                results["auc_roc"], results["auc_pr"], results["mcc"])
    logger.info("Threshold=%.2f", best_t)
    logger.info("\n%s", classification_report(test_true, test_preds,
                target_names=["Non-glycosylated", "Glycosylated"], zero_division=0))

    return model, results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    X_win, X_glob, y, pids = build_site_dataset()

    model, results = train_and_evaluate(X_win, X_glob, y, pids, device=device, epochs=300)

    # Save
    output_dir = Path("experiments_v2/nlinked_site_cnn")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved to %s", output_dir)


if __name__ == "__main__":
    main()
