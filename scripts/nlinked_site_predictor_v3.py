#!/usr/bin/env python3
"""N-linked glycosylation site predictor v3 — per-residue ESM2 + CNN.

Uses per-residue ESM2 embeddings around each N-X-S/T motif.
This provides 3D structure-aware context that captures surface accessibility
and local folding environment.

Architecture:
- Per-residue ESM2 window: [2*W+3, 1280] → 1D-CNN → attention pool
- Sequence context features: [12]
- Combined → MLP → binary prediction
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
import torch.nn.functional as F_func
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

WINDOW = 15  # ±15 residues around N
AA_LIST = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def extract_global_features(seq: str, pos: int) -> torch.Tensor:
    L = max(len(seq), 1)
    third_aa = seq[pos + 2] if pos + 2 < len(seq) else "?"
    n_motifs = len(re.findall(r"N[^P][ST]", seq))
    return torch.tensor([
        1.0 if third_aa == "S" else 0.0,  # sequon type
        1.0 if third_aa == "T" else 0.0,
        pos / L,  # relative position
        (len(seq) - pos) / L,  # distance from C-terminus
        n_motifs / L * 100,  # sequon density
        len(seq) / 1000.0,  # normalized length
        1.0 if pos < 30 else 0.0,  # signal peptide proxy
        seq.count("N") / L,
        (seq.count("S") + seq.count("T")) / L,
        seq.count("P") / L,
        np.log1p(len(seq)),
        n_motifs,
    ], dtype=torch.float32)


def build_dataset(
    data_dir: str = "data_clean",
    window: int = WINDOW,
    use_per_residue: bool = True,
):
    """Build dataset with per-residue ESM2 embeddings."""
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

    n_skip = 0
    n_found = 0

    for uid in proteins:
        seq = seq_dict.get(f"{uid}-1", seq_dict.get(uid, None))
        if seq is None:
            n_skip += 1
            continue

        # Load per-residue embedding
        per_res = None
        if use_per_residue:
            pr_path = perresidue_dir / f"{uid}.pt"
            if pr_path.exists():
                per_res = torch.load(pr_path, map_location="cpu", weights_only=True).float()
                n_found += 1
            else:
                n_skip += 1
                continue

        confirmed = set(n_sites[n_sites["uniprot_id"] == uid]["site_position"].values)

        for m in re.finditer(r"N[^P][ST]", seq):
            pos = m.start()
            pos_1idx = pos + 1

            # ESM2 per-residue window
            if per_res is not None:
                L = per_res.shape[0]
                esm_win = torch.zeros(total_len, per_res.shape[1])
                for i in range(total_len):
                    seq_idx = pos - window + i
                    if 0 <= seq_idx < L:
                        esm_win[i] = per_res[seq_idx]
                esm_windows.append(esm_win)

            # One-hot encoding
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

    logger.info("Per-residue ESM2: found=%d, skipped=%d", n_found, n_skip)

    X_esm = torch.stack(esm_windows) if esm_windows else None  # [N, W, 1280]
    X_onehot = torch.stack(onehot_windows)  # [N, W, 21]
    X_glob = torch.stack(globals_)          # [N, 12]
    y = torch.tensor(labels, dtype=torch.float32)

    # Standardize global features
    g_mean = X_glob.mean(dim=0)
    g_std = X_glob.std(dim=0).clamp(min=1e-6)
    X_glob = (X_glob - g_mean) / g_std

    logger.info("Dataset: %d motifs", len(y))
    if X_esm is not None:
        logger.info("  ESM2 window: %s", X_esm.shape)
    logger.info("  One-hot window: %s", X_onehot.shape)
    logger.info("  Global: %s", X_glob.shape)
    logger.info("  Positive: %d (%.1f%%)", int(y.sum()), 100 * y.mean())

    return X_esm, X_onehot, X_glob, y, pids


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SitePredictorV3(nn.Module):
    """Per-residue ESM2 + one-hot CNN + global features → site prediction."""

    def __init__(
        self,
        esm_dim: int = 1280,
        onehot_dim: int = 21,
        global_dim: int = 12,
        window_len: int = 2 * WINDOW + 3,
        esm_proj_dim: int = 128,
        cnn_filters: int = 64,
        kernel_sizes: Tuple[int, ...] = (3, 5, 7),
        hidden: int = 256,
        dropout: float = 0.3,
        use_esm: bool = True,
    ):
        super().__init__()
        self.use_esm = use_esm

        # ESM2 per-residue branch
        if use_esm:
            self.esm_proj = nn.Sequential(
                nn.Linear(esm_dim, esm_proj_dim),
                nn.LayerNorm(esm_proj_dim),
                nn.GELU(),
            )
            self.esm_convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(esm_proj_dim, cnn_filters, k, padding=k // 2),
                    nn.BatchNorm1d(cnn_filters),
                    nn.GELU(),
                )
                for k in kernel_sizes
            ])
            self.esm_attn = nn.Linear(cnn_filters * len(kernel_sizes), 1)
            esm_out_dim = cnn_filters * len(kernel_sizes)
        else:
            esm_out_dim = 0

        # One-hot CNN branch
        self.oh_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(onehot_dim, cnn_filters, k, padding=k // 2),
                nn.BatchNorm1d(cnn_filters),
                nn.GELU(),
            )
            for k in kernel_sizes
        ])
        self.oh_attn = nn.Linear(cnn_filters * len(kernel_sizes), 1)
        oh_out_dim = cnn_filters * len(kernel_sizes)

        # Classifier
        total_dim = esm_out_dim + oh_out_dim + global_dim
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def _cnn_attn_pool(self, x, convs, attn):
        """Apply multi-scale CNN + attention pooling."""
        # x: [B, L, D] → [B, D, L]
        x = x.permute(0, 2, 1)
        conv_outs = [conv(x) for conv in convs]  # [B, filters, L]
        h = torch.cat(conv_outs, dim=1)  # [B, total_filters, L]
        h_t = h.permute(0, 2, 1)  # [B, L, total_filters]
        w = torch.softmax(attn(h_t).squeeze(-1), dim=-1)  # [B, L]
        return (h_t * w.unsqueeze(-1)).sum(dim=1)  # [B, total_filters]

    def forward(
        self,
        x_esm: Optional[torch.Tensor],
        x_onehot: torch.Tensor,
        x_glob: torch.Tensor,
    ) -> torch.Tensor:
        parts = []

        if self.use_esm and x_esm is not None:
            esm_proj = self.esm_proj(x_esm)  # [B, L, proj_dim]
            esm_pooled = self._cnn_attn_pool(esm_proj, self.esm_convs, self.esm_attn)
            parts.append(esm_pooled)

        oh_pooled = self._cnn_attn_pool(x_onehot, self.oh_convs, self.oh_attn)
        parts.append(oh_pooled)
        parts.append(x_glob)

        combined = torch.cat(parts, dim=-1)
        return self.classifier(combined).squeeze(-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_evaluate(
    X_esm, X_onehot, X_glob, y, pids,
    device="cpu", epochs=200, lr=5e-4, batch_size=256,
):
    use_esm = X_esm is not None

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

    def to_dev(t, idx):
        return t[idx].to(device) if t is not None else None

    Xe_tr = to_dev(X_esm, train_idx)
    Xo_tr, Xg_tr, y_tr = X_onehot[train_idx].to(device), X_glob[train_idx].to(device), y[train_idx].to(device)
    Xe_va = to_dev(X_esm, val_idx)
    Xo_va, Xg_va, y_va = X_onehot[val_idx].to(device), X_glob[val_idx].to(device), y[val_idx].to(device)
    Xe_te = to_dev(X_esm, test_idx)
    Xo_te, Xg_te, y_te = X_onehot[test_idx].to(device), X_glob[test_idx].to(device), y[test_idx].to(device)

    logger.info("Train pos: %.1f%%, Test pos: %.1f%%", 100 * y_tr.mean(), 100 * y_te.mean())

    pos_w = (len(y_tr) - y_tr.sum()) / y_tr.sum().clamp(min=1)
    model = SitePredictorV3(use_esm=use_esm).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %d (%.1f K)", n_params, n_params / 1000)

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
        perm = torch.randperm(y_tr.shape[0])
        total_loss = 0.0
        for i in range(0, y_tr.shape[0], batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            esm_batch = Xe_tr[idx] if Xe_tr is not None else None
            logits = model(esm_batch, Xo_tr[idx], Xg_tr[idx])
            loss = criterion(logits, y_tr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(idx)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(Xe_va, Xo_va, Xg_va).sigmoid()
            val_preds = (val_logits > 0.5).cpu().numpy()
            val_true = y_va.cpu().numpy()
            val_f1 = f1_score(val_true, val_preds, zero_division=0)

        if (epoch + 1) % 20 == 0:
            logger.info("Epoch %d: loss=%.4f, val_F1=%.4f",
                         epoch + 1, total_loss / len(train_idx), val_f1)

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
        val_s = model(Xe_va, Xo_va, Xg_va).sigmoid().cpu().numpy()
        val_t = y_va.cpu().numpy()
    best_t, best_f = 0.5, 0.0
    for t in np.arange(0.2, 0.8, 0.01):
        f = f1_score(val_t, (val_s > t).astype(float), zero_division=0)
        if f > best_f:
            best_f = f
            best_t = t

    with torch.no_grad():
        test_s = model(Xe_te, Xo_te, Xg_te).sigmoid().cpu().numpy()
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
        "use_esm_perresidue": use_esm,
    }

    logger.info("\n=== Test Results (v3%s) ===", " + ESM2 per-residue" if use_esm else " one-hot only")
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

    X_esm, X_onehot, X_glob, y, pids = build_dataset()

    if X_esm is not None:
        logger.info("\n--- Running with per-residue ESM2 ---")
        model, results = train_and_evaluate(
            X_esm, X_onehot, X_glob, y, pids, device=device, epochs=300,
        )
    else:
        logger.warning("No per-residue ESM2 found, using one-hot only")
        model, results = train_and_evaluate(
            None, X_onehot, X_glob, y, pids, device=device, epochs=300,
        )

    output_dir = Path("experiments_v2/nlinked_site_v3")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved to %s", output_dir)


if __name__ == "__main__":
    main()
