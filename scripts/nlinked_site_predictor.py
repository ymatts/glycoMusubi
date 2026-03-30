#!/usr/bin/env python3
"""N-linked glycosylation site predictor.

Predicts whether an N-X-S/T sequon is actually glycosylated.
Uses ESM2 mean-pool embedding + local sequence context features.

Dataset: 9,383 N-X-S/T motifs across 1,415 proteins
  - 3,821 glycosylated (40.7%)
  - 5,562 non-glycosylated (59.3%)
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
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Amino acid properties
AA_HYDROPHOBICITY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
AA_LIST = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}


def extract_site_features(
    seq: str, pos: int, window: int = 15
) -> torch.Tensor:
    """Extract features around an N-X-S/T motif at position `pos` (0-indexed).

    Returns tensor of shape [feature_dim]:
    - Local AA composition in ±window [20]
    - Local AA property means (hydrophobicity, molecular weight proxy) [4]
    - Sequon type: N-X-S=1 vs N-X-T=0 [1]
    - X residue identity (the residue between N and S/T) [20]
    - Position features [5]
    - Extended context features [12]
    """
    L = len(seq)
    start = max(0, pos - window)
    end = min(L, pos + 3 + window)  # motif is 3 residues: N-X-S/T
    local_seq = seq[start:end]

    # 1. Local AA composition [20]
    local_len = max(len(local_seq), 1)
    aa_comp = torch.zeros(20)
    for aa in local_seq:
        idx = AA_TO_IDX.get(aa)
        if idx is not None:
            aa_comp[idx] += 1
    aa_comp /= local_len

    # 2. Local AA properties [4]
    hydro_vals = [AA_HYDROPHOBICITY.get(aa, 0) for aa in local_seq]
    mean_hydro = np.mean(hydro_vals) if hydro_vals else 0
    std_hydro = np.std(hydro_vals) if hydro_vals else 0
    # N-terminal side vs C-terminal side hydrophobicity
    mid = pos - start
    n_side = [AA_HYDROPHOBICITY.get(aa, 0) for aa in local_seq[:mid]]
    c_side = [AA_HYDROPHOBICITY.get(aa, 0) for aa in local_seq[mid + 3:]]
    n_hydro = np.mean(n_side) if n_side else 0
    c_hydro = np.mean(c_side) if c_side else 0
    props = torch.tensor([mean_hydro, std_hydro, n_hydro, c_hydro], dtype=torch.float32)

    # 3. Sequon type [1]: N-X-S (1) vs N-X-T (0)
    third_aa = seq[pos + 2] if pos + 2 < L else "?"
    sequon_type = torch.tensor([1.0 if third_aa == "S" else 0.0])

    # 4. X residue (between N and S/T) [20]
    x_aa = seq[pos + 1] if pos + 1 < L else "?"
    x_onehot = torch.zeros(20)
    x_idx = AA_TO_IDX.get(x_aa)
    if x_idx is not None:
        x_onehot[x_idx] = 1.0

    # 5. Position features [5]
    rel_pos = pos / max(L, 1)
    dist_n = pos / max(L, 1)
    dist_c = (L - pos) / max(L, 1)
    log_len = np.log1p(L)

    # Number of N-X-S/T motifs in the protein
    n_motifs = len(re.findall(r"N[^P][ST]", seq))
    motif_density = n_motifs / max(L, 1) * 100

    pos_feat = torch.tensor(
        [rel_pos, dist_n, dist_c, log_len, motif_density], dtype=torch.float32
    )

    # 6. Extended context [12]: AA composition in upstream/downstream 5-mers
    up5 = seq[max(0, pos - 5) : pos]
    down5 = seq[pos + 3 : min(L, pos + 8)]

    def charged_frac(s):
        return sum(1 for c in s if c in "DEKRH") / max(len(s), 1)

    def polar_frac(s):
        return sum(1 for c in s if c in "STCYNQ") / max(len(s), 1)

    def aromatic_frac(s):
        return sum(1 for c in s if c in "FWY") / max(len(s), 1)

    ext_feat = torch.tensor([
        charged_frac(up5), polar_frac(up5), aromatic_frac(up5),
        charged_frac(down5), polar_frac(down5), aromatic_frac(down5),
        # Proline near the site (blocks glycosylation)
        1.0 if "P" in seq[max(0, pos - 2) : pos] else 0.0,
        1.0 if "P" in seq[pos + 3 : min(L, pos + 5)] else 0.0,
        # Adjacent sequons
        len(re.findall(r"N[^P][ST]", seq[max(0, pos - 20) : pos])) / max(1, min(pos, 20)),
        len(re.findall(r"N[^P][ST]", seq[pos + 3 : min(L, pos + 23)])) / max(1, min(L - pos - 3, 20)),
        # Signal peptide proxy (first 30 residues)
        1.0 if pos < 30 else 0.0,
        # Transmembrane proxy (long hydrophobic stretch nearby)
        1.0 if mean_hydro > 1.5 else 0.0,
    ], dtype=torch.float32)

    return torch.cat([aa_comp, props, sequon_type, x_onehot, pos_feat, ext_feat])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def build_site_dataset(data_dir: str = "data_clean"):
    """Build N-linked glycosylation site prediction dataset."""
    data_dir = Path(data_dir)

    sites = pd.read_csv(data_dir / "uniprot_sites.tsv", sep="\t")
    n_sites = sites[sites["site_type"] == "N-linked"]
    seqs_df = pd.read_csv(data_dir / "protein_sequences.tsv", sep="\t")
    seq_dict = dict(zip(seqs_df["uniprot_id"], seqs_df["sequence"]))

    with open(data_dir / "esm2_cache" / "id_to_idx.json") as f:
        id_to_idx = json.load(f)
    esm2_dir = data_dir / "esm2_cache"

    proteins = n_sites["uniprot_id"].unique()
    features = []
    labels = []
    metadata = []  # (protein_id, position, seq)

    for uid in proteins:
        # Get sequence
        seq = seq_dict.get(f"{uid}-1", seq_dict.get(uid, None))
        if seq is None:
            continue

        # Get ESM2 embedding
        esm_idx = None
        for key in [f"{uid}-1", uid]:
            if key in id_to_idx:
                esm_idx = id_to_idx[key]
                break
        if esm_idx is None:
            continue
        pt_path = esm2_dir / f"{esm_idx}.pt"
        if not pt_path.exists():
            continue
        esm_emb = torch.load(pt_path, map_location="cpu", weights_only=True)
        if esm_emb.dim() == 2:
            esm_emb = esm_emb.mean(dim=0)

        # Confirmed glycosylated positions (1-indexed)
        confirmed = set(n_sites[n_sites["uniprot_id"] == uid]["site_position"].values)

        # Find all N-X-S/T motifs
        for m in re.finditer(r"N[^P][ST]", seq):
            pos = m.start()  # 0-indexed
            pos_1idx = pos + 1  # 1-indexed for matching

            site_feat = extract_site_features(seq, pos)
            feat = torch.cat([esm_emb, site_feat])
            features.append(feat)

            label = 1.0 if pos_1idx in confirmed else 0.0
            labels.append(label)
            metadata.append((uid, pos_1idx, seq[pos:pos+3]))

    X = torch.stack(features)
    y = torch.tensor(labels, dtype=torch.float32)

    # Standardize features
    mean = X.mean(dim=0)
    std = X.std(dim=0).clamp(min=1e-6)
    X = (X - mean) / std

    logger.info("Site dataset: %d motifs, %d features", X.shape[0], X.shape[1])
    logger.info("Positive (glycosylated): %d (%.1f%%)", int(y.sum()), 100 * y.mean())

    return X, y, metadata, mean, std


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        return x + self.net(x)


class SitePredictor(nn.Module):
    def __init__(self, input_dim, hidden=512, n_blocks=3, dropout=0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        self.blocks = nn.Sequential(*[ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.head(self.blocks(self.proj(x))).squeeze(-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_evaluate(
    X: torch.Tensor,
    y: torch.Tensor,
    device: str = "cpu",
    epochs: int = 200,
    lr: float = 3e-4,
    batch_size: int = 512,
    protein_split: bool = True,
    metadata: Optional[List] = None,
) -> Tuple[SitePredictor, Dict]:
    """Train with protein-level split (no data leakage)."""
    if protein_split and metadata:
        # Split by protein ID
        pids = [m[0] for m in metadata]
        unique_pids = sorted(set(pids))
        np.random.seed(42)
        np.random.shuffle(unique_pids)

        n_train = int(len(unique_pids) * 0.8)
        n_val = int(len(unique_pids) * 0.1)
        train_pids = set(unique_pids[:n_train])
        val_pids = set(unique_pids[n_train:n_train + n_val])
        test_pids = set(unique_pids[n_train + n_val:])

        train_idx = [i for i, pid in enumerate(pids) if pid in train_pids]
        val_idx = [i for i, pid in enumerate(pids) if pid in val_pids]
        test_idx = [i for i, pid in enumerate(pids) if pid in test_pids]
    else:
        n = X.shape[0]
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
        n_train, n_val = int(n * 0.8), int(n * 0.1)
        train_idx = perm[:n_train]
        val_idx = perm[n_train:n_train + n_val]
        test_idx = perm[n_train + n_val:]

    train_idx = torch.tensor(train_idx, dtype=torch.long)
    val_idx = torch.tensor(val_idx, dtype=torch.long)
    test_idx = torch.tensor(test_idx, dtype=torch.long)

    X_tr, y_tr = X[train_idx].to(device), y[train_idx].to(device)
    X_va, y_va = X[val_idx].to(device), y[val_idx].to(device)
    X_te, y_te = X[test_idx].to(device), y[test_idx].to(device)

    logger.info("Split: train=%d (%d proteins), val=%d, test=%d",
                len(train_idx), len(set(metadata[i][0] for i in train_idx.tolist())) if metadata else 0,
                len(val_idx), len(test_idx))
    logger.info("Train pos rate: %.1f%%, Val: %.1f%%, Test: %.1f%%",
                100 * y_tr.mean(), 100 * y_va.mean(), 100 * y_te.mean())

    pos_weight = torch.tensor([(len(y_tr) - y_tr.sum()) / y_tr.sum().clamp(min=1)], device=device)
    model = SitePredictor(X.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    warmup = 10
    def lr_lambda(epoch):
        if epoch < warmup:
            return epoch / warmup
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup) / max(epochs - warmup, 1)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_f1, best_state = 0.0, None
    patience, no_improve = 30, 0

    for epoch in range(epochs):
        model.train()
        perm_tr = torch.randperm(X_tr.shape[0])
        total_loss = 0.0
        for i in range(0, X_tr.shape[0], batch_size):
            idx = perm_tr[i:i + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(X_tr[idx]), y_tr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(idx)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_scores = model(X_va).sigmoid()
            val_preds = (val_scores > 0.5).cpu().numpy()
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

    # Threshold optimization
    with torch.no_grad():
        val_s = model(X_va).sigmoid().cpu().numpy()
        val_t = y_va.cpu().numpy()
    best_t, best_f = 0.5, 0.0
    for t in np.arange(0.2, 0.8, 0.01):
        f = f1_score(val_t, (val_s > t).astype(float), zero_division=0)
        if f > best_f:
            best_f = f
            best_t = t

    # Test evaluation
    with torch.no_grad():
        test_scores = model(X_te).sigmoid().cpu().numpy()
        test_true = y_te.cpu().numpy()
    test_preds = (test_scores > best_t).astype(float)

    results = {
        "f1": f1_score(test_true, test_preds),
        "precision": precision_score(test_true, test_preds),
        "recall": recall_score(test_true, test_preds),
        "accuracy": accuracy_score(test_true, test_preds),
        "auc_roc": roc_auc_score(test_true, test_scores),
        "auc_pr": average_precision_score(test_true, test_scores),
        "mcc": matthews_corrcoef(test_true.astype(int), test_preds.astype(int)),
        "threshold": best_t,
        "n_test": len(test_true),
        "n_positive": int(test_true.sum()),
    }

    logger.info("\n=== Test Results ===")
    logger.info("F1=%.4f, Prec=%.4f, Rec=%.4f, Acc=%.4f",
                results["f1"], results["precision"], results["recall"], results["accuracy"])
    logger.info("AUC-ROC=%.4f, AUC-PR=%.4f, MCC=%.4f",
                results["auc_roc"], results["auc_pr"], results["mcc"])
    logger.info("Threshold=%.2f", best_t)
    logger.info("\n%s", classification_report(test_true, test_preds,
                target_names=["Non-glycosylated", "Glycosylated"], zero_division=0))

    return model, results


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def run_baselines(X: torch.Tensor, y: torch.Tensor, metadata: List):
    """Simple baselines for comparison."""
    # Baseline 1: Always predict positive
    always_pos = np.ones(len(y))
    f1_pos = f1_score(y.numpy(), always_pos)
    logger.info("Baseline (always positive): F1=%.4f", f1_pos)

    # Baseline 2: Random (class-balanced)
    np.random.seed(42)
    rate = y.mean().item()
    random_preds = (np.random.random(len(y)) < rate).astype(float)
    f1_rand = f1_score(y.numpy(), random_preds)
    logger.info("Baseline (random balanced): F1=%.4f", f1_rand)

    # Baseline 3: Sequon type (N-X-S vs N-X-T)
    # Literature: N-X-T has higher glycosylation rate than N-X-S
    sequon_preds = np.array([1.0 if m[2][2] == "T" else 0.0 for m in metadata])
    f1_sequon = f1_score(y.numpy(), sequon_preds)
    logger.info("Baseline (N-X-T only): F1=%.4f", f1_sequon)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    X, y, metadata, mean, std = build_site_dataset()

    # Baselines
    run_baselines(X, y, metadata)

    # Train
    model, results = train_and_evaluate(
        X, y, device=device, epochs=300, metadata=metadata, protein_split=True,
    )

    # Save
    output_dir = Path("experiments_v2/nlinked_site_predictor")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_mean": mean,
        "feature_std": std,
        "results": results,
    }, output_dir / "model.pt")

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Saved to %s", output_dir)


if __name__ == "__main__":
    main()
