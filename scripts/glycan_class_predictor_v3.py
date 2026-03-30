#!/usr/bin/env python3
"""Glycan function class predictor v3 — focused on biologically meaningful classes.

Key insight from data analysis:
- Glycosphingolipid co-occurs with N-linked 99.9% of the time → noise
- Only N-linked and O-linked are biologically meaningful for protein-level prediction
- 48.7% of proteins are exclusive to one class

Three evaluation modes:
1. Binary: N-linked yes/no, O-linked yes/no (separate classifiers)
2. Exclusive 3-class: N-linked vs O-linked vs Other (clean subset)
3. Soft regression: Predict class fractions

Usage:
    python scripts/glycan_class_predictor_v3.py
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score as sk_f1,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading (shared)
# ---------------------------------------------------------------------------

def load_all_data(data_dir: str = "data_clean"):
    data_dir = Path(data_dir)

    edges = pd.read_csv(data_dir / "edges_glycan_protein.tsv", sep="\t")
    edges = edges[["glycan_id", "protein_id"]].drop_duplicates()
    func = pd.read_csv(data_dir / "glycan_function_labels.tsv", sep="\t")
    func = func[["glycan_id", "function_term"]].drop_duplicates()
    merged = edges.merge(func, on="glycan_id", how="inner")

    sites = pd.read_csv(data_dir / "uniprot_sites.tsv", sep="\t")
    site_counts: Dict[str, Dict[str, int]] = {}
    for uid, group in sites.groupby("uniprot_id"):
        site_counts[uid] = group["site_type"].value_counts().to_dict()

    esm2_dir = data_dir / "esm2_cache"
    with open(esm2_dir / "id_to_idx.json") as f:
        id_to_idx = json.load(f)

    seqs = pd.read_csv(data_dir / "protein_sequences.tsv", sep="\t")
    sequences = dict(zip(seqs["uniprot_id"], seqs["sequence"]))

    return merged, site_counts, id_to_idx, esm2_dir, sequences


def _resolve_esm2_idx(pid: str, id_to_idx: Dict[str, int]) -> Optional[int]:
    if pid in id_to_idx:
        return id_to_idx[pid]
    base = pid.split("-")[0]
    if f"{base}-1" in id_to_idx:
        return id_to_idx[f"{base}-1"]
    if base in id_to_idx:
        return id_to_idx[base]
    return None


def extract_sequence_features(seq: str) -> torch.Tensor:
    L = max(len(seq), 1)
    seq_upper = seq.upper() if seq else ""
    nxst = len(re.findall(r"N[^P][ST]", seq_upper))
    nxs = len(re.findall(r"N[^P]S", seq_upper))
    nxt = len(re.findall(r"N[^P]T", seq_upper))
    aa_counts = {aa: seq_upper.count(aa) for aa in "NSTPWAILMFVDEKRHYCGQ"}
    features = [
        nxst / L,
        nxs / L,
        nxt / L,
        aa_counts.get("N", 0) / L,
        aa_counts.get("S", 0) / L,
        aa_counts.get("T", 0) / L,
        (aa_counts.get("S", 0) + aa_counts.get("T", 0)) / L,
        aa_counts.get("W", 0) / L,
        aa_counts.get("Y", 0) / L,
        aa_counts.get("C", 0) / L,
        aa_counts.get("G", 0) / L,
        aa_counts.get("Q", 0) / L,
        np.log1p(L),
        sum(aa_counts.get(aa, 0) for aa in "AILMFWV") / L,
        sum(aa_counts.get(aa, 0) for aa in "DEKRH") / L,
        aa_counts.get("P", 0) / L,
    ]
    return torch.tensor(features, dtype=torch.float32)


def load_features(
    proteins: List[str],
    site_counts: Dict[str, Dict[str, int]],
    id_to_idx: Dict[str, int],
    esm2_dir: Path,
    sequences: Dict[str, str],
) -> Tuple[torch.Tensor, List[str]]:
    """Load ESM2 + site + sequence features for given protein list."""
    features = []
    valid = []
    for pid in proteins:
        esm_idx = _resolve_esm2_idx(pid, id_to_idx)
        if esm_idx is None:
            continue
        pt_path = esm2_dir / f"{esm_idx}.pt"
        if not pt_path.exists():
            continue
        emb = torch.load(pt_path, map_location="cpu", weights_only=True)
        if emb.dim() == 2:
            emb = emb.mean(dim=0)

        base_id = pid.split("-")[0]
        sc = site_counts.get(base_id, {})
        site_feat = torch.tensor([
            sc.get("N-linked", 0),
            sc.get("O-linked", 0),
            sc.get("C-mannosylation", 0),
            sc.get("glycosylation", 0),
        ], dtype=torch.float32)

        seq = sequences.get(pid, sequences.get(f"{base_id}-1", ""))
        seq_feat = extract_sequence_features(seq)

        features.append(torch.cat([emb, site_feat, seq_feat]))
        valid.append(pid)

    X = torch.stack(features)
    # Standardize
    mean = X.mean(dim=0)
    std = X.std(dim=0).clamp(min=1e-6)
    X = (X - mean) / std
    return X, valid


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        return x + self.net(x)


class BinaryClassifier(nn.Module):
    def __init__(self, input_dim, hidden=256, n_blocks=2, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        self.blocks = nn.Sequential(*[ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.head(self.blocks(self.proj(x))).squeeze(-1)


class MultiClassifier(nn.Module):
    def __init__(self, input_dim, n_classes, hidden=256, n_blocks=2, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        self.blocks = nn.Sequential(*[ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):
        return self.head(self.blocks(self.proj(x)))


# ---------------------------------------------------------------------------
# Experiment 1: Binary classifiers (N-linked, O-linked)
# ---------------------------------------------------------------------------

def run_binary_classifier(
    X: torch.Tensor,
    y: torch.Tensor,
    class_name: str,
    device: str = "cpu",
    epochs: int = 200,
) -> Dict:
    """Train binary classifier for one class."""
    n = X.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    n_train, n_val = int(n * 0.8), int(n * 0.1)
    tr, va, te = perm[:n_train], perm[n_train:n_train+n_val], perm[n_train+n_val:]

    X_tr, y_tr = X[tr].to(device), y[tr].to(device)
    X_va, y_va = X[va].to(device), y[va].to(device)
    X_te, y_te = X[te].to(device), y[te].to(device)

    pos_count = y_tr.sum()
    neg_count = len(y_tr) - pos_count
    pos_weight = torch.tensor([neg_count / pos_count.clamp(min=1)], device=device)

    model = BinaryClassifier(X.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_f1, best_state = 0.0, None
    patience, no_improve = 30, 0

    for epoch in range(epochs):
        model.train()
        perm_tr = torch.randperm(X_tr.shape[0])
        total_loss = 0.0
        for i in range(0, X_tr.shape[0], 256):
            idx = perm_tr[i:i+256]
            optimizer.zero_grad()
            logits = model(X_tr[idx])
            loss = criterion(logits, y_tr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            val_preds = (model(X_va).sigmoid() > 0.5).cpu().numpy()
            val_true = y_va.cpu().numpy()
            val_f1 = sk_f1(val_true, val_preds, zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()

    # Threshold search
    with torch.no_grad():
        val_scores = model(X_va).sigmoid().cpu().numpy()
        val_true = y_va.cpu().numpy()
    best_t, best_f = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.02):
        f = sk_f1(val_true, (val_scores > t).astype(float), zero_division=0)
        if f > best_f:
            best_f = f
            best_t = t

    with torch.no_grad():
        test_scores = model(X_te).sigmoid().cpu().numpy()
        test_true = y_te.cpu().numpy()
    test_preds = (test_scores > best_t).astype(float)

    f1 = sk_f1(test_true, test_preds, zero_division=0)
    prec = precision_score(test_true, test_preds, zero_division=0)
    rec = recall_score(test_true, test_preds, zero_division=0)
    acc = accuracy_score(test_true, test_preds)
    try:
        auc = roc_auc_score(test_true, test_scores)
    except ValueError:
        auc = 0.0

    logger.info(
        "[%s] F1=%.4f, Prec=%.4f, Rec=%.4f, Acc=%.4f, AUC=%.4f, threshold=%.2f",
        class_name, f1, prec, rec, acc, auc, best_t,
    )
    return {"f1": f1, "precision": prec, "recall": rec, "accuracy": acc, "auc": auc,
            "threshold": best_t, "support": int(test_true.sum()), "n_test": len(test_true)}


# ---------------------------------------------------------------------------
# Experiment 2: Exclusive 3-class (clean subset)
# ---------------------------------------------------------------------------

def run_exclusive_classifier(
    X: torch.Tensor,
    y_labels: np.ndarray,
    class_names: List[str],
    device: str = "cpu",
    epochs: int = 200,
) -> Dict:
    """3-class classifier on proteins exclusive to one class."""
    n = X.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    n_train, n_val = int(n * 0.8), int(n * 0.1)
    tr, va, te = perm[:n_train], perm[n_train:n_train+n_val], perm[n_train+n_val:]

    y = torch.tensor(y_labels, dtype=torch.long)
    X_tr, y_tr = X[tr].to(device), y[tr].to(device)
    X_va, y_va = X[va].to(device), y[va].to(device)
    X_te, y_te = X[te].to(device), y[te].to(device)

    # Class weights
    class_counts = torch.bincount(y_tr, minlength=len(class_names)).float()
    weights = (class_counts.sum() / class_counts.clamp(min=1)).to(device)

    model = MultiClassifier(X.shape[1], len(class_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_val_f1, best_state = 0.0, None
    patience, no_improve = 30, 0

    for epoch in range(epochs):
        model.train()
        perm_tr = torch.randperm(X_tr.shape[0])
        for i in range(0, X_tr.shape[0], 256):
            idx = perm_tr[i:i+256]
            optimizer.zero_grad()
            loss = criterion(model(X_tr[idx]), y_tr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_va).argmax(dim=-1).cpu().numpy()
            val_true = y_va.cpu().numpy()
            val_f1 = sk_f1(val_true, val_preds, average="macro", zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_preds = model(X_te).argmax(dim=-1).cpu().numpy()
        test_true = y_te.cpu().numpy()

    macro_f1 = sk_f1(test_true, test_preds, average="macro", zero_division=0)
    acc = accuracy_score(test_true, test_preds)

    logger.info("[Exclusive 3-class] Macro F1=%.4f, Acc=%.4f", macro_f1, acc)
    report = classification_report(test_true, test_preds, target_names=class_names, zero_division=0)
    logger.info("\n%s", report)

    return {"macro_f1": macro_f1, "accuracy": acc}


# ---------------------------------------------------------------------------
# Experiment 3: Fraction regression
# ---------------------------------------------------------------------------

def run_fraction_regression(
    X: torch.Tensor,
    y_fractions: torch.Tensor,
    class_names: List[str],
    device: str = "cpu",
    epochs: int = 200,
) -> Dict:
    """Predict fraction of each glycan class per protein."""
    n = X.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    n_train, n_val = int(n * 0.8), int(n * 0.1)
    tr, va, te = perm[:n_train], perm[n_train:n_train+n_val], perm[n_train+n_val:]

    X_tr, y_tr = X[tr].to(device), y_fractions[tr].to(device)
    X_va, y_va = X[va].to(device), y_fractions[va].to(device)
    X_te, y_te = X[te].to(device), y_fractions[te].to(device)

    model = MultiClassifier(X.shape[1], len(class_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)

    best_val_loss, best_state = float("inf"), None
    patience, no_improve = 30, 0

    for epoch in range(epochs):
        model.train()
        perm_tr = torch.randperm(X_tr.shape[0])
        for i in range(0, X_tr.shape[0], 256):
            idx = perm_tr[i:i+256]
            optimizer.zero_grad()
            # Softmax output vs. target fractions → KL divergence
            logits = model(X_tr[idx])
            log_probs = F.log_softmax(logits, dim=-1)
            loss = F.kl_div(log_probs, y_tr[idx], reduction="batchmean")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_va)
            val_log_probs = F.log_softmax(val_logits, dim=-1)
            val_loss = F.kl_div(val_log_probs, y_va, reduction="batchmean").item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_probs = F.softmax(model(X_te), dim=-1).cpu().numpy()
        test_true = y_te.cpu().numpy()

    # Evaluate: dominant class accuracy (argmax)
    pred_dominant = test_probs.argmax(axis=1)
    true_dominant = test_true.argmax(axis=1)
    dominant_acc = accuracy_score(true_dominant, pred_dominant)
    dominant_f1 = sk_f1(true_dominant, pred_dominant, average="macro", zero_division=0)

    # MAE per class
    mae = np.abs(test_probs - test_true).mean(axis=0)

    logger.info("[Fraction Regression] Dominant class accuracy=%.4f, F1=%.4f", dominant_acc, dominant_f1)
    for i, cls in enumerate(class_names):
        logger.info("  %s: MAE=%.4f, pred_mean=%.3f, true_mean=%.3f",
                     cls, mae[i], test_probs[:, i].mean(), test_true[:, i].mean())

    return {"dominant_accuracy": dominant_acc, "dominant_f1": dominant_f1,
            "mae": {cls: float(mae[i]) for i, cls in enumerate(class_names)}}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    merged, site_counts, id_to_idx, esm2_dir, sequences = load_all_data()

    # Build per-protein labels
    prot_class_counts = (
        merged.groupby(["protein_id", "function_term"])
        .size()
        .reset_index(name="count")
    )
    prot_total = prot_class_counts.groupby("protein_id")["count"].sum().reset_index(name="total")
    prot_class_counts = prot_class_counts.merge(prot_total, on="protein_id")

    all_proteins = sorted(prot_class_counts["protein_id"].unique())
    logger.info("Total proteins: %d", len(all_proteins))

    # Load features
    X, valid_proteins = load_features(all_proteins, site_counts, id_to_idx, esm2_dir, sequences)
    logger.info("Features loaded: %d proteins, %d dims", X.shape[0], X.shape[1])

    # Build label mappings for valid proteins
    valid_set = set(valid_proteins)
    pid_to_idx = {p: i for i, p in enumerate(valid_proteins)}

    results = {}

    # ========================================
    # Experiment 1: Binary classifiers
    # ========================================
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 1: Binary classifiers")
    logger.info("=" * 60)

    prot_labels = merged.groupby("protein_id")["function_term"].apply(set).to_dict()

    for cls_name in ["N-linked", "O-linked"]:
        y = torch.zeros(len(valid_proteins))
        for i, pid in enumerate(valid_proteins):
            if pid in prot_labels and cls_name in prot_labels[pid]:
                y[i] = 1.0
        n_pos = int(y.sum())
        logger.info("%s: %d positive / %d total (%.1f%%)",
                     cls_name, n_pos, len(y), 100 * n_pos / len(y))
        r = run_binary_classifier(X, y, cls_name, device=device, epochs=200)
        results[f"binary_{cls_name}"] = r

    # ========================================
    # Experiment 2: Exclusive 3-class
    # ========================================
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 2: Exclusive 3-class classifier")
    logger.info("=" * 60)

    target_classes = ["N-linked", "O-linked", "Other"]
    exclusive_proteins = []
    exclusive_labels = []

    for pid in valid_proteins:
        if pid not in prot_labels:
            continue
        funcs = prot_labels[pid] & set(target_classes)
        if len(funcs) == 1:
            cls = funcs.pop()
            exclusive_proteins.append(pid)
            exclusive_labels.append(target_classes.index(cls))

    logger.info("Exclusive proteins: %d", len(exclusive_proteins))
    for i, cls in enumerate(target_classes):
        cnt = sum(1 for l in exclusive_labels if l == i)
        logger.info("  %s: %d", cls, cnt)

    excl_indices = [pid_to_idx[p] for p in exclusive_proteins]
    X_excl = X[excl_indices]
    y_excl = np.array(exclusive_labels)
    r = run_exclusive_classifier(X_excl, y_excl, target_classes, device=device, epochs=200)
    results["exclusive_3class"] = r

    # ========================================
    # Experiment 3: Fraction regression
    # ========================================
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 3: Fraction regression")
    logger.info("=" * 60)

    frac_classes = ["N-linked", "O-linked", "Other"]
    y_frac = torch.zeros(len(valid_proteins), len(frac_classes))
    for i, pid in enumerate(valid_proteins):
        sub = prot_class_counts[prot_class_counts["protein_id"] == pid]
        total = sub["count"].sum()
        if total == 0:
            y_frac[i, 0] = 1.0  # default N-linked
            continue
        for j, cls in enumerate(frac_classes):
            row = sub[sub["function_term"] == cls]
            if not row.empty:
                y_frac[i, j] = row["count"].values[0] / total
        # Normalize to sum=1 (redistribute Glycosphingolipid etc.)
        s = y_frac[i].sum()
        if s > 0:
            y_frac[i] /= s
        else:
            y_frac[i, 0] = 1.0

    r = run_fraction_regression(X, y_frac, frac_classes, device=device, epochs=200)
    results["fraction_regression"] = r

    # ========================================
    # Summary
    # ========================================
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    logger.info("Binary N-linked: F1=%.4f, AUC=%.4f",
                results["binary_N-linked"]["f1"], results["binary_N-linked"]["auc"])
    logger.info("Binary O-linked: F1=%.4f, AUC=%.4f",
                results["binary_O-linked"]["f1"], results["binary_O-linked"]["auc"])
    logger.info("Exclusive 3-class: Macro F1=%.4f, Acc=%.4f",
                results["exclusive_3class"]["macro_f1"], results["exclusive_3class"]["accuracy"])
    logger.info("Fraction regression: Dominant Acc=%.4f, F1=%.4f",
                results["fraction_regression"]["dominant_accuracy"],
                results["fraction_regression"]["dominant_f1"])

    # Save
    output_dir = Path("experiments_v2/glycan_class_v3")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
