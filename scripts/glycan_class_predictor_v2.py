#!/usr/bin/env python3
"""Glycan function class predictor v2 — improved features and training.

Improvements over v1:
- Sequence motif features (N-X-S/T count, S/T fraction, etc.)
- Feature standardization
- Focal loss for class imbalance
- Residual MLP architecture
- Focus on 4 active classes (N-linked, O-linked, Glycosphingolipid, Other)
- Label smoothing

Usage:
    python scripts/glycan_class_predictor_v2.py
    python scripts/glycan_class_predictor_v2.py --use-kg-emb --epochs 300
"""

from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Focus on classes with meaningful protein support
FUNCTION_CLASSES = [
    "N-linked",
    "O-linked",
    "Glycosphingolipid",
    "Other",
    "C-linked",
]
NUM_CLASSES = len(FUNCTION_CLASSES)


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalBCELoss(nn.Module):
    """Focal loss for multi-label classification."""

    def __init__(self, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        probs = logits.sigmoid()
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_protein_function_labels(data_dir: Path) -> Dict[str, Set[str]]:
    edges = pd.read_csv(data_dir / "edges_glycan_protein.tsv", sep="\t")
    edges = edges[["glycan_id", "protein_id"]].drop_duplicates()
    func = pd.read_csv(data_dir / "glycan_function_labels.tsv", sep="\t")
    func = func[["glycan_id", "function_term"]].drop_duplicates()
    merged = edges.merge(func, on="glycan_id", how="inner")
    return merged.groupby("protein_id")["function_term"].apply(set).to_dict()


def load_site_counts(data_dir: Path) -> Dict[str, Dict[str, int]]:
    sites = pd.read_csv(data_dir / "uniprot_sites.tsv", sep="\t")
    result: Dict[str, Dict[str, int]] = {}
    for uid, group in sites.groupby("uniprot_id"):
        result[uid] = group["site_type"].value_counts().to_dict()
    return result


def load_esm2_cache(data_dir: Path):
    esm2_dir = data_dir / "esm2_cache"
    with open(esm2_dir / "id_to_idx.json") as f:
        id_to_idx = json.load(f)
    return id_to_idx, esm2_dir


def load_protein_sequences(data_dir: Path) -> Dict[str, str]:
    """Return {protein_id: sequence}."""
    df = pd.read_csv(data_dir / "protein_sequences.tsv", sep="\t")
    return dict(zip(df["uniprot_id"], df["sequence"]))


def _resolve_esm2_idx(pid: str, id_to_idx: Dict[str, int]) -> Optional[int]:
    if pid in id_to_idx:
        return id_to_idx[pid]
    base = pid.split("-")[0]
    if f"{base}-1" in id_to_idx:
        return id_to_idx[f"{base}-1"]
    if base in id_to_idx:
        return id_to_idx[base]
    return None


# ---------------------------------------------------------------------------
# Sequence motif features
# ---------------------------------------------------------------------------

def extract_sequence_features(seq: str) -> torch.Tensor:
    """Extract glycosylation-relevant features from amino acid sequence.

    Returns tensor of shape [12]:
      - N-X-S/T motif count (N-linked sequon)
      - N-X-S motif count
      - N-X-T motif count
      - Fraction of N residues
      - Fraction of S residues
      - Fraction of T residues
      - Fraction of S+T residues
      - Fraction of W residues (tryptophan, common in glycosyltransferase recognition)
      - Log sequence length
      - Fraction hydrophobic (AILMFWV)
      - Fraction charged (DEKRH)
      - Fraction of P (proline, blocks N-linked)
    """
    L = len(seq) if seq else 1
    seq_upper = seq.upper() if seq else ""

    # N-X-S/T sequon (N-linked glycosylation motif, X != P)
    nxst = len(re.findall(r"N[^P][ST]", seq_upper))
    nxs = len(re.findall(r"N[^P]S", seq_upper))
    nxt = len(re.findall(r"N[^P]T", seq_upper))

    aa_counts = {aa: seq_upper.count(aa) for aa in "NSTPWAILMFVDEKRH"}

    features = [
        nxst / max(L, 1),                          # sequon density
        nxs / max(L, 1),
        nxt / max(L, 1),
        aa_counts.get("N", 0) / L,                 # Asn fraction
        aa_counts.get("S", 0) / L,                 # Ser fraction
        aa_counts.get("T", 0) / L,                 # Thr fraction
        (aa_counts.get("S", 0) + aa_counts.get("T", 0)) / L,  # S+T fraction
        aa_counts.get("W", 0) / L,                 # Trp fraction
        np.log1p(L),                                # log length
        sum(aa_counts.get(aa, 0) for aa in "AILMFWV") / L,  # hydrophobic
        sum(aa_counts.get(aa, 0) for aa in "DEKRH") / L,    # charged
        aa_counts.get("P", 0) / L,                 # Pro fraction
    ]
    return torch.tensor(features, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_dataset(
    prot_funcs: Dict[str, Set[str]],
    site_counts: Dict[str, Dict[str, int]],
    id_to_idx: Dict[str, int],
    esm2_dir: Path,
    sequences: Dict[str, str],
    kg_emb: Optional[Tuple[torch.Tensor, Dict[str, int]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Build (X, y, protein_ids)."""
    class_to_idx = {c: i for i, c in enumerate(FUNCTION_CLASSES)}
    proteins = sorted(prot_funcs.keys())

    features: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    valid_proteins: List[str] = []

    kg_emb_tensor, kg_node_map = (None, None) if kg_emb is None else kg_emb
    n_skip = 0

    for pid in proteins:
        esm_idx = _resolve_esm2_idx(pid, id_to_idx)
        if esm_idx is None:
            n_skip += 1
            continue
        pt_path = esm2_dir / f"{esm_idx}.pt"
        if not pt_path.exists():
            n_skip += 1
            continue

        emb = torch.load(pt_path, map_location="cpu", weights_only=True)
        if emb.dim() == 2:
            emb = emb.mean(dim=0)

        # Site features [4]
        base_id = pid.split("-")[0]
        sc = site_counts.get(base_id, {})
        site_feat = torch.tensor([
            sc.get("N-linked", 0),
            sc.get("O-linked", 0),
            sc.get("C-mannosylation", 0),
            sc.get("glycosylation", 0),
        ], dtype=torch.float32)

        # Sequence motif features [12]
        seq = sequences.get(pid, sequences.get(f"{base_id}-1", ""))
        seq_feat = extract_sequence_features(seq)

        parts = [emb, site_feat, seq_feat]

        # Optional KG embedding
        if kg_emb_tensor is not None and kg_node_map is not None:
            kg_idx = kg_node_map.get(pid)
            if kg_idx is not None and kg_idx < kg_emb_tensor.shape[0]:
                parts.append(kg_emb_tensor[kg_idx])
            else:
                parts.append(torch.zeros(kg_emb_tensor.shape[1]))

        feat = torch.cat(parts)
        features.append(feat)

        # Multi-hot target (only active classes)
        target = torch.zeros(NUM_CLASSES)
        for func_name in prot_funcs[pid]:
            if func_name in class_to_idx:
                target[class_to_idx[func_name]] = 1.0
        targets.append(target)
        valid_proteins.append(pid)

    if n_skip > 0:
        logger.warning("Skipped %d proteins (no ESM2 embedding)", n_skip)

    X = torch.stack(features)
    y = torch.stack(targets)

    # Standardize features
    mean = X.mean(dim=0)
    std = X.std(dim=0).clamp(min=1e-6)
    X = (X - mean) / std

    logger.info("Dataset: %d proteins, %d features, %d classes", X.shape[0], X.shape[1], y.shape[1])
    dist = {FUNCTION_CLASSES[i]: int(y[:, i].sum()) for i in range(NUM_CLASSES)}
    logger.info("Class distribution: %s", dist)

    return X, y, valid_proteins, mean, std


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class GlycanClassifierV2(nn.Module):
    """Residual MLP multi-label classifier."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = NUM_CLASSES,
        hidden_dim: int = 512,
        n_blocks: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.blocks(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def site_heuristic_baseline(
    prot_funcs: Dict[str, Set[str]],
    site_counts: Dict[str, Dict[str, int]],
) -> None:
    from sklearn.metrics import f1_score as sk_f1

    class_to_idx = {c: i for i, c in enumerate(FUNCTION_CLASSES)}
    y_true_list, y_pred_list = [], []

    for pid, funcs in prot_funcs.items():
        true = np.zeros(NUM_CLASSES)
        for f in funcs:
            if f in class_to_idx:
                true[class_to_idx[f]] = 1.0
        y_true_list.append(true)

        pred = np.zeros(NUM_CLASSES)
        base_id = pid.split("-")[0]
        sc = site_counts.get(base_id, {})
        if sc.get("N-linked", 0) > 0:
            pred[class_to_idx["N-linked"]] = 1.0
        if sc.get("O-linked", 0) > 0:
            pred[class_to_idx["O-linked"]] = 1.0
        if sc.get("C-mannosylation", 0) > 0 and "C-linked" in class_to_idx:
            pred[class_to_idx["C-linked"]] = 1.0
        if pred.sum() == 0:
            pred[class_to_idx["N-linked"]] = 1.0
        y_pred_list.append(pred)

    y_true = np.stack(y_true_list)
    y_pred = np.stack(y_pred_list)
    macro = sk_f1(y_true, y_pred, average="macro", zero_division=0)
    micro = sk_f1(y_true, y_pred, average="micro", zero_division=0)
    logger.info("=== Site Heuristic Baseline ===")
    logger.info("Macro F1: %.4f, Micro F1: %.4f", macro, micro)
    for i, cls in enumerate(FUNCTION_CLASSES):
        f1 = sk_f1(y_true[:, i], y_pred[:, i], zero_division=0)
        logger.info("  %s: F1=%.4f (support=%d)", cls, f1, int(y_true[:, i].sum()))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_evaluate(
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 300,
    lr: float = 5e-4,
    batch_size: int = 256,
    device: str = "cpu",
    gamma: float = 2.0,
    label_smoothing: float = 0.05,
) -> Tuple[GlycanClassifierV2, float, float, Dict]:
    from sklearn.metrics import classification_report, f1_score as sk_f1

    n = X.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(42))

    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    X_train, y_train = X[train_idx].to(device), y[train_idx].to(device)
    X_val, y_val = X[val_idx].to(device), y[val_idx].to(device)
    X_test, y_test = X[test_idx].to(device), y[test_idx].to(device)

    logger.info("Split: train=%d, val=%d, test=%d", len(train_idx), len(val_idx), len(test_idx))

    model = GlycanClassifierV2(X.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    # Warmup + cosine
    warmup_epochs = 10
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    pos_weight = (y_train.shape[0] - y_train.sum(0)) / y_train.sum(0).clamp(min=1)
    criterion = FocalBCELoss(gamma=gamma, pos_weight=pos_weight, label_smoothing=label_smoothing)

    best_val_f1 = 0.0
    best_state = None
    patience = 40
    no_improve = 0

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * xb.shape[0]
        scheduler.step()

        avg_loss = total_loss / len(train_idx)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_preds = (val_logits.sigmoid() > 0.5).cpu().numpy()
            val_true = y_val.cpu().numpy()
            val_f1 = sk_f1(val_true, val_preds, average="macro", zero_division=0)

        if (epoch + 1) % 30 == 0:
            logger.info("Epoch %d: loss=%.4f, val_macro_F1=%.4f", epoch + 1, avg_loss, val_f1)

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

    # Per-class threshold search on val set
    best_thresholds = [0.5] * NUM_CLASSES
    with torch.no_grad():
        val_logits = model(X_val).sigmoid().cpu().numpy()
        val_true = y_val.cpu().numpy()

    for i in range(NUM_CLASSES):
        if val_true[:, i].sum() == 0:
            continue
        best_t, best_f = 0.5, 0.0
        for t in np.arange(0.1, 0.9, 0.02):
            preds = (val_logits[:, i] > t).astype(float)
            f = sk_f1(val_true[:, i], preds, zero_division=0)
            if f > best_f:
                best_f = f
                best_t = t
        best_thresholds[i] = best_t
    logger.info("Thresholds: %s", {c: f"{t:.2f}" for c, t in zip(FUNCTION_CLASSES, best_thresholds)})

    # Test
    with torch.no_grad():
        test_logits = model(X_test).sigmoid().cpu().numpy()
        test_true = y_test.cpu().numpy()

    test_preds = np.zeros_like(test_logits)
    for i in range(NUM_CLASSES):
        test_preds[:, i] = (test_logits[:, i] > best_thresholds[i]).astype(float)

    macro_f1 = sk_f1(test_true, test_preds, average="macro", zero_division=0)
    micro_f1 = sk_f1(test_true, test_preds, average="micro", zero_division=0)

    # Weighted macro (only classes with support)
    active_f1s = []
    for i in range(NUM_CLASSES):
        if test_true[:, i].sum() > 0:
            active_f1s.append(sk_f1(test_true[:, i], test_preds[:, i], zero_division=0))
    active_macro = np.mean(active_f1s) if active_f1s else 0.0

    logger.info("\n=== Test Results ===")
    logger.info("Best val macro F1: %.4f", best_val_f1)
    logger.info("Test macro F1: %.4f (active classes: %.4f)", macro_f1, active_macro)
    logger.info("Test micro F1: %.4f", micro_f1)

    report = classification_report(
        test_true, test_preds, target_names=FUNCTION_CLASSES, zero_division=0,
    )
    logger.info("\n%s", report)

    results = {}
    for i, cls_name in enumerate(FUNCTION_CLASSES):
        n_pos = int(test_true[:, i].sum())
        n_pred = int(test_preds[:, i].sum())
        f1 = sk_f1(test_true[:, i], test_preds[:, i], zero_division=0)
        logger.info("  %s: support=%d, predicted=%d, F1=%.4f, threshold=%.2f",
                     cls_name, n_pos, n_pred, f1, best_thresholds[i])
        results[cls_name] = {"support": n_pos, "predicted": n_pred, "f1": f1,
                             "threshold": best_thresholds[i]}

    return model, macro_f1, micro_f1, results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_clean")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal loss gamma")
    parser.add_argument("--use-kg-emb", action="store_true")
    parser.add_argument("--kg-checkpoint", default="experiments_v2/glycokgnet_inductive_r8")
    parser.add_argument("--output-dir", default="experiments_v2/glycan_class_v2")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    data_dir = Path(args.data_dir)
    prot_funcs = load_protein_function_labels(data_dir)
    site_counts = load_site_counts(data_dir)
    id_to_idx, esm2_dir = load_esm2_cache(data_dir)
    sequences = load_protein_sequences(data_dir)

    logger.info("Proteins: %d, with site data: %d, with sequences: %d",
                len(prot_funcs), len(site_counts), len(sequences))

    # Baseline
    site_heuristic_baseline(prot_funcs, site_counts)

    # KG embeddings
    kg_emb = None
    if args.use_kg_emb:
        kg_dir = Path(args.kg_checkpoint)
        ds_path = kg_dir / "dataset.pt"
        if ds_path.exists():
            logger.info("Loading KG embeddings from %s", kg_dir)
            ds = torch.load(ds_path, map_location="cpu", weights_only=False)
            node_mappings = ds["node_mappings"]
            protein_map = node_mappings.get("protein", {})
            protein_feat = ds["data"]["protein"].x
            kg_emb = (protein_feat.detach(), protein_map)
            logger.info("KG protein embeddings: %s", protein_feat.shape)

    # Build dataset
    X, y, proteins, mean, std = build_dataset(
        prot_funcs, site_counts, id_to_idx, esm2_dir, sequences, kg_emb,
    )

    # Train
    model, macro_f1, micro_f1, results = train_and_evaluate(
        X, y, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        device=device, gamma=args.gamma,
    )

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_mean": mean,
        "feature_std": std,
        "function_classes": FUNCTION_CLASSES,
    }, output_dir / "classifier.pt")

    summary = {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "per_class": results,
        "num_proteins": X.shape[0],
        "feature_dim": X.shape[1],
        "use_kg_emb": args.use_kg_emb,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Saved to %s", output_dir)
    logger.info("Final: macro_F1=%.4f, micro_F1=%.4f", macro_f1, micro_f1)


if __name__ == "__main__":
    main()
