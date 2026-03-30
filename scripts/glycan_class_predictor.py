#!/usr/bin/env python3
"""Glycan function class predictor from protein features.

Multi-label classification: protein → {N-linked, O-linked, GAG, ...}
Uses ESM2 mean-pool embeddings + site residue count features.

Usage:
    python scripts/glycan_class_predictor.py
    python scripts/glycan_class_predictor.py --epochs 300 --use-kg-emb
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

FUNCTION_CLASSES = [
    "N-linked",
    "O-linked",
    "GAG",
    "Glycosphingolipid",
    "Human Milk Oligosaccharide",
    "GPI anchor",
    "C-linked",
    "Other",
]
NUM_CLASSES = len(FUNCTION_CLASSES)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_protein_function_labels(
    data_dir: Path,
) -> Dict[str, Set[str]]:
    """Return {protein_id: {function_term, ...}}."""
    edges = pd.read_csv(data_dir / "edges_glycan_protein.tsv", sep="\t")
    edges = edges[["glycan_id", "protein_id"]].drop_duplicates()
    func = pd.read_csv(data_dir / "glycan_function_labels.tsv", sep="\t")
    func = func[["glycan_id", "function_term"]].drop_duplicates()
    merged = edges.merge(func, on="glycan_id", how="inner")
    return merged.groupby("protein_id")["function_term"].apply(set).to_dict()


def load_site_counts(data_dir: Path) -> Dict[str, Dict[str, int]]:
    """Return {uniprot_base_id: {site_type: count}}."""
    sites = pd.read_csv(data_dir / "uniprot_sites.tsv", sep="\t")
    result: Dict[str, Dict[str, int]] = {}
    for uid, group in sites.groupby("uniprot_id"):
        result[uid] = group["site_type"].value_counts().to_dict()
    return result


def load_esm2_cache(data_dir: Path):
    """Return (id_to_idx dict, cache directory path)."""
    esm2_dir = data_dir / "esm2_cache"
    with open(esm2_dir / "id_to_idx.json") as f:
        id_to_idx = json.load(f)
    return id_to_idx, esm2_dir


def _resolve_esm2_idx(
    pid: str, id_to_idx: Dict[str, int]
) -> Optional[int]:
    """Resolve protein_id to ESM2 cache index."""
    if pid in id_to_idx:
        return id_to_idx[pid]
    base = pid.split("-")[0]
    if f"{base}-1" in id_to_idx:
        return id_to_idx[f"{base}-1"]
    if base in id_to_idx:
        return id_to_idx[base]
    return None


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_dataset(
    prot_funcs: Dict[str, Set[str]],
    site_counts: Dict[str, Dict[str, int]],
    id_to_idx: Dict[str, int],
    esm2_dir: Path,
    kg_emb: Optional[Tuple[torch.Tensor, Dict[str, int]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Build (X, y, protein_ids).

    Features: ESM2 [1280] + site_counts [4] (+ KG emb [256] if provided).
    Target: multi-hot [8].
    """
    class_to_idx = {c: i for i, c in enumerate(FUNCTION_CLASSES)}
    proteins = sorted(prot_funcs.keys())

    features: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    valid_proteins: List[str] = []

    kg_emb_tensor, kg_node_map = (None, None) if kg_emb is None else kg_emb
    n_skip_esm = 0

    for pid in proteins:
        esm_idx = _resolve_esm2_idx(pid, id_to_idx)
        if esm_idx is None:
            n_skip_esm += 1
            continue
        pt_path = esm2_dir / f"{esm_idx}.pt"
        if not pt_path.exists():
            n_skip_esm += 1
            continue

        emb = torch.load(pt_path, map_location="cpu", weights_only=True)
        if emb.dim() == 2:
            emb = emb.mean(dim=0)

        # Site features
        base_id = pid.split("-")[0]
        sc = site_counts.get(base_id, {})
        site_feat = torch.tensor(
            [
                sc.get("N-linked", 0),
                sc.get("O-linked", 0),
                sc.get("C-mannosylation", 0),
                sc.get("glycosylation", 0),
            ],
            dtype=torch.float32,
        )

        parts = [emb, site_feat]

        # Optional KG embedding
        if kg_emb_tensor is not None and kg_node_map is not None:
            kg_idx = kg_node_map.get(pid)
            if kg_idx is not None and kg_idx < kg_emb_tensor.shape[0]:
                parts.append(kg_emb_tensor[kg_idx])
            else:
                parts.append(torch.zeros(kg_emb_tensor.shape[1]))

        feat = torch.cat(parts)
        features.append(feat)

        # Multi-hot target
        target = torch.zeros(NUM_CLASSES)
        for func_name in prot_funcs[pid]:
            if func_name in class_to_idx:
                target[class_to_idx[func_name]] = 1.0
        targets.append(target)
        valid_proteins.append(pid)

    if n_skip_esm > 0:
        logger.warning("Skipped %d proteins (no ESM2 embedding)", n_skip_esm)

    X = torch.stack(features)
    y = torch.stack(targets)
    logger.info(
        "Dataset: %d proteins, %d features, %d classes",
        X.shape[0], X.shape[1], y.shape[1],
    )
    dist = {FUNCTION_CLASSES[i]: int(y[:, i].sum()) for i in range(NUM_CLASSES)}
    logger.info("Class distribution: %s", dist)
    return X, y, valid_proteins


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GlycanClassifier(nn.Module):
    """MLP multi-label classifier."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = NUM_CLASSES,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Baseline: site residue heuristic
# ---------------------------------------------------------------------------

def site_heuristic_baseline(
    prot_funcs: Dict[str, Set[str]],
    site_counts: Dict[str, Dict[str, int]],
) -> None:
    """Predict function class from site residue type only."""
    from sklearn.metrics import f1_score as sk_f1

    class_to_idx = {c: i for i, c in enumerate(FUNCTION_CLASSES)}
    y_true_list = []
    y_pred_list = []

    for pid, funcs in prot_funcs.items():
        base_id = pid.split("-")[0]
        sc = site_counts.get(base_id, {})

        true = np.zeros(NUM_CLASSES)
        for f in funcs:
            if f in class_to_idx:
                true[class_to_idx[f]] = 1.0
        y_true_list.append(true)

        pred = np.zeros(NUM_CLASSES)
        # Heuristic: if N-linked sites → predict N-linked
        if sc.get("N-linked", 0) > 0:
            pred[class_to_idx["N-linked"]] = 1.0
        if sc.get("O-linked", 0) > 0:
            pred[class_to_idx["O-linked"]] = 1.0
        if sc.get("C-mannosylation", 0) > 0:
            pred[class_to_idx["C-linked"]] = 1.0
        # No site data → predict most common (N-linked)
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
# Training loop
# ---------------------------------------------------------------------------

def train_and_evaluate(
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 256,
    device: str = "cpu",
    threshold_search: bool = True,
) -> Tuple[GlycanClassifier, float, float, Dict]:
    """Train with 80/10/10 split."""
    from sklearn.metrics import (
        classification_report,
        f1_score as sk_f1,
    )

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

    logger.info(
        "Split: train=%d, val=%d, test=%d", len(train_idx), len(val_idx), len(test_idx)
    )

    model = GlycanClassifier(X.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Class-balanced loss
    pos_weight = (y_train.shape[0] - y_train.sum(0)) / y_train.sum(0).clamp(min=1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_f1 = 0.0
    best_state = None
    patience = 30
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
            optimizer.step()
            total_loss += loss.item() * xb.shape[0]
        scheduler.step()

        avg_loss = total_loss / len(train_idx)

        # Validate
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_preds = (val_logits.sigmoid() > 0.5).cpu().numpy()
            val_true = y_val.cpu().numpy()
            val_f1 = sk_f1(val_true, val_preds, average="macro", zero_division=0)

        if (epoch + 1) % 20 == 0:
            logger.info(
                "Epoch %d: loss=%.4f, val_macro_F1=%.4f", epoch + 1, avg_loss, val_f1
            )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    # Load best model
    model.load_state_dict(best_state)
    model.eval()

    # Threshold search on val set
    best_thresholds = [0.5] * NUM_CLASSES
    if threshold_search:
        with torch.no_grad():
            val_logits = model(X_val).sigmoid().cpu().numpy()
            val_true = y_val.cpu().numpy()

        for i in range(NUM_CLASSES):
            best_t, best_f = 0.5, 0.0
            for t in np.arange(0.1, 0.9, 0.05):
                preds = (val_logits[:, i] > t).astype(float)
                f = sk_f1(val_true[:, i], preds, zero_division=0)
                if f > best_f:
                    best_f = f
                    best_t = t
            best_thresholds[i] = best_t
        logger.info("Optimized thresholds: %s", dict(zip(FUNCTION_CLASSES, best_thresholds)))

    # Test evaluation
    with torch.no_grad():
        test_logits = model(X_test).sigmoid().cpu().numpy()
        test_true = y_test.cpu().numpy()

    test_preds = np.zeros_like(test_logits)
    for i in range(NUM_CLASSES):
        test_preds[:, i] = (test_logits[:, i] > best_thresholds[i]).astype(float)

    macro_f1 = sk_f1(test_true, test_preds, average="macro", zero_division=0)
    micro_f1 = sk_f1(test_true, test_preds, average="micro", zero_division=0)

    logger.info("\n=== Test Results ===")
    logger.info("Best val macro F1: %.4f", best_val_f1)
    logger.info("Test macro F1: %.4f", macro_f1)
    logger.info("Test micro F1: %.4f", micro_f1)

    report = classification_report(
        test_true,
        test_preds,
        target_names=FUNCTION_CLASSES,
        zero_division=0,
    )
    logger.info("\n%s", report)

    # Per-class detail
    results = {}
    for i, cls_name in enumerate(FUNCTION_CLASSES):
        n_pos = int(test_true[:, i].sum())
        n_pred = int(test_preds[:, i].sum())
        f1 = sk_f1(test_true[:, i], test_preds[:, i], zero_division=0)
        prec = sk_f1(test_true[:, i], test_preds[:, i], zero_division=0)
        logger.info("  %s: support=%d, predicted=%d, F1=%.4f", cls_name, n_pos, n_pred, f1)
        results[cls_name] = {"support": n_pos, "predicted": n_pred, "f1": f1}

    return model, macro_f1, micro_f1, results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Glycan function class predictor")
    parser.add_argument("--data-dir", default="data_clean", help="Data directory")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--use-kg-emb", action="store_true", help="Add KG-learned embeddings")
    parser.add_argument("--kg-checkpoint", default="experiments_v2/glycokgnet_inductive_r8")
    parser.add_argument("--output-dir", default="experiments_v2/glycan_class_predictor")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    data_dir = Path(args.data_dir)

    # Load data
    prot_funcs = load_protein_function_labels(data_dir)
    site_counts = load_site_counts(data_dir)
    id_to_idx, esm2_dir = load_esm2_cache(data_dir)

    logger.info("Proteins with labels: %d", len(prot_funcs))
    logger.info("Proteins with site data: %d", len(site_counts))

    # Baseline
    site_heuristic_baseline(prot_funcs, site_counts)

    # Optional KG embeddings
    kg_emb = None
    if args.use_kg_emb:
        kg_dir = Path(args.kg_checkpoint)
        ds_path = kg_dir / "dataset.pt"
        ckpt_path = kg_dir / "best.pt"
        if ds_path.exists() and ckpt_path.exists():
            logger.info("Loading KG embeddings from %s", kg_dir)
            ds = torch.load(ds_path, map_location="cpu", weights_only=False)
            node_mappings = ds["node_mappings"]
            protein_map = node_mappings.get("protein", {})
            # Get protein embeddings from dataset features
            protein_feat = ds["data"]["protein"].x  # [num_proteins, 256]
            kg_emb = (protein_feat.detach(), protein_map)
            logger.info("KG protein embeddings: %s", protein_feat.shape)
        else:
            logger.warning("KG checkpoint not found at %s", kg_dir)

    # Build dataset
    X, y, proteins = build_dataset(prot_funcs, site_counts, id_to_idx, esm2_dir, kg_emb)

    # Train
    model, macro_f1, micro_f1, results = train_and_evaluate(
        X, y, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, device=device,
    )

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "classifier.pt")

    summary = {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "per_class": results,
        "num_proteins": X.shape[0],
        "feature_dim": X.shape[1],
        "use_kg_emb": args.use_kg_emb,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Results saved to %s", output_dir)
    logger.info("Final: macro_F1=%.4f, micro_F1=%.4f", macro_f1, micro_f1)


if __name__ == "__main__":
    main()
