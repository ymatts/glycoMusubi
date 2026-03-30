#!/usr/bin/env python3
"""Glycan Composition Predictor: Site → Structural Features.

Predicts interpretable glycan structural features from protein site context:
1. Monosaccharide composition (GlcNAc, Man, Gal, Fuc, NeuAc, etc.)
2. Core structure type (high-mannose, complex, hybrid)
3. Branching pattern (mono/bi/tri/tetra-antennary)

This addresses "WHY does this structure attach at this site?" by predicting
the biosynthetic processing outcome from the protein sequence context.

Usage:
    python scripts/glycan_composition_predictor.py
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT = Path(__file__).resolve().parent.parent


# ── WURCS parsing ──────────────────────────────────────────────────────────


# Key monosaccharides by WURCS residue patterns
MONO_PATTERNS = {
    "GlcNAc": r"a2122h.*2\*NCC",   # N-Acetylglucosamine
    "GalNAc": r"a2112h.*2\*NCC",   # N-Acetylgalactosamine
    "Man":    r"a1122h-1[ab]_1-5(?!.*\*)",  # Mannose (no modifications)
    "Gal":   r"a2112h-1[ab]_1-5(?!.*\*)",  # Galactose (no modifications)
    "Fuc":   r"a1221m-1[ab]_1-5",  # Fucose (6-deoxy)
    "NeuAc": r"a2112h-1[abx]_1-5.*5\*NCCO",  # Sialic acid (Neu5Ac)
    "NeuGc": r"a2112h.*5\*NCCO/7=O",  # Neu5Gc
    "Xyl":   r"a1122h-1[abx]_1-4",  # Xylose (5-carbon)
    "GlcA":  r"a1122h.*6\*OC",     # Glucuronic acid
}


def parse_wurcs_composition(wurcs: str) -> dict:
    """Parse WURCS string to monosaccharide composition."""
    comp = Counter()
    if not isinstance(wurcs, str) or not wurcs.startswith("WURCS"):
        return comp

    # Extract residue definitions between [ ]
    residues_str = wurcs.split("/")
    if len(residues_str) < 3:
        return comp

    # Get the residue blocks (parts 2..N-2 contain residue defs)
    # WURCS format: version/counts/[res1][res2]...[resN]/topology/linkage
    # The residues are in the section after counts, between [ ]
    residue_section = "/".join(residues_str[2:])
    residue_defs = re.findall(r"\[([^\]]+)\]", residue_section)

    # Get topology to count occurrences
    # topology is the part after residue defs, like "1-2-3-2-3"
    topo_match = re.search(r"\]/(\d[\d-]*)/", wurcs)
    if topo_match:
        topo = topo_match.group(1)
        residue_counts = [int(x) for x in topo.split("-")]
    else:
        residue_counts = list(range(1, len(residue_defs) + 1))

    # Count each residue type
    for res_num in residue_counts:
        if res_num < 1 or res_num > len(residue_defs):
            continue
        res_def = residue_defs[res_num - 1]

        matched = False
        for mono_name, pattern in MONO_PATTERNS.items():
            if re.search(pattern, res_def):
                comp[mono_name] += 1
                matched = True
                break
        if not matched:
            comp["Other"] += 1

    return comp


def classify_nlinked_core(comp: dict) -> str:
    """Classify N-linked glycan core type from composition."""
    man = comp.get("Man", 0)
    glcnac = comp.get("GlcNAc", 0)
    gal = comp.get("Gal", 0)
    neua = comp.get("NeuAc", 0)
    fuc = comp.get("Fuc", 0)

    total = sum(comp.values())
    if total < 3:
        return "minimal"

    # High-mannose: mostly Man with 2 GlcNAc core
    if man >= 5 and gal == 0 and neua == 0:
        return "high_mannose"

    # Complex: has Gal and/or NeuAc
    if gal > 0 or neua > 0:
        if man >= 3:
            return "hybrid"
        return "complex"

    # Pauci-mannose
    if man >= 2 and man <= 4 and glcnac <= 3:
        return "pauci_mannose"

    return "other"


def count_antennae(wurcs: str) -> int:
    """Estimate branching (antennae count) from WURCS linkage."""
    if not isinstance(wurcs, str):
        return 0
    # Count unique branch points in linkage section
    parts = wurcs.split("/")
    if len(parts) < 5:
        return 1
    linkage = parts[-1]
    # Count number of _ (branch points)
    branches = linkage.count("_")
    if branches <= 2:
        return 1  # mono-antennary
    elif branches <= 4:
        return 2  # bi-antennary
    elif branches <= 6:
        return 3  # tri-antennary
    else:
        return 4  # tetra-antennary


# ── Feature building ───────────────────────────────────────────────────────


MONO_NAMES = ["GlcNAc", "Man", "Gal", "Fuc", "NeuAc", "GalNAc",
              "NeuGc", "Xyl", "GlcA", "Other"]
CORE_TYPES = ["high_mannose", "complex", "hybrid", "pauci_mannose", "minimal", "other"]


def build_glycan_features(wurcs_dict: dict) -> dict:
    """Build composition, core type, and branching features for each glycan."""
    features = {}
    for glycan_id, wurcs in wurcs_dict.items():
        comp = parse_wurcs_composition(wurcs)
        core = classify_nlinked_core(comp)
        antennae = count_antennae(wurcs)

        # Composition vector (10-dim)
        comp_vec = torch.zeros(len(MONO_NAMES))
        for i, name in enumerate(MONO_NAMES):
            comp_vec[i] = comp.get(name, 0)

        # Log-transform composition (better for regression)
        comp_vec_log = torch.log1p(comp_vec)

        features[glycan_id] = {
            "composition": comp_vec,
            "composition_log": comp_vec_log,
            "core_type": CORE_TYPES.index(core),
            "antennae": min(antennae, 4),
            "total_residues": sum(comp.values()),
        }
    return features


# ── Model ──────────────────────────────────────────────────────────────────


class GlycanCompositionPredictor(nn.Module):
    """Predicts glycan composition from protein site context."""

    def __init__(self, esm2_dim=1280, hidden=512, n_mono=10,
                 n_core_types=6, dropout=0.2):
        super().__init__()

        # Shared encoder: site_local + global_mean
        self.encoder = nn.Sequential(
            nn.Linear(esm2_dim * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Composition head (multi-target regression)
        self.comp_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, n_mono),
        )

        # Core type head (classification)
        self.core_head = nn.Sequential(
            nn.Linear(hidden, hidden // 4),
            nn.GELU(),
            nn.Linear(hidden // 4, n_core_types),
        )

        # Antennae head (regression)
        self.antenna_head = nn.Sequential(
            nn.Linear(hidden, hidden // 4),
            nn.GELU(),
            nn.Linear(hidden // 4, 1),
        )

    def forward(self, site_local, global_mean):
        x = torch.cat([site_local, global_mean], dim=-1)
        h = self.encoder(x)
        comp = self.comp_head(h)      # [B, 10] log composition
        core = self.core_head(h)      # [B, 6] logits
        ant = self.antenna_head(h)    # [B, 1]
        return comp, core, ant


# ── Training & Evaluation ─────────────────────────────────────────────────


def train_and_evaluate():
    logger.info("=" * 70)
    logger.info("  Glycan Composition Predictor")
    logger.info("=" * 70)

    # 1. Load WURCS structures
    struct_df = pd.read_csv(PROJECT / "data_raw/glytoucan_structures.tsv", sep="\t")
    wurcs_dict = dict(zip(struct_df["glycan_id"], struct_df["structure"]))
    logger.info("WURCS structures: %d", len(wurcs_dict))

    # 2. Build glycan features
    glycan_features = build_glycan_features(wurcs_dict)
    logger.info("Glycan features built for %d glycans", len(glycan_features))

    # Check composition parsing quality
    n_parsed = sum(1 for f in glycan_features.values() if f["total_residues"] > 0)
    logger.info("Successfully parsed: %d / %d (%.1f%%)",
                n_parsed, len(glycan_features),
                100 * n_parsed / len(glycan_features))

    # Core type distribution
    core_counts = Counter(f["core_type"] for f in glycan_features.values())
    logger.info("Core type distribution:")
    for ct, name in enumerate(CORE_TYPES):
        logger.info("  %s: %d", name, core_counts.get(ct, 0))

    # 3. Load site-glycan data (N-linked)
    site_df = pd.read_csv(PROJECT / "data_clean/glyconnect_site_glycans.tsv", sep="\t")
    site_df["glyc_type_clean"] = site_df["glycosylation_type"].apply(
        lambda x: x.split(";")[0] if ";" in str(x) else str(x)
    )
    ndf = site_df[site_df["glyc_type_clean"] == "N-linked"].copy()

    # Filter to glycans with WURCS and parsed features
    ndf = ndf[ndf["glytoucan_ac"].isin(glycan_features)]
    ndf = ndf[ndf["glytoucan_ac"].apply(
        lambda g: glycan_features[g]["total_residues"] > 0
    )]
    logger.info("N-linked triples with parsed WURCS: %d", len(ndf))

    # 4. ESM2
    esm2_dir = PROJECT / "data_clean/esm2_cache"
    with open(esm2_dir / "id_to_idx.json") as f:
        esm2_id_to_idx = json.load(f)
    canonical_to_esm2 = {}
    for esm_id in esm2_id_to_idx:
        canonical = esm_id.split("-")[0]
        if canonical not in canonical_to_esm2 or esm_id.endswith("-1"):
            canonical_to_esm2[canonical] = esm_id

    per_residue_dir = PROJECT / "data_clean/esm2_perresidue"

    ndf = ndf[ndf["uniprot_ac"].isin(canonical_to_esm2)]
    logger.info("After ESM2 filter: %d triples", len(ndf))

    # 5. Split
    all_prots = sorted(ndf["uniprot_ac"].unique())
    np.random.seed(42)
    np.random.shuffle(all_prots)
    n = len(all_prots)
    train_prots = set(all_prots[:int(0.8 * n)])
    val_prots = set(all_prots[int(0.8 * n):int(0.9 * n)])
    test_prots = set(all_prots[int(0.9 * n):])

    # 6. Build samples
    logger.info("Building samples...")

    def _build(prot_set):
        samples = []
        sub = ndf[ndf["uniprot_ac"].isin(prot_set)]
        for prot_id, grp in sub.groupby("uniprot_ac"):
            esm2_id = canonical_to_esm2.get(prot_id)
            if not esm2_id:
                continue
            esm_idx = esm2_id_to_idx[esm2_id]
            global_mean = torch.load(esm2_dir / f"{esm_idx}.pt",
                                     map_location="cpu", weights_only=True)
            if isinstance(global_mean, dict):
                global_mean = list(global_mean.values())[0]

            # Per-residue
            pr_data = None
            for name in [prot_id, esm2_id, prot_id.split("-")[0]]:
                pr_path = per_residue_dir / f"{name}.pt"
                if pr_path.exists():
                    pr_data = torch.load(pr_path, map_location="cpu", weights_only=True)
                    if isinstance(pr_data, dict):
                        pr_data = pr_data.get("embeddings", pr_data.get("embedding"))
                    break

            for _, row in grp.iterrows():
                pos = int(row["site_position"]) - 1
                gf = glycan_features[row["glytoucan_ac"]]

                if pr_data is not None and pr_data.dim() >= 2:
                    L = pr_data.shape[0]
                    p = min(max(pos, 0), L - 1)
                    s, e = max(0, p - 16), min(L, p + 17)
                    local = pr_data[s:e]
                    c = p - s
                    d = torch.abs(torch.arange(local.shape[0], dtype=torch.float) - c)
                    w = torch.exp(-0.15 * d)
                    w /= w.sum()
                    site_local = (local * w.unsqueeze(1)).sum(0)
                else:
                    site_local = global_mean

                samples.append({
                    "site_local": site_local,
                    "global_mean": global_mean,
                    "comp_log": gf["composition_log"],
                    "core_type": gf["core_type"],
                    "antennae": float(gf["antennae"]),
                    "glycan_id": row["glytoucan_ac"],
                })
        return samples

    train_samples = _build(train_prots)
    val_samples = _build(val_prots)
    test_samples = _build(test_prots)
    logger.info("Samples: %d train, %d val, %d test",
                len(train_samples), len(val_samples), len(test_samples))

    # 7. Train
    model = GlycanCompositionPredictor().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
    batch_size = 512
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(80):
        model.train()
        indices = np.random.permutation(len(train_samples))
        total_loss = 0.0
        n_batches = 0

        for bs in range(0, len(indices), batch_size):
            batch = [train_samples[i] for i in indices[bs:bs + batch_size]]
            sl = torch.stack([s["site_local"] for s in batch]).to(DEVICE)
            gm = torch.stack([s["global_mean"] for s in batch]).to(DEVICE)
            comp_target = torch.stack([s["comp_log"] for s in batch]).to(DEVICE)
            core_target = torch.tensor([s["core_type"] for s in batch],
                                       dtype=torch.long, device=DEVICE)
            ant_target = torch.tensor([s["antennae"] for s in batch],
                                      dtype=torch.float, device=DEVICE)

            comp_pred, core_pred, ant_pred = model(sl, gm)

            loss_comp = F.mse_loss(comp_pred, comp_target)
            loss_core = F.cross_entropy(core_pred, core_target)
            loss_ant = F.mse_loss(ant_pred.squeeze(), ant_target)

            loss = loss_comp + loss_core + 0.5 * loss_ant

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            val_m = _evaluate(model, val_samples)
            logger.info("Epoch %d/80  loss=%.4f  val: comp_cos=%.4f  core_acc=%.4f  ant_mae=%.4f",
                        epoch + 1, total_loss / n_batches,
                        val_m["comp_cosine"], val_m["core_accuracy"], val_m["ant_mae"])
            val_loss = -val_m["comp_cosine"] + (1 - val_m["core_accuracy"])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    # 8. Test
    logger.info("\n" + "=" * 60)
    logger.info("TEST RESULTS")
    logger.info("=" * 60)
    test_m = _evaluate(model, test_samples, detailed=True)

    for k, v in test_m.items():
        if isinstance(v, float):
            logger.info("  %s: %.4f", k, v)
        elif isinstance(v, dict):
            logger.info("  %s:", k)
            for kk, vv in v.items():
                logger.info("    %s: %.4f", kk, vv)

    # 9. Baselines
    logger.info("\n" + "─" * 60)
    logger.info("BASELINES")
    logger.info("─" * 60)

    # Mean composition baseline
    train_comps = torch.stack([s["comp_log"] for s in train_samples])
    mean_comp = train_comps.mean(dim=0)

    # Majority core type baseline
    train_cores = [s["core_type"] for s in train_samples]
    majority_core = Counter(train_cores).most_common(1)[0][0]

    # Mean antennae baseline
    train_ants = [s["antennae"] for s in train_samples]
    mean_ant = np.mean(train_ants)

    # Evaluate baselines
    test_comps = torch.stack([s["comp_log"] for s in test_samples])
    test_cores = [s["core_type"] for s in test_samples]
    test_ants = [s["antennae"] for s in test_samples]

    # Composition cosine (mean prediction)
    cos = F.cosine_similarity(
        mean_comp.unsqueeze(0).expand(len(test_samples), -1),
        test_comps, dim=-1
    ).mean().item()
    core_acc = sum(1 for c in test_cores if c == majority_core) / len(test_cores)
    ant_mae = np.mean(np.abs(np.array(test_ants) - mean_ant))

    logger.info("  Mean prediction baseline:")
    logger.info("    comp_cosine=%.4f  core_acc=%.4f  ant_mae=%.4f",
                cos, core_acc, ant_mae)

    # Save
    output_dir = PROJECT / "experiments_v2/glycan_composition"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "model": "GlycanCompositionPredictor",
        "n_train": len(train_samples),
        "n_test": len(test_samples),
        "test_metrics": test_m,
        "baselines": {
            "mean_comp_cosine": cos,
            "majority_core_accuracy": core_acc,
            "mean_ant_mae": ant_mae,
        },
    }
    # Convert tensors/non-serializable to float
    def _to_serializable(obj):
        if isinstance(obj, (torch.Tensor, np.floating)):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_serializable(v) for v in obj]
        return obj

    with open(output_dir / "results.json", "w") as f:
        json.dump(_to_serializable(results), f, indent=2)
    torch.save(model.state_dict(), output_dir / "model.pt")
    logger.info("\nSaved to %s", output_dir)

    return test_m


def _evaluate(model, samples, batch_size=512, detailed=False):
    model.eval()
    all_comp_pred, all_comp_true = [], []
    all_core_pred, all_core_true = [], []
    all_ant_pred, all_ant_true = [], []

    with torch.no_grad():
        for bs in range(0, len(samples), batch_size):
            batch = samples[bs:bs + batch_size]
            sl = torch.stack([s["site_local"] for s in batch]).to(DEVICE)
            gm = torch.stack([s["global_mean"] for s in batch]).to(DEVICE)

            comp_pred, core_pred, ant_pred = model(sl, gm)

            all_comp_pred.append(comp_pred.cpu())
            all_comp_true.append(torch.stack([s["comp_log"] for s in batch]))
            all_core_pred.append(core_pred.argmax(dim=-1).cpu())
            all_core_true.extend([s["core_type"] for s in batch])
            all_ant_pred.append(ant_pred.squeeze().cpu())
            all_ant_true.extend([s["antennae"] for s in batch])

    comp_pred = torch.cat(all_comp_pred)
    comp_true = torch.cat(all_comp_true)
    core_pred = torch.cat(all_core_pred).numpy()
    core_true = np.array(all_core_true)
    ant_pred = torch.cat(all_ant_pred).numpy()
    ant_true = np.array(all_ant_true)

    # Composition: cosine similarity
    comp_cos = F.cosine_similarity(comp_pred, comp_true, dim=-1).mean().item()

    # Composition: per-monosaccharide MAE
    comp_mae = (comp_pred - comp_true).abs().mean(dim=0)

    # Core type accuracy
    core_acc = float((core_pred == core_true).mean())

    # Antennae MAE
    ant_mae = float(np.abs(ant_pred - ant_true).mean())

    metrics = {
        "comp_cosine": comp_cos,
        "comp_mse": float(F.mse_loss(comp_pred, comp_true)),
        "core_accuracy": core_acc,
        "ant_mae": ant_mae,
    }

    if detailed:
        # Per-monosaccharide MAE
        metrics["per_mono_mae"] = {
            MONO_NAMES[i]: float(comp_mae[i]) for i in range(len(MONO_NAMES))
        }

        # Per-core-type accuracy
        for ct_idx, ct_name in enumerate(CORE_TYPES):
            mask = core_true == ct_idx
            if mask.sum() > 0:
                metrics[f"core_{ct_name}_acc"] = float(
                    (core_pred[mask] == ct_idx).mean()
                )
                metrics[f"core_{ct_name}_n"] = int(mask.sum())

        # Top-1 and top-2 core accuracy
        # Recompute with raw logits for top-k
        all_core_logits = []
        with torch.no_grad():
            for bs in range(0, len(samples), batch_size):
                batch = samples[bs:bs + batch_size]
                sl = torch.stack([s["site_local"] for s in batch]).to(DEVICE)
                gm = torch.stack([s["global_mean"] for s in batch]).to(DEVICE)
                _, core_logits, _ = model(sl, gm)
                all_core_logits.append(core_logits.cpu())
        core_logits = torch.cat(all_core_logits)
        top2 = core_logits.topk(2, dim=-1).indices.numpy()
        core_top2_acc = float(np.mean([
            core_true[i] in top2[i] for i in range(len(core_true))
        ]))
        metrics["core_top2_accuracy"] = core_top2_acc

        # Composition R² per monosaccharide
        for i, name in enumerate(MONO_NAMES):
            pred_i = comp_pred[:, i].numpy()
            true_i = comp_true[:, i].numpy()
            ss_res = ((pred_i - true_i) ** 2).sum()
            ss_tot = ((true_i - true_i.mean()) ** 2).sum()
            r2 = 1 - ss_res / (ss_tot + 1e-8) if ss_tot > 1e-8 else 0.0
            metrics[f"r2_{name}"] = float(r2)

    return metrics


if __name__ == "__main__":
    train_and_evaluate()
