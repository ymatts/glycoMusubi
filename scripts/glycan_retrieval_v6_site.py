#!/usr/bin/env python3
"""Glycan Retrieval v6: Site-Level Prediction.

Predicts which glycan attaches at a specific site on a protein,
using per-residue ESM2 features at the site position.

Key advance over v3 (protein-level):
- Training on 109K (protein, site) → glycan triples (vs ~16K protein→glycan)
- Per-residue ESM2 local window features at each site
- Site context (residue type, glycosylation type) as additional features
- Can distinguish different glycans at different sites on the same protein

Usage:
    python scripts/glycan_retrieval_v6_site.py
"""

from __future__ import annotations

import json
import logging
import sys
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


# ── Model ──────────────────────────────────────────────────────────────────


class SiteGlycanRetriever(nn.Module):
    """Two-tower model: site encoder × glycan encoder."""

    def __init__(
        self,
        esm2_dim: int = 1280,
        glycan_dim: int = 256,
        embed_dim: int = 256,
        hidden: int = 768,
        n_glycans: int = 0,
        n_glyc_types: int = 4,  # N-linked, O-linked, C-linked, unknown
        window_size: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size

        # Site encoder: local window ESM2 features + global context
        # Input: site_local (esm2_dim) + global_mean (esm2_dim) + glyc_type_emb (32)
        site_input_dim = esm2_dim * 2 + 32

        self.glyc_type_emb = nn.Embedding(n_glyc_types, 32)

        self.site_encoder = nn.Sequential(
            nn.Linear(site_input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
        )

        # Glycan encoder
        self.glycan_proj = nn.Sequential(
            nn.Linear(glycan_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
        )

        # Popularity bias
        self.glycan_bias = nn.Embedding(n_glycans, 1)

        # Temperature
        self.temp = nn.Parameter(torch.tensor(0.07))

    def encode_site(self, site_local: torch.Tensor, global_mean: torch.Tensor,
                    glyc_type_idx: torch.Tensor) -> torch.Tensor:
        """Encode a glycosylation site.

        Args:
            site_local: [B, esm2_dim] - weighted average of local window
            global_mean: [B, esm2_dim] - protein mean-pooled ESM2
            glyc_type_idx: [B] - glycosylation type index
        """
        type_emb = self.glyc_type_emb(glyc_type_idx)  # [B, 32]
        x = torch.cat([site_local, global_mean, type_emb], dim=-1)
        emb = self.site_encoder(x)
        return F.normalize(emb, dim=-1)

    def encode_glycan(self, g_feat: torch.Tensor) -> torch.Tensor:
        emb = self.glycan_proj(g_feat)
        return F.normalize(emb, dim=-1)

    def score(self, s_emb: torch.Tensor, g_emb: torch.Tensor,
              g_idx: torch.Tensor) -> torch.Tensor:
        """Score site-glycan pairs."""
        temp = self.temp.clamp(min=0.01)
        sim = (s_emb * g_emb).sum(dim=-1) / temp
        bias = self.glycan_bias(g_idx).squeeze(-1)
        return sim + bias

    def score_all(self, s_emb: torch.Tensor, all_g_emb: torch.Tensor,
                  all_bias: torch.Tensor) -> torch.Tensor:
        """Score site against all glycans. [B, K]
        all_bias: [1, K] or [K]
        """
        temp = self.temp.clamp(min=0.01)
        sim = s_emb @ all_g_emb.T / temp  # [B, K]
        if all_bias.dim() == 1:
            all_bias = all_bias.unsqueeze(0)  # [1, K]
        return sim + all_bias  # [B, K]


# ── Feature extraction ─────────────────────────────────────────────────────


def extract_site_features(
    per_residue_path: Path,
    site_pos: int,
    window: int = 16,
) -> torch.Tensor | None:
    """Extract weighted local window from per-residue ESM2."""
    if not per_residue_path.exists():
        return None
    data = torch.load(per_residue_path, map_location="cpu", weights_only=True)
    if isinstance(data, dict):
        emb = data.get("embeddings", data.get("embedding"))
    else:
        emb = data
    if emb is None or emb.dim() < 2:
        return None

    L = emb.shape[0]
    pos = min(max(site_pos, 0), L - 1)
    start = max(0, pos - window)
    end = min(L, pos + window + 1)
    local = emb[start:end]  # [W, D]

    # Center-weighted average
    center_idx = pos - start
    distances = torch.abs(torch.arange(local.shape[0], dtype=torch.float) - center_idx)
    weights = torch.exp(-0.15 * distances)
    weights = weights / weights.sum()
    return (local * weights.unsqueeze(1)).sum(dim=0)  # [D]


GLYC_TYPE_MAP = {"N-linked": 0, "O-linked": 1, "C-linked": 2}


def load_data():
    """Load site-glycan triples and features."""
    # 1. Load site-glycan triples
    site_df = pd.read_csv(PROJECT / "data_clean/glyconnect_site_glycans.tsv", sep="\t")
    logger.info("Site-glycan triples: %d", len(site_df))

    # Normalize glyc type
    site_df["glyc_type_clean"] = site_df["glycosylation_type"].apply(
        lambda x: x.split(";")[0] if ";" in str(x) else str(x)
    )

    # 2. Load KG glycan embeddings
    ckpt_path = PROJECT / "experiments_v2/glycokgnet_inductive_r8/best.pt"
    ds_path = PROJECT / "experiments_v2/glycokgnet_inductive_r8/dataset.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    ds = torch.load(ds_path, map_location="cpu", weights_only=False)

    glycan_name_to_idx = ds["node_mappings"]["glycan"]
    state_key = "model_state_dict" if "model_state_dict" in ckpt else "model_state"
    glycan_emb_all = ckpt[state_key]["node_embeddings.glycan.weight"]
    logger.info("KG glycan embeddings: %s", glycan_emb_all.shape)

    # 3. Load ESM2 index (individual .pt files per protein)
    esm2_dir = PROJECT / "data_clean/esm2_cache"
    with open(esm2_dir / "id_to_idx.json") as f:
        esm2_id_to_idx = json.load(f)

    per_residue_dir = PROJECT / "data_clean/esm2_perresidue"

    # 4. Build canonical→ESM2 ID map (site data has canonical, ESM2 has isoform)
    canonical_to_esm2 = {}
    for esm_id in esm2_id_to_idx:
        canonical = esm_id.split("-")[0]
        # Prefer -1 isoform if multiple exist
        if canonical not in canonical_to_esm2 or esm_id.endswith("-1"):
            canonical_to_esm2[canonical] = esm_id

    valid_proteins = set(canonical_to_esm2.keys())
    valid_glycans = set(glycan_name_to_idx.keys())
    logger.info("ESM2 canonical proteins: %d, valid glycans: %d",
                len(valid_proteins), len(valid_glycans))

    before = len(site_df)
    site_df = site_df[
        site_df["uniprot_ac"].isin(valid_proteins)
        & site_df["glytoucan_ac"].isin(valid_glycans)
    ].copy()
    logger.info("After filtering: %d / %d triples (%.1f%%)",
                len(site_df), before, 100 * len(site_df) / before)

    # 5. Build glycan index (only glycans that appear in site data)
    used_glycans = sorted(site_df["glytoucan_ac"].unique())
    glycan_local = {g: i for i, g in enumerate(used_glycans)}
    glycan_feats = torch.stack([
        glycan_emb_all[glycan_name_to_idx[g]] for g in used_glycans
    ])
    logger.info("Candidate glycans: %d, features: %s", len(used_glycans), glycan_feats.shape)

    # 6. Compute glycan frequency for popularity bias init
    glycan_freq = site_df["glytoucan_ac"].value_counts()
    freq_tensor = torch.zeros(len(used_glycans))
    for g, i in glycan_local.items():
        freq_tensor[i] = glycan_freq.get(g, 0)
    log_freq = torch.log1p(freq_tensor)
    log_freq = (log_freq - log_freq.mean()) / (log_freq.std() + 1e-8)

    return {
        "site_df": site_df,
        "glycan_local": glycan_local,
        "glycan_feats": glycan_feats,
        "glycan_names": used_glycans,
        "log_freq": log_freq,
        "esm2_id_to_idx": esm2_id_to_idx,
        "canonical_to_esm2": canonical_to_esm2,
        "esm2_dir": esm2_dir,
        "per_residue_dir": per_residue_dir,
    }


def build_samples(data: dict, split_proteins: set) -> list:
    """Build (site_local, global_mean, glyc_type, glycan_idx) samples."""
    site_df = data["site_df"]
    glycan_local = data["glycan_local"]
    esm2_id_to_idx = data["esm2_id_to_idx"]
    canonical_to_esm2 = data["canonical_to_esm2"]
    esm2_dir = data["esm2_dir"]
    per_residue_dir = data["per_residue_dir"]

    split_df = site_df[site_df["uniprot_ac"].isin(split_proteins)]
    samples = []
    n_no_perresidue = 0

    # Group by protein for efficiency
    for prot_id, grp in split_df.groupby("uniprot_ac"):
        esm2_id = canonical_to_esm2.get(prot_id)
        if esm2_id is None:
            continue
        esm_idx = esm2_id_to_idx[esm2_id]
        # Load mean-pooled ESM2 from individual file
        global_mean = torch.load(esm2_dir / f"{esm_idx}.pt", map_location="cpu", weights_only=True)
        if isinstance(global_mean, dict):
            global_mean = global_mean.get("embedding", list(global_mean.values())[0])

        # Try to load per-residue
        pr_path = per_residue_dir / f"{prot_id}.pt"
        pr_data = None
        if not pr_path.exists():
            # Try with isoform suffix
            pr_path = per_residue_dir / f"{esm2_id}.pt"
        if not pr_path.exists():
            # Try canonical
            canonical = prot_id.split("-")[0]
            pr_path = per_residue_dir / f"{canonical}.pt"
        if pr_path.exists():
            pr_data = torch.load(pr_path, map_location="cpu", weights_only=True)
            if isinstance(pr_data, dict):
                pr_data = pr_data.get("embeddings", pr_data.get("embedding"))

        for _, row in grp.iterrows():
            pos = int(row["site_position"]) - 1  # 0-indexed
            glyc_type = GLYC_TYPE_MAP.get(row["glyc_type_clean"], 3)
            g_idx = glycan_local[row["glytoucan_ac"]]

            # Extract site-local features
            if pr_data is not None and pr_data.dim() >= 2:
                L = pr_data.shape[0]
                p = min(max(pos, 0), L - 1)
                start = max(0, p - 16)
                end = min(L, p + 17)
                local = pr_data[start:end]
                center = p - start
                dists = torch.abs(torch.arange(local.shape[0], dtype=torch.float) - center)
                wts = torch.exp(-0.15 * dists)
                wts = wts / wts.sum()
                site_local = (local * wts.unsqueeze(1)).sum(dim=0)
            else:
                site_local = global_mean  # fallback
                n_no_perresidue += 1

            samples.append((site_local, global_mean, glyc_type, g_idx))

    if n_no_perresidue > 0:
        logger.info("  %d samples used mean-pool fallback (no per-residue)", n_no_perresidue)
    return samples


# ── Training ───────────────────────────────────────────────────────────────


def train_model(data: dict, train_samples: list, val_samples: list,
                epochs: int = 60, batch_size: int = 512, lr: float = 3e-4,
                n_negatives: int = 63, hard_neg_ratio: float = 0.3):
    """Train site-level retrieval model."""
    glycan_feats = data["glycan_feats"].to(DEVICE)
    n_glycans = len(glycan_feats)

    model = SiteGlycanRetriever(
        n_glycans=n_glycans,
        embed_dim=256,
        hidden=768,
        dropout=0.2,
    ).to(DEVICE)

    # Init popularity bias
    with torch.no_grad():
        model.glycan_bias.weight.copy_(data["log_freq"].unsqueeze(1))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Pre-encode glycan embeddings
    all_g_emb = None
    hard_neg_scores = None

    best_val_mrr = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()

        # Refresh glycan embeddings and hard negatives every 10 epochs
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                all_g_emb = model.encode_glycan(glycan_feats)  # [K, D]
                all_bias = model.glycan_bias.weight.squeeze()  # [K]
            model.train()

        # Shuffle training samples
        indices = np.random.permutation(len(train_samples))
        total_loss = 0.0
        n_batches = 0

        for batch_start in range(0, len(indices), batch_size):
            batch_idx = indices[batch_start:batch_start + batch_size]
            batch = [train_samples[i] for i in batch_idx]

            site_locals = torch.stack([s[0] for s in batch]).to(DEVICE)
            global_means = torch.stack([s[1] for s in batch]).to(DEVICE)
            glyc_types = torch.tensor([s[2] for s in batch], dtype=torch.long, device=DEVICE)
            pos_g_idx = torch.tensor([s[3] for s in batch], dtype=torch.long, device=DEVICE)

            B = len(batch)

            # Encode site
            s_emb = model.encode_site(site_locals, global_means, glyc_types)  # [B, D]

            # Sample negatives (mix of random + hard)
            n_hard = int(n_negatives * hard_neg_ratio)
            n_rand = n_negatives - n_hard

            # Random negatives
            neg_idx = torch.randint(0, n_glycans, (B, n_rand), device=DEVICE)

            # Hard negatives from top scores
            if all_g_emb is not None and n_hard > 0:
                with torch.no_grad():
                    scores = model.score_all(s_emb, all_g_emb, all_bias.unsqueeze(0))  # [B, K]
                    # Mask out positive
                    for i in range(B):
                        scores[i, pos_g_idx[i]] = -1e9
                    _, hard_indices = scores.topk(n_hard, dim=-1)  # [B, n_hard]
                neg_idx = torch.cat([neg_idx, hard_indices], dim=1)  # [B, n_negatives]

            # Positive scores
            pos_g_feat = glycan_feats[pos_g_idx]  # [B, glycan_dim]
            pos_g_emb = model.encode_glycan(pos_g_feat)  # [B, D]
            pos_scores = model.score(s_emb, pos_g_emb, pos_g_idx)  # [B]

            # Negative scores
            neg_g_feat = glycan_feats[neg_idx.reshape(-1)]  # [B*N, glycan_dim]
            neg_g_emb = model.encode_glycan(neg_g_feat).reshape(B, -1, model.embed_dim)  # [B, N, D]
            neg_scores_sim = (s_emb.unsqueeze(1) * neg_g_emb).sum(-1) / model.temp.clamp(min=0.01)
            neg_bias = model.glycan_bias(neg_idx).squeeze(-1)  # [B, N]
            neg_scores = neg_scores_sim + neg_bias  # [B, N]

            # InfoNCE loss
            logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)  # [B, 1+N]
            labels = torch.zeros(B, dtype=torch.long, device=DEVICE)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_metrics = evaluate(model, val_samples, glycan_feats, data)
            val_mrr = val_metrics["mrr"]
            logger.info("Epoch %d/%d  loss=%.4f  val_MRR=%.4f  val_H@10=%.4f",
                        epoch + 1, epochs, avg_loss, val_mrr, val_metrics["hits@10"])
            if val_mrr > best_val_mrr:
                best_val_mrr = val_mrr
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif (epoch + 1) % 10 == 0:
            logger.info("Epoch %d/%d  loss=%.4f", epoch + 1, epochs, avg_loss)

    if best_state is not None:
        model.load_state_dict(best_state)
    logger.info("Best val MRR: %.4f", best_val_mrr)
    return model


# ── Evaluation ─────────────────────────────────────────────────────────────


def evaluate(model, samples, glycan_feats, data, batch_size=256):
    """Evaluate site-level retrieval."""
    model.eval()
    glycan_feats_d = glycan_feats.to(DEVICE)

    with torch.no_grad():
        all_g_emb = model.encode_glycan(glycan_feats_d)  # [K, D]
        all_bias = model.glycan_bias.weight.squeeze().unsqueeze(0)  # [1, K]

    ranks = []

    with torch.no_grad():
        for batch_start in range(0, len(samples), batch_size):
            batch = samples[batch_start:batch_start + batch_size]
            site_locals = torch.stack([s[0] for s in batch]).to(DEVICE)
            global_means = torch.stack([s[1] for s in batch]).to(DEVICE)
            glyc_types = torch.tensor([s[2] for s in batch], dtype=torch.long, device=DEVICE)
            pos_g_idx = torch.tensor([s[3] for s in batch], dtype=torch.long, device=DEVICE)

            s_emb = model.encode_site(site_locals, global_means, glyc_types)
            scores = model.score_all(s_emb, all_g_emb, all_bias)  # [B, K]

            for i in range(len(batch)):
                true_score = scores[i, pos_g_idx[i]].item()
                rank = (scores[i] > true_score).sum().item() + 1
                ranks.append(rank)

    ranks = np.array(ranks, dtype=float)
    metrics = {
        "mrr": float(np.mean(1.0 / ranks)),
        "hits@1": float(np.mean(ranks <= 1)),
        "hits@3": float(np.mean(ranks <= 3)),
        "hits@10": float(np.mean(ranks <= 10)),
        "hits@50": float(np.mean(ranks <= 50)),
        "mr": float(np.mean(ranks)),
        "n": len(ranks),
    }
    return metrics


def evaluate_per_type(model, samples, glycan_feats, data):
    """Evaluate separately for N-linked and O-linked."""
    results = {}
    for type_name, type_idx in [("N-linked", 0), ("O-linked", 1)]:
        type_samples = [s for s in samples if s[2] == type_idx]
        if len(type_samples) < 10:
            continue
        m = evaluate(model, type_samples, glycan_feats, data)
        m["n_samples"] = len(type_samples)
        results[type_name] = m
        logger.info("  %s (%d): MRR=%.4f  H@1=%.4f  H@10=%.4f  H@50=%.4f",
                     type_name, len(type_samples), m["mrr"], m["hits@1"],
                     m["hits@10"], m["hits@50"])
    return results


def evaluate_popularity_baseline(samples, data):
    """Popularity baseline: rank by training frequency."""
    glycan_freq = data["site_df"][
        data["site_df"]["uniprot_ac"].isin(data.get("train_proteins", set()))
    ]["glytoucan_ac"].value_counts()

    freq_scores = torch.zeros(len(data["glycan_local"]))
    for g, i in data["glycan_local"].items():
        freq_scores[i] = glycan_freq.get(g, 0)

    ranks = []
    for s in samples:
        true_idx = s[3]
        true_score = freq_scores[true_idx].item()
        rank = (freq_scores > true_score).sum().item() + 1
        ranks.append(rank)

    ranks = np.array(ranks, dtype=float)
    return {
        "mrr": float(np.mean(1.0 / ranks)),
        "hits@1": float(np.mean(ranks <= 1)),
        "hits@10": float(np.mean(ranks <= 10)),
        "hits@50": float(np.mean(ranks <= 50)),
        "n": len(ranks),
    }


# ── Comparison with v3 protein-level ───────────────────────────────────────


def compare_with_protein_level(model, test_samples, glycan_feats, data):
    """Aggregate site-level predictions to protein-level for v3 comparison."""
    model.eval()
    glycan_feats_d = glycan_feats.to(DEVICE)

    with torch.no_grad():
        all_g_emb = model.encode_glycan(glycan_feats_d)
        all_bias = model.glycan_bias.weight.squeeze().unsqueeze(0)

    # Group test samples by protein
    site_df = data["site_df"]
    test_prots = data["test_proteins"]
    test_df = site_df[site_df["uniprot_ac"].isin(test_prots)]

    # For each protein, get its unique glycans and best rank across sites
    protein_glycans = defaultdict(set)
    for _, row in test_df.iterrows():
        protein_glycans[row["uniprot_ac"]].add(row["glytoucan_ac"])

    # For protein-level MRR: for each (protein, glycan) pair,
    # take the best rank across all sites of that protein
    protein_ranks = []
    protein_samples_by_prot = defaultdict(list)
    for s in test_samples:
        # We need protein ID - reconstruct from sample data
        # Unfortunately we don't store protein_id in samples directly
        pass

    logger.info("  (protein-level comparison requires v3 co-evaluation, skipping)")


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    logger.info("=" * 70)
    logger.info("  Glycan Retrieval v6: Site-Level Prediction")
    logger.info("=" * 70)

    # Load data
    data = load_data()
    site_df = data["site_df"]

    # Protein-level split (80/10/10) — same seed as v3
    all_proteins = sorted(site_df["uniprot_ac"].unique())
    np.random.seed(42)
    np.random.shuffle(all_proteins)
    n = len(all_proteins)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train_proteins = set(all_proteins[:n_train])
    val_proteins = set(all_proteins[n_train:n_train + n_val])
    test_proteins = set(all_proteins[n_train + n_val:])

    data["train_proteins"] = train_proteins
    data["test_proteins"] = test_proteins

    logger.info("Split: %d train / %d val / %d test proteins",
                len(train_proteins), len(val_proteins), len(test_proteins))

    # Count triples per split
    for name, pset in [("train", train_proteins), ("val", val_proteins), ("test", test_proteins)]:
        n_triples = len(site_df[site_df["uniprot_ac"].isin(pset)])
        logger.info("  %s: %d triples", name, n_triples)

    # Build samples
    logger.info("Building training samples...")
    train_samples = build_samples(data, train_proteins)
    logger.info("  Train: %d samples", len(train_samples))

    logger.info("Building validation samples...")
    val_samples = build_samples(data, val_proteins)
    logger.info("  Val: %d samples", len(val_samples))

    logger.info("Building test samples...")
    test_samples = build_samples(data, test_proteins)
    logger.info("  Test: %d samples", len(test_samples))

    # Train
    logger.info("\n" + "─" * 60)
    logger.info("Training site-level retrieval model...")
    logger.info("─" * 60)
    model = train_model(data, train_samples, val_samples, epochs=60)

    # Evaluate
    logger.info("\n" + "─" * 60)
    logger.info("Test Evaluation")
    logger.info("─" * 60)

    test_metrics = evaluate(model, test_samples, data["glycan_feats"], data)
    logger.info("Overall: MRR=%.4f  H@1=%.4f  H@3=%.4f  H@10=%.4f  H@50=%.4f  MR=%.1f  (n=%d)",
                test_metrics["mrr"], test_metrics["hits@1"], test_metrics["hits@3"],
                test_metrics["hits@10"], test_metrics["hits@50"], test_metrics["mr"],
                test_metrics["n"])

    logger.info("\nPer glycosylation type:")
    per_type = evaluate_per_type(model, test_samples, data["glycan_feats"], data)

    # Popularity baseline
    logger.info("\nPopularity baseline:")
    pop_metrics = evaluate_popularity_baseline(test_samples, data)
    logger.info("  Pop: MRR=%.4f  H@1=%.4f  H@10=%.4f  H@50=%.4f",
                pop_metrics["mrr"], pop_metrics["hits@1"],
                pop_metrics["hits@10"], pop_metrics["hits@50"])

    # Save results
    output_dir = PROJECT / "experiments_v2/glycan_retrieval_v6"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "model": "SiteGlycanRetriever v6",
        "description": "Site-level glycan prediction with per-residue ESM2",
        "n_train_samples": len(train_samples),
        "n_test_samples": len(test_samples),
        "n_candidate_glycans": len(data["glycan_local"]),
        "test_metrics": test_metrics,
        "per_type": per_type,
        "popularity_baseline": pop_metrics,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    torch.save(model.state_dict(), output_dir / "model.pt")
    logger.info("\nSaved to %s", output_dir)

    # Summary comparison
    logger.info("\n" + "=" * 70)
    logger.info("  COMPARISON")
    logger.info("=" * 70)
    logger.info("  v3 protein-level:  MRR=0.118  H@10=0.228  (3,345 candidates)")
    logger.info("  v6 site-level:     MRR=%.3f  H@10=%.3f  (%d candidates)",
                test_metrics["mrr"], test_metrics["hits@10"],
                len(data["glycan_local"]))
    logger.info("  Popularity:        MRR=%.3f  H@10=%.3f",
                pop_metrics["mrr"], pop_metrics["hits@10"])

    return results


if __name__ == "__main__":
    main()
