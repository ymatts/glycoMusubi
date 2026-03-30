#!/usr/bin/env python3
"""Glycan Retrieval v6b: N-linked focused site-level + function features.

Improvements over v6:
- N-linked only (cleaner signal, 2,359 candidates)
- Function-class matching from v3
- Glycan structural cluster evaluation
- Longer training (100 epochs)
- Type-stratified popularity baseline
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT = Path(__file__).resolve().parent.parent


class SiteGlycanRetrieverV6b(nn.Module):
    """Site-level retriever with function matching."""

    def __init__(self, esm2_dim=1280, glycan_dim=256, embed_dim=256,
                 hidden=768, n_glycans=0, n_func=8, dropout=0.2):
        super().__init__()
        self.embed_dim = embed_dim

        # Site encoder: site_local + global_mean
        self.site_encoder = nn.Sequential(
            nn.Linear(esm2_dim * 2, hidden),
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

        # Function matching (from v3)
        self.site_func_proj = nn.Linear(n_func, embed_dim // 4)
        self.glycan_func_proj = nn.Linear(n_func, embed_dim // 4)

        self.temp = nn.Parameter(torch.tensor(0.07))

    def encode_site(self, site_local, global_mean):
        x = torch.cat([site_local, global_mean], dim=-1)
        return F.normalize(self.site_encoder(x), dim=-1)

    def encode_glycan(self, g_feat):
        return F.normalize(self.glycan_proj(g_feat), dim=-1)

    def score_all(self, s_emb, all_g_emb, all_bias, s_func=None, all_g_func=None):
        """Score site against all glycans. [B, K]"""
        temp = self.temp.clamp(min=0.01)
        sim = s_emb @ all_g_emb.T / temp
        scores = sim + all_bias
        if s_func is not None and all_g_func is not None:
            sf = self.site_func_proj(s_func)    # [B, D/4]
            gf = self.glycan_func_proj(all_g_func)  # [K, D/4]
            func_score = sf @ gf.T              # [B, K]
            scores = scores + func_score
        return scores


def load_nlinked_data():
    """Load N-linked site-glycan data with function features."""
    site_df = pd.read_csv(PROJECT / "data_clean/glyconnect_site_glycans.tsv", sep="\t")
    site_df["glyc_type_clean"] = site_df["glycosylation_type"].apply(
        lambda x: x.split(";")[0] if ";" in str(x) else str(x)
    )
    # N-linked only
    site_df = site_df[site_df["glyc_type_clean"] == "N-linked"].copy()
    logger.info("N-linked triples: %d", len(site_df))

    # KG glycan embeddings
    ckpt = torch.load(PROJECT / "experiments_v2/glycokgnet_inductive_r8/best.pt",
                      map_location="cpu", weights_only=True)
    ds = torch.load(PROJECT / "experiments_v2/glycokgnet_inductive_r8/dataset.pt",
                    map_location="cpu", weights_only=False)
    glycan_name_to_idx = ds["node_mappings"]["glycan"]
    state_key = "model_state_dict" if "model_state_dict" in ckpt else "model_state"
    glycan_emb_all = ckpt[state_key]["node_embeddings.glycan.weight"]

    # ESM2
    esm2_dir = PROJECT / "data_clean/esm2_cache"
    with open(esm2_dir / "id_to_idx.json") as f:
        esm2_id_to_idx = json.load(f)
    canonical_to_esm2 = {}
    for esm_id in esm2_id_to_idx:
        canonical = esm_id.split("-")[0]
        if canonical not in canonical_to_esm2 or esm_id.endswith("-1"):
            canonical_to_esm2[canonical] = esm_id

    per_residue_dir = PROJECT / "data_clean/esm2_perresidue"

    # Function labels
    func_df = pd.read_csv(PROJECT / "data_clean/glycan_function_labels.tsv", sep="\t")
    func_terms = sorted(func_df["function_term"].unique())
    func_term_to_idx = {t: i for i, t in enumerate(func_terms)}
    glycan_func = defaultdict(lambda: torch.zeros(len(func_terms)))
    for _, row in func_df.iterrows():
        g = row["glycan_id"]
        t = row["function_term"]
        if t in func_term_to_idx:
            glycan_func[g][func_term_to_idx[t]] = 1.0

    # Filter
    valid_prots = set(canonical_to_esm2.keys())
    valid_glycans = set(glycan_name_to_idx.keys())
    before = len(site_df)
    site_df = site_df[
        site_df["uniprot_ac"].isin(valid_prots) &
        site_df["glytoucan_ac"].isin(valid_glycans)
    ].copy()
    logger.info("After filtering: %d / %d", len(site_df), before)

    # Build glycan index
    used_glycans = sorted(site_df["glytoucan_ac"].unique())
    glycan_local = {g: i for i, g in enumerate(used_glycans)}
    glycan_feats = torch.stack([glycan_emb_all[glycan_name_to_idx[g]] for g in used_glycans])
    glycan_func_feats = torch.stack([glycan_func[g] for g in used_glycans])
    logger.info("N-linked glycans: %d", len(used_glycans))

    # Frequency
    freq = site_df["glytoucan_ac"].value_counts()
    freq_tensor = torch.zeros(len(used_glycans))
    for g, i in glycan_local.items():
        freq_tensor[i] = freq.get(g, 0)
    log_freq = torch.log1p(freq_tensor)
    log_freq = (log_freq - log_freq.mean()) / (log_freq.std() + 1e-8)

    return {
        "site_df": site_df,
        "glycan_local": glycan_local,
        "glycan_feats": glycan_feats,
        "glycan_func_feats": glycan_func_feats,
        "glycan_names": used_glycans,
        "glycan_emb_all": glycan_emb_all,
        "glycan_name_to_idx": glycan_name_to_idx,
        "log_freq": log_freq,
        "esm2_id_to_idx": esm2_id_to_idx,
        "canonical_to_esm2": canonical_to_esm2,
        "esm2_dir": PROJECT / "data_clean/esm2_cache",
        "per_residue_dir": per_residue_dir,
        "func_terms": func_terms,
        "glycan_func": glycan_func,
    }


def build_samples(data, split_proteins):
    """Build (site_local, global_mean, glycan_idx, protein_func) samples."""
    site_df = data["site_df"]
    glycan_local = data["glycan_local"]
    esm2_id_to_idx = data["esm2_id_to_idx"]
    canonical_to_esm2 = data["canonical_to_esm2"]
    esm2_dir = data["esm2_dir"]
    per_residue_dir = data["per_residue_dir"]
    glycan_func = data["glycan_func"]

    split_df = site_df[site_df["uniprot_ac"].isin(split_proteins)]
    samples = []
    n_fallback = 0

    for prot_id, grp in split_df.groupby("uniprot_ac"):
        esm2_id = canonical_to_esm2.get(prot_id)
        if esm2_id is None:
            continue
        esm_idx = esm2_id_to_idx[esm2_id]
        global_mean = torch.load(esm2_dir / f"{esm_idx}.pt", map_location="cpu", weights_only=True)
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

        # Compute protein-level function vector from its glycans (train only)
        prot_glycans = grp["glytoucan_ac"].unique()
        prot_func = torch.zeros(len(data["func_terms"]))
        for g in prot_glycans:
            prot_func = torch.max(prot_func, glycan_func[g])

        for _, row in grp.iterrows():
            pos = int(row["site_position"]) - 1
            g_idx = glycan_local[row["glytoucan_ac"]]

            if pr_data is not None and pr_data.dim() >= 2:
                L = pr_data.shape[0]
                p = min(max(pos, 0), L - 1)
                start = max(0, p - 16)
                end = min(L, p + 17)
                local = pr_data[start:end]
                center = p - start
                dists = torch.abs(torch.arange(local.shape[0], dtype=torch.float) - center)
                wts = torch.exp(-0.15 * dists)
                wts /= wts.sum()
                site_local = (local * wts.unsqueeze(1)).sum(0)
            else:
                site_local = global_mean
                n_fallback += 1

            samples.append((site_local, global_mean, g_idx, prot_func))

    if n_fallback:
        logger.info("  %d mean-pool fallbacks", n_fallback)
    return samples


def train_model(data, train_samples, val_samples, epochs=100, batch_size=512,
                lr=3e-4, n_neg=63, hard_neg_ratio=0.3):
    glycan_feats = data["glycan_feats"].to(DEVICE)
    glycan_func_feats = data["glycan_func_feats"].to(DEVICE)
    n_glycans = len(glycan_feats)

    model = SiteGlycanRetrieverV6b(
        n_glycans=n_glycans,
        n_func=len(data["func_terms"]),
        embed_dim=256,
        hidden=768,
    ).to(DEVICE)

    with torch.no_grad():
        model.glycan_bias.weight.copy_(data["log_freq"].unsqueeze(1))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    all_g_emb = None
    best_val_mrr = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                all_g_emb = model.encode_glycan(glycan_feats)
            model.train()

        indices = np.random.permutation(len(train_samples))
        total_loss = 0.0
        n_batches = 0

        for bs in range(0, len(indices), batch_size):
            batch = [train_samples[i] for i in indices[bs:bs + batch_size]]
            B = len(batch)

            site_locals = torch.stack([s[0] for s in batch]).to(DEVICE)
            global_means = torch.stack([s[1] for s in batch]).to(DEVICE)
            pos_g_idx = torch.tensor([s[2] for s in batch], dtype=torch.long, device=DEVICE)
            prot_func = torch.stack([s[3] for s in batch]).to(DEVICE)

            s_emb = model.encode_site(site_locals, global_means)

            # Negatives
            n_hard = int(n_neg * hard_neg_ratio)
            n_rand = n_neg - n_hard
            neg_idx = torch.randint(0, n_glycans, (B, n_rand), device=DEVICE)

            if all_g_emb is not None and n_hard > 0:
                with torch.no_grad():
                    all_bias = model.glycan_bias.weight.squeeze()
                    scores = model.score_all(s_emb, all_g_emb, all_bias,
                                             prot_func, glycan_func_feats)
                    for i in range(B):
                        scores[i, pos_g_idx[i]] = -1e9
                    _, hard = scores.topk(n_hard, dim=-1)
                neg_idx = torch.cat([neg_idx, hard], dim=1)

            # Positive
            pos_g_emb = model.encode_glycan(glycan_feats[pos_g_idx])
            temp = model.temp.clamp(min=0.01)
            pos_sim = (s_emb * pos_g_emb).sum(-1) / temp
            pos_bias = model.glycan_bias(pos_g_idx).squeeze(-1)
            pos_func = (model.site_func_proj(prot_func) *
                       model.glycan_func_proj(glycan_func_feats[pos_g_idx])).sum(-1)
            pos_scores = pos_sim + pos_bias + pos_func

            # Negative
            flat_neg = neg_idx.reshape(-1)
            neg_g_emb = model.encode_glycan(glycan_feats[flat_neg]).reshape(B, -1, 256)
            neg_sim = (s_emb.unsqueeze(1) * neg_g_emb).sum(-1) / temp
            neg_bias = model.glycan_bias(neg_idx).squeeze(-1)
            neg_func = (model.site_func_proj(prot_func).unsqueeze(1) *
                       model.glycan_func_proj(glycan_func_feats[flat_neg]).reshape(B, -1, 64)).sum(-1)
            neg_scores = neg_sim + neg_bias + neg_func

            logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
            loss = F.cross_entropy(logits, torch.zeros(B, dtype=torch.long, device=DEVICE))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            val_m = evaluate(model, val_samples, data)
            logger.info("Epoch %d/%d  loss=%.4f  val_MRR=%.4f  H@10=%.4f",
                        epoch + 1, epochs, total_loss / n_batches,
                        val_m["mrr"], val_m["hits@10"])
            if val_m["mrr"] > best_val_mrr:
                best_val_mrr = val_m["mrr"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    logger.info("Best val MRR: %.4f", best_val_mrr)
    return model


def evaluate(model, samples, data, batch_size=256):
    model.eval()
    glycan_feats = data["glycan_feats"].to(DEVICE)
    glycan_func_feats = data["glycan_func_feats"].to(DEVICE)

    with torch.no_grad():
        all_g_emb = model.encode_glycan(glycan_feats)
        all_bias = model.glycan_bias.weight.squeeze()

    ranks = []
    with torch.no_grad():
        for bs in range(0, len(samples), batch_size):
            batch = samples[bs:bs + batch_size]
            site_locals = torch.stack([s[0] for s in batch]).to(DEVICE)
            global_means = torch.stack([s[1] for s in batch]).to(DEVICE)
            pos_g_idx = torch.tensor([s[2] for s in batch], dtype=torch.long, device=DEVICE)
            prot_func = torch.stack([s[3] for s in batch]).to(DEVICE)

            s_emb = model.encode_site(site_locals, global_means)
            scores = model.score_all(s_emb, all_g_emb, all_bias, prot_func, glycan_func_feats)

            for i in range(len(batch)):
                true_score = scores[i, pos_g_idx[i]].item()
                rank = (scores[i] > true_score).sum().item() + 1
                ranks.append(rank)

    ranks = np.array(ranks, dtype=float)
    return {
        "mrr": float(np.mean(1.0 / ranks)),
        "hits@1": float(np.mean(ranks <= 1)),
        "hits@3": float(np.mean(ranks <= 3)),
        "hits@10": float(np.mean(ranks <= 10)),
        "hits@50": float(np.mean(ranks <= 50)),
        "mr": float(np.mean(ranks)),
        "n": len(ranks),
    }


def evaluate_cluster_level(model, samples, data, n_clusters=20):
    """Evaluate at glycan structural cluster level."""
    glycan_feats = data["glycan_feats"]

    # Cluster glycan embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(glycan_feats.numpy())

    # Get per-sample cluster assignment
    model.eval()
    glycan_feats_d = glycan_feats.to(DEVICE)
    glycan_func_feats = data["glycan_func_feats"].to(DEVICE)

    with torch.no_grad():
        all_g_emb = model.encode_glycan(glycan_feats_d)
        all_bias = model.glycan_bias.weight.squeeze()

    cluster_ranks = []
    with torch.no_grad():
        for bs in range(0, len(samples), 256):
            batch = samples[bs:bs + 256]
            site_locals = torch.stack([s[0] for s in batch]).to(DEVICE)
            global_means = torch.stack([s[1] for s in batch]).to(DEVICE)
            pos_g_idx = torch.tensor([s[2] for s in batch], dtype=torch.long, device=DEVICE)
            prot_func = torch.stack([s[3] for s in batch]).to(DEVICE)

            s_emb = model.encode_site(site_locals, global_means)
            scores = model.score_all(s_emb, all_g_emb, all_bias, prot_func, glycan_func_feats)

            # Aggregate scores to cluster level
            n_g = len(glycan_feats)
            cluster_scores_np = np.full((len(batch), n_clusters), -1e9)
            scores_np = scores.cpu().numpy()
            for c in range(n_clusters):
                mask = cluster_labels == c
                if mask.any():
                    cluster_scores_np[:, c] = scores_np[:, mask].max(axis=1)

            for i in range(len(batch)):
                true_cluster = cluster_labels[pos_g_idx[i].item()]
                true_score = cluster_scores_np[i, true_cluster]
                rank = (cluster_scores_np[i] > true_score).sum() + 1
                cluster_ranks.append(rank)

    cluster_ranks = np.array(cluster_ranks, dtype=float)
    return {
        "cluster_mrr": float(np.mean(1.0 / cluster_ranks)),
        "cluster_h@1": float(np.mean(cluster_ranks <= 1)),
        "cluster_h@3": float(np.mean(cluster_ranks <= 3)),
        "cluster_h@5": float(np.mean(cluster_ranks <= 5)),
        "cluster_h@10": float(np.mean(cluster_ranks <= 10)),
        "n_clusters": n_clusters,
        "n": len(cluster_ranks),
    }


def evaluate_popularity_baseline(samples, data):
    """N-linked popularity baseline."""
    train_prots = data.get("train_proteins", set())
    train_df = data["site_df"][data["site_df"]["uniprot_ac"].isin(train_prots)]
    freq = train_df["glytoucan_ac"].value_counts()

    freq_scores = torch.zeros(len(data["glycan_local"]))
    for g, i in data["glycan_local"].items():
        freq_scores[i] = freq.get(g, 0)

    ranks = []
    for s in samples:
        true_idx = s[2]
        rank = (freq_scores > freq_scores[true_idx]).sum().item() + 1
        ranks.append(rank)

    ranks = np.array(ranks, dtype=float)
    return {
        "mrr": float(np.mean(1.0 / ranks)),
        "hits@1": float(np.mean(ranks <= 1)),
        "hits@10": float(np.mean(ranks <= 10)),
        "hits@50": float(np.mean(ranks <= 50)),
        "n": len(ranks),
    }


def main():
    logger.info("=" * 70)
    logger.info("  Glycan Retrieval v6b: N-linked Site-Level")
    logger.info("=" * 70)

    data = load_nlinked_data()
    site_df = data["site_df"]

    # Split
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

    for name, pset in [("train", train_proteins), ("val", val_proteins), ("test", test_proteins)]:
        n_t = len(site_df[site_df["uniprot_ac"].isin(pset)])
        logger.info("%s: %d proteins, %d triples", name, len(pset), n_t)

    logger.info("Building samples...")
    train_samples = build_samples(data, train_proteins)
    val_samples = build_samples(data, val_proteins)
    test_samples = build_samples(data, test_proteins)
    logger.info("Samples: %d train / %d val / %d test",
                len(train_samples), len(val_samples), len(test_samples))

    # Train
    model = train_model(data, train_samples, val_samples, epochs=100)

    # Test
    logger.info("\n" + "=" * 60)
    logger.info("TEST RESULTS (N-linked only)")
    logger.info("=" * 60)

    test_m = evaluate(model, test_samples, data)
    logger.info("Site-level: MRR=%.4f  H@1=%.4f  H@3=%.4f  H@10=%.4f  H@50=%.4f  MR=%.1f",
                test_m["mrr"], test_m["hits@1"], test_m["hits@3"],
                test_m["hits@10"], test_m["hits@50"], test_m["mr"])

    # Cluster
    logger.info("\nCluster-level (K=20):")
    cluster_m = evaluate_cluster_level(model, test_samples, data, n_clusters=20)
    logger.info("  MRR=%.4f  H@1=%.4f  H@3=%.4f  H@5=%.4f  H@10=%.4f",
                cluster_m["cluster_mrr"], cluster_m["cluster_h@1"],
                cluster_m["cluster_h@3"], cluster_m["cluster_h@5"],
                cluster_m["cluster_h@10"])

    # Popularity baseline
    pop_m = evaluate_popularity_baseline(test_samples, data)
    logger.info("\nPopularity baseline: MRR=%.4f  H@1=%.4f  H@10=%.4f",
                pop_m["mrr"], pop_m["hits@1"], pop_m["hits@10"])

    # Save
    output_dir = PROJECT / "experiments_v2/glycan_retrieval_v6b"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "model": "SiteGlycanRetrieverV6b (N-linked)",
        "n_train": len(train_samples),
        "n_test": len(test_samples),
        "n_candidates": len(data["glycan_local"]),
        "test": test_m,
        "cluster_k20": cluster_m,
        "popularity": pop_m,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    torch.save(model.state_dict(), output_dir / "model.pt")

    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON (N-linked)")
    logger.info("=" * 60)
    logger.info("  v3 protein-level (func-restricted): MRR=0.118  H@10=0.228  (3,345 cands)")
    logger.info("  v6  site-level:                     MRR=0.075  H@10=0.182  (2,359 cands)")
    logger.info("  v6b site+func (N-linked only):      MRR=%.3f  H@10=%.3f  (%d cands)",
                test_m["mrr"], test_m["hits@10"], len(data["glycan_local"]))
    logger.info("  Popularity (N-linked):              MRR=%.3f  H@10=%.3f",
                pop_m["mrr"], pop_m["hits@10"])
    logger.info("\n  Cluster-level:")
    logger.info("  v3 protein-level: H@5=0.526, H@10=0.919")
    logger.info("  v6b site-level:   H@5=%.3f, H@10=%.3f",
                cluster_m["cluster_h@5"], cluster_m["cluster_h@10"])


if __name__ == "__main__":
    main()
