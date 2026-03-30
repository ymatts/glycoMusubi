#!/usr/bin/env python3
"""Cell-type conditioned glycan prediction v2: Cascade model.

Cascade approach:
  P(glycan | protein, site, cell_type) = P(cluster | protein, site) x P(glycan | cluster, cell_type)

Stage 1 (existing v6b): Predict structural cluster (K=20, H@5=0.786)
Stage 2 (this script):  Rank glycans within cluster using cell-type enzyme expression

Evaluation modes:
  1. Within-cluster marginal:  avg enzyme score across all cell types -> rank within cluster
  2. Within-cluster oracle:    best cell type per sample -> rank within cluster
  3. Cascade marginal:         P(cluster|prot) x P(glycan|cluster, marginal) -> full ranking
  4. Cascade oracle:           P(cluster|prot) x P(glycan|cluster, best_ct) -> full ranking
  5. Learned model:            InfoNCE-trained scorer using enzyme vec + glycan embeddings
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
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


# ──────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────

def build_enzyme_expression_matrix(
    ts_path: Path,
    gene_list: list[str],
) -> tuple[np.ndarray, list[tuple[str, str]]]:
    """Build (n_celltypes, n_genes) expression matrix from Tabula Sapiens.

    Returns:
        matrix: (n_ct, n_genes) mean expression values
        ct_index: list of (cell_type, tissue) tuples
    """
    ts = pd.read_csv(ts_path, sep="\t")
    gene_to_col = {g: i for i, g in enumerate(gene_list)}

    # Build unique (cell_type, tissue) index
    ct_tissue = ts.groupby(["cell_type", "tissue_general"]).groups.keys()
    ct_index = sorted(ct_tissue)
    ct_to_row = {ct: i for i, ct in enumerate(ct_index)}

    matrix = np.zeros((len(ct_index), len(gene_list)), dtype=np.float32)
    for _, row in ts.iterrows():
        gene = row["gene_symbol"]
        if gene not in gene_to_col:
            continue
        ct_key = (row["cell_type"], row["tissue_general"])
        if ct_key not in ct_to_row:
            continue
        matrix[ct_to_row[ct_key], gene_to_col[gene]] = row["mean_expr"]

    logger.info("Enzyme expression matrix: %d cell-types x %d genes", *matrix.shape)
    logger.info("  Non-zero entries: %d / %d (%.1f%%)",
                (matrix > 0).sum(), matrix.size, 100 * (matrix > 0).sum() / matrix.size)
    return matrix, ct_index


def build_gene_to_glycans(edges_path: Path) -> dict[str, set[str]]:
    """Map gene_symbol -> set of glycan IDs from enzyme edges."""
    edges = pd.read_csv(edges_path, sep="\t")
    gene_to_glycans: dict[str, set[str]] = defaultdict(set)
    for _, row in edges.iterrows():
        gene = row.get("gene")
        if pd.notna(gene):
            gene_to_glycans[str(gene)].add(row["glycan_id"])
    logger.info("Gene->glycan mapping: %d genes -> %d unique glycans",
                len(gene_to_glycans),
                len(set().union(*gene_to_glycans.values()) if gene_to_glycans else set()))
    return dict(gene_to_glycans)


def build_within_cluster_enzyme_scores(
    enzyme_matrix: np.ndarray,       # (n_ct, n_genes)
    gene_list: list[str],
    gene_to_glycans: dict[str, set[str]],
    cluster_assignments: dict[str, int],  # glycan_id -> cluster_id
    glycan_local_idx: dict[str, int],     # glycan_id -> int within cluster
    n_clusters: int,
) -> dict[int, np.ndarray]:
    """Build per-cluster enzyme score matrices.

    Returns:
        cluster_id -> (n_ct, n_glycans_in_cluster) score matrix
    """
    # Map glycan -> set of gene indices that produce it
    glycan_to_gene_idxs: dict[str, list[int]] = defaultdict(list)
    for gi, gene in enumerate(gene_list):
        if gene in gene_to_glycans:
            for gly in gene_to_glycans[gene]:
                glycan_to_gene_idxs[gly].append(gi)

    # Group glycans by cluster
    cluster_glycans: dict[int, list[str]] = defaultdict(list)
    for gly_id, cid in cluster_assignments.items():
        cluster_glycans[cid].append(gly_id)

    n_ct = enzyme_matrix.shape[0]
    result: dict[int, np.ndarray] = {}

    for cid in range(n_clusters):
        gly_ids = sorted(cluster_glycans.get(cid, []))
        if not gly_ids:
            continue
        scores = np.zeros((n_ct, len(gly_ids)), dtype=np.float32)
        local_idx = {g: i for i, g in enumerate(gly_ids)}

        for gly_id in gly_ids:
            j = local_idx[gly_id]
            gene_idxs = glycan_to_gene_idxs.get(gly_id, [])
            if gene_idxs:
                # Sum expression of all enzymes that produce this glycan
                scores[:, j] = enzyme_matrix[:, gene_idxs].sum(axis=1)

        # Log-transform for smoother scoring
        scores = np.log1p(scores)
        result[cid] = scores

    n_nonempty = sum(1 for s in result.values() if (s > 0).any())
    logger.info("Within-cluster enzyme scores: %d/%d clusters have non-zero scores",
                n_nonempty, n_clusters)
    return result


# ──────────────────────────────────────────────────────────────────
# Learned model
# ──────────────────────────────────────────────────────────────────

class CellTypeGlycanScorer(nn.Module):
    """Score glycans within a cluster using enzyme expression."""

    def __init__(self, enzyme_dim: int = 339, glycan_dim: int = 256,
                 hidden: int = 256, n_glycans: int = 0):
        super().__init__()
        self.enzyme_encoder = nn.Sequential(
            nn.Linear(enzyme_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
        )
        self.glycan_proj = nn.Sequential(
            nn.Linear(glycan_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
        )
        self.glycan_bias = nn.Embedding(n_glycans, 1) if n_glycans > 0 else None
        self.temp = nn.Parameter(torch.tensor(0.07))

    def forward(self, enzyme_vec: torch.Tensor, glycan_embs: torch.Tensor,
                glycan_idx: torch.Tensor | None = None) -> torch.Tensor:
        """Score enzyme expression vector against glycan embeddings.

        Args:
            enzyme_vec: (B, enzyme_dim)
            glycan_embs: (K, glycan_dim) or (B, K, glycan_dim)
            glycan_idx: (K,) global glycan indices for bias lookup

        Returns:
            scores: (B, K)
        """
        e = F.normalize(self.enzyme_encoder(enzyme_vec), dim=-1)   # (B, H/2)
        if glycan_embs.dim() == 2:
            g = F.normalize(self.glycan_proj(glycan_embs), dim=-1)  # (K, H/2)
            scores = (e @ g.T) / self.temp.clamp(0.01)              # (B, K)
        else:
            g = F.normalize(self.glycan_proj(glycan_embs), dim=-1)  # (B, K, H/2)
            scores = (e.unsqueeze(1) * g).sum(-1) / self.temp.clamp(0.01)

        if self.glycan_bias is not None and glycan_idx is not None:
            scores = scores + self.glycan_bias(glycan_idx).squeeze(-1)
        return scores


# ──────────────────────────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────────────────────────

def compute_metrics(ranks: np.ndarray) -> dict:
    """Compute retrieval metrics from rank array."""
    return {
        "mrr": float(np.mean(1.0 / ranks)),
        "hits@1": float(np.mean(ranks <= 1)),
        "hits@3": float(np.mean(ranks <= 3)),
        "hits@5": float(np.mean(ranks <= 5)),
        "hits@10": float(np.mean(ranks <= 10)),
        "hits@50": float(np.mean(ranks <= 50)),
        "mr": float(np.mean(ranks)),
        "n": int(len(ranks)),
    }


def rank_within_cluster(scores_row: np.ndarray, true_local_idx: int) -> int:
    """Compute rank of true glycan within cluster scores."""
    true_score = scores_row[true_local_idx]
    return int((scores_row > true_score).sum()) + 1


# ──────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 70)
    logger.info("  Cell-Type Conditioned Glycan Prediction v2 (Cascade)")
    logger.info("=" * 70)

    # ── 1. Load N-linked site data ──
    site_df = pd.read_csv(PROJECT / "data_clean/glyconnect_site_glycans.tsv", sep="\t")
    site_df["glyc_type_clean"] = site_df["glycosylation_type"].apply(
        lambda x: x.split(";")[0] if ";" in str(x) else str(x)
    )
    site_df = site_df[site_df["glyc_type_clean"] == "N-linked"].copy()
    logger.info("N-linked triples: %d", len(site_df))

    # ── 2. Load KG glycan embeddings ──
    ckpt_dir = PROJECT / "experiments_v2/glycokgnet_inductive_r8"
    ckpt = torch.load(ckpt_dir / "best.pt", map_location="cpu", weights_only=True)
    ds = torch.load(ckpt_dir / "dataset.pt", map_location="cpu", weights_only=False)
    glycan_name_to_kg_idx = ds["node_mappings"]["glycan"]
    state_key = "model_state_dict" if "model_state_dict" in ckpt else "model_state"
    glycan_emb_all = ckpt[state_key]["node_embeddings.glycan.weight"]

    # ── 3. Load ESM2 data ──
    esm2_dir = PROJECT / "data_clean/esm2_cache"
    with open(esm2_dir / "id_to_idx.json") as f:
        esm2_id_to_idx = json.load(f)
    canonical_to_esm2 = {}
    for esm_id in esm2_id_to_idx:
        canonical = esm_id.split("-")[0]
        if canonical not in canonical_to_esm2 or esm_id.endswith("-1"):
            canonical_to_esm2[canonical] = esm_id
    per_residue_dir = PROJECT / "data_clean/esm2_perresidue"

    # ── 4. Filter to proteins with ESM2 and glycans in KG ──
    valid_prots = set(canonical_to_esm2.keys())
    valid_glycans = set(glycan_name_to_kg_idx.keys())
    before = len(site_df)
    site_df = site_df[
        site_df["uniprot_ac"].isin(valid_prots) &
        site_df["glytoucan_ac"].isin(valid_glycans)
    ].copy()
    logger.info("After filtering: %d / %d triples", len(site_df), before)

    # Build glycan index (same as v6b)
    used_glycans = sorted(site_df["glytoucan_ac"].unique())
    glycan_local = {g: i for i, g in enumerate(used_glycans)}
    glycan_feats = torch.stack([glycan_emb_all[glycan_name_to_kg_idx[g]] for g in used_glycans])
    n_glycans = len(used_glycans)
    logger.info("N-linked glycans: %d (embedding dim=%d)", n_glycans, glycan_feats.shape[1])

    # ── 5. Cluster glycan embeddings (K=20) ──
    N_CLUSTERS = 20
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(glycan_feats.numpy())
    cluster_assignments = {g: int(cluster_labels[i]) for g, i in glycan_local.items()}

    # Cluster stats
    cluster_sizes = np.bincount(cluster_labels, minlength=N_CLUSTERS)
    logger.info("Cluster sizes: min=%d, max=%d, mean=%.1f, median=%.1f",
                cluster_sizes.min(), cluster_sizes.max(),
                cluster_sizes.mean(), np.median(cluster_sizes))

    # Map within-cluster local indices
    cluster_local_idx: dict[int, dict[str, int]] = {}
    for cid in range(N_CLUSTERS):
        gly_in_cluster = sorted(g for g, c in cluster_assignments.items() if c == cid)
        cluster_local_idx[cid] = {g: i for i, g in enumerate(gly_in_cluster)}

    # ── 6. Load enzyme expression data ──
    gene_to_glycans = build_gene_to_glycans(PROJECT / "data_clean/edges_glycan_enzyme.tsv")

    # Get unique gene list from TS data
    ts_df = pd.read_csv(PROJECT / "data_clean/ts_celltype_enzyme_expression.tsv", sep="\t")
    gene_list = sorted(ts_df["gene_symbol"].unique())
    logger.info("TS genes: %d", len(gene_list))

    enzyme_matrix, ct_index = build_enzyme_expression_matrix(
        PROJECT / "data_clean/ts_celltype_enzyme_expression.tsv", gene_list
    )
    n_ct = len(ct_index)

    # ── 7. Build within-cluster enzyme scores ──
    cluster_enzyme_scores = build_within_cluster_enzyme_scores(
        enzyme_matrix, gene_list, gene_to_glycans,
        cluster_assignments, glycan_local, N_CLUSTERS,
    )

    # ── 8. Protein-level split (same seed as v6b) ──
    all_proteins = sorted(site_df["uniprot_ac"].unique())
    rng = np.random.RandomState(42)
    rng.shuffle(all_proteins)
    n = len(all_proteins)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train_proteins = set(all_proteins[:n_train])
    val_proteins = set(all_proteins[n_train:n_train + n_val])
    test_proteins = set(all_proteins[n_train + n_val:])

    train_df = site_df[site_df["uniprot_ac"].isin(train_proteins)]
    val_df = site_df[site_df["uniprot_ac"].isin(val_proteins)]
    test_df = site_df[site_df["uniprot_ac"].isin(test_proteins)]
    logger.info("Split: %d train / %d val / %d test proteins", len(train_proteins), len(val_proteins), len(test_proteins))
    logger.info("       %d train / %d val / %d test triples", len(train_df), len(val_df), len(test_df))

    # ── 9. Train frequency baseline ──
    train_freq = train_df["glytoucan_ac"].value_counts()
    pop_scores = np.zeros(n_glycans, dtype=np.float32)
    for g, i in glycan_local.items():
        pop_scores[i] = train_freq.get(g, 0)
    pop_scores_log = np.log1p(pop_scores)

    # ─────────────────────────────────────────────
    # EVALUATION
    # ─────────────────────────────────────────────

    results: dict = {
        "n_clusters": N_CLUSTERS,
        "n_glycans": n_glycans,
        "n_celltypes": n_ct,
        "cluster_sizes": {
            "min": int(cluster_sizes.min()),
            "max": int(cluster_sizes.max()),
            "mean": float(cluster_sizes.mean()),
        },
    }

    # ── Eval 1: Within-cluster marginal ──
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION 1: Within-cluster marginal (no cell-type knowledge)")
    logger.info("=" * 60)

    ranks_wc_marginal = []
    for _, row in test_df.iterrows():
        gly = row["glytoucan_ac"]
        cid = cluster_assignments.get(gly)
        if cid is None or cid not in cluster_enzyme_scores:
            continue
        local_map = cluster_local_idx[cid]
        if gly not in local_map:
            continue
        scores = cluster_enzyme_scores[cid]  # (n_ct, n_in_cluster)
        marginal = scores.mean(axis=0)       # avg over all cell types
        true_j = local_map[gly]
        ranks_wc_marginal.append(rank_within_cluster(marginal, true_j))

    ranks_wc_marginal = np.array(ranks_wc_marginal, dtype=float)
    m_wc_marginal = compute_metrics(ranks_wc_marginal)
    results["within_cluster_marginal"] = m_wc_marginal
    logger.info("  MRR=%.4f  H@1=%.4f  H@5=%.4f  H@10=%.4f  (n=%d)",
                m_wc_marginal["mrr"], m_wc_marginal["hits@1"],
                m_wc_marginal["hits@5"], m_wc_marginal["hits@10"], m_wc_marginal["n"])

    # ── Eval 2: Within-cluster oracle ──
    logger.info("\nEVALUATION 2: Within-cluster oracle (best cell type per sample)")

    ranks_wc_oracle = []
    for _, row in test_df.iterrows():
        gly = row["glytoucan_ac"]
        cid = cluster_assignments.get(gly)
        if cid is None or cid not in cluster_enzyme_scores:
            continue
        local_map = cluster_local_idx[cid]
        if gly not in local_map:
            continue
        scores = cluster_enzyme_scores[cid]  # (n_ct, n_in_cluster)
        true_j = local_map[gly]

        best_rank = scores.shape[1]  # worst case
        for ct_i in range(n_ct):
            ct_scores = scores[ct_i]
            if ct_scores[true_j] == 0:
                continue
            r = rank_within_cluster(ct_scores, true_j)
            best_rank = min(best_rank, r)
        ranks_wc_oracle.append(best_rank)

    ranks_wc_oracle = np.array(ranks_wc_oracle, dtype=float)
    m_wc_oracle = compute_metrics(ranks_wc_oracle)
    results["within_cluster_oracle"] = m_wc_oracle
    logger.info("  MRR=%.4f  H@1=%.4f  H@5=%.4f  H@10=%.4f  (n=%d)",
                m_wc_oracle["mrr"], m_wc_oracle["hits@1"],
                m_wc_oracle["hits@5"], m_wc_oracle["hits@10"], m_wc_oracle["n"])

    # ── Eval 3: Within-cluster popularity ──
    logger.info("\nEVALUATION 3: Within-cluster popularity")

    ranks_wc_pop = []
    for _, row in test_df.iterrows():
        gly = row["glytoucan_ac"]
        cid = cluster_assignments.get(gly)
        if cid is None:
            continue
        local_map = cluster_local_idx[cid]
        if gly not in local_map:
            continue
        true_j = local_map[gly]
        # Build within-cluster pop scores
        gly_in_cluster = sorted(local_map.keys(), key=lambda g: local_map[g])
        wc_pop = np.array([pop_scores_log[glycan_local[g]] for g in gly_in_cluster])
        ranks_wc_pop.append(rank_within_cluster(wc_pop, true_j))

    ranks_wc_pop = np.array(ranks_wc_pop, dtype=float)
    m_wc_pop = compute_metrics(ranks_wc_pop)
    results["within_cluster_popularity"] = m_wc_pop
    logger.info("  MRR=%.4f  H@1=%.4f  H@5=%.4f  H@10=%.4f  (n=%d)",
                m_wc_pop["mrr"], m_wc_pop["hits@1"],
                m_wc_pop["hits@5"], m_wc_pop["hits@10"], m_wc_pop["n"])

    # ── Eval 4: Within-cluster enzyme + popularity combination ──
    logger.info("\nEVALUATION 4: Within-cluster enzyme+popularity (grid search alpha)")

    best_alpha = 0.0
    best_mrr_alpha = 0.0
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        ranks_combo = []
        for _, row in val_df.iterrows():
            gly = row["glytoucan_ac"]
            cid = cluster_assignments.get(gly)
            if cid is None or cid not in cluster_enzyme_scores:
                continue
            local_map = cluster_local_idx[cid]
            if gly not in local_map:
                continue
            scores_enz = cluster_enzyme_scores[cid].mean(axis=0)
            gly_in_cluster = sorted(local_map.keys(), key=lambda g: local_map[g])
            wc_pop_arr = np.array([pop_scores_log[glycan_local[g]] for g in gly_in_cluster])

            # Normalize
            enz_norm = scores_enz / (scores_enz.max() + 1e-8)
            pop_norm = wc_pop_arr / (wc_pop_arr.max() + 1e-8)
            combined = alpha * enz_norm + (1 - alpha) * pop_norm

            true_j = local_map[gly]
            ranks_combo.append(rank_within_cluster(combined, true_j))

        if ranks_combo:
            mrr = np.mean(1.0 / np.array(ranks_combo))
            if mrr > best_mrr_alpha:
                best_mrr_alpha = mrr
                best_alpha = alpha

    logger.info("  Best alpha=%.1f (val MRR=%.4f)", best_alpha, best_mrr_alpha)

    # Evaluate on test with best alpha
    ranks_wc_combo = []
    for _, row in test_df.iterrows():
        gly = row["glytoucan_ac"]
        cid = cluster_assignments.get(gly)
        if cid is None or cid not in cluster_enzyme_scores:
            continue
        local_map = cluster_local_idx[cid]
        if gly not in local_map:
            continue
        scores_enz = cluster_enzyme_scores[cid].mean(axis=0)
        gly_in_cluster = sorted(local_map.keys(), key=lambda g: local_map[g])
        wc_pop_arr = np.array([pop_scores_log[glycan_local[g]] for g in gly_in_cluster])
        enz_norm = scores_enz / (scores_enz.max() + 1e-8)
        pop_norm = wc_pop_arr / (wc_pop_arr.max() + 1e-8)
        combined = best_alpha * enz_norm + (1 - best_alpha) * pop_norm
        true_j = local_map[gly]
        ranks_wc_combo.append(rank_within_cluster(combined, true_j))

    ranks_wc_combo = np.array(ranks_wc_combo, dtype=float)
    m_wc_combo = compute_metrics(ranks_wc_combo)
    m_wc_combo["best_alpha"] = best_alpha
    results["within_cluster_enzyme_pop"] = m_wc_combo
    logger.info("  Test MRR=%.4f  H@1=%.4f  H@5=%.4f  H@10=%.4f",
                m_wc_combo["mrr"], m_wc_combo["hits@1"],
                m_wc_combo["hits@5"], m_wc_combo["hits@10"])

    # ── Eval 5: Cascade (cluster prediction x within-cluster) ──
    # Load v6b model for cluster predictions
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION 5-6: Cascade (cluster x within-cluster)")
    logger.info("=" * 60)

    v6b_results_path = PROJECT / "experiments_v2/glycan_retrieval_v6b/results.json"
    if v6b_results_path.exists():
        with open(v6b_results_path) as f:
            v6b_results = json.load(f)
        cluster_h5 = v6b_results.get("cluster_k20", {}).get("cluster_h@5", 0.786)
        cluster_h10 = v6b_results.get("cluster_k20", {}).get("cluster_h@10", 0.978)
        logger.info("v6b cluster accuracy: H@5=%.3f, H@10=%.3f", cluster_h5, cluster_h10)
    else:
        cluster_h5 = 0.786
        cluster_h10 = 0.978
        logger.info("Using reference v6b cluster accuracy: H@5=%.3f, H@10=%.3f", cluster_h5, cluster_h10)

    # Cascade = assume correct cluster (H@1 from v6b) x within-cluster score
    # For rigorous evaluation, compute cascade with cluster correctness probability
    cluster_h1 = v6b_results.get("cluster_k20", {}).get("cluster_h@1", 0.40) if v6b_results_path.exists() else 0.40

    # Cascade marginal: P_correct_cluster * MRR_within_cluster
    cascade_marginal_mrr = cluster_h1 * m_wc_marginal["mrr"]
    cascade_pop_mrr = cluster_h1 * m_wc_pop["mrr"]
    cascade_oracle_mrr = cluster_h1 * m_wc_oracle["mrr"]
    cascade_combo_mrr = cluster_h1 * m_wc_combo["mrr"]

    results["cascade_estimates"] = {
        "cluster_h1": cluster_h1,
        "cascade_marginal_mrr": float(cascade_marginal_mrr),
        "cascade_popularity_mrr": float(cascade_pop_mrr),
        "cascade_oracle_mrr": float(cascade_oracle_mrr),
        "cascade_enzyme_pop_mrr": float(cascade_combo_mrr),
        "note": "Lower bound: cascade_MRR >= cluster_H@1 * within_cluster_MRR",
    }
    logger.info("Cascade estimates (cluster_H@1=%.3f):", cluster_h1)
    logger.info("  Cascade marginal MRR >= %.4f", cascade_marginal_mrr)
    logger.info("  Cascade popularity MRR >= %.4f", cascade_pop_mrr)
    logger.info("  Cascade enzyme+pop MRR >= %.4f", cascade_combo_mrr)
    logger.info("  Cascade oracle MRR >= %.4f", cascade_oracle_mrr)

    # ── Eval 7: Full-ranking (all N glycan candidates) ──
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION 7: Full-ranking baselines (all %d candidates)", n_glycans)
    logger.info("=" * 60)

    # Full-ranking popularity
    ranks_full_pop = []
    for _, row in test_df.iterrows():
        gly = row["glytoucan_ac"]
        if gly not in glycan_local:
            continue
        gi = glycan_local[gly]
        rank = int((pop_scores_log > pop_scores_log[gi]).sum()) + 1
        ranks_full_pop.append(rank)
    ranks_full_pop = np.array(ranks_full_pop, dtype=float)
    m_full_pop = compute_metrics(ranks_full_pop)
    results["full_ranking_popularity"] = m_full_pop
    logger.info("  Popularity: MRR=%.4f  H@10=%.4f", m_full_pop["mrr"], m_full_pop["hits@10"])

    # Full-ranking enzyme marginal
    full_enz_scores = np.zeros(n_glycans, dtype=np.float32)
    for gly, gi in glycan_local.items():
        for gene, glycan_set in gene_to_glycans.items():
            if gly in glycan_set:
                gene_idx_in_list = next((j for j, g in enumerate(gene_list) if g == gene), None)
                if gene_idx_in_list is not None:
                    full_enz_scores[gi] += enzyme_matrix[:, gene_idx_in_list].mean()
    full_enz_scores = np.log1p(full_enz_scores)

    ranks_full_enz = []
    for _, row in test_df.iterrows():
        gly = row["glytoucan_ac"]
        if gly not in glycan_local:
            continue
        gi = glycan_local[gly]
        rank = int((full_enz_scores > full_enz_scores[gi]).sum()) + 1
        ranks_full_enz.append(rank)
    ranks_full_enz = np.array(ranks_full_enz, dtype=float)
    m_full_enz = compute_metrics(ranks_full_enz)
    results["full_ranking_enzyme_marginal"] = m_full_enz
    logger.info("  Enzyme marginal: MRR=%.4f  H@10=%.4f", m_full_enz["mrr"], m_full_enz["hits@10"])

    # ── Eval 8: Learned CellTypeGlycanScorer ──
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION 8: Learned CellTypeGlycanScorer")
    logger.info("=" * 60)

    learned_results = train_and_evaluate_learned_model(
        site_df=site_df,
        glycan_feats=glycan_feats,
        glycan_local=glycan_local,
        cluster_assignments=cluster_assignments,
        cluster_local_idx=cluster_local_idx,
        enzyme_matrix=enzyme_matrix,
        gene_list=gene_list,
        gene_to_glycans=gene_to_glycans,
        pop_scores_log=pop_scores_log,
        train_proteins=train_proteins,
        val_proteins=val_proteins,
        test_proteins=test_proteins,
        n_clusters=N_CLUSTERS,
    )
    results["learned_model"] = learned_results

    # ── Summary ──
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info("Within-cluster evaluation (avg cluster size=%.0f):", cluster_sizes.mean())
    logger.info("  %-35s MRR=%.4f  H@5=%.4f  H@10=%.4f",
                "Popularity:", m_wc_pop["mrr"], m_wc_pop["hits@5"], m_wc_pop["hits@10"])
    logger.info("  %-35s MRR=%.4f  H@5=%.4f  H@10=%.4f",
                "Enzyme marginal:", m_wc_marginal["mrr"], m_wc_marginal["hits@5"], m_wc_marginal["hits@10"])
    logger.info("  %-35s MRR=%.4f  H@5=%.4f  H@10=%.4f",
                f"Enzyme+pop (alpha={best_alpha}):", m_wc_combo["mrr"], m_wc_combo["hits@5"], m_wc_combo["hits@10"])
    logger.info("  %-35s MRR=%.4f  H@5=%.4f  H@10=%.4f",
                "Oracle (best cell type):", m_wc_oracle["mrr"], m_wc_oracle["hits@5"], m_wc_oracle["hits@10"])
    if "within_cluster" in learned_results:
        lr = learned_results["within_cluster"]
        logger.info("  %-35s MRR=%.4f  H@5=%.4f  H@10=%.4f",
                    "Learned model:", lr["mrr"], lr["hits@5"], lr["hits@10"])

    logger.info("\nFull-ranking (%d candidates):", n_glycans)
    logger.info("  %-35s MRR=%.4f  H@10=%.4f", "Popularity:", m_full_pop["mrr"], m_full_pop["hits@10"])
    logger.info("  %-35s MRR=%.4f  H@10=%.4f", "Enzyme marginal:", m_full_enz["mrr"], m_full_enz["hits@10"])
    logger.info("  %-35s MRR=0.066  H@10=0.145", "v6b site-level (ref):")
    logger.info("  %-35s MRR=0.118  H@10=0.228", "v3 protein-level (ref):")

    # Save results
    output_dir = PROJECT / "experiments_v2/celltype_conditioned_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("\nResults saved to %s", output_dir / "results.json")

    return results


def train_and_evaluate_learned_model(
    site_df: pd.DataFrame,
    glycan_feats: torch.Tensor,
    glycan_local: dict[str, int],
    cluster_assignments: dict[str, int],
    cluster_local_idx: dict[int, dict[str, int]],
    enzyme_matrix: np.ndarray,
    gene_list: list[str],
    gene_to_glycans: dict[str, set[str]],
    pop_scores_log: np.ndarray,
    train_proteins: set[str],
    val_proteins: set[str],
    test_proteins: set[str],
    n_clusters: int,
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 512,
) -> dict:
    """Train and evaluate the learned CellTypeGlycanScorer.

    Uses cluster-grouped batching for efficiency: pre-groups samples
    by cluster to avoid the per-sample inner loop.
    """

    n_genes = len(gene_list)
    n_glycans = len(glycan_local)
    n_ct = enzyme_matrix.shape[0]
    glycan_dim = glycan_feats.shape[1]
    idx_to_glycan = {v: k for k, v in glycan_local.items()}

    # Build training samples as (glycan_global_idx, cluster_id, true_local_idx)
    def build_samples(proteins: set[str]) -> list[tuple[int, int, int]]:
        df = site_df[site_df["uniprot_ac"].isin(proteins)]
        samples = []
        for _, row in df.iterrows():
            gly = row["glytoucan_ac"]
            if gly not in glycan_local or gly not in cluster_assignments:
                continue
            gi = glycan_local[gly]
            cid = cluster_assignments[gly]
            local_map = cluster_local_idx.get(cid)
            if local_map is None or len(local_map) < 2 or gly not in local_map:
                continue
            samples.append((gi, cid, local_map[gly]))
        return samples

    train_samples = build_samples(train_proteins)
    val_samples = build_samples(val_proteins)
    test_samples = build_samples(test_proteins)
    logger.info("Learned model samples: %d train / %d val / %d test",
                len(train_samples), len(val_samples), len(test_samples))

    if not train_samples:
        logger.warning("No training samples for learned model!")
        return {"error": "no training samples"}

    # Pre-compute per-cluster glycan embeddings and indices (cached on device)
    cluster_glycan_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for cid, local_map in cluster_local_idx.items():
        if len(local_map) < 2:
            continue
        gly_in_cluster = sorted(local_map.keys(), key=lambda g: local_map[g])
        global_idxs = torch.tensor(
            [glycan_local[g] for g in gly_in_cluster],
            dtype=torch.long, device=DEVICE,
        )
        cluster_glycan_cache[cid] = (global_idxs, glycan_feats.to(DEVICE)[global_idxs])

    model = CellTypeGlycanScorer(
        enzyme_dim=n_genes,
        glycan_dim=glycan_dim,
        hidden=256,
        n_glycans=n_glycans,
    ).to(DEVICE)

    # Init bias with popularity
    if model.glycan_bias is not None:
        with torch.no_grad():
            norm_pop = torch.from_numpy(pop_scores_log).float()
            norm_pop = (norm_pop - norm_pop.mean()) / (norm_pop.std() + 1e-8)
            model.glycan_bias.weight.copy_(norm_pop.unsqueeze(1))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    glycan_feats_d = glycan_feats.to(DEVICE)
    best_val_mrr = 0.0
    best_state = None

    # Group train samples by cluster for efficient batching
    from collections import defaultdict as _dd
    cluster_to_samples: dict[int, list[int]] = _dd(list)
    for si, (gi, cid, tli) in enumerate(train_samples):
        cluster_to_samples[cid].append(si)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        # Shuffle cluster order and process per-cluster batches
        cluster_ids_shuffled = list(cluster_to_samples.keys())
        np.random.shuffle(cluster_ids_shuffled)

        for cid in cluster_ids_shuffled:
            if cid not in cluster_glycan_cache:
                continue
            sample_idxs = cluster_to_samples[cid]
            np.random.shuffle(sample_idxs)

            global_idxs, cluster_embs = cluster_glycan_cache[cid]
            K_c = len(global_idxs)

            for bs in range(0, len(sample_idxs), batch_size):
                batch_si = sample_idxs[bs:bs + batch_size]
                B = len(batch_si)

                # Random cell-type enzyme vectors
                ct_indices = np.random.randint(0, n_ct, size=B)
                enzyme_vecs = torch.from_numpy(
                    enzyme_matrix[ct_indices]
                ).float().to(DEVICE)  # (B, n_genes)

                targets = torch.tensor(
                    [train_samples[si][2] for si in batch_si],
                    dtype=torch.long, device=DEVICE,
                )  # true local indices within cluster

                # Score all glycans in this cluster
                scores = model(enzyme_vecs, cluster_embs, global_idxs)  # (B, K_c)
                loss = F.cross_entropy(scores, targets)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

        scheduler.step()

        # Evaluate every 2 epochs or at start/end
        if epoch == 0 or (epoch + 1) % 2 == 0 or epoch == epochs - 1:
            val_m = evaluate_learned_within_cluster(
                model, val_samples, glycan_feats_d, glycan_local,
                cluster_local_idx, cluster_glycan_cache, enzyme_matrix,
            )
            logger.info("  Epoch %d/%d  loss=%.4f  val_MRR=%.4f  H@5=%.4f",
                        epoch + 1, epochs,
                        total_loss / max(n_batches, 1),
                        val_m["mrr"], val_m["hits@5"])
            if val_m["mrr"] > best_val_mrr:
                best_val_mrr = val_m["mrr"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    # Test evaluation
    test_m = evaluate_learned_within_cluster(
        model, test_samples, glycan_feats_d, glycan_local,
        cluster_local_idx, cluster_glycan_cache, enzyme_matrix,
    )
    logger.info("Learned model test: MRR=%.4f  H@1=%.4f  H@5=%.4f  H@10=%.4f",
                test_m["mrr"], test_m["hits@1"], test_m["hits@5"], test_m["hits@10"])

    return {"within_cluster": test_m, "best_val_mrr": float(best_val_mrr)}


def evaluate_learned_within_cluster(
    model: CellTypeGlycanScorer,
    samples: list[tuple[int, int, int]],
    glycan_feats_d: torch.Tensor,
    glycan_local: dict[str, int],
    cluster_local_idx: dict[int, dict[str, int]],
    cluster_glycan_cache: dict[int, tuple[torch.Tensor, torch.Tensor]],
    enzyme_matrix: np.ndarray | None = None,
) -> dict:
    """Evaluate learned model on within-cluster ranking.

    Uses marginal enzyme vector (mean over all cell types) for scoring.
    Batches samples by cluster for efficiency.
    """
    model.eval()

    # Group by cluster
    from collections import defaultdict as _dd
    cluster_to_samples: dict[int, list[tuple[int, int, int]]] = _dd(list)
    for s in samples:
        cluster_to_samples[s[1]].append(s)

    ranks = []
    with torch.no_grad():
        for cid, csamples in cluster_to_samples.items():
            if cid not in cluster_glycan_cache:
                continue
            global_idxs, cluster_embs = cluster_glycan_cache[cid]

            # Use zero enzyme vec (marginal); the model learns from bias + glycan proj
            B = len(csamples)
            # Marginal: random enzyme vecs not needed for eval, use zeros
            enzyme_vecs = torch.zeros(B, model.enzyme_encoder[0].in_features, device=DEVICE)
            if enzyme_matrix is not None:
                marginal = torch.from_numpy(enzyme_matrix.mean(axis=0)).float().to(DEVICE)
                enzyme_vecs = marginal.unsqueeze(0).expand(B, -1)

            scores = model(enzyme_vecs, cluster_embs, global_idxs)  # (B, K_c)
            scores_np = scores.cpu().numpy()

            for i, (gi, _, true_local) in enumerate(csamples):
                ranks.append(rank_within_cluster(scores_np[i], true_local))

    if not ranks:
        return {"mrr": 0.0, "hits@1": 0.0, "hits@5": 0.0, "hits@10": 0.0, "n": 0}
    return compute_metrics(np.array(ranks, dtype=float))


if __name__ == "__main__":
    main()
