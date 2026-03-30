#!/usr/bin/env python3
"""Pathway-based glycan prediction: glycoPathAI × glycoMusubi × Tabula Sapiens.

Integrates:
1. glycoPathAI: pathway inference (ILP + GFlowNet) using reaction triplets
2. glycoMusubi: glycan structural classification → GlyTouCan ID mapping
3. Tabula Sapiens: cell-type-specific enzyme expression

Pipeline:
  cell_type → TS enzyme expression
           → glycoPathAI pathway scoring (reachable structures from Man5)
           → structural class → GlyTouCan ID candidates
           → score × popularity → ranked glycan predictions

Evaluation: MRR/H@10 on N-linked site-glycan triples (same test set as v3/v6b).
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parent.parent


# ──────────────────────────────────────────────
# 1. Data loading
# ──────────────────────────────────────────────

def load_reaction_triplets() -> list[tuple[str, str, str]]:
    df = pd.read_csv(PROJECT / "data_clean/reaction_triplets.tsv", sep="\t")
    return list(zip(df["enzyme_gene"], df["reactant"], df["product"]))


def load_ts_expression() -> pd.DataFrame:
    return pd.read_csv(
        PROJECT / "data_clean/ts_celltype_enzyme_expression.tsv", sep="\t"
    )


def load_site_glycan_data() -> pd.DataFrame:
    df = pd.read_csv(PROJECT / "data_clean/glyconnect_site_glycans.tsv", sep="\t")
    return df[df["glycosylation_type"] == "N-linked"].copy()


def classify_nlinked_glycans() -> dict[str, str]:
    """Classify all N-linked glycans by structural type using WURCS."""
    structs = pd.read_csv(PROJECT / "data_raw/glytoucan_structures.tsv", sep="\t")
    func_labels = pd.read_csv(
        PROJECT / "data_clean/glycan_function_labels.tsv", sep="\t"
    )
    nlinked_ids = set(
        func_labels[func_labels["function_term"] == "N-linked"]["glycan_id"]
    )

    classifications = {}
    for _, row in structs.iterrows():
        gid = row["glycan_id"]
        if gid not in nlinked_ids:
            continue
        wurcs = str(row["structure"])
        if not wurcs.startswith("WURCS="):
            classifications[gid] = "unknown"
            continue

        m = re.match(r"WURCS=2\.0/(\d+),(\d+),(\d+)/", wurcs)
        if not m:
            classifications[gid] = "unknown"
            continue
        n_total = int(m.group(2))

        has_fuc = "a1221m" in wurcs or "1221m" in wurcs
        has_sia = "AUx" in wurcs or "Aun" in wurcs or "a2112h-1x" in wurcs
        n_hexnac = wurcs.count("NCC/3=O")

        if n_total <= 5 and n_hexnac <= 2 and not has_fuc and not has_sia:
            classifications[gid] = "high_mannose"
        elif 5 < n_total <= 11 and n_hexnac <= 2 and not has_fuc and not has_sia:
            classifications[gid] = "high_mannose"
        elif n_hexnac == 3 and not has_sia and not has_fuc:
            classifications[gid] = "hybrid"
        elif n_hexnac > 2 and not has_sia and not has_fuc:
            classifications[gid] = "complex_agalacto"
        elif has_fuc and has_sia:
            classifications[gid] = "complex_fucosialylated"
        elif has_fuc and not has_sia:
            classifications[gid] = "complex_fucosylated"
        elif has_sia and not has_fuc:
            classifications[gid] = "complex_sialylated"
        else:
            classifications[gid] = "other"

    return classifications


# ──────────────────────────────────────────────
# 2. Pathway reachability scoring
# ──────────────────────────────────────────────

# Map reaction DB structures → structural classes
STRUCTURE_TO_CLASS = {
    "Man9GlcNAc2": "high_mannose",
    "Man8GlcNAc2": "high_mannose",
    "Man7GlcNAc2": "high_mannose",
    "Man6GlcNAc2": "high_mannose",
    "Man5GlcNAc2": "high_mannose",
    "GlcNAcMan5GlcNAc2": "hybrid",
    "GlcNAcMan3GlcNAc2": "hybrid",
    "GalGlcNAcMan5GlcNAc2": "hybrid",
    "SiaGalGlcNAcMan5GlcNAc2": "hybrid",
    "GlcNAc2Man3GlcNAc2": "complex_agalacto",
    "BisGlcNAc2Man3GlcNAc2": "complex_agalacto",
    "GlcNAc3Man3GlcNAc2": "complex_agalacto",
    "GlcNAc4Man3GlcNAc2": "complex_agalacto",
    "Gal1GlcNAc2Man3GlcNAc2": "complex_galactosylated",
    "Gal2GlcNAc2Man3GlcNAc2": "complex_galactosylated",
    "Gal3GlcNAc3Man3GlcNAc2": "complex_galactosylated",
    "Sia1Gal2GlcNAc2Man3GlcNAc2": "complex_sialylated",
    "Sia2Gal2GlcNAc2Man3GlcNAc2": "complex_sialylated",
    "a23Sia1Gal2GlcNAc2Man3GlcNAc2": "complex_sialylated",
    "Sia1Gal3GlcNAc3Man3GlcNAc2": "complex_sialylated",
    "FucGlcNAc2Man3GlcNAc2": "complex_fucosylated",
    "FucGal2GlcNAc2Man3GlcNAc2": "complex_fucosialylated",
    "FucSia2Gal2GlcNAc2Man3GlcNAc2": "complex_fucosialylated",
    "LeXGal2GlcNAc2Man3GlcNAc2": "complex_fucosylated",
    "HGal2GlcNAc2Man3GlcNAc2": "complex_fucosylated",
}


def compute_pathway_structure_scores(
    triplets: list[tuple[str, str, str]],
    enzyme_expression: dict[str, float],
    root: str = "Man5GlcNAc2",
) -> dict[str, float]:
    """Compute per-structure pathway scores using expression-weighted BFS.

    Score = mean expression of enzymes along the best path from root.
    Returns {structure_name: score}.
    """
    adj: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for gene, substrate, product in triplets:
        adj[substrate].append((gene, product))

    # BFS: track (cumulative_expression, n_steps) for mean calculation
    structure_info: dict[str, tuple[float, int]] = {root: (0.0, 0)}
    structure_scores: dict[str, float] = {root: 1.0}
    queue = [root]
    visited = {root}

    while queue:
        current = queue.pop(0)
        cum_expr, n_steps = structure_info[current]

        for gene, product in adj.get(current, []):
            expr = enzyme_expression.get(gene, 0.0)
            new_cum = cum_expr + expr
            new_steps = n_steps + 1
            new_score = new_cum / new_steps  # Mean expression along path

            if product not in visited or new_score > structure_scores.get(product, 0):
                structure_scores[product] = new_score
                structure_info[product] = (new_cum, new_steps)
                if product not in visited:
                    visited.add(product)
                    queue.append(product)

    return structure_scores


# ──────────────────────────────────────────────
# 3. glycoPathAI pipeline integration
# ──────────────────────────────────────────────

def run_glycopathai_for_celltype(
    triplets: list[tuple[str, str, str]],
    enzyme_expression: dict[str, float],
    root: str = "Man5GlcNAc2",
) -> dict[str, float]:
    """Run glycoPathAI pipeline with cell-type-specific enzyme expression.

    Returns: {structure_name: pathway_score}
    """
    try:
        from glycopath.triplet import TripletDB
        from glycopath.constraints import BiosyntheticConstraints
        from glycopath.expression import ExpressionData
        from glycopath.scorer import PathwayScorer

        db = TripletDB(triplets)

        # Find reachable glycans via BFS
        reachable = set()
        queue = [root]
        reachable.add(root)
        while queue:
            node = queue.pop(0)
            for rxn in db.applicable(node):
                if rxn.product not in reachable:
                    reachable.add(rxn.product)
                    queue.append(rxn.product)

        # Score each reachable structure using PathwayScorer
        expr_data = ExpressionData(enzyme_expression)
        scorer = PathwayScorer(expression_data=expr_data)

        # For each reachable product, find a path from root and score it
        # Use BFS to find all paths
        from glycopath.gfn_env import BiosynState

        structure_scores = {}
        # Simple BFS path finding
        parent_map: dict[str, tuple[str, str, str]] = {}  # product → (gene, substrate, product)
        bfs_queue = [root]
        bfs_visited = {root}
        while bfs_queue:
            node = bfs_queue.pop(0)
            for rxn in db.applicable(node):
                if rxn.product not in bfs_visited:
                    bfs_visited.add(rxn.product)
                    parent_map[rxn.product] = (rxn.enzyme, rxn.reactant, rxn.product)
                    bfs_queue.append(rxn.product)

        for target in reachable:
            if target == root:
                structure_scores[target] = 1.0
                continue
            # Reconstruct path
            path = []
            current = target
            while current in parent_map:
                step = parent_map[current]
                path.append(step)
                current = step[1]  # substrate
            path.reverse()

            if not path:
                continue

            state = BiosynState(glycan=target, path=tuple(path))
            scored = scorer.score(state)
            structure_scores[target] = scored.total_score

        return structure_scores

    except ImportError:
        logger.warning("glycoPathAI not installed, using BFS reachability only")
        return {}


# ──────────────────────────────────────────────
# 4. Integrated prediction
# ──────────────────────────────────────────────

# ──────────────────────────────────────────────
# Composition-based structure → GlyTouCan mapping
# ──────────────────────────────────────────────

# Known compositions for pathway structures (Hex, HexNAc, Fuc, NeuAc)
PATHWAY_COMPOSITION = {
    "Man5GlcNAc2":                    (5, 2, 0, 0),
    "Man6GlcNAc2":                    (6, 2, 0, 0),
    "Man7GlcNAc2":                    (7, 2, 0, 0),
    "Man8GlcNAc2":                    (8, 2, 0, 0),
    "Man9GlcNAc2":                    (9, 2, 0, 0),
    "GlcNAcMan5GlcNAc2":             (5, 3, 0, 0),
    "GlcNAcMan3GlcNAc2":             (3, 3, 0, 0),
    "GlcNAc2Man3GlcNAc2":            (3, 4, 0, 0),
    "BisGlcNAc2Man3GlcNAc2":         (3, 5, 0, 0),
    "GlcNAc3Man3GlcNAc2":            (3, 5, 0, 0),
    "GlcNAc4Man3GlcNAc2":            (3, 6, 0, 0),
    "Gal1GlcNAc2Man3GlcNAc2":        (4, 4, 0, 0),
    "Gal2GlcNAc2Man3GlcNAc2":        (5, 4, 0, 0),
    "Gal3GlcNAc3Man3GlcNAc2":        (6, 5, 0, 0),
    "Sia1Gal2GlcNAc2Man3GlcNAc2":    (5, 4, 0, 1),
    "Sia2Gal2GlcNAc2Man3GlcNAc2":    (5, 4, 0, 2),
    "a23Sia1Gal2GlcNAc2Man3GlcNAc2": (5, 4, 0, 1),
    "FucGlcNAc2Man3GlcNAc2":         (3, 4, 1, 0),
    "FucGal2GlcNAc2Man3GlcNAc2":     (5, 4, 1, 0),
    "FucSia2Gal2GlcNAc2Man3GlcNAc2": (5, 4, 1, 2),
    "LeXGal2GlcNAc2Man3GlcNAc2":     (5, 4, 1, 0),
    "HGal2GlcNAc2Man3GlcNAc2":       (5, 4, 1, 0),
    "GalGlcNAcMan5GlcNAc2":          (6, 3, 0, 0),
    "SiaGalGlcNAcMan5GlcNAc2":       (6, 3, 0, 1),
}


def parse_wurcs_composition(wurcs: str) -> tuple[int, int, int, int] | None:
    """Parse WURCS to extract (Hex, HexNAc, Fuc, NeuAc) counts."""
    if not isinstance(wurcs, str) or not wurcs.startswith("WURCS="):
        return None
    m = re.match(r"WURCS=2\.0/(\d+),(\d+),(\d+)/", wurcs)
    if not m:
        return None

    # Count monosaccharide types from residue list
    parts = wurcs.split("/")
    if len(parts) < 3:
        return None
    residues = parts[2].strip("[]").split("][")

    hex_count = 0
    hexnac_count = 0
    fuc_count = 0
    neuac_count = 0

    # Count in connectivity section (4th part)
    if len(parts) >= 4:
        conn = parts[3]  # e.g., "1-2-3-3-2-3"
        residue_indices = conn.split("-")
        for idx_str in residue_indices:
            try:
                idx = int(idx_str) - 1
            except ValueError:
                continue
            if idx < len(residues):
                res = residues[idx]
                if "NCC/3=O" in res:  # HexNAc
                    hexnac_count += 1
                elif "a1221m" in res or "1221m" in res:  # Fucose
                    fuc_count += 1
                elif "AUx" in res or "Aun" in res or "a2112h-1x" in res:  # NeuAc
                    neuac_count += 1
                elif "1122h" in res:  # Hex (Man/Gal/Glc)
                    hex_count += 1
                elif "2122h" in res and "NCC" not in res:  # Hex variant
                    hex_count += 1

    return (hex_count, hexnac_count, fuc_count, neuac_count)


def build_glycan_composition_map(
    glycan_ids: list[str], structs_df: pd.DataFrame
) -> dict[str, tuple[int, int, int, int]]:
    """Build GlyTouCan ID → composition map."""
    wurcs_map = {}
    for _, row in structs_df.iterrows():
        wurcs_map[row["glycan_id"]] = row["structure"]

    result = {}
    for gid in glycan_ids:
        wurcs = wurcs_map.get(gid)
        if wurcs:
            comp = parse_wurcs_composition(wurcs)
            if comp:
                result[gid] = comp
    return result


def composition_similarity(c1: tuple, c2: tuple) -> float:
    """Cosine similarity between two composition vectors."""
    a = np.array(c1, dtype=float)
    b = np.array(c2, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def build_glycan_scores(
    structure_scores: dict[str, float],
    glycan_compositions: dict[str, tuple],
    popularity: dict[str, float],
    alpha: float = 0.5,
) -> dict[str, float]:
    """Score each GlyTouCan glycan by composition similarity to pathway structures.

    For each candidate glycan:
      pathway_score = max over all pathway structures of
          (composition_similarity × structure_pathway_score)
    final = α × pathway_score + (1-α) × popularity
    """
    scores = {}
    for gid, comp in glycan_compositions.items():
        best_pathway = 0.0
        for struct_name, struct_score in structure_scores.items():
            struct_comp = PATHWAY_COMPOSITION.get(struct_name)
            if struct_comp is None:
                continue
            sim = composition_similarity(comp, struct_comp)
            best_pathway = max(best_pathway, sim * struct_score)

        pop_score = popularity.get(gid, 0.0)
        scores[gid] = alpha * best_pathway + (1 - alpha) * pop_score
    return scores


def evaluate(
    site_glycans: pd.DataFrame,
    glycan_compositions: dict[str, tuple],
    triplets: list[tuple[str, str, str]],
    ts_expression: pd.DataFrame,
    popularity: dict[str, float],
    all_glycans: list[str],
):
    """Evaluate pathway-based prediction on N-linked test set."""
    # Build cell-type → gene → expression
    ct_gene_expr: dict[str, dict[str, float]] = defaultdict(dict)
    for _, row in ts_expression.iterrows():
        ct = row["cell_type"]
        gene = row["gene_symbol"]
        expr = row["mean_expr"]
        if expr > 0:
            ct_gene_expr[ct][gene] = expr

    # Normalize expression per cell type (max = 1)
    for ct in ct_gene_expr:
        max_expr = max(ct_gene_expr[ct].values()) if ct_gene_expr[ct] else 1.0
        if max_expr > 0:
            ct_gene_expr[ct] = {g: v / max_expr for g, v in ct_gene_expr[ct].items()}

    logger.info("Cell types with expression data: %d", len(ct_gene_expr))

    # Pre-compute structure scores for each cell type
    ct_struct_scores: dict[str, dict[str, float]] = {}
    for ct, expr in ct_gene_expr.items():
        ct_struct_scores[ct] = compute_pathway_structure_scores(triplets, expr)

    # Marginalized structure scores (average over all cell types)
    all_structs = set()
    for scores in ct_struct_scores.values():
        all_structs.update(scores.keys())

    marginalized_scores = {}
    for struct in all_structs:
        vals = [ct_struct_scores[ct].get(struct, 0) for ct in ct_struct_scores]
        marginalized_scores[struct] = np.mean(vals)

    logger.info("Marginalized structure scores (top 15):")
    for struct, score in sorted(marginalized_scores.items(), key=lambda x: -x[1])[:15]:
        logger.info("  %s: %.4f", struct, score)

    # Prepare test data
    # Group by (protein, site) → glycan set
    test_groups = defaultdict(set)
    for _, row in site_glycans.iterrows():
        key = (row.get("uniprot_ac", ""), row.get("site_position", 0))
        test_groups[key].add(row["glytoucan_ac"])

    logger.info("Test protein-site groups: %d", len(test_groups))

    # Evaluate different configurations
    glycan_idx = {g: i for i, g in enumerate(all_glycans)}
    n_glycans = len(all_glycans)

    configs = {
        "popularity_only": {"alpha": 0.0, "struct_scores": marginalized_scores},
        "pathway_only": {"alpha": 1.0, "struct_scores": marginalized_scores},
        "pathway+pop_0.3": {"alpha": 0.3, "struct_scores": marginalized_scores},
        "pathway+pop_0.5": {"alpha": 0.5, "struct_scores": marginalized_scores},
        "pathway+pop_0.7": {"alpha": 0.7, "struct_scores": marginalized_scores},
    }

    # Add per-cell-type oracle
    oracle_results = {"mrr": [], "h10": []}

    for config_name, config in configs.items():
        alpha = config["alpha"]
        struct_scores = config["struct_scores"]

        # Build glycan scores
        glycan_scores = build_glycan_scores(
            struct_scores, glycan_compositions, popularity, alpha
        )

        # Score vector
        score_vec = np.array([glycan_scores.get(g, 0.0) for g in all_glycans])

        mrrs = []
        h10s = []

        for (prot, site), true_glycans in test_groups.items():
            # Rank all glycans
            # Add small random tiebreaker
            scores = score_vec + np.random.uniform(0, 1e-10, n_glycans)
            ranking = np.argsort(-scores)

            # Find best rank of any true glycan
            best_rank = n_glycans
            for true_g in true_glycans:
                if true_g in glycan_idx:
                    idx = glycan_idx[true_g]
                    rank = np.where(ranking == idx)[0][0] + 1
                    best_rank = min(best_rank, rank)

            if best_rank <= n_glycans:
                mrrs.append(1.0 / best_rank)
                h10s.append(1.0 if best_rank <= 10 else 0.0)

        mrr = np.mean(mrrs) if mrrs else 0.0
        h10 = np.mean(h10s) if h10s else 0.0
        logger.info("  %-25s MRR=%.4f  H@10=%.4f", config_name, mrr, h10)

    # Oracle: sample 500 test cases × top-20 cell types for speed
    logger.info("\nOracle cell-type evaluation (sampled)...")
    sample_keys = list(test_groups.keys())
    np.random.seed(42)
    if len(sample_keys) > 500:
        sample_keys = [sample_keys[i] for i in np.random.choice(len(sample_keys), 500, replace=False)]

    # Pre-rank cell types by total enzyme expression breadth
    ct_breadth = {ct: len(expr) for ct, expr in ct_gene_expr.items()}
    top_cts = sorted(ct_breadth, key=ct_breadth.get, reverse=True)[:20]

    oracle_mrrs = []
    oracle_h10s = []

    for prot_site in sample_keys:
        true_glycans = test_groups[prot_site]
        best_mrr = 0.0
        best_h10 = 0.0

        for ct in top_cts:
            cs = ct_struct_scores.get(ct, {})
            glycan_scores = build_glycan_scores(cs, glycan_compositions, popularity, alpha=0.5)
            scores = np.array([glycan_scores.get(g, 0.0) for g in all_glycans])
            scores += np.random.uniform(0, 1e-10, n_glycans)
            ranking = np.argsort(-scores)

            for true_g in true_glycans:
                if true_g in glycan_idx:
                    idx = glycan_idx[true_g]
                    rank = np.where(ranking == idx)[0][0] + 1
                    this_mrr = 1.0 / rank
                    if this_mrr > best_mrr:
                        best_mrr = this_mrr
                        best_h10 = 1.0 if rank <= 10 else 0.0

        oracle_mrrs.append(best_mrr)
        oracle_h10s.append(best_h10)

    oracle_mrr = np.mean(oracle_mrrs)
    oracle_h10 = np.mean(oracle_h10s)
    logger.info("  Oracle (best CT, α=0.5): MRR=%.4f  H@10=%.4f (500 samples × 20 CTs)", oracle_mrr, oracle_h10)

    # glycoPathAI full pipeline (if available)
    logger.info("\nglycPathAI full pipeline evaluation...")
    try:
        # Run for a few cell types and show pathway scores
        sample_cts = sorted(ct_gene_expr.keys())[:3]
        for ct in sample_cts:
            scores = run_glycopathai_for_celltype(triplets, ct_gene_expr[ct])
            if scores:
                logger.info("  Cell type: %s — reachable: %d structures", ct, len(scores))
                for struct, score in sorted(scores.items(), key=lambda x: -x[1])[:5]:
                    cls = STRUCTURE_TO_CLASS.get(struct, "?")
                    logger.info("    %s (%s): %.4f", struct, cls, score)
    except Exception as e:
        logger.warning("  glycoPathAI pipeline failed: %s", e)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY (N-linked, %d candidates)", n_glycans)
    logger.info("=" * 60)
    for config_name, config in configs.items():
        alpha = config["alpha"]
        class_scores = config["class_scores"]
        glycan_scores = build_glycan_scores(
            class_scores, glycan_classes, popularity, alpha
        )
        score_vec = np.array([glycan_scores.get(g, 0.0) for g in all_glycans])
        mrrs = []
        h10s = []
        for (prot, site), true_glycans in test_groups.items():
            scores = score_vec + np.random.uniform(0, 1e-10, n_glycans)
            ranking = np.argsort(-scores)
            best_rank = n_glycans
            for true_g in true_glycans:
                if true_g in glycan_idx:
                    idx = glycan_idx[true_g]
                    rank = np.where(ranking == idx)[0][0] + 1
                    best_rank = min(best_rank, rank)
            if best_rank <= n_glycans:
                mrrs.append(1.0 / best_rank)
                h10s.append(1.0 if best_rank <= 10 else 0.0)
        mrr = np.mean(mrrs) if mrrs else 0
        h10 = np.mean(h10s) if h10s else 0
        logger.info("  %-25s MRR=%.4f  H@10=%.4f (α=%.1f)", config_name, mrr, h10, alpha)
    logger.info("  %-25s MRR=%.4f  H@10=%.4f", "Oracle (best CT)", oracle_mrr, oracle_h10)
    logger.info("  ────────────────────────────────────")
    logger.info("  %-25s MRR=0.1180  H@10=0.2280  (reference)", "v3 protein-level")
    logger.info("  %-25s MRR=0.0660  H@10=0.1450  (reference)", "v6b site-level")


def main():
    logger.info("=" * 70)
    logger.info("  Pathway-Based Glycan Predictor: glycoPathAI × glycoMusubi × TS")
    logger.info("=" * 70)

    # Load data
    triplets = load_reaction_triplets()
    logger.info("Reaction triplets: %d", len(triplets))

    ts = load_ts_expression()
    logger.info("TS expression: %d rows", len(ts))

    site_glycans = load_site_glycan_data()
    logger.info("N-linked site-glycan triples: %d", len(site_glycans))

    # Build glycan composition map
    structs_df = pd.read_csv(PROJECT / "data_raw/glytoucan_structures.tsv", sep="\t")
    observed_glycans_set = set(site_glycans["glytoucan_ac"].unique())
    glycan_compositions = build_glycan_composition_map(
        list(observed_glycans_set), structs_df
    )
    logger.info("Glycan compositions parsed: %d / %d observed",
                len(glycan_compositions), len(observed_glycans_set))

    # Build popularity scores
    glycan_counts = site_glycans["glytoucan_ac"].value_counts()
    total = glycan_counts.sum()
    popularity = {gid: cnt / total for gid, cnt in glycan_counts.items()}
    logger.info("Popularity scores: %d glycans", len(popularity))

    # All N-linked glycan candidates — restrict to those with composition
    all_glycans = sorted(glycan_compositions.keys())
    logger.info("Candidate glycans: %d (with composition)", len(all_glycans))

    # Evaluate
    evaluate(site_glycans, glycan_compositions, triplets, ts, popularity, all_glycans)


if __name__ == "__main__":
    main()
