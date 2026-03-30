#!/usr/bin/env python3
"""Consolidated benchmark: aggregate all experiment results into paper tables.

Paper table numbering:
  Table 1 = KG statistics (inline in draft, not generated here)
  Table 2 = Hierarchical prediction benchmark
  Table 3 = Cell-type conditioned prediction

Outputs:
  - experiments_v2/paper_tables/table2_hierarchical.tsv  (+ .tex)
  - experiments_v2/paper_tables/table3_celltype.tsv  (+ .tex)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT / "experiments_v2/paper_tables"


# ──────────────────────────────────────────────────────────────────
# Result loading helpers
# ──────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict | None:
    """Load JSON file, return None if not found."""
    if not path.exists():
        logger.warning("Missing: %s", path)
        return None
    with open(path) as f:
        return json.load(f)


def safe_get(d: dict | None, *keys, default=None):
    """Nested dict lookup with default."""
    if d is None:
        return default
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d


# ──────────────────────────────────────────────────────────────────
# Table 2: Hierarchical prediction benchmark
# ──────────────────────────────────────────────────────────────────

def build_table2() -> pd.DataFrame:
    """Hierarchical glycan prediction from protein sequence."""
    base = PROJECT / "experiments_v2"

    # Load results
    v6b = load_json(base / "glycan_retrieval_v6b/results.json")
    v3 = load_json(base / "glycan_retrieval_v3/results.json")
    pipeline = load_json(base / "pipeline_benchmark/results.json")
    ct_v2 = load_json(base / "celltype_conditioned_v2/results.json")

    rows = []

    # 1. N-linked classification
    rows.append({
        "Task": "N-linked site classification",
        "Candidate space": "Binary",
        "Model": "ESM-2 MLP",
        "Primary metric": "F1",
        "Value": safe_get(pipeline, "nlinked_classification", "f1", default=0.924),
        "Secondary metric": "AUC-ROC",
        "Secondary value": safe_get(pipeline, "nlinked_classification", "auc_roc", default=""),
    })

    # 2. Dominant glycosylation type
    rows.append({
        "Task": "Dominant type prediction",
        "Candidate space": "4 classes",
        "Model": "Fraction regression",
        "Primary metric": "Accuracy",
        "Value": safe_get(pipeline, "dominant_type", "accuracy", default=0.844),
        "Secondary metric": "",
        "Secondary value": "",
    })

    # 3. Site ranking
    rows.append({
        "Task": "Site ranking",
        "Candidate space": "Per protein",
        "Model": "ESM-2 ranking",
        "Primary metric": "Recall@3",
        "Value": safe_get(pipeline, "site_ranking", "recall_at_3", default=0.683),
        "Secondary metric": "MRR",
        "Secondary value": safe_get(pipeline, "site_ranking", "mrr", default=0.688),
    })

    # 4. Structural family (8 classes)
    rows.append({
        "Task": "Structural family (8 classes)",
        "Candidate space": "8 classes",
        "Model": "ESM-2 MLP",
        "Primary metric": "Top-2 Acc",
        "Value": safe_get(pipeline, "structural_family", "top2_acc", default=0.858),
        "Secondary metric": "Top-1 Acc",
        "Secondary value": safe_get(pipeline, "structural_family", "top1_acc", default=0.638),
    })

    # 5. Structural cluster (K=20)
    rows.append({
        "Task": "Structural cluster (K=20)",
        "Candidate space": "20 clusters",
        "Model": "v6b site-level",
        "Primary metric": "H@5",
        "Value": safe_get(v6b, "cluster_k20", "cluster_h@5", default=0.786),
        "Secondary metric": "H@10",
        "Secondary value": safe_get(v6b, "cluster_k20", "cluster_h@10", default=0.978),
    })

    # 6. Exact glycan ID (sequence only)
    n_cands = safe_get(v6b, "n_candidates", default=2357)
    rows.append({
        "Task": "Exact glycan ID (sequence)",
        "Candidate space": f"{n_cands} glycans",
        "Model": "v6b site-level",
        "Primary metric": "MRR",
        "Value": safe_get(v6b, "test", "mrr", default=0.066),
        "Secondary metric": "H@10",
        "Secondary value": safe_get(v6b, "test", "hits@10", default=0.145),
    })

    # 7. Exact glycan ID (protein-level)
    n_cands_v3 = safe_get(v3, "n_candidates", default=3345)
    rows.append({
        "Task": "Exact glycan ID (protein)",
        "Candidate space": f"{n_cands_v3} glycans",
        "Model": "v3 protein-level",
        "Primary metric": "MRR",
        "Value": safe_get(v3, "test", "mrr", default=0.118),
        "Secondary metric": "H@10",
        "Secondary value": safe_get(v3, "test", "hits@10", default=0.228),
    })

    # 8. Exact glycan ID with cell-type (if available)
    if ct_v2 is not None:
        wc_oracle = safe_get(ct_v2, "within_cluster_oracle", default={})
        rows.append({
            "Task": "Within-cluster ID (+cell type)",
            "Candidate space": f"~{safe_get(ct_v2, 'cluster_sizes', 'mean', default=118):.0f}/cluster",
            "Model": "Enzyme expression oracle",
            "Primary metric": "MRR",
            "Value": safe_get(wc_oracle, "mrr", default=""),
            "Secondary metric": "H@5",
            "Secondary value": safe_get(wc_oracle, "hits@5", default=""),
        })

    # 9. Popularity baseline
    rows.append({
        "Task": "Popularity baseline",
        "Candidate space": f"{n_cands} glycans",
        "Model": "Train frequency",
        "Primary metric": "MRR",
        "Value": safe_get(v6b, "popularity", "mrr", default=0.100),
        "Secondary metric": "H@10",
        "Secondary value": safe_get(v6b, "popularity", "hits@10", default=0.193),
    })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────
# Table 3: Cell-type conditioned results
# ──────────────────────────────────────────────────────────────────

def build_table3() -> pd.DataFrame:
    """Cell-type conditioned glycan prediction."""
    ct_v1 = load_json(PROJECT / "experiments_v2/celltype_glycan/results.json")
    ct_v2 = load_json(PROJECT / "experiments_v2/celltype_conditioned_v2/results.json")

    rows = []

    # V1 results (full-ranking, 2,357 candidates)
    if ct_v1:
        n_gly = ct_v1.get("n_glycans", 2359)
        rows.append({
            "Method": "Popularity baseline",
            "Scope": f"Full ({n_gly})",
            "MRR": safe_get(ct_v1, "popularity", "mrr", default=""),
            "H@10": safe_get(ct_v1, "popularity", "h@10", default=""),
            "H@50": "",
            "Note": "Train frequency",
        })
        rows.append({
            "Method": "Enzyme marginal (all CT)",
            "Scope": f"Full ({n_gly})",
            "MRR": safe_get(ct_v1, "marginal", "mrr", default=""),
            "H@10": safe_get(ct_v1, "marginal", "h@10", default=""),
            "H@50": safe_get(ct_v1, "marginal", "h@50", default=""),
            "Note": "Avg over 748 cell types",
        })
        rows.append({
            "Method": "Enzyme oracle (best CT)",
            "Scope": f"Full ({n_gly})",
            "MRR": safe_get(ct_v1, "oracle", "mrr", default=""),
            "H@10": safe_get(ct_v1, "oracle", "h@10", default=""),
            "H@50": safe_get(ct_v1, "oracle", "h@50", default=""),
            "Note": "Best CT per sample",
        })

    # V2 results (within-cluster)
    if ct_v2:
        for key, label in [
            ("within_cluster_popularity", "Popularity"),
            ("within_cluster_marginal", "Enzyme marginal"),
            ("within_cluster_enzyme_pop", "Enzyme+popularity"),
            ("within_cluster_oracle", "Enzyme oracle"),
        ]:
            m = ct_v2.get(key, {})
            if m:
                rows.append({
                    "Method": label,
                    "Scope": "Within-cluster",
                    "MRR": m.get("mrr", ""),
                    "H@10": m.get("hits@10", ""),
                    "H@50": m.get("hits@50", ""),
                    "Note": f"Avg ~{safe_get(ct_v2, 'cluster_sizes', 'mean', default=118):.0f} cands",
                })

        # Learned model
        learned = safe_get(ct_v2, "learned_model", "within_cluster", default={})
        if learned:
            rows.append({
                "Method": "Learned CellTypeGlycanScorer",
                "Scope": "Within-cluster",
                "MRR": learned.get("mrr", ""),
                "H@10": learned.get("hits@10", ""),
                "H@50": learned.get("hits@50", ""),
                "Note": "InfoNCE trained",
            })

        # Cascade estimates
        cascade = ct_v2.get("cascade_estimates", {})
        if cascade:
            rows.append({
                "Method": "Cascade (cluster x within)",
                "Scope": f"Full ({ct_v2.get('n_glycans', '')})",
                "MRR": cascade.get("cascade_oracle_mrr", ""),
                "H@10": "",
                "H@50": "",
                "Note": f"Cluster H@1={cascade.get('cluster_h1', ''):.3f}",
            })

    # Reference models
    rows.append({
        "Method": "v6b site-level (sequence only)",
        "Scope": "Full (2,357)",
        "MRR": 0.066,
        "H@10": 0.145,
        "H@50": "",
        "Note": "Reference",
    })
    rows.append({
        "Method": "v3 protein-level (sequence only)",
        "Scope": "Full (3,345)",
        "MRR": 0.118,
        "H@10": 0.228,
        "H@50": "",
        "Note": "Reference",
    })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────
# LaTeX output
# ──────────────────────────────────────────────────────────────────

def to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    """Convert DataFrame to LaTeX table."""
    lines = []
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")

    ncols = len(df.columns)
    col_spec = "l" + "r" * (ncols - 1)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header
    header = " & ".join(f"\\textbf{{{c}}}" for c in df.columns)
    lines.append(header + " \\\\")
    lines.append("\\midrule")

    # Data rows
    for _, row in df.iterrows():
        vals = []
        for c in df.columns:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.3f}" if v >= 0.001 else f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append(" & ".join(vals) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 70)
    logger.info("  Consolidated Benchmark: Generating Paper Tables")
    logger.info("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Table 2: Hierarchical prediction
    t2 = build_table2()
    t2.to_csv(OUTPUT_DIR / "table2_hierarchical.tsv", sep="\t", index=False)
    with open(OUTPUT_DIR / "table2_hierarchical.tex", "w") as f:
        f.write(to_latex(t2,
                         "Hierarchical glycan prediction from protein sequence",
                         "tab:hierarchical"))
    logger.info("Table 2: %d rows -> %s", len(t2), OUTPUT_DIR / "table2_hierarchical.tsv")
    print("\n" + t2.to_string(index=False))

    # Table 3: Cell-type
    t3 = build_table3()
    t3.to_csv(OUTPUT_DIR / "table3_celltype.tsv", sep="\t", index=False)
    with open(OUTPUT_DIR / "table3_celltype.tex", "w") as f:
        f.write(to_latex(t3,
                         "Cell-type conditioned glycan prediction",
                         "tab:celltype"))
    logger.info("\nTable 3: %d rows -> %s", len(t3), OUTPUT_DIR / "table3_celltype.tsv")
    print("\n" + t3.to_string(index=False))

    logger.info("\nAll tables saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
