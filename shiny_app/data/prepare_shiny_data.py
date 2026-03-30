#!/usr/bin/env python3
"""Prepare pre-computed data for the glycoMusubi Shiny app.

Generates:
  - umap_glycan_embeddings.tsv  (UMAP of KG glycan embeddings)
  - umap_protein_embeddings.tsv (UMAP of ESM-2 protein embeddings)
  - glycan_hierarchy.tsv        (parent_of / child_of edges)
  - retrieval_rankings.tsv      (top-100 glycan retrievals per test protein)
  - site_predictions.tsv        (N-linked site prediction results)

Usage:
  python prepare_shiny_data.py [--exp-dir ../../experiments_v2]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──
SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR.parent.parent  # glycoMusubi/
KG_DIR = BASE / "kg"
DATA_DIR = BASE / "data_clean"
EXP_DIR = BASE / "experiments_v2"
OUT_DIR = SCRIPT_DIR


def load_kg_edges():
    """Load KG edges from parquet or TSV."""
    pq = KG_DIR / "edges.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    return pd.read_csv(KG_DIR / "edges.tsv", sep="\t")


def load_kg_nodes():
    """Load KG nodes from parquet or TSV."""
    pq = KG_DIR / "nodes.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    return pd.read_csv(KG_DIR / "nodes.tsv", sep="\t")


# ══════════════════════════════════════════════════════════════
# 1. Glycan hierarchy
# ══════════════════════════════════════════════════════════════
def prepare_hierarchy(edges: pd.DataFrame):
    """Extract glycan hierarchy edges."""
    print("Preparing glycan hierarchy...")
    hier = edges[edges["relation"].isin(
        ["parent_of", "child_of", "subsumes", "subsumed_by"]
    )][["source_id", "target_id", "relation"]].copy()
    out = OUT_DIR / "glycan_hierarchy.tsv"
    hier.to_csv(out, sep="\t", index=False)
    print(f"  → {len(hier):,} hierarchy edges → {out}")


# ══════════════════════════════════════════════════════════════
# 2. UMAP embeddings
# ══════════════════════════════════════════════════════════════
def prepare_umap_glycan(exp_dir: Path):
    """Compute UMAP from KG glycan embeddings."""
    print("Preparing glycan UMAP...")
    try:
        from umap import UMAP
        import torch
    except ImportError:
        print("  SKIP: umap-learn or torch not installed")
        return

    # Find best inductive model
    candidates = sorted(exp_dir.glob("glycokgnet_inductive_*/best.pt"))
    if not candidates:
        print("  SKIP: no GlycoKGNet checkpoint found")
        return

    ckpt_path = candidates[-1]
    print(f"  Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Extract glycan entity embeddings
    # The exact key depends on model architecture; try common patterns
    emb = None
    for key in ["entity_embeddings.weight", "glycan_embeddings", "embeddings"]:
        if key in ckpt:
            emb = ckpt[key].numpy()
            break

    if emb is None and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        for key in state:
            if "entity" in key and "embed" in key:
                emb = state[key].numpy()
                break

    if emb is None:
        print("  SKIP: cannot extract glycan embeddings from checkpoint")
        return

    # Load node list to get glycan IDs
    nodes = load_kg_nodes()
    glycans = nodes[nodes["node_type"] == "glycan"]["node_id"].tolist()

    # Use first N embeddings matching glycan count (heuristic)
    n = min(len(glycans), emb.shape[0])
    emb_sub = emb[:n]
    glycan_ids = glycans[:n]

    print(f"  Running UMAP on {n} glycan embeddings ({emb_sub.shape[1]}d)...")
    reducer = UMAP(n_components=2, n_neighbors=30, min_dist=0.3, random_state=42)
    coords = reducer.fit_transform(emb_sub)

    # Try to get cluster assignments
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=20, random_state=42, n_init=10)
        clusters = km.fit_predict(emb_sub)
    except Exception:
        clusters = np.zeros(n, dtype=int)

    # Try to get glycan type info
    glycans_df = pd.read_csv(DATA_DIR / "glycans_clean.tsv", sep="\t")
    type_map = {}
    if "glycan_type" in glycans_df.columns:
        type_map = dict(zip(glycans_df["node_id"], glycans_df["glycan_type"]))

    df = pd.DataFrame({
        "id": glycan_ids,
        "umap_x": coords[:, 0],
        "umap_y": coords[:, 1],
        "cluster": clusters,
        "glycan_type": [type_map.get(g, "unknown") for g in glycan_ids],
    })

    out = OUT_DIR / "umap_glycan_embeddings.tsv"
    df.to_csv(out, sep="\t", index=False)
    print(f"  → {len(df):,} glycan UMAP coordinates → {out}")


def prepare_umap_protein():
    """Compute UMAP from ESM-2 protein embeddings."""
    print("Preparing protein UMAP...")
    try:
        from umap import UMAP
        import torch
    except ImportError:
        print("  SKIP: umap-learn or torch not installed")
        return

    cache_dir = DATA_DIR / "esm2_cache"
    if not cache_dir.exists():
        print("  SKIP: ESM-2 cache not found")
        return

    pt_files = sorted(cache_dir.glob("*.pt"))
    if not pt_files:
        print("  SKIP: no .pt files in ESM-2 cache")
        return

    print(f"  Loading {len(pt_files)} ESM-2 embeddings...")
    ids = []
    embs = []
    for f in pt_files:
        try:
            t = torch.load(f, map_location="cpu", weights_only=False)
            if isinstance(t, torch.Tensor):
                if t.dim() == 1:
                    embs.append(t.numpy())
                else:
                    embs.append(t.mean(dim=0).numpy())
                ids.append(f.stem)
        except Exception:
            continue

    if len(embs) < 10:
        print("  SKIP: too few embeddings loaded")
        return

    emb_matrix = np.stack(embs)
    print(f"  Running UMAP on {len(ids)} protein embeddings ({emb_matrix.shape[1]}d)...")
    reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    coords = reducer.fit_transform(emb_matrix)

    df = pd.DataFrame({
        "id": ids,
        "umap_x": coords[:, 0],
        "umap_y": coords[:, 1],
    })

    out = OUT_DIR / "umap_protein_embeddings.tsv"
    df.to_csv(out, sep="\t", index=False)
    print(f"  → {len(df):,} protein UMAP coordinates → {out}")


# ══════════════════════════════════════════════════════════════
# 3. Retrieval rankings
# ══════════════════════════════════════════════════════════════
def prepare_retrieval_rankings(exp_dir: Path):
    """Extract pre-computed retrieval rankings from experiment results."""
    print("Preparing retrieval rankings...")

    # Look for retrieval experiment output
    retrieval_dirs = sorted(exp_dir.glob("glycan_retrieval_*"))
    if not retrieval_dirs:
        print("  SKIP: no retrieval experiment directories found")
        return

    best_dir = retrieval_dirs[-1]
    rankings_file = best_dir / "test_rankings.tsv"
    eval_file = best_dir / "eval_payload.pt"

    if rankings_file.exists():
        df = pd.read_csv(rankings_file, sep="\t")
        # Keep top 100 per query
        if "rank" in df.columns:
            df = df[df["rank"] <= 100]
        out = OUT_DIR / "retrieval_rankings.tsv"
        df.to_csv(out, sep="\t", index=False)
        print(f"  → {len(df):,} retrieval rankings → {out}")
    elif eval_file.exists():
        try:
            import torch
            payload = torch.load(eval_file, map_location="cpu", weights_only=False)
            if isinstance(payload, dict) and "rankings" in payload:
                df = pd.DataFrame(payload["rankings"])
                out = OUT_DIR / "retrieval_rankings.tsv"
                df.to_csv(out, sep="\t", index=False)
                print(f"  → {len(df):,} retrieval rankings → {out}")
            else:
                print("  SKIP: eval_payload.pt doesn't contain rankings")
        except Exception as e:
            print(f"  SKIP: error loading eval_payload.pt: {e}")
    else:
        print("  SKIP: no rankings file found")


# ══════════════════════════════════════════════════════════════
# 4. Site predictions
# ══════════════════════════════════════════════════════════════
def prepare_site_predictions(exp_dir: Path):
    """Extract N-linked site prediction results."""
    print("Preparing site predictions...")

    site_dirs = sorted(exp_dir.glob("nlinked_*"))
    if not site_dirs:
        print("  SKIP: no N-linked experiment directories found")
        return

    best_dir = site_dirs[-1]
    pred_file = best_dir / "test_predictions.tsv"
    results_file = best_dir / "results.json"

    if pred_file.exists():
        df = pd.read_csv(pred_file, sep="\t")
        out = OUT_DIR / "site_predictions.tsv"
        df.to_csv(out, sep="\t", index=False)
        print(f"  → {len(df):,} site predictions → {out}")
    elif results_file.exists():
        results = json.loads(results_file.read_text())
        if "predictions" in results:
            df = pd.DataFrame(results["predictions"])
            out = OUT_DIR / "site_predictions.tsv"
            df.to_csv(out, sep="\t", index=False)
            print(f"  → {len(df):,} site predictions → {out}")
        else:
            print("  SKIP: results.json doesn't contain predictions")
    else:
        print("  SKIP: no prediction files found")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Prepare Shiny app data")
    parser.add_argument("--exp-dir", type=Path, default=EXP_DIR)
    parser.add_argument("--skip-umap", action="store_true",
                        help="Skip UMAP computation (slow)")
    args = parser.parse_args()

    print(f"Base directory: {BASE}")
    print(f"Output directory: {OUT_DIR}")
    print()

    edges = load_kg_edges()

    # Always run these (fast)
    prepare_hierarchy(edges)
    prepare_retrieval_rankings(args.exp_dir)
    prepare_site_predictions(args.exp_dir)

    # UMAP (slow, optional)
    if not args.skip_umap:
        prepare_umap_glycan(args.exp_dir)
        prepare_umap_protein()
    else:
        print("Skipping UMAP (--skip-umap)")

    print("\nDone. Run the Shiny app with: Rscript -e 'shiny::runApp(\"shiny_app\")'")


if __name__ == "__main__":
    main()
