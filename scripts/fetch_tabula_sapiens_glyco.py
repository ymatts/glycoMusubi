#!/usr/bin/env python3
"""Fetch glycosyltransferase expression from Tabula Sapiens via CELLxGENE Census.

Builds a (cell_type × enzyme_gene) expression matrix for glycosylation enzymes,
then maps to our KG enzymes via gene symbol → UniProt.

Output: data_clean/ts_celltype_enzyme_expression.tsv
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parent.parent


def get_glyco_gene_symbols() -> list[str]:
    """Get glycosylation enzyme gene symbols from our curated list."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "glycosylation_genes", PROJECT / "scripts/glycosylation_genes.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ALL_GENE_SYMBOLS


def fetch_from_census(gene_symbols: list[str]) -> pd.DataFrame:
    """Query CELLxGENE Census for gene expression in Tabula Sapiens."""
    import cellxgene_census

    logger.info("Opening CELLxGENE Census...")

    with cellxgene_census.open_soma() as census:
        # Find Tabula Sapiens dataset IDs
        datasets = census["census_info"]["datasets"].read(
            value_filter="collection_name == 'Tabula Sapiens'"
        ).concat().to_pandas()
        logger.info("Tabula Sapiens datasets: %d", len(datasets))
        if len(datasets) > 0:
            show_cols = [c for c in ["dataset_title", "cell_count", "dataset_total_cell_count"]
                         if c in datasets.columns]
            logger.info("  Columns: %s", list(datasets.columns))
            if show_cols:
                logger.info("  %s", datasets[show_cols].head(5).to_string())
            ts_dataset_ids = datasets["dataset_id"].tolist()
        else:
            logger.warning("No Tabula Sapiens datasets found, using all human data")
            ts_dataset_ids = None

        # Build gene symbol filter
        # Census uses feature_name for gene symbols
        gene_filter = "feature_name in [" + ",".join(f"'{g}'" for g in gene_symbols) + "]"

        # Build dataset filter if we found TS
        obs_filter = "is_primary_data == True"
        if ts_dataset_ids:
            ds_filter = " or ".join(f"dataset_id == '{did}'" for did in ts_dataset_ids)
            obs_filter = f"({obs_filter}) and ({ds_filter})"

        logger.info("Fetching expression data for %d genes...", len(gene_symbols))
        logger.info("  obs_filter: %s", obs_filter[:200])

        adata = cellxgene_census.get_anndata(
            census=census,
            organism="Homo sapiens",
            var_value_filter=gene_filter,
            obs_value_filter=obs_filter,
            column_names={
                "obs": ["cell_type", "tissue", "tissue_general", "assay",
                        "disease", "dataset_id"],
            },
        )

    logger.info("Fetched AnnData: %s", adata.shape)
    logger.info("  Cells: %d", adata.n_obs)
    logger.info("  Genes matched: %d / %d", adata.n_vars, len(gene_symbols))
    logger.info("  Cell types: %d", adata.obs["cell_type"].nunique())
    logger.info("  Tissues: %d", adata.obs["tissue_general"].nunique())

    return adata


def aggregate_by_celltype(adata) -> pd.DataFrame:
    """Aggregate expression to cell_type level.

    Returns DataFrame with columns:
        cell_type, tissue_general, gene_symbol, mean_expr, frac_expressing, n_cells
    """
    import scipy.sparse as sp

    logger.info("Aggregating by cell_type × tissue...")

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()

    gene_names = adata.var["feature_name"].values if "feature_name" in adata.var.columns \
        else adata.var_names.values

    obs = adata.obs.copy()
    # Convert categoricals to string for concatenation
    obs["cell_type"] = obs["cell_type"].astype(str)
    obs["tissue_general"] = obs["tissue_general"].astype(str)
    obs["ct_tissue"] = obs["cell_type"] + "||" + obs["tissue_general"]

    records = []
    for ct_tissue, group_idx in obs.groupby("ct_tissue").groups.items():
        idx_arr = group_idx.values if hasattr(group_idx, 'values') else list(group_idx)

        # Get integer positions for iloc
        positions = [obs.index.get_loc(i) for i in idx_arr]
        sub_X = X[positions]
        n_cells = len(positions)

        cell_type, tissue = ct_tissue.split("||", 1)

        for j, gene in enumerate(gene_names):
            col = sub_X[:, j]
            mean_expr = float(col.mean())
            frac_expr = float((col > 0).mean())

            if mean_expr > 0 or frac_expr > 0:
                records.append({
                    "cell_type": cell_type,
                    "tissue_general": tissue,
                    "gene_symbol": gene,
                    "mean_expr": mean_expr,
                    "frac_expressing": frac_expr,
                    "n_cells": n_cells,
                })

    df = pd.DataFrame(records)
    logger.info("Aggregated: %d records, %d cell_type-tissue combos, %d genes",
                len(df), df.groupby(["cell_type", "tissue_general"]).ngroups,
                df["gene_symbol"].nunique())
    return df


def map_to_kg_enzymes(agg_df: pd.DataFrame) -> pd.DataFrame:
    """Map gene symbols to KG enzyme UniProt IDs."""
    # Load KG enzyme list
    enzymes_path = PROJECT / "data_clean" / "enzymes_clean.tsv"
    if enzymes_path.exists():
        enz_df = pd.read_csv(enzymes_path, sep="\t")
        kg_enzymes = set(enz_df.iloc[:, 0])
    else:
        # Fallback: load from edges
        edges = pd.read_csv(PROJECT / "data_clean/edges_glycan_enzyme.tsv", sep="\t")
        kg_enzymes = set(edges["source"])
        logger.info("KG enzymes from edges: %d", len(kg_enzymes))

    # We need gene_symbol → UniProt mapping
    # Try fetching from UniProt ID mapping or use our existing data
    # For now, build from existing protein data
    prot_path = PROJECT / "data_clean/protein_sequences.tsv"
    gene_to_uniprot = {}
    if prot_path.exists():
        prot_df = pd.read_csv(prot_path, sep="\t")
        if "gene_name" in prot_df.columns:
            for _, row in prot_df.iterrows():
                gene = row.get("gene_name", "")
                uid = row.get("uniprot_id", row.iloc[0])
                if pd.notna(gene) and gene:
                    gene_to_uniprot[gene] = uid

    logger.info("Gene→UniProt mappings: %d", len(gene_to_uniprot))
    logger.info("KG enzymes: %d", len(kg_enzymes))

    # Add UniProt column
    agg_df["uniprot_id"] = agg_df["gene_symbol"].map(gene_to_uniprot)
    agg_df["in_kg"] = agg_df["uniprot_id"].isin(kg_enzymes)

    n_mapped = agg_df["uniprot_id"].notna().sum()
    n_in_kg = agg_df["in_kg"].sum()
    logger.info("Mapped to UniProt: %d / %d records", n_mapped, len(agg_df))
    logger.info("In KG: %d records", n_in_kg)

    return agg_df


def main():
    logger.info("=" * 70)
    logger.info("  Fetch Tabula Sapiens Glycosylation Enzyme Expression")
    logger.info("=" * 70)

    # 1. Get gene list
    gene_symbols = get_glyco_gene_symbols()
    logger.info("Glycosylation genes: %d", len(gene_symbols))

    # 2. Fetch from Census
    adata = fetch_from_census(gene_symbols)

    # 3. Aggregate by cell type
    agg_df = aggregate_by_celltype(adata)

    # 4. Map to KG
    agg_df = map_to_kg_enzymes(agg_df)

    # 5. Save
    output_path = PROJECT / "data_clean/ts_celltype_enzyme_expression.tsv"
    agg_df.to_csv(output_path, sep="\t", index=False)
    logger.info("Saved to %s", output_path)

    # 6. Summary stats
    logger.info("\n" + "─" * 60)
    logger.info("Summary:")
    logger.info("  Unique cell types: %d", agg_df["cell_type"].nunique())
    logger.info("  Unique tissues: %d", agg_df["tissue_general"].nunique())
    logger.info("  Unique genes: %d", agg_df["gene_symbol"].nunique())
    logger.info("  Cell-type-tissue combos: %d",
                agg_df.groupby(["cell_type", "tissue_general"]).ngroups)

    # Top expressed glyco-enzymes
    top_genes = agg_df.groupby("gene_symbol")["mean_expr"].mean().nlargest(20)
    logger.info("\n  Top 20 most expressed glyco-enzymes:")
    for gene, expr in top_genes.items():
        logger.info("    %s: %.2f", gene, expr)


if __name__ == "__main__":
    main()
