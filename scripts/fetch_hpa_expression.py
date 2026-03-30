#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fetch_hpa_expression.py

Downloads Human Protein Atlas (HPA) tissue expression data and builds a
protein x tissue nTPM expression matrix for proteins/enzymes in the glycoMusubi
knowledge graph.

Data sources:
  - rna_tissue_consensus.tsv.zip  : Consensus RNA expression per tissue
  - proteinatlas.tsv.zip          : Gene-level annotation with Uniprot mapping

Pipeline:
  1. Download & unzip HPA data to data_raw/
  2. Load consensus expression (Gene, Gene name, Tissue, nTPM)
  3. Load proteinatlas.tsv for Ensembl -> UniProt mapping
  4. Pivot to (UniProt x Tissue) nTPM matrix
  5. Filter to proteins/enzymes present in the KG
  6. Save filtered matrix to data_clean/hpa_tissue_expression.tsv

Usage:
    python scripts/fetch_hpa_expression.py
    python scripts/fetch_hpa_expression.py --skip-download  # reuse cached files
"""

import argparse
import logging
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Set

import pandas as pd

logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data_raw"
DATA_CLEAN = BASE_DIR / "data_clean"

# HPA download URLs (current as of 2026-03; main site moved RNA TSVs under /tsv/)
RNA_TISSUE_URL = "https://www.proteinatlas.org/download/tsv/rna_tissue_consensus.tsv.zip"
PROTEINATLAS_URL = "https://www.proteinatlas.org/download/proteinatlas.tsv.zip"

# Downloaded / extracted file paths
RNA_TISSUE_ZIP = DATA_RAW / "rna_tissue_consensus.tsv.zip"
RNA_TISSUE_TSV = DATA_RAW / "rna_tissue_consensus.tsv"
PROTEINATLAS_ZIP = DATA_RAW / "proteinatlas.tsv.zip"
PROTEINATLAS_TSV = DATA_RAW / "proteinatlas.tsv"

# KG edge files for protein ID collection
EDGES_GLYCAN_ENZYME = DATA_CLEAN / "edges_glycan_enzyme.tsv"
EDGES_GLYCAN_PROTEIN = DATA_CLEAN / "edges_glycan_protein.tsv"

# Output
OUTPUT_TSV = DATA_CLEAN / "hpa_tissue_expression.tsv"

# Regex to strip isoform suffix (e.g., P01588-1 -> P01588)
ISOFORM_SUFFIX_RE = re.compile(r"-\d+$")


def strip_isoform(uniprot_id: str) -> str:
    """Strip the isoform suffix from a UniProt accession."""
    return ISOFORM_SUFFIX_RE.sub("", uniprot_id)


def download_file(url: str, dest: Path) -> None:
    """
    Download a file using wget if it does not already exist.

    Args:
        url: URL to download.
        dest: Destination file path.
    """
    if dest.exists():
        logger.info("File already exists, skipping download: %s", dest.name)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", url, dest)

    cmd = [
        "wget",
        "--quiet",
        "--user-agent", "Mozilla/5.0 (GlycoKG pipeline)",
        "-O", str(dest),
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("wget failed (exit %d): %s", result.returncode, result.stderr)
        raise RuntimeError(f"Download failed for {url}")

    size_mb = dest.stat().st_size / (1024 * 1024)
    logger.info("Downloaded %s (%.1f MB)", dest.name, size_mb)


def unzip_file(zip_path: Path, extract_dir: Path) -> Path:
    """
    Extract a single-file zip archive.

    Args:
        zip_path: Path to the zip file.
        extract_dir: Directory to extract into.

    Returns:
        Path to the extracted file.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        logger.info("Zip contains: %s", names)
        zf.extractall(extract_dir)

    extracted = extract_dir / names[0]
    size_mb = extracted.stat().st_size / (1024 * 1024)
    logger.info("Extracted %s (%.1f MB)", extracted.name, size_mb)
    return extracted


def load_kg_protein_ids() -> Set[str]:
    """
    Load unique protein/enzyme UniProt IDs from KG edge files.

    Strips isoform suffixes to get canonical accessions for matching
    against HPA data.

    Returns:
        Set of canonical UniProt accessions (without isoform suffix).
    """
    ids: Set[str] = set()
    isoform_ids: Set[str] = set()

    if EDGES_GLYCAN_ENZYME.exists():
        df = pd.read_csv(EDGES_GLYCAN_ENZYME, sep="\t", usecols=["enzyme_id"])
        raw = df["enzyme_id"].dropna().astype(str).str.strip().unique()
        isoform_ids.update(raw)
        logger.info("Loaded %d enzyme IDs from %s", len(raw), EDGES_GLYCAN_ENZYME.name)
    else:
        logger.warning("File not found: %s", EDGES_GLYCAN_ENZYME)

    if EDGES_GLYCAN_PROTEIN.exists():
        df = pd.read_csv(EDGES_GLYCAN_PROTEIN, sep="\t", usecols=["protein_id"])
        raw = df["protein_id"].dropna().astype(str).str.strip().unique()
        isoform_ids.update(raw)
        logger.info("Loaded %d protein IDs from %s", len(raw), EDGES_GLYCAN_PROTEIN.name)
    else:
        logger.warning("File not found: %s", EDGES_GLYCAN_PROTEIN)

    # Strip isoform suffixes for matching
    for uid in isoform_ids:
        ids.add(strip_isoform(uid))

    logger.info(
        "KG proteins: %d isoform IDs -> %d canonical accessions",
        len(isoform_ids),
        len(ids),
    )
    return ids


def load_uniprot_mapping(proteinatlas_tsv: Path) -> pd.DataFrame:
    """
    Load Ensembl -> UniProt mapping from the proteinatlas.tsv file.

    proteinatlas.tsv has 'Gene' (gene symbol), 'Ensembl' (ENSG ID), and
    'Uniprot' columns.  We need the Ensembl column because
    rna_tissue_consensus.tsv uses Ensembl gene IDs in its 'Gene' column.

    Only reads 'Ensembl' and 'Uniprot' columns to keep memory usage low.

    Args:
        proteinatlas_tsv: Path to the extracted proteinatlas.tsv.

    Returns:
        DataFrame with columns ['Ensembl', 'Uniprot'], rows with valid IDs.
    """
    logger.info("Loading UniProt mapping from %s (reading 2 columns only)...", proteinatlas_tsv.name)

    # Peek at the header to find the exact column names
    header_df = pd.read_csv(proteinatlas_tsv, sep="\t", nrows=0)
    columns = list(header_df.columns)
    logger.info("proteinatlas.tsv columns (%d): %s", len(columns), columns[:10])

    # Find the Uniprot and Ensembl columns (case-insensitive matching)
    uniprot_col = None
    ensembl_col = None
    for col in columns:
        if col.lower() == "uniprot":
            uniprot_col = col
        if col.lower() == "ensembl":
            ensembl_col = col

    if uniprot_col is None:
        logger.error("Could not find Uniprot column. Available: %s", columns)
        raise ValueError("No Uniprot column found in proteinatlas.tsv")

    if ensembl_col is None:
        logger.error("Could not find Ensembl column. Available: %s", columns)
        raise ValueError("No Ensembl column found in proteinatlas.tsv")

    logger.info("Using columns: ensembl='%s', uniprot='%s'", ensembl_col, uniprot_col)

    df = pd.read_csv(
        proteinatlas_tsv,
        sep="\t",
        usecols=[ensembl_col, uniprot_col],
        dtype=str,
    )
    df = df.rename(columns={ensembl_col: "Ensembl", uniprot_col: "Uniprot"})
    df = df.dropna(subset=["Uniprot", "Ensembl"])
    df = df[df["Uniprot"].str.strip().str.len() > 0]
    df["Uniprot"] = df["Uniprot"].str.strip()
    df["Ensembl"] = df["Ensembl"].str.strip()

    logger.info("UniProt mapping: %d genes with UniProt IDs", len(df))
    return df


def build_expression_matrix(
    rna_tissue_tsv: Path,
    mapping_df: pd.DataFrame,
    kg_proteins: Set[str],
) -> pd.DataFrame:
    """
    Build a (UniProt x Tissue) nTPM expression matrix filtered to KG proteins.

    Args:
        rna_tissue_tsv: Path to rna_tissue_consensus.tsv.
        mapping_df: DataFrame with Gene -> Uniprot mapping.
        kg_proteins: Set of canonical UniProt accessions in the KG.

    Returns:
        DataFrame with UniProt as index and tissue names as columns,
        values are nTPM expression levels.
    """
    logger.info("Loading RNA tissue consensus data from %s...", rna_tissue_tsv.name)
    rna_df = pd.read_csv(rna_tissue_tsv, sep="\t", dtype={"Gene": str, "nTPM": float})
    logger.info("RNA tissue data: %d rows, columns: %s", len(rna_df), list(rna_df.columns))

    n_genes = rna_df["Gene"].nunique()
    n_tissues = rna_df["Tissue"].nunique()
    logger.info("Genes: %d, Tissues: %d", n_genes, n_tissues)

    # The rna_tissue_consensus 'Gene' column contains Ensembl IDs (ENSG...).
    # Merge with the Ensembl -> UniProt mapping from proteinatlas.tsv.
    logger.info("Merging expression data with UniProt mapping (Gene -> Ensembl)...")
    merged = rna_df.merge(
        mapping_df[["Ensembl", "Uniprot"]],
        left_on="Gene",
        right_on="Ensembl",
        how="inner",
    )
    logger.info("After merge: %d rows (%d unique UniProt IDs)",
                len(merged), merged["Uniprot"].nunique())

    # Some UniProt entries may have multiple Ensembl genes mapped.
    # For proteins with multiple Ensembl mappings, take the max nTPM per tissue.
    # Filter to KG proteins
    logger.info("Filtering to KG proteins...")
    merged_kg = merged[merged["Uniprot"].isin(kg_proteins)]
    logger.info("After KG filter: %d rows (%d unique UniProt IDs)",
                len(merged_kg), merged_kg["Uniprot"].nunique())

    if merged_kg.empty:
        logger.error("No KG proteins found in HPA data!")
        raise ValueError("Empty expression matrix after filtering")

    # Pivot to matrix: rows = UniProt, columns = Tissue, values = nTPM
    logger.info("Pivoting to expression matrix...")
    matrix = merged_kg.pivot_table(
        index="Uniprot",
        columns="Tissue",
        values="nTPM",
        aggfunc="max",  # Take max if multiple Ensembl genes map to same UniProt
    )

    # Fill NaN with 0 (no expression detected)
    matrix = matrix.fillna(0.0)

    # Sort by UniProt ID
    matrix = matrix.sort_index()

    logger.info(
        "Expression matrix: %d proteins x %d tissues",
        matrix.shape[0],
        matrix.shape[1],
    )

    # Report coverage
    matched = len(set(matrix.index) & kg_proteins)
    coverage = matched / len(kg_proteins) * 100 if kg_proteins else 0
    logger.info(
        "KG coverage: %d / %d proteins (%.1f%%)",
        matched,
        len(kg_proteins),
        coverage,
    )

    return matrix


def save_matrix(matrix: pd.DataFrame, output_path: Path) -> None:
    """
    Save the expression matrix to a TSV file.

    The index (UniProt ID) is saved as the first column named 'uniprot_id'.

    Args:
        matrix: Expression DataFrame with UniProt as index.
        output_path: Path to write the TSV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.index.name = "uniprot_id"
    matrix.to_csv(output_path, sep="\t", float_format="%.1f")
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Saved expression matrix to %s (%.2f MB)", output_path, size_mb)


def main() -> None:
    """Main entry point for HPA tissue expression download and processing."""
    parser = argparse.ArgumentParser(
        description="Download HPA tissue expression data and build protein x tissue matrix."
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step; reuse existing files in data_raw/.",
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("=" * 60)
    logger.info("HPA Tissue Expression Pipeline")
    logger.info("=" * 60)

    # Step 1: Download
    if not args.skip_download:
        logger.info("--- Step 1: Download HPA data ---")
        download_file(RNA_TISSUE_URL, RNA_TISSUE_ZIP)
        download_file(PROTEINATLAS_URL, PROTEINATLAS_ZIP)
    else:
        logger.info("--- Step 1: Skipping download (--skip-download) ---")

    # Step 2: Unzip
    logger.info("--- Step 2: Extract archives ---")
    if not RNA_TISSUE_TSV.exists():
        unzip_file(RNA_TISSUE_ZIP, DATA_RAW)
    else:
        logger.info("Already extracted: %s", RNA_TISSUE_TSV.name)

    if not PROTEINATLAS_TSV.exists():
        unzip_file(PROTEINATLAS_ZIP, DATA_RAW)
    else:
        logger.info("Already extracted: %s", PROTEINATLAS_TSV.name)

    # Step 3: Load KG protein IDs
    logger.info("--- Step 3: Load KG protein IDs ---")
    kg_proteins = load_kg_protein_ids()
    if not kg_proteins:
        logger.error("No protein IDs found in KG edge files. Aborting.")
        sys.exit(1)

    # Step 4: Load UniProt mapping from proteinatlas.tsv
    logger.info("--- Step 4: Load Ensembl -> UniProt mapping ---")
    mapping_df = load_uniprot_mapping(PROTEINATLAS_TSV)

    # Step 5: Build expression matrix
    logger.info("--- Step 5: Build expression matrix ---")
    matrix = build_expression_matrix(RNA_TISSUE_TSV, mapping_df, kg_proteins)

    # Step 6: Save
    logger.info("--- Step 6: Save output ---")
    save_matrix(matrix, OUTPUT_TSV)

    # Summary
    logger.info("=" * 60)
    logger.info("Done. Output: %s", OUTPUT_TSV)
    logger.info("Matrix shape: %d proteins x %d tissues", matrix.shape[0], matrix.shape[1])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
