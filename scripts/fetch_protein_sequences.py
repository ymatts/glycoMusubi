#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fetch_protein_sequences.py

Fetches protein sequences from the UniProt REST API for all proteins and enzymes
in the glycoMusubi knowledge graph. Reads protein IDs from cleaned entity tables,
strips isoform suffixes for API queries, and outputs a TSV with sequences.

Supports resume: skips IDs already present in the output file.

Usage:
    python scripts/fetch_protein_sequences.py
    python scripts/fetch_protein_sequences.py --limit 50
"""

import os
import sys
import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm

# Allow imports from scripts/utils/
sys.path.insert(0, os.path.dirname(__file__))

from utils.api_client import APIClient, APIFetchResult

logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
PROTEINS_TSV = BASE_DIR / "data_clean" / "proteins_clean.tsv"
ENZYMES_TSV = BASE_DIR / "data_clean" / "enzymes_clean.tsv"
OUTPUT_TSV = BASE_DIR / "data_clean" / "protein_sequences.tsv"
FAILED_IDS_LOG = BASE_DIR / "logs" / "failed_sequence_ids.txt"

UNIPROT_API_URL = "https://rest.uniprot.org/uniprotkb/{accession}.json"

# Regex to strip isoform suffix (e.g., P01588-1 -> P01588)
ISOFORM_SUFFIX_RE = re.compile(r"-\d+$")


def strip_isoform(uniprot_id: str) -> str:
    """
    Strip the isoform suffix from a UniProt accession.

    Args:
        uniprot_id: UniProt accession, possibly with isoform suffix (e.g. P01588-1).

    Returns:
        Canonical accession without isoform suffix (e.g. P01588).
    """
    return ISOFORM_SUFFIX_RE.sub("", uniprot_id)


def load_protein_ids() -> List[str]:
    """
    Load unique protein IDs from proteins_clean.tsv and enzymes_clean.tsv.

    Returns:
        Sorted list of unique protein/enzyme IDs (with isoform suffixes).
    """
    ids: Set[str] = set()

    if PROTEINS_TSV.exists():
        df = pd.read_csv(PROTEINS_TSV, sep="\t", usecols=["protein_id"])
        protein_ids = df["protein_id"].dropna().astype(str).str.strip()
        ids.update(protein_ids)
        logger.info("Loaded %d protein IDs from %s", len(protein_ids), PROTEINS_TSV.name)
    else:
        logger.warning("Proteins file not found: %s", PROTEINS_TSV)

    if ENZYMES_TSV.exists():
        df = pd.read_csv(ENZYMES_TSV, sep="\t", usecols=["enzyme_id"])
        enzyme_ids = df["enzyme_id"].dropna().astype(str).str.strip()
        ids.update(enzyme_ids)
        logger.info("Loaded %d enzyme IDs from %s", len(enzyme_ids), ENZYMES_TSV.name)
    else:
        logger.warning("Enzymes file not found: %s", ENZYMES_TSV)

    sorted_ids = sorted(ids)
    logger.info("Total unique IDs to fetch: %d", len(sorted_ids))
    return sorted_ids


def load_existing_sequences() -> Dict[str, Tuple[str, int]]:
    """
    Load already-fetched sequences from the output file for resume support.

    Returns:
        Dict mapping uniprot_id -> (sequence, length).
    """
    existing: Dict[str, Tuple[str, int]] = {}

    if OUTPUT_TSV.exists():
        df = pd.read_csv(OUTPUT_TSV, sep="\t")
        for _, row in df.iterrows():
            uid = str(row["uniprot_id"]).strip()
            seq = str(row["sequence"]).strip()
            length = int(row["length"])
            existing[uid] = (seq, length)
        logger.info("Loaded %d existing sequences from %s (resume mode)", len(existing), OUTPUT_TSV.name)

    return existing


def fetch_sequence(client: APIClient, accession: str) -> Optional[Tuple[str, int]]:
    """
    Fetch a single protein sequence from UniProt REST API.

    Args:
        client: APIClient instance with rate limiting.
        accession: Canonical UniProt accession (no isoform suffix).

    Returns:
        Tuple of (sequence_string, sequence_length) or None on failure.
    """
    url = UNIPROT_API_URL.format(accession=accession)

    try:
        data = client.get_json(url)
    except Exception as e:
        logger.error("Exception fetching %s: %s", accession, e)
        return None

    if data is None:
        return None

    # Extract sequence from response JSON
    seq_obj = data.get("sequence")
    if seq_obj is None:
        logger.warning("No 'sequence' key in response for %s", accession)
        return None

    seq_value = seq_obj.get("value")
    if not seq_value:
        logger.warning("Empty sequence value for %s", accession)
        return None

    return (seq_value, len(seq_value))


def fetch_all_sequences(
    protein_ids: List[str],
    existing: Dict[str, Tuple[str, int]],
    limit: Optional[int] = None,
) -> Tuple[List[Dict[str, str]], APIFetchResult]:
    """
    Fetch sequences for all protein IDs, skipping those already fetched.

    Args:
        protein_ids: List of UniProt IDs (with isoform suffixes).
        existing: Dict of already-fetched sequences for resume support.
        limit: Optional limit on number of IDs to fetch (for testing).

    Returns:
        Tuple of (list of result dicts, APIFetchResult for tracking).
    """
    # Determine which IDs still need fetching
    to_fetch = [pid for pid in protein_ids if pid not in existing]

    if limit is not None:
        to_fetch = to_fetch[:limit]
        logger.info("Limiting fetch to %d IDs (--limit %d)", len(to_fetch), limit)

    skipped = len(protein_ids) - len(to_fetch) if limit is None else len(existing)
    logger.info(
        "IDs to fetch: %d | Already fetched (skipped): %d",
        len(to_fetch),
        skipped,
    )

    result = APIFetchResult()
    rows: List[Dict[str, str]] = []

    # Include existing sequences in the output
    for uid, (seq, length) in existing.items():
        rows.append({"uniprot_id": uid, "sequence": seq, "length": str(length)})

    if not to_fetch:
        logger.info("All sequences already fetched. Nothing to do.")
        return rows, result

    # Deduplicate canonical accessions to avoid redundant API calls.
    # Multiple isoform IDs (e.g. P01588-1, P01588-2) map to the same
    # canonical accession; we only need to fetch once per accession.
    canonical_to_isoforms: Dict[str, List[str]] = {}
    for pid in to_fetch:
        canon = strip_isoform(pid)
        canonical_to_isoforms.setdefault(canon, []).append(pid)

    unique_accessions = sorted(canonical_to_isoforms.keys())
    logger.info(
        "Unique canonical accessions to query: %d (from %d isoform IDs)",
        len(unique_accessions),
        len(to_fetch),
    )

    with APIClient(rate_limit=10.0) as client:
        for accession in tqdm(unique_accessions, desc="Fetching sequences", unit="seq"):
            seq_result = fetch_sequence(client, accession)

            isoform_ids = canonical_to_isoforms[accession]

            if seq_result is not None:
                seq_value, seq_length = seq_result
                for pid in isoform_ids:
                    rows.append({
                        "uniprot_id": pid,
                        "sequence": seq_value,
                        "length": str(seq_length),
                    })
                    result.add_success(pid)
            else:
                for pid in isoform_ids:
                    result.add_failure(pid, "fetch_error", f"Failed to fetch sequence for {accession}")

    return rows, result


def save_sequences(rows: List[Dict[str, str]]) -> None:
    """
    Save fetched sequences to the output TSV file.

    Args:
        rows: List of dicts with keys uniprot_id, sequence, length.
    """
    OUTPUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["uniprot_id", "sequence", "length"])
    df = df.drop_duplicates(subset=["uniprot_id"], keep="first")
    df = df.sort_values("uniprot_id").reset_index(drop=True)
    df.to_csv(OUTPUT_TSV, sep="\t", index=False)
    logger.info("Saved %d sequences to %s", len(df), OUTPUT_TSV)


def report_coverage(total_ids: int, result: APIFetchResult, existing_count: int) -> None:
    """
    Report coverage statistics.

    Args:
        total_ids: Total number of unique protein/enzyme IDs.
        result: APIFetchResult from the current fetch run.
        existing_count: Number of sequences already present before this run.
    """
    fetched = len(result.success) + existing_count
    coverage = (fetched / total_ids * 100) if total_ids > 0 else 0.0

    logger.info("=" * 60)
    logger.info("Sequence Fetch Summary")
    logger.info("=" * 60)
    logger.info("Total unique protein/enzyme IDs: %d", total_ids)
    logger.info("Previously fetched (resume):     %d", existing_count)
    logger.info("Newly fetched (this run):        %d", len(result.success))
    logger.info("Failed (this run):               %d", len(result.failed))
    logger.info("Coverage:                        %.1f%% (%d / %d)", coverage, fetched, total_ids)
    logger.info("=" * 60)

    if coverage < 95.0:
        logger.warning("Coverage %.1f%% is below 95%% target!", coverage)
    else:
        logger.info("Coverage meets the >= 95%% target.")

    if result.failed:
        logger.info(result.summary())


def main() -> None:
    """Main entry point for fetching protein sequences."""
    parser = argparse.ArgumentParser(
        description="Fetch protein sequences from UniProt REST API."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of IDs to fetch (for testing).",
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("Starting protein sequence fetch")

    # Load all protein/enzyme IDs
    protein_ids = load_protein_ids()
    if not protein_ids:
        logger.error("No protein IDs found. Check input files.")
        sys.exit(1)

    # Load existing sequences for resume support
    existing = load_existing_sequences()

    # Fetch sequences
    rows, result = fetch_all_sequences(protein_ids, existing, limit=args.limit)

    # Save results
    save_sequences(rows)

    # Save failed IDs for retry
    if result.failed:
        FAILED_IDS_LOG.parent.mkdir(parents=True, exist_ok=True)
        result.save_failed_ids(str(FAILED_IDS_LOG))

    # Report coverage
    report_coverage(len(protein_ids), result, len(existing))


if __name__ == "__main__":
    main()
