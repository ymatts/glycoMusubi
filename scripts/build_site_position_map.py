"""Build a protein-to-site-positions mapping from UniProt glycosylation sites.

Phase 1 of the glycoMusubi inductive prediction system.  Reads site-level
annotations from ``data_clean/uniprot_sites.tsv`` and produces a compact
mapping that associates each protein (UniProt ID) with its known
glycosylation sites.

Output
------
``data_clean/site_positions_map.pt`` containing:
  - ``"protein_to_sites"``: Dict[str, List[Tuple[int, str, str]]]
      Maps each uniprot_id to a list of (position, residue, site_type) tuples.
  - ``"stats"``: Dict with ``total_proteins`` and ``total_sites`` counts.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch

logger = logging.getLogger(__name__)

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data_clean" / "uniprot_sites.tsv"
OUTPUT_PATH = PROJECT_ROOT / "data_clean" / "site_positions_map.pt"


def main() -> None:
    """Read UniProt site annotations and build a protein-to-sites mapping."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("Reading site data from %s", INPUT_PATH)
    if not INPUT_PATH.exists():
        logger.error("Input file not found: %s", INPUT_PATH)
        raise FileNotFoundError(INPUT_PATH)

    df = pd.read_csv(INPUT_PATH, sep="\t")
    logger.info("Loaded %d site records", len(df))

    required_cols = {"uniprot_id", "site_position", "site_residue", "site_type"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # Build mapping: protein_id -> list of (position, residue, site_type)
    protein_to_sites: Dict[str, List[Tuple[int, str, str]]] = defaultdict(list)

    for _, row in df.iterrows():
        uid = str(row["uniprot_id"])
        try:
            position = int(row["site_position"])
        except (ValueError, TypeError):
            logger.debug(
                "Skipping row with non-integer site_position: %s",
                row["site_position"],
            )
            continue
        residue = str(row["site_residue"]) if pd.notna(row["site_residue"]) else ""
        site_type = str(row["site_type"]) if pd.notna(row["site_type"]) else ""

        protein_to_sites[uid].append((position, residue, site_type))

    # Convert defaultdict to plain dict for serialisation
    protein_to_sites_dict: Dict[str, List[Tuple[int, str, str]]] = dict(
        protein_to_sites
    )

    total_proteins = len(protein_to_sites_dict)
    total_sites = sum(len(v) for v in protein_to_sites_dict.values())
    avg_sites = total_sites / total_proteins if total_proteins > 0 else 0.0

    stats = {
        "total_proteins": total_proteins,
        "total_sites": total_sites,
    }

    cache = {
        "protein_to_sites": protein_to_sites_dict,
        "stats": stats,
    }

    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
    torch.save(cache, OUTPUT_PATH)

    # Report
    logger.info("Proteins with sites:      %d", total_proteins)
    logger.info("Total sites:              %d", total_sites)
    logger.info("Average sites per protein: %.1f", avg_sites)
    logger.info("Map saved to %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
