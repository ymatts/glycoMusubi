"""Precompute WURCS feature vectors for all glycans.

Phase 1 of the glycoMusubi inductive prediction system.  Reads glycan
structures from ``data_clean/glycans_clean.tsv``, extracts a 24-dimensional
biochemical feature vector for each glycan via
:func:`glycoMusubi.embedding.encoders.glycan_encoder.extract_wurcs_features`,
and saves the result as a torch tensor cache for downstream embedding and
prediction tasks.

Output
------
``data_clean/wurcs_features_cache.pt`` containing:
  - ``"features"``: Tensor of shape ``[num_glycans, 24]``
  - ``"glycan_ids"``: list of glycan ID strings
  - ``"id_to_idx"``: dict mapping glycan_id -> integer index
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Ensure project root is on the import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from tqdm import tqdm

from glycoMusubi.embedding.encoders.glycan_encoder import extract_wurcs_features

logger = logging.getLogger(__name__)

# Paths (relative to project root)
INPUT_PATH = PROJECT_ROOT / "data_clean" / "glycans_clean.tsv"
OUTPUT_PATH = PROJECT_ROOT / "data_clean" / "wurcs_features_cache.pt"

FEATURE_DIM = 24


def main() -> None:
    """Read glycan structures, extract WURCS features, and save cache."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("Reading glycan data from %s", INPUT_PATH)
    if not INPUT_PATH.exists():
        logger.error("Input file not found: %s", INPUT_PATH)
        raise FileNotFoundError(INPUT_PATH)

    df = pd.read_csv(INPUT_PATH, sep="\t")
    logger.info("Loaded %d glycans", len(df))

    if "glycan_id" not in df.columns or "structure" not in df.columns:
        raise ValueError(
            f"Expected columns 'glycan_id' and 'structure', "
            f"got {list(df.columns)}"
        )

    glycan_ids: list[str] = []
    feature_vectors: list[torch.Tensor] = []
    id_to_idx: dict[str, int] = {}
    non_zero_count = 0

    for idx, row in tqdm(
        df.iterrows(), total=len(df), desc="Extracting WURCS features"
    ):
        gid = str(row["glycan_id"])
        wurcs = str(row["structure"]) if pd.notna(row["structure"]) else ""

        feat = extract_wurcs_features(wurcs)

        glycan_ids.append(gid)
        feature_vectors.append(feat)
        id_to_idx[gid] = len(glycan_ids) - 1

        if feat.abs().sum().item() > 0:
            non_zero_count += 1

    # Stack into a single tensor [num_glycans, 24]
    features_tensor = torch.stack(feature_vectors, dim=0)

    # Save cache
    cache = {
        "features": features_tensor,
        "glycan_ids": glycan_ids,
        "id_to_idx": id_to_idx,
    }

    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
    torch.save(cache, OUTPUT_PATH)

    # Report
    total = len(glycan_ids)
    coverage = (non_zero_count / total * 100) if total > 0 else 0.0
    logger.info("Glycans processed:        %d", total)
    logger.info("Non-zero feature vectors: %d", non_zero_count)
    logger.info("Coverage:                 %.1f%%", coverage)
    logger.info("Feature tensor shape:     %s", list(features_tensor.shape))
    logger.info("Cache saved to %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
