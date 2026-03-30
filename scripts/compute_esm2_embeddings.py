#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute ESM-2 protein embeddings and cache them to disk.

Reads protein sequences from ``data_clean/protein_sequences.tsv``,
runs inference through the ``esm2_t33_650M_UR50D`` model, and saves
per-protein mean-pooled embeddings as individual ``.pt`` files under
``data_clean/esm2_cache/``.

Each file ``{index}.pt`` contains a 1-D tensor of shape ``[1280]``
(mean-pooled) or, when ``--per_residue`` is used, a 2-D tensor of
shape ``[L, 1280]`` in float16 (per-residue embeddings).
A mapping file ``id_to_idx.json`` maps UniProt accession IDs to the
integer row indices used as filenames.

Usage
-----
::

    python scripts/compute_esm2_embeddings.py
    python scripts/compute_esm2_embeddings.py --batch_size 8 --device cuda

Sequences longer than 1022 tokens are truncated to 1022 (the ESM-2
positional-embedding limit minus the two special tokens).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_TSV = PROJECT_ROOT / "data_clean" / "protein_sequences.tsv"
CACHE_DIR = PROJECT_ROOT / "data_clean" / "esm2_cache"
ID_MAP_FILE = CACHE_DIR / "id_to_idx.json"

ESM2_MODEL_NAME = "esm2_t33_650M_UR50D"
ESM2_DIM = 1280
MAX_SEQ_LEN = 1022  # ESM-2 max positions minus BOS/EOS tokens


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_device(requested: str) -> str:
    """Return a concrete device string (``cuda`` or ``cpu``)."""
    if requested == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return requested


def _load_sequences(tsv_path: Path) -> List[Tuple[int, str, str]]:
    """Load protein sequences from TSV.

    Returns
    -------
    list of (row_index, uniprot_id, sequence)
    """
    import pandas as pd

    if not tsv_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {tsv_path}\n"
            "Ensure data_clean/protein_sequences.tsv exists (columns: uniprot_id, sequence, length)."
        )

    df = pd.read_csv(tsv_path, sep="\t", dtype=str)

    required_cols = {"uniprot_id", "sequence"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Input TSV is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    records: List[Tuple[int, str, str]] = []
    for idx, row in df.iterrows():
        uid = str(row["uniprot_id"]).strip()
        seq = str(row["sequence"]).strip()
        if uid and seq and seq.lower() != "nan":
            records.append((int(idx), uid, seq))

    return records


def _load_existing_cache(cache_dir: Path) -> set:
    """Return the set of integer indices that already have cached embeddings."""
    cached: set = set()
    if not cache_dir.exists():
        return cached
    for pt_file in cache_dir.glob("*.pt"):
        stem = pt_file.stem
        try:
            cached.add(int(stem))
        except ValueError:
            continue
    return cached


# ---------------------------------------------------------------------------
# Embedding computation
# ---------------------------------------------------------------------------

def compute_embeddings(
    records: List[Tuple[int, str, str]],
    batch_size: int = 4,
    device: str = "cpu",
    cache_dir: Optional[Path] = None,
    save_per_residue: bool = False,
) -> int:
    """Compute and cache ESM-2 embeddings for the given protein records.

    Parameters
    ----------
    records : list of (index, uniprot_id, sequence)
        Proteins to embed. Sequences are truncated to ``MAX_SEQ_LEN``.
    batch_size : int
        Number of sequences per inference batch.
    device : str
        ``"cuda"`` or ``"cpu"``.
    cache_dir : Path, optional
        Output directory for cached embeddings.
    save_per_residue : bool
        If True, save per-residue ``[L, 1280]`` tensors in float16
        instead of mean-pooled ``[1280]`` tensors.

    Returns
    -------
    int
        Number of embeddings successfully computed and saved.
    """
    if cache_dir is None:
        cache_dir = CACHE_DIR
    try:
        import torch
        import esm
    except ImportError as exc:
        raise ImportError(
            "The 'esm' library is required but not installed.\n"
            "Install it with:  pip install fair-esm\n"
            "See https://github.com/facebookresearch/esm for details."
        ) from exc

    from tqdm import tqdm

    # Load model and alphabet
    logger.info("Loading ESM-2 model: %s", ESM2_MODEL_NAME)
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model = model.eval().to(device)

    # Number of layers for last-layer extraction
    num_layers = 33

    cache_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    num_batches = (len(records) + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_start in tqdm(
            range(0, len(records), batch_size),
            total=num_batches,
            desc="Computing ESM-2 embeddings",
            unit="batch",
        ):
            batch_records = records[batch_start : batch_start + batch_size]

            # Prepare data for the batch converter: list of (label, sequence)
            esm_data: List[Tuple[str, str]] = []
            batch_indices: List[int] = []
            for idx, uid, seq in batch_records:
                # Truncate to MAX_SEQ_LEN if needed
                truncated_seq = seq[:MAX_SEQ_LEN]
                esm_data.append((uid, truncated_seq))
                batch_indices.append(idx)

            _, _, batch_tokens = batch_converter(esm_data)
            batch_tokens = batch_tokens.to(device)

            results = model(batch_tokens, repr_layers=[num_layers], return_contacts=False)
            token_representations = results["representations"][num_layers]
            # token_representations shape: [B, L+2, 1280] (includes BOS and EOS tokens)

            for i, row_idx in enumerate(batch_indices):
                seq_len = len(esm_data[i][1])
                # Extract per-residue embeddings (skip BOS at position 0)
                per_residue = token_representations[i, 1 : seq_len + 1, :]  # [L, 1280]

                out_path = cache_dir / f"{row_idx}.pt"
                if save_per_residue:
                    torch.save(per_residue.cpu().to(torch.float16), out_path)
                else:
                    mean_emb = per_residue.mean(dim=0)  # [1280]
                    torch.save(mean_emb.cpu(), out_path)
                saved_count += 1

    return saved_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compute ESM-2 protein embeddings and cache to disk.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of sequences per inference batch (default: 4).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for inference (default: auto).",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(INPUT_TSV),
        help=f"Path to protein_sequences.tsv (default: {INPUT_TSV}).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(CACHE_DIR),
        help=f"Output cache directory (default: {CACHE_DIR}).",
    )
    parser.add_argument(
        "--per_residue",
        action="store_true",
        default=False,
        help="Save per-residue [L, 1280] embeddings in float16 instead of mean-pooled [1280].",
    )
    args = parser.parse_args(argv)

    device = _resolve_device(args.device)
    logger.info("Device: %s", device)

    input_path = Path(args.input)
    cache_dir = Path(args.output_dir)

    # Load sequences
    logger.info("Loading protein sequences from %s", input_path)
    all_records = _load_sequences(input_path)
    logger.info("Loaded %d protein sequences", len(all_records))

    if not all_records:
        logger.warning("No protein sequences found. Exiting.")
        return

    # Build and save id-to-index mapping
    cache_dir.mkdir(parents=True, exist_ok=True)
    id_to_idx: Dict[str, int] = {uid: idx for idx, uid, _ in all_records}
    id_map_path = cache_dir / "id_to_idx.json"
    with open(id_map_path, "w") as f:
        json.dump(id_to_idx, f, indent=2)
    logger.info("Saved id_to_idx mapping (%d entries) to %s", len(id_to_idx), id_map_path)

    # Check for existing cached embeddings (resume support)
    existing = _load_existing_cache(cache_dir)
    records_to_compute = [
        (idx, uid, seq) for idx, uid, seq in all_records if idx not in existing
    ]

    if existing:
        logger.info(
            "Found %d existing cached embeddings; %d remaining to compute",
            len(existing),
            len(records_to_compute),
        )

    if not records_to_compute:
        logger.info("All embeddings already cached. Nothing to do.")
        return

    # Compute embeddings
    t0 = time.time()
    num_saved = compute_embeddings(
        records=records_to_compute,
        batch_size=args.batch_size,
        device=device,
        cache_dir=cache_dir,
        save_per_residue=args.per_residue,
    )
    elapsed = time.time() - t0

    # Summary
    proteins_per_sec = num_saved / elapsed if elapsed > 0 else float("inf")
    logger.info("=" * 60)
    logger.info("ESM-2 embedding computation complete")
    logger.info("  Proteins embedded : %d", num_saved)
    logger.info("  Total time        : %.2f s", elapsed)
    logger.info("  Throughput        : %.2f proteins/sec", proteins_per_sec)
    logger.info("  Cache directory   : %s", cache_dir)
    logger.info("  ID mapping        : %s", id_map_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
