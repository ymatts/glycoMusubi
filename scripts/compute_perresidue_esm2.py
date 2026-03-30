#!/usr/bin/env python3
"""Compute per-residue ESM2 embeddings for N-linked glycosylation site prediction.

Only processes proteins with confirmed N-linked glycosylation sites.
Saves per-protein tensor files: data_clean/esm2_perresidue/{uid}.pt → [L, 1280]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    data_dir = Path("data_clean")
    output_dir = data_dir / "esm2_perresidue"
    output_dir.mkdir(exist_ok=True)

    # Load site data
    sites = pd.read_csv(data_dir / "uniprot_sites.tsv", sep="\t")
    n_sites = sites[sites["site_type"] == "N-linked"]
    target_uids = set(n_sites["uniprot_id"].unique())

    # Also include O-linked for future use
    o_sites = sites[sites["site_type"] == "O-linked"]
    target_uids.update(o_sites["uniprot_id"].unique())

    # Load sequences
    seqs_df = pd.read_csv(data_dir / "protein_sequences.tsv", sep="\t")
    seq_dict = dict(zip(seqs_df["uniprot_id"], seqs_df["sequence"]))

    # Find sequences for target proteins
    uid_seq_pairs = []
    for uid in sorted(target_uids):
        seq = seq_dict.get(f"{uid}-1", seq_dict.get(uid, None))
        if seq is None:
            continue
        # Skip if already computed
        out_path = output_dir / f"{uid}.pt"
        if out_path.exists():
            continue
        uid_seq_pairs.append((uid, seq))

    logger.info("Proteins to process: %d (skipping already computed)", len(uid_seq_pairs))
    if not uid_seq_pairs:
        logger.info("All done!")
        return

    # Load ESM2
    import esm
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model = model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info("ESM2 loaded on %s", device)

    # Process in batches
    max_len = 1022  # ESM2 max sequence length
    batch_size = 4  # conservative for per-residue (much more memory)

    for i in range(0, len(uid_seq_pairs), batch_size):
        batch_pairs = uid_seq_pairs[i:i + batch_size]
        batch_data = [(uid, seq[:max_len]) for uid, seq in batch_pairs]

        _, _, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            representations = results["representations"][33]  # [B, L+2, 1280]

        for j, (uid, seq) in enumerate(batch_pairs):
            L = min(len(seq), max_len)
            # Exclude BOS and EOS tokens
            per_residue = representations[j, 1:L + 1, :].cpu().half()  # [L, 1280] in float16
            out_path = output_dir / f"{uid}.pt"
            torch.save(per_residue, out_path)

        if (i // batch_size + 1) % 50 == 0:
            logger.info("Processed %d / %d proteins", min(i + batch_size, len(uid_seq_pairs)),
                         len(uid_seq_pairs))

    logger.info("Done! Per-residue embeddings saved to %s", output_dir)


if __name__ == "__main__":
    main()
