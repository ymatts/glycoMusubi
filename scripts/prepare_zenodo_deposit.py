#!/usr/bin/env python3
"""Prepare Zenodo deposit: package KG files, model checkpoints, and metadata.

Creates a compressed archive suitable for Zenodo upload.

Usage:
    python scripts/prepare_zenodo_deposit.py
    python scripts/prepare_zenodo_deposit.py --output-dir /tmp/zenodo_deposit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import tarfile
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def collect_files(staging_dir: Path) -> list[dict]:
    """Collect files for deposit and copy to staging directory."""
    manifest = []

    # ── 1. Knowledge Graph files ──
    kg_dir = staging_dir / "kg"
    kg_dir.mkdir(parents=True, exist_ok=True)

    kg_files = [
        "kg/nodes.tsv",
        "kg/nodes.parquet",
        "kg/edges.tsv",
        "kg/edges.parquet",
    ]
    for f in kg_files:
        src = PROJECT / f
        if src.exists():
            dst = kg_dir / src.name
            shutil.copy2(src, dst)
            manifest.append({
                "file": f"kg/{src.name}",
                "size_bytes": dst.stat().st_size,
                "sha256": sha256_file(dst),
                "description": f"Knowledge graph {src.suffix.upper()} file",
            })
            logger.info("  Copied: %s (%.1f MB)", f, dst.stat().st_size / 1e6)
        else:
            logger.warning("  Missing: %s", f)

    # ── 2. Clean data (edge and node tables) ──
    data_dir = staging_dir / "data_clean"
    data_dir.mkdir(parents=True, exist_ok=True)

    data_files = [
        "data_clean/glyconnect_site_glycans.tsv",
        "data_clean/edges_glycan_enzyme.tsv",
        "data_clean/ts_celltype_enzyme_expression.tsv",
        "data_clean/glycan_function_labels.tsv",
        "data_clean/glycowork_subgraph_pairs.tsv",
    ]
    for f in data_files:
        src = PROJECT / f
        if src.exists():
            dst = data_dir / src.name
            shutil.copy2(src, dst)
            manifest.append({
                "file": f"data_clean/{src.name}",
                "size_bytes": dst.stat().st_size,
                "sha256": sha256_file(dst),
            })
            logger.info("  Copied: %s", f)

    # ── 3. Model checkpoints ──
    models_dir = staging_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_files = [
        ("experiments_v2/glycokgnet_inductive_r2/best.pt", "Best KG embedding model"),
        ("experiments_v2/glycan_retrieval_v6b/model.pt", "Site-level retrieval model"),
    ]
    for f, desc in model_files:
        src = PROJECT / f
        if src.exists():
            dst = models_dir / src.name
            shutil.copy2(src, dst)
            manifest.append({
                "file": f"models/{src.name}",
                "size_bytes": dst.stat().st_size,
                "sha256": sha256_file(dst),
                "description": desc,
            })
            logger.info("  Copied: %s (%.1f MB)", f, dst.stat().st_size / 1e6)

    # ── 4. Result files ──
    results_dir = staging_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    result_files = [
        "experiments_v2/glycan_retrieval_v6b/results.json",
        "experiments_v2/glycan_retrieval_v3/results.json",
        "experiments_v2/celltype_conditioned_v2/results.json",
        "experiments_v2/pipeline_benchmark/results.json",
    ]
    for f in result_files:
        src = PROJECT / f
        if src.exists():
            dst = results_dir / src.name
            shutil.copy2(src, dst)
            manifest.append({
                "file": f"results/{src.name}",
                "size_bytes": dst.stat().st_size,
                "sha256": sha256_file(dst),
            })

    # ── 5. Paper tables and figures ──
    for subdir in ["paper_tables", "paper_figures"]:
        src_dir = PROJECT / "experiments_v2" / subdir
        if src_dir.exists():
            dst_dir = staging_dir / subdir
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
            for p in dst_dir.rglob("*"):
                if p.is_file():
                    manifest.append({
                        "file": f"{subdir}/{p.name}",
                        "size_bytes": p.stat().st_size,
                        "sha256": sha256_file(p),
                    })

    return manifest


def create_metadata(staging_dir: Path, manifest: list[dict]):
    """Create Zenodo metadata files."""
    metadata = {
        "title": "glycoMusubi: Knowledge Graph and Embedding Framework for Glycobiology",
        "description": (
            "Data deposit for the glycoMusubi paper. Contains the constructed knowledge graph "
            "(78K nodes, 2.5M edges, 10 node types, 14 relation types), trained model "
            "checkpoints, experiment results, and paper figures."
        ),
        "creators": [{"name": "Matsui Lab"}],
        "keywords": [
            "glycobiology", "knowledge graph", "graph embedding",
            "glycosylation", "bioinformatics",
        ],
        "license": "MIT",
        "upload_type": "dataset",
        "access_right": "open",
        "created": datetime.now().isoformat(),
        "files": manifest,
    }

    with open(staging_dir / "zenodo_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # README
    readme = (
        "# glycoMusubi Data Deposit\n\n"
        "## Contents\n\n"
        "- `kg/` — Knowledge graph (nodes.tsv, edges.tsv, Parquet)\n"
        "- `data_clean/` — Processed data tables\n"
        "- `models/` — Trained model checkpoints\n"
        "- `results/` — Experiment result files\n"
        "- `paper_tables/` — Paper tables (TSV + LaTeX)\n"
        "- `paper_figures/` — Paper figures (PDF + PNG)\n\n"
        "## Reproduction\n\n"
        "```bash\n"
        "git clone https://github.com/ymatts/glycoMusubi\n"
        "cd glycoMusubi\n"
        "python scripts/reproduce_paper.py --dry-run\n"
        "```\n\n"
        "## License\n\nMIT\n"
    )
    with open(staging_dir / "README.md", "w") as f:
        f.write(readme)


def main():
    parser = argparse.ArgumentParser(description="Prepare Zenodo deposit")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT / "experiments_v2/zenodo_deposit")
    args = parser.parse_args()

    output_dir = args.output_dir
    staging_dir = output_dir / "glycoMusubi_deposit"

    logger.info("Preparing Zenodo deposit...")
    logger.info("Output: %s", output_dir)

    # Clean staging area
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    # Collect files
    manifest = collect_files(staging_dir)
    logger.info("Collected %d files", len(manifest))

    # Create metadata
    create_metadata(staging_dir, manifest)
    logger.info("Metadata created")

    # Create tar.gz archive
    archive_path = output_dir / "glycoMusubi_deposit.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(staging_dir, arcname="glycoMusubi_deposit")
    logger.info("Archive: %s (%.1f MB)", archive_path, archive_path.stat().st_size / 1e6)

    # Total size
    total_size = sum(f["size_bytes"] for f in manifest)
    logger.info("Total deposit size: %.1f MB (%d files)", total_size / 1e6, len(manifest))


if __name__ == "__main__":
    main()
