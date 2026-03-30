#!/usr/bin/env python3
"""Reproduce all paper experiments in sequence.

Usage:
    python scripts/reproduce_paper.py              # Run all stages
    python scripts/reproduce_paper.py --dry-run     # Print commands only
    python scripts/reproduce_paper.py --stage 3     # Run from stage 3
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parent.parent

STAGES = [
    {
        "name": "KG construction",
        "command": "python scripts/pipeline.py",
        "description": "Build knowledge graph from public databases",
        "expected_output": "kg/nodes.tsv",
    },
    {
        "name": "KG embedding (GlycoKGNet R2)",
        "command": (
            "python scripts/embedding_pipeline.py"
            " -e glycokgnet_inductive_r2"
            " --output-dir experiments_v2/glycokgnet_inductive_r2"
        ),
        "description": "Train best KG embedding model (inductive)",
        "expected_output": "experiments_v2/glycokgnet_inductive_r2/best.pt",
    },
    {
        "name": "N-linked site classification",
        "command": "python scripts/glycan_class_predictor_v3.py",
        "description": "ESM-2 MLP for N-linked/O-linked classification",
        "expected_output": "experiments_v2/glycan_class_v3",
    },
    {
        "name": "Site ranking",
        "command": "python scripts/nlinked_site_predictor_v5.py",
        "description": "ESM-2 ranking model for glycosylation sites",
        "expected_output": "experiments_v2/nlinked_site_v5",
    },
    {
        "name": "Glycan retrieval (protein-level)",
        "command": "python scripts/glycan_retrieval_v3.py",
        "description": "Protein-level glycan retrieval with function matching",
        "expected_output": "experiments_v2/glycan_retrieval_v3/results.json",
    },
    {
        "name": "Glycan retrieval (site-level)",
        "command": "python scripts/glycan_retrieval_v6b_nlinked.py",
        "description": "Site-level N-linked glycan retrieval with clustering",
        "expected_output": "experiments_v2/glycan_retrieval_v6b/results.json",
    },
    {
        "name": "Cell-type conditioned prediction",
        "command": "python scripts/celltype_conditioned_glycan_v2.py",
        "description": "Cascade model with cell-type enzyme expression",
        "expected_output": "experiments_v2/celltype_conditioned_v2/results.json",
    },
    {
        "name": "Consolidated benchmark",
        "command": "python scripts/consolidated_benchmark.py",
        "description": "Aggregate results into paper tables",
        "expected_output": "experiments_v2/paper_tables/table2_hierarchical.tsv",
    },
    {
        "name": "Figure generation",
        "command": "python scripts/paper_figures/generate_all.py",
        "description": "Generate all paper figures",
        "expected_output": "experiments_v2/paper_figures/fig1_kg_benchmark.pdf",
    },
]


def run_stage(stage: dict, dry_run: bool = False) -> bool:
    """Run a single stage. Returns True on success."""
    logger.info("=" * 60)
    logger.info("Stage: %s", stage["name"])
    logger.info("  %s", stage["description"])
    logger.info("  Command: %s", stage["command"])

    if dry_run:
        logger.info("  [DRY RUN] Skipped")
        return True

    start = time.time()
    try:
        result = subprocess.run(
            stage["command"],
            shell=True,
            cwd=str(PROJECT),
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout per stage
        )
        elapsed = time.time() - start

        if result.returncode == 0:
            logger.info("  Completed in %.1f seconds", elapsed)
            # Check expected output
            expected = PROJECT / stage["expected_output"]
            if expected.exists():
                logger.info("  Output verified: %s", stage["expected_output"])
            else:
                logger.warning("  Expected output missing: %s", stage["expected_output"])
            return True
        else:
            logger.error("  FAILED (exit code %d)", result.returncode)
            if result.stderr:
                for line in result.stderr.strip().split("\n")[-10:]:
                    logger.error("    %s", line)
            return False

    except subprocess.TimeoutExpired:
        logger.error("  TIMEOUT after 2 hours")
        return False
    except Exception as e:
        logger.error("  ERROR: %s", e)
        return False


def main():
    parser = argparse.ArgumentParser(description="Reproduce paper experiments")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--stage", type=int, default=1, help="Start from stage N (1-indexed)")
    args = parser.parse_args()

    logger.info("glycoMusubi Paper Reproduction Pipeline")
    logger.info("Stages: %d total, starting from stage %d", len(STAGES), args.stage)
    if args.dry_run:
        logger.info("Mode: DRY RUN")

    results = []
    for i, stage in enumerate(STAGES, 1):
        if i < args.stage:
            logger.info("Skipping stage %d: %s", i, stage["name"])
            continue

        success = run_stage(stage, dry_run=args.dry_run)
        results.append((stage["name"], success))

        if not success and not args.dry_run:
            logger.error("Pipeline stopped at stage %d. Use --stage %d to resume.", i, i)
            break

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for name, success in results:
        status = "OK" if success else "FAILED"
        logger.info("  [%s] %s", status, name)

    n_ok = sum(1 for _, s in results if s)
    logger.info("\n%d/%d stages completed successfully", n_ok, len(results))

    return 0 if all(s for _, s in results) else 1


if __name__ == "__main__":
    sys.exit(main())
