#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pipeline.py

Single entry point for the glycoMusubi ETL pipeline.
Orchestrates all stages: download -> clean -> build -> validate (+ label_audit)

Usage:
    python scripts/pipeline.py                    # Run full pipeline
    python scripts/pipeline.py --stage download   # Run specific stage
    python scripts/pipeline.py --stage clean build validate  # Run multiple stages
    python scripts/pipeline.py --stage label_audit  # Early label feasibility check
    python scripts/pipeline.py --auto-fix         # Enable auto-fix in validation
    python scripts/pipeline.py --dry-run          # Show what would be executed
    python scripts/pipeline.py --workers 4        # Use 4 worker processes
    python scripts/pipeline.py --no-parallel      # Disable multiprocessing
"""

import os
import sys
import argparse
import logging
from contextlib import contextmanager
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(__file__))

from utils.config_loader import load_config

config = load_config()

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", config.directories.logs)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "pipeline.log"))
    ]
)
logger = logging.getLogger(__name__)


STAGES = ['download', 'clean', 'build', 'validate', 'label_audit']


@contextmanager
def _isolated_argv(argv0: str):
    """
    Temporarily isolate sys.argv so sub-scripts with argparse do not
    accidentally parse pipeline.py arguments.
    """
    original_argv = sys.argv[:]
    sys.argv = [argv0]
    try:
        yield
    finally:
        sys.argv = original_argv


@dataclass
class PipelineResult:
    """Container for pipeline execution results."""
    stage: str
    success: bool
    duration_seconds: float
    message: str = ""
    error: Optional[str] = None


@dataclass
class PipelineReport:
    """Container for full pipeline execution report."""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    results: List[PipelineResult] = field(default_factory=list)
    
    def add_result(self, result: PipelineResult):
        self.results.append(result)
    
    @property
    def all_success(self) -> bool:
        return all(r.success for r in self.results)
    
    @property
    def total_duration(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return sum(r.duration_seconds for r in self.results)
    
    def to_text(self) -> str:
        lines = [
            "=" * 60,
            "GLYCO-KG PIPELINE EXECUTION REPORT",
            f"Start: {self.start_time.isoformat()}",
            f"End: {self.end_time.isoformat() if self.end_time else 'N/A'}",
            f"Status: {'SUCCESS' if self.all_success else 'FAILED'}",
            f"Total Duration: {self.total_duration:.2f}s",
            "=" * 60,
            "",
            "STAGE RESULTS:",
            "-" * 40,
        ]
        
        for r in self.results:
            status = "OK" if r.success else "FAILED"
            lines.append(f"  [{status}] {r.stage}: {r.duration_seconds:.2f}s")
            if r.message:
                lines.append(f"       {r.message}")
            if r.error:
                lines.append(f"       ERROR: {r.error}")
        
        return "\n".join(lines)


def run_download() -> PipelineResult:
    """Execute the download stage."""
    start = datetime.now()
    try:
        import download_data
        with _isolated_argv("download_data.py"):
            download_data.main()
        duration = (datetime.now() - start).total_seconds()
        return PipelineResult(
            stage="download",
            success=True,
            duration_seconds=duration,
            message="Data downloaded successfully"
        )
    except Exception as e:
        duration = (datetime.now() - start).total_seconds()
        logger.error(f"Download stage failed: {e}")
        return PipelineResult(
            stage="download",
            success=False,
            duration_seconds=duration,
            error=str(e)
        )


def run_clean() -> PipelineResult:
    """Execute the clean stage."""
    start = datetime.now()
    try:
        import clean_data
        with _isolated_argv("clean_data.py"):
            clean_data.main()
        duration = (datetime.now() - start).total_seconds()
        return PipelineResult(
            stage="clean",
            success=True,
            duration_seconds=duration,
            message="Data cleaned successfully"
        )
    except Exception as e:
        duration = (datetime.now() - start).total_seconds()
        logger.error(f"Clean stage failed: {e}")
        return PipelineResult(
            stage="clean",
            success=False,
            duration_seconds=duration,
            error=str(e)
        )


def run_build() -> PipelineResult:
    """Execute the build stage."""
    start = datetime.now()
    try:
        import build_kg
        with _isolated_argv("build_kg.py"):
            build_kg.main()
        duration = (datetime.now() - start).total_seconds()
        return PipelineResult(
            stage="build",
            success=True,
            duration_seconds=duration,
            message="Knowledge graph built successfully"
        )
    except Exception as e:
        duration = (datetime.now() - start).total_seconds()
        logger.error(f"Build stage failed: {e}")
        return PipelineResult(
            stage="build",
            success=False,
            duration_seconds=duration,
            error=str(e)
        )


def run_validate(auto_fix: bool = False) -> PipelineResult:
    """Execute the validate stage."""
    start = datetime.now()
    try:
        import validate_kg
        validate_kg.main(auto_fix=auto_fix)
        duration = (datetime.now() - start).total_seconds()
        return PipelineResult(
            stage="validate",
            success=True,
            duration_seconds=duration,
            message=f"Validation completed (auto_fix={auto_fix})"
        )
    except Exception as e:
        duration = (datetime.now() - start).total_seconds()
        logger.error(f"Validate stage failed: {e}")
        return PipelineResult(
            stage="validate",
            success=False,
            duration_seconds=duration,
            error=str(e)
        )


def run_label_audit() -> PipelineResult:
    """Execute quick label-data feasibility audit stage."""
    start = datetime.now()
    try:
        import label_data_audit
        with _isolated_argv("label_data_audit.py"):
            label_data_audit.main()
        duration = (datetime.now() - start).total_seconds()
        return PipelineResult(
            stage="label_audit",
            success=True,
            duration_seconds=duration,
            message="Label feasibility audit completed"
        )
    except Exception as e:
        duration = (datetime.now() - start).total_seconds()
        logger.error(f"Label audit stage failed: {e}")
        return PipelineResult(
            stage="label_audit",
            success=False,
            duration_seconds=duration,
            error=str(e)
        )


STAGE_RUNNERS = {
    'download': run_download,
    'clean': run_clean,
    'build': run_build,
    'validate': run_validate,
    'label_audit': run_label_audit,
}


def run_pipeline(
    stages: Optional[List[str]] = None,
    auto_fix: bool = False,
    dry_run: bool = False,
    stop_on_error: bool = True,
    parallel: Optional[bool] = None,
    workers: Optional[int] = None
) -> PipelineReport:
    """
    Run the glycoMusubi pipeline.
    
    Args:
        stages: List of stages to run. If None, runs all stages.
        auto_fix: Enable auto-fix in validation stage.
        dry_run: If True, only show what would be executed.
        stop_on_error: If True, stop pipeline on first error.
        parallel: Enable multiprocessing. If None, uses config default.
        workers: Number of worker processes. If None, auto-detects.
    
    Returns:
        PipelineReport with execution results.
    """
    if stages is None:
        stages = STAGES
    
    for stage in stages:
        if stage not in STAGES:
            raise ValueError(f"Unknown stage: {stage}. Valid stages: {STAGES}")
    
    report = PipelineReport()
    
    logger.info("=" * 60)
    logger.info("GLYCO-KG PIPELINE STARTING")
    logger.info(f"Stages: {stages}")
    logger.info(f"Auto-fix: {auto_fix}")
    logger.info(f"Dry-run: {dry_run}")
    logger.info(f"Parallel: {parallel}")
    logger.info(f"Workers: {workers if workers else 'auto'}")
    logger.info("=" * 60)
    
    if dry_run:
        logger.info("DRY RUN - No stages will be executed")
        for stage in stages:
            logger.info(f"  Would run: {stage}")
        return report
    
    for stage in stages:
        logger.info(f"\n{'='*40}")
        logger.info(f"STAGE: {stage.upper()}")
        logger.info(f"{'='*40}")
        
        if stage == 'validate':
            result = run_validate(auto_fix=auto_fix)
        else:
            runner = STAGE_RUNNERS[stage]
            result = runner()
        
        report.add_result(result)
        
        if result.success:
            logger.info(f"Stage {stage} completed successfully in {result.duration_seconds:.2f}s")
        else:
            logger.error(f"Stage {stage} failed: {result.error}")
            if stop_on_error:
                logger.error("Stopping pipeline due to error")
                break
    
    report.end_time = datetime.now()
    
    report_path = os.path.join(LOG_DIR, "pipeline_report.txt")
    with open(report_path, 'w') as f:
        f.write(report.to_text())
    
    logger.info("\n" + report.to_text())
    logger.info(f"\nPipeline report saved to: {report_path}")
    
    return report


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="glycoMusubi ETL Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/pipeline.py                     # Run full pipeline
  python scripts/pipeline.py --stage download    # Run only download
  python scripts/pipeline.py --stage clean build # Run clean and build
  python scripts/pipeline.py --stage label_audit # Quick label feasibility check
  python scripts/pipeline.py --auto-fix          # Enable auto-fix in validation
  python scripts/pipeline.py --dry-run           # Show what would be executed
  python scripts/pipeline.py --no-stop-on-error  # Continue on errors
        """
    )
    
    parser.add_argument(
        '--stage', '-s',
        nargs='+',
        choices=STAGES,
        help=f"Stages to run (default: all). Choices: {STAGES}"
    )
    
    parser.add_argument(
        '--auto-fix',
        action='store_true',
        help="Enable auto-fix in validation stage"
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Show what would be executed without running"
    )
    
    parser.add_argument(
        '--no-stop-on-error',
        action='store_true',
        help="Continue pipeline even if a stage fails"
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count or config.parallel.workers)"
    )
    
    parser.add_argument(
        '--parallel',
        dest='parallel',
        action='store_true',
        default=None,
        help="Enable multiprocessing (default)"
    )
    
    parser.add_argument(
        '--no-parallel',
        dest='parallel',
        action='store_false',
        help="Disable multiprocessing"
    )
    
    args = parser.parse_args()
    
    # Determine effective parallel settings
    parallel_enabled = args.parallel
    if parallel_enabled is None:
        parallel_enabled = config.parallel.enabled
    
    workers = args.workers
    if workers is None:
        workers = config.parallel.workers
    
    # Set environment variables for sub-scripts
    import os
    os.environ['GLYCO_KG_PARALLEL'] = str(parallel_enabled).lower()
    if workers is not None:
        os.environ['GLYCO_KG_WORKERS'] = str(workers)
    os.environ['GLYCO_KG_PARALLEL_SEED'] = str(config.parallel.seed)
    os.environ['GLYCO_KG_BATCH_SIZE'] = str(config.parallel.batch_size)
    
    report = run_pipeline(
        stages=args.stage,
        auto_fix=args.auto_fix,
        dry_run=args.dry_run,
        stop_on_error=not args.no_stop_on_error,
        parallel=parallel_enabled,
        workers=workers
    )
    
    sys.exit(0 if report.all_success else 1)


if __name__ == "__main__":
    main()
