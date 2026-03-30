#!/usr/bin/env python3
"""Compare baseline metrics across experiments.

Supports both:
- downstream metrics (`downstream_metrics.json`)
- link-prediction metrics (`metrics.json`)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DEFAULT_EXPERIMENTS = ["baseline_transe", "baseline_distmult", "baseline_rotate"]
DEFAULT_TASKS = [
    "glycan_protein_interaction",
    "drug_target_identification",
    "binding_site_prediction",
    "disease_association_prediction",
    "glycan_function_prediction",
    "immunogenicity",
]

# Task-specific priority metrics to display first.
PRIMARY_METRICS = {
    "glycan_protein_interaction": ["auc_roc", "auc_pr", "f1_optimal"],
    "drug_target_identification": ["auc_roc", "enrichment_factor@1%"],
    "binding_site_prediction": ["residue_auc", "site_f1"],
    "disease_association_prediction": ["auc_roc", "recall@10", "ndcg@10"],
    "glycan_function_prediction": ["mean_accuracy", "mean_f1", "mean_mcc"],
    "immunogenicity": ["auc_roc", "auc_pr", "f1_optimal"],
}

LINKPRED_SECTIONS = ("metrics", "head_metrics", "tail_metrics")
PRIMARY_LINKPRED = {
    "metrics": ["mrr", "hits@1", "hits@3", "hits@10", "mr"],
    "head_metrics": ["mrr", "hits@1", "hits@3", "hits@10", "mr"],
    "tail_metrics": ["mrr", "hits@1", "hits@3", "hits@10", "mr"],
}

# Heuristic grade thresholds (metric-specific) for quick quality checks.
# "higher is better" by default. Value ranges are interpreted as:
# - >= high: good
# - >= low: normal
# - else: low
METRIC_GRADE_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "mrr": (0.28, 0.35),
    "hits@1": (0.20, 0.30),
    "hits@3": (0.28, 0.38),
    "hits@10": (0.38, 0.48),
    # MR is smaller is better.
    "mr": (1800.0, 1400.0),

    "auc_roc": (0.85, 0.95),
    "auc_pr": (0.80, 0.90),
    "f1_optimal": (0.70, 0.85),
    "recall@10": (0.65, 0.80),
    "ndcg@10": (0.65, 0.80),
    "recall@20": (0.65, 0.80),
    "ndcg@20": (0.65, 0.80),
    "recall@50": (0.75, 0.90),
    "ndcg@50": (0.75, 0.90),
    "residue_auc": (0.85, 0.93),
    "site_f1": (0.70, 0.85),
    "mean_accuracy": (0.80, 0.90),
    "mean_f1": (0.75, 0.85),
    "mean_mcc": (0.70, 0.85),
    "enrichment_factor@1%": (3.0, 8.0),
}

LOWER_BETTER_METRICS = {"mr"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("experiments"),
        help="Root directory that contains experiment outputs.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=DEFAULT_EXPERIMENTS,
        help="Experiment names to compare.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
        help="Downstream tasks to include (used in downstream mode).",
    )
    parser.add_argument(
        "--input-name",
        default="downstream_metrics.json",
        help="Metrics JSON file name inside each experiment directory.",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "downstream", "linkpred"),
        default="auto",
        help="Comparison mode. auto infers from JSON payload shape.",
    )
    parser.add_argument(
        "--out-tsv",
        type=Path,
        default=Path("logs/baseline_comparison.tsv"),
        help="Output TSV path.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("logs/baseline_comparison.md"),
        help="Output Markdown path.",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Append per-metric good/normal/low grade columns using threshold-based heuristics.",
    )
    return parser.parse_args()


def load_metrics(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload if isinstance(payload, dict) else {}


def infer_mode(payloads: Iterable[Dict], requested_mode: str) -> str:
    if requested_mode != "auto":
        return requested_mode
    for p in payloads:
        if not isinstance(p, dict) or not p:
            continue
        if "metrics" in p and isinstance(p.get("metrics"), dict):
            return "linkpred"
        # downstream payload usually contains task dictionaries
        if any(isinstance(v, dict) for v in p.values()):
            return "downstream"
    return "downstream"


def downstream_metric_columns(tasks: Iterable[str], payloads: Iterable[Dict[str, Dict]]) -> List[Tuple[str, str]]:
    cols: List[Tuple[str, str]] = []
    seen = set()
    for task in tasks:
        for metric in PRIMARY_METRICS.get(task, []):
            key = (task, metric)
            if key not in seen:
                cols.append(key)
                seen.add(key)
        # Add remaining metrics found in payloads.
        extras = set()
        for data in payloads:
            task_payload = data.get(task, {})
            if isinstance(task_payload, dict):
                extras.update(task_payload.keys())
        for metric in sorted(extras):
            key = (task, metric)
            if key not in seen:
                cols.append(key)
                seen.add(key)
    return cols


def linkpred_metric_columns(payloads: Iterable[Dict]) -> List[Tuple[str, str]]:
    cols: List[Tuple[str, str]] = []
    seen = set()
    for section in LINKPRED_SECTIONS:
        for metric in PRIMARY_LINKPRED.get(section, []):
            key = (section, metric)
            if key not in seen:
                cols.append(key)
                seen.add(key)
        extras = set()
        for p in payloads:
            sec = p.get(section, {})
            if isinstance(sec, dict):
                extras.update(sec.keys())
        for metric in sorted(extras):
            key = (section, metric)
            if key not in seen:
                cols.append(key)
                seen.add(key)

    # Keep num_triples at end if present.
    if any("num_triples" in p for p in payloads):
        cols.append(("summary", "num_triples"))
    return cols


def format_val(v) -> str:
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def _to_float(value) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def classify_metric(metric_name: str, value) -> str:
    """Return a quick quality grade for a metric value."""
    normalized = _to_float(value)
    if normalized is None:
        return "NA"

    thresholds = METRIC_GRADE_THRESHOLDS.get(metric_name)
    if thresholds is None:
        return "normal"

    low, high = thresholds
    if metric_name in LOWER_BETTER_METRICS:
        if normalized <= high:
            return "good"
        if normalized <= low:
            return "normal"
        return "low"

    if normalized >= high:
        return "good"
    if normalized >= low:
        return "normal"
    return "low"


def to_markdown(headers: List[str], rows: List[List[str]]) -> str:
    sep = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    payload_by_exp: Dict[str, Dict] = {}
    for exp in args.experiments:
        in_path = args.experiments_dir / exp / args.input_name
        payload_by_exp[exp] = load_metrics(in_path)

    mode = infer_mode(payload_by_exp.values(), args.mode)
    if mode == "linkpred":
        cols = linkpred_metric_columns(payload_by_exp.values())
    else:
        cols = downstream_metric_columns(args.tasks, payload_by_exp.values())
    headers = ["experiment"] + [f"{scope}.{metric}" for scope, metric in cols]
    if args.judge:
        headers.extend([f"{scope}.{metric}_grade" for scope, metric in cols])
    rows: List[List[str]] = []
    grade_summary: Dict[str, Dict[str, int]] = {}

    for exp in args.experiments:
        payload = payload_by_exp.get(exp, {})
        row = [exp]
        grade_row: List[str] = []
        good_cnt = 0
        normal_cnt = 0
        low_cnt = 0
        for scope, metric in cols:
            if mode == "linkpred":
                if scope == "summary" and metric == "num_triples":
                    value = payload.get("num_triples", "NA")
                    row.append(format_val(value))
                    if args.judge:
                        grade_row.append("NA")
                    continue
                sec = payload.get(scope, {})
                if isinstance(sec, dict) and metric in sec:
                    value = sec[metric]
                    row.append(format_val(value))
                    if args.judge:
                        grade = classify_metric(metric, value)
                        grade_row.append(grade)
                        if grade == "good":
                            good_cnt += 1
                        elif grade == "normal":
                            normal_cnt += 1
                        elif grade == "low":
                            low_cnt += 1
                else:
                    row.append("NA")
                    if args.judge:
                        grade_row.append("NA")
            else:
                task_payload = payload.get(scope, {})
                if isinstance(task_payload, dict) and metric in task_payload:
                    value = task_payload[metric]
                    row.append(format_val(value))
                    if args.judge:
                        grade = classify_metric(metric, value)
                        grade_row.append(grade)
                        if grade == "good":
                            good_cnt += 1
                        elif grade == "normal":
                            normal_cnt += 1
                        elif grade == "low":
                            low_cnt += 1
                else:
                    row.append("NA")
                    if args.judge:
                        grade_row.append("NA")
        rows.append(row + grade_row)
        grade_summary[exp] = {
            "good": good_cnt,
            "normal": normal_cnt,
            "low": low_cnt,
        }

    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)

    with args.out_tsv.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(headers) + "\n")
        for row in rows:
            fh.write("\t".join(row) + "\n")

    with args.out_md.open("w", encoding="utf-8") as fh:
        fh.write(to_markdown(headers, rows))

    print(f"Saved TSV: {args.out_tsv}")
    print(f"Saved MD : {args.out_md}")
    print("Mode:", mode)
    print("Experiments:", ", ".join(args.experiments))
    if args.judge:
        print("Judgment summary:")
        for exp in args.experiments:
            stats = grade_summary.get(exp, {"good": 0, "normal": 0, "low": 0})
            print(f"  {exp}: good={stats['good']} normal={stats['normal']} low={stats['low']}")


if __name__ == "__main__":
    main()
