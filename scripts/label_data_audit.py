#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Quick feasibility audit for downstream label data.

This script is designed for early go/no-go decisions before long training runs.
It checks whether immunogenicity / glycan-function labels are realistically
obtainable from currently available local data files.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd


GLYTOUCAN_RE = re.compile(r"^G\d{5}[A-Z]{2}$")
DEFAULT_GLYCANML_URL = (
    "https://torchglycan.s3.us-east-2.amazonaws.com/downstream/glycan_immunogenicity.csv"
)


@dataclass
class TaskAudit:
    task: str
    status: str
    reason: str
    raw_rows: int = 0
    usable_rows: int = 0
    mapped_rows: int = 0
    in_kg_rows: int = 0
    notes: str = ""


def _load_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", low_memory=False)


def _norm_text(v) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip()


def _norm_iupac(v) -> str:
    return "".join(_norm_text(v).split()).lower()


def _pick_column(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    cols = set(columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def _parse_binary(v) -> Optional[int]:
    if pd.isna(v):
        return None
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "positive", "pos", "immunogenic"}:
        return 1
    if s in {"0", "false", "no", "negative", "neg", "non-immunogenic", "non_immunogenic"}:
        return 0
    try:
        x = float(s)
        if x == 1.0:
            return 1
        if x == 0.0:
            return 0
    except ValueError:
        pass
    return None


def _load_whitelist(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = _load_tsv(path)
    if "glycan_id" not in df.columns:
        return set()
    return {str(x).strip() for x in df["glycan_id"].dropna().astype(str)}


def _load_mapping_tables(paths: List[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        if not path.exists():
            continue
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path, low_memory=False)
        else:
            df = pd.read_csv(path, sep="\t", low_memory=False)

        glytoucan_col = _pick_column(df.columns, ("glytoucan_id", "glytoucan_ac", "glycan_id"))
        if not glytoucan_col:
            continue

        source_id_col = _pick_column(df.columns, ("source_id", "glycanml_id", "id", "glycan_id"))
        source_iupac_col = _pick_column(df.columns, ("source_iupac", "iupac", "glycan", "iupac_condensed"))

        for _, r in df.iterrows():
            gid = _norm_text(r.get(glytoucan_col, ""))
            if not gid:
                continue
            rows.append(
                {
                    "source_id": _norm_text(r.get(source_id_col, "")) if source_id_col else "",
                    "source_iupac_norm": _norm_iupac(r.get(source_iupac_col, "")) if source_iupac_col else "",
                    "glytoucan_id": gid,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["source_id", "source_iupac_norm", "glytoucan_id"])
    return pd.DataFrame(rows).drop_duplicates()


def audit_immunogenicity(
    glycanml_path: Path,
    mapping_paths: List[Path],
    whitelist: set[str],
    min_rows: int,
) -> TaskAudit:
    if not glycanml_path.exists():
        return TaskAudit(
            task="immunogenicity",
            status="BLOCKED",
            reason=f"Missing source file: {glycanml_path}",
            notes=f"Download from {DEFAULT_GLYCANML_URL}",
        )

    src = pd.read_csv(glycanml_path, low_memory=False)
    raw_rows = len(src)

    id_col = _pick_column(src.columns, ("glycan_id", "source_id", "id"))
    iupac_col = _pick_column(src.columns, ("glycan", "iupac", "iupac_condensed"))
    label_col = _pick_column(src.columns, ("immunogenicity", "label", "y", "target", "class"))

    if not label_col:
        return TaskAudit(
            task="immunogenicity",
            status="BLOCKED",
            reason="No label column found in GlycanML file",
            raw_rows=raw_rows,
        )

    src["label"] = src[label_col].map(_parse_binary)
    src["source_id"] = src[id_col].map(_norm_text) if id_col else ""
    src["source_iupac_norm"] = src[iupac_col].map(_norm_iupac) if iupac_col else ""
    src = src[src["label"].isin([0, 1])].copy()
    usable_rows = len(src)

    # Direct GlyTouCan IDs in source (rare for GlycanML, but check anyway).
    direct = 0
    if id_col:
        direct = int(src["source_id"].str.match(GLYTOUCAN_RE).sum())

    mapping = _load_mapping_tables(mapping_paths)
    by_id = mapping[mapping["source_id"] != ""][["source_id", "glytoucan_id"]].drop_duplicates()
    by_iupac = mapping[mapping["source_iupac_norm"] != ""][["source_iupac_norm", "glytoucan_id"]].drop_duplicates()

    m = src.merge(by_id, on="source_id", how="left")
    m = m.rename(columns={"glytoucan_id": "glytoucan_from_id"})
    m = m.merge(by_iupac, on="source_iupac_norm", how="left")
    m = m.rename(columns={"glytoucan_id": "glytoucan_from_iupac"})
    m["glytoucan_id"] = m["glytoucan_from_id"]
    fill_mask = m["glytoucan_id"].isna() | (m["glytoucan_id"] == "")
    m.loc[fill_mask, "glytoucan_id"] = m.loc[fill_mask, "glytoucan_from_iupac"]
    m["glytoucan_id"] = m["glytoucan_id"].map(_norm_text)

    mapped_rows = int((m["glytoucan_id"] != "").sum()) + direct
    if whitelist:
        in_kg_rows = int(m["glytoucan_id"].isin(whitelist).sum())
    else:
        in_kg_rows = int((m["glytoucan_id"] != "").sum())

    if in_kg_rows >= min_rows:
        return TaskAudit(
            task="immunogenicity",
            status="READY",
            reason=f"Sufficient mapped labels in KG ({in_kg_rows} >= {min_rows})",
            raw_rows=raw_rows,
            usable_rows=usable_rows,
            mapped_rows=mapped_rows,
            in_kg_rows=in_kg_rows,
            notes=f"mapping_tables={sum(int(p.exists()) for p in mapping_paths)}",
        )
    if in_kg_rows > 0:
        return TaskAudit(
            task="immunogenicity",
            status="PARTIAL",
            reason=f"Some labels mapped but below target ({in_kg_rows} < {min_rows})",
            raw_rows=raw_rows,
            usable_rows=usable_rows,
            mapped_rows=mapped_rows,
            in_kg_rows=in_kg_rows,
            notes="Expand mapping table or relax minimum threshold.",
        )
    return TaskAudit(
        task="immunogenicity",
        status="BLOCKED",
        reason="No labels mapped to GlyTouCan IDs in current KG",
        raw_rows=raw_rows,
        usable_rows=usable_rows,
        mapped_rows=mapped_rows,
        in_kg_rows=in_kg_rows,
        notes="Manual/curated mapping (source_id or IUPAC -> GlyTouCan) is required.",
    )


def audit_glycan_function(glycan_details_path: Path, whitelist: set[str], min_rows: int) -> TaskAudit:
    if not glycan_details_path.exists():
        return TaskAudit(
            task="glycan_function",
            status="BLOCKED",
            reason=f"Missing source file: {glycan_details_path}",
        )

    with glycan_details_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    if isinstance(payload, dict):
        items = payload.items()
    elif isinstance(payload, list):
        items = ((str(x.get("glycan_id", "")), x) for x in payload if isinstance(x, dict))
    else:
        return TaskAudit(task="glycan_function", status="BLOCKED", reason="Unsupported glycan_details format")

    rows = []
    total = 0
    for gid, detail in items:
        if not gid or not isinstance(detail, dict):
            continue
        total += 1
        cls = detail.get("classification", [])
        if not isinstance(cls, list):
            continue
        for c in cls:
            if not isinstance(c, dict):
                continue
            ctype = c.get("type", {})
            if isinstance(ctype, dict):
                name = _norm_text(ctype.get("name", ""))
                cid = _norm_text(ctype.get("id", ""))
            else:
                name = _norm_text(ctype)
                cid = ""
            if not name:
                continue
            rows.append(
                {
                    "glycan_id": gid,
                    "function_term": name,
                    "function_id": cid,
                }
            )

    if not rows:
        return TaskAudit(
            task="glycan_function",
            status="BLOCKED",
            reason="No classification/function annotations found in glycan_details",
            raw_rows=total,
        )

    df = pd.DataFrame(rows).drop_duplicates(subset=["glycan_id", "function_term"])
    raw_rows = len(df)
    in_kg_rows = int(df["glycan_id"].isin(whitelist).sum()) if whitelist else raw_rows

    if in_kg_rows >= min_rows:
        status = "READY"
        reason = f"Sufficient function labels in KG ({in_kg_rows} >= {min_rows})"
    elif in_kg_rows > 0:
        status = "PARTIAL"
        reason = f"Function labels exist but below target ({in_kg_rows} < {min_rows})"
    else:
        status = "BLOCKED"
        reason = "Function labels do not overlap with current KG glycan IDs"

    return TaskAudit(
        task="glycan_function",
        status=status,
        reason=reason,
        raw_rows=raw_rows,
        usable_rows=raw_rows,
        mapped_rows=raw_rows,
        in_kg_rows=in_kg_rows,
        notes=f"unique_terms={df['function_term'].nunique()}",
    )


def _write_markdown(path: Path, audits: List[TaskAudit]) -> None:
    headers = ["task", "status", "reason", "raw_rows", "usable_rows", "mapped_rows", "in_kg_rows", "notes"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for a in audits:
        vals = [str(getattr(a, h)) for h in headers]
        lines.append("| " + " | ".join(vals) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Early feasibility audit for label data.")
    parser.add_argument(
        "--glycanml-input",
        default="data_raw/glycan_immunogenicity_glycanml.csv",
        help="Path to GlycanML immunogenicity CSV",
    )
    parser.add_argument(
        "--mapping-input",
        action="append",
        default=[],
        help="Mapping tables for GlycanML -> GlyTouCan (repeatable)",
    )
    parser.add_argument(
        "--glycan-details-input",
        default="data_raw/glygen_glycan_details.json",
        help="Path to GlyGen glycan details JSON",
    )
    parser.add_argument(
        "--glycan-whitelist",
        default="data_clean/glycans_clean.tsv",
        help="Path to KG glycan whitelist",
    )
    parser.add_argument(
        "--min-immunogenicity-rows",
        type=int,
        default=200,
        help="Threshold for READY status in immunogenicity",
    )
    parser.add_argument(
        "--min-function-rows",
        type=int,
        default=1000,
        help="Threshold for READY status in glycan_function",
    )
    parser.add_argument(
        "--json-out",
        default="logs/label_data_audit.json",
        help="JSON audit report path",
    )
    parser.add_argument(
        "--md-out",
        default="logs/label_data_audit.md",
        help="Markdown audit report path",
    )
    args = parser.parse_args()

    mapping_paths = [Path(p) for p in args.mapping_input]
    if not mapping_paths:
        mapping_paths = [Path("data_clean/glycanml_to_glytoucan.tsv")]

    whitelist = _load_whitelist(Path(args.glycan_whitelist))
    audits = [
        audit_immunogenicity(
            glycanml_path=Path(args.glycanml_input),
            mapping_paths=mapping_paths,
            whitelist=whitelist,
            min_rows=args.min_immunogenicity_rows,
        ),
        audit_glycan_function(
            glycan_details_path=Path(args.glycan_details_input),
            whitelist=whitelist,
            min_rows=args.min_function_rows,
        ),
    ]

    report = {
        "summary": {
            "ready_tasks": [a.task for a in audits if a.status == "READY"],
            "partial_tasks": [a.task for a in audits if a.status == "PARTIAL"],
            "blocked_tasks": [a.task for a in audits if a.status == "BLOCKED"],
        },
        "tasks": [asdict(a) for a in audits],
    }

    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_markdown(Path(args.md_out), audits)

    print("Label data feasibility audit")
    for a in audits:
        print(f"- {a.task}: {a.status} | {a.reason}")
        print(
            f"  rows(raw/usable/mapped/in_kg)=({a.raw_rows}/{a.usable_rows}/{a.mapped_rows}/{a.in_kg_rows})"
        )
        if a.notes:
            print(f"  note: {a.notes}")
    print(f"Saved JSON: {json_out}")
    print(f"Saved MD  : {args.md_out}")


if __name__ == "__main__":
    main()
