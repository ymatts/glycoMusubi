#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Prepare glycan immunogenicity labels for glycoMusubi.

This utility converts a source table (e.g. from GlycanML assets) into:
  data_clean/glycan_immunogenicity.tsv

Output schema:
  glycan_id<TAB>label
where label is binary (0/1).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

import pandas as pd


DEFAULT_ID_CANDIDATES = (
    "glycan_id",
    "glytoucan_ac",
    "glytoucan_id",
    "glytoucan",
    "id",
)

DEFAULT_LABEL_CANDIDATES = (
    "label",
    "immunogenicity",
    "immunogenic",
    "y",
    "target",
    "class",
)

DEFAULT_POSITIVE = ("1", "true", "yes", "positive", "pos", "immunogenic")
DEFAULT_NEGATIVE = ("0", "false", "no", "negative", "neg", "non-immunogenic", "non_immunogenic")


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".tsv", ".txt"):
        return pd.read_csv(path, sep="\t", low_memory=False)
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict):
            return pd.DataFrame(data.values() if all(isinstance(v, dict) for v in data.values()) else [data])
        raise ValueError(f"Unsupported JSON structure: {type(data)}")
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return pd.DataFrame(rows)
    raise ValueError(f"Unsupported file extension: {suffix}")


def _find_column(columns: Iterable[str], explicit: Optional[str], candidates: Iterable[str], kind: str) -> str:
    cols = list(columns)
    if explicit:
        if explicit not in cols:
            raise ValueError(f"{kind} column '{explicit}' not found. Available: {cols}")
        return explicit
    for c in candidates:
        if c in cols:
            return c
    raise ValueError(f"Could not infer {kind} column. Available: {cols}")


def _parse_label(raw, positive: Set[str], negative: Set[str]) -> Optional[int]:
    if pd.isna(raw):
        return None

    text = str(raw).strip()
    if not text:
        return None

    low = text.lower()
    if low in positive:
        return 1
    if low in negative:
        return 0

    try:
        num = float(text)
        if num == 1.0:
            return 1
        if num == 0.0:
            return 0
    except ValueError:
        pass
    return None


def _load_whitelist(path: Path, id_col: str) -> Set[str]:
    if not path.exists():
        return set()
    df = pd.read_csv(path, sep="\t", low_memory=False)
    if id_col not in df.columns:
        raise ValueError(f"Whitelist id column '{id_col}' not found in {path}")
    return {str(x).strip() for x in df[id_col].dropna().astype(str)}


def _dedupe_with_conflict_check(rows: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    conflict_count = 0
    grouped: Dict[str, int] = {}
    for _, r in rows.iterrows():
        gid = r["glycan_id"]
        label = int(r["label"])
        if gid in grouped and grouped[gid] != label:
            conflict_count += 1
        grouped[gid] = label
    out = pd.DataFrame(
        {"glycan_id": list(grouped.keys()), "label": list(grouped.values())}
    )
    return out, conflict_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert source labels to data_clean/glycan_immunogenicity.tsv")
    parser.add_argument("--input", required=True, help="Input file path (.tsv/.csv/.json/.jsonl)")
    parser.add_argument(
        "--output",
        default="data_clean/glycan_immunogenicity.tsv",
        help="Output TSV path",
    )
    parser.add_argument("--glycan-id-col", default=None, help="Source glycan ID column")
    parser.add_argument("--label-col", default=None, help="Source label column")
    parser.add_argument(
        "--glycan-whitelist",
        default="data_clean/glycans_clean.tsv",
        help="TSV containing valid glycan IDs for filtering (optional)",
    )
    parser.add_argument(
        "--whitelist-id-col",
        default="glycan_id",
        help="ID column name in whitelist table",
    )
    parser.add_argument(
        "--no-whitelist-filter",
        action="store_true",
        help="Do not filter labels by glycoMusubi glycan IDs",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    df = _read_table(in_path)
    id_col = _find_column(df.columns, args.glycan_id_col, DEFAULT_ID_CANDIDATES, "glycan_id")
    label_col = _find_column(df.columns, args.label_col, DEFAULT_LABEL_CANDIDATES, "label")

    positive = {v.lower() for v in DEFAULT_POSITIVE}
    negative = {v.lower() for v in DEFAULT_NEGATIVE}

    work = df[[id_col, label_col]].copy()
    work["glycan_id"] = work[id_col].where(work[id_col].notna(), "").astype(str).str.strip()
    work.loc[work["glycan_id"].str.lower().isin({"nan", "none", "null"}), "glycan_id"] = ""
    work["label"] = work[label_col].apply(lambda x: _parse_label(x, positive, negative))

    invalid_label = int(work["label"].isna().sum())
    empty_id = int((work["glycan_id"] == "").sum())

    work = work[(work["glycan_id"] != "") & work["label"].notna()][["glycan_id", "label"]]
    work["label"] = work["label"].astype(int)

    whitelist_filtered = 0
    if not args.no_whitelist_filter:
        whitelist = _load_whitelist(Path(args.glycan_whitelist), args.whitelist_id_col)
        if whitelist:
            before = len(work)
            work = work[work["glycan_id"].isin(whitelist)]
            whitelist_filtered = before - len(work)

    deduped, conflicts = _dedupe_with_conflict_check(work)
    deduped = deduped.sort_values("glycan_id").reset_index(drop=True)
    deduped.to_csv(out_path, sep="\t", index=False)

    print(f"Input rows: {len(df)}")
    print(f"Parsed rows: {len(work)}")
    print(f"Invalid labels skipped: {invalid_label}")
    print(f"Empty glycan_id skipped: {empty_id}")
    print(f"Whitelist filtered: {whitelist_filtered}")
    print(f"Conflicting duplicates resolved (last wins): {conflicts}")
    print(f"Output rows: {len(deduped)}")
    print(f"Saved: {out_path}")
    if not deduped.empty:
        print("Label counts:", deduped["label"].value_counts().to_dict())


if __name__ == "__main__":
    main()
