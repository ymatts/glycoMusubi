#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Map GlycanML immunogenicity labels to GlyTouCan IDs used by glycoMusubi.

Input:
- GlycanML immunogenicity CSV (contains internal glycan_id, IUPAC, immunogenicity)
- One or more mapping tables that include either:
  - GlycanML internal ID -> GlyTouCan ID
  - IUPAC -> GlyTouCan ID

Output:
- data_clean/glycan_immunogenicity.tsv (glycan_id, label)
- Optional mapping diagnostics in logs/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd


GLYTOUCAN_CANDIDATES = ("glytoucan_ac", "glytoucan_id", "glytoucan", "glycan_id")
INTERNAL_ID_CANDIDATES = ("glycanml_id", "glycan_id", "id", "source_id")
IUPAC_CANDIDATES = ("glycan", "iupac", "iupac_condensed", "structure")
LABEL_CANDIDATES = ("immunogenicity", "label", "y", "target", "class")


def _read_table(path: Path) -> pd.DataFrame:
    sfx = path.suffix.lower()
    if sfx == ".csv":
        return pd.read_csv(path, low_memory=False)
    if sfx in (".tsv", ".txt"):
        return pd.read_csv(path, sep="\t", low_memory=False)
    if sfx == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict):
            if all(isinstance(v, dict) for v in data.values()):
                return pd.DataFrame(data.values())
            return pd.DataFrame([data])
        raise ValueError(f"Unsupported JSON payload: {type(data)}")
    raise ValueError(f"Unsupported extension for {path}")


def _pick_column(columns: Iterable[str], explicit: Optional[str], candidates: Sequence[str], name: str) -> str:
    cols = list(columns)
    if explicit:
        if explicit not in cols:
            raise ValueError(f"{name} column '{explicit}' not found. Available={cols}")
        return explicit
    for c in candidates:
        if c in cols:
            return c
    raise ValueError(f"Could not infer {name} column. Available={cols}")


def _norm_text(v) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip()


def _norm_iupac(v) -> str:
    s = _norm_text(v)
    # Best-effort canonicalization for matching across minor formatting differences.
    return "".join(s.split()).lower()


def _parse_label(v) -> Optional[int]:
    if pd.isna(v):
        return None
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "positive", "pos", "immunogenic"):
        return 1
    if s in ("0", "false", "no", "negative", "neg", "non-immunogenic", "non_immunogenic"):
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


def _load_whitelist(path: Path, id_col: str) -> set[str]:
    if not path.exists():
        return set()
    df = pd.read_csv(path, sep="\t", low_memory=False)
    if id_col not in df.columns:
        raise ValueError(f"Whitelist column '{id_col}' not found in {path}")
    return {str(x).strip() for x in df[id_col].dropna().astype(str)}


def _build_mapping_from_file(path: Path) -> pd.DataFrame:
    df = _read_table(path)
    glytoucan_col = _pick_column(df.columns, None, GLYTOUCAN_CANDIDATES, "glytoucan_id")

    internal_col = None
    for c in INTERNAL_ID_CANDIDATES:
        if c in df.columns and c != glytoucan_col:
            internal_col = c
            break

    iupac_col = None
    for c in IUPAC_CANDIDATES:
        if c in df.columns:
            iupac_col = c
            break

    if internal_col is None and iupac_col is None:
        raise ValueError(
            f"{path}: no usable mapping key found. Need one of internal ID columns {INTERNAL_ID_CANDIDATES} "
            f"or IUPAC columns {IUPAC_CANDIDATES}."
        )

    out = pd.DataFrame()
    out["glytoucan_id"] = df[glytoucan_col].map(_norm_text)
    out = out[out["glytoucan_id"] != ""]
    if internal_col is not None:
        out["source_id"] = df[internal_col].map(_norm_text)
    else:
        out["source_id"] = ""
    if iupac_col is not None:
        out["source_iupac"] = df[iupac_col].map(_norm_text)
        out["source_iupac_norm"] = df[iupac_col].map(_norm_iupac)
    else:
        out["source_iupac"] = ""
        out["source_iupac_norm"] = ""
    return out.drop_duplicates()


def main() -> None:
    parser = argparse.ArgumentParser(description="Map GlycanML immunogenicity labels to GlyTouCan IDs.")
    parser.add_argument(
        "--glycanml-input",
        default="data_raw/glycan_immunogenicity_glycanml.csv",
        help="GlycanML immunogenicity CSV path",
    )
    parser.add_argument(
        "--mapping-input",
        action="append",
        default=[],
        help="Mapping table path. May be specified multiple times.",
    )
    parser.add_argument(
        "--output",
        default="data_clean/glycan_immunogenicity.tsv",
        help="Output TSV path",
    )
    parser.add_argument(
        "--report-path",
        default="logs/immunogenicity_mapping_report.tsv",
        help="Diagnostics TSV path",
    )
    parser.add_argument(
        "--template-path",
        default="logs/immunogenicity_mapping_template.tsv",
        help="Template TSV path for manual curation when mapping is incomplete",
    )
    parser.add_argument(
        "--glycan-whitelist",
        default="data_clean/glycans_clean.tsv",
        help="Whitelist TSV (glycoMusubi glycans) for output filtering",
    )
    parser.add_argument(
        "--whitelist-id-col",
        default="glycan_id",
        help="ID column name in whitelist TSV",
    )
    parser.add_argument(
        "--no-whitelist-filter",
        action="store_true",
        help="Disable filtering to known glycoMusubi glycan IDs.",
    )
    args = parser.parse_args()

    gml_path = Path(args.glycanml_input)
    if not gml_path.exists():
        raise FileNotFoundError(f"GlycanML input not found: {gml_path}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    template_path = Path(args.template_path)
    template_path.parent.mkdir(parents=True, exist_ok=True)

    src = _read_table(gml_path)
    id_col = _pick_column(src.columns, None, INTERNAL_ID_CANDIDATES, "source glycan_id")
    iupac_col = _pick_column(src.columns, None, IUPAC_CANDIDATES, "source IUPAC")
    label_col = _pick_column(src.columns, None, LABEL_CANDIDATES, "label")

    src["source_id"] = src[id_col].map(_norm_text)
    src["source_iupac"] = src[iupac_col].map(_norm_text)
    src["source_iupac_norm"] = src[iupac_col].map(_norm_iupac)
    src["label"] = src[label_col].map(_parse_label)

    raw_rows = len(src)
    src = src[src["label"].isin([0, 1])].copy()
    valid_label_rows = len(src)

    mapping_tables: list[pd.DataFrame] = []
    for mp in args.mapping_input:
        p = Path(mp)
        if not p.exists():
            raise FileNotFoundError(f"Mapping input not found: {p}")
        mapping_tables.append(_build_mapping_from_file(p))

    if mapping_tables:
        mapping = pd.concat(mapping_tables, ignore_index=True).drop_duplicates()
    else:
        mapping = pd.DataFrame(columns=["source_id", "source_iupac", "source_iupac_norm", "glytoucan_id"])

    # Match priority: internal ID first, then normalized IUPAC.
    by_id = mapping[mapping["source_id"] != ""][["source_id", "glytoucan_id"]].drop_duplicates()
    by_iupac = mapping[mapping["source_iupac_norm"] != ""][["source_iupac_norm", "glytoucan_id"]].drop_duplicates()

    merged = src.copy()
    merged = merged.merge(by_id, on="source_id", how="left", suffixes=("", "_id"))
    merged = merged.rename(columns={"glytoucan_id": "glytoucan_from_id"})
    merged = merged.merge(by_iupac, on="source_iupac_norm", how="left", suffixes=("", "_iupac"))
    merged = merged.rename(columns={"glytoucan_id": "glytoucan_from_iupac"})

    merged["glytoucan_id"] = merged["glytoucan_from_id"]
    mask = merged["glytoucan_id"].isna() | (merged["glytoucan_id"] == "")
    merged.loc[mask, "glytoucan_id"] = merged.loc[mask, "glytoucan_from_iupac"]
    merged["glytoucan_id"] = merged["glytoucan_id"].map(_norm_text)

    if not args.no_whitelist_filter:
        whitelist = _load_whitelist(Path(args.glycan_whitelist), args.whitelist_id_col)
        if whitelist:
            merged["in_whitelist"] = merged["glytoucan_id"].isin(whitelist)
        else:
            merged["in_whitelist"] = True
    else:
        merged["in_whitelist"] = True

    merged["mapped"] = merged["glytoucan_id"] != ""
    merged["usable"] = merged["mapped"] & merged["in_whitelist"]

    final = merged[merged["usable"]][["glytoucan_id", "label"]].copy()
    final = final.rename(columns={"glytoucan_id": "glycan_id"})

    # Resolve conflicting labels per glycan (last wins for determinism).
    conflict_count = 0
    grouped = {}
    for _, row in final.iterrows():
        gid = row["glycan_id"]
        lab = int(row["label"])
        if gid in grouped and grouped[gid] != lab:
            conflict_count += 1
        grouped[gid] = lab
    final = pd.DataFrame({"glycan_id": list(grouped.keys()), "label": list(grouped.values())})
    final = final.sort_values("glycan_id").reset_index(drop=True)
    final.to_csv(out_path, sep="\t", index=False)

    report_cols = [
        "source_id",
        "source_iupac",
        "label",
        "glytoucan_from_id",
        "glytoucan_from_iupac",
        "glytoucan_id",
        "mapped",
        "in_whitelist",
        "usable",
    ]
    merged[report_cols].to_csv(report_path, sep="\t", index=False)

    # Emit manual template for unresolved rows.
    unresolved = merged[~merged["mapped"]][["source_id", "source_iupac", "label"]].drop_duplicates()
    unresolved["glytoucan_id"] = ""
    unresolved.to_csv(template_path, sep="\t", index=False)

    print(f"Input rows: {raw_rows}")
    print(f"Rows with valid labels: {valid_label_rows}")
    print(f"Mapping tables: {len(mapping_tables)}")
    print(f"Mapped rows: {int(merged['mapped'].sum())}")
    print(f"Rows in whitelist: {int((merged['mapped'] & merged['in_whitelist']).sum())}")
    print(f"Conflicting duplicate labels resolved (last wins): {conflict_count}")
    print(f"Output rows: {len(final)}")
    print(f"Saved output: {out_path}")
    print(f"Saved report: {report_path}")
    print(f"Saved unresolved template: {template_path}")
    if len(final):
        print("Label counts:", final["label"].value_counts().to_dict())


if __name__ == "__main__":
    main()

