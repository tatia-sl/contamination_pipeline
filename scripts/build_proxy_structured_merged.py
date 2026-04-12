#!/usr/bin/env python3
"""
scripts/build_proxy_structured_merged.py

Merge per-source structured proxy CSVs into a single corpus for SLex.

Single-pass pipeline context
─────────────────────────────
In the v4 pipeline, run_proxy_builder.py produces
proxy_structured_github.csv and proxy_structured_kaggle.csv directly
during the collection run (one download per file).
This script is the final merge step only — it no longer orchestrates
collection or calls extract_structured_proxy_data.py.

Normal usage
────────────
    python3 scripts/build_proxy_structured_merged.py \\
        --config configs/run_config.yaml

Flags
─────
    --github_csv    override path to GitHub structured CSV
    --kaggle_csv    override path to Kaggle structured CSV
    --merged_out    override path for merged output CSV
    --dedupe_mode   none | summary_norm | doc_summary_hash  (default: summary_norm)
    --dry_run       validate inputs and print stats without writing output

Inputs  (resolved from config, overridable via flags)
────────────────────────────────────────────────────
    data/proxies/proxy_structured_github.csv
    data/proxies/proxy_structured_kaggle.csv

Outputs
────────────────────────────────────────────────────
    data/proxies/proxy_structured_merged.csv   → SLex (proxy_column=summary_ref)
    outputs/proxy_structured_merged_build_summary.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml


STRUCTURED_COLS = [
    "item_id",
    "xsum_id",
    "split",
    "source",
    "source_detail",
    "source_sha256",
    "source_query",
    "source_repo",
    "document",
    "summary_ref",
]

# Columns that must exist in v3-era CSVs (without the new provenance fields).
# If any of the new columns are missing we add them as None so the merge still
# works — this keeps backward compatibility with CSVs produced by the old
# extract_structured_proxy_data.py.
LEGACY_REQUIRED_COLS = ["item_id", "xsum_id", "split", "source", "source_detail",
                         "document", "summary_ref"]
NEW_PROVENANCE_COLS  = ["source_sha256", "source_query", "source_repo"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_summary_for_dedupe(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


def read_structured_csv(path: str, label: str) -> pd.DataFrame:
    """
    Read a per-source structured CSV with backward-compatible column handling.
    Missing provenance columns (added in v4) are filled with None so that
    v3-era CSVs can be merged without errors.
    """
    p = Path(path)
    if not p.exists():
        print(f"  [{label}] not found at {path} — using empty frame")
        return pd.DataFrame(columns=STRUCTURED_COLS)

    df = pd.read_csv(p, dtype=str)

    # Back-compat: ensure all expected columns exist
    for col in STRUCTURED_COLS:
        if col not in df.columns:
            df[col] = None

    # Keep only canonical columns in canonical order
    df = df[STRUCTURED_COLS].copy()
    n = len(df)
    print(f"  [{label}] {n} rows  ← {path}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Merge structured proxy CSVs for SLex",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config",      default="configs/run_config.yaml")
    ap.add_argument("--github_csv",  default=None, help="Override GitHub structured CSV path")
    ap.add_argument("--kaggle_csv",  default=None, help="Override Kaggle structured CSV path")
    ap.add_argument("--merged_out",  default=None, help="Override merged output path")
    ap.add_argument("--summary_out", default=None, help="Override summary JSON path")
    ap.add_argument(
        "--dedupe_mode",
        choices=["none", "summary_norm", "doc_summary_hash"],
        default="summary_norm",   # changed from "none" — see notes in review
        help="Deduplication strategy (default: summary_norm)",
    )
    ap.add_argument("--dry_run", action="store_true",
                    help="Validate and print stats without writing output")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    pb  = cfg.get("proxy_builder", {})
    out_dir = pb.get("output_dir", "data/proxies")

    github_csv = args.github_csv  or pb.get("github_structured_out",
                                             f"{out_dir}/proxy_structured_github.csv")
    kaggle_csv = args.kaggle_csv  or pb.get("kaggle_structured_out",
                                             f"{out_dir}/proxy_structured_kaggle.csv")
    merged_out = args.merged_out  or pb.get("merged_out",
                                             f"{out_dir}/proxy_structured_merged.csv")
    summary_out = args.summary_out or "outputs/proxy_structured_merged_build_summary.json"

    started_at = utc_now()

    if not args.quiet:
        print("=" * 60)
        print("build_proxy_structured_merged  (merge-only, v4 pipeline)")
        print("=" * 60)
        print(f"  GitHub CSV  : {github_csv}")
        print(f"  Kaggle CSV  : {kaggle_csv}")
        print(f"  Merged out  : {merged_out}")
        print(f"  Dedupe mode : {args.dedupe_mode}")
        print()

    # ── Read inputs ───────────────────────────────────────────────────────────
    df_gh = read_structured_csv(github_csv, "github")
    df_kg = read_structured_csv(kaggle_csv, "kaggle")

    merged = pd.concat([df_gh, df_kg], ignore_index=True)
    rows_before = len(merged)

    if not args.quiet:
        print(f"\n  Total rows before dedupe: {rows_before}")

    # ── Deduplication ─────────────────────────────────────────────────────────
    if args.dedupe_mode == "summary_norm":
        merged["_norm"] = normalize_summary_for_dedupe(merged["summary_ref"])
        merged = merged.drop_duplicates(subset=["_norm"], keep="first").drop(columns=["_norm"])

    elif args.dedupe_mode == "doc_summary_hash":
        key = merged["document"].fillna("").astype(str) + "||" + \
              merged["summary_ref"].fillna("").astype(str)
        merged["_hash"] = pd.util.hash_pandas_object(key, index=False).astype(str)
        merged = merged.drop_duplicates(subset=["_hash"], keep="first").drop(columns=["_hash"])

    merged = merged.reset_index(drop=True)
    rows_after = len(merged)

    if not args.quiet:
        print(f"  Rows after  dedupe ({args.dedupe_mode}): {rows_after}  "
              f"(removed {rows_before - rows_after})")

    # ── Write output ──────────────────────────────────────────────────────────
    if not args.dry_run:
        ensure_parent(merged_out)
        merged.to_csv(merged_out, index=False, encoding="utf-8")
        if not args.quiet:
            print(f"\n  Written: {merged_out}")
    else:
        if not args.quiet:
            print("\n  dry_run=True — output not written")

    # ── Provenance quality check ──────────────────────────────────────────────
    has_sha = merged["source_sha256"].notna().sum() if "source_sha256" in merged.columns else 0
    pct_sha = f"{has_sha / rows_after * 100:.1f}%" if rows_after else "n/a"

    # ── Summary JSON ──────────────────────────────────────────────────────────
    norm = normalize_summary_for_dedupe(merged["summary_ref"])
    summary: Dict[str, Any] = {
        "started_at_utc":          started_at,
        "finished_at_utc":         utc_now(),
        "config":                  args.config,
        "github_csv":              github_csv,
        "kaggle_csv":              kaggle_csv,
        "merged_out":              merged_out,
        "dedupe_mode":             args.dedupe_mode,
        "dry_run":                 args.dry_run,
        "rows_github":             int(len(df_gh)),
        "rows_kaggle":             int(len(df_kg)),
        "rows_before_dedupe":      int(rows_before),
        "rows_after_dedupe":       int(rows_after),
        "rows_removed_by_dedupe":  int(rows_before - rows_after),
        "rows_empty_summary":      int(
            (merged["summary_ref"].fillna("").astype(str).str.strip() == "").sum()
        ),
        "rows_unique_summary_norm": int(norm.nunique(dropna=False)),
        "rows_with_sha256":        int(has_sha),
        "pct_rows_with_sha256":    pct_sha,
        "source_counts":           merged["source"].fillna("unknown")
                                        .value_counts(dropna=False).to_dict(),
    }

    if not args.dry_run:
        ensure_parent(summary_out)
        with open(summary_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    if not args.quiet:
        print(f"\n  Provenance coverage (source_sha256): {pct_sha} of rows")
        print(f"  Summary JSON: {summary_out}")
        print("\nNext step:")
        print(f"  python3 scripts/run_lexical_detector.py \\")
        print(f"    --config {args.config} \\")
        print(f"    --proxy_path {merged_out} \\")
        print(f"    --proxy_column summary_ref \\")
        print(f"    --prefix structured_merged")
        print("=" * 60)
        print("Done.")


if __name__ == "__main__":
    main()
