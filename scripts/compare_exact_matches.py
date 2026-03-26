#!/usr/bin/env python3
"""
scripts/compare_exact_matches.py
----------------------------------
Compares exact match (EM=1) rows across multiple model probe runs.

For each model, extracts the xsum_ids where EM=1, then computes:
  - per-model exact match sets
  - pairwise overlaps between all model pairs
  - union of all exact matches
  - rows matched by ALL models simultaneously
  - rows matched by only ONE model (model-specific memorisation)

Also prints the actual prefix + gold_suffix + completion for each
exact match row, grouped by overlap pattern — the key diagnostic
for distinguishing trivial suffixes from genuine memorisation.

Usage:
    python compare_exact_matches.py --config run_config.yaml

Or with explicit paths:
    python compare_exact_matches.py \
        --parquets runs/v5_mem_gpt4.parquet \
                   runs/v5_mem_gpt35turbo.parquet \
                   runs/v5_mem_gpt4omini.parquet \
                   runs/v5_mem_gemini25flash.parquet \
        --model_ids gpt4 gpt35turbo gpt4omini gemini25flash \
        --master master_table_xsum_n300_seed42_v4_dcq4_frozen.parquet
"""

import argparse
import unicodedata
import re
import sys
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Set

import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = " ".join(s.split())
    return s.strip()


def extract_gold_suffix(ref_full: str, prefix: str) -> str:
    ref_full = normalize_text(ref_full)
    prefix   = normalize_text(prefix)
    if not ref_full or not prefix:
        return ""
    if ref_full.startswith(prefix):
        return ref_full[len(prefix):].strip()
    return ""


def sep(char: str = "-", width: int = 68) -> str:
    return char * width


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_from_config(config_path: str):
    """Load model_ids and parquet paths from run_config.yaml."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    mem_cfg = cfg["memorization"]
    master  = cfg["project"]["frozen_master_table_path"]

    model_ids = []
    parquets  = []
    for m in cfg["models"]:
        mid = m["model_id"]
        pq  = mem_cfg["outputs"]["parquet"].replace("{model_id}", mid)
        if Path(pq).exists():
            model_ids.append(mid)
            parquets.append(pq)
        else:
            print(f"[SKIP] {mid}: parquet not found at {pq}")

    return model_ids, parquets, master


def load_em_sets(
    model_ids: List[str],
    parquets:  List[str],
    master_path: str,
) -> tuple:
    """
    Returns:
        master_df   — master table with prefix_ref and summary_ref_norm
        model_dfs   — dict {model_id: df with EM column}
        em_sets     — dict {model_id: set of xsum_ids with EM=1}
    """
    master_df = pd.read_parquet(master_path)

    model_dfs: Dict[str, pd.DataFrame] = {}
    em_sets:   Dict[str, Set[str]]     = {}

    for mid, pq in zip(model_ids, parquets):
        df     = pd.read_parquet(pq)
        col_em = f"EM_{mid}"

        if col_em not in df.columns:
            print(f"[WARN] Column '{col_em}' not found in {pq} — skipping")
            continue

        em_vals = pd.to_numeric(df[col_em], errors="coerce")
        exact_mask = em_vals == 1

        if "xsum_id" in df.columns:
            ids = set(df.loc[exact_mask, "xsum_id"].astype(str).tolist())
        else:
            ids = set(df.index[exact_mask].astype(str).tolist())

        model_dfs[mid] = df
        em_sets[mid]   = ids
        print(f"  {mid:20s}: {len(ids):3d} exact match rows")

    return master_df, model_dfs, em_sets


# ---------------------------------------------------------------------------
# Overlap analysis
# ---------------------------------------------------------------------------

def compute_overlaps(em_sets: Dict[str, Set[str]]) -> None:
    model_ids = list(em_sets.keys())

    print(f"\n{sep()}")
    print("  Pairwise overlaps")
    print(sep())

    for a, b in combinations(model_ids, 2):
        overlap = em_sets[a] & em_sets[b]
        union   = em_sets[a] | em_sets[b]
        jaccard = len(overlap) / len(union) if union else 0.0
        print(f"  {a:20s} ∩ {b:20s} = {len(overlap):3d} rows  "
              f"(Jaccard: {jaccard:.3f})")

    all_models_overlap = set.intersection(*em_sets.values()) if em_sets else set()
    any_model_union    = set.union(*em_sets.values())         if em_sets else set()

    print(f"\n  Union (any model):        {len(any_model_union):3d} unique xsum_ids")
    print(f"  Intersection (all models):{len(all_models_overlap):3d} xsum_ids  "
          f"← same row matched by ALL models")

    return all_models_overlap, any_model_union


# ---------------------------------------------------------------------------
# Per-row diagnostic
# ---------------------------------------------------------------------------

def print_row_detail(
    xsum_id:    str,
    master_df:  pd.DataFrame,
    model_dfs:  Dict[str, pd.DataFrame],
    em_sets:    Dict[str, Set[str]],
    label:      str,
) -> None:
    """Print prefix, gold_suffix, and completions for one xsum_id."""

    master_row = master_df[master_df["xsum_id"].astype(str) == str(xsum_id)]
    if master_row.empty:
        print(f"    [xsum_id {xsum_id} not found in master table]")
        return

    row      = master_row.iloc[0]
    prefix   = normalize_text(str(row.get("prefix_ref", "")))
    ref      = normalize_text(str(row.get("summary_ref_norm", "")))
    suffix   = extract_gold_suffix(ref, prefix)
    doc_norm = str(row.get("document_norm", ""))[:120]

    print(f"\n  xsum_id : {xsum_id}  [{label}]")
    print(f"  prefix  : {prefix[:90]}")
    print(f"  suffix  : {suffix[:90]}  ({len(suffix.split())} words)")
    print(f"  doc     : {doc_norm}...")
    print()

    for mid, df in model_dfs.items():
        col_comp = f"mem_completion_{mid}"
        col_em   = f"EM_{mid}"
        row_df   = df[df["xsum_id"].astype(str) == str(xsum_id)]

        if row_df.empty:
            continue

        r      = row_df.iloc[0]
        em_val = int(pd.to_numeric(r.get(col_em, 0), errors="coerce") or 0)
        comp   = normalize_text(str(r.get(col_comp, "")))[:90]
        flag   = "EM=1" if em_val == 1 else "    "

        print(f"    [{flag}] {mid:20s}: {comp}")


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------

def run_report(
    model_ids:   List[str],
    parquets:    List[str],
    master_path: str,
) -> None:
    print(f"\n{sep('=')}")
    print("  Exact match overlap analysis")
    print(sep("="))
    print(f"  Models   : {', '.join(model_ids)}")
    print(f"  Master   : {master_path}")
    print(f"  n        : 296 rows")
    print(sep("="))

    print(f"\n{sep()}")
    print("  Exact match counts per model")
    print(sep())

    master_df, model_dfs, em_sets = load_em_sets(
        model_ids, parquets, master_path
    )

    if not em_sets:
        print("No model data loaded. Check paths.")
        return

    all_overlap, any_union = compute_overlaps(em_sets)

    # Classify each xsum_id in the union
    print(f"\n{sep()}")
    print("  Row-level breakdown")
    print(sep())

    # Group rows by which models matched them
    pattern_groups: Dict[str, List[str]] = {}
    for xid in sorted(any_union):
        matched_by = tuple(
            mid for mid in model_ids if xid in em_sets.get(mid, set())
        )
        key = " + ".join(matched_by)
        pattern_groups.setdefault(key, []).append(xid)

    for pattern, xids in sorted(pattern_groups.items(),
                                 key=lambda x: -len(x[1])):
        n_models = len(pattern.split(" + "))
        marker = (
            "ALL MODELS"   if n_models == len(model_ids) else
            "2+ models"    if n_models > 1 else
            "unique"
        )
        print(f"\n  [{marker}] matched by: {pattern}  ({len(xids)} rows)")

        for xid in xids:
            print_row_detail(
                xsum_id=xid,
                master_df=master_df,
                model_dfs=model_dfs,
                em_sets=em_sets,
                label=marker,
            )

    # Summary verdict
    n_shared    = sum(1 for xids in pattern_groups.values()
                      if len(xids[0].split()) == 0  # placeholder
                      or len(pattern_groups) > 0)
    n_all       = len(all_overlap)
    n_union     = len(any_union)
    n_unique    = sum(
        len(xids) for pat, xids in pattern_groups.items()
        if len(pat.split(" + ")) == 1
    )
    n_shared_2p = n_union - n_unique

    print(f"\n{sep('=')}")
    print("  Summary")
    print(sep("="))
    print(f"  Total unique rows with EM=1 across all models : {n_union}")
    print(f"  Rows matched by 2+ models (shared signal)     : {n_shared_2p}")
    print(f"  Rows matched by ALL models                    : {n_all}")
    print(f"  Rows matched by exactly 1 model (unique)      : {n_unique}")
    print()
    if n_all > 0:
        print("  Rows matched by ALL models are the strongest contamination")
        print("  signal — they are unlikely to be trivially predictable since")
        print("  independent models from different providers all reproduce them.")
    if n_unique > 0:
        print(f"  {n_unique} rows unique to one model suggest model-specific")
        print("  memorisation rather than universally predictable suffixes.")
    if n_shared_2p == 0 and n_all == 0:
        print("  No shared rows — each model reproduced different instances.")
        print("  This weakens the contamination hypothesis; suffixes may be")
        print("  predictable from context for different inputs per model.")
    print(sep("="))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare exact match rows across model probe runs."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=str,
                       help="Path to run_config.yaml (auto-discovers paths)")
    group.add_argument("--parquets", type=str, nargs="+",
                       help="Explicit list of probe parquet paths")

    parser.add_argument("--model_ids", type=str, nargs="+",
                        help="Model IDs matching --parquets order "
                             "(required with --parquets)")
    parser.add_argument("--master", type=str,
                        help="Master table parquet path "
                             "(required with --parquets)")
    args = parser.parse_args()

    if args.config:
        model_ids, parquets, master = load_from_config(args.config)
    else:
        if not args.model_ids or not args.master:
            parser.error(
                "--model_ids and --master are required when using --parquets"
            )
        model_ids = args.model_ids
        parquets  = args.parquets
        master    = args.master

    run_report(model_ids, parquets, master)


if __name__ == "__main__":
    main()
