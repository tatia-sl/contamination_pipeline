"""
load_benchmarks_bbc2025.py
==========================
Creates a frozen master table for the BBC News 2025 baseline dataset
from RealTimeData/bbc_news_alltime.

Default dataset slice:
  - dataset: RealTimeData/bbc_news_alltime
  - config:  2025-06
  - split:   train

The output schema is intentionally compatible with the existing detector
scripts, including the legacy column name xsum_id for the article identifier.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata

import numpy as np
import pandas as pd
from datasets import load_dataset


DEFAULT_SEED = 42
DEFAULT_N_ITEMS = 300
DEFAULT_DATASET_NAME = "RealTimeData/bbc_news_alltime"
DEFAULT_DATASET_CONFIG = "2025-06"
DEFAULT_DATASET_SPLIT = "train"
DEFAULT_OUT_PARQUET = "master_table_bbc2025_n300_seed42_v1.parquet"
DEFAULT_OUT_INDICES = "bbc2025_indices_seed42_n300.json"


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s))
    return re.sub(r"\s+", " ", s).strip()


def make_prefix(summary: str, frac: float = 0.4, min_tokens: int = 12) -> str:
    toks = summary.split()
    m = max(min_tokens, int(len(toks) * frac))
    m = min(m, max(1, len(toks) - 1))
    return " ".join(toks[:m])


def make_control_prefix(prefix: str, seed: int = 123) -> str:
    rnd = random.Random(seed)
    toks = prefix.split()
    rnd.shuffle(toks)
    return " ".join(toks)


def valid(row: dict) -> bool:
    doc = row.get("content", "") or ""
    summary = row.get("description", "") or ""
    return (
        len(doc.strip()) >= 200
        and len(summary.strip()) >= 20
        and len(summary.strip()) <= 300
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BBC 2025 master table.")
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset_config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--dataset_split", default=DEFAULT_DATASET_SPLIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n_items", type=int, default=DEFAULT_N_ITEMS)
    parser.add_argument("--out_parquet", default=DEFAULT_OUT_PARQUET)
    parser.add_argument("--out_indices", default=DEFAULT_OUT_INDICES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(
        f"Loading {args.dataset_name} "
        f"(config={args.dataset_config!r}, split={args.dataset_split!r}) ..."
    )
    ds = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
    print(f"Total rows : {len(ds)}")
    print(f"Columns    : {ds.column_names}")

    random.seed(args.seed)
    indices = random.sample(range(len(ds)), min(args.n_items * 4, len(ds)))
    subset = ds.select(indices)

    with open(args.out_indices, "w", encoding="utf-8") as f:
        json.dump(indices, f)
    print(f"Saved {len(indices)} pre-filter indices -> {args.out_indices}")

    subset = subset.filter(valid)
    print(f"After filtering : {len(subset)} items")

    if len(subset) > args.n_items:
        subset = subset.select(range(args.n_items))
    print(f"Final sample    : {len(subset)} items")

    rows = []
    for idx, ex in enumerate(subset):
        rows.append({
            "item_id": idx,
            "xsum_id": ex.get("link", f"bbc2025_{idx}"),
            "split": args.dataset_split,
            "document": ex["content"],
            "summary_ref": ex["description"],
            "title": ex.get("title", ""),
            "published_date": ex.get("published_date", ""),
        })

    df = pd.DataFrame(rows)
    print(f"\nDataFrame shape : {df.shape}")
    print(df[["xsum_id", "summary_ref"]].head(3).to_string())

    df["document_norm"] = df["document"].map(normalize_text)
    df["summary_ref_norm"] = df["summary_ref"].map(normalize_text)
    df["prefix_ref"] = df["summary_ref_norm"].map(
        lambda s: make_prefix(s, frac=0.4, min_tokens=12)
    )
    df["control_prefix"] = df["prefix_ref"].map(
        lambda p: make_control_prefix(p, seed=123)
    )

    df["dcq_A_canonical"] = df["summary_ref_norm"]
    df["dcq_B_para1"] = ""
    df["dcq_C_para2"] = ""
    df["dcq_D_para3"] = ""
    df["dcq_E_para4"] = ""
    df["dcq_choice"] = ""
    df["mem_completion"] = ""
    df["stability_outputs_json"] = ""

    for col in [
        "CPS", "EM", "NE", "UAR", "mNED",
        "MaxSpanLen", "NgramHits", "ProxyCount",
        "SLex", "SSem", "SMem", "SProb",
        "RiskScore",
    ]:
        df[col] = np.nan

    df["RiskLevel"] = ""
    df["Confidence"] = ""

    df.to_parquet(args.out_parquet, index=False)
    print(f"\nSaved  : {args.out_parquet}")
    print(f"Shape  : {df.shape}")
    print(f"Columns: {list(df.columns)}")

    misaligned = df[
        ~df.apply(lambda r: r["summary_ref_norm"].startswith(r["prefix_ref"]), axis=1)
    ]
    if len(misaligned) > 0:
        print(f"\nWARNING: {len(misaligned)} rows where prefix_ref is not aligned!")
        print(misaligned[["xsum_id", "summary_ref_norm", "prefix_ref"]].head(3).to_string())
    else:
        print(f"\nPrefix alignment check: OK — all {len(df)} rows aligned")

    lengths = df["summary_ref_norm"].str.split().str.len()
    print(
        f"\nSummary length (tokens): min={lengths.min()}, "
        f"mean={lengths.mean():.1f}, max={lengths.max()}"
    )
    print("First 5 summaries:")
    for summary in df["summary_ref_norm"].head(5):
        print(f"  [{len(summary.split())} tokens] {summary[:100]}")


if __name__ == "__main__":
    main()
