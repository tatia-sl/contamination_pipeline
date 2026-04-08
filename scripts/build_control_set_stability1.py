#!/usr/bin/env python3
"""
scripts/build_control_set_stability.py

Builds the control document set for the stability detector control baseline pass.

The control set is sampled from CNN/DailyMail test split (news domain) to
provide a non-XSum baseline of comparable text type for stability metrics.
Rows are sampled reproducibly by seed and normalized with the same text
normalization routine as the main pipeline.

Output:
  A parquet file with columns:
    control_id      -- deterministic identifier (ctrl_XXXX)
    document_norm   -- normalized article text (same normalization as master table)
    source_dataset  -- "cnn_dailymail"
    source_split    -- "test"
    source_index    -- original index in CNN/DailyMail test split

Usage:
  python scripts/build_control_set_stability.py \\
      --output data/control_set_cnn_n296_seed42.parquet \\
      --n 296 \\
      --seed 42

Requirements:
  pip install datasets
"""

import argparse
import unicodedata
import random
from pathlib import Path

import pandas as pd


def normalize_text(s: str) -> str:
    """Identical normalization to the main pipeline (run_stability_detector.py)."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = " ".join(s.split())
    return s.strip()


def build_control_set(n: int, seed: int) -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "The 'datasets' library is required: pip install datasets"
        )

    print("Loading CNN/DailyMail test split...")
    ds = load_dataset("cnn_dailymail", "3.0.0", split="test")
    print(f"  CNN/DailyMail test split size: {len(ds)} articles")

    if n > len(ds):
        raise ValueError(
            f"Requested n={n} exceeds CNN/DailyMail test split size ({len(ds)})"
        )

    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), n)
    indices_sorted = sorted(indices)

    print(f"  Sampling {n} articles with seed={seed}...")

    records = []
    for rank, orig_idx in enumerate(indices_sorted):
        article = ds[orig_idx]["article"]
        doc_norm = normalize_text(article)

        if not doc_norm or len(doc_norm) < 100:
            # Skip degenerate entries — extremely rare in CNN/DailyMail
            print(f"  Warning: skipping short/empty article at index {orig_idx}")
            continue

        records.append({
            "control_id":     f"ctrl_{rank:04d}",
            "document_norm":  doc_norm,
            "source_dataset": "cnn_dailymail",
            "source_split":   "test",
            "source_index":   orig_idx,
        })

    df = pd.DataFrame(records)
    print(f"  Retained {len(df)} control documents after quality check")
    print(f"  Avg document length: {df['document_norm'].str.len().mean():.0f} chars")
    print(f"  Min document length: {df['document_norm'].str.len().min()} chars")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=str,
        default="data/control_set_cnn_n296_seed42.parquet",
        help="Output parquet path"
    )
    parser.add_argument("--n",    type=int, default=296, help="Number of control documents")
    parser.add_argument("--seed", type=int, default=42,  help="Random seed")
    args = parser.parse_args()

    df = build_control_set(n=args.n, seed=args.seed)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)

    print(f"\nControl set saved to: {args.output}")
    print(f"Rows: {len(df)}")
    print(df[["control_id", "source_index"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
