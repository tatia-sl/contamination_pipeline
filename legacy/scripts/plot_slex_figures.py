#!/usr/bin/env python3
"""
Build dissertation-friendly SLex figures from a lexical parquet file.

Produces:
1) SLex level distribution (0/1/2/3) - bar chart
2) Match-strength distribution - histogram (MaxSpanLen preferred, else NgramHits)
3) MaxSpanLen by SLex level - boxplot (if MaxSpanLen exists)
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def infer_tag_from_path(path: Path) -> str:
    # e.g. runs/v3_lexical_structured_merged.parquet -> structured_merged
    stem = path.stem
    prefix = "v3_lexical_"
    if stem.startswith(prefix) and len(stem) > len(prefix):
        return stem[len(prefix):]
    return "default"


def pick_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def pick_prefixed_col(df: pd.DataFrame, base: str, tag: str, extra_candidates: list[str]) -> Optional[str]:
    candidates = []
    if tag:
        candidates.extend([f"{base}_{tag}", f"{base}{tag}"])
    candidates.extend(extra_candidates)
    return pick_col(df, candidates)


def main() -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from exc

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="runs/v3_lexical_structured_merged.parquet",
        help="Path to lexical parquet",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional tag for prefixed columns and output names (default: inferred from input path)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs",
        help="Directory for output figures",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Histogram bins",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {in_path}")

    df = pd.read_parquet(in_path)
    tag = args.tag or infer_tag_from_path(in_path)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    col_slex = pick_prefixed_col(
        df, "SLex", tag, ["SLex", "slex", "lex_level", "SLex_level"]
    )
    col_span = pick_prefixed_col(
        df, "MaxSpanLen", tag, ["MaxSpanLen", "max_span_len", "MaxSpan", "span_max"]
    )
    col_ngram = pick_prefixed_col(
        df, "NgramHits", tag, ["NgramHits", "ngram_hits", "NGramHits", "hits_13gram"]
    )

    print("Detected columns:", {"SLex": col_slex, "MaxSpanLen": col_span, "NgramHits": col_ngram})

    if col_slex is None:
        raise ValueError("Could not find SLex column. Check df.columns or pass --tag.")

    # 1) SLex level distribution (fixed order 0..3 for comparability)
    slex_vals = pd.to_numeric(df[col_slex], errors="coerce")
    counts = slex_vals.value_counts(dropna=False)
    ordered_levels = [0, 1, 2, 3]
    y = [int(counts.get(level, 0)) for level in ordered_levels]

    plt.figure(figsize=(7, 4.5))
    plt.bar([str(x) for x in ordered_levels], y)
    plt.title("SLex Level Distribution")
    plt.xlabel("SLex level")
    plt.ylabel("Number of instances")
    plt.tight_layout()
    out1 = out_dir / f"fig_slex_levels_{tag}.png"
    plt.savefig(out1, dpi=200)
    plt.close()

    # 2) Match-strength distribution
    metric_col = col_span if col_span is not None else col_ngram
    if metric_col is None:
        raise ValueError("Could not find MaxSpanLen or NgramHits column.")

    x = pd.to_numeric(df[metric_col], errors="coerce").dropna()
    plt.figure(figsize=(7, 4.5))
    plt.hist(x, bins=args.bins)
    plt.title(f"Distribution of {metric_col}")
    plt.xlabel(metric_col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    metric_slug = metric_col
    suffix = f"_{tag}"
    if tag and metric_slug.endswith(suffix):
        metric_slug = metric_slug[: -len(suffix)]
    out2 = out_dir / f"fig_slex_strength_hist_{metric_slug}_{tag}.png"
    plt.savefig(out2, dpi=200)
    plt.close()

    # 3) Boxplot: MaxSpanLen by SLex level (if MaxSpanLen exists)
    out3 = None
    if col_span is not None:
        tmp = df[[col_slex, col_span]].copy()
        tmp[col_slex] = pd.to_numeric(tmp[col_slex], errors="coerce")
        tmp[col_span] = pd.to_numeric(tmp[col_span], errors="coerce")
        tmp = tmp.dropna(subset=[col_slex, col_span])

        levels = [lv for lv in [0, 1, 2, 3] if (tmp[col_slex] == lv).any()]
        data = [tmp.loc[tmp[col_slex] == lv, col_span].values for lv in levels]

        if data:
            plt.figure(figsize=(7, 4.5))
            plt.boxplot(data, tick_labels=[str(lv) for lv in levels], showfliers=False)
            plt.title("MaxSpanLen by SLex Level")
            plt.xlabel("SLex level")
            plt.ylabel("MaxSpanLen (characters)")
            plt.tight_layout()
            out3 = out_dir / f"fig_slex_boxplot_maxspan_by_level_{tag}.png"
            plt.savefig(out3, dpi=200)
            plt.close()

    print("Saved:")
    print(f"- {out1}")
    print(f"- {out2}")
    if out3:
        print(f"- {out3}")


if __name__ == "__main__":
    main()
