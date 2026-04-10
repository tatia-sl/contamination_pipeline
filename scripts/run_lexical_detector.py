#!/usr/bin/env python3
"""
scripts/run_lexical_detector.py

Lexical Proxy Detector (SLex) using a public proxy corpus (XSum mirrors).

Input:
- Frozen master table (FINAL parquet)
- Proxy corpus: data/proxies/proxy_structured_merged.csv (`proxy_column=summary_ref`)

Output:
- runs/v3_lexical.parquet
- logs/v3_lexical.jsonl
- outputs/v3_lexical_summary.json

Metrics:
- MaxSpanLen: max length of longest common substring (character-level) between reference summary and any proxy summary
- NgramHits: number of token 13-grams from reference summary that appear in proxy corpus
- ProxyCount: number of distinct proxy summaries with MaxSpanLen >= 50 (computed over candidate set; diagnostic only)

Rubric (levels 0..3):
0: no matches
1: (30 <= MaxSpanLen < 50) OR few short n-gram hits (1-2)
2: (MaxSpanLen >= 50) OR (NgramHits >= 3)
3: (MaxSpanLen >= 100)

Notes:
- Uses an inverted index over proxy token 13-grams to generate a small candidate set per item.
- Computes MaxSpanLen via longest common substring DP only over candidates (fast enough).
"""

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple, Optional

import pandas as pd
import yaml


# -----------------------
# Utilities
# -----------------------

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def write_json(path: str, payload: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def log_jsonl(path: str, payload: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

def format_path(template: str, model_id: Optional[str] = None) -> str:
    # lexical stage is model-agnostic; keep compatibility if you want templating
    if model_id is None:
        return template.replace("{model_id}", "")
    return template.replace("{model_id}", model_id)

def norm_prefix(prefix: str) -> str:
    """
    Normalize optional column prefix.
    Accepts "", "web", "web_", "web__" and returns "" or "web_".
    """
    if prefix is None:
        return ""
    p = str(prefix).strip()
    if not p:
        return ""
    p = p.rstrip("_")
    return p + "_"

def canonicalize_text(s: str) -> str:
    """
    Normalize text for lexical comparison.

    Goals:
    - make Unicode accents/ligatures comparable to ASCII forms
    - normalize smart quotes/dashes/currency punctuation
    - collapse whitespace deterministically
    """
    if s is None:
        return ""
    s = str(s)
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00a3": " GBP ",
        "\u20ac": " EUR ",
        "\u00a0": " ",
    }
    for src, dst in replacements.items():
        s = s.replace(src, dst)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s: str) -> List[str]:
    # simple, deterministic tokenization
    return re.findall(r"[A-Za-z0-9]+", canonicalize_text(s))

def ngrams(tokens: List[str], k: int) -> List[Tuple[str, ...]]:
    if len(tokens) < k:
        return []
    return [tuple(tokens[i:i+k]) for i in range(len(tokens) - k + 1)]


# -----------------------
# Longest Common Substring (character-level)
# -----------------------
def longest_common_substring_len(a: str, b: str) -> int:
    """
    Length of the Longest Common Substring between a and b (contiguous).
    O(len(a)*len(b)) DP with 2 rows; ok for short strings (XSum summaries).
    """
    if not a or not b:
        return 0
    # Ensure a is the shorter string for memory efficiency
    if len(a) > len(b):
        a, b = b, a

    prev = [0] * (len(a) + 1)
    best = 0
    for cb in b:
        curr = [0] * (len(a) + 1)
        for i, ca in enumerate(a, start=1):
            if ca == cb:
                curr[i] = prev[i-1] + 1
                if curr[i] > best:
                    best = curr[i]
        prev = curr
    return best


# -----------------------
# Rubric mapping
# -----------------------
def map_to_SLex(max_span: int, ngram_hits: int, proxy_count: int) -> int:
    # Apply top-down to avoid overlaps.
    # ProxyCount is retained as a diagnostic metric but is not used in the
    # level mapping because proxy deduplication intentionally removes repeated
    # copies of the same summary across sources.
    if max_span >= 100:
        return 3
    if (max_span >= 50) or (ngram_hits >= 3):
        return 2
    if (30 <= max_span < 50) or (ngram_hits in (1, 2)):
        return 1
    return 0


def map_to_SLex_aggregate(slex_vals: pd.Series) -> int:
    """
    Aggregate-level SLex rubric.

    Rules:
    - 0 if no positive items
    - 1 if only isolated level-1 items
    - 2 if at least one level-2 item or at least 5% positive items
    - 3 if at least one level-3 item or at least 10% level-2/3 items
    """
    vals = pd.to_numeric(slex_vals, errors="coerce").dropna()
    if vals.empty:
        return 0

    total = len(vals)
    positive = (vals > 0).sum()
    level2_or_3 = (vals >= 2).sum()
    has_level2 = bool((vals == 2).any())
    has_level3 = bool((vals == 3).any())

    positive_frac = positive / total if total > 0 else 0.0
    level23_frac = level2_or_3 / total if total > 0 else 0.0

    if positive == 0:
        return 0
    if has_level3 or level23_frac >= 0.10:
        return 3
    if has_level2 or positive_frac >= 0.05:
        return 2
    return 1


# -----------------------
# Proxy index building
# -----------------------
def load_proxy_lines(proxy_path: str, max_lines: Optional[int] = None, proxy_column: Optional[str] = None) -> List[str]:
    """
    Load proxy corpus.
    - .txt: one summary per line
    - .csv/.tsv: try summary-like column (proxy_column or heuristic)
    """
    path = Path(proxy_path)
    suffix = path.suffix.lower()
    if suffix in [".csv", ".tsv"]:
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep, dtype=str, on_bad_lines="skip", nrows=max_lines)
        cand_cols = [proxy_column] + ["summary_ref", "summary", "target", "highlights", "output", "prediction"] if proxy_column else ["summary_ref", "summary", "target", "highlights", "output", "prediction"]
        col = next((c for c in cand_cols if c in df.columns), None)
        if col is None:
            # fallback to first column
            col = df.columns[0]
        series = df[col].dropna().astype(str).str.strip()
        lines = [canonicalize_text(ln) for ln in series.tolist() if str(ln).strip()]
        return lines[:max_lines] if max_lines is not None else lines

    lines: List[str] = []
    with open(proxy_path, "r", encoding="utf-8") as f:
        for i, ln in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            ln = canonicalize_text(ln)
            if ln:
                lines.append(ln)
    return lines

def build_ngram_inverted_index(
    proxy_lines: List[str],
    k: int = 13,
    max_ngrams_per_line: int = 200
) -> Tuple[Dict[Tuple[str, ...], List[int]], List[Set[Tuple[str, ...]]]]:
    """
    Returns:
      inv: ngram -> list of proxy indices where it appears
      line_ngrams: list of sets of ngrams per proxy line (for fast membership)
    """
    inv: Dict[Tuple[str, ...], List[int]] = {}
    line_ngrams: List[Set[Tuple[str, ...]]] = []

    for idx, line in enumerate(proxy_lines):
        toks = tokenize(line)
        ngs = ngrams(toks, k)
        # cap to avoid pathological long lines (rare)
        if len(ngs) > max_ngrams_per_line:
            ngs = ngs[:max_ngrams_per_line]
        ng_set = set(ngs)
        line_ngrams.append(ng_set)
        for g in ng_set:
            inv.setdefault(g, []).append(idx)

    return inv, line_ngrams


def candidate_proxy_indices(
    ref_ngrams: List[Tuple[str, ...]],
    inv: Dict[Tuple[str, ...], List[int]],
    max_candidates: int = 200
) -> List[int]:
    """
    Get candidate proxy indices by union of posting lists for matching ngrams.
    If too many, keep those with highest frequency of hits.
    """
    if not ref_ngrams:
        return []

    hit_counts: Dict[int, int] = {}
    for g in set(ref_ngrams):
        for idx in inv.get(g, []):
            hit_counts[idx] = hit_counts.get(idx, 0) + 1

    if not hit_counts:
        return []

    # sort by descending count
    sorted_ids = sorted(hit_counts.items(), key=lambda x: (-x[1], x[0]))
    return [i for i, _ in sorted_ids[:max_candidates]]


# -----------------------
# Main lexical run
# -----------------------
def run_lexical(
    df: pd.DataFrame,
    proxy_lines: List[str],
    inv: Dict[Tuple[str, ...], List[int]],
    line_ngrams: List[Set[Tuple[str, ...]]],
    limit: Optional[int],
    save_every: int,
    out_parquet: str,
    log_path: str,
    prefix: str,
    k: int = 13,
    max_candidates: int = 200
) -> Dict[str, Any]:
    pfx = norm_prefix(prefix)
    col_maxspan = f"MaxSpanLen_{pfx}".replace("__", "_").rstrip("_") if pfx else "MaxSpanLen"
    col_nghits = f"NgramHits_{pfx}".replace("__", "_").rstrip("_") if pfx else "NgramHits"
    col_pcount = f"ProxyCount_{pfx}".replace("__", "_").rstrip("_") if pfx else "ProxyCount"
    col_slex = f"SLex_{pfx}".replace("__", "_").rstrip("_") if pfx else "SLex"
    col_slex_agg = f"SLex_aggregate_{pfx}".replace("__", "_").rstrip("_") if pfx else "SLex_aggregate"

    # Output columns (model-agnostic, prefix-aware)
    for c in [col_maxspan, col_nghits, col_pcount, col_slex, col_slex_agg]:
        if c not in df.columns:
            # use integer nullable dtype to accept numeric writes
            df[c] = pd.Series([pd.NA] * len(df), dtype="Int64")

    processed_new = 0
    failures = 0

    for ridx, row in df.iterrows():
        # Resume-friendly: skip if this prefix's SLex already computed (numeric)
        existing = row.get(col_slex, "")
        if isinstance(existing, (int, float)) and pd.notna(existing):
            continue
        if isinstance(existing, str) and existing.strip() in ["0", "1", "2", "3"]:
            continue

        xsum_id = str(row.get("xsum_id", ridx))
        ref = row.get("summary_ref_norm", "")

        if not isinstance(ref, str) or not ref.strip():
            failures += 1
            log_jsonl(log_path, {
                "row": int(ridx),
                "xsum_id": xsum_id,
                "status": "error_missing_summary_ref_norm",
                "prefix": pfx.rstrip("_") if pfx else "",
            })
            continue

        ref = canonicalize_text(ref)
        ref_toks = tokenize(ref)
        ref_ngs = ngrams(ref_toks, k)

        # Candidate set based on n-gram overlap
        cand_ids = candidate_proxy_indices(ref_ngs, inv, max_candidates=max_candidates)

        # Compute NgramHits using candidate membership
        ngram_hits = 0
        if ref_ngs and cand_ids:
            # count unique ref ngrams that appear in any candidate line
            ref_ng_set = set(ref_ngs)
            for g in ref_ng_set:
                # if any candidate contains g
                if any(g in line_ngrams[cid] for cid in cand_ids):
                    ngram_hits += 1

        # Compute MaxSpanLen and ProxyCount (span>=50) over candidates.
        # If no candidates from ngrams, we can still do a light fallback:
        # take 0 (conservative) rather than scanning all proxy.
        max_span = 0
        proxy_count = 0
        if cand_ids:
            for cid in cand_ids:
                span = longest_common_substring_len(ref, proxy_lines[cid])
                if span > max_span:
                    max_span = span
                if span >= 50:
                    proxy_count += 1

        slex = map_to_SLex(max_span, ngram_hits, proxy_count)

        df.at[ridx, col_maxspan] = int(max_span)
        df.at[ridx, col_nghits] = int(ngram_hits)
        df.at[ridx, col_pcount] = int(proxy_count)
        df.at[ridx, col_slex] = int(slex)

        log_jsonl(log_path, {
            "row": int(ridx),
            "xsum_id": xsum_id,
            "status": "ok",
            "MaxSpanLen": int(max_span),
            "NgramHits": int(ngram_hits),
            "ProxyCount": int(proxy_count),
            "SLex": int(slex),
            "candidates": int(len(cand_ids)),
            "prefix": pfx.rstrip("_") if pfx else "",
        })

        processed_new += 1

        if save_every and processed_new > 0 and (processed_new % save_every == 0):
            ensure_parent_dir(out_parquet)
            df.to_parquet(out_parquet, index=False)
            print(f"Saved progress: {processed_new} new rows processed -> {out_parquet}")

        if limit is not None and processed_new >= limit:
            break

    slex_vals = pd.to_numeric(df[col_slex], errors="coerce")
    valid = slex_vals.notna()
    slex_aggregate = map_to_SLex_aggregate(slex_vals[valid]) if valid.any() else 0
    df[col_slex_agg] = pd.Series([slex_aggregate] * len(df), dtype="Int64")

    ensure_parent_dir(out_parquet)
    df.to_parquet(out_parquet, index=False)

    summary = {
        "processed_new": processed_new,
        "failures": failures,
        "valid_items": int(valid.sum()),
        "SLex_counts": {int(k): int(v) for k, v in slex_vals[valid].value_counts().items()} if valid.any() else {},
        "SLex_aggregate": int(slex_aggregate),
        "MaxSpanLen_mean": float(pd.to_numeric(df[col_maxspan], errors="coerce")[valid].mean()) if valid.any() else None,
        "NgramHits_mean": float(pd.to_numeric(df[col_nghits], errors="coerce")[valid].mean()) if valid.any() else None,
        "ProxyCount_mean": float(pd.to_numeric(df[col_pcount], errors="coerce")[valid].mean()) if valid.any() else None,
        "total_rows": int(len(df)),
        "prefix": pfx.rstrip("_") if pfx else "",
        "columns": {
            "MaxSpanLen": col_maxspan,
            "NgramHits": col_nghits,
            "ProxyCount": col_pcount,
            "SLex": col_slex,
            "SLex_aggregate": col_slex_agg,
        },
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to run_config.yaml")
    parser.add_argument("--proxy_path", type=str, default=None,
                        help="Override proxy corpus path (.txt per line OR .csv/.tsv with summary column). "
                             "Default: proxy_builder.merged_out from config.")
    parser.add_argument("--proxy_column", type=str, default=None,
                        help="Column name to use when proxy_path is CSV/TSV (default: summary_ref/summary/...)")
    parser.add_argument("--prefix", type=str, default="",
                        help="Optional column prefix for outputs. Example: --prefix web_ writes SLex_web, etc. "
                             "Default '' preserves legacy columns (SLex, MaxSpanLen, ...).")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N NEW rows (pilot runs)")
    parser.add_argument("--save_every", type=int, default=25, help="Save every N processed rows")
    parser.add_argument("--max_candidates", type=int, default=200, help="Max proxy candidates per item")
    parser.add_argument("--max_proxy_lines", type=int, default=None, help="Debug: cap proxy lines loaded")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    master_path = cfg["project"]["frozen_master_table_path"]
    proxy_cfg = cfg.get("proxy_builder", {})
    default_proxy_path = proxy_cfg.get("merged_out", "data/proxies/proxy_structured_merged.csv")
    df = pd.read_parquet(master_path)

    # Prefix-aware artifact names.
    # If no prefix is provided, always write to canonical legacy paths.
    p = norm_prefix(args.prefix).rstrip("_")

    if p:
        tag = p
        out_parquet = f"runs/v3_lexical_{tag}.parquet"
        log_path = f"logs/v3_lexical_{tag}.jsonl"
        summary_json = f"outputs/v3_lexical_summary_{tag}.json"
    else:
        tag = ""
        out_parquet = "runs/v3_lexical.parquet"
        log_path = "logs/v3_lexical.jsonl"
        summary_json = "outputs/v3_lexical_summary.json"

    proxy_path = Path(args.proxy_path or default_proxy_path)
    if not proxy_path.exists():
        alt = Path(default_proxy_path)
        if alt.exists():
            print(f"Proxy corpus not found at {proxy_path}, using {alt}")
            proxy_path = alt
        else:
            raise FileNotFoundError(f"Proxy corpus not found: {proxy_path}")

    print("Loading proxy corpus:", proxy_path)
    proxy_lines = load_proxy_lines(str(proxy_path), max_lines=args.max_proxy_lines, proxy_column=args.proxy_column)
    print("Proxy lines loaded:", len(proxy_lines))

    print("Building 13-gram inverted index (may take a bit)...")
    t0 = time.time()
    inv, line_ngrams = build_ngram_inverted_index(proxy_lines, k=13)
    print(f"Index built: {len(inv)} unique 13-grams in {time.time() - t0:.1f}s")

    t1 = time.time()
    results = run_lexical(
        df=df,
        proxy_lines=proxy_lines,
        inv=inv,
        line_ngrams=line_ngrams,
        limit=args.limit,
        save_every=args.save_every,
        out_parquet=out_parquet,
        log_path=log_path,
        prefix=args.prefix,
        max_candidates=args.max_candidates,
    )
    elapsed_s = time.time() - t1

    summary = {
        "stage": "lexical_proxy",
        "dataset_path": master_path,
        "proxy_path": str(proxy_path),
        "prefix": norm_prefix(args.prefix).rstrip("_") if args.prefix else "",
        "n_rows_total": int(len(df)),
        "processed_new": results["processed_new"],
        "failures": results["failures"],
        "valid_items": results["valid_items"],
        "SLex_counts": results["SLex_counts"],
        "SLex_aggregate": results["SLex_aggregate"],
        "MaxSpanLen_mean": results["MaxSpanLen_mean"],
        "NgramHits_mean": results["NgramHits_mean"],
        "ProxyCount_mean": results["ProxyCount_mean"],
        "k_ngram": 13,
        "max_candidates": args.max_candidates,
        "elapsed_seconds": elapsed_s,
        "out_parquet": out_parquet,
        "log_jsonl": log_path,
        "artifact_tag": tag,
    }

    write_json(summary_json, summary)

    print("Done.")
    print(f"Processed new: {results['processed_new']}; failures: {results['failures']}")
    print("SLex_counts:", results["SLex_counts"])
    print("SLex_aggregate:", results["SLex_aggregate"])
    print("Output:", out_parquet)
    print("Summary:", summary_json)
    print("Log:", log_path)


if __name__ == "__main__":
    main()
