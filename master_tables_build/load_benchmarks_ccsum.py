"""
load_benchmarks_ccsum.py
========================
Creates a frozen master table for ccsum/ccsum_summary_only.

Dataset slice:
  - dataset: ccsum/ccsum_summary_only
  - split:   test
  - rows:    300

The summary-only release does not include the full article body. To keep the
existing detector schema compatible, document/document_norm are populated from
available metadata: article_title, url, article_domain, and published date.
This is enough for prefix-based SMem checks, but it is not a full article
summarization input for SProb.
"""

from __future__ import annotations

import argparse
import html
import json
import random
import re
import unicodedata
from html.parser import HTMLParser

import numpy as np
import pandas as pd
import requests
from datasets import load_dataset


DEFAULT_SEED = 42
DEFAULT_N_ITEMS = 300
DEFAULT_DATASET_NAME = "ccsum/ccsum_summary_only"
DEFAULT_DATASET_SPLIT = "test"
DEFAULT_OUT_PARQUET = "master_table_ccsum_n300_seed42_v1.parquet"
DEFAULT_OUT_INDICES = "ccsum_test_indices_seed42_n300.json"
DEFAULT_SAMPLE_BUFFER_MULTIPLIER = 10
DEFAULT_MIN_DOCUMENT_CHARS = 200
DEFAULT_FETCH_TIMEOUT_S = 20


class ParagraphExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._in_p = False
        self._skip_depth = 0
        self._buf: list[str] = []
        self.paragraphs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        elif tag == "p" and self._skip_depth == 0:
            self._in_p = True
            self._buf = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "p" and self._in_p:
            text = normalize_text(html.unescape(" ".join(self._buf)))
            if len(text) >= 40:
                self.paragraphs.append(text)
            self._in_p = False
            self._buf = []

    def handle_data(self, data: str) -> None:
        if self._in_p and self._skip_depth == 0:
            self._buf.append(data)


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
    summary = normalize_text(row.get("summary", "") or "")
    title = normalize_text(row.get("article_title", "") or "")
    return (
        len(summary) >= 20
        and len(summary) <= 900
        and len(summary.split()) >= 13
        and len(title) >= 10
    )


def fetch_article_text(url: str, timeout_s: int = DEFAULT_FETCH_TIMEOUT_S) -> str:
    if not url:
        return ""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
        )
    }
    try:
        r = requests.get(url, headers=headers, timeout=timeout_s)
        if r.status_code != 200:
            return ""
        content_type = r.headers.get("content-type", "").lower()
        if "html" not in content_type and "text" not in content_type:
            return ""
        parser = ParagraphExtractor()
        parser.feed(r.text)
        return normalize_text(" ".join(parser.paragraphs))
    except Exception:
        return ""


def metadata_document(row: dict) -> str:
    parts = [
        row.get("article_title", ""),
        row.get("url", ""),
        row.get("article_domain", ""),
        str(row.get("date_publish", "") or ""),
    ]
    return normalize_text(" ".join(str(p) for p in parts if p))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CCSum summary-only master table.")
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset_split", default=DEFAULT_DATASET_SPLIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n_items", type=int, default=DEFAULT_N_ITEMS)
    parser.add_argument("--out_parquet", default=DEFAULT_OUT_PARQUET)
    parser.add_argument("--out_indices", default=DEFAULT_OUT_INDICES)
    parser.add_argument("--sample_buffer_multiplier", type=int, default=DEFAULT_SAMPLE_BUFFER_MULTIPLIER)
    parser.add_argument("--min_document_chars", type=int, default=DEFAULT_MIN_DOCUMENT_CHARS)
    parser.add_argument("--fetch_timeout_s", type=int, default=DEFAULT_FETCH_TIMEOUT_S)
    parser.add_argument(
        "--no_fetch_articles",
        action="store_true",
        help="Do not fetch article pages by URL; store metadata in document instead.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading {args.dataset_name} (split={args.dataset_split!r}) ...")
    ds = load_dataset(args.dataset_name, split=args.dataset_split)
    print(f"Total rows : {len(ds)}")
    print(f"Columns    : {ds.column_names}")

    random.seed(args.seed)
    indices = random.sample(
        range(len(ds)),
        min(args.n_items * args.sample_buffer_multiplier, len(ds)),
    )
    subset = ds.select(indices)

    with open(args.out_indices, "w", encoding="utf-8") as f:
        json.dump(indices, f)
    print(f"Saved {len(indices)} pre-filter indices -> {args.out_indices}")

    subset = subset.filter(valid)
    print(f"After filtering : {len(subset)} items")

    if len(subset) < args.n_items:
        raise RuntimeError(
            f"Only {len(subset)} valid rows after filtering, need {args.n_items}. "
            "Increase sampling buffer or relax valid()."
        )

    rows = []
    fetch_articles = not args.no_fetch_articles
    fetch_failures = 0

    for ex in subset:
        document = ""
        if fetch_articles:
            document = fetch_article_text(
                str(ex.get("url", "") or ""),
                timeout_s=args.fetch_timeout_s,
            )
            if len(document) < args.min_document_chars:
                fetch_failures += 1
                continue
        else:
            document = metadata_document(ex)

        if len(document) < args.min_document_chars:
            continue

        idx = len(rows)
        rows.append({
            "item_id": idx,
            "xsum_id": ex.get("id", f"ccsum_{idx}"),
            "split": args.dataset_split,
            "document": document,
            "summary_ref": ex["summary"],
            "title": ex.get("article_title", ""),
            "url": ex.get("url", ""),
            "published_date": str(ex.get("date_publish", "") or ""),
            "article_domain": ex.get("article_domain", ""),
            "summary_domain": ex.get("summary_domain", ""),
            "abstractiveness_bin": ex.get("abstractiveness_bin", ""),
            "summary_word_count": ex.get("summary_word_count", np.nan),
        })

        if len(rows) >= args.n_items:
            break

    if len(rows) < args.n_items:
        raise RuntimeError(
            f"Only {len(rows)} rows with usable documents, need {args.n_items}. "
            "Increase --sample_buffer_multiplier, lower --min_document_chars, "
            "or use --no_fetch_articles for summary-only compatibility mode."
        )

    print(f"Final sample    : {len(rows)} items")
    if fetch_articles:
        print(f"Article fetch failures/skips : {fetch_failures}")

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
