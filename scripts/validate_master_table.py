#!/usr/bin/env python3
"""
validate_master_table.py
------------------------
Pre-run validation of the master table for the SMem memorization probe detector.
Checks all conditions required for a correct reference pass and control pass.

Usage:
    python validate_master_table.py --table path/to/master_table.parquet

Checks performed:
  1.  Required columns presence
  2.  Null / empty field counts
  3.  prefix_ref → summary_ref_norm alignment (startswith)
  4.  gold_suffix extraction success rate
  5.  gold_suffix length distribution (chars and estimated tokens)
  6.  prefix_ref length distribution (words and chars)
  7.  Suspicious prefix lengths (too short / too long relative to full summary)
  8.  normalize_text side-effects on prefix_ref and summary_ref_norm
  9.  'split' field value distribution and fallback rate
  10. xsum_id uniqueness
  11. control_prefix presence and alignment behaviour (fallback rate)
  12. Overall readiness verdict
"""

import argparse
import unicodedata
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Text utilities (mirrors detector code exactly)
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


def estimate_tokens(text: str) -> int:
    """
    Rough token count estimate: ~4 chars per token for English.
    Good enough for a safety-margin check; no tokenizer required.
    """
    return max(1, len(text) // 4)


def check_nfkc_changes(text: str) -> list:
    """Return list of (original_char, normalised_char) pairs that differ."""
    changes = []
    for ch in text:
        norm = unicodedata.normalize("NFKC", ch)
        if norm != ch:
            changes.append((ch, norm))
    return changes


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

PASS  = "  [PASS]"
WARN  = "  [WARN]"
FAIL  = "  [FAIL]"
INFO  = "  [INFO]"
SEP   = "-" * 68


def section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def check_required_columns(df: pd.DataFrame) -> bool:
    section("CHECK 1 — Required columns")
    required = {
        "reference_pass":  ["xsum_id", "prefix_ref", "summary_ref_norm"],
        "control_pass":    ["control_prefix"],
        "prompt_template": ["split"],
    }
    all_ok = True
    for group, cols in required.items():
        for col in cols:
            if col in df.columns:
                print(f"{PASS}  '{col}'  ({group})")
            else:
                print(f"{FAIL}  '{col}' MISSING  ({group})")
                all_ok = False
    return all_ok


def check_nulls(df: pd.DataFrame) -> bool:
    section("CHECK 2 — Null / empty field counts")
    cols = ["xsum_id", "prefix_ref", "summary_ref_norm", "control_prefix", "split"]
    all_ok = True
    for col in cols:
        if col not in df.columns:
            print(f"{WARN}  '{col}' not in table — skipped")
            continue
        null_count  = df[col].isna().sum()
        empty_count = (df[col].astype(str).str.strip() == "").sum() - null_count
        empty_count = max(0, empty_count)
        total_bad   = null_count + empty_count
        status = PASS if total_bad == 0 else FAIL
        if total_bad > 0:
            all_ok = False
        print(f"{status}  '{col}':  nulls={null_count}  empty={empty_count}  "
              f"total_bad={total_bad}/{len(df)}")
    return all_ok


def check_alignment(df: pd.DataFrame) -> tuple:
    section("CHECK 3 — prefix_ref → summary_ref_norm alignment (startswith)")

    if "prefix_ref" not in df.columns or "summary_ref_norm" not in df.columns:
        print(f"{FAIL}  Required columns missing — cannot check alignment")
        return False, pd.Series(dtype=bool)

    aligned = df.apply(
        lambda r: bool(
            normalize_text(str(r["summary_ref_norm"])).startswith(
                normalize_text(str(r["prefix_ref"]))
            )
            and normalize_text(str(r["prefix_ref"]))
        ),
        axis=1,
    )
    n_ok   = aligned.sum()
    n_fail = (~aligned).sum()
    status = PASS if n_fail == 0 else FAIL

    print(f"{status}  Aligned: {n_ok}/{len(df)}   Not aligned: {n_fail}/{len(df)}")

    if n_fail > 0:
        print(f"\n  First 3 misaligned rows:")
        for _, row in df[~aligned].head(3).iterrows():
            p   = normalize_text(str(row["prefix_ref"]))[:70]
            ref = normalize_text(str(row["summary_ref_norm"]))[:70]
            print(f"    xsum_id={row.get('xsum_id', '?')}")
            print(f"      prefix_ref:       {p}")
            print(f"      summary_ref_norm: {ref}")

    return n_fail == 0, aligned


def check_gold_suffix(df: pd.DataFrame, aligned: pd.Series) -> bool:
    section("CHECK 4 — gold_suffix extraction success rate")

    if "prefix_ref" not in df.columns or "summary_ref_norm" not in df.columns:
        print(f"{FAIL}  Required columns missing")
        return False

    suffixes = df.apply(
        lambda r: extract_gold_suffix(
            str(r["summary_ref_norm"]), str(r["prefix_ref"])
        ),
        axis=1,
    )
    empty_suffix = (suffixes == "").sum()
    ok_suffix    = (suffixes != "").sum()
    status = PASS if empty_suffix == 0 else FAIL

    print(f"{status}  Extracted OK: {ok_suffix}/{len(df)}   "
          f"Empty (would be skipped): {empty_suffix}/{len(df)}")
    return empty_suffix == 0


def check_suffix_length(df: pd.DataFrame, max_tokens: int = 128) -> bool:
    section(f"CHECK 5 — gold_suffix length vs max_tokens={max_tokens}")

    if "prefix_ref" not in df.columns or "summary_ref_norm" not in df.columns:
        print(f"{FAIL}  Required columns missing")
        return False

    suffixes = df.apply(
        lambda r: extract_gold_suffix(
            str(r["summary_ref_norm"]), str(r["prefix_ref"])
        ),
        axis=1,
    )
    valid = suffixes[suffixes != ""]
    if valid.empty:
        print(f"{WARN}  No valid suffixes to measure")
        return False

    char_lens  = valid.str.len()
    token_ests = valid.apply(estimate_tokens)

    risky = (token_ests > max_tokens).sum()
    status = PASS if risky == 0 else WARN

    print(f"{INFO}  gold_suffix char length:  "
          f"min={char_lens.min()}  median={char_lens.median():.0f}  "
          f"mean={char_lens.mean():.1f}  max={char_lens.max()}  "
          f"p90={char_lens.quantile(0.90):.0f}")
    print(f"{INFO}  estimated token count:    "
          f"min={token_ests.min()}  median={token_ests.median():.0f}  "
          f"mean={token_ests.mean():.1f}  max={token_ests.max()}  "
          f"p90={token_ests.quantile(0.90):.0f}")
    print(f"{status}  Suffixes likely exceeding max_tokens ({max_tokens}): "
          f"{risky}/{len(valid)}")

    if risky > 0:
        long_ids = df.loc[token_ests[token_ests > max_tokens].index, "xsum_id"].tolist()
        print(f"  xsum_ids at risk: {long_ids[:10]}")

    return risky == 0


def check_prefix_length(df: pd.DataFrame) -> bool:
    section("CHECK 6 — prefix_ref length distribution")

    if "prefix_ref" not in df.columns or "summary_ref_norm" not in df.columns:
        print(f"{FAIL}  Required columns missing")
        return False

    prefix_words = df["prefix_ref"].apply(
        lambda x: len(normalize_text(str(x)).split())
    )
    summary_words = df["summary_ref_norm"].apply(
        lambda x: len(normalize_text(str(x)).split())
    )
    # Ratio: what fraction of the full summary does the prefix cover
    ratio = prefix_words / summary_words.replace(0, 1)

    too_short = (prefix_words < 4).sum()   # <4 words: trivial completion
    too_long  = (ratio > 0.85).sum()        # >85% of summary: trivial EM

    status = PASS if (too_short == 0 and too_long == 0) else WARN

    print(f"{INFO}  prefix word count:  "
          f"min={prefix_words.min()}  median={prefix_words.median():.0f}  "
          f"mean={prefix_words.mean():.1f}  max={prefix_words.max()}")
    print(f"{INFO}  prefix/summary ratio:  "
          f"min={ratio.min():.2f}  median={ratio.median():.2f}  "
          f"mean={ratio.mean():.2f}  max={ratio.max():.2f}")
    print(f"{status}  Too short (<4 words): {too_short}   "
          f"Too long (>85% of summary): {too_long}")

    return too_short == 0 and too_long == 0


def check_normalize_side_effects(df: pd.DataFrame) -> bool:
    section("CHECK 7 — normalize_text NFKC side-effects")

    cols_to_check = ["prefix_ref", "summary_ref_norm"]
    all_ok = True
    for col in cols_to_check:
        if col not in df.columns:
            continue
        changed_rows  = 0
        example_chars: list = []
        for val in df[col].dropna():
            changes = check_nfkc_changes(str(val))
            if changes:
                changed_rows += 1
                if len(example_chars) < 6:
                    example_chars.extend(changes[:2])

        status = PASS if changed_rows == 0 else WARN
        print(f"{status}  '{col}': rows with NFKC-changed chars: "
              f"{changed_rows}/{len(df)}")
        if example_chars:
            unique_ex = list({orig: norm for orig, norm in example_chars}.items())[:4]
            for orig, norm in unique_ex:
                print(f"         e.g. U+{ord(orig):04X} {repr(orig)} "
                      f"→ {repr(norm)}  "
                      f"(name: {unicodedata.name(orig, 'unknown')})")
            if changed_rows > 0:
                all_ok = False  # WARN not FAIL, but flag it

    return all_ok


def check_split_field(df: pd.DataFrame) -> bool:
    section("CHECK 8 — 'split' field values")

    if "split" not in df.columns:
        print(f"{WARN}  'split' column not found — "
              f"detector will use fallback 'test' for all rows")
        return True

    values     = df["split"].astype(str).str.strip()
    dist       = values.value_counts().to_dict()
    fallback_n = (values == "").sum() + df["split"].isna().sum()

    known_splits = {"train", "validation", "test"}
    unexpected   = {v for v in dist if v not in known_splits and v != "nan"}

    status = PASS if (fallback_n == 0 and not unexpected) else WARN

    print(f"{INFO}  Value distribution: {dist}")
    print(f"{status}  Rows using fallback 'test': {fallback_n}/{len(df)}")
    if unexpected:
        print(f"{WARN}  Unexpected split values: {unexpected}")

    return fallback_n == 0 and not unexpected


def check_xsum_id_uniqueness(df: pd.DataFrame) -> bool:
    section("CHECK 9 — xsum_id uniqueness")

    if "xsum_id" not in df.columns:
        print(f"{FAIL}  'xsum_id' column not found")
        return False

    total    = len(df)
    n_unique = df["xsum_id"].nunique()
    dupes    = total - n_unique
    status   = PASS if dupes == 0 else FAIL

    print(f"{status}  Total rows: {total}   Unique xsum_ids: {n_unique}   "
          f"Duplicates: {dupes}")

    if dupes > 0:
        dup_ids = df[df["xsum_id"].duplicated(keep=False)]["xsum_id"].unique()
        print(f"  Duplicate xsum_ids (first 10): {list(dup_ids[:10])}")

    return dupes == 0


def check_control_prefix(df: pd.DataFrame) -> bool:
    section("CHECK 10 — control_prefix alignment and fallback rate")

    if "control_prefix" not in df.columns:
        print(f"{WARN}  'control_prefix' column not found — "
              f"control pass will fail entirely")
        return False

    if "summary_ref_norm" not in df.columns:
        print(f"{FAIL}  'summary_ref_norm' missing — cannot check control alignment")
        return False

    # Mirrors run_control_pass logic exactly
    aligned_ctrl = df.apply(
        lambda r: bool(
            normalize_text(str(r["summary_ref_norm"])).startswith(
                normalize_text(str(r["control_prefix"]))
            )
            and normalize_text(str(r["control_prefix"]))
        ),
        axis=1,
    )
    n_aligned  = aligned_ctrl.sum()
    n_fallback = (~aligned_ctrl).sum()

    # Fallback means ctrl_suffix = full summary_ref_norm — see analysis notes
    if n_fallback == len(df):
        print(f"{INFO}  control_prefix is word-shuffled (never a true prefix) — "
              f"fallback fires on all {len(df)} rows.")
        print(f"{INFO}  This is expected for shuffle-based control design.")
        print(f"{INFO}  ctrl_suffix = full summary_ref_norm on every row.")
        print(f"{WARN}  EM_control will be systematically near 0 → "
              f"contrast ratio may be inflated. Conservative bias (fewer false positives).")
    elif n_fallback > 0:
        print(f"{WARN}  Fallback fires on {n_fallback}/{len(df)} rows "
              f"(ctrl_suffix = full ref).")
    else:
        print(f"{PASS}  control_prefix aligns on all rows — "
              f"no fallback needed.")

    # Check control_prefix is non-trivial (not same as prefix_ref)
    if "prefix_ref" in df.columns:
        same_as_ref = (
            df["control_prefix"].astype(str).str.strip()
            == df["prefix_ref"].astype(str).str.strip()
        ).sum()
        status = PASS if same_as_ref == 0 else FAIL
        print(f"{status}  control_prefix == prefix_ref: {same_as_ref}/{len(df)}")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate SMem detector master table before a probe run."
    )
    parser.add_argument(
        "--table", type=str, required=True,
        help="Path to the frozen master table parquet file.",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=128,
        help="max_tokens from decoding config (default: 128).",
    )
    args = parser.parse_args()

    path = Path(args.table)
    if not path.exists():
        print(f"[FAIL] File not found: {path}")
        sys.exit(1)

    print(f"\n{'='*68}")
    print(f"  SMem detector — master table validation")
    print(f"  Table : {path.name}")
    print(f"{'='*68}")

    df = pd.read_parquet(path)
    print(f"\n{INFO}  Rows: {len(df)}   Columns: {len(df.columns)}")
    print(f"{INFO}  Columns: {df.columns.tolist()}")

    results = {}

    results["columns"]        = check_required_columns(df)
    results["nulls"]          = check_nulls(df)
    ok_align, aligned_mask    = check_alignment(df)
    results["alignment"]      = ok_align
    results["gold_suffix"]    = check_gold_suffix(df, aligned_mask)
    results["suffix_length"]  = check_suffix_length(df, args.max_tokens)
    results["prefix_length"]  = check_prefix_length(df)
    results["nfkc"]           = check_normalize_side_effects(df)
    results["split"]          = check_split_field(df)
    results["xsum_id"]        = check_xsum_id_uniqueness(df)
    results["control"]        = check_control_prefix(df)

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    section("OVERALL VERDICT")

    critical = ["columns", "nulls", "alignment", "gold_suffix", "xsum_id"]
    advisory = ["suffix_length", "prefix_length", "nfkc", "split", "control"]

    crit_fails = [k for k in critical if not results[k]]
    adv_warns  = [k for k in advisory if not results[k]]

    if not crit_fails:
        print(f"{PASS}  All critical checks passed — "
              f"detector can run on this table.")
    else:
        print(f"{FAIL}  Critical checks FAILED: {crit_fails}")
        print(f"        Fix these before running the detector.")

    if adv_warns:
        print(f"{WARN}  Advisory checks with issues: {adv_warns}")
        print(f"        Review WARN messages above before proceeding.")
    else:
        print(f"{PASS}  All advisory checks clean.")

    print()
    sys.exit(0 if not crit_fails else 1)


if __name__ == "__main__":
    main()
