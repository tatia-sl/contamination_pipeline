#!/usr/bin/env python3
"""
scripts/run_mem_probe.py  (control-baseline enabled)

Memorization Probing Detector (SMem) runner for API-only contamination pipeline.

SIGNAL ASSIGNMENT follows the methodology's aggregate-level rubric:

  Aggregate metrics computed over all valid items:
    EM_rate    = mean(EM_ref)   -- proportion of exact reference reconstructions
    NE_rate    = mean(NE_ref)   -- proportion of near-exact reference reconstructions
    EM_control = mean(EM_ctrl)  -- proportion of exact control reconstructions
    NE_control = mean(NE_ctrl)  -- proportion of near-exact control reconstructions

  Model-level SMem (used for methodology reporting):
    Core rule:
      SMem = 0  iff exact_count == 0
      (no exact reconstructions observed in reference pass)

    If exact_count >= 1:
      SMem = 1  when EM_rate < 0.05 AND NE_rate < 0.15
      SMem = 2  when EM_rate >= 0.05 OR  NE_rate >= 0.15
      SMem = 3  when EM_rate >= 0.15 AND NE_rate >= 0.35
                and contrast condition is satisfied:
                EM_rate / EM_control >= 2.0

    Contrast handling:
      - if use_control_prefix = false: level-3 candidate is capped at SMem = 2
      - if use_control_prefix = true and EM_control == 0: level-3 is allowed

  Per-item SMem (used for item-level risk score in v7):
    SMem_item = 3  if EM_ref == 1
    SMem_item = 2  if NED_ref <= 0.10
    SMem_item = 1  if NED_ref <= 0.25
    SMem_item = 0  otherwise

COLUMNS WRITTEN to parquet (per model_id):
  mem_completion_{model_id}       -- raw reference completion
  mem_completion_ctrl_{model_id}  -- raw control completion (if use_control)
  EM_{model_id}                   -- binary exact match on reference (0/1)
  EM_ctrl_{model_id}              -- binary exact match on control (0/1)
  NED_{model_id}                  -- normalized edit distance, reference [0,1]
  NED_ctrl_{model_id}             -- normalized edit distance, control [0,1]
  NE_{model_id}                   -- binary near-exact on reference (NED<=0.10)
  NE_ctrl_{model_id}              -- binary near-exact on control
  SMem_{model_id}                 -- item-level signal (0-3), used in risk scoring

TERMINOLOGY:
  NED (Normalized Edit Distance): continuous in [0,1].
  NE  (Near-Exact):               binary, 1 when NED <= 0.10.
  EM  (Exact Match):              binary, 1 when completion == gold_suffix exactly.

CLI FLAGS:
  --config        path to run_config.yaml  (required)
  --model_id      model ID from config     (required)
  --limit         max NEW rows to process  (pilot runs)
  --control_only  run only control pass    (reference already done)

RESUME: skips rows where EM_{model_id} is already filled.
        For control_only mode, skips rows where EM_ctrl_{model_id} is filled.
"""

import argparse
import json
import time
import hashlib
import unicodedata
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prompts import MEM_PROMPT_TEMPLATE
from src.clients.openai_client import OpenAIClient


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def log_jsonl(path: str, payload: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = " ".join(s.split())
    return s.strip()


# ---------------------------------------------------------------------------
# Refusal / response-type classifier
# ---------------------------------------------------------------------------

REFUSAL_PATTERNS = {
    "dataset_access_disclaimer": [
        "i'm not able to access the full dataset",
        "i cannot access the dataset",
        "the actual completion may vary",
        "i don't have access to the original",
    ],
    "capability_refusal": [
        "i cannot provide",
        "i'm not able to",
        "i don't have access to",
        "i am unable to",
        "as an ai, i cannot",
    ],
    "meta_refusal": [
        "as an ai language model",
        "i should note that",
        "i want to clarify",
        "i must point out",
        "it's important to note that i",
    ],
}


def classify_response(completion_norm: str) -> Dict[str, Any]:
    """
    Classify a model completion as a valid attempt or a refusal variant.

    Categories:
      completion                 -- normal completion, usable for EM/NED scoring
      dataset_access_disclaimer  -- model disclaims access to the dataset/benchmark
      capability_refusal         -- generic capability refusal ("I cannot provide...")
      meta_refusal               -- meta-commentary instead of completion
      empty                      -- empty or whitespace-only response

    Returns dict with keys: response_type, is_refusal, is_valid_completion.
    Note: is_refusal=True does NOT set EM=1 or affect NED scoring;
    SMem_item is still computed as 0 for refusals, but SMem_interpretability
    is downgraded in the summary when refusal_rate is high.
    """
    if not isinstance(completion_norm, str) or not completion_norm.strip():
        return {
            "response_type": "empty",
            "is_refusal": True,
            "is_valid_completion": False,
        }
    text = completion_norm.lower().strip()
    for rtype, patterns in REFUSAL_PATTERNS.items():
        if any(p in text for p in patterns):
            return {
                "response_type": rtype,
                "is_refusal": True,
                "is_valid_completion": False,
            }
    return {
        "response_type": "completion",
        "is_refusal": False,
        "is_valid_completion": True,
    }


def extract_gold_suffix(ref_full: str, prefix: str) -> str:
    ref_full = normalize_text(ref_full)
    prefix   = normalize_text(prefix)
    if not ref_full or not prefix:
        return ""
    if ref_full.startswith(prefix):
        return ref_full[len(prefix):].strip()
    return ""


def build_mem_prompt(prefix: str, split_name: str) -> str:
    return MEM_PROMPT_TEMPLATE.format(SPLIT_NAME=split_name, PREFIX=prefix)


def select_client(model_cfg: Dict[str, Any]):
    provider  = model_cfg["provider"].lower()
    model_name = model_cfg["model_name"]
    api_key_var = model_cfg.get("env", {}).get("api_key_var")
    api_cfg    = model_cfg.get("api", {}) or {}

    if provider == "openai":
        api_key = os.environ.get(api_key_var or "OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(f"Missing {api_key_var or 'OPENAI_API_KEY'} env var")
        return OpenAIClient(
            model=model_name, api_key=api_key,
            api_mode=api_cfg.get("mode", "chat_completions"),
        )
    if provider == "openrouter":
        api_key = os.environ.get(api_key_var or "OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(f"Missing {api_key_var or 'OPENROUTER_API_KEY'} env var")
        base_url = api_cfg.get("base_url", "https://openrouter.ai/api/v1")
        extra_headers = {}
        if r := os.environ.get("OPENROUTER_HTTP_REFERER"):
            extra_headers["HTTP-Referer"] = r
        if t := os.environ.get("OPENROUTER_X_TITLE"):
            extra_headers["X-Title"] = t
        return OpenAIClient(
            model=model_name, api_key=api_key, base_url=base_url,
            extra_headers=extra_headers or None,
            api_mode=api_cfg.get("mode", "chat_completions"),
        )
    if provider == "groq":
        api_key = os.environ.get(api_key_var or "GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(f"Missing {api_key_var or 'GROQ_API_KEY'} env var")
        base_url = api_cfg.get("base_url", "https://api.groq.com/openai/v1")
        return OpenAIClient(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            api_mode=api_cfg.get("mode", "chat_completions"),
        )
    if provider == "gemini":
        from src.clients.gemini_client import GeminiClient
        return GeminiClient(model=model_name)

    raise ValueError(f"Unsupported provider: {provider}")


def format_path(template: str, model_id: str) -> str:
    return template.replace("{model_id}", model_id)


def normalized_edit_distance(a: str, b: str) -> float:
    """Character-level Levenshtein distance normalized to [0,1]."""
    a = a or ""
    b = b or ""
    if a == b:
        return 0.0
    if not a:
        return 1.0
    if not b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(min(curr[j-1]+1, prev[j]+1, prev[j-1]+(0 if ca==cb else 1)))
        prev = curr
    return float(prev[-1]) / float(max(len(a), len(b)))


def wilson_ci(count: int, n: int, z: float = 1.96) -> tuple:
    """
    Wilson score interval for a proportion count/n at confidence level z.

    Preferred over the normal (Wald) interval because it remains valid when
    n is small or the proportion is close to 0 or 1 — both common situations
    in contamination probing where EM_rate is typically very low.

    Args:
        count: number of successes (e.g. exact matches).
        n:     total number of valid items.
        z:     z-score for the desired confidence level (default 1.96 → 95%).

    Returns:
        (lower, upper) bounds, each rounded to 6 decimal places and clamped
        to [0, 1].
    """
    if n == 0:
        return (0.0, 0.0)
    p = count / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    margin = (z / denom) * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)) ** 0.5)
    lo = max(0.0, centre - margin)
    hi = min(1.0, centre + margin)
    return (round(lo, 6), round(hi, 6))


def map_to_SMem_item(em: int, ned: float) -> int:
    """
    Item-level SMem mapping used for per-item risk score computation in v7.
    Applied independently to each item regardless of control baseline.

      SMem = 3  if EM == 1          (exact reconstruction)
      SMem = 2  if NED <= 0.10      (near-verbatim)
      SMem = 1  if NED <= 0.25      (weak similarity)
      SMem = 0  otherwise
    """
    if em == 1:
        return 3
    if ned <= 0.10:
        return 2
    if ned <= 0.25:
        return 1
    return 0


def map_to_SMem_aggregate(
    em_rate: float,
    ne_rate: float,
    em_control: Optional[float],
    ne_control: Optional[float],
    use_control: bool,
    exact_count: int = 0,
) -> int:
    """
    Model-level SMem mapping per methodology aggregate rubric.
    Used for methodology-level reporting; NOT used in item-level risk scoring.

    Core rule: SMem = 0 if and only if exact_count == 0 (no verbatim
    reconstructions observed). Any observed exact match anchors the minimum
    level at 1 regardless of how low the rates are. This prevents large
    sample sizes from suppressing real contamination evidence via rate dilution.

    Level thresholds (applied only when exact_count > 0):
      SMem = 1  exact_count >= 1  AND EM_rate < 0.05 AND NE_rate < 0.15
      SMem = 2  exact_count >= 1  AND (EM_rate >= 0.05 OR NE_rate >= 0.15)
      SMem = 3  exact_count >= 1  AND EM_rate >= 0.15 AND NE_rate >= 0.35
                AND (EM_rate / EM_control >= 2.0 if use_control else capped at 2)
    """
    # SMem = 0 only when there are zero exact matches
    if exact_count == 0:
        return 0

    # From here: exact_count >= 1, minimum level is 1
    # Check for level 3 candidate first
    if em_rate >= 0.15 and ne_rate >= 0.35:
        if use_control and em_control is not None and em_control > 0:
            contrast = em_rate / em_control
            if contrast >= 2.0:
                return 3
            else:
                return 2
        elif use_control and (em_control is None or em_control == 0):
            return 3
        else:
            return 2

    # Level 2: either EM or NE crosses the level-2 threshold
    if em_rate >= 0.05 or ne_rate >= 0.15:
        return 2

    # Level 1: exact_count > 0 but rates below level-2 thresholds
    return 1


# ---------------------------------------------------------------------------
# Core probe logic — reference pass
# ---------------------------------------------------------------------------

def run_reference_pass(
    df: pd.DataFrame,
    client,
    model_id: str,
    decoding: Dict[str, Any],
    limit: Optional[int],
    sleep_s: float,
    save_every: int,
    out_parquet: str,
    log_path: str,
) -> Dict[str, Any]:
    """
    Probe each item with its reference prefix and record EM, NED, NE, SMem_item.
    Returns aggregate statistics over all valid items.
    """
    col_comp  = f"mem_completion_{model_id}"
    col_em    = f"EM_{model_id}"
    col_ned   = f"NED_{model_id}"
    col_ne    = f"NE_{model_id}"
    col_smem  = f"SMem_{model_id}"

    col_rtype   = f"response_type_{model_id}"
    col_refusal = f"is_refusal_{model_id}"
    col_valid   = f"is_valid_completion_{model_id}"

    for col, dtype in [
        (col_comp,    "object"),
        (col_em,      "Int64"),
        (col_ned,     "Float64"),
        (col_ne,      "Int64"),
        (col_smem,    "Int64"),
        (col_rtype,   "object"),
        (col_refusal, "Int64"),
        (col_valid,   "Int64"),
    ]:
        if col not in df.columns:
            df[col] = pd.array([pd.NA] * len(df), dtype=dtype) if dtype != "object" else ""
        elif dtype != "object":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)

    processed_new = 0
    failures = 0
    scanned = 0

    for idx, row in df.iterrows():
        scanned += 1
        # Resume: skip if EM already computed
        existing = row.get(col_em)
        if pd.notna(existing):
            continue

        item_key   = str(row.get("xsum_id", idx))
        prefix     = row.get("prefix_ref", "")
        ref        = row.get("summary_ref_norm", "")
        split_name = str(row.get("split", "test")).strip() or "test"

        if not isinstance(prefix, str) or not prefix.strip():
            failures += 1
            log_jsonl(log_path, {"xsum_id": item_key, "pass": "reference",
                                  "status": "error_missing_prefix_ref"})
            continue

        if not isinstance(ref, str) or not ref.strip():
            failures += 1
            log_jsonl(log_path, {"xsum_id": item_key, "pass": "reference",
                                  "status": "error_missing_summary_ref"})
            continue

        prefix_n   = normalize_text(prefix)
        ref_n      = normalize_text(ref)
        gold_suffix = extract_gold_suffix(ref_n, prefix_n)

        if not gold_suffix:
            failures += 1
            log_jsonl(log_path, {"xsum_id": item_key, "pass": "reference",
                                  "status": "error_prefix_not_aligned"})
            continue

        try:
            comp   = client.generate_text(
                prompt=build_mem_prompt(prefix_n, split_name),
                temperature=float(decoding["temperature"]),
                top_p=float(decoding["top_p"]),
                max_tokens=int(decoding["max_tokens"]),
            )
            comp_n     = normalize_text(comp)
            resp_class = classify_response(comp_n)
            em         = 1 if comp_n == gold_suffix else 0
            ned        = normalized_edit_distance(comp_n, gold_suffix)
            ne         = 1 if ned <= 0.10 else 0
            smem       = map_to_SMem_item(em=em, ned=ned)

            df.at[idx, col_comp]    = comp
            df.at[idx, col_em]      = int(em)
            df.at[idx, col_ned]     = float(ned)
            df.at[idx, col_ne]      = int(ne)
            df.at[idx, col_smem]    = int(smem)
            df.at[idx, col_rtype]   = resp_class["response_type"]
            df.at[idx, col_refusal] = int(resp_class["is_refusal"])
            df.at[idx, col_valid]   = int(resp_class["is_valid_completion"])

            log_jsonl(log_path, {
                "xsum_id": item_key, "pass": "reference", "status": "ok",
                "EM": em, "NED": round(ned, 6), "NE": ne, "SMem_item": smem,
                "response_type": resp_class["response_type"],
                "is_refusal": resp_class["is_refusal"],
                "gold_suffix": gold_suffix, "completion_norm": comp_n,
            })
            processed_new += 1

        except Exception as e:
            import traceback
            failures += 1
            log_jsonl(log_path, {
                "xsum_id": item_key, "pass": "reference", "status": "api_error",
                "error_type": type(e).__name__, "traceback": traceback.format_exc(),
            })

        if save_every and processed_new > 0 and processed_new % save_every == 0:
            ensure_parent_dir(out_parquet)
            df.to_parquet(out_parquet, index=False)
            print(
                f"Saved REFERENCE progress: scanned={scanned}, "
                f"processed={processed_new}, failures={failures} -> {out_parquet}",
                flush=True,
            )

        if limit is not None and processed_new >= limit:
            break

        time.sleep(float(sleep_s))

    ensure_parent_dir(out_parquet)
    df.to_parquet(out_parquet, index=False)
    print(
        f"REFERENCE pass done: scanned={scanned}, processed={processed_new}, "
        f"failures={failures} -> {out_parquet}",
        flush=True,
    )

    return _aggregate_stats(df, col_em, col_ned, col_ne, col_smem,
                            processed_new, failures)


# ---------------------------------------------------------------------------
# Core probe logic — control pass
# ---------------------------------------------------------------------------

def run_control_pass(
    df: pd.DataFrame,
    client,
    model_id: str,
    decoding: Dict[str, Any],
    limit: Optional[int],
    sleep_s: float,
    save_every: int,
    out_parquet: str,
    log_path: str,
) -> Dict[str, Any]:
    """
    Probe each item with its control prefix and record EM_ctrl, NED_ctrl, NE_ctrl.
    The control suffix target is derived from control_prefix + remainder of summary_ref_norm.
    Returns aggregate statistics over valid control items.
    """
    col_comp_ctrl = f"mem_completion_ctrl_{model_id}"
    col_em_ctrl   = f"EM_ctrl_{model_id}"
    col_ned_ctrl  = f"NED_ctrl_{model_id}"
    col_ne_ctrl   = f"NE_ctrl_{model_id}"

    for col, dtype in [
        (col_comp_ctrl, "object"),
        (col_em_ctrl,   "Int64"),
        (col_ned_ctrl,  "Float64"),
        (col_ne_ctrl,   "Int64"),
    ]:
        if col not in df.columns:
            df[col] = pd.array([pd.NA] * len(df), dtype=dtype) if dtype != "object" else ""
        elif dtype != "object":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)

    processed_new = 0
    failures = 0
    scanned = 0

    for idx, row in df.iterrows():
        scanned += 1
        # Resume: skip if EM_ctrl already computed
        existing = row.get(col_em_ctrl)
        if pd.notna(existing):
            continue

        item_key   = str(row.get("xsum_id", idx))
        ctrl       = row.get("control_prefix", "")
        ref        = row.get("summary_ref_norm", "")
        split_name = str(row.get("split", "test")).strip() or "test"

        if not isinstance(ctrl, str) or not ctrl.strip():
            failures += 1
            log_jsonl(log_path, {"xsum_id": item_key, "pass": "control",
                                  "status": "error_missing_control_prefix"})
            continue

        if not isinstance(ref, str) or not ref.strip():
            failures += 1
            log_jsonl(log_path, {"xsum_id": item_key, "pass": "control",
                                  "status": "error_missing_summary_ref"})
            continue

        ctrl_n     = normalize_text(ctrl)
        ref_n      = normalize_text(ref)
        # Use same suffix-extraction logic: control suffix = what follows control_prefix
        ctrl_suffix = extract_gold_suffix(ref_n, ctrl_n)

        if not ctrl_suffix:
            # control_prefix may not align with the same reference — use full ref as target
            # This is an acceptable approximation: we measure how well the model reproduces
            # the reference summary when given a non-contaminated prompt.
            ctrl_suffix = ref_n

        try:
            comp_ctrl  = client.generate_text(
                prompt=build_mem_prompt(ctrl_n, split_name),
                temperature=float(decoding["temperature"]),
                top_p=float(decoding["top_p"]),
                max_tokens=int(decoding["max_tokens"]),
            )
            comp_ctrl_n = normalize_text(comp_ctrl)
            em_ctrl     = 1 if comp_ctrl_n == ctrl_suffix else 0
            ned_ctrl    = normalized_edit_distance(comp_ctrl_n, ctrl_suffix)
            ne_ctrl     = 1 if ned_ctrl <= 0.10 else 0

            df.at[idx, col_comp_ctrl] = comp_ctrl
            df.at[idx, col_em_ctrl]   = int(em_ctrl)
            df.at[idx, col_ned_ctrl]  = float(ned_ctrl)
            df.at[idx, col_ne_ctrl]   = int(ne_ctrl)

            log_jsonl(log_path, {
                "xsum_id": item_key, "pass": "control", "status": "ok",
                "EM_ctrl": em_ctrl, "NED_ctrl": round(ned_ctrl, 6), "NE_ctrl": ne_ctrl,
                "ctrl_suffix": ctrl_suffix, "completion_norm": comp_ctrl_n,
            })
            processed_new += 1

        except Exception as e:
            import traceback
            failures += 1
            log_jsonl(log_path, {
                "xsum_id": item_key, "pass": "control", "status": "api_error",
                "error_type": type(e).__name__, "traceback": traceback.format_exc(),
            })

        if save_every and processed_new > 0 and processed_new % save_every == 0:
            ensure_parent_dir(out_parquet)
            df.to_parquet(out_parquet, index=False)
            print(
                f"Saved CONTROL progress: scanned={scanned}, "
                f"processed={processed_new}, failures={failures} -> {out_parquet}",
                flush=True,
            )

        if limit is not None and processed_new >= limit:
            break

        time.sleep(float(sleep_s))

    ensure_parent_dir(out_parquet)
    df.to_parquet(out_parquet, index=False)
    print(
        f"CONTROL pass done: scanned={scanned}, processed={processed_new}, "
        f"failures={failures} -> {out_parquet}",
        flush=True,
    )

    em_ctrl_vals  = pd.to_numeric(df[col_em_ctrl],  errors="coerce")
    ned_ctrl_vals = pd.to_numeric(df[col_ned_ctrl], errors="coerce")
    ne_ctrl_vals  = pd.to_numeric(df[col_ne_ctrl],  errors="coerce")
    valid_ctrl    = em_ctrl_vals.notna()

    return {
        "processed_new": processed_new,
        "failures": failures,
        "valid_items": int(valid_ctrl.sum()),
        "EM_control": float(em_ctrl_vals[valid_ctrl].mean())  if valid_ctrl.any() else None,
        "NE_control": float(ne_ctrl_vals[valid_ctrl].mean())  if valid_ctrl.any() else None,
        "NED_ctrl_mean": float(ned_ctrl_vals[valid_ctrl].mean()) if valid_ctrl.any() else None,
    }


# ---------------------------------------------------------------------------
# Aggregate stats helper
# ---------------------------------------------------------------------------

def _aggregate_stats(
    df: pd.DataFrame,
    col_em: str,
    col_ned: str,
    col_ne: str,
    col_smem: str,
    processed_new: int,
    failures: int,
) -> Dict[str, Any]:
    em_vals  = pd.to_numeric(df[col_em],  errors="coerce")
    ned_vals = pd.to_numeric(df[col_ned], errors="coerce")
    ne_vals  = pd.to_numeric(df[col_ne],  errors="coerce")
    valid    = em_vals.notna() & ned_vals.notna()

    n                = int(valid.sum())
    exact_count      = int(em_vals[valid].sum())
    ne_count         = int(ne_vals[valid].sum())
    near_exact_count = ne_count - exact_count
    non_match_count  = n - exact_count - near_exact_count

    # Wilson score 95 % CI for EM_rate and NE_rate.
    # More reliable than the normal (Wald) interval when n is small or the
    # proportion is near 0, which is the typical regime for EM in probing.
    em_ci = wilson_ci(exact_count, n) if n > 0 else (None, None)
    ne_ci = wilson_ci(ne_count,    n) if n > 0 else (None, None)

    return {
        "processed_new": processed_new,
        "failures": failures,
        "valid_items": n,
        "EM_rate":    float(em_vals[valid].mean())          if valid.any() else None,
        "EM_ci_lo":   em_ci[0],
        "EM_ci_hi":   em_ci[1],
        "NE_rate":    float(ne_vals[valid].mean())          if valid.any() else None,
        "NE_ci_lo":   ne_ci[0],
        "NE_ci_hi":   ne_ci[1],
        "NED_mean":   float(ned_vals[valid].mean())         if valid.any() else None,
        "NED_median": float(ned_vals[valid].median())       if valid.any() else None,
        "NED_p10":    float(ned_vals[valid].quantile(0.10)) if valid.any() else None,
        "NED_p90":    float(ned_vals[valid].quantile(0.90)) if valid.any() else None,
        "exact_count":      exact_count,
        "near_exact_count": near_exact_count,
        "non_match_count":  non_match_count,
        "dominant_SMem_item": (
            int(df[col_smem].dropna().mode()[0])
            if df[col_smem].notna().any() else None
        ),
    }


def _ensure_reference_item_signals(df: pd.DataFrame, model_id: str) -> int:
    """
    Ensure per-item reference memorization signals are present.
    Backfills missing NE/SMem from EM+NED when possible.

    Returns:
      number of rows where SMem was backfilled
    """
    col_em = f"EM_{model_id}"
    col_ned = f"NED_{model_id}"
    col_ne = f"NE_{model_id}"
    col_smem = f"SMem_{model_id}"

    if col_em not in df.columns or col_ned not in df.columns:
        return 0

    # Ensure target columns exist and are numeric nullable.
    if col_ne not in df.columns:
        df[col_ne] = pd.array([pd.NA] * len(df), dtype="Int64")
    else:
        df[col_ne] = pd.to_numeric(df[col_ne], errors="coerce").astype("Int64")

    if col_smem not in df.columns:
        df[col_smem] = pd.array([pd.NA] * len(df), dtype="Int64")
    else:
        df[col_smem] = pd.to_numeric(df[col_smem], errors="coerce").astype("Int64")

    em_vals = pd.to_numeric(df[col_em], errors="coerce")
    ned_vals = pd.to_numeric(df[col_ned], errors="coerce")
    can_compute = em_vals.notna() & ned_vals.notna()

    # Backfill NE where missing from NED threshold.
    ne_missing = df[col_ne].isna() & can_compute
    if ne_missing.any():
        df.loc[ne_missing, col_ne] = (ned_vals[ne_missing] <= 0.10).astype(int)

    # Backfill SMem where missing from EM+NED item mapping.
    smem_missing = df[col_smem].isna() & can_compute
    if smem_missing.any():
        backfilled = [
            map_to_SMem_item(int(em), float(ned))
            for em, ned in zip(em_vals[smem_missing], ned_vals[smem_missing])
        ]
        df.loc[smem_missing, col_smem] = backfilled
        return int(smem_missing.sum())

    return 0


def _aggregate_reference_from_existing(df: pd.DataFrame, model_id: str) -> Dict[str, Any]:
    """
    Compute reference aggregate metrics from existing parquet columns.
    Returns empty dict if required columns are unavailable.
    """
    col_em = f"EM_{model_id}"
    col_ned = f"NED_{model_id}"
    col_ne = f"NE_{model_id}"
    col_smem = f"SMem_{model_id}"

    if any(c not in df.columns for c in [col_em, col_ned, col_ne, col_smem]):
        return {}

    stats = _aggregate_stats(
        df=df,
        col_em=col_em,
        col_ned=col_ned,
        col_ne=col_ne,
        col_smem=col_smem,
        processed_new=0,
        failures=0,
    )
    return stats


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       type=str, required=True)
    parser.add_argument("--model_id",     type=str, required=True)
    parser.add_argument("--limit",        type=int, default=None)
    parser.add_argument("--control_only", action="store_true",
                        help="Run only the control prefix pass (reference already done)")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    model_cfg = next(
        (m for m in cfg["models"] if m["model_id"] == args.model_id), None
    )
    if model_cfg is None:
        raise ValueError(f"model_id='{args.model_id}' not found in config")

    mem_cfg    = cfg["memorization"]
    decoding   = mem_cfg["decoding"]
    use_control = bool(mem_cfg.get("use_control_prefix", False))
    sleep_s    = float(mem_cfg["runtime"]["sleep_s"])
    save_every = int(mem_cfg["runtime"]["save_every"])

    out_parquet  = format_path(mem_cfg["outputs"]["parquet"],      args.model_id)
    log_path     = format_path(mem_cfg["outputs"]["log_jsonl"],    args.model_id)
    summary_json = format_path(mem_cfg["outputs"]["summary_json"], args.model_id)

    master_path = cfg["project"]["frozen_master_table_path"]
    if Path(out_parquet).exists():
        print(f"[{args.model_id}] Resuming from existing output: {out_parquet}")
        df = pd.read_parquet(out_parquet)
    else:
        df = pd.read_parquet(master_path)

    # For control-only and resume scenarios, ensure item-level reference signal
    # columns are available for downstream risk integration.
    smem_backfilled = _ensure_reference_item_signals(df, args.model_id)
    if smem_backfilled > 0:
        ensure_parent_dir(out_parquet)
        df.to_parquet(out_parquet, index=False)
        print(f"[{args.model_id}] Backfilled SMem for {smem_backfilled} rows from existing EM/NED.")

    client = select_client(model_cfg)
    t0 = time.time()

    ref_stats = _aggregate_reference_from_existing(df, args.model_id)
    ctrl_stats = {}

    if not args.control_only:
        print(f"[{args.model_id}] Running REFERENCE pass...")
        ref_stats = run_reference_pass(
            df=df, client=client, model_id=args.model_id,
            decoding=decoding, limit=args.limit,
            sleep_s=sleep_s, save_every=save_every,
            out_parquet=out_parquet, log_path=log_path,
        )
        # Reload parquet to get updated df for control pass
        df = pd.read_parquet(out_parquet)
        print(f"  EM_rate={ref_stats['EM_rate']:.4f}  "
              f"NE_rate={ref_stats['NE_rate']:.4f}  "
              f"exact={ref_stats['exact_count']}  "
              f"near_exact={ref_stats['near_exact_count']}")
    elif not ref_stats:
        print(f"[{args.model_id}] control_only: reference columns not found in existing parquet; "
              "SMem-based risk integration will be unavailable until reference pass is run.")

    if use_control or args.control_only:
        print(f"[{args.model_id}] Running CONTROL pass...")
        ctrl_stats = run_control_pass(
            df=df, client=client, model_id=args.model_id,
            decoding=decoding, limit=args.limit,
            sleep_s=sleep_s, save_every=save_every,
            out_parquet=out_parquet, log_path=log_path,
        )
        df = pd.read_parquet(out_parquet)
        print(f"  EM_control={ctrl_stats['EM_control']}  "
              f"NE_control={ctrl_stats['NE_control']}")

    # Compute aggregate SMem (model-level, for methodology reporting)
    em_rate  = ref_stats.get("EM_rate")
    ne_rate  = ref_stats.get("NE_rate")
    em_ctrl  = ctrl_stats.get("EM_control")
    ne_ctrl  = ctrl_stats.get("NE_control")

    smem_aggregate  = None
    contrast_ratio  = None
    em_level        = None
    ne_level        = None
    signal_conflict = None

    if em_rate is not None and ne_rate is not None:
        smem_aggregate = map_to_SMem_aggregate(
            em_rate=em_rate, ne_rate=ne_rate,
            em_control=em_ctrl, ne_control=ne_ctrl,
            use_control=(use_control or args.control_only),
            exact_count=ref_stats.get("exact_count", 0) or 0,
        )
        if em_ctrl and em_ctrl > 0:
            contrast_ratio = round(em_rate / em_ctrl, 4)

        # Individual signal levels — used to detect EM/NE disagreement
        em_level = (0 if em_rate < 0.01 else
                    1 if em_rate < 0.05 else
                    2 if em_rate < 0.15 else 3)
        ne_level = (0 if ne_rate < 0.05 else
                    1 if ne_rate < 0.15 else
                    2 if ne_rate < 0.35 else 3)
        signal_conflict = (em_level != ne_level)

    elapsed_s = time.time() - t0

    # ------------------------------------------------------------------
    # Refusal diagnostics
    # ------------------------------------------------------------------
    col_refusal_diag = f"is_refusal_{args.model_id}"
    col_valid_diag   = f"is_valid_completion_{args.model_id}"
    col_rtype_diag   = f"response_type_{args.model_id}"

    refusal_rate          = None
    valid_completion_rate = None
    refusal_breakdown     = None
    smem_interpretability = "UNKNOWN"

    if col_refusal_diag in df.columns:
        n_total   = len(df)
        n_refusal = int(df[col_refusal_diag].fillna(0).sum())
        n_valid   = int(df[col_valid_diag].fillna(0).sum()) if col_valid_diag in df.columns else None
        refusal_rate          = round(n_refusal / n_total, 4) if n_total else None
        valid_completion_rate = round(n_valid   / n_total, 4) if (n_valid is not None and n_total) else None

        if col_rtype_diag in df.columns:
            refusal_breakdown = (
                df[df[col_refusal_diag] == 1][col_rtype_diag]
                .value_counts()
                .to_dict()
            )

        if refusal_rate is not None:
            if refusal_rate >= 0.50:
                smem_interpretability = "LOW"
            elif refusal_rate >= 0.25:
                smem_interpretability = "MODERATE"
            else:
                smem_interpretability = "HIGH"

    summary = {
        "stage": "memorization_probe_v2",
        "model_id": args.model_id,
        "provider": model_cfg["provider"],
        "model_name": model_cfg["model_name"],
        "dataset_path": master_path,
        "n_rows_total": int(len(df)),
        # Reference pass
        "processed_new_ref":  ref_stats.get("processed_new"),
        "failures_ref":       ref_stats.get("failures"),
        "valid_items":        ref_stats.get("valid_items"),
        "EM_rate":            ref_stats.get("EM_rate"),
        "EM_ci_lo":           ref_stats.get("EM_ci_lo"),
        "EM_ci_hi":           ref_stats.get("EM_ci_hi"),
        "NE_rate":            ref_stats.get("NE_rate"),
        "NE_ci_lo":           ref_stats.get("NE_ci_lo"),
        "NE_ci_hi":           ref_stats.get("NE_ci_hi"),
        "NED_mean":           ref_stats.get("NED_mean"),
        "NED_median":         ref_stats.get("NED_median"),
        "NED_p10":            ref_stats.get("NED_p10"),
        "NED_p90":            ref_stats.get("NED_p90"),
        "exact_count":        ref_stats.get("exact_count"),
        "near_exact_count":   ref_stats.get("near_exact_count"),
        "non_match_count":    ref_stats.get("non_match_count"),
        "dominant_SMem_item": ref_stats.get("dominant_SMem_item"),
        # Control pass
        "use_control_prefix":   use_control or args.control_only,
        "processed_new_ctrl":   ctrl_stats.get("processed_new"),
        "failures_ctrl":        ctrl_stats.get("failures"),
        "EM_control":           em_ctrl,
        "NE_control":           ne_ctrl,
        "NED_ctrl_mean":        ctrl_stats.get("NED_ctrl_mean"),
        # Aggregate SMem (model-level, methodology rubric)
        "SMem_aggregate":       smem_aggregate,
        "contrast_ratio_EM":    contrast_ratio,
        "em_level":             em_level,
        "ne_level":             ne_level,
        "signal_conflict":      signal_conflict,
        # Refusal diagnostics
        "refusal_rate":           refusal_rate,
        "valid_completion_rate":  valid_completion_rate,
        "refusal_breakdown":      refusal_breakdown,
        "SMem_interpretability":  smem_interpretability,
        # Run metadata
        "decoding":         decoding,
        "elapsed_seconds":  elapsed_s,
        "out_parquet":      out_parquet,
        "log_jsonl":        log_path,
    }

    ensure_parent_dir(summary_json)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Model: {args.model_id} ({model_cfg['model_name']})")
    print(f"SMem_aggregate (model-level): {smem_aggregate}  "
          f"contrast_ratio_EM: {contrast_ratio}  "
          f"em_level: {em_level}  ne_level: {ne_level}  "
          f"signal_conflict: {signal_conflict}")
    print(f"Output: {out_parquet}")
    print(f"Summary: {summary_json}")
    print(f"Log: {log_path}")

    # Post-check block: always print the 6 key aggregate fields.
    print("\n[POST-CHECK] Memorization aggregate fields")
    print(f"  EM_rate: {summary.get('EM_rate')}  "
          f"95% CI: [{summary.get('EM_ci_lo')}, {summary.get('EM_ci_hi')}]  "
          f"n={summary.get('valid_items')}")
    print(f"  NE_rate: {summary.get('NE_rate')}  "
          f"95% CI: [{summary.get('NE_ci_lo')}, {summary.get('NE_ci_hi')}]")
    print(f"  EM_control: {summary.get('EM_control')}")
    print(f"  NE_control: {summary.get('NE_control')}")
    print(f"  SMem_aggregate: {summary.get('SMem_aggregate')}")
    print(f"  contrast_ratio_EM: {summary.get('contrast_ratio_EM')}")
    print(f"  refusal_rate: {summary.get('refusal_rate')}  "
          f"valid_completion_rate: {summary.get('valid_completion_rate')}  "
          f"SMem_interpretability: {summary.get('SMem_interpretability')}")


if __name__ == "__main__":
    main()
