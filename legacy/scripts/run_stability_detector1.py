#!/usr/bin/env python3
"""
scripts/run_stability_detector.py

Stability-based Probability Proxy (SProb) runner for API-only contamination pipeline.

Methodology: Dong et al. (2024) CDD — Contamination Detection via output Distribution.
No external control set or baseline. Detection is fully self-contained per document:
greedy anchor (temperature=0) serves as the internal reference point.

v1 changes:
  - adds greedy anchor generation (temperature=0)
  - switches mNED to token-level normalized edit distance
  - SProb=0 only when ALL signals are clean
  - any weak evidence already maps to SProb>=1

v2 changes:
  - [FIX 2] API calls wrapped in exponential backoff retry (3 attempts,
      delays 1s / 4s / 16s). Prevents silent item loss on transient API errors.
  - [FIX 3] Partial-output recovery: existing stochastic outputs reloaded from
      parquet; only missing samples are fetched. Greedy reused if already present.

v4 changes (current):
  - [ARCH] Replaced _band_absolute + _band_anchor + max() with a single
      unified map_to_SProb function (Variant B).
  - SProb=0 requires ALL FOUR signals clean simultaneously.
  - SProb=3/2/1 use AND on stochastic axis (UAR+mNED must both be weak)
      and OR on anchor axis (peak_eps or anchor_mNED alone is sufficient).
  - Eliminates false positives where one borderline metric raised SProb
      despite the other three being clean.

SProb mapping (first match wins):
  SProb=0  UAR>0.60 AND mNED>0.25 AND anchor_mNED>=0.25 AND peak_eps<0.25
  SProb=3  (UAR<0.20 AND mNED<0.08) OR peak_eps>=0.75 OR anchor_mNED<0.08
  SProb=2  (UAR<0.40 AND mNED<0.15) OR peak_eps>=0.50 OR anchor_mNED<0.15
  SProb=1  residual (some signal, below level-2 thresholds)
"""

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prompts import STABILITY_PROMPT_TEMPLATE
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


def format_path(template: str, model_id: str) -> str:
    return template.replace("{model_id}", model_id)


def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = " ".join(s.split())
    return s.strip()


def get_document_field(row: pd.Series) -> str:
    for col in ["document_norm", "document"]:
        if col in row and isinstance(row[col], str) and row[col].strip():
            return row[col]
    return ""


def build_stability_prompt(document: str) -> str:
    return STABILITY_PROMPT_TEMPLATE.format(DOCUMENT=document)


# ---------------------------------------------------------------------------
# FIX 2: Retry helper with exponential backoff
# ---------------------------------------------------------------------------

def generate_with_retry(
    client,
    *,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_attempts: int = 3,
    base_delay: float = 1.0,
) -> str:
    """Call client.generate_text with exponential backoff on failure.

    Delays: 1s, 4s, 16s (base_delay * 4^attempt).
    Raises the last exception if all attempts are exhausted.
    """
    last_exc: Exception = RuntimeError("No attempts made")
    for attempt in range(max_attempts):
        try:
            result = client.generate_text(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            return (result or "").strip()
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                delay = base_delay * (4 ** attempt)
                print(
                    f"    [retry] attempt {attempt + 1}/{max_attempts} failed "
                    f"({type(exc).__name__}), retrying in {delay:.0f}s..."
                )
                time.sleep(delay)
    raise last_exc


# ---------------------------------------------------------------------------
# Client selection
# ---------------------------------------------------------------------------

def select_client(model_cfg: Dict[str, Any]):
    provider = model_cfg["provider"].lower()
    model_name = model_cfg["model_name"]
    api_key_var = model_cfg.get("env", {}).get("api_key_var")
    api_cfg = model_cfg.get("api", {}) or {}

    if provider == "openai":
        api_key = os.environ.get(api_key_var or "OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(f"Missing {api_key_var or 'OPENAI_API_KEY'} env var")
        return OpenAIClient(
            model=model_name,
            api_key=api_key,
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
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            extra_headers=extra_headers or None,
            api_mode=api_cfg.get("mode", "chat_completions"),
        )
    if provider == "gemini":
        from src.clients.gemini_client import GeminiClient
        return GeminiClient(model=model_name)

    raise ValueError(f"Unsupported provider: {provider}")


# ---------------------------------------------------------------------------
# Token-level utilities
# ---------------------------------------------------------------------------

def regex_tokenize(text: str) -> List[str]:
    text = normalize_text(text).lower()
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def encode_tokens(text: str, token_encoder=None) -> List[Any]:
    text = normalize_text(text).lower()
    if token_encoder is None:
        return regex_tokenize(text)

    if callable(token_encoder):
        out = token_encoder(text)
        return list(out) if out is not None else []

    if hasattr(token_encoder, "encode"):
        try:
            return list(token_encoder.encode(text, add_special_tokens=False))
        except TypeError:
            try:
                return list(token_encoder.encode(text))
            except Exception:
                pass

    if hasattr(token_encoder, "encode_ordinary"):
        return list(token_encoder.encode_ordinary(text))

    raise TypeError(
        "Unsupported token_encoder. Pass callable(text)->tokens or tokenizer with .encode(...)."
    )


def normalized_token_edit_distance(a: str, b: str, token_encoder=None) -> float:
    ta = encode_tokens(a, token_encoder=token_encoder)
    tb = encode_tokens(b, token_encoder=token_encoder)

    if ta == tb:
        return 0.0
    if not ta or not tb:
        return 1.0

    prev = list(range(len(tb) + 1))
    for i, xa in enumerate(ta, start=1):
        curr = [i]
        for j, xb in enumerate(tb, start=1):
            curr.append(min(
                curr[j - 1] + 1,
                prev[j] + 1,
                prev[j - 1] + (0 if xa == xb else 1),
            ))
        prev = curr
    return float(prev[-1]) / float(max(len(ta), len(tb)))


# ---------------------------------------------------------------------------
# Stability metrics
# ---------------------------------------------------------------------------

def compute_uar(outputs: List[str]) -> float:
    normed = [normalize_text(x).lower() for x in outputs if isinstance(x, str)]
    if not normed:
        return float("nan")
    return len(set(normed)) / len(normed)


def compute_mned_pairwise(
    outputs: List[str],
    max_pairs: Optional[int] = None,
    token_encoder=None,
) -> float:
    normed = [normalize_text(x).lower() for x in outputs if isinstance(x, str)]
    n = len(normed)
    if n < 2:
        return float("nan")

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if max_pairs is not None and len(pairs) > max_pairs:
        step = max(1, len(pairs) // max_pairs)
        pairs = pairs[::step][:max_pairs]

    dists = [
        normalized_token_edit_distance(normed[i], normed[j], token_encoder=token_encoder)
        for i, j in pairs
    ]
    return float(sum(dists) / len(dists)) if dists else float("nan")


def compute_anchor_metrics(
    outputs: List[str],
    greedy_output: str,
    *,
    anchor_eps: float = 0.15,
    token_encoder=None,
) -> Tuple[float, float]:
    normed = [normalize_text(x).lower() for x in outputs if isinstance(x, str)]
    greedy = normalize_text(greedy_output).lower()
    if not normed or not greedy:
        return float("nan"), float("nan")

    dists = [normalized_token_edit_distance(x, greedy, token_encoder=token_encoder) for x in normed]
    anchor_mned = float(sum(dists) / len(dists))
    peak_eps = float(sum(d <= anchor_eps for d in dists) / len(dists))
    return anchor_mned, peak_eps


# ---------------------------------------------------------------------------
# SProb mapping
# ---------------------------------------------------------------------------

def map_to_SProb(
    *,
    uar: float,
    mned: float,
    anchor_mned: float,
    peak_eps: float,
) -> int:
    """Map four stability metrics to a single contamination signal (0..3).

    Follows Dong et al. (2024) CDD: greedy anchor (temperature=0) is the
    internal reference point — no external baseline required.

    Rules (evaluated top-to-bottom, first match wins):

      SProb=0  All four signals clean:
                 UAR > 0.60  AND  mNED > 0.25
                 AND  anchor_mNED >= 0.25  AND  peak_eps < 0.25

      SProb=3  Strong memorization signal — ANY of:
                 (UAR < 0.20  AND  mNED < 0.08)   stochastic outputs collapse
                 OR  peak_eps >= 0.75              75%+ outputs cluster at greedy
                 OR  anchor_mNED < 0.08            outputs near-identical to greedy

      SProb=2  Moderate signal — ANY of:
                 (UAR < 0.40  AND  mNED < 0.15)   both stochastic signals weak
                 OR  peak_eps >= 0.50              half outputs cluster at greedy
                 OR  anchor_mNED < 0.15            outputs moderately close to greedy

      SProb=1  Residual: some signal present but below level-2 thresholds.

    NaN in any input -> SProb=1 (data incomplete, conservatively non-zero).
    """
    if any(pd.isna(v) for v in (uar, mned, anchor_mned, peak_eps)):
        return 1

    # SProb=0: all four signals indicate clean behaviour
    if uar > 0.60 and mned > 0.25 and anchor_mned >= 0.25 and peak_eps < 0.25:
        return 0

    # SProb=3: strong memorization signal on either axis
    if (uar < 0.20 and mned < 0.08) or peak_eps >= 0.75 or anchor_mned < 0.08:
        return 3

    # SProb=2: moderate signal on either axis
    if (uar < 0.40 and mned < 0.15) or peak_eps >= 0.50 or anchor_mned < 0.15:
        return 2

    # SProb=1: something is off but below level-2 thresholds
    return 1


# ---------------------------------------------------------------------------
# Shared collector
# ---------------------------------------------------------------------------

def collect_stability_metrics(
    *,
    client,
    prompt: str,
    decoding: Dict[str, Any],
    N: int,
    sleep_s: float,
    max_pairs: int = 435,
    anchor_eps: float = 0.15,
    token_encoder=None,
    existing_outputs: Optional[List[str]] = None,
    existing_greedy: Optional[str] = None,
) -> Dict[str, Any]:
    """Collect N stochastic outputs + 1 greedy output, then compute metrics.

    FIX 2: every API call goes through generate_with_retry (3 attempts,
    exponential backoff 1s/4s/16s) — transient errors no longer lose the item.

    FIX 3: partial recovery — if existing_outputs is provided (from a previous
    interrupted run), only the missing (N - len(existing_outputs)) samples are
    fetched. existing_greedy is reused if present, otherwise re-fetched.
    """
    # --- FIX 3: restore already-collected outputs ---
    outputs: List[str] = list(existing_outputs) if existing_outputs else []
    n_missing = N - len(outputs)

    for i in range(n_missing):
        # FIX 2: use retry wrapper instead of bare client call
        out = generate_with_retry(
            client,
            prompt=prompt,
            temperature=float(decoding["temperature"]),
            top_p=float(decoding["top_p"]),
            max_tokens=int(decoding["max_tokens"]),
        )
        outputs.append(out)
        if i < n_missing - 1:
            time.sleep(float(sleep_s))

    # --- FIX 3: reuse greedy if already present, else fetch ---
    if existing_greedy and existing_greedy.strip():
        greedy_output = existing_greedy.strip()
    else:
        greedy_output = generate_with_retry(
            client,
            prompt=prompt,
            temperature=0.0,
            top_p=1.0,
            max_tokens=int(decoding["max_tokens"]),
        )

    uar = compute_uar(outputs)
    mned = compute_mned_pairwise(outputs, max_pairs=max_pairs, token_encoder=token_encoder)
    anchor_mned, peak_eps = compute_anchor_metrics(
        outputs,
        greedy_output,
        anchor_eps=anchor_eps,
        token_encoder=token_encoder,
    )

    return {
        "outputs": outputs,
        "greedy_output": greedy_output,
        "UAR": uar,
        "mNED": mned,
        "anchor_mNED": anchor_mned,
        "peak_eps": peak_eps,
    }


# ---------------------------------------------------------------------------
# Aggregate stats helper
# ---------------------------------------------------------------------------

def _aggregate_stats(
    df: pd.DataFrame,
    col_uar: str,
    col_mned: str,
    col_sprob: str,
    processed_new: int,
    failures: int,
) -> Dict[str, Any]:
    uar_vals = pd.to_numeric(df[col_uar], errors="coerce")
    mned_vals = pd.to_numeric(df[col_mned], errors="coerce")
    sprob_vals = pd.to_numeric(df[col_sprob], errors="coerce")
    valid = uar_vals.notna() & mned_vals.notna()
    n_valid = int(valid.sum())

    sprob_dist = {}
    for level in range(4):
        count = int((sprob_vals == level).sum())
        sprob_dist[f"SProb_{level}_count"] = count
        sprob_dist[f"SProb_{level}_pct"] = round(100.0 * count / n_valid, 2) if n_valid else 0.0

    dominant = int(sprob_vals.dropna().mode()[0]) if sprob_vals.notna().any() else None

    return {
        "processed_new": processed_new,
        "failures": failures,
        "valid_items": n_valid,
        "UAR_mean": float(uar_vals[valid].mean()) if valid.any() else None,
        "UAR_median": float(uar_vals[valid].median()) if valid.any() else None,
        "UAR_p10": float(uar_vals[valid].quantile(0.10)) if valid.any() else None,
        "UAR_p90": float(uar_vals[valid].quantile(0.90)) if valid.any() else None,
        "mNED_mean": float(mned_vals[valid].mean()) if valid.any() else None,
        "mNED_median": float(mned_vals[valid].median()) if valid.any() else None,
        "mNED_p10": float(mned_vals[valid].quantile(0.10)) if valid.any() else None,
        "mNED_p90": float(mned_vals[valid].quantile(0.90)) if valid.any() else None,
        "dominant_SProb": dominant,
        **sprob_dist,
    }


# ---------------------------------------------------------------------------
# Reference pass
# ---------------------------------------------------------------------------

def run_reference_pass(
    df: pd.DataFrame,
    client,
    model_id: str,
    decoding: Dict[str, Any],
    N: int,
    limit: Optional[int],
    sleep_s: float,
    save_every: int,
    out_parquet: str,
    log_path: str,
    max_pairs: int = 435,
    anchor_eps: float = 0.15,
    token_encoder=None,
) -> Dict[str, Any]:
    col_out = f"stability_outputs_json_{model_id}"
    col_greedy = f"greedy_output_{model_id}"
    col_uar = f"UAR_{model_id}"
    col_mned = f"mNED_{model_id}"
    col_anchor = f"anchor_mNED_{model_id}"
    col_peak = f"peak_eps_{model_id}"
    col_sprob = f"SProb_{model_id}"

    for col, dtype in [
        (col_out, "object"),
        (col_greedy, "object"),
        (col_uar, "Float64"),
        (col_mned, "Float64"),
        (col_anchor, "Float64"),
        (col_peak, "Float64"),
        (col_sprob, "Int64"),
    ]:
        if col not in df.columns:
            if dtype == "object":
                df[col] = ""
            else:
                df[col] = pd.array([pd.NA] * len(df), dtype=dtype)
        elif dtype != "object":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)

    processed_new = 0
    failures = 0

    for idx, row in df.iterrows():
        existing_raw = row.get(col_out, "")
        existing_greedy = row.get(col_greedy, "")

        # FIX 3: parse any partial outputs already saved
        existing_outputs: Optional[List[str]] = None
        if isinstance(existing_raw, str) and existing_raw.strip().startswith("["):
            try:
                parsed = json.loads(existing_raw)
                if isinstance(parsed, list) and len(parsed) > 0:
                    existing_outputs = [str(x) for x in parsed]
            except (json.JSONDecodeError, ValueError):
                existing_outputs = None

        # Skip only when N outputs are complete AND greedy is present
        if (
            existing_outputs is not None
            and len(existing_outputs) >= N
            and isinstance(existing_greedy, str)
            and existing_greedy.strip()
        ):
            continue

        item_key = str(row.get("xsum_id", idx))
        doc = get_document_field(row)
        if not doc:
            failures += 1
            log_jsonl(log_path, {"xsum_id": item_key, "pass": "reference", "status": "error_missing_document"})
            continue

        prompt = build_stability_prompt(normalize_text(doc))

        try:
            metrics = collect_stability_metrics(
                client=client,
                prompt=prompt,
                decoding=decoding,
                N=N,
                sleep_s=sleep_s,
                max_pairs=max_pairs,
                anchor_eps=anchor_eps,
                token_encoder=token_encoder,
                existing_outputs=existing_outputs,
                existing_greedy=existing_greedy if isinstance(existing_greedy, str) else None,
            )

            sprob = map_to_SProb(
                uar=metrics["UAR"],
                mned=metrics["mNED"],
                anchor_mned=metrics["anchor_mNED"],
                peak_eps=metrics["peak_eps"],
            )

            df.at[idx, col_out] = json.dumps(metrics["outputs"], ensure_ascii=False)
            df.at[idx, col_greedy] = metrics["greedy_output"]
            df.at[idx, col_uar] = float(metrics["UAR"])
            df.at[idx, col_mned] = float(metrics["mNED"])
            df.at[idx, col_anchor] = float(metrics["anchor_mNED"])
            df.at[idx, col_peak] = float(metrics["peak_eps"])
            df.at[idx, col_sprob] = int(sprob)

            n_fetched = N - (len(existing_outputs) if existing_outputs else 0)
            log_jsonl(log_path, {
                "xsum_id": item_key,
                "pass": "reference",
                "status": "ok",
                "UAR": round(float(metrics["UAR"]), 6),
                "mNED": round(float(metrics["mNED"]), 6),
                "anchor_mNED": round(float(metrics["anchor_mNED"]), 6),
                "peak_eps": round(float(metrics["peak_eps"]), 6),
                "SProb": int(sprob),
                "N_collected": len(metrics["outputs"]),
                "N_fetched_this_run": n_fetched,
            })
            processed_new += 1

        except Exception as e:
            import traceback
            failures += 1
            log_jsonl(log_path, {
                "xsum_id": item_key,
                "pass": "reference",
                "status": "api_error",
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
            })

        if save_every and processed_new > 0 and processed_new % save_every == 0:
            ensure_parent_dir(out_parquet)
            df.to_parquet(out_parquet, index=False)
            print(f"  Saved: {processed_new} rows -> {out_parquet}")

        if limit is not None and processed_new >= limit:
            break

    ensure_parent_dir(out_parquet)
    df.to_parquet(out_parquet, index=False)
    return _aggregate_stats(df, col_uar, col_mned, col_sprob, processed_new, failures)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    master_path = cfg["project"]["frozen_master_table_path"]
    model_cfg = next((m for m in cfg["models"] if m["model_id"] == args.model_id), None)
    if model_cfg is None:
        raise ValueError(f"model_id='{args.model_id}' not found in config")

    stab_cfg = cfg["stability"]
    decoding = stab_cfg["decoding"]
    N = int(stab_cfg["N_samples"])
    sleep_s = float(stab_cfg["runtime"]["sleep_s"])
    save_every = int(stab_cfg["runtime"]["save_every"])
    max_pairs = int(stab_cfg.get("max_pairs", 435))
    anchor_eps = float(stab_cfg.get("anchor_eps", 0.15))

    out_parquet = format_path(stab_cfg["outputs"]["parquet"], args.model_id)
    log_path = format_path(stab_cfg["outputs"]["log_jsonl"], args.model_id)
    summary_json = format_path(stab_cfg["outputs"]["summary_json"], args.model_id)

    client = select_client(model_cfg)
    token_encoder = None
    t0 = time.time()

    df = pd.read_parquet(out_parquet) if Path(out_parquet).exists() else pd.read_parquet(master_path)
    print(f"[{args.model_id}] Running stability detector (N={N} stochastic + 1 greedy per item)...")
    print(f"  Method: Dong et al. (2024) CDD — self-contained, no external baseline.")

    ref_stats = run_reference_pass(
        df=df,
        client=client,
        model_id=args.model_id,
        decoding=decoding,
        N=N,
        limit=args.limit,
        sleep_s=sleep_s,
        save_every=save_every,
        out_parquet=out_parquet,
        log_path=log_path,
        max_pairs=max_pairs,
        anchor_eps=anchor_eps,
        token_encoder=token_encoder,
    )
    elapsed_s = time.time() - t0

    df_final = pd.read_parquet(out_parquet)
    col_sprob = f"SProb_{args.model_id}"
    sprob3_total = int((pd.to_numeric(df_final.get(col_sprob, pd.Series()), errors="coerce") == 3).sum())

    summary = {
        "stage": "stability_v4",
        "method": "Dong et al. (2024) CDD — no external baseline",
        "model_id": args.model_id,
        "provider": model_cfg["provider"],
        "model_name": model_cfg["model_name"],
        "dataset_path": master_path,
        "n_rows_total": int(len(df_final)),
        **ref_stats,
        "SProb3_total": sprob3_total,
        "decoding": decoding,
        "N_samples": N,
        "max_pairs": max_pairs,
        "anchor_eps": anchor_eps,
        "elapsed_seconds": elapsed_s,
        "out_parquet": out_parquet,
        "log_jsonl": log_path,
    }

    ensure_parent_dir(summary_json)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Model: {args.model_id} ({model_cfg['model_name']})")
    print(f"UAR_mean={ref_stats['UAR_mean']:.4f}  mNED_mean={ref_stats['mNED_mean']:.4f}")
    print("SProb dist: " + "  ".join(f"L{i}={ref_stats[f'SProb_{i}_count']}" for i in range(4)))
    print(f"SProb=3 total={sprob3_total}")
    print(f"Output:  {out_parquet}")
    print(f"Summary: {summary_json}")
    print(f"Log:     {log_path}")


if __name__ == "__main__":
    main()
