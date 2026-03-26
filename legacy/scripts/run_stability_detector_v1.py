#!/usr/bin/env python3
"""
scripts/run_stability_detector_v1.py

Stability-based Probability Proxy (SProb) runner for API-only contamination pipeline.

v1 changes:
  - adds greedy anchor generation (temperature=0)
  - switches mNED to token-level normalized edit distance
  - SProb=0 only when ALL signals are clean
  - any weak evidence already maps to SProb>=1

Core per-item metrics:
  UAR          = unique stochastic outputs / N
  mNED         = mean pairwise token-level NED across stochastic samples
  anchor_mNED  = mean token-level NED from stochastic outputs to greedy anchor
  peak_eps     = share of stochastic outputs within eps distance of greedy anchor

Control baseline:
  UAR_control, mNED_control, anchor_mNED_control, peak_eps_control

SProb mapping:
  SProb = max(abs_band, contrast_band, anchor_band)
  where SProb=0 only if every band is 0.
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

def _band_absolute(uar: float, mned: float) -> int:
    if pd.isna(uar) or pd.isna(mned):
        return 1
    if uar > 0.60 and mned > 0.25:
        return 0
    if (0.40 <= uar <= 0.60) or (0.15 <= mned <= 0.25):
        return 1
    if (0.20 <= uar < 0.40) or (0.08 <= mned < 0.15):
        return 2
    return 3


def _band_contrast(
    uar: float,
    mned: float,
    uar_control: Optional[float],
    mned_control: Optional[float],
) -> Tuple[int, bool]:
    if (
        uar_control is None
        or mned_control is None
        or pd.isna(uar)
        or pd.isna(mned)
        or uar <= 0
        or mned <= 0
    ):
        return 0, False

    c_uar = float(uar_control / uar) if uar > 0 else float("inf")
    c_mned = float(mned_control / mned) if mned > 0 else float("inf")
    c = max(c_uar, c_mned)

    if c >= 2.0:
        return 3, True
    if c >= 1.5:
        return 2, True
    if c >= 1.25:
        return 1, True
    return 0, False


def _band_anchor(anchor_mned: float, peak_eps: float) -> int:
    if pd.isna(anchor_mned) or pd.isna(peak_eps):
        return 1

    if anchor_mned >= 0.25 and peak_eps < 0.25:
        return 0

    band = 1
    if anchor_mned < 0.15 or peak_eps >= 0.50:
        band = max(band, 2)
    if anchor_mned < 0.08 or peak_eps >= 0.75:
        band = max(band, 3)
    return band


def map_to_SProb(
    *,
    uar: float,
    mned: float,
    anchor_mned: float,
    peak_eps: float,
    uar_control: Optional[float] = None,
    mned_control: Optional[float] = None,
) -> Tuple[int, bool]:
    abs_band = _band_absolute(uar, mned)
    contrast_band, contrast_met = _band_contrast(uar, mned, uar_control, mned_control)
    anchor_band = _band_anchor(anchor_mned, peak_eps)
    sprob = max(abs_band, contrast_band, anchor_band)
    return int(sprob), bool(contrast_met)


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
) -> Dict[str, Any]:
    outputs: List[str] = []

    for _ in range(N):
        out = client.generate_text(
            prompt=prompt,
            temperature=float(decoding["temperature"]),
            top_p=float(decoding["top_p"]),
            max_tokens=int(decoding["max_tokens"]),
        )
        outputs.append((out or "").strip())
        time.sleep(float(sleep_s))

    greedy_output = client.generate_text(
        prompt=prompt,
        temperature=0.0,
        top_p=1.0,
        max_tokens=int(decoding["max_tokens"]),
    )
    greedy_output = (greedy_output or "").strip()

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
# Control pass
# ---------------------------------------------------------------------------

def run_control_pass(
    df_control: pd.DataFrame,
    client,
    model_id: str,
    decoding: Dict[str, Any],
    N: int,
    limit: Optional[int],
    sleep_s: float,
    save_every: int,
    control_parquet: str,
    log_path: str,
    max_pairs: int = 435,
    anchor_eps: float = 0.15,
    token_encoder=None,
) -> Dict[str, Any]:
    col_ctrl_out = f"ctrl_outputs_json_{model_id}"
    col_ctrl_greedy = f"greedy_ctrl_{model_id}"
    col_ctrl_uar = f"UAR_ctrl_{model_id}"
    col_ctrl_mned = f"mNED_ctrl_{model_id}"
    col_ctrl_anchor = f"anchor_mNED_ctrl_{model_id}"
    col_ctrl_peak = f"peak_eps_ctrl_{model_id}"

    for col, dtype in [
        (col_ctrl_out, "object"),
        (col_ctrl_greedy, "object"),
        (col_ctrl_uar, "Float64"),
        (col_ctrl_mned, "Float64"),
        (col_ctrl_anchor, "Float64"),
        (col_ctrl_peak, "Float64"),
    ]:
        if col not in df_control.columns:
            if dtype == "object":
                df_control[col] = ""
            else:
                df_control[col] = pd.array([pd.NA] * len(df_control), dtype=dtype)
        elif dtype != "object":
            df_control[col] = pd.to_numeric(df_control[col], errors="coerce").astype(dtype)

    processed_new = 0
    failures = 0

    for idx, row in df_control.iterrows():
        existing = row.get(col_ctrl_out, "")
        if isinstance(existing, str) and existing.strip().startswith("[") and len(existing.strip()) > 10:
            continue

        item_key = str(row.get("control_id", idx))
        doc = row.get("document_norm", "")
        if not isinstance(doc, str) or not doc.strip():
            failures += 1
            log_jsonl(log_path, {"control_id": item_key, "pass": "control", "status": "error_missing_document"})
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
            )

            df_control.at[idx, col_ctrl_out] = json.dumps(metrics["outputs"], ensure_ascii=False)
            df_control.at[idx, col_ctrl_greedy] = metrics["greedy_output"]
            df_control.at[idx, col_ctrl_uar] = float(metrics["UAR"])
            df_control.at[idx, col_ctrl_mned] = float(metrics["mNED"])
            df_control.at[idx, col_ctrl_anchor] = float(metrics["anchor_mNED"])
            df_control.at[idx, col_ctrl_peak] = float(metrics["peak_eps"])

            log_jsonl(log_path, {
                "control_id": item_key,
                "pass": "control",
                "status": "ok",
                "UAR": round(float(metrics["UAR"]), 6),
                "mNED": round(float(metrics["mNED"]), 6),
                "anchor_mNED": round(float(metrics["anchor_mNED"]), 6),
                "peak_eps": round(float(metrics["peak_eps"]), 6),
            })
            processed_new += 1

        except Exception as e:
            import traceback
            failures += 1
            log_jsonl(log_path, {
                "control_id": item_key,
                "pass": "control",
                "status": "api_error",
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
            })

        if save_every and processed_new > 0 and processed_new % save_every == 0:
            ensure_parent_dir(control_parquet)
            df_control.to_parquet(control_parquet, index=False)
            print(f"  Control saved: {processed_new} rows -> {control_parquet}")

        if limit is not None and processed_new >= limit:
            break

    ensure_parent_dir(control_parquet)
    df_control.to_parquet(control_parquet, index=False)

    uar_v = pd.to_numeric(df_control[col_ctrl_uar], errors="coerce")
    mned_v = pd.to_numeric(df_control[col_ctrl_mned], errors="coerce")
    anchor_v = pd.to_numeric(df_control[col_ctrl_anchor], errors="coerce")
    peak_v = pd.to_numeric(df_control[col_ctrl_peak], errors="coerce")
    valid = uar_v.notna() & mned_v.notna()

    return {
        "processed_new": processed_new,
        "failures": failures,
        "valid_items": int(valid.sum()),
        "UAR_control": float(uar_v[valid].mean()) if valid.any() else None,
        "mNED_control": float(mned_v[valid].mean()) if valid.any() else None,
        "anchor_mNED_control": float(anchor_v[valid].mean()) if valid.any() else None,
        "peak_eps_control": float(peak_v[valid].mean()) if valid.any() else None,
        "UAR_ctrl_median": float(uar_v[valid].median()) if valid.any() else None,
        "mNED_ctrl_median": float(mned_v[valid].median()) if valid.any() else None,
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
    uar_control: Optional[float],
    mned_control: Optional[float],
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
    col_delta_uar = f"delta_UAR_{model_id}"
    col_mned_ratio = f"mNED_ratio_{model_id}"
    col_contrast_met = f"contrast_met_{model_id}"

    for col, dtype in [
        (col_out, "object"),
        (col_greedy, "object"),
        (col_uar, "Float64"),
        (col_mned, "Float64"),
        (col_anchor, "Float64"),
        (col_peak, "Float64"),
        (col_sprob, "Int64"),
        (col_delta_uar, "Float64"),
        (col_mned_ratio, "Float64"),
        (col_contrast_met, "object"),
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
        existing = row.get(col_out, "")
        if isinstance(existing, str) and existing.strip().startswith("[") and len(existing.strip()) > 10:
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
            )

            sprob, contrast_met = map_to_SProb(
                uar=metrics["UAR"],
                mned=metrics["mNED"],
                anchor_mned=metrics["anchor_mNED"],
                peak_eps=metrics["peak_eps"],
                uar_control=uar_control,
                mned_control=mned_control,
            )

            delta_uar = (
                metrics["UAR"] - uar_control
                if uar_control is not None and not pd.isna(metrics["UAR"])
                else None
            )
            mned_ratio = (
                metrics["mNED"] / mned_control
                if mned_control is not None and mned_control > 0 and not pd.isna(metrics["mNED"])
                else None
            )

            df.at[idx, col_out] = json.dumps(metrics["outputs"], ensure_ascii=False)
            df.at[idx, col_greedy] = metrics["greedy_output"]
            df.at[idx, col_uar] = float(metrics["UAR"])
            df.at[idx, col_mned] = float(metrics["mNED"])
            df.at[idx, col_anchor] = float(metrics["anchor_mNED"])
            df.at[idx, col_peak] = float(metrics["peak_eps"])
            df.at[idx, col_sprob] = int(sprob)
            df.at[idx, col_contrast_met] = str(contrast_met)
            if delta_uar is not None:
                df.at[idx, col_delta_uar] = float(delta_uar)
            if mned_ratio is not None:
                df.at[idx, col_mned_ratio] = float(mned_ratio)

            log_jsonl(log_path, {
                "xsum_id": item_key,
                "pass": "reference",
                "status": "ok",
                "UAR": round(float(metrics["UAR"]), 6),
                "mNED": round(float(metrics["mNED"]), 6),
                "anchor_mNED": round(float(metrics["anchor_mNED"]), 6),
                "peak_eps": round(float(metrics["peak_eps"]), 6),
                "SProb": int(sprob),
                "contrast_met": bool(contrast_met),
                "delta_UAR": round(delta_uar, 6) if delta_uar is not None else None,
                "mNED_ratio": round(mned_ratio, 6) if mned_ratio is not None else None,
                "N_collected": len(metrics["outputs"]),
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
    parser.add_argument("--control_only", action="store_true", help="Run only control pass (reference already done)")
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
    use_control = bool(stab_cfg.get("use_control_baseline", False))
    control_set_path = stab_cfg.get("control_set_path", "")
    max_pairs = int(stab_cfg.get("max_pairs", 435))
    anchor_eps = float(stab_cfg.get("anchor_eps", 0.15))

    out_parquet = format_path(stab_cfg["outputs"]["parquet"], args.model_id)
    log_path = format_path(stab_cfg["outputs"]["log_jsonl"], args.model_id)
    summary_json = format_path(stab_cfg["outputs"]["summary_json"], args.model_id)
    control_parquet = format_path(
        stab_cfg["outputs"].get("control_parquet", "runs/v6_stability_ctrl_{model_id}.parquet"),
        args.model_id,
    )

    client = select_client(model_cfg)
    token_encoder = None
    t0 = time.time()

    ctrl_stats: Dict[str, Any] = {}
    uar_control = None
    mned_control = None
    anchor_mned_control = None
    peak_eps_control = None

    if use_control or args.control_only:
        if not control_set_path or not Path(control_set_path).exists():
            raise FileNotFoundError(
                f"Control set not found: '{control_set_path}'. Run control set builder first."
            )
        df_control = pd.read_parquet(control_parquet) if Path(control_parquet).exists() else pd.read_parquet(control_set_path)
        print(f"[{args.model_id}] Running CONTROL pass (N={N} stochastic + 1 greedy per item)...")
        ctrl_stats = run_control_pass(
            df_control=df_control,
            client=client,
            model_id=args.model_id,
            decoding=decoding,
            N=N,
            limit=args.limit,
            sleep_s=sleep_s,
            save_every=save_every,
            control_parquet=control_parquet,
            log_path=log_path,
            max_pairs=max_pairs,
            anchor_eps=anchor_eps,
            token_encoder=token_encoder,
        )
        uar_control = ctrl_stats["UAR_control"]
        mned_control = ctrl_stats["mNED_control"]
        anchor_mned_control = ctrl_stats.get("anchor_mNED_control")
        peak_eps_control = ctrl_stats.get("peak_eps_control")
        if uar_control is not None and mned_control is not None:
            print(f"  UAR_control={uar_control:.4f}  mNED_control={mned_control:.4f}")
        if anchor_mned_control is not None and peak_eps_control is not None:
            print(f"  anchor_mNED_control={anchor_mned_control:.4f}  peak_eps_control={peak_eps_control:.4f}")

        if args.control_only:
            control_only_summary_json = summary_json.replace(
                "v6_stability_summary_", "v6_stability_ctrl_summary_"
            ).replace("v7_stability_summary_", "v6_stability_ctrl_summary_")
            ensure_parent_dir(control_only_summary_json)
            with open(control_only_summary_json, "w", encoding="utf-8") as f:
                json.dump({
                    "stage": "stability_v7_control_only",
                    "model_id": args.model_id,
                    "provider": model_cfg["provider"],
                    "model_name": model_cfg["model_name"],
                    "control_set_path": control_set_path,
                    **ctrl_stats,
                    "control_parquet": control_parquet,
                    "log_jsonl": log_path,
                }, f, ensure_ascii=False, indent=2)
            print(f"Control-only run complete. Summary: {control_only_summary_json}")
            return

    df = pd.read_parquet(out_parquet) if Path(out_parquet).exists() else pd.read_parquet(master_path)
    print(f"[{args.model_id}] Running REFERENCE pass (N={N} stochastic + 1 greedy per item)...")
    if uar_control is not None:
        print(f"  Control baseline: UAR_control={uar_control:.4f}, mNED_control={mned_control:.4f}")
    else:
        print("  No control baseline — only absolute and anchor bands will be used.")

    ref_stats = run_reference_pass(
        df=df,
        client=client,
        model_id=args.model_id,
        decoding=decoding,
        N=N,
        uar_control=uar_control,
        mned_control=mned_control,
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
    col_contrast_met = f"contrast_met_{args.model_id}"
    col_sprob = f"SProb_{args.model_id}"
    contrast_confirmed = int((df_final.get(col_contrast_met, pd.Series(dtype=str)) == "True").sum())
    sprob3_total = int((pd.to_numeric(df_final.get(col_sprob, pd.Series()), errors="coerce") == 3).sum())

    summary = {
        "stage": "stability_v7",
        "model_id": args.model_id,
        "provider": model_cfg["provider"],
        "model_name": model_cfg["model_name"],
        "dataset_path": master_path,
        "n_rows_total": int(len(df_final)),
        **ref_stats,
        "use_control_baseline": use_control,
        "UAR_control": uar_control,
        "mNED_control": mned_control,
        "anchor_mNED_control": anchor_mned_control,
        "peak_eps_control": peak_eps_control,
        "UAR_ctrl_median": ctrl_stats.get("UAR_ctrl_median"),
        "mNED_ctrl_median": ctrl_stats.get("mNED_ctrl_median"),
        "SProb3_total": sprob3_total,
        "SProb3_contrast_confirmed": contrast_confirmed,
        "decoding": decoding,
        "N_samples": N,
        "max_pairs": max_pairs,
        "anchor_eps": anchor_eps,
        "elapsed_seconds": elapsed_s,
        "out_parquet": out_parquet,
        "control_parquet": control_parquet if (use_control or args.control_only) else None,
        "log_jsonl": log_path,
    }

    ensure_parent_dir(summary_json)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Model: {args.model_id} ({model_cfg['model_name']})")
    print(f"UAR_mean={ref_stats['UAR_mean']:.4f}  mNED_mean={ref_stats['mNED_mean']:.4f}")
    print("SProb dist: " + "  ".join(f"L{i}={ref_stats[f'SProb_{i}_count']}" for i in range(4)))
    print(f"SProb=3 total={sprob3_total}  contrast_confirmed={contrast_confirmed}")
    print(f"Output:  {out_parquet}")
    print(f"Summary: {summary_json}")
    print(f"Log:     {log_path}")


if __name__ == "__main__":
    main()
