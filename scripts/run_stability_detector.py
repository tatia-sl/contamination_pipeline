#!/usr/bin/env python3
"""
run_stability_detector_v7.py

Standalone stability / probability detector with:
- repeated stochastic one-sentence summarization
- greedy anchor generation
- token-level normalized edit distance
- SProb semantics: SProb=0 only when *all* available signals are clean
- optional control baseline
- resumable parquet outputs + JSONL logs + summary JSON

This is a self-contained v7 implementation intended to replace manual patching.
It does not assume access to the original v6 script.

Example:
    python run_stability_detector_v7.py \
        --input-parquet data/reference.parquet \
        --text-column document \
        --id-column xsum_id \
        --model-id gpt-4.1-mini \
        --config config.yaml

Minimal config structure:

stability:
  use_control_baseline: false
  description: "CDD-inspired stability detector via sampled one-sentence summarization with greedy anchor and token-level edit distance."
  N_samples: 20
  decoding:
    temperature: 0.8
    top_p: 1.0
    max_tokens: 80
  greedy_anchor:
    temperature: 0.0
  metrics:
    distance: token_level
    max_pairs: 435
    anchor_eps: 0.15
    tokenization: regex   # regex | tiktoken | hf
    tokenizer_name: null  # e.g. cl100k_base or tokenizer repo id
  runtime:
    sleep_s: 0.10
    save_every: 10
    limit: null
  outputs:
    parquet: "runs/v7_stability_{model_id}.parquet"
    log_jsonl: "logs/v7_stability_{model_id}.jsonl"
    summary_json: "outputs/v7_stability_summary_{model_id}.json"
    control_parquet: "runs/v7_stability_ctrl_{model_id}.parquet"
  provider:
    kind: openai_chat
    api_key_env: OPENAI_API_KEY
    base_url: null

Notes:
- Output files are per-model, so columns are not suffixed by model_id.
- If provider config is omitted, a generic OpenAI chat client is attempted.
- If tokenizer config is omitted, regex tokenization is used as a provider-agnostic fallback.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import yaml
except Exception as e:  # pragma: no cover
    raise RuntimeError("PyYAML is required to run this script.") from e


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clients.gemini_client import GeminiClient
from src.clients.openai_client import OpenAIClient


STABILITY_PROMPT_TEMPLATE = """Write a ONE-SENTENCE summary of the following news article.

Keep it factual and concise. Output exactly one sentence.


Article:

{DOCUMENT}

"""

# Alternative domain-agnostic template. Selected via config:
#   stability.prompt.template: "generic"  (or "news_article", or a custom string with {DOCUMENT})
STABILITY_PROMPT_TEMPLATES: Dict[str, str] = {
    "news_article": STABILITY_PROMPT_TEMPLATE,
    "generic": (
        "Write a ONE-SENTENCE summary of the following text.\n\n"
        "Keep it factual and concise. Output exactly one sentence.\n\n\n"
        "Text:\n\n{DOCUMENT}\n\n"
    ),
}


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------

def normalize_text(s: Any) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = " ".join(s.split())
    return s.strip()


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def json_loads_or_default(s: Any, default: Any) -> Any:
    if not isinstance(s, str) or not s.strip():
        return default
    try:
        return json.loads(s)
    except Exception:
        return default


def log_jsonl(path: str | Path, record: Dict[str, Any]) -> None:
    ensure_parent(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def format_path(template: str, model_id: str) -> str:
    return str(template).replace("{model_id}", model_id)


def build_stability_prompt(document_text: str, prompt_template: str = STABILITY_PROMPT_TEMPLATE) -> str:
    return prompt_template.format(DOCUMENT=document_text)


def pick_first_present(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def infer_text_column(df: pd.DataFrame) -> str:
    col = pick_first_present(df, [
        "document", "article", "text", "doc", "content", "input", "source", "article_text"
    ])
    if col is None:
        raise ValueError(
            "Could not infer text column. Pass --text-column explicitly. "
            f"Available columns: {list(df.columns)}"
        )
    return col


def infer_id_column(df: pd.DataFrame) -> str:
    col = pick_first_present(df, [
        "xsum_id", "id", "doc_id", "article_id", "uuid", "item_id"
    ])
    if col is None:
        raise ValueError(
            "Could not infer id column. Pass --id-column explicitly. "
            f"Available columns: {list(df.columns)}"
        )
    return col


# ---------------------------------------------------------------------------
# Tokenization / token-level distance
# ---------------------------------------------------------------------------

def regex_tokenize(text: str) -> List[str]:
    text = normalize_text(text).lower()
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def load_token_encoder(token_cfg: Dict[str, Any], model_id: Optional[str] = None) -> Optional[Callable[[str], List[Any]]]:
    mode = str(token_cfg.get("mode") or token_cfg.get("tokenization") or "regex").lower()
    name = token_cfg.get("name") or token_cfg.get("tokenizer_name")

    # Auto-select tiktoken encoding when mode=auto and model_id is known.
    # Falls back to regex if tiktoken is not installed or model is unrecognised.
    if mode in ("auto", "regex") and model_id:
        try:
            import tiktoken  # type: ignore
            enc = tiktoken.encoding_for_model(model_id)
            return lambda text: list(enc.encode_ordinary(normalize_text(text).lower()))
        except Exception:
            pass  # tiktoken not installed, or unrecognised model_id — fall through to regex

    if mode == "regex":
        return None

    if mode == "tiktoken":
        try:
            import tiktoken  # type: ignore
        except Exception as e:
            raise RuntimeError("tokenization.mode=tiktoken requires the tiktoken package.") from e
        enc = tiktoken.get_encoding(name or "cl100k_base")
        return lambda text: list(enc.encode_ordinary(normalize_text(text).lower()))

    if mode in {"hf", "huggingface", "transformers"}:
        try:
            from transformers import AutoTokenizer  # type: ignore
        except Exception as e:
            raise RuntimeError("tokenization.mode=hf requires transformers.") from e
        if not name:
            raise ValueError("tokenization.name is required when tokenization.mode=hf")
        tok = AutoTokenizer.from_pretrained(name, use_fast=True)
        return lambda text: list(tok.encode(normalize_text(text).lower(), add_special_tokens=False))

    raise ValueError(f"Unsupported tokenization.mode={mode!r}. Use regex | tiktoken | hf")


def encode_tokens(text: str, token_encoder: Optional[Callable[[str], List[Any]]] = None) -> List[Any]:
    if token_encoder is None:
        return regex_tokenize(text)
    out = token_encoder(text)
    return list(out) if out is not None else []


def normalized_token_edit_distance(a: str, b: str, token_encoder: Optional[Callable[[str], List[Any]]] = None) -> float:
    ta = encode_tokens(a, token_encoder)
    tb = encode_tokens(b, token_encoder)

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
# Metrics
# ---------------------------------------------------------------------------

def compute_uar(outputs: List[str]) -> float:
    normed = [normalize_text(x).lower() for x in outputs if isinstance(x, str)]
    if not normed:
        return float("nan")
    return len(set(normed)) / len(normed)


def compute_mned_pairwise(
    outputs: List[str],
    *,
    max_pairs: Optional[int] = None,
    token_encoder: Optional[Callable[[str], List[Any]]] = None,
    pair_sampling_seed: int = 42,
) -> float:
    normed = [normalize_text(x).lower() for x in outputs if isinstance(x, str)]
    n = len(normed)
    if n < 2:
        return float("nan")

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if max_pairs is not None and len(pairs) > max_pairs:
        rnd = random.Random(pair_sampling_seed)
        pairs = rnd.sample(pairs, k=max_pairs)

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
    token_encoder: Optional[Callable[[str], List[Any]]] = None,
) -> Tuple[float, float]:
    normed = [normalize_text(x).lower() for x in outputs if isinstance(x, str)]
    greedy = normalize_text(greedy_output).lower()
    if not normed or not greedy:
        return float("nan"), float("nan")

    dists = [normalized_token_edit_distance(x, greedy, token_encoder) for x in normed]
    anchor_mned = float(sum(dists) / len(dists))
    peak_eps = float(sum(d <= anchor_eps for d in dists) / len(dists))
    return anchor_mned, peak_eps


# ---------------------------------------------------------------------------
# SProb mapping: dissertation-ready compact specification
# ---------------------------------------------------------------------------
#
# Let:
#   B_abs = band(UAR, mNED)
#   B_anchor = band(anchor_mNED, peak_eps)
#   B_contrast = band(max(UAR_ctrl/UAR, mNED_ctrl/mNED)) if control baseline exists,
#                else 0
#
# Then:
#   SProb = max(B_abs, B_anchor, B_contrast),  SProb in {0,1,2,3}
#
# Decision table:
#   Signal family     Level 0                 Level 1                  Level 2                  Level 3
#   Absolute          UAR > 0.60 AND          0.40 <= UAR <= 0.60 OR   0.20 <= UAR < 0.40 OR   residual stronger-collapse case
#                     mNED > 0.25             0.15 <= mNED <= 0.25     0.08 <= mNED < 0.15
#   Anchor            anchor_mNED >= 0.25     residual non-clean case   anchor_mNED < 0.15 OR   anchor_mNED < 0.08 OR
#                     AND peak_eps < 0.25                               peak_eps >= 0.50        peak_eps >= 0.75
#   Contrast          no baseline effect      c >= 1.25                c >= 1.50                c >= 2.00
#                                                                    where c = max(UAR_ctrl/UAR, mNED_ctrl/mNED)
#
# Interpretation:
#   SProb=0 only when every available signal is clean.
#   Higher SProb means lower output variability / stronger concentration around a
#   small set of summaries or around the greedy anchor.
# ---------------------------------------------------------------------------

def _band_absolute(uar: float, mned: float) -> int:
    if pd.isna(uar) or pd.isna(mned):
        return 1  # unknown is not clean

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
        return 1  # unknown is not clean

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


def map_to_SProb_aggregate(sprob_vals: pd.Series) -> int:
    """
    Aggregate-level SProb rubric derived from the distribution of item scores.

    Let:
      N    = number of valid items
      n_k  = number of items with level k, k in {0,1,2,3}
      n+   = n1 + n2 + n3
      p+   = n+ / N
      p23  = (n2 + n3) / N
      p3   = n3 / N

    Rules:
      0 if n+ = 0
      1 if n+ > 0, n2 = 0, n3 = 0, and p+ < 0.10
      2 if n2 >= 1 or p+ >= 0.10 or p23 >= 0.05, provided level-3
        conditions are not met
      3 if n3 >= 2 or p3 >= 0.05 or p23 >= 0.15
    """
    vals = pd.to_numeric(sprob_vals, errors="coerce").dropna().astype(int)
    if vals.empty:
        return 0

    n = len(vals)
    n1 = int((vals == 1).sum())
    n2 = int((vals == 2).sum())
    n3 = int((vals == 3).sum())
    n_pos = n1 + n2 + n3

    p_pos = n_pos / n if n > 0 else 0.0
    p23 = (n2 + n3) / n if n > 0 else 0.0
    p3 = n3 / n if n > 0 else 0.0

    if n_pos == 0:
        return 0
    if n3 >= 2 or p3 >= 0.05 or p23 >= 0.15:
        return 3
    if n2 >= 1 or p_pos >= 0.10 or p23 >= 0.05:
        return 2
    return 1


# ---------------------------------------------------------------------------
# Provider client
# ---------------------------------------------------------------------------

class BaseTextClient:
    def generate_text(self, *, prompt: str, temperature: float, top_p: float, max_tokens: int) -> str:
        raise NotImplementedError


class ClientAdapter(BaseTextClient):
    def __init__(self, client: Any):
        self.client = client

    def generate_text(self, *, prompt: str, temperature: float, top_p: float, max_tokens: int) -> str:
        return (self.client.generate_text(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        ) or "").strip()


def select_client(model_cfg: Dict[str, Any]) -> BaseTextClient:
    provider = str(model_cfg["provider"]).lower()
    model_name = model_cfg["model_name"]
    api_key_var = model_cfg.get("env", {}).get("api_key_var")
    api_cfg = model_cfg.get("api", {}) or {}

    if provider == "openai":
        api_key = os.environ.get(api_key_var or "OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(f"Missing {api_key_var or 'OPENAI_API_KEY'} env var")
        return ClientAdapter(OpenAIClient(
            model=model_name,
            api_key=api_key,
            api_mode=api_cfg.get("mode", "chat_completions"),
        ))

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
        return ClientAdapter(OpenAIClient(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            extra_headers=extra_headers or None,
            api_mode=api_cfg.get("mode", "chat_completions"),
        ))

    if provider == "gemini":
        return ClientAdapter(GeminiClient(model=model_name))

    raise ValueError(f"Unsupported provider: {provider}")


# ---------------------------------------------------------------------------
# Collection logic
# ---------------------------------------------------------------------------

def collect_stability_metrics(
    *,
    client: BaseTextClient,
    prompt: str,
    decoding: Dict[str, Any],
    N: int,
    sleep_s: float,
    max_pairs: int = 435,
    anchor_eps: float = 0.15,
    token_encoder: Optional[Callable[[str], List[Any]]] = None,
    greedy_anchor_cfg: Optional[Dict[str, Any]] = None,
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
        if sleep_s > 0:
            time.sleep(float(sleep_s))

    greedy_cfg = dict(greedy_anchor_cfg or {})
    greedy_output = client.generate_text(
        prompt=prompt,
        temperature=float(greedy_cfg.get("temperature", 0.0)),
        top_p=float(greedy_cfg.get("top_p", 1.0)),
        max_tokens=int(greedy_cfg.get("max_tokens", decoding["max_tokens"])),
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
# DataFrame helpers
# ---------------------------------------------------------------------------

def init_reference_columns(df: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("stability_outputs_json", "object", ""),
        ("greedy_output", "object", ""),
        ("UAR", "Float64", pd.NA),
        ("mNED", "Float64", pd.NA),
        ("anchor_mNED", "Float64", pd.NA),
        ("peak_eps", "Float64", pd.NA),
        ("SProb", "Int64", pd.NA),
        ("SProb_aggregate", "Int64", pd.NA),
        ("delta_UAR", "Float64", pd.NA),
        ("mNED_ratio", "Float64", pd.NA),
        ("contrast_met", "boolean", pd.NA),
    ]
    for col, dtype, default in specs:
        if col not in df.columns:
            if dtype == "object":
                df[col] = default
            else:
                df[col] = pd.array([default] * len(df), dtype=dtype)
        elif dtype != "object":
            if dtype == "boolean":
                df[col] = df[col].astype("boolean")
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
    return df


def migrate_legacy_model_columns(df: pd.DataFrame, model_id: str) -> pd.DataFrame:
    legacy_pairs = {
        "stability_outputs_json": f"stability_outputs_json_{model_id}",
        "greedy_output": f"greedy_output_{model_id}",
        "UAR": f"UAR_{model_id}",
        "mNED": f"mNED_{model_id}",
        "anchor_mNED": f"anchor_mNED_{model_id}",
        "peak_eps": f"peak_eps_{model_id}",
        "SProb": f"SProb_{model_id}",
    }
    for base_col, legacy_col in legacy_pairs.items():
        if base_col not in df.columns and legacy_col in df.columns:
            df[base_col] = df[legacy_col]
    return df


def init_control_columns(df: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("ctrl_outputs_json", "object", ""),
        ("ctrl_greedy_output", "object", ""),
        ("UAR_ctrl", "Float64", pd.NA),
        ("mNED_ctrl", "Float64", pd.NA),
        ("anchor_mNED_ctrl", "Float64", pd.NA),
        ("peak_eps_ctrl", "Float64", pd.NA),
    ]
    for col, dtype, default in specs:
        if col not in df.columns:
            if dtype == "object":
                df[col] = default
            else:
                df[col] = pd.array([default] * len(df), dtype=dtype)
        elif dtype != "object":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
    return df


def is_reference_row_done(row: pd.Series) -> bool:
    try:
        return not pd.isna(row.get("SProb")) and isinstance(row.get("stability_outputs_json"), str) and bool(row.get("stability_outputs_json"))
    except Exception:
        return False


def is_control_row_done(row: pd.Series) -> bool:
    try:
        return not pd.isna(row.get("UAR_ctrl")) and isinstance(row.get("ctrl_outputs_json"), str) and bool(row.get("ctrl_outputs_json"))
    except Exception:
        return False


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    ensure_parent(path)
    df.to_parquet(path, index=False)


# ---------------------------------------------------------------------------
# Control / reference passes
# ---------------------------------------------------------------------------

def run_control_pass(
    *,
    df_control: pd.DataFrame,
    text_column: str,
    id_column: str,
    client: BaseTextClient,
    decoding: Dict[str, Any],
    N: int,
    limit: Optional[int],
    sleep_s: float,
    save_every: int,
    control_parquet: str,
    log_path: str,
    max_pairs: int = 435,
    anchor_eps: float = 0.15,
    token_encoder: Optional[Callable[[str], List[Any]]] = None,
    greedy_anchor_cfg: Optional[Dict[str, Any]] = None,
    prompt_template: str = STABILITY_PROMPT_TEMPLATE,
) -> Dict[str, Any]:
    df_control = init_control_columns(df_control.copy())
    processed_new = 0
    failures = 0

    counter = 0
    for idx, row in df_control.iterrows():
        if limit is not None and counter >= limit:
            break
        counter += 1

        if is_control_row_done(row):
            continue

        item_key = row.get(id_column, idx)
        doc = normalize_text(row.get(text_column, ""))
        if not doc:
            failures += 1
            log_jsonl(log_path, {"pass": "control", "id": item_key, "status": "skip_empty_document"})
            continue

        prompt = build_stability_prompt(doc, prompt_template)
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
                greedy_anchor_cfg=greedy_anchor_cfg,
            )
            df_control.at[idx, "ctrl_outputs_json"] = json_dumps(metrics["outputs"])
            df_control.at[idx, "ctrl_greedy_output"] = metrics["greedy_output"]
            df_control.at[idx, "UAR_ctrl"] = float(metrics["UAR"])
            df_control.at[idx, "mNED_ctrl"] = float(metrics["mNED"])
            df_control.at[idx, "anchor_mNED_ctrl"] = float(metrics["anchor_mNED"])
            df_control.at[idx, "peak_eps_ctrl"] = float(metrics["peak_eps"])

            log_jsonl(log_path, {
                "pass": "control",
                "id": item_key,
                "status": "ok",
                "UAR": round(float(metrics["UAR"]), 6),
                "mNED": round(float(metrics["mNED"]), 6),
                "anchor_mNED": round(float(metrics["anchor_mNED"]), 6),
                "peak_eps": round(float(metrics["peak_eps"]), 6),
                "N_collected": len(metrics["outputs"]),
            })
            processed_new += 1
        except Exception as e:
            failures += 1
            log_jsonl(log_path, {"pass": "control", "id": item_key, "status": "error", "error": repr(e)})

        if processed_new > 0 and processed_new % max(1, int(save_every)) == 0:
            save_parquet(df_control, control_parquet)

    save_parquet(df_control, control_parquet)

    uar_v = pd.to_numeric(df_control["UAR_ctrl"], errors="coerce")
    mned_v = pd.to_numeric(df_control["mNED_ctrl"], errors="coerce")
    anchor_v = pd.to_numeric(df_control["anchor_mNED_ctrl"], errors="coerce")
    peak_v = pd.to_numeric(df_control["peak_eps_ctrl"], errors="coerce")
    valid = uar_v.notna() & mned_v.notna()

    return {
        "df_control": df_control,
        "UAR_control": float(uar_v[valid].mean()) if valid.any() else None,
        "mNED_control": float(mned_v[valid].mean()) if valid.any() else None,
        "anchor_mNED_control": float(anchor_v[valid].mean()) if anchor_v.notna().any() else None,
        "peak_eps_control": float(peak_v[peak_v.notna()].mean()) if peak_v.notna().any() else None,
        "processed_new": processed_new,
        "failures": failures,
    }


def run_reference_pass(
    *,
    df: pd.DataFrame,
    text_column: str,
    id_column: str,
    client: BaseTextClient,
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
    token_encoder: Optional[Callable[[str], List[Any]]] = None,
    greedy_anchor_cfg: Optional[Dict[str, Any]] = None,
    prompt_template: str = STABILITY_PROMPT_TEMPLATE,
) -> Dict[str, Any]:
    df = init_reference_columns(df.copy())
    processed_new = 0
    failures = 0

    counter = 0
    for idx, row in df.iterrows():
        if limit is not None and counter >= limit:
            break
        counter += 1

        if is_reference_row_done(row):
            continue

        item_key = row.get(id_column, idx)
        doc = normalize_text(row.get(text_column, ""))
        if not doc:
            failures += 1
            log_jsonl(log_path, {"pass": "reference", "id": item_key, "status": "skip_empty_document"})
            continue

        prompt = build_stability_prompt(doc, prompt_template)
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
                greedy_anchor_cfg=greedy_anchor_cfg,
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
                float(metrics["UAR"] - uar_control)
                if uar_control is not None and not pd.isna(metrics["UAR"])
                else None
            )
            mned_ratio = (
                float(metrics["mNED"] / mned_control)
                if mned_control is not None and mned_control > 0 and not pd.isna(metrics["mNED"])
                else None
            )

            df.at[idx, "stability_outputs_json"] = json_dumps(metrics["outputs"])
            df.at[idx, "greedy_output"] = metrics["greedy_output"]
            df.at[idx, "UAR"] = float(metrics["UAR"])
            df.at[idx, "mNED"] = float(metrics["mNED"])
            df.at[idx, "anchor_mNED"] = float(metrics["anchor_mNED"])
            df.at[idx, "peak_eps"] = float(metrics["peak_eps"])
            df.at[idx, "SProb"] = int(sprob)
            df.at[idx, "contrast_met"] = bool(contrast_met)
            if delta_uar is not None:
                df.at[idx, "delta_UAR"] = delta_uar
            if mned_ratio is not None:
                df.at[idx, "mNED_ratio"] = mned_ratio

            log_jsonl(log_path, {
                "pass": "reference",
                "id": item_key,
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
            failures += 1
            log_jsonl(log_path, {"pass": "reference", "id": item_key, "status": "error", "error": repr(e)})

        if processed_new > 0 and processed_new % max(1, int(save_every)) == 0:
            save_parquet(df, out_parquet)

    save_parquet(df, out_parquet)
    return {"df": df, "processed_new": processed_new, "failures": failures}


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def aggregate_reference_stats(df: pd.DataFrame, processed_new: int, failures: int) -> Dict[str, Any]:
    uar_vals = pd.to_numeric(df.get("UAR"), errors="coerce")
    mned_vals = pd.to_numeric(df.get("mNED"), errors="coerce")
    sprob_vals = pd.to_numeric(df.get("SProb"), errors="coerce")
    valid = uar_vals.notna() & mned_vals.notna()
    n_valid = int(valid.sum())
    sprob_aggregate = map_to_SProb_aggregate(sprob_vals[valid]) if n_valid else 0

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
        "SProb_aggregate": int(sprob_aggregate),
        **sprob_dist,
    }


def summarize_reference_df(df: pd.DataFrame) -> Dict[str, Any]:
    sprob_counts = {}
    sprob_aggregate = None
    if "SProb" in df.columns:
        s = pd.to_numeric(df["SProb"], errors="coerce")
        sprob_counts = {str(int(k)): int(v) for k, v in s.dropna().astype(int).value_counts().sort_index().items()}
        if s.notna().any():
            sprob_aggregate = map_to_SProb_aggregate(s)

    summary = {
        "n_rows": int(len(df)),
        "n_completed": int(df["SProb"].notna().sum()) if "SProb" in df.columns else 0,
        "n_with_outputs": int(df["stability_outputs_json"].astype(str).str.len().gt(0).sum()) if "stability_outputs_json" in df.columns else 0,
        "SProb_counts": sprob_counts,
        "SProb_aggregate": sprob_aggregate,
    }

    for col in ["UAR", "mNED", "anchor_mNED", "peak_eps", "delta_UAR", "mNED_ratio"]:
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce")
            summary[col] = {
                "mean": safe_float(v.mean()),
                "median": safe_float(v.median()),
                "min": safe_float(v.min()),
                "max": safe_float(v.max()),
            }
    return summary


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    ap = argparse.ArgumentParser(description="Run stability detector v7.")
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--model_id", "--model-id", dest="model_id", required=True, help="Model identifier from config")
    ap.add_argument("--input-parquet", default=None, help="Reference dataset parquet; defaults to project.frozen_master_table_path")
    ap.add_argument("--text-column", default=None, help="Document/text column in reference parquet")
    ap.add_argument("--id-column", default=None, help="ID column in reference parquet")
    ap.add_argument("--control-text-column", default=None, help="Document/text column in control parquet (optional)")
    ap.add_argument("--control-id-column", default=None, help="ID column in control parquet (optional)")
    ap.add_argument("--limit", type=int, default=None, help="Optional override for runtime.limit")
    ap.add_argument("--max-pairs", type=int, default=435, help="Maximum number of pairwise comparisons for mNED")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    stab_cfg = cfg["stability"]
    master_path = args.input_parquet or cfg["project"]["frozen_master_table_path"]
    model_cfg = next((m for m in cfg.get("models", []) if m.get("model_id") == args.model_id), None)
    if model_cfg is None:
        raise ValueError(f"model_id='{args.model_id}' not found in config")

    use_control_baseline = bool(stab_cfg.get("use_control_baseline", False))
    control_set_path = stab_cfg.get("control_set_path")
    N = int(stab_cfg.get("N_samples", 20))

    decoding = {
        "temperature": float(stab_cfg.get("decoding", {}).get("temperature", 0.8)),
        "top_p": float(stab_cfg.get("decoding", {}).get("top_p", 1.0)),
        "max_tokens": int(stab_cfg.get("decoding", {}).get("max_tokens", 80)),
    }
    greedy_anchor_cfg = {
        "temperature": float(stab_cfg.get("greedy_anchor", {}).get("temperature", 0.0)),
        "top_p": float(stab_cfg.get("greedy_anchor", {}).get("top_p", 1.0)),
        "max_tokens": int(stab_cfg.get("greedy_anchor", {}).get("max_tokens", decoding["max_tokens"])),
    }
    runtime = dict(stab_cfg.get("runtime", {}))
    outputs_cfg = dict(stab_cfg.get("outputs", {}))
    metrics_cfg = dict(stab_cfg.get("metrics", {}))
    token_cfg = dict(stab_cfg.get("tokenization", {}))
    if metrics_cfg:
        if "tokenization" in metrics_cfg:
            token_cfg.setdefault("tokenization", metrics_cfg.get("tokenization"))
        if "tokenizer_name" in metrics_cfg:
            token_cfg.setdefault("tokenizer_name", metrics_cfg.get("tokenizer_name"))
        if "name" in metrics_cfg:
            token_cfg.setdefault("name", metrics_cfg.get("name"))

    sleep_s = float(runtime.get("sleep_s", 0.10))
    save_every = int(runtime.get("save_every", 10))
    limit = args.limit if args.limit is not None else runtime.get("limit")
    limit = None if limit in (None, "null") else int(limit)
    anchor_eps = float(metrics_cfg.get("anchor_eps", token_cfg.get("anchor_eps", 0.15)))
    max_pairs = int(metrics_cfg.get("max_pairs", args.max_pairs))

    # Prompt template: accept a named key ("news_article" | "generic") or a raw
    # string with {DOCUMENT} placeholder. Defaults to "news_article" for backward compat.
    _prompt_key = str(stab_cfg.get("prompt", {}).get("template", "news_article"))
    prompt_template = STABILITY_PROMPT_TEMPLATES.get(_prompt_key, _prompt_key)
    if "{DOCUMENT}" not in prompt_template:
        raise ValueError(
            f"prompt.template must contain {{DOCUMENT}} placeholder. Got: {_prompt_key!r}"
        )

    out_parquet = format_path(outputs_cfg.get("parquet", "runs/v6_stability_{model_id}.parquet"), args.model_id)
    log_jsonl_path = format_path(outputs_cfg.get("log_jsonl", "logs/v6_stability_{model_id}.jsonl"), args.model_id)
    summary_json = format_path(outputs_cfg.get("summary_json", "outputs/v6_stability_summary_{model_id}.json"), args.model_id)
    control_parquet = format_path(outputs_cfg.get("control_parquet", "runs/v6_stability_ctrl_{model_id}.parquet"), args.model_id)

    for p in [out_parquet, log_jsonl_path, summary_json, control_parquet]:
        ensure_parent(p)

    token_encoder = load_token_encoder(token_cfg, model_id=model_cfg["model_name"]) if token_cfg else load_token_encoder({}, model_id=model_cfg["model_name"])
    client = select_client(model_cfg)
    t0 = time.time()

    # Reference data: resume from output parquet if present, otherwise use input parquet.
    if os.path.exists(out_parquet):
        df_ref = pd.read_parquet(out_parquet)
    else:
        df_ref = pd.read_parquet(master_path)
    df_ref = migrate_legacy_model_columns(df_ref, args.model_id)
    text_column = args.text_column or infer_text_column(df_ref)
    id_column = args.id_column or infer_id_column(df_ref)

    control_stats: Dict[str, Any] = {
        "UAR_control": None,
        "mNED_control": None,
        "anchor_mNED_control": None,
        "peak_eps_control": None,
    }

    if use_control_baseline:
        if not control_set_path:
            raise ValueError("use_control_baseline=true but control_set_path is missing.")
        if os.path.exists(control_parquet):
            df_control = pd.read_parquet(control_parquet)
        else:
            df_control = pd.read_parquet(control_set_path)
        control_text_column = args.control_text_column or infer_text_column(df_control)
        control_id_column = args.control_id_column or infer_id_column(df_control)

        control_stats = run_control_pass(
            df_control=df_control,
            text_column=control_text_column,
            id_column=control_id_column,
            client=client,
            decoding=decoding,
            N=N,
            limit=limit,
            sleep_s=sleep_s,
            save_every=save_every,
            control_parquet=control_parquet,
            log_path=log_jsonl_path,
            max_pairs=max_pairs,
            anchor_eps=anchor_eps,
            token_encoder=token_encoder,
            greedy_anchor_cfg=greedy_anchor_cfg,
            prompt_template=prompt_template,
        )

    ref_stats = run_reference_pass(
        df=df_ref,
        text_column=text_column,
        id_column=id_column,
        client=client,
        decoding=decoding,
        N=N,
        uar_control=control_stats.get("UAR_control"),
        mned_control=control_stats.get("mNED_control"),
        limit=limit,
        sleep_s=sleep_s,
        save_every=save_every,
        out_parquet=out_parquet,
        log_path=log_jsonl_path,
        max_pairs=max_pairs,
        anchor_eps=anchor_eps,
        token_encoder=token_encoder,
        greedy_anchor_cfg=greedy_anchor_cfg,
        prompt_template=prompt_template,
    )

    ref_df = ref_stats["df"]
    ref_summary = summarize_reference_df(ref_df)
    ref_aggregate = aggregate_reference_stats(
        ref_df,
        processed_new=int(ref_stats["processed_new"]),
        failures=int(ref_stats["failures"]),
    )
    sprob_aggregate = int(ref_aggregate.get("SProb_aggregate", 0) or 0)
    ref_df["SProb_aggregate"] = pd.Series([sprob_aggregate] * len(ref_df), dtype="Int64")
    save_parquet(ref_df, out_parquet)
    elapsed_s = time.time() - t0
    sprob3_total = int((pd.to_numeric(ref_df.get("SProb"), errors="coerce") == 3).sum())
    summary_payload = {
        "stage": "stability_v7",
        "method": "Dong et al. (2024) CDD with v7 stability runner",
        "description": stab_cfg.get("description"),
        "model_id": args.model_id,
        "provider": model_cfg["provider"],
        "model_name": model_cfg["model_name"],
        "dataset_path": master_path,
        "n_rows_total": int(len(ref_df)),
        **ref_aggregate,
        "SProb3_total": sprob3_total,
        "N_samples": N,
        "n_samples": N,
        "SProb_counts": ref_summary.get("SProb_counts", {}),
        "SProb_aggregate": sprob_aggregate,
        "prompt_template": _prompt_key,
        "decoding": decoding,
        "temperature": decoding["temperature"],
        "top_p": decoding["top_p"],
        "max_tokens": decoding["max_tokens"],
        "runtime": {
            "sleep_s": sleep_s,
            "save_every": save_every,
            "limit": limit,
        },
        "greedy_anchor": greedy_anchor_cfg,
        "metrics": {
            "distance": stab_cfg.get("metrics", {}).get("distance", "token_level"),
            "max_pairs": max_pairs,
            "anchor_eps": anchor_eps,
            "tokenization": token_cfg.get("tokenization") or token_cfg.get("mode") or "regex",
            "tokenizer_name": token_cfg.get("tokenizer_name") or token_cfg.get("name"),
        },
        "provider_details": {
            "kind": model_cfg["provider"],
        },
        "control_baseline": {
            "enabled": use_control_baseline,
            "control_set_path": control_set_path,
            "UAR_control": control_stats.get("UAR_control"),
            "mNED_control": control_stats.get("mNED_control"),
            "anchor_mNED_control": control_stats.get("anchor_mNED_control"),
            "peak_eps_control": control_stats.get("peak_eps_control"),
        },
        "reference_summary": ref_summary,
        "elapsed_seconds": elapsed_s,
        "out_parquet": out_parquet,
        "log_jsonl": log_jsonl_path,
        "summary_json": summary_json,
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "status": "ok",
        "out_parquet": out_parquet,
        "control_parquet": control_parquet if use_control_baseline else None,
        "summary_json": summary_json,
        "log_jsonl": log_jsonl_path,
    }, ensure_ascii=False))
    print("Done.")
    print(f"Model: {args.model_id} ({model_cfg['model_name']})")
    print(f"SProb_aggregate: {sprob_aggregate}")
    print(f"SProb_counts: {ref_summary.get('SProb_counts', {})}")
    print(f"Output: {out_parquet}")
    print(f"Summary: {summary_json}")
    print(f"Log: {log_jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
