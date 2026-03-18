#!/usr/bin/env python3
"""
scripts/run_stability_detector.py

Stability-based Probability Proxy (SProb) runner for API-only contamination pipeline.

Features:
- Reads configs/run_config.yaml
- Supports OpenAI (gpt-4o-mini) and Gemini (gemini-1.5-flash via API key)
- --model_id selects one model per run
- --limit for pilot runs (process at most N NEW rows)
- Writes:
  - parquet with appended stability columns
  - jsonl log of per-item status
  - summary json (aggregate UAR/mNED stats + run metadata)
- Resume-friendly: skips rows where model-specific stability outputs already exist

Assumptions:
- Frozen master table has columns:
  - xsum_id
  - document_norm (preferred) or document
- Prompt template is stored in src/prompts.py as STABILITY_PROMPT_TEMPLATE
"""

import argparse
import json
import time
import unicodedata
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
import yaml

# Ensure project root is on sys.path for `import src.*` when running from scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prompts import STABILITY_PROMPT_TEMPLATE
from src.clients.openai_client import OpenAIClient


# -----------------------
# Utilities
# -----------------------

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
    # prefer normalized document if present
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

    if provider == "openai":
        api_key = os.environ.get(api_key_var or "OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(f"Missing {api_key_var or 'OPENAI_API_KEY'} env var")
        return OpenAIClient(model=model_name, api_key=api_key)
    if provider == "gemini":
        from src.clients.gemini_client import GeminiClient
        return GeminiClient(model=model_name)

    raise ValueError(f"Unsupported provider: {provider}")


def normalized_edit_distance(a: str, b: str) -> float:
    a = a or ""
    b = b or ""
    if a == b:
        return 0.0
    if len(a) == 0 or len(b) == 0:
        return 1.0

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(ins, delete, sub))
        prev = curr

    dist = prev[-1]
    denom = max(len(a), len(b))
    return float(dist) / float(denom)


def compute_uar(outputs: List[str]) -> float:
    """
    Unique Answer Rate: unique(normalized outputs) / N
    """
    normed = [normalize_text(x).lower() for x in outputs if isinstance(x, str)]
    if not normed:
        return float("nan")
    unique = len(set(normed))
    return unique / len(normed)


def compute_mned_pairwise(outputs: List[str], max_pairs: Optional[int] = None) -> float:
    """
    Mean normalized edit distance across pairs.
    If max_pairs is set, subsample pairs deterministically for speed.
    """
    normed = [normalize_text(x).lower() for x in outputs if isinstance(x, str)]
    n = len(normed)
    if n < 2:
        return float("nan")

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))

    # Deterministic subsample to cap compute cost
    if max_pairs is not None and len(pairs) > max_pairs:
        step = max(1, len(pairs) // max_pairs)
        pairs = pairs[::step][:max_pairs]

    dists = []
    for i, j in pairs:
        dists.append(normalized_edit_distance(normed[i], normed[j]))
    return float(sum(dists) / len(dists)) if dists else float("nan")


def map_to_SProb(uar: float, mned: float) -> int:
    """
    Map probability-concentration evidence to level 0..3.

    Higher SProb = lower output variability / stronger peakedness.
    Methodology-aligned thresholds:
      - Level 0: UAR > 0.60 AND mNED > 0.25
      - Level 1: 0.40 <= UAR <= 0.60 AND 0.15 <= mNED <= 0.25
      - Level 2: 0.20 <= UAR < 0.40 OR 0.08 <= mNED < 0.15
      - Level 3: UAR < 0.20 OR mNED < 0.08

    Note:
    - This implementation aligns the *direction* of SProb with the dissertation
      methodology: lower UAR / lower mNED imply higher contamination-consistent
      concentration.
    - The control-based contrast condition for Level 3 (>= 2x vs control),
      described in the methodology, is not operationalized in this script yet.
    """
    if pd.isna(uar) or pd.isna(mned):
        return 0

    if (uar < 0.20) or (mned < 0.08):
        return 3
    if (0.20 <= uar < 0.40) or (0.08 <= mned < 0.15):
        return 2
    if (0.40 <= uar <= 0.60) and (0.15 <= mned <= 0.25):
        return 1
    if (uar > 0.60) and (mned > 0.25):
        return 0

    # Conservative fallback for mixed boundary cases.
    return 1


# -----------------------
# Core Stability logic
# -----------------------

def run_stability(
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
    max_pairs: int = 435,  # 30 choose 2 = 435 (full pairwise)
) -> Dict[str, Any]:
    """
    Adds model-specific stability columns:
      stability_outputs_json_{model_id}
      UAR_{model_id}
      mNED_{model_id}
      SProb_{model_id}
    """
    col_out = f"stability_outputs_json_{model_id}"
    col_uar = f"UAR_{model_id}"
    col_mned = f"mNED_{model_id}"
    col_sprob = f"SProb_{model_id}"

    # Ensure columns exist with consistent dtypes
    if col_out not in df.columns:
        df[col_out] = ""

    if col_uar not in df.columns:
        df[col_uar] = pd.Series([pd.NA] * len(df), dtype="Float64")
    else:
        df[col_uar] = pd.to_numeric(df[col_uar], errors="coerce").astype("Float64")

    if col_mned not in df.columns:
        df[col_mned] = pd.Series([pd.NA] * len(df), dtype="Float64")
    else:
        df[col_mned] = pd.to_numeric(df[col_mned], errors="coerce").astype("Float64")

    if col_sprob not in df.columns:
        df[col_sprob] = pd.Series([pd.NA] * len(df), dtype="Int64")
    else:
        df[col_sprob] = pd.to_numeric(df[col_sprob], errors="coerce").astype("Int64")

    processed_new = 0
    failures = 0

    for idx, row in df.iterrows():
        # Resume-friendly: skip if outputs already exist for this model
        existing = row.get(col_out, "")
        if isinstance(existing, str) and existing.strip().startswith("[") and len(existing.strip()) > 10:
            continue

        item_key = str(row.get("xsum_id", idx))
        doc = get_document_field(row)
        if not doc:
            failures += 1
            log_jsonl(log_path, {
                "row": int(idx),
                "xsum_id": item_key,
                "status": "error_missing_document",
            })
            continue

        doc_n = normalize_text(doc)
        prompt = build_stability_prompt(doc_n)

        outputs: List[str] = []
        try:
            for s in range(N):
                out = client.generate_text(
                    prompt=prompt,
                    temperature=float(decoding["temperature"]),
                    top_p=float(decoding["top_p"]),
                    max_tokens=int(decoding["max_tokens"]),
                )
                outputs.append((out or "").strip())
                time.sleep(float(sleep_s))

            uar = compute_uar(outputs)
            mned = compute_mned_pairwise(outputs, max_pairs=max_pairs)
            sprob = map_to_SProb(uar, mned)

            df.at[idx, col_out] = json.dumps(outputs, ensure_ascii=False)
            df.at[idx, col_uar] = float(uar)
            df.at[idx, col_mned] = float(mned)
            df.at[idx, col_sprob] = int(sprob)

            log_jsonl(log_path, {
                "row": int(idx),
                "xsum_id": item_key,
                "status": "ok",
                "UAR": float(uar),
                "mNED": float(mned),
                "SProb": int(sprob),
                "N": int(N),
            })

            processed_new += 1

        except Exception as e:
            import traceback
            failures += 1
            log_jsonl(log_path, {
                "row": int(idx),
                "xsum_id": item_key,
                "status": "api_error",
                "error_type": type(e).__name__,
                "error_repr": repr(e),
                "traceback": traceback.format_exc(),
                "n_collected": len(outputs),
            })

        if save_every and processed_new > 0 and (processed_new % save_every == 0):
            ensure_parent_dir(out_parquet)
            df.to_parquet(out_parquet, index=False)
            print(f"Saved progress: {processed_new} new rows processed -> {out_parquet}")

        if limit is not None and processed_new >= limit:
            break

    ensure_parent_dir(out_parquet)
    df.to_parquet(out_parquet, index=False)

    uar_vals = pd.to_numeric(df[col_uar], errors="coerce")
    mned_vals = pd.to_numeric(df[col_mned], errors="coerce")
    valid = uar_vals.notna() & mned_vals.notna()

    summary = {
        "processed_new": processed_new,
        "failures": failures,
        "valid_items": int(valid.sum()),
        "UAR_mean": float(uar_vals[valid].mean()) if valid.any() else None,
        "UAR_median": float(uar_vals[valid].median()) if valid.any() else None,
        "mNED_mean": float(mned_vals[valid].mean()) if valid.any() else None,
        "mNED_median": float(mned_vals[valid].median()) if valid.any() else None,
        "UAR_p10": float(uar_vals[valid].quantile(0.10)) if valid.any() else None,
        "UAR_p90": float(uar_vals[valid].quantile(0.90)) if valid.any() else None,
        "mNED_p10": float(mned_vals[valid].quantile(0.10)) if valid.any() else None,
        "mNED_p90": float(mned_vals[valid].quantile(0.90)) if valid.any() else None,
        "columns_added": [col_out, col_uar, col_mned, col_sprob],
        "total_rows": int(len(df)),
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to run_config.yaml")
    parser.add_argument("--model_id", type=str, required=True, help="Model ID from run_config.yaml (e.g., gpt4omini)")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N NEW rows (pilot runs)")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    master_path = cfg["project"]["frozen_master_table_path"]

    model_cfg = None
    for m in cfg["models"]:
        if m["model_id"] == args.model_id:
            model_cfg = m
            break
    if model_cfg is None:
        raise ValueError(f"model_id='{args.model_id}' not found in config")

    stab_cfg = cfg["stability"]
    decoding = stab_cfg["decoding"]
    N = int(stab_cfg["N_samples"])
    sleep_s = float(stab_cfg["runtime"]["sleep_s"])
    save_every = int(stab_cfg["runtime"]["save_every"])

    out_parquet = format_path(stab_cfg["outputs"]["parquet"], args.model_id)
    log_path = format_path(stab_cfg["outputs"]["log_jsonl"], args.model_id)
    summary_json = format_path(stab_cfg["outputs"]["summary_json"], args.model_id)

    # Resume-friendly loading:
    # continue from existing stage output if present; otherwise start from frozen master table.
    if Path(out_parquet).exists():
        df = pd.read_parquet(out_parquet)
        print(f"Resuming from existing output: {out_parquet}")
    else:
        df = pd.read_parquet(master_path)
        print(f"Starting from master table: {master_path}")

    client = select_client(model_cfg)

    t0 = time.time()
    results = run_stability(
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
    )
    elapsed_s = time.time() - t0

    summary = {
        "stage": "stability",
        "model_id": args.model_id,
        "provider": model_cfg["provider"],
        "model_name": model_cfg["model_name"],
        "dataset_path": master_path,
        "n_rows_total": int(len(df)),
        "processed_new": results["processed_new"],
        "failures": results["failures"],
        "valid_items": results["valid_items"],
        "UAR_mean": results["UAR_mean"],
        "UAR_median": results["UAR_median"],
        "mNED_mean": results["mNED_mean"],
        "mNED_median": results["mNED_median"],
        "UAR_p10": results["UAR_p10"],
        "UAR_p90": results["UAR_p90"],
        "mNED_p10": results["mNED_p10"],
        "mNED_p90": results["mNED_p90"],
        "decoding": decoding,
        "N_samples": N,
        "elapsed_seconds": elapsed_s,
        "out_parquet": out_parquet,
        "log_jsonl": log_path,
    }

    ensure_parent_dir(summary_json)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Model: {args.model_id} ({model_cfg['model_name']})")
    print(f"Processed new: {results['processed_new']}; failures: {results['failures']}")
    print(f"UAR_mean: {results['UAR_mean']}; mNED_mean: {results['mNED_mean']}")
    print("Output:", out_parquet)
    print("Summary:", summary_json)
    print("Log:", log_path)


if __name__ == "__main__":
    main()
