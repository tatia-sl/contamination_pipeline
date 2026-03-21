#!/usr/bin/env python3
"""
scripts/run_mem_probe.py

Memorization Probing Detector (SMem) runner for API-only contamination pipeline.

Features:
- Reads configs/run_config.yaml
- Supports OpenAI (gpt-4o-mini) and Gemini (gemini-1.5-flash via API key)
- --model_id selects one model per run
- --limit for pilot runs (process at most N NEW rows)
- Writes:
  - parquet with appended memorization columns
  - jsonl log of per-item status
  - summary json (aggregate EM/NE stats + run metadata)
- Resume-friendly: skips rows where model-specific EM is already filled

Assumptions:
- Frozen master table has columns:
  - xsum_id
  - split
  - prefix_ref
  - control_prefix
  - summary_ref_norm
- Prompt template is stored in src/prompts.py as MEM_PROMPT_TEMPLATE
"""

import argparse
import json
import time
import hashlib
import unicodedata
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import pandas as pd
import yaml

# Ensure project root is on sys.path for `import src.*` when running from scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prompts import MEM_PROMPT_TEMPLATE
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


def stable_int_from_str(s: str) -> int:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]
    return int(h, 16)


def normalize_text(s: str) -> str:
    """
    Match your dataset normalization style.
    Keep it minimal, deterministic, and consistent across models.
    """
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = " ".join(s.split())
    return s.strip()


def extract_gold_suffix(ref_full: str, prefix: str) -> str:
    """
    Extract the expected continuation (gold suffix) from the normalized full reference
    and normalized prefix.

    If the normalized reference starts with the normalized prefix, return the remaining
    tail. Otherwise return an empty string to signal alignment failure.
    """
    ref_full = normalize_text(ref_full)
    prefix = normalize_text(prefix)

    if not ref_full or not prefix:
        return ""

    if ref_full.startswith(prefix):
        return ref_full[len(prefix):].strip()

    return ""


def build_mem_prompt(prefix: str, split_name: str) -> str:
    return MEM_PROMPT_TEMPLATE.format(
        SPLIT_NAME=split_name,
        PREFIX=prefix,
    )


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


def format_path(template: str, model_id: str) -> str:
    return template.replace("{model_id}", model_id)


def normalized_edit_distance(a: str, b: str) -> float:
    """
    Normalized Levenshtein distance in [0,1].
    0 = identical, 1 = completely different.
    Implementation: classic DP, O(len(a)*len(b)) for short strings (summaries) is OK.
    """
    a = a or ""
    b = b or ""
    if a == b:
        return 0.0
    if len(a) == 0:
        return 1.0
    if len(b) == 0:
        return 1.0

    # DP with two rows
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


def map_to_SMem(em: int, ne: float) -> int:
    """
    Map memorization evidence to level 0..3.
    Adjust thresholds here only if you have a frozen rubric already.
    Conservative defaults:
      - Level 3: Exact match (EM=1)
      - Level 2: Near-verbatim (NE <= 0.10)
      - Level 1: Weak similarity (NE <= 0.25)
      - Level 0: Otherwise
    """
    if em == 1:
        return 3
    if ne <= 0.10:
        return 2
    if ne <= 0.25:
        return 1
    return 0


# -----------------------
# Core SMem logic
# -----------------------

def run_mem_probe(
    df: pd.DataFrame,
    client,
    model_id: str,
    decoding: Dict[str, Any],
    use_control: bool,
    limit: Optional[int],
    sleep_s: float,
    save_every: int,
    out_parquet: str,
    log_path: str
) -> Dict[str, Any]:
    """
    Adds model-specific memorization columns:
      mem_completion_{model_id}
      mem_completion_ctrl_{model_id} (optional)
      EM_{model_id}
      NE_{model_id}
      SMem_{model_id}
    """
    col_comp = f"mem_completion_{model_id}"
    col_comp_ctrl = f"mem_completion_ctrl_{model_id}"
    col_em = f"EM_{model_id}"
    col_ne = f"NE_{model_id}"
    col_smem = f"SMem_{model_id}"

    # Ensure columns exist with consistent dtypes
    if col_comp not in df.columns:
        df[col_comp] = ""
    if use_control and col_comp_ctrl not in df.columns:
        df[col_comp_ctrl] = ""

    if col_em not in df.columns:
        df[col_em] = pd.Series([pd.NA] * len(df), dtype="Int64")
    else:
        df[col_em] = pd.to_numeric(df[col_em], errors="coerce").astype("Int64")

    if col_ne not in df.columns:
        df[col_ne] = pd.Series([pd.NA] * len(df), dtype="Float64")
    else:
        df[col_ne] = pd.to_numeric(df[col_ne], errors="coerce").astype("Float64")

    if col_smem not in df.columns:
        df[col_smem] = pd.Series([pd.NA] * len(df), dtype="Int64")
    else:
        df[col_smem] = pd.to_numeric(df[col_smem], errors="coerce").astype("Int64")

    processed_new = 0
    failures = 0

    for idx, row in df.iterrows():
        # Resume-friendly: skip if EM already computed for this model
        existing_em = row.get(col_em, "")
        if isinstance(existing_em, (int, float)) and pd.notna(existing_em):
            # already computed numeric EM
            continue
        if isinstance(existing_em, str) and existing_em.strip() in ["0", "1"]:
            continue

        item_key = str(row.get("xsum_id", idx))

        prefix = row.get("prefix_ref", "")
        ref = row.get("summary_ref_norm", "")
        split_name = row.get("split", "test")
        if not isinstance(split_name, str) or not split_name.strip():
            split_name = "test"
        split_name = split_name.strip()

        if not isinstance(prefix, str) or not prefix.strip():
            failures += 1
            log_jsonl(log_path, {
                "row": int(idx),
                "xsum_id": item_key,
                "status": "error_missing_prefix_ref",
            })
            continue

        if not isinstance(ref, str) or not ref.strip():
            failures += 1
            log_jsonl(log_path, {
                "row": int(idx),
                "xsum_id": item_key,
                "status": "error_missing_summary_ref",
            })
            continue

        prefix_n = normalize_text(prefix)
        ref_n = normalize_text(ref)
        gold_suffix = extract_gold_suffix(ref_n, prefix_n)

        if not gold_suffix:
            failures += 1
            log_jsonl(log_path, {
                "row": int(idx),
                "xsum_id": item_key,
                "status": "error_prefix_not_aligned_with_reference",
            })
            continue

        prompt = build_mem_prompt(prefix_n, split_name)

        try:
            comp = client.generate_text(
                prompt=prompt,
                temperature=float(decoding["temperature"]),
                top_p=float(decoding["top_p"]),
                max_tokens=int(decoding["max_tokens"]),
            )
            comp_n = normalize_text(comp)

            # Compare completion against the gold suffix (the expected second piece)
            em = 1 if comp_n == gold_suffix else 0
            ne = normalized_edit_distance(comp_n, gold_suffix)
            smem = map_to_SMem(em=em, ne=ne)

            df.at[idx, col_comp] = comp
            df.at[idx, col_em] = int(em)
            df.at[idx, col_ne] = float(ne)
            df.at[idx, col_smem] = int(smem)

            # Optional control prefix probing
            comp_ctrl = None
            if use_control:
                ctrl = row.get("control_prefix", "")
                if isinstance(ctrl, str) and ctrl.strip():
                    ctrl_n = normalize_text(ctrl)
                    prompt_ctrl = build_mem_prompt(ctrl_n, split_name)
                    comp_ctrl = client.generate_text(
                        prompt=prompt_ctrl,
                        temperature=float(decoding["temperature"]),
                        top_p=float(decoding["top_p"]),
                        max_tokens=int(decoding["max_tokens"]),
                    )
                    df.at[idx, col_comp_ctrl] = comp_ctrl

            log_jsonl(log_path, {
                "row": int(idx),
                "xsum_id": item_key,
                "status": "ok",
                "EM": int(em),
                "NE": float(ne),
                "SMem": int(smem),
                "prefix_ref_norm": prefix_n,
                "gold_suffix": gold_suffix,
                "completion_norm": comp_n,
                "has_ctrl": bool(use_control),
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
            })

        if save_every and processed_new > 0 and (processed_new % save_every == 0):
            ensure_parent_dir(out_parquet)
            df.to_parquet(out_parquet, index=False)
            print(f"Saved progress: {processed_new} new rows processed -> {out_parquet}")

        if limit is not None and processed_new >= limit:
            break

        time.sleep(float(sleep_s))

    ensure_parent_dir(out_parquet)
    df.to_parquet(out_parquet, index=False)

    # Aggregate stats over valid EM/NE
    em_vals = pd.to_numeric(df[col_em], errors="coerce")
    ne_vals = pd.to_numeric(df[col_ne], errors="coerce")
    valid = em_vals.notna() & ne_vals.notna()

    summary = {
        "processed_new": processed_new,
        "failures": failures,
        "valid_items": int(valid.sum()),
        "EM_rate": float(em_vals[valid].mean()) if valid.any() else None,
        "NE_mean": float(ne_vals[valid].mean()) if valid.any() else None,
        "NE_median": float(ne_vals[valid].median()) if valid.any() else None,
        "NE_p10": float(ne_vals[valid].quantile(0.10)) if valid.any() else None,
        "NE_p90": float(ne_vals[valid].quantile(0.90)) if valid.any() else None,
        "columns_added": [col_comp, col_em, col_ne, col_smem] + ([col_comp_ctrl] if use_control else []),
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
    df = pd.read_parquet(master_path)

    model_cfg = None
    for m in cfg["models"]:
        if m["model_id"] == args.model_id:
            model_cfg = m
            break
    if model_cfg is None:
        raise ValueError(f"model_id='{args.model_id}' not found in config")

    mem_cfg = cfg["memorization"]
    decoding = mem_cfg["decoding"]
    use_control = bool(mem_cfg.get("use_control_prefix", True))
    sleep_s = float(mem_cfg["runtime"]["sleep_s"])
    save_every = int(mem_cfg["runtime"]["save_every"])

    out_parquet = format_path(mem_cfg["outputs"]["parquet"], args.model_id)
    log_path = format_path(mem_cfg["outputs"]["log_jsonl"], args.model_id)
    summary_json = format_path(mem_cfg["outputs"]["summary_json"], args.model_id)

    client = select_client(model_cfg)

    t0 = time.time()
    results = run_mem_probe(
        df=df,
        client=client,
        model_id=args.model_id,
        decoding=decoding,
        use_control=use_control,
        limit=args.limit,
        sleep_s=sleep_s,
        save_every=save_every,
        out_parquet=out_parquet,
        log_path=log_path,
    )
    elapsed_s = time.time() - t0

    summary = {
        "stage": "memorization_probe",
        "model_id": args.model_id,
        "provider": model_cfg["provider"],
        "model_name": model_cfg["model_name"],
        "dataset_path": master_path,
        "n_rows_total": int(len(df)),
        "processed_new": results["processed_new"],
        "failures": results["failures"],
        "valid_items": results["valid_items"],
        "EM_rate": results["EM_rate"],
        "NE_mean": results["NE_mean"],
        "NE_median": results["NE_median"],
        "NE_p10": results["NE_p10"],
        "NE_p90": results["NE_p90"],
        "decoding": decoding,
        "use_control_prefix": use_control,
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
    print(f"EM_rate: {results['EM_rate']}; NE_mean: {results['NE_mean']}")
    print("Output:", out_parquet)
    print("Summary:", summary_json)
    print("Log:", log_path)


if __name__ == "__main__":
    main()
