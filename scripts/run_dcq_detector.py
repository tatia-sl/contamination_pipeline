#!/usr/bin/env python3
"""
scripts/run_dcq_detector.py

DCQ Semantic Detector (SSem) runner for API-only contamination pipeline.

Features:
- Reads configs/run_config.yaml
- Supports OpenAI (gpt-4o-mini) and Gemini (gemini-1.5-flash via API key)
- --model_id selects one model per run
- --limit for pilot runs (process at most N NEW rows)
- Writes:
  - parquet with appended DCQ columns
  - jsonl log of per-item status
  - summary json (CPS + run metadata)
- Robust parsing of A/B/C/D even if output is "A.", "Answer: A", etc.

Assumptions:
- Frozen master table has columns:
  - xsum_id (string or int)
  - document_norm (or document)  -> we prefer document_norm if present
  - summary_ref_norm
  - dcq_A_canonical, dcq_B_para1, dcq_C_para2, dcq_D_para3
- Prompt template is stored in src/prompts.py as DCQ_PROMPT_TEMPLATE

Run examples:
  python3 scripts/run_dcq_detector.py --config configs/run_config.yaml --model_id gpt4omini --limit 20
  python3 scripts/run_dcq_detector.py --config configs/run_config.yaml --model_id gemini15flash --limit 20
"""

import argparse
import json
import os
import re
import sys
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Tuple, List

import pandas as pd
import yaml
import unicodedata


# Ensure project root is on sys.path for `import src.*` when running from scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prompts import DCQ_PROMPT_TEMPLATE
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
    # stable 32-bit integer derived from string
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]
    return int(h, 16)


def parse_choice_abcd(text: str) -> str:
    """
    Extract a single choice letter A/B/C/D from model output.
    Accepts outputs like:
      "A", "A.", "Answer: A", "I choose B", "(C)", "Option D"
    Returns: "A"/"B"/"C"/"D" or "" if cannot parse.
    """
    if not text:
        return ""
    t = text.strip().upper()

    # Fast path: if the first non-space char is A/B/C/D
    m = re.match(r"^\s*([ABCD])\b", t)
    if m:
        return m.group(1)

    # Common patterns
    # "ANSWER: A", "OPTION B", "CHOICE C"
    m = re.search(r"\b(ANSWER|OPTION|CHOICE)\s*[:\-]?\s*([ABCD])\b", t)
    if m:
        return m.group(2)

    # Parenthesized: "(A)"
    m = re.search(r"\(\s*([ABCD])\s*\)", t)
    if m:
        return m.group(1)

    # Any standalone letter token (avoid matching inside words)
    m = re.search(r"\b([ABCD])\b", t)
    if m:
        return m.group(1)

    return ""

def redact_secrets(text: str) -> str:
    """
    Best-effort redaction for bearer/API tokens in logs.
    """
    if not isinstance(text, str):
        return ""
    redacted = text
    redacted = re.sub(r"Bearer\s+[A-Za-z0-9_\-\.]+", "Bearer [REDACTED]", redacted)
    redacted = re.sub(r"sk-[A-Za-z0-9_\-\.]+", "sk-[REDACTED]", redacted)
    return redacted

def map_ssem_from_cps(cps: float) -> int:
    """
    Map global CPS to SSem level.
    """
    if cps < 0.35:
        return 0
    if cps < 0.45:
        return 1
    if cps < 0.60:
        return 2
    return 3


def build_dcq_prompt(document: str, A: str, B: str, C: str, D: str) -> str:
    return DCQ_PROMPT_TEMPLATE.format(DOCUMENT=document, A=A, B=B, C=C, D=D)


def get_document_field(row: pd.Series) -> str:
    # prefer normalized document if present
    for col in ["document_norm", "document", "doc_norm", "article", "text"]:
        if col in row and isinstance(row[col], str) and row[col].strip():
            return row[col]
    return ""


def resolve_prompt_ref(_ref: str) -> str:
    """
    run_config.yaml stores e.g. "src/prompts.py::DCQ_PROMPT_TEMPLATE".
    We already import DCQ_PROMPT_TEMPLATE, so this is only a sanity check hook.
    """
    return _ref


def select_client(model_cfg: Dict[str, Any]):
    provider = model_cfg["provider"].lower()
    model_name = model_cfg["model_name"]
    api_key_var = model_cfg.get("env", {}).get("api_key_var")

    if provider == "openai":
        api_key = os.environ.get(api_key_var or "OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(f"Missing {api_key_var or 'OPENAI_API_KEY'} env var")
        return OpenAIClient(api_key=api_key, model=model_name)
    if provider == "gemini":
        # Requires GEMINI_API_KEY
        from src.clients.gemini_client import GeminiClient
        return GeminiClient(model=model_name)

    raise ValueError(f"Unsupported provider: {provider}")


def format_path(template: str, model_id: str) -> str:
    return template.replace("{model_id}", model_id)

def normalize_for_api(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    # common punctuation normalization
    text = text.replace("—", "-").replace("–", "-")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    return text


# -----------------------
# Core DCQ logic
# -----------------------

def shuffled_options_for_item(
    item_key: str,
    canonical: str,
    p1: str, p2: str, p3: str,
    base_seed: int
) -> Tuple[Dict[str, str], str, List[str]]:
    """
    Returns:
      opts: dict mapping "A"/"B"/"C"/"D" -> option_text
      canonical_pos: "A"/"B"/"C"/"D"
      order: list of labels ["A","B","C","D"] describing which source went where, e.g. ["S","P1","P3","P2"]
    """
    candidates = [("S", canonical), ("P1", p1), ("P2", p2), ("P3", p3)]

    # Deterministic shuffle per item: combine base seed with item_key hash
    seed_int = base_seed ^ stable_int_from_str(item_key)
    rng = seed_int  # simple deterministic shuffle via Fisher-Yates using LCG

    def rand32(x: int) -> int:
        # LCG (Numerical Recipes) for deterministic pseudo-randomness
        return (1664525 * x + 1013904223) & 0xFFFFFFFF

    # Fisher-Yates
    arr = candidates[:]
    for j in range(len(arr) - 1, 0, -1):
        rng = rand32(rng)
        k = rng % (j + 1)
        arr[j], arr[k] = arr[k], arr[j]

    labels = ["A", "B", "C", "D"]
    opts = {labels[idx]: arr[idx][1] for idx in range(4)}
    order = [arr[idx][0] for idx in range(4)]  # which source ended up at each position

    canonical_pos = labels[order.index("S")]
    return opts, canonical_pos, order


def run_dcq(
    df: pd.DataFrame,
    client,
    model_id: str,
    decoding: Dict[str, Any],
    base_seed: int,
    limit: int,
    sleep_s: float,
    save_every: int,
    out_parquet: str,
    log_path: str
) -> Dict[str, Any]:
    """
    Adds DCQ columns:
      dcq_order_{model_id}
      dcq_canonical_pos_{model_id}
      dcq_choice_{model_id}
      dcq_win_{model_id}
      dcq_raw_{model_id}
    Computes CPS over processed + existing.
    """
    # Column names per model to keep both models in the same df if desired
    col_order = f"dcq_order_{model_id}"
    col_cpos = f"dcq_canonical_pos_{model_id}"
    col_choice = f"dcq_choice_{model_id}"
    col_win = f"dcq_win_{model_id}"
    col_raw = f"dcq_raw_{model_id}"

    # Ensure columns exist with consistent dtypes
    for c in [col_order, col_cpos, col_choice, col_raw]:
        if c not in df.columns:
            df[c] = ""
    if col_win not in df.columns or df[col_win].dtype != "Int64":
        df[col_win] = pd.Series([pd.NA] * len(df), dtype="Int64")

    processed_new = 0
    failures = 0

    # Iterate rows (resume-friendly: skip if already has a parsed choice)
    for idx, row in df.iterrows():
        #existing_choice = str(row.get(col_choice, "")).strip()
        #if existing_choice in ["A", "B", "C", "D"]:
            #continue
    # resume-friendly: skip only if model-specific DCQ already exists
        existing_choice = row.get(col_choice, "")
        if isinstance(existing_choice, str) and existing_choice.strip() in ["A", "B", "C", "D"]:
            continue


        #document = get_document_field(row)
        document = normalize_for_api(get_document_field(row))

        if not document:
            failures += 1
            log_jsonl(log_path, {
                "row": int(idx),
                "xsum_id": str(row.get("xsum_id", "")),
                "status": "error_missing_document",
            })
            continue

        canonical = row.get("dcq_A_canonical") or row.get("summary_ref_norm") or ""
        p1 = row.get("dcq_B_para1", "")
        p2 = row.get("dcq_C_para2", "")
        p3 = row.get("dcq_D_para3", "")

        if not (isinstance(canonical, str) and isinstance(p1, str) and isinstance(p2, str) and isinstance(p3, str)):
            failures += 1
            log_jsonl(log_path, {
                "row": int(idx),
                "xsum_id": str(row.get("xsum_id", "")),
                "status": "error_bad_types",
            })
            continue

        # Deterministic shuffle per item using xsum_id if present, else row index
        item_key = str(row.get("xsum_id", idx))
        opts, canonical_pos, order = shuffled_options_for_item(
            item_key=item_key,
            canonical=canonical,
            p1=p1, p2=p2, p3=p3,
            base_seed=base_seed
        )

        prompt = build_dcq_prompt(document=document, A=opts["A"], B=opts["B"], C=opts["C"], D=opts["D"])

        try:
            raw = client.generate_text(
                prompt=prompt,
                temperature=float(decoding["temperature"]),
                top_p=float(decoding["top_p"]),
                max_tokens=int(decoding["max_tokens"]),
            )
            choice = parse_choice_abcd(raw)

            if choice not in ["A", "B", "C", "D"]:
                failures += 1
                df.at[idx, col_order] = json.dumps(order, ensure_ascii=False)
                df.at[idx, col_cpos] = canonical_pos
                df.at[idx, col_choice] = ""
                df.at[idx, col_win] = pd.NA
                df.at[idx, col_raw] = raw

                log_jsonl(log_path, {
                    "row": int(idx),
                    "xsum_id": item_key,
                    "status": "parse_failed",
                    "raw": raw[:2000],
                    "canonical_pos": canonical_pos,
                    "order": order,
                })
            else:
                win = 1 if choice == canonical_pos else 0

                df.at[idx, col_order] = json.dumps(order, ensure_ascii=False)
                df.at[idx, col_cpos] = canonical_pos
                df.at[idx, col_choice] = choice
                df.at[idx, col_win] = int(win)
                df.at[idx, col_raw] = raw

                log_jsonl(log_path, {
                    "row": int(idx),
                    "xsum_id": item_key,
                    "status": "ok",
                    "choice": choice,
                    "win": win,
                    "canonical_pos": canonical_pos,
                    "order": order,
                })

                processed_new += 1

        except Exception as e:
            import traceback
            failures += 1
            tb = redact_secrets(traceback.format_exc())
            err_repr = redact_secrets(repr(e))
            log_jsonl(log_path, {
                "row": int(idx),
                "xsum_id": item_key,
                "status": "api_error",
                "error_type": type(e).__name__,
                "error_repr": err_repr,
                "traceback": tb,
            })


        if save_every and processed_new > 0 and (processed_new % save_every == 0):
            ensure_parent_dir(out_parquet)
            df.to_parquet(out_parquet, index=False)
            print(f"Saved progress: {processed_new} new rows processed -> {out_parquet}")

        if limit is not None and processed_new >= limit:
            break

        time.sleep(float(sleep_s))

    # CPS over rows with valid win (exclude parse failures)
    wins = pd.to_numeric(df[col_win], errors="coerce")
    valid = wins.notna()
    cps = float(wins[valid].mean()) if valid.any() else None
    ssem_level = map_ssem_from_cps(cps) if cps is not None else pd.NA

    # Persist SSem both in generic and model-specific columns for downstream compatibility.
    col_ssem_model = f"SSem_{model_id}"
    df["SSem"] = pd.Series([ssem_level] * len(df), dtype="Int64")
    df[col_ssem_model] = pd.Series([ssem_level] * len(df), dtype="Int64")

    # Final save (after SSem is populated)
    ensure_parent_dir(out_parquet)
    df.to_parquet(out_parquet, index=False)

    return {
        "model_id": model_id,
        "processed_new": processed_new,
        "failures": failures,
        "cps": cps,
        "ssem_level": None if cps is None else int(ssem_level),
        "valid_items_for_cps": int(valid.sum()),
        "total_rows": int(len(df)),
        "columns_added": [col_order, col_cpos, col_choice, col_win, col_raw, "SSem", col_ssem_model],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to run_config.yaml")
    parser.add_argument("--model_id", type=str, required=True, help="Model ID from run_config.yaml (e.g., gpt4omini)")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N NEW rows (pilot runs)")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    # Resolve dataset path
    master_path = cfg["project"]["frozen_master_table_path"]
    df = pd.read_parquet(master_path)

    # Find model config
    model_cfg = None
    for m in cfg["models"]:
        if m["model_id"] == args.model_id:
            model_cfg = m
            break
    if model_cfg is None:
        raise ValueError(f"model_id='{args.model_id}' not found in config")

    # Sanity check prompt ref exists in config (optional)
    _ = resolve_prompt_ref(cfg.get("prompts", {}).get("dcq_prompt_path", ""))

    # DCQ settings
    dcq_cfg = cfg["dcq"]
    decoding = dcq_cfg["decoding"]
    base_seed = int(dcq_cfg["option_shuffle_seed"])
    sleep_s = float(dcq_cfg["runtime"]["sleep_s"])
    save_every = int(dcq_cfg["runtime"]["save_every"])

    # Output paths
    out_parquet = format_path(dcq_cfg["outputs"]["parquet"], args.model_id)
    log_path = format_path(dcq_cfg["outputs"]["log_jsonl"], args.model_id)
    summary_json = format_path(dcq_cfg["outputs"]["summary_json"], args.model_id)

    # Create client
    client = select_client(model_cfg)

    # Run
    t0 = time.time()
    results = run_dcq(
        df=df,
        client=client,
        model_id=args.model_id,
        decoding=decoding,
        base_seed=base_seed,
        limit=args.limit,
        sleep_s=sleep_s,
        save_every=save_every,
        out_parquet=out_parquet,
        log_path=log_path,
    )
    elapsed_s = time.time() - t0

    # Write summary
    summary = {
        "stage": "dcq",
        "model_id": args.model_id,
        "provider": model_cfg["provider"],
        "model_name": model_cfg["model_name"],
        "dataset_path": master_path,
        "n_rows_total": results["total_rows"],
        "processed_new": results["processed_new"],
        "failures": results["failures"],
        "valid_items_for_cps": results["valid_items_for_cps"],
        "CPS": results["cps"],
        "SSem_level_from_CPS": results["ssem_level"],
        "decoding": decoding,
        "option_shuffle_seed": base_seed,
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
    print(f"CPS: {results['cps']} (valid items: {results['valid_items_for_cps']})")
    print("Output:", out_parquet)
    print("Summary:", summary_json)
    print("Log:", log_path)


if __name__ == "__main__":
    main()
