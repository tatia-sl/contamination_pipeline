#!/usr/bin/env python3
"""
scripts/run_dcq_detector.py

Bias-compensated DCQ runner:
1) BDQ  -> detect positional bias without canonical option
2) BCQ  -> place canonical into one least-preferred position
3) CPS  -> computed from BCQ wins
4) SSem -> mapped from BCQ CPS

Assumptions:
- Frozen table has columns:
  - xsum_id
  - document_norm (or document)
  - summary_ref_norm
  - dcq_A_canonical
  - dcq_B_para1
  - dcq_C_para2
  - dcq_D_para3
  - dcq_E_para4
- Prompt template in src/prompts.py supports A-E.
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import traceback
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

# Ensure project root is on sys.path for `import src.*` when running from scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clients.openai_client import OpenAIClient
from src.prompts import DCQ_PROMPT_TEMPLATE


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



def parse_choice_abcde(text: str) -> str:
    """
    Extract a single choice letter A/B/C/D/E from model output.
    """
    if not text:
        return ""

    t = text.strip().upper()

    m = re.match(r"^\s*([ABCDE])\b", t)
    if m:
        return m.group(1)

    patterns = [
        r"\b(ANSWER|OPTION|CHOICE)\s*[:\-]?\s*([ABCDE])\b",
        r"\(\s*([ABCDE])\s*\)",
        r"\b([ABCDE])\b",
    ]
    for pat in patterns:
        m = re.search(pat, t)
        if m:
            return m.group(2) if len(m.groups()) > 1 else m.group(1)

    return ""



def valid_choice_set() -> set[str]:
    return {"A", "B", "C", "D", "E"}



def redact_secrets(text: str) -> str:
    if not isinstance(text, str):
        return ""
    redacted = text
    redacted = re.sub(r"Bearer\s+[A-Za-z0-9_\-\.]+", "Bearer [REDACTED]", redacted)
    redacted = re.sub(r"sk-[A-Za-z0-9_\-\.]+", "sk-[REDACTED]", redacted)
    return redacted



def map_ssem_from_cps(cps: float) -> int:
    if cps < 0.35:
        return 0
    if cps < 0.45:
        return 1
    if cps < 0.60:
        return 2
    return 3



def build_quiz_prompt(document: str, A: str, B: str, C: str, D: str, E: str) -> str:
    return DCQ_PROMPT_TEMPLATE.format(DOCUMENT=document, A=A, B=B, C=C, D=D, E=E)



def build_quiz_retry_prompt(base_prompt: str) -> str:
    return base_prompt + "\n\nIMPORTANT: Reply with exactly one letter only: A, B, C, D, or E."



def get_document_field(row: pd.Series) -> str:
    for col in ["document_norm", "document", "doc_norm", "article", "text"]:
        if col in row and isinstance(row[col], str) and row[col].strip():
            return row[col]
    return ""



def select_client(model_cfg: Dict[str, Any]):
    provider = str(model_cfg.get("provider", "")).lower()
    model_name = model_cfg["model_name"]
    api_key_var = model_cfg.get("env", {}).get("api_key_var")

    if provider == "openai":
        api_key = os.environ.get(api_key_var or "OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(f"Missing {api_key_var or 'OPENAI_API_KEY'} env var")
        return OpenAIClient(api_key=api_key, model=model_name)

    if provider == "gemini":
        from src.clients.gemini_client import GeminiClient

        return GeminiClient(model=model_name)

    raise ValueError(f"Unsupported provider: {provider}")



def format_path(template: str, model_id: str) -> str:
    return template.replace("{model_id}", model_id)



def normalize_for_api(text: Any) -> str:
    if text is None:
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    s = s.replace("—", "-").replace("–", "-")
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("‘", "'").replace("’", "'")
    return s.strip()

def validate_dcq_input_table(df: pd.DataFrame, dataset_path: str) -> None:
    """
    Validate that the input table is compatible with BDQ/BCQ mode.
    BDQ requires four paraphrases (B/C/D/E) and therefore needs dcq_E_para4.
    """
    required_cols = [
        "xsum_id",
        "summary_ref_norm",
        "dcq_A_canonical",
        "dcq_B_para1",
        "dcq_C_para2",
        "dcq_D_para3",
        "dcq_E_para4",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            "BDQ/BCQ input schema is incomplete. Missing columns: "
            + ", ".join(missing)
            + f". Dataset: {dataset_path}"
        )

    p4_filled = int(df["dcq_E_para4"].astype(str).str.strip().ne("").sum())
    if p4_filled == 0:
        raise RuntimeError(
            "Column dcq_E_para4 is empty for all rows. "
            "BDQ/BCQ mode requires 4 paraphrases (B/C/D/E). "
            f"Dataset: {dataset_path}. "
            "Use a dcq4 frozen table in run_config.yaml."
        )



def call_model_text(client, prompt: str, decoding: Dict[str, Any]) -> str:
    return client.generate_text(
        prompt=prompt,
        temperature=float(decoding["temperature"]),
        top_p=float(decoding["top_p"]),
        max_tokens=int(decoding["max_tokens"]),
    )



def execute_quiz_item(client, prompt: str, decoding: Dict[str, Any]) -> Dict[str, Any]:
    raw = call_model_text(client=client, prompt=prompt, decoding=decoding)
    choice = parse_choice_abcde(raw)

    if choice in valid_choice_set():
        return {
            "status": "ok",
            "choice": choice,
            "raw": raw,
            "raw_retry": "",
            "retried": False,
        }

    retry_prompt = build_quiz_retry_prompt(prompt)
    raw_retry = call_model_text(client=client, prompt=retry_prompt, decoding=decoding)
    choice_retry = parse_choice_abcde(raw_retry)

    if choice_retry in valid_choice_set():
        return {
            "status": "ok_after_retry",
            "choice": choice_retry,
            "raw": raw,
            "raw_retry": raw_retry,
            "retried": True,
        }

    return {
        "status": "parse_failed",
        "choice": "",
        "raw": raw,
        "raw_retry": raw_retry,
        "retried": True,
    }


# -----------------------
# Option builders
# -----------------------


def get_item_material(row: pd.Series) -> Dict[str, str]:
    canonical = normalize_for_api(row.get("dcq_A_canonical") or row.get("summary_ref_norm") or "")
    return {
        "canonical": canonical,
        "P1": normalize_for_api(row.get("dcq_B_para1", "")),
        "P2": normalize_for_api(row.get("dcq_C_para2", "")),
        "P3": normalize_for_api(row.get("dcq_D_para3", "")),
        "P4": normalize_for_api(row.get("dcq_E_para4", "")),
    }



def build_bdq_options_for_item(
    item_key: str,
    paraphrases: List[Tuple[str, str]],
    base_seed: int,
) -> Tuple[Dict[str, str], List[str]]:
    """
    BDQ:
      A-D = all 4 paraphrases (shuffled)
      E   = None option
    """
    rng = random.Random(base_seed + stable_int_from_str(f"BDQ::{item_key}"))
    arr = paraphrases[:]
    rng.shuffle(arr)

    order = [label for label, _ in arr]
    opts = {
        "A": arr[0][1],
        "B": arr[1][1],
        "C": arr[2][1],
        "D": arr[3][1],
        "E": "None of the provided options.",
    }
    return opts, order



def choose_three_paraphrases(
    item_key: str,
    paraphrases: List[Tuple[str, str]],
    base_seed: int,
) -> List[Tuple[str, str]]:
    rng = random.Random(base_seed + stable_int_from_str(f"BCQ_PICK::{item_key}"))
    arr = paraphrases[:]
    rng.shuffle(arr)
    return arr[:3]



def build_bcq_options_for_item(
    item_key: str,
    canonical: str,
    paraphrases: List[Tuple[str, str]],
    base_seed: int,
    canonical_target_pos: str,
) -> Tuple[Dict[str, str], List[str], str]:
    """
    BCQ:
      canonical forced into least-preferred position among A-D
      remaining A-D are 3 of 4 paraphrases
      E is none-option
    """
    chosen = choose_three_paraphrases(item_key=item_key, paraphrases=paraphrases, base_seed=base_seed)

    rng = random.Random(base_seed + stable_int_from_str(f"BCQ_FILL::{item_key}::{canonical_target_pos}"))
    shuffled = chosen[:]
    rng.shuffle(shuffled)

    slots = ["A", "B", "C", "D"]
    rem = [s for s in slots if s != canonical_target_pos]

    opts: Dict[str, str] = {canonical_target_pos: canonical}
    order_map: Dict[str, str] = {canonical_target_pos: "CANONICAL"}

    for slot, (label, text) in zip(rem, shuffled):
        opts[slot] = text
        order_map[slot] = label

    opts["E"] = "None of the provided options."
    order = [order_map["A"], order_map["B"], order_map["C"], order_map["D"]]
    return opts, order, canonical_target_pos


# -----------------------
# DataFrame helpers
# -----------------------


def ensure_text_column(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        df[col] = ""



def ensure_int_column(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        df[col] = pd.Series([pd.NA] * len(df), dtype="Int64")
    else:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")



def ensure_bool_column(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        df[col] = pd.Series([pd.NA] * len(df), dtype="boolean")
    else:
        df[col] = df[col].astype("boolean")


# -----------------------
# Stage 1: BDQ
# -----------------------


def run_bdq(
    df: pd.DataFrame,
    client,
    model_id: str,
    decoding: Dict[str, Any],
    base_seed: int,
    limit: Optional[int],
    sleep_s: float,
    save_every: int,
    out_parquet: str,
    log_path: str,
) -> Dict[str, Any]:
    col_order = f"bdq_order_{model_id}"
    col_choice = f"bdq_choice_{model_id}"
    col_raw = f"bdq_raw_{model_id}"
    col_raw_retry = f"bdq_raw_retry_{model_id}"
    col_retried = f"bdq_retried_{model_id}"
    col_status = f"bdq_status_{model_id}"

    for c in [col_order, col_choice, col_raw, col_raw_retry, col_status]:
        ensure_text_column(df, c)
    ensure_bool_column(df, col_retried)

    processed_new = 0
    failures = 0
    scanned_new = 0

    for idx, row in df.iterrows():
        if str(row.get(col_choice, "")).strip() in valid_choice_set():
            continue

        scanned_new += 1
        item_key = str(row.get("xsum_id", idx))
        document = normalize_for_api(get_document_field(row))
        mat = get_item_material(row)
        paraphrases = [("P1", mat["P1"]), ("P2", mat["P2"]), ("P3", mat["P3"]), ("P4", mat["P4"])]

        if not document or any(not txt for _, txt in paraphrases):
            failures += 1
            df.at[idx, col_status] = "missing_input"
            log_jsonl(log_path, {"stage": "bdq", "row": int(idx), "xsum_id": item_key, "status": "missing_input"})
            continue

        opts, order = build_bdq_options_for_item(item_key=item_key, paraphrases=paraphrases, base_seed=base_seed)
        prompt = build_quiz_prompt(document=document, **opts)

        try:
            res = execute_quiz_item(client=client, prompt=prompt, decoding=decoding)
            df.at[idx, col_order] = json.dumps(order, ensure_ascii=False)
            df.at[idx, col_choice] = res["choice"]
            df.at[idx, col_raw] = res["raw"]
            df.at[idx, col_raw_retry] = res["raw_retry"]
            df.at[idx, col_retried] = bool(res["retried"])
            df.at[idx, col_status] = res["status"]

            if res["status"] == "parse_failed":
                failures += 1

            processed_new += 1
            log_jsonl(
                log_path,
                {
                    "stage": "bdq",
                    "row": int(idx),
                    "xsum_id": item_key,
                    "status": res["status"],
                    "choice": res["choice"],
                    "retried": bool(res["retried"]),
                    "order": order,
                },
            )
        except Exception as e:  # noqa: BLE001
            failures += 1
            tb = redact_secrets(traceback.format_exc())
            err = redact_secrets(repr(e))
            df.at[idx, col_status] = "api_error"
            log_jsonl(
                log_path,
                {
                    "stage": "bdq",
                    "row": int(idx),
                    "xsum_id": item_key,
                    "status": "api_error",
                    "error_type": type(e).__name__,
                    "error_repr": err,
                    "traceback": tb,
                    "client_meta": getattr(client, "last_response_meta", None),
                },
            )

        if save_every and scanned_new > 0 and (scanned_new % save_every == 0):
            ensure_parent_dir(out_parquet)
            df.to_parquet(out_parquet, index=False)
            print(
                f"Saved BDQ progress: scanned={scanned_new}, "
                f"processed={processed_new}, failures={failures} -> {out_parquet}"
            )

        if limit is not None and scanned_new >= limit:
            break

        if sleep_s > 0:
            time.sleep(sleep_s)

    ensure_parent_dir(out_parquet)
    df.to_parquet(out_parquet, index=False)

    valid = df[col_choice].astype(str).str.strip()
    valid = valid[valid.isin(["A", "B", "C", "D", "E"])]
    counts = {k: int((valid == k).sum()) for k in ["A", "B", "C", "D", "E"]}

    return {
        "scanned_new": scanned_new,
        "processed_new": processed_new,
        "failures": failures,
        "valid_items": int(len(valid)),
        "choice_counts": counts,
    }


# -----------------------
# Bias analysis
# -----------------------


def identify_non_preferred_position(df: pd.DataFrame, model_id: str) -> Dict[str, Any]:
    col_choice = f"bdq_choice_{model_id}"

    valid = df[col_choice].astype(str).str.strip()
    valid = valid[valid.isin(["A", "B", "C", "D", "E"])]

    counts = {k: int((valid == k).sum()) for k in ["A", "B", "C", "D", "E"]}
    n = int(len(valid))

    least = min(["A", "B", "C", "D"], key=lambda x: (counts[x], x))
    rates = {k: (counts[k] / n if n else 0.0) for k in ["A", "B", "C", "D", "E"]}

    return {
        "bdq_counts": counts,
        "bdq_rates": rates,
        "valid_items": n,
        "least_preferred_position": least,
        "least_preferred_rate": rates[least],
    }


# -----------------------
# Stage 2: BCQ
# -----------------------


def run_single_bcq(
    df: pd.DataFrame,
    client,
    model_id: str,
    decoding: Dict[str, Any],
    base_seed: int,
    canonical_target_pos: str,
    limit: Optional[int],
    sleep_s: float,
    save_every: int,
    out_parquet: str,
    log_path: str,
) -> Dict[str, Any]:
    col_target = f"bcq_target_pos_{model_id}"
    col_order = f"bcq_order_{model_id}"
    col_cpos = f"bcq_canonical_pos_{model_id}"
    col_choice = f"bcq_choice_{model_id}"
    col_win = f"bcq_win_{model_id}"
    col_raw = f"bcq_raw_{model_id}"
    col_raw_retry = f"bcq_raw_retry_{model_id}"
    col_retried = f"bcq_retried_{model_id}"
    col_status = f"bcq_status_{model_id}"

    for c in [col_target, col_order, col_cpos, col_choice, col_raw, col_raw_retry, col_status]:
        ensure_text_column(df, c)
    ensure_int_column(df, col_win)
    ensure_bool_column(df, col_retried)

    processed_new = 0
    failures = 0
    scanned_new = 0

    for idx, row in df.iterrows():
        if str(row.get(col_choice, "")).strip() in valid_choice_set():
            continue

        scanned_new += 1
        item_key = str(row.get("xsum_id", idx))
        document = normalize_for_api(get_document_field(row))
        mat = get_item_material(row)
        canonical = mat["canonical"]
        paraphrases = [("P1", mat["P1"]), ("P2", mat["P2"]), ("P3", mat["P3"]), ("P4", mat["P4"])]

        if not document or not canonical or any(not txt for _, txt in paraphrases):
            failures += 1
            df.at[idx, col_status] = "missing_input"
            log_jsonl(
                log_path,
                {
                    "stage": "bcq",
                    "row": int(idx),
                    "xsum_id": item_key,
                    "status": "missing_input",
                    "target_pos": canonical_target_pos,
                },
            )
            continue

        opts, order, canonical_pos = build_bcq_options_for_item(
            item_key=item_key,
            canonical=canonical,
            paraphrases=paraphrases,
            base_seed=base_seed,
            canonical_target_pos=canonical_target_pos,
        )
        prompt = build_quiz_prompt(document=document, **opts)

        try:
            res = execute_quiz_item(client=client, prompt=prompt, decoding=decoding)
            choice = res["choice"]
            win = int(choice == canonical_pos) if choice else pd.NA

            df.at[idx, col_target] = canonical_target_pos
            df.at[idx, col_order] = json.dumps(order, ensure_ascii=False)
            df.at[idx, col_cpos] = canonical_pos
            df.at[idx, col_choice] = choice
            df.at[idx, col_win] = win
            df.at[idx, col_raw] = res["raw"]
            df.at[idx, col_raw_retry] = res["raw_retry"]
            df.at[idx, col_retried] = bool(res["retried"])
            df.at[idx, col_status] = res["status"]

            if res["status"] == "parse_failed":
                failures += 1

            processed_new += 1
            log_jsonl(
                log_path,
                {
                    "stage": "bcq",
                    "row": int(idx),
                    "xsum_id": item_key,
                    "status": res["status"],
                    "choice": choice,
                    "win": None if win is pd.NA else int(win),
                    "retried": bool(res["retried"]),
                    "target_pos": canonical_target_pos,
                    "canonical_pos": canonical_pos,
                    "order": order,
                },
            )
        except Exception as e:  # noqa: BLE001
            failures += 1
            tb = redact_secrets(traceback.format_exc())
            err = redact_secrets(repr(e))
            df.at[idx, col_status] = "api_error"
            log_jsonl(
                log_path,
                {
                    "stage": "bcq",
                    "row": int(idx),
                    "xsum_id": item_key,
                    "status": "api_error",
                    "error_type": type(e).__name__,
                    "error_repr": err,
                    "traceback": tb,
                    "client_meta": getattr(client, "last_response_meta", None),
                },
            )

        if save_every and scanned_new > 0 and (scanned_new % save_every == 0):
            ensure_parent_dir(out_parquet)
            df.to_parquet(out_parquet, index=False)
            print(
                f"Saved BCQ progress: scanned={scanned_new}, "
                f"processed={processed_new}, failures={failures} -> {out_parquet}"
            )

        if limit is not None and scanned_new >= limit:
            break

        if sleep_s > 0:
            time.sleep(sleep_s)

    ensure_parent_dir(out_parquet)
    df.to_parquet(out_parquet, index=False)

    choices = df[col_choice].astype(str).str.strip()
    valid_mask = choices.isin(["A", "B", "C", "D", "E"])
    wins = pd.to_numeric(df.loc[valid_mask, col_win], errors="coerce").dropna()

    return {
        "scanned_new": scanned_new,
        "processed_new": processed_new,
        "failures": failures,
        "valid_items": int(len(wins)),
        "bcq_cps": float(wins.mean()) if len(wins) else 0.0,
        "target_pos": canonical_target_pos,
    }


# -----------------------
# Finalization
# -----------------------


def write_legacy_dcq_aliases(df: pd.DataFrame, model_id: str) -> pd.DataFrame:
    """
    Keep downstream compatibility by mirroring BCQ outputs into legacy dcq_* columns.
    """
    bcq_order = f"bcq_order_{model_id}"
    bcq_cpos = f"bcq_canonical_pos_{model_id}"
    bcq_choice = f"bcq_choice_{model_id}"
    bcq_win = f"bcq_win_{model_id}"
    bcq_raw = f"bcq_raw_{model_id}"

    dcq_order = f"dcq_order_{model_id}"
    dcq_cpos = f"dcq_canonical_pos_{model_id}"
    dcq_choice = f"dcq_choice_{model_id}"
    dcq_win = f"dcq_win_{model_id}"
    dcq_raw = f"dcq_raw_{model_id}"

    for c in [dcq_order, dcq_cpos, dcq_choice, dcq_raw]:
        ensure_text_column(df, c)
    ensure_int_column(df, dcq_win)

    df[dcq_order] = df[bcq_order]
    df[dcq_cpos] = df[bcq_cpos]
    df[dcq_choice] = df[bcq_choice]
    df[dcq_win] = pd.to_numeric(df[bcq_win], errors="coerce").astype("Int64")
    df[dcq_raw] = df[bcq_raw]
    return df



def aggregate_results(df: pd.DataFrame, model_id: str, least_pos: str, bdq_counts: Dict[str, int]) -> Dict[str, Any]:
    col_choice = f"bcq_choice_{model_id}"
    col_win = f"bcq_win_{model_id}"

    valid_choices = df[col_choice].astype(str).str.strip()
    valid_mask = valid_choices.isin(["A", "B", "C", "D", "E"])
    wins = pd.to_numeric(df.loc[valid_mask, col_win], errors="coerce").dropna()

    bcq_valid_items = int(len(wins))
    cps = float(wins.mean()) if bcq_valid_items else None
    ssem = map_ssem_from_cps(cps) if cps is not None else pd.NA

    bdq_valid_items = int(sum(bdq_counts.values()))
    pe = (bdq_counts.get(least_pos, 0) / bdq_valid_items) if bdq_valid_items else 0.0
    po = cps if cps is not None else 0.0
    kappa_min = ((po - pe) / (1 - pe)) if bdq_valid_items and pe < 1 else None

    return {
        "CPS": cps,
        "SSem_level_from_CPS": None if pd.isna(ssem) else int(ssem),
        "least_preferred_position": least_pos,
        "bdq_valid_items": bdq_valid_items,
        "bdq_counts": bdq_counts,
        "bdq_rates": {
            k: (bdq_counts[k] / bdq_valid_items if bdq_valid_items else 0.0)
            for k in ["A", "B", "C", "D", "E"]
        },
        "bcq_valid_items": bcq_valid_items,
        "kappa_min": kappa_min,
    }



def main() -> None:
    parser = argparse.ArgumentParser(description="Bias-compensated DCQ runner (BDQ + one-position BCQ)")
    parser.add_argument("--config", type=str, required=True, help="Path to run_config.yaml")
    parser.add_argument("--model_id", type=str, required=True, help="Model ID from run_config.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N NEW rows per stage")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    master_path = cfg["project"]["frozen_master_table_path"]
    dcq_cfg = cfg["dcq"]

    model_cfg = None
    for m in cfg["models"]:
        if m["model_id"] == args.model_id:
            model_cfg = m
            break
    if model_cfg is None:
        raise ValueError(f"model_id='{args.model_id}' not found in config")

    decoding = dcq_cfg["decoding"]
    base_seed = int(dcq_cfg["option_shuffle_seed"])
    sleep_s = float(dcq_cfg["runtime"]["sleep_s"])
    save_every = int(dcq_cfg["runtime"]["save_every"])

    out_parquet = format_path(dcq_cfg["outputs"]["parquet"], args.model_id)
    log_path = format_path(dcq_cfg["outputs"]["log_jsonl"], args.model_id)
    summary_json = format_path(dcq_cfg["outputs"]["summary_json"], args.model_id)

    if Path(out_parquet).exists():
        df = pd.read_parquet(out_parquet)
        print(f"Resuming from existing output: {out_parquet}")
    else:
        df = pd.read_parquet(master_path)
        print(f"Starting from master table: {master_path}")

    # Stop early if dataset is not compatible with BDQ/BCQ prerequisites.
    validate_dcq_input_table(df=df, dataset_path=master_path)

    client = select_client(model_cfg)

    t0 = time.time()

    bdq = run_bdq(
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
    print(
        f"BDQ stage done: scanned={bdq['scanned_new']}, "
        f"processed={bdq['processed_new']}, failures={bdq['failures']}, "
        f"valid_items={bdq['valid_items']}"
    )

    df = pd.read_parquet(out_parquet)

    bias = identify_non_preferred_position(df=df, model_id=args.model_id)
    least_pos = bias["least_preferred_position"]

    bcq = run_single_bcq(
        df=df,
        client=client,
        model_id=args.model_id,
        decoding=decoding,
        base_seed=base_seed,
        canonical_target_pos=least_pos,
        limit=args.limit,
        sleep_s=sleep_s,
        save_every=save_every,
        out_parquet=out_parquet,
        log_path=log_path,
    )
    print(
        f"BCQ stage done: scanned={bcq['scanned_new']}, "
        f"processed={bcq['processed_new']}, failures={bcq['failures']}, "
        f"valid_items={bcq['valid_items']}"
    )

    df = pd.read_parquet(out_parquet)

    final = aggregate_results(
        df=df,
        model_id=args.model_id,
        least_pos=least_pos,
        bdq_counts=bias["bdq_counts"],
    )

    ssem_level = final["SSem_level_from_CPS"]
    df["SSem"] = pd.Series([ssem_level] * len(df), dtype="Int64")
    df[f"SSem_{args.model_id}"] = pd.Series([ssem_level] * len(df), dtype="Int64")

    df = write_legacy_dcq_aliases(df=df, model_id=args.model_id)

    ensure_parent_dir(out_parquet)
    df.to_parquet(out_parquet, index=False)

    elapsed_s = time.time() - t0

    summary = {
        "stage": "dcq",
        "mode": "bias_compensated_light",
        "model_id": args.model_id,
        "provider": model_cfg["provider"],
        "model_name": model_cfg["model_name"],
        "dataset_path": master_path,
        "n_rows_total": int(len(df)),
        "processed_new": int(bdq["processed_new"] + bcq["processed_new"]),
        "failures": int(bdq["failures"] + bcq["failures"]),
        "valid_items_for_cps": final["bcq_valid_items"],
        "CPS": final["CPS"],
        "SSem_level_from_CPS": final["SSem_level_from_CPS"],
        "decoding": decoding,
        "option_shuffle_seed": base_seed,
        "elapsed_seconds": elapsed_s,
        "out_parquet": out_parquet,
        "log_jsonl": log_path,
        "bdq": bdq,
        "bias_info": bias,
        "bcq": bcq,
        "least_preferred_position": final["least_preferred_position"],
        "bdq_valid_items": final["bdq_valid_items"],
        "bdq_counts": final["bdq_counts"],
        "bdq_rates": final["bdq_rates"],
        "bcq_valid_items": final["bcq_valid_items"],
        "kappa_min": final["kappa_min"],
    }

    ensure_parent_dir(summary_json)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Model: {args.model_id} ({model_cfg['model_name']})")
    print(f"Least preferred position (BDQ): {final['least_preferred_position']}")
    print(f"BCQ CPS: {final['CPS']} (valid items: {final['bcq_valid_items']})")
    print(f"SSem: {final['SSem_level_from_CPS']}")
    print(f"kappa_min: {final['kappa_min']}")
    print("Output:", out_parquet)
    print("Summary:", summary_json)
    print("Log:", log_path)


if __name__ == "__main__":
    main()
