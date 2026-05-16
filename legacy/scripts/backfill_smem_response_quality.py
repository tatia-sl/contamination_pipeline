#!/usr/bin/env python3
"""
Offline backfill for SMem response-quality diagnostics.

Reads existing v5_mem JSONL logs, classifies `completion_norm` with the same
classifier used by run_mem_probe.py, and writes these parquet columns:
  - response_type_{model_id}
  - is_refusal_{model_id}
  - is_valid_completion_{model_id}

No model/API calls are made.
"""

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def template_path(path: str, model_id: str) -> str:
    return str(path).replace("{model_id}", model_id)


def mem_outputs(cfg: Dict[str, Any], model_id: str) -> Dict[str, str]:
    mem_cfg = cfg.get("memorization") or cfg.get("mem") or {}
    outputs = mem_cfg.get("outputs", {}) if isinstance(mem_cfg, dict) else {}
    return {
        "parquet": template_path(outputs.get("parquet", "runs/v5_mem_{model_id}.parquet"), model_id),
        "log_jsonl": template_path(outputs.get("log_jsonl", "logs/v5_mem_{model_id}.jsonl"), model_id),
        "summary_json": template_path(
            outputs.get("summary_json") or outputs.get("summary") or "outputs/v5_mem_summary_{model_id}.json",
            model_id,
        ),
    }


def load_classifier():
    path = Path(__file__).resolve().parent / "run_mem_probe.py"
    spec = importlib.util.spec_from_file_location("run_mem_probe_for_backfill", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load classifier from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.classify_response


def read_reference_log(log_path: str, classify_response) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    with open(log_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("pass") != "reference" or rec.get("status") != "ok":
                continue
            item_id = rec.get("xsum_id")
            completion_norm = rec.get("completion_norm")
            if item_id is None or completion_norm is None:
                continue
            resp = classify_response(completion_norm)
            records[str(item_id)] = {
                "response_type": resp["response_type"],
                "is_refusal": int(bool(resp["is_refusal"])),
                "is_valid_completion": int(bool(resp["is_valid_completion"])),
                "source_line": line_no,
            }
    return records


def interpretability_from_refusal_rate(refusal_rate: Optional[float]) -> str:
    if refusal_rate is None:
        return "UNKNOWN"
    if refusal_rate >= 0.50:
        return "LOW"
    if refusal_rate >= 0.25:
        return "MODERATE"
    return "HIGH"


def update_summary(path: str, df: pd.DataFrame, model_id: str, matched: int, unmatched: int) -> None:
    summary: Dict[str, Any] = {}
    p = Path(path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            summary = json.load(f)

    col_refusal = f"is_refusal_{model_id}"
    col_valid = f"is_valid_completion_{model_id}"
    col_rtype = f"response_type_{model_id}"

    n_total = len(df)
    n_refusal = int(pd.to_numeric(df[col_refusal], errors="coerce").fillna(0).sum())
    n_valid = int(pd.to_numeric(df[col_valid], errors="coerce").fillna(0).sum())
    refusal_rate = round(n_refusal / n_total, 4) if n_total else None
    valid_completion_rate = round(n_valid / n_total, 4) if n_total else None

    refusal_breakdown = (
        df[pd.to_numeric(df[col_refusal], errors="coerce").fillna(0).astype(int) == 1][col_rtype]
        .dropna()
        .value_counts()
        .to_dict()
    )

    summary.update({
        "refusal_rate": refusal_rate,
        "valid_completion_rate": valid_completion_rate,
        "refusal_breakdown": refusal_breakdown,
        "SMem_interpretability": interpretability_from_refusal_rate(refusal_rate),
        "response_quality_backfill": {
            "status": "done",
            "matched_reference_log_rows": matched,
            "unmatched_parquet_rows": unmatched,
            "note": "Offline backfill from existing v5_mem JSONL logs; no API calls.",
        },
    })

    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill SMem refusal/valid-completion diagnostics from existing logs.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--parquet", default="")
    ap.add_argument("--log_jsonl", default="")
    ap.add_argument("--summary_json", default="")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    outputs = mem_outputs(cfg, args.model_id)
    parquet_path = args.parquet or outputs["parquet"]
    log_path = args.log_jsonl or outputs["log_jsonl"]
    summary_path = args.summary_json or outputs["summary_json"]

    if not Path(parquet_path).exists():
        raise FileNotFoundError(f"Missing parquet: {parquet_path}")
    if not Path(log_path).exists():
        raise FileNotFoundError(f"Missing JSONL log: {log_path}")

    classify_response = load_classifier()
    records = read_reference_log(log_path, classify_response)
    if not records:
        raise RuntimeError(f"No reference completion_norm records found in {log_path}")

    df = pd.read_parquet(parquet_path)
    if "xsum_id" not in df.columns:
        raise KeyError("Parquet must contain xsum_id column")

    col_rtype = f"response_type_{args.model_id}"
    col_refusal = f"is_refusal_{args.model_id}"
    col_valid = f"is_valid_completion_{args.model_id}"
    for col in (col_rtype, col_refusal, col_valid):
        if col not in df.columns:
            df[col] = pd.NA

    matched = 0
    for idx, row in df.iterrows():
        rec = records.get(str(row.get("xsum_id")))
        if not rec:
            continue
        df.at[idx, col_rtype] = rec["response_type"]
        df.at[idx, col_refusal] = rec["is_refusal"]
        df.at[idx, col_valid] = rec["is_valid_completion"]
        matched += 1

    df[col_refusal] = pd.to_numeric(df[col_refusal], errors="coerce").astype("Int64")
    df[col_valid] = pd.to_numeric(df[col_valid], errors="coerce").astype("Int64")
    unmatched = int(len(df) - matched)

    if not args.dry_run:
        df.to_parquet(parquet_path, index=False)
        update_summary(summary_path, df, args.model_id, matched, unmatched)

    print(json.dumps({
        "status": "dry_run" if args.dry_run else "ok",
        "model_id": args.model_id,
        "parquet": parquet_path,
        "log_jsonl": log_path,
        "summary_json": summary_path,
        "reference_log_records": len(records),
        "matched_rows": matched,
        "unmatched_rows": unmatched,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
