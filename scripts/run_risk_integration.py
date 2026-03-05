#!/usr/bin/env python3
"""
scripts/run_risk_integration.py

Step E — Evidence Integration & Risk (per model)

Combines:
- SLex (from runs/v3_lexical.parquet)  [model-agnostic exposure context]
- SSem (from runs/v4_dcq_{model_id}.parquet)
- SMem (from runs/v5_mem_{model_id}.parquet)
- SProb (from runs/v6_stability_{model_id}.parquet)

Outputs (per model):
- runs/v7_risk_{model_id}.parquet
- runs/v7_risk_{model_id}.csv
- outputs/v7_risk_summary_{model_id}.json
- logs/v7_risk_{model_id}.jsonl

Design principles:
- SLex is treated as exposure context (not direct contamination evidence).
- RiskScore uses ONLY model-dependent signals: SSem, SMem, SProb (weighted sum).
- SLex is used as a gating factor for RiskLevel interpretation.

Default integration (can be adjusted in one place):
- weights: w_sem=0.4, w_mem=0.4, w_prob=0.2
- gating: if SLex<=1 => RiskLevel=Low regardless of RiskScore
- mapping (when SLex>=2):
    RiskScore < 1.0 -> Low
    1.0 <= RiskScore < 2.0 -> Medium
    RiskScore >= 2.0 -> High
- Confidence = (# of strong signals >=2 among {SSem,SMem,SProb}) / 3
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd
import yaml


# -----------------------
# Utilities
# -----------------------

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def write_json(path: str, payload: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def log_jsonl(path: str, payload: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

def format_path(template: str, model_id: str) -> str:
    return template.replace("{model_id}", model_id)

def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


# -----------------------
# Risk integration rules
# -----------------------

def compute_risk_score(
    ssem: float,
    smem: float,
    sprob: float,
    w_sem: float,
    w_mem: float,
    w_prob: float
) -> float:
    # Missing signals treated conservatively as 0
    ssem = 0.0 if pd.isna(ssem) else float(ssem)
    smem = 0.0 if pd.isna(smem) else float(smem)
    sprob = 0.0 if pd.isna(sprob) else float(sprob)
    return (w_sem * ssem) + (w_mem * smem) + (w_prob * sprob)

def compute_confidence(ssem: float, smem: float, sprob: float) -> float:
    vals = [
        0.0 if pd.isna(ssem) else float(ssem),
        0.0 if pd.isna(smem) else float(smem),
        0.0 if pd.isna(sprob) else float(sprob),
    ]
    strong = sum(1 for v in vals if v >= 2.0)
    return strong / 3.0

def map_risk_level(slex: float, risk_score: float) -> str:
    """
    Exposure gating:
      if SLex <= 1 => Low
      else use RiskScore thresholds
    """
    slex = 0.0 if pd.isna(slex) else float(slex)
    risk_score = 0.0 if pd.isna(risk_score) else float(risk_score)

    if slex <= 1.0:
        return "Low"
    # slex >= 2
    if risk_score < 1.0:
        return "Low"
    if risk_score < 2.0:
        return "Medium"
    return "High"


# -----------------------
# Loading + merging
# -----------------------

def load_and_select(path: str, cols_keep: List[str], stage: str) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"[{stage}] Missing file: {path}")
    df = pd.read_parquet(path)
    missing = [c for c in cols_keep if c not in df.columns]
    if missing:
        raise ValueError(f"[{stage}] Missing columns in {path}: {missing}")
    return df[cols_keep].copy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to configs/run_config.yaml")
    parser.add_argument("--model_id", type=str, required=True, help="Model ID (e.g., gpt4omini, gemini15flash)")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    model_id = args.model_id

    # Paths (align with your actual filenames)
    # Lexical is model-agnostic
    lexical_path = "runs/v3_lexical.parquet"

    # Model-specific runs (from previous steps)
    dcq_path = f"runs/v4_dcq_{model_id}.parquet"
    mem_path = f"runs/v5_mem_{model_id}.parquet"
    stab_path = f"runs/v6_stability_{model_id}.parquet"

    out_parquet = f"runs/v7_risk_{model_id}.parquet"
    out_csv = f"runs/v7_risk_{model_id}.csv"
    out_summary = f"outputs/v7_risk_summary_{model_id}.json"
    out_log = f"logs/v7_risk_{model_id}.jsonl"

    # Weights (single place to adjust)
    w_sem = float(cfg.get("risk_integration", {}).get("weights", {}).get("w_sem", 0.4))
    w_mem = float(cfg.get("risk_integration", {}).get("weights", {}).get("w_mem", 0.4))
    w_prob = float(cfg.get("risk_integration", {}).get("weights", {}).get("w_prob", 0.2))

    # --------
    # Load minimal columns from each stage
    # --------
    df_lex = load_and_select(
        lexical_path,
        cols_keep=["xsum_id", "MaxSpanLen", "NgramHits", "ProxyCount", "SLex"],
        stage="SLex"
    )

    # DCQ / Semantic (SSem)
    # Prefer model-specific columns if present; fall back to base columns.
    df_dcq_full = pd.read_parquet(dcq_path)
    ssem_col = f"SSem_{model_id}" if f"SSem_{model_id}" in df_dcq_full.columns else ("SSem" if "SSem" in df_dcq_full.columns else None)
    cps_col = f"CPS_{model_id}" if f"CPS_{model_id}" in df_dcq_full.columns else ("CPS" if "CPS" in df_dcq_full.columns else None)
    if ssem_col is None:
        raise ValueError(f"[SSem] Could not find SSem column in {dcq_path}")
    cols_dcq = ["xsum_id", ssem_col] + ([cps_col] if cps_col else [])
    df_dcq = df_dcq_full[cols_dcq].copy()
    df_dcq = df_dcq.rename(columns={ssem_col: "SSem", **({cps_col: "CPS"} if cps_col else {})})

    # Memorization (SMem) + EM/NE
    df_mem_full = pd.read_parquet(mem_path)
    smem_col = f"SMem_{model_id}" if f"SMem_{model_id}" in df_mem_full.columns else ("SMem" if "SMem" in df_mem_full.columns else None)
    em_col = f"EM_{model_id}" if f"EM_{model_id}" in df_mem_full.columns else ("EM" if "EM" in df_mem_full.columns else None)
    ne_col = f"NE_{model_id}" if f"NE_{model_id}" in df_mem_full.columns else ("NE" if "NE" in df_mem_full.columns else None)
    if smem_col is None:
        raise ValueError(f"[SMem] Could not find SMem column in {mem_path}")
    cols_mem = ["xsum_id", smem_col] + ([em_col] if em_col else []) + ([ne_col] if ne_col else [])
    df_mem = df_mem_full[cols_mem].copy()
    rename_mem = {smem_col: "SMem"}
    if em_col: rename_mem[em_col] = "EM"
    if ne_col: rename_mem[ne_col] = "NE"
    df_mem = df_mem.rename(columns=rename_mem)

    # Stability (SProb) + UAR/mNED
    df_stab_full = pd.read_parquet(stab_path)
    sprob_col = f"SProb_{model_id}" if f"SProb_{model_id}" in df_stab_full.columns else ("SProb" if "SProb" in df_stab_full.columns else None)
    uar_col = f"UAR_{model_id}" if f"UAR_{model_id}" in df_stab_full.columns else ("UAR" if "UAR" in df_stab_full.columns else None)
    mned_col = f"mNED_{model_id}" if f"mNED_{model_id}" in df_stab_full.columns else ("mNED" if "mNED" in df_stab_full.columns else None)
    if sprob_col is None:
        raise ValueError(f"[SProb] Could not find SProb column in {stab_path}")
    cols_stab = ["xsum_id", sprob_col] + ([uar_col] if uar_col else []) + ([mned_col] if mned_col else [])
    df_stab = df_stab_full[cols_stab].copy()
    rename_stab = {sprob_col: "SProb"}
    if uar_col: rename_stab[uar_col] = "UAR"
    if mned_col: rename_stab[mned_col] = "mNED"
    df_stab = df_stab.rename(columns=rename_stab)

    # --------
    # Merge all signals on xsum_id
    # --------
    df = df_lex.merge(df_dcq, on="xsum_id", how="left")
    df = df.merge(df_mem, on="xsum_id", how="left")
    df = df.merge(df_stab, on="xsum_id", how="left")

    # Ensure numeric types for signal levels
    df["SLex"] = to_num(df["SLex"])
    df["SSem"] = to_num(df["SSem"])
    df["SMem"] = to_num(df["SMem"])
    df["SProb"] = to_num(df["SProb"])

    # Compute risk fields
    risk_scores = []
    risk_levels = []
    confidences = []

    for _, r in df.iterrows():
        rs = compute_risk_score(r["SSem"], r["SMem"], r["SProb"], w_sem, w_mem, w_prob)
        cl = compute_confidence(r["SSem"], r["SMem"], r["SProb"])
        rl = map_risk_level(r["SLex"], rs)
        risk_scores.append(rs)
        risk_levels.append(rl)
        confidences.append(cl)

    df["RiskScore"] = risk_scores
    df["RiskLevel"] = risk_levels
    df["Confidence"] = confidences

    # Optional: keep weights used (helps reproducibility)
    df["w_sem"] = w_sem
    df["w_mem"] = w_mem
    df["w_prob"] = w_prob

    # --------
    # Write outputs
    # --------
    ensure_parent_dir(out_parquet)
    df.to_parquet(out_parquet, index=False)

    ensure_parent_dir(out_csv)
    df.to_csv(out_csv, index=False, encoding="utf-8")

    # Write a small summary
    level_counts = df["RiskLevel"].value_counts().to_dict()
    summary = {
        "stage": "risk_integration",
        "model_id": model_id,
        "inputs": {
            "lexical": lexical_path,
            "dcq": dcq_path,
            "mem": mem_path,
            "stability": stab_path,
        },
        "weights": {"w_sem": w_sem, "w_mem": w_mem, "w_prob": w_prob},
        "n_rows": int(len(df)),
        "risk_level_counts": level_counts,
        "risk_score_mean": float(df["RiskScore"].mean()),
        "risk_score_median": float(df["RiskScore"].median()),
        "confidence_mean": float(df["Confidence"].mean()),
        "missing_rates": {
            "SSem_missing": float(df["SSem"].isna().mean()),
            "SMem_missing": float(df["SMem"].isna().mean()),
            "SProb_missing": float(df["SProb"].isna().mean()),
            "SLex_missing": float(df["SLex"].isna().mean()),
        },
        "outputs": {
            "parquet": out_parquet,
            "csv": out_csv,
        }
    }
    write_json(out_summary, summary)

    # Minimal log record for traceability
    log_jsonl(out_log, {
        "status": "done",
        "model_id": model_id,
        "n_rows": int(len(df)),
        "risk_level_counts": level_counts,
        "out_parquet": out_parquet,
        "out_csv": out_csv,
    })

    print("Done.")
    print("Model:", model_id)
    print("Risk levels:", level_counts)
    print("Output parquet:", out_parquet)
    print("Output csv:", out_csv)
    print("Summary:", out_summary)
    print("Log:", out_log)


if __name__ == "__main__":
    main()
