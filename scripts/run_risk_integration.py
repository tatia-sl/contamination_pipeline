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

Current implementation:
- Merges SLex + SSem + SMem + SProb on xsum_id.
- Coerces missing numeric signal values to 0.
- Computes normalized levels:
    SLex_n = SLex / 3
    SSem_n = SSem / 3
    SMem_n = SMem / 3
    SProb_n = SProb / 3
- Computes base score (0..100):
    RiskScore_raw = 100 * (
        w_lex * SLex_n +
        w_sem * SSem_n +
        w_mem * SMem_n +
        w_prob * SProb_n
    )

Weights:
- Read from cfg.risk_integration.weights with defaults:
    w_lex=0.35, w_sem=0.20, w_mem=0.30, w_prob=0.15
- Must sum to 1.0 (strict check).

Overrides:
- direct_evidence:
    if (SLex == 3) OR (SMem == 3), enforce RiskScore >= 50
- single_signal_caution:
    if (SProb >= 2) AND max(SLex, SSem, SMem) <= 1, cap RiskScore <= 49

Risk levels:
- Low:      RiskScore < 25
- Medium:   25 <= RiskScore < 50
- High:     50 <= RiskScore < 75
- Critical: RiskScore >= 75

Confidence:
- strong = count(signals >= 2) over {SLex, SSem, SMem, SProb}
- weak   = count(signals >= 1) over {SLex, SSem, SMem, SProb}
- High   if (strong >= 2) and (SLex >= 2 or SMem >= 2)
- Medium if weak >= 2
- Low    otherwise
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
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def derive_smem_item(em_series: pd.Series, ne_or_ned_series: pd.Series, treat_as_binary_ne: bool) -> pd.Series:
    """
    Derive item-level SMem from EM + (NE or NED) when SMem column is missing.
    Rules mirror run_mem_probe item mapping:
      SMem = 3 if EM == 1
      SMem = 2 if near-exact (NE==1 OR NED<=0.10)
      SMem = 1 if NED<=0.25 (only available for continuous distance)
      SMem = 0 otherwise
    """
    em = pd.to_numeric(em_series, errors="coerce")
    x = pd.to_numeric(ne_or_ned_series, errors="coerce")

    smem = pd.Series(pd.NA, index=em.index, dtype="Float64")
    smem = smem.mask((em == 1), 3.0)

    if treat_as_binary_ne:
        smem = smem.mask((smem.isna()) & (x == 1), 2.0)
        smem = smem.mask((smem.isna()) & x.notna(), 0.0)
    else:
        smem = smem.mask((smem.isna()) & (x <= 0.10), 2.0)
        smem = smem.mask((smem.isna()) & (x <= 0.25), 1.0)
        smem = smem.mask((smem.isna()) & x.notna(), 0.0)

    return pd.to_numeric(smem, errors="coerce")


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

    risk_cfg = cfg.get("risk_integration", {})

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
    ned_col = f"NED_{model_id}" if f"NED_{model_id}" in df_mem_full.columns else ("NED" if "NED" in df_mem_full.columns else None)

    if smem_col is None:
        # Compatibility fallback:
        # derive SMem from EM + NED/NE for control-only or legacy mem outputs.
        source_col = ned_col or ne_col
        if em_col is None or source_col is None:
            raise ValueError(
                f"[SMem] Could not find SMem in {mem_path}, and cannot derive it "
                f"(need EM + NED/NE columns; found EM={em_col}, NE={ne_col}, NED={ned_col})."
            )
        source_vals = pd.to_numeric(df_mem_full[source_col], errors="coerce").dropna()
        unique_vals = set(source_vals.unique().tolist())
        treat_as_binary_ne = (source_col == ne_col) and unique_vals.issubset({0.0, 1.0})
        df_mem_full["SMem__derived"] = derive_smem_item(
            em_series=df_mem_full[em_col],
            ne_or_ned_series=df_mem_full[source_col],
            treat_as_binary_ne=treat_as_binary_ne,
        )
        smem_col = "SMem__derived"

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

    # Missing rates before numeric coercion/fill
    missing_rates = {
        "SSem_missing": float(df["SSem"].isna().mean()),
        "SMem_missing": float(df["SMem"].isna().mean()),
        "SProb_missing": float(df["SProb"].isna().mean()),
        "SLex_missing": float(df["SLex"].isna().mean()),
    }

    # Ensure numeric types for signal levels
    sem_col = "SSem"
    mem_col = "SMem"
    prob_col = "SProb"

    df["SLex"] = to_num(df["SLex"])
    df["SSem"] = to_num(df[sem_col])
    df["SMem"] = to_num(df[mem_col])
    df["SProb"] = to_num(df[prob_col])

    # Weights
    w_lex = float(risk_cfg.get("weights", {}).get("lex", 0.35))
    w_sem = float(risk_cfg.get("weights", {}).get("sem", 0.20))
    w_mem = float(risk_cfg.get("weights", {}).get("mem", 0.30))
    w_prob = float(risk_cfg.get("weights", {}).get("prob", 0.15))

    w_sum = w_lex + w_sem + w_mem + w_prob
    if abs(w_sum - 1.0) > 1e-9:
        raise ValueError(f"Risk weights must sum to 1.0, got {w_sum}")

    # Normalized signals
    df["SLex_n"] = df["SLex"] / 3.0
    df["SSem_n"] = df["SSem"] / 3.0
    df["SMem_n"] = df["SMem"] / 3.0
    df["SProb_n"] = df["SProb"] / 3.0

    # Base risk score (0..100)
    df["RiskScore_raw"] = 100.0 * (
        w_lex * df["SLex_n"] +
        w_sem * df["SSem_n"] +
        w_mem * df["SMem_n"] +
        w_prob * df["SProb_n"]
    )

    # Override rules
    df["override_direct_evidence"] = ((df["SLex"] == 3) | (df["SMem"] == 3))
    df["override_single_signal_caution"] = (
        (df["SProb"] >= 2) &
        (df[["SLex", "SSem", "SMem"]].max(axis=1) <= 1)
    )

    df["RiskScore"] = df["RiskScore_raw"]
    df.loc[df["override_direct_evidence"], "RiskScore"] = df.loc[df["override_direct_evidence"], "RiskScore"].clip(lower=50.0)

    mask_caution = (~df["override_direct_evidence"]) & (df["override_single_signal_caution"])
    df.loc[mask_caution, "RiskScore"] = df.loc[mask_caution, "RiskScore"].clip(upper=49.0)

    def risk_level(r: float) -> str:
        if r < 25:
            return "Low"
        if r < 50:
            return "Medium"
        if r < 75:
            return "High"
        return "Critical"

    df["RiskLevel"] = df["RiskScore"].apply(risk_level)

    # Confidence (categorical)
    strong = (
        (df["SLex"] >= 2).astype(int) +
        (df["SSem"] >= 2).astype(int) +
        (df["SMem"] >= 2).astype(int) +
        (df["SProb"] >= 2).astype(int)
    )
    weak = (
        (df["SLex"] >= 1).astype(int) +
        (df["SSem"] >= 1).astype(int) +
        (df["SMem"] >= 1).astype(int) +
        (df["SProb"] >= 1).astype(int)
    )
    df["Confidence"] = "Low"
    df.loc[weak >= 2, "Confidence"] = "Medium"
    df.loc[(strong >= 2) & ((df["SLex"] >= 2) | (df["SMem"] >= 2)), "Confidence"] = "High"

    # Keep weights used
    df["w_lex"] = w_lex
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
        "weights": {"w_lex": w_lex, "w_sem": w_sem, "w_mem": w_mem, "w_prob": w_prob},
        "n_rows": int(len(df)),
        "risk_level_counts": level_counts,
        "risk_score_mean": float(df["RiskScore"].mean()),
        "risk_score_median": float(df["RiskScore"].median()),
        "confidence_counts": df["Confidence"].value_counts().to_dict(),
        "override_counts": {
            "direct_evidence": int(df["override_direct_evidence"].sum()),
            "single_signal_caution": int(df["override_single_signal_caution"].sum()),
        },
        "missing_rates": missing_rates,
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
