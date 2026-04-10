#!/usr/bin/env python3
"""
scripts/run_risk_integration.py

Step E — Evidence Integration & Risk (per model)

Combines:
- SLex (from cfg.lexical.outputs.parquet, fallback: runs/v3_lexical.parquet)
  [benchmark-level exposure context — NOT included in CRS formula]
- SSem (from runs/v4_dcq_{model_id}.parquet)
- SMem (from runs/v5_mem_{model_id}.parquet)
- SProb (from runs/v6_stability_{model_id}.parquet)

Outputs (per model):
- runs/v7_risk_{model_id}.parquet   — record-level data (one row per xsum_id)
- runs/v7_risk_{model_id}.csv       — same, CSV format
- outputs/v7_risk_summary_{model_id}.json  — aggregated per-model summary
- logs/v7_risk_{model_id}.jsonl     — process-level execution journal

Methodology (v4.1):
─────────────────────────────────────────────────────────────────────────────
CRS Formula (model-level detectors only):
    CRS_raw = 0.35 × (SSem/3) + 0.35 × (SMem/3) + 0.30 × (SProb/3)

    SLex is intentionally excluded. It characterises a property of the
    benchmark, not model behaviour. It is carried through for traceability
    and displayed separately in the report as a benchmark-level exposure prior.

Safety Override (per-example):
    If any of {SSem, SMem, SProb} == 3 for a given example:
        CRS_example = max(CRS_raw_example, 0.60)
    The floor is applied at the example level before aggregation.
    The median CRS is therefore computed over already-floored values —
    the override propagates into the per-model CRS proportionally to the
    share of examples that triggered it.

Risk Levels (qualitative, fixed thresholds on CRS ∈ [0, 1]):
    LOW:      CRS < 0.25
    MODERATE: 0.25 <= CRS < 0.50
    HIGH:     0.50 <= CRS < 0.75
    CRITICAL: CRS >= 0.75

Confidence Estimate (reliability of the CRS assessment):
    Operates on {SSem, SMem, SProb} only — SLex excluded.

    coverage       = count(score > 0 in {SSem, SMem, SProb}) / 3
    agreement      = 1 − variance(SSem, SMem, SProb) / 3
    confidence_raw = 0.5 × coverage + 0.5 × agreement   ∈ [0, 1]
    confidence_pct = round(confidence_raw × 100)

    confidence_level:
        HIGH if confidence_pct >= 70
        LOW  if confidence_pct < 70

    Confidence is diagnostically relevant only at HIGH and CRITICAL.
    At LOW or MODERATE, low confidence is recorded but does not affect
    the recommended action. conflicting_evidence is only flagged at
    HIGH/CRITICAL + LOW confidence.

Aggregation (per-model from per-example):
    CRS and Confidence are computed per example, then aggregated to a single
    per-model value using the median. The median is preferred over the mean
    because it is robust to outlier examples with extreme signal values.
    Per-example distributions are retained in runs/ for full traceability.
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

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


def resolve_lexical_path(cfg: Dict[str, Any]) -> str:
    """
    Resolve lexical parquet path from config, with backward-compatible fallback.
    """
    lexical_cfg = cfg.get("lexical", {}) or {}
    outputs_cfg = lexical_cfg.get("outputs", {}) or {}
    path = (
        outputs_cfg.get("parquet")
        or outputs_cfg.get("out_parquet")
        or "runs/v3_lexical.parquet"
    )
    return str(path)


def derive_smem_item(
    em_series: pd.Series,
    ne_or_ned_series: pd.Series,
    treat_as_binary_ne: bool,
) -> pd.Series:
    """
    Derive item-level SMem from EM + (NE or NED) when SMem column is missing.
    Mirrors run_mem_probe item mapping:
        SMem = 3 if EM == 1
        SMem = 2 if near-exact (NE==1 OR NED<=0.10)
        SMem = 1 if NED<=0.25  (only for continuous NED)
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


# ─────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────

def load_and_select(
    path: str, cols_keep: List[str], stage: str
) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"[{stage}] Missing file: {path}")
    df = pd.read_parquet(path)
    missing = [c for c in cols_keep if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{stage}] Missing columns in {path}: {missing}"
        )
    return df[cols_keep].copy()


# ─────────────────────────────────────────────
# Per-example CRS computation
# ─────────────────────────────────────────────

# Fixed weights — model-level detectors only.
# SLex is excluded from CRS by design.
W_SEM  = 0.35
W_MEM  = 0.35
W_PROB = 0.30

# Safety override floor
OVERRIDE_FLOOR = 0.60

# Risk level thresholds (CRS ∈ [0, 1])
RISK_THRESHOLDS = [
    (0.25, "LOW"),
    (0.50, "MODERATE"),
    (0.75, "HIGH"),
]

# Confidence threshold
CONFIDENCE_THRESHOLD = 70


def compute_crs_raw(ssem_n: pd.Series, smem_n: pd.Series, sprob_n: pd.Series) -> pd.Series:
    """
    Weighted linear combination of normalised model-level detector scores.
    SLex is intentionally excluded.
    """
    return W_SEM * ssem_n + W_MEM * smem_n + W_PROB * sprob_n


def apply_safety_override(
    crs_raw: pd.Series,
    ssem: pd.Series,
    smem: pd.Series,
    sprob: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    If any model-level detector returns 3, floor CRS at OVERRIDE_FLOOR.
    Returns (crs_final, override_active_bool_series).
    """
    override_active = (ssem == 3) | (smem == 3) | (sprob == 3)
    crs_final = crs_raw.copy()
    crs_final = crs_final.where(
        ~override_active,
        crs_raw.clip(lower=OVERRIDE_FLOOR),
    )
    # Ensure [0, 1] range
    crs_final = crs_final.clip(lower=0.0, upper=1.0)
    return crs_final, override_active


def map_risk_level(crs: pd.Series) -> pd.Series:
    """
    Map CRS values to qualitative risk levels.
    Thresholds: LOW < 0.25 <= MODERATE < 0.50 <= HIGH < 0.75 <= CRITICAL
    """
    conditions = [
        crs < 0.25,
        (crs >= 0.25) & (crs < 0.50),
        (crs >= 0.50) & (crs < 0.75),
        crs >= 0.75,
    ]
    choices = ["LOW", "MODERATE", "HIGH", "CRITICAL"]
    return pd.Series(
        np.select(conditions, choices, default="MODERATE"),
        index=crs.index,
    )


def compute_confidence(
    ssem: pd.Series, smem: pd.Series, sprob: pd.Series
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Compute per-example Confidence from three model-level detectors.
    SLex is excluded — it is not a behavioural signal.

    Returns:
        coverage       — proportion of active detectors (score > 0), ∈ [0, 1]
        agreement      — 1 − normalised variance, ∈ [0, 1]
        confidence_pct — integer percentage
        confidence_level — "HIGH" (>= threshold) or "LOW"
    """
    active = (ssem > 0).astype(float) + (smem > 0).astype(float) + (sprob > 0).astype(float)
    coverage = active / 3.0

    mean_score = (ssem + smem + sprob) / 3.0
    variance = (
        (ssem - mean_score) ** 2
        + (smem - mean_score) ** 2
        + (sprob - mean_score) ** 2
    ) / 3.0
    # Normalise variance by max possible (3.0) to get [0, 1] range
    agreement = (1.0 - variance / 3.0).clip(lower=0.0, upper=1.0)

    confidence_raw = 0.5 * coverage + 0.5 * agreement
    confidence_pct = (confidence_raw * 100).round().astype(int)

    confidence_level = pd.Series(
        np.where(confidence_pct >= CONFIDENCE_THRESHOLD, "HIGH", "LOW"),
        index=ssem.index,
    )

    return coverage, agreement, confidence_pct, confidence_level


# ─────────────────────────────────────────────
# Per-model aggregation
# ─────────────────────────────────────────────

def aggregate_model(df: pd.DataFrame, model_id: str) -> Dict[str, Any]:
    """
    Aggregate per-example results to a single per-model summary.

    Aggregation method: median — robust to outlier examples with extreme
    signal values. Per-example distributions are retained in runs/ for
    full traceability.

    Safety override — per-example semantics:
        floor 0.60 is applied to each example where any detector == 3.
        The median CRS is computed over already-floored per-example values,
        so the override propagates into the aggregated result proportionally
        to the share of examples that triggered it.
        safety_override_active=True if any example triggered the override.
        override_example_count and override_example_pct quantify the scope.

    Confidence interpretation by risk level:
        Confidence is diagnostically relevant only at HIGH and CRITICAL.
        At LOW or MODERATE, low confidence is recorded but does not
        affect the recommended action or require a special caveat.
        conflicting_evidence is therefore only flagged at HIGH/CRITICAL.

    Returns a dict suitable for inclusion in the outputs/ JSON summary.
    """
    n_total = len(df)

    # Aggregate CRS: median across all examples
    crs_model = float(df["CRS"].median())
    crs_model = round(crs_model, 4)

    # Aggregate detector scores: median per detector
    ssem_model  = float(df["SSem"].median())
    smem_model  = float(df["SMem"].median())
    sprob_model = float(df["SProb"].median())
    slex_model  = float(df["SLex"].median())

    # Re-derive model-level risk level from aggregated CRS
    if crs_model < 0.25:
        risk_level = "LOW"
    elif crs_model < 0.50:
        risk_level = "MODERATE"
    elif crs_model < 0.75:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"

    # Safety override — per-model flag (does NOT modify CRS)
    override_example_count = int(df["override_active"].sum())
    override_example_pct   = round(override_example_count / n_total * 100, 1) if n_total > 0 else 0.0
    # True if any example triggered it; a low CRS alongside True means
    # the override fired on a minority of examples — inspect runs/.
    safety_override_active = override_example_count > 0

    # Confidence: recompute from aggregated (median) detector scores
    # to get a single model-level Confidence value
    ssem_s  = pd.Series([ssem_model])
    smem_s  = pd.Series([smem_model])
    sprob_s = pd.Series([sprob_model])

    coverage_s, agreement_s, conf_pct_s, conf_level_s = compute_confidence(
        ssem_s, smem_s, sprob_s
    )
    confidence_pct   = int(conf_pct_s.iloc[0])
    coverage_val     = round(float(coverage_s.iloc[0]), 4)
    agreement_val    = round(float(agreement_s.iloc[0]), 4)
    confidence_level = str(conf_level_s.iloc[0])

    # conflicting_evidence: relevant only at HIGH/CRITICAL.
    # At LOW/MODERATE, low confidence is informational and does not
    # change the recommended action or require a special caveat.
    conflicting_evidence = (
        risk_level in ("HIGH", "CRITICAL")
        and confidence_level == "LOW"
    )

    # Risk level distribution across examples
    risk_level_counts = df["RiskLevel"].value_counts().to_dict()

    # Confidence level distribution across examples
    confidence_counts = df["ConfidenceLevel"].value_counts().to_dict()

    return {
        # ── Primary report fields ──────────────────────────────────────────
        "model_id":   model_id,
        "CRS":        crs_model,
        "risk_level": risk_level,

        # ── Safety override (per-example) ────────────────────────────────────
        # Median CRS is computed over already-floored per-example values.
        # override_example_count/pct quantify how many examples triggered it.
        "safety_override_active":  safety_override_active,
        "override_example_count":  override_example_count,
        "override_example_pct":    override_example_pct,

        # ── Confidence ────────────────────────────────────────────────────
        "confidence_pct":        confidence_pct,
        "confidence_level":      confidence_level,
        "coverage":              coverage_val,
        "signal_agreement_pct":  round(agreement_val * 100),
        # conflicting_evidence: only True at HIGH/CRITICAL with LOW confidence.
        # Informational at LOW/MODERATE — no action change.
        "conflicting_evidence":  conflicting_evidence,

        # ── Detector scores (median, for report detector bars) ─────────────
        "SSem_score":  round(ssem_model,  4),
        "SMem_score":  round(smem_model,  4),
        "SProb_score": round(sprob_model, 4),
        "SLex_score":  round(slex_model,  4),

        # ── CRS distribution across examples ──────────────────────────────
        "CRS_mean":   round(float(df["CRS"].mean()),   4),
        "CRS_median": round(float(df["CRS"].median()), 4),
        "CRS_min":    round(float(df["CRS"].min()),    4),
        "CRS_max":    round(float(df["CRS"].max()),    4),

        # ── Weights used (for traceability) ───────────────────────────────
        "weights": {
            "w_sem":  W_SEM,
            "w_mem":  W_MEM,
            "w_prob": W_PROB,
            "note":   "SLex excluded from CRS formula — benchmark-level signal only",
        },

        # ── Distributions ─────────────────────────────────────────────────
        "n_total":             n_total,
        "risk_level_counts":   risk_level_counts,
        "confidence_counts":   confidence_counts,
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to configs/run_config.yaml",
    )
    parser.add_argument(
        "--model_id", type=str, required=True,
        help="Model ID (e.g., gpt4omini, gemini25flash)",
    )
    args = parser.parse_args()

    cfg      = load_yaml(args.config)
    model_id = args.model_id

    # ── Paths (unchanged from v1 for project compatibility) ───────────────
    lexical_path = resolve_lexical_path(cfg)
    dcq_path     = f"runs/v4_dcq_{model_id}.parquet"
    mem_path     = f"runs/v5_mem_{model_id}.parquet"
    stab_path    = f"runs/v6_stability_{model_id}.parquet"

    out_parquet = f"runs/v7_risk_{model_id}.parquet"
    out_csv     = f"runs/v7_risk_{model_id}.csv"
    out_summary = f"outputs/v7_risk_summary_{model_id}.json"
    out_log     = f"logs/v7_risk_{model_id}.jsonl"

    # ── Load SLex (benchmark-level, model-agnostic) ───────────────────────
    df_lex = load_and_select(
        lexical_path,
        cols_keep=["xsum_id", "MaxSpanLen", "NgramHits", "ProxyCount", "SLex"],
        stage="SLex",
    )

    # ── Load SSem ─────────────────────────────────────────────────────────
    df_dcq_full = pd.read_parquet(dcq_path)
    ssem_col = (
        f"SSem_{model_id}" if f"SSem_{model_id}" in df_dcq_full.columns
        else ("SSem" if "SSem" in df_dcq_full.columns else None)
    )
    cps_col = (
        f"CPS_{model_id}" if f"CPS_{model_id}" in df_dcq_full.columns
        else ("CPS" if "CPS" in df_dcq_full.columns else None)
    )
    if ssem_col is None:
        raise ValueError(f"[SSem] Could not find SSem column in {dcq_path}")

    cols_dcq = ["xsum_id", ssem_col] + ([cps_col] if cps_col else [])
    df_dcq = df_dcq_full[cols_dcq].copy()
    df_dcq = df_dcq.rename(
        columns={ssem_col: "SSem", **({cps_col: "CPS"} if cps_col else {})}
    )

    # ── Load SMem ─────────────────────────────────────────────────────────
    df_mem_full = pd.read_parquet(mem_path)
    smem_col = (
        f"SMem_{model_id}" if f"SMem_{model_id}" in df_mem_full.columns
        else ("SMem" if "SMem" in df_mem_full.columns else None)
    )
    em_col = (
        f"EM_{model_id}" if f"EM_{model_id}" in df_mem_full.columns
        else ("EM" if "EM" in df_mem_full.columns else None)
    )
    ne_col = (
        f"NE_{model_id}" if f"NE_{model_id}" in df_mem_full.columns
        else ("NE" if "NE" in df_mem_full.columns else None)
    )
    ned_col = (
        f"NED_{model_id}" if f"NED_{model_id}" in df_mem_full.columns
        else ("NED" if "NED" in df_mem_full.columns else None)
    )

    if smem_col is None:
        # Compatibility fallback: derive SMem from EM + NED/NE
        source_col = ned_col or ne_col
        if em_col is None or source_col is None:
            raise ValueError(
                f"[SMem] Could not find SMem in {mem_path}, and cannot derive it "
                f"(need EM + NED/NE; found EM={em_col}, NE={ne_col}, NED={ned_col})."
            )
        source_vals = pd.to_numeric(
            df_mem_full[source_col], errors="coerce"
        ).dropna()
        unique_vals = set(source_vals.unique().tolist())
        treat_as_binary_ne = (source_col == ne_col) and unique_vals.issubset({0.0, 1.0})
        df_mem_full["SMem__derived"] = derive_smem_item(
            em_series=df_mem_full[em_col],
            ne_or_ned_series=df_mem_full[source_col],
            treat_as_binary_ne=treat_as_binary_ne,
        )
        smem_col = "SMem__derived"

    cols_mem = (
        ["xsum_id", smem_col]
        + ([em_col]  if em_col  else [])
        + ([ne_col]  if ne_col  else [])
    )
    df_mem = df_mem_full[cols_mem].copy()
    rename_mem: Dict[str, str] = {smem_col: "SMem"}
    if em_col: rename_mem[em_col] = "EM"
    if ne_col: rename_mem[ne_col] = "NE"
    df_mem = df_mem.rename(columns=rename_mem)

    # ── Load SProb ────────────────────────────────────────────────────────
    df_stab_full = pd.read_parquet(stab_path)
    sprob_col = (
        f"SProb_{model_id}" if f"SProb_{model_id}" in df_stab_full.columns
        else ("SProb" if "SProb" in df_stab_full.columns else None)
    )
    uar_col = (
        f"UAR_{model_id}" if f"UAR_{model_id}" in df_stab_full.columns
        else ("UAR" if "UAR" in df_stab_full.columns else None)
    )
    mned_col = (
        f"mNED_{model_id}" if f"mNED_{model_id}" in df_stab_full.columns
        else ("mNED" if "mNED" in df_stab_full.columns else None)
    )
    if sprob_col is None:
        raise ValueError(f"[SProb] Could not find SProb column in {stab_path}")

    cols_stab = (
        ["xsum_id", sprob_col]
        + ([uar_col]  if uar_col  else [])
        + ([mned_col] if mned_col else [])
    )
    df_stab = df_stab_full[cols_stab].copy()
    rename_stab: Dict[str, str] = {sprob_col: "SProb"}
    if uar_col:  rename_stab[uar_col]  = "UAR"
    if mned_col: rename_stab[mned_col] = "mNED"
    df_stab = df_stab.rename(columns=rename_stab)

    # ── Merge all signals on xsum_id ──────────────────────────────────────
    df = df_lex.merge(df_dcq, on="xsum_id", how="left")
    df = df.merge(df_mem,  on="xsum_id", how="left")
    df = df.merge(df_stab, on="xsum_id", how="left")

    # Record missing rates before numeric coercion
    missing_rates = {
        "SSem_missing":  float(df["SSem"].isna().mean()),
        "SMem_missing":  float(df["SMem"].isna().mean()),
        "SProb_missing": float(df["SProb"].isna().mean()),
        "SLex_missing":  float(df["SLex"].isna().mean()),
    }

    # Coerce to numeric, fill NaN with 0
    for col in ["SLex", "SSem", "SMem", "SProb"]:
        df[col] = to_num(df[col])

    # ── 1. Normalise model-level scores to [0, 1] ─────────────────────────
    # SLex_n is retained for traceability in runs/ but not used in CRS.
    df["SLex_n"]  = df["SLex"]  / 3.0
    df["SSem_n"]  = df["SSem"]  / 3.0
    df["SMem_n"]  = df["SMem"]  / 3.0
    df["SProb_n"] = df["SProb"] / 3.0

    # ── 2. Compute CRS_raw (SSem + SMem + SProb only) ─────────────────────
    df["CRS_raw"] = compute_crs_raw(df["SSem_n"], df["SMem_n"], df["SProb_n"])

    # ── 3. Apply safety override ──────────────────────────────────────────
    df["CRS"], df["override_active"] = apply_safety_override(
        df["CRS_raw"], df["SSem"], df["SMem"], df["SProb"]
    )

    # ── 4. Map to qualitative risk level ──────────────────────────────────
    df["RiskLevel"] = map_risk_level(df["CRS"])

    # ── 5. Compute per-example Confidence ────────────────────────────────
    (
        df["coverage"],
        df["signal_agreement"],
        df["confidence_pct"],
        df["ConfidenceLevel"],
    ) = compute_confidence(df["SSem"], df["SMem"], df["SProb"])

    # ── 6. Store weights for per-row traceability ─────────────────────────
    df["w_sem"]  = W_SEM
    df["w_mem"]  = W_MEM
    df["w_prob"] = W_PROB

    # ── 7. Write record-level artifacts (runs/) ───────────────────────────
    ensure_parent_dir(out_parquet)
    df.to_parquet(out_parquet, index=False)

    ensure_parent_dir(out_csv)
    df.to_csv(out_csv, index=False, encoding="utf-8")

    # ── 8. Aggregate to per-model summary ─────────────────────────────────
    model_summary = aggregate_model(df, model_id)

    # ── 9. Build full outputs/ JSON summary ───────────────────────────────
    summary = {
        "stage":    "risk_integration",
        "pipeline_version": "4.1.0",
        "model_id": model_id,

        # Input artifact paths (for traceability)
        "inputs": {
            "lexical":   lexical_path,
            "dcq":       dcq_path,
            "mem":       mem_path,
            "stability": stab_path,
        },

        # Per-model aggregated results (primary report fields)
        "model": model_summary,

        # Data quality
        "n_rows":       int(len(df)),
        "missing_rates": missing_rates,

        # Output artifact paths
        "outputs": {
            "parquet": out_parquet,
            "csv":     out_csv,
        },
    }
    write_json(out_summary, summary)

    # ── 10. Write process log entry (logs/) ───────────────────────────────
    log_jsonl(out_log, {
        "status":                  "done",
        "model_id":                model_id,
        "n_rows":                  int(len(df)),
        "CRS":                     model_summary["CRS"],
        "risk_level":              model_summary["risk_level"],
        "confidence_pct":          model_summary["confidence_pct"],
        "confidence_level":        model_summary["confidence_level"],
        "conflicting_evidence":    model_summary["conflicting_evidence"],
        "safety_override_active":  model_summary["safety_override_active"],
        "override_example_count":  model_summary["override_example_count"],
        "override_example_pct":    model_summary["override_example_pct"],
        "risk_level_counts":       model_summary["risk_level_counts"],
        "out_parquet":             out_parquet,
        "out_csv":                 out_csv,
        "out_summary":             out_summary,
    })

    # ── Console output ────────────────────────────────────────────────────
    override_note = (
        f"True ({model_summary['override_example_count']} examples, "
        f"{model_summary['override_example_pct']}% of dataset) — "
        f"CRS unchanged, inspect runs/ for per-example detail"
        if model_summary["safety_override_active"]
        else "False"
    )
    conf_note = (
        " [conflicting evidence — strengthened caveat required]"
        if model_summary["conflicting_evidence"]
        else ""
    )
    print("Done.")
    print(f"Model:              {model_id}")
    print(f"CRS:                {model_summary['CRS']:.4f}")
    print(f"Risk level:         {model_summary['risk_level']}")
    print(f"Confidence:         {model_summary['confidence_pct']}% ({model_summary['confidence_level']}){conf_note}")
    print(f"Coverage:           {round(model_summary['coverage'] * 100)}%")
    print(f"Signal agreement:   {model_summary['signal_agreement_pct']}%")
    print(f"Safety override:    {override_note}")
    print(f"Risk level counts:  {model_summary['risk_level_counts']}")
    print(f"Output parquet:     {out_parquet}")
    print(f"Output CSV:         {out_csv}")
    print(f"Summary:            {out_summary}")
    print(f"Log:                {out_log}")


if __name__ == "__main__":
    main()
