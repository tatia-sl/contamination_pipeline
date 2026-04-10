#!/usr/bin/env python3
"""
scripts/run_risk_integration.py

Step E — Evidence Integration & Risk (per model)

Input sources:
─────────────────────────────────────────────────────────────────────────────
CRS inputs — aggregate scores from previous stage summary JSONs:
    SLex  <- outputs/v3_lexical_summary.json               field: SLex_aggregate
    SSem  <- outputs/v4_dcq_summary_{model_id}.json        field: SSem_aggregate
    SMem  <- outputs/v5_mem_summary_{model_id}.json        field: SMem_aggregate
    SProb <- outputs/v6_stability_summary_{model_id}.json  field: SProb_aggregate

    These are the final integer scores (0-3) produced by each detector
    after applying their respective thresholding rules to the full dataset.
    CRS is computed directly from these aggregate values — no re-aggregation.

Confidence inputs — all four aggregate scores:
    SLex_aggregate, SSem_aggregate, SMem_aggregate, SProb_aggregate.
    SLex contributes as the exposure component: high benchmark availability
    in open sources raises the prior probability of contamination and
    increases confidence in the assessment. No parquet required.
-----------------------------------------------------------------------------

Outputs (per model):
    outputs/v7_risk_summary_{model_id}.json  -- per-model assessment (primary)
    logs/v7_risk_{model_id}.jsonl            -- process-level execution journal

Methodology (v4.2):
-----------------------------------------------------------------------------
CRS Formula:
    CRS_raw = 0.35 * (SSem_aggregate/3)
            + 0.35 * (SMem_aggregate/3)
            + 0.30 * (SProb_aggregate/3)

    SLex is intentionally excluded. It characterises a property of the
    benchmark, not model behaviour. It is carried for traceability and
    displayed separately as a benchmark-level exposure prior.

Safety Override:
    If any of {SSem_aggregate, SMem_aggregate, SProb_aggregate} == 3:
        CRS = max(CRS_raw, 0.60)
    Applied at the aggregate level — a maximum signal from any single
    model-level detector floors the overall CRS at 0.60.

Risk Levels (fixed thresholds on CRS in [0, 1]):
    LOW:      CRS < 0.25
    MODERATE: 0.25 <= CRS < 0.50
    HIGH:     0.50 <= CRS < 0.75
    CRITICAL: CRS >= 0.75

Confidence Estimate:
    Measures the reliability of the CRS assessment. Composed of three equal
    components — all four aggregate scores contribute.

    coverage   = count(score > 0 in {SSem, SMem, SProb}) / 3
                 How many model-level detectors produced a signal.

    agreement  = 1 - variance(SSem, SMem, SProb) / 3
                 How consistently the model-level detectors agree.

    exposure   = SLex_aggregate / 3
                 How widely the benchmark was available in open sources.
                 High exposure increases the prior probability of contamination
                 and therefore increases confidence that model signals are
                 being interpreted in the correct context.

    confidence_raw = (coverage + agreement + exposure) / 3
    confidence_pct = round(confidence_raw * 100)

    confidence_level:
        HIGH if confidence_pct >= 70
        LOW  if confidence_pct < 70

    Confidence is diagnostically relevant only at HIGH and CRITICAL.
    At LOW or MODERATE it is recorded but does not change the recommendation.
    conflicting_evidence is flagged only at HIGH/CRITICAL + LOW confidence.
-----------------------------------------------------------------------------
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

W_SEM  = 0.35
W_MEM  = 0.35
W_PROB = 0.30

OVERRIDE_FLOOR       = 0.60
CONFIDENCE_THRESHOLD = 70


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: str, stage: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"[{stage}] Missing summary JSON: {path}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


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


def to_num_scalar(value: Any, field: str, stage: str) -> float:
    """Coerce a scalar value from a JSON summary to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"[{stage}] Field '{field}' is not numeric: {value!r}"
        )


# ─────────────────────────────────────────────
# Aggregate score extraction from summary JSONs
# ─────────────────────────────────────────────

def extract_slex(summary: Dict[str, Any]) -> float:
    """Extract SLex_aggregate from v3_lexical_summary.json."""
    for key in ("SLex_aggregate", "SLex"):
        if key in summary:
            return to_num_scalar(summary[key], key, "SLex")
    raise KeyError(
        f"[SLex] Could not find 'SLex_aggregate' or 'SLex'. "
        f"Available keys: {list(summary.keys())}"
    )


def extract_ssem(summary: Dict[str, Any]) -> float:
    """Extract SSem_aggregate from v4_dcq_summary_{model_id}.json."""
    for key in ("SSem_aggregate", "SSem"):
        if key in summary:
            return to_num_scalar(summary[key], key, "SSem")
    raise KeyError(
        f"[SSem] Could not find 'SSem_aggregate' or 'SSem'. "
        f"Available keys: {list(summary.keys())}"
    )


def extract_smem(summary: Dict[str, Any]) -> float:
    """Extract SMem_aggregate from v5_mem_summary_{model_id}.json."""
    for key in ("SMem_aggregate", "SMem"):
        if key in summary:
            return to_num_scalar(summary[key], key, "SMem")
    raise KeyError(
        f"[SMem] Could not find 'SMem_aggregate' or 'SMem'. "
        f"Available keys: {list(summary.keys())}"
    )


def extract_sprob(summary: Dict[str, Any]) -> float:
    """Extract SProb_aggregate from v6_stability_summary_{model_id}.json."""
    for key in ("SProb_aggregate", "SProb"):
        if key in summary:
            return to_num_scalar(summary[key], key, "SProb")
    raise KeyError(
        f"[SProb] Could not find 'SProb_aggregate' or 'SProb'. "
        f"Available keys: {list(summary.keys())}"
    )


# ─────────────────────────────────────────────
# CRS computation (aggregate level)
# ─────────────────────────────────────────────

def compute_crs(
    ssem: float, smem: float, sprob: float
) -> Tuple[float, float, bool]:
    """
    Compute CRS from aggregate detector scores.

    Returns:
        crs_raw         weighted sum before override, in [0, 1]
        crs             final CRS after safety override, in [0, 1]
        override_active True if any detector == 3
    """
    crs_raw = W_SEM * (ssem / 3.0) + W_MEM * (smem / 3.0) + W_PROB * (sprob / 3.0)
    override_active = any(s == 3.0 for s in (ssem, smem, sprob))
    crs = max(crs_raw, OVERRIDE_FLOOR) if override_active else crs_raw
    crs = float(np.clip(crs, 0.0, 1.0))
    return round(crs_raw, 6), round(crs, 6), override_active


def map_risk_level(crs: float) -> str:
    """Map CRS value to qualitative risk level."""
    if crs < 0.25:
        return "LOW"
    if crs < 0.50:
        return "MODERATE"
    if crs < 0.75:
        return "HIGH"
    return "CRITICAL"


# ─────────────────────────────────────────────
# Confidence computation (per-sample SProb data)
# ─────────────────────────────────────────────

def compute_confidence(
    slex: float,
    ssem: float,
    smem: float,
    sprob: float,
) -> Tuple[float, float, float, int, str]:
    """
    Compute Confidence from all four aggregate detector scores.
    Measures the reliability of the CRS assessment.

    Three equal components:

        coverage   = count(score > 0 in {SSem, SMem, SProb}) / 3
                     How many model-level detectors produced a signal.

        agreement  = 1 - variance(SSem, SMem, SProb) / 3
                     How consistently the model-level detectors agree.
                     Normalised by max possible variance (3.0).

        exposure   = SLex / 3
                     Benchmark availability in open sources. High exposure
                     raises the prior probability of contamination and
                     increases confidence that signals are contextually valid.

        confidence_raw = (coverage + agreement + exposure) / 3
        confidence_pct = round(confidence_raw * 100)

    Returns:
        coverage           in [0, 1]
        agreement          in [0, 1]
        exposure           in [0, 1]
        confidence_pct     integer 0-100
        confidence_level   "HIGH" or "LOW"
    """
    active   = sum(1 for s in (ssem, smem, sprob) if s > 0)
    coverage = active / 3.0

    mean_score = (ssem + smem + sprob) / 3.0
    variance   = (
        (ssem - mean_score)**2
        + (smem - mean_score)**2
        + (sprob - mean_score)**2
    ) / 3.0
    agreement = float(np.clip(1.0 - variance / 3.0, 0.0, 1.0))

    exposure = float(np.clip(slex / 3.0, 0.0, 1.0))

    confidence_raw   = (coverage + agreement + exposure) / 3.0
    confidence_pct   = int(round(confidence_raw * 100))
    confidence_level = "HIGH" if confidence_pct >= CONFIDENCE_THRESHOLD else "LOW"

    return (
        round(coverage, 4),
        round(agreement, 4),
        round(exposure, 4),
        confidence_pct,
        confidence_level,
    )


# ─────────────────────────────────────────────
# Path resolution
# ─────────────────────────────────────────────

def resolve_input_paths(
    cfg: Dict[str, Any], model_id: str
) -> Tuple[str, str, str, str]:
    """
    Resolve paths to the four stage summary JSONs.
    Uses config where present; falls back to conventional paths.
    No parquet is required — Confidence is computed from aggregate scores only.
    """
    def _summary(stage_key: str, default: str) -> str:
        val = str(
            cfg.get(stage_key, {}).get("outputs", {}).get("summary", default)
        )
        return val.replace("{model_id}", model_id)

    lex_summary  = _summary("lexical",   "outputs/v3_lexical_summary.json")
    dcq_summary  = _summary("dcq",       f"outputs/v4_dcq_summary_{model_id}.json")
    mem_summary  = _summary("mem",       f"outputs/v5_mem_summary_{model_id}.json")
    stab_summary = _summary("stability", f"outputs/v6_stability_summary_{model_id}.json")

    return lex_summary, dcq_summary, mem_summary, stab_summary


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

    # Output paths — unchanged for project compatibility
    out_summary = f"outputs/v7_risk_summary_{model_id}.json"
    out_log     = f"logs/v7_risk_{model_id}.jsonl"

    # ── 1. Resolve input paths ────────────────────────────────────────────
    (
        lex_summary_path,
        dcq_summary_path,
        mem_summary_path,
        stab_summary_path,
    ) = resolve_input_paths(cfg, model_id)

    # ── 2. Load aggregate scores from summary JSONs ───────────────────────
    lex_summary  = load_json(lex_summary_path,  "SLex")
    dcq_summary  = load_json(dcq_summary_path,  "SSem")
    mem_summary  = load_json(mem_summary_path,  "SMem")
    stab_summary = load_json(stab_summary_path, "SProb")

    slex_agg  = extract_slex(lex_summary)
    ssem_agg  = extract_ssem(dcq_summary)
    smem_agg  = extract_smem(mem_summary)
    sprob_agg = extract_sprob(stab_summary)

    for name, val in (
        ("SLex", slex_agg), ("SSem", ssem_agg),
        ("SMem", smem_agg), ("SProb", sprob_agg),
    ):
        if not (0.0 <= val <= 3.0):
            raise ValueError(
                f"[{name}] Aggregate score out of expected range [0, 3]: {val}"
            )

    # ── 3. Compute CRS from aggregate scores ─────────────────────────────
    crs_raw, crs, override_active = compute_crs(ssem_agg, smem_agg, sprob_agg)
    risk_level = map_risk_level(crs)

    # ── 4. Compute Confidence from all four aggregate scores ─────────────
    # SLex contributes as the exposure component — high benchmark availability
    # in open sources raises the prior and increases confidence in the assessment.
    coverage, agreement, exposure, confidence_pct, confidence_level = compute_confidence(
        slex_agg, ssem_agg, smem_agg, sprob_agg
    )

    # conflicting_evidence: relevant only at HIGH/CRITICAL
    conflicting_evidence = (
        risk_level in ("HIGH", "CRITICAL") and confidence_level == "LOW"
    )

    # ── 6. Write outputs/ summary JSON ────────────────────────────────────
    summary = {
        "stage":            "risk_integration",
        "pipeline_version": "4.2.0",
        "model_id":         model_id,

        # Input sources for traceability
        "inputs": {
            "SLex_source":  lex_summary_path,
            "SSem_source":  dcq_summary_path,
            "SMem_source":  mem_summary_path,
            "SProb_source": stab_summary_path,
        },

        # Primary report fields
        "CRS_raw":    crs_raw,
        "CRS":        crs,
        "risk_level": risk_level,

        # Detector aggregate scores
        "SLex_aggregate":  slex_agg,
        "SSem_aggregate":  ssem_agg,
        "SMem_aggregate":  smem_agg,
        "SProb_aggregate": sprob_agg,

        # Safety override
        "safety_override_active": override_active,

        # Confidence — reliability of the CRS assessment
        # coverage:  share of active model-level detectors (score > 0) / 3
        # agreement: inter-detector consistency = 1 - normalised variance
        # exposure:  SLex / 3 — benchmark availability in open sources
        "confidence_pct":       confidence_pct,
        "confidence_level":     confidence_level,
        "coverage":             coverage,
        "signal_agreement":     agreement,
        "exposure":             exposure,
        "conflicting_evidence": conflicting_evidence,

        # Weights for traceability
        "weights": {
            "w_sem":  W_SEM,
            "w_mem":  W_MEM,
            "w_prob": W_PROB,
            "note":   "SLex excluded from CRS — benchmark-level signal only",
        },

        "outputs": {
            "summary": out_summary,
        },
    }
    write_json(out_summary, summary)

    # ── 7. Write process log entry (logs/) ───────────────────────────────
    log_jsonl(out_log, {
        "status":                 "done",
        "model_id":               model_id,
        "CRS":                    crs,
        "risk_level":             risk_level,
        "safety_override_active": override_active,
        "confidence_pct":         confidence_pct,
        "confidence_level":       confidence_level,
        "conflicting_evidence":   conflicting_evidence,
        "signal_agreement":       agreement,
        "exposure":               exposure,
        "aggregate_scores": {
            "SLex":  slex_agg,
            "SSem":  ssem_agg,
            "SMem":  smem_agg,
            "SProb": sprob_agg,
        },
        "out_summary": out_summary,
    })

    # ── Console output ────────────────────────────────────────────────────
    override_note = (
        f"True — SSem={ssem_agg} SMem={smem_agg} SProb={sprob_agg}, "
        f"CRS floored at {OVERRIDE_FLOOR}"
        if override_active else "False"
    )
    conf_note = (
        " [conflicting evidence — investigate before escalation]"
        if conflicting_evidence else ""
    )
    print("Done.")
    print(f"Model:               {model_id}")
    print(f"Aggregate scores:    SLex={slex_agg}  SSem={ssem_agg}  SMem={smem_agg}  SProb={sprob_agg}")
    print(f"CRS raw:             {crs_raw:.4f}")
    print(f"CRS:                 {crs:.4f}")
    print(f"Risk level:          {risk_level}")
    print(f"Safety override:     {override_note}")
    print(f"Confidence:          {confidence_pct}% ({confidence_level}){conf_note}")
    print(f"Coverage:            {round(coverage * 100)}%")
    print(f"Signal agreement:    {round(agreement * 100)}%")
    print(f"Exposure (SLex):     {round(exposure * 100)}%")
    print(f"Summary:             {out_summary}")
    print(f"Log:                 {out_log}")


if __name__ == "__main__":
    main()
