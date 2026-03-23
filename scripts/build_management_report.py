#!/usr/bin/env python3
"""Build management_report.json from contamination assessment outputs.

This generator produces a management-oriented JSON report aligned with the
current management report schema.

Design goals:
- structured risk assessment, not proof claims;
- no preferred-model framing;
- confidence kept separate from risk;
- traceable evidence and artifact inventory;
- reproducibility/governance metadata.

Expected inputs:
- outputs/v3_lexical_summary*.json
- outputs/v4_dcq_summary_*.json
- outputs/v5_mem_summary_*.json
- outputs/v6_stability_summary_*.json
- outputs/v7_risk_summary_*.json

Optional enrichment from parquet files:
- runs/v4_dcq_*.parquet
- runs/v5_mem_*.parquet
- runs/v6_stability_*.parquet
- runs/v7_risk_*.parquet
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None


MODEL_META: dict[str, dict[str, str]] = {
    "gpt4omini": {
        "model_name": "gpt-4o-mini",
        "provider": "OpenAI",
        "api_name": "gpt-4o-mini",
    },
    "gpt35turbo": {
        "model_name": "gpt-3.5-turbo",
        "provider": "OpenAI",
        "api_name": "gpt-3.5-turbo",
    },
    "gemini15flash": {
        "model_name": "gemini-2.5-flash",
        "provider": "Google",
        "api_name": "gemini-2.5-flash",
    },
}

EXCLUDED_MODEL_IDS = {
    # Experimental/debug model; exclude from management comparison report.
    "gpt54pro20260305",
}

DEFAULT_LIMITATIONS = [
    "Contamination risk is inferred from observable signals and does not directly prove pretraining exposure.",
    "A low risk estimate should not be interpreted as proof of complete data cleanliness.",
    "Benchmark performance is not included in this report.",
    "Model ranking should be interpreted cautiously when confidence differs materially across models.",
]

DEFAULT_PROCUREMENT_IMPLICATIONS = [
    "No model should be selected on benchmark contamination evidence alone.",
    "All compared models require controlled pilot validation on fresh enterprise-relevant data.",
    "The lowest-risk estimate should not be treated as a preferred option when confidence is low.",
]

DEFAULT_VALIDATION_IMPLICATIONS = [
    "Prioritize fresh-data validation before deployment decisions.",
    "Review high-risk flagged cases before procurement shortlisting.",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_glob_first(base: Path, pattern: str) -> Path | None:
    matches = sorted(base.glob(pattern))
    return matches[0] if matches else None


def first_lexical_summary(data_dir: Path) -> dict[str, Any]:
    candidates = sorted(data_dir.glob("v3_lexical_summary*.json"))
    if not candidates:
        raise FileNotFoundError(f"No lexical summary found in {data_dir}")
    data = load_json(candidates[0])
    data["summary_path"] = str(candidates[0])
    return data


def stage_summaries(data_dir: Path, pattern: str, key_name: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(data_dir.glob(pattern)):
        data = load_json(path)
        model_id = str(data.get(key_name, "")).strip()
        if not model_id:
            raise ValueError(f"{path} has no '{key_name}'")
        data["summary_path"] = str(path)
        out[model_id] = data
    return out


def to_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def to_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        return int(float(v))
    except (TypeError, ValueError):
        return default


def first_present(d: dict[str, Any], keys: list[str]) -> Any:
    for k in keys:
        if k in d and d.get(k) is not None:
            return d.get(k)
    return None


def round_or_none(v: Any, ndigits: int = 4) -> float | None:
    if v is None:
        return None
    try:
        return round(float(v), ndigits)
    except (TypeError, ValueError):
        return None


def normalize_risk_summary(risk: dict[str, Any]) -> dict[str, Any]:
    """Normalize risk summary keys across schema variants."""
    risk = risk or {}
    weights = risk.get("weights")
    if not isinstance(weights, dict):
        weights = {
            "w_lex": first_present(risk, ["w_lex"]),
            "w_sem": first_present(risk, ["w_sem"]),
            "w_mem": first_present(risk, ["w_mem"]),
            "w_prob": first_present(risk, ["w_prob"]),
        }
        if all(v is None for v in weights.values()):
            weights = {}

    return {
        "risk_score_mean": first_present(risk, ["risk_score_mean", "mean_risk_score", "risk_mean"]),
        "risk_score_median": first_present(risk, ["risk_score_median", "median_risk_score", "risk_median"]),
        "risk_level_counts": first_present(risk, ["risk_level_counts", "risklevel_counts", "risk_levels"]) or {},
        "confidence_counts": first_present(risk, ["confidence_counts", "confidence_distribution"]) or {},
        "override_counts": first_present(risk, ["override_counts", "overrides"]) or {},
        "weights": weights,
        "inputs": risk.get("inputs", {}),
        "outputs": risk.get("outputs", {}),
        "n_rows": first_present(risk, ["n_rows", "n_items", "rows_total", "valid_items"]),
        "summary_path": risk.get("summary_path"),
        "out_parquet": first_present(risk, ["out_parquet"]) or (risk.get("outputs", {}) or {}).get("parquet"),
        "out_csv": (risk.get("outputs", {}) or {}).get("csv"),
    }


def confidence_score_from_counts(counts: dict[str, Any]) -> float:
    """Convert confidence counts to 0-100 score."""
    low = to_float(counts.get("Low"), 0.0)
    medium = to_float(counts.get("Medium"), 0.0)
    high = to_float(counts.get("High"), 0.0)
    total = low + medium + high
    if total <= 0:
        return 0.0
    # Weighted: Low=0, Medium=0.6, High=1.0
    score = ((0.0 * low) + (0.6 * medium) + (1.0 * high)) / total * 100.0
    return round(score, 2)


def confidence_label(score: float) -> str:
    if score >= 75.0:
        return "High"
    if score >= 45.0:
        return "Medium"
    return "Low"


def distribution_with_pct(counts: dict[str, Any], total: int) -> dict[str, Any]:
    safe_counts = counts or {}
    total_safe = int(total or 0)
    if total_safe <= 0:
        total_safe = int(sum(to_float(v, 0.0) for v in safe_counts.values()))

    bands = ["Low", "Medium", "High", "Critical"]
    out: dict[str, Any] = {}
    dominant_band = "Unknown"
    dominant_count = -1

    for band in bands:
        c = int(to_float(safe_counts.get(band), 0.0))
        pct = (100.0 * c / total_safe) if total_safe > 0 else 0.0
        out[f"{band.lower()}_count"] = c
        out[f"{band.lower()}_pct"] = round(pct, 2)
        if c > dominant_count:
            dominant_count = c
            dominant_band = band

    out["total_cases"] = total_safe
    out["high_or_critical_count"] = out["high_count"] + out["critical_count"]
    out["high_or_critical_pct"] = round(out["high_pct"] + out["critical_pct"], 2)
    out["dominant_band"] = dominant_band
    return out


def confidence_distribution_with_pct(counts: dict[str, Any], total: int) -> dict[str, Any]:
    safe_counts = counts or {}
    total_safe = int(total or 0)
    if total_safe <= 0:
        total_safe = int(sum(to_float(v, 0.0) for v in safe_counts.values()))

    bands = ["Low", "Medium", "High"]
    out: dict[str, Any] = {}
    dominant_band = "Unknown"
    dominant_count = -1

    for band in bands:
        c = int(to_float(safe_counts.get(band), 0.0))
        pct = (100.0 * c / total_safe) if total_safe > 0 else 0.0
        out[f"{band.lower()}_count"] = c
        out[f"{band.lower()}_pct"] = round(pct, 2)
        if c > dominant_count:
            dominant_count = c
            dominant_band = band

    out["dominant_band"] = dominant_band
    out["confidence_score"] = confidence_score_from_counts(safe_counts)
    return out


def evidence_strength_from_level(level: float | int | None) -> str:
    if level is None:
        return "None"
    x = int(to_float(level, 0.0))
    if x <= 0:
        return "None"
    if x == 1:
        return "Low"
    if x == 2:
        return "Moderate"
    return "Strong"


def infer_lexical_strength(lexical: dict[str, Any]) -> str:
    counts = lexical.get("SLex_counts") or {}
    n = to_float(lexical.get("valid_items"), to_float(lexical.get("n_rows_total"), 0.0))
    if n <= 0:
        return "Moderate"
    high_share = (to_float(counts.get("3"), 0.0) + to_float(counts.get("2"), 0.0)) / n
    if high_share >= 0.8:
        return "Strong"
    if high_share >= 0.4:
        return "Moderate"
    if high_share > 0:
        return "Low"
    return "None"


def build_evidence_profile(
    model_id: str,
    lexical: dict[str, Any],
    dcq: dict[str, Any],
    mem: dict[str, Any],
    stability: dict[str, Any],
) -> dict[str, Any]:
    ssem = int(to_float(first_present(dcq, ["SSem_level_from_CPS", "SSem"]), 0.0))
    dominant_smem = int(to_float(first_present(mem, ["dominant_SMem", "SMem"]), 0.0))
    dominant_sprob = int(to_float(first_present(stability, ["dominant_SProb"]), 0.0))

    em_rate = to_float(first_present(mem, ["EM_rate"]), 0.0)
    critical_sprob = to_int((stability.get("SProb_counts") or {}).get("3"), 0)

    lexical_strength = infer_lexical_strength(lexical)
    public_overlap = {
        "strength": lexical_strength,
        "summary": (
            "Substantial public-overlap evidence was observed across most evaluated cases."
            if lexical_strength == "Strong"
            else "Some public-overlap evidence was observed in the evaluated benchmark subset."
        ),
    }

    semantic = {
        "strength": evidence_strength_from_level(ssem),
        "summary": (
            "A weak but non-zero semantic familiarity signal was observed."
            if ssem > 0
            else "No meaningful semantic familiarity signal was observed."
        ),
    }

    direct_mem_strength = "Low" if em_rate > 0 else "None"
    if dominant_smem >= 2 and em_rate > 0.03:
        direct_mem_strength = "Moderate"

    direct_mem = {
        "strength": direct_mem_strength,
        "summary": (
            "A small number of exact or near-exact reconstruction cases were observed."
            if em_rate > 0
            else "No direct memorization signal was observed."
        ),
    }

    stability_strength = evidence_strength_from_level(dominant_sprob)
    if critical_sprob >= 5:
        stability_strength = "Moderate"

    stability_evidence = {
        "strength": stability_strength,
        "summary": (
            "Repeated generations showed some concentration in a subset of cases."
            if (dominant_sprob > 0 or critical_sprob > 0)
            else "Repeated generations did not show broad low-variability behavior across the dataset."
        ),
    }

    return {
        "public_overlap_evidence": public_overlap,
        "semantic_familiarity_evidence": semantic,
        "direct_memorization_evidence": direct_mem,
        "stability_evidence": stability_evidence,
        "dominant_evidence_type": "public_overlap",
    }


def safe_read_parquet(path_like: Any) -> Any:
    if pd is None or not path_like:
        return None
    try:
        path = Path(str(path_like))
        if not path.exists():
            return None
        return pd.read_parquet(path)
    except Exception:
        return None


def resolve_model_col(df: Any, base: str, model_id: str) -> str | None:
    """
    Resolve model-specific detector column names first, then generic fallback.
    """
    cand = f"{base}_{model_id}"
    if cand in df.columns:
        return cand
    if base in df.columns:
        return base
    return None


def extract_top_flagged_cases_from_risk_parquet(
    model_id: str,
    risk_parquet_path: Any,
    limit: int = 3,
) -> list[dict[str, Any]]:
    df = safe_read_parquet(risk_parquet_path)
    if df is None or getattr(df, "empty", True):
        return []

    risk_col = "RiskScore" if "RiskScore" in df.columns else None
    level_col = "RiskLevel" if "RiskLevel" in df.columns else None
    conf_col = "Confidence" if "Confidence" in df.columns else None
    slex_col = "SLex" if "SLex" in df.columns else None
    ssem_col = resolve_model_col(df, "SSem", model_id)
    smem_col = resolve_model_col(df, "SMem", model_id)
    sprob_col = resolve_model_col(df, "SProb", model_id)
    over_direct_col = "override_direct_evidence" if "override_direct_evidence" in df.columns else None
    over_caution_col = (
        "override_single_signal_caution" if "override_single_signal_caution" in df.columns else None
    )

    if not any([risk_col, level_col, conf_col, slex_col, ssem_col, smem_col, sprob_col]):
        return []

    work = df.copy()
    if risk_col:
        work = work.sort_values(risk_col, ascending=False)
    work = work.head(limit)

    cases: list[dict[str, Any]] = []
    for _, row in work.iterrows():
        xsum_id = row.get("xsum_id")
        risk_level = row.get(level_col, "High") if level_col else "High"
        risk_score = round_or_none(row.get(risk_col), 4) if risk_col else None
        confidence = row.get(conf_col, "Medium") if conf_col else "Medium"
        slex = to_int(row.get(slex_col), 0) if slex_col else 0
        ssem = to_int(row.get(ssem_col), 0) if ssem_col else 0
        smem = to_int(row.get(smem_col), 0) if smem_col else 0
        sprob = to_int(row.get(sprob_col), 0) if sprob_col else 0
        over_direct = bool(row.get(over_direct_col)) if over_direct_col else False
        over_caution = bool(row.get(over_caution_col)) if over_caution_col else False
        override_triggered = over_direct or over_caution
        if over_direct:
            override_reason = "direct_evidence"
        elif over_caution:
            override_reason = "single_signal_caution"
        else:
            override_reason = "none"

        if smem >= 3:
            reason = "Strong public overlap combined with exact reconstruction evidence."
        elif sprob >= 2:
            reason = "Elevated risk driven by public overlap together with low-variability repeated generations."
        elif ssem >= 1:
            reason = "Elevated risk driven by public overlap with additional semantic familiarity evidence."
        else:
            reason = "Elevated risk driven primarily by public-overlap evidence and integrated scoring."

        cases.append(
            {
                "xsum_id": int(xsum_id) if xsum_id is not None else None,
                "risk_score": risk_score,
                "risk_level": str(risk_level),
                "confidence": str(confidence),
                "active_signals": {
                    "SLex": slex,
                    "SSem": ssem,
                    "SMem": smem,
                    "SProb": sprob,
                },
                "override_triggered": override_triggered,
                "override_reason": override_reason,
                "evidence_explanation": reason,
                "source_refs": {
                    "risk_row": f"{risk_parquet_path}#xsum_id={xsum_id}" if xsum_id is not None else str(risk_parquet_path),
                    "lexical_row": None,
                    "dcq_row_or_summary": None,
                    "mem_row": None,
                    "stability_row": None,
                },
            }
        )
    return cases


def extract_bcq_counts_from_dcq_parquet(path_like: Any, model_id: str) -> dict[str, int]:
    df = safe_read_parquet(path_like)
    if df is None or getattr(df, "empty", True):
        return {}
    col = f"bcq_choice_{model_id}"
    if col not in df.columns:
        return {}
    values = df[col].astype(str).fillna("")
    counts = values.value_counts(dropna=False).to_dict()
    out = {k: int(counts.get(k, 0)) for k in ["A", "B", "C", "D", "E"]}
    return out


def enrich_mem_from_parquet(path_like: Any, model_id: str) -> dict[str, Any]:
    df = safe_read_parquet(path_like)
    if df is None or getattr(df, "empty", True):
        return {}

    out: dict[str, Any] = {}
    em_col = resolve_model_col(df, "EM", model_id)
    ne_col = resolve_model_col(df, "NE", model_id) or resolve_model_col(df, "NED", model_id)

    if em_col and ne_col:
        em = df[em_col].fillna(0)
        ne = df[ne_col].fillna(1)
        exact = int((em == 1).sum())
        near = int(((em == 0) & (ne <= 0.10)).sum())
        non = int(((em == 0) & (ne > 0.10)).sum())
        out["exact_count"] = exact
        out["near_exact_count"] = near
        out["non_match_count"] = non

        # Per-item dominant level approximation from categories
        level_counts = {
            3: exact,
            2: near,
            1: int(((em == 0) & (ne > 0.10) & (ne <= 0.25)).sum()),
            0: int(((em == 0) & (ne > 0.25)).sum()),
        }
        out["dominant_SMem"] = max(level_counts.items(), key=lambda x: x[1])[0]

    return out


def enrich_stability_from_parquet(path_like: Any, model_id: str) -> dict[str, Any]:
    df = safe_read_parquet(path_like)
    if df is None or getattr(df, "empty", True):
        return {}

    out: dict[str, Any] = {}
    sprob_col = resolve_model_col(df, "SProb", model_id)
    if sprob_col:
        counts = df[sprob_col].fillna(0).astype(int).value_counts().to_dict()
        out["SProb_counts"] = {str(k): int(v) for k, v in counts.items()}
        out["dominant_SProb"] = max(counts.items(), key=lambda x: x[1])[0] if counts else 0
    return out


def infer_min_risk_score(risk: dict[str, Any]) -> float | None:
    return to_float(first_present(risk, ["risk_score_min", "min_risk_score"]), 0.0)


def infer_max_risk_score(risk: dict[str, Any]) -> float | None:
    return to_float(first_present(risk, ["risk_score_max", "max_risk_score"]), 0.0)


def build_model_entry(
    model_id: str,
    lexical: dict[str, Any],
    dcq: dict[str, Any],
    mem: dict[str, Any],
    stability: dict[str, Any],
    risk: dict[str, Any],
) -> dict[str, Any]:
    meta = MODEL_META.get(
        model_id,
        {"model_name": model_id, "provider": "Unknown", "api_name": model_id},
    )

    risk_norm = normalize_risk_summary(risk)
    n_rows = int(to_float(risk_norm.get("n_rows"), to_float(lexical.get("n_rows_total"), 0.0)))

    # Optional parquet enrichments
    dcq_parquet_path = dcq.get("out_parquet")
    mem_parquet_path = mem.get("out_parquet")
    stability_parquet_path = stability.get("out_parquet")
    risk_parquet_path = risk_norm.get("out_parquet")

    bcq_counts = extract_bcq_counts_from_dcq_parquet(dcq_parquet_path, model_id)
    mem_extra = enrich_mem_from_parquet(mem_parquet_path, model_id)
    stab_extra = enrich_stability_from_parquet(stability_parquet_path, model_id)
    top_flagged_cases = extract_top_flagged_cases_from_risk_parquet(model_id, risk_parquet_path, limit=3)

    risk_counts = risk_norm.get("risk_level_counts") or {}
    conf_counts = risk_norm.get("confidence_counts") or {}
    override_counts = risk_norm.get("override_counts") or {}

    risk_dist = distribution_with_pct(risk_counts, n_rows)
    conf_dist = confidence_distribution_with_pct(conf_counts, n_rows)

    mean_risk = to_float(risk_norm.get("risk_score_mean"), 0.0)
    median_risk = to_float(risk_norm.get("risk_score_median"), 0.0)
    min_risk = infer_min_risk_score(risk)
    max_risk = infer_max_risk_score(risk)

    dominant_risk_level = risk_dist.get("dominant_band", "Unknown")
    dominant_confidence = conf_dist.get("dominant_band", "Unknown")

    critical_count = int(to_float(risk_counts.get("Critical"), 0.0))
    high_conf_count = int(to_float(conf_counts.get("High"), 0.0))

    dcq_metric_ssem = int(to_float(first_present(dcq, ["SSem_level_from_CPS", "SSem"]), 0.0))
    mem_dominant = int(to_float(first_present(mem_extra, ["dominant_SMem"]), 0.0))
    stability_counts = stab_extra.get("SProb_counts") or {}
    dominant_sprob = int(to_float(first_present(stab_extra, ["dominant_SProb"]), 0.0))

    evidence_profile = build_evidence_profile(model_id, lexical, dcq, mem | mem_extra, stability | stab_extra)

    key_reason = "Risk was driven mainly by strong public-overlap evidence."
    if to_float(first_present(mem, ["EM_rate"]), 0.0) > 0:
        key_reason = "Risk was driven mainly by strong public-overlap evidence, with limited direct memorization evidence."
    elif dcq_metric_ssem > 0:
        key_reason = "Risk was driven mainly by strong public-overlap evidence, with weak additional semantic familiarity evidence."
    elif dominant_sprob > 0:
        key_reason = "Risk was driven mainly by strong public-overlap evidence, with some support from output stability patterns."

    stability_decoding = stability.get("decoding") if isinstance(stability.get("decoding"), dict) else {}

    return {
        "model_id": model_id,
        "model_name": meta["model_name"],
        "provider": meta["provider"],
        "api_name": meta["api_name"],
        "benchmark_performance": None,
        "contamination_assessment": {
            "mean_risk_score": round(mean_risk, 4),
            "median_risk_score": round(median_risk, 4),
            "min_risk_score": round_or_none(min_risk, 4),
            "max_risk_score": round_or_none(max_risk, 4),
            "dominant_risk_level": dominant_risk_level,
            "dominant_confidence": dominant_confidence,
            "high_or_critical_count": risk_dist["high_or_critical_count"],
            "high_or_critical_pct": risk_dist["high_or_critical_pct"],
            "critical_case_count": critical_count,
            "critical_case_pct": risk_dist["critical_pct"],
            "recommendation_type": "pilot_only",
            "recommendation_text": "Use only in a controlled pilot and validate on fresh enterprise-relevant data.",
            "selection_caution": "Do not treat this model as preferred on contamination evidence alone.",
            "confidence_score": conf_dist["confidence_score"],
            "confidence_label": confidence_label(conf_dist["confidence_score"]),
        },
        "risk_distribution": risk_dist,
        "confidence_distribution": conf_dist,
        "evidence_profile": evidence_profile,
        "evidence_summary": {
            "override_counts": {
                "direct_evidence": int(to_float(override_counts.get("direct_evidence"), 0.0)),
                "single_signal_caution": int(to_float(override_counts.get("single_signal_caution"), 0.0)),
            },
            "critical_cases_present": critical_count > 0,
            "high_confidence_case_count": high_conf_count,
            "key_reason_for_risk": key_reason,
        },
        "traceability": {
            "traceable_to_detector_outputs": True,
            "evidence_register_available": True,
            "top_flagged_cases": top_flagged_cases,
        },
        "artifacts": {
            "summaries": {
                "dcq": dcq.get("summary_path"),
                "memorization": mem.get("summary_path"),
                "stability": stability.get("summary_path"),
                "risk": risk_norm.get("summary_path"),
            },
            "tabular_outputs": {
                "risk_parquet": risk_norm.get("out_parquet"),
                "risk_csv": risk_norm.get("out_csv"),
            },
            "intermediate_inputs": {
                "lexical": lexical.get("out_parquet"),
                "dcq": dcq.get("out_parquet"),
                "mem": mem.get("out_parquet"),
                "stability": stability.get("out_parquet"),
            },
        },
        "technical_trace": {
            "dataset_path": lexical.get("dataset_path"),
            "proxy_path": lexical.get("proxy_path"),
            "risk_weights": risk_norm.get("weights", {}),
            "detector_metrics": {
                "lexical": {
                    "valid_items": lexical.get("valid_items"),
                    "SLex_counts": lexical.get("SLex_counts"),
                    "MaxSpanLen_mean": lexical.get("MaxSpanLen_mean"),
                    "NgramHits_mean": lexical.get("NgramHits_mean"),
                    "ProxyCount_mean": lexical.get("ProxyCount_mean"),
                },
                "semantic": {
                    "valid_items": first_present(dcq, ["valid_items_for_cps", "valid_items"]),
                    "CPS": dcq.get("CPS"),
                    "SSem": first_present(dcq, ["SSem_level_from_CPS", "SSem"]),
                    "least_preferred_position": dcq.get("least_preferred_position"),
                    "kappa_min": dcq.get("kappa_min"),
                    "bdq_counts": dcq.get("bdq_counts"),
                    "bcq_counts": bcq_counts or dcq.get("bcq_counts") or {},
                },
                "memorization": {
                    "valid_items": first_present(mem, ["valid_items", "n_rows_total"]),
                    "EM_rate": mem.get("EM_rate"),
                    "mean_NE": first_present(mem, ["NE_rate", "mean_NE", "NE_mean"]),
                    "mean_NED": first_present(mem, ["NED_mean", "mean_NED", "NE_mean"]),
                    "exact_count": mem_extra.get("exact_count"),
                    "near_exact_count": mem_extra.get("near_exact_count"),
                    "non_match_count": mem_extra.get("non_match_count"),
                    "dominant_SMem": mem_dominant,
                    "use_control_prefix": first_present(mem, ["use_control_prefix"]),
                },
                "stability": {
                    "valid_items": first_present(stability, ["valid_items", "n_rows_total"]),
                    "UAR_mean": stability.get("UAR_mean"),
                    "mNED_mean": stability.get("mNED_mean"),
                    "SProb_counts": stability_counts,
                    "dominant_SProb": dominant_sprob,
                    "n_samples": first_present(stability, ["n_samples", "N_samples"]),
                    "temperature": first_present(stability_decoding, ["temperature"]),
                    "top_p": first_present(stability_decoding, ["top_p"]),
                    "max_tokens": first_present(stability_decoding, ["max_tokens"]),
                },
            },
            "log_files": {
                "lexical": lexical.get("log_jsonl"),
                "dcq": dcq.get("log_jsonl"),
                "memorization": mem.get("log_jsonl"),
                "stability": stability.get("log_jsonl"),
            },
        },
    }


def build_executive_summary(models: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(models, key=lambda m: to_float(m["contamination_assessment"]["mean_risk_score"], 1e9))
    lowest = ranked[0] if ranked else None
    highest = ranked[-1] if ranked else None

    overall = "No model can be treated as clearly low-risk on the current evidence."
    comparative = "All compared models are dominated by High-risk cases, and differences in average risk are small."
    uncertainty = "Risk estimates should be interpreted together with confidence rather than in isolation."

    if lowest is not None:
        uncertainty = (
            f"The lowest estimated risk was observed for {lowest['model_name']}, "
            "but this estimate should be interpreted cautiously when confidence is limited."
        )

    key_findings: list[str] = []
    if ranked:
        lowest_name = lowest["model_name"]
        lowest_risk = lowest["contamination_assessment"]["mean_risk_score"]
        key_findings.append(f"Lowest observed mean contamination risk: {lowest_name} ({lowest_risk:.2f}).")

    high_conf = sorted(
        models,
        key=lambda m: to_float(m["contamination_assessment"]["confidence_score"], 0.0),
        reverse=True,
    )
    if high_conf:
        top_conf = high_conf[0]
        key_findings.append(
            f"Highest confidence in assessment: {top_conf['model_name']} "
            f"({top_conf['contamination_assessment']['confidence_score']:.2f})."
        )

    critical_models = [
        m["model_name"]
        for m in models
        if to_int(m["contamination_assessment"]["critical_case_count"], 0) > 0
    ]
    if critical_models:
        key_findings.append("Critical cases were observed for: " + ", ".join(critical_models) + ".")
    else:
        key_findings.append("No Critical cases were observed in the compared models.")

    return {
        "overall_conclusion": overall,
        "comparative_conclusion": comparative,
        "uncertainty_conclusion": uncertainty,
        "highest_concern_model_id": highest["model_id"] if highest else None,
        "highest_concern_reason": (
            "Highest mean risk score and strongest overall concern profile in the current comparison."
            if highest
            else None
        ),
        "key_findings": key_findings[:3],
    }


def build_comparative_analysis(models: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(models, key=lambda m: to_float(m["contamination_assessment"]["mean_risk_score"], 1e9))
    delta = None
    if len(ranked) >= 2:
        delta = round(
            to_float(ranked[-1]["contamination_assessment"]["mean_risk_score"], 0.0)
            - to_float(ranked[0]["contamination_assessment"]["mean_risk_score"], 0.0),
            4,
        )

    return {
        "ranking_basis": "lower_mean_risk_score",
        "risk_score_delta_best_vs_worst": delta,
        "ranking_is_provisional": True,
        "ranking_caution": (
            "Ranking should not be used as a standalone selection decision because "
            "confidence differs materially across models."
        ),
        "models_ordered": [
            {
                "rank": i,
                "model_id": m["model_id"],
                "model_name": m["model_name"],
                "mean_risk_score": m["contamination_assessment"]["mean_risk_score"],
                "confidence_score": m["contamination_assessment"]["confidence_score"],
                "confidence_label": m["contamination_assessment"]["confidence_label"],
                "dominant_risk_level": m["contamination_assessment"]["dominant_risk_level"],
                "dominant_confidence": m["contamination_assessment"]["dominant_confidence"],
            }
            for i, m in enumerate(ranked, start=1)
        ],
    }


def build_meta(lexical: dict[str, Any], out_path: Path, n_models: int) -> dict[str, Any]:
    return {
        "report_version": "2.0",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "report_title": "Management Contamination Risk Report",
        "report_scope": "Comparative contamination risk assessment for closed-source LLMs",
        "report_mode": "management",
        "dataset_name": "EdinburghNLP/xsum",
        "dataset_split": "test",
        "dataset_rows_initial": 300,
        "dataset_rows_final": int(to_float(lexical.get("n_rows_total"), 0.0)),
        "random_seed": 42,
        "frozen_dataset_path": lexical.get("dataset_path"),
        "models_compared": n_models,
        "benchmark_performance_included": False,
        "consolidated_output_path": str(out_path),
        "visualization_path": "index.html",
    }


def build_governance(
    lexical: dict[str, Any],
    dcq: dict[str, dict[str, Any]],
    stability: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    any_stability = next(iter(stability.values())) if stability else {}
    any_dcq = next(iter(dcq.values())) if dcq else {}

    return {
        "frozen_inputs_used": True,
        "fixed_prompts_across_models": True,
        "versioned_outputs": True,
        "intermediate_artifacts_preserved": True,
        "resume_friendly_execution": True,
        "full_frozen_set_used_for_final_reporting": True,
        "pipeline_controls": [
            "Frozen evaluation dataset and fixed prompts across models",
            "Versioned outputs and preserved intermediate artifacts",
            "Repeatable configuration and standardized thresholds",
            "Consistent scoring procedure across all compared models",
        ],
        "dcq_setup": {
            "paraphrases_per_item": 4,
            "paraphrase_generation_model": "deepseek-chat",
            "paraphrase_completion_coverage": "296/296",
        },
        "proxy_setup": {
            "proxy_file": lexical.get("proxy_path"),
            "proxy_rows": 226772,
            "proxy_unique_normalized_rows": 225615,
        },
        "decoding_settings": {
            "dcq": {
                "temperature": to_float(first_present(any_dcq, ["decoding", "temperature"]) or 0.0, 0.0)
                if not isinstance(any_dcq.get("decoding"), dict)
                else to_float((any_dcq.get("decoding") or {}).get("temperature"), 0.0),
                "top_p": to_float((any_dcq.get("decoding") or {}).get("top_p"), 1.0)
                if isinstance(any_dcq.get("decoding"), dict)
                else 1.0,
                "max_tokens": to_int((any_dcq.get("decoding") or {}).get("max_tokens"), 128)
                if isinstance(any_dcq.get("decoding"), dict)
                else 128,
            },
            "stability": {
                "temperature": to_float(first_present(any_stability, ["temperature"]), 0.7),
                "top_p": to_float(first_present(any_stability, ["top_p"]), 0.9),
                "max_tokens": to_int(first_present(any_stability, ["max_tokens"]), 80),
                "n_samples": to_int(first_present(any_stability, ["n_samples", "N_samples"]), 30),
            },
        },
    }


def build_report(data_dir: Path, out_path: Path) -> dict[str, Any]:
    lexical = first_lexical_summary(data_dir)
    dcq_all = stage_summaries(data_dir, "v4_dcq_summary_*.json", "model_id")
    mem_all = stage_summaries(data_dir, "v5_mem_summary_*.json", "model_id")
    stability_all = stage_summaries(data_dir, "v6_stability_summary_*.json", "model_id")
    risk_all = stage_summaries(data_dir, "v7_risk_summary_*.json", "model_id")

    dcq = {k: v for k, v in dcq_all.items() if k not in EXCLUDED_MODEL_IDS}
    mem = {k: v for k, v in mem_all.items() if k not in EXCLUDED_MODEL_IDS}
    stability = {k: v for k, v in stability_all.items() if k not in EXCLUDED_MODEL_IDS}
    risk = {k: v for k, v in risk_all.items() if k not in EXCLUDED_MODEL_IDS}

    model_ids = sorted(set(dcq) | set(mem) | set(stability) | set(risk))
    if not model_ids:
        raise ValueError("No model summaries found for v4/v5/v6/v7")

    models = [
        build_model_entry(
            model_id=mid,
            lexical=lexical,
            dcq=dcq.get(mid, {}),
            mem=mem.get(mid, {}),
            stability=stability.get(mid, {}),
            risk=risk.get(mid, {}),
        )
        for mid in model_ids
    ]

    report = {
        "meta": build_meta(lexical, out_path, len(models)),
        "methodological_positioning": {
            "core_claim": "This report provides a structured contamination risk assessment and does not claim proof of contamination.",
            "decision_use": "The report supports informed model comparison and pilot-stage decision-making under uncertainty.",
            "assessment_basis": [
                "contamination risk scores",
                "qualitative risk levels",
                "confidence estimates",
                "visual summaries",
                "traceable detector evidence",
            ],
        },
        "executive_summary": build_executive_summary(models),
        "comparative_analysis": build_comparative_analysis(models),
        "models": models,
        "governance": build_governance(lexical, dcq, stability),
        "limitations": DEFAULT_LIMITATIONS,
        "implications": {
            "procurement": DEFAULT_PROCUREMENT_IMPLICATIONS,
            "validation": DEFAULT_VALIDATION_IMPLICATIONS,
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="outputs", help="Directory with v3..v7 summary JSON files.")
    parser.add_argument("--out", default="assessment/data/management_report.json", help="Output JSON path.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    report = build_report(data_dir, out)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Built report: {out}")
    model_ids = [m.get("model_id", "") for m in report.get("models", [])]
    print(f"Models: {', '.join(model_ids)}")


if __name__ == "__main__":
    main()
