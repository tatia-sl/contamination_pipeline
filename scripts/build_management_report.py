#!/usr/bin/env python3
"""Build management-ready assessment/data/management_report.json from summary files."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def first_lexical_summary(data_dir: Path) -> dict[str, Any]:
    candidates = sorted(data_dir.glob("v3_lexical_summary*.json"))
    if not candidates:
        raise FileNotFoundError(f"No lexical summary found in {data_dir}")
    return load_json(candidates[0])


def stage_summaries(data_dir: Path, pattern: str, key_name: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(data_dir.glob(pattern)):
        data = load_json(path)
        model_id = str(data.get(key_name, "")).strip()
        if not model_id:
            raise ValueError(f"{path} has no '{key_name}'")
        out[model_id] = data
    return out


def to_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def risk_label(score: float) -> str:
    if score < 25.0:
        return "Low"
    if score < 50.0:
        return "Moderate"
    if score < 75.0:
        return "Elevated"
    return "Critical"


def confidence_score_from_counts(counts: dict[str, Any]) -> float:
    counts = counts or {}
    high = to_float(counts.get("High"), 0.0)
    medium = to_float(counts.get("Medium"), 0.0)
    low = to_float(counts.get("Low"), 0.0)
    total = high + medium + low
    if total <= 0:
        return 0.0
    return round((high * 100.0 + medium * 60.0 + low * 30.0) / total, 2)


def confidence_label(score: float) -> str:
    if score >= 85.0:
        return "High"
    if score >= 60.0:
        return "Medium"
    return "Low"


def distribution_with_pct(counts: dict[str, Any], total: int) -> dict[str, Any]:
    safe_counts = counts or {}
    normalized_counts = dict(safe_counts)
    if "Medium" in normalized_counts and "Moderate" not in normalized_counts:
        normalized_counts["Moderate"] = normalized_counts.pop("Medium")

    total_safe = int(total or 0)
    if total_safe <= 0:
        total_safe = int(sum(to_float(v, 0.0) for v in normalized_counts.values()))

    bands = ["Low", "Moderate", "High", "Critical"]
    out: dict[str, Any] = {"total_cases": total_safe}

    dominant_band = "Unknown"
    dominant_count = -1
    high_or_critical_count = 0

    for band in bands:
        c = int(to_float(normalized_counts.get(band), 0.0))
        pct = (100.0 * c / total_safe) if total_safe > 0 else 0.0
        out[f"{band.lower()}_count"] = c
        out[f"{band.lower()}_pct"] = round(pct, 2)

        if c > dominant_count:
            dominant_count = c
            dominant_band = band

        if band in ("High", "Critical"):
            high_or_critical_count += c

    out["high_or_critical_count"] = high_or_critical_count
    out["high_or_critical_pct"] = (
        round(100.0 * high_or_critical_count / total_safe, 2) if total_safe > 0 else 0.0
    )
    out["dominant_band"] = dominant_band
    return out


def recommendation_type(integrated_risk_score: float, min_score_among_models: float, has_critical: bool) -> str:
    if integrated_risk_score <= min_score_among_models and integrated_risk_score < 60.0:
        return "shortlist"
    if has_critical or integrated_risk_score >= 60.0:
        return "pilot_only"
    return "monitor"


def recommendation_text(rec_type: str) -> str:
    if rec_type == "shortlist":
        return "Preferred candidate for shortlist and controlled deployment validation."
    if rec_type == "pilot_only":
        return "Use only in a controlled pilot and validate on fresh enterprise-relevant data."
    return "Further validation required before relying on benchmark results."


def business_reliability(integrated_risk_score: float, min_score_among_models: float, confidence_score: float) -> str:
    if confidence_score < 60.0:
        return "Low"
    if integrated_risk_score <= min_score_among_models:
        return "Medium"
    return "Low"


def calc_lexical_score(lexical: dict[str, Any]) -> float:
    counts = lexical.get("SLex_counts") or {}
    total = int(to_float(lexical.get("valid_items"), 0.0))
    if total <= 0:
        total = int(sum(to_float(v, 0.0) for v in counts.values()))
    if total <= 0:
        return 0.0
    weighted = 0.0
    for level, count in counts.items():
        weighted += to_float(level, 0.0) * to_float(count, 0.0)
    return round((weighted / total) / 3.0 * 100.0, 2)


def calc_mem_score(em_rate: float, ne_mean: float) -> float:
    score = (em_rate * 60.0 + max(0.0, 1.0 - ne_mean) * 40.0) * 100.0
    return round(min(max(score, 0.0), 100.0), 2)


def calc_stability_score(uar_mean: float, mned_mean: float) -> float:
    score = (max(0.0, 1.0 - uar_mean) * 70.0 + max(0.0, mned_mean) * 30.0) * 100.0
    return round(min(max(score, 0.0), 100.0), 2)


def build_model_card(
    model_id: str,
    lexical: dict[str, Any],
    dcq: dict[str, Any],
    mem: dict[str, Any],
    stability: dict[str, Any],
    risk: dict[str, Any],
    management_report_path: str,
    min_risk_score: float,
) -> dict[str, Any]:
    model_name = (
        dcq.get("model_name")
        or mem.get("model_name")
        or stability.get("model_name")
        or model_id
    )

    cps = to_float(dcq.get("CPS"), 0.0)
    em_rate = to_float(mem.get("EM_rate"), 0.0)
    ne_mean = to_float(mem.get("NE_mean"), 0.0)
    uar_mean = to_float(stability.get("UAR_mean"), 0.0)
    mned_mean = to_float(stability.get("mNED_mean"), 0.0)

    risk_score_mean = to_float(risk.get("risk_score_mean"), 0.0)
    risk_score_median = to_float(risk.get("risk_score_median"), 0.0)
    risk_counts = risk.get("risk_level_counts") or {}
    conf_counts = risk.get("confidence_counts") or {}
    over_counts = risk.get("override_counts") or {}

    n_rows = int(to_float(risk.get("n_rows"), to_float(lexical.get("n_rows_total"), 0.0)))

    conf_score = confidence_score_from_counts(conf_counts)
    has_critical = int(to_float(risk_counts.get("Critical"), 0.0)) > 0
    rec_type = recommendation_type(risk_score_mean, min_risk_score, has_critical)

    high_conf_count = int(to_float(conf_counts.get("High"), 0.0))
    med_conf_count = int(to_float(conf_counts.get("Medium"), 0.0))
    low_conf_count = int(to_float(conf_counts.get("Low"), 0.0))

    lexical_score = calc_lexical_score(lexical)
    mem_score = calc_mem_score(em_rate, ne_mean)
    stab_score = calc_stability_score(uar_mean, mned_mean)

    return {
        "model_id": model_id,
        "model_name": model_name,
        "benchmark_performance": None,
        "contamination_assessment": {
            "integrated_risk_score": round(risk_score_mean, 4),
            "integrated_risk_label": risk_label(risk_score_mean),
            "confidence_score": conf_score,
            "confidence_label": confidence_label(conf_score),
            "business_reliability": business_reliability(risk_score_mean, min_risk_score, conf_score),
            "recommendation_type": rec_type,
            "recommendation": recommendation_text(rec_type),
        },
        "signal_profile": {
            "lexical": {
                "score": lexical_score,
                "label": "Elevated" if lexical_score >= 50.0 else "Moderate",
                "source_fields": {
                    "valid_items": lexical.get("valid_items"),
                    "MaxSpanLen_mean": lexical.get("MaxSpanLen_mean"),
                    "SLex_counts": lexical.get("SLex_counts"),
                },
            },
            "semantic": {
                "score": round(cps * 100.0, 2),
                "label": risk_label(cps * 100.0),
                "source_fields": {
                    "CPS": dcq.get("CPS"),
                    "SSem_level_from_CPS": dcq.get("SSem_level_from_CPS"),
                },
            },
            "memorization": {
                "score": mem_score,
                "label": risk_label(mem_score),
                "source_fields": {
                    "EM_rate": mem.get("EM_rate"),
                    "NE_mean": mem.get("NE_mean"),
                },
            },
            "stability": {
                "score": stab_score,
                "label": risk_label(stab_score),
                "source_fields": {
                    "UAR_mean": stability.get("UAR_mean"),
                    "mNED_mean": stability.get("mNED_mean"),
                },
            },
        },
        "risk_distribution": distribution_with_pct(risk_counts, n_rows),
        "evidence_summary": {
            "override_counts": over_counts,
            "confidence_counts": conf_counts,
            "high_confidence_count": high_conf_count,
            "medium_confidence_count": med_conf_count,
            "low_confidence_count": low_conf_count,
        },
        "technical_trace": {
            "proxy_path": lexical.get("proxy_path"),
            "inputs": risk.get("inputs", {}),
            "out_parquet": {
                "lexical": lexical.get("out_parquet"),
                "dcq": dcq.get("out_parquet"),
                "mem": mem.get("out_parquet"),
                "stability": stability.get("out_parquet"),
                "risk": (risk.get("outputs") or {}).get("parquet"),
            },
            "log_jsonl": {
                "lexical": lexical.get("log_jsonl"),
                "dcq": dcq.get("log_jsonl"),
                "mem": mem.get("log_jsonl"),
                "stability": stability.get("log_jsonl"),
            },
            "CPS": dcq.get("CPS"),
            "EM_rate": mem.get("EM_rate"),
            "NE_mean": mem.get("NE_mean"),
            "UAR_mean": stability.get("UAR_mean"),
            "mNED_mean": stability.get("mNED_mean"),
            "risk_score_mean": risk.get("risk_score_mean"),
            "risk_score_median": risk.get("risk_score_median"),
            "management_report": management_report_path,
        },
    }


def build_executive_summary(models: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(models, key=lambda m: to_float(m["contamination_assessment"].get("integrated_risk_score")))
    preferred = ranked[0] if ranked else None
    highest_risk = ranked[-1] if ranked else None

    findings: list[str] = []
    if preferred:
        findings.append(
            f"Preferred model by integrated risk is {preferred['model_name']} "
            f"({preferred['contamination_assessment']['integrated_risk_score']:.2f})."
        )
    if highest_risk and preferred and highest_risk["model_id"] != preferred["model_id"]:
        findings.append(
            f"Highest integrated risk is {highest_risk['model_name']} "
            f"({highest_risk['contamination_assessment']['integrated_risk_score']:.2f})."
        )
    high_critical = [
        (m["model_name"], to_float(m["risk_distribution"].get("high_or_critical_pct"), 0.0))
        for m in models
    ]
    if high_critical:
        dominant = max(high_critical, key=lambda x: x[1])
        findings.append(f"Max High/Critical share is {dominant[1]:.2f}% for {dominant[0]}.")

    return {
        "preferred_model_id": preferred["model_id"] if preferred else None,
        "preferred_model_name": preferred["model_name"] if preferred else None,
        "highest_risk_model_id": highest_risk["model_id"] if highest_risk else None,
        "highest_risk_model_name": highest_risk["model_name"] if highest_risk else None,
        "key_findings": findings[:3],
    }


def build_comparative_analysis(models: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(models, key=lambda m: to_float(m["contamination_assessment"].get("integrated_risk_score")))

    rows = []
    for i, m in enumerate(ranked, start=1):
        ass = m["contamination_assessment"]
        rows.append(
            {
                "rank_by_lower_risk": i,
                "model_id": m["model_id"],
                "model_name": m["model_name"],
                "integrated_risk_score": ass.get("integrated_risk_score"),
                "integrated_risk_label": ass.get("integrated_risk_label"),
                "confidence_score": ass.get("confidence_score"),
                "recommendation_type": ass.get("recommendation_type"),
            }
        )

    risk_delta = None
    if len(ranked) >= 2:
        risk_delta = round(
            to_float(ranked[-1]["contamination_assessment"].get("integrated_risk_score"))
            - to_float(ranked[0]["contamination_assessment"].get("integrated_risk_score")),
            4,
        )

    return {
        "ranking": rows,
        "risk_score_delta_best_vs_worst": risk_delta,
    }


def build_report(data_dir: Path, out_path: Path) -> dict[str, Any]:
    lexical = first_lexical_summary(data_dir)
    dcq = stage_summaries(data_dir, "v4_dcq_summary_*.json", "model_id")
    mem = stage_summaries(data_dir, "v5_mem_summary_*.json", "model_id")
    stability = stage_summaries(data_dir, "v6_stability_summary_*.json", "model_id")
    risk = stage_summaries(data_dir, "v7_risk_summary_*.json", "model_id")

    # Prefer models that already reached risk integration stage.
    model_ids = sorted(risk.keys())
    if not model_ids:
        model_ids = sorted(set(dcq) | set(mem) | set(stability))
    if not model_ids:
        raise ValueError("No model summaries found for v4/v5/v6/v7")

    min_risk_score = min(
        to_float((risk.get(mid, {}) or {}).get("risk_score_mean"), 0.0) for mid in model_ids
    )

    model_cards = [
        build_model_card(
            model_id=mid,
            lexical=lexical,
            dcq=dcq.get(mid, {}),
            mem=mem.get(mid, {}),
            stability=stability.get(mid, {}),
            risk=risk.get(mid, {}),
            management_report_path=str(out_path),
            min_risk_score=min_risk_score,
        )
        for mid in model_ids
    ]

    executive = build_executive_summary(model_cards)
    comparative = build_comparative_analysis(model_cards)

    preferred_id = executive.get("preferred_model_id")
    decision_items = []
    for m in model_cards:
        ass = m["contamination_assessment"]
        decision_items.append(
            {
                "model_id": m["model_id"],
                "model_name": m["model_name"],
                "recommendation_type": ass.get("recommendation_type"),
                "business_reliability": ass.get("business_reliability"),
                "integrated_risk_score": ass.get("integrated_risk_score"),
                "is_preferred": m["model_id"] == preferred_id,
            }
        )

    evidence = []
    for m in model_cards:
        evidence.append(
            {
                "model_id": m["model_id"],
                "model_name": m["model_name"],
                "risk_label": m["contamination_assessment"].get("integrated_risk_label"),
                "high_or_critical_pct": m["risk_distribution"].get("high_or_critical_pct"),
                "override_counts": m["evidence_summary"].get("override_counts"),
                "confidence_counts": m["evidence_summary"].get("confidence_counts"),
                "dominant_evidence": "Signal-level convergence from lexical, semantic, memorization, and stability detectors.",
                "why_it_matters": "Helps assess whether benchmark gains may be inflated by prior exposure risk.",
            }
        )

    preferred_name = executive.get("preferred_model_name") or "Preferred model"
    highest_name = executive.get("highest_risk_model_name") or "Highest-risk model"

    return {
        "meta": {
            "generated_at": date.today().isoformat(),
            "title": "Management Contamination Report",
            "dataset_name": "XSum evaluation subset",
            "benchmark_name": "Comparative LLM benchmark review",
            "notes": "Automatically generated from contamination assessment outputs.",
            "dataset_path": lexical.get("dataset_path"),
            "dataset_rows": lexical.get("n_rows_total"),
            "models_compared": len(model_cards),
        },
        "executive_summary": executive,
        "decision_summary": {
            "preferred_model_id": preferred_id,
            "model_decisions": decision_items,
        },
        "models": model_cards,
        "comparative_analysis": comparative,
        "evidence_highlights": evidence,
        "auditability": {
            "dataset_frozen_path": lexical.get("dataset_path"),
            "proxy_path": lexical.get("proxy_path"),
            "controls": [
                "Frozen evaluation dataset and fixed prompts across models",
                "Versioned outputs and preserved intermediate artifacts",
                "Repeatable configuration and standardized thresholds",
                "Consistent scoring procedure across all compared models",
            ],
            "artifacts": {
                "lexical_out_parquet": lexical.get("out_parquet"),
                "lexical_log_jsonl": lexical.get("log_jsonl"),
                "management_report_path": str(out_path),
            },
            "decoding_by_stage": {
                "dcq": {mid: (dcq.get(mid, {}) or {}).get("decoding") for mid in model_ids},
                "mem": {mid: (mem.get(mid, {}) or {}).get("decoding") for mid in model_ids},
                "stability": {mid: (stability.get(mid, {}) or {}).get("decoding") for mid in model_ids},
            },
        },
        "implications": {
            "procurement": [
                f"{preferred_name} is the stronger shortlist candidate in the current comparison.",
                f"{highest_name} should not be selected on benchmark results alone.",
                "Both models require validation on fresh enterprise-relevant data before deployment.",
            ],
            "deployment_guidance": (
                "Use shortlist models for controlled deployment; keep pilot_only models in "
                "limited-scope evaluation with additional guardrails."
            ),
            "governance_note": (
                "Integrated risk is contamination-focused and should be combined with quality, "
                "cost, and compliance criteria before final model selection."
            ),
        },
        "limitations": [
            "Benchmark performance is not available in this report and is left null.",
            "Risk interpretation is derived from aggregated detector summaries, not raw instance adjudication.",
            "Signal labels are management-layer abstractions over technical metrics.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="assessment/data")
    parser.add_argument("--out", default="assessment/data/management_report.json")
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
