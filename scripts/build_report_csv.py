#!/usr/bin/env python3
"""
scripts/build_report_csv.py

Collects data from all pipeline stage summary JSONs and assembles
a single CSV file used by the HTML report page.

Input sources per model:
    outputs/v3_lexical_summary.json               — SLex (shared, benchmark-level)
    outputs/v4_dcq_summary_{model_id}.json         — SSem
    outputs/v5_mem_summary_{model_id}.json         — SMem
    outputs/v6_stability_summary_{model_id}.json   — SProb + UAR/mNED stats
    outputs/v7_risk_summary_{model_id}.json        — CRS, Confidence, Risk level

Output:
    outputs/report_data.csv                        — one row per model

CSV columns:
    # Identification
    model_id, run_date, pipeline_version, benchmark

    # Benchmark exposure (shared)
    # MaxSpanLen/NgramHits/ProxyCount: read as exact key or _mean variant
    SLex_aggregate, MaxSpanLen, NgramHits, ProxyCount, SLex_label,
    sources_reviewed, lexical_items_found, lexical_items_total,
    lexical_items_found_pct, lexical_strong_overlap_items,
    lexical_strong_overlap_pct

    # Detector scores
    SSem_aggregate, SMem_aggregate, SProb_aggregate

    # CRS & Risk
    CRS_raw, CRS, risk_level, safety_override_active

    # Confidence components
    confidence_pct, confidence_level, coverage,
    signal_agreement, exposure, conflicting_evidence

    # SSem supporting metrics
    # CPS_mean: read from "CPS" key (project) or "CPS_mean" fallback
    CPS_mean, kappa_min_mean

    # SMem supporting metrics
    EM_rate, NED_mean

    # SProb supporting metrics
    # B_abs/B_anchor: read from direct keys or reference_summary nested fields
    UAR_mean, mNED_mean, B_abs, B_anchor

    # Artifact paths (for traceability links)
    # runs_parquet: empty for v7 (risk integration writes no parquet)
    runs_parquet, outputs_summary, logs_jsonl
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def load_json(path: str) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: str) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe(d: Optional[Dict], *keys, default=""):
    """Safely navigate nested dict keys, return default if any key missing."""
    if d is None:
        return default
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key, None)
        if d is None:
            return default
    return d if d != "" else default


def fmt(val, decimals=4):
    """Format numeric value to string, return empty string if None."""
    if val is None or val == "":
        return ""
    try:
        f = float(val)
        if decimals == 0:
            return str(int(round(f)))
        return str(round(f, decimals))
    except (TypeError, ValueError):
        return str(val)


def slex_label(score) -> str:
    try:
        s = float(score)
    except (TypeError, ValueError):
        return ""
    if s == 0:
        return "Not detected"
    if s == 1:
        return "Weak presence"
    if s == 2:
        return "Moderate presence"
    return "Widely available"


def as_int(val, default: int = 0) -> int:
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default


def pct(part: int, total: int) -> str:
    if total <= 0:
        return ""
    return str(int(round(part / total * 100)))


def source_name_from_path(path: str) -> str:
    name = Path(str(path)).stem.lower()
    if "github" in name:
        return "GitHub"
    if "kaggle" in name:
        return "Kaggle"
    return Path(str(path)).stem.replace("_", " ").title()


def build_sources_reviewed(proxy_merge: Dict[str, Any]) -> str:
    sources: List[str] = []
    for key in ("github_csv", "kaggle_csv"):
        value = proxy_merge.get(key, "")
        if value:
            sources.append(source_name_from_path(value))
    if not sources:
        sources = [str(k).title() for k in (proxy_merge.get("source_counts") or {}).keys()]
    sources = list(dict.fromkeys(sources))
    if len(sources) <= 1:
        return sources[0] if sources else ""
    return ", ".join(sources[:-1]) + " and " + sources[-1]


def build_repro_metadata(config_path: str) -> Dict[str, str]:
    """
    Build reproducibility-card values from versioned project metadata.

    These fields are intentionally duplicated into report_data.csv so the HTML
    report remains a pure static renderer and does not need direct YAML/JSON
    access beyond the CSV.
    """
    cfg = load_yaml(config_path) or {}
    project = cfg.get("project", {}) if isinstance(cfg, dict) else {}
    proxy_merge = load_json("outputs/proxy_structured_merged_build_summary.json") or {}
    proxy_build = load_json("outputs/proxy_build_summary_external.json") or {}

    dataset_name = project.get("dataset_name", "XSum")
    split = proxy_build.get("split_filter") or "test"
    frozen_path = project.get("frozen_master_table_path", "")
    n_items = project.get("n_items_expected", "")
    dataset_version = project.get("dataset_version", "")
    seed = project.get("global_seed", "")

    frozen_eval = frozen_path
    if frozen_path and n_items != "":
        frozen_eval = f"{frozen_path} ({n_items} items)"

    proxy_out = proxy_merge.get("merged_out", "")
    proxy_rows = proxy_merge.get("rows_after_dedupe", "")
    proxy_version = proxy_out
    if proxy_out and proxy_rows != "":
        proxy_version = f"{proxy_out} ({proxy_rows} rows)"

    return {
        "benchmark_split": f"{dataset_name} / {split}",
        "frozen_evaluation_set": frozen_eval,
        "configuration_profile": config_path,
        "proxy_corpus_version": proxy_version,
        "random_seed": str(seed),
        "dataset_version": str(dataset_version),
        "sources_reviewed": build_sources_reviewed(proxy_merge),
    }


# ─────────────────────────────────────────────
# Per-model data collection
# ─────────────────────────────────────────────

def collect_model(
    model_id: str,
    lex_summary: Dict[str, Any],
    repro: Dict[str, str],
    run_date: str,
    pipeline_version: str,
    benchmark: str,
) -> Dict[str, str]:
    """
    Load all stage summary JSONs for a model and assemble one CSV row.
    """
    dcq_s   = load_json(f"outputs/v4_dcq_summary_{model_id}.json")
    mem_s   = load_json(f"outputs/v5_mem_summary_{model_id}.json")
    stab_s  = load_json(f"outputs/v6_stability_summary_{model_id}.json")
    risk_s  = load_json(f"outputs/v7_risk_summary_{model_id}.json")

    # ── Benchmark-level (shared) ──────────────────────────────────────────
    slex_agg = safe(lex_summary, "SLex_aggregate") or safe(lex_summary, "SLex")
    slex_counts = safe(lex_summary, "SLex_counts", default={})
    if not isinstance(slex_counts, dict):
        slex_counts = {}
    lexical_total = as_int(safe(lex_summary, "valid_items") or safe(lex_summary, "n_rows_total"))
    lexical_items_found = sum(as_int(v) for k, v in slex_counts.items() if as_int(k, -1) > 0)
    lexical_strong_overlap_items = as_int(slex_counts.get("3", slex_counts.get(3, 0)))

    # ── Risk integration primary fields ──────────────────────────────────
    crs_raw          = safe(risk_s, "CRS_raw")
    crs              = safe(risk_s, "CRS")
    risk_level       = safe(risk_s, "risk_level")
    override_active  = safe(risk_s, "safety_override_active")
    conf_pct         = safe(risk_s, "confidence_pct")
    conf_level       = safe(risk_s, "confidence_level")
    coverage         = safe(risk_s, "coverage")
    signal_agreement = safe(risk_s, "signal_agreement")
    exposure         = safe(risk_s, "exposure")
    conflict         = safe(risk_s, "conflicting_evidence")

    ssem_agg  = safe(risk_s, "SSem_aggregate")
    smem_agg  = safe(risk_s, "SMem_aggregate")
    sprob_agg = safe(risk_s, "SProb_aggregate")

    # Fall back to detector summaries if risk summary missing
    if ssem_agg == "":
        ssem_agg  = safe(dcq_s,  "SSem_aggregate") or safe(dcq_s,  "SSem")
    if smem_agg == "":
        smem_agg  = safe(mem_s,  "SMem_aggregate") or safe(mem_s,  "SMem")
    if sprob_agg == "":
        sprob_agg = safe(stab_s, "SProb_aggregate") or safe(stab_s, "SProb")

    # ── SSem supporting metrics ───────────────────────────────────────────
    # Project stores CPS as "CPS" (not "CPS_mean"); try both for compatibility
    cps_mean       = safe(dcq_s, "CPS") or safe(dcq_s, "CPS_mean") or safe(dcq_s, "cps_mean")
    kappa_min_mean = safe(dcq_s, "kappa_min_mean") or safe(dcq_s, "kappa_min")

    # ── SMem supporting metrics ───────────────────────────────────────────
    em_rate  = safe(mem_s, "EM_rate")  or safe(mem_s, "em_rate")
    ned_mean = safe(mem_s, "NED_mean") or safe(mem_s, "ned_mean")

    # ── SProb supporting metrics ──────────────────────────────────────────
    uar_mean  = safe(stab_s, "UAR_mean")  or safe(stab_s, "uar_mean")
    mned_mean = safe(stab_s, "mNED_mean") or safe(stab_s, "mned_mean")
    # B_abs / B_anchor are only populated if explicitly stored by the stability
    # detector. Do not substitute raw anchor_mNED summaries here; those are input
    # metrics, not band scores.
    b_abs    = (safe(stab_s, "B_abs")    or safe(stab_s, "b_abs"))
    b_anchor = (safe(stab_s, "B_anchor") or safe(stab_s, "b_anchor"))

    # ── Artifact paths ────────────────────────────────────────────────────
    # v7 risk integration writes summary JSON + log only (no parquet).
    # runs_parquet is kept in the CSV schema for forward-compatibility but
    # left empty so the report page does not link to a non-existent file.
    runs_parquet    = ""
    outputs_summary = f"outputs/v7_risk_summary_{model_id}.json"
    logs_jsonl      = f"logs/v7_risk_{model_id}.jsonl"

    return {
        # Identification
        "model_id":          model_id,
        "run_date":          run_date,
        "pipeline_version":  pipeline_version,
        "benchmark":         benchmark,
        "benchmark_split":   repro.get("benchmark_split", ""),
        "frozen_evaluation_set": repro.get("frozen_evaluation_set", ""),
        "configuration_profile": repro.get("configuration_profile", ""),
        "proxy_corpus_version":  repro.get("proxy_corpus_version", ""),
        "random_seed":       repro.get("random_seed", ""),
        "dataset_version":   repro.get("dataset_version", ""),

        # Benchmark exposure
        # Field names: try exact key first, then _mean variant (project convention)
        "SLex_aggregate":    fmt(slex_agg),
        "MaxSpanLen":        fmt(safe(lex_summary, "MaxSpanLen")      or safe(lex_summary, "MaxSpanLen_mean"), 0),
        "NgramHits":         fmt(safe(lex_summary, "NgramHits")       or safe(lex_summary, "NgramHits_mean"),  0),
        "ProxyCount":        fmt(safe(lex_summary, "ProxyCount")      or safe(lex_summary, "ProxyCount_mean"), 0),
        "SLex_label":        slex_label(slex_agg),
        "sources_reviewed":  repro.get("sources_reviewed", ""),
        "lexical_items_found": fmt(lexical_items_found, 0),
        "lexical_items_total": fmt(lexical_total, 0),
        "lexical_items_found_pct": pct(lexical_items_found, lexical_total),
        "lexical_strong_overlap_items": fmt(lexical_strong_overlap_items, 0),
        "lexical_strong_overlap_pct": pct(lexical_strong_overlap_items, lexical_total),

        # Detector scores
        "SSem_aggregate":    fmt(ssem_agg,  1),
        "SMem_aggregate":    fmt(smem_agg,  1),
        "SProb_aggregate":   fmt(sprob_agg, 1),

        # CRS & Risk
        "CRS_raw":                fmt(crs_raw),
        "CRS":                    fmt(crs),
        "risk_level":             str(risk_level),
        "safety_override_active": str(override_active),

        # Confidence
        "confidence_pct":        fmt(conf_pct,   0),
        "confidence_level":      str(conf_level),
        "coverage":              fmt(coverage),
        "signal_agreement":      fmt(signal_agreement),
        "exposure":              fmt(exposure),
        "conflicting_evidence":  str(conflict),

        # SSem supporting
        "CPS_mean":       fmt(cps_mean),
        "kappa_min_mean": fmt(kappa_min_mean),

        # SMem supporting
        "EM_rate":  fmt(em_rate),
        "NED_mean": fmt(ned_mean),

        # SProb supporting
        "UAR_mean":  fmt(uar_mean),
        "mNED_mean": fmt(mned_mean),
        "B_abs":     fmt(b_abs,    1),
        "B_anchor":  fmt(b_anchor, 1),

        # Artifact paths
        "runs_parquet":    runs_parquet,
        "outputs_summary": outputs_summary,
        "logs_jsonl":      logs_jsonl,
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

COLUMNS = [
    "model_id", "run_date", "pipeline_version", "benchmark",
    "benchmark_split", "frozen_evaluation_set", "configuration_profile",
    "proxy_corpus_version", "random_seed", "dataset_version",
    "SLex_aggregate", "MaxSpanLen", "NgramHits", "ProxyCount", "SLex_label",
    "sources_reviewed", "lexical_items_found", "lexical_items_total",
    "lexical_items_found_pct", "lexical_strong_overlap_items",
    "lexical_strong_overlap_pct",
    "SSem_aggregate", "SMem_aggregate", "SProb_aggregate",
    "CRS_raw", "CRS", "risk_level", "safety_override_active",
    "confidence_pct", "confidence_level", "coverage",
    "signal_agreement", "exposure", "conflicting_evidence",
    "CPS_mean", "kappa_min_mean",
    "EM_rate", "NED_mean",
    "UAR_mean", "mNED_mean", "B_abs", "B_anchor",
    "runs_parquet", "outputs_summary", "logs_jsonl",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_ids", nargs="+", required=True,
        help="List of model IDs to include (e.g. gpt4omini gemini25flash)",
    )
    parser.add_argument(
        "--benchmark", type=str, default="XSum",
        help="Benchmark name (default: XSum)",
    )
    parser.add_argument(
        "--run_date", type=str, default="",
        help="Run date string (e.g. 2026-04-08). Auto-detected from risk summary if omitted.",
    )
    parser.add_argument(
        "--out", type=str, default="outputs/report_data.csv",
        help="Output CSV path (default: outputs/report_data.csv)",
    )
    parser.add_argument(
        "--config", type=str, default="configs/run_config.yaml",
        help="Versioned pipeline YAML config (default: configs/run_config.yaml)",
    )
    args = parser.parse_args()

    # Load shared lexical summary (benchmark-level, same for all models)
    lex_summary = load_json("outputs/v3_lexical_summary.json") or {}
    repro = build_repro_metadata(args.config)

    # Auto-detect pipeline version and run date from first available risk summary
    pipeline_version = "4.2.0"
    run_date = args.run_date
    for mid in args.model_ids:
        rs = load_json(f"outputs/v7_risk_summary_{mid}.json")
        if rs:
            pipeline_version = rs.get("pipeline_version", pipeline_version)
            if not run_date:
                # Try to extract date from log or use today
                import datetime
                run_date = datetime.date.today().isoformat()
            break

    rows: List[Dict[str, str]] = []
    missing: List[str] = []

    for model_id in args.model_ids:
        risk_path = Path(f"outputs/v7_risk_summary_{model_id}.json")
        if not risk_path.exists():
            print(f"[WARN] Missing risk summary for {model_id} — skipping")
            missing.append(model_id)
            continue
        row = collect_model(
            model_id=model_id,
            lex_summary=lex_summary,
            repro=repro,
            run_date=run_date,
            pipeline_version=pipeline_version,
            benchmark=args.benchmark,
        )
        rows.append(row)
        print(f"  [{model_id}] CRS={row['CRS']}  risk={row['risk_level']}  confidence={row['confidence_pct']}%")

    if not rows:
        print("No data collected — exiting without writing CSV.")
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {len(rows)} model(s) written to {args.out}")
    if missing:
        print(f"Skipped (missing risk summary): {missing}")


if __name__ == "__main__":
    main()
