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
    assessment/data/report_data.csv                — one row per model

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
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
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


def template_path(path: str, model_id: str) -> str:
    return str(path).replace("{model_id}", model_id)


def stage_summary_path(
    cfg: Dict[str, Any],
    stage_key: str,
    default: str,
    model_id: str,
) -> str:
    outputs = cfg.get(stage_key, {}).get("outputs", {}) if isinstance(cfg, dict) else {}
    path = outputs.get("summary") or outputs.get("summary_json") or default
    return template_path(str(path), model_id)


def stage_output_path(
    cfg: Dict[str, Any],
    stage_key: str,
    output_key: str,
    default: str,
    model_id: str,
) -> str:
    outputs = cfg.get(stage_key, {}).get("outputs", {}) if isinstance(cfg, dict) else {}
    path = outputs.get(output_key, default)
    return template_path(str(path), model_id)


def risk_summary_path(cfg: Dict[str, Any], model_id: str) -> str:
    outputs = cfg.get("risk_integration", {}).get("outputs", {}) if isinstance(cfg, dict) else {}
    path = outputs.get("summary_json", f"outputs/v7_risk_summary_{model_id}.json")
    return template_path(str(path), model_id)


def risk_log_path(cfg: Dict[str, Any], model_id: str) -> str:
    outputs = cfg.get("risk_integration", {}).get("outputs", {}) if isinstance(cfg, dict) else {}
    path = outputs.get("log_jsonl", f"logs/v7_risk_{model_id}.jsonl")
    return template_path(str(path), model_id)


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


def compact_dict(val: Any) -> str:
    if not isinstance(val, dict) or not val:
        return ""
    return "; ".join(f"{k}:{v}" for k, v in val.items())


def quality_value(
    quality: Dict[str, Any],
    detector: str,
    field: str,
    default: Any = "",
) -> Any:
    if not isinstance(quality, dict):
        return default
    det = quality.get(detector, {})
    if not isinstance(det, dict):
        return default
    return det.get(field, default)


def quality_reasons(quality: Dict[str, Any], detector: str) -> str:
    reasons = quality_value(quality, detector, "reasons", default=[])
    if isinstance(reasons, list):
        return "; ".join(str(x) for x in reasons if str(x))
    return str(reasons) if reasons else ""


def detector_quality_from_summary(detector: str, summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(summary, dict):
        return {}

    def num(key: str) -> float:
        try:
            value = summary.get(key)
            if value in (None, ""):
                return 0.0
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    if detector == "SMem":
        failures = num("failures_ref") + num("failures_ctrl")
        processed = num("processed_new_ref") + num("processed_new_ctrl")
    else:
        failures = num("failures")
        processed = num("processed_new")

    valid_items = (
        num("valid_items")
        or num("valid_items_for_cps")
        or num("bcq_valid_items")
        or num("bdq_valid_items")
    )
    total_calls = processed + failures
    failure_rate = failures / total_calls if total_calls else 0.0
    valid_response_rate = valid_items / (valid_items + failures) if (valid_items + failures) else None

    reasons: List[str] = []
    caveat = False
    if failure_rate >= 0.10:
        caveat = True
        reasons.append(f"failure_rate={failure_rate:.3f}")
    if valid_response_rate is not None and valid_response_rate < 0.90:
        caveat = True
        reasons.append(f"valid_response_rate={valid_response_rate:.3f}")

    refusal_rate = None
    if detector == "SMem" and summary.get("refusal_rate") not in (None, ""):
        refusal_rate = num("refusal_rate")
        if refusal_rate >= 0.25:
            caveat = True
            reasons.append(f"refusal_rate={refusal_rate:.3f}")

    if failure_rate >= 0.25 or (valid_response_rate is not None and valid_response_rate < 0.75) or (refusal_rate is not None and refusal_rate >= 0.50):
        interp = "LOW"
    elif caveat:
        interp = "MODERATE"
    else:
        interp = "HIGH"

    return {
        "interpretability": interp,
        "evidence_caveat": caveat,
        "reasons": reasons,
    }


SMEM_REFUSAL_PATTERNS = {
    "dataset_access_disclaimer": [
        "i'm not able to access the full dataset",
        "i cannot access the dataset",
        "the actual completion may vary",
        "i don't have access to the original",
    ],
    "capability_refusal": [
        "i cannot provide",
        "i'm not able to",
        "i don't have access to",
        "i am unable to",
        "as an ai, i cannot",
    ],
    "meta_refusal": [
        "as an ai language model",
        "i should note that",
        "i want to clarify",
        "i must point out",
        "it's important to note that i",
    ],
}


def classify_smem_response(completion_norm: Any) -> Dict[str, Any]:
    if not isinstance(completion_norm, str) or not completion_norm.strip():
        return {
            "response_type": "empty",
            "is_refusal": True,
            "is_valid_completion": False,
        }
    text = completion_norm.lower().strip()
    for rtype, patterns in SMEM_REFUSAL_PATTERNS.items():
        if any(p in text for p in patterns):
            return {
                "response_type": rtype,
                "is_refusal": True,
                "is_valid_completion": False,
            }
    return {
        "response_type": "completion",
        "is_refusal": False,
        "is_valid_completion": True,
    }


def smem_interp_from_refusal_rate(refusal_rate: Optional[float]) -> str:
    if refusal_rate is None:
        return ""
    if refusal_rate >= 0.50:
        return "LOW"
    if refusal_rate >= 0.25:
        return "MODERATE"
    return "HIGH"


def smem_quality_from_counts(
    *,
    n_total: int,
    n_refusal: int,
    n_valid: int,
    refusal_breakdown: Dict[str, int],
) -> Dict[str, Any]:
    if n_total <= 0:
        return {}
    refusal_rate = n_refusal / n_total
    valid_completion_rate = n_valid / n_total
    reasons: List[str] = []
    caveat = False
    if refusal_rate >= 0.25:
        caveat = True
        reasons.append(f"refusal_rate={refusal_rate:.3f}")
    interp = smem_interp_from_refusal_rate(refusal_rate)
    return {
        "interpretability": interp,
        "evidence_caveat": caveat,
        "reasons": reasons,
        "refusal_rate": round(refusal_rate, 4),
        "valid_completion_rate": round(valid_completion_rate, 4),
        "refusal_breakdown": refusal_breakdown,
    }


def smem_quality_from_parquet(path: str, model_id: str) -> Dict[str, Any]:
    p = Path(str(path))
    if not p.exists():
        return {}
    try:
        df = pd.read_parquet(p)
    except Exception:
        return {}

    col_refusal = f"is_refusal_{model_id}"
    col_valid = f"is_valid_completion_{model_id}"
    col_rtype = f"response_type_{model_id}"
    if col_refusal not in df.columns or col_valid not in df.columns:
        return {}

    n_total = len(df)
    refusal_series = pd.to_numeric(df[col_refusal], errors="coerce").fillna(0).astype(int)
    valid_series = pd.to_numeric(df[col_valid], errors="coerce").fillna(0).astype(int)
    refusal_breakdown: Dict[str, int] = {}
    if col_rtype in df.columns:
        refusal_breakdown = (
            df[refusal_series == 1][col_rtype]
            .dropna()
            .astype(str)
            .value_counts()
            .to_dict()
        )
    return smem_quality_from_counts(
        n_total=n_total,
        n_refusal=int(refusal_series.sum()),
        n_valid=int(valid_series.sum()),
        refusal_breakdown=refusal_breakdown,
    )


def smem_quality_from_log(path: str) -> Dict[str, Any]:
    p = Path(str(path))
    if not p.exists():
        return {}
    n_total = 0
    n_refusal = 0
    n_valid = 0
    refusal_breakdown: Dict[str, int] = {}
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("pass") != "reference" or rec.get("status") != "ok":
                continue
            completion_norm = rec.get("completion_norm")
            if completion_norm is None:
                continue
            cls = classify_smem_response(completion_norm)
            n_total += 1
            if cls["is_refusal"]:
                n_refusal += 1
                rtype = str(cls["response_type"])
                refusal_breakdown[rtype] = refusal_breakdown.get(rtype, 0) + 1
            if cls["is_valid_completion"]:
                n_valid += 1
    return smem_quality_from_counts(
        n_total=n_total,
        n_refusal=n_refusal,
        n_valid=n_valid,
        refusal_breakdown=refusal_breakdown,
    )


def parse_choice_abcde(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    s = text.strip()
    if not s:
        return ""
    m = re.match(r"^\s*([ABCDE])(?:[\)\].:\s]|$)", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    window = s[:120]
    m = re.search(
        r"(?:answer|choice|option|selected|select|pick|picked|my answer)\s*(?:is|:)?\s*([ABCDE])\b",
        window,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()
    letters = re.findall(r"\b([ABCDE])\b", window, flags=re.IGNORECASE)
    unique = {x.upper() for x in letters}
    return unique.pop() if len(unique) == 1 else ""


def ssem_quality_from_parquet(path: str, model_id: str) -> Dict[str, Any]:
    p = Path(str(path))
    if not p.exists():
        return {}
    try:
        df = pd.read_parquet(p)
    except Exception:
        return {}

    stages: List[Dict[str, Any]] = []
    for prefix in ("bdq", "bcq"):
        raw_col = f"{prefix}_raw_{model_id}"
        retry_col = f"{prefix}_raw_retry_{model_id}"
        choice_col = f"{prefix}_choice_{model_id}"
        if raw_col not in df.columns or choice_col not in df.columns:
            continue
        raw = df[raw_col].fillna("").astype(str)
        retry = df[retry_col].fillna("").astype(str) if retry_col in df.columns else pd.Series([""] * len(df))
        choice = df[choice_col].fillna("").astype(str).str.strip().str.upper()

        attempted = raw.str.strip().ne("") | retry.str.strip().ne("") | choice.ne("")
        denom = int(attempted.sum())
        if denom == 0:
            continue

        initial_non_choice = raw.map(parse_choice_abcde).eq("") & raw.str.strip().ne("")
        final_parse_failure = ~choice.isin(["A", "B", "C", "D", "E"])
        stages.append({
            "stage": prefix.upper(),
            "attempted": denom,
            "initial_non_choice": int((initial_non_choice & attempted).sum()),
            "parse_failures": int((final_parse_failure & attempted).sum()),
        })

    attempted_total = sum(x["attempted"] for x in stages)
    if attempted_total == 0:
        return {}
    non_choice = sum(x["initial_non_choice"] for x in stages)
    parse_failures = sum(x["parse_failures"] for x in stages)
    non_choice_rate = non_choice / attempted_total
    parse_failure_rate = parse_failures / attempted_total

    reasons: List[str] = []
    caveat = False
    if parse_failure_rate >= 0.10:
        caveat = True
        reasons.append(f"SSem_parse_failure_rate={parse_failure_rate:.3f}")
    if non_choice_rate >= 0.10:
        caveat = True
        reasons.append(f"SSem_non_choice_rate={non_choice_rate:.3f}")

    if parse_failure_rate >= 0.25 or non_choice_rate >= 0.25:
        interp = "LOW"
    elif caveat:
        interp = "MODERATE"
    else:
        interp = "HIGH"

    return {
        "interpretability": interp,
        "evidence_caveat": caveat,
        "reasons": reasons,
        "SSem_parse_failure_rate": round(parse_failure_rate, 4),
        "SSem_non_choice_rate": round(non_choice_rate, 4),
        "SSem_parse_failures": parse_failures,
        "SSem_non_choice_count": non_choice,
        "SSem_choice_attempts": attempted_total,
        "SSem_stage_breakdown": stages,
    }


SPROB_REFUSAL_PATTERNS = [
    "i cannot summarize",
    "i can't summarize",
    "i am unable to summarize",
    "i'm unable to summarize",
    "i cannot provide",
    "i'm not able to",
    "i don't have access",
    "as an ai language model",
    "as an ai,",
    "no article was provided",
    "the text is not provided",
]


SPROB_META_PREFIXES = [
    "here is a one-sentence summary",
    "here's a one-sentence summary",
    "one-sentence summary:",
    "summary:",
    "the summary is:",
]


def _load_json_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    return []


def _sentence_count(text: str) -> int:
    text = str(text or "").strip()
    if not text:
        return 0
    chunks = re.split(r"(?<=[.!?])\s+", text)
    return len([x for x in chunks if x.strip()])


def _sprob_output_type(text: Any) -> str:
    if not isinstance(text, str) or not text.strip():
        return "empty"
    s = text.strip()
    lower = s.lower()
    if any(p in lower for p in SPROB_REFUSAL_PATTERNS):
        return "refusal_or_disclaimer"
    if any(lower.startswith(p) for p in SPROB_META_PREFIXES):
        return "meta_answer"
    if "\n" in s and _sentence_count(s) > 1:
        return "non_summary"
    if _sentence_count(s) > 1:
        return "multi_sentence"
    return "valid_summary"


def sprob_quality_from_parquet(path: str, model_id: str) -> Dict[str, Any]:
    p = Path(str(path))
    if not p.exists():
        return {}
    try:
        df = pd.read_parquet(p)
    except Exception:
        return {}

    if "stability_outputs_json" not in df.columns:
        return {}

    counts: Dict[str, int] = {}
    total = 0
    rows_with_outputs = 0
    rows_with_issue = 0

    for _, row in df.iterrows():
        outputs = _load_json_list(row.get("stability_outputs_json"))
        greedy = row.get("greedy_output")
        if isinstance(greedy, str) and greedy.strip():
            outputs.append(greedy)
        if not outputs:
            continue
        rows_with_outputs += 1
        row_issue = False
        for output in outputs:
            otype = _sprob_output_type(output)
            counts[otype] = counts.get(otype, 0) + 1
            total += 1
            if otype != "valid_summary":
                row_issue = True
        if row_issue:
            rows_with_issue += 1

    if total == 0:
        return {}

    def rate(name: str) -> float:
        return counts.get(name, 0) / total

    empty_rate = rate("empty")
    refusal_rate = rate("refusal_or_disclaimer")
    meta_rate = rate("meta_answer")
    non_summary_rate = rate("non_summary")
    multi_sentence_rate = rate("multi_sentence")
    valid_summary_rate = rate("valid_summary")
    row_issue_rate = rows_with_issue / rows_with_outputs if rows_with_outputs else 0.0

    reasons: List[str] = []
    caveat = False
    issue_rates = {
        "SProb_empty_output_rate": empty_rate,
        "SProb_refusal_or_disclaimer_rate": refusal_rate,
        "SProb_meta_answer_rate": meta_rate,
        "SProb_non_summary_rate": non_summary_rate,
        "SProb_multi_sentence_rate": multi_sentence_rate,
        "SProb_row_issue_rate": row_issue_rate,
    }
    for name, value in issue_rates.items():
        if value >= 0.10:
            caveat = True
            reasons.append(f"{name}={value:.3f}")

    if any(value >= 0.25 for value in issue_rates.values()) or valid_summary_rate < 0.75:
        interp = "LOW"
    elif caveat or valid_summary_rate < 0.90:
        interp = "MODERATE"
    else:
        interp = "HIGH"

    return {
        "interpretability": interp,
        "evidence_caveat": caveat,
        "reasons": reasons,
        "SProb_valid_summary_rate": round(valid_summary_rate, 4),
        "SProb_empty_output_rate": round(empty_rate, 4),
        "SProb_refusal_or_disclaimer_rate": round(refusal_rate, 4),
        "SProb_meta_answer_rate": round(meta_rate, 4),
        "SProb_non_summary_rate": round(non_summary_rate, 4),
        "SProb_multi_sentence_rate": round(multi_sentence_rate, 4),
        "SProb_row_issue_rate": round(row_issue_rate, 4),
        "SProb_quality_counts": counts,
        "SProb_outputs_checked": total,
        "SProb_rows_with_outputs": rows_with_outputs,
    }


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


def count_csv_rows(path: str) -> str:
    p = Path(str(path))
    if not p.exists():
        return ""
    with open(p, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            next(reader)
        except StopIteration:
            return "0"
        return str(sum(1 for _ in reader))


def build_repro_metadata(config_path: str) -> Dict[str, str]:
    """
    Build reproducibility-card values from versioned project metadata.

    These fields are intentionally duplicated into report_data.csv so the HTML
    report remains a pure static renderer and does not need direct YAML/JSON
    access beyond the CSV.
    """
    cfg = load_yaml(config_path) or {}
    project = cfg.get("project", {}) if isinstance(cfg, dict) else {}
    proxy_cfg = cfg.get("proxy_builder", {}) if isinstance(cfg, dict) else {}
    proxy_merge = load_json(
        proxy_cfg.get("merge_summary_out", "outputs/proxy_structured_merged_build_summary.json")
    ) or {}
    proxy_build = load_json(
        proxy_cfg.get("summary_out_json", "outputs/proxy_build_summary_external.json")
    ) or {}

    dataset_name = project.get("dataset_name", "XSum")
    split = proxy_build.get("split_filter") or "test"
    frozen_path = project.get("frozen_master_table_path", "")
    n_items = project.get("n_items_expected", "")
    dataset_version = project.get("dataset_version", "")
    seed = project.get("global_seed", "")

    frozen_eval = frozen_path
    if frozen_path and n_items != "":
        frozen_eval = f"{frozen_path} ({n_items} items)"

    proxy_out = proxy_merge.get("merged_out", "") or proxy_cfg.get("merged_out", "")
    proxy_rows = proxy_merge.get("rows_after_dedupe", "")
    if proxy_rows == "" and proxy_out:
        proxy_rows = count_csv_rows(proxy_out)
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
        "sources_reviewed": build_sources_reviewed({
            **proxy_merge,
            "github_csv": proxy_merge.get("github_csv", "") or proxy_cfg.get("github_structured_out", ""),
            "kaggle_csv": proxy_merge.get("kaggle_csv", "") or proxy_cfg.get("kaggle_structured_out", ""),
        }),
    }


# ─────────────────────────────────────────────
# Per-model data collection
# ─────────────────────────────────────────────

def collect_model(
    model_id: str,
    cfg: Dict[str, Any],
    lex_summary: Dict[str, Any],
    repro: Dict[str, str],
    run_date: str,
    pipeline_version: str,
    benchmark: str,
) -> Dict[str, str]:
    """
    Load all stage summary JSONs for a model and assemble one CSV row.
    """
    dcq_path = stage_summary_path(cfg, "dcq", f"outputs/v4_dcq_summary_{model_id}.json", model_id)
    dcq_parquet_path = stage_output_path(cfg, "dcq", "parquet", f"runs/v4_dcq_{model_id}.parquet", model_id)
    mem_path = stage_summary_path(cfg, "mem", f"outputs/v5_mem_summary_{model_id}.json", model_id)
    if not Path(mem_path).exists():
        mem_path = stage_summary_path(cfg, "memorization", f"outputs/v5_mem_summary_{model_id}.json", model_id)
    mem_parquet_path = stage_output_path(cfg, "memorization", "parquet", f"runs/v5_mem_{model_id}.parquet", model_id)
    mem_log_path = stage_output_path(cfg, "memorization", "log_jsonl", f"logs/v5_mem_{model_id}.jsonl", model_id)
    stab_path = stage_summary_path(cfg, "stability", f"outputs/v6_stability_summary_{model_id}.json", model_id)
    stab_parquet_path = stage_output_path(cfg, "stability", "parquet", f"runs/v6_stability_{model_id}.parquet", model_id)
    risk_path = risk_summary_path(cfg, model_id)

    dcq_s   = load_json(dcq_path)
    mem_s   = load_json(mem_path)
    stab_s  = load_json(stab_path)
    risk_s  = load_json(risk_path)

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
    detector_quality = safe(risk_s, "detector_quality", default={})
    if not isinstance(detector_quality, dict):
        detector_quality = {}
    if not detector_quality:
        detector_quality = {
            "SSem": detector_quality_from_summary("SSem", dcq_s),
            "SMem": detector_quality_from_summary("SMem", mem_s),
            "SProb": detector_quality_from_summary("SProb", stab_s),
        }
    ssem_parquet_quality = ssem_quality_from_parquet(dcq_parquet_path, model_id)
    if ssem_parquet_quality:
        existing_ssem_quality = detector_quality.get("SSem", {})
        merged_reasons = []
        for source in (existing_ssem_quality, ssem_parquet_quality):
            reasons = source.get("reasons", []) if isinstance(source, dict) else []
            if isinstance(reasons, list):
                merged_reasons.extend(str(x) for x in reasons if str(x))
        merged_ssem_quality = {
            **existing_ssem_quality,
            **ssem_parquet_quality,
            "evidence_caveat": bool(existing_ssem_quality.get("evidence_caveat")) or bool(ssem_parquet_quality.get("evidence_caveat")),
            "reasons": list(dict.fromkeys(merged_reasons)),
        }
        if existing_ssem_quality.get("interpretability") == "LOW" or ssem_parquet_quality.get("interpretability") == "LOW":
            merged_ssem_quality["interpretability"] = "LOW"
        elif existing_ssem_quality.get("interpretability") == "MODERATE" or ssem_parquet_quality.get("interpretability") == "MODERATE":
            merged_ssem_quality["interpretability"] = "MODERATE"
        else:
            merged_ssem_quality["interpretability"] = ssem_parquet_quality.get("interpretability", existing_ssem_quality.get("interpretability", ""))
        detector_quality["SSem"] = merged_ssem_quality
    smem_offline_quality = smem_quality_from_parquet(mem_parquet_path, model_id) or smem_quality_from_log(mem_log_path)
    if smem_offline_quality:
        existing_smem_quality = detector_quality.get("SMem", {})
        merged_reasons = []
        for source in (existing_smem_quality, smem_offline_quality):
            reasons = source.get("reasons", []) if isinstance(source, dict) else []
            if isinstance(reasons, list):
                merged_reasons.extend(str(x) for x in reasons if str(x))
        merged_smem_quality = {
            **existing_smem_quality,
            **smem_offline_quality,
            "evidence_caveat": bool(existing_smem_quality.get("evidence_caveat")) or bool(smem_offline_quality.get("evidence_caveat")),
            "reasons": list(dict.fromkeys(merged_reasons)),
        }
        if existing_smem_quality.get("interpretability") == "LOW" or smem_offline_quality.get("interpretability") == "LOW":
            merged_smem_quality["interpretability"] = "LOW"
        elif existing_smem_quality.get("interpretability") == "MODERATE" or smem_offline_quality.get("interpretability") == "MODERATE":
            merged_smem_quality["interpretability"] = "MODERATE"
        else:
            merged_smem_quality["interpretability"] = smem_offline_quality.get("interpretability", existing_smem_quality.get("interpretability", ""))
        detector_quality["SMem"] = merged_smem_quality
    sprob_parquet_quality = sprob_quality_from_parquet(stab_parquet_path, model_id)
    if sprob_parquet_quality:
        existing_sprob_quality = detector_quality.get("SProb", {})
        merged_reasons = []
        for source in (existing_sprob_quality, sprob_parquet_quality):
            reasons = source.get("reasons", []) if isinstance(source, dict) else []
            if isinstance(reasons, list):
                merged_reasons.extend(str(x) for x in reasons if str(x))
        merged_sprob_quality = {
            **existing_sprob_quality,
            **sprob_parquet_quality,
            "evidence_caveat": bool(existing_sprob_quality.get("evidence_caveat")) or bool(sprob_parquet_quality.get("evidence_caveat")),
            "reasons": list(dict.fromkeys(merged_reasons)),
        }
        if existing_sprob_quality.get("interpretability") == "LOW" or sprob_parquet_quality.get("interpretability") == "LOW":
            merged_sprob_quality["interpretability"] = "LOW"
        elif existing_sprob_quality.get("interpretability") == "MODERATE" or sprob_parquet_quality.get("interpretability") == "MODERATE":
            merged_sprob_quality["interpretability"] = "MODERATE"
        else:
            merged_sprob_quality["interpretability"] = sprob_parquet_quality.get("interpretability", existing_sprob_quality.get("interpretability", ""))
        detector_quality["SProb"] = merged_sprob_quality

    # ── SSem supporting metrics ───────────────────────────────────────────
    # Project stores CPS as "CPS" (not "CPS_mean"); try both for compatibility
    cps_mean       = safe(dcq_s, "CPS") or safe(dcq_s, "CPS_mean") or safe(dcq_s, "cps_mean")
    kappa_min_mean = safe(dcq_s, "kappa_min_mean") or safe(dcq_s, "kappa_min")

    # ── SMem supporting metrics ───────────────────────────────────────────
    em_rate  = safe(mem_s, "EM_rate")  or safe(mem_s, "em_rate")
    ned_mean = safe(mem_s, "NED_mean") or safe(mem_s, "ned_mean")
    smem_diag = safe(risk_s, "SMem_diagnostics", default={})
    if not isinstance(smem_diag, dict):
        smem_diag = {}
    refusal_rate = safe(smem_diag, "refusal_rate")
    if refusal_rate == "":
        refusal_rate = safe(mem_s, "refusal_rate")
    if refusal_rate == "":
        refusal_rate = quality_value(detector_quality, "SMem", "refusal_rate")
    valid_completion_rate = safe(smem_diag, "valid_completion_rate")
    if valid_completion_rate == "":
        valid_completion_rate = safe(mem_s, "valid_completion_rate")
    if valid_completion_rate == "":
        valid_completion_rate = quality_value(detector_quality, "SMem", "valid_completion_rate")
    smem_interpretability = safe(smem_diag, "SMem_interpretability")
    if smem_interpretability == "":
        smem_interpretability = safe(mem_s, "SMem_interpretability")
    if smem_interpretability == "":
        smem_interpretability = quality_value(detector_quality, "SMem", "interpretability")
    refusal_breakdown = safe(smem_diag, "refusal_breakdown", default={})
    if not refusal_breakdown:
        refusal_breakdown = safe(mem_s, "refusal_breakdown", default={})
    if not refusal_breakdown:
        refusal_breakdown = quality_value(detector_quality, "SMem", "refusal_breakdown", default={})
    smem_evidence_caveat = safe(risk_s, "SMem_evidence_caveat")
    if smem_evidence_caveat == "":
        smem_evidence_caveat = safe(smem_diag, "SMem_evidence_caveat")
    if smem_evidence_caveat == "":
        smem_evidence_caveat = quality_value(detector_quality, "SMem", "evidence_caveat")

    ssem_interpretability = quality_value(detector_quality, "SSem", "interpretability")
    ssem_parse_failure_rate = quality_value(detector_quality, "SSem", "SSem_parse_failure_rate")
    ssem_non_choice_rate = quality_value(detector_quality, "SSem", "SSem_non_choice_rate")
    smem_quality_interp = quality_value(detector_quality, "SMem", "interpretability")
    sprob_interpretability = quality_value(detector_quality, "SProb", "interpretability")
    sprob_valid_summary_rate = quality_value(detector_quality, "SProb", "SProb_valid_summary_rate")
    sprob_empty_output_rate = quality_value(detector_quality, "SProb", "SProb_empty_output_rate")
    sprob_refusal_or_disclaimer_rate = quality_value(detector_quality, "SProb", "SProb_refusal_or_disclaimer_rate")
    sprob_meta_answer_rate = quality_value(detector_quality, "SProb", "SProb_meta_answer_rate")
    sprob_non_summary_rate = quality_value(detector_quality, "SProb", "SProb_non_summary_rate")
    sprob_multi_sentence_rate = quality_value(detector_quality, "SProb", "SProb_multi_sentence_rate")
    if smem_quality_interp:
        smem_interpretability = smem_quality_interp
    ssem_evidence_caveat = safe(risk_s, "SSem_evidence_caveat")
    if ssem_evidence_caveat == "":
        ssem_evidence_caveat = quality_value(detector_quality, "SSem", "evidence_caveat")
    sprob_evidence_caveat = safe(risk_s, "SProb_evidence_caveat")
    if sprob_evidence_caveat == "":
        sprob_evidence_caveat = quality_value(detector_quality, "SProb", "evidence_caveat")

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
    outputs_summary = risk_path
    logs_jsonl      = risk_log_path(cfg, model_id)

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
        "SSem_interpretability": str(ssem_interpretability),
        "SSem_evidence_caveat": str(ssem_evidence_caveat),
        "SSem_quality_reasons": quality_reasons(detector_quality, "SSem"),
        "SSem_parse_failure_rate": fmt(ssem_parse_failure_rate),
        "SSem_non_choice_rate": fmt(ssem_non_choice_rate),

        # SMem supporting
        "EM_rate":  fmt(em_rate),
        "NED_mean": fmt(ned_mean),
        "refusal_rate": fmt(refusal_rate),
        "valid_completion_rate": fmt(valid_completion_rate),
        "SMem_interpretability": str(smem_interpretability),
        "refusal_breakdown": compact_dict(refusal_breakdown),
        "SMem_evidence_caveat": str(smem_evidence_caveat),
        "SMem_quality_reasons": quality_reasons(detector_quality, "SMem"),

        # SProb supporting
        "UAR_mean":  fmt(uar_mean),
        "mNED_mean": fmt(mned_mean),
        "B_abs":     fmt(b_abs,    1),
        "B_anchor":  fmt(b_anchor, 1),
        "SProb_interpretability": str(sprob_interpretability),
        "SProb_evidence_caveat": str(sprob_evidence_caveat),
        "SProb_quality_reasons": quality_reasons(detector_quality, "SProb"),
        "SProb_valid_summary_rate": fmt(sprob_valid_summary_rate),
        "SProb_empty_output_rate": fmt(sprob_empty_output_rate),
        "SProb_refusal_or_disclaimer_rate": fmt(sprob_refusal_or_disclaimer_rate),
        "SProb_meta_answer_rate": fmt(sprob_meta_answer_rate),
        "SProb_non_summary_rate": fmt(sprob_non_summary_rate),
        "SProb_multi_sentence_rate": fmt(sprob_multi_sentence_rate),

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
    "SSem_interpretability", "SSem_evidence_caveat", "SSem_quality_reasons",
    "SSem_parse_failure_rate", "SSem_non_choice_rate",
    "EM_rate", "NED_mean", "refusal_rate", "valid_completion_rate",
    "SMem_interpretability", "refusal_breakdown", "SMem_evidence_caveat",
    "SMem_quality_reasons",
    "UAR_mean", "mNED_mean", "B_abs", "B_anchor",
    "SProb_interpretability", "SProb_evidence_caveat", "SProb_quality_reasons",
    "SProb_valid_summary_rate", "SProb_empty_output_rate",
    "SProb_refusal_or_disclaimer_rate", "SProb_meta_answer_rate",
    "SProb_non_summary_rate", "SProb_multi_sentence_rate",
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
        "--out", type=str, default="assessment/data/report_data.csv",
        help="Output CSV path (default: assessment/data/report_data.csv)",
    )
    parser.add_argument(
        "--config", type=str, default="configs/run_config.yaml",
        help="Versioned pipeline YAML config (default: configs/run_config.yaml)",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config) or {}

    # Load shared lexical summary (benchmark-level, same for all models)
    lex_summary_path = stage_summary_path(
        cfg, "lexical", "outputs/v3_lexical_summary.json", args.model_ids[0]
    )
    lex_summary = load_json(lex_summary_path) or {}
    repro = build_repro_metadata(args.config)

    # Auto-detect pipeline version and run date from first available risk summary
    pipeline_version = "4.2.0"
    run_date = args.run_date
    for mid in args.model_ids:
        rs = load_json(risk_summary_path(cfg, mid))
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
        risk_path = Path(risk_summary_path(cfg, model_id))
        if not risk_path.exists():
            print(f"[WARN] Missing risk summary for {model_id} — skipping")
            missing.append(model_id)
            continue
        row = collect_model(
            model_id=model_id,
            cfg=cfg,
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
