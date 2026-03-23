#!/usr/bin/env python3
"""
scripts/run_risk_and_visualize.py

End-of-pipeline module:
- merges detector outputs (SLex + model signals SSem/SMem/SProb)
- computes normalized risk R(ei) and Risk Level with overrides
- computes Confidence (High/Medium/Low) per your rules
- exports:
    runs/v7_risk_{model_id}.parquet
    runs/v7_risk_{model_id}.csv
    outputs/v7_summary_{model_id}.json
- visualizes:
    outputs/fig_{model_id}_signals_hist.png
    outputs/fig_{model_id}_risk_hist.png
    outputs/fig_{model_id}_risk_level_bar.png
    outputs/fig_{model_id}_signals_bar.png

Implements exactly your final steps:
5) Normalization:
   ŜLex = SLex/3, ŜSem = SSem/3, ŜMem = SMem/3, ŜProb = SProb/3
6) Risk:
   R = 100 * (0.35ŜLex + 0.20ŜSem + 0.30ŜMem + 0.15ŜProb)
7) Overrides:
   - if SLex=3 OR SMem=3 => enforce R >= 50
   - if SProb>=2 AND others<=1 => cap R <= 49
8) Levels:
   Low (<25), Medium (25–49), High (50–74), Critical (>=75)
9) Confidence:
   High if >=2 signals >=2 AND (SLex>=2 OR SMem>=2)
   Medium if >=2 signals >=1
   else Low
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# IO helpers
# ----------------------------

def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def write_json(path: str, payload: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def update_summary_table_csv(summary_row: Dict[str, Any], out_path: str) -> None:
    """
    Create/append/update a per-model summary table.
    If model_id already exists, overwrite that row.
    """
    ensure_parent_dir(out_path)
    new_df = pd.DataFrame([summary_row])

    if Path(out_path).exists():
        old = pd.read_csv(out_path)
        if "model_id" in old.columns:
            old = old[old["model_id"] != summary_row["model_id"]]
        merged = pd.concat([old, new_df], ignore_index=True)
    else:
        merged = new_df

    merged.to_csv(out_path, index=False, encoding="utf-8")


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def safe_read_parquet(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_parquet(path)

def resolve_signal_col(df: pd.DataFrame, base: str, model_id: str) -> str:
    """
    Allow both styles:
      - base (e.g., "SSem") or
      - model-specific (e.g., "SSem_gpt4omini")
    """
    c1 = f"{base}_{model_id}"
    if c1 in df.columns:
        return c1
    if base in df.columns:
        return base
    raise ValueError(f"Could not find column '{base}' or '{c1}' in dataframe columns.")


# ----------------------------
# Risk computation (exact rubric)
# ----------------------------

def normalize_level(x: float) -> float:
    # x in {0,1,2,3} -> [0,1]
    if pd.isna(x):
        return 0.0
    return float(x) / 3.0

def compute_risk_row(slex: float, ssem: float, smem: float, sprob: float) -> float:
    # Step 5 normalization
    sh_lex = normalize_level(slex)
    sh_sem = normalize_level(ssem)
    sh_mem = normalize_level(smem)
    sh_prob = normalize_level(sprob)

    # Step 6 base risk
    r = 100.0 * (0.35*sh_lex + 0.20*sh_sem + 0.30*sh_mem + 0.15*sh_prob)

    # Step 7 overrides
    # 7a) enforce >=50 if SLex=3 OR SMem=3
    if (not pd.isna(slex) and float(slex) == 3.0) or (not pd.isna(smem) and float(smem) == 3.0):
        r = max(r, 50.0)

    # 7b) cap <=49 if SProb>=2 AND others<=1
    # "others" = SLex, SSem, SMem
    if (not pd.isna(sprob) and float(sprob) >= 2.0):
        others = [
            0.0 if pd.isna(slex) else float(slex),
            0.0 if pd.isna(ssem) else float(ssem),
            0.0 if pd.isna(smem) else float(smem),
        ]
        if all(v <= 1.0 for v in others):
            r = min(r, 49.0)

    # clamp just in case
    r = max(0.0, min(100.0, r))
    return float(r)

def assign_risk_level(r: float) -> str:
    # Step 8
    if r < 25.0:
        return "Low"
    if r < 50.0:
        return "Medium"
    if r < 75.0:
        return "High"
    return "Critical"

def assign_confidence(slex: float, ssem: float, smem: float, sprob: float) -> str:
    # Step 9
    vals = [
        0.0 if pd.isna(slex) else float(slex),
        0.0 if pd.isna(ssem) else float(ssem),
        0.0 if pd.isna(smem) else float(smem),
        0.0 if pd.isna(sprob) else float(sprob),
    ]
    n_ge2 = sum(1 for v in vals if v >= 2.0)
    n_ge1 = sum(1 for v in vals if v >= 1.0)

    if n_ge2 >= 2 and ((0.0 if pd.isna(slex) else float(slex)) >= 2.0 or (0.0 if pd.isna(smem) else float(smem)) >= 2.0):
        return "High"
    if n_ge1 >= 2:
        return "Medium"
    return "Low"


# ----------------------------
# Visualization
# ----------------------------

def save_hist(values: pd.Series, title: str, xlabel: str, out_path: str, bins: int = 20) -> None:
    ensure_parent_dir(out_path)
    plt.figure()
    plt.hist(values.dropna().astype(float), bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_bar_counts(counts: Dict[Any, int], title: str, xlabel: str, ylabel: str, out_path: str) -> None:
    ensure_parent_dir(out_path)
    keys = list(counts.keys())
    vals = [counts[k] for k in keys]
    plt.figure()
    plt.bar([str(k) for k in keys], vals)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def counts_levels(series: pd.Series, levels=(0,1,2,3)) -> Dict[int, int]:
    s = pd.to_numeric(series, errors="coerce")
    out = {}
    for lv in levels:
        out[int(lv)] = int((s == lv).sum())
    return out

def save_onepage_signals_and_risk_figure(
    d: pd.DataFrame,
    model_id: str,
    out_path: str
) -> None:
    """
    One-page figure with 5 bar charts:
    SLex, SSem, SMem, SProb level counts + RiskLevel counts.
    Suitable for Results chapter.
    """
    ensure_parent_dir(out_path)

    # Prepare counts
    slex = counts_levels(d["SLex"])
    ssem = counts_levels(d["SSem"])
    smem = counts_levels(d["SMem"])
    sprob = counts_levels(d["SProb"])
    rlevels = d["RiskLevel"].value_counts().to_dict()

    # Order RiskLevel consistently
    rl_order = ["Low", "Medium", "High", "Critical"]
    rlevels_ordered = {k: int(rlevels.get(k, 0)) for k in rl_order}

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fig.suptitle(f"{model_id}: Signals and Integrated Risk", fontsize=14)

    # Helper for signal bars
    def plot_signal(ax, counts_dict, title, xlabel):
        keys = [0, 1, 2, 3]
        vals = [counts_dict.get(k, 0) for k in keys]
        ax.bar([str(k) for k in keys], vals)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")

    plot_signal(axes[0, 0], slex, "SLex (Lexical exposure)", "Level")
    plot_signal(axes[0, 1], ssem, "SSem (Semantic preference)", "Level")
    plot_signal(axes[0, 2], smem, "SMem (Memorization)", "Level")
    plot_signal(axes[1, 0], sprob, "SProb (Stability/Probability)", "Level")

    # RiskLevel bar
    ax = axes[1, 1]
    ax.bar(list(rlevels_ordered.keys()), list(rlevels_ordered.values()))
    ax.set_title("RiskLevel (Integrated)")
    ax.set_xlabel("RiskLevel")
    ax.set_ylabel("Count")

    # Leave last subplot empty but clean
    axes[1, 2].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=250)
    plt.close()


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True, help="e.g., gpt4omini or gemini15flash")
    ap.add_argument("--lex_path", default="runs/v3_lexical.parquet")
    ap.add_argument("--dcq_path", default=None, help="default: runs/v4_dcq_{model_id}.parquet")
    ap.add_argument("--mem_path", default=None, help="default: runs/v5_mem_{model_id}.parquet")
    ap.add_argument("--stab_path", default=None, help="default: runs/v6_stability_{model_id}.parquet")
    ap.add_argument("--out_prefix", default=None, help="default: runs/v7_risk_{model_id}")
    args = ap.parse_args()

    model_id = args.model_id
    dcq_path = args.dcq_path or f"runs/v4_dcq_{model_id}.parquet"
    mem_path = args.mem_path or f"runs/v5_mem_{model_id}.parquet"
    stab_path = args.stab_path or f"runs/v6_stability_{model_id}.parquet"
    out_prefix = args.out_prefix or f"runs/v7_risk_{model_id}"

    out_parquet = f"{out_prefix}.parquet"
    out_csv = f"{out_prefix}.csv"
    out_summary = f"outputs/v7_summary_{model_id}.json"

    # --- load
    df_lex = safe_read_parquet(args.lex_path)
    df_dcq = safe_read_parquet(dcq_path)
    df_mem = safe_read_parquet(mem_path)
    df_stab = safe_read_parquet(stab_path)

    # --- select signal columns robustly
    slex_col = resolve_signal_col(df_lex, "SLex", model_id="")  # SLex should be base
    # resolve_signal_col expects model_id; for SLex use direct
    if "SLex" not in df_lex.columns:
        raise ValueError("Lexical parquet must contain 'SLex' column.")

    ssem_col = resolve_signal_col(df_dcq, "SSem", model_id)
    smem_col = resolve_signal_col(df_mem, "SMem", model_id)
    sprob_col = resolve_signal_col(df_stab, "SProb", model_id)

    # optional supporting metrics (if present)
    cps_col = None
    for cand in [f"CPS_{model_id}", "CPS"]:
        if cand in df_dcq.columns:
            cps_col = cand
            break
    em_col = None
    for cand in [f"EM_{model_id}", "EM"]:
        if cand in df_mem.columns:
            em_col = cand
            break
    ne_col = None
    for cand in [f"NE_{model_id}", "NE"]:
        if cand in df_mem.columns:
            ne_col = cand
            break
    uar_col = None
    for cand in [f"UAR_{model_id}", "UAR"]:
        if cand in df_stab.columns:
            uar_col = cand
            break
    mned_col = None
    for cand in [f"mNED_{model_id}", "mNED"]:
        if cand in df_stab.columns:
            mned_col = cand
            break

    # --- merge on xsum_id
    keep_lex = ["xsum_id", "SLex", "MaxSpanLen", "NgramHits", "ProxyCount"]
    keep_dcq = ["xsum_id", ssem_col] + ([cps_col] if cps_col else [])
    keep_mem = ["xsum_id", smem_col] + ([em_col] if em_col else []) + ([ne_col] if ne_col else [])
    keep_stab = ["xsum_id", sprob_col] + ([uar_col] if uar_col else []) + ([mned_col] if mned_col else [])

    d = df_lex[keep_lex].copy()
    d = d.merge(df_dcq[keep_dcq].copy(), on="xsum_id", how="left")
    d = d.merge(df_mem[keep_mem].copy(), on="xsum_id", how="left")
    d = d.merge(df_stab[keep_stab].copy(), on="xsum_id", how="left")

    # standardize signal column names
    d = d.rename(columns={
        ssem_col: "SSem",
        smem_col: "SMem",
        sprob_col: "SProb",
        **({cps_col: "CPS"} if cps_col else {}),
        **({em_col: "EM"} if em_col else {}),
        **({ne_col: "NE"} if ne_col else {}),
        **({uar_col: "UAR"} if uar_col else {}),
        **({mned_col: "mNED"} if mned_col else {}),
    })

    # numeric conversions
    for c in ["SLex","SSem","SMem","SProb","MaxSpanLen","NgramHits","ProxyCount","CPS","EM","NE","UAR","mNED"]:
        if c in d.columns:
            d[c] = to_num(d[c])

    # --- compute risk + confidence
    risks = []
    levels = []
    confs = []
    for _, r in d.iterrows():
        rr = compute_risk_row(r["SLex"], r["SSem"], r["SMem"], r["SProb"])
        risks.append(rr)
        levels.append(assign_risk_level(rr))
        confs.append(assign_confidence(r["SLex"], r["SSem"], r["SMem"], r["SProb"]))

    d["Risk"] = risks          # R(ei)
    d["RiskLevel"] = levels    # L
    d["Confidence"] = confs    # C

    # --- summary table (one CSV for all models)
    summary_table_path = "outputs/v7_models_summary.csv"

    level_counts = d["RiskLevel"].value_counts()
    n = len(d)

    summary_row = {
        "model_id": model_id,
        "n_items": n,
        "mean_Risk": float(d["Risk"].mean()),
        "median_Risk": float(d["Risk"].median()),
        "p10_Risk": float(d["Risk"].quantile(0.10)),
        "p90_Risk": float(d["Risk"].quantile(0.90)),
        "pct_Low": float(level_counts.get("Low", 0) / n * 100.0),
        "pct_Medium": float(level_counts.get("Medium", 0) / n * 100.0),
        "pct_High": float(level_counts.get("High", 0) / n * 100.0),
        "pct_Critical": float(level_counts.get("Critical", 0) / n * 100.0),
        "confidence_High_pct": float((d["Confidence"] == "High").mean() * 100.0),
        "confidence_Medium_pct": float((d["Confidence"] == "Medium").mean() * 100.0),
        "confidence_Low_pct": float((d["Confidence"] == "Low").mean() * 100.0),
    }

    update_summary_table_csv(summary_row, summary_table_path)
    print("Summary table updated:", summary_table_path)


    # --- export
    ensure_parent_dir(out_parquet)
    d.to_parquet(out_parquet, index=False)
    ensure_parent_dir(out_csv)
    d.to_csv(out_csv, index=False, encoding="utf-8")

    # --- summaries per detector + overall
    s = {
        "model_id": model_id,
        "n": int(len(d)),
        "signals_level_counts": {
            "SLex": counts_levels(d["SLex"]),
            "SSem": counts_levels(d["SSem"]),
            "SMem": counts_levels(d["SMem"]),
            "SProb": counts_levels(d["SProb"]),
        },
        "risk": {
            "mean": float(d["Risk"].mean()),
            "median": float(d["Risk"].median()),
            "p10": float(d["Risk"].quantile(0.10)),
            "p90": float(d["Risk"].quantile(0.90)),
            "level_counts": d["RiskLevel"].value_counts().to_dict(),
            "confidence_counts": d["Confidence"].value_counts().to_dict(),
        },
        "inputs": {
            "lex": args.lex_path,
            "dcq": dcq_path,
            "mem": mem_path,
            "stability": stab_path,
        },
        "outputs": {
            "parquet": out_parquet,
            "csv": out_csv,
        },
        "rubric": {
            "normalization": "ŜLex=SLex/3; ŜSem=SSem/3; ŜMem=SMem/3; ŜProb=SProb/3",
            "risk_formula": "R=100*(0.35ŜLex+0.20ŜSem+0.30ŜMem+0.15ŜProb)",
            "overrides": [
                "if SLex=3 or SMem=3 => enforce R>=50",
                "if SProb>=2 and others<=1 => cap R<=49",
            ],
            "levels": "Low(<25), Medium(25–49), High(50–74), Critical(>=75)",
            "confidence": "High if >=2 signals>=2 and (SLex>=2 or SMem>=2); Medium if >=2 signals>=1; else Low",
        }
    }
    write_json(out_summary, s)

    # --- visualizations (PNG)
    # Signal histograms (0..3)
    save_bar_counts(s["signals_level_counts"]["SLex"], f"{model_id}: SLex levels", "SLex", "Count", f"outputs/fig_{model_id}_SLex_bar.png")
    save_bar_counts(s["signals_level_counts"]["SSem"], f"{model_id}: SSem levels", "SSem", "Count", f"outputs/fig_{model_id}_SSem_bar.png")
    save_bar_counts(s["signals_level_counts"]["SMem"], f"{model_id}: SMem levels", "SMem", "Count", f"outputs/fig_{model_id}_SMem_bar.png")
    save_bar_counts(s["signals_level_counts"]["SProb"], f"{model_id}: SProb levels", "SProb", "Count", f"outputs/fig_{model_id}_SProb_bar.png")

    # Risk histogram and RiskLevel bar
    save_hist(d["Risk"], f"{model_id}: Risk (R) distribution", "Risk (0..100)", f"outputs/fig_{model_id}_risk_hist.png", bins=20)
    save_bar_counts(d["RiskLevel"].value_counts().to_dict(), f"{model_id}: RiskLevel counts", "RiskLevel", "Count", f"outputs/fig_{model_id}_risklevel_bar.png")

    # One-page figure for Results
    onepage_path = f"outputs/fig_{model_id}_onepage_signals_risk.png"
    save_onepage_signals_and_risk_figure(d, model_id, onepage_path)
    print("One-page figure:", onepage_path)


    print("Done.")
    print("Model:", model_id)
    print("Outputs:")
    print(" -", out_parquet)
    print(" -", out_csv)
    print(" -", out_summary)
    print("Figures: outputs/fig_...png")


if __name__ == "__main__":
    main()
