"""
load_benchmarks_bbc2024.py
==========================
Creates a frozen master table for the BBC News 2024 dataset
(RealTimeData/bbc_news_alltime, config='2024-06').

Dataset schema:
  - content     -> document  (full article text)
  - description -> summary_ref (one-sentence factual summary, BBC style)
  - title       -> stored for reference only

Why this dataset for the SProb baseline experiment:
  - Articles published June 2024: after the knowledge cutoff of both
    evaluated models (GPT-4o-mini: Oct 2023; Gemini 2.5 Flash: early 2025
    training data likely excludes June 2024 BBC articles at this granularity).
    Contamination is excluded by a temporal criterion, not merely assumed.
  - Same source (BBC), same genre (news), same one-sentence summary format
    as XSum -> task structure is held constant across both datasets.
    Any difference in UAR/mNED is attributable to benchmark exposure,
    not to a change in task difficulty or summary style.
  - 3,450 rows available in the 2024-06 config -> ample for a 100-item sample.

Usage:
    python3 data/load_benchmarks_bbc2024.py

Output files:
    master_table_bbc2024_n100_seed42_v1.parquet
    bbc2024_indices_seed42_n100.json
"""

from datasets import load_dataset
import random
import pandas as pd
import json
import re
import unicodedata
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────
SEED           = 42
N_ITEMS        = 100
DATASET_NAME   = "RealTimeData/bbc_news_alltime"
DATASET_CONFIG = "2024-06"    # articles published June 2024 — after GPT-4o-mini cutoff
DATASET_SPLIT  = "train"      # this dataset has only a train split

OUT_PARQUET = "master_table_bbc2024_n100_seed42_v1.parquet"
OUT_INDICES = "bbc2024_indices_seed42_n100.json"

# ── Load ──────────────────────────────────────────────────────────────────────
print(f"Loading {DATASET_NAME} (config='{DATASET_CONFIG}', split='{DATASET_SPLIT}') ...")
ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
print(f"Total rows : {len(ds)}")
print(f"Columns    : {ds.column_names}")

# ── Filter ────────────────────────────────────────────────────────────────────
def valid(x):
    doc  = x.get("content",     "") or ""
    summ = x.get("description", "") or ""
    # Same length constraints as XSum: doc >= 200 chars, summary >= 20 chars.
    # Upper bound on description: exclude rare multi-sentence descriptions
    # (> 300 chars) to maintain single-sentence format consistent with XSum.
    return (
        len(doc.strip())  >= 200
        and len(summ.strip()) >= 20
        and len(summ.strip()) <= 300
    )

# Sample with a buffer before filtering, then trim to N_ITEMS
random.seed(SEED)
indices = random.sample(range(len(ds)), min(N_ITEMS * 4, len(ds)))
subset  = ds.select(indices)

with open(OUT_INDICES, "w", encoding="utf-8") as f:
    json.dump(indices, f)
print(f"Saved {len(indices)} pre-filter indices -> {OUT_INDICES}")

subset = subset.filter(valid)
print(f"After filtering : {len(subset)} items")

if len(subset) > N_ITEMS:
    subset = subset.select(range(N_ITEMS))
print(f"Final sample    : {len(subset)} items")

# ── Build DataFrame ───────────────────────────────────────────────────────────
rows = []
for idx, ex in enumerate(subset):
    rows.append({
        "item_id":     idx,
        # NOTE: all pipeline scripts expect a column named "xsum_id".
        # We store the article URL (unique per article) as the identifier.
        "xsum_id":     ex.get("link", f"bbc2024_{idx}"),
        "split":       DATASET_SPLIT,
        # document = full article body
        "document":    ex["content"],
        # summary_ref = one-sentence description (BBC editorial summary)
        # This is the direct analogue of XSum's "summary" field:
        # factual, single-sentence, BBC news register.
        "summary_ref": ex["description"],
        # stored for reference; not used by any detector
        "title":           ex.get("title", ""),
        "published_date":  ex.get("published_date", ""),
    })

df = pd.DataFrame(rows)
print(f"\nDataFrame shape : {df.shape}")
print(df[["xsum_id", "summary_ref"]].head(3).to_string())

# ── Normalisation — identical to load_benchmarks.py ──────────────────────────
def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

df["document_norm"]    = df["document"].map(normalize_text)
df["summary_ref_norm"] = df["summary_ref"].map(normalize_text)

# ── Reference prefix for memorisation probing — identical to load_benchmarks.py
def make_prefix(summary: str, frac: float = 0.4, min_tokens: int = 12) -> str:
    toks = summary.split()
    m    = max(min_tokens, int(len(toks) * frac))
    m    = min(m, max(1, len(toks) - 1))
    return " ".join(toks[:m])

df["prefix_ref"] = df["summary_ref_norm"].map(
    lambda s: make_prefix(s, frac=0.4, min_tokens=12)
)

# ── Control prefix — identical to load_benchmarks.py ─────────────────────────
def make_control_prefix(prefix: str, seed: int = 123) -> str:
    rnd  = random.Random(seed)
    toks = prefix.split()
    rnd.shuffle(toks)
    return " ".join(toks)

df["control_prefix"] = df["prefix_ref"].map(
    lambda p: make_control_prefix(p, seed=123)
)

# ── DCQ columns — empty, filled by generate_dcq_paraphrases_bbc2024.py ───────
df["dcq_A_canonical"]        = df["summary_ref_norm"]
df["dcq_B_para1"]            = ""
df["dcq_C_para2"]            = ""
df["dcq_D_para3"]            = ""
df["dcq_E_para4"]            = ""
df["dcq_choice"]             = ""
df["mem_completion"]         = ""
df["stability_outputs_json"] = ""

# ── Metric columns — empty, populated by detectors ───────────────────────────
metric_cols = [
    "CPS", "EM", "NE", "UAR", "mNED",
    "MaxSpanLen", "NgramHits", "ProxyCount",
    "SLex", "SSem", "SMem", "SProb",
    "RiskScore",
]
for c in metric_cols:
    df[c] = np.nan

df["RiskLevel"]  = ""
df["Confidence"] = ""

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_parquet(OUT_PARQUET, index=False)
print(f"\nSaved  : {OUT_PARQUET}")
print(f"Shape  : {df.shape}")
print(f"Columns: {list(df.columns)}")

# Prefix alignment check
misaligned = df[
    ~df.apply(lambda r: r["summary_ref_norm"].startswith(r["prefix_ref"]), axis=1)
]
if len(misaligned) > 0:
    print(f"\nWARNING: {len(misaligned)} rows where prefix_ref is not aligned!")
    print(misaligned[["xsum_id", "summary_ref_norm", "prefix_ref"]].head(3).to_string())
else:
    print(f"\nPrefix alignment check: OK — all {len(df)} rows aligned")

# Show summary length distribution for verification
lengths = df["summary_ref_norm"].str.split().str.len()
print(f"\nSummary length (tokens): min={lengths.min()}, "
      f"mean={lengths.mean():.1f}, max={lengths.max()}")
print("First 5 summaries:")
for s in df["summary_ref_norm"].head(5):
    print(f"  [{len(s.split())} tokens] {s[:100]}")
