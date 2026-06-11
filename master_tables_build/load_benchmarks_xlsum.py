"""
load_benchmarks_xlsum.py
========================
Creates a frozen master table for the XL-Sum (English) dataset.

Dataset:  csebuetnlp/xlsum, config="english", split="test"
Schema:   { "id", "url", "title", "summary", "text" }
  - "text"    -> document  (news article body)
  - "summary" -> summary_ref (single-sentence summary, analogous to XSum)

Usage:
    python3 data/load_benchmarks_xlsum.py

Output files (place in data/ directory):
    master_table_xlsum_n100_seed42_v1.parquet
    xlsum_en_test_indices_seed42_n100.json

Rationale for choosing XL-Sum as the baseline dataset:
    - Same source (BBC) and same genre as XSum; single-sentence summary format
      ensures task structure is held constant across both datasets.
    - Published in 2021 but substantially less represented on GitHub/Kaggle
      than XSum, making contamination less likely for the evaluated models.
    - Methodologically compatible with the existing pipeline without code changes.
    - Enables the SProb baseline comparison:
        if UAR on XL-Sum >> UAR on XSum collapse items  ->  XSum collapse is
          benchmark-specific, supporting the contamination interpretation.
        if UAR on XL-Sum ≈  UAR on XSum collapse items  ->  collapse reflects
          task structure (short single-sentence BBC summaries), warranting a
          more cautious dual-interpretation in Discussion 5.2.4.
"""

from datasets import load_dataset
import random
import pandas as pd
import json
import re
import unicodedata
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────
SEED         = 42
N_ITEMS      = 100           # baseline experiment; smaller than the main XSum eval (296)
DATASET_NAME = "csebuetnlp/xlsum"
DATASET_CONFIG = "english"
DATASET_SPLIT  = "test"

OUT_PARQUET  = "master_table_xlsum_n100_seed42_v1.parquet"
OUT_INDICES  = "xlsum_en_test_indices_seed42_n100.json"

# ── Load dataset ──────────────────────────────────────────────────────────────
print(f"Loading {DATASET_NAME} ({DATASET_CONFIG}, {DATASET_SPLIT}) ...")
ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
print(f"Total test items : {len(ds)}")
print(f"Columns          : {ds.column_names}")

# ── Sample — draw with a buffer to account for items lost in filtering ────────
random.seed(SEED)
indices = random.sample(range(len(ds)), min(N_ITEMS * 3, len(ds)))
subset  = ds.select(indices)

# Save pre-filter indices for full reproducibility
with open(OUT_INDICES, "w", encoding="utf-8") as f:
    json.dump(indices, f)
print(f"Saved {len(indices)} pre-filter indices -> {OUT_INDICES}")

# ── Filter — same criteria as XSum (load_benchmarks.py) ──────────────────────
def valid(x):
    doc  = x.get("text",    "") or ""
    summ = x.get("summary", "") or ""
    # Minimum lengths match the original XSum filter: doc >= 200, summary >= 20.
    # Upper bound on summary length: XL-Sum occasionally produces multi-sentence
    # summaries; capping at 300 characters keeps the task structure consistent
    # with XSum (single-sentence format required for SMem prefix probing).
    return len(doc) >= 200 and len(summ) >= 20 and len(summ) <= 300

subset = subset.filter(valid)
print(f"After filtering  : {len(subset)} items")

if len(subset) > N_ITEMS:
    subset = subset.select(range(N_ITEMS))
print(f"Final sample     : {len(subset)} items")

# ── Build DataFrame ───────────────────────────────────────────────────────────
rows = []
for idx, ex in enumerate(subset):
    rows.append({
        "item_id":     idx,
        # NOTE: all pipeline scripts expect a column named "xsum_id".
        # We store the XL-Sum article ID here to keep the schema compatible.
        "xsum_id":     ex.get("id", f"xlsum_en_{idx}"),
        "split":       DATASET_SPLIT,
        "document":    ex["text"],      # article body
        "summary_ref": ex["summary"],   # single-sentence reference summary
        "title":       ex.get("title", ""),  # stored for reference; not used by detectors
    })

df = pd.DataFrame(rows)
print(f"\nDataFrame shape  : {df.shape}")
print(df[["xsum_id", "document", "summary_ref"]].head(2).to_string())

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
    m    = min(m, max(1, len(toks) - 1))   # always leave at least one token to complete
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

# ── DCQ columns — empty, filled by generate_dcq_paraphrases_xlsum.py ─────────
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

# Alignment sanity check: prefix_ref must be an exact prefix of summary_ref_norm.
# Misalignment causes run_mem_probe.py to log error_prefix_not_aligned for those rows.
misaligned = df[
    ~df.apply(lambda r: r["summary_ref_norm"].startswith(r["prefix_ref"]), axis=1)
]
if len(misaligned) > 0:
    print(f"\nWARNING: {len(misaligned)} rows where prefix_ref is not aligned!")
    print(misaligned[["xsum_id", "summary_ref_norm", "prefix_ref"]].head(3).to_string())
else:
    print(f"\nPrefix alignment check: OK — all {len(df)} rows aligned")
