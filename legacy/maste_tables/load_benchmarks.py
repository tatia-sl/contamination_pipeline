from datasets import load_dataset
import random
import pandas as pd
import json
import re
import unicodedata
import numpy as np

SEED = 42
N_ITEMS = 300

ds = load_dataset("EdinburghNLP/xsum")
test = ds["test"]

random.seed(SEED)
indices = random.sample(range(len(test)), N_ITEMS)
subset = test.select(indices)

with open("xsum_test_indices_seed42_n300.json", "w", encoding="utf-8") as f:
    json.dump(indices, f)
def valid(x):
    return len(x["document"]) >= 200 and len(x["summary"]) >= 20

subset = subset.filter(valid)


rows = []
for idx, ex in enumerate(subset):
    rows.append({
        "item_id": idx,
        "xsum_id": ex["id"],
        "split": "test",
        "document": ex["document"],
        "summary_ref": ex["summary"],
    })

df = pd.DataFrame(rows)
print(df.shape)
df.head(2)


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

df["document_norm"] = df["document"].map(normalize_text)
df["summary_ref_norm"] = df["summary_ref"].map(normalize_text)
def make_prefix(summary: str, frac: float = 0.4, min_tokens: int = 12) -> str:
    toks = summary.split()
    m = max(min_tokens, int(len(toks) * frac))
    m = min(m, max(1, len(toks) - 1))  # чтобы было что дописывать
    return " ".join(toks[:m])

df["prefix_ref"] = df["summary_ref_norm"].map(lambda s: make_prefix(s, frac=0.4, min_tokens=12))


def make_control_prefix(prefix: str, seed: int = 123) -> str:
    rnd = random.Random(seed)
    toks = prefix.split()
    rnd.shuffle(toks)
    return " ".join(toks)

df["control_prefix"] = df["prefix_ref"].map(lambda p: make_control_prefix(p, seed=123))
df["dcq_A_canonical"] = df["summary_ref_norm"]
df["dcq_B_para1"] = ""
df["dcq_C_para2"] = ""
df["dcq_D_para3"] = ""
df["dcq_E_para4"] = ""
df["dcq_choice"] = ""          # A/B/C/D
df["mem_completion"] = ""      # что дописала модель
df["stability_outputs_json"] = ""  # список из 30 в JSON строке


metric_cols = [
    "CPS", "EM", "NE", "UAR", "mNED",
    "MaxSpanLen", "NgramHits", "ProxyCount",
    "SLex", "SSem", "SMem", "SProb",
    "RiskScore"
]
for c in metric_cols:
    df[c] = np.nan

df["RiskLevel"] = ""
df["Confidence"] = ""
df.to_csv("master_table_xsum_n300_seed42_v1.csv", index=False)
df.to_parquet("master_table_xsum_n300_seed42_v1.parquet", index=False)
