from datasets import load_dataset
import re
import unicodedata
from pathlib import Path

OUT_RAW = Path("data/proxies/xsum_proxy_summaries_raw.txt")
OUT_NORM = Path("data/proxies/xsum_proxy_summaries_norm_dedup.txt")

def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def main():
    ds = load_dataset("EdinburghNLP/xsum")  # public HF mirror
    summaries = []

    for split in ["train", "validation", "test"]:
        for ex in ds[split]:
            summaries.append(ex["summary"])

    OUT_RAW.parent.mkdir(parents=True, exist_ok=True)

    # raw dump
    with OUT_RAW.open("w", encoding="utf-8") as f:
        for s in summaries:
            f.write(s.replace("\n", " ").strip() + "\n")

    # normalized + dedup
    seen = set()
    kept = []
    for s in summaries:
        s2 = norm(s)
        if s2 and s2 not in seen:
            seen.add(s2)
            kept.append(s2)

    with OUT_NORM.open("w", encoding="utf-8") as f:
        for s in kept:
            f.write(s + "\n")

    print("HF proxy built.")
    print("Raw lines:", len(summaries))
    print("Norm+dedup lines:", len(kept))
    print("Files:", OUT_RAW, OUT_NORM)

if __name__ == "__main__":
    main()
