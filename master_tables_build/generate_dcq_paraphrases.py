import json
import re
import time
import argparse
import sys
from pathlib import Path
import pandas as pd

# Ensure project root is on sys.path when running as a script from /data
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openai import APIStatusError
from src.clients.deepseek_client import DeepSeekClient
from src.prompts import DEEPSEEK_PARAPHRASE_PROMPT


DEFAULT_IN_PATH = "master_table_xsum_n300_seed42_v1.parquet"
DEFAULT_OUT_PATH = "master_table_xsum_n300_seed42_v3_dcq4_frozen.parquet"
DEFAULT_LOG_PATH = "dcq4_paraphrase_generation_log.jsonl"


def parse_four_lines(text: str) -> list[str]:
    # remove leading/trailing whitespace and split into non-empty lines
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

    # if the model returned bullets/numbers, strip common prefixes
    cleaned = []
    for ln in lines:
        ln = re.sub(r"^(\d+[\).\s]+|[-•]\s+)", "", ln).strip()
        if ln:
            cleaned.append(ln)

    # keep first 4 non-empty lines
    if len(cleaned) < 4:
        raise ValueError(f"Expected 4 lines, got {len(cleaned)}: {cleaned}")
    return cleaned[:4]


def is_single_sentence(s: str) -> bool:
    # heuristic: allow one terminal punctuation group
    # (XSum summaries may contain abbreviations; keep it permissive)
    return len(re.findall(r"[.!?]", s)) <= 2


def log_jsonl(path: str, payload: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def main():
    parser = argparse.ArgumentParser(description="Generate DCQ paraphrases (k=4) using DeepSeek and freeze into master table.")
    parser.add_argument("--in_path", type=str, default=DEFAULT_IN_PATH)
    parser.add_argument("--out_path", type=str, default=DEFAULT_OUT_PATH)
    parser.add_argument("--log_path", type=str, default=DEFAULT_LOG_PATH)

    parser.add_argument("--limit", type=int, default=None, help="Process at most this many NEW rows (resume-friendly).")
    parser.add_argument("--save_every", type=int, default=10, help="Save parquet after every N processed rows.")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep between API calls (seconds).")

    args = parser.parse_args()

    df = pd.read_parquet(args.in_path)
    client = DeepSeekClient(model="deepseek-chat")

    failures = 0
    warnings = 0
    done = 0  # how many NEW rows processed in this run

    for i, row in df.iterrows():
        # skip if already filled (resume-friendly)
        already_done = all(
            isinstance(row.get(col), str) and row.get(col, "").strip()
            for col in ["dcq_B_para1", "dcq_C_para2", "dcq_D_para3", "dcq_E_para4"]
        )
        if already_done:
            continue

        summary = row["summary_ref_norm"]
        prompt = DEEPSEEK_PARAPHRASE_PROMPT.format(SUMMARY=summary)

        try:
            out = client.generate_text(prompt, temperature=args.temperature, top_p=args.top_p)
            p1, p2, p3, p4 = parse_four_lines(out)

            single_sentence_warning = not (
                is_single_sentence(p1)
                and is_single_sentence(p2)
                and is_single_sentence(p3)
                and is_single_sentence(p4)
            )

            norms = {
                normalize_text(summary),
                normalize_text(p1),
                normalize_text(p2),
                normalize_text(p3),
                normalize_text(p4),
            }
            duplicate_warning = len(norms) < 5

            warning = single_sentence_warning or duplicate_warning
            if warning:
                warnings += 1

            df.at[i, "dcq_A_canonical"] = summary
            df.at[i, "dcq_B_para1"] = p1
            df.at[i, "dcq_C_para2"] = p2
            df.at[i, "dcq_D_para3"] = p3
            df.at[i, "dcq_E_para4"] = p4

            log_jsonl(args.log_path, {
                "row": int(i),
                "xsum_id": row["xsum_id"],
                "status": "ok_with_warning" if warning else "ok",
                "single_sentence_warning": bool(single_sentence_warning),
                "duplicate_warning": bool(duplicate_warning),
            })

            done += 1

        except APIStatusError as e:
            # Graceful stop on insufficient balance (402): save progress and exit cleanly
            msg = str(e)
            if "402" in msg or "Insufficient Balance" in msg:
                df.to_parquet(args.out_path, index=False)
                log_jsonl(args.log_path, {
                    "row": int(i),
                    "xsum_id": row.get("xsum_id", None),
                    "status": "stopped_insufficient_balance",
                    "error": msg,
                    "processed_in_this_run": done,
                    "failures_in_this_run": failures,
                    "warnings_in_this_run": warnings,
                })
                print("Stopped: Insufficient Balance (402). Progress saved to:", args.out_path)
                print(f"Processed in this run: {done}; failures: {failures}; warnings: {warnings}")
                return

            # Other API errors: log and continue
            failures += 1
            log_jsonl(args.log_path, {
                "row": int(i),
                "xsum_id": row.get("xsum_id", None),
                "status": "api_error",
                "error": msg,
            })

        except Exception as e:
            failures += 1
            log_jsonl(args.log_path, {
                "row": int(i),
                "xsum_id": row.get("xsum_id", None),
                "status": "error",
                "error": str(e),
            })

        # periodic save based on processed count (not i)
        if args.save_every and done > 0 and (done % args.save_every == 0):
            df.to_parquet(args.out_path, index=False)
            print(f"Saved progress: {done} rows processed -> {args.out_path}")

        # stop if reached limit
        if args.limit is not None and done >= args.limit:
            break

        # be polite to the API
        time.sleep(args.sleep)

    # final save
    df.to_parquet(args.out_path, index=False)
    print("Done.")
    print(f"Processed in this run: {done}; failures: {failures}; warnings: {warnings}")
    print("Output:", args.out_path)


if __name__ == "__main__":
    main()
