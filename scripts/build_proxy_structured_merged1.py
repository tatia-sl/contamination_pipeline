#!/usr/bin/env python3
"""
Build reproducible structured proxy corpus for lexical detector.

Pipeline:
1) Optional collection via run_proxy_builder_improved.py
2) Structured extraction from Kaggle downloads -> proxy_structured_kaggle.csv
3) Structured re-parse from GitHub manifest -> proxy_structured_github.csv
4) Merge -> proxy_structured_merged.csv
5) Write build summary JSON

This script closes the reproducibility gap by orchestrating all steps in one place.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: List[str], quiet: bool = False) -> None:
    if not quiet:
        print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def normalize_summary_for_dedupe(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


def resolve_manifest_path(cfg: Dict[str, Any], user_manifest: str | None) -> str:
    if user_manifest:
        return user_manifest

    cfg_manifest = str(cfg.get("proxy_builder", {}).get("manifest_out_jsonl", "")).strip()
    candidates = [
        cfg_manifest,
        "data/proxies/proxy_sources_manifest_external.jsonl",
        "data/proxies/proxy_sources_manifest_external_2026-02-06.jsonl",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    return candidates[0] if candidates[0] else "data/proxies/proxy_sources_manifest_external.jsonl"


def read_csv_if_exists(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["item_id", "xsum_id", "split", "source", "source_detail", "document", "summary_ref"])
    return pd.read_csv(p, dtype=str)


def write_empty_structured_csv(path: str) -> None:
    cols = ["item_id", "xsum_id", "split", "source", "source_detail", "document", "summary_ref"]
    ensure_parent(path)
    pd.DataFrame(columns=cols).to_csv(path, index=False, encoding="utf-8")


def read_json_if_exists(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def main() -> None:
    ap = argparse.ArgumentParser(description="Reproducible builder for proxy_structured_merged.csv")
    ap.add_argument("--config", default="configs/run_config.yaml")

    ap.add_argument("--structured_out", default="data/proxies/proxy_structured_kaggle.csv")
    ap.add_argument("--github_structured_out", default="data/proxies/proxy_structured_github.csv")
    ap.add_argument("--merged_out", default="data/proxies/proxy_structured_merged.csv")
    ap.add_argument("--summary_out", default="outputs/proxy_structured_merged_build_summary.json")

    ap.add_argument("--manifest", default=None, help="Path to proxy manifest JSONL (auto-resolved if omitted)")
    ap.add_argument("--kaggle_dir", default="data/proxies/kaggle_tmp")
    ap.add_argument("--github_rate_limit_delay", type=float, default=2.0)
    ap.add_argument("--github_max_files", type=int, default=None)

    ap.add_argument("--dedupe_mode", choices=["none", "summary_norm", "doc_summary_hash"], default="none")

    ap.add_argument("--skip_collect", action="store_true")
    ap.add_argument("--skip_extract_kaggle", action="store_true")
    ap.add_argument("--skip_reparse_github", action="store_true")
    ap.add_argument("--skip_merge", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    master_table = str(cfg.get("project", {}).get("frozen_master_table_path", "")).strip()
    manifest = resolve_manifest_path(cfg, args.manifest)
    proxy_build_summary = str(cfg.get("proxy_builder", {}).get("summary_out_json", "outputs/proxy_build_summary_external.json")).strip()

    root = Path(__file__).resolve().parents[1]
    py = sys.executable

    if not args.skip_collect:
        run_cmd([py, str(root / "scripts" / "run_proxy_builder_improved.py"), "--config", args.config], quiet=args.quiet)

    proxy_build_meta = read_json_if_exists(proxy_build_summary)
    kaggle_collection_ran = bool(proxy_build_meta.get("kaggle_collection_ran"))
    kaggle_effective_status = str(proxy_build_meta.get("kaggle_effective_status", "unknown"))

    if not args.skip_extract_kaggle:
        if not args.skip_collect and not kaggle_collection_ran:
            write_empty_structured_csv(args.structured_out)
            if not args.quiet:
                print(
                    "Warning: current collection did not run Kaggle "
                    f"(status={kaggle_effective_status}); "
                    f"writing empty {args.structured_out} to avoid stale kaggle_tmp reuse"
                )
        else:
            cmd = [
                py,
                str(root / "scripts" / "extract_structured_proxy_data.py"),
                "--kaggle-dir",
                args.kaggle_dir,
                "--output",
                args.structured_out,
            ]
            if master_table:
                cmd.extend(["--master-table", master_table])
            if args.quiet:
                cmd.append("--quiet")
            run_cmd(cmd, quiet=args.quiet)

    github_step_status = "skipped" if args.skip_reparse_github else "not_started"
    github_step_error = None
    if not args.skip_reparse_github:
        if not Path(manifest).exists():
            raise FileNotFoundError(f"Manifest not found: {manifest}")
        cmd = [
            py,
            str(root / "scripts" / "extract_structured_proxy_data.py"),
            "--manifest",
            manifest,
            "--output",
            args.github_structured_out,
            "--github-rate-limit-delay",
            str(args.github_rate_limit_delay),
        ]
        if args.github_max_files is not None:
            cmd.extend(["--github-max-files", str(args.github_max_files)])
        cmd = [x for x in cmd if x]
        if args.quiet:
            cmd.append("--quiet")
        try:
            run_cmd(cmd, quiet=args.quiet)
            github_step_status = "ok"
        except subprocess.CalledProcessError as e:
            # Best-effort mode: keep pipeline running even if GitHub structured extraction finds 0 rows.
            # This makes Kaggle-only builds reproducible without manual retries.
            github_step_status = "failed_best_effort"
            github_step_error = str(e)
            write_empty_structured_csv(args.github_structured_out)
            if not args.quiet:
                print(
                    "Warning: GitHub structured extraction failed; "
                    f"continuing with empty {args.github_structured_out}"
                )

    summary: Dict[str, Any] = {
        "started_at_utc": utc_now(),
        "config": args.config,
        "master_table": master_table,
        "manifest": manifest,
        "proxy_build_summary": proxy_build_summary,
        "kaggle_effective_status": kaggle_effective_status,
        "kaggle_collection_ran": kaggle_collection_ran,
        "kaggle_dir": args.kaggle_dir,
        "structured_out": args.structured_out,
        "github_structured_out": args.github_structured_out,
        "github_step_status": github_step_status,
        "github_step_error": github_step_error,
        "merged_out": args.merged_out,
        "dedupe_mode": args.dedupe_mode,
    }

    if not args.skip_merge:
        df_k = read_csv_if_exists(args.structured_out)
        df_g = read_csv_if_exists(args.github_structured_out)

        base_cols = ["item_id", "xsum_id", "split", "source", "source_detail", "document", "summary_ref"]
        for c in base_cols:
            if c not in df_k.columns:
                df_k[c] = ""
            if c not in df_g.columns:
                df_g[c] = ""

        df_k = df_k[base_cols].copy()
        df_g = df_g[base_cols].copy()

        merged = pd.concat([df_k, df_g], ignore_index=True)
        before = len(merged)

        if args.dedupe_mode == "summary_norm":
            merged["_norm"] = normalize_summary_for_dedupe(merged["summary_ref"])
            merged = merged.drop_duplicates(subset=["_norm"], keep="first").drop(columns=["_norm"])
        elif args.dedupe_mode == "doc_summary_hash":
            k = merged["document"].fillna("").astype(str) + "||" + merged["summary_ref"].fillna("").astype(str)
            merged["_hash"] = pd.util.hash_pandas_object(k, index=False).astype(str)
            merged = merged.drop_duplicates(subset=["_hash"], keep="first").drop(columns=["_hash"])

        ensure_parent(args.merged_out)
        merged.to_csv(args.merged_out, index=False, encoding="utf-8")

        norm = normalize_summary_for_dedupe(merged["summary_ref"])
        summary.update(
            {
                "rows_kaggle_structured": int(len(df_k)),
                "rows_github_structured": int(len(df_g)),
                "rows_merged_before_dedupe": int(before),
                "rows_merged_after_dedupe": int(len(merged)),
                "rows_removed_by_dedupe": int(before - len(merged)),
                "rows_empty_summary": int((merged["summary_ref"].fillna("").astype(str).str.strip() == "").sum()),
                "rows_unique_summary_norm": int(norm.nunique(dropna=False)),
                "source_rows": merged["source"].fillna("").value_counts(dropna=False).to_dict(),
            }
        )

    summary["finished_at_utc"] = utc_now()

    ensure_parent(args.summary_out)
    with open(args.summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if not args.quiet:
        print("Done.")
        print("Summary:", args.summary_out)
        if not args.skip_merge:
            print("Merged CSV:", args.merged_out)


if __name__ == "__main__":
    main()
