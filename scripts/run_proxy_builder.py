#!/usr/bin/env python3
"""
scripts/run_proxy_builder.py

Build external proxy corpus for lexical detector from:
- GitHub (Search API)
- Kaggle (kaggle API)

Outputs:
- proxy_out_txt: normalized + deduped summary-like lines (one per line)
- manifest_out_jsonl: provenance records (search hits, downloads, extracts, errors)
- summary_out_json: aggregate stats for reproducibility

Notes:
- This is a *collection step*. For reproducibility, freeze the outputs and use them offline in run_lexical_detector.py.
- Respects file size limits and extension allowlists.
"""

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import requests
import yaml


# -----------------------
# Utils
# -----------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def write_json(path: str, payload: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def append_jsonl(path: str, payload: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def normalize_line(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    # keep printable ASCII-ish (avoid weird binaries)
    s = re.sub(r"[^\x20-\x7E]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", s.lower())

def looks_like_summary(line: str, min_tokens: int, max_tokens: int, max_periods: int) -> bool:
    toks = tokenize(line)
    if len(toks) < min_tokens or len(toks) > max_tokens:
        return False
    if line.count(".") > max_periods:
        return False
    return True

def extract_summary_like_lines(text: str, min_tokens: int, max_tokens: int, max_periods: int) -> List[str]:
    out: List[str] = []
    for ln in text.splitlines():
        ln = normalize_line(ln)
        if not ln:
            continue
        if looks_like_summary(ln, min_tokens, max_tokens, max_periods):
            out.append(ln)
    return out

def ext_ok(filename: str, allowed: List[str]) -> bool:
    f = filename.lower()
    return any(f.endswith(ext) for ext in allowed)

def safe_unlink(p: Path) -> None:
    try:
        if p.exists():
            p.unlink()
    except Exception:
        pass


# -----------------------
# Query building
# -----------------------

def build_queries(df: pd.DataFrame, cfg: Dict[str, Any]) -> List[str]:
    pb = cfg["proxy_builder"]
    mode = pb.get("query_mode", "ids_and_keywords")
    keywords = pb.get("keywords", [])
    id_cap = int(pb.get("id_query_cap", 80))

    queries: List[str] = []

    # broad keywords
    for kw in keywords:
        queries.append(f'{kw} (summary OR summaries)')

    # targeted ids
    if mode == "ids_and_keywords":
        if "xsum_id" in df.columns:
            ids = df["xsum_id"].astype(str).tolist()
            for xsum_id in ids[:id_cap]:
                queries.append(f'"{xsum_id}" xsum')
    return queries


# -----------------------
# GitHub client
# -----------------------

@dataclass
class GitHubCfg:
    enabled: bool
    token_env: str
    per_query_max_results: int
    max_pages: int
    sleep_seconds: float
    allowed_extensions: List[str]
    deny_repo_substrings: List[str]
    max_file_bytes: int

class GitHubClient:
    def __init__(self, token: str):
        self.base = "https://api.github.com"
        self.s = requests.Session()
        self.s.headers.update({
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "xsum-proxy-builder"
        })

    def search_code(self, q: str, per_page: int = 30, page: int = 1) -> Dict[str, Any]:
        url = f"{self.base}/search/code"
        r = self.s.get(url, params={"q": q, "per_page": per_page, "page": page}, timeout=30)
        r.raise_for_status()
        return r.json()

    @staticmethod
    def to_raw_url(html_url: str) -> Optional[str]:
        m = re.match(r"https://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.*)$", html_url)
        if not m:
            return None
        owner, repo, branch, path = m.group(1), m.group(2), m.group(3), m.group(4)
        return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"

    def fetch_raw(self, raw_url: str) -> Optional[bytes]:
        r = self.s.get(raw_url, timeout=30)
        if r.status_code != 200:
            return None
        return r.content


# -----------------------
# Kaggle client (optional dependency)
# -----------------------

class KaggleClient:
    def __init__(self):
        # Lazy import so script works without kaggle installed
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Kaggle enabled but kaggle package is not available. "
                "Install with: pip install kaggle"
            ) from e

        self.api = KaggleApi()
        self.api.authenticate()

    def list_datasets(self, search: str, max_results: int) -> List[Any]:
        # returns Dataset instances (from kaggle)
        return list(self.api.dataset_list(search=search, max_results=max_results))

    def list_files(self, dataset: str) -> List[Any]:
        return list(self.api.dataset_list_files(dataset).files)

    def download_file(self, dataset: str, file_name: str, out_dir: str) -> Path:
        ensure_dir(out_dir)
        # force=True overwrites
        self.api.dataset_download_file(dataset, file_name, path=out_dir, force=True, quiet=True)
        # Kaggle adds .zip sometimes depending on file type; but dataset_download_file returns exact file
        # Kaggle API typically downloads as <file_name>.zip if compressed.
        # We'll handle both possibilities.
        p = Path(out_dir) / file_name
        if p.exists():
            return p
        z = Path(out_dir) / (file_name + ".zip")
        if z.exists():
            return z
        # fallback: find newest file in dir
        files = sorted(Path(out_dir).glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)
        if files:
            return files[0]
        raise FileNotFoundError(f"Could not find downloaded Kaggle file for {dataset}:{file_name}")


# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str, help="Path to run_config.yaml")
    ap.add_argument("--dry_run", action="store_true", help="Search/log only; skip downloads/extraction")
    ap.add_argument("--force_reset_manifest", action="store_true", help="Delete existing manifest before run")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    master_path = cfg["project"]["frozen_master_table_path"]
    df = pd.read_parquet(master_path)

    pb = cfg["proxy_builder"]
    out_dir = pb.get("output_dir", "data/proxies")
    ensure_dir(out_dir)

    proxy_out_txt = pb.get("proxy_out_txt", f"{out_dir}/xsum_proxy_summaries_norm_dedup_external.txt")
    manifest_out = pb.get("manifest_out_jsonl", f"{out_dir}/proxy_sources_manifest_external.jsonl")
    summary_out = pb.get("summary_out_json", "outputs/proxy_build_summary_external.json")

    if args.force_reset_manifest:
        safe_unlink(Path(manifest_out))

    extraction = pb.get("extraction", {})
    min_tokens = int(extraction.get("min_tokens", 20))
    max_tokens = int(extraction.get("max_tokens", 120))
    max_periods = int(extraction.get("max_periods", 6))

    # ---- GitHub cfg ----
    gh_raw = pb.get("github", {})
    gh_cfg = GitHubCfg(
        enabled=bool(gh_raw.get("enabled", True)),
        token_env=str(gh_raw.get("token_env", "GITHUB_TOKEN")),
        per_query_max_results=int(gh_raw.get("per_query_max_results", 30)),
        max_pages=int(gh_raw.get("max_pages", 2)),
        sleep_seconds=float(gh_raw.get("sleep_seconds", 2.0)),
        allowed_extensions=list(gh_raw.get("allowed_extensions", [".txt", ".jsonl", ".json", ".csv", ".tsv", ".md"])),
        deny_repo_substrings=list(gh_raw.get("deny_repo_substrings", [])),
        max_file_bytes=int(gh_raw.get("max_file_bytes", 500000)),
    )

    gh: Optional[GitHubClient] = None
    if gh_cfg.enabled:
        token = os.getenv(gh_cfg.token_env, "").strip()
        if not token:
            raise RuntimeError(f"GitHub enabled but {gh_cfg.token_env} is not set")
        gh = GitHubClient(token)

    # ---- Kaggle cfg ----
    kg_raw = pb.get("kaggle", {})
    kg_enabled = bool(kg_raw.get("enabled", False))
    kg_max_datasets = int(kg_raw.get("max_datasets", 10))
    kg_dataset_keywords = list(kg_raw.get("dataset_keywords", ["xsum"]))
    kg_file_allow_exts = list(kg_raw.get("file_allow_extensions", [".txt", ".jsonl", ".json", ".csv", ".tsv"]))
    kg_max_file_bytes = int(kg_raw.get("max_file_bytes", 500000))
    kg_sleep = float(kg_raw.get("sleep_seconds", 2.0))

    kaggle_client: Optional[KaggleClient] = None
    if kg_enabled and not args.dry_run:
        kaggle_client = KaggleClient()

    # ---- Build queries ----
    queries = build_queries(df, cfg)
    collected_lines: List[str] = []

    stats = {
        "github": {"files_considered": 0, "hits_logged": 0, "files_downloaded": 0, "extract_events": 0, "failures": 0},
        "kaggle": {"datasets_considered": 0, "files_considered": 0, "files_downloaded": 0, "extract_events": 0, "failures": 0},
    }

    started_at = utc_now()
    append_jsonl(manifest_out, {
        "ts": utc_now(),
        "type": "run_start",
        "master_table": master_path,
        "n_master_rows": int(len(df)),
        "dry_run": bool(args.dry_run),
        "config_path": args.config
    })

    # -----------------------
    # GitHub collection
    # -----------------------
    if gh is not None:
        for q in queries:
            # Qualifiers reduce junk; code search is limited by GitHub anyway
            q_full = f'{q} in:file (xsum OR bbc)'
            for page in range(1, gh_cfg.max_pages + 1):
                try:
                    res = gh.search_code(q_full, per_page=gh_cfg.per_query_max_results, page=page)
                except Exception as e:
                    stats["github"]["failures"] += 1
                    append_jsonl(manifest_out, {"ts": utc_now(), "type": "github_search_error", "query": q_full, "page": page, "error": str(e)})
                    time.sleep(gh_cfg.sleep_seconds)
                    continue

                items = res.get("items", []) or []
                if not items:
                    break

                for it in items:
                    stats["github"]["files_considered"] += 1
                    name = it.get("name", "")
                    repo = (it.get("repository") or {}).get("full_name", "")
                    html_url = it.get("html_url", "")
                    path = it.get("path", "")

                    if gh_cfg.deny_repo_substrings:
                        repo_l = (repo or "").lower()
                        if any(d in repo_l for d in gh_cfg.deny_repo_substrings):
                            continue

                    if not ext_ok(name, gh_cfg.allowed_extensions):
                        continue

                    stats["github"]["hits_logged"] += 1
                    base = {
                        "ts": utc_now(),
                        "type": "github_hit",
                        "query": q_full,
                        "repo": repo,
                        "path": path,
                        "name": name,
                        "html_url": html_url,
                    }
                    append_jsonl(manifest_out, base)

                    if args.dry_run:
                        continue

                    raw_url = GitHubClient.to_raw_url(html_url)
                    if not raw_url:
                        append_jsonl(manifest_out, {**base, "type": "github_download_skip", "reason": "raw_url_parse_failed"})
                        continue

                    try:
                        raw = gh.fetch_raw(raw_url)
                        if raw is None:
                            append_jsonl(manifest_out, {**base, "type": "github_download_skip", "reason": "http_not_200", "raw_url": raw_url})
                            continue
                        if len(raw) > gh_cfg.max_file_bytes:
                            append_jsonl(manifest_out, {**base, "type": "github_download_skip", "reason": "too_large", "bytes": len(raw), "raw_url": raw_url})
                            continue

                        stats["github"]["files_downloaded"] += 1
                        h = sha256_bytes(raw)

                        append_jsonl(manifest_out, {**base, "type": "github_download_ok", "raw_url": raw_url, "sha256": h, "bytes": len(raw)})

                        text = raw.decode("utf-8", errors="ignore")
                        lines = extract_summary_like_lines(text, min_tokens, max_tokens, max_periods)
                        if lines:
                            collected_lines.extend(lines)
                            stats["github"]["extract_events"] += 1
                            append_jsonl(manifest_out, {**base, "type": "github_extract_ok", "n_lines": len(lines), "sha256": h})

                    except Exception as e:
                        stats["github"]["failures"] += 1
                        append_jsonl(manifest_out, {**base, "type": "github_download_error", "raw_url": raw_url, "error": str(e)})

                    time.sleep(gh_cfg.sleep_seconds)

                time.sleep(gh_cfg.sleep_seconds)

    # -----------------------
    # Kaggle collection
    # -----------------------
    if kg_enabled:
        if args.dry_run:
            append_jsonl(manifest_out, {"ts": utc_now(), "type": "kaggle_note", "note": "dry_run: skipping kaggle downloads/extraction"})
        else:
            if kaggle_client is None:
                kaggle_client = KaggleClient()

            tmp_dir = Path(out_dir) / "kaggle_tmp"
            ensure_dir(str(tmp_dir))

            for kw in kg_dataset_keywords:
                try:
                    datasets = kaggle_client.list_datasets(search=kw, max_results=kg_max_datasets)
                except Exception as e:
                    stats["kaggle"]["failures"] += 1
                    append_jsonl(manifest_out, {"ts": utc_now(), "type": "kaggle_dataset_search_error", "query": kw, "error": str(e)})
                    time.sleep(kg_sleep)
                    continue

                for ds in datasets:
                    stats["kaggle"]["datasets_considered"] += 1
                    # Kaggle dataset ref typically like "owner/dataset-name"
                    ds_ref = getattr(ds, "ref", None) or getattr(ds, "datasetRef", None) or str(ds)

                    base_ds = {"ts": utc_now(), "type": "kaggle_dataset_hit", "query": kw, "dataset": ds_ref}
                    append_jsonl(manifest_out, base_ds)

                    try:
                        files = kaggle_client.list_files(ds_ref)
                    except Exception as e:
                        stats["kaggle"]["failures"] += 1
                        append_jsonl(manifest_out, {**base_ds, "type": "kaggle_list_files_error", "error": str(e)})
                        time.sleep(kg_sleep)
                        continue

                    for f in files:
                        # f.name is file name
                        fname = getattr(f, "name", None) or str(f)
                        stats["kaggle"]["files_considered"] += 1

                        if not ext_ok(fname, kg_file_allow_exts):
                            continue

                        try:
                            p = kaggle_client.download_file(ds_ref, fname, out_dir=str(tmp_dir))
                            # if zip, skip to keep it simple (you can extend later to unzip)
                            if p.suffix.lower() == ".zip":
                                append_jsonl(manifest_out, {**base_ds, "type": "kaggle_download_skip", "file": fname, "reason": "zip_not_supported_yet"})
                                continue

                            b = p.read_bytes()
                            if len(b) > kg_max_file_bytes:
                                append_jsonl(manifest_out, {**base_ds, "type": "kaggle_download_skip", "file": fname, "reason": "too_large", "bytes": len(b)})
                                continue

                            stats["kaggle"]["files_downloaded"] += 1
                            h = sha256_bytes(b)
                            append_jsonl(manifest_out, {**base_ds, "type": "kaggle_download_ok", "file": fname, "sha256": h, "bytes": len(b)})

                            text = b.decode("utf-8", errors="ignore")
                            lines = extract_summary_like_lines(text, min_tokens, max_tokens, max_periods)
                            if lines:
                                collected_lines.extend(lines)
                                stats["kaggle"]["extract_events"] += 1
                                append_jsonl(manifest_out, {**base_ds, "type": "kaggle_extract_ok", "file": fname, "n_lines": len(lines), "sha256": h})

                        except Exception as e:
                            stats["kaggle"]["failures"] += 1
                            append_jsonl(manifest_out, {**base_ds, "type": "kaggle_download_error", "file": fname, "error": str(e)})

                        time.sleep(kg_sleep)

                time.sleep(kg_sleep)

    # -----------------------
    # Finalize: dedup & write proxy txt
    # -----------------------
    uniq = sorted(set(collected_lines))
    Path(proxy_out_txt).parent.mkdir(parents=True, exist_ok=True)
    with open(proxy_out_txt, "w", encoding="utf-8") as f:
        for ln in uniq:
            f.write(ln + "\n")

    summary = {
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "master_table": master_path,
        "n_master_rows": int(len(df)),
        "n_queries": len(queries),
        "dry_run": bool(args.dry_run),
        "proxy_out_txt": proxy_out_txt,
        "manifest_out_jsonl": manifest_out,
        "lines_collected_raw": len(collected_lines),
        "lines_unique": len(uniq),
        "stats": stats,
    }
    write_json(summary_out, summary)

    append_jsonl(manifest_out, {"ts": utc_now(), "type": "run_end", **summary})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
