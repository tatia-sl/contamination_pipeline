#!/usr/bin/env python3
"""
scripts/run_proxy_builder_v4.py

Build external proxy corpus for lexical detector from:
- GitHub (Search API)
- Kaggle (kaggle API)

Architecture: single-pass pipeline
  Each file is downloaded exactly once. During extraction, structured rows
  (item_id, xsum_id, split, source, source_detail, source_sha256,
  source_query, source_repo, document, summary_ref) are accumulated
  alongside the plain text lines and written directly to per-source CSVs.
  This eliminates the previous two-pass architecture where extract_structured_proxy_data.py
  re-downloaded the same files from the manifest.

  Downstream: build_proxy_structured_merged.py reads the per-source CSVs
  produced here and merges them — extract_structured_proxy_data.py is no
  longer needed in the normal collection pipeline.

Provenance
  Every row in the output CSVs carries source_sha256, which links it to the
  corresponding github_download_ok / kaggle_download_ok record in manifest.jsonl.
  From that record the full context is recoverable: repo, path, query, timestamp,
  file size, xsum_like_reason, extractor stats.

Outputs (configurable via proxy_builder.* in run_config.yaml):
  data/proxies/proxy_structured_github.csv      structured rows, GitHub source
  data/proxies/proxy_structured_kaggle.csv      structured rows, Kaggle source
  data/proxies/proxy_sources_manifest_*.jsonl   full provenance audit trail
  outputs/proxy_build_summary_*.json            aggregate stats
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import re
import sys
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -----------------------
# Structured row schema
# -----------------------

STRUCTURED_COLS = [
    "item_id",       # stable id: sha256[:12] of source_detail
    "xsum_id",       # matched XSum id from hint (None if not found)
    "split",         # "test" if hint contained split/test signal, else None
    "source",        # "github" | "kaggle"
    "source_detail", # raw_url (GitHub) or "dataset:filename" (Kaggle)
    "source_sha256", # sha256 of the downloaded bytes — links row → manifest
    "source_query",  # search query that found this file
    "source_repo",   # repo full_name (GitHub) or dataset ref (Kaggle)
    "document",      # full document text (None for summary-only sources)
    "summary_ref",   # the extracted summary-like line
]


def make_structured_row(
    summary_ref: str,
    source: str,
    source_detail: str,
    source_sha256: str,
    source_query: str,
    source_repo: str,
    hint_test: Dict[str, Any],
    xsum_id: Optional[str] = None,
    document: Optional[str] = None,
) -> Dict[str, Any]:
    """Build one structured corpus row with full provenance fields."""
    matched = hint_test.get("hint_matched_test_ids_regex_sample", [])
    resolved_xsum_id = xsum_id or (matched[0] if matched else None)
    split_val = "test" if hint_test.get("hint_has_split_test") else None
    item_id = f"{source[:2]}_{sha256_bytes(source_detail.encode())[:12]}"
    return {
        "item_id":       item_id,
        "xsum_id":       resolved_xsum_id,
        "split":         split_val,
        "source":        source,
        "source_detail": source_detail,
        "source_sha256": source_sha256,
        "source_query":  source_query,
        "source_repo":   source_repo,
        "document":      document,
        "summary_ref":   summary_ref,
    }

import pandas as pd
import requests
import yaml
from requests import HTTPError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not installed. Install with 'pip install tqdm' for progress bars")


# -----------------------
# Logging Setup
# -----------------------

def setup_logging(log_file: Optional[str] = None, verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


# -----------------------
# Manifest writer (buffered)
# -----------------------

class ManifestWriter:
    """
    Buffered JSONL manifest writer.
    Opens the file once and keeps the handle open for the duration of the run,
    avoiding per-record open/close overhead.
    """

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def write(self, payload: Dict[str, Any]) -> None:
        self._fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "ManifestWriter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


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


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def normalize_line(s: str) -> str:
    """Normalize to lowercase ASCII-printable, collapse whitespace."""
    s = s.strip().lower()
    # FIX: single pass — previous version called re.sub(whitespace) twice
    s = re.sub(r"[^\x20-\x7E]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", s.lower())


def looks_like_summary(line: str, min_tokens: int, max_tokens: int, max_periods: int) -> bool:
    """
    Heuristic filter: keep lines that plausibly look like news summaries.

    Changes vs previous version:
    - Added alpha_ratio check: at least 50% of tokens must be alphabetic.
      This rejects index rows, numeric sequences, and delimiter noise that
      would otherwise pass length/period checks.
    """
    toks = tokenize(line)
    n = len(toks)
    if n < min_tokens or n > max_tokens:
        return False
    if line.count(".") > max_periods:
        return False
    # FIX: reject lines that are mostly numbers/punctuation (e.g. "1 2 3 4 ...")
    alpha = sum(1 for t in toks if t.isalpha())
    if n > 0 and alpha / n < 0.5:
        return False
    return True


def extract_summary_like_lines(
    text: str, min_tokens: int, max_tokens: int, max_periods: int
) -> List[str]:
    out: List[str] = []
    for ln in text.splitlines():
        ln = normalize_line(ln)
        if not ln:
            continue
        if looks_like_summary(ln, min_tokens, max_tokens, max_periods):
            out.append(ln)
    return out


def extract_text_from_tabular(
    raw: bytes,
    filename: str,
    min_tokens: int,
    max_tokens: int,
    max_periods: int,
    max_rows: int = 2000,
    max_values_per_col: int = 500,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Extract summary-like text from CSV/TSV files.
    Prioritises columns whose names suggest summaries/predictions.
    Falls back to plain-text extraction if pandas parse fails.
    """
    name = (filename or "").lower()
    sep = "\t" if name.endswith(".tsv") else ","
    if not name.endswith(".csv") and not name.endswith(".tsv"):
        try:
            sample = raw[:4096].decode("utf-8", errors="ignore")
            if sample.count("\t") > sample.count(","):
                sep = "\t"
        except Exception:
            sep = ","

    try:
        df_tab = pd.read_csv(
            io.BytesIO(raw),
            sep=sep,
            nrows=max_rows,
            dtype=str,
            on_bad_lines="skip",
            engine="python",
        )
    except Exception:
        try:
            text = raw.decode("utf-8", errors="ignore")
        except Exception:
            return [], {"rows": 0, "kept_lines": 0, "errors": 1, "reason": "decode_failed", "filename": filename}
        lines = extract_summary_like_lines(text, min_tokens, max_tokens, max_periods)
        return lines, {"rows": 0, "kept_lines": len(lines), "errors": 0, "reason": "fallback_text", "filename": filename}

    if df_tab is None or df_tab.empty:
        return [], {"rows": 0, "kept_lines": 0, "errors": 0, "reason": "empty_df", "filename": filename}

    preferred_patterns = [
        "summary", "summ", "prediction", "pred", "generated", "generation",
        "output", "decoded", "hyp", "hypothesis", "model", "candidate",
    ]
    cols = [c for c in df_tab.columns if c is not None]
    cols_l = [str(c).lower() for c in cols]
    preferred_cols = [c for c, cl in zip(cols, cols_l) if any(p in cl for p in preferred_patterns)]
    candidate_cols = preferred_cols if preferred_cols else cols

    out: List[str] = []

    for c in candidate_cols:
        values = df_tab[c].dropna().astype(str).tolist()
        if not values:
            continue
        short = sum(1 for x in values[:200] if len(tokenize(x)) < 8)
        if short > 150 and c not in preferred_cols:
            continue
        for v in values[:max_values_per_col]:
            v = normalize_line(v)
            if v and looks_like_summary(v, min_tokens, max_tokens, max_periods):
                out.append(v)

    out = sorted(set(out))
    return out, {"rows": int(len(df_tab)), "kept_lines": int(len(out)), "errors": 0, "filename": filename}


def extract_text_from_jsonl(
    raw_bytes: bytes,
    filename: str,
    min_tokens: int,
    max_tokens: int,
    max_periods: int,
    prefer_fields: Optional[List[str]] = None,
    require_any_key: Optional[List[str]] = None,
    max_lines: int = 50000,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Parse JSONL and extract summary-like text from selected fields.
    """
    if prefer_fields is None:
        prefer_fields = ["summary", "target", "reference", "highlights", "output", "prediction", "decoded", "text"]
    if require_any_key is None:
        require_any_key = []

    try:
        text = raw_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return [], {"parsed": 0, "kept_objs": 0, "kept_lines": 0, "errors": 1, "reason": "decode_failed"}

    out: List[str] = []
    parsed = errors = kept_objs = 0

    for i, ln in enumerate(text.splitlines()):
        if i >= max_lines:
            break
        ln = ln.strip()
        if not ln or not (ln.startswith("{") and ln.endswith("}")):
            continue
        try:
            obj = json.loads(ln)
            parsed += 1
        except Exception:
            errors += 1
            continue

        if require_any_key and not any(k in obj for k in require_any_key):
            continue
        kept_objs += 1

        for f in prefer_fields:
            v = obj.get(f)
            if v is None or isinstance(v, (dict, list)):
                continue
            v = normalize_line(str(v))
            if v and looks_like_summary(v, min_tokens, max_tokens, max_periods):
                out.append(v)

    out = sorted(set(out))
    return out, {"parsed": parsed, "kept_objs": kept_objs, "kept_lines": len(out), "errors": errors, "filename": filename}


def dispatch_extractor(
    raw: bytes,
    name: str,
    min_tokens: int,
    max_tokens: int,
    max_periods: int,
    require_xsum_key: bool = False,
) -> Tuple[List[str], str, Optional[Dict[str, Any]]]:
    """
    Route to the appropriate extractor based on file extension.
    Returns (lines, extractor_name, stats).

    NOTE: .parquet is intentionally NOT handled here — the format requires
    a columnar deserialiser and cannot be decoded as UTF-8 text. Add a
    dedicated branch with pd.read_parquet(io.BytesIO(raw)) if needed.
    """
    name_l = name.lower()
    if name_l.endswith(".jsonl"):
        require = ["xsum_id"] if require_xsum_key else []
        lines, stats = extract_text_from_jsonl(
            raw_bytes=raw,
            filename=name,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            max_periods=max_periods,
            require_any_key=require,
        )
        return lines, "jsonl", stats
    elif name_l.endswith(".csv") or name_l.endswith(".tsv"):
        lines, stats = extract_text_from_tabular(
            raw=raw,
            filename=name,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            max_periods=max_periods,
        )
        return lines, "tabular", stats
    else:
        text = raw.decode("utf-8", errors="ignore")
        lines = extract_summary_like_lines(text, min_tokens, max_tokens, max_periods)
        return lines, "text", None


def ext_ok(filename: str, allowed: List[str]) -> bool:
    f = filename.lower()
    return any(f.endswith(ext) for ext in allowed)


def safe_unlink(p: Path) -> None:
    try:
        if p.exists():
            p.unlink()
    except Exception:
        pass


def safe_hint_from_bytes(raw: bytes, max_bytes: int = 8192) -> str:
    try:
        if not raw:
            return ""
        head = raw[:max_bytes]
        tail = raw[-max_bytes:] if len(raw) > max_bytes else b""
        combined = head + (b"\n\n---TAIL---\n\n" + tail if tail else b"")
        return combined.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def compute_hint_test_signals(
    hint: str, test_ids: Optional[List[str]] = None, max_matches: int = 3
) -> Dict[str, Any]:
    """
    FIX: removed legacy plain substring match for test_ids (high false-positive rate
    for short numeric IDs). Only regex with word-boundary / id= patterns is used.
    """
    hint_l = (hint or "").lower()
    has_split_test = ("split" in hint_l) and ("test" in hint_l)
    matched_regex: List[str] = []
    if test_ids:
        for tid in test_ids:
            if not tid:
                continue
            pattern = rf'(?i)(\b{re.escape(tid)}\b|id\s*[:=]\s*"?{re.escape(tid)}"?)'
            if re.search(pattern, hint):
                matched_regex.append(tid)
            if len(matched_regex) >= max_matches:
                break
    return {
        "hint_has_split_test": bool(has_split_test),
        "hint_has_any_test_id_regex": bool(matched_regex),
        "hint_matched_test_ids_regex_sample": matched_regex[:max_matches],
    }


def is_xsum_like_hit(repo: str, path: str, name: str, hint: str) -> Tuple[bool, str]:
    """
    Heuristic: decide whether a downloaded artifact is plausibly XSum-related.
    Precision-first ordering.
    """
    repo_l, path_l, name_l, hint_l = (
        (repo or "").lower(), (path or "").lower(),
        (name or "").lower(), (hint or "").lower(),
    )
    if "xsum_id" in hint_l:
        return True, "hint_contains_xsum_id"
    if "xsum" in hint_l and ("summary" in hint_l or "document" in hint_l or "article" in hint_l):
        return True, "hint_contains_xsum_and_schema"
    if "xsum" in repo_l or "xsum" in path_l or "xsum" in name_l:
        return True, "meta_contains_xsum"
    return False, "no_xsum_signal"


# -----------------------
# Search Cache (JSONL-based)
# -----------------------

class SearchCache:
    """
    JSONL-backed cache for GitHub search results.

    FIX: replaced pickle serialisation with JSONL for cross-version portability
    and human-readable inspection. The cache file can be opened and audited
    without any Python code.
    """

    def __init__(self, cache_file: str, max_age_hours: int = 24) -> None:
        self.cache_file = Path(cache_file)
        self.max_age_hours = max_age_hours
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.cache_file.exists():
            return
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        self.cache[entry["key"]] = entry
                    except Exception:
                        continue
            logging.info(f"Loaded cache with {len(self.cache)} entries from {self.cache_file}")
        except Exception as e:
            logging.warning(f"Failed to load cache: {e}")
            self.cache = {}

    def _save(self) -> None:
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                for entry in self.cache.values():
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logging.warning(f"Failed to save cache: {e}")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        entry = self.cache.get(key)
        if entry is None:
            return None
        try:
            cached_time = datetime.fromisoformat(entry["timestamp"])
            age = datetime.now(timezone.utc) - cached_time
            if age.total_seconds() > self.max_age_hours * 3600:
                logging.debug(f"Cache expired for key: {key}")
                return None
        except Exception:
            return None
        logging.debug(f"Cache hit for key: {key}")
        return entry["data"]

    def set(self, key: str, data: Dict[str, Any]) -> None:
        self.cache[key] = {"key": key, "timestamp": utc_now(), "data": data}
        self._save()


# -----------------------
# Config Validation
# -----------------------

def validate_config(cfg: Dict[str, Any]) -> None:
    for key in ("project", "proxy_builder"):
        if key not in cfg:
            raise ValueError(f"Missing required config section: {key}")

    if "frozen_master_table_path" not in cfg["project"]:
        raise ValueError("Missing project.frozen_master_table_path")

    master_path = cfg["project"]["frozen_master_table_path"]
    if not Path(master_path).exists():
        raise ValueError(f"Master table not found: {master_path}")

    pb = cfg["proxy_builder"]

    if pb.get("github", {}).get("enabled"):
        token_env = pb["github"].get("token_env", "GITHUB_TOKEN")
        token = os.getenv(token_env, "").strip()
        if not token:
            raise ValueError(
                f"GitHub enabled but {token_env} not set.\n"
                f"Set it with: export {token_env}='your_token_here'"
            )
        logging.info(f"✓ {token_env} found (length: {len(token)})")

    # FIX 3: Kaggle import check is now a WARNING, not a hard error.
    # This preserves compatibility with environments that only need the GitHub
    # collection path (e.g. CI, build_proxy_structured_merged.py orchestration).
    if pb.get("kaggle", {}).get("enabled"):
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore  # noqa: F401
            logging.info("✓ kaggle package available")
        except ImportError:
            logging.warning(
                "Kaggle is enabled in config but the 'kaggle' package is not installed. "
                "Kaggle collection will be skipped at runtime. "
                "Install with: pip install kaggle"
            )

    # FIX 2: Warn about config values that contradict the hardened defaults in this script.
    # These are not fatal — the config is the source of truth — but surface the drift
    # explicitly so it shows up in logs and is auditable.
    gh_raw = pb.get("github", {})
    _warn_config_drift(gh_raw, "rate_limit_threshold", recommended_min=20, op="lt",
                       msg="rate_limit_threshold < 20 risks frequent 403/429 errors on GitHub Search API (10 req/min limit).")
    _warn_config_drift(gh_raw, "max_file_bytes", recommended_max=1_000_000, op="gt",
                       msg="max_file_bytes > 1 000 000 loads large files into memory before size check; consider ≤ 1 MB.")
    ext_raw = pb.get("extraction", {})
    _warn_config_drift(ext_raw, "max_periods", recommended_max=4, op="gt",
                       msg="max_periods > 4 allows multi-sentence blocks into proxy corpus (noise for SLex).")
    _warn_config_drift(pb, "id_query_cap", recommended_max=40, op="gt",
                       msg="id_query_cap > 40 generates many API calls; at 10 req/min this adds significant wall time.")
    keywords: List[str] = pb.get("keywords", [])
    if "split test" in [k.strip().lower() for k in keywords]:
        logging.warning(
            "Config keyword 'split test' is very broad and not XSum-specific. "
            "It will match any ML repo with train/test splits. Consider removing it."
        )

    logging.info("✓ Configuration validated")


def _warn_config_drift(
    section: Dict[str, Any],
    key: str,
    op: str,
    msg: str,
    recommended_min: Optional[int] = None,
    recommended_max: Optional[int] = None,
) -> None:
    """Emit a warning if a config value crosses a known-bad threshold."""
    val = section.get(key)
    if val is None:
        return
    try:
        v = float(val)
    except (TypeError, ValueError):
        return
    triggered = (op == "lt" and recommended_min is not None and v < recommended_min) or \
                (op == "gt" and recommended_max is not None and v > recommended_max)
    if triggered:
        logging.warning(f"Config drift [{key}={val}]: {msg}")


# -----------------------
# Query building
# -----------------------

def build_queries(df: pd.DataFrame, cfg: Dict[str, Any]) -> List[str]:
    pb = cfg["proxy_builder"]
    mode = pb.get("query_mode", "ids_and_keywords")
    keywords: List[str] = pb.get("keywords", [])
    id_cap = int(pb.get("id_query_cap", 40))

    queries: List[str] = []
    for kw in keywords:
        kw = kw.strip()
        if not kw:
            continue
        queries.append(f"{kw} summary")
        queries.append(f"{kw} summaries")

    if mode == "ids_and_keywords" and "xsum_id" in df.columns:
        for xsum_id in df["xsum_id"].astype(str).tolist()[:id_cap]:
            queries.append(f'"{xsum_id}"')
            queries.append(f'"{xsum_id}" xsum')

    return queries


def build_advanced_queries(df: pd.DataFrame, cfg: Dict[str, Any]) -> List[str]:
    queries = build_queries(df, cfg)
    advanced = [
        '"xsum_id" extension:jsonl',
        '"xsum_id" "summary" extension:jsonl',
        '"xsum_id" "split" "test" extension:jsonl',
        '"xsum_id" extension:csv',
        '"xsum_id" "summary" extension:csv',
        'path:data xsum_id extension:jsonl',
        'path:dataset xsum_id extension:jsonl',
        '"xsum" "xsum_id" "summary" extension:jsonl',
        "filename:xsum extension:jsonl",
        "filename:xsum extension:csv",
    ]
    queries.extend(advanced)
    logging.info(f"Built {len(queries)} search queries ({len(advanced)} advanced)")
    return queries


# -----------------------
# HTTP Session with Retry
# -----------------------

def create_session_with_retry(retries: int = 3, backoff_factor: float = 1.0) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


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
    # FIX: path_deny_substrings is now config-driven instead of hard-coded.
    # Document the exclusion decision explicitly so it is auditable in the thesis.
    path_deny_substrings: List[str]
    max_file_bytes: int
    rate_limit_threshold: int = 20  # FIX: raised from 5; Search API = 10 req/min


class GitHubClient:
    def __init__(self, token: str, use_retry: bool = True) -> None:
        self.base = "https://api.github.com"
        self.s = create_session_with_retry() if use_retry else requests.Session()
        self.s.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "xsum-proxy-builder/3.0",
            }
        )
        self.search_remaining: Optional[int] = None
        self.search_reset: Optional[int] = None
        self.search_limit: Optional[int] = None

    def check_rate_limit(self) -> Dict[str, Any]:
        r = self.s.get(f"{self.base}/rate_limit", timeout=30)
        r.raise_for_status()
        search = r.json()["resources"].get("search", {})
        self.search_remaining = int(search.get("remaining", 0))
        self.search_reset = int(search.get("reset", int(time.time()) + 60))
        self.search_limit = int(search.get("limit", 0))
        reset_time = datetime.fromtimestamp(self.search_reset, tz=timezone.utc)
        return {
            "remaining": self.search_remaining,
            "limit": self.search_limit,
            "reset_at": reset_time.isoformat(),
            "reset_in_seconds": max(0, (reset_time - datetime.now(timezone.utc)).total_seconds()),
            "bucket": "search",
        }

    def wait_if_needed(self, threshold: int = 20) -> None:
        if self.search_remaining is None or self.search_reset is None:
            self.check_rate_limit()

        exhausted = self.search_remaining == 0
        low = self.search_remaining is not None and self.search_remaining < threshold

        if exhausted or low:
            wait_time = self.search_reset - time.time() + 10
            if wait_time > 0:
                reason = "exhausted" if exhausted else f"low ({self.search_remaining}/{self.search_limit})"
                logging.warning(
                    f"GitHub Search rate limit {reason}. Waiting {wait_time:.0f}s until reset..."
                )
                time.sleep(wait_time)
                self.check_rate_limit()
                logging.info(f"Rate limit restored: {self.search_remaining}/{self.search_limit}")

    def verify_token(self) -> bool:
        try:
            limit_info = self.check_rate_limit()
            logging.info(
                f"✓ GitHub API accessible: {limit_info['remaining']}/{limit_info['limit']} remaining"
            )
            r = self.s.get(f"{self.base}/user", timeout=10)
            if r.status_code == 200:
                logging.info(f"  Authenticated as: {r.json().get('login', 'unknown')}")
            return True
        except Exception as e:
            logging.error(f"✗ Token verification failed: {e}")
            return False

    def search_code(self, q: str, per_page: int = 30, page: int = 1) -> Dict[str, Any]:
        r = self.s.get(
            f"{self.base}/search/code",
            params={"q": q, "per_page": per_page, "page": page},
            timeout=30,
        )
        logging.debug(
            f"Rate headers: remaining={r.headers.get('X-RateLimit-Remaining')} "
            f"reset={r.headers.get('X-RateLimit-Reset')}"
        )
        r.raise_for_status()
        if "X-RateLimit-Remaining" in r.headers:
            self.search_remaining = int(r.headers["X-RateLimit-Remaining"])
        if "X-RateLimit-Reset" in r.headers:
            self.search_reset = int(r.headers["X-RateLimit-Reset"])
        if "X-RateLimit-Limit" in r.headers:
            self.search_limit = int(r.headers["X-RateLimit-Limit"])
        return r.json()

    @staticmethod
    def to_raw_url(html_url: str) -> Optional[str]:
        m = re.match(r"https://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.*)$", html_url)
        if not m:
            return None
        owner, repo, branch, path = m.group(1), m.group(2), m.group(3), m.group(4)
        return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"

    def get_content_length(self, raw_url: str) -> Optional[int]:
        """
        FIX: HEAD request to check Content-Length before downloading.
        Avoids pulling large files into memory only to discard them.
        Returns None if the server does not advertise Content-Length.
        """
        try:
            r = self.s.head(raw_url, timeout=10, allow_redirects=True)
            cl = r.headers.get("Content-Length")
            return int(cl) if cl else None
        except Exception:
            return None

    def fetch_raw(self, raw_url: str) -> Optional[bytes]:
        r = self.s.get(raw_url, timeout=30)
        if r.status_code != 200:
            return None
        return r.content


# -----------------------
# Kaggle client
# -----------------------

class KaggleClient:
    def __init__(self) -> None:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
        except Exception as e:
            raise RuntimeError("Kaggle package not available. pip install kaggle") from e
        self.api = KaggleApi()
        self.api.authenticate()
        logging.info("✓ Kaggle API authenticated")

    def list_datasets(self, search: str, max_results: int) -> List[Any]:
        return list(self.api.dataset_list(search=search))[:max_results]

    def list_files(self, dataset: str) -> List[Any]:
        return list(self.api.dataset_list_files(dataset).files)

    def download_file(self, dataset: str, file_name: str, out_dir: str) -> Path:
        ensure_dir(out_dir)
        self.api.dataset_download_file(dataset, file_name, path=out_dir, force=True, quiet=True)
        p = Path(out_dir) / file_name
        if p.exists():
            return p
        z = Path(out_dir) / (file_name + ".zip")
        if z.exists():
            return z
        files = sorted(Path(out_dir).glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)
        if files:
            return files[0]
        raise FileNotFoundError(f"Could not find downloaded Kaggle file for {dataset}:{file_name}")


# -----------------------
# GitHub collection
# -----------------------

def collect_from_github(
    gh: GitHubClient,
    cfg: GitHubCfg,
    queries: List[str],
    extraction_params: Dict[str, int],
    manifest: ManifestWriter,
    cache: Optional[SearchCache] = None,
    dry_run: bool = False,
    test_ids: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns (structured_rows, stats).
    Each row is a dict matching STRUCTURED_COLS — includes full provenance.
    """
    structured_rows: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "files_considered": 0,
        "hits_logged": 0,
        "files_downloaded": 0,
        "extract_events": 0,
        "failures": 0,
        "cache_hits": 0,
        "cache_misses": 0,
    }

    min_t = extraction_params["min_tokens"]
    max_t = extraction_params["max_tokens"]
    max_p = extraction_params["max_periods"]

    queries_iter = tqdm(queries, desc="GitHub queries") if TQDM_AVAILABLE else queries

    for q in queries_iter:
        gh.wait_if_needed(threshold=cfg.rate_limit_threshold)

        # FIX: avoid appending "xsum" when the query already contains it
        q_full = q if "xsum" in q.lower() else f"{q} in:file xsum"

        for page in range(1, cfg.max_pages + 1):
            page_key = f"gh_{hashlib.md5((q_full + str(page)).encode()).hexdigest()}"

            res = None
            if cache:
                res = cache.get(page_key)
                # FIX: increment counter only on actual hit
                if res is not None:
                    stats["cache_hits"] += 1
                    logging.debug(f"Cache hit: {q_full[:60]}... page {page}")
                else:
                    stats["cache_misses"] += 1

            if res is None:
                try:
                    res = gh.search_code(q_full, per_page=cfg.per_query_max_results, page=page)
                    if cache:
                        cache.set(page_key, res)
                except HTTPError as e:
                    status = getattr(e.response, "status_code", None)
                    stats["failures"] += 1
                    manifest.write({
                        "ts": utc_now(), "type": "github_search_error",
                        "query": q_full, "page": page,
                        "status_code": status, "error": str(e),
                    })
                    if status in (403, 429):
                        logging.warning(f"Rate limited (status={status}), waiting for reset…")
                        try:
                            gh.check_rate_limit()
                            gh.wait_if_needed(threshold=1)
                            res = gh.search_code(q_full, per_page=cfg.per_query_max_results, page=page)
                            if cache:
                                cache.set(page_key, res)
                        except Exception as e2:
                            logging.error(f"Retry failed for '{q_full}' page {page}: {e2}")
                            time.sleep(cfg.sleep_seconds)
                            continue
                    else:
                        logging.error(f"Search error for '{q_full}' page {page}: {e}")
                        time.sleep(cfg.sleep_seconds)
                        continue
                except Exception as e:
                    stats["failures"] += 1
                    manifest.write({
                        "ts": utc_now(), "type": "github_search_error",
                        "query": q_full, "page": page, "error": str(e),
                    })
                    logging.error(f"Search error for '{q_full}' page {page}: {e}")
                    time.sleep(cfg.sleep_seconds)
                    continue

            items = (res or {}).get("items") or []
            if not items:
                break

            items_iter = tqdm(items, desc=f"  Page {page}", leave=False) if TQDM_AVAILABLE else items

            for it in items_iter:
                stats["files_considered"] += 1
                name = it.get("name", "")
                repo = (it.get("repository") or {}).get("full_name", "")
                html_url = it.get("html_url", "")
                path = it.get("path", "")
                path_l = (path or "").lower()

                # FIX: path denylist is now config-driven (cfg.path_deny_substrings)
                if cfg.path_deny_substrings and any(bad in path_l for bad in cfg.path_deny_substrings):
                    continue

                if cfg.deny_repo_substrings:
                    repo_l = (repo or "").lower()
                    if any(d in repo_l for d in cfg.deny_repo_substrings):
                        continue

                if not ext_ok(name, cfg.allowed_extensions):
                    continue

                stats["hits_logged"] += 1
                base = {
                    "ts": utc_now(), "type": "github_hit",
                    "query": q_full, "repo": repo,
                    "path": path, "name": name, "html_url": html_url,
                }
                manifest.write(base)

                if dry_run:
                    continue

                raw_url = GitHubClient.to_raw_url(html_url)
                if not raw_url:
                    manifest.write({**base, "type": "github_download_skip", "reason": "raw_url_parse_failed"})
                    continue

                # FIX: HEAD check before downloading to avoid loading oversized files into memory
                cl = gh.get_content_length(raw_url)
                if cl is not None and cl > cfg.max_file_bytes:
                    manifest.write({
                        **base, "type": "github_download_skip",
                        "reason": "too_large_content_length", "bytes": cl, "raw_url": raw_url,
                    })
                    continue

                try:
                    raw = gh.fetch_raw(raw_url)
                    if raw is None:
                        manifest.write({**base, "type": "github_download_skip", "reason": "http_not_200", "raw_url": raw_url})
                        continue

                    if len(raw) > cfg.max_file_bytes:
                        manifest.write({
                            **base, "type": "github_download_skip",
                            "reason": "too_large", "bytes": len(raw), "raw_url": raw_url,
                        })
                        continue

                    stats["files_downloaded"] += 1
                    h = sha256_bytes(raw)
                    hint = safe_hint_from_bytes(raw)
                    xsum_like, xsum_like_reason = is_xsum_like_hit(repo=repo, path=path, name=name, hint=hint)
                    hint_test = compute_hint_test_signals(hint=hint, test_ids=test_ids)

                    manifest.write({
                        **base, "type": "github_download_ok",
                        "raw_url": raw_url, "sha256": h, "bytes": len(raw),
                        "xsum_like_gate": bool(xsum_like), "xsum_like_reason": xsum_like_reason,
                        **hint_test,
                    })

                    lines, extractor, ext_stats = dispatch_extractor(
                        raw=raw, name=name,
                        min_tokens=min_t, max_tokens=max_t, max_periods=max_p,
                        require_xsum_key=True,
                    )

                    event_type = "github_extract_ok" if lines else "github_extract_empty"
                    if lines:
                        stats["extract_events"] += 1
                        for line in lines:
                            structured_rows.append(make_structured_row(
                                summary_ref=line,
                                source="github",
                                source_detail=raw_url,
                                source_sha256=h,
                                source_query=q_full,
                                source_repo=repo,
                                hint_test=hint_test,
                            ))

                    manifest.write({
                        **base, "type": event_type,
                        "n_lines": len(lines), "sha256": h, "extractor": extractor,
                        "xsum_like_gate": bool(xsum_like), "xsum_like_reason": xsum_like_reason,
                        "ext_stats": ext_stats, **hint_test,
                    })

                except Exception as e:
                    stats["failures"] += 1
                    manifest.write({
                        **base, "type": "github_download_error",
                        "raw_url": raw_url, "error": str(e),
                    })
                    logging.error(f"Download error for {raw_url}: {e}")

                time.sleep(cfg.sleep_seconds)

            time.sleep(cfg.sleep_seconds)

    return structured_rows, stats


# -----------------------
# Kaggle collection
# -----------------------

def collect_from_kaggle(
    kaggle_client: KaggleClient,
    cfg: Dict[str, Any],
    extraction_params: Dict[str, int],
    manifest: ManifestWriter,
    dry_run: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns (structured_rows, stats).
    Each row is a dict matching STRUCTURED_COLS — includes full provenance.
    """
    structured_rows: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "datasets_considered": 0,
        "files_considered": 0,
        "files_downloaded": 0,
        "extract_events": 0,
        "failures": 0,
    }

    if dry_run:
        manifest.write({"ts": utc_now(), "type": "kaggle_note", "note": "dry_run: skipping kaggle"})
        return structured_rows, stats

    min_t = extraction_params["min_tokens"]
    max_t = extraction_params["max_tokens"]
    max_p = extraction_params["max_periods"]

    kg_max_datasets = cfg.get("max_datasets", 10)
    kg_dataset_keywords: List[str] = cfg.get("dataset_keywords", ["xsum"])
    kg_file_allow_exts: List[str] = cfg.get("file_allow_extensions", [".txt", ".jsonl", ".json", ".csv", ".tsv"])
    kg_max_file_bytes = int(cfg.get("max_file_bytes", 1_000_000))
    kg_sleep = float(cfg.get("sleep_seconds", 3.0))

    tmp_dir = Path(cfg.get("temp_dir", "data/proxies/kaggle_tmp"))
    ensure_dir(str(tmp_dir))

    keywords_iter = tqdm(kg_dataset_keywords, desc="Kaggle keywords") if TQDM_AVAILABLE else kg_dataset_keywords

    for kw in keywords_iter:
        try:
            datasets = kaggle_client.list_datasets(search=kw, max_results=kg_max_datasets)
        except Exception as e:
            stats["failures"] += 1
            manifest.write({"ts": utc_now(), "type": "kaggle_dataset_search_error", "query": kw, "error": str(e)})
            logging.error(f"Kaggle dataset search error for '{kw}': {e}")
            time.sleep(kg_sleep)
            continue

        datasets_iter = (
            tqdm(datasets, desc=f"  Datasets for '{kw}'", leave=False) if TQDM_AVAILABLE else datasets
        )

        for ds in datasets_iter:
            stats["datasets_considered"] += 1
            ds_ref = getattr(ds, "ref", None) or getattr(ds, "datasetRef", None) or str(ds)
            base_ds = {"ts": utc_now(), "type": "kaggle_dataset_hit", "query": kw, "dataset": ds_ref}
            manifest.write(base_ds)

            try:
                files = kaggle_client.list_files(ds_ref)
            except Exception as e:
                stats["failures"] += 1
                manifest.write({**base_ds, "type": "kaggle_list_files_error", "error": str(e)})
                logging.error(f"Kaggle list files error for {ds_ref}: {e}")
                time.sleep(kg_sleep)
                continue

            for f in files:
                fname = getattr(f, "name", None) or str(f)
                stats["files_considered"] += 1

                if not ext_ok(fname, kg_file_allow_exts):
                    continue

                def handle_bytes(b: bytes, label: str) -> None:
                    if len(b) > kg_max_file_bytes:
                        manifest.write({
                            **base_ds, "type": "kaggle_download_skip",
                            "file": label, "reason": "too_large", "bytes": len(b),
                        })
                        return
                    h = sha256_bytes(b)
                    manifest.write({
                        **base_ds, "type": "kaggle_download_ok",
                        "file": label, "sha256": h, "bytes": len(b),
                    })
                    lines, extractor, ext_stats = dispatch_extractor(
                        raw=b, name=label.split(":")[-1],
                        min_tokens=min_t, max_tokens=max_t, max_periods=max_p,
                        require_xsum_key=False,
                    )
                    if lines:
                        stats["extract_events"] += 1
                        # Kaggle has no search query / rate-limit signals —
                        # construct a minimal hint_test so make_structured_row works.
                        hint_test_kg: Dict[str, Any] = {
                            "hint_has_split_test": False,
                            "hint_has_any_test_id_regex": False,
                            "hint_matched_test_ids_regex_sample": [],
                        }
                        source_detail = f"{ds_ref}:{label}"
                        for line in lines:
                            structured_rows.append(make_structured_row(
                                summary_ref=line,
                                source="kaggle",
                                source_detail=source_detail,
                                source_sha256=h,
                                source_query=kw,
                                source_repo=ds_ref,
                                hint_test=hint_test_kg,
                            ))
                        manifest.write({
                            **base_ds, "type": "kaggle_extract_ok",
                            "file": label, "n_lines": len(lines),
                            "sha256": h, "extractor": extractor, "ext_stats": ext_stats,
                        })

                try:
                    p = kaggle_client.download_file(ds_ref, fname, out_dir=str(tmp_dir))

                    if p.suffix.lower() == ".zip":
                        try:
                            with zipfile.ZipFile(p, "r") as zf:
                                for info in zf.infolist():
                                    if info.is_dir():
                                        continue
                                    if not ext_ok(info.filename, kg_file_allow_exts):
                                        continue
                                    if info.file_size > kg_max_file_bytes:
                                        manifest.write({
                                            **base_ds, "type": "kaggle_download_skip",
                                            "file": info.filename, "reason": "too_large",
                                            "bytes": info.file_size,
                                        })
                                        continue
                                    with zf.open(info) as fzip:
                                        b = fzip.read()
                                        stats["files_downloaded"] += 1
                                        handle_bytes(b, f"{fname}:{info.filename}")
                        except Exception as e:
                            stats["failures"] += 1
                            manifest.write({**base_ds, "type": "kaggle_zip_error", "file": fname, "error": str(e)})
                        continue

                    b = p.read_bytes()
                    stats["files_downloaded"] += 1
                    handle_bytes(b, fname)

                except Exception as e:
                    stats["failures"] += 1
                    manifest.write({**base_ds, "type": "kaggle_download_error", "file": fname, "error": str(e)})
                    logging.error(f"Kaggle download error for {ds_ref}:{fname}: {e}")

                time.sleep(kg_sleep)

            time.sleep(kg_sleep)

    return structured_rows, stats


# -----------------------
# Main
# -----------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build proxy corpus from GitHub and Kaggle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", required=True, type=str, help="Path to run_config.yaml")
    ap.add_argument("--dry_run", action="store_true", help="Search/log only; skip downloads/extraction")
    ap.add_argument("--force_reset_manifest", action="store_true", help="Delete existing manifest before run")
    ap.add_argument("--use_cache", action="store_true", help="Use search results cache (JSONL-backed)")
    ap.add_argument("--cache_max_age_hours", type=int, default=24)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--log_file", type=str, default=None)
    ap.add_argument("--advanced_queries", action="store_true", help="Use advanced GitHub search queries")
    args = ap.parse_args()

    setup_logging(log_file=args.log_file, verbose=args.verbose)
    logging.info("=" * 60)
    logging.info("Proxy Corpus Builder v4.0 — single-pass structured output")
    logging.info("=" * 60)

    cfg = load_yaml(args.config)
    validate_config(cfg)

    # Load master table
    master_path = cfg["project"]["frozen_master_table_path"]
    logging.info(f"Loading master table from: {master_path}")
    df = pd.read_parquet(master_path)
    logging.info(f"  Loaded {len(df)} rows")

    # FIX: single split filter (previous version applied it twice, corrupting before/after counts)
    rows_total = len(df)
    if "split" in df.columns:
        df = df[df["split"].astype(str).str.lower() == "test"].copy()
        logging.info(f"  Filtered to split=='test': {len(df)} rows (was {rows_total})")
    else:
        logging.warning("  Column 'split' not found; proceeding without split filtering")

    rows_used = len(df)

    # Build test_ids for hint matching
    test_ids: List[str] = []
    if "xsum_id" in df.columns:
        test_ids = df["xsum_id"].dropna().astype(str).tolist()
        logging.info(f"  Prepared test_ids: {len(test_ids)}")

    pb = cfg["proxy_builder"]
    out_dir = pb.get("output_dir", "data/proxies")
    ensure_dir(out_dir)

    manifest_path = pb.get("manifest_out_jsonl", f"{out_dir}/proxy_sources_manifest_external.jsonl")
    summary_out = pb.get("summary_out_json", "outputs/proxy_build_summary_external.json")

    if args.force_reset_manifest:
        safe_unlink(Path(manifest_path))
        logging.info("Manifest reset.")

    extraction = pb.get("extraction", {})
    extraction_params = {
        "min_tokens": int(extraction.get("min_tokens", 12)),
        "max_tokens": int(extraction.get("max_tokens", 140)),
        "max_periods": int(extraction.get("max_periods", 4)),
    }
    logging.info(f"Extraction params: {extraction_params}")

    cache: Optional[SearchCache] = None
    if args.use_cache:
        cache_file = f"{out_dir}/search_cache.jsonl"
        cache = SearchCache(cache_file, max_age_hours=args.cache_max_age_hours)
        logging.info(f"Cache enabled (max age: {args.cache_max_age_hours}h) → {cache_file}")

    # GitHub config
    gh_raw = pb.get("github", {})
    gh_cfg = GitHubCfg(
        enabled=bool(gh_raw.get("enabled", True)),
        token_env=str(gh_raw.get("token_env", "GITHUB_TOKEN")),
        per_query_max_results=int(gh_raw.get("per_query_max_results", 20)),
        max_pages=int(gh_raw.get("max_pages", 2)),
        sleep_seconds=float(gh_raw.get("sleep_seconds", 2.0)),
        # FIX 1: .parquet is stripped at runtime even if present in config —
        # dispatch_extractor has no parquet branch and would silently produce garbage
        # via the UTF-8 text fallback.  Remove this guard once a pd.read_parquet branch
        # is added to dispatch_extractor.
        allowed_extensions=[
            e for e in gh_raw.get("allowed_extensions", [".txt", ".jsonl", ".json", ".csv", ".tsv"])
            if e.lower() != ".parquet"
        ],
        deny_repo_substrings=list(gh_raw.get("deny_repo_substrings", [])),
        path_deny_substrings=list(gh_raw.get("path_deny_substrings", [])),
        max_file_bytes=int(gh_raw.get("max_file_bytes", 1_000_000)),
        rate_limit_threshold=int(gh_raw.get("rate_limit_threshold", 20)),
    )

    gh: Optional[GitHubClient] = None
    if gh_cfg.enabled:
        token = os.getenv(gh_cfg.token_env, "").strip()
        if not token:
            raise RuntimeError(f"GitHub enabled but {gh_cfg.token_env} is not set")
        gh = GitHubClient(token, use_retry=True)
        if not gh.verify_token():
            raise RuntimeError("GitHub token verification failed")

    kg_raw = pb.get("kaggle", {})
    kg_enabled = bool(kg_raw.get("enabled", False))
    kaggle_client: Optional[KaggleClient] = None
    if kg_enabled and not args.dry_run:
        logging.info("Initializing Kaggle client…")
        # FIX 3: graceful degradation — if the kaggle package is absent, skip Kaggle
        # collection and continue with GitHub-only mode.  This preserves compatibility
        # with environments where only GitHub collection is needed (CI, orchestration
        # via build_proxy_structured_merged.py) without requiring pip install kaggle.
        try:
            kaggle_client = KaggleClient()
        except RuntimeError as e:
            logging.warning(f"Kaggle client unavailable, skipping Kaggle collection: {e}")
            kg_enabled = False

    queries = build_advanced_queries(df, cfg) if args.advanced_queries else build_queries(df, cfg)
    logging.info(f"Total queries: {len(queries)}")

    all_stats: Dict[str, Any] = {"github": {}, "kaggle": {}}
    started_at = utc_now()

    with ManifestWriter(manifest_path) as manifest:
        manifest.write({
            "ts": utc_now(), "type": "run_start",
            "master_table": master_path,
            "n_master_rows": rows_used,
            "dry_run": bool(args.dry_run),
            "config_path": args.config,
            "use_cache": args.use_cache,
            "advanced_queries": args.advanced_queries,
        })

        # GitHub
        if gh is not None:
            logging.info("\n" + "=" * 60)
            logging.info("Starting GitHub collection…")
            gh_rows, gh_stats = collect_from_github(
                gh=gh, cfg=gh_cfg, queries=queries,
                extraction_params=extraction_params,
                manifest=manifest, cache=cache,
                dry_run=args.dry_run, test_ids=test_ids,
            )
            all_stats["github"] = gh_stats
            logging.info(
                f"GitHub done: {gh_stats['files_downloaded']} downloaded, "
                f"{len(gh_rows)} rows, {gh_stats['failures']} failures"
            )
        else:
            gh_rows = []

        # Kaggle
        if kg_enabled:
            logging.info("\n" + "=" * 60)
            logging.info("Starting Kaggle collection…")
            if kaggle_client is None and not args.dry_run:
                kaggle_client = KaggleClient()
            kg_rows, kg_stats = collect_from_kaggle(
                kaggle_client=kaggle_client,  # type: ignore[arg-type]
                cfg=kg_raw,
                extraction_params=extraction_params,
                manifest=manifest,
                dry_run=args.dry_run,
            )
            all_stats["kaggle"] = kg_stats
            logging.info(
                f"Kaggle done: {kg_stats['files_downloaded']} downloaded, "
                f"{len(kg_rows)} rows, {kg_stats['failures']} failures"
            )
        else:
            kg_rows = []

        # ── Write per-source structured CSVs ──────────────────────────────────
        # These are the direct inputs to build_proxy_structured_merged.py.
        # extract_structured_proxy_data.py is no longer needed in normal runs.
        logging.info("\n" + "=" * 60)
        logging.info("Writing structured CSVs…")

        def write_structured_csv(rows: List[Dict[str, Any]], path: str, label: str) -> int:
            """Deduplicate by summary_ref + source_sha256, write CSV, return row count."""
            ensure_dir(str(Path(path).parent))
            if not rows:
                pd.DataFrame(columns=STRUCTURED_COLS).to_csv(path, index=False, encoding="utf-8")
                logging.info(f"  {label}: 0 rows (empty) → {path}")
                return 0
            df_out = pd.DataFrame(rows, columns=STRUCTURED_COLS)
            before = len(df_out)
            df_out = df_out.drop_duplicates(
                subset=["summary_ref", "source_sha256"], keep="first"
            ).reset_index(drop=True)
            dropped = before - len(df_out)
            if not args.dry_run:
                df_out.to_csv(path, index=False, encoding="utf-8")
            logging.info(
                f"  {label}: {len(df_out)} rows "
                f"(deduped {dropped} within-source) → {path}"
            )
            return len(df_out)

        pb_cfg = cfg["proxy_builder"]
        out_dir_path = pb_cfg.get("output_dir", "data/proxies")
        github_csv  = pb_cfg.get("github_structured_out",
                                  f"{out_dir_path}/proxy_structured_github.csv")
        kaggle_csv  = pb_cfg.get("kaggle_structured_out",
                                  f"{out_dir_path}/proxy_structured_kaggle.csv")

        n_gh = write_structured_csv(gh_rows,  github_csv,  "GitHub")
        n_kg = write_structured_csv(kg_rows,  kaggle_csv,  "Kaggle")
        n_total_rows = n_gh + n_kg

        # ── Build summary ─────────────────────────────────────────────────────
        all_summary_refs = [r["summary_ref"] for r in gh_rows + kg_rows]
        n_raw = len(all_summary_refs)
        n_uniq = len(set(all_summary_refs))

        summary = {
            "started_at_utc": started_at,
            "finished_at_utc": utc_now(),
            "master_table": master_path,
            "n_master_rows_total": rows_total,
            "n_master_rows_used": rows_used,
            "split_filter": "test" if "split" in df.columns else None,
            "n_queries": len(queries),
            "dry_run": bool(args.dry_run),
            "use_cache": args.use_cache,
            "advanced_queries": args.advanced_queries,
            "manifest_out_jsonl": manifest_path,
            "github_structured_out": github_csv,
            "kaggle_structured_out": kaggle_csv,
            "rows_github": n_gh,
            "rows_kaggle": n_kg,
            "rows_total": n_total_rows,
            "summary_refs_raw": n_raw,
            "summary_refs_unique": n_uniq,
            "dedup_rate": f"{(1 - n_uniq / n_raw) * 100:.2f}%" if n_raw else "0%",
            "stats": all_stats,
        }

        write_json(summary_out, summary)
        manifest.write({"ts": utc_now(), "type": "run_end", **summary})

    logging.info("=" * 60)
    logging.info(f"Summary : {summary_out}")
    logging.info(f"Manifest: {manifest_path}")
    logging.info(f"GitHub  : {github_csv}  ({n_gh} rows)")
    logging.info(f"Kaggle  : {kaggle_csv}  ({n_kg} rows)")
    logging.info("Next step: python3 scripts/build_proxy_structured_merged.py --config configs/run_config.yaml")
    logging.info("Done!")


if __name__ == "__main__":
    main()
