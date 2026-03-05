#!/usr/bin/env python3
"""
scripts/run_proxy_builder_improved.py

Build external proxy corpus for lexical detector from:
- GitHub (Search API)
- Kaggle (kaggle API)

Improvements:
- Rate limit monitoring and auto-wait
- Retry logic with exponential backoff
- Search results caching
- Progress bars with tqdm
- Better error handling and logging
- Config validation
- Advanced search queries

Outputs:
- proxy_out_txt: normalized + deduped summary-like lines (one per line)
- manifest_out_jsonl: provenance records (search hits, downloads, extracts, errors)
- summary_out_json: aggregate stats for reproducibility

Notes:
- This is a *collection step*. For reproducibility, freeze the outputs and use them offline.
- Respects file size limits and extension allowlists.
"""

import argparse
import hashlib
import json
import logging
import os
import pickle
import re
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

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
    """Configure logging"""
    level = logging.DEBUG if verbose else logging.INFO
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )


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

    Strategy:
    - parse with pandas (robust, on_bad_lines='skip')
    - prefer columns whose names look like summaries/predictions
    - fallback to any object/string columns
    - normalize + filter with looks_like_summary()

    Returns: list of normalized lines.
    """
    name = (filename or "").lower()

    # detect delimiter
    sep = ","
    if name.endswith(".tsv"):
        sep = "\t"
    elif name.endswith(".csv"):
        sep = ","
    else:
        # best-effort: sniff for tabs if many
        try:
            sample = raw[:4096].decode("utf-8", errors="ignore")
            if sample.count("\t") > sample.count(","):
                sep = "\t"
        except Exception:
            sep = ","

    # Try parse
    try:
        import io
        buf = io.BytesIO(raw)
        # engine="python" is more permissive with weird CSVs
        df_tab = pd.read_csv(
            buf,
            sep=sep,
            nrows=max_rows,
            dtype=str,
            on_bad_lines="skip",
            engine="python"
        )
    except Exception:
        # fallback: treat as plain text lines
        try:
            text = raw.decode("utf-8", errors="ignore")
        except Exception:
            return [], {"rows": 0, "kept_lines": 0, "errors": 1, "reason": "decode_failed"}
        lines = extract_summary_like_lines(text, min_tokens, max_tokens, max_periods)
        stats = {
            "rows": 0,
            "kept_lines": len(lines),
            "errors": 0,
            "reason": "fallback_text",
            "filename": filename,
        }
        return lines, stats

    if df_tab is None or df_tab.empty:
        return [], {"rows": 0, "kept_lines": 0, "errors": 0, "reason": "empty_df", "filename": filename}

    # Column prioritization
    preferred_patterns = [
        "summary", "summ", "prediction", "pred", "generated", "generation",
        "output", "decoded", "hyp", "hypothesis", "model", "candidate"
    ]

    cols = [c for c in df_tab.columns if c is not None]
    cols_l = [str(c).lower() for c in cols]

    preferred_cols = []
    for c, cl in zip(cols, cols_l):
        if any(p in cl for p in preferred_patterns):
            preferred_cols.append(c)

    # fallback to all columns if no preferred
    candidate_cols = preferred_cols if preferred_cols else cols

    out: List[str] = []

    def add_value(v: str) -> None:
        if v is None:
            return
        v = str(v)
        v = normalize_line(v)
        if not v:
            return
        if looks_like_summary(v, min_tokens, max_tokens, max_periods):
            out.append(v)

    # Extract up to max_values_per_col per column
    for c in candidate_cols:
        series = df_tab[c]
        # only strings
        values = series.dropna().astype(str).tolist()
        if not values:
            continue

        # reduce obvious noise: if most values are very short, skip column
        short = sum(1 for x in values[:200] if len(tokenize(x)) < 8)
        if short > 150 and c in preferred_cols:
            # even preferred column can be junk, but be slightly tolerant
            pass
        elif short > 150 and c not in preferred_cols:
            continue

        # sample / cap
        for v in values[:max_values_per_col]:
            add_value(v)

    # Dedup in-file
    out = sorted(set(out))
    stats = {
        "rows": int(len(df_tab)),
        "kept_lines": int(len(out)),
        "errors": 0,
        "filename": filename,
    }
    return out, stats


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

    Returns:
      (lines, stats)

    - prefer_fields: fields to extract text from (in priority order)
    - require_any_key: if provided, only keep JSON objects that contain at least one of these keys
      (useful to gate to XSum-like objects, e.g., ['xsum_id'])
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
    parsed = 0
    kept_objs = 0
    errors = 0

    for i, ln in enumerate(text.splitlines()):
        if i >= max_lines:
            break
        ln = ln.strip()
        if not ln:
            continue
        # quick reject non-json lines
        if not (ln.startswith("{") and ln.endswith("}")):
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

        # Extract from preferred fields
        for f in prefer_fields:
            v = obj.get(f)
            if v is None:
                continue
            # allow nested structures but only stringify if it's a scalar-ish value
            if isinstance(v, (dict, list)):
                continue
            v = normalize_line(str(v))
            if not v:
                continue
            if looks_like_summary(v, min_tokens, max_tokens, max_periods):
                out.append(v)

    out = sorted(set(out))
    stats = {
        "parsed": parsed,
        "kept_objs": kept_objs,
        "kept_lines": len(out),
        "errors": errors,
        "filename": filename,
    }
    return out, stats


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
    """
    Small content preview used for cheap signature checks (schema/header keys).
    Uses head+tail to reduce false negatives when identifiers are not at the beginning.
    Keeps reproducibility (stored via derived boolean/reason, not raw bytes).
    """
    try:
        if not raw:
            return ""
        n = int(max_bytes)
        head = raw[:n]
        tail = raw[-n:] if len(raw) > n else b""
        if tail:
            b = head + b"\n\n---TAIL---\n\n" + tail
        else:
            b = head
        return b.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def compute_hint_test_signals(hint: str, test_ids: Optional[List[str]] = None, max_matches: int = 3) -> Dict[str, Any]:
    hint_l = (hint or "").lower()
    has_split_test = ("split" in hint_l) and ("test" in hint_l)
    matched: List[str] = []
    matched_regex: List[str] = []
    if test_ids:
        for tid in test_ids:
            if not tid:
                continue
            # simple substring (legacy)
            if tid in hint:
                matched.append(tid)
            # regex with word boundaries / id=: patterns to avoid spurious digit matches
            pattern = rf'(?i)(\b{re.escape(tid)}\b|id\s*[:=]\s*"?{re.escape(tid)}"?)'
            if re.search(pattern, hint):
                matched_regex.append(tid)
            if len(matched_regex) >= max_matches:
                break
    return {
        "hint_has_split_test": bool(has_split_test),
        "hint_has_any_test_id": bool(len(matched) > 0),
        "hint_has_any_test_id_regex": bool(len(matched_regex) > 0),
        "hint_matched_test_ids_sample": matched,
        "hint_matched_test_ids_regex_sample": matched_regex[:max_matches],
    }

def is_xsum_like_hit(repo: str, path: str, name: str, hint: str) -> Tuple[bool, str]:
    """
    Heuristic: decide whether a downloaded artifact is plausibly XSum-related.
    Returns: (flag, reason)

    Precision-first ordering:
      1) Strong schema cues in content preview (xsum_id; xsum + summary/document)
      2) Metadata cues in repo/path/name (contains 'xsum')
    """
    repo_l = (repo or "").lower()
    path_l = (path or "").lower()
    name_l = (name or "").lower()
    hint_l = (hint or "").lower()

    if "xsum_id" in hint_l:
        return True, "hint_contains_xsum_id"
    if ("xsum" in hint_l) and (("summary" in hint_l) or ("document" in hint_l) or ("article" in hint_l)):
        return True, "hint_contains_xsum_and_schema"
    if ("xsum" in repo_l) or ("xsum" in path_l) or ("xsum" in name_l):
        return True, "meta_contains_xsum"
    return False, "no_xsum_signal"


# -----------------------
# Caching
# -----------------------

class SearchCache:
    """Cache for GitHub search results"""
    
    def __init__(self, cache_file: str, max_age_hours: int = 24):
        self.cache_file = Path(cache_file)
        self.max_age_hours = max_age_hours
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._load()
    
    def _load(self) -> None:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                logging.info(f"Loaded cache with {len(self.cache)} entries")
            except Exception as e:
                logging.warning(f"Failed to load cache: {e}")
                self.cache = {}
    
    def _save(self) -> None:
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logging.warning(f"Failed to save cache: {e}")
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        cached_time = datetime.fromisoformat(entry['timestamp'])
        age = datetime.now(timezone.utc) - cached_time
        
        if age.total_seconds() > self.max_age_hours * 3600:
            logging.debug(f"Cache expired for key: {key}")
            return None
        
        logging.debug(f"Cache hit for key: {key}")
        return entry['data']
    
    def set(self, key: str, data: Dict[str, Any]) -> None:
        self.cache[key] = {
            'timestamp': utc_now(),
            'data': data
        }
        self._save()


# -----------------------
# Config Validation
# -----------------------

def validate_config(cfg: Dict[str, Any]) -> None:
    """Validate configuration before running"""
    required = ["project", "proxy_builder"]
    for key in required:
        if key not in cfg:
            raise ValueError(f"Missing required config section: {key}")
    
    if "frozen_master_table_path" not in cfg["project"]:
        raise ValueError("Missing project.frozen_master_table_path")
    
    master_path = cfg["project"]["frozen_master_table_path"]
    if not Path(master_path).exists():
        raise ValueError(f"Master table not found: {master_path}")
    
    pb = cfg["proxy_builder"]
    
    # GitHub validation
    if pb.get("github", {}).get("enabled"):
        gh = pb["github"]
        token_env = gh.get("token_env", "GITHUB_TOKEN")
        token = os.getenv(token_env, "").strip()
        if not token:
            raise ValueError(
                f"GitHub enabled but {token_env} not set in environment.\n"
                f"Set it with: export {token_env}='your_token_here'"
            )
        logging.info(f"✓ {token_env} found (length: {len(token)})")
    
    # Kaggle validation
    if pb.get("kaggle", {}).get("enabled"):
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
        except ImportError:
            raise ValueError(
                "Kaggle enabled but kaggle package not installed.\n"
                "Install with: pip install kaggle"
            )
    
    logging.info("✓ Configuration validated")


# -----------------------
# Query building
# -----------------------

def build_queries(df: pd.DataFrame, cfg: Dict[str, Any]) -> List[str]:
    pb = cfg["proxy_builder"]
    mode = pb.get("query_mode", "ids_and_keywords")
    keywords = pb.get("keywords", [])
    id_cap = int(pb.get("id_query_cap", 80))

    queries: List[str] = []

    # broad keywords — keep syntax simple for GitHub code search
    for kw in keywords:
        kw = kw.strip()
        if not kw:
            continue
        # Two variants: plain and with "summary"
        queries.append(f'{kw} summary')
        queries.append(f'{kw} summaries')

    # targeted ids
    if mode == "ids_and_keywords":
        if "xsum_id" in df.columns:
            ids = df["xsum_id"].astype(str).tolist()
            for xsum_id in ids[:id_cap]:
                # Avoid parentheses/OR to comply with GitHub code search syntax
                queries.append(f'"{xsum_id}"')
                queries.append(f'"{xsum_id}" xsum')

    return queries


def build_advanced_queries(df: pd.DataFrame, cfg: Dict[str, Any]) -> List[str]:
    """Build advanced search queries with GitHub-specific syntax"""
    queries = build_queries(df, cfg)
    
    # Add advanced patterns
    advanced = [
        # Precision-first: prefer XSum signature keys over generic "document/summary"
        '"xsum_id" extension:jsonl',
        '"xsum_id" "summary" extension:jsonl',
        '"xsum_id" "split" "test" extension:jsonl',
        '"xsum_id" extension:csv',
        '"xsum_id" "summary" extension:csv',
        '"xsum_id" extension:parquet',

        # Additional XSum-oriented patterns (still relatively precise)
        'path:data xsum_id extension:jsonl',
        'path:dataset xsum_id extension:jsonl',
        '"xsum" "xsum_id" "summary" extension:jsonl',

        # Keep some broad-but-less-harmful queries (optional)
        'filename:xsum extension:jsonl',
        'filename:xsum extension:csv',
    ]
    
    queries.extend(advanced)
    logging.info(f"Built {len(queries)} search queries ({len(advanced)} advanced)")
    
    return queries


# -----------------------
# HTTP Session with Retry
# -----------------------

def create_session_with_retry(retries: int = 3, backoff_factor: float = 1.0) -> requests.Session:
    """Create requests session with retry logic"""
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
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
    max_file_bytes: int
    rate_limit_threshold: int = 100  # Wait if remaining < this


class GitHubClient:
    def __init__(self, token: str, use_retry: bool = True):
        self.base = "https://api.github.com"
        if use_retry:
            self.s = create_session_with_retry()
        else:
            self.s = requests.Session()
        
        self.s.headers.update({
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "xsum-proxy-builder/2.0"
        })

        # Track Search API bucket explicitly (code search uses the "search" bucket, not "core")
        self.search_remaining: Optional[int] = None
        self.search_reset: Optional[int] = None
        self.search_limit: Optional[int] = None

    def check_rate_limit(self) -> Dict[str, Any]:
        """Check current rate limit status"""
        url = f"{self.base}/rate_limit"
        r = self.s.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()

        # IMPORTANT: /search/* endpoints use the "search" bucket
        search = data["resources"].get("search", {})
        self.search_remaining = int(search.get("remaining", 0))
        self.search_reset = int(search.get("reset", int(time.time()) + 60))
        self.search_limit = int(search.get("limit", 0))

        reset_time = datetime.fromtimestamp(self.search_reset, tz=timezone.utc)
        time_until_reset = (reset_time - datetime.now(timezone.utc)).total_seconds()

        return {
            "remaining": self.search_remaining,
            "limit": self.search_limit,
            "reset_at": reset_time.isoformat(),
            "reset_in_seconds": max(0, time_until_reset),
            "bucket": "search",
        }
    
    def wait_if_needed(self, threshold: int = 100) -> None:
        """Wait if rate limit is too low"""
        # Ensure we have search bucket info
        if self.search_remaining is None or self.search_reset is None:
            self.check_rate_limit()

        # If empty, always wait until reset (prevents 403 spam)
        if self.search_remaining == 0 and self.search_reset:
            wait_time = self.search_reset - time.time() + 10  # buffer
            if wait_time > 0:
                logging.warning(
                    f"GitHub Search rate limit exhausted ({self.search_remaining}/{self.search_limit}). "
                    f"Waiting {wait_time:.0f}s until reset..."
                )
                time.sleep(wait_time)
                self.check_rate_limit()
                logging.info(f"Rate limit restored: {self.search_remaining}/{self.search_limit}")
            return

        # Otherwise wait when low
        if self.search_remaining is not None and self.search_remaining < threshold:
            wait_time = self.search_reset - time.time() + 10  # buffer
            if wait_time > 0:
                logging.warning(
                    f"GitHub Search rate limit low ({self.search_remaining}/{self.search_limit}). "
                    f"Waiting {wait_time:.0f}s until reset..."
                )
                time.sleep(wait_time)
                self.check_rate_limit()
                logging.info(f"Rate limit restored: {self.search_remaining}/{self.search_limit}")

    def verify_token(self) -> bool:
        """Verify that the token is valid and working"""
        try:
            limit_info = self.check_rate_limit()
            logging.info(
                f"✓ GitHub API accessible ({limit_info.get('bucket','search')}): "
                f"{limit_info['remaining']}/{limit_info['limit']} remaining"
            )
            
            # Get user info
            r = self.s.get(f"{self.base}/user", timeout=10)
            if r.status_code == 200:
                user = r.json()
                logging.info(f"  Authenticated as: {user.get('login', 'unknown')}")
            
            return True
        except Exception as e:
            logging.error(f"✗ Token verification failed: {e}")
            return False

    def search_code(self, q: str, per_page: int = 30, page: int = 1) -> Dict[str, Any]:
        url = f"{self.base}/search/code"
        r = self.s.get(url, params={"q": q, "per_page": per_page, "page": page}, timeout=30)

        logging.debug(
            f"Rate headers: remaining={r.headers.get('X-RateLimit-Remaining')} "
            f"reset={r.headers.get('X-RateLimit-Reset')} "
            f"limit={r.headers.get('X-RateLimit-Limit')}"
        )

        r.raise_for_status()

        # Update rate limit from headers
        if "X-RateLimit-Remaining" in r.headers:
            self.search_remaining = int(r.headers["X-RateLimit-Remaining"])
        if "X-RateLimit-Reset" in r.headers:
            self.search_reset = int(r.headers["X-RateLimit-Reset"])
        if "X-RateLimit-Limit" in r.headers:
            self.search_limit = int(r.headers["X-RateLimit-Limit"])

        return r.json()


    @staticmethod
    def to_raw_url(html_url: str) -> Optional[str]:
        """Convert GitHub blob URL to raw content URL"""
        m = re.match(r"https://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.*)$", html_url)
        if not m:
            return None
        owner, repo, branch, path = m.group(1), m.group(2), m.group(3), m.group(4)
        return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"

    def fetch_raw(self, raw_url: str) -> Optional[bytes]:
        """Fetch raw file content"""
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
        logging.info("✓ Kaggle API authenticated")

    def list_datasets(self, search: str, max_results: int) -> List[Any]:
        # returns Dataset instances (from kaggle)
        ds = list(self.api.dataset_list(search=search))
        return ds[:max_results]


    def list_files(self, dataset: str) -> List[Any]:
        return list(self.api.dataset_list_files(dataset).files)

    def download_file(self, dataset: str, file_name: str, out_dir: str) -> Path:
        ensure_dir(out_dir)
        # force=True overwrites
        self.api.dataset_download_file(dataset, file_name, path=out_dir, force=True, quiet=True)
        
        # Try to find the downloaded file
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
# Main Collection Logic
# -----------------------

def collect_from_github(
    gh: GitHubClient,
    cfg: GitHubCfg,
    queries: List[str],
    extraction_params: Dict[str, int],
    manifest_out: str,
    cache: Optional[SearchCache] = None,
    dry_run: bool = False,
    test_ids: Optional[List[str]] = None,
) -> Tuple[List[str], Dict[str, int]]:
    """
    Collect proxy corpus from GitHub
    
    Returns:
        Tuple of (collected_lines, stats)
    """
    collected_lines: List[str] = []
    stats = {
        "files_considered": 0,
        "hits_logged": 0,
        "files_downloaded": 0,
        "extract_events": 0,
        "failures": 0,
        "cache_hits": 0,
        "cache_misses": 0
    }
    
    min_tokens = extraction_params['min_tokens']
    max_tokens = extraction_params['max_tokens']
    max_periods = extraction_params['max_periods']
    
    # Progress bar setup
    queries_iter = tqdm(queries, desc="GitHub queries") if TQDM_AVAILABLE else queries
    
    for q in queries_iter:
        # Check rate limit before each query
        gh.wait_if_needed(threshold=cfg.rate_limit_threshold)
        
        # Build full query with qualifiers
        # Keep qualifiers simple to satisfy GitHub code search syntax
        q_full = f'{q} in:file xsum'
        
        for page in range(1, cfg.max_pages + 1):
            page_key = f"gh_search_{hashlib.md5((q_full + str(page)).encode()).hexdigest()}"
            
            res = None
            if cache:
                res = cache.get(page_key)
                stats["cache_hits"] += 1
                if res:
                    logging.debug(f"Using cached results for query: {q_full[:50]}... page {page}")
                else:
                    stats["cache_hits"] -= 1  # revert hit increment if miss
            
            if res is None:
                stats["cache_misses"] += 1
                try:
                    res = gh.search_code(q_full, per_page=cfg.per_query_max_results, page=page)
                    if cache:
                        cache.set(page_key, res)
                except HTTPError as e:
                    status = getattr(e.response, "status_code", None)
                    if status in (403, 429):
                        stats["failures"] += 1
                        append_jsonl(manifest_out, {
                            "ts": utc_now(),
                            "type": "github_search_error",
                            "query": q_full,
                            "page": page,
                            "status_code": status,
                            "error": str(e),
                            "note": "rate_limited_or_forbidden; will wait and retry once"
                        })
                        logging.warning(
                            f"Search rate limited (status={status}) for '{q_full[:80]}...' page {page}. "
                            "Waiting for reset and retrying once..."
                        )
                        try:
                            gh.check_rate_limit()
                            gh.wait_if_needed(threshold=1)
                            res = gh.search_code(q_full, per_page=cfg.per_query_max_results, page=page)
                            if cache:
                                cache.set(page_key, res)
                        except Exception as e2:
                            stats["failures"] += 1
                            append_jsonl(manifest_out, {
                                "ts": utc_now(),
                                "type": "github_search_error",
                                "query": q_full,
                                "page": page,
                                "status_code": status,
                                "error": str(e2),
                                "note": "retry_failed_after_wait"
                            })
                            logging.error(f"Retry failed for '{q_full}' page {page}: {e2}")
                            time.sleep(cfg.sleep_seconds)
                            continue
                    else:
                        stats["failures"] += 1
                        append_jsonl(manifest_out, {
                            "ts": utc_now(),
                            "type": "github_search_error",
                            "query": q_full,
                            "page": page,
                            "status_code": status,
                            "error": str(e)
                        })
                        logging.error(f"Search error for '{q_full}' page {page}: {e}")
                        time.sleep(cfg.sleep_seconds)
                        continue
                except Exception as e:
                    stats["failures"] += 1
                    append_jsonl(manifest_out, {
                        "ts": utc_now(),
                        "type": "github_search_error",
                        "query": q_full,
                        "page": page,
                        "error": str(e)
                    })
                    logging.error(f"Search error for '{q_full}' page {page}: {e}")
                    time.sleep(cfg.sleep_seconds)
                    continue

            items = res.get("items", []) or []
            if not items:
                break

            # Progress bar for items
            items_iter = tqdm(items, desc=f"  Page {page}", leave=False) if TQDM_AVAILABLE else items
            
            for it in items_iter:
                stats["files_considered"] += 1
                name = it.get("name", "")
                repo = (it.get("repository") or {}).get("full_name", "")
                html_url = it.get("html_url", "")
                path = it.get("path", "")
                path_l = (path or "").lower()
                if any(bad in path_l for bad in ["eval", "evaluation", "results", "outputs", "metrics"]):
                    continue


                # Filter by repo substrings
                if cfg.deny_repo_substrings:
                    repo_l = (repo or "").lower()
                    if any(d in repo_l for d in cfg.deny_repo_substrings):
                        continue

                # Filter by extension
                if not ext_ok(name, cfg.allowed_extensions):
                    continue

                stats["hits_logged"] += 1
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

                if dry_run:
                    continue

                # Download and extract
                raw_url = GitHubClient.to_raw_url(html_url)
                if not raw_url:
                    append_jsonl(manifest_out, {
                        **base,
                        "type": "github_download_skip",
                        "reason": "raw_url_parse_failed"
                    })
                    continue

                try:
                    raw = gh.fetch_raw(raw_url)
                    if raw is None:
                        append_jsonl(manifest_out, {
                            **base,
                            "type": "github_download_skip",
                            "reason": "http_not_200",
                            "raw_url": raw_url
                        })
                        continue
                    
                    if len(raw) > cfg.max_file_bytes:
                        append_jsonl(manifest_out, {
                            **base,
                            "type": "github_download_skip",
                            "reason": "too_large",
                            "bytes": len(raw),
                            "raw_url": raw_url
                        })
                        continue

                    stats["files_downloaded"] += 1
                    h = sha256_bytes(raw)
                    hint = safe_hint_from_bytes(raw, max_bytes=8192)
                    xsum_like, xsum_like_reason = is_xsum_like_hit(
                        repo=repo, path=path, name=name, hint=hint
                    )
                    hint_test = compute_hint_test_signals(hint=hint, test_ids=test_ids, max_matches=3)

                    append_jsonl(manifest_out, {
                        **base,
                        "type": "github_download_ok",
                        "raw_url": raw_url,
                        "sha256": h,
                        "bytes": len(raw),
                        "xsum_like_gate": bool(xsum_like),
                        "xsum_like_reason": xsum_like_reason,
                        "hint_has_xsum_id": ("xsum_id" in (hint or "").lower()),
                        "hint_has_split_test": hint_test["hint_has_split_test"],
                        "hint_has_any_test_id": hint_test["hint_has_any_test_id"],
                        "hint_has_any_test_id_regex": hint_test["hint_has_any_test_id_regex"],
                        "hint_matched_test_ids_sample": hint_test["hint_matched_test_ids_sample"],
                    })

                    # Extract summary-like lines with format-aware extraction + XSum gating
                    extractor = "text"
                    jsonl_stats: Optional[Dict[str, Any]] = None
                    tabular_stats: Optional[Dict[str, Any]] = None
                    name_l = name.lower()
                    if name_l.endswith(".jsonl"):
                        extractor = "jsonl"
                        lines, jsonl_stats = extract_text_from_jsonl(
                            raw_bytes=raw,
                            filename=name,
                            min_tokens=min_tokens,
                            max_tokens=max_tokens,
                            max_periods=max_periods,
                            require_any_key=["xsum_id"],
                        )
                        logging.debug(f"[jsonl] {name}: {jsonl_stats}")
                    elif name_l.endswith(".csv") or name_l.endswith(".tsv"):
                        extractor = "tabular"
                        lines, tabular_stats = extract_text_from_tabular(
                            raw=raw,
                            filename=name,
                            min_tokens=min_tokens,
                            max_tokens=max_tokens,
                            max_periods=max_periods,
                        )
                    else:
                        text = raw.decode("utf-8", errors="ignore")
                        lines = extract_summary_like_lines(text, min_tokens, max_tokens, max_periods)

                    # Always log extraction result
                    if lines and len(lines) > 0:
                        collected_lines.extend(lines)
                        stats["extract_events"] += 1
                        event_type = "github_extract_ok"
                    else:
                        event_type = "github_extract_empty"

                    append_jsonl(manifest_out, {
                        **base,
                        "type": event_type,
                        "n_lines": int(len(lines) if lines else 0),
                        "sha256": h,
                        "extractor": extractor,
                        "xsum_like_gate": bool(xsum_like),
                        "xsum_like_reason": xsum_like_reason,
                        "hint_has_xsum_id": ("xsum_id" in (hint or "").lower()),
                        "hint_has_split_test": hint_test["hint_has_split_test"],
                        "hint_has_any_test_id": hint_test["hint_has_any_test_id"],
                        "hint_has_any_test_id_regex": hint_test["hint_has_any_test_id_regex"],
                        "hint_matched_test_ids_sample": hint_test["hint_matched_test_ids_sample"],
                        "hint_matched_test_ids_regex_sample": hint_test["hint_matched_test_ids_regex_sample"],
                        "jsonl_stats": jsonl_stats,
                        "tabular_stats": tabular_stats,
                    })

                except Exception as e:
                    stats["failures"] += 1
                    append_jsonl(manifest_out, {
                        **base,
                        "type": "github_download_error",
                        "raw_url": raw_url,
                        "error": str(e)
                    })
                    logging.error(f"Download error for {raw_url}: {e}")

                time.sleep(cfg.sleep_seconds)

            time.sleep(cfg.sleep_seconds)
    
    return collected_lines, stats


def collect_from_kaggle(
    kaggle_client: KaggleClient,
    cfg: Dict[str, Any],
    extraction_params: Dict[str, int],
    manifest_out: str,
    dry_run: bool = False
    ) -> Tuple[List[str], Dict[str, int]]:
    """
    Collect proxy corpus from Kaggle
    
    Returns:
        Tuple of (collected_lines, stats)
    """
    collected_lines: List[str] = []
    stats = {
        "datasets_considered": 0,
        "files_considered": 0,
        "files_downloaded": 0,
        "extract_events": 0,
        "failures": 0
    }
    
    if dry_run:
        append_jsonl(manifest_out, {
            "ts": utc_now(),
            "type": "kaggle_note",
            "note": "dry_run: skipping kaggle downloads/extraction"
        })
        return collected_lines, stats
    
    min_tokens = extraction_params['min_tokens']
    max_tokens = extraction_params['max_tokens']
    max_periods = extraction_params['max_periods']
    
    kg_max_datasets = cfg.get("max_datasets", 10)
    kg_dataset_keywords = cfg.get("dataset_keywords", ["xsum"])
    kg_file_allow_exts = cfg.get("file_allow_extensions", [".txt", ".jsonl", ".json", ".csv", ".tsv"])
    kg_max_file_bytes = cfg.get("max_file_bytes", 500000)
    kg_sleep = cfg.get("sleep_seconds", 2.0)
    
    tmp_dir = Path(cfg.get("temp_dir", "data/proxies/kaggle_tmp"))
    ensure_dir(str(tmp_dir))
    
    keywords_iter = tqdm(kg_dataset_keywords, desc="Kaggle keywords") if TQDM_AVAILABLE else kg_dataset_keywords
    
    for kw in keywords_iter:
        try:
            datasets = kaggle_client.list_datasets(search=kw, max_results=kg_max_datasets)
        except Exception as e:
            stats["failures"] += 1
            append_jsonl(manifest_out, {
                "ts": utc_now(),
                "type": "kaggle_dataset_search_error",
                "query": kw,
                "error": str(e)
            })
            logging.error(f"Kaggle dataset search error for '{kw}': {e}")
            time.sleep(kg_sleep)
            continue

        datasets_iter = tqdm(datasets, desc=f"  Datasets for '{kw}'", leave=False) if TQDM_AVAILABLE else datasets
        
        for ds in datasets_iter:
            stats["datasets_considered"] += 1
            ds_ref = getattr(ds, "ref", None) or getattr(ds, "datasetRef", None) or str(ds)

            base_ds = {
                "ts": utc_now(),
                "type": "kaggle_dataset_hit",
                "query": kw,
                "dataset": ds_ref
            }
            append_jsonl(manifest_out, base_ds)

            try:
                files = kaggle_client.list_files(ds_ref)
            except Exception as e:
                stats["failures"] += 1
                append_jsonl(manifest_out, {
                    **base_ds,
                    "type": "kaggle_list_files_error",
                    "error": str(e)
                })
                logging.error(f"Kaggle list files error for {ds_ref}: {e}")
                time.sleep(kg_sleep)
                continue

            for f in files:
                fname = getattr(f, "name", None) or str(f)
                stats["files_considered"] += 1

                if not ext_ok(fname, kg_file_allow_exts):
                    continue

                try:
                    p = kaggle_client.download_file(ds_ref, fname, out_dir=str(tmp_dir))
                    
                    # Skip zips
                    def handle_bytes(b: bytes, label: str) -> None:
                        if len(b) > kg_max_file_bytes:
                            append_jsonl(manifest_out, {
                                **base_ds,
                                "type": "kaggle_download_skip",
                                "file": label,
                                "reason": "too_large",
                                "bytes": len(b)
                            })
                            return
                        h = sha256_bytes(b)
                        append_jsonl(manifest_out, {
                            **base_ds,
                            "type": "kaggle_download_ok",
                            "file": label,
                            "sha256": h,
                            "bytes": len(b)
                        })
                        text = b.decode("utf-8", errors="ignore")
                        lines = extract_summary_like_lines(text, min_tokens, max_tokens, max_periods)
                        if lines:
                            collected_lines.extend(lines)
                            stats["extract_events"] += 1
                            append_jsonl(manifest_out, {
                                **base_ds,
                                "type": "kaggle_extract_ok",
                                "file": label,
                                "n_lines": len(lines),
                                "sha256": h
                            })

                    if p.suffix.lower() == ".zip":
                        try:
                            with zipfile.ZipFile(p, "r") as zf:
                                for info in zf.infolist():
                                    if info.is_dir():
                                        continue
                                    if not ext_ok(info.filename, kg_file_allow_exts):
                                        continue
                                    if info.file_size > kg_max_file_bytes:
                                        append_jsonl(manifest_out, {
                                            **base_ds,
                                            "type": "kaggle_download_skip",
                                            "file": info.filename,
                                            "reason": "too_large",
                                            "bytes": info.file_size
                                        })
                                        continue
                                    with zf.open(info) as fzip:
                                        b = fzip.read()
                                        stats["files_downloaded"] += 1
                                        handle_bytes(b, f"{fname}:{info.filename}")
                        except Exception as e:
                            stats["failures"] += 1
                            append_jsonl(manifest_out, {
                                **base_ds,
                                "type": "kaggle_zip_error",
                                "file": fname,
                                "error": str(e)
                            })
                        continue

                    b = p.read_bytes()
                    stats["files_downloaded"] += 1
                    handle_bytes(b, fname)

                except Exception as e:
                    stats["failures"] += 1
                    append_jsonl(manifest_out, {
                        **base_ds,
                        "type": "kaggle_download_error",
                        "file": fname,
                        "error": str(e)
                    })
                    logging.error(f"Kaggle download error for {ds_ref}:{fname}: {e}")

                time.sleep(kg_sleep)

        time.sleep(kg_sleep)
    
    return collected_lines, stats


# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser(
        description="Build proxy corpus from GitHub and Kaggle",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True, type=str, help="Path to run_config.yaml")
    ap.add_argument("--dry_run", action="store_true", help="Search/log only; skip downloads/extraction")
    ap.add_argument("--force_reset_manifest", action="store_true", help="Delete existing manifest before run")
    ap.add_argument("--use_cache", action="store_true", help="Use search results cache")
    ap.add_argument("--cache_max_age_hours", type=int, default=24, help="Cache max age in hours")
    ap.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    ap.add_argument("--log_file", type=str, default=None, help="Log file path")
    ap.add_argument("--advanced_queries", action="store_true", help="Use advanced search queries")
    args = ap.parse_args()

    # Setup logging
    setup_logging(log_file=args.log_file, verbose=args.verbose)
    
    logging.info("="*60)
    logging.info("Proxy Corpus Builder - Enhanced Version")
    logging.info("="*60)

    # Load and validate config
    logging.info(f"Loading config from: {args.config}")
    cfg = load_yaml(args.config)
    validate_config(cfg)

    # Load master table
    master_path = cfg["project"]["frozen_master_table_path"]
    logging.info(f"Loading master table from: {master_path}")
    df = pd.read_parquet(master_path)
    logging.info(f"  Loaded {len(df)} rows")
    # Filter to test split if available
    split_value = "test"
    if "split" in df.columns:
        before = len(df)
        df = df[df["split"].astype(str) == split_value].copy()
        logging.info(f"  Filtered to split=='{split_value}': {len(df)} rows (was {before})")
    # Build test-id list for hint matching
    test_ids: List[str] = []
    if "xsum_id" in df.columns:
        test_ids = [str(x) for x in df["xsum_id"].dropna().astype(str).tolist()]
        logging.info(f"  Prepared test_ids for hint matching: {len(test_ids)} ids")

    # Use only test split for proxy building (avoid collecting signals for train/val)
    master_rows_total = len(df)
    if "split" in df.columns:
        before = len(df)
        df = df[df["split"].astype(str).str.lower() == "test"].copy()
        after = len(df)
        logging.info(f"  Filtered to split=='test': {after} rows (was {before})")
    else:
        before = len(df)
        after = len(df)
        logging.warning("  Column 'split' not found in master table; proceeding without split filtering")

    pb = cfg["proxy_builder"]
    out_dir = pb.get("output_dir", "data/proxies")
    ensure_dir(out_dir)

    proxy_out_txt = pb.get("proxy_out_txt", f"{out_dir}/xsum_proxy_summaries_norm_dedup_external.txt")
    manifest_out = pb.get("manifest_out_jsonl", f"{out_dir}/proxy_sources_manifest_external.jsonl")
    summary_out = pb.get("summary_out_json", "outputs/proxy_build_summary_external.json")

    if args.force_reset_manifest:
        logging.info("Resetting manifest file")
        safe_unlink(Path(manifest_out))

    # Extraction parameters
    extraction = pb.get("extraction", {})
    extraction_params = {
        'min_tokens': int(extraction.get("min_tokens", 20)),
        'max_tokens': int(extraction.get("max_tokens", 120)),
        'max_periods': int(extraction.get("max_periods", 6))
    }
    logging.info(f"Extraction params: {extraction_params}")

    # Setup cache
    cache = None
    if args.use_cache:
        cache_file = f"{out_dir}/search_cache.pkl"
        cache = SearchCache(cache_file, max_age_hours=args.cache_max_age_hours)
        logging.info(f"Cache enabled (max age: {args.cache_max_age_hours}h)")

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
        rate_limit_threshold=int(gh_raw.get("rate_limit_threshold", 100))
    )

    gh: Optional[GitHubClient] = None
    if gh_cfg.enabled:
        token = os.getenv(gh_cfg.token_env, "").strip()
        if not token:
            raise RuntimeError(f"GitHub enabled but {gh_cfg.token_env} is not set")
        
        logging.info("Initializing GitHub client...")
        gh = GitHubClient(token, use_retry=True)
        
        if not gh.verify_token():
            raise RuntimeError("GitHub token verification failed")

    # ---- Kaggle cfg ----
    kg_raw = pb.get("kaggle", {})
    kg_enabled = bool(kg_raw.get("enabled", False))
    
    kaggle_client: Optional[KaggleClient] = None
    if kg_enabled and not args.dry_run:
        logging.info("Initializing Kaggle client...")
        kaggle_client = KaggleClient()

    # ---- Build queries ----
    if args.advanced_queries:
        logging.info("Building advanced queries...")
        queries = build_advanced_queries(df, cfg)
    else:
        logging.info("Building standard queries...")
        queries = build_queries(df, cfg)
    
    logging.info(f"Total queries: {len(queries)}")

    collected_lines: List[str] = []
    all_stats = {
        "github": {},
        "kaggle": {}
    }

    started_at = utc_now()
    append_jsonl(manifest_out, {
        "ts": utc_now(),
        "type": "run_start",
        "master_table": master_path,
        "n_master_rows": int(len(df)),
        "dry_run": bool(args.dry_run),
        "config_path": args.config,
        "use_cache": args.use_cache,
        "advanced_queries": args.advanced_queries
    })

    # -----------------------
    # GitHub collection
    # -----------------------
    if gh is not None:
        logging.info("\n" + "="*60)
        logging.info("Starting GitHub collection...")
        logging.info("="*60)
        
        gh_lines, gh_stats = collect_from_github(
            gh=gh,
            cfg=gh_cfg,
            queries=queries,
            extraction_params=extraction_params,
            manifest_out=manifest_out,
            cache=cache,
            dry_run=args.dry_run,
            test_ids=test_ids,
        )
        
        collected_lines.extend(gh_lines)
        all_stats["github"] = gh_stats
        
        logging.info(f"\nGitHub collection complete:")
        logging.info(f"  Files considered: {gh_stats['files_considered']}")
        logging.info(f"  Files downloaded: {gh_stats['files_downloaded']}")
        logging.info(f"  Lines extracted: {len(gh_lines)}")
        logging.info(f"  Failures: {gh_stats['failures']}")
        if cache:
            logging.info(f"  Cache hits: {gh_stats['cache_hits']}")
            logging.info(f"  Cache misses: {gh_stats['cache_misses']}")

    # -----------------------
    # Kaggle collection
    # -----------------------
    if kg_enabled:
        logging.info("\n" + "="*60)
        logging.info("Starting Kaggle collection...")
        logging.info("="*60)
        
        if kaggle_client is None and not args.dry_run:
            kaggle_client = KaggleClient()
        
        if kaggle_client or args.dry_run:
            kg_lines, kg_stats = collect_from_kaggle(
                kaggle_client=kaggle_client,
                cfg=kg_raw,
                extraction_params=extraction_params,
                manifest_out=manifest_out,
                dry_run=args.dry_run
            )
            
            collected_lines.extend(kg_lines)
            all_stats["kaggle"] = kg_stats
            
            logging.info(f"\nKaggle collection complete:")
            logging.info(f"  Datasets considered: {kg_stats['datasets_considered']}")
            logging.info(f"  Files downloaded: {kg_stats['files_downloaded']}")
            logging.info(f"  Lines extracted: {len(kg_lines)}")
            logging.info(f"  Failures: {kg_stats['failures']}")

    # -----------------------
    # Finalize: dedup & write proxy txt
    # -----------------------
    logging.info("\n" + "="*60)
    logging.info("Finalizing corpus...")
    logging.info("="*60)
    
    logging.info(f"Total lines collected (before dedup): {len(collected_lines)}")
    
    uniq = sorted(set(collected_lines))
    logging.info(f"Unique lines: {len(uniq)}")
    logging.info(f"Duplicates removed: {len(collected_lines) - len(uniq)}")
    
    if args.dry_run:
        logging.info("dry_run=True: skipping writing proxy_out_txt")
    else:
        Path(proxy_out_txt).parent.mkdir(parents=True, exist_ok=True)
        with open(proxy_out_txt, "w", encoding="utf-8") as f:
            for ln in uniq:
                f.write(ln + "\n")
        logging.info(f"Written to: {proxy_out_txt}")


    # Summary
    summary = {
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "master_table": master_path,
        "n_master_rows": int(len(df)),
        "n_queries": len(queries),
        "dry_run": bool(args.dry_run),
        "use_cache": args.use_cache,
        "advanced_queries": args.advanced_queries,
        "proxy_out_txt": proxy_out_txt,
        "manifest_out_jsonl": manifest_out,
        "lines_collected_raw": len(collected_lines),
        "lines_unique": len(uniq),
        "dedup_rate": f"{(1 - len(uniq)/len(collected_lines))*100:.2f}%" if collected_lines else "0%",
        "stats": all_stats,
        "master_rows_total": int(before),
        "rows_used_after_split_filter": int(after),
        "split_filter_value": "test" if "split" in df.columns else None,
    }
    
    write_json(summary_out, summary)
    append_jsonl(manifest_out, {"ts": utc_now(), "type": "run_end", **summary})
    
    logging.info("\n" + "="*60)
    logging.info("Summary:")
    logging.info("="*60)
    logging.info(json.dumps(summary, ensure_ascii=False, indent=2))
    logging.info("="*60)
    logging.info(f"Summary written to: {summary_out}")
    logging.info("Done!")


if __name__ == "__main__":
    main()
