#!/usr/bin/env python3
"""
extract_structured_proxy_data.py

Extract structured proxy data from:
1. GitHub manifest (JSONL with download metadata)
2. Kaggle downloaded files (CSV, JSON, JSONL)

Output: Clean CSV with columns:
- item_id: unique identifier
- xsum_id: XSum document ID (if matched)
- split: train/val/test (if available)
- document: source text
- summary_ref: reference summary

Usage:
    python extract_structured_proxy_data.py \
        --manifest proxy_sources_manifest_external.jsonl \
        --kaggle-dir data/proxies/kaggle_tmp \
        --output proxy_structured_kaggle.csv
"""

import argparse
import csv
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import requests


def extract_xsum_id(text: str) -> Optional[str]:
    """
    Extract XSum ID from text.
    Patterns: "xsum_id", "id:", document IDs like "11223344"
    """
    # Pattern 1: Explicit xsum_id field
    match = re.search(r'["\']?xsum_id["\']?\s*[:=]\s*["\']?(\d+)["\']?', text, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Pattern 2: Generic id field with 8 digits
    match = re.search(r'["\']?id["\']?\s*[:=]\s*["\']?(\d{8})["\']?', text)
    if match:
        return match.group(1)
    
    # Pattern 3: Standalone 8-digit ID
    match = re.search(r'\b(\d{8})\b', text)
    if match:
        return match.group(1)
    
    return None


def detect_split(text: str, xsum_id: Optional[str] = None) -> Optional[str]:
    """Detect train/val/test split from text or ID"""
    text_lower = text.lower()
    
    # Explicit split field
    if 'split' in text_lower:
        if 'test' in text_lower:
            return 'test'
        if 'train' in text_lower:
            return 'train'
        if 'val' in text_lower or 'validation' in text_lower:
            return 'validation'
    
    # Based on XSum ID (if we have the mapping)
    # This would require loading the master table
    
    return None


def parse_json_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse single JSON line, handling various formats"""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        # Try to fix common issues
        # Remove trailing commas
        line = re.sub(r',\s*}', '}', line)
        line = re.sub(r',\s*]', ']', line)
        try:
            return json.loads(line)
        except:
            return None


def extract_from_json_object(obj: Dict[str, Any], source_info: Optional[str] = None) -> Optional[Dict[str, str]]:
    """
    Extract document and summary from JSON object.
    
    Common patterns:
    - {"document": "...", "summary": "..."}
    - {"text": "...", "summary": "..."}
    - {"src": "...", "tgt": "..."}
    - {"article": "...", "highlights": "..."}
    """
    
    # Field name mappings
    doc_fields = ['document', 'text', 'article', 'src', 'source_text', 'input', 'content']
    sum_fields = ['summary', 'sum_sents', 'tgt', 'target', 'highlights', 'abstract', 'reference', 'output']
    id_fields = ['id', 'xsum_id', 'doc_id', 'article_id', 'item_id', 'bbcid']
    split_fields = ['split', 'set', 'subset']
    
    result = {}
    
    # Extract document
    for field in doc_fields:
        if field in obj and obj[field]:
            val = obj[field]
            if isinstance(val, list):
                val = " ".join(str(x) for x in val)
            elif isinstance(val, dict):
                continue
            result['document'] = str(val).strip()
            break
    
    # Extract summary
    for field in sum_fields:
        if field in obj and obj[field]:
            val = obj[field]
            if isinstance(val, list):
                val = " ".join(str(x) for x in val)
            elif isinstance(val, dict):
                continue
            result['summary'] = str(val).strip()
            break
    
    # Extract ID
    for field in id_fields:
        if field in obj and obj[field]:
            val = obj[field]
            if isinstance(val, (list, dict)):
                continue
            result['id'] = str(val).strip()
            break
    
    # Extract split
    for field in split_fields:
        if field in obj and obj[field]:
            result['split'] = str(obj[field]).strip().lower()
            break
    
    # Add source info
    if source_info:
        result['source_info'] = source_info
    
    # Only return if we have both document and summary
    if 'document' in result and 'summary' in result:
        return result
    
    return None


def process_json_file(filepath: Path, source_info: str) -> List[Dict[str, str]]:
    """Process JSON file (single object or array)"""
    records = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle array of objects
        if isinstance(data, list):
            for obj in data:
                if isinstance(obj, dict):
                    extracted = extract_from_json_object(obj, source_info)
                    if extracted:
                        records.append(extracted)
        
        # Handle single object
        elif isinstance(data, dict):
            extracted = extract_from_json_object(data, source_info)
            if extracted:
                records.append(extracted)
    
    except Exception as e:
        print(f"  ⚠ Error processing {filepath.name}: {e}")
    
    return records


def process_jsonl_file(filepath: Path, source_info: str) -> List[Dict[str, str]]:
    """Process JSONL file (one JSON object per line)"""
    records = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                obj = parse_json_line(line)
                if obj:
                    extracted = extract_from_json_object(obj, source_info)
                    if extracted:
                        records.append(extracted)
    
    except Exception as e:
        print(f"  ⚠ Error processing {filepath.name}: {e}")
    
    return records


def process_csv_file(filepath: Path, source_info: str) -> List[Dict[str, str]]:
    """Process CSV/TSV file"""
    records = []
    
    # Detect delimiter
    sep = '\t' if filepath.suffix.lower() == '.tsv' else ','
    
    try:
        df = pd.read_csv(filepath, sep=sep, dtype=str, on_bad_lines='skip', engine='python')
        
        # Find relevant columns
        doc_cols = [c for c in df.columns if any(x in str(c).lower() for x in ['document', 'text', 'article', 'src', 'source_text', 'input'])]
        sum_cols = [c for c in df.columns if any(x in str(c).lower() for x in ['summary', 'tgt', 'target', 'highlights', 'abstract', 'reference'])]
        id_cols = [c for c in df.columns if any(x in str(c).lower() for x in ['id', 'xsum'])]
        split_cols = [c for c in df.columns if any(x in str(c).lower() for x in ['split', 'set'])]
        
        doc_col = doc_cols[0] if doc_cols else None
        sum_col = sum_cols[0] if sum_cols else None
        id_col = id_cols[0] if id_cols else None
        split_col = split_cols[0] if split_cols else None
        
        if doc_col and sum_col:
            for _, row in df.iterrows():
                record = {
                    'document': str(row[doc_col]).strip() if pd.notna(row[doc_col]) else '',
                    'summary': str(row[sum_col]).strip() if pd.notna(row[sum_col]) else '',
                    'source_info': source_info
                }
                
                if id_col and pd.notna(row[id_col]):
                    record['id'] = str(row[id_col]).strip()
                
                if split_col and pd.notna(row[split_col]):
                    record['split'] = str(row[split_col]).strip().lower()
                
                # Only keep if both doc and summary are non-empty
                if record['document'] and record['summary']:
                    records.append(record)
    
    except Exception as e:
        print(f"  ⚠ Error processing {filepath.name}: {e}")
    
    return records


def process_file(filepath: Path, dataset_name: Optional[str] = None) -> List[Dict[str, str]]:
    """Process single file based on extension"""
    suffix = filepath.suffix.lower()
    
    # Create source info
    source_info = f"kaggle:{dataset_name}:{filepath.name}" if dataset_name else f"kaggle:{filepath.name}"
    
    if suffix == '.jsonl':
        return process_jsonl_file(filepath, source_info)
    elif suffix == '.json':
        return process_json_file(filepath, source_info)
    elif suffix in ['.csv', '.tsv']:
        return process_csv_file(filepath, source_info)
    else:
        return []


def parse_json_content_bytes(raw: bytes, source_info: str) -> List[Dict[str, str]]:
    """Parse JSON bytes and extract structured records."""
    records: List[Dict[str, str]] = []
    try:
        data = json.loads(raw.decode("utf-8", errors="ignore"))
    except Exception:
        return records

    if isinstance(data, list):
        for obj in data:
            if isinstance(obj, dict):
                extracted = extract_from_json_object(obj, source_info)
                if extracted:
                    records.append(extracted)
    elif isinstance(data, dict):
        extracted = extract_from_json_object(data, source_info)
        if extracted:
            records.append(extracted)

    return records


def parse_jsonl_content_bytes(raw: bytes, source_info: str) -> List[Dict[str, str]]:
    """Parse JSONL bytes and extract structured records."""
    records: List[Dict[str, str]] = []
    try:
        text = raw.decode("utf-8", errors="ignore")
    except Exception:
        return records

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = parse_json_line(line)
        if not obj:
            continue
        extracted = extract_from_json_object(obj, source_info)
        if extracted:
            records.append(extracted)
    return records


def parse_csv_content_bytes(raw: bytes, filename: str, source_info: str) -> List[Dict[str, str]]:
    """Parse CSV/TSV bytes and extract structured records."""
    records: List[Dict[str, str]] = []
    sep = "\t" if filename.lower().endswith(".tsv") else ","
    try:
        import io

        df = pd.read_csv(
            io.BytesIO(raw),
            sep=sep,
            dtype=str,
            on_bad_lines="skip",
            engine="python",
        )
    except Exception:
        return records

    doc_cols = [c for c in df.columns if any(x in str(c).lower() for x in ["document", "text", "article", "src", "source_text", "input", "content"])]
    sum_cols = [c for c in df.columns if any(x in str(c).lower() for x in ["summary", "sum_sents", "tgt", "target", "highlights", "abstract", "reference", "output"])]
    id_cols = [c for c in df.columns if any(x in str(c).lower() for x in ["id", "xsum"])]
    split_cols = [c for c in df.columns if any(x in str(c).lower() for x in ["split", "set", "subset"])]

    doc_col = doc_cols[0] if doc_cols else None
    sum_col = sum_cols[0] if sum_cols else None
    id_col = id_cols[0] if id_cols else None
    split_col = split_cols[0] if split_cols else None

    if not doc_col or not sum_col:
        return records

    for _, row in df.iterrows():
        record = {
            "document": str(row[doc_col]).strip() if pd.notna(row[doc_col]) else "",
            "summary": str(row[sum_col]).strip() if pd.notna(row[sum_col]) else "",
            "source_info": source_info,
        }
        if id_col and pd.notna(row[id_col]):
            record["id"] = str(row[id_col]).strip()
        if split_col and pd.notna(row[split_col]):
            record["split"] = str(row[split_col]).strip().lower()
        if record["document"] and record["summary"]:
            records.append(record)
    return records


def download_bytes(url: str, timeout: int = 30, headers: Optional[Dict[str, str]] = None) -> Optional[bytes]:
    try:
        r = requests.get(url, timeout=timeout, headers=headers or {})
        if r.status_code == 200:
            return r.content
    except Exception:
        return None
    return None


def extract_from_kaggle_dir(kaggle_dir: str, verbose: bool = True) -> List[Dict[str, str]]:
    """Extract all structured data from Kaggle downloads directory"""
    
    kaggle_path = Path(kaggle_dir)
    if not kaggle_path.exists():
        print(f"⚠ Kaggle directory not found: {kaggle_dir}")
        return []
    
    all_records = []
    
    # Supported extensions
    extensions = ['.json', '.jsonl', '.csv', '.tsv']
    
    if verbose:
        print(f"\n📁 Scanning Kaggle directory: {kaggle_dir}")
    
    # Try to detect dataset name from directory structure
    # Pattern: .../kaggle_tmp/dataset-name/file.ext or .../kaggle_tmp/file.ext
    for ext in extensions:
        files = list(kaggle_path.glob(f'*{ext}'))
        
        # Also check subdirectories
        files.extend(kaggle_path.glob(f'*/*{ext}'))
        
        if verbose and files:
            print(f"\n  {ext.upper()} files: {len(files)}")
        
        for filepath in files:
            if verbose:
                print(f"    Processing: {filepath.name}")
            
            # Detect dataset name from path
            dataset_name = None
            if filepath.parent != kaggle_path:
                dataset_name = filepath.parent.name
            
            records = process_file(filepath, dataset_name)
            
            if records:
                if verbose:
                    print(f"      ✓ Extracted {len(records)} records")
                all_records.extend(records)
            else:
                if verbose:
                    print(f"      ⚠ No valid records found")
    
    return all_records


def extract_from_manifest(
    manifest_path: str,
    github_token: Optional[str] = None,
    rate_limit_delay: float = 2.0,
    max_files: Optional[int] = None,
    metadata_only: bool = False,
    verbose: bool = True,
) -> List[Dict[str, str]]:
    """
    Extract data from GitHub manifest.
    
    Manifest contains:
    - github_download_ok: successful downloads with sha256
    - github_extract_ok: successful text extraction
    
    Note: This function tracks metadata. For actual content extraction,
    you would need to re-download or have cached the raw files.
    """
    
    if not Path(manifest_path).exists():
        print(f"⚠ Manifest not found: {manifest_path}")
        return []
    
    records: List[Dict[str, str]] = []
    github_files = {}  # sha -> event
    
    if verbose:
        print(f"\n📋 Processing GitHub manifest: {manifest_path}")
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                event = json.loads(line.strip())
                
                # Track successful downloads for content extraction
                if event.get('type') == 'github_download_ok':
                    repo = event.get('repo', 'unknown')
                    path = event.get('path', 'unknown')
                    sha256 = event.get('sha256', '')
                    
                    source_info = f"github:{repo}:{path}"
                    
                    github_files[sha256] = {
                        'source_info': source_info,
                        'repo': repo,
                        'path': path,
                        'raw_url': event.get('raw_url', ''),
                        'bytes': event.get('bytes', 0)
                    }
            
            except:
                continue
    
    if verbose:
        print(f"  ℹ Tracked {len(github_files)} GitHub downloads")

    if metadata_only:
        if verbose:
            print("  ℹ metadata_only=True, skipping download/parse")
        return records

    allowed_ext = {"json", "jsonl", "csv", "tsv"}
    headers = {"Authorization": f"Bearer {github_token}"} if github_token else {}

    items = list(github_files.values())
    if max_files is not None and max_files >= 0:
        items = items[:max_files]

    extracted = 0
    for i, event in enumerate(items, start=1):
        raw_url = event.get("raw_url", "")
        repo = event.get("repo", "unknown")
        path = event.get("path", "unknown")
        name = event.get("name", "unknown")
        sha_expected = event.get("sha256", "")

        if not raw_url:
            continue

        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if ext not in allowed_ext:
            continue

        if verbose:
            print(f"    [{i}/{len(items)}] GitHub parse: {repo}/{path}")

        raw = download_bytes(raw_url, timeout=30, headers=headers)
        if not raw:
            continue

        if sha_expected:
            sha_actual = hashlib.sha256(raw).hexdigest()
            if sha_actual != sha_expected:
                if verbose:
                    print("      ⚠ hash mismatch, skip")
                continue

        source_info = f"github:{repo}:{path}"
        if ext == "json":
            recs = parse_json_content_bytes(raw, source_info)
        elif ext == "jsonl":
            recs = parse_jsonl_content_bytes(raw, source_info)
        else:
            recs = parse_csv_content_bytes(raw, name, source_info)

        if recs:
            records.extend(recs)
            extracted += len(recs)
            if verbose:
                print(f"      ✓ extracted {len(recs)} records")

        if rate_limit_delay > 0:
            time.sleep(rate_limit_delay)

    if verbose:
        print(f"  ✓ Extracted structured GitHub records: {extracted:,}")

    return records


def deduplicate_records(records: List[Dict[str, str]], verbose: bool = True) -> List[Dict[str, str]]:
    """Remove duplicate records based on content hash"""
    
    seen_hashes = set()
    unique_records = []
    
    for record in records:
        # Create hash from document + summary
        content = record['document'] + '||' + record['summary']
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_records.append(record)
    
    if verbose:
        removed = len(records) - len(unique_records)
        print(f"\n🔄 Deduplication:")
        print(f"  Original: {len(records):,}")
        print(f"  Unique: {len(unique_records):,}")
        print(f"  Duplicates removed: {removed:,}")
    
    return unique_records


def create_structured_csv(
    records: List[Dict[str, str]],
    output_path: str,
    master_table_path: Optional[str] = None,
    verbose: bool = True
) -> None:
    """Create final structured CSV"""
    
    # Load master table if provided (for split mapping)
    xsum_id_to_split = {}
    if master_table_path and Path(master_table_path).exists():
        if verbose:
            print(f"\n📊 Loading master table: {master_table_path}")
        
        try:
            df_master = pd.read_parquet(master_table_path)
            if 'xsum_id' in df_master.columns and 'split' in df_master.columns:
                xsum_id_to_split = dict(zip(
                    df_master['xsum_id'].astype(str),
                    df_master['split'].astype(str)
                ))
                if verbose:
                    print(f"  ✓ Loaded {len(xsum_id_to_split):,} ID-split mappings")
        except Exception as e:
            if verbose:
                print(f"  ⚠ Error loading master table: {e}")
    
    # Prepare final records
    final_records = []
    
    for i, record in enumerate(records):
        # Generate unique item_id
        item_id = f"proxy_{i+1:06d}"
        
        # Extract or generate xsum_id
        xsum_id = record.get('id') or extract_xsum_id(record['document'] + record['summary'])
        
        # Determine split
        split = record.get('split')
        if not split and xsum_id and xsum_id in xsum_id_to_split:
            split = xsum_id_to_split[xsum_id]
        
        # Parse source info
        source_info = record.get('source_info', '')
        source_type = ''
        source_detail = ''
        
        if source_info:
            # Format: "kaggle:dataset-name:file.ext" or "github:owner/repo:path/to/file"
            parts = source_info.split(':', 2)
            if len(parts) >= 1:
                source_type = parts[0]  # 'kaggle' or 'github'
            if len(parts) >= 2:
                source_detail = ':'.join(parts[1:])  # rest of the info
        
        final_records.append({
            'item_id': item_id,
            'xsum_id': xsum_id or '',
            'split': split or '',
            'source': source_type or 'unknown',
            'source_detail': source_detail or '',
            'document': record['document'],
            'summary_ref': record['summary']
        })
    
    # Write CSV
    fieldnames = ['item_id', 'xsum_id', 'split', 'source', 'source_detail', 'document', 'summary_ref']
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_records)
    
    if verbose:
        print(f"\n💾 Output CSV created: {output_path}")
        print(f"  Records: {len(final_records):,}")
        print(f"  Columns: {', '.join(fieldnames)}")
        
        # Statistics
        with_xsum_id = sum(1 for r in final_records if r['xsum_id'])
        with_split = sum(1 for r in final_records if r['split'])
        
        # Source statistics
        from collections import Counter
        source_counts = Counter(r['source'] for r in final_records)
        
        print(f"\n📈 Statistics:")
        print(f"  With XSum ID: {with_xsum_id:,} ({with_xsum_id/len(final_records)*100:.1f}%)")
        print(f"  With split: {with_split:,} ({with_split/len(final_records)*100:.1f}%)")
        print(f"\n📊 Source distribution:")
        for source, count in source_counts.most_common():
            pct = count / len(final_records) * 100
            print(f"  {source:10s}: {count:>6,} ({pct:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Extract structured proxy data from GitHub manifest and Kaggle files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from Kaggle directory only
  python extract_structured_proxy_data.py \\
      --kaggle-dir data/proxies/kaggle_tmp \\
      --output proxy_structured_kaggle.csv
  
  # Include manifest and master table for split mapping
  python extract_structured_proxy_data.py \\
      --manifest data/proxies/proxy_sources_manifest_external.jsonl \\
      --kaggle-dir data/proxies/kaggle_tmp \\
      --master-table data/master_table.parquet \\
      --output proxy_structured_kaggle.csv
  
  # Silent mode
  python extract_structured_proxy_data.py \\
      --kaggle-dir data/proxies/kaggle_tmp \\
      --output proxy_structured_kaggle.csv \\
      --quiet
        """
    )
    
    parser.add_argument(
        '--manifest',
        type=str,
        help='Path to GitHub manifest JSONL file'
    )
    
    parser.add_argument(
        '--github-token',
        type=str,
        help='GitHub token (optional; or set GITHUB_TOKEN env var)'
    )

    parser.add_argument(
        '--github-rate-limit-delay',
        type=float,
        default=2.0,
        help='Delay between GitHub raw file requests (seconds)'
    )

    parser.add_argument(
        '--github-max-files',
        type=int,
        default=None,
        help='Optional cap on number of GitHub files parsed from manifest'
    )

    parser.add_argument(
        '--manifest-metadata-only',
        action='store_true',
        help='Only read GitHub metadata from manifest, do not download/parse content'
    )
    
    parser.add_argument(
        '--kaggle-dir',
        type=str,
        help='Path to Kaggle downloads directory'
    )
    
    parser.add_argument(
        '--master-table',
        type=str,
        help='Path to master table (for split mapping)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output CSV file path'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    if not args.manifest and not args.kaggle_dir:
        print("Error: Must provide at least one of --manifest or --kaggle-dir")
        return 1
    
    # Extract from all sources
    all_records = []
    
    if args.manifest:
        import os
        github_token = args.github_token or os.getenv("GITHUB_TOKEN")
        manifest_records = extract_from_manifest(
            args.manifest,
            github_token=github_token,
            rate_limit_delay=float(args.github_rate_limit_delay),
            max_files=args.github_max_files,
            metadata_only=bool(args.manifest_metadata_only),
            verbose=verbose,
        )
        all_records.extend(manifest_records)
    
    if args.kaggle_dir:
        kaggle_records = extract_from_kaggle_dir(args.kaggle_dir, verbose=verbose)
        all_records.extend(kaggle_records)
    
    if not all_records:
        print("\n⚠ No records extracted from any source!")
        return 1
    
    # Deduplicate
    unique_records = deduplicate_records(all_records, verbose=verbose)
    
    # Create output CSV
    create_structured_csv(
        unique_records,
        args.output,
        master_table_path=args.master_table,
        verbose=verbose
    )
    
    if verbose:
        print("\n✅ Done!")
    
    return 0


if __name__ == "__main__":
    exit(main())
