#!/usr/bin/env python3
"""
reparse_github_from_manifest.py

Re-download and parse GitHub files from manifest to extract structured data.

The manifest contains URLs and metadata for files that were downloaded but
only extracted as "summary-like lines", not as structured document-summary pairs.

This script:
1. Reads the manifest
2. Finds github_download_ok events
3. Re-downloads files from raw_url
4. Parses them properly for structured data (JSON/JSONL/CSV)
5. Outputs structured CSV

Usage:
    python reparse_github_from_manifest.py \
        --manifest proxy_sources_manifest_external.jsonl \
        --output github_structured.csv
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import requests
import pandas as pd


def parse_json_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse single JSON line"""
    try:
        return json.loads(line)
    except:
        line = re.sub(r',\s*}', '}', line)
        line = re.sub(r',\s*]', ']', line)
        try:
            return json.loads(line)
        except:
            return None


def extract_from_json_object(obj: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Extract document and summary from JSON object"""
    doc_fields = ['document', 'text', 'article', 'src', 'source_text', 'input', 'content']
    sum_fields = ['summary', 'sum_sents', 'tgt', 'target', 'highlights', 'abstract', 'reference', 'output']
    id_fields = ['id', 'bbcid', 'xsum_id', 'doc_id', 'article_id']
    split_fields = ['split', 'set', 'subset']
    
    result = {}
    
    for field in doc_fields:
        if field in obj and obj[field]:
            val = obj[field]
            if isinstance(val, list):
                val = " ".join(str(x) for x in val)
            elif isinstance(val, dict):
                continue
            result['document'] = str(val).strip()
            break
    
    for field in sum_fields:
        if field in obj and obj[field]:
            val = obj[field]
            if isinstance(val, list):
                val = " ".join(str(x) for x in val)
            elif isinstance(val, dict):
                continue
            result['summary'] = str(val).strip()
            break
    
    for field in id_fields:
        if field in obj and obj[field]:
            val = obj[field]
            if isinstance(val, (list, dict)):
                continue
            result['id'] = str(val).strip()
            break
    
    for field in split_fields:
        if field in obj and obj[field]:
            val = obj[field]
            if isinstance(val, (list, dict)):
                continue
            result['split'] = str(val).strip().lower()
            break
    
    if 'document' in result and 'summary' in result:
        return result
    
    return None


def parse_jsonl_content(content: bytes, source_info: str) -> List[Dict[str, str]]:
    """Parse JSONL content"""
    records = []
    
    try:
        text = content.decode('utf-8', errors='ignore')
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            
            obj = parse_json_line(line)
            if obj:
                extracted = extract_from_json_object(obj)
                if extracted:
                    extracted['source_info'] = source_info
                    records.append(extracted)
    except Exception as e:
        pass
    
    return records


def parse_json_content(content: bytes, source_info: str) -> List[Dict[str, str]]:
    """Parse JSON content"""
    records = []
    
    try:
        data = json.loads(content.decode('utf-8', errors='ignore'))
        
        if isinstance(data, list):
            for obj in data:
                if isinstance(obj, dict):
                    extracted = extract_from_json_object(obj)
                    if extracted:
                        extracted['source_info'] = source_info
                        records.append(extracted)
        
        elif isinstance(data, dict):
            extracted = extract_from_json_object(data)
            if extracted:
                extracted['source_info'] = source_info
                records.append(extracted)
    
    except Exception as e:
        pass
    
    return records


def parse_csv_content(content: bytes, filename: str, source_info: str) -> List[Dict[str, str]]:
    """Parse CSV/TSV content"""
    records = []
    
    sep = '\t' if filename.lower().endswith('.tsv') else ','
    
    try:
        import io
        df = pd.read_csv(
            io.BytesIO(content),
            sep=sep,
            dtype=str,
            on_bad_lines='skip',
            engine='python'
        )
        
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
                
                if record['document'] and record['summary']:
                    records.append(record)
    
    except Exception as e:
        pass
    
    return records


def download_file(url: str, timeout: int = 30) -> Optional[bytes]:
    """Download file from URL"""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            return response.content
    except:
        pass
    
    return None


def reparse_github_files(
    manifest_path: str,
    github_token: Optional[str] = None,
    rate_limit_delay: float = 2.0,
    verbose: bool = True
) -> List[Dict[str, str]]:
    """
    Re-parse GitHub files from manifest
    
    Args:
        manifest_path: Path to manifest JSONL
        github_token: Optional GitHub token for higher rate limits
        rate_limit_delay: Delay between requests (seconds)
        verbose: Print progress
        
    Returns:
        List of extracted records
    """
    
    if not Path(manifest_path).exists():
        print(f"❌ Manifest not found: {manifest_path}")
        return []
    
    # Read manifest
    with open(manifest_path, 'r') as f:
        events = [json.loads(line) for line in f if line.strip()]
    
    # Filter download_ok events
    downloads = [e for e in events if e.get('type') == 'github_download_ok']
    
    if verbose:
        print(f"\n📋 Анализ манифеста")
        print(f"{'='*70}")
        print(f"Всего событий: {len(events):,}")
        print(f"Успешных загрузок GitHub: {len(downloads):,}")
    
    # Group by sha256 to avoid duplicates
    unique_files = {}
    for event in downloads:
        sha = event.get('sha256')
        if sha and sha not in unique_files:
            unique_files[sha] = event
    
    if verbose:
        print(f"Уникальных файлов: {len(unique_files):,}")
    
    # Setup headers
    headers = {}
    if github_token:
        headers['Authorization'] = f'Bearer {github_token}'
    
    all_records = []
    
    if verbose:
        print(f"\n📥 Повторная загрузка и парсинг файлов...")
        print(f"{'='*70}")
    
    for i, (sha, event) in enumerate(unique_files.items(), 1):
        repo = event.get('repo', 'unknown')
        path = event.get('path', 'unknown')
        filename = event.get('name', 'unknown')
        raw_url = event.get('raw_url')
        size = event.get('bytes', 0)
        
        if not raw_url:
            continue
        
        # Determine file type
        ext = filename.split('.')[-1].lower() if '.' in filename else ''
        
        # Only process structured files
        if ext not in ['json', 'jsonl', 'csv', 'tsv']:
            if verbose and i % 10 == 0:
                print(f"  [{i}/{len(unique_files)}] Пропущено (тип: {ext}): {repo}/{path}")
            continue
        
        if verbose:
            print(f"  [{i}/{len(unique_files)}] {repo}/{path} ({size/1024:.1f} KB)")
        
        # Download
        content = download_file(raw_url, timeout=30)
        
        if not content:
            if verbose:
                print(f"    ⚠ Не удалось загрузить")
            time.sleep(rate_limit_delay)
            continue
        
        # Verify hash
        downloaded_sha = hashlib.sha256(content).hexdigest()
        if downloaded_sha != sha:
            if verbose:
                print(f"    ⚠ Hash mismatch!")
            time.sleep(rate_limit_delay)
            continue
        
        # Parse based on extension
        source_info = f"github:{repo}:{path}"
        records = []
        
        if ext == 'jsonl':
            records = parse_jsonl_content(content, source_info)
        elif ext == 'json':
            records = parse_json_content(content, source_info)
        elif ext in ['csv', 'tsv']:
            records = parse_csv_content(content, filename, source_info)
        
        if records:
            all_records.extend(records)
            if verbose:
                print(f"    ✓ Извлечено {len(records)} записей")
        else:
            if verbose:
                print(f"    ⚠ Нет структурированных данных")
        
        time.sleep(rate_limit_delay)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"📊 Итого извлечено записей: {len(all_records):,}")
    
    return all_records


def main():
    parser = argparse.ArgumentParser(
        description="Re-parse GitHub files from manifest to extract structured data",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--manifest',
        type=str,
        required=True,
        help='Path to proxy_sources_manifest_external.jsonl'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output CSV file'
    )
    
    parser.add_argument(
        '--github-token',
        type=str,
        help='GitHub token (or use GITHUB_TOKEN env var)'
    )
    
    parser.add_argument(
        '--rate-limit-delay',
        type=float,
        default=2.0,
        help='Delay between requests in seconds (default: 2.0)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress output'
    )
    
    args = parser.parse_args()
    
    # Get GitHub token
    github_token = args.github_token or os.getenv('GITHUB_TOKEN')
    
    verbose = not args.quiet
    
    # Re-parse files
    records = reparse_github_files(
        manifest_path=args.manifest,
        github_token=github_token,
        rate_limit_delay=args.rate_limit_delay,
        verbose=verbose
    )
    
    if not records:
        print("\n⚠ No structured records extracted!")
        return 1
    
    # Deduplicate
    if verbose:
        print(f"\n🔄 Дедупликация...")
    
    seen = set()
    unique_records = []
    
    for record in records:
        content = record['document'] + '||' + record['summary']
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        
        if content_hash not in seen:
            seen.add(content_hash)
            unique_records.append(record)
    
    if verbose:
        removed = len(records) - len(unique_records)
        print(f"  Оригинал: {len(records):,}")
        print(f"  Уникальных: {len(unique_records):,}")
        print(f"  Удалено дубликатов: {removed:,}")
    
    # Create output CSV
    if verbose:
        print(f"\n💾 Создание CSV...")
    
    final_records = []
    
    for i, record in enumerate(unique_records):
        # Parse source info
        source_info = record.get('source_info', '')
        parts = source_info.split(':', 2)
        source_type = parts[0] if len(parts) >= 1 else 'github'
        source_detail = ':'.join(parts[1:]) if len(parts) >= 2 else ''
        
        final_records.append({
            'item_id': f'github_{i+1:06d}',
            'xsum_id': record.get('id', ''),
            'split': record.get('split', ''),
            'source': source_type,
            'source_detail': source_detail,
            'document': record['document'],
            'summary_ref': record['summary']
        })
    
    # Write CSV
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = ['item_id', 'xsum_id', 'split', 'source', 'source_detail', 'document', 'summary_ref']
    
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_records)
    
    if verbose:
        print(f"  ✓ Сохранено в: {args.output}")
        print(f"  Записей: {len(final_records):,}")
        
        with_xsum = sum(1 for r in final_records if r['xsum_id'])
        with_split = sum(1 for r in final_records if r['split'])
        
        print(f"\n📈 Статистика:")
        print(f"  С XSum ID: {with_xsum:,} ({with_xsum/len(final_records)*100:.1f}%)")
        print(f"  Со split: {with_split:,} ({with_split/len(final_records)*100:.1f}%)")
        
        print(f"\n✅ Готово!")
    
    return 0


if __name__ == "__main__":
    exit(main())
