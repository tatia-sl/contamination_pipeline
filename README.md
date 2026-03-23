# Contamination Detection Pipeline

This repository contains an API-based pipeline for estimating potential training-data contamination in large language models (LLMs) using the XSum benchmark.

The pipeline combines multiple evidence signals and produces an integrated risk score per item and per model.

## What the project does

For each XSum example, the project computes four contamination-related signals:

- `SLex` (Lexical exposure): overlap against an external proxy corpus built from GitHub and Kaggle sources via their APIs (plus related public mirrors/dumps).
- `SSem` (Semantic preference): DCQ (Discriminative Choice Question) selection among canonical vs paraphrased summaries.
- `SMem` (Memorization): prefix-completion probing against the frozen reference summary.
- `SProb` (Stability/Probability): instability under repeated stochastic one-sentence generation.

These signals are then merged into a final per-item risk output (`RiskScore`, `RiskLevel`, `Confidence`).

## Repository layout

- `configs/run_config.yaml`: main run configuration (dataset paths, models, decoding, output templates).
- `src/prompts.py`: fixed prompt templates used by detector stages.
- `src/clients/`: API clients for OpenAI, Gemini, and DeepSeek.
- `scripts/`: executable pipeline stages and utilities.
- `data/`: frozen dataset tables and proxy artifacts.
- `runs/`: stage outputs (`.parquet` / `.csv`).
- `logs/`: per-stage JSONL logs.
- `outputs/`: summaries and visualizations.

## Main pipeline stages

1. Proxy corpus build (optional refresh; collected via GitHub API and Kaggle API)
   - `scripts/run_proxy_builder_improved.py`
2. Lexical detector (`SLex`)
   - `scripts/run_lexical_detector.py`
3. DCQ semantic detector (`SSem`)
   - `scripts/run_dcq_detector.py`
4. Memorization probe (`SMem`)
   - `scripts/run_mem_probe.py`
5. Stability detector (`SProb`)
   - `scripts/run_stability_detector.py`
6. Risk integration and reporting
   - `scripts/run_risk_integration.py`
   - `scripts/build_management_report.py`

## Current risk logic (implemented)

`scripts/run_risk_integration.py` currently computes:

- normalization: `SLex_n=SLex/3`, `SSem_n=SSem/3`, `SMem_n=SMem/3`, `SProb_n=SProb/3`
- base score:
  - `RiskScore_raw = 100 * (0.35*SLex_n + 0.20*SSem_n + 0.30*SMem_n + 0.15*SProb_n)`
- overrides:
  - if `SLex=3` or `SMem=3` => enforce `RiskScore >= 50`
  - if `SProb>=2` and max(`SLex`,`SSem`,`SMem`)<=1 => cap `RiskScore <= 49`
- levels:
  - `Low` if `<25`
  - `Medium` if `25..49.999`
  - `High` if `50..74.999`
  - `Critical` if `>=75`
- confidence:
  - `High` if at least 2 strong signals (`>=2`) and (`SLex>=2` or `SMem>=2`)
  - `Medium` if at least 2 weak signals (`>=1`)
  - else `Low`

## Requirements

Python 3.10+ is recommended.

Install dependencies used by scripts (minimum set inferred from the codebase):

```bash
pip install pandas pyarrow pyyaml matplotlib numpy openai google-genai requests tqdm
```

## Environment variables

Set API keys for the providers you want to run:

```bash
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."
export OPENROUTER_API_KEY="..." # optional, only if a model in config uses provider=openrouter
export DEEPSEEK_API_KEY="..."   # optional, only for DeepSeek utilities
export GITHUB_TOKEN="..."        # optional, for proxy builder GitHub search
export KAGGLE_USERNAME="..."     # optional, for proxy builder Kaggle API access
export KAGGLE_KEY="..."          # optional, for proxy builder Kaggle API access
```

## Quick start

Run from repository root:

```bash
# 1) (Optional) rebuild external proxy corpus
python3 scripts/run_proxy_builder_improved.py --config configs/run_config.yaml

# 2) Lexical signal (model-agnostic)
python3 scripts/run_lexical_detector.py --config configs/run_config.yaml

# 3) Model-dependent signals
python3 scripts/run_dcq_detector.py --config configs/run_config.yaml --model_id gpt4omini
python3 scripts/run_mem_probe.py --config configs/run_config.yaml --model_id gpt4omini
# 3a) Build control set once (required for stability control baseline)
python3 scripts/build_control_set_stability.py --output data/control_set_cnn_n296_seed42.parquet --n 296 --seed 42
python3 scripts/run_stability_detector.py --config configs/run_config.yaml --model_id gpt4omini

# 4) Integrated risk
python3 scripts/run_risk_integration.py --config configs/run_config.yaml --model_id gpt4omini
# 5) Build management report JSON for assessment page
python3 scripts/build_management_report.py --data-dir outputs --out assessment/data/management_report.json
```

Use `--limit N` on detector scripts for pilot runs.

## Single-command reproducible proxy build

Build/update the structured proxy corpus with one command:

```bash
python3 scripts/build_proxy_structured_merged.py --config configs/run_config.yaml
```

Useful variants:

```bash
# Merge only (reuse existing proxy_structured_kaggle.csv + proxy_structured_github.csv)
python3 scripts/build_proxy_structured_merged.py \
  --skip_collect \
  --skip_extract_kaggle \
  --skip_reparse_github

# Rebuild and deduplicate by normalized summary text
python3 scripts/build_proxy_structured_merged.py \
  --config configs/run_config.yaml \
  --dedupe_mode summary_norm
```

Default outputs:

- `data/proxies/proxy_structured_kaggle.csv`
- `data/proxies/proxy_structured_github.csv`
- `data/proxies/proxy_structured_merged.csv`
- `outputs/proxy_structured_merged_build_summary.json`

## Inputs and outputs

Primary dataset input is configured in `configs/run_config.yaml`, currently pointing to:

- `master_table_xsum_n300_seed42_v3_dcq4_frozen_FINAL.parquet`

Typical outputs:

- Stage runs: `runs/v3_*.parquet` ... `runs/v7_*.parquet`
- Logs: `logs/*.jsonl`
- Summaries/figures: `outputs/*.json`, `outputs/*.png`

Primary risk integration outputs:

- `runs/v7_risk_{model_id}.parquet`
- `runs/v7_risk_{model_id}.csv`
- `outputs/v7_risk_summary_{model_id}.json`
- `logs/v7_risk_{model_id}.jsonl`

Management/reporting outputs:

- `assessment/data/management_report.json` (built by `scripts/build_management_report.py`)
- `assessment/index.html` (reads management report JSON and updates dashboard values client-side)

## Current run status in repository (2026-03-23)

Completed stage files exist for all 3 reporting models:

- `gpt4omini`: `runs/v4_dcq_gpt4omini.parquet`, `runs/v5_mem_gpt4omini.parquet`, `runs/v6_stability_gpt4omini.parquet`, `runs/v7_risk_gpt4omini.parquet`
- `gpt35turbo`: `runs/v4_dcq_gpt35turbo.parquet`, `runs/v5_mem_gpt35turbo.parquet`, `runs/v6_stability_gpt35turbo.parquet`, `runs/v7_risk_gpt35turbo.parquet`
- `gemini15flash`: `runs/v4_dcq_gemini15flash.parquet`, `runs/v5_mem_gemini15flash.parquet`, `runs/v6_stability_gemini15flash.parquet`, `runs/v7_risk_gemini15flash.parquet`

From latest risk summaries:

- `outputs/v7_risk_summary_gpt4omini.json`
  - `risk_level_counts`: `{High: 244, Medium: 44, Low: 8}`
  - `risk_score_mean`: `46.3457`
  - `confidence_counts`: `{Medium: 282, High: 7, Low: 7}`
- `outputs/v7_risk_summary_gpt35turbo.json`
  - `risk_level_counts`: `{High: 243, Medium: 10, Low: 43}`
  - `risk_score_mean`: `45.1633`
  - `confidence_counts`: `{Low: 249, Medium: 34, High: 13}`
- `outputs/v7_risk_summary_gemini15flash.json`
  - `risk_level_counts`: `{High: 248, Medium: 41, Low: 6, Critical: 1}`
  - `risk_score_mean`: `47.6971`
  - `confidence_counts`: `{Medium: 260, High: 34, Low: 2}`

## Latest lexical run on current external proxy corpus

Run tag: `structured_merged`  
Proxy corpus: `data/proxies/proxy_structured_merged.csv` (`proxy_column=summary_ref`)  
Summary file: `outputs/v3_lexical_summary_structured_merged.json`

- `n_rows_total`: 296
- `processed_new`: 296
- `failures`: 0
- `SLex_counts`: `{3: 241, 2: 46, 0: 9}`
- `MaxSpanLen_mean`: 122.5642
- `NgramHits_mean`: 9.7061
- `ProxyCount_mean`: 0.9865
- `elapsed_seconds`: 1.2512

## Reproducibility notes

- Prompt templates are stored in code (`src/prompts.py`) and intended to remain fixed across model comparisons.
- The pipeline is resume-friendly: existing stage columns are reused and completed incrementally.
- Keep the frozen master table and proxy artifacts versioned/frozen when producing comparable reports.
