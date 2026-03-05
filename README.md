# Contamination Detection Pipeline

This repository contains an API-based pipeline for estimating potential training-data contamination in large language models (LLMs) using the XSum benchmark.

The pipeline combines multiple evidence signals and produces an integrated risk score per item and per model.

## What the project does

For each XSum example, the project computes four contamination-related signals:

- `SLex` (Lexical exposure): overlap against an external proxy corpus (GitHub/Kaggle mirrors and related dumps).
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

1. Proxy corpus build (optional refresh)
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
   - `scripts/run_risk_and_visualize.py`

## Requirements

Python 3.10+ is recommended.

Install dependencies used by scripts (minimum set inferred from the codebase):

```bash
pip install pandas pyarrow pyyaml matplotlib openai google-genai requests tqdm
```

## Environment variables

Set API keys for the providers you want to run:

```bash
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."
export DEEPSEEK_API_KEY="..."   # optional, only for DeepSeek utilities
export GITHUB_TOKEN="..."        # optional, for proxy builder GitHub search
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
python3 scripts/run_stability_detector.py --config configs/run_config.yaml --model_id gpt4omini

# 4) Integrated risk
python3 scripts/run_risk_integration.py --config configs/run_config.yaml --model_id gpt4omini
python3 scripts/run_risk_and_visualize.py --config configs/run_config.yaml --model_id gpt4omini
```

Use `--limit N` on detector scripts for pilot runs.

## Inputs and outputs

Primary dataset input is configured in `configs/run_config.yaml`, currently pointing to:

- `data/master_table_xsum_n300_seed42_v2_dcq_frozen_FINAL.parquet`

Typical outputs:

- Stage runs: `runs/v3_*.parquet` ... `runs/v7_*.parquet`
- Logs: `logs/*.jsonl`
- Summaries/figures: `outputs/*.json`, `outputs/*.png`

## Reproducibility notes

- Prompt templates are stored in code (`src/prompts.py`) and intended to remain fixed across model comparisons.
- The pipeline is resume-friendly: existing stage columns are reused and completed incrementally.
- Keep the frozen master table and proxy artifacts versioned/frozen when producing comparable reports.

