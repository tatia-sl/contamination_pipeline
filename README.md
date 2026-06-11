# Assessing Data Contamination Risk In Proprietary LLMs: A Reproducible Black-Box Detection Pipeline

This repository contains an API-based pipeline for estimating potential benchmark contamination in large language models. The active versioned experiments are the XSum summarization benchmark and an isolated BBC2025 benchmark run with 300 frozen items. Earlier CCSum, BBC2024, XL-Sum, and exploratory artefacts have been moved under `legacy/`.

The project is designed for black-box model evaluation: it does not require access to model weights, logits, training data, or internal activations. All model-dependent measurements are collected through provider APIs and are stored as reproducible artefacts.

## Overview

The pipeline computes four contamination-related detector signals and integrates their aggregate outputs into a model-level risk assessment:

- `SLex` — lexical exposure against an external proxy corpus built from GitHub and Kaggle sources.
- `SSem` — semantic familiarity measured through discriminative choice probing over canonical and paraphrased summaries.
- `SMem` — memorization measured through prefix-completion probing.
- `SProb` — output stability/concentration measured through repeated stochastic summarization.

Each detector produces item-level evidence where appropriate and an aggregate detector-level signal:

- `SLex_aggregate`
- `SSem_aggregate`
- `SMem_aggregate`
- `SProb_aggregate`

The final `scripts/run_risk_integration.py` stage consumes these aggregate detector outputs from summary JSON files and produces a model-level composite contamination risk score (`CRS`), qualitative risk level, and confidence estimate.

## Repository Layout

- `configs/run_config.yaml` — active XSum configuration file.
- `configs/run_config_bbc2025.yaml` — active BBC2025 configuration file with isolated output directories.
- `src/prompts.py` — fixed prompt templates used by detector stages.
- `src/clients/` — API clients for OpenAI-compatible providers, Gemini, and DeepSeek-related generation utilities.
- `scripts/` — executable pipeline stages.
- `data/` — XSum proxy corpus files and related source artefacts.
- `proxy_corpus_bbc2025/` — BBC2025 proxy corpus files.
- `runs/`, `outputs/`, `logs/` — XSum row-level parquet outputs, aggregate summaries, and JSONL execution logs.
- `runs_bbc2025/`, `outputs_bbc2025/`, `logs_bbc2025/` — isolated BBC2025 run artefacts.
- `legacy/` — archived scripts and old run artefacts.

## Core Experimental Artefacts

The central XSum reproducibility substrate is:

```text
master_table_xsum_n300_seed42_v4_dcq4_frozen.parquet
```

This frozen master table contains the fixed XSum evaluation sample and all detector-specific task materials:

- `xsum_id`
- `split`
- `document`
- `summary_ref`
- `document_norm`
- `summary_ref_norm`
- `prefix_ref`
- `control_prefix`
- `dcq_A_canonical`
- `dcq_B_para1`
- `dcq_C_para2`
- `dcq_D_para3`
- `dcq_E_para4`

The table currently contains `296` evaluation items. It is frozen before model evaluation so that all detectors and models operate on the same benchmark substrate.

The BBC2025 reproducibility substrate is:

```text
master_table_bbc2025_n300_seed42_v3_dcq4_frozen.parquet
```

This table contains `300` frozen BBC News 2025 evaluation items and the same detector-specific columns used by the XSum pipeline.

## Active Pipeline Stages

1. Proxy corpus construction
   - `scripts/run_proxy_builder.py`
   - `scripts/build_proxy_structured_merged.py`

2. Lexical detector (`SLex`)
   - `scripts/run_lexical_detector.py`

3. Semantic detector (`SSem`)
   - `scripts/run_dcq_detector.py`

4. Memorization detector (`SMem`)
   - `scripts/run_mem_probe.py`

5. Stability detector (`SProb`)
   - `scripts/run_stability_detector.py`

6. Risk integration
   - `scripts/run_risk_integration.py`

Reporting and publication helpers are downstream of the detector contract:

- `scripts/build_report_csv.py`
- `scripts/build_pages_artifacts_manifest.py`

## Proxy Corpus Construction

The proxy corpus is built in two explicit steps:

```bash
python3 scripts/run_proxy_builder.py --config configs/run_config.yaml
python3 scripts/build_proxy_structured_merged.py --config configs/run_config.yaml
```

The first step collects and structures candidate proxy records from GitHub and Kaggle. The second step merges the per-source structured files and deduplicates records by normalized summary text.

Primary proxy outputs:

- `data/proxies/proxy_structured_github.csv`
- `data/proxies/proxy_structured_kaggle.csv`
- `data/proxies/proxy_structured_merged.csv`
- `data/proxies/proxy_sources_manifest_external_2026_02_06.jsonl`
- `outputs/proxy_build_summary_external.json`
- `outputs/proxy_structured_merged_build_summary.json`

Compressed GitHub archives such as `.tar.gz` and `.tgz` are intentionally not processed by the active proxy builder. This is a practical constraint due to archive size and the absence of safe archive extraction in the current collector. External dataset dissemination is nevertheless captured through other structured sources, especially Kaggle-hosted XSum-format files.

## Detector Outputs

The project separates row-level evidence, aggregate summaries, and execution traces:

- `runs/` — parquet tables with item-level detector evidence.
- `outputs/` — JSON summaries with aggregate detector outputs.
- `logs/` — JSONL execution traces.

## Report-Time Response Verification

`scripts/build_report_csv.py` performs a downstream response-quality verification pass before assembling the static report CSV. This pass does not recompute detector scores or the final CRS. Instead, it checks whether model outputs are interpretable enough for the detector evidence to be trusted in the report.

The verification combines detector summary JSON fields with offline checks over parquet and JSONL artefacts:

- `SSem` — checks BDQ/BCQ raw responses and final parsed choices, then reports `SSem_parse_failure_rate`, `SSem_non_choice_rate`, `SSem_interpretability`, `SSem_evidence_caveat`, and `SSem_quality_reasons`.
- `SMem` — classifies reference completions as valid completions, empty outputs, capability refusals, or meta refusals, then reports `refusal_rate`, `valid_completion_rate`, `refusal_breakdown`, `SMem_interpretability`, `SMem_evidence_caveat`, and `SMem_quality_reasons`.
- `SProb` — checks sampled summaries and greedy anchor outputs for empty outputs, refusals/disclaimers, meta answers, non-summary text, and multi-sentence responses, then reports `SProb_valid_summary_rate`, issue rates, `SProb_interpretability`, `SProb_evidence_caveat`, and `SProb_quality_reasons`.

Interpretability is reported as `HIGH`, `MODERATE`, or `LOW`. `*_evidence_caveat` flags cases where failure, refusal, parsing, or output-format problems may limit how confidently the corresponding detector evidence should be interpreted.

## Sharing Results Via GitHub

The repository supports a hybrid sharing mode for management review:

- GitHub Pages for a presentation-ready HTML view.
- Direct access to all tracked run artefacts from `runs/`, `outputs/`, and `logs/`.

The entry point is the repository root `index.html`, which links to:

- `assessment/contamination_report.html` — management-facing report.
- `assessment/contamination_report_bbc2025.html` — BBC2025 management-facing report.
- `artifacts/index.html` — file browser for all published run outputs.

Pages deployment is handled by `.github/workflows/deploy-pages.yml`. On every push to `main`, the workflow:

1. Regenerates `assessment/data/artifacts_manifest.json`.
2. Publishes `assessment/`, XSum `runs/`, `outputs/`, `logs/`, `artifacts/`, and the root `index.html` to GitHub Pages.

The BBC2025 HTML report is present under `assessment/`, but the current Pages workflow does not publish `runs_bbc2025/`, `outputs_bbc2025/`, or `logs_bbc2025/`. The report can still be opened locally from the repository root with a simple HTTP server.

To publish updated results:

```bash
python3 scripts/build_pages_artifacts_manifest.py
git add assessment/ artifacts/ runs/ outputs/ logs/ index.html .github/workflows/deploy-pages.yml .gitignore README.md
git commit -m "Publish updated contamination results"
git push origin main
```

Then enable GitHub Pages in the repository settings to deploy from GitHub Actions.

### SLex

Command:

```bash
python3 scripts/run_lexical_detector.py \
  --config configs/run_config.yaml \
  --proxy_column summary_ref
```

Primary outputs:

- `runs/v3_lexical.parquet`
- `outputs/v3_lexical_summary.json`
- `logs/v3_lexical.jsonl`

Important fields:

- item-level: `MaxSpanLen`, `NgramHits`, `ProxyCount`, `SLex`
- aggregate: `SLex_aggregate`

Current item-level mapping:

```text
SLex = 3 if MaxSpanLen >= 100
SLex = 2 if MaxSpanLen >= 50 or NgramHits >= 3
SLex = 1 if 30 <= MaxSpanLen < 50 or NgramHits in {1, 2}
SLex = 0 otherwise
```

`ProxyCount` is retained as a diagnostic metric but is not used in the current `SLex` level mapping.

Aggregate-level mapping:

```text
SLex_aggregate = 0 if no positive items
SLex_aggregate = 1 if only isolated level-1 items
SLex_aggregate = 2 if at least one level-2 item or at least 5% positive items
SLex_aggregate = 3 if at least one level-3 item or at least 10% level-2/3 items
```

### SSem

Command:

```bash
python3 scripts/run_dcq_detector.py --config configs/run_config.yaml --model_id gpt4omini
```

Primary outputs:

- `runs/v4_dcq_{model_id}.parquet`
- `outputs/v4_dcq_summary_{model_id}.json`
- `logs/v4_dcq_{model_id}.jsonl`

Important summary fields:

- `CPS`
- `pe`
- `kappa_min`
- `e_rate`
- `SSem_aggregate`
- `SSem` retained for compatibility

`SSem_aggregate` is derived from aggregated BCQ/BDQ metrics, not from a median of item-level values.

### SMem

Command:

```bash
python3 scripts/run_mem_probe.py --config configs/run_config.yaml --model_id gpt4omini
```

Primary outputs:

- `runs/v5_mem_{model_id}.parquet`
- `outputs/v5_mem_summary_{model_id}.json`
- `logs/v5_mem_{model_id}.jsonl`

Important fields:

- item-level: `EM_{model_id}`, `NED_{model_id}`, `NE_{model_id}`, `SMem_{model_id}`
- aggregate summary: `EM_rate`, `NE_rate`, `EM_control`, `SMem_aggregate`

Aggregate-level mapping:

```text
SMem_aggregate = 0 if exact_count = 0
SMem_aggregate = 1 if exact_count >= 1, EM_rate < 0.05, NE_rate < 0.15
SMem_aggregate = 2 if exact_count >= 1 and (EM_rate >= 0.05 or NE_rate >= 0.15)
SMem_aggregate = 3 if exact_count >= 1, EM_rate >= 0.15, NE_rate >= 0.35, and the control-baseline rule permits level 3
```

If `use_control_prefix` is enabled and `EM_control > 0`, level 3 additionally requires `EM_rate / EM_control >= 2.0`. If `use_control_prefix` is enabled and `EM_control` is missing or zero, level 3 is allowed for candidates that meet the `EM_rate` and `NE_rate` thresholds. Without a control baseline, level-3 candidates are capped at 2.

### SProb

Command:

```bash
python3 scripts/run_stability_detector.py --config configs/run_config.yaml --model_id gpt4omini
```

Primary outputs:

- `runs/v6_stability_{model_id}.parquet`
- `outputs/v6_stability_summary_{model_id}.json`
- `logs/v6_stability_{model_id}.jsonl`

Important fields:

- item-level: `UAR`, `mNED`, `anchor_mNED`, `peak_eps`, `SProb`
- aggregate: `SProb_aggregate`

Item-level `SProb`:

```text
SProb = max(B_abs, B_anchor, B_contrast)
```

where:

- `B_abs` is derived from `UAR` and `mNED`
- `B_anchor` is derived from `anchor_mNED` and `peak_eps`
- `B_contrast` is optional and uses a control baseline when enabled

By default, the main XSum configuration does not enable a stability control baseline, so `B_contrast = 0` for standard runs.

Aggregate-level mapping:

```text
Let N be the number of valid items.
Let nk be the number of items with SProb = k.
Let n+ = n1 + n2 + n3.
Let p+ = n+ / N, p23 = (n2 + n3) / N, and p3 = n3 / N.

SProb_aggregate = 0 if n+ = 0
SProb_aggregate = 1 if n+ > 0, n2 = 0, n3 = 0, and p+ < 0.10
SProb_aggregate = 2 if n2 >= 1 or p+ >= 0.10 or p23 >= 0.05, provided level-3 conditions are not met
SProb_aggregate = 3 if n3 >= 2 or p3 >= 0.05 or p23 >= 0.15
```

## Risk Integration

Command:

```bash
python3 scripts/run_risk_integration.py --config configs/run_config.yaml --model_id gpt4omini
```

Primary outputs:

- `outputs/v7_risk_summary_{model_id}.json`
- `logs/v7_risk_{model_id}.jsonl`

The risk integration stage consumes aggregate detector outputs from summary JSON files:

- `outputs/v3_lexical_summary.json` -> `SLex_aggregate`
- `outputs/v4_dcq_summary_{model_id}.json` -> `SSem_aggregate`
- `outputs/v5_mem_summary_{model_id}.json` -> `SMem_aggregate`
- `outputs/v6_stability_summary_{model_id}.json` -> `SProb_aggregate`

Current CRS formula:

```text
CRS_raw =
    0.35 * (SSem_aggregate / 3)
  + 0.35 * (SMem_aggregate / 3)
  + 0.30 * (SProb_aggregate / 3)
```

`SLex_aggregate` is excluded from `CRS` because it is a benchmark-level exposure signal rather than a model-behaviour signal. It is used in the confidence calculation as an exposure prior.

Safety override:

```text
If any of {SSem_aggregate, SMem_aggregate, SProb_aggregate} == 3:
    CRS = max(CRS_raw, 0.60)
else:
    CRS = CRS_raw
```

Risk levels:

```text
LOW      if CRS < 0.25
MODERATE if 0.25 <= CRS < 0.50
HIGH     if 0.50 <= CRS < 0.75
CRITICAL if CRS >= 0.75
```

Confidence:

```text
coverage  = count(score > 0 in {SSem, SMem, SProb}) / 3
agreement = 1 - variance(SSem, SMem, SProb) / 3
exposure  = SLex_aggregate / 3

confidence = (coverage + agreement + exposure) / 3
```

## Quick Start

Run from the repository root.

```bash
# 1) Optional: rebuild external proxy corpus
python3 scripts/run_proxy_builder.py --config configs/run_config.yaml
python3 scripts/build_proxy_structured_merged.py --config configs/run_config.yaml

# 2) Model-independent lexical stage
python3 scripts/run_lexical_detector.py \
  --config configs/run_config.yaml \
  --proxy_column summary_ref

# 3) Model-dependent detector stages
python3 scripts/run_dcq_detector.py --config configs/run_config.yaml --model_id gpt4omini
python3 scripts/run_mem_probe.py --config configs/run_config.yaml --model_id gpt4omini
python3 scripts/run_stability_detector.py --config configs/run_config.yaml --model_id gpt4omini

# 4) Aggregate risk integration
python3 scripts/run_risk_integration.py --config configs/run_config.yaml --model_id gpt4omini
```

Use `--limit N` on detector scripts for pilot runs where supported.

For the active BBC2025 run, use the isolated configuration and output paths:

```bash
python3 scripts/run_proxy_builder.py --config configs/run_config_bbc2025.yaml
python3 scripts/build_proxy_structured_merged.py --config configs/run_config_bbc2025.yaml

python3 scripts/run_lexical_detector.py \
  --config configs/run_config_bbc2025.yaml \
  --proxy_column summary_ref

python3 scripts/run_dcq_detector.py --config configs/run_config_bbc2025.yaml --model_id gpt4omini
python3 scripts/run_mem_probe.py --config configs/run_config_bbc2025.yaml --model_id gpt4omini
python3 scripts/run_stability_detector.py --config configs/run_config_bbc2025.yaml --model_id gpt4omini
python3 scripts/run_risk_integration.py --config configs/run_config_bbc2025.yaml --model_id gpt4omini

python3 scripts/build_report_csv.py \
  --config configs/run_config_bbc2025.yaml \
  --model_ids gpt4omini gemini25flash llama31_8b \
  --benchmark BBC2025 \
  --out assessment/data/report_data_bbc2025.csv
```

## Requirements

Python 3.10+ is recommended.

The codebase uses the following main libraries:

```bash
pip install pandas pyarrow pyyaml numpy openai google-genai requests
```

Optional dependencies:

```bash
pip install tiktoken transformers kaggle
```

- `tiktoken` is used only when `stability.tokenization: "tiktoken"`.
- `transformers` is used only when `stability.tokenization: "hf"`.
- `kaggle` is used by the proxy builder when Kaggle collection is enabled.

## Environment Variables

Set only the keys needed for the providers and stages you run:

```bash
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."
export OPENROUTER_API_KEY="..."
export DEEPSEEK_API_KEY="..."
export GITHUB_TOKEN="..."
export KAGGLE_USERNAME="..."
export KAGGLE_KEY="..."
```

## Reproducibility Notes

- The frozen master table is the central evaluation substrate.
- Main XSum runtime settings are controlled through `configs/run_config.yaml`.
- BBC2025 runtime settings are controlled through `configs/run_config_bbc2025.yaml` and write to isolated `*_bbc2025` directories.
- Archived benchmark branches and exploratory scripts are stored under `legacy/`.
- Detector prompts are stored in `src/prompts.py`.
- Long-running stages write checkpoint parquet files and JSONL logs.
- Model-facing stages are API-only and may be affected by provider-side changes or backend nondeterminism.
- For comparable runs, keep the frozen master table, proxy corpus, configuration, prompts, and detector thresholds fixed.
