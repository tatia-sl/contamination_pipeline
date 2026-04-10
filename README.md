# Contamination Detection Pipeline

This repository contains an API-based pipeline for estimating potential benchmark contamination in large language models using the XSum summarization benchmark.

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

The final `run_risk_integration.py` stage consumes these aggregate detector outputs from summary JSON files and produces a model-level composite contamination risk score (`CRS`), qualitative risk level, and confidence estimate.

## Repository Layout

- `configs/run_config.yaml` — main versioned configuration file.
- `src/prompts.py` — fixed prompt templates used by detector stages.
- `src/clients/` — API clients for OpenAI-compatible providers and Gemini.
- `scripts/` — executable pipeline stages.
- `data/` — proxy corpus files and related source artefacts.
- `runs/` — row-level parquet outputs from detector stages.
- `outputs/` — aggregate JSON summaries and final reports.
- `logs/` — JSONL execution logs.
- `legacy/` — archived scripts and old run artefacts.

## Core Experimental Artefact

The central reproducibility substrate is:

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

## Active Pipeline Stages

1. Proxy corpus construction
   - `scripts/run_proxy_builder_v4.py`
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

`scripts/build_management_report.py` is downstream reporting code and may be updated separately. It is not part of the detector contract.

## Proxy Corpus Construction

The proxy corpus is built in two explicit steps:

```bash
python3 scripts/run_proxy_builder_v4.py --config configs/run_config.yaml
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
SMem_aggregate = 3 if exact_count >= 1, EM_rate >= 0.15, NE_rate >= 0.35, and contrast >= 2x
```

If `EM_control = 0`, the contrast is undefined and level 3 is assigned conservatively. Without a control baseline, level-3 candidates are capped at 2.

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
python3 scripts/run_proxy_builder_v4.py --config configs/run_config.yaml
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
- All major runtime settings are controlled through `configs/run_config.yaml`.
- Detector prompts are stored in `src/prompts.py`.
- Long-running stages write checkpoint parquet files and JSONL logs.
- Model-facing stages are API-only and may be affected by provider-side changes or backend nondeterminism.
- For comparable runs, keep the frozen master table, proxy corpus, configuration, prompts, and detector thresholds fixed.
