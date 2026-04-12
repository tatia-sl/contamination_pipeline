# Contamination Risk Assessment Pipeline

This repository implements an API-only pipeline for assessing possible benchmark contamination in closed-source large language models. The current benchmark is `EdinburghNLP/XSum`, evaluated on a frozen XSum test subset.

The pipeline is designed for black-box evaluation. It does not require model weights, logits, internal activations, or access to provider training data. All model-dependent evidence is collected through provider APIs and stored as auditable artifacts.

## Current Project Contract

The active configuration file is:

```text
configs/run_config.yaml
```

The frozen evaluation substrate is:

```text
master_table_xsum_n300_seed42_v4_dcq4_frozen.parquet
```

The frozen table contains `296` XSum test items and includes:

- identity/source fields: `xsum_id`, `split`, `document`, `summary_ref`
- normalized fields: `document_norm`, `summary_ref_norm`
- memorization fields: `prefix_ref`, `control_prefix`
- semantic-choice fields: canonical and paraphrased summary variants for DCQ

The active final proxy corpus is:

```text
data/proxies/proxy_structured_merged.csv
```

The active report data file is:

```text
assessment/data/report_data.csv
```

The active HTML management report is:

```text
assessment/contamination_report.html
```

## Repository Layout

```text
configs/       Versioned pipeline configuration.
src/           Prompt templates and API clients.
scripts/       Executable pipeline stages.
data/          Benchmark preparation files and proxy corpus artifacts.
runs/          Item-level detector outputs in parquet format.
outputs/       Aggregate detector summaries and risk summaries in JSON.
logs/          JSONL execution logs for audit and resume/debugging.
assessment/    Static HTML report and report CSV data.
legacy/        Archived scripts, old outputs, and superseded artifacts.
```

## Detector Signals

The pipeline produces four contamination-related signals.

| Signal | Level | Meaning |
|---|---:|---|
| `SLex` | benchmark-level | Public lexical exposure of the benchmark in reviewed external sources. |
| `SSem` | model-level | Semantic familiarity: preference for benchmark-original wording over equivalent alternatives. |
| `SMem` | model-level | Direct reconstruction: ability to reproduce benchmark content from a partial prompt. |
| `SProb` | model-level | Output concentration: unusually similar repeated outputs under stochastic sampling. |

Each detector produces item-level evidence where appropriate and an aggregate signal:

```text
SLex_aggregate
SSem_aggregate
SMem_aggregate
SProb_aggregate
```

The final risk integration stage combines `SSem_aggregate`, `SMem_aggregate`, and `SProb_aggregate` into a composite contamination risk score (`CRS`). `SLex_aggregate` is not included directly in the CRS formula because it is a benchmark-exposure prior, not model-behaviour evidence. It is used as context and contributes to confidence/exposure interpretation.

## Active Pipeline Stages

Run all commands from the repository root.

### 1. Proxy Corpus Construction

```bash
python3 scripts/run_proxy_builder.py --config configs/run_config.yaml
python3 scripts/build_proxy_structured_merged.py --config configs/run_config.yaml
```

The proxy builder collects candidate external records from GitHub and Kaggle, extracts summary-like content, preserves provenance, and writes structured per-source CSVs. The merge stage deduplicates by normalized summary text and writes the final proxy corpus.

Primary artifacts:

```text
data/proxies/proxy_structured_github.csv
data/proxies/proxy_structured_kaggle.csv
data/proxies/proxy_structured_merged.csv
data/proxies/proxy_sources_manifest_external_2026_02_06.jsonl
outputs/proxy_build_summary_external.json
outputs/proxy_structured_merged_build_summary.json
```

Compressed archives such as `.tar.gz` and `.tgz` are not processed by the active proxy builder. This is an intentional practical constraint due to archive size and safe extraction concerns. Dataset dissemination is nevertheless captured through structured public sources, especially Kaggle-hosted XSum-format files.

### 2. Lexical Exposure Detector (`SLex`)

```bash
python3 scripts/run_lexical_detector.py \
  --config configs/run_config.yaml \
  --proxy_column summary_ref
```

Primary artifacts:

```text
runs/v3_lexical.parquet
outputs/v3_lexical_summary.json
logs/v3_lexical.jsonl
```

The detector compares benchmark reference summaries against the structured proxy corpus using normalized text and n-gram/span overlap. It produces item-level `SLex` values and a benchmark-level `SLex_aggregate`.

Current item-level mapping:

```text
SLex = 3 if MaxSpanLen >= 100
SLex = 2 if MaxSpanLen >= 50 or NgramHits >= 3
SLex = 1 if 30 <= MaxSpanLen < 50 or NgramHits in {1, 2}
SLex = 0 otherwise
```

Current aggregate mapping:

```text
SLex_aggregate = 0 if no positive items
SLex_aggregate = 1 if only isolated level-1 items
SLex_aggregate = 2 if at least one level-2 item or at least 5% positive items
SLex_aggregate = 3 if at least one level-3 item or at least 10% level-2/3 items
```

### 3. Semantic Familiarity Detector (`SSem`)

```bash
python3 scripts/run_dcq_detector.py --config configs/run_config.yaml --model_id gpt4omini
```

Primary artifacts:

```text
runs/v4_dcq_{model_id}.parquet
outputs/v4_dcq_summary_{model_id}.json
logs/v4_dcq_{model_id}.jsonl
```

The detector presents the model with discriminative choice questions over canonical and paraphrased summaries. It aggregates BDQ/BCQ outcomes into model-level metrics such as `CPS`, `p_e`, `kappa_min`, and `e_rate`, then assigns `SSem_aggregate`.

### 4. Direct Reconstruction Detector (`SMem`)

```bash
python3 scripts/run_mem_probe.py --config configs/run_config.yaml --model_id gpt4omini
```

Primary artifacts:

```text
runs/v5_mem_{model_id}.parquet
outputs/v5_mem_summary_{model_id}.json
logs/v5_mem_{model_id}.jsonl
```

The detector probes whether the model can reconstruct the benchmark reference summary from a partial prefix. It computes exact match (`EM`), near-exact reconstruction (`NE`), normalized edit distance (`NED`), and a control-prefix baseline where enabled.

Aggregate-level mapping:

```text
SMem_aggregate = 0 if exact_count = 0
SMem_aggregate = 1 if exact_count >= 1, EM_rate < 0.05, NE_rate < 0.15
SMem_aggregate = 2 if exact_count >= 1 and (EM_rate >= 0.05 or NE_rate >= 0.15)
SMem_aggregate = 3 if exact_count >= 1, EM_rate >= 0.15, NE_rate >= 0.35, and contrast >= 2x
```

### 5. Output Concentration Detector (`SProb`)

```bash
python3 scripts/run_stability_detector.py --config configs/run_config.yaml --model_id gpt4omini
```

Primary artifacts:

```text
runs/v6_stability_{model_id}.parquet
outputs/v6_stability_summary_{model_id}.json
logs/v6_stability_{model_id}.jsonl
```

The detector repeatedly samples summaries from the model under stochastic decoding and compares the resulting outputs with each other and with a deterministic greedy anchor. It computes `UAR`, `mNED`, `anchor_mNED`, `peak_eps`, item-level `SProb`, and model-level `SProb_aggregate`.

Aggregate-level mapping:

```text
Let N be the number of valid items.
Let nk be the number of items assigned SProb = k.
Let n+ = n1 + n2 + n3.
Let p+ = n+ / N, p23 = (n2 + n3) / N, and p3 = n3 / N.

SProb_aggregate = 0 if n+ = 0
SProb_aggregate = 1 if n+ > 0, n2 = 0, n3 = 0, and p+ < 0.10
SProb_aggregate = 2 if n2 >= 1 or p+ >= 0.10 or p23 >= 0.05, provided level-3 conditions are not met
SProb_aggregate = 3 if n3 >= 2 or p3 >= 0.05 or p23 >= 0.15
```

### 6. Risk Integration

```bash
python3 scripts/run_risk_integration.py --config configs/run_config.yaml --model_id gpt4omini
```

Primary artifacts:

```text
outputs/v7_risk_summary_{model_id}.json
logs/v7_risk_{model_id}.jsonl
```

The risk integration stage consumes:

```text
outputs/v3_lexical_summary.json
outputs/v4_dcq_summary_{model_id}.json
outputs/v5_mem_summary_{model_id}.json
outputs/v6_stability_summary_{model_id}.json
```

Current CRS formula:

```text
CRS_raw =
    0.35 * (SSem_aggregate / 3)
  + 0.35 * (SMem_aggregate / 3)
  + 0.30 * (SProb_aggregate / 3)
```

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

## Management Report

The project includes a static HTML management report:

```text
assessment/contamination_report.html
```

It also includes a visual workflow overview of the full pipeline:

```text
assessment/full_pipeline_overview.html
```

The page reads report data from:

```text
assessment/data/report_data.csv
```

Generate or refresh the report CSV after detector and risk-integration runs:

```bash
python3 scripts/build_report_csv.py \
  --model_ids gpt4omini gemini25flash \
  --benchmark XSum
```

The default output path is:

```text
assessment/data/report_data.csv
```

To include an additional model, first run stages `SSem`, `SMem`, `SProb`, and `risk integration` for that model, then add the model id to the CSV build command:

```bash
python3 scripts/build_report_csv.py \
  --model_ids gpt4omini gemini25flash NEW_MODEL_ID \
  --benchmark XSum
```

Serve the report from the repository root:

```bash
python3 -m http.server 8080
```

Open:

```text
http://localhost:8080/assessment/contamination_report.html
```

The report is a single-page static management view. It includes:

- About this report
- Benchmark Exposure / public availability prior
- Model cards with CRS and detector bars
- Detector signals for the selected model
- Overall risk score and confidence
- Recommended actions for management and ML teams
- Reproducibility fixed settings
- Artifacts by detector

The report is intentionally framed as a risk assessment. It does not claim proof of training-data inclusion.

## Quick Start: Full Run For One Model

```bash
# Optional: rebuild external proxy corpus
python3 scripts/run_proxy_builder.py --config configs/run_config.yaml
python3 scripts/build_proxy_structured_merged.py --config configs/run_config.yaml

# Model-independent lexical stage
python3 scripts/run_lexical_detector.py \
  --config configs/run_config.yaml \
  --proxy_column summary_ref

# Model-dependent detector stages
python3 scripts/run_dcq_detector.py --config configs/run_config.yaml --model_id gpt4omini
python3 scripts/run_mem_probe.py --config configs/run_config.yaml --model_id gpt4omini
python3 scripts/run_stability_detector.py --config configs/run_config.yaml --model_id gpt4omini

# Risk integration
python3 scripts/run_risk_integration.py --config configs/run_config.yaml --model_id gpt4omini

# Report CSV
python3 scripts/build_report_csv.py --model_ids gpt4omini --benchmark XSum
```

Use `--limit N` for pilot runs where supported by the relevant detector.

## Environment

Python 3.10+ is recommended.

Main dependencies:

```bash
pip install pandas pyarrow pyyaml numpy openai google-genai requests
```

Optional dependencies:

```bash
pip install kaggle tiktoken transformers
```

Provider and data-source credentials are read from environment variables:

```bash
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."
export DEEPSEEK_API_KEY="..."
export GITHUB_TOKEN="..."
export KAGGLE_USERNAME="..."
export KAGGLE_KEY="..."
```

Set only the variables required for the stages being run.

## Reproducibility Notes

- `configs/run_config.yaml` is the active versioned configuration profile.
- `master_table_xsum_n300_seed42_v4_dcq4_frozen.parquet` is the fixed evaluation substrate.
- `src/prompts.py` stores fixed prompt templates.
- Detector stages write parquet checkpoints, JSON summaries, and JSONL execution logs.
- The report CSV duplicates key metadata so the HTML page remains a static renderer.
- API-only evaluation can still be affected by provider-side model changes, backend updates, rate limits, or nondeterministic serving behaviour.
- Comparable runs require fixed master table, fixed proxy corpus, fixed config, fixed prompts, and stable detector thresholds.

## Active Models In Current Config

The current configuration includes:

```text
gpt4omini     -> gpt-4o-mini
gemini25flash -> gemini-2.5-flash
gpt35turbo    -> gpt-3.5-turbo
gpt4          -> gpt-4
```

Only models with completed detector summaries and risk summaries should be included in `build_report_csv.py`.
