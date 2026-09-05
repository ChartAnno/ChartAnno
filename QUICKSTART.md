# Quick Start

This guide runs from the `ChartAnno/` directory.

## 1. Install Dependencies

```bash
cd ChartAnno
python3 -m pip install -r requirements.txt
```

## 2. Set an API Key

For OpenAI:

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
```

For another provider, set a provider-specific environment variable and pass it with `--api-key-env`:

```bash
export PROVIDER_API_KEY="YOUR_API_KEY"
```

## 3. Download the Dataset

The dataset is hosted on HuggingFace and is gated. Click **Request access** on the [repo page](https://huggingface.co/datasets/chartanno/ChartAnno), then download with a read token ([settings/tokens](https://huggingface.co/settings/tokens)):

```bash
pip install -U huggingface_hub
hf login --token $HF_TOKEN
hf download chartanno/ChartAnno chartanno.tar.gz --repo-type dataset --local-dir .
tar -xzvf chartanno.tar.gz    # extracts to ./data

# optional D3.js / SVG extension (code-only, 120 charts x 3 levels per backend)
hf download chartanno/ChartAnno chartanno_d3_svg.tar.gz --repo-type dataset --local-dir .
tar -xzvf chartanno_d3_svg.tar.gz    # extracts to ./d3_svg_data
```

## 4. Smoke Test Without API Calls

*(Note: `--source-model` is used to specify the identifier/slug of the evaluated model for output folders, result files, and final summary reports.)*

Print the full command sequence:

```bash
./run_pipeline.sh \
  --steps all \
  --model gpt-5.4 \
  --source-model gpt54 \
  --max-data-rows 5 \
  --dry-run
```

Print the generation targets only:

```bash
MODE=both MODEL=gpt-5.4 MAX_DATA_ROWS=2 DRY_RUN=1 bash tasks/run_tasks.sh
```

## 5. Run a Small End-to-End Test

Run only a few rows before launching the full benchmark:

```bash
./run_pipeline.sh \
  --steps all \
  --model gpt-5.4 \
  --source-model gpt54 \
  --max-data-rows 10
```

This writes intermediate files under `outputs/`.

`--max-data-rows` is applied to each input setting, so `--max-data-rows 2` runs 2 `Input: code` rows and 2 `Input: code+Image` rows.

To run only specific instruction levels, pass `--levels intent` (or `--levels intent,operation`). The filter applies to generation; later steps process only what was generated.

## 6. Run the Full Benchmark

```bash
./run_pipeline.sh \
  --steps all \
  --model gpt-5.4 \
  --source-model gpt54 \
  --base-url https://api.openai.com/v1
```

This command runs generation, rendering, rule-based evaluation, LLM-judged evaluation, and single-model score merging for `gpt54`.

It writes:

```text
outputs/runs/gpt54/test_code/                         # generated Python code
outputs/runs/gpt54/test_images/                       # rendered generated charts
outputs/runs/gpt54/analysis/metric_summary/           # rule-based scores
outputs/runs/gpt54/api/semantic_design/               # LLM-judged scores
results/per_model_combined_csv/gpt54.csv              # merged scores for this model
```

The main single-model score files are:

```text
outputs/runs/gpt54/analysis/metric_summary/low_level_scores_by_model_stage.csv
outputs/runs/gpt54/api/semantic_design/high_level_scores_by_model_stage.csv
results/per_model_combined_csv/gpt54.csv
```

After evaluating one or more models, rebuild the cross-model summary tables with:

```bash
./run_pipeline.sh --steps summarize-model-results
```

This writes:

```text
results/summary/model_results_table.csv             # all metrics by model, input setting, and level
results/summary/model_results_table_4_metrics.csv   # compact table with the four headline metrics
results/summary/model_results_stage_summary.csv     # per-level summary for each model/input setting
results/summary/model_results_overall_summary.csv   # overall summary across levels
```

## Use Another Model Provider

For Kimi/Gemini/Claude through the same Chat Completions request format, change only the model name, base URL, and API key environment variable.

```bash
export PROVIDER_API_KEY="YOUR_API_KEY"

./run_pipeline.sh \
  --steps all \
  --model gemini31pro \
  --source-model your_model_slug \
  --base-url https://your-provider.example/v1 \
  --api-key-env PROVIDER_API_KEY
```

## Use a Separate Judge Model

Generation and judging can use different models:

```bash
./run_pipeline.sh \
  --steps judge-prompts,judge \
  --source-model gpt54 \
  --model gpt-5.4 \
  --judge-model gpt-5.4 \
  --judge-base-url https://api.openai.com/v1
```

## Common Workflows

Evaluate D3/SVG extension outputs (Matplotlib-free backends; see `evaluation/rule_based_d3/README.md` and `evaluation/rule_based_svg/README.md`):

```bash
python evaluation/rule_based_d3/scripts/run_all_metrics.py \
  --gt-root d3_svg_data/code/d3/gt \
  --test-root /path/to/d3_predictions \
  --baseline-root d3_svg_data/code/d3/removed
```

Prepare evaluator assets only:

```bash
./run_pipeline.sh --steps prepare
```

Generate both code and code+image outputs:

```bash
./run_pipeline.sh \
  --steps code,code-image \
  --model gpt-5.4 \
  --source-model gpt54
```

Render generated code:

```bash
./run_pipeline.sh --steps render
```

Run only evaluation and summaries after code/images already exist:

```bash
./run_pipeline.sh \
  --steps rule,judge-prompts,judge,summarize-judge,summarize-current \
  --model gpt-5.4 \
  --source-model gpt54
```

## Step Reference

| Step | Purpose |
| --- | --- |
| `prepare` | Materialize evaluator assets from JSONL |
| `code` | Generate code for the `Input: code` task |
| `code-image` | Generate code for the `Input: code+Image` task |
| `render` | Render generated Python files to chart images |
| `rule` | Run rule-based metrics |
| `judge-prompts` | Generate judge prompt files |
| `judge` | Call the LLM judge |
| `summarize-judge` | Aggregate judge JSON files |
| `summarize-current` | Merge rule-based and LLM-judged scores |
| `summarize-model-results` | Aggregate bundled model result CSVs |

## Metrics

Main outputs include:

```text
execution_success_rate
chart_fidelity, annotation_matching, color_matching
semantic_faithfulness, semantic_clarity
visual_clarity, annotation_organization_quality, attention_guidance
structural_compliance, semantic_consistency, design_effectiveness
```

Aggregate formulas:

```text
structural_compliance = chart_fidelity                  # intent
structural_compliance = mean(chart_fidelity, annotation_matching, color_matching)
semantic_consistency  = mean(semantic_faithfulness, semantic_clarity)
design_effectiveness  = mean(visual_clarity, annotation_organization_quality, attention_guidance)
```

Rule-based scores are in `[0, 1]`; LLM-judged scores are in `[0, 5]`.
