# Scripts

This directory contains helper scripts used by `run_pipeline.py`. All commands below assume the current working directory is `ChartAnno/`.

## `materialize_eval_assets.py`

Materializes evaluator assets from `data/input_code_image.jsonl` (row metadata such as `category` and `sample_id` is read from the inlined JSONL fields).

```bash
python3 scripts/materialize_eval_assets.py --repo-root .
```

Main outputs:

```text
outputs/eval_assets/
  dataset_code/
  dataset_code_removed/
  dataset_image_new/
  dataset_image_removed/
```

This script is called by:

```bash
./run_pipeline.sh --steps prepare
```

## `summarize_semantic_design.py`

Aggregates LLM-judge JSON outputs under `outputs/api/semantic_design/`.

```bash
python3 scripts/summarize_semantic_design.py \
  --root outputs/api/semantic_design \
  --dataset-image-root outputs/eval_assets/dataset_image_new \
  --only gpt54
```

Main outputs:

```text
outputs/api/semantic_design/
  _summary_by_model_mode_stage_missing_as_zero.csv
  _ranking_by_mode_stage_missing_as_zero.csv
  _overall_model_ranking_missing_as_zero.csv
  _vlm_vs_llm_relative_gain_missing_as_zero.csv
  _per_sample_semantic_design_missing_as_zero.csv
```

This script is called by:

```bash
./run_pipeline.sh --steps summarize-judge
```

## `summarize_current_model.py`

Merges rule-based and LLM-judged summaries for one evaluated model.

```bash
python3 scripts/summarize_current_model.py \
  --repo-root . \
  --source-model gpt54
```

Main outputs:

```text
results/per_model_combined_csv/
  gpt54.csv
```

This script is called by:

```bash
./run_pipeline.sh --steps summarize-current
```

## `summarize_model_results.py`

Aggregates bundled model per-model CSV files and recomputes all-model averages.

```bash
python3 scripts/summarize_model_results.py
```

Main outputs:

```text
results/
  summary/model_results_table.csv
  summary/model_results_table_4_metrics.csv
  summary/model_results_stage_summary.csv
  summary/model_results_overall_summary.csv
```

This script is called by:

```bash
./run_pipeline.sh --steps summarize-model-results
```

