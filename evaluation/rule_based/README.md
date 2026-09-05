# Annotation Evaluation Toolkit

This directory contains the low-level evaluation code for chart annotation generation. The evaluator lives here, while benchmark chart code, rendered images, extracted annotations, and metric outputs live under a separate `PROJECT_PATH`.

This directory is the Matplotlib backend. D3 and standalone SVG are maintained
as parallel backends in `evaluation/rule_based_d3` and
`evaluation/rule_based_svg`.

The public entrypoint uses a thin shell script to call the Python runner, computing all low-level metrics under `outputs/analysis`.

## Evaluation Flow

```text
chart code / structured annotation JSON
        │
        ▼
1. Element Extraction        annotation_eval/extraction/
   Optional by default. Runs only when full extraction is requested.
        │
        ▼
2. Metric Computation        annotation_eval/metrics/
   Chart Fidelity, Annotation Matching, Color Matching
        │
        ▼
3. Pipeline Summary          annotation_eval/pipeline/run_all_metrics.py
   One model/stage summary CSV
```

By default, `scripts/run_evaluation_all.sh` skips element extraction and annotation diff. It reads existing structured annotation JSON files and computes all scores. Use `scripts/run_full_extraction_and_evaluation.sh` only when structured annotations need to be regenerated from chart code.

## Repository Layout

```text
.
├── README.md
├── requirements.txt
├── configs/
│   └── pipeline_config.example.json
├── scripts/
│   ├── run_evaluation_all.sh
│   ├── run_evaluation_all.py
│   ├── run_full_extraction_and_evaluation.sh
│   └── run_evaluation_model_list.sh
└── annotation_eval/
    ├── config.py
    ├── extraction/
    │   ├── annotation_schema.py
    │   ├── element_extractor.py
    │   ├── diff_extractor.py
    │   ├── diff_pipeline.py
    │   ├── raw_elements.py
    │   ├── raw_diff.py
    │   ├── subtraction.py
    │   ├── runtime.py
    │   ├── geometry.py
    │   └── heuristics/
    ├── metrics/
    │   ├── chart_fidelity.py
    │   ├── annotation_matching.py
    │   ├── color_matching.py
    │   ├── text_extraction.py
    │   └── text_relationship.py
    └── pipeline/
        └── run_all_metrics.py
```

## Module Overview

- **`scripts/`**: Shell entrypoints for evaluating single models, lists of models, and controlling whether to skip extraction or reuse existing artifacts.
- **`annotation_eval/extraction/`**: Handles parsing Matplotlib figures and extracting structured annotations (`enclosure`, `connector`, `text`, `glyph`, `color`, `indicator`, `geometric`). It executes the generated Python code safely and uses differential heuristics to isolate added annotations from the base chart.
- **`annotation_eval/metrics/`**: Contains the core rule-based scoring logic:
  - `chart_fidelity.py`: Ensures protected baseline data and layout are preserved.
  - `annotation_matching.py`: Computes Jaccard-style matching for structural compliance.
  - `color_matching.py`: Computes CIEDE2000 color similarity.
- **`annotation_eval/pipeline/`**: Orchestrates the extraction and metric computation processes, producing the final summary CSV files.

## Expected Project Layout

Default score-only evaluation expects:

```text
${PROJECT_PATH}/
├── dataset_code_removed/
│   └── <Category>/<ChartID>.py
├── test_code/
│   └── <Category>/<ChartID>/<code|code+image>/<ChartID>_<code|code_image>_<stage>.py
├── test_code_image/
│   └── <Category>/<ChartID>/<code|code+image>/<ChartID>_<code|code_image>_<stage>.png
├── outputs/annotations/
│   ├── final_gt_structured/
│   ├── final_test_llm_minus_removed_structured/
│   └── final_test_vlm_minus_removed_structured/
└── outputs/
```

Full extraction additionally expects:

```text
${PROJECT_PATH}/
├── dataset_code/
│   └── <Category>/<ChartID>.py
├── dataset_code_removed/
│   └── <Category>/<ChartID>.py
└── test_code/
    └── <Category>/<ChartID>/<code|code+image>/<ChartID>_<code|code_image>_<stage>.py
```

`stage` must be one of `intent`, `operation`, or `implementation`. Legacy files named with `task` are still accepted and normalized to `intent`. The rendered image directory defines the scoring universe. If a generated image is missing, that sample is counted as missing and receives score `0` in the final summary.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Generated chart scripts may import additional plotting packages. Install those packages in the same environment before running evaluation.

## One-Command Evaluation

Default score-only evaluation:

```bash
PROJECT_PATH=/path/to/your/project \
TEST_CODE_DIR=test_code \
TEST_IMAGE_DIR=test_code_image \
bash scripts/run_evaluation_all.sh
```

For the current generated-output naming pattern:

```bash
PROJECT_PATH=/path/to/your/project \
TEST_CODE_DIR=test_code_gemini3flash \
TEST_IMAGE_DIR=test_code_image_gemini3flash \
bash scripts/run_evaluation_all.sh
```

Regenerate structured annotations from chart code, then evaluate:

```bash
PROJECT_PATH=/path/to/your/project bash scripts/run_full_extraction_and_evaluation.sh
```

Reuse existing metric artifacts without recomputing metric scripts:

```bash
SKIP_RUN=1 bash scripts/run_evaluation_all.sh
```

Evaluate multiple model folders:

```bash
PROJECT_PATH=/path/to/your/project bash scripts/run_evaluation_model_list.sh
```

## Main Outputs

For the standalone rule-based entrypoint, `outputs/analysis/` is the default analysis root. In the full `run_pipeline.sh` workflow, the same structure is written under `outputs/runs/<source-model>/analysis/`.

```text
outputs/analysis/
├── intermediates/
│   ├── chart_fidelity/
│   │   ├── per_chart/                 # Per-sample JSON result files
│   │   └── chart_fidelity_all.csv     # Combined results used by pipeline
│   ├── annotation_matching/
│   │   ├── per_chart/                 # Per-sample JSON result files
│   │   ├── annotation_matching_results.json
│   │   ├── annotation_matching_summary.csv
│   │   └── annotation_matching_per_file.csv
│   ├── color_matching/
│   │   ├── per_chart/                 # Per-sample JSON result files
│   │   ├── color_matching_results.json
│   │   ├── color_matching_summary.csv
│   │   └── color_matching_per_file.csv
│   ├── text_extraction/
│   └── text_relationship/
└── metric_summary/
    ├── low_level_scores_by_model_stage.csv
    └── low_level_scores_per_sample.csv
```

The final summary tables are:

```text
outputs/analysis/metric_summary/low_level_scores_by_model_stage.csv
outputs/analysis/metric_summary/low_level_scores_per_sample.csv
```

## Path Overrides

The shell entrypoint accepts path overrides through environment variables:

```bash
PROJECT_PATH=/path/to/your/project
DATASET_CODE_DIR=dataset_code
DATASET_CODE_REMOVED_DIR=dataset_code_removed
TEST_CODE_DIR=test_code
TEST_IMAGE_DIR=test_code_image
ANNOTATIONS_DIR=outputs/annotations
ANALYSIS_DIR=outputs/analysis
OUTPUT_CSV=outputs/analysis/metric_summary/custom.csv
SHARED_TEXT_EXTRACTION_DIR=
MPLCONFIGDIR=outputs/.mplconfig
```

All relative paths are resolved under `PROJECT_PATH`. Chart scripts are executed with `PROJECT_PATH` as the working directory, so relative paths inside chart code continue to work.
