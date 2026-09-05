# D3 Rule-Based Evaluation

This directory is the standalone D3 backend, parallel to `evaluation/rule_based`.
It evaluates paired D3/HTML programs that render SVG.

```text
candidate.html + removed.html
  -> headless Chrome
  -> SVG DOM Raw IR
  -> object-level diff
  -> seven-category Semantic IR
  -> annotation Jaccard + color matching + chart fidelity + bbox
```

The D3 ground-truth and removed-baseline code ships with the dataset
(`chartanno_d3_svg.tar.gz`, see the [HuggingFace dataset](https://huggingface.co/datasets/chartanno/ChartAnno)):
`d3_svg_data/code/d3/gt/` (annotated GT) and `d3_svg_data/code/d3/removed/` (unannotated baseline).

Evaluate a set of model predictions:

```bash
python evaluation/rule_based_d3/scripts/run_all_metrics.py \
  --gt-root /path/to/d3_gt \
  --test-root /path/to/d3_predictions \
  --baseline-root /path/to/d3_removed
```

The default data locations mirror the Matplotlib dataset roles:

```text
outputs/eval_assets/dataset_code_D3/          # GT
outputs/eval_assets/dataset_removed_D3/       # Removed baseline
outputs/eval_assets/test_code_D3/             # Test/model predictions
```

With the default paths populated, the command can be run without path flags.
Outputs mirror the Matplotlib evaluator under `evaluation/rule_based_d3/outputs/`:

```text
annotations/raw_gt_minus_removed/
annotations/raw_test_minus_removed/
annotations/final_gt_structured/
annotations/final_test_minus_removed_structured/
analysis/annotation_matching/
analysis/color_matching/
analysis/chart_fidelity/
analysis/summary.json
visualizations/screenshots/
visualizations/bboxes/
```

Jaccard and Color compare `(D3 GT - D3 removed)` against
`(D3 prediction - D3 removed)`. Fidelity compares the D3 prediction with the
same D3 removed baseline. No Matplotlib GT is used.

The backend implementation is in `d3_eval/`. `scripts/run_d3_diff.py` runs a
single pair; `scripts/run_all_metrics.py` is the canonical full entrypoint.

Requirements: Python with `playwright` (headless Chrome) plus `npm install` for
the local D3 bundle; standalone `.js` charts are wrapped into HTML automatically
(the local `node_modules/d3` is preferred, with a CDN fallback).
