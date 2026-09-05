# SVG Rule-Based Evaluation

This directory is the standalone SVG backend, parallel to `evaluation/rule_based`.
It evaluates paired SVG documents independently of D3.

```text
candidate.svg + removed.svg
  -> headless Chrome
  -> SVG DOM Raw IR
  -> object-level diff
  -> seven-category Semantic IR
  -> annotation Jaccard + color matching + chart fidelity + bbox
```

The SVG ground-truth and removed-baseline documents ship with the dataset
(`chartanno_d3_svg.tar.gz`, see the [HuggingFace dataset](https://huggingface.co/datasets/chartanno/ChartAnno)):
`d3_svg_data/code/svg/gt/` (annotated GT) and `d3_svg_data/code/svg/removed/` (unannotated baseline).

Evaluate a set of model predictions:

```bash
python evaluation/rule_based_svg/scripts/run_all_metrics.py \
  --gt-root /path/to/svg_gt \
  --test-root /path/to/svg_predictions \
  --baseline-root /path/to/svg_removed
```

The default data locations mirror the Matplotlib dataset roles:

```text
outputs/eval_assets/dataset_code_SVG/          # GT
outputs/eval_assets/dataset_removed_SVG/       # Removed baseline
outputs/eval_assets/test_code_SVG/             # Test/model predictions
```

With the default paths populated, the command can be run without path flags.
Outputs mirror the Matplotlib evaluator under annotations, analysis, and
visualizations subdirectories in `evaluation/rule_based_svg/outputs/`.

The SVG backend has its own `svg_eval/` package and does not import `d3_eval`.
Jaccard and Color compare `(SVG GT - SVG removed)` against
`(SVG prediction - SVG removed)`. Fidelity compares the SVG prediction with
the same SVG removed baseline. No Matplotlib GT is used.

`scripts/convert_matplotlib_to_svg.py` converts Matplotlib charts into SVG
pairs for adapter smoke tests. Requirement: Python with `playwright`
(headless Chrome).
