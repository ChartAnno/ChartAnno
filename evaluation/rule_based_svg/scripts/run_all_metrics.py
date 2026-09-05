#!/usr/bin/env python3
"""Run SVG-native GT/prediction rule-based evaluation."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


TOOL_ROOT = Path(__file__).resolve().parents[1]


def _find_repo_root() -> Path:
    """Locate the repo containing evaluation/rule_based (works in the release too)."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "evaluation" / "rule_based" / "annotation_eval").is_dir():
            return parent
    raise RuntimeError("Could not locate the ChartAnno repository root")


REPO_ROOT = _find_repo_root()
MPL_RULE_BASED_ROOT = REPO_ROOT / "evaluation" / "rule_based"
for import_root in (TOOL_ROOT, MPL_RULE_BASED_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from annotation_eval.extraction.subtraction import pre_dedupe_annotation_dict  # noqa: E402
from svg_eval.metrics import compute_rule_based_metrics  # noqa: E402
from svg_eval.pipeline import extract_diffed_svg_bundle  # noqa: E402
from svg_eval.runtime import SVGBrowserRuntime  # noqa: E402


SEMANTIC_KIND = {
    "1_enclosure": "annotation_bbox",
    "2_connector": "annotation_arrow",
    "3_text": "text",
    "4_glyph": "collection",
    "5_color": "patch",
    "6_indicator": "line",
    "7_geometric": "axes",
}


def _dump(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _flatten_semantic(semantic: dict[str, list]) -> list[dict]:
    flattened = []
    for feature, items in semantic.items():
        for index, item in enumerate(items, start=1):
            if not isinstance(item, dict) or not isinstance(item.get("bbox"), (list, tuple)):
                continue
            record = dict(item)
            record["kind"] = SEMANTIC_KIND[feature]
            record["element_id"] = f"{feature}:{index}"
            flattened.append(record)
    return flattened


def _chart_relative(prediction: Path, prediction_root: Path, gt_root: Path) -> Path:
    relative = prediction.relative_to(prediction_root)
    if (gt_root / relative).is_file():
        return relative
    category = relative.parts[0] if len(relative.parts) > 1 else prediction.parent.name
    match = re.match(r"^([A-Za-z]+_\d+)", prediction.stem)
    if match is None:
        raise ValueError(f"Cannot infer chart id from prediction: {prediction}")
    return Path(category) / f"{match.group(1)}.svg"


def _aggregate(files: list[dict]) -> dict:
    intersection = sum(item["annotation_matching"]["overall"]["intersection"] for item in files)
    union = sum(item["annotation_matching"]["overall"]["union"] for item in files)
    color_gt = sum(item["color_matching"]["overall"]["gt_count"] for item in files)
    color_pred = sum(item["color_matching"]["overall"]["pred_count"] for item in files)
    color_similarity = sum(item["color_matching"]["overall"]["max_similarity"] for item in files)
    precision = color_similarity / color_pred if color_pred else (1.0 if color_gt == 0 else 0.0)
    recall = color_similarity / color_gt if color_gt else (1.0 if color_pred == 0 else 0.0)
    color_f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    count = len(files)
    fidelity_passed = sum(item["chart_fidelity"]["chart_fidelity"] for item in files)
    coverage = sum(item["chart_fidelity"]["protected_element_coverage"] for item in files)
    return {
        "samples": count,
        "annotation_matching": {
            "intersection": intersection,
            "union": union,
            "jaccard": round(1.0 if union == 0 else intersection / union, 6),
        },
        "color_matching": {
            "gt_count": color_gt,
            "pred_count": color_pred,
            "max_similarity": round(color_similarity, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(color_f1, 6),
        },
        "chart_fidelity": {
            "passed": fidelity_passed,
            "pass_rate": round(1.0 if count == 0 else fidelity_passed / count, 6),
            "mean_protected_element_coverage": round(1.0 if count == 0 else coverage / count, 6),
        },
    }


def main() -> int:
    assets = REPO_ROOT / "outputs" / "eval_assets"
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Jaccard/Color compare (SVG GT - SVG removed) with "
            "(SVG prediction - SVG removed)."
        ),
    )
    parser.add_argument("--gt-root", default=str(assets / "dataset_code_SVG"), help="Real SVG ground-truth root")
    parser.add_argument("--baseline-root", default=str(assets / "dataset_removed_SVG"), help="SVG removed/baseline root")
    parser.add_argument("--test-root", default=str(assets / "test_code_SVG"), help="SVG prediction root")
    parser.add_argument("--output-root", default=str(TOOL_ROOT / "outputs"))
    parser.add_argument("--sample", action="append", help="Relative prediction SVG path; may be repeated")
    args = parser.parse_args()

    gt_root = Path(args.gt_root).resolve()
    baseline_root = Path(args.baseline_root).resolve()
    prediction_root = Path(args.test_root).resolve()
    output_root = Path(args.output_root).resolve()
    predictions = (
        [prediction_root / relative for relative in args.sample]
        if args.sample
        else sorted(prediction_root.rglob("*.svg"))
    )
    if not predictions:
        raise SystemExit(
            f"No SVG predictions found under {prediction_root}. Formal metrics require "
            "real SVG GT, SVG test outputs, and SVG removed baselines."
        )

    annotations_root = output_root / "annotations"
    analysis_root = output_root / "analysis"
    visualizations_root = output_root / "visualizations"

    results = []
    with SVGBrowserRuntime() as runtime:
        for prediction in predictions:
            prediction_relative = prediction.relative_to(prediction_root)
            chart_relative = _chart_relative(prediction, prediction_root, gt_root)
            gt_candidate = gt_root / chart_relative
            baseline = baseline_root / chart_relative
            for required in (gt_candidate, baseline, prediction):
                if not required.is_file():
                    raise FileNotFoundError(required)

            gt_bundle = extract_diffed_svg_bundle(gt_candidate, baseline, runtime=runtime)
            prediction_bundle = extract_diffed_svg_bundle(
                prediction,
                baseline,
                candidate_screenshot=visualizations_root / "screenshots" / prediction_relative.with_suffix(".png"),
                runtime=runtime,
            )
            gt_semantic = pre_dedupe_annotation_dict(gt_bundle["diffed_semantic"])
            prediction_semantic = pre_dedupe_annotation_dict(prediction_bundle["diffed_semantic"])
            metrics = compute_rule_based_metrics(
                gt_semantic,
                prediction_semantic,
                baseline_raw=prediction_bundle["baseline_raw"],
                candidate_raw=prediction_bundle["candidate_raw"],
                baseline_canvas=prediction_bundle["baseline"].get("canvas"),
                candidate_canvas=prediction_bundle["candidate"].get("canvas"),
            )

            runtime.render_bbox_overlay(
                prediction,
                _flatten_semantic(prediction_semantic),
                visualizations_root / "bboxes" / prediction_relative.with_suffix(".png"),
                title="SVG prediction diffed semantic bboxes",
            )
            output_relative = prediction_relative.with_suffix(".json")
            chart_json = chart_relative.with_suffix(".json")
            _dump(annotations_root / "raw_gt_minus_removed" / chart_json, gt_bundle)
            _dump(annotations_root / "raw_test_minus_removed" / output_relative, prediction_bundle)
            _dump(annotations_root / "final_gt_structured" / chart_json, gt_semantic)
            _dump(annotations_root / "final_test_minus_removed_structured" / output_relative, prediction_semantic)
            _dump(analysis_root / "annotation_matching" / output_relative, metrics["annotation_matching"])
            _dump(analysis_root / "color_matching" / output_relative, metrics["color_matching"])
            _dump(analysis_root / "chart_fidelity" / output_relative, metrics["chart_fidelity"])

            item = {
                "prediction": prediction_relative.with_suffix("").as_posix(),
                "chart": chart_relative.with_suffix("").as_posix(),
                **metrics,
            }
            results.append(item)
            print(json.dumps({
                "prediction": item["prediction"],
                "chart": item["chart"],
                "jaccard": metrics["annotation_matching"]["overall"]["jaccard"],
                "color_f1": metrics["color_matching"]["overall"]["f1"],
                "chart_fidelity": metrics["chart_fidelity"]["chart_fidelity"],
            }, ensure_ascii=False))

    summary = {
        "backend": "svg-dom",
        "mode": "evaluation",
        "comparison": "SVG GT diff vs SVG prediction diff; both use the same SVG removed baseline",
        **_aggregate(results),
        "files": results,
    }
    _dump(analysis_root / "summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("backend", "mode", "samples", "annotation_matching", "color_matching", "chart_fidelity")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
