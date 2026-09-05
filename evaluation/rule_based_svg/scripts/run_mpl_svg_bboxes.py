#!/usr/bin/env python3
"""Diff Matplotlib-generated SVG pairs and render classified semantic bboxes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = next(
    parent
    for parent in SCRIPT_PATH.parents
    if (parent / "outputs" / "eval_assets").is_dir()
)
SVG_DOM_TOOL_ROOT = REPO_ROOT / "evaluation" / "rule_based_svg"
if str(SVG_DOM_TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(SVG_DOM_TOOL_ROOT))

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


def _svg_src_prefix(semantic: dict[str, list]) -> None:
    for items in semantic.values():
        for item in items:
            if isinstance(item, dict) and str(item.get("src", "")).startswith("svg_"):
                item["src"] = "mpl_svg_" + item["src"][len("svg_"):]


def _flatten(semantic: dict[str, list]) -> list[dict]:
    records = []
    for feature_key, items in semantic.items():
        for index, item in enumerate(items, start=1):
            if not isinstance(item, dict) or not isinstance(item.get("bbox"), (list, tuple)):
                continue
            record = dict(item)
            record["kind"] = SEMANTIC_KIND[feature_key]
            record["element_id"] = f"{feature_key}:{index}"
            records.append(record)
    return records


def main() -> int:
    assets = REPO_ROOT / "outputs" / "eval_assets"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", default=str(assets / "dataset_code_MPL_SVG"))
    parser.add_argument("--baseline-root", default=str(assets / "dataset_removed_MPL_SVG"))
    parser.add_argument("--output-root", default=str(assets / "mpl_svg_bbox_visualizations"))
    args = parser.parse_args()

    candidate_root = Path(args.candidate_root).resolve()
    baseline_root = Path(args.baseline_root).resolve()
    output_root = Path(args.output_root).resolve()
    summaries = []

    with SVGBrowserRuntime() as runtime:
        for candidate in sorted(candidate_root.rglob("*.svg")):
            relative = candidate.relative_to(candidate_root)
            baseline = baseline_root / relative
            if not baseline.is_file():
                raise FileNotFoundError(baseline)
            bundle = extract_diffed_svg_bundle(candidate, baseline, runtime=runtime)
            bundle["backend"] = "matplotlib-svg"
            _svg_src_prefix(bundle["diffed_semantic"])
            semantic_records = _flatten(bundle["diffed_semantic"])

            runtime.render_bbox_overlay(
                candidate,
                semantic_records,
                output_root / "semantic_diff" / relative.with_suffix(".png"),
                title="Matplotlib SVG diffed semantic bboxes",
            )

            payload = {
                "backend": "matplotlib-svg",
                "candidate": str(candidate),
                "baseline": str(baseline),
                "classification_input_artist_ids": bundle["raw_diff"]["allowed_artist_ids"],
                "classification_input_records": bundle["classification_input_records"],
                "diffed_semantic": bundle["diffed_semantic"],
                "semantic_bbox_records": semantic_records,
            }
            json_path = output_root / "bbox_json" / relative.with_suffix(".json")
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

            summary = {
                "sample": relative.with_suffix("").as_posix(),
                "raw_diff_count": len(bundle["classification_input_records"]),
                "semantic_bbox_count": len(semantic_records),
                "semantic_counts": {key: len(value) for key, value in bundle["diffed_semantic"].items()},
            }
            summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False))

    summary_path = output_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(summaries)} Matplotlib-SVG bbox results to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
