#!/usr/bin/env python3
"""Render bbox overlays for every paired D3 candidate/baseline example."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


TOOL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "outputs" / "eval_assets").is_dir()
)
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

from d3_eval.pipeline import extract_diffed_d3_bundle  # noqa: E402
from d3_eval.runtime import D3BrowserRuntime  # noqa: E402


SEMANTIC_KIND = {
    "1_enclosure": "annotation_bbox",
    "2_connector": "annotation_arrow",
    "3_text": "text",
    "4_glyph": "collection",
    "5_color": "patch",
    "6_indicator": "line",
    "7_geometric": "axes",
}


def _flatten_semantic_bboxes(semantic: dict[str, list]) -> list[dict]:
    records = []
    for feature_key, items in semantic.items():
        if not isinstance(items, list):
            continue
        for index, item in enumerate(items, start=1):
            if not isinstance(item, dict) or not isinstance(item.get("bbox"), (list, tuple)):
                continue
            record = dict(item)
            record["kind"] = SEMANTIC_KIND.get(feature_key, "patch")
            record["element_id"] = f"{feature_key}:{index}"
            record["semantic_category"] = feature_key
            records.append(record)
    return records


def main() -> int:
    assets = REPO_ROOT / "outputs" / "eval_assets"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", default=str(assets / "dataset_code_D3_examples"))
    parser.add_argument("--baseline-root", default=str(assets / "dataset_removed_D3_examples"))
    parser.add_argument("--output-root", default=str(assets / "d3_bbox_visualizations"))
    parser.add_argument("--pattern", default="*.html", help="Input filename glob, e.g. *.html or *.svg")
    args = parser.parse_args()

    candidate_root = Path(args.candidate_root).resolve()
    baseline_root = Path(args.baseline_root).resolve()
    output_root = Path(args.output_root).resolve()

    summaries = []
    with D3BrowserRuntime() as runtime:
        for candidate in sorted(candidate_root.rglob(args.pattern)):
            relative = candidate.relative_to(candidate_root)
            baseline = baseline_root / relative
            if not baseline.is_file():
                raise FileNotFoundError(baseline)
            bundle = extract_diffed_d3_bundle(candidate, baseline, runtime=runtime)
            semantic_records = _flatten_semantic_bboxes(bundle["diffed_semantic"])
            png_relative = relative.with_suffix(".png")
            json_relative = relative.with_suffix(".json")
            runtime.render_bbox_overlay(
                candidate,
                semantic_records,
                output_root / "semantic_diff" / png_relative,
                title="Diffed semantic annotation bboxes",
            )
            bbox_payload = {
                "sample": relative.with_suffix("").as_posix(),
                "canvas": bundle["candidate"]["canvas"],
                "classification_input_artist_ids": bundle["raw_diff"]["allowed_artist_ids"],
                "diffed_semantic": bundle["diffed_semantic"],
                "semantic_bbox_records": semantic_records,
            }
            json_path = output_root / "bbox_json" / json_relative
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(bbox_payload, indent=2, ensure_ascii=False), encoding="utf-8")
            summary = {
                "sample": bbox_payload["sample"],
                "semantic_diff_bbox_count": len(semantic_records),
            }
            summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False))

    summary_path = output_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
