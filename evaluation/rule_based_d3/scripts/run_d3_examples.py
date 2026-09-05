#!/usr/bin/env python3
"""Run every paired HTML adapter example."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-root",
        default=str(REPO_ROOT / "outputs" / "eval_assets" / "dataset_code_D3_examples"),
    )
    parser.add_argument(
        "--baseline-root",
        default=str(REPO_ROOT / "outputs" / "eval_assets" / "dataset_removed_D3_examples"),
    )
    parser.add_argument("--output-root", default=str(TOOL_ROOT / "outputs" / "examples"))
    args = parser.parse_args()

    candidate_root = Path(args.candidate_root).resolve()
    baseline_root = Path(args.baseline_root).resolve()
    output_root = Path(args.output_root).resolve()
    pairs = []
    for candidate in sorted(candidate_root.rglob("*.html")):
        relative = candidate.relative_to(candidate_root)
        baseline = baseline_root / relative
        if not baseline.is_file():
            raise FileNotFoundError(f"Missing paired baseline: {baseline}")
        pairs.append((relative, candidate, baseline))

    summaries = []
    with D3BrowserRuntime() as runtime:
        for relative, candidate, baseline in pairs:
            stem = relative.with_suffix("")
            result = extract_diffed_d3_bundle(
                candidate,
                baseline,
                candidate_screenshot=output_root / "screenshots" / relative.with_suffix(".png"),
                runtime=runtime,
            )
            output = output_root / "ir" / relative.with_suffix(".json")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            counts = {key: len(value) for key, value in result["diffed_semantic"].items()}
            summaries.append({"sample": stem.as_posix(), "semantic_counts": counts})
            print(json.dumps(summaries[-1], ensure_ascii=False))

    summary_path = output_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(summaries)} examples to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
