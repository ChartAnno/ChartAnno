#!/usr/bin/env python3
"""Run the D3 evaluator for one candidate/baseline pair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


TOOL_ROOT = Path(__file__).resolve().parents[1]
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

from d3_eval import extract_diffed_d3_bundle  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, help="Annotated/candidate D3 HTML file")
    parser.add_argument("--baseline", required=True, help="Removed/baseline D3 HTML file")
    parser.add_argument("--output", required=True, help="Output JSON bundle")
    parser.add_argument("--candidate-screenshot", help="Optional candidate PNG path")
    parser.add_argument("--baseline-screenshot", help="Optional baseline PNG path")
    args = parser.parse_args()

    result = extract_diffed_d3_bundle(
        args.candidate,
        args.baseline,
        candidate_screenshot=args.candidate_screenshot,
        baseline_screenshot=args.baseline_screenshot,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    counts = {key: len(value) for key, value in result["diffed_semantic"].items()}
    print(json.dumps({"output": str(output), "semantic_counts": counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

