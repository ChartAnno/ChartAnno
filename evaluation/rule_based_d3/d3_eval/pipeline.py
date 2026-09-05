"""End-to-end D3 raw-IR, diff, and semantic-IR pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

from .runtime import D3BrowserRuntime
from .semantic import append_matched_color_diffs, classify_unmatched_records


def _find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "evaluation" / "rule_based" / "annotation_eval").is_dir():
            return parent
    raise RuntimeError("Could not locate the ChartAnno repository root")


_REPO_ROOT = _find_repo_root()
_MPL_EVAL_ROOT = _REPO_ROOT / "evaluation" / "rule_based"
if str(_MPL_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MPL_EVAL_ROOT))

from annotation_eval.extraction.raw_diff import build_raw_diff_result, to_jsonable_raw_diff  # noqa: E402


def extract_diffed_d3_bundle(
    candidate_file: str | Path,
    baseline_file: str | Path,
    *,
    candidate_screenshot: str | Path | None = None,
    baseline_screenshot: str | Path | None = None,
    runtime: D3BrowserRuntime | None = None,
) -> dict:
    """Render two D3 programs and return raw, delta, and semantic IR artifacts."""

    owns_runtime = runtime is None
    if owns_runtime:
        runtime = D3BrowserRuntime()
        runtime.__enter__()
    assert runtime is not None
    try:
        candidate_result = runtime.extract(candidate_file, screenshot_path=candidate_screenshot)
        baseline_result = runtime.extract(baseline_file, screenshot_path=baseline_screenshot)
        candidate_raw = candidate_result["records"]
        baseline_raw = baseline_result["records"]
        raw_diff = build_raw_diff_result(candidate_raw, baseline_raw, ignore_visual_color=True)
        allowed_artist_ids = {int(value) for value in raw_diff["allowed_artist_ids"]}
        classification_records = [
            record
            for record in candidate_raw
            if record.get("artist_id") is not None and int(record["artist_id"]) in allowed_artist_ids
        ]
        semantic = classify_unmatched_records(classification_records)
        append_matched_color_diffs(semantic, raw_diff.get("matched_pairs", []))
        return {
            "backend": "d3-svg",
            "candidate": candidate_result,
            "baseline": baseline_result,
            "candidate_raw": candidate_raw,
            "baseline_raw": baseline_raw,
            "raw_diff": to_jsonable_raw_diff(raw_diff),
            "classification_input_records": classification_records,
            "diffed_semantic": semantic,
        }
    finally:
        if owns_runtime:
            runtime.__exit__(None, None, None)


__all__ = ["extract_diffed_d3_bundle"]
