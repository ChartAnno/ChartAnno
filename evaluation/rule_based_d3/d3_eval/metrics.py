"""Rule-based metrics for the D3 backend."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "evaluation" / "rule_based" / "annotation_eval").is_dir()
)
MPL_RULE_BASED_ROOT = REPO_ROOT / "evaluation" / "rule_based"
if str(MPL_RULE_BASED_ROOT) not in sys.path:
    sys.path.insert(0, str(MPL_RULE_BASED_ROOT))


def compute_rule_based_metrics(
    gt_semantic: dict[str, list],
    predicted_semantic: dict[str, list],
    *,
    baseline_raw: list[dict],
    candidate_raw: list[dict],
    baseline_canvas: dict | None = None,
    candidate_canvas: dict | None = None,
) -> dict:
    from annotation_eval.extraction.raw_diff import build_raw_diff_result
    from annotation_eval.metrics.annotation_matching import _compute_file_jaccard
    from annotation_eval.metrics.color_matching import _score_one_file

    return {
        "annotation_matching": _compute_file_jaccard(gt_semantic, predicted_semantic),
        "color_matching": _score_one_file(gt_semantic, predicted_semantic),
        "chart_fidelity": compute_chart_fidelity(
            baseline_raw,
            candidate_raw,
            baseline_canvas=baseline_canvas,
            candidate_canvas=candidate_canvas,
        ),
    }

FIDELITY_IGNORED_ROLES = {"axis-decoration", "legend-key", "indicator-marker"}
FIDELITY_PROTECTED_KINDS = {"axes", "line", "figure_line", "patch", "collection"}


def _protected_records(records: list[dict]) -> list[dict]:
    return [
        record
        for record in records
        if (record.get("kind") in FIDELITY_PROTECTED_KINDS or record.get("semantic_role") in FIDELITY_PROTECTED_KINDS)
        if str(record.get("semantic_role") or "") not in FIDELITY_IGNORED_ROLES
    ]


def _canvas_ratio(canvas: dict | None) -> float | None:
    if not isinstance(canvas, dict):
        return None
    try:
        width = float(canvas["width"])
        height = float(canvas["height"])
    except (KeyError, TypeError, ValueError):
        return None
    return width / height if height > 0 else None


def compute_chart_fidelity(
    baseline_raw: list[dict],
    candidate_raw: list[dict],
    *,
    baseline_canvas: dict | None = None,
    candidate_canvas: dict | None = None,
) -> dict:
    """Measure whether protected baseline objects survive in the candidate."""
    from annotation_eval.extraction.raw_diff import build_raw_diff_result

    protected = _protected_records(baseline_raw)
    reverse_diff = build_raw_diff_result(protected, candidate_raw, ignore_visual_color=True)
    missing = _protected_records(reverse_diff.get("unmatched_records", []))
    protected_count = len(protected)
    missing_count = len(missing)
    coverage = (
        1.0
        if protected_count == 0
        else max(0.0, (protected_count - missing_count) / protected_count)
    )

    baseline_ratio = _canvas_ratio(baseline_canvas)
    candidate_ratio = _canvas_ratio(candidate_canvas)
    ratio_preserved = (
        True
        if baseline_ratio is None or candidate_ratio is None
        else abs(baseline_ratio - candidate_ratio) <= 0.01
    )

    passed = missing_count == 0 and ratio_preserved
    return {
        "chart_fidelity": 1 if passed else 0,
        "protected_element_coverage": round(coverage, 6),
        "protected_count": protected_count,
        "preserved_count": protected_count - missing_count,
        "missing_count": missing_count,
        "canvas_ratio_preserved": ratio_preserved,
        "baseline_canvas_ratio": None if baseline_ratio is None else round(baseline_ratio, 6),
        "candidate_canvas_ratio": None if candidate_ratio is None else round(candidate_ratio, 6),
        "missing_records": missing,
    }



__all__ = ["compute_chart_fidelity", "compute_rule_based_metrics"]
