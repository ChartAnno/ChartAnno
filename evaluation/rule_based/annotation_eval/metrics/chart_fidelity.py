#!/usr/bin/env python3
"""Score chart fidelity by comparing protected data against removed baselines.

Protected elements:
- data-bearing artists already present in dataset_code_removed
- figure ratio already present in dataset_code_removed
- explicit aspect constraints already present in dataset_code_removed
- axes layout already present in dataset_code_removed

Ignored by design:
- added annotations or auxiliary elements
- color/style changes
- extra axes or extra artists, as long as protected baseline elements remain intact

Outputs:
- outputs/analysis/chart_fidelity/<Category>/<ChartID>/<LLM|VLM>/<stem>_chart_fidelity.json
- outputs/analysis/chart_fidelity/chart_fidelity_all.csv
- outputs/analysis/chart_fidelity/_summary.json
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import matplotlib as mpl
import matplotlib.collections as mcollections
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = (
    Path(os.path.abspath(sys.argv[0])).parents[2]
    if sys.argv and sys.argv[0]
    else Path(__file__).resolve().parents[2]
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from annotation_eval.config import get_path

DEFAULT_REMOVED_ROOT = REPO_ROOT / "dataset_code_removed"
DEFAULT_TEST_ROOT = get_path(REPO_ROOT, "test_code_dir", "test_code")
DEFAULT_OUT_ROOT = REPO_ROOT / "outputs" / "analysis" / "chart_fidelity"

LEGACY_ROOT = os.environ.get("ANNOTATION_EVAL_LEGACY_ROOT", "")
MODEL_NAMES = ("LLM", "VLM")
MODEL_DIR_ALIASES = {
    "code": "LLM",
    "code+image": "VLM",
}
LAYERS = ("intent", "operation", "implementation")

FIGURE_RATIO_ATOL = 0.01
AXIS_EDGE_ATOL = 0.05
AXIS_AREA_RATIO_MIN = 0.85
AXIS_IOU_MIN = 0.85
AXIS_CENTER_DELTA_MAX = 0.05
AXIS_WIDTH_RATIO_MIN = 0.80
AXIS_HEIGHT_RATIO_MIN = 0.80
AXIS_SINGLE_AXIS_PRESERVED_RATIO_MIN = 0.95
AXIS_UNIFORM_RATIO_GAP_MAX = 0.05
POINT_PREC = 6
VERTEX_RESAMPLE_POINTS = 256
VALUE_RTOL = 1e-4
VALUE_ATOL = 1e-6


@dataclass(frozen=True)
class Job:
    category: str
    chart_id: str
    source_model: str
    layer: str
    removed_path: Path
    candidate_path: Path
    out_path: Path


def _round_float(value: float, precision: int = POINT_PREC) -> float:
    if math.isnan(value) or math.isinf(value):
        return value
    return round(float(value), precision)


def _normalize_scalar(value: Any) -> float | str | None:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if isinstance(value, np.datetime64):
        value = value.astype("datetime64[ms]").astype(object)
    if isinstance(value, (dt.datetime, dt.date)):
        return float(mdates.date2num(value))
    if isinstance(value, str):
        return value
    try:
        return float(value)
    except Exception:
        return str(value)


def _normalize_array(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if np.issubdtype(arr.dtype, np.datetime64):
        arr = mdates.date2num(arr.astype("datetime64[ms]").astype(object))
        return np.asarray(arr, dtype=float)

    normalized: list[float] = []
    for item in arr.tolist():
        item_norm = _normalize_scalar(item)
        if isinstance(item_norm, str) or item_norm is None:
            raise TypeError(f"Non-numeric value encountered: {item!r}")
        normalized.append(float(item_norm))
    return np.asarray(normalized, dtype=float)


def _allclose(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    return bool(np.allclose(a, b, rtol=VALUE_RTOL, atol=VALUE_ATOL, equal_nan=True))


def _bounds_area(bounds: tuple[float, float, float, float]) -> float:
    _, _, w, h = bounds
    return max(float(w), 0.0) * max(float(h), 0.0)


def _bounds_area_ratio(base: tuple[float, float, float, float], cand: tuple[float, float, float, float]) -> float:
    base_area = _bounds_area(base)
    cand_area = _bounds_area(cand)
    max_area = max(base_area, cand_area)
    if max_area <= VALUE_ATOL:
        return 1.0
    return min(base_area, cand_area) / max_area


def _bounds_iou(base: tuple[float, float, float, float], cand: tuple[float, float, float, float]) -> float:
    bx, by, bw, bh = base
    cx, cy, cw, ch = cand
    bx2, by2 = float(bx) + float(bw), float(by) + float(bh)
    cx2, cy2 = float(cx) + float(cw), float(cy) + float(ch)
    inter_w = max(0.0, min(bx2, cx2) - max(float(bx), float(cx)))
    inter_h = max(0.0, min(by2, cy2) - max(float(by), float(cy)))
    inter = inter_w * inter_h
    union = _bounds_area(base) + _bounds_area(cand) - inter
    if union <= VALUE_ATOL:
        return 1.0
    return inter / union


def _bounds_max_delta(base: tuple[float, float, float, float], cand: tuple[float, float, float, float]) -> float:
    return max(abs(float(b) - float(c)) for b, c in zip(base, cand))


def _bounds_center_delta(base: tuple[float, float, float, float], cand: tuple[float, float, float, float]) -> float:
    bx, by, bw, bh = base
    cx, cy, cw, ch = cand
    base_center = (float(bx) + float(bw) / 2.0, float(by) + float(bh) / 2.0)
    cand_center = (float(cx) + float(cw) / 2.0, float(cy) + float(ch) / 2.0)
    return max(abs(base_center[0] - cand_center[0]), abs(base_center[1] - cand_center[1]))


def _bounds_size_ratios(
    base: tuple[float, float, float, float], cand: tuple[float, float, float, float]
) -> tuple[float, float]:
    _, _, bw, bh = base
    _, _, cw, ch = cand
    width_max = max(abs(float(bw)), abs(float(cw)))
    height_max = max(abs(float(bh)), abs(float(ch)))
    width_ratio = 1.0 if width_max <= VALUE_ATOL else min(abs(float(bw)), abs(float(cw))) / width_max
    height_ratio = 1.0 if height_max <= VALUE_ATOL else min(abs(float(bh)), abs(float(ch))) / height_max
    return width_ratio, height_ratio


def _bounds_layout_matches(
    base: tuple[float, float, float, float], cand: tuple[float, float, float, float]
) -> tuple[bool, float, float, float, float, float, float]:
    max_delta = _bounds_max_delta(base, cand)
    area_ratio = _bounds_area_ratio(base, cand)
    iou = _bounds_iou(base, cand)
    center_delta = _bounds_center_delta(base, cand)
    width_ratio, height_ratio = _bounds_size_ratios(base, cand)
    small_adjustment = max_delta <= AXIS_EDGE_ATOL and area_ratio >= AXIS_AREA_RATIO_MIN
    high_overlap = iou >= AXIS_IOU_MIN and area_ratio >= AXIS_AREA_RATIO_MIN
    centered_rescale = (
        center_delta <= AXIS_CENTER_DELTA_MAX
        and width_ratio >= AXIS_WIDTH_RATIO_MIN
        and height_ratio >= AXIS_HEIGHT_RATIO_MIN
        and abs(width_ratio - height_ratio) <= AXIS_UNIFORM_RATIO_GAP_MAX
    )
    single_axis_rescale = center_delta <= AXIS_CENTER_DELTA_MAX and (
        (
            width_ratio >= AXIS_SINGLE_AXIS_PRESERVED_RATIO_MIN
            and height_ratio >= AXIS_HEIGHT_RATIO_MIN
        )
        or (
            height_ratio >= AXIS_SINGLE_AXIS_PRESERVED_RATIO_MIN
            and width_ratio >= AXIS_WIDTH_RATIO_MIN
        )
    )
    return (
        small_adjustment or high_overlap or centered_rescale or single_axis_rescale,
        max_delta,
        area_ratio,
        iou,
        center_delta,
        width_ratio,
        height_ratio,
    )


def _format_bounds(bounds: tuple[float, float, float, float]) -> str:
    return ",".join(f"{float(v):.4f}" for v in bounds)


def _path_arc_resample(points: np.ndarray, n: int = VERTEX_RESAMPLE_POINTS) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Expected Nx2 points.")
    if len(points) == 0:
        return np.zeros((0, 2), dtype=float)
    if len(points) == 1:
        return np.repeat(points.astype(float), n, axis=0)

    deltas = np.diff(points.astype(float), axis=0)
    seg = np.linalg.norm(deltas, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= 0:
        return np.repeat(points[:1].astype(float), n, axis=0)

    sample = np.linspace(0.0, total, n)
    x = np.interp(sample, cum, points[:, 0].astype(float))
    y = np.interp(sample, cum, points[:, 1].astype(float))
    return np.column_stack([x, y])


def _resampled_signature(points: np.ndarray, n: int = VERTEX_RESAMPLE_POINTS) -> list[list[float]]:
    resampled = _path_arc_resample(points, n=n)
    return [[_round_float(x), _round_float(y)] for x, y in resampled]


def _rect_signature(rect: mpatches.Rectangle) -> str:
    payload = {
        "x": _round_float(rect.get_x()),
        "y": _round_float(rect.get_y()),
        "w": _round_float(rect.get_width()),
        "h": _round_float(rect.get_height()),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _wedge_signature(wedge: mpatches.Wedge) -> str:
    payload = {
        "r": _round_float(wedge.r),
        "theta1": _round_float(wedge.theta1),
        "theta2": _round_float(wedge.theta2),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _point_signature(x: float, y: float) -> str:
    payload = {
        "x": _round_float(x),
        "y": _round_float(y),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _marker_is_none(marker: Any) -> bool:
    return str(marker).strip().lower() in {"", "none", "null", " "}


def _linestyle_is_none(style: Any) -> bool:
    return str(style).strip().lower() in {"", "none", "null", " "}


def _is_data_line(ax, line) -> bool:
    if not line.get_visible():
        return False
    if not line.get_clip_on():
        return False

    try:
        x = _normalize_array(line.get_xdata(orig=False))
        y = _normalize_array(line.get_ydata(orig=False))
    except Exception:
        return False
    if len(x) != len(y) or len(x) == 0:
        return False

    if len(x) <= 2 and _marker_is_none(line.get_marker()):
        return False

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    in_view = (
        np.any((x >= min(xlim) - VALUE_ATOL) & (x <= max(xlim) + VALUE_ATOL))
        and np.any((y >= min(ylim) - VALUE_ATOL) & (y <= max(ylim) + VALUE_ATOL))
    )
    if not in_view and _linestyle_is_none(line.get_linestyle()):
        return False

    return True


def _line_descriptor(line) -> dict[str, Any] | None:
    try:
        x = _normalize_array(line.get_xdata(orig=False))
        y = _normalize_array(line.get_ydata(orig=False))
    except Exception:
        return None
    if len(x) != len(y) or len(x) == 0:
        return None

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) == 0:
        return None

    points = np.column_stack([x, y])
    monotonic = bool(np.all(np.diff(x) >= 0) or np.all(np.diff(x) <= 0))
    return {
        "points": points,
        "monotonic_x": monotonic,
        "x_range": (_round_float(float(np.min(x))), _round_float(float(np.max(x)))),
        "signature": _resampled_signature(points),
    }


def _poly_descriptor(path) -> dict[str, Any] | None:
    vertices = np.asarray(getattr(path, "vertices", None))
    if vertices.size == 0:
        return None
    if vertices.ndim != 2 or vertices.shape[1] != 2:
        return None
    mask = np.isfinite(vertices[:, 0]) & np.isfinite(vertices[:, 1])
    vertices = vertices[mask]
    if len(vertices) == 0:
        return None
    return {
        "points": vertices.astype(float),
        "signature": _resampled_signature(vertices.astype(float)),
    }


def _line_match(base: dict[str, Any], cand: dict[str, Any]) -> bool:
    base_points = base["points"]
    cand_points = cand["points"]

    if len(base_points) == 1 and len(cand_points) == 1:
        return _allclose(base_points, cand_points)

    if base["monotonic_x"] and cand["monotonic_x"]:
        bx = base_points[:, 0]
        by = base_points[:, 1]
        cx = cand_points[:, 0]
        cy = cand_points[:, 1]

        if cx[0] > bx[0] + VALUE_ATOL or cx[-1] < bx[-1] - VALUE_ATOL:
            return False

        order = np.argsort(cx)
        cx = cx[order]
        cy = cy[order]
        uniq_x, uniq_idx = np.unique(cx, return_index=True)
        uniq_y = cy[uniq_idx]
        if len(uniq_x) == 1:
            return False

        interp_y = np.interp(bx, uniq_x, uniq_y)
        return bool(np.allclose(by, interp_y, rtol=VALUE_RTOL, atol=VALUE_ATOL))

    base_sig = np.asarray(base["signature"], dtype=float)
    cand_sig = np.asarray(cand["signature"], dtype=float)
    return _allclose(base_sig, cand_sig)


def _line_cover_mask(base: dict[str, Any], cand: dict[str, Any]) -> np.ndarray:
    base_points = base["points"]
    cand_points = cand["points"]
    covered = np.zeros(len(base_points), dtype=bool)

    if not base["monotonic_x"] or not cand["monotonic_x"]:
        return covered

    bx = base_points[:, 0]
    by = base_points[:, 1]
    cx = cand_points[:, 0]
    cy = cand_points[:, 1]

    order = np.argsort(cx)
    cx = cx[order]
    cy = cy[order]
    uniq_x, uniq_idx = np.unique(cx, return_index=True)
    uniq_y = cy[uniq_idx]
    if len(uniq_x) == 0:
        return covered

    if len(uniq_x) == 1:
        mask = np.isclose(bx, uniq_x[0], rtol=VALUE_RTOL, atol=VALUE_ATOL)
        if np.any(mask):
            covered[mask] = np.isclose(by[mask], uniq_y[0], rtol=VALUE_RTOL, atol=VALUE_ATOL)
        return covered

    in_range = (bx >= uniq_x[0] - VALUE_ATOL) & (bx <= uniq_x[-1] + VALUE_ATOL)
    if not np.any(in_range):
        return covered

    interp_y = np.interp(bx[in_range], uniq_x, uniq_y)
    covered[in_range] = np.isclose(by[in_range], interp_y, rtol=VALUE_RTOL, atol=VALUE_ATOL)
    return covered


def _line_covered_by_candidates(base: dict[str, Any], cand_items: list[dict[str, Any]]) -> bool:
    for cand in cand_items:
        if _line_match(base, cand):
            return True

    if not base["monotonic_x"]:
        return False

    covered = np.zeros(len(base["points"]), dtype=bool)
    for cand in cand_items:
        covered |= _line_cover_mask(base, cand)
        if bool(np.all(covered)):
            return True
    return False


def _poly_match(base: dict[str, Any], cand: dict[str, Any]) -> bool:
    base_sig = np.asarray(base["signature"], dtype=float)
    cand_sig = np.asarray(cand["signature"], dtype=float)
    return _allclose(base_sig, cand_sig)


def _counter_is_subset(base: Counter[str], cand: Counter[str]) -> tuple[bool, list[str]]:
    missing: list[str] = []
    for key, need in base.items():
        have = cand.get(key, 0)
        if have < need:
            missing.append(f"{key} x{need - have}")
    return (len(missing) == 0, missing)


def _match_descriptors(
    base_items: list[dict[str, Any]],
    cand_items: list[dict[str, Any]],
    matcher,
) -> tuple[bool, str | None]:
    used = [False] * len(cand_items)
    for base in base_items:
        found = False
        for idx, cand in enumerate(cand_items):
            if used[idx]:
                continue
            if matcher(base, cand):
                used[idx] = True
                found = True
                break
        if not found:
            return False, "descriptor_not_matched"
    return True, None


def _normalize_aspect_value(value: Any) -> float | str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        return None if text == "auto" else text
    try:
        return _round_float(float(value))
    except Exception:
        return str(value)


def _extract_axis_spec(ax) -> dict[str, Any]:
    box_aspect = None
    try:
        raw_box_aspect = ax.get_box_aspect()
        if raw_box_aspect is not None:
            box_aspect = _round_float(float(raw_box_aspect))
    except Exception:
        box_aspect = None

    axis_bounds = tuple(_round_float(float(v), 4) for v in ax.get_position().bounds)

    rectangles: list[str] = []
    wedges: list[str] = []
    for patch in ax.patches:
        if patch is ax.patch or not patch.get_visible():
            continue
        if isinstance(patch, mpatches.Rectangle):
            rectangles.append(_rect_signature(patch))
        elif isinstance(patch, mpatches.Wedge):
            wedges.append(_wedge_signature(patch))

    scatter_points: list[str] = []
    for collection in ax.collections:
        if not collection.get_visible():
            continue
        if isinstance(collection, mcollections.PathCollection):
            try:
                offsets = np.asarray(collection.get_offsets(), dtype=float)
            except Exception:
                continue
            if offsets.ndim != 2 or offsets.shape[1] != 2:
                continue
            for x, y in offsets:
                if np.isfinite(x) and np.isfinite(y):
                    scatter_points.append(_point_signature(float(x), float(y)))

    line_descriptors: list[dict[str, Any]] = []
    for line in ax.lines:
        if not _is_data_line(ax, line):
            continue
        descriptor = _line_descriptor(line)
        if descriptor is not None:
            line_descriptors.append(descriptor)

    return {
        "explicit_aspect": _normalize_aspect_value(ax.get_aspect()),
        "box_aspect": box_aspect,
        "axis_bounds": axis_bounds,
        "rectangles": Counter(rectangles),
        "wedges": Counter(wedges),
        "scatter_points": Counter(scatter_points),
        "lines": line_descriptors,
    }


def _extract_figure_spec(fig) -> dict[str, Any]:
    try:
        fig.canvas.draw()
    except Exception:
        pass

    size = fig.get_size_inches()
    figure_ratio = None
    if len(size) >= 2 and float(size[1]) != 0.0:
        figure_ratio = float(size[0]) / float(size[1])

    return {
        "figure_ratio": figure_ratio,
        "axes": [_extract_axis_spec(ax) for ax in fig.axes],
    }


def _axis_matches(base: dict[str, Any], cand: dict[str, Any], reasons: list[str]) -> bool:
    if base["explicit_aspect"] is not None and cand["explicit_aspect"] != base["explicit_aspect"]:
        reasons.append(f"aspect_mismatch:{base['explicit_aspect']}!={cand['explicit_aspect']}")
        return False

    if base["box_aspect"] is not None and cand["box_aspect"] != base["box_aspect"]:
        reasons.append(f"box_aspect_mismatch:{base['box_aspect']}!={cand['box_aspect']}")
        return False

    layout_ok, max_delta, area_ratio, iou, center_delta, width_ratio, height_ratio = _bounds_layout_matches(
        base["axis_bounds"], cand["axis_bounds"]
    )
    if not layout_ok:
        reasons.append(
            "axis_position_mismatch:"
            f"iou={iou:.3f},area_ratio={area_ratio:.3f},max_delta={max_delta:.4f},"
            f"center_delta={center_delta:.4f},width_ratio={width_ratio:.3f},height_ratio={height_ratio:.3f},"
            f"{_format_bounds(base['axis_bounds'])}!={_format_bounds(cand['axis_bounds'])}"
        )
        return False

    for key in ("rectangles", "wedges", "scatter_points"):
        ok, missing = _counter_is_subset(base[key], cand[key])
        if not ok:
            reasons.append(f"{key}_missing:{'|'.join(missing[:5])}")
            return False

    for base_line in base["lines"]:
        if not _line_covered_by_candidates(base_line, cand["lines"]):
            reasons.append("line_mismatch")
            return False

    return True


def _compare_specs(base: dict[str, Any], cand: dict[str, Any]) -> tuple[bool, dict[str, Any], list[str]]:
    reasons: list[str] = []
    checks = {
        "figure_ratio": True,
        "axes": True,
    }

    if base["figure_ratio"] is not None:
        cand_ratio = cand["figure_ratio"]
        if cand_ratio is None or abs(base["figure_ratio"] - cand_ratio) > FIGURE_RATIO_ATOL:
            checks["figure_ratio"] = False
            reasons.append(
                f"figure_ratio_mismatch:{base['figure_ratio']:.4f}!={cand_ratio if cand_ratio is None else f'{cand_ratio:.4f}'}"
            )

    used = [False] * len(cand["axes"])
    for base_axis in base["axes"]:
        matched = False
        best_reason = "protected_axis_not_matched"
        best_ratio_delta = float("inf")
        for idx, cand_axis in enumerate(cand["axes"]):
            if used[idx]:
                continue
            axis_reasons: list[str] = []
            if _axis_matches(base_axis, cand_axis, axis_reasons):
                used[idx] = True
                matched = True
                break
            current_delta = _bounds_center_delta(base_axis["axis_bounds"], cand_axis["axis_bounds"])
            if axis_reasons and current_delta < best_ratio_delta:
                best_ratio_delta = current_delta
                best_reason = axis_reasons[0]
        if not matched:
            checks["axes"] = False
            reasons.append(best_reason)
            break

    passed = all(checks.values())
    return passed, checks, reasons


def _read_source(path: Path, project_root: Path) -> str:
    source = path.read_text(encoding="utf-8")
    if LEGACY_ROOT and LEGACY_ROOT in source:
        source = source.replace(LEGACY_ROOT, str(project_root))
    return source


def _exec_source_get_figure(source_code: str, file_path: Path):
    module_name = f"cfp_{file_path.stem}_{abs(hash(str(file_path)))}"
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(file_path)
    module.__dict__["__name__"] = "__main__"

    seed_injection = """
import numpy as np
import random
np.random.seed(42)
random.seed(42)
"""
    modified_source = seed_injection + source_code

    with patch("matplotlib.pyplot.show"), patch(
        "matplotlib.pyplot.close", lambda *a, **k: None
    ), patch("matplotlib.pyplot.savefig", lambda *a, **k: None), patch(
        "matplotlib.figure.Figure.savefig", lambda *a, **k: None
    ):
        exec(compile(modified_source, str(file_path), "exec"), module.__dict__)

    if len(plt.get_fignums()) == 0:
        return None
    return plt.gcf()

def _load_figure_spec(path: Path, project_root: Path) -> dict[str, Any]:
    real_close = plt.close
    try:
        mpl.rcdefaults()
        real_close("all")
        source = _read_source(path, project_root)
        fig = _exec_source_get_figure(source, path)
        if fig is None:
            raise RuntimeError("No matplotlib figure produced.")
        return _extract_figure_spec(fig)
    finally:
        real_close("all")


def _layer_from_stem(stem: str) -> str:
    lower = stem.lower()
    if lower.endswith("_task"):
        return "intent"
    for layer in LAYERS:
        if lower.endswith(f"_{layer}"):
            return layer
    raise ValueError(f"Cannot infer layer from stem: {stem}")


def _iter_jobs(
    removed_root: Path,
    test_root: Path,
    out_root: Path,
    only_category: str = "",
    only_model: str = "",
    only_layer: str = "",
) -> list[Job]:
    jobs: list[Job] = []
    for path in sorted(test_root.rglob("*.py")):
        parts = path.relative_to(test_root).parts
        if len(parts) != 4:
            continue
        category, chart_id, model_dir, filename = parts
        model = MODEL_DIR_ALIASES.get(model_dir)
        if model not in MODEL_NAMES:
            continue
        layer = _layer_from_stem(path.stem)
        if only_category and category != only_category:
            continue
        if only_model and model != only_model:
            continue
        if only_layer and layer != only_layer:
            continue

        removed_path = removed_root / category / f"{chart_id}.py"
        out_path = out_root / "per_chart" / category / chart_id / model / f"{chart_id}_{layer}.json"
        jobs.append(
            Job(
                category=category,
                chart_id=chart_id,
                source_model=model,
                layer=layer,
                removed_path=removed_path,
                candidate_path=path,
                out_path=out_path,
            )
        )
    return jobs


def _write_job_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score chart fidelity against dataset_code_removed."
    )
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--removed-root", type=Path, default=DEFAULT_REMOVED_ROOT)
    parser.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--only-category", default="", help="Filter by category name.")
    parser.add_argument("--only-model", default="", help="Filter by model name: LLM or VLM.")
    parser.add_argument("--only-layer", default="", help="Filter by layer name.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip jobs whose output JSON already exists.",
    )
    return parser


def main() -> int:
    args = build_cli().parse_args()

    only_model = args.only_model.upper().strip()
    if only_model and only_model not in MODEL_NAMES:
        raise SystemExit(f"Invalid --only-model: {args.only_model}")
    only_layer = args.only_layer.lower().strip()
    if only_layer and only_layer not in LAYERS:
        raise SystemExit(f"Invalid --only-layer: {args.only_layer}")

    jobs = _iter_jobs(
        removed_root=args.removed_root,
        test_root=args.test_root,
        out_root=args.out_root,
        only_category=args.only_category.strip(),
        only_model=only_model,
        only_layer=only_layer,
    )
    if not jobs:
        print("No jobs found.")
        args.out_root.mkdir(parents=True, exist_ok=True)
        csv_path = args.out_root / "chart_fidelity_all.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["category", "chart_id", "model", "layer", "chart_fidelity", "parse_ok", "status", "reasons"],
            )
            writer.writeheader()
        summary_path = args.out_root / "_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "jobs": 0,
                    "passed": 0,
                    "failed": 0,
                    "errors": 0,
                    "csv": str(csv_path),
                    "out_root": str(args.out_root),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"Wrote: {csv_path}")
        print(f"Wrote: {summary_path}")
        return 0

    args.out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    passed = 0
    failed = 0
    errors = 0

    for idx, job in enumerate(jobs, start=1):
        print(f"[{idx}/{len(jobs)}] {job.category}/{job.chart_id}/{job.source_model}/{job.layer}")

        if args.skip_existing and job.out_path.exists():
            try:
                payload = json.loads(job.out_path.read_text(encoding="utf-8"))
                item = ((payload.get("result") or {}).get("results") or [{}])[0]
                cf = item.get("chart_fidelity", item.get("cf"))
                rows.append(
                    {
                        "category": job.category,
                        "chart_id": job.chart_id,
                        "model": job.source_model,
                        "layer": job.layer,
                        "chart_fidelity": cf,
                        "parse_ok": payload.get("parse_ok", False),
                        "status": "skipped_existing",
                        "reasons": "|".join(item.get("reasons") or []),
                    }
                )
                continue
            except Exception:
                pass

        payload: dict[str, Any]
        try:
            if not job.removed_path.exists():
                raise FileNotFoundError(f"Removed baseline missing: {job.removed_path}")

            base_spec = _load_figure_spec(job.removed_path, args.project_root)
            cand_spec = _load_figure_spec(job.candidate_path, args.project_root)
            matched, checks, reasons = _compare_specs(base_spec, cand_spec)
            cf = 1 if matched else 0

            if matched:
                passed += 1
            else:
                failed += 1

            payload = {
                "meta": {
                    "category": job.category,
                    "chart_id": job.chart_id,
                    "source_model": job.source_model,
                    "layer": job.layer,
                    "removed_path": str(job.removed_path),
                    "candidate_path": str(job.candidate_path),
                    "source": "chart_fidelity.py",
                },
                "parse_ok": True,
                "result": {
                    "results": [
                        {
                            "cf": cf,
                            "chart_fidelity": cf,
                            "checks": checks,
                            "reasons": reasons,
                            "baseline_axes": len(base_spec["axes"]),
                            "candidate_axes": len(cand_spec["axes"]),
                        }
                    ]
                },
            }
            status = "ok"
        except Exception as exc:
            errors += 1
            payload = {
                "meta": {
                    "category": job.category,
                    "chart_id": job.chart_id,
                    "source_model": job.source_model,
                    "layer": job.layer,
                    "removed_path": str(job.removed_path),
                    "candidate_path": str(job.candidate_path),
                    "source": "chart_fidelity.py",
                },
                "parse_ok": False,
                "error": str(exc),
                "result": {
                    "results": [
                        {
                            "cf": 0,
                            "chart_fidelity": 0,
                            "checks": {
                                "figure_ratio": False,
                                "figure_texts": False,
                                "axes": False,
                            },
                            "reasons": [str(exc)],
                        }
                    ]
                },
            }
            status = "error"

        _write_job_result(job.out_path, payload)
        item = ((payload.get("result") or {}).get("results") or [{}])[0]
        rows.append(
            {
                "category": job.category,
                "chart_id": job.chart_id,
                "model": job.source_model,
                "layer": job.layer,
                "chart_fidelity": item.get("chart_fidelity", item.get("cf")),
                "parse_ok": payload.get("parse_ok", False),
                "status": status,
                "reasons": "|".join(item.get("reasons") or []),
            }
        )

    csv_path = args.out_root / "chart_fidelity_all.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["category", "chart_id", "model", "layer", "chart_fidelity", "parse_ok", "status", "reasons"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "jobs": len(jobs),
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "csv": str(csv_path),
        "out_root": str(args.out_root),
    }
    summary_path = args.out_root / "_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {summary_path}")
    print(f"Jobs: {len(jobs)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Errors: {errors}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
