"""Shared subtraction and deduplication utilities for annotation JSON files."""

import json
import math
import os
import re
from collections import Counter
from typing import Any

from annotation_eval.extraction.annotation_schema import FEATURE_ORDER

BBOX_CENTER_TOL = 0.02
BBOX_SIZE_TOL = 0.05
BBOX_IOU_TOL = 0.45
BBOX_BASE_COVERAGE_TOL = 0.75
NUMERIC_TOL = 1e-3
DEDUPE_BBOX_IOU_TOL = 0.90
DEDUPE_CENTER_TOL = 0.01
DEDUPE_SIZE_TOL = 0.02

# Line-like bbox IoU dilation.
# BBox is in normalized figure coordinates, so 4px is converted with a reference canvas.
REFERENCE_CANVAS_PX = 1200.0
LINE_IOU_DILATE_PX = 4.0
LINE_IOU_DILATE_NORM = LINE_IOU_DILATE_PX / REFERENCE_CANVAS_PX
LINE_IOU_CATEGORIES = {"2_connector", "6_indicator"}

CATEGORY_BBOX_TOLERANCE = {
    "1_enclosure": {"center": 0.10, "size": 0.20, "iou": 0.20},
    "2_connector": {"center": 0.05, "size": 0.10, "iou": 0.30},
    "3_text": {"center": 0.08, "size": 0.20, "iou": 0.10},
    "4_glyph": {"center": 0.04, "size": 0.08, "iou": 0.35},
    "6_indicator": {"center": 0.08, "size": 0.15, "iou": 0.25},
    "7_geometric": {"center": 0.10, "size": 0.20, "iou": 0.25},
}

CATEGORY_IDENTITY_KEYS = {
    "1_enclosure": ("src", "type", "marker"),
    "2_connector": ("src", "type"),
    "4_glyph": ("src", "marker", "type"),
    "5_color": ("src", "type"),
    "6_indicator": ("src", "type", "linestyle", "is_full_span"),
    "7_geometric": ("src", "type"),
}

DEDUPE_EXCLUDE_CATEGORIES = {"3_text", "5_color"}
STAGE_FILE_PATTERN = re.compile(r"^([A-Za-z]+_\d+)_(task|intent|operation|implementation)$")


def _normalize_number(x: Any) -> Any:
    if isinstance(x, float):
        if math.isclose(x, 0.0, abs_tol=1e-12):
            return 0.0
        return round(x, 6)
    return x


def _canonicalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _canonicalize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_canonicalize(v) for v in obj]
    return _normalize_number(obj)


def generate_fingerprint(item_dict: dict[str, Any]) -> str:
    return json.dumps(_canonicalize(item_dict), sort_keys=True, ensure_ascii=False)


def order_feature_dict(data: dict[str, Any]) -> dict[str, Any]:
    ordered = {}
    for key in FEATURE_ORDER:
        ordered[key] = data.get(key, [])
    for key in sorted(k for k in data.keys() if k not in FEATURE_ORDER):
        ordered[key] = data[key]
    return ordered


def _text_content_key(item: Any) -> str | None:
    if not isinstance(item, dict):
        return None
    content = item.get("content")
    if not isinstance(content, str):
        return None
    return " ".join(content.split())


def _to_float(v: Any) -> float | None:
    try:
        return float(v)
    except Exception:
        return None


def _parse_bbox(item: Any) -> tuple[float, float, float, float] | None:
    bbox = item.get("bbox") if isinstance(item, dict) else item
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        return None
    vals = [_to_float(x) for x in bbox]
    if any(v is None for v in vals):
        return None
    x, y, w, h = vals
    return x, y, max(0.0, w), max(0.0, h)


def _parse_bbox_field(item: Any, field_name: str) -> tuple[float, float, float, float] | None:
    if not isinstance(item, dict):
        return None
    bbox = item.get(field_name)
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        return None
    vals = [_to_float(x) for x in bbox]
    if any(v is None for v in vals):
        return None
    x, y, w, h = vals
    return x, y, max(0.0, w), max(0.0, h)


def _has_bbox_field(item: Any, field_name: str) -> bool:
    return _parse_bbox_field(item, field_name) is not None


def _normalize_color(color: Any) -> Any:
    if isinstance(color, list):
        return tuple(_normalize_number(float(c)) if isinstance(c, (int, float)) else c for c in color)
    if isinstance(color, tuple):
        return tuple(_normalize_number(float(c)) if isinstance(c, (int, float)) else c for c in color)
    return color


def _bbox_iou(ab: tuple[float, float, float, float], bb: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = ab
    bx, by, bw, bh = bb
    ax1, ay1 = ax + aw, ay + ah
    bx1, by1 = bx + bw, by + bh
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _is_line_like_bbox(box: tuple[float, float, float, float]) -> bool:
    _, _, w, h = box
    return (w <= 1e-9) or (h <= 1e-9)


def _dilate_bbox_for_iou(
    box: tuple[float, float, float, float],
    *,
    category: str | None = None,
) -> tuple[float, float, float, float]:
    if category not in LINE_IOU_CATEGORIES:
        return box
    if not _is_line_like_bbox(box):
        return box

    x, y, w, h = box
    d = float(LINE_IOU_DILATE_NORM)
    x2 = x - d
    y2 = y - d
    w2 = max(0.0, w + 2.0 * d)
    h2 = max(0.0, h + 2.0 * d)
    return x2, y2, w2, h2


def _bbox_iou_for_match(
    ab: tuple[float, float, float, float],
    bb: tuple[float, float, float, float],
    *,
    category: str | None = None,
) -> float:
    a2 = _dilate_bbox_for_iou(ab, category=category)
    b2 = _dilate_bbox_for_iou(bb, category=category)
    return _bbox_iou(a2, b2)


def _bbox_intersection_area(
    ab: tuple[float, float, float, float], bb: tuple[float, float, float, float]
) -> float:
    ax, ay, aw, ah = ab
    bx, by, bw, bh = bb
    ax1, ay1 = ax + aw, ay + ah
    bx1, by1 = bx + bw, by + bh
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    return iw * ih


def _bbox_duplicate_for_dedupe(a: Any, b: Any, category: str | None = None) -> bool:
    ab = _parse_bbox(a)
    bb = _parse_bbox(b)
    if ab is None or bb is None:
        return False

    _, _, aw, ah = ab
    _, _, bw, bh = bb
    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)

    if area_a > 0.0 and area_b > 0.0:
        return _bbox_iou_for_match(ab, bb, category=category) >= DEDUPE_BBOX_IOU_TOL

    ax, ay, aw, ah = ab
    bx, by, bw, bh = bb
    acx, acy = ax + aw / 2.0, ay + ah / 2.0
    bcx, bcy = bx + bw / 2.0, by + bh / 2.0
    return (
        abs(acx - bcx) <= DEDUPE_CENTER_TOL
        and abs(acy - bcy) <= DEDUPE_CENTER_TOL
        and abs(aw - bw) <= DEDUPE_SIZE_TOL
        and abs(ah - bh) <= DEDUPE_SIZE_TOL
    )


def _category_bbox_tolerance(category: str | None) -> tuple[float, float, float]:
    conf = CATEGORY_BBOX_TOLERANCE.get(category, {})
    return (
        float(conf.get("center", BBOX_CENTER_TOL)),
        float(conf.get("size", BBOX_SIZE_TOL)),
        float(conf.get("iou", BBOX_IOU_TOL)),
    )


def _style_match_for_axes_bbox(a: Any, b: Any, *, category: str | None = None) -> bool:
    if not (isinstance(a, dict) and isinstance(b, dict)):
        return False

    if category == "1_enclosure":
        keys = ("color", "alpha")
    elif category == "2_connector":
        keys = ("linestyle", "color", "arrow_style", "text_content", "orientation")
    elif category == "6_indicator":
        keys = ("linestyle", "color", "is_full_span", "orientation")
    else:
        keys = ("linestyle", "color")

    for key in keys:
        if key not in a or key not in b:
            continue
        av = a.get(key)
        bv = b.get(key)
        if av is None or bv is None:
            continue
        if key == "color":
            if _normalize_color(av) != _normalize_color(bv):
                return False
            continue
        if key == "alpha":
            if not (
                isinstance(av, (int, float))
                and isinstance(bv, (int, float))
                and abs(float(av) - float(bv)) <= NUMERIC_TOL
            ):
                return False
            continue
        if av != bv:
            return False

    return True


def _axes_bbox_close_by_category(a: Any, b: Any, *, category: str | None = None) -> bool:
    if not (isinstance(a, dict) and isinstance(b, dict)):
        return False

    ab = _parse_bbox_field(a, "axes_bbox")
    bb = _parse_bbox_field(b, "axes_bbox")
    if ab is None or bb is None:
        return False

    a_idx = a.get("ax_index")
    b_idx = b.get("ax_index")
    if a_idx is not None and b_idx is not None and a_idx != b_idx:
        return False

    if not _style_match_for_axes_bbox(a, b, category=category):
        return False

    center_tol, size_tol, iou_tol = _category_bbox_tolerance(category)
    ax, ay, aw, ah = ab
    bx, by, bw, bh = bb
    acx, acy = ax + aw / 2.0, ay + ah / 2.0
    bcx, bcy = bx + bw / 2.0, by + bh / 2.0

    if (
        abs(acx - bcx) <= center_tol
        and abs(acy - bcy) <= center_tol
        and abs(aw - bw) <= size_tol
        and abs(ah - bh) <= size_tol
    ):
        return True

    return _bbox_iou_for_match(ab, bb, category=category) >= iou_tol


def _normalize_identity_value(v: Any) -> Any:
    if isinstance(v, str):
        return v.strip().lower()
    return v


def _element_identity_match(a: Any, b: Any, category: str | None = None) -> bool:
    if not (isinstance(a, dict) and isinstance(b, dict)):
        return False

    keys = CATEGORY_IDENTITY_KEYS.get(category, ("src", "type", "marker"))
    shared = 0
    for k in keys:
        if k not in a or k not in b:
            continue
        av = a.get(k)
        bv = b.get(k)
        if av is None or bv is None:
            continue
        shared += 1
        if _normalize_identity_value(av) != _normalize_identity_value(bv):
            return False
    return shared > 0


def _bbox_close(a: Any, b: Any, category: str | None = None) -> bool:
    if (
        category in {"1_enclosure", "2_connector", "3_text", "6_indicator"}
        and _has_bbox_field(a, "axes_bbox")
        and _has_bbox_field(b, "axes_bbox")
    ):
        return _axes_bbox_close_by_category(a, b, category=category)

    ab = _parse_bbox(a)
    bb = _parse_bbox(b)
    if ab is None or bb is None:
        return False

    center_tol, size_tol, iou_tol = _category_bbox_tolerance(category)
    ax, ay, aw, ah = ab
    bx, by, bw, bh = bb
    acx, acy = ax + aw / 2.0, ay + ah / 2.0
    bcx, bcy = bx + bw / 2.0, by + bh / 2.0

    if (
        abs(acx - bcx) <= center_tol
        and abs(acy - bcy) <= center_tol
        and abs(aw - bw) <= size_tol
        and abs(ah - bh) <= size_tol
    ):
        return True
    return _bbox_iou_for_match(ab, bb, category=category) >= iou_tol


def _value_match(a: Any, b: Any, key_hint: str | None = None, category: str | None = None) -> bool:
    if key_hint in {"bbox", "overlap_bbox"}:
        return _bbox_close(a, b, category=category)

    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= NUMERIC_TOL

    if isinstance(a, dict) and isinstance(b, dict):
        if key_hint is None and category is not None and not _element_identity_match(a, b, category=category):
            return False

        if (
            key_hint is None
            and _element_identity_match(a, b, category=category)
            and _bbox_close(a, b, category=category)
        ):
            return True

        if set(a.keys()) != set(b.keys()):
            return False

        for k in a.keys():
            if k == "color":
                if _normalize_color(a.get(k)) != _normalize_color(b.get(k)):
                    return False
                continue
            if not _value_match(a.get(k), b.get(k), key_hint=k, category=category):
                return False
        return True

    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_value_match(x, y, category=category) for x, y in zip(a, b))

    return a == b


def _text_item_match(a: Any, b: Any) -> bool:
    ka = _text_content_key(a)
    kb = _text_content_key(b)
    if ka is None or kb is None or ka != kb:
        return False

    if isinstance(a, dict) and isinstance(b, dict):
        a_idx = a.get("ax_index")
        b_idx = b.get("ax_index")
        if a_idx is not None and b_idx is not None and a_idx != b_idx:
            return False

    if _bbox_close(a, b, category="3_text"):
        return True

    if _parse_bbox(a) is None and _parse_bbox(b) is None:
        return True
    return False


def _item_match_for_subtraction(gt_item: Any, base_item: Any, category: str | None = None) -> bool:
    if category == "3_text":
        return _text_item_match(gt_item, base_item)
    if _value_match(gt_item, base_item, category=category):
        return True

    if not (
        isinstance(gt_item, dict)
        and isinstance(base_item, dict)
        and _element_identity_match(gt_item, base_item, category=category)
    ):
        return False

    gb = _parse_bbox(gt_item)
    bb = _parse_bbox(base_item)
    if gb is None or bb is None:
        return False

    inter = _bbox_intersection_area(gb, bb)
    base_area = max(0.0, bb[2]) * max(0.0, bb[3])
    if base_area <= 0.0:
        return False

    return (inter / base_area) >= BBOX_BASE_COVERAGE_TOL


def _subtract_by_tolerant_match(
    gt_items: list[Any], base_items: list[Any], category: str | None = None
) -> list[Any]:
    base_remaining = list(base_items)
    kept = []

    for gt_item in gt_items:
        matched_idx = None
        for i, base_item in enumerate(base_remaining):
            if _item_match_for_subtraction(gt_item, base_item, category=category):
                matched_idx = i
                break

        if matched_idx is not None:
            base_remaining.pop(matched_idx)
        else:
            kept.append(gt_item)

    return kept


def _dedupe_items_by_identity_bbox(items: list[Any], category: str | None = None) -> list[Any]:
    deduped = []
    for item in items:
        duplicate = False
        for kept in deduped:
            if (
                _element_identity_match(item, kept, category=category)
                and _bbox_duplicate_for_dedupe(item, kept, category=category)
            ):
                duplicate = True
                break
        if not duplicate:
            deduped.append(item)
    return deduped


def pre_dedupe_annotation_dict(data: Any) -> Any:
    if not isinstance(data, dict):
        return data

    cleaned = {}
    for category, items in data.items():
        if not isinstance(items, list):
            cleaned[category] = items
            continue
        if category in DEDUPE_EXCLUDE_CATEGORIES:
            cleaned[category] = items
            continue
        cleaned[category] = _dedupe_items_by_identity_bbox(items, category=category)

    return order_feature_dict(cleaned)


def _subtract_items(category: str, gt_items: list[Any], base_items: list[Any]) -> list[Any]:
    if category != "3_text":
        if category in {"6_indicator", "7_geometric"}:
            gt_items = _dedupe_items_by_identity_bbox(gt_items, category=category)
            base_items = _dedupe_items_by_identity_bbox(base_items, category=category)
        return _subtract_by_tolerant_match(gt_items, base_items, category=category)

    base_remaining = list(base_items)
    kept = []
    for item in gt_items:
        matched_idx = None
        for i, base_item in enumerate(base_remaining):
            if _text_item_match(item, base_item):
                matched_idx = i
                break
        if matched_idx is not None:
            base_remaining.pop(matched_idx)
        else:
            kept.append(item)

    return kept


def subtract_annotation_dict(gt_data: dict[str, Any], base_data: dict[str, Any]) -> dict[str, Any]:
    gt_data = pre_dedupe_annotation_dict(gt_data)
    base_data = pre_dedupe_annotation_dict(base_data)

    result_data = {}
    all_categories = set(gt_data.keys()) | set(base_data.keys())
    for category in all_categories:
        gt_items = gt_data.get(category, [])
        base_items = base_data.get(category, [])
        result_data[category] = _subtract_items(category, gt_items, base_items)

    return order_feature_dict(result_data)


def infer_removed_basename(test_json_filename: str) -> str:
    stem = os.path.splitext(test_json_filename)[0]
    m = STAGE_FILE_PATTERN.match(stem)
    if m:
        return f"{m.group(1)}.json"

    parts = stem.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}.json"
    return f"{stem}.json"


def infer_chart_id(filename: str) -> str:
    stem = os.path.splitext(filename)[0]
    m = STAGE_FILE_PATTERN.match(stem)
    if m:
        return m.group(1)

    parts = stem.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return stem


def build_structured_output_path(
    output_root: str,
    category: str,
    filename: str,
    model_label: str,
) -> str:
    chart_id = infer_chart_id(filename)
    return os.path.join(output_root, category, chart_id, model_label, filename)


__all__ = [
    "FEATURE_ORDER",
    "build_structured_output_path",
    "generate_fingerprint",
    "infer_chart_id",
    "infer_removed_basename",
    "order_feature_dict",
    "pre_dedupe_annotation_dict",
    "subtract_annotation_dict",
]
