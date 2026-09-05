import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from annotation_eval.config import get_path

OUTPUT_ROOT = REPO_ROOT / "outputs"
ANNOTATION_ROOT = OUTPUT_ROOT / "annotations"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "analysis" / "text_relationship"
DEFAULT_ALL_TEXT_ROOT = OUTPUT_ROOT / "analysis" / "text_extraction"

CANVAS_BOX = (0.0, 0.0, 1.0, 1.0)
STAGE_FILE_PATTERN = re.compile(
    r"^([A-Za-z]+_\d+)(?:_(?:llm|vlm|code|code_image))?_(task|intent|operation|implementation)$",
    re.IGNORECASE,
)
SIDECAR_SOURCE_TAGS = {
    "artist_gt": "artist_gt",
    "artist_removed": "artist_removed",
    "artist_removed_test": "artist_removed_test",
    "artist_test_llm": "artist_test_llm",
    "artist_test_vlm": "artist_test_vlm",
    "final_gt_structured": "artist_gt",
    "final_test_llm_minus_removed_structured": "artist_test_llm",
    "final_test_vlm_minus_removed_structured": "artist_test_vlm",
    "raw_test_llm": "artist_test_llm",
    "raw_test_vlm": "artist_test_vlm",
    "raw_test_llm_minus_removed": "artist_test_llm",
    "raw_test_vlm_minus_removed": "artist_test_vlm",
    "raw_removed_test": "artist_removed_test",
}


def _iter_json_files(root: str):
    for dirpath, _, filenames in os.walk(root):
        for filename in sorted(filenames):
            if filename.endswith(".json"):
                yield os.path.join(dirpath, filename)


def _read_json(path: str) -> Optional[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _to_float(value) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _parse_bbox_xywh(item: Dict) -> Optional[Tuple[float, float, float, float]]:
    if not isinstance(item, dict):
        return None

    bbox = item.get("bbox")
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        return None

    vals = [_to_float(v) for v in bbox]
    if any(v is None for v in vals):
        return None

    x, y, w, h = vals
    return x, y, max(0.0, w), max(0.0, h)


def _xywh_to_xyxy(x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
    return x, y, x + w, y + h


def _bbox_center_and_size(item: Dict) -> Optional[Tuple[float, float, float, float]]:
    bbox = item.get("bbox_xywh")
    if not (isinstance(bbox, list) and len(bbox) == 4):
        return None
    x, y, w, h = bbox
    try:
        x = float(x)
        y = float(y)
        w = max(0.0, float(w))
        h = max(0.0, float(h))
    except Exception:
        return None
    return x + w / 2.0, y + h / 2.0, w, h


def _text_bbox_match_score(current: Dict, removed: Dict) -> float:
    current_bbox = _bbox_center_and_size(current)
    removed_bbox = _bbox_center_and_size(removed)
    if current_bbox is None or removed_bbox is None:
        return float("inf")

    ccx, ccy, cw, ch = current_bbox
    rcx, rcy, rw, rh = removed_bbox
    center_dx = abs(ccx - rcx)
    center_dy = abs(ccy - rcy)
    size_dw = abs(cw - rw)
    size_dh = abs(ch - rh)
    return center_dx + center_dy + 0.5 * (size_dw + size_dh)


def _normalize_text(content: str) -> Optional[str]:
    if not isinstance(content, str):
        return None
    text = " ".join(content.split())
    return text if text else None


def _area_xyxy(box: Tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = box
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _intersection_xyxy(
    a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]
) -> Optional[Tuple[float, float, float, float, float]]:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = ix1 - ix0
    ih = iy1 - iy0
    if iw <= 0.0 or ih <= 0.0:
        return None
    return ix0, iy0, ix1, iy1, iw * ih


def _union_area_xyxy(rects: List[Tuple[float, float, float, float]]) -> float:
    normalized = []
    for rect in rects:
        if not rect or len(rect) != 4:
            continue
        x0, y0, x1, y1 = rect
        if x1 <= x0 or y1 <= y0:
            continue
        normalized.append((float(x0), float(y0), float(x1), float(y1)))

    if not normalized:
        return 0.0

    xs = sorted({x0 for x0, _, x1, _ in normalized} | {x1 for _, _, x1, _ in normalized})
    if len(xs) < 2:
        return 0.0

    total = 0.0
    for i in range(len(xs) - 1):
        sx0 = xs[i]
        sx1 = xs[i + 1]
        if sx1 <= sx0:
            continue

        intervals = []
        for x0, y0, x1, y1 in normalized:
            if x0 < sx1 and x1 > sx0:
                intervals.append((y0, y1))

        if not intervals:
            continue

        intervals.sort()
        covered_y = 0.0
        cur_y0, cur_y1 = intervals[0]
        for y0, y1 in intervals[1:]:
            if y0 <= cur_y1:
                cur_y1 = max(cur_y1, y1)
            else:
                covered_y += max(0.0, cur_y1 - cur_y0)
                cur_y0, cur_y1 = y0, y1
        covered_y += max(0.0, cur_y1 - cur_y0)
        total += (sx1 - sx0) * covered_y

    return total


def _outside_canvas_rects(
    box: Tuple[float, float, float, float],
    canvas_box: Tuple[float, float, float, float] = CANVAS_BOX,
) -> List[Tuple[float, float, float, float]]:
    x0, y0, x1, y1 = box
    cx0, cy0, cx1, cy1 = canvas_box

    ix0 = min(max(x0, cx0), cx1)
    iy0 = min(max(y0, cy0), cy1)
    ix1 = min(max(x1, cx0), cx1)
    iy1 = min(max(y1, cy0), cy1)

    rects: List[Tuple[float, float, float, float]] = []

    if x0 < ix0:
        rects.append((x0, y0, ix0, y1))
    if ix1 < x1:
        rects.append((ix1, y0, x1, y1))
    if y0 < iy0 and ix0 < ix1:
        rects.append((ix0, y0, ix1, iy0))
    if iy1 < y1 and ix0 < ix1:
        rects.append((ix0, iy1, ix1, y1))

    return [rect for rect in rects if _area_xyxy(rect) > 0.0]


def _classify_by_overlap_ratio(
    ratio: float, mild_max_ratio: float, moderate_max_ratio: float
) -> str:
    if ratio <= mild_max_ratio:
        return "mild"
    if ratio <= moderate_max_ratio:
        return "moderate"
    return "severe"


def _new_severity_counts() -> Dict[str, int]:
    return {
        "mild": 0,
        "moderate": 0,
        "severe": 0,
    }


def _severity_ratio_pct(counts: Dict[str, int], total: int) -> Dict[str, float]:
    if total <= 0:
        return {k: 0.0 for k in counts.keys()}
    return {k: round((v / total) * 100.0, 3) for k, v in counts.items()}


def _extract_text_items(data: Dict) -> List[Dict]:
    raw_items = data.get("3_text", [])
    return _extract_text_items_from_raw(raw_items)


def _extract_text_items_from_raw(raw_items: List[Dict]) -> List[Dict]:
    if not isinstance(raw_items, list):
        return []

    extracted = []
    for idx, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue

        text = _normalize_text(item.get("content"))
        if text is None:
            continue

        bbox_xywh = _parse_bbox_xywh(item)
        if bbox_xywh is None:
            continue
        bbox_xyxy = _xywh_to_xyxy(*bbox_xywh)

        extracted.append(
            {
                "index": idx,
                "content": text,
                "bbox_xywh": [round(v, 6) for v in bbox_xywh],
                "bbox_xyxy": [round(v, 6) for v in bbox_xyxy],
            }
        )
    return extracted


def _load_sidecar_all_text(
    all_text_root: str, source_root: str, file_rel: str
) -> Optional[List[Dict]]:
    if not all_text_root:
        return None

    source_name = os.path.basename(os.path.normpath(source_root))
    source_tag = SIDECAR_SOURCE_TAGS.get(source_name, source_name)
    sidecar_path = os.path.join(all_text_root, source_tag, file_rel)
    sidecar_data = _read_json(sidecar_path)
    if sidecar_data is None:
        return None

    raw_items = sidecar_data.get("all_text", [])
    items = _extract_text_items_from_raw(raw_items)
    return items if items else []


def _normalize_relpath_for_sidecar(file_rel: str) -> str:
    parts = Path(file_rel).parts
    if len(parts) >= 4 and parts[2] in {"GT", "LLM", "VLM"}:
        return os.path.join(parts[0], parts[-1])
    return file_rel


def _infer_removed_relpath(file_rel: str) -> str:
    file_rel = _normalize_relpath_for_sidecar(file_rel)
    category, filename = os.path.split(file_rel)
    stem, _ = os.path.splitext(filename)
    m = STAGE_FILE_PATTERN.match(stem)
    if m:
        return os.path.join(category, f"{m.group(1)}.json")
    return file_rel


def _split_original_and_added(
    current_texts: List[Dict], removed_texts: List[Dict]
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    current_by_content: Dict[str, List[Dict]] = {}
    removed_by_content: Dict[str, List[Dict]] = {}

    for t in current_texts:
        current_by_content.setdefault(t["content"], []).append(t)
    for t in removed_texts:
        removed_by_content.setdefault(t["content"], []).append(t)

    original_indexes = set()

    for content, current_group in current_by_content.items():
        removed_group = removed_by_content.get(content, [])
        if not removed_group:
            continue

        if len(current_group) == 1 and len(removed_group) == 1:
            original_indexes.add(current_group[0]["index"])
            continue

        remaining_removed = removed_group[:]
        for current_item in sorted(current_group, key=lambda item: item["index"]):
            if not remaining_removed:
                break
            best_idx = min(
                range(len(remaining_removed)),
                key=lambda idx: _text_bbox_match_score(current_item, remaining_removed[idx]),
            )
            original_indexes.add(current_item["index"])
            remaining_removed.pop(best_idx)

    all_texts = []
    added_texts = []
    for t in current_texts:
        role = "original_text" if t["index"] in original_indexes else "added_text"
        item = {
            "index": t["index"],
            "content": t["content"],
            "bbox_xywh": t["bbox_xywh"],
            "bbox_xyxy": t["bbox_xyxy"],
            "role": role,
        }
        all_texts.append(item)
        if role != "original_text":
            added_texts.append(item)

    original_texts = [t for t in all_texts if t["role"] == "original_text"]
    return all_texts, original_texts, added_texts


def _build_current_reference_texts(
    current_sidecar_texts: List[Dict],
    added_texts: List[Dict],
    min_match_ratio: float = 0.5,
) -> List[Dict]:
    """
    Build original-text reference pool in CURRENT image coordinates.

    We identify sidecar entries that correspond to added annotations by
    matching on text content + geometric coverage, then exclude them.
    Remaining sidecar texts are treated as original/base-chart texts.
    """
    if not current_sidecar_texts:
        return []

    consumed_sidecar_idx = set()

    for added in added_texts:
        a_box = tuple(added["bbox_xyxy"])
        a_area = _area_xyxy(a_box)
        if a_area <= 0.0:
            continue

        best_idx = None
        best_ratio = 0.0

        for i, cand in enumerate(current_sidecar_texts):
            if i in consumed_sidecar_idx:
                continue
            if cand.get("content") != added.get("content"):
                continue

            c_box = tuple(cand["bbox_xyxy"])
            inter = _intersection_xyxy(a_box, c_box)
            if inter is None:
                continue

            _, _, _, _, inter_area = inter
            ratio = inter_area / a_area
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = i

        if best_idx is not None and best_ratio >= min_match_ratio:
            consumed_sidecar_idx.add(best_idx)

    original_ref_texts = []
    for i, t in enumerate(current_sidecar_texts):
        if i in consumed_sidecar_idx:
            continue
        original_ref_texts.append(
            {
                "index": i,
                "content": t["content"],
                "bbox_xywh": t["bbox_xywh"],
                "bbox_xyxy": t["bbox_xyxy"],
                "role": "original_text",
            }
        )

    return original_ref_texts


def _detect_out_of_canvas(texts: List[Dict]) -> List[Dict]:
    out_items = []
    for t in texts:
        x0, y0, x1, y1 = t["bbox_xyxy"]
        sides = []
        if x0 < 0.0:
            sides.append("left")
        if y0 < 0.0:
            sides.append("bottom")
        if x1 > 1.0:
            sides.append("right")
        if y1 > 1.0:
            sides.append("top")
        if not sides:
            continue

        box = (float(x0), float(y0), float(x1), float(y1))
        area = _area_xyxy(box)
        inter = _intersection_xyxy(box, CANVAS_BOX)
        inside_area = inter[4] if inter is not None else 0.0
        outside_area = max(0.0, area - inside_area)
        outside_pct = 0.0 if area <= 0.0 else (outside_area / area) * 100.0

        out_items.append(
            {
                "index": t["index"],
                "content": t["content"],
                "role": t["role"],
                "bbox_xyxy": t["bbox_xyxy"],
                "outside_sides": sides,
                "outside_pct": round(outside_pct, 3),
            }
        )
    return out_items


def _build_overlap_record(
    left: Dict,
    right: Dict,
    left_ratio: float,
    right_ratio: Optional[float] = None,
    intersection_box: Optional[Tuple[float, float, float, float]] = None,
) -> Dict:
    ratio = left_ratio if right_ratio is None else max(left_ratio, right_ratio)
    rec = {
        "left_index": left["index"],
        "right_index": right["index"],
        "left_ratio": round(left_ratio, 6),
        "ratio": round(ratio, 6),
    }
    if right_ratio is not None:
        rec["right_ratio"] = round(right_ratio, 6)
    if intersection_box is not None:
        rec["intersection_bbox_xyxy"] = [round(v, 6) for v in intersection_box]
    return rec


def _collect_overlaps(
    left_items: List[Dict],
    right_items: List[Dict],
    min_overlap_ratio: float,
    same_pool: bool = False,
) -> List[Dict]:
    pairs = []

    if same_pool:
        for i in range(len(left_items)):
            left = left_items[i]
            left_box = tuple(left["bbox_xyxy"])
            left_area = _area_xyxy(left_box)
            for j in range(i + 1, len(left_items)):
                right = left_items[j]
                right_box = tuple(right["bbox_xyxy"])
                inter = _intersection_xyxy(left_box, right_box)
                if inter is None:
                    continue
                _, _, _, _, inter_area = inter
                right_area = _area_xyxy(right_box)
                if left_area <= 0.0 or right_area <= 0.0:
                    continue
                left_ratio = inter_area / left_area
                right_ratio = inter_area / right_area
                if max(left_ratio, right_ratio) <= min_overlap_ratio:
                    continue
                pairs.append(
                    _build_overlap_record(
                        left,
                        right,
                        left_ratio,
                        right_ratio,
                        intersection_box=inter[:4],
                    )
                )
    else:
        for left in left_items:
            left_box = tuple(left["bbox_xyxy"])
            left_area = _area_xyxy(left_box)
            if left_area <= 0.0:
                continue
            for right in right_items:
                right_box = tuple(right["bbox_xyxy"])
                inter = _intersection_xyxy(left_box, right_box)
                if inter is None:
                    continue
                _, _, _, _, inter_area = inter
                left_ratio = inter_area / left_area
                if left_ratio <= min_overlap_ratio:
                    continue
                pairs.append(
                    _build_overlap_record(
                        left,
                        right,
                        left_ratio,
                        intersection_box=inter[:4],
                    )
                )

    pairs.sort(key=lambda x: (-x["ratio"], x["left_index"], x["right_index"]))
    return pairs


def _analyze_file(
    file_path: str,
    artist_root: str,
    removed_root: str,
    all_text_root: str,
    min_overlap_ratio: float,
    mild_max_ratio: float,
    moderate_max_ratio: float,
) -> Optional[Dict]:
    current_data = _read_json(file_path)
    if current_data is None:
        return None

    file_rel = os.path.relpath(file_path, artist_root)
    current_texts = _extract_text_items(current_data)
    current_sidecar_texts = _load_sidecar_all_text(
        all_text_root,
        artist_root,
        _normalize_relpath_for_sidecar(file_rel),
    )

    removed_rel = _infer_removed_relpath(file_rel)
    removed_path = os.path.join(removed_root, removed_rel) if removed_root else ""
    removed_data = _read_json(removed_path) if removed_path and os.path.exists(removed_path) else None
    removed_texts = _load_sidecar_all_text(all_text_root, removed_root, removed_rel)
    if removed_texts is None:
        removed_texts = _extract_text_items(removed_data) if removed_data is not None else []

    all_texts, original_texts, added_texts = _split_original_and_added(
        current_texts,
        removed_texts,
    )

    if current_sidecar_texts is not None:
        original_texts = _build_current_reference_texts(current_sidecar_texts, added_texts)

    added_vs_original_overlaps = _collect_overlaps(
        added_texts,
        original_texts,
        min_overlap_ratio=min_overlap_ratio,
        same_pool=False,
    )
    added_vs_added_overlaps = _collect_overlaps(
        added_texts,
        added_texts,
        min_overlap_ratio=min_overlap_ratio,
        same_pool=True,
    )
    out_of_canvas_texts_all = _detect_out_of_canvas(all_texts)
    out_of_canvas_added = [t for t in out_of_canvas_texts_all if t.get("role") == "added_text"]

    overlapping_added_indices = set()
    overlapping_added_max_iou = {}

    for pair in added_vs_original_overlaps:
        overlapping_added_indices.add(pair["left_index"])
        left_idx = pair["left_index"]
        ratio = pair["left_ratio"]
        prev = overlapping_added_max_iou.get(left_idx, -1.0)
        if ratio > prev:
            overlapping_added_max_iou[left_idx] = ratio

    for pair in added_vs_added_overlaps:
        overlapping_added_indices.add(pair["left_index"])
        overlapping_added_indices.add(pair["right_index"])
        li = pair["left_index"]
        ri = pair["right_index"]
        lr = pair["left_ratio"]
        rr = pair["right_ratio"]
        if lr > overlapping_added_max_iou.get(li, -1.0):
            overlapping_added_max_iou[li] = lr
        if rr > overlapping_added_max_iou.get(ri, -1.0):
            overlapping_added_max_iou[ri] = rr

    overlapping_added_text_severity_counts = _new_severity_counts()
    for idx in overlapping_added_indices:
        max_ratio = overlapping_added_max_iou.get(idx, 0.0)
        s = _classify_by_overlap_ratio(max_ratio, mild_max_ratio, moderate_max_ratio)
        overlapping_added_text_severity_counts[s] += 1

    added_count = len(added_texts)
    overlapping_added_count = len(overlapping_added_indices)
    out_of_canvas_added_count = len(out_of_canvas_added)
    overlap_ratio = 0.0 if added_count == 0 else (overlapping_added_count / added_count) * 100.0
    out_canvas_ratio = 0.0 if added_count == 0 else (out_of_canvas_added_count / added_count) * 100.0
    canvas_area = _area_xyxy(CANVAS_BOX)

    overlap_rects = []
    for pair in added_vs_original_overlaps:
        bbox = pair.get("intersection_bbox_xyxy")
        if isinstance(bbox, list) and len(bbox) == 4:
            overlap_rects.append(tuple(float(v) for v in bbox))
    for pair in added_vs_added_overlaps:
        bbox = pair.get("intersection_bbox_xyxy")
        if isinstance(bbox, list) and len(bbox) == 4:
            overlap_rects.append(tuple(float(v) for v in bbox))
    overlap_global_pct = (
        0.0 if canvas_area <= 0.0 else (_union_area_xyxy(overlap_rects) / canvas_area) * 100.0
    )

    out_rects = []
    for t in out_of_canvas_added:
        box = t.get("bbox_xyxy")
        if isinstance(box, list) and len(box) == 4:
            out_rects.extend(_outside_canvas_rects(tuple(float(v) for v in box)))
    out_global_pct = (
        0.0 if canvas_area <= 0.0 else (_union_area_xyxy(out_rects) / canvas_area) * 100.0
    )

    return {
        "file": file_rel,
        "anno_text_n": added_count,
        "anno_overlap_n": overlapping_added_count,
        "anno_overlap_pct": round(overlap_ratio, 3),
        "overlap_global_pct": round(overlap_global_pct, 3),
        "overlap_severity_pct": _severity_ratio_pct(
            overlapping_added_text_severity_counts, added_count
        ),
        "off_canvas_n": out_of_canvas_added_count,
        "off_canvas_pct": round(out_canvas_ratio, 3),
        "out_global_pct": round(out_global_pct, 3),
        "_overlap_severity_cnt": overlapping_added_text_severity_counts,
    }


def _make_summary(file_results: List[Dict]) -> Dict:
    total_added = sum(r["anno_text_n"] for r in file_results)
    total_overlap_added = sum(r["anno_overlap_n"] for r in file_results)
    total_out_canvas_added = sum(r["off_canvas_n"] for r in file_results)

    summary = {
        "files_n": len(file_results),
        "anno_text_n": total_added,
        "anno_overlap_n": total_overlap_added,
        "anno_overlap_pct": round(
            0.0 if total_added == 0 else (total_overlap_added / total_added) * 100.0, 3
        ),
        "overlap_global_pct": round(
            0.0
            if not file_results
            else sum(r.get("overlap_global_pct", 0.0) for r in file_results) / len(file_results),
            3,
        ),
        "overlap_severity_pct": _new_severity_counts(),
        "off_canvas_n": total_out_canvas_added,
        "off_canvas_pct": round(
            0.0 if total_added == 0 else (total_out_canvas_added / total_added) * 100.0, 3
        ),
        "out_global_pct": round(
            0.0
            if not file_results
            else sum(r.get("out_global_pct", 0.0) for r in file_results) / len(file_results),
            3,
        ),
    }

    total_added_text_severity_counts = _new_severity_counts()
    for r in file_results:
        for k in total_added_text_severity_counts:
            total_added_text_severity_counts[k] += r["_overlap_severity_cnt"][k]

    summary["overlap_severity_pct"] = _severity_ratio_pct(
        total_added_text_severity_counts,
        summary["anno_text_n"],
    )

    return summary


def _model_label(job_name: str, artist_root: str) -> str:
    jr = job_name.lower()
    ar = artist_root.lower()
    if "llm" in jr or "llm" in ar:
        return "LLM"
    if "vlm" in jr or "vlm" in ar:
        return "VLM"
    if "code+image" in ar or "code_image" in ar:
        return "VLM"
    if "code" in ar:
        return "LLM"
    if "gt" in jr or "gt" in ar:
        return "GT"
    return "MODEL"


def _split_chart_stage(file_rel: str) -> Tuple[str, str, str]:
    parts = Path(file_rel).parts
    if len(parts) >= 4 and parts[2] in {"GT", "LLM", "VLM", "code", "code+image"}:
        label = {"code": "llm", "code+image": "vlm"}.get(parts[2], parts[2].lower())
        return parts[0], parts[1], label

    file_rel = _normalize_relpath_for_sidecar(file_rel)
    category, filename = os.path.split(file_rel)
    category = category or "uncategorized"
    stem, _ = os.path.splitext(filename)
    m = STAGE_FILE_PATTERN.match(stem)
    if m:
        stage = "intent" if m.group(2).lower() == "task" else m.group(2).lower()
        return category, m.group(1), stage
    return category, stem, "gt"


def _write_chart_model_outputs(output_root: str, model: str, public_file_results: List[Dict]) -> None:
    os.makedirs(output_root, exist_ok=True)
    for item in public_file_results:
        file_rel = item.get("file", "")
        category, chart_id, _ = _split_chart_stage(file_rel)
        filename = os.path.basename(file_rel) or "result.json"
        model_dir = os.path.join(output_root, category, chart_id, model)
        os.makedirs(model_dir, exist_ok=True)

        # Cleanup legacy generic names from older outputs in the same folder.
        for legacy_name in ("intent.json", "task.json", "operation.json", "implementation.json", "gt.json"):
            legacy_path = os.path.join(model_dir, legacy_name)
            if legacy_name != filename and os.path.exists(legacy_path):
                try:
                    os.remove(legacy_path)
                except Exception:
                    pass

        out_path = os.path.join(model_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(item, f, indent=2, ensure_ascii=False)


def _run_one_job(
    job_name: str,
    artist_root: str,
    removed_root: str,
    output_root: str,
    all_text_root: str,
    only_file: str,
    min_overlap_ratio: float,
    mild_max_ratio: float,
    moderate_max_ratio: float,
) -> Dict:
    if not os.path.isdir(artist_root):
        print(f"[skip] {job_name}: artist root not found -> {artist_root}")
        return {"job_name": job_name, "skipped": True, "reason": "artist_root_missing"}

    files = list(_iter_json_files(artist_root))
    if only_file:
        target = os.path.normpath(only_file)
        files = [
            p
            for p in files
            if os.path.normpath(os.path.relpath(p, artist_root)) == target
        ]

    file_results = []
    for path in files:
        item = _analyze_file(
            file_path=path,
            artist_root=artist_root,
            removed_root=removed_root,
            all_text_root=all_text_root,
            min_overlap_ratio=min_overlap_ratio,
            mild_max_ratio=mild_max_ratio,
            moderate_max_ratio=moderate_max_ratio,
        )
        if item is not None:
            file_results.append(item)

    file_results.sort(key=lambda x: x["file"])
    summary = _make_summary(file_results)
    public_file_results = []
    for item in file_results:
        public_file_results.append({k: v for k, v in item.items() if not k.startswith("_")})
    model = _model_label(job_name, artist_root)
    if output_root:
        _write_chart_model_outputs(output_root, model, public_file_results)

    print(
        f"[{job_name}] files={summary['files_n']} "
        f"anno={summary['anno_text_n']} "
        f"overlap={summary['anno_overlap_n']} "
        f"m/m/s(%)={summary['overlap_severity_pct']['mild']}/"
        f"{summary['overlap_severity_pct']['moderate']}/"
        f"{summary['overlap_severity_pct']['severe']} "
        f"overlap_pct={summary['anno_overlap_pct']}% "
        f"overlap_global_pct={summary['overlap_global_pct']}% "
        f"out_canvas_pct={summary['off_canvas_pct']}% "
        f"out_global_pct={summary['out_global_pct']}%"
    )
    if output_root:
        print(f"Saved: {output_root}/{model}")

    return {
        "job_name": job_name,
        "skipped": False,
        "artist_root": artist_root,
        "removed_root": removed_root,
        "output_root": output_root,
        "model": model,
        "summary": summary,
    }


def build_cli():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze text relationships for Artist JSON files: all texts, original-vs-added split "
            "(based on removed set), overlaps, and out-of-canvas detection."
        )
    )
    parser.add_argument(
        "--project-root",
        default=str(REPO_ROOT),
        help="Project root used to resolve default outputs/annotations paths.",
    )
    parser.add_argument("--run-all", action="store_true", help="Run GT/test_llm/test_vlm jobs once.")
    parser.add_argument("--artist-root", default="", help="Current Artist root for single-job mode.")
    parser.add_argument("--removed-root", default="", help="Removed Artist root for single-job mode.")
    parser.add_argument("--output-json", default="", help="Deprecated. If set, its parent directory is used as output root.")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory for --run-all mode.",
    )
    parser.add_argument(
        "--text-extraction-root",
        default="",
        help=(
            "Optional sidecar root for text extraction results. "
            "Expected layout: <root>/<source_tag>/<Category>/<file>.json "
            "(e.g., artist_test_llm/Density/Density_1_llm_implementation.json). "
            "If missing, fallback to 3_text in artist json."
        ),
    )
    parser.add_argument("--only-file", default="", help="Optional relative file filter, e.g. Area/Area_27.json")
    parser.add_argument(
        "--min-overlap-ratio",
        type=float,
        default=0.05,
        help="Keep overlaps with (intersection / anno text area) > value. Default: 0.05",
    )
    parser.add_argument(
        "--mild-max-ratio",
        type=float,
        default=0.2,
        help="Upper bound for mild overlap ratio. Default: 0.2",
    )
    parser.add_argument(
        "--moderate-max-ratio",
        type=float,
        default=0.5,
        help="Upper bound for moderate overlap ratio. Default: 0.5",
    )
    return parser


def main():
    args = build_cli().parse_args()
    project_root = Path(args.project_root).resolve()
    annotation_root = get_path(project_root, "annotations_dir", "outputs/annotations")
    analysis_root = get_path(project_root, "analysis_dir", "outputs/analysis")
    default_output_dir = analysis_root / "text_relationship"
    default_all_text_root = analysis_root / "text_extraction"
    min_overlap_ratio = args.min_overlap_ratio
    mild_max_ratio = args.mild_max_ratio
    moderate_max_ratio = args.moderate_max_ratio

    if min_overlap_ratio < 0 or min_overlap_ratio >= 1:
        raise ValueError("--min-overlap-ratio must be in [0, 1)")
    if mild_max_ratio <= 0 or mild_max_ratio >= 1:
        raise ValueError("--mild-max-ratio must be in (0, 1)")
    if moderate_max_ratio <= 0 or moderate_max_ratio >= 1:
        raise ValueError("--moderate-max-ratio must be in (0, 1)")
    if moderate_max_ratio < mild_max_ratio:
        raise ValueError("--moderate-max-ratio must be >= --mild-max-ratio")

    output_root = args.output_dir or str(default_output_dir)
    if args.output_json:
        output_root = os.path.dirname(args.output_json) or args.output_dir

    all_text_root = args.text_extraction_root or str(default_all_text_root)

    if args.run_all or not args.artist_root:
        jobs = [
            (
                "gt",
                str(annotation_root / "final_gt_structured"),
                str(annotation_root / "artist_removed"),
            ),
            (
                "test_llm",
                str(annotation_root / "final_test_llm_minus_removed_structured"),
                str(annotation_root / "raw_removed_test"),
            ),
            (
                "test_vlm",
                str(annotation_root / "final_test_vlm_minus_removed_structured"),
                str(annotation_root / "raw_removed_test"),
            ),
        ]
        for job_name, artist_root, removed_root in jobs:
            _run_one_job(
                job_name=job_name,
                artist_root=artist_root,
                removed_root=removed_root,
                output_root=output_root,
                all_text_root=all_text_root,
                only_file=args.only_file,
                min_overlap_ratio=min_overlap_ratio,
                mild_max_ratio=mild_max_ratio,
                moderate_max_ratio=moderate_max_ratio,
            )
        return

    if not args.removed_root:
        raise ValueError("--removed-root is required in single-job mode.")
    _run_one_job(
        job_name="single",
        artist_root=args.artist_root,
        removed_root=args.removed_root,
        output_root=output_root,
        all_text_root=all_text_root,
        only_file=args.only_file,
        min_overlap_ratio=min_overlap_ratio,
        mild_max_ratio=mild_max_ratio,
        moderate_max_ratio=moderate_max_ratio,
    )


if __name__ == "__main__":
    main()
