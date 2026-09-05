import json
import math

REFERENCE_CANVAS_PX = 1200.0
LINE_IOU_DILATE_PX = 4.0
LINE_IOU_DILATE_NORM = LINE_IOU_DILATE_PX / REFERENCE_CANVAS_PX

RAW_KIND_TOLERANCE = {
    "axes": {"center": 0.08, "size": 0.12, "iou": 0.25},
    "text": {"center": 0.05, "size": 0.16, "iou": 0.10},
    "annotation_bbox": {"center": 0.06, "size": 0.18, "iou": 0.12},
    "annotation_arrow": {"center": 0.06, "size": 0.14, "iou": 0.15},
    "line": {"center": 0.05, "size": 0.12, "iou": 0.20},
    "figure_line": {"center": 0.05, "size": 0.12, "iou": 0.20},
    "patch": {"center": 0.05, "size": 0.10, "iou": 0.30},
    "collection": {"center": 0.06, "size": 0.14, "iou": 0.20},
    "table_cell": {"center": 0.03, "size": 0.08, "iou": 0.40},
}

GROUP_COLLECTION_TYPES = {"FillBetweenPolyCollection", "PolyCollection"}
GROUP_LINE_KINDS = {"line", "figure_line"}
MARKER_NONE_VALUES = {"None", "none", "", None}


def _normalize_text(text):
    if not isinstance(text, str):
        return None
    return " ".join(text.strip().lower().split())


def _to_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _parse_bbox(box):
    if not (isinstance(box, (list, tuple)) and len(box) == 4):
        return None
    vals = [_to_float(v) for v in box]
    if any(v is None for v in vals):
        return None
    if any(not math.isfinite(v) for v in vals):
        return None
    x, y, w, h = vals
    return x, y, max(0.0, w), max(0.0, h)


def _best_bbox(record):
    axes_bbox = _parse_bbox(record.get("axes_bbox"))
    if axes_bbox is not None:
        return axes_bbox
    return _parse_bbox(record.get("bbox"))


def _container_axes_bbox(record):
    return _parse_bbox(record.get("container_axes_bbox"))


def _same_container_axes(a, b):
    a_box = _container_axes_bbox(a)
    b_box = _container_axes_bbox(b)
    if a_box is None or b_box is None:
        return False
    return _bbox_match(a_box, b_box, "axes")


def _is_line_like(box):
    if box is None:
        return False
    _, _, w, h = box
    return w <= 1e-9 or h <= 1e-9


def _line_orientation(box):
    if box is None:
        return None
    _, _, w, h = box
    is_vertical = w <= 1e-9 and h > 1e-9
    is_horizontal = h <= 1e-9 and w > 1e-9
    if is_vertical:
        return "vertical"
    if is_horizontal:
        return "horizontal"
    if w <= 1e-9 and h <= 1e-9:
        return "point"
    return None


def _dilate_for_iou(box, kind):
    if box is None:
        return None
    if kind not in {"line", "figure_line", "collection"} or not _is_line_like(box):
        return box
    x, y, w, h = box
    d = float(LINE_IOU_DILATE_NORM)
    return x - d, y - d, max(0.0, w + 2.0 * d), max(0.0, h + 2.0 * d)


def _bbox_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax1, ay1 = ax + aw, ay + ah
    bx1, by1 = bx + bw, by + bh
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = aw * ah + bw * bh - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _bbox_score(a_box, b_box, kind):
    if a_box is None or b_box is None:
        return math.inf
    ax, ay, aw, ah = a_box
    bx, by, bw, bh = b_box
    acx, acy = ax + aw / 2.0, ay + ah / 2.0
    bcx, bcy = bx + bw / 2.0, by + bh / 2.0
    center = abs(acx - bcx) + abs(acy - bcy)
    size = abs(aw - bw) + abs(ah - bh)
    iou = _bbox_iou(_dilate_for_iou(a_box, kind), _dilate_for_iou(b_box, kind))
    return center + size + (1.0 - iou)


def _bbox_contains(outer_box, inner_box, tol=0.03):
    if outer_box is None or inner_box is None:
        return False
    try:
        ox, oy, ow, oh = [float(v) for v in outer_box]
        ix, iy, iw, ih = [float(v) for v in inner_box]
    except Exception:
        return False
    return (
        ix >= ox - tol
        and iy >= oy - tol
        and ix + iw <= ox + ow + tol
        and iy + ih <= oy + oh + tol
    )


def _union_bbox(boxes):
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        return None
    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[0] + box[2] for box in boxes)
    y1 = max(box[1] + box[3] for box in boxes)
    return (x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0))


def _boxes_overlap(a_box, b_box):
    if a_box is None or b_box is None:
        return False
    ax, ay, aw, ah = a_box
    bx, by, bw, bh = b_box
    return (ax < bx + bw) and (bx < ax + aw) and (ay < by + bh) and (by < ay + ah)


def _bbox_area(box):
    if box is None:
        return 0.0
    _, _, w, h = box
    return max(0.0, float(w)) * max(0.0, float(h))


def _bbox_match(a_box, b_box, kind):
    if a_box is None or b_box is None:
        return False
    ax, ay, aw, ah = a_box
    bx, by, bw, bh = b_box
    acx, acy = ax + aw / 2.0, ay + ah / 2.0
    bcx, bcy = bx + bw / 2.0, by + bh / 2.0
    tol = RAW_KIND_TOLERANCE.get(kind, RAW_KIND_TOLERANCE["patch"])
    iou = _bbox_iou(_dilate_for_iou(a_box, kind), _dilate_for_iou(b_box, kind))
    center_dx = abs(acx - bcx)
    center_dy = abs(acy - bcy)
    size_dw = abs(aw - bw)
    size_dh = abs(ah - bh)
    if center_dx <= tol["center"] and center_dy <= tol["center"] and size_dw <= tol["size"] and size_dh <= tol["size"]:
        if iou >= tol["iou"]:
            return True
    if kind == "collection" and size_dw <= tol["size"] and size_dh <= tol["size"]:
        x_overlap = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
        y_overlap = max(0.0, min(ay + ah, by + bh) - max(ay, by))
        min_w = min(aw, bw)
        min_h = min(ah, bh)
        x_overlap_ratio = x_overlap / min_w if min_w > 1e-9 else 0.0
        y_overlap_ratio = y_overlap / min_h if min_h > 1e-9 else 0.0
        if x_overlap_ratio >= 0.95 and y_overlap_ratio >= 0.90 and iou >= 0.70:
            return True
    if iou >= tol["iou"]:
        if center_dx <= tol["center"] and center_dy <= tol["center"]:
            return True
    if _is_line_like(a_box) and _is_line_like(b_box):
        orient_a = _line_orientation(a_box)
        orient_b = _line_orientation(b_box)
        if orient_a != orient_b:
            return False
        if orient_a == "vertical":
            return abs(acx - bcx) <= max(tol["center"], 0.03)
        if orient_a == "horizontal":
            return abs(acy - bcy) <= max(tol["center"], 0.03)
        return abs(acx - bcx) <= max(tol["center"], 0.03) and abs(acy - bcy) <= max(tol["center"], 0.03)
    return False


def _round_json(value):
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, list):
        return [_round_json(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_round_json(v) for v in value)
    return value


def _style_equal(a, b):
    return _round_json(a) == _round_json(b)


def _close_scalar(a, b, tol=1e-3):
    if a is None or b is None:
        return True
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return _round_json(a) == _round_json(b)


def _close_point(a, b, tol=1e-3):
    if a is None or b is None:
        return True
    if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) >= 2 and len(b) >= 2):
        return _round_json(a) == _round_json(b)
    try:
        return abs(float(a[0]) - float(b[0])) <= tol and abs(float(a[1]) - float(b[1])) <= tol
    except Exception:
        return _round_json(a) == _round_json(b)


def _is_marker_only_line(record):
    if record.get("kind") not in {"line", "figure_line"}:
        return False
    if str(record.get("linestyle")) not in MARKER_NONE_VALUES:
        return False
    if str(record.get("marker")) in MARKER_NONE_VALUES:
        return False
    return int(record.get("n_points") or 0) > 1


def _is_path_collection(record):
    return record.get("kind") == "collection" and record.get("type") == "PathCollection"


def _is_marker_line_path_collection_match(line_record, collection_record):
    if not _is_marker_only_line(line_record) or not _is_path_collection(collection_record):
        return False
    sig_a = line_record.get("point_signature")
    sig_b = collection_record.get("offset_signature")
    if sig_a is None or sig_b is None:
        return False
    if _round_json(sig_a) != _round_json(sig_b):
        return False
    count_a = int(line_record.get("n_points") or 0)
    count_b = int(collection_record.get("item_count") or 0)
    if count_a > 0 and count_b > 0 and count_a != count_b:
        return False
    a_ax = line_record.get("ax_index")
    b_ax = collection_record.get("ax_index")
    if a_ax is not None and b_ax is not None and a_ax != b_ax:
        if not _same_container_axes(line_record, collection_record):
            return False
    return True


def _match_text_records(a, b):
    if _normalize_text(a.get("content")) != _normalize_text(b.get("content")):
        return False
    role_a = a.get("text_role")
    role_b = b.get("text_role")
    if role_a and role_b and role_a != role_b:
        return False
    if role_a in {"x_tick_label", "y_tick_label"} and role_a == role_b:
        return True
    return _bbox_match(_best_bbox(a), _best_bbox(b), "text")


def _match_annotation_bbox_records(a, b):
    if _normalize_text(a.get("text_content")) != _normalize_text(b.get("text_content")):
        return False
    return _bbox_match(_best_bbox(a), _best_bbox(b), "annotation_bbox")


def _match_annotation_arrow_records(a, b):
    text_a = _normalize_text(a.get("text_content"))
    text_b = _normalize_text(b.get("text_content"))
    if text_a and text_b and text_a != text_b:
        return False
    style_a = a.get("arrow_style")
    style_b = b.get("arrow_style")
    if style_a and style_b and str(style_a) != str(style_b):
        return False
    return _bbox_match(_best_bbox(a), _best_bbox(b), "annotation_arrow")


def _match_line_records(a, b, *, ignore_visual_color=False):
    if str(a.get("linestyle")) != str(b.get("linestyle")):
        return False
    if str(a.get("marker")) != str(b.get("marker")):
        return False
    label_a = _normalize_text(a.get("label"))
    label_b = _normalize_text(b.get("label"))
    if label_a and label_b and label_a != label_b:
        return False
    n_points_a = a.get("n_points")
    n_points_b = b.get("n_points")
    if (
        isinstance(n_points_a, int)
        and isinstance(n_points_b, int)
        and min(n_points_a, n_points_b) <= 8
        and abs(n_points_a - n_points_b) > 1
    ):
        return False
    if not ignore_visual_color:
        color_a = a.get("color")
        color_b = b.get("color")
        if color_a is not None and color_b is not None and not _style_equal(color_a, color_b):
            return False
    sig_a = a.get("data_signature")
    sig_b = b.get("data_signature")
    if sig_a is not None and sig_b is not None and _round_json(sig_a) == _round_json(sig_b):
        return True
    return _bbox_match(_best_bbox(a), _best_bbox(b), a.get("kind"))


def _match_patch_records(a, b):
    if a.get("type") != b.get("type"):
        return False
    if bool(a.get("fill")) != bool(b.get("fill")):
        return False
    if a.get("type") == "Wedge":
        if not _close_point(a.get("center"), b.get("center"), tol=1e-3):
            return False
        if not _close_scalar(a.get("r"), b.get("r"), tol=1e-3):
            return False
        if not _close_scalar(a.get("theta1"), b.get("theta1"), tol=1e-3):
            return False
        if not _close_scalar(a.get("theta2"), b.get("theta2"), tol=1e-3):
            return False
        if not _close_scalar(a.get("width"), b.get("width"), tol=1e-3):
            return False
    return _bbox_match(_best_bbox(a), _best_bbox(b), "patch")


def _match_collection_records(a, b):
    if a.get("type") != b.get("type"):
        return False
    count_a = a.get("item_count")
    count_b = b.get("item_count")
    if (
        isinstance(count_a, int)
        and isinstance(count_b, int)
        and min(count_a, count_b) <= 12
        and abs(count_a - count_b) > 1
    ):
        return False
    if a.get("type") == "PathCollection":
        sig_a = a.get("offset_signature")
        sig_b = b.get("offset_signature")
        if sig_a is not None and sig_b is not None:
            return _round_json(sig_a) == _round_json(sig_b)
    if a.get("type") in GROUP_COLLECTION_TYPES:
        sig_a = a.get("path_signature")
        sig_b = b.get("path_signature")
        if sig_a is not None and sig_b is not None:
            return _round_json(sig_a) == _round_json(sig_b)
    return _bbox_match(_best_bbox(a), _best_bbox(b), "collection")


def _match_table_cell_records(a, b):
    if _normalize_text(a.get("content")) != _normalize_text(b.get("content")):
        return False
    return _bbox_match(_best_bbox(a), _best_bbox(b), "table_cell")


def _records_match(a, b, *, ignore_visual_color=False):
    kind_a = a.get("kind")
    kind_b = b.get("kind")

    if kind_a != kind_b:
        if _is_marker_line_path_collection_match(a, b) or _is_marker_line_path_collection_match(b, a):
            return True
        return False

    kind = kind_a

    a_ax = a.get("ax_index")
    b_ax = b.get("ax_index")
    if a_ax is not None and b_ax is not None and a_ax != b_ax:
        if not _same_container_axes(a, b):
            return False

    if kind == "text":
        return _match_text_records(a, b)
    if kind == "annotation_bbox":
        return _match_annotation_bbox_records(a, b)
    if kind == "annotation_arrow":
        return _match_annotation_arrow_records(a, b)
    if kind in {"line", "figure_line"}:
        return _match_line_records(a, b, ignore_visual_color=ignore_visual_color)
    if kind == "patch":
        return _match_patch_records(a, b)
    if kind == "collection":
        return _match_collection_records(a, b)
    if kind == "table_cell":
        return _match_table_cell_records(a, b)

    return _bbox_match(_best_bbox(a), _best_bbox(b), kind)


def _group_context_key(record):
    kind = record.get("kind")
    rtype = record.get("type")
    ax_index = record.get("ax_index")
    if ax_index is not None:
        return (kind, rtype, ax_index)
    container = _container_axes_bbox(record)
    if container is not None:
        container = tuple(round(float(v), 3) for v in container)
    return (kind, rtype, None, container)


def _bbox_span_ratios(part_box, whole_box):
    if part_box is None or whole_box is None:
        return None, None
    _, _, part_w, part_h = part_box
    _, _, whole_w, whole_h = whole_box
    width_ratio = (part_w / whole_w) if whole_w > 1e-9 else 0.0
    height_ratio = (part_h / whole_h) if whole_h > 1e-9 else 0.0
    return width_ratio, height_ratio


def _is_small_fragment_box(part_box, whole_box):
    if part_box is None or whole_box is None:
        return False
    px, py, pw, ph = part_box
    wx, wy, ww, wh = whole_box
    part_area = _bbox_area(part_box)
    whole_area = _bbox_area(whole_box)
    if part_area <= 1e-9 or whole_area <= 1e-9:
        return False
    area_ratio = part_area / whole_area
    width_ratio, height_ratio = _bbox_span_ratios(part_box, whole_box)
    if width_ratio is None or height_ratio is None:
        return False
    if area_ratio > 0.35:
        return False
    if height_ratio <= 0.35 and width_ratio >= 0.9:
        return False
    touch_tol = 0.05
    touches_x_boundary = abs(px - wx) <= touch_tol or abs((px + pw) - (wx + ww)) <= touch_tol
    touches_y_boundary = abs(py - wy) <= touch_tol or abs((py + ph) - (wy + wh)) <= touch_tol
    if width_ratio <= 0.35 and touches_x_boundary:
        return True
    if height_ratio <= 0.35 and touches_y_boundary:
        return True
    return False


def _dominant_and_fragment_boxes(gt_records, removed_box):
    if not gt_records or removed_box is None:
        return None, None
    scored = []
    for record in gt_records:
        gt_box = _best_bbox(record)
        if gt_box is None:
            continue
        score = _bbox_score(gt_box, removed_box, record.get("kind"))
        scored.append((score, record, gt_box))
    if not scored:
        return None, None
    scored.sort(key=lambda item: item[0])
    base_record, base_box = scored[0][1], scored[0][2]
    fragment_boxes = []
    for _, record, gt_box in scored[1:]:
        if not _is_small_fragment_box(gt_box, removed_box):
            return None, None
        fragment_boxes.append((record, gt_box))
    if not fragment_boxes:
        return None, None
    return (base_record, base_box), fragment_boxes


def _collection_group_candidates(gt_records, removed_record):
    removed_box = _best_bbox(removed_record)
    if removed_box is None:
        return []
    candidates = []
    for record in gt_records:
        if record.get("kind") != "collection":
            continue
        if record.get("type") != removed_record.get("type"):
            continue
        if record.get("type") not in GROUP_COLLECTION_TYPES:
            continue
        gt_box = _best_bbox(record)
        if gt_box is None:
            continue
        if _bbox_contains(removed_box, gt_box, tol=0.05) or _boxes_overlap(gt_box, removed_box):
            candidates.append(record)
    return candidates


def _collection_group_match(gt_records, removed_record):
    if len(gt_records) < 2:
        return False
    removed_box = _best_bbox(removed_record)
    if removed_box is None:
        return False
    base_info, fragment_boxes = _dominant_and_fragment_boxes(gt_records, removed_box)
    if base_info is None or fragment_boxes is None:
        return False
    gt_boxes = [_best_bbox(record) for record in gt_records]
    gt_union_box = _union_bbox(gt_boxes)
    if gt_union_box is None:
        return False
    if _bbox_match(gt_union_box, removed_box, "collection"):
        return True
    iou = _bbox_iou(_dilate_for_iou(gt_union_box, "collection"), _dilate_for_iou(removed_box, "collection"))
    gt_area = _bbox_area(gt_union_box)
    removed_area = _bbox_area(removed_box)
    coverage = gt_area / removed_area if removed_area > 1e-9 else 0.0
    return _bbox_contains(removed_box, gt_union_box, tol=0.05) and iou >= 0.45 and coverage >= 0.65


def _line_group_candidates(gt_records, removed_record):
    removed_box = _best_bbox(removed_record)
    if removed_box is None:
        return []
    if int(removed_record.get("n_points") or 0) < 3:
        return []
    candidates = []
    removed_kind = removed_record.get("kind")
    removed_linestyle = str(removed_record.get("linestyle"))
    removed_marker = str(removed_record.get("marker"))
    removed_label = removed_record.get("label")
    for record in gt_records:
        if record.get("kind") != removed_kind:
            continue
        if int(record.get("n_points") or 0) < 3:
            continue
        if str(record.get("marker")) != removed_marker:
            continue
        if str(record.get("linestyle")) != removed_linestyle:
            continue
        if record.get("label") != removed_label:
            continue
        gt_box = _best_bbox(record)
        if gt_box is None:
            continue
        if _bbox_contains(removed_box, gt_box, tol=0.05) or _boxes_overlap(gt_box, removed_box):
            candidates.append(record)
    return candidates


def _line_group_match(gt_records, removed_record):
    if len(gt_records) < 2:
        return False
    removed_box = _best_bbox(removed_record)
    if removed_box is None:
        return False
    base_info, fragment_boxes = _dominant_and_fragment_boxes(gt_records, removed_box)
    if base_info is None or fragment_boxes is None:
        return False
    gt_boxes = [_best_bbox(record) for record in gt_records]
    gt_union_box = _union_bbox(gt_boxes)
    if gt_union_box is None:
        return False
    if _bbox_match(gt_union_box, removed_box, removed_record.get("kind")):
        return True
    iou = _bbox_iou(
        _dilate_for_iou(gt_union_box, removed_record.get("kind")),
        _dilate_for_iou(removed_box, removed_record.get("kind")),
    )
    gt_area = _bbox_area(gt_union_box)
    removed_area = _bbox_area(removed_box)
    coverage = gt_area / removed_area if removed_area > 1e-9 else 0.0
    return _bbox_contains(removed_box, gt_union_box, tol=0.05) and iou >= 0.45 and coverage >= 0.65


def _build_group_matches(unmatched_gt_records, unmatched_removed_records):
    gt_groups = {}
    for record in unmatched_gt_records:
        if record.get("kind") == "collection":
            if record.get("type") not in GROUP_COLLECTION_TYPES:
                continue
        elif record.get("kind") in GROUP_LINE_KINDS:
            if str(record.get("marker")) not in {"None", "none", "", " ", "null"}:
                continue
        else:
            continue
        gt_groups.setdefault(_group_context_key(record), []).append(record)

    removed_groups = {}
    for removed_idx, record in unmatched_removed_records:
        if record.get("kind") == "collection":
            if record.get("type") not in GROUP_COLLECTION_TYPES:
                continue
        elif record.get("kind") in GROUP_LINE_KINDS:
            if str(record.get("marker")) not in {"None", "none", "", " ", "null"}:
                continue
        else:
            continue
        removed_groups.setdefault(_group_context_key(record), []).append((removed_idx, record))

    group_matched_pairs = []
    used_gt_artist_ids = set()
    used_removed_indexes = set()

    for key, gt_records in gt_groups.items():
        removed_candidates = removed_groups.get(key, [])
        if not removed_candidates:
            continue
        for removed_idx, removed_record in removed_candidates:
            if removed_idx in used_removed_indexes:
                continue
            if removed_record.get("kind") == "collection":
                raw_candidates = _collection_group_candidates(gt_records, removed_record)
                group_ok = _collection_group_match
            else:
                raw_candidates = _line_group_candidates(gt_records, removed_record)
                group_ok = _line_group_match
            candidates = [
                record
                for record in raw_candidates
                if record.get("artist_id") is not None and int(record.get("artist_id")) not in used_gt_artist_ids
            ]
            if not group_ok(candidates, removed_record):
                continue
            group_matched_pairs.append((list(candidates), [removed_record]))
            for record in candidates:
                artist_id = record.get("artist_id")
                if artist_id is not None:
                    used_gt_artist_ids.add(int(artist_id))
            used_removed_indexes.add(removed_idx)
            break

    return group_matched_pairs, used_gt_artist_ids, used_removed_indexes


def _freeze_key(value):
    """Recursively convert lists/dicts to hashable tuples.

    `a == b` (in the sense the linear scan used) holds iff the frozen keys are
    equal, so index lookups stay semantically identical to the old `!=` checks.
    """
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_key(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((str(k), _freeze_key(v)) for k, v in value.items()))
    return value


def _index_by_key(records_with_idx, key_fn):
    """Group `(idx, record)` pairs by an exact-match key (records keep list order)."""
    index = {}
    for removed_idx, removed_record in records_with_idx:
        key = key_fn(removed_record)
        if key is None:
            continue
        index.setdefault(_freeze_key(key), []).append((removed_idx, removed_record))
    return index


def _kind_key_fn(kind):
    if kind == "text":
        return _text_match_key
    if kind in {"line", "figure_line"}:
        return _line_match_key
    if kind == "collection":
        return _collection_match_key
    return None


def _key_index_hits(record, kind_key_index):
    """Exact-key candidates for `record`, ordered exactly like a linear scan of
    the same-kind removed list would produce (an O(1) hash lookup instead)."""
    kind = record.get("kind")
    key_fn = _kind_key_fn(kind)
    index = kind_key_index.get(kind)
    if key_fn is None or not index:
        return ()
    key = key_fn(record)
    if key is None:
        return ()
    return index.get(_freeze_key(key), ())


def _text_match_key(record):
    if record.get("kind") != "text":
        return None
    content = _normalize_text(record.get("content"))
    if not content:
        return None
    return (record.get("text_role"), content)


def _line_match_key(record):
    if record.get("kind") not in {"line", "figure_line"}:
        return None
    sig = record.get("data_signature")
    if not sig:
        return None
    return (
        record.get("kind"),
        str(record.get("linestyle")),
        str(record.get("marker")),
        _normalize_text(record.get("label")),
        tuple(_round_json(sig)),
    )


def _collection_match_key(record):
    if record.get("kind") != "collection":
        return None
    rtype = record.get("type")
    if rtype == "PathCollection":
        sig = record.get("offset_signature")
        if sig:
            return ("PathCollection", tuple(_round_json(sig)))
        return None
    if rtype in GROUP_COLLECTION_TYPES:
        sig = record.get("path_signature")
        if sig:
            return (rtype, tuple(tuple(tuple(pt) for pt in path) for path in _round_json(sig)))
    return None


def build_raw_diff_result(gt_records, removed_records, *, ignore_visual_color=False):
    pre_group_matched_pairs, pre_group_used_gt_artist_ids, pre_group_used_removed_indexes = _build_group_matches(
        gt_records,
        list(enumerate(removed_records)),
    )
    used_removed_indexes = set(pre_group_used_removed_indexes)
    unmatched = []
    matched_pairs = []
    group_matched_pairs = list(pre_group_matched_pairs)

    removed_by_kind = {}
    for idx, record in enumerate(removed_records):
        removed_by_kind.setdefault(record.get("kind"), []).append((idx, record))

    # Hash indexes for the exact-key fast paths below; semantically identical
    # to scanning `removed_by_kind[kind]` linearly (lists keep removed order),
    # but O(1) per lookup instead of O(n) — critical for DOM-scale inputs
    # (D3/SVG charts extract ~10k records).
    kind_key_index = {}
    for kind, pairs in removed_by_kind.items():
        key_fn = _kind_key_fn(kind)
        if key_fn is not None:
            kind_key_index[kind] = _index_by_key(pairs, key_fn)

    for record in gt_records:
        artist_id = record.get("artist_id")
        if artist_id is not None and int(artist_id) in pre_group_used_gt_artist_ids:
            continue
        kind = record.get("kind")
        candidates = list(removed_by_kind.get(kind, []))
        if kind in {"line", "figure_line"}:
            marker = str(record.get("marker"))
            linestyle = str(record.get("linestyle"))
            if marker not in {"None", "none", "", None} and linestyle in {"None", "none", "", None}:
                candidates.extend(removed_by_kind.get("collection", []))
        elif kind == "collection" and record.get("type") == "PathCollection":
            candidates.extend(removed_by_kind.get("line", []))
        best_idx = None
        best_score = math.inf

        if kind in {"text", "line", "figure_line", "collection"}:
            exact_candidates = []
            a_ax = record.get("ax_index")
            for removed_idx, removed_record in _key_index_hits(record, kind_key_index):
                if removed_idx in used_removed_indexes:
                    continue
                b_ax = removed_record.get("ax_index")
                if a_ax is not None and b_ax is not None and a_ax != b_ax:
                    if not _same_container_axes(record, removed_record):
                        continue
                exact_candidates.append((removed_idx, removed_record))
            if len(exact_candidates) == 1:
                removed_idx, removed_record = exact_candidates[0]
                used_removed_indexes.add(removed_idx)
                matched_pairs.append((record, removed_record))
                continue

        for removed_idx, removed_record in candidates:
            if removed_idx in used_removed_indexes:
                continue
            if not _records_match(
                record,
                removed_record,
                ignore_visual_color=ignore_visual_color,
            ):
                continue
            score = _bbox_score(_best_bbox(record), _best_bbox(removed_record), kind)
            if score < best_score:
                best_score = score
                best_idx = removed_idx

        if best_idx is None:
            unmatched.append(record)
        else:
            used_removed_indexes.add(best_idx)
            matched_pairs.append((record, removed_records[best_idx]))

    unmatched_removed_records = [
        (idx, record)
        for idx, record in enumerate(removed_records)
        if idx not in used_removed_indexes
    ]
    fallback_group_matched_pairs, group_used_gt_artist_ids, group_used_removed_indexes = _build_group_matches(
        unmatched,
        unmatched_removed_records,
    )
    group_matched_pairs.extend(fallback_group_matched_pairs)
    if group_used_gt_artist_ids:
        unmatched = [
            record
            for record in unmatched
            if record.get("artist_id") is None or int(record.get("artist_id")) not in group_used_gt_artist_ids
        ]
    used_removed_indexes.update(group_used_removed_indexes)

    gt_axes_records = [record for record in gt_records if record.get("kind") == "axes"]
    allowed_artist_ids = set()
    allowed_axes_ids = set()
    unmatched_axes_ids = set()
    for record in unmatched:
        artist_id = record.get("artist_id")
        if artist_id is not None:
            allowed_artist_ids.add(int(artist_id))
        ax_artist_id = record.get("ax_artist_id")
        if ax_artist_id is not None:
            allowed_axes_ids.add(int(ax_artist_id))
        if record.get("kind") == "axes" and artist_id is not None:
            allowed_axes_ids.add(int(artist_id))
            record_box = _best_bbox(record)
            record_area = _bbox_area(record_box)
            for other in gt_axes_records:
                other_id = other.get("artist_id")
                if other_id is None or int(other_id) == int(artist_id):
                    continue
                other_box = _best_bbox(other)
                other_area = _bbox_area(other_box)
                if record_area <= 0.0 or other_area <= 0.0:
                    continue
                if not _boxes_overlap(record_box, other_box):
                    continue
                if record_area >= other_area * 0.95:
                    continue
                unmatched_axes_ids.add(int(artist_id))
                break

    return {
        "unmatched_records": unmatched,
        "allowed_artist_ids": allowed_artist_ids,
        "allowed_axes_ids": allowed_axes_ids,
        "unmatched_axes_ids": unmatched_axes_ids,
        "matched_pairs": matched_pairs,
        "group_matched_pairs": group_matched_pairs,
    }


def to_jsonable_raw_diff(diff_result):
    return {
        "unmatched_records": diff_result.get("unmatched_records", []),
        "allowed_artist_ids": sorted(diff_result.get("allowed_artist_ids", [])),
        "allowed_axes_ids": sorted(diff_result.get("allowed_axes_ids", [])),
        "unmatched_axes_ids": sorted(diff_result.get("unmatched_axes_ids", [])),
        "matched_pairs": [
            {"gt": gt, "removed": removed}
            for gt, removed in diff_result.get("matched_pairs", [])
        ],
        "group_matched_pairs": [
            {"gt": gt_records, "removed": removed_records}
            for gt_records, removed_records in diff_result.get("group_matched_pairs", [])
        ],
    }


def raw_records_to_jsonable(records):
    return json.loads(json.dumps(records))
