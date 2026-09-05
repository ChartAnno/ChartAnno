"""Map diffed D3/SVG raw objects to the seven ChartAnno semantic categories."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable


FEATURE_KEYS = (
    "1_enclosure",
    "2_connector",
    "3_text",
    "4_glyph",
    "5_color",
    "6_indicator",
    "7_geometric",
)


ROLE_TO_FEATURE = {
    "enclosure": "1_enclosure",
    "connector": "2_connector",
    "arrow": "2_connector",
    "text": "3_text",
    "glyph": "4_glyph",
    "color": "5_color",
    "indicator": "6_indicator",
    "reference-line": "6_indicator",
    "geometric": "7_geometric",
    "inset": "7_geometric",
    "zoom": "7_geometric",
}


def new_annotation_ir() -> dict[str, list]:
    return {key: [] for key in FEATURE_KEYS}


def _feature_from_record(record: dict) -> dict:
    item = {
        "src": f"d3_{record.get('semantic_role') or record.get('tag') or record.get('kind')}",
    }
    for key in (
        "ax_index",
        "bbox",
        "axes_bbox",
        "color",
        "facecolor",
        "edgecolor",
        "alpha",
        "linestyle",
        "marker",
        "type",
        "content",
        "text_content",
    ):
        if record.get(key) is not None:
            item[key] = record[key]
    box = record.get("axes_bbox") or record.get("bbox")
    if isinstance(box, (list, tuple)) and len(box) == 4:
        width, height = float(box[2]), float(box[3])
        if width <= 0.01 and height > 0.01:
            item["orientation"] = "vertical"
        elif height <= 0.01 and width > 0.01:
            item["orientation"] = "horizontal"
    return item


def _infer_feature_key(record: dict) -> str | None:
    role = str(record.get("semantic_role") or "").strip().lower()
    if role in {"indicator-marker", "legend-key", "axis-decoration"}:
        return None
    if role in ROLE_TO_FEATURE:
        return ROLE_TO_FEATURE[role]

    kind = record.get("kind")
    tag = record.get("tag")
    if kind == "text":
        return "3_text"
    if kind == "annotation_bbox":
        return "1_enclosure"
    if kind == "annotation_arrow":
        return "2_connector"
    if kind == "axes":
        return "7_geometric"
    if kind == "line":
        box = record.get("axes_bbox") or record.get("bbox")
        if isinstance(box, (list, tuple)) and len(box) == 4:
            width, height = float(box[2]), float(box[3])
            if bool(record.get("is_full_span")) or (width >= 0.8 and height <= 0.03) or (height >= 0.8 and width <= 0.03):
                return "6_indicator"
            if int(record.get("n_points") or 0) >= 3 and (width >= 0.3 or height >= 0.3):
                return "6_indicator"
            if record.get("ax_index") is None:
                return "2_connector"
        return "2_connector"
    if kind == "collection" and record.get("type") == "PathCollection":
        return "4_glyph"
    if kind == "patch":
        if tag in {"circle", "ellipse", "polygon"}:
            return "4_glyph"
        box = record.get("axes_bbox") or record.get("bbox")
        if isinstance(box, (list, tuple)) and len(box) == 4:
            width, height = float(box[2]), float(box[3])
            if min(width, height) <= 0.03 and max(width, height) >= 0.04:
                return "2_connector"
        if not bool(record.get("fill")) and record.get("edgecolor") is not None:
            return "1_enclosure"
        if bool(record.get("fill")) and isinstance(box, (list, tuple)) and len(box) == 4:
            width, height = float(box[2]), float(box[3])
            if min(width, height) <= 0.08 and 0.04 <= max(width, height) <= 0.15:
                return "2_connector"
            if width >= 0.2 and height >= 0.2:
                return "1_enclosure"
    return None


def _box(record: dict, key: str):
    value = record.get(key)
    if not (isinstance(value, (list, tuple)) and len(value) == 4):
        return None
    try:
        return tuple(float(part) for part in value)
    except (TypeError, ValueError):
        return None


def _union_box(records: Iterable[dict], key: str):
    boxes = [box for record in records if (box := _box(record, key)) is not None]
    if not boxes:
        return None
    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[0] + box[2] for box in boxes)
    y1 = max(box[1] + box[3] for box in boxes)
    return [x0, y0, x1 - x0, y1 - y0]


def _line_rectangle_groups(records: list[dict], excluded_axes: set) -> tuple[list[dict], set[int]]:
    """Merge four axis-aligned Matplotlib line2d sides into one enclosure."""

    merged = []
    consumed: set[int] = set()
    axes_indexes = {
        record.get("ax_index")
        for record in records
        if record.get("kind") == "line" and record.get("ax_index") not in excluded_axes
    }
    for ax_index in axes_indexes:
        candidates = [
            record
            for record in records
            if record.get("kind") == "line"
            and record.get("ax_index") == ax_index
            and str(record.get("semantic_role") or "") != "axis-decoration"
            and _box(record, "axes_bbox") is not None
        ]
        horizontals = [record for record in candidates if _box(record, "axes_bbox")[3] <= 0.01 and _box(record, "axes_bbox")[2] >= 0.05]
        verticals = [record for record in candidates if _box(record, "axes_bbox")[2] <= 0.01 and _box(record, "axes_bbox")[3] >= 0.05]
        for h_pair in combinations(horizontals, 2):
            h0, h1 = [_box(record, "axes_bbox") for record in h_pair]
            if abs(h0[0] - h1[0]) > 0.015 or abs(h0[2] - h1[2]) > 0.015:
                continue
            y0, y1 = sorted((h0[1], h1[1]))
            if y1 - y0 < 0.03:
                continue
            x0, x1 = h0[0], h0[0] + h0[2]
            for v_pair in combinations(verticals, 2):
                v0, v1 = [_box(record, "axes_bbox") for record in v_pair]
                vx0, vx1 = sorted((v0[0], v1[0]))
                if abs(v0[1] - v1[1]) > 0.015 or abs(v0[3] - v1[3]) > 0.015:
                    continue
                if abs(vx0 - x0) > 0.015 or abs(vx1 - x1) > 0.015:
                    continue
                if abs(v0[1] - y0) > 0.015 or abs((v0[1] + v0[3]) - y1) > 0.015:
                    continue
                group = list(h_pair) + list(v_pair)
                ids = {int(record["artist_id"]) for record in group if record.get("artist_id") is not None}
                if ids & consumed:
                    continue
                enclosure = dict(group[0])
                enclosure["kind"] = "annotation_bbox"
                enclosure["semantic_role"] = "enclosure"
                enclosure["bbox"] = _union_box(group, "bbox")
                enclosure["axes_bbox"] = _union_box(group, "axes_bbox")
                merged.append(enclosure)
                consumed.update(ids)
    return merged, consumed


def _merge_text_paragraphs(text_items: list[dict]) -> list[dict]:
    if not text_items or len(text_items) <= 1:
        return text_items

    def get_box(item):
        return item.get("axes_bbox") or item.get("bbox")

    items_with_box = [item for item in text_items if get_box(item) is not None]
    items_no_box = [item for item in text_items if get_box(item) is None]

    if not items_with_box:
        return text_items

    sorted_items = sorted(items_with_box, key=lambda it: (-get_box(it)[1], get_box(it)[0]))

    clusters: list[list[dict]] = []
    for item in sorted_items:
        box = get_box(item)
        text_str = str(item.get("content") or item.get("text_content") or "").strip()
        is_short_label = len(text_str) <= 2 and not text_str.startswith("•")

        placed = False
        if not is_short_label:
            for cluster in clusters:
                last_item = cluster[-1]
                last_box = get_box(last_item)
                last_str = str(last_item.get("content") or last_item.get("text_content") or "").strip()
                if len(last_str) <= 2 and not last_str.startswith("•"):
                    continue

                x_close = (
                    abs(box[0] - last_box[0]) <= 0.08
                    or abs((box[0] + box[2] / 2.0) - (last_box[0] + last_box[2] / 2.0)) <= 0.08
                )
                y_gap = abs(last_box[1] - (box[1] + box[3]))
                if y_gap > 0.05:
                    y_gap = abs(box[1] - (last_box[1] + last_box[3]))

                if x_close and y_gap <= 0.05:
                    cluster.append(item)
                    placed = True
                    break

        if not placed:
            clusters.append([item])

    merged_items = list(items_no_box)
    for cluster in clusters:
        if len(cluster) == 1:
            merged_items.append(cluster[0])
        else:
            base = dict(cluster[0])
            contents = []
            for it in cluster:
                txt = str(it.get("content") or it.get("text_content") or "").strip()
                if txt:
                    contents.append(txt)
            merged_content = " ".join(contents)
            if "content" in base:
                base["content"] = merged_content
            if "text_content" in base:
                base["text_content"] = merged_content

            boxes = [get_box(it) for it in cluster if get_box(it) is not None]
            if boxes:
                x0 = min(b[0] for b in boxes)
                y0 = min(b[1] for b in boxes)
                x1 = max(b[0] + b[2] for b in boxes)
                y1 = max(b[1] + b[3] for b in boxes)
                merged_box = [x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)]
                if "axes_bbox" in base:
                    base["axes_bbox"] = merged_box
                if "bbox" in base:
                    base["bbox"] = merged_box
            merged_items.append(base)

    return merged_items


def classify_unmatched_records(records: Iterable[dict]) -> dict[str, list]:
    records = list(records)
    semantic = new_annotation_ir()
    # An unmatched nested axes is represented once as a geometric annotation.
    # Its child marks are rendering payload, not separate annotations.
    geometric_ax_indexes = {
        record.get("ax_index")
        for record in records
        if record.get("kind") == "axes" and record.get("ax_index") is not None
    }
    merged_enclosures, consumed_artist_ids = _line_rectangle_groups(records, geometric_ax_indexes)
    for record in merged_enclosures:
        semantic["1_enclosure"].append(_feature_from_record(record))
    for record in records:
        if record.get("kind") != "axes" and record.get("ax_index") in geometric_ax_indexes:
            continue
        artist_id = record.get("artist_id")
        if artist_id is not None and int(artist_id) in consumed_artist_ids:
            continue
        feature_key = _infer_feature_key(record)
        if feature_key is None:
            continue
        semantic[feature_key].append(_feature_from_record(record))

    if semantic.get("3_text"):
        semantic["3_text"] = _merge_text_paragraphs(semantic["3_text"])

    return semantic


def _primary_color(record: dict):
    kind = record.get("kind")
    if kind in {"line", "figure_line", "collection", "text", "annotation_arrow"}:
        return record.get("color")
    if kind in {"patch", "annotation_bbox", "table_cell"}:
        if record.get("fill") and record.get("facecolor") is not None:
            return record.get("facecolor")
        return record.get("edgecolor")
    return None


def append_matched_color_diffs(semantic: dict[str, list], matched_pairs: Iterable[tuple[dict, dict]]) -> None:
    seen = set()
    for candidate, baseline in matched_pairs:
        if str(candidate.get("semantic_role") or "") in {"legend-key", "axis-decoration"}:
            continue
        candidate_color = _primary_color(candidate)
        baseline_color = _primary_color(baseline)
        if candidate_color is None or baseline_color is None or candidate_color == baseline_color:
            continue
        key = (candidate.get("artist_id"), tuple(candidate_color))
        if key in seen:
            continue
        seen.add(key)
        item = _feature_from_record(candidate)
        item["src"] = f"d3_{candidate.get('kind')}_color_change"
        item["color"] = candidate_color
        semantic["5_color"].append(item)


__all__ = [
    "append_matched_color_diffs",
    "classify_unmatched_records",
    "new_annotation_ir",
]
