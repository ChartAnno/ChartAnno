import matplotlib.pyplot as plt

from annotation_eval.extraction.element_extractor import ChartAnnotationExtractor
from annotation_eval.extraction.runtime import load_chart_figures, save_rendered_figure
from annotation_eval.extraction.raw_diff import _best_bbox, _bbox_score, build_raw_diff_result
from annotation_eval.extraction.raw_elements import ChartRawElementExtractor


def _round_color_key(color):
    if isinstance(color, list):
        return tuple(round(float(v), 6) for v in color)
    if isinstance(color, tuple):
        return tuple(round(float(v), 6) for v in color)
    return color


def _bbox_key(box):
    if isinstance(box, list):
        return tuple(box)
    return box


def _normalize_color_value(color):
    key = _round_color_key(color)
    if not isinstance(key, tuple):
        return None
    if len(key) < 3:
        return None
    try:
        rgba = tuple(float(v) for v in key[:4])
    except Exception:
        return None
    if len(rgba) == 4 and rgba[3] <= 0.0:
        return None
    return key


def _color_compare_key(color):
    value = _normalize_color_value(color)
    if value is None or len(value) < 3:
        return None
    return tuple(float(v) for v in value[:3])


def _record_primary_color(record):
    kind = record.get("kind")
    if kind in {"line", "figure_line", "collection", "text"}:
        return record.get("color"), "color"
    if kind == "table_cell":
        return record.get("facecolor"), "facecolor"
    if kind == "annotation_bbox":
        face = _normalize_color_value(record.get("facecolor"))
        if face is not None:
            return record.get("facecolor"), "facecolor"
        edge = _normalize_color_value(record.get("edgecolor"))
        if edge is not None:
            return record.get("edgecolor"), "edgecolor"
        return None, None
    if kind == "patch":
        face = _normalize_color_value(record.get("facecolor"))
        if bool(record.get("fill")) and face is not None:
            return record.get("facecolor"), "facecolor"
        edge = _normalize_color_value(record.get("edgecolor"))
        if edge is not None:
            return record.get("edgecolor"), "edgecolor"
        return None, None
    return None, None


def _is_semantic_line_color_candidate(record):
    if record.get("kind") not in {"line", "figure_line"}:
        return False
    try:
        n_points = int(record.get("n_points") or 0)
    except Exception:
        n_points = 0
    if n_points < 3:
        return False
    axes_bbox = record.get("axes_bbox")
    if isinstance(axes_bbox, (list, tuple)) and len(axes_bbox) == 4:
        try:
            _, _, w, h = [float(v) for v in axes_bbox]
        except Exception:
            return True
        if (w <= 0.03 and h >= 0.85) or (h <= 0.03 and w >= 0.85):
            return False
    return True


def _best_group_removed_record(gt_record, removed_records):
    best_removed = None
    best_score = float("inf")
    for removed_record in removed_records:
        score = _bbox_score(_best_bbox(gt_record), _best_bbox(removed_record), gt_record.get("kind"))
        if score < best_score:
            best_score = score
            best_removed = removed_record
    return best_removed


def _build_color_matches(raw_matched_pairs, group_matched_pairs):
    color_matches = []
    seen = set()

    merged_pairs = list(raw_matched_pairs)
    for gt_records, removed_records in group_matched_pairs:
        if not removed_records:
            continue
        for gt_record in gt_records:
            removed_record = _best_group_removed_record(gt_record, removed_records)
            if removed_record is None:
                continue
            merged_pairs.append((gt_record, removed_record))

    for gt_record, removed_record in merged_pairs:
        kind = gt_record.get("kind")
        gt_color_value, _ = _record_primary_color(gt_record)
        removed_color_value, _ = _record_primary_color(removed_record)
        gt_color = _color_compare_key(gt_color_value)
        removed_color = _color_compare_key(removed_color_value)
        if gt_color is None or removed_color is None or gt_color == removed_color:
            continue

        key = (
            gt_record.get("artist_id"),
            gt_record.get("kind"),
            gt_record.get("ax_index"),
            _bbox_key(gt_record.get("bbox")),
            _bbox_key(gt_record.get("axes_bbox")),
        )
        if key in seen:
            continue
        seen.add(key)
        color_matches.append((gt_record, removed_record))

    return color_matches


def _append_color_item(existing, semantic_data, item):
    dedupe_key = (
        item.get("src"),
        item.get("ax_index"),
        _bbox_key(item.get("bbox")),
        _bbox_key(item.get("axes_bbox")),
        _round_color_key(item.get("color")),
    )
    if dedupe_key in existing:
        return
    existing.add(dedupe_key)
    semantic_data.setdefault("5_color", []).append(item)


def _append_matched_color_diffs(semantic_data, color_matches):
    if not color_matches:
        return semantic_data

    existing = set()
    for item in semantic_data.get("5_color", []):
        if not isinstance(item, dict):
            continue
        existing.add(
            (
                item.get("src"),
                item.get("ax_index"),
                _bbox_key(item.get("bbox")),
                _bbox_key(item.get("axes_bbox")),
                _round_color_key(item.get("color")),
            )
        )

    for gt_record, removed_record in color_matches:
        kind = gt_record.get("kind")
        if kind in {"line", "figure_line"} and not _is_semantic_line_color_candidate(gt_record):
            continue
        gt_color_value, gt_color_attr = _record_primary_color(gt_record)
        removed_color_value, _ = _record_primary_color(removed_record)
        gt_color = _color_compare_key(gt_color_value)
        removed_color = _color_compare_key(removed_color_value)
        if gt_color is None or removed_color is None or gt_color == removed_color:
            continue

        bbox = gt_record.get("bbox")
        axes_bbox = gt_record.get("axes_bbox")
        if (
            kind == "collection"
            and gt_record.get("type") == "PathCollection"
            and isinstance(gt_record.get("point_boxes"), list)
            and len(gt_record.get("point_boxes")) > 0
            and int(gt_record.get("item_count") or 0) <= 20
        ):
            for point_box in gt_record.get("point_boxes", []):
                item = {
                    "src": f"{kind}_{gt_color_attr}_change",
                    "ax_index": gt_record.get("ax_index"),
                    "bbox": point_box.get("bbox"),
                    "axes_bbox": point_box.get("axes_bbox"),
                    "color": gt_color_value,
                    "type": gt_record.get("type"),
                }
                _append_color_item(existing, semantic_data, item)
            continue
        item = {
            "src": f"{kind}_{gt_color_attr}_change",
            "ax_index": gt_record.get("ax_index"),
            "bbox": bbox,
            "axes_bbox": axes_bbox,
            "color": gt_color_value,
        }
        if kind == "text":
            item["content"] = gt_record.get("content")
        elif kind == "annotation_bbox":
            item["text_content"] = gt_record.get("text_content")
        elif kind in {"line", "figure_line"}:
            item["linestyle"] = gt_record.get("linestyle")
            item["marker"] = gt_record.get("marker")
        elif kind == "collection":
            item["type"] = gt_record.get("type")
        elif kind == "patch":
            item["type"] = gt_record.get("type")
        elif kind == "table_cell":
            item["content"] = gt_record.get("content")
        elif kind == "annotation_arrow":
            item["text_content"] = gt_record.get("text_content")
            item["arrow_style"] = gt_record.get("arrow_style")

        _append_color_item(existing, semantic_data, item)

    return semantic_data


def extract_diffed_annotation_bundle(
    source_file: str,
    removed_file: str,
    project_root: str,
    *,
    render_output_path: str | None = None,
    removed_reference_file: str | None = None,
    ast_strip_grid_calls: bool = False,
    ast_remove_removed_draw_calls: bool = False,
):
    try:
        source_fig, render_fig = load_chart_figures(
            source_file,
            project_root,
            removed_reference_file=removed_reference_file,
            ast_strip_grid_calls=ast_strip_grid_calls,
            ast_remove_removed_draw_calls=ast_remove_removed_draw_calls,
        )
        if source_fig is None:
            return None

        removed_fig, _ = load_chart_figures(
            removed_file,
            project_root,
            ast_strip_grid_calls=ast_strip_grid_calls,
        )
        if removed_fig is None:
            return None

        gt_raw = ChartRawElementExtractor(source_fig).extract()
        removed_raw = ChartRawElementExtractor(removed_fig).extract()
        raw_diff = build_raw_diff_result(gt_raw, removed_raw, ignore_visual_color=True)
        color_matches = _build_color_matches(
            raw_diff.get("matched_pairs", []),
            raw_diff.get("group_matched_pairs", []),
        )
        diffed_extractor = ChartAnnotationExtractor(
            source_fig,
            allowed_artist_ids=raw_diff["allowed_artist_ids"],
            allowed_axes_ids=raw_diff["allowed_axes_ids"],
            geometric_axes_ids=raw_diff["unmatched_axes_ids"],
        )
        diffed_semantic = diffed_extractor.extract()
        diffed_semantic["5_color"] = []
        diffed_semantic = _append_matched_color_diffs(
            diffed_semantic,
            color_matches,
        )

        save_rendered_figure(render_fig, render_output_path)

        return {
            "gt_raw": gt_raw,
            "removed_raw": removed_raw,
            "raw_diff": raw_diff,
            "diffed_semantic": diffed_semantic,
        }
    finally:
        plt.close("all")
