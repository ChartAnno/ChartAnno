import matplotlib.collections as mcollections
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np


class IndicatorAnnotationMixin:
    def _is_light_color(self, color, threshold=0.92):
        try:
            r, g, b, _ = mcolors.to_rgba(color)
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            return luminance >= threshold
        except Exception:
            return False

    def _is_chromatic_color(self, color, threshold=0.08):
        try:
            r, g, b, _ = mcolors.to_rgba(color)
            return (max(r, g, b) - min(r, g, b)) >= threshold
        except Exception:
            return False

    def _get_line_label_text(self, line):
        try:
            label = line.get_label()
        except Exception:
            label = None
        if not isinstance(label, str):
            return None
        label = label.strip()
        if not label or label.startswith("_"):
            return None
        return label

    def _is_reference_like_label(self, label):
        if not isinstance(label, str):
            return False
        normalized = " ".join(label.lower().replace("-", " ").replace("_", " ").split())
        if not normalized:
            return False
        reference_terms = {
            "average",
            "avg",
            "mean",
            "median",
            "baseline",
            "reference",
            "benchmark",
            "target",
            "threshold",
        }
        tokens = set(normalized.split())
        return any(term in normalized for term in reference_terms) or bool(tokens & reference_terms)

    def _has_companion_main_line(self, ax, line_infos, current_idx):
        dashed_styles = {"--", ":", "-.", "dashed", "dotted", "dashdot"}
        for other_idx, other in enumerate(line_infos):
            if other_idx == current_idx:
                continue
            if int(other.get("n_points", 0)) < 3:
                continue
            other_line = ax.lines[other_idx]
            other_ls = str(other_line.get_linestyle()).strip().lower()
            if other_ls in dashed_styles:
                continue
            if self._is_light_color(other_line.get_color()):
                continue
            return True
        return False

    def _is_reference_indicator_line(
        self,
        ax,
        idx,
        line,
        info,
        line_infos,
        *,
        has_bar_like_rect,
        has_filled_area,
    ):
        if has_bar_like_rect:
            return False

        ls_normalized = str(line.get_linestyle()).strip().lower()
        dashed_styles = {"--", ":", "-.", "dashed", "dotted", "dashdot"}
        if ls_normalized not in dashed_styles:
            return False
        if line.get_marker() not in self.NONE_MARKERS:
            return False
        if int(info.get("n_points", 0)) < 3:
            return False
        if self._is_light_color(line.get_color()):
            return False

        label = self._get_line_label_text(line)
        if self._is_reference_like_label(label):
            return True

        if not bool(info.get("is_axis_spanning", False)):
            return False

        if has_filled_area and self._has_companion_main_line(ax, line_infos, idx):
            return True

        return False

    def _is_diffed_line_indicator_fallback(self, line, info):
        if self.allowed_artist_ids is None:
            return False
        if line.get_marker() not in self.NONE_MARKERS:
            return False
        if int(info.get("n_points", 0)) < 3:
            return False
        ls_normalized = str(line.get_linestyle()).strip().lower()
        if ls_normalized in {"none", "", "null"}:
            return False
        return True

    def _is_axis_aligned_bracket_polyline(self, xdata, ydata):
        try:
            x = np.asarray(xdata, dtype=float).ravel()
            y = np.asarray(ydata, dtype=float).ravel()
            if x.size < 3 or y.size < 3 or x.size != y.size:
                return False
            ux = np.unique(np.round(x, 6))
            uy = np.unique(np.round(y, 6))
            return len(ux) == 2 and len(uy) == 2
        except Exception:
            return False

    def _is_open_bracket_polyline(self, xdata, ydata):
        try:
            x = np.asarray(xdata, dtype=float).ravel()
            y = np.asarray(ydata, dtype=float).ravel()
            if x.size < 4 or y.size < 4 or x.size != y.size:
                return False
            dx = np.diff(x)
            dy = np.diff(y)
            step_is_axis_aligned = np.all(
                (np.isclose(dx, 0.0, atol=1e-9) & (~np.isclose(dy, 0.0, atol=1e-9)))
                | ((~np.isclose(dx, 0.0, atol=1e-9)) & np.isclose(dy, 0.0, atol=1e-9))
            )
            if not step_is_axis_aligned:
                return False
            ux = np.unique(np.round(x, 6))
            uy = np.unique(np.round(y, 6))
            return len(ux) <= 3 and len(uy) <= 3
        except Exception:
            return False

    def _is_brace_polyline(self, xdata, ydata):
        try:
            x = np.asarray(xdata, dtype=float).ravel()
            y = np.asarray(ydata, dtype=float).ravel()
            if x.size != 5 or y.size != 5:
                return False
            return (
                np.isclose(x[0], x[1], atol=1e-9)
                and np.isclose(x[3], x[4], atol=1e-9)
                and (x[1] < x[2] < x[3])
                and np.isclose(y[0], y[4], atol=1e-9)
                and np.isclose(y[1], y[3], atol=1e-9)
                and (y[0] < y[1] < y[2])
            )
        except Exception:
            return False

    def _is_arrow_like_polygon(self, patch):
        if not isinstance(patch, mpatches.Polygon):
            return False
        try:
            xy = np.asarray(patch.get_xy(), dtype=float)
        except Exception:
            return False
        if xy.ndim != 2 or xy.shape[0] < 3:
            return False
        pts = xy[:, :2]
        if pts.shape[0] > 1 and np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
        if pts.shape[0] == 3:
            return True
        if pts.shape[0] < 5 or pts.shape[0] > 8:
            return False

        rounded = np.round(pts, 6)
        ux, x_counts = np.unique(rounded[:, 0], return_counts=True)
        uy, y_counts = np.unique(rounded[:, 1], return_counts=True)

        def has_arrow_tier(levels, counts):
            if len(levels) < 3 or len(levels) > 4:
                return False
            singleton_levels = [
                levels[idx]
                for idx, count in enumerate(counts)
                if int(count) == 1
            ]
            if not singleton_levels:
                return False
            level_min = float(np.min(levels))
            level_max = float(np.max(levels))
            if not any(
                np.isclose(level, level_min) or np.isclose(level, level_max)
                for level in singleton_levels
            ):
                return False
            return int(np.max(counts)) >= 2

        return has_arrow_tier(ux, x_counts) or has_arrow_tier(uy, y_counts)

    def _is_bar_like_rectangle_bbox(self, bbox, ax_bw, ax_bh):
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            return False
        try:
            px, py, pw, ph = [float(v) for v in bbox]
        except Exception:
            return False
        if pw <= 0 or ax_bw <= 0 or ax_bh <= 0:
            return False
        ratio_w = pw / ax_bw
        ratio_h = max(0.0, ph) / ax_bh
        if (
            (ratio_h >= 0.85 and ratio_w >= 0.25)
            or (ratio_w >= 0.85 and ratio_h >= 0.15)
        ):
            return False
        is_vertical_bar = ratio_w <= 0.55 and 0.0 < ratio_h < 0.98
        is_horizontal_bar = ratio_h <= 0.22 and 0.0 < ratio_w < 0.98
        if abs(ax_bw - 1.0) < 1e-9 and abs(ax_bh - 1.0) < 1e-9:
            intersects_axis = (
                (px + pw) > 0.0
                and px < 1.0
                and (py + ph) > 0.0
                and py < 1.0
            )
            if intersects_axis:
                if ratio_h <= 0.22 and (px < 0.0 or (px + pw) > 1.0):
                    is_horizontal_bar = True
                if ratio_w <= 0.55 and (py < 0.0 or (py + ph) > 1.0):
                    is_vertical_bar = True
        return is_vertical_bar or is_horizontal_bar

    def _is_contextual_bar_like_rectangle_bbox(self, bbox, ax_bw, ax_bh, bar_like_axes_boxes=None):
        if not self._is_bar_like_rectangle_bbox(bbox, ax_bw, ax_bh):
            return False
        if not bar_like_axes_boxes:
            return True
        try:
            px, py, pw, ph = [float(v) for v in bbox]
        except Exception:
            return True
        valid_boxes = []
        for candidate in bar_like_axes_boxes:
            if not (isinstance(candidate, (list, tuple)) and len(candidate) == 4):
                continue
            try:
                valid_boxes.append(tuple(float(v) for v in candidate))
            except Exception:
                continue
        if len(valid_boxes) < 2:
            return True
        typical_width = float(np.median([max(0.0, bw) for _, _, bw, _ in valid_boxes]))
        typical_height = float(np.median([max(0.0, bh) for _, _, _, bh in valid_boxes]))
        if typical_width <= 0.0 or typical_height <= 0.0:
            return True

        def overlap_len(a0, a1, b0, b1):
            return max(0.0, min(a1, b1) - max(a0, b0))

        overlap_x_count = 0
        overlap_y_count = 0
        for bx, by, bw, bh in valid_boxes:
            if overlap_len(px, px + pw, bx, bx + bw) >= min(pw, bw) * 0.25:
                overlap_x_count += 1
            if overlap_len(py, py + ph, by, by + bh) >= min(ph, bh) * 0.25:
                overlap_y_count += 1

        spans_multiple_vertical_bars = (
            pw >= typical_width * 2.2 and ph >= typical_height * 0.3 and overlap_x_count >= 2
        )
        spans_multiple_horizontal_bars = (
            ph >= typical_height * 2.2 and pw >= typical_width * 0.7 and overlap_y_count >= 2
        )
        return not (spans_multiple_vertical_bars or spans_multiple_horizontal_bars)

    def _is_violin_like_axis(self, ax):
        try:
            collections = list(getattr(ax, "collections", []) or [])
        except Exception:
            return False

        body_boxes = []
        centers = []
        for coll in collections:
            if not isinstance(coll, mcollections.PolyCollection):
                continue
            if isinstance(coll, (mcollections.LineCollection, mcollections.PathCollection)):
                continue
            axes_bbox = self._get_artist_axes_bbox_fingerprint(coll, ax)
            if not (isinstance(axes_bbox, (list, tuple)) and len(axes_bbox) == 4):
                continue
            try:
                x, y, w, h = [float(v) for v in axes_bbox]
            except Exception:
                continue
            if w <= 0.0 or h <= 0.0:
                continue
            color = self._collection_facecolor(coll)
            if color is None or self._alpha_from_color(color) <= 0.0:
                continue
            if self._is_near_white_fill(color):
                continue
            if w > 0.22 or h < 0.08:
                continue
            body_boxes.append((x, y, w, h))
            centers.append(round(x + w / 2.0, 3))

        return len(body_boxes) >= 2 and len(set(centers)) >= 2

    def _is_background_rectangle(self, patch, bbox, ax_bw, ax_bh, bar_like_axes_boxes=None, ax=None):
        if not isinstance(patch, mpatches.Rectangle):
            return False
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            return False
        if self._is_contextual_bar_like_rectangle_bbox(bbox, ax_bw, ax_bh, bar_like_axes_boxes):
            return False
        try:
            _, _, pw, ph = [float(v) for v in bbox]
        except Exception:
            return False
        if pw <= 0 or ph <= 0 or ax_bw <= 0 or ax_bh <= 0:
            return False
        ratio_w = pw / ax_bw
        ratio_h = ph / ax_bh
        area_ratio = ratio_w * ratio_h
        try:
            zorder = float(patch.get_zorder())
        except Exception:
            zorder = 0.0

        if zorder > 1.5:
            return False

        is_axis_spanning_band = (
            (ratio_h >= 0.95 and ratio_w >= 0.01)
            or (ratio_w >= 0.95 and ratio_h >= 0.01)
        )
        return is_axis_spanning_band or (
            ratio_w >= 0.08 and (area_ratio >= 0.02 or ratio_h >= 0.18)
        )

    def _is_errorbar_segment(self, segment_axes_bbox, bar_axes_boxes, orientation=None):
        if not (isinstance(segment_axes_bbox, (list, tuple)) and len(segment_axes_bbox) == 4):
            return False
        sx, sy, sw, sh = [float(v) for v in segment_axes_bbox]
        if orientation == "vertical":
            if sw > 0.03 or sh <= 0.015:
                return False
            center_x = sx + sw / 2.0
            seg_top = sy + sh
            for bx, by, bw, bh in bar_axes_boxes:
                pad = max(bw * 0.15, 0.01)
                if (bx - pad) <= center_x <= (bx + bw + pad):
                    bar_top = by + bh
                    if (
                        abs(bar_top - sy) <= 0.08
                        or abs(bar_top - seg_top) <= 0.08
                        or (bar_top <= seg_top and (seg_top - bar_top) <= 0.12)
                    ):
                        return True
            return False

        if orientation == "horizontal":
            if sh > 0.03 or sw <= 0.015:
                return False
            center_y = sy + sh / 2.0
            seg_left = sx
            seg_right = sx + sw
            for bx, by, bw, bh in bar_axes_boxes:
                pad = max(bh * 0.35, 0.02)
                if (by - pad) <= center_y <= (by + bh + pad):
                    bar_right = bx + bw
                    if (
                        (seg_left - 0.04) <= bar_right <= (seg_right + 0.04)
                        or abs(bar_right - seg_left) <= 0.08
                        or abs(bar_right - seg_right) <= 0.08
                    ):
                        return True
        return False

    def _line_collection_color(self, coll):
        try:
            edge = coll.get_edgecolor()
            if edge is not None and len(edge) > 0:
                return tuple(edge[0])
        except Exception:
            pass
        try:
            color = coll.get_color()
            if color is not None and len(color) > 0:
                return tuple(color[0])
        except Exception:
            pass
        return (0.0, 0.0, 0.0, 1.0)

    def _segment_bbox_from_transform(self, seg, transform):
        try:
            arr = np.asarray(seg, dtype=float)
            if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
                return None
            pts = arr[:, :2]
            disp = transform.transform(pts)
            x0 = float(np.min(disp[:, 0]))
            y0 = float(np.min(disp[:, 1]))
            x1 = float(np.max(disp[:, 0]))
            y1 = float(np.max(disp[:, 1]))
            return (
                round(x0 / self.canvas_width_px, 6),
                round(y0 / self.canvas_height_px, 6),
                round(max(0.0, x1 - x0) / self.canvas_width_px, 6),
                round(max(0.0, y1 - y0) / self.canvas_height_px, 6),
            )
        except Exception:
            return None

    def _segment_axes_bbox_from_transform(self, seg, transform, ax):
        try:
            arr = np.asarray(seg, dtype=float)
            if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
                return None
            pts = arr[:, :2]
            disp = transform.transform(pts)
            x0 = float(np.min(disp[:, 0]))
            y0 = float(np.min(disp[:, 1]))
            x1 = float(np.max(disp[:, 0]))
            y1 = float(np.max(disp[:, 1]))
            return self._display_bounds_to_axes_norm(x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0), ax)
        except Exception:
            return None

    def _bbox_overlap_with_pad(self, a, b, pad=None):
        if pad is None:
            pad = self.BRACKET_FRAGMENT_MERGE_PAD
        try:
            ax, ay, aw, ah = [float(v) for v in a]
            bx, by, bw, bh = [float(v) for v in b]
            ax0, ay0, ax1, ay1 = ax - pad, ay - pad, ax + aw + pad, ay + ah + pad
            bx0, by0, bx1, by1 = bx - pad, by - pad, bx + bw + pad, by + bh + pad
            return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)
        except Exception:
            return False

    def _has_nearby_text_anchor(self, ax, bbox, axes_bbox):
        target_box = axes_bbox if isinstance(axes_bbox, (list, tuple)) and len(axes_bbox) == 4 else bbox
        if not (isinstance(target_box, (list, tuple)) and len(target_box) == 4):
            return False
        for text_artist in self._iter_ax_text_artists(ax):
            try:
                if not text_artist.get_visible():
                    continue
            except Exception:
                pass
            try:
                content = text_artist.get_text()
            except Exception:
                content = None
            if not (isinstance(content, str) and content.strip()):
                continue
            text_axes_bbox = self._get_text_axes_bbox_fingerprint(text_artist, ax)
            text_box = (
                text_axes_bbox
                if isinstance(text_axes_bbox, (list, tuple)) and len(text_axes_bbox) == 4
                else self._get_text_bbox_fingerprint(text_artist)
            )
            if not (isinstance(text_box, (list, tuple)) and len(text_box) == 4):
                continue
            if self._bbox_overlap_with_pad(target_box, text_box, pad=0.02):
                return True
        return False

    def _merge_bracket_indicator_fragments(self, start_idx):
        items = self.features["6_indicator"][start_idx:]
        if not items:
            return

        candidate_indices = [
            i
            for i, it in enumerate(items)
            if isinstance(it, dict)
            and it.get("_merge_bracket_candidate")
            and isinstance(it.get("bbox"), (list, tuple))
            and len(it.get("bbox")) == 4
        ]
        if len(candidate_indices) < self.BRACKET_FRAGMENT_MIN_PARTS:
            for it in items:
                if isinstance(it, dict):
                    it.pop("_merge_bracket_candidate", None)
            return

        group_key_to_indices = {}
        for idx in candidate_indices:
            it = items[idx]
            key = (str(it.get("color")), str(it.get("linestyle")))
            group_key_to_indices.setdefault(key, []).append(idx)

        components = []
        for group_indices in group_key_to_indices.values():
            group_set = set(group_indices)
            visited = set()
            for root in sorted(group_indices):
                if root in visited:
                    continue
                stack = [root]
                visited.add(root)
                comp = []
                while stack:
                    cur = stack.pop()
                    comp.append(cur)
                    cur_bbox = items[cur].get("bbox")
                    for nxt in group_indices:
                        if nxt in visited:
                            continue
                        if nxt not in group_set:
                            continue
                        nxt_bbox = items[nxt].get("bbox")
                        if self._bbox_overlap_with_pad(cur_bbox, nxt_bbox):
                            visited.add(nxt)
                            stack.append(nxt)
                components.append(sorted(comp))

        component_by_min_idx = {min(comp): comp for comp in components if comp}
        all_component_indices = {idx for comp in components for idx in comp}
        kept = []

        for idx, item in enumerate(items):
            if idx not in all_component_indices:
                if isinstance(item, dict):
                    item = dict(item)
                    item.pop("_merge_bracket_candidate", None)
                kept.append(item)
                continue

            comp = component_by_min_idx.get(idx)
            if comp is None:
                continue

            if len(comp) < self.BRACKET_FRAGMENT_MIN_PARTS:
                for cidx in comp:
                    sub_item = items[cidx]
                    if isinstance(sub_item, dict):
                        sub_item = dict(sub_item)
                        sub_item.pop("_merge_bracket_candidate", None)
                    kept.append(sub_item)
                continue

            comp_items = [items[cidx] for cidx in comp]
            x0 = min(float(ci["bbox"][0]) for ci in comp_items)
            y0 = min(float(ci["bbox"][1]) for ci in comp_items)
            x1 = max(float(ci["bbox"][0]) + float(ci["bbox"][2]) for ci in comp_items)
            y1 = max(float(ci["bbox"][1]) + float(ci["bbox"][3]) for ci in comp_items)

            merged = dict(comp_items[0])
            merged.pop("_merge_bracket_candidate", None)
            merged["bbox"] = [
                round(x0, 6),
                round(y0, 6),
                round(max(0.0, x1 - x0), 6),
                round(max(0.0, y1 - y0), 6),
            ]
            axes_boxes = [
                ci.get("axes_bbox")
                for ci in comp_items
                if isinstance(ci.get("axes_bbox"), (list, tuple)) and len(ci.get("axes_bbox")) == 4
            ]
            if axes_boxes:
                ax0 = min(float(ci[0]) for ci in axes_boxes)
                ay0 = min(float(ci[1]) for ci in axes_boxes)
                ax1 = max(float(ci[0]) + float(ci[2]) for ci in axes_boxes)
                ay1 = max(float(ci[1]) + float(ci[3]) for ci in axes_boxes)
                merged["axes_bbox"] = [
                    round(ax0, 6),
                    round(ay0, 6),
                    round(max(0.0, ax1 - ax0), 6),
                    round(max(0.0, ay1 - ay0), 6),
                ]
            else:
                merged["axes_bbox"] = None
            merged["orientation"] = self._line_like_orientation(merged.get("axes_bbox") or merged.get("bbox"))
            kept.append(merged)

        self.features["6_indicator"] = (
            self.features["6_indicator"][:start_idx] + kept
        )

    def _filter_grid_like_indicators(self):
        grouped = {}
        for idx, item in enumerate(self.features["6_indicator"]):
            if not isinstance(item, dict):
                continue
            if item.get("_keep_diff_full_span"):
                continue
            if not item.get("is_full_span"):
                continue
            orientation = item.get("orientation")
            if orientation not in {"horizontal", "vertical"}:
                continue
            key = (
                item.get("ax_index"),
                item.get("src"),
                str(item.get("color")),
                str(item.get("linestyle")),
                orientation,
            )
            grouped.setdefault(key, []).append(idx)

        remove_indexes = set()
        for indexes in grouped.values():
            if len(indexes) >= self.GRIDLIKE_INDICATOR_REPEAT_MIN:
                remove_indexes.update(indexes)

        if remove_indexes:
            self.features["6_indicator"] = [
                item
                for idx, item in enumerate(self.features["6_indicator"])
                if idx not in remove_indexes
            ]

    def _extract_line_stage(self, ax, ax_index, ctx):
        line_infos = ctx["line_infos"]
        has_bar_like_rect = ctx["has_bar_like_rect"]
        has_filled_area = ctx["has_filled_area"]
        has_annotation = ctx["has_annotation"]
        overlay_indicator_indexes = ctx["overlay_indicator_indexes"]
        for idx, line in enumerate(ax.lines):
            if not self._is_artist_allowed(line):
                continue
            info = line_infos[idx] if idx < len(line_infos) else {}
            ls = line.get_linestyle()
            is_full_span = bool(info.get("is_full_span", False))
            bbox = info.get("bbox")

            xdata = line.get_xdata()
            ydata = line.get_ydata()
            is_two_point_line = bool(info.get("is_two_point_line", len(xdata) <= 2 and len(ydata) <= 2))
            is_line_like_bbox = False
            if bbox is not None:
                _, _, bw, bh = bbox
                is_line_like_bbox = (bw <= 0.004) or (bh <= 0.004)
            marker = line.get_marker()
            marker_is_none = marker in self.NONE_MARKERS
            axes_bbox = self._get_artist_axes_bbox_fingerprint(line, ax)
            orientation = self._line_like_orientation(axes_bbox or bbox)
            is_axis_boundary_line = False
            is_diff_full_span_dashed = (
                self.allowed_artist_ids is not None
                and is_full_span
                and orientation in {"horizontal", "vertical"}
                and str(ls).strip().lower() in {"--", ":", "-.", "dashed", "dotted", "dashdot"}
            )
            if (
                (not is_diff_full_span_dashed)
                and bool(getattr(line, "get_clip_on", lambda: True)())
                and isinstance(axes_bbox, (list, tuple))
                and len(axes_bbox) == 4
                and orientation in {"horizontal", "vertical"}
            ):
                try:
                    ax_x, ax_y, ax_w, ax_h = [float(v) for v in axes_bbox]
                except Exception:
                    ax_x = ax_y = ax_w = ax_h = 0.0
                if orientation == "horizontal" and ax_w >= 0.95:
                    is_axis_boundary_line = abs(ax_y) <= 0.02 or abs((ax_y + ax_h) - 1.0) <= 0.02
                elif orientation == "vertical" and ax_h >= 0.95:
                    is_axis_boundary_line = abs(ax_x) <= 0.02 or abs((ax_x + ax_w) - 1.0) <= 0.02
            if is_axis_boundary_line:
                continue

            keep_as_indicator = False
            is_bracket_fragment_indicator = False
            is_text_bound_guide_polyline = False
            try:
                line_width = float(line.get_linewidth() or 0.0)
            except Exception:
                line_width = 0.0
            ls_normalized = str(ls).strip().lower()
            is_dashed_indicator_style = ls_normalized in {"--", ":", "-.", "dashed", "dotted", "dashdot"}
            if self._is_reference_indicator_line(
                ax, idx, line, info, line_infos,
                has_bar_like_rect=has_bar_like_rect,
                has_filled_area=has_filled_area,
            ):
                keep_as_indicator = True
            elif (
                has_bar_like_rect
                and is_dashed_indicator_style
                and int(info.get("n_points", 0)) >= 3
                and (not self._is_light_color(line.get_color()))
            ):
                keep_as_indicator = True
            elif (
                has_bar_like_rect
                and idx in overlay_indicator_indexes
                and (not self._is_light_color(line.get_color()))
                and ls_normalized not in {"none", "", "null"}
            ):
                keep_as_indicator = True
            elif is_full_span and is_two_point_line and is_line_like_bbox:
                if not self._is_light_color(line.get_color()):
                    keep_as_indicator = True
                    if (
                        (not info.get("is_axis_spanning", False))
                        and self._is_chromatic_color(line.get_color())
                    ):
                        is_bracket_fragment_indicator = True
            elif marker_is_none and is_two_point_line and info.get("is_axis_spanning", False):
                if not self._is_light_color(line.get_color()):
                    keep_as_indicator = True
            else:
                if (
                    idx in overlay_indicator_indexes
                    and (not self._is_light_color(line.get_color()))
                    and ls_normalized not in {"none", "", "null"}
                ):
                    if not (
                        self._has_nearby_text_anchor(ax, bbox, axes_bbox)
                        and (
                            self._is_axis_aligned_bracket_polyline(xdata, ydata)
                            or self._is_open_bracket_polyline(xdata, ydata)
                            or self._is_brace_polyline(xdata, ydata)
                        )
                    ):
                        keep_as_indicator = True
                elif (
                    (not has_annotation)
                    and marker_is_none
                    and is_two_point_line
                    and (not is_full_span)
                    and (not info.get("is_axis_spanning", False))
                    and (not self._is_light_color(line.get_color()))
                    and (line_width >= 1.5 or self._is_chromatic_color(line.get_color()))
                ):
                    keep_as_indicator = True
                    is_bracket_fragment_indicator = True
                elif (
                    (not has_annotation)
                    and marker_is_none
                    and int(info.get("n_points", 0)) >= 3
                    and (not is_full_span)
                    and (not info.get("is_axis_spanning", False))
                    and (self._is_chromatic_color(line.get_color()) or line_width <= 1.2)
                    and (
                        self._is_axis_aligned_bracket_polyline(xdata, ydata)
                        or self._is_open_bracket_polyline(xdata, ydata)
                        or self._is_brace_polyline(xdata, ydata)
                    )
                ):
                    if self._has_nearby_text_anchor(ax, bbox, axes_bbox):
                        is_text_bound_guide_polyline = True
                    else:
                        keep_as_indicator = True
                elif self._is_diffed_line_indicator_fallback(line, info):
                    if self._has_nearby_text_anchor(ax, bbox, axes_bbox):
                        is_text_bound_guide_polyline = True
                    else:
                        keep_as_indicator = True

            if keep_as_indicator:
                indicator_item = {
                    "src": "line_indicator",
                    "ax_index": ax_index,
                    "linestyle": ls,
                    "color": self._round_color(line.get_color()),
                    "is_full_span": is_full_span,
                    "bbox": bbox,
                    "axes_bbox": axes_bbox,
                    "orientation": self._line_like_orientation(axes_bbox or bbox),
                }
                if (
                    self.allowed_artist_ids is not None
                    and is_diff_full_span_dashed
                ):
                    indicator_item["_keep_diff_full_span"] = True
                if is_bracket_fragment_indicator:
                    indicator_item["_merge_bracket_candidate"] = True
                self.features["6_indicator"].append(indicator_item)
                continue

            try:
                linewidth = float(line.get_linewidth() or 0.0)
            except Exception:
                linewidth = 0.0
            non_annotation_guide_line = (
                (not has_annotation)
                and is_two_point_line
                and (not is_full_span)
                and marker_is_none
                and linewidth <= 1.5
                and (not info.get("is_axis_spanning", False))
                and is_line_like_bbox
                and (not self._is_light_color(line.get_color()))
                and (len(ax.texts) >= 6 or (not bool(line.get_clip_on())))
            )
            is_rect_frame_edge = False
            if (
                self.allowed_artist_ids is not None
                and (not has_annotation)
                and is_two_point_line
                and (not is_full_span)
                and marker_is_none
                and linewidth <= 1.5
                and (not info.get("is_axis_spanning", False))
                and is_line_like_bbox
                and (not self._is_light_color(line.get_color()))
                and isinstance(axes_bbox, (list, tuple))
                and len(axes_bbox) == 4
            ):
                orientation = self._line_like_orientation(axes_bbox or bbox)
                if orientation == "horizontal":
                    span = float(axes_bbox[2])
                    is_rect_frame_edge = 0.15 <= span <= 0.85
                elif orientation == "vertical":
                    span = float(axes_bbox[3])
                    is_rect_frame_edge = 0.15 <= span <= 0.85
            is_short_guide_line = (
                has_annotation
                and is_two_point_line
                and (not is_full_span)
                and marker_is_none
                and linewidth <= 2.5
                and (not self._is_light_color(line.get_color()))
            ) or non_annotation_guide_line or is_rect_frame_edge or is_text_bound_guide_polyline
            if is_short_guide_line:
                axes_bbox = self._get_artist_axes_bbox_fingerprint(line, ax)
                self.features["2_connector"].append(
                    {
                        "src": "line_connector",
                        "ax_index": ax_index,
                        "linestyle": ls,
                        "color": self._round_color(line.get_color()),
                        "bbox": bbox,
                        "axes_bbox": axes_bbox,
                        "orientation": self._line_like_orientation(axes_bbox or bbox),
                    }
                )
