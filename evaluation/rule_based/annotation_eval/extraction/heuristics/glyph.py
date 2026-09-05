from collections import Counter

import matplotlib.collections as mcollections
import numpy as np


class GlyphAnnotationMixin:
    def _scatter_radius_px(self, scatter_size):
        try:
            radius_pt = (max(float(scatter_size), 0.0) ** 0.5) / 2.0
            return radius_pt * float(self.fig.dpi) / 72.0
        except Exception:
            return 0.0

    def _markersize_to_radius_px(self, marker_size_pt):
        try:
            return max(float(marker_size_pt), 0.0) * float(self.fig.dpi) / 144.0
        except Exception:
            return 0.0

    def _extract_collection_stage(self, ax, ax_index, ctx):
        ax_bw = ctx["ax_bw"]
        ax_bh = ctx["ax_bh"]
        bar_like_axes_boxes = ctx["bar_like_axes_boxes"]
        errorbar_cap_color_keys = ctx["errorbar_cap_color_keys"]
        has_bar_like_rect = ctx["has_bar_like_rect"]
        is_violin_like_axis = ctx["is_violin_like_axis"]
        line_infos = ctx["line_infos"]

        for coll in ax.collections:
            if not self._is_artist_allowed(coll):
                continue
            if not isinstance(coll, mcollections.PolyCollection):
                continue
            if isinstance(coll, (mcollections.LineCollection, mcollections.PathCollection)):
                continue
            if self.allowed_artist_ids is None:
                continue
            bbox, axes_bbox = self._get_artist_clipped_bbox_fingerprints(coll, ax)
            if bbox is None:
                continue
            color = self._collection_facecolor(coll)
            if color is None:
                continue
            if self._alpha_from_color(color) <= 0.0:
                continue
            if self._is_near_white_fill(color):
                continue
            self.features["1_enclosure"].append(
                {
                    "src": "poly_fill_enclosure",
                    "ax_index": ax_index,
                    "bbox": bbox,
                    "axes_bbox": axes_bbox,
                    "color": self._round_color(color),
                    "alpha": float(self._alpha_from_color(color)),
                }
            )

        for coll in ax.collections:
            if not self._is_artist_allowed(coll):
                continue
            if not isinstance(coll, mcollections.LineCollection):
                continue
            coll_color = self._line_collection_color(coll)
            if self._is_light_color(coll_color):
                continue
            try:
                linewidths = np.asarray(coll.get_linewidths(), dtype=float).ravel()
                line_width = float(np.max(linewidths)) if linewidths.size > 0 else 1.0
            except Exception:
                line_width = 1.0
            transform = coll.get_transform() if hasattr(coll, "get_transform") else ax.transData
            try:
                segments = coll.get_segments()
            except Exception:
                segments = []
            segment_count = len(segments)
            try:
                coll_linestyle = str(coll.get_linestyle())
            except Exception:
                coll_linestyle = "None"
            generic_errorbar_collection = (
                segment_count <= 12
                and str(self._round_color(coll_color)) in errorbar_cap_color_keys
            )
            if line_width > 1.8 and not generic_errorbar_collection:
                continue

            for seg in segments:
                bbox = self._segment_bbox_from_transform(seg, transform)
                if bbox is None:
                    continue
                _, _, bw, bh = bbox
                spans_x = ax_bw > 0 and float(bw) >= ax_bw * self.AXIS_SPAN_RATIO_INDICATOR
                spans_y = ax_bh > 0 and float(bh) >= ax_bh * self.AXIS_SPAN_RATIO_INDICATOR
                if spans_x or spans_y:
                    keep_as_collection_indicator = (
                        segment_count <= 3
                        and line_width >= 0.8
                        and (not self._is_light_color(coll_color))
                        and coll_linestyle.strip().lower() not in {"none", "", "null"}
                    )
                    if keep_as_collection_indicator:
                        axes_bbox = self._segment_axes_bbox_from_transform(seg, transform, ax)
                        self.features["6_indicator"].append(
                            {
                                "src": "line_collection_indicator",
                                "ax_index": ax_index,
                                "linestyle": coll_linestyle,
                                "color": self._round_color(coll_color),
                                "is_full_span": True,
                                "bbox": bbox,
                                "axes_bbox": axes_bbox,
                                "orientation": self._line_like_orientation(axes_bbox or bbox),
                            }
                        )
                    continue

                is_line_like = (float(bw) <= 0.004) or (float(bh) <= 0.004)
                if not is_line_like:
                    continue
                axes_bbox = self._segment_axes_bbox_from_transform(seg, transform, ax)
                orientation = self._line_like_orientation(axes_bbox or bbox)
                ls_normalized = coll_linestyle.strip().lower()
                is_dashed_collection = (
                    ls_normalized in {"--", "-.", ":", "dashed", "dashdot", "dotted"}
                    or ("[" in ls_normalized and "]" in ls_normalized and "none" not in ls_normalized)
                )
                substantial_span = (
                    (orientation == "vertical" and ax_bh > 0 and float(bh) >= ax_bh * 0.25)
                    or (orientation == "horizontal" and ax_bw > 0 and float(bw) >= ax_bw * 0.25)
                )
                if generic_errorbar_collection and orientation in {"horizontal", "vertical"}:
                    self.features["6_indicator"].append(
                        {
                            "src": "errorbar_indicator",
                            "type": "errorbar",
                            "ax_index": ax_index,
                            "linestyle": coll_linestyle,
                            "color": self._round_color(coll_color),
                            "is_full_span": False,
                            "bbox": bbox,
                            "axes_bbox": axes_bbox,
                            "orientation": orientation,
                        }
                    )
                elif (
                    errorbar_cap_color_keys
                    and orientation in {"horizontal", "vertical"}
                    and segment_count >= 4
                    and line_width <= 1.8
                ):
                    self.features["6_indicator"].append(
                        {
                            "src": "errorbar_indicator",
                            "type": "errorbar",
                            "ax_index": ax_index,
                            "linestyle": coll_linestyle,
                            "color": self._round_color(coll_color),
                            "is_full_span": False,
                            "bbox": bbox,
                            "axes_bbox": axes_bbox,
                            "orientation": orientation,
                        }
                    )
                elif (
                    orientation in {"horizontal", "vertical"}
                    and is_dashed_collection
                    and substantial_span
                    and not self._is_light_color(coll_color)
                ):
                    self.features["6_indicator"].append(
                        {
                            "src": "line_collection_indicator",
                            "ax_index": ax_index,
                            "linestyle": coll_linestyle,
                            "color": self._round_color(coll_color),
                            "is_full_span": False,
                            "bbox": bbox,
                            "axes_bbox": axes_bbox,
                            "orientation": orientation,
                        }
                    )
                elif (
                    has_bar_like_rect
                    and self._is_errorbar_segment(axes_bbox, bar_like_axes_boxes, orientation=orientation)
                ):
                    self.features["6_indicator"].append(
                        {
                            "src": "errorbar_indicator",
                            "type": "errorbar",
                            "ax_index": ax_index,
                            "linestyle": coll_linestyle,
                            "color": self._round_color(coll_color),
                            "is_full_span": False,
                            "bbox": bbox,
                            "axes_bbox": axes_bbox,
                            "orientation": orientation,
                        }
                    )
                elif is_violin_like_axis and orientation in {"horizontal", "vertical"}:
                    self.features["6_indicator"].append(
                        {
                            "src": "violin_center_indicator",
                            "type": "violin_center",
                            "ax_index": ax_index,
                            "linestyle": coll_linestyle,
                            "color": self._round_color(coll_color),
                            "is_full_span": False,
                            "bbox": bbox,
                            "axes_bbox": axes_bbox,
                            "orientation": orientation,
                        }
                    )
                else:
                    self.features["2_connector"].append(
                        {
                            "src": "line_collection_connector",
                            "ax_index": ax_index,
                            "linestyle": coll_linestyle,
                            "color": self._round_color(coll_color),
                            "bbox": bbox,
                            "axes_bbox": axes_bbox,
                            "orientation": orientation,
                        }
                    )

        marker_none_linestyles = {"none", "", " ", "null"}
        line_marker_candidates = []
        marker_candidate_point_total = 0
        series_line_point_total = 0
        for info in line_infos:
            n_points = int(info.get("n_points", 0))
            if n_points <= 0 or bool(info.get("is_full_span", False)):
                continue
            if (
                bool(info.get("is_axis_spanning", False))
                and n_points >= 3
                and bool(info.get("marker_is_none", False))
            ):
                series_line_point_total += n_points

        for line in ax.lines:
            if not self._is_artist_allowed(line):
                continue
            marker = line.get_marker()
            if marker in self.NONE_MARKERS or str(marker) in self.ERRORBAR_MARKER_SYMBOLS:
                continue
            xdata = np.ravel(np.asarray(line.get_xdata(orig=False)))
            ydata = np.ravel(np.asarray(line.get_ydata(orig=False)))
            if len(xdata) == 0 or len(ydata) == 0:
                continue
            if len(xdata) != len(ydata):
                n = min(len(xdata), len(ydata))
                xdata = xdata[:n]
                ydata = ydata[:n]
            n_points = len(xdata)
            if n_points > 20:
                continue
            linestyle = str(line.get_linestyle()).lower()
            is_marker_only = linestyle in marker_none_linestyles
            is_sparse_marker_line = is_marker_only or (n_points <= 2)
            if not is_sparse_marker_line:
                continue
            line_marker_candidates.append({"line": line, "xdata": xdata, "ydata": ydata, "marker": marker})
            marker_candidate_point_total += n_points

        keep_sparse_marker_lines = True
        if series_line_point_total > 0:
            sparse_cap = max(2, int(np.ceil(series_line_point_total * 0.5)))
            keep_sparse_marker_lines = (
                marker_candidate_point_total < series_line_point_total
                and marker_candidate_point_total <= sparse_cap
            )

        if keep_sparse_marker_lines:
            for item in line_marker_candidates:
                line = item["line"]
                marker = item["marker"]
                xdata = item["xdata"]
                ydata = item["ydata"]
                transform = line.get_transform() if hasattr(line, "get_transform") else ax.transData
                radius_px = self._markersize_to_radius_px(line.get_markersize())
                for x, y in zip(xdata, ydata):
                    if np.ma.is_masked(x) or np.ma.is_masked(y):
                        continue
                    try:
                        x = float(x)
                        y = float(y)
                    except Exception:
                        continue
                    bbox = self._point_bbox_from_transform(x, y, transform, radius_px=radius_px)
                    if bbox is None:
                        continue
                    marker_color = line.get_markeredgecolor()
                    if isinstance(marker_color, np.ndarray):
                        marker_color = tuple(marker_color.tolist())
                    elif isinstance(marker_color, list):
                        marker_color = tuple(marker_color)
                    if marker_color is None or (isinstance(marker_color, str) and marker_color.lower() == "none"):
                        marker_color = line.get_color()
                    if isinstance(marker_color, np.ndarray):
                        marker_color = tuple(marker_color.tolist())
                    elif isinstance(marker_color, list):
                        marker_color = tuple(marker_color)
                    marker_str = str(marker)
                    if (
                        self.allowed_artist_ids is not None
                        and len(xdata) == 1
                        and marker_str in self.TRIANGLE_CONNECTOR_MARKERS
                    ):
                        self.features["2_connector"].append(
                            {
                                "src": "triangle_marker_connector",
                                "marker": marker_str,
                                "ax_index": ax_index,
                                "bbox": bbox,
                                "color": self._round_color(marker_color),
                            }
                        )
                    else:
                        self.features["4_glyph"].append(
                            {
                                "src": "line_marker",
                                "marker": marker_str,
                                "bbox": bbox,
                                "color": self._round_color(marker_color),
                            }
                        )

        for coll in ax.collections:
            if not self._is_artist_allowed(coll):
                continue
            if not isinstance(coll, mcollections.PathCollection):
                continue
            offsets = coll.get_offsets()
            if offsets is None or len(offsets) == 0:
                continue
            total_pts = len(offsets)
            sizes_raw = coll.get_sizes()
            try:
                sizes_arr = np.asarray(sizes_raw, dtype=float).ravel()
            except Exception:
                sizes_arr = np.array([], dtype=float)
            if sizes_arr.size == 0:
                sizes_arr = np.full(total_pts, 36.0, dtype=float)
            elif sizes_arr.size == 1 and total_pts > 1:
                sizes_arr = np.full(total_pts, float(sizes_arr[0]), dtype=float)
            elif sizes_arr.size < total_pts:
                sizes_arr = np.pad(sizes_arr, (0, total_pts - sizes_arr.size), mode="edge")
            else:
                sizes_arr = sizes_arr[:total_pts]
            size_keys = [round(float(s), 6) for s in sizes_arr]
            size_counts = Counter(size_keys)
            mode_size_key = size_counts.most_common(1)[0][0]
            offset_transform = coll.get_offset_transform() if hasattr(coll, "get_offset_transform") else ax.transData

            for i, pt in enumerate(offsets):
                if i >= len(sizes_arr) or len(pt) < 2:
                    break
                if np.ma.is_masked(pt[0]) or np.ma.is_masked(pt[1]):
                    continue
                s = float(sizes_arr[i])
                s_key = round(s, 6)
                is_size_outlier = (
                    s_key != mode_size_key and (size_counts[s_key] / float(max(total_pts, 1)) < 0.2)
                )
                keep_as_glyph = is_size_outlier or (total_pts <= 20)
                if not keep_as_glyph:
                    continue
                bbox = self._point_bbox_from_transform(
                    pt[0], pt[1], offset_transform, radius_px=self._scatter_radius_px(s)
                )
                if bbox is None:
                    continue
                try:
                    facecolors = coll.get_facecolors()
                    edgecolors = coll.get_edgecolors()
                except Exception:
                    facecolors = edgecolors = None

                def _pick_collection_color(arr, idx):
                    try:
                        if arr is None or len(arr) == 0:
                            return None
                        use_idx = idx if len(arr) == total_pts else 0
                        return tuple(float(v) for v in arr[use_idx][:4])
                    except Exception:
                        return None

                face_color = _pick_collection_color(facecolors, i)
                edge_color = _pick_collection_color(edgecolors, i)
                try:
                    face_alpha = float(face_color[3]) if face_color is not None else 0.0
                except Exception:
                    face_alpha = 0.0
                is_large_hollow_circle = (
                    total_pts == 1
                    and s >= 500.0
                    and face_alpha <= 0.0
                    and edge_color is not None
                    and self._alpha_from_color(edge_color) > 0.0
                )
                if is_large_hollow_circle:
                    self.features["1_enclosure"].append(
                        {
                            "src": "hollow_scatter_enclosure",
                            "ax_index": ax_index,
                            "bbox": bbox,
                            "color": self._round_color(edge_color),
                            "alpha": float(self._alpha_from_color(edge_color)),
                        }
                    )
                    continue
                record = {"src": "scatter_outlier" if is_size_outlier else "scatter_point", "bbox": bbox, "size": s}
                if face_color is not None:
                    record["facecolor"] = self._round_color(face_color)
                if edge_color is not None:
                    record["edgecolor"] = self._round_color(edge_color)
                if face_color is not None and self._alpha_from_color(face_color) > 0.0:
                    record["color"] = self._round_color(face_color)
                elif edge_color is not None and self._alpha_from_color(edge_color) > 0.0:
                    record["color"] = self._round_color(edge_color)
                if is_size_outlier:
                    record["note"] = "Size Outlier"
                self.features["4_glyph"].append(record)
