"""Extract structured annotation elements from a matplotlib figure."""

import matplotlib.collections as mcollections
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.text as mtext
import matplotlib.transforms as mtransforms
import numpy as np

from annotation_eval.extraction.heuristics.color import ColorAnnotationMixin
from annotation_eval.extraction.heuristics.connector import ConnectorAnnotationMixin
from annotation_eval.extraction.heuristics.enclosure import EnclosureAnnotationMixin
from annotation_eval.extraction.heuristics.geometric import GeometricAnnotationMixin
from annotation_eval.extraction.heuristics.glyph import GlyphAnnotationMixin
from annotation_eval.extraction.heuristics.indicator import IndicatorAnnotationMixin
from annotation_eval.extraction.heuristics.text import TextAnnotationMixin
from annotation_eval.extraction.annotation_schema import new_annotation_dict
from annotation_eval.extraction.geometry import (
    bbox_to_axes_norm,
    bbox_to_fig_norm,
    clip_display_bounds_to_axes,
    display_bounds_to_axes_norm,
    display_bounds_to_fig_norm,
)


class ChartAnnotationExtractor(
    EnclosureAnnotationMixin,
    ConnectorAnnotationMixin,
    TextAnnotationMixin,
    GlyphAnnotationMixin,
    ColorAnnotationMixin,
    IndicatorAnnotationMixin,
    GeometricAnnotationMixin,
):
    AXIS_SPAN_RATIO_INDICATOR = 0.80
    NONE_MARKERS = {None, "", "None", "none", " ", "null"}
    BRACKET_FRAGMENT_MERGE_PAD = 0.0025
    BRACKET_FRAGMENT_MIN_PARTS = 3
    GRIDLIKE_INDICATOR_REPEAT_MIN = 4
    ERRORBAR_MARKER_SYMBOLS = {"_", "|"}
    TRIANGLE_CONNECTOR_MARKERS = {"v", "^", "<", ">", "1", "2", "3", "4"}

    def __init__(self, fig, *, allowed_artist_ids=None, allowed_axes_ids=None, geometric_axes_ids=None):
        self.fig = fig
        self.axes = self._collect_axes(fig)
        self._seen_text_enclosure_patches = set()
        self.allowed_artist_ids = (
            {int(v) for v in allowed_artist_ids}
            if allowed_artist_ids is not None
            else None
        )
        self.allowed_axes_ids = (
            {int(v) for v in allowed_axes_ids}
            if allowed_axes_ids is not None
            else None
        )
        self.geometric_axes_ids = (
            {int(v) for v in geometric_axes_ids}
            if geometric_axes_ids is not None
            else None
        )

        try:
            self.fig.canvas.draw()
        except Exception:
            pass

        try:
            self.renderer = self.fig.canvas.get_renderer()
        except Exception:
            from matplotlib.backends.backend_agg import FigureCanvasAgg

            canvas = FigureCanvasAgg(self.fig)
            canvas.draw()
            self.renderer = canvas.get_renderer()

        self.canvas_width_px, self.canvas_height_px = self._get_canvas_size()

        self.features = new_annotation_dict()

    def _is_artist_allowed(self, artist):
        if artist is None:
            return False
        if self.allowed_artist_ids is None:
            return True
        return id(artist) in self.allowed_artist_ids

    def _is_ax_allowed(self, ax):
        if ax is None:
            return False
        if self.allowed_axes_ids is None:
            return True
        return id(ax) in self.allowed_axes_ids

    def _is_ax_allowed_for_geometric(self, ax):
        if ax is None:
            return False
        if self.geometric_axes_ids is None:
            return self._is_ax_allowed(ax)
        return id(ax) in self.geometric_axes_ids

    def _collect_axes(self, fig):
        axes = []
        seen = set()

        def visit(ax):
            ax_id = id(ax)
            if ax_id in seen:
                return
            seen.add(ax_id)
            axes.append(ax)
            for child_ax in getattr(ax, "child_axes", []) or []:
                visit(child_ax)

        for ax in fig.axes:
            visit(ax)
        return axes

    def _get_canvas_size(self):
        try:
            width = float(getattr(self.renderer, "width", 0.0))
            height = float(getattr(self.renderer, "height", 0.0))
            if width > 0 and height > 0:
                return width, height

            width, height = self.fig.canvas.get_width_height()
            return float(max(width, 1)), float(max(height, 1))
        except Exception:
            width, height = self.fig.get_size_inches() * self.fig.dpi
            return float(max(width, 1)), float(max(height, 1))

    def _display_to_fig_norm(self, x_disp, y_disp):
        return (
            round(float(x_disp) / self.canvas_width_px, 6),
            round(float(y_disp) / self.canvas_height_px, 6),
        )

    def _center_size_to_bbox(self, cx_disp, cy_disp, w_px=0.0, h_px=0.0):
        try:
            x0 = float(cx_disp) - float(w_px) / 2.0
            y0 = float(cy_disp) - float(h_px) / 2.0
            return (
                round(x0 / self.canvas_width_px, 6),
                round(y0 / self.canvas_height_px, 6),
                round(float(w_px) / self.canvas_width_px, 6),
                round(float(h_px) / self.canvas_height_px, 6),
            )
        except Exception:
            return None

    def _point_bbox_from_transform(self, x, y, transform, radius_px=0.0):
        try:
            disp = transform.transform((float(x), float(y)))
            size_px = max(float(radius_px), 0.0) * 2.0
            return self._center_size_to_bbox(disp[0], disp[1], size_px, size_px)
        except Exception:
            return None

    def _normalize_coord(self, x, y, transform=None):
        try:
            if isinstance(x, (str, np.str_)) or isinstance(y, (str, np.str_)):
                return (x, y)

            if transform is None:
                display_pt = (float(x), float(y))
            else:
                display_pt = transform.transform((float(x), float(y)))

            return self._display_to_fig_norm(display_pt[0], display_pt[1])
        except Exception:
            return (x, y)

    def _bbox_to_fig_norm(self, bbox):
        return bbox_to_fig_norm(bbox, self.canvas_width_px, self.canvas_height_px)

    def _display_bounds_to_fig_norm(self, x0, y0, w, h):
        return display_bounds_to_fig_norm((x0, y0, w, h), self.canvas_width_px, self.canvas_height_px)

    def _display_bounds_to_axes_norm(self, x0, y0, w, h, ax):
        try:
            ax_bbox = ax.get_window_extent(self.renderer)
        except Exception:
            return None
        return display_bounds_to_axes_norm((x0, y0, w, h), ax_bbox)

    def _clip_display_bounds_to_axes(self, x0, y0, w, h, ax):
        try:
            ax_bbox = ax.get_window_extent(self.renderer)
        except Exception:
            return None
        return clip_display_bounds_to_axes((x0, y0, w, h), ax_bbox, allow_zero_area=False)

    def _get_artist_clipped_bbox_fingerprints(self, artist, ax):
        try:
            bbox = artist.get_window_extent(self.renderer)
            x0, y0, w, h = bbox.bounds
        except Exception:
            return None, None
        clipped = self._clip_display_bounds_to_axes(x0, y0, w, h, ax)
        if clipped is None:
            return None, None
        fig_bbox = self._display_bounds_to_fig_norm(*clipped)
        axes_bbox = self._display_bounds_to_axes_norm(*clipped, ax)
        return fig_bbox, axes_bbox

    def _bbox_to_axes_norm(self, bbox, ax):
        try:
            ax_bbox = ax.get_window_extent(self.renderer)
        except Exception:
            return None
        return bbox_to_axes_norm(bbox, ax_bbox)

    def _get_artist_bbox_fingerprint(self, artist):
        try:
            bbox = artist.get_window_extent(self.renderer)
            return self._bbox_to_fig_norm(bbox)
        except Exception:
            return None

    def _get_artist_axes_bbox_fingerprint(self, artist, ax):
        try:
            bbox = artist.get_window_extent(self.renderer)
            return self._bbox_to_axes_norm(bbox, ax)
        except Exception:
            return None

    def _get_text_bbox_fingerprint(self, text_artist):
        try:
            if isinstance(text_artist, mtext.Annotation):
                bbox = mtext.Text.get_window_extent(text_artist, self.renderer)
            else:
                bbox = text_artist.get_window_extent(self.renderer)
            return self._bbox_to_fig_norm(bbox)
        except Exception:
            return None

    def _get_text_axes_bbox_fingerprint(self, text_artist, ax):
        try:
            if isinstance(text_artist, mtext.Annotation):
                bbox = mtext.Text.get_window_extent(text_artist, self.renderer)
            else:
                bbox = text_artist.get_window_extent(self.renderer)
            return self._bbox_to_axes_norm(bbox, ax)
        except Exception:
            return None

    def _bbox_union(self, boxes):
        valid = []
        for box in boxes:
            if not (isinstance(box, (list, tuple)) and len(box) == 4):
                continue
            try:
                x, y, w, h = [float(v) for v in box]
            except Exception:
                continue
            valid.append((x, y, max(0.0, w), max(0.0, h)))
        if not valid:
            return None
        x0 = min(x for x, _, _, _ in valid)
        y0 = min(y for _, y, _, _ in valid)
        x1 = max(x + w for x, _, w, _ in valid)
        y1 = max(y + h for _, y, _, h in valid)
        return [
            round(x0, 6),
            round(y0, 6),
            round(max(0.0, x1 - x0), 6),
            round(max(0.0, y1 - y0), 6),
        ]

    def _alpha_from_color(self, color):
        try:
            return float(mcolors.to_rgba(color)[3])
        except Exception:
            return 0.0

    def _bbox_intersects_unit_square(self, bbox):
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            return False
        try:
            x, y, w, h = [float(v) for v in bbox]
        except Exception:
            return False
        return (x + w) > 0.0 and x < 1.0 and (y + h) > 0.0 and y < 1.0

    def _round_color(self, color):
        if color is None:
            return "none"
        try:
            rgba = mcolors.to_rgba(color)
            return tuple(round(c, 3) for c in rgba)
        except Exception:
            return str(color)

    def _build_ax_stage_context(self, ax):
        indicator_start_idx = len(self.features["6_indicator"])
        feature_start_idx = {key: len(self.features[key]) for key in self.features}
        exclude_texts = [
            ax.title,
            getattr(ax, "_left_title", None),
            getattr(ax, "_right_title", None),
            ax.xaxis.label,
            ax.yaxis.label,
        ]
        suppress_geometric_axis_payload = (
            self.geometric_axes_ids is not None and id(ax) in self.geometric_axes_ids
        )
        processed_patches = set()
        pie_geo_targets = self._select_pie_geometric_wedges(ax)
        has_annotation = any(
            isinstance(t, mtext.Annotation) and bool(getattr(t, "arrowprops", None))
            for t in ax.texts
        )
        has_filled_area = any(
            isinstance(coll, mcollections.PolyCollection)
            for coll in ax.collections
        )
        has_scatter_collection = any(
            isinstance(coll, mcollections.PathCollection)
            for coll in ax.collections
        )
        is_violin_like_axis = self._is_violin_like_axis(ax)
        ax_bbox = self._get_artist_bbox_fingerprint(ax)
        ax_bw = float(ax_bbox[2]) if isinstance(ax_bbox, (list, tuple)) and len(ax_bbox) == 4 else 0.0
        ax_bh = float(ax_bbox[3]) if isinstance(ax_bbox, (list, tuple)) and len(ax_bbox) == 4 else 0.0

        bar_like_rect_count = 0
        bar_like_axes_boxes = []
        if ax_bw > 0 and ax_bh > 0:
            for p in ax.patches:
                if not isinstance(p, mpatches.Rectangle) or p is ax.patch:
                    continue
                pb_axes = self._get_artist_axes_bbox_fingerprint(p, ax)
                bbox_for_bar_like = pb_axes
                bbox_w = 1.0
                bbox_h = 1.0
                if not (isinstance(pb_axes, (list, tuple)) and len(pb_axes) == 4):
                    pb = self._get_artist_bbox_fingerprint(p)
                    if not (isinstance(pb, (list, tuple)) and len(pb) == 4):
                        continue
                    bbox_for_bar_like = pb
                    bbox_w = ax_bw
                    bbox_h = ax_bh
                pw = float(bbox_for_bar_like[2])
                ph = float(bbox_for_bar_like[3])
                if pw <= 0 or ph <= 0:
                    continue
                if self._is_bar_like_rectangle_bbox(bbox_for_bar_like, bbox_w, bbox_h):
                    bar_like_rect_count += 1
                    if isinstance(pb_axes, (list, tuple)) and len(pb_axes) == 4:
                        bar_like_axes_boxes.append(tuple(float(v) for v in pb_axes))
        has_bar_like_rect = bar_like_rect_count >= 3

        line_infos = []
        errorbar_cap_color_keys = set()
        for line in ax.lines:
            bbox = self._get_artist_bbox_fingerprint(line)
            line_w = float(bbox[2]) if isinstance(bbox, (list, tuple)) and len(bbox) == 4 else 0.0
            line_h = float(bbox[3]) if isinstance(bbox, (list, tuple)) and len(bbox) == 4 else 0.0
            spans_x = ax_bw > 0 and line_w >= ax_bw * self.AXIS_SPAN_RATIO_INDICATOR
            spans_y = ax_bh > 0 and line_h >= ax_bh * self.AXIS_SPAN_RATIO_INDICATOR
            marker = line.get_marker()
            marker_is_none = marker in self.NONE_MARKERS
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            is_two_point_line = len(xdata) <= 2 and len(ydata) <= 2
            is_full_span = False
            try:
                trans = line.get_transform()
                if isinstance(trans, mtransforms.BlendedGenericTransform):
                    is_full_span = True
            except Exception:
                pass
            if str(marker) in self.ERRORBAR_MARKER_SYMBOLS:
                try:
                    color_key = str(self._round_color(line.get_color()))
                except Exception:
                    color_key = None
                if color_key:
                    errorbar_cap_color_keys.add(color_key)
            line_infos.append(
                {
                    "bbox": bbox,
                    "marker_is_none": marker_is_none,
                    "is_two_point_line": is_two_point_line,
                    "is_full_span": is_full_span,
                    "is_axis_spanning": spans_x or spans_y,
                    "n_points": min(len(xdata), len(ydata)),
                }
            )

        axis_spanning_main_or_overlay = [
            i
            for i, info in enumerate(line_infos)
            if info["is_axis_spanning"]
            and (not info["is_full_span"])
            and int(info["n_points"]) >= 3
        ]
        overlay_indicator_indexes = set(axis_spanning_main_or_overlay) if has_bar_like_rect else set()

        return {
            "indicator_start_idx": indicator_start_idx,
            "feature_start_idx": feature_start_idx,
            "exclude_texts": exclude_texts,
            "suppress_geometric_axis_payload": suppress_geometric_axis_payload,
            "processed_patches": processed_patches,
            "pie_geo_targets": pie_geo_targets,
            "has_annotation": has_annotation,
            "has_filled_area": has_filled_area,
            "has_scatter_collection": has_scatter_collection,
            "is_violin_like_axis": is_violin_like_axis,
            "ax_bw": ax_bw,
            "ax_bh": ax_bh,
            "bar_like_axes_boxes": bar_like_axes_boxes,
            "has_bar_like_rect": has_bar_like_rect,
            "line_infos": line_infos,
            "errorbar_cap_color_keys": errorbar_cap_color_keys,
            "overlay_indicator_indexes": overlay_indicator_indexes,
        }

    def _finalize_ax_stage(self, ax_index, ctx):
        self._merge_bracket_indicator_fragments(ctx["indicator_start_idx"])
        self._merge_line_connector_rectangles(ax_index, ctx["feature_start_idx"]["2_connector"])
        if ctx["suppress_geometric_axis_payload"]:
            self._prune_geometric_axis_payload(ax_index, ctx["feature_start_idx"])

    def _extract_from_ax(self, ax, ax_index=None):
        ctx = self._build_ax_stage_context(ax)
        self._extract_line_stage(ax, ax_index, ctx)
        self._extract_collection_stage(ax, ax_index, ctx)
        self._extract_patch_stage(ax, ax_index, ctx)
        self._extract_color_stage(ax, ax_index, ctx)
        self._extract_text_stage(ax, ax_index, ctx)
        self._finalize_ax_stage(ax_index, ctx)

    def extract(self):
        if len(self.axes) > 1:
            self._check_geometric_structures()
        self._extract_figure_level_connectors()

        for ax_index, ax in enumerate(self.axes):
            if not self._is_ax_allowed(ax):
                continue
            self._extract_from_ax(ax, ax_index=ax_index)

        self._extract_figure_level_texts()
        self._filter_grid_like_indicators()
        return self.features
