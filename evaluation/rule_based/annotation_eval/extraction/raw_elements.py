import matplotlib.collections as mcollections
import matplotlib.lines as mlines
import matplotlib.offsetbox as moffsetbox
import matplotlib.patches as mpatches
import matplotlib.text as mtext
import numpy as np

from annotation_eval.extraction.element_extractor import ChartAnnotationExtractor
from annotation_eval.extraction.geometry import (
    clip_display_bounds_to_axes,
    display_bounds_to_fig_norm,
    is_valid_norm_bbox,
    norm_bbox_intersects_unit,
    norm_bbox_positive_visible_extent,
)


class ChartRawElementExtractor(ChartAnnotationExtractor):
    def __init__(self, fig):
        super().__init__(fig)
        self.records = []

    def _is_secondary_shared_axis_text(self, text_role, ax, ax_index):
        if ax is None or ax_index is None or ax_index <= 0:
            return False
        if text_role in {"x_tick_label", "x_axis_label"}:
            try:
                shared_axes = ax.get_shared_x_axes()
            except Exception:
                shared_axes = None
        elif text_role in {"y_tick_label", "y_axis_label"}:
            try:
                shared_axes = ax.get_shared_y_axes()
            except Exception:
                shared_axes = None
        else:
            return False
        if shared_axes is None:
            return False
        for prev_ax in self.axes[:ax_index]:
            try:
                if shared_axes.joined(ax, prev_ax):
                    return True
            except Exception:
                continue
        return False

    def _infer_text_role(self, artist, ax):
        if artist is None:
            return None
        if ax is None:
            return "figure_text"
        if artist is ax.title:
            return "title"
        if artist is getattr(ax, "_left_title", None):
            return "left_title"
        if artist is getattr(ax, "_right_title", None):
            return "right_title"
        if artist is ax.xaxis.label:
            return "x_axis_label"
        if artist is ax.yaxis.label:
            return "y_axis_label"
        for label in ax.get_xticklabels():
            if artist is label:
                return "x_tick_label"
        for label in ax.get_yticklabels():
            if artist is label:
                return "y_tick_label"
        legend = ax.get_legend()
        if legend is not None:
            if artist is legend.get_title():
                return "legend_title"
            for text in legend.get_texts():
                if artist is text:
                    return "legend_text"
        return "text"

    def _append_record(self, kind, artist, *, ax=None, ax_index=None, bbox=None, axes_bbox=None, **kwargs):
        record = {
            "kind": kind,
            "artist_id": id(artist) if artist is not None else None,
        }
        if ax is not None:
            record["ax_index"] = ax_index
            record["ax_artist_id"] = id(ax)
            ax_bbox = self._get_artist_bbox_fingerprint(ax)
            if ax_bbox is not None:
                record["container_axes_bbox"] = ax_bbox
        if bbox is not None:
            record["bbox"] = bbox
        if axes_bbox is not None:
            record["axes_bbox"] = axes_bbox
        for key, value in kwargs.items():
            if value is not None:
                record[key] = value
        self.records.append(record)

    def _append_text_record(self, artist, seen_artist_ids=None, seen_tick_keys=None, *, ax=None, ax_index=None):
        if artist is None:
            return
        artist_id = id(artist)
        if seen_artist_ids is not None:
            if artist_id in seen_artist_ids:
                return
            seen_artist_ids.add(artist_id)

        content = self._get_textlike_content(artist)
        if not (isinstance(content, str) and content.strip()):
            return

        bbox = self._get_text_bbox_fingerprint(artist)
        if bbox is None:
            try:
                bbox = self._point_bbox_from_transform(
                    artist.get_position()[0],
                    artist.get_position()[1],
                    artist.get_transform(),
                    radius_px=0.0,
                )
            except Exception:
                bbox = None
        if bbox is None:
            return

        text_role = self._infer_text_role(artist, ax)
        if self._is_secondary_shared_axis_text(text_role, ax, ax_index):
            return
        if text_role in {"x_tick_label", "y_tick_label"} and seen_tick_keys is not None:
            try:
                bbox_key = tuple(round(float(v), 2) for v in bbox)
            except Exception:
                bbox_key = tuple(bbox) if isinstance(bbox, (list, tuple)) else bbox
            dedupe_key = (text_role, content, bbox_key)
            if dedupe_key in seen_tick_keys:
                return
            seen_tick_keys.add(dedupe_key)

        axes_bbox = self._get_text_axes_bbox_fingerprint(artist, ax) if ax is not None else None
        bbox_patch = artist.get_bbox_patch() if hasattr(artist, "get_bbox_patch") else None
        self._append_record(
            "text",
            artist,
            ax=ax,
            ax_index=ax_index,
            bbox=bbox,
            axes_bbox=axes_bbox,
            content=content,
            text_role=text_role,
            color=self._round_color(self._get_textlike_color(artist)),
            has_bbox_patch=bool(bbox_patch is not None),
            bbox_fill_color=(
                self._round_color(bbox_patch.get_facecolor())
                if bbox_patch is not None
                else None
            ),
            bbox_edge_color=(
                self._round_color(bbox_patch.get_edgecolor())
                if bbox_patch is not None
                else None
            ),
        )

    def _append_annotation_arrow_record(self, artist, seen_artist_ids=None, *, ax=None, ax_index=None):
        if not isinstance(artist, mtext.Annotation):
            return
        if not getattr(artist, "arrowprops", None):
            return

        dedupe_key = ("annotation_arrow", id(artist))
        if seen_artist_ids is not None:
            if dedupe_key in seen_artist_ids:
                return
            seen_artist_ids.add(dedupe_key)

        arrow_patch = getattr(artist, "arrow_patch", None)
        bbox = self._get_artist_bbox_fingerprint(arrow_patch) if arrow_patch is not None else None
        if bbox is None:
            bbox = self._get_text_bbox_fingerprint(artist)
        if bbox is None:
            return

        axes_bbox = None
        if ax is not None:
            axes_bbox = (
                self._get_artist_axes_bbox_fingerprint(arrow_patch, ax)
                if arrow_patch is not None
                else None
            ) or self._get_text_axes_bbox_fingerprint(artist, ax)

        content = self._get_textlike_content(artist)
        arrowprops = getattr(artist, "arrowprops", {}) or {}
        self._append_record(
            "annotation_arrow",
            artist,
            ax=ax,
            ax_index=ax_index,
            bbox=bbox,
            axes_bbox=axes_bbox,
            text_content=content if isinstance(content, str) and content.strip() else None,
            arrow_style=str(arrowprops),
            color=self._round_color(arrowprops.get("color") or arrowprops.get("edgecolor", "black")),
        )

    def _append_annotation_bbox_record(self, artist, *, ax=None, ax_index=None):
        if not isinstance(artist, moffsetbox.AnnotationBbox):
            return
        patch = getattr(artist, "patch", None)
        bbox = self._get_artist_bbox_fingerprint(patch) if patch is not None else None
        if bbox is None:
            bbox = self._get_artist_bbox_fingerprint(artist)
        if bbox is None:
            return
        axes_bbox = None
        if ax is not None:
            axes_bbox = (
                self._get_artist_axes_bbox_fingerprint(patch, ax)
                if patch is not None
                else None
            ) or self._get_artist_axes_bbox_fingerprint(artist, ax)

        self._append_record(
            "annotation_bbox",
            artist,
            ax=ax,
            ax_index=ax_index,
            bbox=bbox,
            axes_bbox=axes_bbox,
            text_content=self._annotation_bbox_text_content(artist),
            facecolor=(
                self._round_color(patch.get_facecolor())
                if patch is not None and hasattr(patch, "get_facecolor")
                else None
            ),
            edgecolor=(
                self._round_color(patch.get_edgecolor())
                if patch is not None and hasattr(patch, "get_edgecolor")
                else None
            ),
            linewidth=(
                float(patch.get_linewidth() or 0.0)
                if patch is not None and hasattr(patch, "get_linewidth")
                else None
            ),
        )

    def _append_line_record(self, line, *, ax=None, ax_index=None, kind="line"):
        if line is None:
            return
        try:
            if not line.get_visible():
                return
        except Exception:
            pass

        bbox = self._get_artist_bbox_fingerprint(line)
        if bbox is None:
            return
        axes_bbox = self._get_artist_axes_bbox_fingerprint(line, ax) if ax is not None else None
        try:
            clip_on = bool(line.get_clip_on())
        except Exception:
            clip_on = True
        if ax is not None and clip_on and axes_bbox is not None and not self._norm_bbox_intersects_unit(axes_bbox):
            return
        try:
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            n_points = min(len(xdata), len(ydata))
        except Exception:
            xdata = None
            ydata = None
            n_points = 0
        is_full_span = False
        try:
            transform = line.get_transform()
            import matplotlib.transforms as mtransforms

            if isinstance(transform, mtransforms.BlendedGenericTransform):
                is_full_span = True
        except Exception:
            pass

        point_signature = None
        data_signature = None
        try:
            pts = self._line_xy_points(line)
            if pts:
                point_signature = sorted(pts)[:64]
                data_signature = self._sample_point_sequence(pts, limit=128)
        except Exception:
            point_signature = None
            data_signature = None

        self._append_record(
            kind,
            line,
            ax=ax,
            ax_index=ax_index,
            bbox=bbox,
            axes_bbox=axes_bbox,
            type=type(line).__name__,
            linestyle=str(line.get_linestyle()),
            marker=str(line.get_marker()),
            color=self._round_color(line.get_color()),
            linewidth=float(line.get_linewidth() or 0.0),
            n_points=n_points,
            point_signature=point_signature,
            data_signature=data_signature,
            label=self._get_line_label_text(line),
            clip_on=clip_on,
            is_full_span=is_full_span,
        )

    def _is_valid_norm_bbox(self, bbox):
        return is_valid_norm_bbox(bbox)

    def _norm_bbox_intersects_unit(self, bbox):
        return norm_bbox_intersects_unit(bbox)

    def _norm_bbox_positive_visible_extent(self, bbox):
        return norm_bbox_positive_visible_extent(bbox)

    def _display_bounds_to_fig_norm(self, x0, y0, w, h):
        return display_bounds_to_fig_norm((x0, y0, w, h), self.canvas_width_px, self.canvas_height_px)

    def _clip_display_bounds_to_axes(self, bounds, ax):
        if bounds is None or ax is None:
            return bounds
        try:
            ax_bbox = ax.get_window_extent(self.renderer)
        except Exception:
            return bounds
        return clip_display_bounds_to_axes(bounds, ax_bbox, allow_zero_area=True)

    def _line_collection_display_bounds(self, coll):
        try:
            segments = coll.get_segments()
        except Exception:
            return None
        if not segments:
            return None

        transform = coll.get_transform()
        min_x = min_y = np.inf
        max_x = max_y = -np.inf

        for segment in segments:
            try:
                pts = np.asarray(segment, dtype=float)
            except Exception:
                continue
            if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 2:
                continue
            pts = pts[:, :2]
            pts = pts[np.isfinite(pts).all(axis=1)]
            if pts.size == 0:
                continue
            try:
                disp = np.asarray(transform.transform(pts), dtype=float)
            except Exception:
                continue
            if disp.ndim != 2 or disp.shape[0] == 0 or disp.shape[1] < 2:
                continue
            disp = disp[:, :2]
            disp = disp[np.isfinite(disp).all(axis=1)]
            if disp.size == 0:
                continue
            min_x = min(min_x, float(np.min(disp[:, 0])))
            min_y = min(min_y, float(np.min(disp[:, 1])))
            max_x = max(max_x, float(np.max(disp[:, 0])))
            max_y = max(max_y, float(np.max(disp[:, 1])))

        if not np.isfinite([min_x, min_y, max_x, max_y]).all():
            return None
        return min_x, min_y, max(0.0, max_x - min_x), max(0.0, max_y - min_y)

    def _path_collection_offset_signature(self, coll, limit=64):
        try:
            offsets = coll.get_offsets()
        except Exception:
            return None
        if offsets is None:
            return None
        try:
            import numpy as np
            arr = np.asarray(offsets, dtype=float)
        except Exception:
            return None
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
            return None
        arr = arr[:, :2]
        arr = arr[np.isfinite(arr).all(axis=1)]
        if arr.size == 0:
            return None
        pts = [(round(float(x), 6), round(float(y), 6)) for x, y in arr[:limit]]
        pts.sort()
        return pts

    def _sample_point_sequence(self, pts, limit=128):
        if not pts:
            return None
        if len(pts) <= limit:
            return list(pts)
        idxs = np.linspace(0, len(pts) - 1, num=limit)
        seen = []
        last = None
        for idx in idxs:
            i = int(round(float(idx)))
            i = max(0, min(len(pts) - 1, i))
            pt = pts[i]
            if pt != last:
                seen.append(pt)
                last = pt
        return seen

    def _line_xy_points(self, line, limit=None):
        pts = []
        try:
            arr = np.asarray(line.get_xydata(), dtype=float)
        except Exception:
            arr = None
        if arr is not None and arr.ndim == 2 and arr.shape[1] >= 2:
            arr = arr[:, :2]
            arr = arr[np.isfinite(arr).all(axis=1)]
            for x, y in arr:
                pts.append((round(float(x), 6), round(float(y), 6)))
        if pts:
            return pts if limit is None else pts[:limit]
        try:
            xdata = line.get_xdata(orig=False)
            ydata = line.get_ydata(orig=False)
        except Exception:
            return None
        try:
            n = min(len(xdata), len(ydata))
        except Exception:
            return None
        for x, y in zip(xdata[:n], ydata[:n]):
            try:
                fx = float(x)
                fy = float(y)
            except Exception:
                continue
            if not np.isfinite(fx) or not np.isfinite(fy):
                continue
            pts.append((round(fx, 6), round(fy, 6)))
        return pts if pts else None

    def _collection_path_signature(self, coll, max_paths=4, max_vertices=256):
        try:
            paths = coll.get_paths()
        except Exception:
            return None
        if not paths:
            return None
        signature = []
        for path_idx, path in enumerate(paths[:max_paths]):
            try:
                verts = np.asarray(path.vertices, dtype=float)
            except Exception:
                continue
            if verts.ndim != 2 or verts.shape[0] == 0 or verts.shape[1] < 2:
                continue
            verts = verts[:, :2]
            verts = verts[np.isfinite(verts).all(axis=1)]
            if verts.size == 0:
                continue
            pts = [(round(float(x), 6), round(float(y), 6)) for x, y in verts]
            pts = self._sample_point_sequence(pts, limit=max_vertices)
            signature.append(pts)
        return signature or None

    def _path_collection_point_boxes(self, coll, ax, limit=64):
        try:
            offsets = np.asarray(coll.get_offsets(), dtype=float)
        except Exception:
            return None
        if offsets.ndim != 2 or offsets.shape[0] == 0 or offsets.shape[1] < 2:
            return None
        offsets = offsets[:, :2]
        offsets = offsets[np.isfinite(offsets).all(axis=1)]
        if offsets.size == 0:
            return None

        total_pts = len(offsets)
        try:
            sizes = np.asarray(coll.get_sizes(), dtype=float).ravel()
        except Exception:
            sizes = np.array([], dtype=float)
        if sizes.size == 0:
            sizes = np.full(total_pts, 36.0, dtype=float)
        elif sizes.size == 1 and total_pts > 1:
            sizes = np.full(total_pts, float(sizes[0]), dtype=float)
        elif sizes.size < total_pts:
            sizes = np.pad(sizes, (0, total_pts - sizes.size), mode="edge")
        else:
            sizes = sizes[:total_pts]

        offset_transform = (
            coll.get_offset_transform()
            if hasattr(coll, "get_offset_transform")
            else ax.transData
        )
        point_boxes = []
        for idx, (pt, size) in enumerate(zip(offsets, sizes)):
            if idx >= limit:
                break
            radius_px = self._scatter_radius_px(float(size))
            bbox = self._point_bbox_from_transform(
                pt[0],
                pt[1],
                offset_transform,
                radius_px=radius_px,
            )
            if bbox is None:
                continue
            axes_bbox = None
            if ax is not None:
                try:
                    x_disp, y_disp = offset_transform.transform((pt[0], pt[1]))
                    axes_bbox = self._display_bounds_to_axes_norm(
                        float(x_disp) - radius_px,
                        float(y_disp) - radius_px,
                        radius_px * 2.0,
                        radius_px * 2.0,
                        ax,
                    )
                except Exception:
                    axes_bbox = None
            point_boxes.append(
                {
                    "bbox": list(bbox) if isinstance(bbox, tuple) else bbox,
                    "axes_bbox": list(axes_bbox) if isinstance(axes_bbox, tuple) else axes_bbox,
                }
            )
        return point_boxes or None

    def _path_collection_display_bounds(self, coll):
        try:
            offsets = np.asarray(coll.get_offsets(), dtype=float)
        except Exception:
            return None
        if offsets.ndim != 2 or offsets.shape[0] == 0 or offsets.shape[1] < 2:
            return None
        offsets = offsets[:, :2]
        offsets = offsets[np.isfinite(offsets).all(axis=1)]
        if offsets.size == 0:
            return None

        try:
            disp = np.asarray(coll.get_offset_transform().transform(offsets), dtype=float)
        except Exception:
            return None
        if disp.ndim != 2 or disp.shape[0] == 0 or disp.shape[1] < 2:
            return None
        disp = disp[:, :2]
        disp = disp[np.isfinite(disp).all(axis=1)]
        if disp.size == 0:
            return None

        radius_px = 0.0
        try:
            sizes = np.asarray(coll.get_sizes(), dtype=float).ravel()
            if sizes.size > 0:
                radius_px = max(radius_px, self._scatter_radius_px(float(np.max(sizes))))
        except Exception:
            pass
        try:
            linewidths = np.asarray(coll.get_linewidths(), dtype=float).ravel()
            if linewidths.size > 0:
                radius_px = max(radius_px, float(np.max(linewidths)) / 2.0)
        except Exception:
            pass

        min_x = float(np.min(disp[:, 0])) - radius_px
        min_y = float(np.min(disp[:, 1])) - radius_px
        max_x = float(np.max(disp[:, 0])) + radius_px
        max_y = float(np.max(disp[:, 1])) + radius_px
        if not np.isfinite([min_x, min_y, max_x, max_y]).all():
            return None
        return min_x, min_y, max(0.0, max_x - min_x), max(0.0, max_y - min_y)

    def _generic_collection_display_bounds(self, coll):
        try:
            paths = coll.get_paths()
        except Exception:
            return None
        if not paths:
            return None

        try:
            transform = coll.get_transform()
        except Exception:
            return None

        min_x = min_y = np.inf
        max_x = max_y = -np.inf

        for path in paths:
            try:
                verts = np.asarray(path.vertices, dtype=float)
            except Exception:
                continue
            if verts.ndim != 2 or verts.shape[0] == 0 or verts.shape[1] < 2:
                continue
            verts = verts[:, :2]
            verts = verts[np.isfinite(verts).all(axis=1)]
            if verts.size == 0:
                continue
            try:
                disp = np.asarray(transform.transform(verts), dtype=float)
            except Exception:
                continue
            if disp.ndim != 2 or disp.shape[0] == 0 or disp.shape[1] < 2:
                continue
            disp = disp[:, :2]
            disp = disp[np.isfinite(disp).all(axis=1)]
            if disp.size == 0:
                continue
            min_x = min(min_x, float(np.min(disp[:, 0])))
            min_y = min(min_y, float(np.min(disp[:, 1])))
            max_x = max(max_x, float(np.max(disp[:, 0])))
            max_y = max(max_y, float(np.max(disp[:, 1])))

        if not np.isfinite([min_x, min_y, max_x, max_y]).all():
            return None
        return min_x, min_y, max(0.0, max_x - min_x), max(0.0, max_y - min_y)

    def _append_collection_record(self, coll, *, ax=None, ax_index=None):
        if coll is None:
            return
        try:
            if not coll.get_visible():
                return
        except Exception:
            pass

        bbox = self._get_artist_bbox_fingerprint(coll)
        axes_bbox = self._get_artist_axes_bbox_fingerprint(coll, ax) if ax is not None else None

        item_count = None
        if isinstance(coll, mcollections.LineCollection):
            try:
                item_count = len(coll.get_segments())
            except Exception:
                item_count = None
            color = self._round_color(self._line_collection_color(coll))
            display_bounds = self._line_collection_display_bounds(coll)
            if display_bounds is not None:
                display_bounds = self._clip_display_bounds_to_axes(display_bounds, ax)
            if display_bounds is not None:
                bbox = self._display_bounds_to_fig_norm(*display_bounds)
                if ax is not None:
                    axes_bbox = self._display_bounds_to_axes_norm(*display_bounds, ax)
        elif isinstance(coll, mcollections.PathCollection):
            try:
                item_count = len(coll.get_offsets())
            except Exception:
                item_count = None
            offset_signature = self._path_collection_offset_signature(coll)
            point_boxes = self._path_collection_point_boxes(coll, ax)
            color = None
            try:
                face = coll.get_facecolors()
                if face is not None and len(face) > 0:
                    color = self._round_color(tuple(face[0]))
            except Exception:
                color = None
            if color is None:
                try:
                    edge = coll.get_edgecolors()
                    if edge is not None and len(edge) > 0:
                        color = self._round_color(tuple(edge[0]))
                except Exception:
                    color = None
            display_bounds = self._path_collection_display_bounds(coll)
            if display_bounds is not None:
                display_bounds = self._clip_display_bounds_to_axes(display_bounds, ax)
            if display_bounds is not None:
                bbox = self._display_bounds_to_fig_norm(*display_bounds)
                if ax is not None:
                    axes_bbox = self._display_bounds_to_axes_norm(*display_bounds, ax)
        else:
            color = None
            try:
                face = coll.get_facecolors()
                if face is not None and len(face) > 0:
                    color = self._round_color(tuple(face[0]))
            except Exception:
                color = None
            display_bounds = self._generic_collection_display_bounds(coll)
            if display_bounds is not None:
                display_bounds = self._clip_display_bounds_to_axes(display_bounds, ax)
            if display_bounds is not None:
                bbox = self._display_bounds_to_fig_norm(*display_bounds)
                if ax is not None:
                    axes_bbox = self._display_bounds_to_axes_norm(*display_bounds, ax)

        try:
            linewidths = np.asarray(coll.get_linewidths(), dtype=float).ravel()
            linewidth = float(np.max(linewidths)) if linewidths.size > 0 else 0.0
        except Exception:
            linewidth = 0.0

        if not self._is_valid_norm_bbox(bbox):
            return
        if (
            isinstance(coll, mcollections.PolyCollection)
            and not isinstance(coll, (mcollections.LineCollection, mcollections.PathCollection))
            and isinstance(axes_bbox, (list, tuple))
            and len(axes_bbox) == 4
            and not self._norm_bbox_positive_visible_extent(axes_bbox)
        ):
            return

        extra_kwargs = {}
        if isinstance(coll, mcollections.PathCollection):
            extra_kwargs["offset_signature"] = offset_signature
            extra_kwargs["point_boxes"] = point_boxes
        elif isinstance(coll, mcollections.PolyCollection):
            extra_kwargs["path_signature"] = self._collection_path_signature(coll)

        self._append_record(
            "collection",
            coll,
            ax=ax,
            ax_index=ax_index,
            bbox=bbox,
            axes_bbox=axes_bbox,
            type=type(coll).__name__,
            color=color,
            item_count=item_count,
            linewidth=linewidth,
            **extra_kwargs,
        )

    def _append_patch_record(self, patch, *, ax=None, ax_index=None):
        if patch is None or patch is getattr(ax, "patch", None):
            return
        try:
            if not patch.get_visible():
                return
        except Exception:
            pass
        bbox = self._get_artist_bbox_fingerprint(patch)
        if bbox is None:
            return
        axes_bbox = self._get_artist_axes_bbox_fingerprint(patch, ax) if ax is not None else None
        try:
            alpha = patch.get_alpha()
        except Exception:
            alpha = None
        extra = {}
        if isinstance(patch, mpatches.Wedge):
            try:
                cx, cy = patch.center
                extra["center"] = [round(float(cx), 6), round(float(cy), 6)]
            except Exception:
                pass
            for attr in ("r", "theta1", "theta2", "width"):
                try:
                    value = getattr(patch, attr)
                except Exception:
                    value = None
                if value is None:
                    continue
                try:
                    extra[attr] = round(float(value), 6)
                except Exception:
                    pass
        self._append_record(
            "patch",
            patch,
            ax=ax,
            ax_index=ax_index,
            bbox=bbox,
            axes_bbox=axes_bbox,
            type=type(patch).__name__,
            facecolor=self._round_color(patch.get_facecolor()) if hasattr(patch, "get_facecolor") else None,
            edgecolor=self._round_color(patch.get_edgecolor()) if hasattr(patch, "get_edgecolor") else None,
            linewidth=float(patch.get_linewidth() or 0.0) if hasattr(patch, "get_linewidth") else None,
            fill=bool(patch.get_fill()) if hasattr(patch, "get_fill") else None,
            alpha=float(alpha) if alpha is not None else None,
            **extra,
        )

    def _append_table_cell_record(self, cell, *, ax=None, ax_index=None):
        if cell is None:
            return
        bbox = self._get_artist_bbox_fingerprint(cell)
        if bbox is None:
            return
        axes_bbox = self._get_artist_axes_bbox_fingerprint(cell, ax) if ax is not None else None
        content = None
        try:
            content = cell.get_text().get_text()
        except Exception:
            content = None
        self._append_record(
            "table_cell",
            cell,
            ax=ax,
            ax_index=ax_index,
            bbox=bbox,
            axes_bbox=axes_bbox,
            facecolor=self._round_color(cell.get_facecolor()) if hasattr(cell, "get_facecolor") else None,
            edgecolor=self._round_color(cell.get_edgecolor()) if hasattr(cell, "get_edgecolor") else None,
            content=content,
        )

    def extract(self):
        seen_text_artist_ids = set()
        seen_tick_keys = set()
        ax_texts = set()
        for ax in self.axes:
            ax_texts.update(ax.texts)
            ax_texts.update(
                [
                    ax.title,
                    getattr(ax, "_left_title", None),
                    getattr(ax, "_right_title", None),
                    ax.xaxis.label,
                    ax.yaxis.label,
                ]
            )

        for ax_index, ax in enumerate(self.axes):
            try:
                pos = ax.get_position()
                ax_bbox = [
                    round(float(pos.x0), 6),
                    round(float(pos.y0), 6),
                    round(float(pos.width), 6),
                    round(float(pos.height), 6),
                ]
            except Exception:
                ax_bbox = self._get_artist_bbox_fingerprint(ax)
            self._append_record(
                "axes",
                ax,
                ax=ax,
                ax_index=ax_index,
                bbox=ax_bbox,
                axes_bbox=None,
                type=type(ax).__name__,
            )

            for line in ax.lines:
                self._append_line_record(line, ax=ax, ax_index=ax_index, kind="line")

            for coll in ax.collections:
                self._append_collection_record(coll, ax=ax, ax_index=ax_index)

            for patch in ax.patches:
                self._append_patch_record(patch, ax=ax, ax_index=ax_index)

            for table in getattr(ax, "tables", []):
                try:
                    cells = table.get_celld()
                except Exception:
                    cells = {}
                for cell in cells.values():
                    self._append_table_cell_record(cell, ax=ax, ax_index=ax_index)

            for artist in getattr(ax, "artists", []) or []:
                inset_parts = self._iter_inset_indicator_patch_parts(artist)
                if inset_parts:
                    for part in inset_parts:
                        self._append_patch_record(part, ax=ax, ax_index=ax_index)
                    continue
                self._append_annotation_bbox_record(artist, ax=ax, ax_index=ax_index)

            for text_artist in self._iter_ax_text_artists(ax):
                self._append_text_record(
                    text_artist,
                    seen_text_artist_ids,
                    seen_tick_keys,
                    ax=ax,
                    ax_index=ax_index,
                )
                self._append_annotation_arrow_record(
                    text_artist,
                    seen_text_artist_ids,
                    ax=ax,
                    ax_index=ax_index,
                )

        seen_fig_text_ids = set()
        for text_artist in self.fig.texts:
            if text_artist in ax_texts:
                continue
            self._append_text_record(text_artist, seen_fig_text_ids)
        self._append_text_record(getattr(self.fig, "_suptitle", None), seen_fig_text_ids)

        seen_fig_line_ids = set()
        candidates = []
        try:
            candidates.extend(list(self.fig.lines))
        except Exception:
            pass
        try:
            candidates.extend(list(self.fig.artists))
        except Exception:
            pass
        for artist in candidates:
            if artist is None:
                continue
            artist_id = id(artist)
            if artist_id in seen_fig_line_ids:
                continue
            seen_fig_line_ids.add(artist_id)
            if isinstance(artist, mlines.Line2D):
                self._append_line_record(artist, kind="figure_line")

        for patch in getattr(self.fig, "patches", []) or []:
            if patch is getattr(self.fig, "patch", None):
                continue
            self._append_patch_record(patch)

        return self.records
