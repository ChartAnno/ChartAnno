import matplotlib.colors as mcolors
import matplotlib.offsetbox as moffsetbox
import matplotlib.patches as mpatches
import numpy as np


class EnclosureAnnotationMixin:
    def _iter_annotation_bbox_text_areas(self, artist):
        if artist is None:
            return

        seen = set()

        def walk(node):
            if node is None:
                return
            node_id = id(node)
            if node_id in seen:
                return
            seen.add(node_id)

            if isinstance(node, moffsetbox.TextArea):
                yield node
                return

            for child in getattr(node, "get_children", lambda: [])():
                yield from walk(child)

        yield from walk(artist)

    def _annotation_bbox_text_content(self, artist):
        parts = []
        for text_area in self._iter_annotation_bbox_text_areas(artist):
            content = self._get_textlike_content(text_area)
            if isinstance(content, str) and content.strip():
                parts.append(content.strip())
        if not parts:
            return None
        return "\n".join(parts)

    def _has_visible_patch_fill(self, patch):
        try:
            return self._alpha_from_color(patch.get_facecolor()) > 0.0
        except Exception:
            return False

    def _is_near_white_fill(self, color, *, luminance_threshold=0.985, chroma_threshold=0.03):
        try:
            r, g, b, a = mcolors.to_rgba(color)
        except Exception:
            return False
        if float(a) <= 0.0:
            return False
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        chroma = max(r, g, b) - min(r, g, b)
        return luminance >= luminance_threshold and chroma <= chroma_threshold

    def _collection_facecolor(self, coll):
        try:
            facecolors = coll.get_facecolors()
        except Exception:
            facecolors = None
        try:
            arr = np.asarray(facecolors, dtype=float)
        except Exception:
            arr = None
        if arr is None or arr.size == 0:
            return None
        if arr.ndim == 1:
            vals = arr[:4]
        else:
            vals = arr[0][:4]
        if len(vals) < 4:
            return None
        return tuple(float(v) for v in vals)

    def _has_visible_patch_edge(self, patch):
        try:
            linewidth = float(patch.get_linewidth() or 0.0)
        except Exception:
            linewidth = 0.0
        if linewidth <= 0.0:
            return False
        try:
            edge = patch.get_edgecolor()
        except Exception:
            return False
        if self._alpha_from_color(edge) <= 0.0:
            return False
        try:
            face = patch.get_facecolor()
        except Exception:
            face = None

        edge_is_near_white = self._is_near_white_fill(edge)
        face_is_near_white = self._is_near_white_fill(face) if face is not None else False

        try:
            edge_rgba = self._round_color(mcolors.to_rgba(edge))
        except Exception:
            edge_rgba = None
        try:
            face_rgba = self._round_color(mcolors.to_rgba(face)) if face is not None else None
        except Exception:
            face_rgba = None

        if edge_is_near_white and (face_is_near_white or (face_rgba is not None and face_rgba == edge_rgba)):
            return False
        return True

    def _is_inset_indicator_artist(self, artist):
        if artist is None:
            return False
        if type(artist).__name__ != "InsetIndicator":
            return False
        return hasattr(artist, "rectangle") and hasattr(artist, "connectors")

    def _iter_inset_indicator_patch_parts(self, artist):
        if not self._is_inset_indicator_artist(artist):
            return []
        parts = []
        try:
            rect = getattr(artist, "rectangle", None)
            if isinstance(rect, mpatches.Patch):
                parts.append(rect)
        except Exception:
            pass
        try:
            for conn in getattr(artist, "connectors", ()) or ():
                if isinstance(conn, mpatches.Patch):
                    parts.append(conn)
        except Exception:
            pass
        return parts

    def _has_semantic_patch_fill(self, patch):
        if not self._has_visible_patch_fill(patch):
            return False
        if self._has_visible_patch_edge(patch):
            return True
        try:
            return not self._is_near_white_fill(patch.get_facecolor())
        except Exception:
            return True

    def _append_annotation_bbox_enclosure(self, artist, *, ax=None, ax_index=None):
        if artist is None:
            return
        patch = getattr(artist, "patch", None)
        if patch is None:
            return
        patch_id = id(patch)
        if patch_id in self._seen_text_enclosure_patches:
            return
        if not (
            self._has_semantic_patch_fill(patch)
            or self._has_visible_patch_edge(patch)
        ):
            return

        bbox = self._get_artist_bbox_fingerprint(patch) or self._get_artist_bbox_fingerprint(artist)
        if bbox is None:
            return
        if not self._bbox_intersects_unit_square(bbox):
            return

        feature = {
            "src": "text_bbox_enclosure",
            "bbox": bbox,
            "color": self._round_color(
                patch.get_facecolor()
                if self._has_semantic_patch_fill(patch)
                else patch.get_edgecolor()
            ),
            "edgecolor": self._round_color(getattr(patch, "get_edgecolor", lambda: None)()),
            "alpha": float(
                patch.get_alpha()
                if patch.get_alpha() is not None
                else (
                    self._alpha_from_color(patch.get_facecolor())
                    or self._alpha_from_color(patch.get_edgecolor())
                )
            ),
        }
        if ax is not None:
            feature["ax_index"] = ax_index
            axes_bbox = self._get_artist_axes_bbox_fingerprint(patch, ax) or self._get_artist_axes_bbox_fingerprint(
                artist, ax
            )
            if axes_bbox is not None:
                feature["axes_bbox"] = axes_bbox

        text_content = self._annotation_bbox_text_content(artist)
        if isinstance(text_content, str) and text_content.strip():
            feature["text_content"] = text_content

        self.features["1_enclosure"].append(feature)
        self._seen_text_enclosure_patches.add(patch_id)

    def _append_text_bbox_enclosure(self, artist, *, ax=None, ax_index=None):
        if artist is None or not hasattr(artist, "get_bbox_patch"):
            return
        try:
            bbox_patch = artist.get_bbox_patch()
        except Exception:
            bbox_patch = None
        if bbox_patch is None:
            return
        patch_id = id(bbox_patch)
        if patch_id in self._seen_text_enclosure_patches:
            return
        if not (
            self._has_semantic_patch_fill(bbox_patch)
            or self._has_visible_patch_edge(bbox_patch)
        ):
            return

        bbox = self._get_text_bbox_fingerprint(artist) or self._get_artist_bbox_fingerprint(bbox_patch)
        if bbox is None:
            return

        feature = {
            "src": "text_bbox_enclosure",
            "bbox": bbox,
            "color": self._round_color(
                bbox_patch.get_facecolor()
                if self._has_semantic_patch_fill(bbox_patch)
                else bbox_patch.get_edgecolor()
            ),
            "edgecolor": self._round_color(getattr(bbox_patch, "get_edgecolor", lambda: None)()),
            "alpha": float(
                bbox_patch.get_alpha()
                if bbox_patch.get_alpha() is not None
                else (
                    self._alpha_from_color(bbox_patch.get_facecolor())
                    or self._alpha_from_color(bbox_patch.get_edgecolor())
                )
            ),
        }
        if ax is not None:
            feature["ax_index"] = ax_index
            axes_bbox = self._get_text_axes_bbox_fingerprint(artist, ax) or self._get_artist_axes_bbox_fingerprint(
                bbox_patch, ax
            )
            if axes_bbox is not None:
                feature["axes_bbox"] = axes_bbox
        try:
            content = artist.get_text()
        except Exception:
            content = None
        if isinstance(content, str) and content.strip():
            feature["text_content"] = content

        self.features["1_enclosure"].append(feature)
        self._seen_text_enclosure_patches.add(patch_id)

    def _extract_patch_stage(self, ax, ax_index, ctx):
        processed_patches = ctx["processed_patches"]
        pie_geo_targets = ctx["pie_geo_targets"]
        has_scatter_collection = ctx["has_scatter_collection"]
        has_bar_like_rect = ctx["has_bar_like_rect"]
        bar_like_axes_boxes = ctx["bar_like_axes_boxes"]
        ax_bw = ctx["ax_bw"]
        ax_bh = ctx["ax_bh"]

        patch_artists = list(ax.patches)
        for artist in getattr(ax, "artists", []) or []:
            patch_artists.extend(self._iter_inset_indicator_patch_parts(artist))

        for p in patch_artists:
            if not self._is_artist_allowed(p):
                continue
            color = self._round_color(p.get_facecolor())
            edge_color = self._round_color(p.get_edgecolor()) if hasattr(p, "get_edgecolor") else None
            alpha = p.get_alpha()
            if alpha is None:
                alpha = self._alpha_from_color(p.get_facecolor()) or self._alpha_from_color(
                    p.get_edgecolor() if hasattr(p, "get_edgecolor") else None
                )
            alpha = alpha if alpha is not None else 1.0
            fill = p.get_fill()
            bbox = self._get_artist_bbox_fingerprint(p)
            axes_bbox = self._get_artist_axes_bbox_fingerprint(p, ax)
            patch_type = type(p).__name__
            if patch_type == "BboxConnector":
                edge_color = self._round_color(p.get_edgecolor()) if hasattr(p, "get_edgecolor") else color
                try:
                    linestyle = str(p.get_linestyle())
                except Exception:
                    linestyle = "solid"
                self.features["2_connector"].append(
                    {
                        "src": "bbox_connector",
                        "ax_index": ax_index,
                        "linestyle": linestyle,
                        "color": edge_color,
                        "bbox": bbox,
                        "axes_bbox": axes_bbox,
                        "orientation": self._line_like_orientation(bbox),
                    }
                )
                processed_patches.add(p)
                continue
            if isinstance(p, (mpatches.FancyArrow, mpatches.Arrow, mpatches.FancyArrowPatch)) or self._is_arrow_like_polygon(p):
                self.features["2_connector"].append(
                    {"src": "patch_arrow", "ax_index": ax_index, "color": color, "bbox": bbox, "axes_bbox": axes_bbox}
                )
                processed_patches.add(p)
                continue
            if isinstance(p, mpatches.Arc):
                edge_color = self._round_color(p.get_edgecolor()) if hasattr(p, "get_edgecolor") else color
                try:
                    linestyle = str(p.get_linestyle())
                except Exception:
                    linestyle = "solid"
                self.features["2_connector"].append(
                    {
                        "src": "arc_connector",
                        "ax_index": ax_index,
                        "bbox": bbox,
                        "axes_bbox": axes_bbox,
                        "color": edge_color,
                        "linestyle": linestyle,
                    }
                )
                processed_patches.add(p)
                continue

            is_enclosure = False
            bbox_for_bar_like = axes_bbox if isinstance(axes_bbox, (list, tuple)) and len(axes_bbox) == 4 else bbox
            bbox_w = 1.0 if bbox_for_bar_like is axes_bbox else ax_bw
            bbox_h = 1.0 if bbox_for_bar_like is axes_bbox else ax_bh
            is_bar_like_rect = isinstance(p, mpatches.Rectangle) and self._is_contextual_bar_like_rectangle_bbox(
                bbox_for_bar_like, bbox_w, bbox_h, bar_like_axes_boxes
            )
            if (
                is_bar_like_rect
                and isinstance(p, mpatches.Rectangle)
                and self._has_visible_patch_edge(p)
                and not self._has_visible_patch_fill(p)
            ):
                is_bar_like_rect = False
            if (
                is_bar_like_rect
                and isinstance(p, mpatches.Rectangle)
                and has_scatter_collection
                and not has_bar_like_rect
            ):
                is_bar_like_rect = False
            is_background_rect = self._is_background_rectangle(
                p,
                bbox_for_bar_like,
                bbox_w,
                bbox_h,
                bar_like_axes_boxes=bar_like_axes_boxes,
                ax=ax,
            )
            if (
                not is_bar_like_rect
                and alpha < 1.0
                and fill
                and self._has_semantic_patch_fill(p)
            ):
                is_enclosure = True
            if (
                not is_enclosure
                and not is_bar_like_rect
                and self._has_visible_patch_edge(p)
                and not self._has_visible_patch_fill(p)
            ):
                is_enclosure = True
            if is_background_rect:
                is_enclosure = True
            if is_enclosure:
                self.features["1_enclosure"].append(
                    {
                        "src": "patch_enclosure",
                        "ax_index": ax_index,
                        "bbox": bbox,
                        "axes_bbox": axes_bbox,
                        "color": color if self._has_semantic_patch_fill(p) else edge_color,
                        "alpha": alpha,
                    }
                )
                processed_patches.add(p)
                continue
            if isinstance(p, mpatches.Wedge):
                if p in pie_geo_targets:
                    self.features["7_geometric"].append({"src": "pie_wedge", "color": color, "bbox": bbox})
                processed_patches.add(p)
