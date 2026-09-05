import matplotlib.offsetbox as moffsetbox
import matplotlib.text as mtext
import numpy as np


class TextAnnotationMixin:
    def _get_textlike_content(self, artist):
        if artist is None:
            return None
        try:
            content = artist.get_text()
        except Exception:
            content = None
        if isinstance(content, str) and content.strip():
            return content
        inner_text = getattr(artist, "_text", None)
        if inner_text is not None and hasattr(inner_text, "get_text"):
            try:
                content = inner_text.get_text()
            except Exception:
                content = None
            if isinstance(content, str) and content.strip():
                return content
        return None

    def _get_textlike_color(self, artist):
        if artist is None:
            return None
        try:
            color = artist.get_color()
        except Exception:
            color = None
        if color is not None:
            return color
        inner_text = getattr(artist, "_text", None)
        if inner_text is not None and hasattr(inner_text, "get_color"):
            try:
                return inner_text.get_color()
            except Exception:
                return None
        return None

    def _append_annotation_bbox_features(self, artist, seen_artist_ids=None, *, ax=None, ax_index=None):
        if not isinstance(artist, moffsetbox.AnnotationBbox):
            return

        for text_area in self._iter_annotation_bbox_text_areas(artist):
            self._append_text_feature(text_area, seen_artist_ids, ax=ax, ax_index=ax_index)

        self._append_annotation_bbox_enclosure(artist, ax=ax, ax_index=ax_index)

        arrow_patch = getattr(artist, "arrow_patch", None)
        if arrow_patch is not None:
            feature = {
                "src": "annotate_arrow",
                "ax_index": ax_index,
                "arrow_style": type(arrow_patch).__name__,
                "color": self._round_color(getattr(arrow_patch, "get_edgecolor", lambda: None)()),
                "bbox": self._get_artist_bbox_fingerprint(arrow_patch),
                "axes_bbox": self._get_artist_axes_bbox_fingerprint(arrow_patch, ax)
                if ax is not None
                else None,
            }
            text_content = self._annotation_bbox_text_content(artist)
            if isinstance(text_content, str) and text_content.strip():
                feature["text_content"] = text_content
            self.features["2_connector"].append(feature)

    def _extract_figure_level_texts(self):
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

        seen_artist_ids = set()
        for t in self.fig.texts:
            if t in ax_texts:
                continue
            if not self._is_artist_allowed(t):
                continue
            self._append_text_feature(t, seen_artist_ids)

        suptitle = getattr(self.fig, "_suptitle", None)
        if suptitle is not None and self._is_artist_allowed(suptitle):
            self._append_text_feature(suptitle, seen_artist_ids)

    def _append_text_feature(self, artist, seen_artist_ids=None, *, ax=None, ax_index=None):
        if artist is None:
            return

        artist_id = id(artist)
        if seen_artist_ids is not None:
            if artist_id in seen_artist_ids:
                return
            seen_artist_ids.add(artist_id)

        try:
            if not artist.get_visible():
                return
        except Exception:
            pass

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

        feature = {
            "content": content,
            "bbox": bbox,
            "color": self._round_color(self._get_textlike_color(artist)),
        }
        if ax is not None:
            feature["ax_index"] = ax_index
            axes_bbox = self._get_text_axes_bbox_fingerprint(artist, ax)
            try:
                clip_on = bool(artist.get_clip_on())
            except Exception:
                clip_on = False
            if clip_on and axes_bbox is not None and not self._bbox_intersects_unit_square(axes_bbox):
                return
            if axes_bbox is not None:
                feature["axes_bbox"] = axes_bbox
        self.features["3_text"].append(feature)
        self._append_text_bbox_enclosure(artist, ax=ax, ax_index=ax_index)

    def _iter_axis_tick_text_artists(self, axis):
        try:
            lo, hi = axis.get_view_interval()
            lo, hi = sorted((float(lo), float(hi)))
        except Exception:
            lo = hi = None

        def _in_view(loc):
            if lo is None or hi is None:
                return True
            try:
                value = float(loc)
            except Exception:
                return True
            eps = max(abs(hi - lo) * 1e-9, 1e-12)
            return (lo - eps) <= value <= (hi + eps)

        ticks = []
        try:
            ticks.extend(axis.get_major_ticks())
        except Exception:
            pass
        try:
            ticks.extend(axis.get_minor_ticks())
        except Exception:
            pass

        for tick in ticks:
            try:
                if not _in_view(tick.get_loc()):
                    continue
            except Exception:
                pass
            for label in (getattr(tick, "label1", None), getattr(tick, "label2", None)):
                if label is None:
                    continue
                try:
                    if not label.get_visible():
                        continue
                except Exception:
                    pass
                yield label

    def _iter_ax_text_artists(self, ax):
        for artist in ax.texts:
            yield artist

        yield ax.title
        yield getattr(ax, "_left_title", None)
        yield getattr(ax, "_right_title", None)
        yield ax.xaxis.label
        yield ax.yaxis.label

        if getattr(ax, "axison", True):
            try:
                xaxis_visible = bool(ax.xaxis.get_visible())
            except Exception:
                xaxis_visible = True
            try:
                yaxis_visible = bool(ax.yaxis.get_visible())
            except Exception:
                yaxis_visible = True
            if xaxis_visible:
                for artist in self._iter_axis_tick_text_artists(ax.xaxis):
                    yield artist
            if yaxis_visible:
                for artist in self._iter_axis_tick_text_artists(ax.yaxis):
                    yield artist

        legend = ax.get_legend()
        if legend is not None:
            yield legend.get_title()
            for artist in legend.get_texts():
                yield artist

    def _extract_text_stage(self, ax, ax_index, ctx):
        exclude_texts = ctx["exclude_texts"]
        seen_text_artist_ids = set()
        for t in ax.texts:
            if t in exclude_texts or not self._is_artist_allowed(t):
                continue
            self._append_text_feature(t, seen_text_artist_ids, ax=ax, ax_index=ax_index)
            content = t.get_text()
            if isinstance(t, mtext.Annotation) and t.arrowprops:
                arrow_color = t.arrowprops.get("color") or t.arrowprops.get("edgecolor", "black")
                if self._alpha_from_color(arrow_color) <= 0.0:
                    continue
                arrow_bbox = self._get_artist_bbox_fingerprint(t.arrow_patch) if t.arrow_patch else None
                feature = {
                    "src": "annotate_arrow",
                    "ax_index": ax_index,
                    "arrow_style": str(t.arrowprops),
                    "color": self._round_color(arrow_color),
                    "bbox": arrow_bbox,
                    "axes_bbox": self._get_artist_axes_bbox_fingerprint(t.arrow_patch, ax) if t.arrow_patch else None,
                }
                if isinstance(content, str) and content.strip():
                    feature["text_content"] = content
                self.features["2_connector"].append(feature)

        for artist in getattr(ax, "artists", []) or []:
            if not self._is_artist_allowed(artist):
                continue
            self._append_annotation_bbox_features(artist, seen_text_artist_ids, ax=ax, ax_index=ax_index)

        for text_artist in self._iter_ax_text_artists(ax):
            if not self._is_artist_allowed(text_artist):
                continue
            self._append_text_feature(text_artist, seen_text_artist_ids, ax=ax, ax_index=ax_index)
