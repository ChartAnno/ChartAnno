import matplotlib.lines as mlines


class ConnectorAnnotationMixin:
    def _resolve_coord_transform(self, ax, coord_spec):
        if coord_spec is None:
            return ax.transData

        if hasattr(coord_spec, "transform"):
            return coord_spec

        if isinstance(coord_spec, str):
            mapping = {
                "data": ax.transData,
                "axes fraction": ax.transAxes,
                "figure fraction": self.fig.transFigure,
            }
            return mapping.get(coord_spec, ax.transData)

        return ax.transData

    def _annotation_target_coord(self, annotation, ax):
        try:
            target_transform = self._resolve_coord_transform(ax, annotation.xycoords)
            return self._normalize_coord(annotation.xy[0], annotation.xy[1], target_transform)
        except Exception:
            return self._normalize_coord(annotation.xy[0], annotation.xy[1], ax.transData)

    def _extract_figure_level_connectors(self):
        seen_artist_ids = set()
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
            artist_id = id(artist)
            if artist_id in seen_artist_ids:
                continue
            seen_artist_ids.add(artist_id)

            if not self._is_artist_allowed(artist):
                continue

            if not isinstance(artist, mlines.Line2D):
                continue
            try:
                if not artist.get_visible():
                    continue
            except Exception:
                pass

            marker = artist.get_marker()
            if marker not in self.NONE_MARKERS:
                continue

            try:
                xdata = artist.get_xdata()
                ydata = artist.get_ydata()
                n_points = min(len(xdata), len(ydata))
            except Exception:
                n_points = 0
            if n_points <= 0 or n_points > 2:
                continue

            try:
                linestyle = str(artist.get_linestyle())
            except Exception:
                linestyle = "None"
            if linestyle.strip().lower() in {"none", "", "null"}:
                continue
            if self._is_light_color(artist.get_color()):
                continue

            bbox = self._get_artist_bbox_fingerprint(artist)
            if bbox is None:
                continue

            self.features["2_connector"].append(
                {
                    "src": "figure_line_connector",
                    "linestyle": linestyle,
                    "color": self._round_color(artist.get_color()),
                    "bbox": bbox,
                    "orientation": self._line_like_orientation(bbox),
                }
            )

    def _line_like_orientation(self, bbox):
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            return None
        try:
            _, _, w, h = [float(v) for v in bbox]
        except Exception:
            return None
        if abs(w) <= 1e-9 and h > 0:
            return "vertical"
        if abs(h) <= 1e-9 and w > 0:
            return "horizontal"
        return None

    def _merge_line_connector_rectangles(self, ax_index, start_idx):
        items = self.features["2_connector"][start_idx:]
        if not items:
            return

        def get_box(item):
            box = item.get("axes_bbox")
            if isinstance(box, (list, tuple)) and len(box) == 4:
                return tuple(float(v) for v in box), "axes_bbox"
            box = item.get("bbox")
            if isinstance(box, (list, tuple)) and len(box) == 4:
                return tuple(float(v) for v in box), "bbox"
            return None, None

        def axis_line(item):
            box, box_key = get_box(item)
            if box is None:
                return None
            x, y, w, h = box
            orientation = item.get("orientation")
            if orientation == "horizontal":
                return {
                    "box_key": box_key,
                    "x0": x,
                    "x1": x + w,
                    "y": y + h / 2.0,
                    "raw": item,
                }
            if orientation == "vertical":
                return {
                    "box_key": box_key,
                    "x": x + w / 2.0,
                    "y0": y,
                    "y1": y + h,
                    "raw": item,
                }
            return None

        def close(a, b, tol=0.03):
            return abs(float(a) - float(b)) <= tol

        candidate_indices = [
            i for i, item in enumerate(items)
            if isinstance(item, dict)
            and item.get("src") == "line_connector"
            and item.get("ax_index") == ax_index
            and item.get("orientation") in {"horizontal", "vertical"}
        ]
        if len(candidate_indices) < 4:
            return

        grouped = {}
        for idx in candidate_indices:
            item = items[idx]
            key = (str(item.get("color")), item.get("ax_index"))
            grouped.setdefault(key, []).append(idx)

        used = set()
        new_enclosures = []

        for group_indices in grouped.values():
            horizontals = []
            verticals = []
            for idx in group_indices:
                parsed = axis_line(items[idx])
                if parsed is None:
                    continue
                if items[idx].get("orientation") == "horizontal":
                    horizontals.append((idx, parsed))
                else:
                    verticals.append((idx, parsed))

            for hi in range(len(horizontals)):
                h_idx_1, h1 = horizontals[hi]
                if h_idx_1 in used:
                    continue
                for hj in range(hi + 1, len(horizontals)):
                    h_idx_2, h2 = horizontals[hj]
                    if h_idx_2 in used:
                        continue
                    if h1["box_key"] != h2["box_key"]:
                        continue
                    x0 = min(h1["x0"], h2["x0"])
                    x1 = max(h1["x1"], h2["x1"])
                    if not (close(h1["x0"], x0) and close(h2["x0"], x0) and close(h1["x1"], x1) and close(h2["x1"], x1)):
                        continue
                    y0 = min(h1["y"], h2["y"])
                    y1 = max(h1["y"], h2["y"])

                    rect_verticals = []
                    for v_idx, v in verticals:
                        if v_idx in used or v["box_key"] != h1["box_key"]:
                            continue
                        if close(v["x"], x0) and close(v["y0"], y0) and close(v["y1"], y1):
                            rect_verticals.append((v_idx, v))
                        elif close(v["x"], x1) and close(v["y0"], y0) and close(v["y1"], y1):
                            rect_verticals.append((v_idx, v))

                    if len(rect_verticals) < 2:
                        continue

                    left = next((item for item in rect_verticals if close(item[1]["x"], x0)), None)
                    right = next((item for item in rect_verticals if close(item[1]["x"], x1)), None)
                    if left is None or right is None:
                        continue

                    used.update({h_idx_1, h_idx_2, left[0], right[0]})
                    raw = items[h_idx_1]
                    enclosure = {
                        "src": "line_rect_enclosure",
                        "ax_index": ax_index,
                        "color": raw.get("color"),
                        h1["box_key"]: [
                            round(x0, 6),
                            round(y0, 6),
                            round(max(0.0, x1 - x0), 6),
                            round(max(0.0, y1 - y0), 6),
                        ],
                    }
                    fig_boxes = []
                    for idx2 in {h_idx_1, h_idx_2, left[0], right[0]}:
                        box = items[idx2].get("bbox")
                        if isinstance(box, (list, tuple)) and len(box) == 4:
                            fig_boxes.append(tuple(float(v) for v in box))
                    if fig_boxes:
                        fx0 = min(b[0] for b in fig_boxes)
                        fy0 = min(b[1] for b in fig_boxes)
                        fx1 = max(b[0] + b[2] for b in fig_boxes)
                        fy1 = max(b[1] + b[3] for b in fig_boxes)
                        enclosure["bbox"] = [
                            round(fx0, 6),
                            round(fy0, 6),
                            round(max(0.0, fx1 - fx0), 6),
                            round(max(0.0, fy1 - fy0), 6),
                        ]
                    new_enclosures.append(enclosure)
                    break

        if not used and not new_enclosures:
            return

        kept_connectors = self.features["2_connector"][:start_idx]
        for idx, item in enumerate(items):
            if idx in used:
                continue
            kept_connectors.append(item)
        self.features["2_connector"] = kept_connectors
        self.features["1_enclosure"].extend(new_enclosures)
