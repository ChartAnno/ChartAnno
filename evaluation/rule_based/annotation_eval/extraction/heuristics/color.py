from collections import Counter

import matplotlib.colors as mcolors
import numpy as np
import matplotlib.patches as mpatches


class ColorAnnotationMixin:
    def _color_key(self, color):
        rounded = self._round_color(color)
        if isinstance(rounded, tuple):
            return tuple(rounded)
        return str(rounded)

    def _select_sparse_color_candidates(self, candidates):
        if not candidates:
            return []

        color_counts = Counter(item["_color_key"] for item in candidates)
        dominant_count = max(color_counts.values())
        total = len(candidates)
        abs_cap = max(2, int(total * 0.15))
        rel_cap = max(1, int(np.ceil(dominant_count * 0.5)))

        selected = []
        for item in candidates:
            cnt = color_counts[item["_color_key"]]
            is_sparse = (cnt <= abs_cap) and (cnt <= rel_cap)
            if not is_sparse:
                continue
            selected.append({k: v for k, v in item.items() if k != "_color_key"})

        return selected

    def _merge_table_cell_candidates(self, candidates):
        non_table = []
        table_groups = {}

        for item in candidates:
            if item.get("src") != "table_cell":
                non_table.append(item)
                continue

            bbox = item.get("bbox")
            if not (
                isinstance(bbox, (list, tuple))
                and len(bbox) == 4
                and all(isinstance(v, (int, float)) for v in bbox)
            ):
                non_table.append(item)
                continue

            x, y, w, h = bbox
            row_key = (
                item.get("_color_key"),
                round(float(y), 4),
                round(float(h), 4),
            )
            table_groups.setdefault(row_key, []).append(item)

        merged = list(non_table)
        for items in table_groups.values():
            x0 = min(float(it["bbox"][0]) for it in items)
            y0 = min(float(it["bbox"][1]) for it in items)
            x1 = max(float(it["bbox"][0]) + float(it["bbox"][2]) for it in items)
            y1 = max(float(it["bbox"][1]) + float(it["bbox"][3]) for it in items)

            merged_item = dict(items[0])
            merged_item["bbox"] = [
                round(x0, 6),
                round(y0, 6),
                round(max(0.0, x1 - x0), 6),
                round(max(0.0, y1 - y0), 6),
            ]
            merged.append(merged_item)

        return merged

    def _merge_gradient_color_candidates(self, candidates):
        passthrough = []
        grouped = {}

        for item in candidates:
            bbox = item.get("bbox")
            if item.get("src") != "bar" or not (
                isinstance(bbox, (list, tuple)) and len(bbox) == 4
            ):
                passthrough.append(item)
                continue
            x, _, w, _ = bbox
            key = (
                item.get("ax_index"),
                round(float(x), 4),
                round(float(w), 4),
            )
            grouped.setdefault(key, []).append(item)

        merged = list(passthrough)
        for items in grouped.values():
            if len(items) < 8:
                merged.extend(items)
                continue

            items = sorted(items, key=lambda it: float(it["bbox"][1]))
            unique_colors = {it.get("_color_key") for it in items}
            heights = [float(it["bbox"][3]) for it in items if float(it["bbox"][3]) > 0]
            avg_h = float(np.mean(heights)) if heights else 0.0
            gaps = []
            for left, right in zip(items, items[1:]):
                left_top = float(left["bbox"][1]) + float(left["bbox"][3])
                right_y = float(right["bbox"][1])
                gaps.append(max(0.0, right_y - left_top))

            if len(unique_colors) < max(6, len(items) // 3) or (
                avg_h > 0.0 and gaps and max(gaps) > avg_h * 2.0
            ):
                merged.extend(items)
                continue

            rgba_list = []
            for item in items:
                try:
                    rgba_list.append(np.asarray(mcolors.to_rgba(item["color"]), dtype=float))
                except Exception:
                    pass
            if rgba_list:
                mean_rgba = tuple(round(float(v), 3) for v in np.mean(rgba_list, axis=0))
            else:
                mean_rgba = items[len(items) // 2]["color"]

            merged.append(
                {
                    "src": "gradient_band",
                    "type": "gradient_band",
                    "ax_index": items[0].get("ax_index"),
                    "bbox": self._bbox_union([it["bbox"] for it in items]),
                    "color": mean_rgba,
                    "_color_key": ("gradient_band",),
                }
            )

        return merged

    def _append_line_color_candidates(self, ax, line_infos, color_candidates, ax_index=None):
        if len(ax.lines) < 4:
            return

        grayscale_count = 0
        for line in ax.lines:
            try:
                rgba = mcolors.to_rgba(line.get_color())
            except Exception:
                continue
            if (max(rgba[:3]) - min(rgba[:3])) < 0.08:
                grayscale_count += 1

        if grayscale_count < len(ax.lines) - 1:
            return

        line_widths = []
        for line in ax.lines:
            try:
                line_widths.append(float(line.get_linewidth() or 0.0))
            except Exception:
                pass
        median_width = float(np.median(line_widths)) if line_widths else 0.0

        for idx, line in enumerate(ax.lines):
            info = line_infos[idx] if idx < len(line_infos) else {}
            bbox = info.get("bbox")
            if bbox is None:
                continue
            if not self._is_artist_allowed(line):
                continue
            color = line.get_color()
            if not self._is_chromatic_color(color):
                continue
            try:
                line_width = float(line.get_linewidth() or 0.0)
            except Exception:
                line_width = 0.0
            if line_width < max(1.8, median_width + 0.4):
                continue
            color_candidates.append(
                {
                    "src": "line_accent",
                    "type": "line_accent",
                    "ax_index": ax_index,
                    "bbox": bbox,
                    "color": self._round_color(color),
                    "_color_key": self._color_key(color),
                }
            )

    def _extract_color_stage(self, ax, ax_index, ctx):
        processed_patches = ctx["processed_patches"]
        line_infos = ctx["line_infos"]
        color_candidates = []
        for p in ax.patches:
            if not self._is_artist_allowed(p) or p in processed_patches:
                continue
            if isinstance(p, mpatches.Rectangle):
                bbox = self._get_artist_bbox_fingerprint(p)
                if bbox is None:
                    continue
                try:
                    rgba = mcolors.to_rgba(p.get_facecolor())
                except Exception:
                    continue
                if float(rgba[3]) <= 0.0:
                    continue
                color_candidates.append(
                    {
                        "src": "bar",
                        "ax_index": ax_index,
                        "bbox": bbox,
                        "color": self._round_color(rgba),
                        "_color_key": self._color_key(rgba),
                    }
                )
        for table in getattr(ax, "tables", []):
            try:
                cells = table.get_celld()
            except Exception:
                continue
            for cell in cells.values():
                if cell is None or not self._is_artist_allowed(cell):
                    continue
                try:
                    rgba = mcolors.to_rgba(cell.get_facecolor())
                except Exception:
                    continue
                if float(rgba[3]) <= 0.0:
                    continue
                bbox = self._get_artist_bbox_fingerprint(cell)
                if bbox is None:
                    continue
                color_candidates.append(
                    {
                        "src": "table_cell",
                        "ax_index": ax_index,
                        "bbox": bbox,
                        "color": self._round_color(rgba),
                        "_color_key": self._color_key(rgba),
                    }
                )
        self._append_line_color_candidates(ax, line_infos, color_candidates, ax_index=ax_index)
        merged_color_candidates = self._merge_gradient_color_candidates(
            self._merge_table_cell_candidates(color_candidates)
        )
        self.features["5_color"].extend(self._select_sparse_color_candidates(merged_color_candidates))
