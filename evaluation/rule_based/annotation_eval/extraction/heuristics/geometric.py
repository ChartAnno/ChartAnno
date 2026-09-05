import matplotlib.patches as mpatches
from collections import Counter


class GeometricAnnotationMixin:
    def _select_pie_geometric_wedges(self, ax):
        wedges = [p for p in ax.patches if isinstance(p, mpatches.Wedge)]
        if len(wedges) < 2:
            return set()

        centers = []
        for w in wedges:
            try:
                cx, cy = w.center
                centers.append((w, float(cx), float(cy)))
            except Exception:
                continue
        if len(centers) < 2:
            return set()

        center_keys = [(round(cx, 4), round(cy, 4)) for _, cx, cy in centers]
        center_counts = Counter(center_keys)
        dominant_center = center_counts.most_common(1)[0][0]

        exploded = [
            w
            for (w, cx, cy), key in zip(centers, center_keys)
            if key != dominant_center
        ]
        if not exploded:
            return set()

        def right_edge_in_fig_norm(wedge):
            bbox = self._get_artist_bbox_fingerprint(wedge)
            if bbox is None:
                return float("-inf")
            x, _, w, _ = bbox
            return float(x) + float(w)

        rightmost = max(exploded, key=right_edge_in_fig_norm)
        return {rightmost}

    def _check_geometric_structures(self):
        if self.geometric_axes_ids is None:
            return
        if not self.geometric_axes_ids:
            return

        def positive_overlap(pos_a, pos_b, eps=1e-9):
            try:
                x_overlap = min(float(pos_a.x1), float(pos_b.x1)) - max(float(pos_a.x0), float(pos_b.x0))
                y_overlap = min(float(pos_a.y1), float(pos_b.y1)) - max(float(pos_a.y0), float(pos_b.y0))
            except Exception:
                return False
            return x_overlap > eps and y_overlap > eps

        axes_info = []
        for i, ax in enumerate(self.axes):
            try:
                if not getattr(ax, "axison", True):
                    continue
                pos = ax.get_position()
                bbox = self._get_artist_bbox_fingerprint(ax)
                if bbox is None:
                    continue
                area = float(pos.width) * float(pos.height)
                axes_info.append((i, ax, pos, bbox, area))
            except Exception:
                pass

        inset_added = set()
        for i, ax_i, b1, bbox_i, area_i in axes_info:
            if not self._is_ax_allowed_for_geometric(ax_i):
                continue
            for j, ax_j, b2, bbox_j, area_j in axes_info:
                if i == j:
                    continue
                if not positive_overlap(b1, b2):
                    continue
                if area_i >= area_j * 0.95:
                    continue
                if i in inset_added:
                    continue

                self.features["7_geometric"].append(
                    {
                        "src": "fig_axes",
                        "type": "Inset/Zoom",
                        "ax_index": i,
                        "overlap_with": j,
                        "bbox": bbox_i,
                        "overlap_bbox": bbox_j,
                    }
                )
                inset_added.add(i)
                break

    def _prune_geometric_axis_payload(self, ax_index, feature_start_idx):
        for key in ["1_enclosure", "3_text", "4_glyph", "5_color", "6_indicator"]:
            self.features[key] = self.features[key][:feature_start_idx[key]]

        kept_connectors = self.features["2_connector"][:feature_start_idx["2_connector"]]
        for item in self.features["2_connector"][feature_start_idx["2_connector"]:]:
            if not isinstance(item, dict):
                continue
            if item.get("ax_index") != ax_index:
                kept_connectors.append(item)
                continue
            if item.get("src") in {"bbox_connector", "patch_arrow"}:
                kept_connectors.append(item)
        self.features["2_connector"] = kept_connectors
