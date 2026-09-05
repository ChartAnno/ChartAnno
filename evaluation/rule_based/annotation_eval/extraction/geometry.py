import numpy as np


def is_valid_norm_bbox(bbox):
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        return False
    try:
        vals = [float(v) for v in bbox]
    except Exception:
        return False
    return all(np.isfinite(v) for v in vals)


def norm_bbox_intersects_unit(bbox):
    if not is_valid_norm_bbox(bbox):
        return False
    x, y, w, h = [float(v) for v in bbox]
    x1 = x + max(0.0, w)
    y1 = y + max(0.0, h)
    return not (x1 < 0.0 or x > 1.0 or y1 < 0.0 or y > 1.0)


def norm_bbox_positive_visible_extent(bbox):
    if not is_valid_norm_bbox(bbox):
        return False
    x, y, w, h = [float(v) for v in bbox]
    x1 = x + max(0.0, w)
    y1 = y + max(0.0, h)
    ix0 = max(0.0, x)
    iy0 = max(0.0, y)
    ix1 = min(1.0, x1)
    iy1 = min(1.0, y1)
    return (ix1 - ix0) > 1e-9 and (iy1 - iy0) > 1e-9


def display_bounds_to_fig_norm(bounds, canvas_width_px, canvas_height_px):
    if bounds is None:
        return None
    try:
        x0, y0, w, h = [float(v) for v in bounds]
        width = float(canvas_width_px)
        height = float(canvas_height_px)
    except Exception:
        return None
    vals = [x0, y0, w, h, width, height]
    if not all(np.isfinite(v) for v in vals):
        return None
    if width <= 0.0 or height <= 0.0:
        return None
    return (
        round(x0 / width, 6),
        round(y0 / height, 6),
        round(w / width, 6),
        round(h / height, 6),
    )


def display_bounds_to_axes_norm(bounds, ax_bbox):
    if bounds is None or ax_bbox is None:
        return None
    try:
        x0, y0, w, h = [float(v) for v in bounds]
        ax_x0 = float(ax_bbox.x0)
        ax_y0 = float(ax_bbox.y0)
        ax_w = float(ax_bbox.width)
        ax_h = float(ax_bbox.height)
    except Exception:
        return None
    if ax_w <= 0.0 or ax_h <= 0.0:
        return None
    return (
        round((x0 - ax_x0) / ax_w, 6),
        round((y0 - ax_y0) / ax_h, 6),
        round(w / ax_w, 6),
        round(h / ax_h, 6),
    )


def clip_display_bounds_to_axes(bounds, ax_bbox, *, allow_zero_area=False):
    if bounds is None or ax_bbox is None:
        return bounds
    try:
        x0, y0, w, h = [float(v) for v in bounds]
        ax_x0 = float(ax_bbox.x0)
        ax_y0 = float(ax_bbox.y0)
        ax_x1 = float(ax_bbox.x1)
        ax_y1 = float(ax_bbox.y1)
    except Exception:
        return bounds

    x1 = x0 + w
    y1 = y0 + h
    ix0 = max(x0, ax_x0)
    iy0 = max(y0, ax_y0)
    ix1 = min(x1, ax_x1)
    iy1 = min(y1, ax_y1)
    if allow_zero_area:
        if ix1 < ix0 or iy1 < iy0:
            return None
    else:
        if ix1 <= ix0 or iy1 <= iy0:
            return None
    return ix0, iy0, ix1 - ix0, iy1 - iy0


def bbox_to_fig_norm(bbox, canvas_width_px, canvas_height_px):
    try:
        bounds = bbox.bounds
    except Exception:
        return None
    return display_bounds_to_fig_norm(bounds, canvas_width_px, canvas_height_px)


def bbox_to_axes_norm(bbox, ax_bbox):
    try:
        bounds = bbox.bounds
    except Exception:
        return None
    return display_bounds_to_axes_norm(bounds, ax_bbox)
