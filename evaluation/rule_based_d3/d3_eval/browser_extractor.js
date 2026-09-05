(() => {
  "use strict";

  const round = (value, digits = 6) => {
    const scale = 10 ** digits;
    return Math.round(Number(value) * scale) / scale;
  };

  const roundBox = (box) => box.map((value) => round(value));

  const isAxesNode = (node) => {
    if (!node) return false;
    if (node.getAttribute("data-ir-kind") === "axes") return true;
    return node.tagName && node.tagName.toLowerCase() === "g" && /^axes_\d+$/.test(node.id || "");
  };

  const MPL_PRIMITIVE_GROUP_SELECTOR = [
    'g[id^="line2d_"]',
    'g[id^="PolyCollection_"]',
    'g[id^="PathCollection_"]',
    'g[id^="LineCollection_"]',
    'g[id^="patch_"]',
  ].join(', ');

  const mplPrimitiveGroupKind = (node) => {
    if (!node || !node.tagName || node.tagName.toLowerCase() !== "g") return null;
    const id = node.id || "";
    if (/^line2d_\d+$/.test(id)) return "line";
    if (/^(PolyCollection|PathCollection|LineCollection)_\d+$/.test(id)) return "collection";
    if (/^patch_\d+$/.test(id)) return "patch";
    return null;
  };

  const parseColor = (value) => {
    if (!value || value === "none" || value === "transparent") return null;
    const match = String(value).match(/^rgba?\(([^)]+)\)$/i);
    if (!match) return null;
    const parts = match[1].split(/[\s,\/]+/).filter(Boolean).map(Number);
    if (parts.length < 3 || parts.slice(0, 3).some((v) => !Number.isFinite(v))) return null;
    const alpha = Number.isFinite(parts[3]) ? parts[3] : 1;
    return [round(parts[0] / 255), round(parts[1] / 255), round(parts[2] / 255), round(alpha)];
  };

  const colorVisible = (color) => Array.isArray(color) && color.length >= 4 && color[3] > 0;

  const parseBoxAttribute = (node, svg) => {
    const raw = node.getAttribute("data-ir-bbox");
    if (!raw) return null;
    const values = raw.split(/[\s,]+/).map(Number);
    if (values.length !== 4 || values.some((v) => !Number.isFinite(v))) return null;
    const viewBox = svg.viewBox && svg.viewBox.baseVal;
    const width = viewBox && viewBox.width ? viewBox.width : svg.getBoundingClientRect().width;
    const height = viewBox && viewBox.height ? viewBox.height : svg.getBoundingClientRect().height;
    if (!(width > 0 && height > 0)) return null;
    return roundBox([
      values[0] / width,
      1 - ((values[1] + values[3]) / height),
      values[2] / width,
      values[3] / height,
    ]);
  };

  const rectToNormalizedBox = (rect, rootRect) => {
    if (!(rootRect.width > 0 && rootRect.height > 0)) return null;
    return roundBox([
      (rect.left - rootRect.left) / rootRect.width,
      1 - ((rect.bottom - rootRect.top) / rootRect.height),
      rect.width / rootRect.width,
      rect.height / rootRect.height,
    ]);
  };

  const elementBox = (node, svg, rootRect) => {
    if (isAxesNode(node)) {
      const explicit = parseBoxAttribute(node, svg);
      if (explicit) return explicit;
      // Matplotlib writes the plotting rectangle as the first direct patch
      // inside each g#axes_N group. Use it rather than the full group extent,
      // which also includes tick labels and annotations.
      const mplAxesPatch = node.querySelector(':scope > g[id^="patch_"] > path, :scope > g[id^="patch_"] > rect');
      if (mplAxesPatch) {
        try {
          return rectToNormalizedBox(mplAxesPatch.getBoundingClientRect(), rootRect);
        } catch (_) {
          // Fall through to the group bounds.
        }
      }
    }
    let rect;
    try {
      rect = node.getBoundingClientRect();
    } catch (_) {
      return null;
    }
    return rectToNormalizedBox(rect, rootRect);
  };

  const localBox = (box, axesBox) => {
    if (!box || !axesBox || !(axesBox[2] > 0 && axesBox[3] > 0)) return null;
    return roundBox([
      (box[0] - axesBox[0]) / axesBox[2],
      (box[1] - axesBox[1]) / axesBox[3],
      box[2] / axesBox[2],
      box[3] / axesBox[3],
    ]);
  };

  const inferTextRole = (node) => {
    const explicit = node.getAttribute("data-text-role");
    if (explicit) return explicit;
    if (node.closest(".x-axis")) return "x_tick_label";
    if (node.closest(".y-axis")) return "y_tick_label";
    if (node.closest('g[id^="xtick_"]')) return "x_tick_label";
    if (node.closest('g[id^="ytick_"]')) return "y_tick_label";
    if (node.closest('g[id^="legend_"]')) return "legend_text";
    if (node.closest(".legend")) return node.classList.contains("legend-title") ? "legend_title" : "legend_text";
    if (node.classList.contains("chart-title") || node.closest(".chart-title")) return "title";
    if (node.classList.contains("x-axis-label")) return "x_axis_label";
    if (node.classList.contains("y-axis-label")) return "y_axis_label";
    return "text";
  };

  const inferKind = (node) => {
    const explicit = node.getAttribute("data-ir-kind");
    if (explicit) return explicit;
    const tag = node.tagName.toLowerCase();
    if (isAxesNode(node)) return "axes";
    const mplGroupKind = mplPrimitiveGroupKind(node);
    if (mplGroupKind) return mplGroupKind;
    if (tag === "text" || tag === "tspan") return "text";
    if (node.closest('g[id^="PolyCollection_"], g[id^="PathCollection_"], g[id^="LineCollection_"]')) return "collection";
    if (node.closest('g[id^="line2d_"]')) return tag === "use" ? "collection" : "line";
    if (tag === "line" || tag === "polyline") return "line";
    if (tag === "circle" || tag === "ellipse") return "collection";
    if (tag === "path" && (node.classList.contains("line") || node.classList.contains("domain"))) return "line";
    return "patch";
  };

  const inferType = (node, kind) => {
    const explicit = node.getAttribute("data-ir-type");
    if (explicit) return explicit;
    const tag = node.tagName.toLowerCase();
    if (/^PolyCollection_\d+$/.test(node.id || "")) return "PolyCollection";
    if (/^PathCollection_\d+$/.test(node.id || "")) return "PathCollection";
    if (/^LineCollection_\d+$/.test(node.id || "")) return "LineCollection";
    if (/^line2d_\d+$/.test(node.id || "")) return "SVGPathElement";
    if (/^patch_\d+$/.test(node.id || "")) return "Path";
    if (node.closest('g[id^="PolyCollection_"]')) return "PolyCollection";
    if (node.closest('g[id^="PathCollection_"]')) return "PathCollection";
    if (node.closest('g[id^="LineCollection_"]')) return "LineCollection";
    if (kind === "collection" && tag === "use") return "PathCollection";
    if (kind === "collection" && (tag === "circle" || tag === "ellipse")) return "PathCollection";
    if (kind === "collection") return "PolyCollection";
    if (kind === "line") return tag === "path" ? "SVGPathElement" : "SVGLineElement";
    if (tag === "rect") return "Rectangle";
    if (tag === "circle") return "Circle";
    if (tag === "ellipse") return "Ellipse";
    if (tag === "polygon") return "Polygon";
    if (tag === "path") return "Path";
    if (kind === "axes") return "SVGGElement";
    return node.constructor ? node.constructor.name : tag;
  };

  const inferSemanticRole = (node) => {
    const explicit = node.getAttribute("data-ir-role");
    if (explicit) return explicit;
    if (node.closest('g[id^="xtick_"], g[id^="ytick_"]')) return "axis-decoration";
    if (node.closest(".legend") && node.tagName.toLowerCase() !== "text") return "legend-key";
    if (node.closest('g[id^="legend_"]') && node.tagName.toLowerCase() !== "text") return "legend-key";
    if (node.closest('g[id^="patch_"]') && node.closest('g[id^="text_"]')) return "enclosure";
    const tokens = Array.from(node.classList || []);
    const joined = tokens.join(" ").toLowerCase();
    if (/annotation[-_ ]?(box|enclosure)|highlight[-_ ]?(box|region)/.test(joined)) return "enclosure";
    if (/annotation[-_ ]?arrow|connector|leader/.test(joined)) return "connector";
    if (/reference[-_ ]?line|indicator|bracket/.test(joined)) return "indicator";
    if (/annotation[-_ ]?(point|glyph)|callout[-_ ]?point/.test(joined)) return "glyph";
    if (/inset|zoom/.test(joined)) return "geometric";
    if (/annotation[-_ ]?text|value[-_ ]?label|callout[-_ ]?label/.test(joined)) return "text";
    return null;
  };

  const pathSignature = (node, limit = 32) => {
    if (typeof node.getTotalLength !== "function" || typeof node.getPointAtLength !== "function") return null;
    let total;
    try {
      total = node.getTotalLength();
    } catch (_) {
      return null;
    }
    if (!Number.isFinite(total) || total <= 0) return null;
    const points = [];
    const count = Math.max(2, Math.min(limit, Number(node.getAttribute("data-count")) || limit));
    for (let index = 0; index < count; index += 1) {
      const point = node.getPointAtLength((total * index) / Math.max(1, count - 1));
      points.push([round(point.x), round(point.y)]);
    }
    return points;
  };

  // Matplotlib line2d paths are emitted as M/L vertex sequences.  Normalizing
  // those vertices to the path's own extent makes the signature invariant to
  // autoscale changes between the candidate and removed renderings.
  const normalizedPathVertexSignature = (node) => {
    const path = node && node.tagName && node.tagName.toLowerCase() === "path"
      ? node
      : (node && node.querySelector ? node.querySelector(":scope > path") : null);
    if (!path) return null;
    const d = String(path.getAttribute("d") || "");
    const chunks = d.match(/[ML][^MLHVCSQTAZ]*/gi) || [];
    const points = [];
    for (const chunk of chunks) {
      const values = chunk.slice(1).match(/[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?/g) || [];
      for (let index = 0; index + 1 < values.length; index += 2) {
        const x = Number(values[index]);
        const y = Number(values[index + 1]);
        if (Number.isFinite(x) && Number.isFinite(y)) points.push([x, y]);
      }
    }
    if (points.length < 2) return null;
    const xs = points.map((point) => point[0]);
    const ys = points.map((point) => point[1]);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const width = maxX - minX;
    const height = maxY - minY;
    return points.map(([x, y]) => [
      round(width > 1e-9 ? (x - minX) / width : 0, 4),
      round(height > 1e-9 ? (y - minY) / height : 0, 4),
    ]);
  };

  const pointSignature = (node) => {
    const tag = node.tagName.toLowerCase();
    if (tag === "g") {
      const path = node.querySelector("path:not(defs path)");
      if (path) return pathSignature(path);
      const uses = Array.from(node.querySelectorAll("use"));
      const points = uses.map((use) => [Number(use.getAttribute("x")), Number(use.getAttribute("y"))])
        .filter((point) => point.every(Number.isFinite))
        .map((point) => point.map((value) => round(value)));
      return points.length ? points : null;
    }
    if (tag === "circle" || tag === "ellipse") {
      const x = Number(node.getAttribute("cx"));
      const y = Number(node.getAttribute("cy"));
      return Number.isFinite(x) && Number.isFinite(y) ? [[round(x), round(y)]] : null;
    }
    if (tag === "use") {
      const x = Number(node.getAttribute("x"));
      const y = Number(node.getAttribute("y"));
      return Number.isFinite(x) && Number.isFinite(y) ? [[round(x), round(y)]] : null;
    }
    return pathSignature(node);
  };

  const domPath = (node, svg) => {
    const parts = [];
    let current = node;
    while (current && current !== svg) {
      const tag = current.tagName.toLowerCase();
      const parent = current.parentElement;
      if (!parent) break;
      const siblings = Array.from(parent.children).filter((child) => child.tagName === current.tagName);
      parts.unshift(`${tag}:${siblings.indexOf(current)}`);
      current = parent;
    }
    return parts.join("/");
  };

  window.__extractD3IR = () => {
    const svg = document.querySelector("svg[data-chart-root], svg");
    if (!svg) throw new Error("No SVG chart root found");
    const rootRect = svg.getBoundingClientRect();
    if (!(rootRect.width > 0 && rootRect.height > 0)) throw new Error("SVG chart root has zero size");

    const selector = [
      '[data-ir-kind="axes"]',
      'g[id^="axes_"]',
      MPL_PRIMITIVE_GROUP_SELECTOR,
      'text', 'line', 'polyline', 'polygon', 'rect', 'circle', 'ellipse', 'path', 'use', 'image'
    ].join(', ');
    const nodes = Array.from(svg.querySelectorAll(selector)).filter((node) => {
      if (node.closest("defs, clipPath, marker, mask, pattern")) return false;
      const mplOwner = node.closest(MPL_PRIMITIVE_GROUP_SELECTOR);
      if (mplOwner && mplOwner !== node) return false;
      const style = getComputedStyle(node);
      if (style.display === "none" || style.visibility === "hidden" || Number(style.opacity) === 0) return false;
      if (node.tagName.toLowerCase() === "text" && !String(node.textContent || "").trim()) return false;
      return true;
    });

    const ids = new Map(nodes.map((node, index) => [node, index + 1]));
    const axesNodes = nodes.filter(isAxesNode);
    const axesIndex = new Map(axesNodes.map((node, index) => [node, index]));
    const axesBoxes = new Map(axesNodes.map((node) => [node, elementBox(node, svg, rootRect)]));

    const records = [];
    for (const node of nodes) {
      const kind = inferKind(node);
      const bbox = elementBox(node, svg, rootRect);
      if (!bbox) continue;
      const axes = kind === "axes" ? node : node.closest('[data-ir-kind="axes"], g[id^="axes_"]');
      const axesBox = axes ? axesBoxes.get(axes) : null;
      const styleNode = node.tagName.toLowerCase() === "g"
        ? (node.querySelector("path, rect, line, polyline, polygon, circle, ellipse, use") || node)
        : node;
      const style = getComputedStyle(styleNode);
      const fill = parseColor(style.fill);
      const stroke = parseColor(style.stroke);
      const opacity = Number.isFinite(Number(style.opacity)) ? Number(style.opacity) : 1;
      const semanticRole = inferSemanticRole(node);
      const type = inferType(node, kind);
      const tag = node.tagName.toLowerCase();

      const record = {
        kind,
        artist_id: ids.get(node),
        element_id: `d3-${String(ids.get(node)).padStart(5, "0")}`,
        dom_path: domPath(node, svg),
        tag,
        type,
        bbox,
      };

      if (axes) {
        record.ax_index = axesIndex.get(axes);
        record.ax_artist_id = ids.get(axes);
        record.container_axes_bbox = axesBox;
        if (kind !== "axes") record.axes_bbox = localBox(bbox, axesBox);
      }
      if (semanticRole) record.semantic_role = semanticRole;
      if (node.getAttribute("class")) record.class_name = node.getAttribute("class");
      const nativeGroup = node.closest("g[id]");
      if (nativeGroup && nativeGroup.id) record.native_group_id = nativeGroup.id;

      const primary = kind === "text" ? fill : (colorVisible(stroke) ? stroke : fill);
      if (primary) record.color = primary;
      if (fill) record.facecolor = fill;
      if (stroke) record.edgecolor = stroke;
      record.fill = colorVisible(fill);
      record.alpha = round(opacity);
      const lineWidth = Number.parseFloat(style.strokeWidth);
      if (Number.isFinite(lineWidth)) record.linewidth = round(lineWidth);

      if (kind === "text") {
        record.content = String(node.textContent || "").trim();
        record.text_role = inferTextRole(node);
        if (node.closest(".annotation, [data-annotation]")) record.semantic_role = record.semantic_role || "text";
      }

      const signature = pointSignature(node);
      if (kind === "line") {
        const mplVertexSignature = mplPrimitiveGroupKind(node) === "line"
          ? normalizedPathVertexSignature(node)
          : null;
        const dash = style.strokeDasharray && style.strokeDasharray !== "none" ? style.strokeDasharray : null;
        record.linestyle = dash ? "--" : "-";
        record.marker = node.getAttribute("data-marker") || (node.querySelector && node.querySelector("use") ? "o" : "None");
        record.n_points = Number(node.getAttribute("data-count"))
          || (mplVertexSignature ? mplVertexSignature.length : (signature ? signature.length : 2));
        if (signature) {
          record.point_signature = signature;
        }
        if (mplVertexSignature) record.data_signature = mplVertexSignature;
        else if (signature) record.data_signature = signature;
        const label = node.getAttribute("data-label");
        if (label) record.label = label;
        record.clip_on = Boolean(node.closest("[clip-path]"));
        record.is_full_span = node.getAttribute("data-full-span") === "true";
      } else if (kind === "collection") {
        const mplGroup = mplPrimitiveGroupKind(node) === "collection";
        record.item_count = Number(node.getAttribute("data-count"))
          || (mplGroup ? node.querySelectorAll("use, path").length : 1);
        if (type === "PathCollection" && signature && !mplGroup) {
          record.offset_signature = signature;
          record.point_boxes = [{bbox, axes_bbox: record.axes_bbox || null}];
        } else if (signature && !mplGroup) {
          record.path_signature = [signature];
        }
      } else if (kind === "annotation_arrow") {
        record.arrow_style = node.getAttribute("marker-end") || node.getAttribute("data-arrow-style") || "arrow";
        const textContent = node.getAttribute("data-text-content");
        if (textContent) record.text_content = textContent;
      } else if (kind === "annotation_bbox") {
        const textContent = node.getAttribute("data-text-content");
        if (textContent) record.text_content = textContent;
      }

      records.push(record);
    }

    return {
      backend: "d3-svg",
      canvas: {width: round(rootRect.width), height: round(rootRect.height)},
      records,
    };
  };
})();
