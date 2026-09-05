from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from annotation_eval.config import get_path


COLOR_FIELDS = (
    "color",
    "facecolor",
    "edgecolor",
    "bbox_fill_color",
    "bbox_edge_color",
    "markerfacecolor",
    "markeredgecolor",
)

HEX_COLOR_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")


def _safe_load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _normalize_number(x):
    if isinstance(x, float):
        if math.isclose(x, 0.0, abs_tol=1e-12):
            return 0.0
        return round(x, 6)
    return x


def _is_number_list(value, min_len=3):
    return (
        isinstance(value, (list, tuple))
        and len(value) >= min_len
        and all(isinstance(x, (int, float)) for x in value[:min_len])
    )


def _parse_hex_color(value: str):
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not HEX_COLOR_RE.match(s):
        return None
    s = s[1:]
    if len(s) in {3, 4}:
        s = "".join(ch * 2 for ch in s)
    if len(s) == 6:
        s = s + "ff"
    if len(s) != 8:
        return None
    r = int(s[0:2], 16) / 255.0
    g = int(s[2:4], 16) / 255.0
    b = int(s[4:6], 16) / 255.0
    a = int(s[6:8], 16) / 255.0
    return r, g, b, a


def _normalize_color_token(value):
    if _is_number_list(value):
        vals = tuple(float(x) for x in value)
        if len(vals) == 3:
            vals = vals + (1.0,)
        if max(vals[:3]) > 1.0 or vals[3] > 1.0:
            vals = tuple(min(255.0, max(0.0, x)) / 255.0 for x in vals)
        return ("rgba", tuple(_normalize_number(x) for x in vals[:4]))

    if isinstance(value, str):
        s = value.strip()
        rgba = _parse_hex_color(s)
        if rgba is not None:
            return ("rgba", tuple(_normalize_number(x) for x in rgba))
        return ("literal", s)

    return None


def _composite_rgba_over_white(rgba):
    r, g, b, a = rgba
    r = r * a + (1.0 - a)
    g = g * a + (1.0 - a)
    b = b * a + (1.0 - a)
    return r, g, b


def _srgb_channel_to_linear(x):
    if x <= 0.04045:
        return x / 12.92
    return ((x + 0.055) / 1.055) ** 2.4


def _srgb_to_lab(rgb):
    r, g, b = (_srgb_channel_to_linear(c) for c in rgb)
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    x /= 0.95047
    y /= 1.00000
    z /= 1.08883

    def f(t):
        delta = 6.0 / 29.0
        if t > delta**3:
            return t ** (1.0 / 3.0)
        return t / (3.0 * delta * delta) + 4.0 / 29.0

    fx = f(x)
    fy = f(y)
    fz = f(z)
    return 116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)


def _delta_e_ciede2000(lab1, lab2):
    l1, a1, b1 = lab1
    l2, a2, b2 = lab2

    c1 = math.hypot(a1, b1)
    c2 = math.hypot(a2, b2)
    c_bar = (c1 + c2) / 2.0
    c_bar7 = c_bar**7
    g = 0.5 * (1.0 - math.sqrt(c_bar7 / (c_bar7 + 25.0**7))) if c_bar > 0 else 0.0

    a1p = (1.0 + g) * a1
    a2p = (1.0 + g) * a2
    c1p = math.hypot(a1p, b1)
    c2p = math.hypot(a2p, b2)

    def hp(ap, bp):
        if math.isclose(ap, 0.0, abs_tol=1e-12) and math.isclose(bp, 0.0, abs_tol=1e-12):
            return 0.0
        angle = math.degrees(math.atan2(bp, ap))
        return angle + 360.0 if angle < 0 else angle

    h1p = hp(a1p, b1)
    h2p = hp(a2p, b2)
    delta_lp = l2 - l1
    delta_cp = c2p - c1p

    if math.isclose(c1p, 0.0, abs_tol=1e-12) or math.isclose(c2p, 0.0, abs_tol=1e-12):
        delta_hp = 0.0
    else:
        delta_hp = h2p - h1p
        if delta_hp > 180.0:
            delta_hp -= 360.0
        elif delta_hp < -180.0:
            delta_hp += 360.0
    delta_hp_term = 2.0 * math.sqrt(c1p * c2p) * math.sin(math.radians(delta_hp / 2.0))

    l_bar_p = (l1 + l2) / 2.0
    c_bar_p = (c1p + c2p) / 2.0

    if math.isclose(c1p, 0.0, abs_tol=1e-12) or math.isclose(c2p, 0.0, abs_tol=1e-12):
        h_bar_p = h1p + h2p
    else:
        h_sum = h1p + h2p
        if abs(h1p - h2p) > 180.0:
            h_bar_p = (h_sum + 360.0) / 2.0 if h_sum < 360.0 else (h_sum - 360.0) / 2.0
        else:
            h_bar_p = h_sum / 2.0

    t = (
        1.0
        - 0.17 * math.cos(math.radians(h_bar_p - 30.0))
        + 0.24 * math.cos(math.radians(2.0 * h_bar_p))
        + 0.32 * math.cos(math.radians(3.0 * h_bar_p + 6.0))
        - 0.20 * math.cos(math.radians(4.0 * h_bar_p - 63.0))
    )
    delta_theta = 30.0 * math.exp(-(((h_bar_p - 275.0) / 25.0) ** 2))
    c_bar_p7 = c_bar_p**7
    r_c = 2.0 * math.sqrt(c_bar_p7 / (c_bar_p7 + 25.0**7)) if c_bar_p > 0 else 0.0
    s_l = 1.0 + (0.015 * ((l_bar_p - 50.0) ** 2)) / math.sqrt(20.0 + ((l_bar_p - 50.0) ** 2))
    s_c = 1.0 + 0.045 * c_bar_p
    s_h = 1.0 + 0.015 * c_bar_p * t
    r_t = -math.sin(math.radians(2.0 * delta_theta)) * r_c

    l_term = delta_lp / s_l
    c_term = delta_cp / s_c
    h_term = delta_hp_term / s_h
    return math.sqrt(l_term * l_term + c_term * c_term + h_term * h_term + r_t * c_term * h_term)


def _color_similarity(color_a, color_b):
    kind_a, value_a = color_a
    kind_b, value_b = color_b
    if kind_a == "rgba" and kind_b == "rgba":
        rgb_a = _composite_rgba_over_white(value_a)
        rgb_b = _composite_rgba_over_white(value_b)
        delta_e = _delta_e_ciede2000(_srgb_to_lab(rgb_a), _srgb_to_lab(rgb_b))
        return max(0.0, 1.0 - (delta_e / 100.0))
    return 1.0 if color_a == color_b else 0.0


def _iter_gt_files(gt_root: Path):
    if not gt_root.exists():
        return
    for path in sorted(gt_root.rglob("*.json")):
        if path.name.startswith("."):
            continue
        if path.parent.name.upper() != "GT" and path.parent.parent == gt_root:
            yield path.parent.name, path.stem, path
        elif path.parent.name.upper() == "GT" and len(path.parents) >= 3:
            yield path.parents[2].name, path.stem, path
        elif len(path.parts) >= 3:
            yield path.parts[-2], path.stem, path


def _stage_from_name(name: str):
    lowered = name.lower()
    if "task" in lowered:
        return "intent"
    for stage in ("intent", "operation", "implementation"):
        if stage in lowered:
            return stage
    return None


def _iter_pred_files(model_root: Path, model: str):
    if not model_root.exists():
        return
    for path in sorted(model_root.rglob("*.json")):
        if path.name.startswith("."):
            continue
        stage = _stage_from_name(path.name)
        if stage is None:
            continue
        parts = path.relative_to(model_root).parts
        category = ""
        chart_id = ""
        if len(parts) >= 4 and parts[2].upper() == model:
            category = parts[0]
            chart_id = parts[1]
        elif len(parts) >= 2:
            category = parts[0]
            m = re.match(r"^([A-Za-z]+_\d+)", path.stem)
            chart_id = m.group(1) if m else path.stem
        if not category or not chart_id:
            continue
        if model not in path.as_posix().upper() and path.parent.name.upper() != model:
            # Keep this permissive; the filename pattern is the main source of truth.
            pass
        yield category, chart_id, stage, path


def _write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _coarse_group_key(feature):
    return feature


def _pick_primary_color(item):
    for color_field in COLOR_FIELDS:
        if color_field not in item:
            continue
        token = _normalize_color_token(item.get(color_field))
        if token is not None:
            return token, color_field
    return None, None


def _extract_grouped_colors(data):
    groups = defaultdict(list)
    for feature, items in data.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            token, color_field = _pick_primary_color(item)
            if token is None:
                continue
            groups[_coarse_group_key(feature)].append(
                {
                    "feature": feature,
                    "color_field": color_field,
                    "color": token,
                }
            )
    return groups


def _hungarian_max_color_match(left_entries, right_entries):
    n = len(left_entries)
    m = len(right_entries)
    if n == 0 or m == 0:
        return []

    swapped = False
    if n > m:
        left_entries, right_entries = right_entries, left_entries
        n, m = m, n
        swapped = True

    pair_priority = []
    for i in range(n):
        row = []
        for j in range(m):
            row.append(_color_similarity(left_entries[i]["color"], right_entries[j]["color"]))
        pair_priority.append(row)

    max_score = max(max(row) for row in pair_priority) if pair_priority else 0.0
    cost = [[max_score - pair_priority[i][j] for j in range(m)] for i in range(n)]
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [-1] * n
    for j in range(1, m + 1):
        if p[j] != 0 and p[j] - 1 < n:
            assignment[p[j] - 1] = j - 1

    pairs = []
    for i, j in enumerate(assignment):
        if j >= 0:
            pairs.append((j, i) if swapped else (i, j))
    return pairs


def _score_one_file(gt_data, pred_data):
    gt_groups = _extract_grouped_colors(gt_data)
    pred_groups = _extract_grouped_colors(pred_data)
    all_keys = sorted(set(gt_groups) | set(pred_groups))

    by_group = {}
    total_gt = 0
    total_pred = 0
    total_similarity = 0.0

    for key in all_keys:
        gt_entries = gt_groups.get(key, [])
        pred_entries = pred_groups.get(key, [])
        pairs = _hungarian_max_color_match(gt_entries, pred_entries)
        similarity = 0.0
        for i, j in pairs:
            similarity += _color_similarity(gt_entries[i]["color"], pred_entries[j]["color"])

        gt_count = len(gt_entries)
        pred_count = len(pred_entries)
        precision = similarity / pred_count if pred_count else (1.0 if gt_count == 0 else 0.0)
        recall = similarity / gt_count if gt_count else (1.0 if pred_count == 0 else 0.0)
        f1 = 0.0 if (precision + recall) == 0 else (2.0 * precision * recall / (precision + recall))

        by_group[key] = {
            "gt_count": gt_count,
            "pred_count": pred_count,
            "matched_pairs": len(pairs),
            "max_similarity": round(similarity, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
        total_gt += gt_count
        total_pred += pred_count
        total_similarity += similarity

    precision = total_similarity / total_pred if total_pred else (1.0 if total_gt == 0 else 0.0)
    recall = total_similarity / total_gt if total_gt else (1.0 if total_pred == 0 else 0.0)
    f1 = 0.0 if (precision + recall) == 0 else (2.0 * precision * recall / (precision + recall))

    return {
        "overall": {
            "gt_count": total_gt,
            "pred_count": total_pred,
            "max_similarity": round(total_similarity, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        },
        "by_group": by_group,
    }


def evaluate_all(gt_root: Path, llm_root: Path, vlm_root: Path, per_chart_dir: Path):
    gt_index = {(category, chart_id): path for category, chart_id, path in _iter_gt_files(gt_root)}
    results = []
    per_file_rows = []
    summary = defaultdict(lambda: {"count": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0})

    for model, pred_root in (("LLM", llm_root), ("VLM", vlm_root)):
        for category, chart_id, stage, pred_path in _iter_pred_files(pred_root, model):
            gt_path = gt_index.get((category, chart_id))
            if gt_path is None:
                continue

            gt_data = _safe_load_json(gt_path)
            pred_data = _safe_load_json(pred_path)
            if not isinstance(gt_data, dict) or not isinstance(pred_data, dict):
                continue

            scored = _score_one_file(gt_data, pred_data)
            result = {
                "model": model,
                "category": category,
                "chart_id": chart_id,
                "stage": stage,
                "gt_file": str(gt_path),
                "pred_file": str(pred_path),
                **scored,
            }
            results.append(result)

            summary[(model, stage)]["count"] += 1
            summary[(model, stage)]["precision"] += result["overall"]["precision"]
            summary[(model, stage)]["recall"] += result["overall"]["recall"]
            summary[(model, stage)]["f1"] += result["overall"]["f1"]

            per_file_rows.append(
                {
                    "model": model,
                    "category": category,
                    "chart_id": chart_id,
                    "stage": stage,
                    "gt_count": result["overall"]["gt_count"],
                    "pred_count": result["overall"]["pred_count"],
                    "max_similarity": result["overall"]["max_similarity"],
                    "precision": result["overall"]["precision"],
                    "recall": result["overall"]["recall"],
                    "f1": result["overall"]["f1"],
                }
            )

            out_path = per_chart_dir / category / chart_id / model / f"{chart_id}_{stage}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_rows = []
    for (model, stage), stats in sorted(summary.items()):
        count = stats["count"]
        summary_rows.append(
            {
                "model": model,
                "stage": stage,
                "count": count,
                "mean_precision": round(stats["precision"] / count, 6) if count else 0.0,
                "mean_recall": round(stats["recall"] / count, 6) if count else 0.0,
                "mean_f1": round(stats["f1"] / count, 6) if count else 0.0,
            }
        )

    return {
        "meta": {
            "metric": "color_matching",
            "grouping": "top-level annotation feature",
            "matching": "maximum bipartite matching on CIEDE2000 color similarity within each feature group",
            "color_unit": "one primary visible color per annotation element",
            "primary_color_rule": "first available field in COLOR_FIELDS order",
            "similarity_formula": "max(0, 1 - deltaE00/100) for numeric/hex colors; exact match for non-hex strings",
        },
        "files": results,
        "summary": summary_rows,
        "per_file_rows": per_file_rows,
    }


def build_cli():
    parser = argparse.ArgumentParser(description="Compute annotation color matching score.")
    parser.add_argument("--project-root")
    parser.add_argument("--gt-root")
    parser.add_argument("--llm-root")
    parser.add_argument("--vlm-root")
    parser.add_argument("--output-json")
    parser.add_argument("--summary-csv")
    parser.add_argument("--file-csv")
    parser.add_argument("--per-chart-dir")
    return parser


def _fill_default_paths(args) -> None:
    dataset_root = Path(args.project_root).resolve() if args.project_root else REPO_ROOT
    annotation_root = get_path(dataset_root, "annotations_dir", "outputs/annotations")
    analysis_root = get_path(dataset_root, "analysis_dir", "outputs/analysis")
    if args.gt_root is None:
        args.gt_root = str(annotation_root / "final_gt_structured")
    if args.llm_root is None:
        args.llm_root = str(annotation_root / "final_test_llm_minus_removed_structured")
    if args.vlm_root is None:
        args.vlm_root = str(annotation_root / "final_test_vlm_minus_removed_structured")
    if args.output_json is None:
        args.output_json = str(analysis_root / "color_matching" / "color_matching_results.json")
    if args.summary_csv is None:
        args.summary_csv = str(analysis_root / "color_matching" / "color_matching_summary.csv")
    if args.file_csv is None:
        args.file_csv = str(analysis_root / "color_matching" / "color_matching_per_file.csv")
    if args.per_chart_dir is None:
        args.per_chart_dir = str(analysis_root / "color_matching" / "per_chart")


def _validate_required_args(args) -> None:
    missing = []
    for field in ("gt_root", "llm_root", "vlm_root", "output_json", "summary_csv", "file_csv", "per_chart_dir"):
        if getattr(args, field) is None:
            missing.append("--" + field.replace("_", "-"))
    if missing:
        raise SystemExit("Missing required arguments: " + ", ".join(missing))


def main():
    args = build_cli().parse_args()
    _fill_default_paths(args)
    _validate_required_args(args)
    results = evaluate_all(
        Path(args.gt_root),
        Path(args.llm_root),
        Path(args.vlm_root),
        Path(args.per_chart_dir),
    )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved: {output_json}")


if __name__ == "__main__":
    main()
