import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from annotation_eval.extraction.annotation_schema import FEATURE_ORDER

EVALUATED_STAGES = {"intent", "operation", "implementation"}

NUMERIC_TOL = 1e-3
BBOX_MATCH_MODE = "exist_only"  # "exist_only" | "tolerant"
BBOX_CENTER_TOL = 0.02
BBOX_SIZE_TOL = 0.05

DEFAULT_ANNOTATION_ROOT = REPO_ROOT / "outputs" / "annotations"
DEFAULT_ANALYSIS_ROOT = REPO_ROOT / "outputs" / "analysis" / "annotation_matching"


def _normalize_number(x):
    if isinstance(x, float):
        if math.isclose(x, 0.0, abs_tol=1e-12):
            return 0.0
        return round(x, 6)
    return x


def _to_float(v):
    try:
        return float(v)
    except Exception:
        return None


def _parse_bbox(item_or_bbox):
    if isinstance(item_or_bbox, dict):
        bbox = item_or_bbox.get("bbox")
    else:
        bbox = item_or_bbox

    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        return None

    vals = [_to_float(v) for v in bbox]
    if any(v is None for v in vals):
        return None

    x, y, w, h = vals
    return x, y, max(0.0, w), max(0.0, h)


def _parse_bbox_field(item, field_name):
    if not isinstance(item, dict):
        return None
    return _parse_bbox(item.get(field_name))


def _bbox_close(a, b):
    ab = _parse_bbox_field(a, "axes_bbox") or _parse_bbox(a)
    bb = _parse_bbox_field(b, "axes_bbox") or _parse_bbox(b)
    if ab is None or bb is None:
        return False

    ax, ay, aw, ah = ab
    bx, by, bw, bh = bb
    acx, acy = ax + aw / 2.0, ay + ah / 2.0
    bcx, bcy = bx + bw / 2.0, by + bh / 2.0

    a_idx = a.get("ax_index") if isinstance(a, dict) else None
    b_idx = b.get("ax_index") if isinstance(b, dict) else None
    if a_idx is not None and b_idx is not None and a_idx != b_idx:
        return False

    center_tol = BBOX_CENTER_TOL
    size_tol = BBOX_SIZE_TOL

    return (
        abs(acx - bcx) <= center_tol
        and abs(acy - bcy) <= center_tol
        and abs(aw - bw) <= size_tol
        and abs(ah - bh) <= size_tol
    )


def _bbox_exist(item_or_bbox):
    return _parse_bbox(item_or_bbox) is not None


def _bbox_match(a, b):
    if BBOX_MATCH_MODE == "exist_only":
        return _bbox_exist(a) and _bbox_exist(b)
    return _bbox_close(a, b)


def _normalize_color(color):
    if isinstance(color, list):
        return tuple(_normalize_number(float(c)) if isinstance(c, (int, float)) else c for c in color)
    if isinstance(color, tuple):
        return tuple(_normalize_number(float(c)) if isinstance(c, (int, float)) else c for c in color)
    return color


def _value_match(a, b, key_hint=None):
    if key_hint in {"bbox", "overlap_bbox"}:
        return _bbox_match(a, b)

    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= NUMERIC_TOL

    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        for k in a.keys():
            if k == "color":
                if _normalize_color(a.get(k)) != _normalize_color(b.get(k)):
                    return False
                continue
            if not _value_match(a.get(k), b.get(k), key_hint=k):
                return False
        return True

    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_value_match(x, y) for x, y in zip(a, b))

    return a == b


def _text_content_key(item):
    if not isinstance(item, dict):
        return None
    content = item.get("content")
    if not isinstance(content, str):
        return None
    # Weak text normalization before content matching.
    text = content.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"[^\w\s%$]+", " ", text.lower())
    return " ".join(text.split())


def _contains_text(gt, pred_blob):
    if not gt:
        return False

    # Avoid false positives for very short text, e.g., "high" matching "higher".
    if len(gt) <= 4:
        return re.search(rf"\b{re.escape(gt)}\b", pred_blob) is not None

    return gt in pred_blob


def _text_match_stats(gt_items, pred_items):
    gt_keys = [_text_content_key(item) for item in gt_items]
    pred_keys = [_text_content_key(item) for item in pred_items]
    gt_keys = [key for key in gt_keys if key]
    pred_keys = [key for key in pred_keys if key]

    pred_blob = " ".join(pred_keys)

    intersection = sum(1 for k in gt_keys if _contains_text(k, pred_blob))

    gt_count = len(gt_keys)
    pred_count = len(pred_keys)

    # Text content coverage with count penalty.
    # This is not strict Jaccard.
    union = max(gt_count, pred_count)

    return gt_count, pred_count, intersection, union

def _item_match(category, a, b):
    if category == "3_text":
        ka = _text_content_key(a)
        kb = _text_content_key(b)
        if ka is not None and kb is not None:
            return ka == kb
    return _value_match(a, b)


def _safe_list(d, key):
    value = d.get(key, [])
    return value if isinstance(value, list) else []


def _max_bipartite_match_count(left_items, right_items, category):
    n = len(left_items)
    m = len(right_items)
    if n == 0 or m == 0:
        return 0

    adj = []
    for li in left_items:
        matches = []
        for j, rj in enumerate(right_items):
            if _item_match(category, li, rj):
                matches.append(j)
        adj.append(matches)

    match_r = [-1] * m

    def dfs(u, seen):
        for v in adj[u]:
            if seen[v]:
                continue
            seen[v] = True
            if match_r[v] == -1 or dfs(match_r[v], seen):
                match_r[v] = u
                return True
        return False

    matched = 0
    for u in range(n):
        seen = [False] * m
        if dfs(u, seen):
            matched += 1
    return matched


def _compute_file_jaccard(gt_data, pred_data):
    by_feature = {}
    total_intersection = 0
    total_union = 0

    for feature in FEATURE_ORDER:
        gt_items = _safe_list(gt_data, feature)
        pred_items = _safe_list(pred_data, feature)
        if feature == "3_text":
            gt_count, pred_count, intersection, union = _text_match_stats(gt_items, pred_items)
        else:
            gt_count = len(gt_items)
            pred_count = len(pred_items)
            intersection = min(gt_count, pred_count)
            union = max(gt_count, pred_count)
        jaccard = 1.0 if union == 0 else intersection / union

        by_feature[feature] = {
            "gt_count": gt_count,
            "pred_count": pred_count,
            "intersection": intersection,
            "union": union,
            "jaccard": round(jaccard, 6),
        }
        total_intersection += intersection
        total_union += union

    overall_jaccard = 1.0 if total_union == 0 else total_intersection / total_union
    return {
        "overall": {
            "intersection": total_intersection,
            "union": total_union,
            "jaccard": round(overall_jaccard, 6),
        },
        "by_feature": by_feature,
    }


def _infer_gt_basename(test_json_filename):
    stem = os.path.splitext(test_json_filename)[0]
    m = re.match(
        r"^([A-Za-z]+_\d+)(?:_(?:llm|vlm|code|code_image))?_(task|intent|operation|implementation)$",
        stem,
        re.IGNORECASE,
    )
    if m:
        stage = "intent" if m.group(2).lower() == "task" else m.group(2).lower()
        return f"{m.group(1)}.json", stage
    return f"{stem}.json", "unknown"


def _iter_model_files(model_root):
    items = []
    for root, _, files in os.walk(model_root):
        for filename in files:
            if not filename.endswith(".json"):
                continue
            path = os.path.join(root, filename)
            rel = os.path.relpath(path, model_root)
            parts = rel.split(os.sep)
            category = parts[0] if parts else "uncategorized"
            items.append((category, filename, path, rel))
    items.sort(key=lambda x: (x[0], x[1], x[3]))
    for item in items:
        yield item


def _build_gt_index(gt_root):
    index = {}
    for root, _, files in os.walk(gt_root):
        for filename in files:
            if not filename.endswith(".json"):
                continue
            path = os.path.join(root, filename)
            rel = os.path.relpath(path, gt_root)
            parts = rel.split(os.sep)
            category = parts[0] if parts else "uncategorized"
            key = (category, filename)

            rel_norm = rel.replace("\\", "/")
            preferred = "/GT/" in f"/{rel_norm}"
            if key not in index or preferred:
                index[key] = path
    return index


def _new_agg_bucket():
    return {
        "files": 0,
        "intersection": 0,
        "union": 0,
        "by_feature": {
            f: {"gt_count": 0, "pred_count": 0, "intersection": 0, "union": 0}
            for f in FEATURE_ORDER
        },
    }


def _finalize_agg_bucket(bucket):
    out = {
        "files": bucket["files"],
        "intersection": bucket["intersection"],
        "union": bucket["union"],
        "jaccard": round((1.0 if bucket["union"] == 0 else bucket["intersection"] / bucket["union"]), 6),
        "by_feature": {},
    }
    for f in FEATURE_ORDER:
        b = bucket["by_feature"][f]
        out["by_feature"][f] = {
            "gt_count": b["gt_count"],
            "pred_count": b["pred_count"],
            "intersection": b["intersection"],
            "union": b["union"],
            "jaccard": round((1.0 if b["union"] == 0 else b["intersection"] / b["union"]), 6),
        }
    return out


def _accumulate_bucket(bucket, file_eval):
    bucket["files"] += 1
    bucket["intersection"] += file_eval["overall"]["intersection"]
    bucket["union"] += file_eval["overall"]["union"]

    for f in FEATURE_ORDER:
        src = file_eval["by_feature"][f]
        dst = bucket["by_feature"][f]
        dst["gt_count"] += src["gt_count"]
        dst["pred_count"] += src["pred_count"]
        dst["intersection"] += src["intersection"]
        dst["union"] += src["union"]


def evaluate_model_against_gt(model_name, gt_root, model_root, gt_index=None):
    files = []
    missing_gt = []
    overall_bucket = _new_agg_bucket()
    by_stage = defaultdict(_new_agg_bucket)
    by_category = defaultdict(_new_agg_bucket)

    if gt_index is None:
        gt_index = _build_gt_index(gt_root)

    for category, filename, model_path, _ in _iter_model_files(model_root):
        gt_name, stage = _infer_gt_basename(filename)

        if stage not in EVALUATED_STAGES:
            continue

        gt_path = gt_index.get((category, gt_name))

        if not gt_path or not os.path.exists(gt_path):
            missing_gt.append(
                {
                    "category": category,
                    "model_file": filename,
                    "expected_gt": gt_name,
                }
            )
            continue

        with open(gt_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
        with open(model_path, "r", encoding="utf-8") as f:
            model_data = json.load(f)

        file_eval = _compute_file_jaccard(gt_data, model_data)
        record = {
            "model": model_name,
            "category": category,
            "stage": stage,
            "gt_file": os.path.relpath(gt_path),
            "pred_file": os.path.relpath(model_path),
            "overall": file_eval["overall"],
            "by_feature": file_eval["by_feature"],
        }
        files.append(record)

        _accumulate_bucket(overall_bucket, file_eval)
        _accumulate_bucket(by_stage[stage], file_eval)
        _accumulate_bucket(by_category[category], file_eval)

    files.sort(key=lambda x: (x["category"], x["gt_file"], x["stage"], x["pred_file"]))

    return {
        "model": model_name,
        "gt_root": gt_root,
        "pred_root": model_root,
        "matched_files": len(files),
        "missing_gt_files": missing_gt,
        "overall": _finalize_agg_bucket(overall_bucket),
        "by_stage": {k: _finalize_agg_bucket(v) for k, v in sorted(by_stage.items())},
        "by_chart_category": {k: _finalize_agg_bucket(v) for k, v in sorted(by_category.items())},
        "files": files,
    }


def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _iter_group_metrics(model_name, result):
    yield model_name, "overall", "overall", result["overall"]
    for stage, bucket in result["by_stage"].items():
        yield model_name, "stage", stage, bucket
    for category, bucket in result["by_chart_category"].items():
        yield model_name, "chart_category", category, bucket


def write_summary_csv(output_data, summary_csv_path):
    _ensure_parent_dir(summary_csv_path)
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "group_type",
                "group_name",
                "files",
                "intersection",
                "union",
                "jaccard",
            ]
        )
        for model_name in ("LLM", "VLM"):
            result = output_data["results"][model_name]
            for m, group_type, group_name, bucket in _iter_group_metrics(model_name, result):
                writer.writerow(
                    [
                        m,
                        group_type,
                        group_name,
                        bucket["files"],
                        bucket["intersection"],
                        bucket["union"],
                        bucket["jaccard"],
                    ]
                )


def write_feature_summary_csv(output_data, feature_csv_path):
    _ensure_parent_dir(feature_csv_path)
    with open(feature_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "group_type",
                "group_name",
                "feature",
                "gt_count",
                "pred_count",
                "intersection",
                "union",
                "jaccard",
            ]
        )
        for model_name in ("LLM", "VLM"):
            result = output_data["results"][model_name]
            for m, group_type, group_name, bucket in _iter_group_metrics(model_name, result):
                for feature in FEATURE_ORDER:
                    item = bucket["by_feature"][feature]
                    writer.writerow(
                        [
                            m,
                            group_type,
                            group_name,
                            feature,
                            item["gt_count"],
                            item["pred_count"],
                            item["intersection"],
                            item["union"],
                            item["jaccard"],
                        ]
                    )


def write_per_file_csv(output_data, file_csv_path):
    _ensure_parent_dir(file_csv_path)
    with open(file_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "category",
                "stage",
                "gt_file",
                "pred_file",
                "intersection",
                "union",
                "jaccard",
            ]
        )
        for model_name in ("LLM", "VLM"):
            files = output_data["results"][model_name]["files"]
            for item in files:
                writer.writerow(
                    [
                        item["model"],
                        item["category"],
                        item["stage"],
                        item["gt_file"],
                        item["pred_file"],
                        item["overall"]["intersection"],
                        item["overall"]["union"],
                        item["overall"]["jaccard"],
                    ]
                    )


def write_per_chart_jsons(output_data, per_chart_dir):
    if not per_chart_dir:
        return

    shutil.rmtree(per_chart_dir, ignore_errors=True)
    os.makedirs(per_chart_dir, exist_ok=True)

    for model_name in ("LLM", "VLM"):
        for item in output_data["results"][model_name]["files"]:
            category = item["category"]
            gt_basename = os.path.basename(item["gt_file"])
            chart_id = os.path.splitext(gt_basename)[0]
            stage = item["stage"]
            out_dir = os.path.join(per_chart_dir, category, chart_id, model_name)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{chart_id}_{stage}.json")
            payload = {
                "model": model_name,
                "category": category,
                "chart_id": chart_id,
                "stage": stage,
                "gt_file": item["gt_file"],
                "pred_file": item["pred_file"],
                "overall": item["overall"],
                "by_feature": item["by_feature"],
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)


def build_cli():
    parser = argparse.ArgumentParser(
        description="Compute Annotation Matching scores using Jaccard-style matching for LLM/VLM outputs."
    )
    parser.add_argument("--gt-root", default=str(DEFAULT_ANNOTATION_ROOT / "final_gt_structured"))
    parser.add_argument(
        "--llm-root",
        default=str(DEFAULT_ANNOTATION_ROOT / "final_test_llm_minus_removed_structured"),
    )
    parser.add_argument(
        "--vlm-root",
        default=str(DEFAULT_ANNOTATION_ROOT / "final_test_vlm_minus_removed_structured"),
    )
    parser.add_argument("--output-json", default=str(DEFAULT_ANALYSIS_ROOT / "annotation_matching_results.json"))
    parser.add_argument("--summary-csv", default=str(DEFAULT_ANALYSIS_ROOT / "annotation_matching_summary.csv"))
    parser.add_argument(
        "--feature-csv",
        default=str(DEFAULT_ANALYSIS_ROOT / "annotation_matching_feature_summary.csv"),
    )
    parser.add_argument("--file-csv", default=str(DEFAULT_ANALYSIS_ROOT / "annotation_matching_per_file.csv"))
    parser.add_argument("--per-chart-dir", default=str(DEFAULT_ANALYSIS_ROOT / "by_chart"))
    return parser


def main():
    args = build_cli().parse_args()

    gt_index = _build_gt_index(args.gt_root)
    llm_result = evaluate_model_against_gt("LLM", args.gt_root, args.llm_root, gt_index=gt_index)
    vlm_result = evaluate_model_against_gt("VLM", args.gt_root, args.vlm_root, gt_index=gt_index)

    output = {
        "settings": {
            "gt_root": args.gt_root,
            "llm_root": args.llm_root,
            "vlm_root": args.vlm_root,
            "evaluated_stages": sorted(EVALUATED_STAGES),
            "features": FEATURE_ORDER,
            "matching_logic": {
                "3_text": "text content coverage with count penalty: concatenate all predicted text elements into one blob; each normalized GT text is counted as matched if it appears in the predicted text blob, with word-boundary matching for very short texts",
                "others": "count-only: intersection=min(gt_count,pred_count), union=max(gt_count,pred_count)",
                "bbox_match_mode": BBOX_MATCH_MODE,
                "bbox_center_tol": BBOX_CENTER_TOL,
                "bbox_size_tol": BBOX_SIZE_TOL,
                "numeric_tol": NUMERIC_TOL,
            },
            "jaccard_formula": "3_text: matched_gt_text_count / max(|GT_texts|, |Pred_texts|); others: min(|GT|,|Pred|) / max(|GT|,|Pred|)",
            "intersection_logic": "3_text counts normalized GT text blocks found in concatenated normalized predictions; others use min/max count rule",
        },
        "results": {
            "LLM": llm_result,
            "VLM": vlm_result,
        },
    }

    _ensure_parent_dir(args.output_json)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved: {args.output_json}")
    for model_name in ("LLM", "VLM"):
        r = output["results"][model_name]["overall"]
        print(
            f"{model_name}: files={output['results'][model_name]['matched_files']} "
            f"intersection={r['intersection']} union={r['union']} jaccard={r['jaccard']}"
        )
        
    per_chart_dir = Path(args.per_chart_dir)
    per_chart_dir.mkdir(parents=True, exist_ok=True)
    for model_name in ("LLM", "VLM"):
        for item in output["results"][model_name].get("files", []):
            category = item["category"]
            stage = item["stage"]
            pred_file = Path(item["pred_file"])
            
            # Extract chart_id from pred_file stem
            parts = pred_file.stem.split("_")
            if len(parts) >= 2:
                chart_id = f"{parts[0]}_{parts[1]}"
            else:
                chart_id = pred_file.stem
                
            item["chart_id"] = chart_id
            
            json_dir = per_chart_dir / category / chart_id / model_name
            json_dir.mkdir(parents=True, exist_ok=True)
            json_path = json_dir / f"{chart_id}_{stage}.json"
            
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(item, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
