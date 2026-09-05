#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import xlsxwriter
except ImportError:
    xlsxwriter = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ROOT = PROJECT_ROOT / "outputs/api/semantic_design"
DEFAULT_DATASET_IMAGE_ROOT = PROJECT_ROOT / "outputs" / "eval_assets" / "dataset_image_new"
DEFAULT_DIFFICULTY_CSV = PROJECT_ROOT / "complexity_analysis/chart_complexity_rule_based.csv"

DEFAULT_OUT = DEFAULT_ROOT / "high_level_scores_by_model_stage.csv"
DEFAULT_RANK_OUT = DEFAULT_ROOT / "_ranking_by_mode_stage_missing_as_zero.csv"
DEFAULT_OVERALL_RANK_OUT = DEFAULT_ROOT / "_overall_model_ranking_missing_as_zero.csv"
DEFAULT_DIFF_OUT = DEFAULT_ROOT / "_vlm_vs_llm_relative_gain_missing_as_zero.csv"
DEFAULT_DIFF_XLSX_OUT = DEFAULT_ROOT / "_vlm_vs_llm_relative_gain_missing_as_zero.xlsx"

DEFAULT_DIFFICULTY_OUT = DEFAULT_ROOT / "_summary_by_difficulty_missing_as_zero.csv"
DEFAULT_DIFFICULTY_DIFF_OUT = DEFAULT_ROOT / "_vlm_vs_llm_by_difficulty_missing_as_zero.csv"
DEFAULT_DIFFICULTY_XLSX_OUT = DEFAULT_ROOT / "_summary_by_difficulty_missing_as_zero.xlsx"
DEFAULT_PER_SAMPLE_OUT = DEFAULT_ROOT / "high_level_scores_per_sample.csv"


MODEL_ORDER = [
    "claude46sonnet",
    "gemini3flash",
    "gemini3pro",
    "gpt54",
    "Kimi-K2.5",
    "gemma-4-31B-it",
    "Qwen3.5-397B-A17B",
    "Qwen3.5-122B-A10B",
    "Qwen3.5-27B",
    "Qwen3.5-9B",
]

MODEL_DISPLAY_NAME = {
    "gpt54": "GPT-5.4",
    "gemini3pro": "Gemini 3 Pro",
    "gemini3flash": "Gemini 3 Flash",
    "claude46sonnet": "Claude Sonnet 4.6",
    "Kimi-K2.5": "Kimi K2.5",
    "gemma-4-31B-it": "Gemma 4 31B",
    "Qwen3.5-397B-A17B": "Qwen3.5 397B",
    "Qwen3.5-122B-A10B": "Qwen3.5 122B",
    "Qwen3.5-27B": "Qwen3.5 27B",
    "Qwen3.5-9B": "Qwen3.5 9B",
}

CLOSED_SOURCE_MODELS = {
    "claude46sonnet",
    "gemini3flash",
    "gemini3pro",
    "gpt54",
    "Kimi-K2.5",
}

OPEN_SOURCE_MODELS = {
    "gemma-4-31B-it",
    "Qwen3.5-397B-A17B",
    "Qwen3.5-122B-A10B",
    "Qwen3.5-27B",
    "Qwen3.5-9B",
}

MODE_ORDER = ["LLM", "VLM"]
STAGE_ORDER = ["intent", "operation", "implementation"]
DIFFICULTY_ORDER = ["simple", "medium", "complex", "unknown"]

BASE_METRICS = ["sf", "sc", "vc", "org", "attn"]
ALL_METRICS = BASE_METRICS + ["semantic_avg", "design_avg", "overall_avg"]

METRIC_DISPLAY_NAME = {
    "sf": "semantic_faithfulness",
    "sc": "semantic_clarity",
    "vc": "visual_clarity",
    "org": "annotation_organization_quality",
    "attn": "attention_guidance",
    "semantic_avg": "semantic_consistency",
    "design_avg": "design_effectiveness",
    "overall_avg": "overall_avg",
}

PUBLIC_FIELD_NAME = {
    "sf": "semantic_faithfulness",
    "sc": "semantic_clarity",
    "vc": "visual_clarity",
    "org": "annotation_organization_quality",
    "attn": "attention_guidance",
    "semantic_avg": "semantic_consistency",
    "design_avg": "design_effectiveness",
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def display_model_name(source_model: str) -> str:
    return MODEL_DISPLAY_NAME.get(source_model, source_model)


def display_stage_name(stage: str) -> str:
    return {
        "intent": "intent",
        "operation": "op",
        "implementation": "imp",
        "overall": "overall",
        "ALL": "ALL",
    }.get(stage, stage)


def model_family(source_model: str) -> str:
    if source_model in CLOSED_SOURCE_MODELS:
        return "closed"
    if source_model in OPEN_SOURCE_MODELS:
        return "open"
    return "unknown"


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def public_field_name(name: str) -> str:
    for src, dst in PUBLIC_FIELD_NAME.items():
        if name == src:
            return dst
        if name.startswith(f"{src}_"):
            return f"{dst}_{name[len(src) + 1:]}"
    if name == "group_mean_semantic_avg":
        return "group_mean_semantic_consistency"
    if name == "group_mean_design_avg":
        return "group_mean_design_effectiveness"
    return name


def public_csv_row(row: dict[str, Any], fieldnames: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in fieldnames:
        public_name = public_field_name(field)
        value = row.get(field, "")
        if field == "metric" and isinstance(value, str):
            value = PUBLIC_FIELD_NAME.get(value, value)
        out[public_name] = value
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    public_fieldnames = [public_field_name(field) for field in fieldnames]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=public_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(public_csv_row(row, fieldnames) for row in rows)


def model_order_idx(source_model: str) -> int:
    return MODEL_ORDER.index(source_model) if source_model in MODEL_ORDER else len(MODEL_ORDER)


def mode_order_idx(mode: str) -> int:
    return MODE_ORDER.index(mode) if mode in MODE_ORDER else len(MODE_ORDER)


def stage_order_idx(stage: str) -> int:
    return STAGE_ORDER.index(stage) if stage in STAGE_ORDER else len(STAGE_ORDER)


def metric_order_idx(metric: str) -> int:
    return ALL_METRICS.index(metric) if metric in ALL_METRICS else len(ALL_METRICS)


def avg(values: list[float]) -> float | str:
    if not values:
        return ""
    return sum(values) / len(values)


def round_or_blank(value: Any, ndigits: int = 4) -> Any:
    if value == "" or value is None:
        return ""
    try:
        return round(float(value), ndigits)
    except Exception:
        return ""


def pct_change(after: Any, before: Any, ndigits: int = 4) -> Any:
    try:
        before_f = float(before)
        after_f = float(after)
        if before_f == 0:
            return ""
        return round((after_f - before_f) / before_f * 100.0, ndigits)
    except Exception:
        return ""


def avg_nonempty(values: list[Any], ndigits: int = 4) -> Any:
    nums: list[float] = []
    for v in values:
        if v == "" or v is None:
            continue
        try:
            nums.append(float(v))
        except Exception:
            continue
    if not nums:
        return ""
    return round(sum(nums) / len(nums), ndigits)


def zero_scores() -> dict[str, float]:
    scores = {m: 0.0 for m in BASE_METRICS}
    scores["semantic_avg"] = 0.0
    scores["design_avg"] = 0.0
    scores["overall_avg"] = 0.0
    return scores


def extract_scores(payload: dict[str, Any]) -> dict[str, float] | None:
    if payload.get("parse_ok") is not True:
        return None

    result = payload.get("result")
    if not isinstance(result, dict):
        return None

    results = result.get("results")
    if not isinstance(results, list) or not results:
        return None

    row = results[0]
    if not isinstance(row, dict):
        return None

    if "org" not in row and "comp" in row:
        row = dict(row)
        row["org"] = row["comp"]

    scores: dict[str, float] = {}
    for metric in BASE_METRICS:
        value = row.get(metric)
        if not isinstance(value, (int, float)):
            return None
        scores[metric] = float(value)

    scores["semantic_avg"] = (scores["sf"] + scores["sc"]) / 2.0
    scores["design_avg"] = (scores["vc"] + scores["org"] + scores["attn"]) / 3.0
    scores["overall_avg"] = sum(scores[m] for m in BASE_METRICS) / len(BASE_METRICS)

    return scores


def load_difficulty_map(path: Path) -> dict[str, str]:
    rows = read_csv_rows(path)
    if not rows:
        print(f"WARN: difficulty CSV not found or empty: {path}")
        return {}

    difficulty_by_key: dict[str, str] = {}

    for row in rows:
        category = row.get("category", "").strip()
        chart_id = row.get("chart_id", "").strip()
        difficulty = row.get("complexity_level_rule", "").strip().lower()

        if not category or not chart_id:
            continue

        if difficulty not in {"simple", "medium", "complex"}:
            difficulty = "unknown"

        difficulty_by_key[f"{category}/{chart_id}"] = difficulty

    return difficulty_by_key


def collect_dataset_charts(dataset_image_root: Path) -> list[dict[str, str]]:
    """
    Collect chart list from:
      dataset_image_new/<Category>/<ChartID>.<ext>

    Also supports:
      dataset_image_new/<Category>/<ChartID>/<ChartID>.<ext>
    """
    charts: dict[str, dict[str, str]] = {}

    if not dataset_image_root.exists():
        raise FileNotFoundError(f"dataset image root not found: {dataset_image_root}")

    for path in sorted(dataset_image_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMG_EXTS:
            continue

        try:
            rel = path.relative_to(dataset_image_root)
        except ValueError:
            continue

        parts = rel.parts
        if len(parts) < 2:
            continue

        category = parts[0]
        chart_id = path.stem

        key = f"{category}/{chart_id}"
        charts[key] = {
            "category": category,
            "chart_id": chart_id,
            "chart_key": key,
            "dataset_image": str(path),
        }

    out = list(charts.values())
    out.sort(key=lambda r: (r["category"], r["chart_id"]))
    return out


def sd_path_for(
    root: Path,
    source_model: str,
    category: str,
    chart_id: str,
    mode: str,
    stage: str,
) -> Path:
    return (
        root
        / source_model
        / category
        / chart_id
        / mode
        / f"{chart_id}_{mode.lower()}_{stage}_sd.json"
    )


def load_one_result_scores(path: Path) -> tuple[dict[str, float], bool, str]:
    """
    Return:
      scores, valid_result, reason

    Missing or invalid result is counted as all-zero.
    """
    if not path.exists():
        return zero_scores(), False, "missing_json"

    payload = read_json(path)
    if payload is None:
        return zero_scores(), False, "bad_json"

    scores = extract_scores(payload)
    if scores is None:
        return zero_scores(), False, "parse_fail_or_invalid_scores"

    return scores, True, "ok"


def sort_key(item: tuple[tuple[str, str, str], dict[str, Any]]) -> tuple:
    source_model, mode, layer = item[0]
    return (
        model_order_idx(source_model),
        mode_order_idx(mode),
        stage_order_idx(layer),
        source_model,
        mode,
        layer,
    )


def build_ranking_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranking_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for row in summary_rows:
        grouped[(str(row["model"]), str(row["stage"]))].append(row)

    for mode in MODE_ORDER:
        for stage in STAGE_ORDER:
            rows = grouped.get((mode, stage), [])
            if not rows:
                continue

            sorted_rows = sorted(
                rows,
                key=lambda r: (
                    -float(r["overall_avg"]),
                    -float(r["semantic_avg"]),
                    -float(r["design_avg"]),
                    model_order_idx(str(r["source_model"])),
                ),
            )

            group_mean_overall = sum(float(r["overall_avg"]) for r in rows) / len(rows)
            group_mean_semantic = sum(float(r["semantic_avg"]) for r in rows) / len(rows)
            group_mean_design = sum(float(r["design_avg"]) for r in rows) / len(rows)

            for rank, row in enumerate(sorted_rows, start=1):
                ranking_rows.append(
                    {
                        "model": mode,
                        "stage": stage,
                        "rank": rank,
                        "source_model": row["source_model"],
                        "count": row["count"],
                        "valid_count": row["valid_count"],
                        "missing_count": row["missing_count"],
                        "sf": row["sf"],
                        "sc": row["sc"],
                        "vc": row["vc"],
                        "org": row["org"],
                        "attn": row["attn"],
                        "semantic_avg": row["semantic_avg"],
                        "design_avg": row["design_avg"],
                        "overall_avg": row["overall_avg"],
                        "group_mean_semantic_avg": round(group_mean_semantic, 4),
                        "group_mean_design_avg": round(group_mean_design, 4),
                        "group_mean_overall_avg": round(group_mean_overall, 4),
                    }
                )

    return ranking_rows


def build_overall_model_ranking_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        grouped[str(row["source_model"])].append(row)

    rows: list[dict[str, Any]] = []
    ordered_models = list(MODEL_ORDER)
    extra_models = sorted(m for m in grouped.keys() if m not in set(MODEL_ORDER))
    ordered_models.extend(extra_models)

    for source_model in ordered_models:
        model_rows = grouped.get(source_model, [])
        if not model_rows:
            continue

        lookup = {
            (str(r["model"]).upper(), str(r["stage"]).lower()): r
            for r in model_rows
        }

        task_values: dict[str, float | str] = {}
        counts: dict[str, int | str] = {}
        valid_counts: dict[str, int | str] = {}
        missing_counts: dict[str, int | str] = {}

        for mode in MODE_ORDER:
            for stage in STAGE_ORDER:
                key_name = f"{mode.lower()}_{stage}"
                r = lookup.get((mode, stage))
                if r is None:
                    task_values[f"{key_name}_overall_avg"] = ""
                    counts[f"{key_name}_count"] = ""
                    valid_counts[f"{key_name}_valid_count"] = ""
                    missing_counts[f"{key_name}_missing_count"] = ""
                else:
                    task_values[f"{key_name}_overall_avg"] = float(r["overall_avg"])
                    counts[f"{key_name}_count"] = int(r["count"])
                    valid_counts[f"{key_name}_valid_count"] = int(r["valid_count"])
                    missing_counts[f"{key_name}_missing_count"] = int(r["missing_count"])

        available_overall = [
            float(v)
            for k, v in task_values.items()
            if k.endswith("_overall_avg") and isinstance(v, (int, float))
        ]

        available_semantic = [float(r["semantic_avg"]) for r in model_rows]
        available_design = [float(r["design_avg"]) for r in model_rows]
        available_sf = [float(r["sf"]) for r in model_rows]
        available_sc = [float(r["sc"]) for r in model_rows]
        available_vc = [float(r["vc"]) for r in model_rows]
        available_org = [float(r["org"]) for r in model_rows]
        available_attn = [float(r["attn"]) for r in model_rows]

        total_count = sum(int(r["count"]) for r in model_rows)
        total_valid_count = sum(int(r["valid_count"]) for r in model_rows)
        total_missing_count = sum(int(r["missing_count"]) for r in model_rows)

        rows.append(
            {
                "source_model": source_model,
                "intent_group_count": len(available_overall),
                "total_sample_count": total_count,
                "total_valid_count": total_valid_count,
                "total_missing_count": total_missing_count,
                "overall_avg": round(sum(available_overall) / len(available_overall), 4)
                if available_overall
                else "",
                "semantic_avg": round(sum(available_semantic) / len(available_semantic), 4)
                if available_semantic
                else "",
                "design_avg": round(sum(available_design) / len(available_design), 4)
                if available_design
                else "",
                "sf": round(sum(available_sf) / len(available_sf), 4) if available_sf else "",
                "sc": round(sum(available_sc) / len(available_sc), 4) if available_sc else "",
                "vc": round(sum(available_vc) / len(available_vc), 4) if available_vc else "",
                "org": round(sum(available_org) / len(available_org), 4) if available_org else "",
                "attn": round(sum(available_attn) / len(available_attn), 4) if available_attn else "",
                **task_values,
                **counts,
                **valid_counts,
                **missing_counts,
            }
        )

    rows = sorted(
        rows,
        key=lambda r: (
            -float(r["overall_avg"]) if r["overall_avg"] != "" else float("inf"),
            -float(r["semantic_avg"]) if r["semantic_avg"] != "" else float("inf"),
            -float(r["design_avg"]) if r["design_avg"] != "" else float("inf"),
            model_order_idx(str(r["source_model"])),
        ),
    )

    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx

    final_rows: list[dict[str, Any]] = []
    for row in rows:
        final_row = {"rank": row.pop("rank")}
        final_row.update(row)
        final_rows.append(final_row)

    return final_rows


def build_vlm_vs_llm_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in summary_rows:
        key = (
            str(row["source_model"]),
            str(row["model"]).upper(),
            str(row["stage"]).lower(),
        )
        lookup[key] = row

    diff_rows: list[dict[str, Any]] = []

    def build_one_row(source_model: str, stage: str) -> dict[str, Any] | None:
        llm = lookup.get((source_model, "LLM", stage))
        vlm = lookup.get((source_model, "VLM", stage))

        if llm is None and vlm is None:
            return None

        row_out: dict[str, Any] = {
            "source_model": source_model,
            "stage": stage,
            "llm_count": llm.get("count") if llm else "",
            "vlm_count": vlm.get("count") if vlm else "",
            "llm_valid_count": llm.get("valid_count") if llm else "",
            "vlm_valid_count": vlm.get("valid_count") if vlm else "",
            "llm_missing_count": llm.get("missing_count") if llm else "",
            "vlm_missing_count": vlm.get("missing_count") if vlm else "",
            "has_both": llm is not None and vlm is not None,
        }

        for metric in ALL_METRICS:
            if llm is None or vlm is None:
                row_out[f"{metric}_llm"] = ""
                row_out[f"{metric}_vlm"] = ""
                row_out[f"{metric}_abs_diff"] = ""
                row_out[f"{metric}_rel_diff_pct"] = ""
                continue

            llm_value = float(llm[metric])
            vlm_value = float(vlm[metric])
            abs_diff = vlm_value - llm_value
            rel_diff_pct = "" if llm_value == 0 else abs_diff / llm_value * 100.0

            row_out[f"{metric}_llm"] = round(llm_value, 4)
            row_out[f"{metric}_vlm"] = round(vlm_value, 4)
            row_out[f"{metric}_abs_diff"] = round(abs_diff, 4)
            row_out[f"{metric}_rel_diff_pct"] = (
                "" if rel_diff_pct == "" else round(float(rel_diff_pct), 4)
            )

        return row_out

    for source_model in MODEL_ORDER:
        for stage in STAGE_ORDER:
            row = build_one_row(source_model, stage)
            if row is not None:
                diff_rows.append(row)

    known = set(MODEL_ORDER)
    extra_models = sorted(
        {str(r["source_model"]) for r in summary_rows if str(r["source_model"]) not in known}
    )

    for source_model in extra_models:
        for stage in STAGE_ORDER:
            row = build_one_row(source_model, stage)
            if row is not None:
                diff_rows.append(row)

    return diff_rows


def build_abs_diff_sheet_rows(diff_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    fieldnames = [
        "source_model",
        "model_family",
        "stage",
        "llm_count",
        "vlm_count",
        "llm_valid_count",
        "vlm_valid_count",
        "llm_missing_count",
        "vlm_missing_count",
        "has_both",
    ]

    for metric in ALL_METRICS:
        fieldnames.append(f"{metric}_diff")

    rows: list[dict[str, Any]] = []

    for row in diff_rows:
        source_model = str(row["source_model"])
        out: dict[str, Any] = {
            "source_model": source_model,
            "model_family": model_family(source_model),
            "stage": row["stage"],
            "llm_count": row["llm_count"],
            "vlm_count": row["vlm_count"],
            "llm_valid_count": row["llm_valid_count"],
            "vlm_valid_count": row["vlm_valid_count"],
            "llm_missing_count": row["llm_missing_count"],
            "vlm_missing_count": row["vlm_missing_count"],
            "has_both": row["has_both"],
        }

        for metric in ALL_METRICS:
            out[f"{metric}_diff"] = row.get(f"{metric}_abs_diff", "")

        rows.append(out)

    def make_avg_row(label: str, family: str, source_rows: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {
            "source_model": label,
            "model_family": family,
            "stage": "ALL",
            "llm_count": "",
            "vlm_count": "",
            "llm_valid_count": "",
            "vlm_valid_count": "",
            "llm_missing_count": "",
            "vlm_missing_count": "",
            "has_both": "",
        }

        for metric in ALL_METRICS:
            key = f"{metric}_diff"
            out[key] = avg_nonempty([r.get(key, "") for r in source_rows], ndigits=4)

        return out

    closed_rows = [r for r in rows if r.get("model_family") == "closed" and r.get("has_both") is True]
    open_rows = [r for r in rows if r.get("model_family") == "open" and r.get("has_both") is True]
    overall_rows = [r for r in rows if r.get("has_both") is True]

    rows.append(make_avg_row("Closed-source avg", "closed", closed_rows))
    rows.append(make_avg_row("Open-source avg", "open", open_rows))
    rows.append(make_avg_row("Overall avg", "overall", overall_rows))

    return rows, fieldnames


def build_rel_diff_sheet_rows(diff_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    fieldnames = [
        "source_model",
        "model_family",
        "stage",
        "llm_count",
        "vlm_count",
        "llm_valid_count",
        "vlm_valid_count",
        "llm_missing_count",
        "vlm_missing_count",
        "has_both",
    ]

    for metric in ALL_METRICS:
        fieldnames.append(f"{metric}_rel_diff_pct")

    rows: list[dict[str, Any]] = []

    for row in diff_rows:
        source_model = str(row["source_model"])
        out: dict[str, Any] = {
            "source_model": source_model,
            "model_family": model_family(source_model),
            "stage": row["stage"],
            "llm_count": row["llm_count"],
            "vlm_count": row["vlm_count"],
            "llm_valid_count": row["llm_valid_count"],
            "vlm_valid_count": row["vlm_valid_count"],
            "llm_missing_count": row["llm_missing_count"],
            "vlm_missing_count": row["vlm_missing_count"],
            "has_both": row["has_both"],
        }

        for metric in ALL_METRICS:
            out[f"{metric}_rel_diff_pct"] = row.get(f"{metric}_rel_diff_pct", "")

        rows.append(out)

    def make_avg_row(label: str, family: str, source_rows: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {
            "source_model": label,
            "model_family": family,
            "stage": "ALL",
            "llm_count": "",
            "vlm_count": "",
            "llm_valid_count": "",
            "vlm_valid_count": "",
            "llm_missing_count": "",
            "vlm_missing_count": "",
            "has_both": "",
        }

        for metric in ALL_METRICS:
            key = f"{metric}_rel_diff_pct"
            out[key] = avg_nonempty([r.get(key, "") for r in source_rows], ndigits=4)

        return out

    closed_rows = [r for r in rows if r.get("model_family") == "closed" and r.get("has_both") is True]
    open_rows = [r for r in rows if r.get("model_family") == "open" and r.get("has_both") is True]
    overall_rows = [r for r in rows if r.get("has_both") is True]

    rows.append(make_avg_row("Closed-source avg", "closed", closed_rows))
    rows.append(make_avg_row("Open-source avg", "open", open_rows))
    rows.append(make_avg_row("Overall avg", "overall", overall_rows))

    return rows, fieldnames


def build_difficulty_score_rows(per_sample_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bucket: dict[tuple[str, str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: {d: [] for d in DIFFICULTY_ORDER}
    )
    count_bucket: dict[tuple[str, str, str, str], dict[str, int]] = defaultdict(
        lambda: {d: 0 for d in DIFFICULTY_ORDER}
    )
    valid_bucket: dict[tuple[str, str, str, str], dict[str, int]] = defaultdict(
        lambda: {d: 0 for d in DIFFICULTY_ORDER}
    )

    for item in per_sample_rows:
        difficulty = str(item.get("difficulty", "unknown")).lower()
        if difficulty not in DIFFICULTY_ORDER:
            difficulty = "unknown"

        for metric in ALL_METRICS:
            key = (
                str(item["source_model"]),
                str(item["mode"]),
                str(item["stage"]),
                metric,
            )
            bucket[key][difficulty].append(float(item[metric]))
            count_bucket[key][difficulty] += 1
            if bool(item.get("valid_result")):
                valid_bucket[key][difficulty] += 1

    rows: list[dict[str, Any]] = []

    for key, values in bucket.items():
        source_model, mode, stage, metric = key

        simple = avg(values["simple"])
        medium = avg(values["medium"])
        complex_v = avg(values["complex"])

        simple_minus_complex = (
            float(simple) - float(complex_v)
            if simple != "" and complex_v != ""
            else ""
        )
        complex_minus_simple = (
            float(complex_v) - float(simple)
            if simple != "" and complex_v != ""
            else ""
        )

        rows.append(
            {
                "source_model": source_model,
                "model_family": model_family(source_model),
                "mode": mode,
                "stage": stage,
                "metric": metric,
                "simple_count": count_bucket[key]["simple"],
                "medium_count": count_bucket[key]["medium"],
                "complex_count": count_bucket[key]["complex"],
                "unknown_count": count_bucket[key]["unknown"],
                "simple_valid_count": valid_bucket[key]["simple"],
                "medium_valid_count": valid_bucket[key]["medium"],
                "complex_valid_count": valid_bucket[key]["complex"],
                "unknown_valid_count": valid_bucket[key]["unknown"],
                "simple_missing_count": count_bucket[key]["simple"] - valid_bucket[key]["simple"],
                "medium_missing_count": count_bucket[key]["medium"] - valid_bucket[key]["medium"],
                "complex_missing_count": count_bucket[key]["complex"] - valid_bucket[key]["complex"],
                "unknown_missing_count": count_bucket[key]["unknown"] - valid_bucket[key]["unknown"],
                "simple": round_or_blank(simple, 4),
                "medium": round_or_blank(medium, 4),
                "complex": round_or_blank(complex_v, 4),
                "simple_minus_complex": round_or_blank(simple_minus_complex, 4),
                "complex_minus_simple": round_or_blank(complex_minus_simple, 4),
                "simple_to_complex_rel_diff_pct": pct_change(complex_v, simple, 4),
            }
        )

    rows.sort(
        key=lambda r: (
            model_order_idx(str(r["source_model"])),
            mode_order_idx(str(r["mode"])),
            stage_order_idx(str(r["stage"])),
            metric_order_idx(str(r["metric"])),
        )
    )

    return rows


def build_difficulty_vlm_llm_rows(difficulty_score_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    for row in difficulty_score_rows:
        key = (
            str(row["source_model"]),
            str(row["mode"]),
            str(row["stage"]),
            str(row["metric"]),
        )
        lookup[key] = row

    source_models = list(MODEL_ORDER)
    extra_models = sorted(
        {
            str(r["source_model"])
            for r in difficulty_score_rows
            if str(r["source_model"]) not in set(MODEL_ORDER)
        }
    )
    source_models.extend(extra_models)

    rows: list[dict[str, Any]] = []

    for source_model in source_models:
        for stage in STAGE_ORDER:
            for metric in ALL_METRICS:
                llm = lookup.get((source_model, "LLM", stage, metric))
                vlm = lookup.get((source_model, "VLM", stage, metric))

                if llm is None and vlm is None:
                    continue

                row: dict[str, Any] = {
                    "source_model": source_model,
                    "model_family": model_family(source_model),
                    "stage": stage,
                    "metric": metric,
                    "has_both": llm is not None and vlm is not None,
                }

                for difficulty in ["simple", "medium", "complex"]:
                    llm_val = "" if llm is None else llm.get(difficulty, "")
                    vlm_val = "" if vlm is None else vlm.get(difficulty, "")

                    row[f"{difficulty}_llm"] = llm_val
                    row[f"{difficulty}_vlm"] = vlm_val

                    try:
                        diff = float(vlm_val) - float(llm_val)
                        row[f"{difficulty}_abs_diff"] = round(diff, 4)
                        row[f"{difficulty}_rel_diff_pct"] = (
                            "" if float(llm_val) == 0 else round(diff / float(llm_val) * 100.0, 4)
                        )
                    except Exception:
                        row[f"{difficulty}_abs_diff"] = ""
                        row[f"{difficulty}_rel_diff_pct"] = ""

                rows.append(row)

    return rows


def build_difficulty_metric_group_rows(difficulty_score_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)

    for row in difficulty_score_rows:
        key = (str(row["source_model"]), str(row["mode"]), str(row["stage"]))
        grouped[key][str(row["metric"])] = row

    rows: list[dict[str, Any]] = []

    for (source_model, mode, stage), metric_map in grouped.items():
        out: dict[str, Any] = {
            "source_model": source_model,
            "model": display_model_name(source_model),
            "model_family": model_family(source_model),
            "mode": mode,
            "stage": display_stage_name(stage),
        }

        for metric in ALL_METRICS:
            src = metric_map.get(metric, {})
            out[f"{metric}_simple"] = src.get("simple", "")
            out[f"{metric}_medium"] = src.get("medium", "")
            out[f"{metric}_complex"] = src.get("complex", "")
            out[f"{metric}_simple_minus_complex"] = src.get("simple_minus_complex", "")
            out[f"{metric}_simple_to_complex_rel_diff_pct"] = src.get("simple_to_complex_rel_diff_pct", "")

        rows.append(out)

    rows.sort(
        key=lambda r: (
            model_order_idx(str(r["source_model"])),
            mode_order_idx(str(r["mode"])),
            stage_order_idx(
                {"intent": "intent", "op": "operation", "imp": "implementation"}.get(
                    str(r["stage"]), str(r["stage"])
                )
            ),
        )
    )

    return rows


def build_difficulty_vlm_llm_metric_group_rows(difficulty_diff_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)

    for row in difficulty_diff_rows:
        key = (str(row["source_model"]), str(row["stage"]))
        grouped[key][str(row["metric"])] = row

    rows: list[dict[str, Any]] = []

    for (source_model, stage), metric_map in grouped.items():
        out: dict[str, Any] = {
            "source_model": source_model,
            "model": display_model_name(source_model),
            "model_family": model_family(source_model),
            "stage": display_stage_name(stage),
        }

        for metric in ALL_METRICS:
            src = metric_map.get(metric, {})
            for difficulty in ["simple", "medium", "complex"]:
                out[f"{metric}_{difficulty}_abs_diff"] = src.get(f"{difficulty}_abs_diff", "")
                out[f"{metric}_{difficulty}_rel_diff_pct"] = src.get(f"{difficulty}_rel_diff_pct", "")

        rows.append(out)

    rows.sort(
        key=lambda r: (
            model_order_idx(str(r["source_model"])),
            stage_order_idx(
                {"intent": "intent", "op": "operation", "imp": "implementation"}.get(
                    str(r["stage"]), str(r["stage"])
                )
            ),
        )
    )

    return rows


def add_avg_rows_score_wide(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = list(rows)

    def make_avg(label: str, family: str, source_rows: list[dict[str, Any]]) -> dict[str, Any]:
        row: dict[str, Any] = {
            "source_model": label,
            "model": label,
            "model_family": family,
            "mode": "ALL",
            "stage": "ALL",
        }

        for metric in ALL_METRICS:
            for suffix in [
                "simple",
                "medium",
                "complex",
                "simple_minus_complex",
                "simple_to_complex_rel_diff_pct",
            ]:
                key = f"{metric}_{suffix}"
                row[key] = avg_nonempty([r.get(key, "") for r in source_rows], ndigits=4)

        return row

    closed = [r for r in rows if r.get("model_family") == "closed"]
    open_ = [r for r in rows if r.get("model_family") == "open"]

    out.append(make_avg("Closed-source avg", "closed", closed))
    out.append(make_avg("Open-source avg", "open", open_))
    out.append(make_avg("Overall avg", "overall", rows))

    return out


def add_avg_rows_diff_wide(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = list(rows)

    def make_avg(label: str, family: str, source_rows: list[dict[str, Any]]) -> dict[str, Any]:
        row: dict[str, Any] = {
            "source_model": label,
            "model": label,
            "model_family": family,
            "stage": "ALL",
        }

        for metric in ALL_METRICS:
            for difficulty in ["simple", "medium", "complex"]:
                for suffix in ["abs_diff", "rel_diff_pct"]:
                    key = f"{metric}_{difficulty}_{suffix}"
                    row[key] = avg_nonempty([r.get(key, "") for r in source_rows], ndigits=4)

        return row

    closed = [r for r in rows if r.get("model_family") == "closed"]
    open_ = [r for r in rows if r.get("model_family") == "open"]

    out.append(make_avg("Closed-source avg", "closed", closed))
    out.append(make_avg("Open-source avg", "open", open_))
    out.append(make_avg("Overall avg", "overall", rows))

    return out


def write_vlm_vs_llm_xlsx(xlsx_path: Path, diff_rows: list[dict[str, Any]]) -> bool:
    if xlsxwriter is None:
        print(f"WARN: XlsxWriter is not installed; skip XLSX output: {xlsx_path}")
        return False

    xlsx_path.parent.mkdir(parents=True, exist_ok=True)

    abs_rows, abs_fieldnames = build_abs_diff_sheet_rows(diff_rows)
    rel_rows, rel_fieldnames = build_rel_diff_sheet_rows(diff_rows)

    workbook = xlsxwriter.Workbook(str(xlsx_path))

    fmt_title = workbook.add_format({"bold": True, "font_size": 14, "align": "left", "valign": "vcenter"})
    fmt_header = workbook.add_format(
        {"bold": True, "bg_color": "#F2F4F7", "border": 1, "align": "center", "valign": "vcenter", "text_wrap": True}
    )
    fmt_text = workbook.add_format({"border": 1, "align": "left", "valign": "vcenter"})
    fmt_num_abs = workbook.add_format({"border": 1, "align": "center", "valign": "vcenter", "num_format": "0.0000"})
    fmt_num_pct = workbook.add_format({"border": 1, "align": "center", "valign": "vcenter", "num_format": '0.00"%"'})
    fmt_int = workbook.add_format({"border": 1, "align": "center", "valign": "vcenter", "num_format": "0"})
    fmt_bool = workbook.add_format({"border": 1, "align": "center", "valign": "vcenter"})
    fmt_pos_abs = workbook.add_format({"font_color": "#C00000", "border": 1, "align": "center", "valign": "vcenter", "num_format": "0.0000"})
    fmt_neg_abs = workbook.add_format({"font_color": "#0050B5", "border": 1, "align": "center", "valign": "vcenter", "num_format": "0.0000"})
    fmt_pos_pct = workbook.add_format({"font_color": "#C00000", "border": 1, "align": "center", "valign": "vcenter", "num_format": '0.00"%"'})
    fmt_neg_pct = workbook.add_format({"font_color": "#0050B5", "border": 1, "align": "center", "valign": "vcenter", "num_format": '0.00"%"'})
    fmt_avg_abs = workbook.add_format({"bold": True, "bg_color": "#FFF2CC", "border": 1, "align": "center", "valign": "vcenter", "num_format": "0.0000"})
    fmt_avg_pct = workbook.add_format({"bold": True, "bg_color": "#FFF2CC", "border": 1, "align": "center", "valign": "vcenter", "num_format": '0.00"%"'})
    fmt_avg_text = workbook.add_format({"bold": True, "bg_color": "#FFF2CC", "border": 1, "align": "left", "valign": "vcenter"})

    def write_sheet(
        *,
        sheet_name: str,
        title: str,
        rows: list[dict[str, Any]],
        fieldnames: list[str],
        is_pct: bool,
    ) -> None:
        ws = workbook.add_worksheet(sheet_name)
        ws.write(0, 0, title, fmt_title)
        ws.freeze_panes(2, 0)

        for col, name in enumerate(fieldnames):
            ws.write(1, col, name, fmt_header)

        for r_idx, row in enumerate(rows, start=2):
            is_avg_row = str(row.get("source_model", "")).endswith("avg")

            for c_idx, name in enumerate(fieldnames):
                value = row.get(name, "")

                if name in {"source_model", "model_family", "stage"}:
                    ws.write(r_idx, c_idx, value, fmt_avg_text if is_avg_row else fmt_text)
                elif name.endswith("_count") or name in {"llm_count", "vlm_count"}:
                    if value == "":
                        ws.write(r_idx, c_idx, "", fmt_avg_pct if is_pct else fmt_avg_abs)
                    else:
                        ws.write_number(r_idx, c_idx, int(value), fmt_int)
                elif name == "has_both":
                    ws.write(r_idx, c_idx, str(value), fmt_avg_text if is_avg_row else fmt_bool)
                else:
                    if value == "":
                        ws.write(r_idx, c_idx, "", fmt_avg_pct if is_pct else fmt_avg_abs)
                        continue

                    numeric_value = float(value)

                    if is_avg_row:
                        ws.write_number(r_idx, c_idx, numeric_value, fmt_avg_pct if is_pct else fmt_avg_abs)
                    elif numeric_value > 0:
                        ws.write_number(r_idx, c_idx, numeric_value, fmt_pos_pct if is_pct else fmt_pos_abs)
                    elif numeric_value < 0:
                        ws.write_number(r_idx, c_idx, numeric_value, fmt_neg_pct if is_pct else fmt_neg_abs)
                    else:
                        ws.write_number(r_idx, c_idx, numeric_value, fmt_num_pct if is_pct else fmt_num_abs)

        ws.autofilter(1, 0, max(1, len(rows) + 1), len(fieldnames) - 1)

        for col, name in enumerate(fieldnames):
            if name == "source_model":
                ws.set_column(col, col, 22)
            elif name == "model_family":
                ws.set_column(col, col, 14)
            elif name == "stage":
                ws.set_column(col, col, 16)
            elif name.endswith("_count") or name == "has_both":
                ws.set_column(col, col, 12)
            else:
                ws.set_column(col, col, 16)

        ws.set_row(0, 24)
        ws.set_row(1, 34)

    write_sheet(
        sheet_name="VLM-vs-LLM",
        title="VLM vs LLM absolute gain, missing results counted as zero",
        rows=abs_rows,
        fieldnames=abs_fieldnames,
        is_pct=False,
    )

    write_sheet(
        sheet_name="VLM-vs-LLM %",
        title="VLM vs LLM relative gain (%), missing results counted as zero",
        rows=rel_rows,
        fieldnames=rel_fieldnames,
        is_pct=True,
    )

    workbook.close()
    return True


def write_difficulty_xlsx(
    xlsx_path: Path,
    difficulty_score_rows: list[dict[str, Any]],
    difficulty_diff_rows: list[dict[str, Any]],
) -> bool:
    if xlsxwriter is None:
        print(f"WARN: XlsxWriter is not installed; skip XLSX output: {xlsx_path}")
        return False

    xlsx_path.parent.mkdir(parents=True, exist_ok=True)

    score_wide_rows = add_avg_rows_score_wide(
        build_difficulty_metric_group_rows(difficulty_score_rows)
    )
    diff_wide_rows = add_avg_rows_diff_wide(
        build_difficulty_vlm_llm_metric_group_rows(difficulty_diff_rows)
    )

    workbook = xlsxwriter.Workbook(str(xlsx_path))

    fmt_title = workbook.add_format({"bold": True, "font_size": 14, "align": "left", "valign": "vcenter"})
    fmt_header_green = workbook.add_format(
        {"bold": True, "bg_color": "#E2F0D9", "border": 1, "align": "center", "valign": "vcenter", "text_wrap": True}
    )
    fmt_header_blue = workbook.add_format(
        {"bold": True, "bg_color": "#D9EAF7", "border": 1, "align": "center", "valign": "vcenter", "text_wrap": True}
    )
    fmt_subheader = workbook.add_format(
        {"bold": True, "bg_color": "#E7E6E6", "border": 1, "align": "center", "valign": "vcenter", "text_wrap": True}
    )

    fmt_text = workbook.add_format({"border": 1, "align": "left", "valign": "vcenter"})
    fmt_center = workbook.add_format({"border": 1, "align": "center", "valign": "vcenter"})
    fmt_num = workbook.add_format({"border": 1, "align": "center", "valign": "vcenter", "num_format": "0.0000"})
    fmt_pct = workbook.add_format({"border": 1, "align": "center", "valign": "vcenter", "num_format": '0.00"%"'})

    fmt_pos_num = workbook.add_format({"font_color": "#C00000", "border": 1, "align": "center", "valign": "vcenter", "num_format": "0.0000"})
    fmt_neg_num = workbook.add_format({"font_color": "#0070C0", "border": 1, "align": "center", "valign": "vcenter", "num_format": "0.0000"})
    fmt_pos_pct = workbook.add_format({"font_color": "#C00000", "border": 1, "align": "center", "valign": "vcenter", "num_format": '0.00"%"'})
    fmt_neg_pct = workbook.add_format({"font_color": "#0070C0", "border": 1, "align": "center", "valign": "vcenter", "num_format": '0.00"%"'})

    fmt_avg_text = workbook.add_format({"bold": True, "bg_color": "#FFF2CC", "border": 1, "align": "left", "valign": "vcenter"})
    fmt_avg_center = workbook.add_format({"bold": True, "bg_color": "#FFF2CC", "border": 1, "align": "center", "valign": "vcenter"})
    fmt_avg_num = workbook.add_format({"bold": True, "bg_color": "#FFF2CC", "border": 1, "align": "center", "valign": "vcenter", "num_format": "0.0000"})
    fmt_avg_pct = workbook.add_format({"bold": True, "bg_color": "#FFF2CC", "border": 1, "align": "center", "valign": "vcenter", "num_format": '0.00"%"'})

    def write_num_cell(ws, r: int, c: int, value: Any, *, is_pct: bool = False, is_diff: bool = False, is_avg: bool = False) -> None:
        if value == "" or value is None:
            ws.write(r, c, "", fmt_avg_center if is_avg else fmt_center)
            return

        try:
            v = float(value)
        except Exception:
            ws.write(r, c, value, fmt_avg_center if is_avg else fmt_center)
            return

        if is_avg:
            ws.write_number(r, c, v, fmt_avg_pct if is_pct else fmt_avg_num)
            return

        if is_diff or is_pct:
            if v > 0:
                ws.write_number(r, c, v, fmt_pos_pct if is_pct else fmt_pos_num)
            elif v < 0:
                ws.write_number(r, c, v, fmt_neg_pct if is_pct else fmt_neg_num)
            else:
                ws.write_number(r, c, v, fmt_pct if is_pct else fmt_num)
        else:
            ws.write_number(r, c, v, fmt_num)

    def write_text_cell(ws, r: int, c: int, value: Any, *, is_avg: bool = False, center: bool = False) -> None:
        if center:
            ws.write(r, c, value, fmt_avg_center if is_avg else fmt_center)
        else:
            ws.write(r, c, value, fmt_avg_text if is_avg else fmt_text)

    # Sheet 1
    ws = workbook.add_worksheet("Scores by Difficulty")
    ws.write(0, 0, "Scores by difficulty, missing results counted as zero", fmt_title)

    left_cols = ["model", "mode", "level"]
    for col, name in enumerate(left_cols):
        ws.merge_range(1, col, 2, col, name, fmt_header_green)

    col = len(left_cols)

    for metric in ALL_METRICS:
        start = col
        end = col + 4
        ws.merge_range(1, start, 1, end, METRIC_DISPLAY_NAME.get(metric, metric), fmt_header_blue)

        subheaders = ["simple", "medium", "complex", "simple-complex", "simple-complex %"]
        for offset, h in enumerate(subheaders):
            ws.write(2, start + offset, h, fmt_subheader)

        col = end + 1

    for r_idx, row in enumerate(score_wide_rows, start=3):
        is_avg = str(row.get("source_model", "")).endswith("avg")

        write_text_cell(ws, r_idx, 0, row.get("model", ""), is_avg=is_avg)
        write_text_cell(ws, r_idx, 1, row.get("mode", ""), is_avg=is_avg, center=True)
        write_text_cell(ws, r_idx, 2, row.get("stage", ""), is_avg=is_avg, center=True)

        col = 3
        for metric in ALL_METRICS:
            write_num_cell(ws, r_idx, col, row.get(f"{metric}_simple", ""), is_avg=is_avg)
            write_num_cell(ws, r_idx, col + 1, row.get(f"{metric}_medium", ""), is_avg=is_avg)
            write_num_cell(ws, r_idx, col + 2, row.get(f"{metric}_complex", ""), is_avg=is_avg)
            write_num_cell(ws, r_idx, col + 3, row.get(f"{metric}_simple_minus_complex", ""), is_diff=True, is_avg=is_avg)
            write_num_cell(ws, r_idx, col + 4, row.get(f"{metric}_simple_to_complex_rel_diff_pct", ""), is_pct=True, is_diff=True, is_avg=is_avg)
            col += 5

    ws.freeze_panes(3, 0)
    ws.autofilter(2, 0, max(2, len(score_wide_rows) + 2), col - 1)
    ws.set_column(0, 0, 28)
    ws.set_column(1, 2, 10)
    ws.set_column(3, col - 1, 14)
    ws.set_row(0, 24)
    ws.set_row(1, 30)
    ws.set_row(2, 26)

    # Sheet 2
    ws2 = workbook.add_worksheet("VLM-LLM by Difficulty")
    ws2.write(0, 0, "VLM-LLM difference by difficulty, missing results counted as zero", fmt_title)

    left_cols_2 = ["model", "level"]
    for c, name in enumerate(left_cols_2):
        ws2.merge_range(1, c, 2, c, name, fmt_header_green)

    col = len(left_cols_2)

    for metric in ALL_METRICS:
        start = col
        end = col + 5
        ws2.merge_range(1, start, 1, end, METRIC_DISPLAY_NAME.get(metric, metric), fmt_header_blue)

        subheaders = ["simple", "simple %", "medium", "medium %", "complex", "complex %"]
        for offset, h in enumerate(subheaders):
            ws2.write(2, start + offset, h, fmt_subheader)

        col = end + 1

    for r_idx, row in enumerate(diff_wide_rows, start=3):
        is_avg = str(row.get("source_model", "")).endswith("avg")

        write_text_cell(ws2, r_idx, 0, row.get("model", ""), is_avg=is_avg)
        write_text_cell(ws2, r_idx, 1, row.get("stage", ""), is_avg=is_avg, center=True)

        col = 2
        for metric in ALL_METRICS:
            for difficulty in ["simple", "medium", "complex"]:
                write_num_cell(
                    ws2,
                    r_idx,
                    col,
                    row.get(f"{metric}_{difficulty}_abs_diff", ""),
                    is_diff=True,
                    is_avg=is_avg,
                )
                write_num_cell(
                    ws2,
                    r_idx,
                    col + 1,
                    row.get(f"{metric}_{difficulty}_rel_diff_pct", ""),
                    is_pct=True,
                    is_diff=True,
                    is_avg=is_avg,
                )
                col += 2

    ws2.freeze_panes(3, 0)
    ws2.autofilter(2, 0, max(2, len(diff_wide_rows) + 2), col - 1)
    ws2.set_column(0, 0, 28)
    ws2.set_column(1, 1, 10)
    ws2.set_column(2, col - 1, 14)
    ws2.set_row(0, 24)
    ws2.set_row(1, 30)
    ws2.set_row(2, 26)

    workbook.close()
    return True


def print_rankings(ranking_rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in ranking_rows:
        grouped[(str(row["model"]), str(row["stage"]))].append(row)

    print()
    print("=" * 80)
    print("Rankings by model × stage")
    print("=" * 80)

    for mode in MODE_ORDER:
        for stage in STAGE_ORDER:
            rows = grouped.get((mode, stage), [])
            if not rows:
                continue

            mean_overall = rows[0]["group_mean_overall_avg"]
            print(f"\n[{mode} / {stage}] mean overall_avg={mean_overall}")

            for row in rows:
                print(
                    f"  #{row['rank']:>2} "
                    f"{row['source_model']:<22} "
                    f"overall={row['overall_avg']} "
                    f"semantic={row['semantic_avg']} "
                    f"design={row['design_avg']} "
                    f"n={row['count']} "
                    f"valid={row['valid_count']} "
                    f"missing={row['missing_count']}"
                )


def print_overall_rankings(overall_rows: list[dict[str, Any]]) -> None:
    print()
    print("=" * 80)
    print("Overall model ranking")
    print("=" * 80)
    print("Macro-average over available mode × stage groups; missing results counted as zero.")

    for row in overall_rows:
        print(
            f"  #{int(row['rank']):>2} "
            f"{str(row['source_model']):<22} "
            f"overall={row['overall_avg']} "
            f"semantic={row['semantic_avg']} "
            f"design={row['design_avg']} "
            f"groups={row['intent_group_count']} "
            f"n={row['total_sample_count']} "
            f"valid={row['total_valid_count']} "
            f"missing={row['total_missing_count']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize semantic_design API results with missing JSON or invalid parse counted as zero."
    )

    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dataset-image-root", type=Path, default=DEFAULT_DATASET_IMAGE_ROOT)
    parser.add_argument("--difficulty-csv", type=Path, default=DEFAULT_DIFFICULTY_CSV)

    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--rank-out", type=Path, default=DEFAULT_RANK_OUT)
    parser.add_argument("--overall-rank-out", type=Path, default=DEFAULT_OVERALL_RANK_OUT)
    parser.add_argument("--diff-out", type=Path, default=DEFAULT_DIFF_OUT)
    parser.add_argument("--diff-xlsx-out", type=Path, default=DEFAULT_DIFF_XLSX_OUT)

    parser.add_argument("--difficulty-out", type=Path, default=DEFAULT_DIFFICULTY_OUT)
    parser.add_argument("--difficulty-diff-out", type=Path, default=DEFAULT_DIFFICULTY_DIFF_OUT)
    parser.add_argument("--difficulty-xlsx-out", type=Path, default=DEFAULT_DIFFICULTY_XLSX_OUT)

    parser.add_argument("--per-sample-out", type=Path, default=DEFAULT_PER_SAMPLE_OUT)

    parser.add_argument(
        "--only",
        nargs="*",
        default=[],
        help="Optional source model names to include.",
    )

    args = parser.parse_args()

    root = args.root.resolve()
    dataset_image_root = args.dataset_image_root.resolve()

    # If --root was overridden from its default, rebase all output paths
    # into the new root so that outputs stay with the data they summarize.
    default_root_resolved = DEFAULT_ROOT.resolve()
    if root != default_root_resolved:
        def _rebase(path: Path) -> Path:
            try:
                rel = path.resolve().relative_to(default_root_resolved)
                return root / rel
            except ValueError:
                return path.resolve()
        out_path = _rebase(args.out)
        rank_out_path = _rebase(args.rank_out)
        overall_rank_out_path = _rebase(args.overall_rank_out)
        diff_out_path = _rebase(args.diff_out)
        diff_xlsx_out_path = _rebase(args.diff_xlsx_out)
        difficulty_out_path = _rebase(args.difficulty_out)
        difficulty_diff_out_path = _rebase(args.difficulty_diff_out)
        difficulty_xlsx_out_path = _rebase(args.difficulty_xlsx_out)
        per_sample_out_path = _rebase(args.per_sample_out)
    else:
        out_path = args.out.resolve()
        rank_out_path = args.rank_out.resolve()
        overall_rank_out_path = args.overall_rank_out.resolve()
        diff_out_path = args.diff_out.resolve()
        diff_xlsx_out_path = args.diff_xlsx_out.resolve()
        difficulty_out_path = args.difficulty_out.resolve()
        difficulty_diff_out_path = args.difficulty_diff_out.resolve()
        difficulty_xlsx_out_path = args.difficulty_xlsx_out.resolve()
        per_sample_out_path = args.per_sample_out.resolve()

    selected_models = set(args.only)
    source_models = [m for m in MODEL_ORDER if not selected_models or m in selected_models]
    if root.exists():
        discovered_models = sorted(
            p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")
        )
        for source_model in discovered_models:
            if source_model not in source_models and (
                not selected_models or source_model in selected_models
            ):
                source_models.append(source_model)
    for source_model in sorted(selected_models):
        if source_model not in source_models:
            source_models.append(source_model)

    difficulty_by_key = load_difficulty_map(args.difficulty_csv.resolve())
    charts = collect_dataset_charts(dataset_image_root)

    groups: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "valid_count": 0,
            "missing_count": 0,
            "sum": {m: 0.0 for m in ALL_METRICS},
        }
    )

    per_sample_rows: list[dict[str, Any]] = []

    expected_total = 0
    valid_results = 0
    missing_or_invalid = 0
    missing_difficulty = 0

    for source_model in source_models:
        for chart in charts:
            category = chart["category"]
            chart_id = chart["chart_id"]
            chart_key = chart["chart_key"]

            difficulty = difficulty_by_key.get(chart_key, "unknown")
            if difficulty == "unknown":
                missing_difficulty += 1

            for mode in MODE_ORDER:
                for stage in STAGE_ORDER:
                    expected_total += 1

                    p = sd_path_for(root, source_model, category, chart_id, mode, stage)
                    scores, valid_result, reason = load_one_result_scores(p)

                    if valid_result:
                        valid_results += 1
                    else:
                        missing_or_invalid += 1

                    key = (source_model, mode, stage)
                    groups[key]["count"] += 1
                    if valid_result:
                        groups[key]["valid_count"] += 1
                    else:
                        groups[key]["missing_count"] += 1

                    for metric in ALL_METRICS:
                        groups[key]["sum"][metric] += scores[metric]

                    per_sample_rows.append(
                        {
                            "source_model": source_model,
                            "category": category,
                            "chart_id": chart_id,
                            "chart_key": chart_key,
                            "dataset_image": chart["dataset_image"],
                            "mode": mode,
                            "stage": stage,
                            "difficulty": difficulty,
                            "sd_path": str(p),
                            "valid_result": valid_result,
                            "missing_or_invalid_reason": reason,
                            **scores,
                        }
                    )

    summary_rows: list[dict[str, Any]] = []

    for (source_model, mode, layer), data in sorted(groups.items(), key=sort_key):
        count = int(data["count"])
        if count <= 0:
            continue

        avg_scores = {
            metric: data["sum"][metric] / count
            for metric in ALL_METRICS
        }

        summary_rows.append(
            {
                "source_model": source_model,
                "model": mode,
                "stage": layer,
                "count": count,
                "valid_count": int(data["valid_count"]),
                "missing_count": int(data["missing_count"]),
                "sf": round(avg_scores["sf"], 4),
                "sc": round(avg_scores["sc"], 4),
                "vc": round(avg_scores["vc"], 4),
                "org": round(avg_scores["org"], 4),
                "attn": round(avg_scores["attn"], 4),
                "semantic_avg": round(avg_scores["semantic_avg"], 4),
                "design_avg": round(avg_scores["design_avg"], 4),
                "overall_avg": round(avg_scores["overall_avg"], 4),
            }
        )

    summary_fieldnames = [
        "source_model",
        "model",
        "stage",
        "count",
        "valid_count",
        "missing_count",
        "sf",
        "sc",
        "vc",
        "org",
        "attn",
        "semantic_avg",
        "design_avg",
        "overall_avg",
    ]
    write_csv(out_path, summary_rows, summary_fieldnames)

    per_sample_fieldnames = [
        "source_model",
        "category",
        "chart_id",
        "chart_key",
        "dataset_image",
        "mode",
        "stage",
        "sf",
        "sc",
        "vc",
        "org",
        "attn",
        "semantic_avg",
        "design_avg",
        "overall_avg",
    ]
    write_csv(per_sample_out_path, per_sample_rows, per_sample_fieldnames)

    print("=" * 80)
    print("Semantic design summary with missing results counted as zero")
    print("=" * 80)
    print(f"Semantic root: {root}")
    print(f"Dataset image root: {dataset_image_root}")
    print(f"Dataset charts: {len(charts)}")
    print(f"Models: {len(source_models)}")
    print(f"Missing/invalid results counted as zero: {missing_or_invalid}")
    print()
    print(f"Saved per-sample: {per_sample_out_path}")
    print(f"Saved summary: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
