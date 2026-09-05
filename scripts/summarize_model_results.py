#!/usr/bin/env python3
"""Summarize bundled per-model CSV results and compute all-model averages."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOTAL_CHARTS = 1200  # total number of charts in the dataset
STAGE_ALIAS = {"task": "intent", "op": "operation", "imp": "implementation"}
METRIC_FIELDS = [
    "execution_success_rate",
    "chart_fidelity",
    "annotation_matching",
    "color_matching",
    "semantic_faithfulness",
    "semantic_clarity",
    "visual_clarity",
    "annotation_organization_quality",
    "attention_guidance",
    "structural_compliance",
    "semantic_consistency",
    "design_effectiveness",
]

SOURCE_ALIASES = {
    "execution_success_rate": ("execution_success_rate", "execution_rate"),
    "chart_fidelity": ("chart_fidelity", "chart_faithfulness"),
    "annotation_matching": ("annotation_matching", "anno_matching"),
    "color_matching": ("color_matching",),
    "semantic_faithfulness": ("semantic_faithfulness",),
    "semantic_clarity": ("semantic_clarity",),
    "visual_clarity": ("visual_clarity",),
    "annotation_organization_quality": ("annotation_organization_quality", "organization"),
    "attention_guidance": ("attention_guidance",),
    "structural_compliance": ("structural_compliance", "low_level_avg"),
    "semantic_consistency": ("semantic_consistency",),
    "design_effectiveness": ("design_effectiveness", "design_space_compatibility"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
        return 0.0 if text == "" else float(text)
    except Exception:
        return 0.0


def norm_stage(value: Any) -> str:
    stage = str(value or "").strip().lower()
    return STAGE_ALIAS.get(stage, stage)


def avg(values: list[float], denominator: int = 0) -> float | str:
    """Average over a fixed denominator. Missing samples count as 0."""
    n = denominator if denominator > 0 else len(values)
    return round(sum(values) / n, 3) if n > 0 else ""


def row_value(row: dict[str, str], field: str) -> float | None:
    for key in SOURCE_ALIASES[field]:
        if key in row:
            value = to_float(row[key])
            if value is not None:
                return value
    return 0.0


def safe_model_name(path: Path) -> str:
    return path.stem.replace("_", " ")


def summarize(files: list[Path], total_charts: int = TOTAL_CHARTS) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: {metric: [] for metric in METRIC_FIELDS}
    )
    counts: dict[tuple[str, str, str], int] = defaultdict(int)

    for path in files:
        model_name = safe_model_name(path)
        for row in read_csv(path):
            mode = str(row.get("mode", "")).strip().upper()
            level = norm_stage(row.get("level"))
            if mode not in {"LLM", "VLM"} or level not in {"intent", "operation", "implementation"}:
                continue
            
            # HOTFIX: Force structural_compliance to chart_fidelity for intent
            if level == "intent":
                row["structural_compliance"] = row.get("chart_fidelity", row.get("chart_faithfulness"))

            key = (model_name, mode, level)
            counts[key] += 1
            for metric in METRIC_FIELDS:
                value = row_value(row, metric)
                if value is not None:
                    grouped[key][metric].append(value)

    stage_rows: list[dict[str, Any]] = []
    for (model_name, mode, level), values in sorted(grouped.items()):
        input_val = "Code" if mode == "LLM" else "Code+Image" if mode == "VLM" else mode
        row = {"model": model_name, "input": input_val, "mode": mode, "level": level, "count": counts[(model_name, mode, level)]}
        for metric in METRIC_FIELDS:
            row[metric] = avg(values[metric], total_charts)
        stage_rows.append(row)

    overall_groups: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {metric: [] for metric in METRIC_FIELDS}
    )
    overall_counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in stage_rows:
        for key in [(row["model"], row["mode"]), (row["model"], "ALL"), ("ALL_MODELS_AVG", row["mode"]), ("ALL_MODELS_AVG", "ALL")]:
            overall_counts[key] += int(row["count"])
            for metric in METRIC_FIELDS:
                value = to_float(row.get(metric))
                if value is not None:
                    overall_groups[key][metric].append(value)

    overall_rows: list[dict[str, Any]] = []
    for (model_name, mode), values in sorted(overall_groups.items()):
        input_val = "Code" if mode == "LLM" else "Code+Image" if mode == "VLM" else mode
        row = {"model": model_name, "input": input_val, "mode": mode, "level": "ALL", "count": overall_counts[(model_name, mode)]}
        for metric in METRIC_FIELDS:
            row[metric] = avg(values[metric])
        overall_rows.append(row)

    return stage_rows, overall_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize model per-model CSV files.")
    parser.add_argument("--results-dir", default=str(ROOT / "results" / "per_model_combined_csv"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "summary")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "results" / "summary" / "model_results_table.csv")
    parser.add_argument("--only", nargs="*", default=[], help="Optional CSV stem/model filters.")
    parser.add_argument("--total-charts", type=int, default=TOTAL_CHARTS,
                        help="Total number of charts in the dataset (denominator for averaging).")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    files = sorted(results_dir.glob("*.csv"))
    if args.only:
        wanted = {item.lower().replace(" ", "_") for item in args.only}
        files = [p for p in files if p.stem.lower() in wanted or safe_model_name(p).lower().replace(" ", "_") in wanted]
    if not files:
        raise SystemExit(f"No per-model CSV files found under {results_dir}")

    stage_rows, overall_rows = summarize(files, args.total_charts)
    output_dir = Path(args.output_dir).resolve()
    fields = ["model", "input", "level", "count", *METRIC_FIELDS]
    combined_rows = stage_rows + overall_rows
    final_out = Path(args.output_csv).resolve()
    write_csv(final_out, combined_rows, fields)
    print(f"Saved: {final_out}")

    stage_out = output_dir / "model_results_stage_summary.csv"
    overall_out = output_dir / "model_results_overall_summary.csv"
    write_csv(stage_out, stage_rows, fields)
    write_csv(overall_out, overall_rows, fields)
    print(f"Saved: {stage_out}")
    print(f"Saved: {overall_out}")
    
    fields_4 = [
        "model",
        "input",
        "level",
        "count",
        "execution_success_rate",
        "structural_compliance",
        "semantic_consistency",
        "design_effectiveness",
    ]
    out_4 = final_out.parent / "model_results_table_4_metrics.csv"
    write_csv(out_4, combined_rows, fields_4)
    print(f"Saved: {out_4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
