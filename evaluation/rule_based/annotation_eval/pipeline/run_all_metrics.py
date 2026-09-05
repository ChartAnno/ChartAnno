from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from annotation_eval.config import get_config_value, get_path

MODEL_NAMES = ("LLM", "VLM")
MODEL_DIR_ALIASES = {
    "code": "LLM",
    "code+image": "VLM",
}
STAGES = ("intent", "operation", "implementation")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _run(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _run_optional(cmd: list[str], cwd: Path, label: str) -> None:
    try:
        _run(cmd, cwd)
    except subprocess.CalledProcessError as exc:
        print(f"WARN optional step failed ({label}); continuing summary: {exc}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _dump_json(path: Path, data, encoder_cls) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, cls=encoder_cls)


def _merge_reused_text_extraction(reused_root: Path | None, out_root: Path) -> None:
    if reused_root is None or not reused_root.exists():
        return
    for tag in ("artist_gt", "artist_removed", "artist_removed_test"):
        src = reused_root / tag
        dst = out_root / tag
        if not src.exists() or dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            dst.symlink_to(src, target_is_directory=True)
        except OSError:
            shutil.copytree(src, dst)


def _collect_test_files(test_root: Path) -> dict[str, dict[str, list[Path]]]:
    model_files: dict[str, dict[str, list[Path]]] = {m: {} for m in MODEL_NAMES}
    for root, _, files in os.walk(test_root):
        root_name = os.path.basename(root)
        if root_name not in MODEL_DIR_ALIASES:
            continue

        rel = os.path.relpath(root, test_root)
        parts = rel.split(os.sep)
        if len(parts) < 3:
            continue

        category = parts[0]
        model = MODEL_DIR_ALIASES.get(parts[2], parts[2])

        for filename in files:
            if not filename.endswith(".py"):
                continue
            model_files[model].setdefault(category, []).append(Path(root) / filename)

    for model in MODEL_NAMES:
        for category in model_files[model]:
            model_files[model][category] = sorted(model_files[model][category])

    return model_files


def _infer_chart_id_from_test_stem(stem: str) -> str:
    parts = stem.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return stem


def _infer_stage_from_test_stem(stem: str) -> str | None:
    lowered = stem.lower()
    if lowered.endswith("_task"):
        return "intent"
    for stage in STAGES:
        if lowered.endswith(f"_{stage}"):
            return stage
    return None


def _expected_code_keys(project_root: Path) -> set[tuple[str, str, str, str]]:
    test_root = get_path(project_root, "test_code_dir", "test_code")
    keys: set[tuple[str, str, str, str]] = set()
    collected = _collect_test_files(test_root)
    for model, by_category in collected.items():
        for category, paths in by_category.items():
            for path in paths:
                stage = _infer_stage_from_test_stem(path.stem)
                if stage is None:
                    continue
                chart_id = _infer_chart_id_from_test_stem(path.stem)
                keys.add((model, category, chart_id, stage))
    return keys


def _expected_chart_keys(project_root: Path) -> set[tuple[str, str]]:
    return {(category, chart_id) for _, category, chart_id, _ in _expected_code_keys(project_root)}


def _rendered_image_exists(
    image_root: Path,
    category: str,
    chart_id: str,
    model: str,
    stage: str,
) -> bool:
    mode_dir = "code" if model == "LLM" else "code+image"
    file_tag = "code" if model == "LLM" else "code_image"
    stem = f"{chart_id}_{file_tag}_{stage}"
    base = image_root / category / chart_id / mode_dir / stem
    return any(base.with_suffix(ext).exists() for ext in IMG_EXTS)


def _expected_image_keys(
    project_root: Path,
    code_keys: set[tuple[str, str, str, str]],
) -> set[tuple[str, str, str, str]]:
    image_root = get_path(project_root, "test_image_dir", "test_code_image")
    if not image_root.exists():
        print(f"WARN image root missing, all metric scores become 0: {image_root}")
        return set()
    return {
        key
        for key in code_keys
        if _rendered_image_exists(image_root, key[1], key[2], key[0], key[3])
    }


def _run_gt_diff_no_render(
    project_root: Path,
    annotations_root: Path,
    skip_existing: bool = False,
    only_charts: set[tuple[str, str]] | None = None,
) -> None:
    from annotation_eval.extraction.diff_pipeline import run_diff_pipeline_for_code_pair
    from annotation_eval.extraction.runtime import NpEncoder, run_extraction_on_file
    from annotation_eval.extraction.subtraction import pre_dedupe_annotation_dict

    dataset_code_dir = get_path(project_root, "dataset_code_dir", "dataset_code")
    removed_code_dir = get_path(project_root, "dataset_code_removed_dir", "dataset_code_removed")

    artist_gt_root = annotations_root / "artist_gt"
    artist_removed_root = annotations_root / "artist_removed"
    final_gt_root = annotations_root / "final_gt_structured"
    raw_gt_root = annotations_root / "raw_gt"
    raw_removed_root = annotations_root / "raw_removed"
    raw_diff_root = annotations_root / "raw_gt_minus_removed"

    if not dataset_code_dir.exists() or not removed_code_dir.exists():
        raise FileNotFoundError(f"dataset_code_dir or dataset_code_removed_dir missing under {project_root}")

    for category_dir in sorted(dataset_code_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        removed_category_dir = removed_code_dir / category
        if not removed_category_dir.exists():
            continue

        for py_path in sorted(category_dir.glob("*.py")):
            stem = py_path.stem
            if only_charts is not None and (category, stem) not in only_charts:
                continue
            removed_py = removed_category_dir / f"{stem}.py"
            if not removed_py.exists():
                continue
            final_output = final_gt_root / category / stem / "GT" / f"{stem}.json"
            if final_output.exists():
                print(f"SKIP GT {category}/{stem}: final output exists")
                continue

            try:
                gt_artist = run_extraction_on_file(str(py_path), str(project_root), render_output_path=None)
                if gt_artist is not None:
                    _dump_json(
                        artist_gt_root / category / f"{stem}.json",
                        pre_dedupe_annotation_dict(gt_artist),
                        NpEncoder,
                    )

                removed_artist = run_extraction_on_file(str(removed_py), str(project_root), render_output_path=None)
                if removed_artist is not None:
                    _dump_json(
                        artist_removed_root / category / f"{stem}.json",
                        pre_dedupe_annotation_dict(removed_artist),
                        NpEncoder,
                    )

                run_diff_pipeline_for_code_pair(
                    source_file=str(py_path),
                    removed_file=str(removed_py),
                    project_root=str(project_root),
                    category=category,
                    filename=f"{stem}.json",
                    model_label="GT",
                    final_output_root=final_gt_root,
                    raw_source_output_root=raw_gt_root,
                    raw_removed_output_root=raw_removed_root,
                    raw_diff_output_root=raw_diff_root,
                    render_output_path=None,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"ERR  GT {category}/{stem}: {exc}")
                continue


def _run_model_diff_no_render(project_root: Path, annotations_root: Path, skip_existing: bool = False) -> None:
    from annotation_eval.extraction.diff_pipeline import run_diff_pipeline_for_code_pair

    test_root = get_path(project_root, "test_code_dir", "test_code")
    removed_root = get_path(project_root, "dataset_code_removed_dir", "dataset_code_removed")

    output_roots = {
        "LLM": annotations_root / "final_test_llm_minus_removed_structured",
        "VLM": annotations_root / "final_test_vlm_minus_removed_structured",
    }
    raw_source_roots = {
        "LLM": annotations_root / "raw_test_llm",
        "VLM": annotations_root / "raw_test_vlm",
    }
    raw_diff_roots = {
        "LLM": annotations_root / "raw_test_llm_minus_removed",
        "VLM": annotations_root / "raw_test_vlm_minus_removed",
    }
    raw_removed_root = annotations_root / "raw_removed_test"

    collected = _collect_test_files(test_root)
    for model_label in MODEL_NAMES:
        for category, files in sorted(collected[model_label].items()):
            for model_source in files:
                stem = model_source.stem
                filename = f"{stem}.json"
                chart_id = _infer_chart_id_from_test_stem(stem)
                removed_source = removed_root / category / f"{chart_id}.py"
                if not removed_source.exists():
                    continue
                final_output = output_roots[model_label] / category / chart_id / model_label / filename
                if skip_existing and final_output.exists():
                    print(f"SKIP {model_label} {category}/{stem}: final output exists")
                    continue

                try:
                    run_diff_pipeline_for_code_pair(
                        source_file=str(model_source),
                        removed_file=str(removed_source),
                        project_root=str(project_root),
                        category=category,
                        filename=filename,
                        model_label=model_label,
                        final_output_root=output_roots[model_label],
                        raw_source_output_root=raw_source_roots[model_label],
                        raw_removed_output_root=raw_removed_root,
                        raw_diff_output_root=raw_diff_roots[model_label],
                        raw_removed_filename=f"{chart_id}.json",
                        render_output_path=None,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"ERR  {model_label} {category}/{stem}: {exc}")
                    continue


def _empty_bucket() -> dict[str, object]:
    return {
        "count": 0,
        "success_count": 0,
        "missing_count": 0,
        "score": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }


def _append_metric_score(
    buckets: dict[tuple[str, str], dict[str, object]],
    model: str,
    stage: str,
    score: float,
    precision: float | None = None,
    recall: float | None = None,
    f1: float | None = None,
    success: bool = True,
) -> None:
    for key in ((model, "overall"), (model, stage)):
        bucket = buckets[key]
        bucket["count"] = int(bucket["count"]) + 1
        if success:
            bucket["success_count"] = int(bucket["success_count"]) + 1
        else:
            bucket["missing_count"] = int(bucket["missing_count"]) + 1
        bucket["score"].append(score)
        if precision is not None:
            bucket["precision"].append(precision)
        if recall is not None:
            bucket["recall"].append(recall)
        if f1 is not None:
            bucket["f1"].append(f1)


def _build_color_summary_rows(
    color_results_json: Path,
    expected_keys: set[tuple[str, str, str, str]],
    image_keys: set[tuple[str, str, str, str]],
) -> list[dict[str, object]]:
    data = _load_json(color_results_json)
    by_key: dict[tuple[str, str, str, str], dict[str, float]] = {}

    for item in data.get("files", []):
        model = item.get("model")
        category = item.get("category")
        chart_id = item.get("chart_id")
        stage = item.get("stage")
        overall = item.get("overall") or {}
        if not model or not category or not chart_id or not stage:
            continue
        by_key[(model, category, chart_id, stage)] = {
            "precision": float(overall.get("precision", 0.0)),
            "recall": float(overall.get("recall", 0.0)),
            "f1": float(overall.get("f1", 0.0)),
        }

    buckets: dict[tuple[str, str], dict[str, object]] = defaultdict(_empty_bucket)
    for model, category, chart_id, stage in sorted(expected_keys):
        if (model, category, chart_id, stage) not in image_keys:
            _append_metric_score(
                buckets,
                model,
                stage,
                score=0.0,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                success=False,
            )
            continue
        values = by_key.get((model, category, chart_id, stage))
        if values is None:
            _append_metric_score(
                buckets,
                model,
                stage,
                score=0.0,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                success=False,
            )
            continue
        _append_metric_score(
            buckets,
            model,
            stage,
            score=values["f1"],
            precision=values["precision"],
            recall=values["recall"],
            f1=values["f1"],
            success=True,
        )

    rows: list[dict[str, object]] = []
    for (model, stage), bucket in sorted(buckets.items()):
        rows.append(
            {
                "metric": "color_matching",
                "model": model,
                "stage": stage,
                "count": bucket["count"],
                "success_count": bucket["success_count"],
                "missing_count": bucket["missing_count"],
                "score": _mean(bucket["score"]),
                "precision": _mean(bucket["precision"]),
                "recall": _mean(bucket["recall"]),
                "f1": _mean(bucket["f1"]),
                "intersection": "",
                "union": "",
            }
        )
    return rows


def _build_annotation_matching_summary_rows(
    annotation_matching_json: Path,
    expected_keys: set[tuple[str, str, str, str]],
    image_keys: set[tuple[str, str, str, str]],
) -> list[dict[str, object]]:
    by_key: dict[tuple[str, str, str, str], float] = {}
    data = _load_json(annotation_matching_json)
    for model, m_data in data.get("results", {}).items():
        if model not in ("LLM", "VLM"):
            continue
        for item in m_data.get("files", []):
            category = item.get("category", "")
            chart_id = item.get("chart_id", "")
            if not chart_id:
                pred_file = Path(item.get("pred_file", ""))
                chart_id = _infer_chart_id_from_test_stem(pred_file.stem)
            stage = item.get("stage", "")
            overall = item.get("overall") or {}
            if not category or not chart_id or not stage:
                continue
            by_key[(model, category, chart_id, stage)] = float(overall.get("jaccard", 0.0))

    buckets: dict[tuple[str, str], dict[str, object]] = defaultdict(_empty_bucket)
    for model, category, chart_id, stage in sorted(expected_keys):
        if (model, category, chart_id, stage) not in image_keys:
            _append_metric_score(buckets, model, stage, score=0.0, success=False)
            continue
        score = by_key.get((model, category, chart_id, stage))
        if score is None:
            _append_metric_score(buckets, model, stage, score=0.0, success=False)
            continue
        _append_metric_score(buckets, model, stage, score=score, success=True)

    rows: list[dict[str, object]] = []
    for (model, stage), bucket in sorted(buckets.items()):
        rows.append(
            {
                "metric": "annotation_matching",
                "model": model,
                "stage": stage,
                "count": bucket["count"],
                "success_count": bucket["success_count"],
                "missing_count": bucket["missing_count"],
                "score": _mean(bucket["score"]),
                "precision": "",
                "recall": "",
                "f1": "",
                "intersection": "",
                "union": "",
            }
        )
    return rows


def _build_chart_fidelity_summary_rows(
    chart_fidelity_csv: Path,
    expected_keys: set[tuple[str, str, str, str]],
    image_keys: set[tuple[str, str, str, str]],
) -> list[dict[str, object]]:
    by_key: dict[tuple[str, str, str, str], tuple[float, bool]] = {}
    if chart_fidelity_csv.exists():
        for row in _read_csv(chart_fidelity_csv):
            model = row.get("model", "").upper()
            category = row.get("category", "")
            chart_id = row.get("chart_id", "")
            stage = row.get("layer", "").lower()
            if not model or not category or not chart_id or not stage:
                continue
            try:
                score = float(row.get("chart_fidelity") or row.get("cf", 0.0))
            except ValueError:
                score = 0.0
            parse_ok = str(row.get("parse_ok", "")).strip().lower() == "true"
            by_key[(model, category, chart_id, stage)] = (score, parse_ok)

    buckets: dict[tuple[str, str], dict[str, object]] = defaultdict(_empty_bucket)
    for model, category, chart_id, stage in sorted(expected_keys):
        if (model, category, chart_id, stage) not in image_keys:
            _append_metric_score(buckets, model, stage, score=0.0, success=False)
            continue
        value = by_key.get((model, category, chart_id, stage))
        if value is None:
            _append_metric_score(buckets, model, stage, score=0.0, success=False)
            continue
        score, parse_ok = value
        _append_metric_score(buckets, model, stage, score=score, success=parse_ok)

    rows: list[dict[str, object]] = []
    for (model, stage), bucket in sorted(buckets.items()):
        rows.append(
            {
                "metric": "chart_fidelity",
                "model": model,
                "stage": stage,
                "count": bucket["count"],
                "success_count": bucket["success_count"],
                "missing_count": bucket["missing_count"],
                "score": _mean(bucket["score"]),
                "precision": "",
                "recall": "",
                "f1": "",
                "intersection": "",
                "union": "",
            }
        )
    return rows


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Annotation Matching, Color Matching, and Chart Fidelity, "
            "then write one model/stage summary CSV."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project/model result root. Defaults to current working directory.",
    )
    parser.add_argument(
        "--run-diff",
        action="store_true",
        help="Run GT/model annotation diff first without bbox, stitched images, or rendered outputs.",
    )
    parser.add_argument(
        "--skip-gt-diff",
        action="store_true",
        help="When --run-diff is set, skip GT-vs-removed diff.",
    )
    parser.add_argument(
        "--skip-model-diff",
        action="store_true",
        help="When --run-diff is set, skip model-vs-removed diff.",
    )
    parser.add_argument(
        "--diff-only",
        action="store_true",
        help="Only run requested diff steps, then exit before metrics and reports.",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Only combine existing metric outputs; do not rerun metric scripts.",
    )
    parser.add_argument(
        "--only-text-report",
        action="store_true",
        help="Only run text extraction and text-relationship reports for prompt inputs, then exit before metric summary.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="When running diff/chart fidelity, skip per-sample outputs that already exist.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Combined summary CSV path. Default: <analysis_dir>/metric_summary/low_level_scores_by_model_stage.csv",
    )
    return parser


def main() -> None:
    args = build_cli().parse_args()
    project_root = args.project_root.resolve()
    annotations_root = get_path(project_root, "annotations_dir", "outputs/annotations")
    gt_annotations_value = get_config_value(project_root, "gt_annotations_dir", "")
    gt_annotations_root = Path(gt_annotations_value) if gt_annotations_value else annotations_root
    if not gt_annotations_root.is_absolute():
        gt_annotations_root = (project_root / gt_annotations_root).resolve()
    shared_text_extraction_value = get_config_value(project_root, "shared_text_extraction_dir", "")
    shared_text_extraction_root = Path(shared_text_extraction_value) if shared_text_extraction_value else None
    if shared_text_extraction_root is not None and not shared_text_extraction_root.is_absolute():
        shared_text_extraction_root = (project_root / shared_text_extraction_root).resolve()
    analysis_root = get_path(project_root, "analysis_dir", "outputs/analysis")
    execution_cwd = get_path(project_root, "execution_cwd", ".")

    intermediates_root = analysis_root / "intermediates"
    annotation_matching_dir = intermediates_root / "annotation_matching"
    color_dir = intermediates_root / "color_matching"
    chart_fidelity_dir = intermediates_root / "chart_fidelity"
    text_extraction_dir = intermediates_root / "text_extraction"
    text_relationship_dir = intermediates_root / "text_relationship"
    annotation_matching_summary_csv = annotation_matching_dir / "annotation_matching_summary.csv"
    annotation_matching_per_file_csv = annotation_matching_dir / "annotation_matching_per_file.csv"
    color_results_json = color_dir / "color_matching_results.json"
    chart_fidelity_csv = chart_fidelity_dir / "chart_fidelity_all.csv"
    output_csv = args.output_csv or (analysis_root / "metric_summary" / "low_level_scores_by_model_stage.csv")
    expected_keys = _expected_code_keys(project_root)
    expected_chart_keys = {(category, chart_id) for _, category, chart_id, _ in expected_keys}

    if args.run_diff:
        if not args.skip_gt_diff:
            _run_gt_diff_no_render(
                project_root,
                annotations_root,
                skip_existing=args.skip_existing,
                only_charts=expected_chart_keys,
            )
        if not args.skip_model_diff:
            _run_model_diff_no_render(project_root, annotations_root, skip_existing=args.skip_existing)

    if args.diff_only:
        return

    if args.only_text_report:
        _run_optional(
            [
                sys.executable,
                str(PROJECT_ROOT / "annotation_eval" / "metrics" / "text_extraction.py"),
                "--project-root",
                str(project_root),
                "--out-root",
                str(text_extraction_dir),
                "--only-tags",
                "artist_test_llm",
                "artist_test_vlm",
                "--ast-remove-removed-draw-calls",
            ],
            execution_cwd,
            "text_extraction",
        )
        _merge_reused_text_extraction(shared_text_extraction_root, text_extraction_dir)
        _run_optional(
            [
                sys.executable,
                str(PROJECT_ROOT / "annotation_eval" / "metrics" / "text_relationship.py"),
                "--project-root",
                str(project_root),
                "--run-all",
                "--output-dir",
                str(text_relationship_dir),
                "--text-extraction-root",
                str(text_extraction_dir),
            ],
            execution_cwd,
            "text_relationship",
        )
        return

    if not args.skip_run:
        annotation_matching_dir.mkdir(parents=True, exist_ok=True)
        _run(
            [
                sys.executable,
                str(PROJECT_ROOT / "annotation_eval" / "metrics" / "annotation_matching.py"),
                "--gt-root",
                str(gt_annotations_root / "final_gt_structured"),
                "--llm-root",
                str(annotations_root / "final_test_llm_minus_removed_structured"),
                "--vlm-root",
                str(annotations_root / "final_test_vlm_minus_removed_structured"),
                "--output-json",
                str(annotation_matching_dir / "annotation_matching_results.json"),
                "--summary-csv",
                str(annotation_matching_summary_csv),
                "--feature-csv",
                str(annotation_matching_dir / "annotation_matching_feature_summary.csv"),
                "--file-csv",
                str(annotation_matching_per_file_csv),
                "--per-chart-dir",
                str(annotation_matching_dir / "per_chart"),
            ],
            execution_cwd,
        )
        _merge_reused_text_extraction(shared_text_extraction_root, text_extraction_dir)
        _run(
            [
                sys.executable,
                str(PROJECT_ROOT / "annotation_eval" / "metrics" / "color_matching.py"),
                "--gt-root",
                str(gt_annotations_root / "final_gt_structured"),
                "--llm-root",
                str(annotations_root / "final_test_llm_minus_removed_structured"),
                "--vlm-root",
                str(annotations_root / "final_test_vlm_minus_removed_structured"),
                "--output-json",
                str(color_dir / "color_matching_results.json"),
                "--summary-csv",
                str(color_dir / "color_matching_summary.csv"),
                "--file-csv",
                str(color_dir / "color_matching_per_file.csv"),
                "--per-chart-dir",
                str(color_dir / "per_chart"),
            ],
            execution_cwd,
        )
        _run(
            [
                sys.executable,
                str(PROJECT_ROOT / "annotation_eval" / "metrics" / "chart_fidelity.py"),
                "--project-root",
                str(project_root),
                "--removed-root",
                str(get_path(project_root, "dataset_code_removed_dir", "dataset_code_removed")),
                "--test-root",
                str(get_path(project_root, "test_code_dir", "test_code")),
                "--out-root",
                str(chart_fidelity_dir),
                *(["--skip-existing"] if args.skip_existing else []),
            ],
            execution_cwd,
        )
        _run(
            [
                sys.executable,
                str(PROJECT_ROOT / "annotation_eval" / "metrics" / "text_extraction.py"),
                "--project-root",
                str(project_root),
                "--out-root",
                str(text_extraction_dir),
                "--only-tags",
                "artist_test_llm",
                "artist_test_vlm",
                "--ast-remove-removed-draw-calls",
            ],
            execution_cwd,
        )
        _run_optional(
            [
                sys.executable,
                str(PROJECT_ROOT / "annotation_eval" / "metrics" / "text_relationship.py"),
                "--project-root",
                str(project_root),
                "--run-all",
                "--output-dir",
                str(text_relationship_dir),
                "--text-extraction-root",
                str(text_extraction_dir),
            ],
            execution_cwd,
            "text_relationship",
        )

    image_keys = _expected_image_keys(project_root, expected_keys)
    print(
        "Summary scoring universe: "
        f"code={len(expected_keys)} image_exists={len(image_keys)} "
        f"image_missing={len(expected_keys) - len(image_keys)}"
    )

    combined_rows = _build_annotation_matching_summary_rows(annotation_matching_dir / "annotation_matching_results.json", expected_keys, image_keys)
    combined_rows.extend(_build_color_summary_rows(color_results_json, expected_keys, image_keys))
    combined_rows.extend(_build_chart_fidelity_summary_rows(chart_fidelity_csv, expected_keys, image_keys))

    _write_csv(
        output_csv,
        combined_rows,
        [
            "metric",
            "model",
            "stage",
            "count",
            "success_count",
            "missing_count",
            "score",
            "precision",
            "recall",
            "f1",
            "intersection",
            "union",
        ],
    )
    print(f"Saved summary: {output_csv}")
    
    # Generate low_level_scores_per_sample.csv
    per_sample_csv = output_csv.parent / "low_level_scores_per_sample.csv"
    per_sample_rows = []
    
    cf_data = {}
    if chart_fidelity_csv.exists():
        for row in _read_csv(chart_fidelity_csv):
            key = (row.get("model", "").upper(), row.get("category", ""), row.get("chart_id", ""), row.get("layer", "").lower())
            cf_data[key] = row.get("chart_fidelity", "")
            
    am_data = {}
    am_json = annotation_matching_dir / "annotation_matching_results.json"
    if am_json.exists():
        for model, m_data in _load_json(am_json).get("results", {}).items():
            if model not in ("LLM", "VLM"): continue
            for item in m_data.get("files", []):
                category = item.get("category", "")
                stage = item.get("stage", "")
                pred_file = Path(item.get("pred_file", ""))
                parts = pred_file.stem.split("_")
                if len(parts) >= 2:
                    chart_id = f"{parts[0]}_{parts[1]}"
                else:
                    chart_id = pred_file.stem
                am_data[(model, category, chart_id, stage)] = item.get("overall", {}).get("jaccard", 0.0)
                        
    cm_data = {}
    if color_results_json.exists():
        for item in _load_json(color_results_json).get("files", []):
            if item.get("model") in ("LLM", "VLM"):
                key = (item["model"], item["category"], item["chart_id"], item["stage"])
                cm_data[key] = item.get("overall", {}).get("f1", 0.0)

    for model, category, chart_id, stage in sorted(expected_keys):
        key = (model, category, chart_id, stage)
        img_exists = 1.0 if key in image_keys else 0.0
        row = {
            "model": model,
            "category": category,
            "chart_id": chart_id,
            "stage": stage,
            "execution_success_rate": img_exists,
            "chart_fidelity": cf_data.get(key, 0.0) if cf_data.get(key, "") != "" else 0.0,
            "annotation_matching": am_data.get(key, 0.0),
            "color_matching": cm_data.get(key, 0.0)
        }
        if stage == "intent":
            row["annotation_matching"] = ""
            row["color_matching"] = ""
        per_sample_rows.append(row)
        
    _write_csv(
        per_sample_csv,
        per_sample_rows,
        ["model", "category", "chart_id", "stage", "execution_success_rate", "chart_fidelity", "annotation_matching", "color_matching"]
    )
    print(f"Saved per-sample: {per_sample_csv}")

if __name__ == "__main__":
    main()
