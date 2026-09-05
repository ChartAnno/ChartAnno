#!/usr/bin/env python3
"""Run the full low-level annotation evaluation from a standalone code repo.

The evaluator repository is intentionally separated from the data/project root.
This wrapper writes a temporary pipeline_config.json that points to the user
project, then invokes the pipeline runner.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


EVAL_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_SCRIPT = EVAL_ROOT / "annotation_eval" / "pipeline" / "run_all_metrics.py"


def _resolve(root: Path, value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else root / path)


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run low-level annotation evaluation and write a combined metric summary."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset-code-dir", default="dataset_code")
    parser.add_argument("--dataset-code-removed-dir", default="dataset_code_removed")
    parser.add_argument("--dataset-image-removed-dir", default="dataset_image_removed")
    parser.add_argument("--test-code-dir", default="test_code")
    parser.add_argument("--test-image-dir", default="test_code_image")
    parser.add_argument("--annotations-dir", default="outputs/annotations")
    parser.add_argument("--analysis-dir", default="outputs/analysis")
    parser.add_argument("--rendered-images-dir", default="outputs/rendered_images")
    parser.add_argument("--gt-annotations-dir", default="")
    parser.add_argument("--shared-text-extraction-dir", default="")
    parser.add_argument("--display-name", default="")
    parser.add_argument("--report-title", default="")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--skip-diff", action="store_true")
    parser.add_argument("--skip-gt-diff", action="store_true")
    parser.add_argument("--skip-model-diff", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--only-text-report", action="store_true")
    parser.add_argument("--diff-only", action="store_true")
    return parser


def main() -> int:
    args = build_cli().parse_args()
    project_root = args.project_root.resolve()

    config = {
        "display_name": args.display_name or project_root.name,
        "report_title": args.report_title or f"{project_root.name} evaluation",
        "dataset_code_dir": _resolve(project_root, args.dataset_code_dir),
        "dataset_code_removed_dir": _resolve(project_root, args.dataset_code_removed_dir),
        "dataset_image_removed_dir": _resolve(project_root, args.dataset_image_removed_dir),
        "test_code_dir": _resolve(project_root, args.test_code_dir),
        "test_image_dir": _resolve(project_root, args.test_image_dir),
        "annotations_dir": _resolve(project_root, args.annotations_dir),
        "rendered_images_dir": _resolve(project_root, args.rendered_images_dir),
        "analysis_dir": _resolve(project_root, args.analysis_dir),
        "gt_annotations_dir": _resolve(project_root, args.gt_annotations_dir)
        if args.gt_annotations_dir
        else "",
        "shared_text_extraction_dir": _resolve(project_root, args.shared_text_extraction_dir)
        if args.shared_text_extraction_dir
        else "",
        "execution_cwd": str(project_root),
    }

    with tempfile.TemporaryDirectory(prefix="annotation_eval_") as tmp:
        tmp_root = Path(tmp)
        (tmp_root / "pipeline_config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        cmd = [
            sys.executable,
            str(SUMMARY_SCRIPT),
            "--project-root",
            str(tmp_root),
        ]
        if not args.skip_diff:
            cmd.append("--run-diff")
        if args.skip_gt_diff:
            cmd.append("--skip-gt-diff")
        if args.skip_model_diff:
            cmd.append("--skip-model-diff")
        if args.skip_run:
            cmd.append("--skip-run")
        if args.skip_existing:
            cmd.append("--skip-existing")
        if args.only_text_report:
            cmd.append("--only-text-report")
        if args.diff_only:
            cmd.append("--diff-only")
        if args.output_csv is not None:
            output_csv = args.output_csv
            if not output_csv.is_absolute():
                output_csv = project_root / output_csv
            cmd.extend(["--output-csv", str(output_csv)])

        print("$", " ".join(cmd))
        return subprocess.run(cmd, cwd=str(project_root), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
