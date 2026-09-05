#!/usr/bin/env python3
"""Run the ChartAnno generation and evaluation pipeline."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ALL_STEPS = (
    "prepare",
    "code",
    "code-image",
    "render",
    "rule",
    "judge-prompts",
    "judge",
    "summarize-judge",
    "summarize-current",
)


def add_if(cmd: list[str], flag: str, value: object | None) -> None:
    if value not in (None, "", 0):
        cmd.extend([flag, str(value)])


def add_bool(cmd: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        cmd.append(flag)


def parse_steps(value: str) -> list[str]:
    aliases = {"llm": "code", "vlm": "code-image", "code+image": "code-image"}
    raw = [aliases.get(item.strip(), item.strip()) for item in value.split(",") if item.strip()]
    if not raw or raw == ["all"]:
        return list(ALL_STEPS)
    valid_steps = set(ALL_STEPS) | {"summarize-model-results"}
    invalid = [item for item in raw if item not in valid_steps]
    if invalid:
        raise ValueError(f"Invalid step(s): {invalid}. Valid steps: {tuple(valid_steps)}")
    return raw


def run_cmd(cmd: list[str], cwd: Path, dry_run: bool) -> None:
    print("$", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def generation_cmd(script: Path, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(script),
        "--repo-root",
        str(args.repo_root),
        "--model",
        args.model,
        "--base-url",
        args.base_url,
        "--api-key-env",
        args.api_key_env,
        "--levels",
        args.levels,
        "--max-qps",
        str(args.max_qps),
        "--timeout",
        str(args.timeout),
        "--max-retries",
        str(args.max_retries),
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
    ]
    add_if(cmd, "--api-key", args.api_key)
    add_if(cmd, "--categories", args.categories)
    add_if(cmd, "--max-data-rows", args.max_data_rows)
    add_bool(cmd, "--overwrite", args.overwrite)
    add_bool(cmd, "--dry-run", args.api_dry_run)
    return cmd


def build_command(step: str, args: argparse.Namespace) -> list[str]:
    root = args.repo_root
    if step == "prepare":
        return [sys.executable, str(root / "scripts" / "materialize_eval_assets.py"), "--repo-root", str(root)]
    if step == "code":
        cmd = generation_cmd(root / "tasks" / "evaluate_llm.py", args)
        cmd.extend(["--output-root", f"outputs/runs/{args.source_model}/test_code"])
        return cmd
    if step == "code-image":
        cmd = generation_cmd(root / "tasks" / "evaluate_vlm.py", args)
        cmd.extend(["--output-root", f"outputs/runs/{args.source_model}/test_code"])
        return cmd
    if step == "render":
        cmd = [
            sys.executable,
            str(root / "tasks" / "render_generated_code.py"),
            "--repo-root",
            str(root),
            "--code-root",
            f"outputs/runs/{args.source_model}/test_code",
            "--image-root",
            f"outputs/runs/{args.source_model}/test_images",
            "--timeout",
            str(args.render_timeout),
        ]
        add_if(cmd, "--max-files", args.max_render_files)
        add_bool(cmd, "--overwrite", args.overwrite)
        return cmd
    if step == "rule":
        cmd = [
            sys.executable,
            str(root / "evaluation" / "rule_based" / "scripts" / "run_evaluation_all.py"),
            "--project-root",
            str(root),
            "--dataset-code-dir",
            "outputs/eval_assets/dataset_code",
            "--dataset-code-removed-dir",
            "outputs/eval_assets/dataset_code_removed",
            "--dataset-image-removed-dir",
            "outputs/eval_assets/dataset_image_removed",
            "--test-code-dir",
            f"outputs/runs/{args.source_model}/test_code",
            "--test-image-dir",
            f"outputs/runs/{args.source_model}/test_images",
            "--annotations-dir",
            f"outputs/runs/{args.source_model}/annotations",
            "--analysis-dir",
            f"outputs/runs/{args.source_model}/analysis",
            "--rendered-images-dir",
            f"outputs/runs/{args.source_model}/rendered_images",
            "--output-csv",
            f"outputs/runs/{args.source_model}/analysis/metric_summary/low_level_scores_by_model_stage.csv",
        ]
        add_bool(cmd, "--skip-diff", args.skip_rule_diff)
        add_bool(cmd, "--skip-existing", args.skip_existing)
        return cmd
    if step == "judge-prompts":
        cmd = [
            sys.executable,
            str(root / "evaluation" / "llm_judged" / "generate_prompts.py"),
            "--repo-root",
            str(root),
            "--source-model",
            args.source_model,
            "--generated-image-root",
            f"outputs/runs/{args.source_model}/test_images",
            "--prompt-output-root",
            f"outputs/runs/{args.source_model}/judge_prompts",
            "--judge-output-root",
            f"outputs/runs/{args.source_model}/api/semantic_design",
        ]
        add_if(cmd, "--max-data-rows", args.max_data_rows)
        return cmd
    if step == "judge":
        cmd = [
            sys.executable,
            str(root / "evaluation" / "llm_judged" / "run_judge.py"),
            "--repo-root",
            str(root),
            "--manifest",
            f"outputs/runs/{args.source_model}/judge_prompts/{args.source_model}/manifest.jsonl",
            "--model",
            args.judge_model or args.model,
            "--base-url",
            args.judge_base_url or args.base_url,
            "--api-key-env",
            args.api_key_env,
            "--max-qps",
            str(args.max_qps),
            "--timeout",
            str(args.timeout),
            "--max-retries",
            str(args.max_retries),
        ]
        add_if(cmd, "--api-key", args.api_key)
        add_if(cmd, "--max-data-rows", args.max_data_rows)
        add_bool(cmd, "--overwrite", args.overwrite)
        add_bool(cmd, "--dry-run", args.api_dry_run)
        return cmd
    if step == "summarize-judge":
        return [
            sys.executable,
            str(root / "scripts" / "summarize_semantic_design.py"),
            "--root",
            f"outputs/runs/{args.source_model}/api/semantic_design",
            "--dataset-image-root",
            str(root / "outputs" / "eval_assets" / "dataset_image_new"),
            "--only",
            args.source_model,
        ]
    if step == "summarize-current":
        return [
            sys.executable,
            str(root / "scripts" / "summarize_current_model.py"),
            "--repo-root",
            str(root),
            "--source-model",
            args.source_model,
            "--run-dir",
            f"outputs/runs/{args.source_model}",
        ]
    if step == "summarize-model-results":
        return [
            sys.executable,
            str(root / "scripts" / "summarize_model_results.py"),
            "--results-dir",
            str(root / "results" / "per_model_combined_csv"),
            "--output-dir",
            str(root / "results" / "summary"),
            "--total-charts",
            str(args.total_charts),
        ]
    raise AssertionError(step)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ChartAnno pipeline steps.")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--steps", default="all", help=f"Comma-separated steps. all={','.join(ALL_STEPS)}")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--source-model", default="gpt54", help="Folder/name used for this model in outputs.")
    parser.add_argument("--judge-model", default="")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--judge-base-url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--total-charts", type=int, default=1200, help="Total charts for averaging denominator.")
    parser.add_argument("--levels", default="intent,operation,implementation")
    parser.add_argument("--categories", default="")
    parser.add_argument("--max-data-rows", type=int, default=0)
    parser.add_argument("--max-qps", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--render-timeout", type=float, default=60.0)
    parser.add_argument("--max-render-files", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-rule-diff", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--api-dry-run", action="store_true", help="Run API scripts in dry-run mode.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    args = parser.parse_args()

    args.repo_root = args.repo_root.resolve()
    steps = parse_steps(args.steps)
    if args.api_key:
        os.environ[args.api_key_env] = args.api_key

    for step in steps:
        print(f"\n== {step} ==")
        run_cmd(build_command(step, args), args.repo_root, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
