#!/usr/bin/env python3
"""Materialize evaluator input folders from the released JSONL files."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_text_if_changed(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = text.rstrip() + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == normalized:
        return
    path.write_text(normalized, encoding="utf-8")


def copy_if_changed(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return
    shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create dataset_code/eval image folders from JSONL.")
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--data-jsonl", default="data/input_code_image.jsonl")
    parser.add_argument("--dataset-code-dir", default="outputs/eval_assets/dataset_code")
    parser.add_argument("--dataset-code-removed-dir", default="outputs/eval_assets/dataset_code_removed")
    parser.add_argument("--dataset-image-dir", default="outputs/eval_assets/dataset_image_new")
    parser.add_argument("--dataset-image-removed-dir", default="outputs/eval_assets/dataset_image_removed")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    data_rows = read_jsonl(resolve(repo_root, args.data_jsonl))

    code_dir = resolve(repo_root, args.dataset_code_dir)
    code_removed_dir = resolve(repo_root, args.dataset_code_removed_dir)
    image_dir = resolve(repo_root, args.dataset_image_dir)
    image_removed_dir = resolve(repo_root, args.dataset_image_removed_dir)

    seen: set[tuple[str, str]] = set()
    for row in data_rows:
        category = str(row["category"])
        sample_id = str(row["sample_id"])
        key = (category, sample_id)
        if key in seen:
            continue
        seen.add(key)

        write_text_if_changed(code_dir / category / f"{sample_id}.py", str(row["GT code"]))
        write_text_if_changed(code_removed_dir / category / f"{sample_id}.py", str(row["GT w/o anno code"]))

        gt_chart = resolve(repo_root, str(row["GT chart"]))
        gt_wo_anno_chart = resolve(repo_root, str(row["GT w/o anno chart"]))
        copy_if_changed(gt_chart, image_dir / category / gt_chart.name)
        copy_if_changed(gt_wo_anno_chart, image_removed_dir / category / gt_wo_anno_chart.name)

    print(f"Materialized {len(seen)} samples under {repo_root / 'outputs/eval_assets'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
