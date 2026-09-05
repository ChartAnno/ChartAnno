#!/usr/bin/env python3
"""Render generated Matplotlib code files into images for rule-based evaluation."""

from __future__ import annotations

import argparse
import os
import runpy
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMG_SUFFIX = ".png"
MODE_DIRS = {"code", "code+image"}


def infer_target(code_root: Path, image_root: Path, path: Path) -> Path | None:
    rel = path.relative_to(code_root)
    parts = rel.parts
    if len(parts) != 4:
        return None
    category, sample_id, mode, filename = parts
    if mode not in MODE_DIRS:
        return None
    return image_root / category / sample_id / mode / Path(filename).with_suffix(IMG_SUFFIX).name


def render_one(code_path: Path, output_path: Path, cwd: Path) -> int:
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    saved = {"done": False}
    real_show = plt.show

    def save_show(*args, **kwargs):  # noqa: ANN001
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig = plt.gcf()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        saved["done"] = True
        return real_show(*args, **kwargs) if False else None

    old_cwd = Path.cwd()
    try:
        os.chdir(cwd)
        plt.close("all")
        plt.show = save_show
        runpy.run_path(str(code_path), run_name="__main__")
        if not saved["done"] and plt.get_fignums():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.gcf().savefig(output_path, dpi=150, bbox_inches="tight")
            saved["done"] = True
    finally:
        plt.show = real_show
        plt.close("all")
        os.chdir(old_cwd)
    return 0 if saved["done"] and output_path.exists() else 1


def parent_main() -> int:
    parser = argparse.ArgumentParser(description="Render generated code tree to PNG images.")
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--code-root", default="outputs/test_code")
    parser.add_argument("--image-root", default="outputs/test_images")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--_render-one", action="store_true")
    parser.add_argument("--_code-path", default="")
    parser.add_argument("--_output-path", default="")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    if args._render_one:
        return render_one(Path(args._code_path).resolve(), Path(args._output_path).resolve(), repo_root)

    code_root = (repo_root / args.code_root).resolve()
    image_root = (repo_root / args.image_root).resolve()
    files = sorted(path for path in code_root.rglob("*.py") if path.is_file())
    if args.max_files > 0:
        files = files[: args.max_files]

    rendered = skipped = failed = 0
    for code_path in files:
        target = infer_target(code_root, image_root, code_path)
        if target is None:
            continue
        if target.exists() and not args.overwrite:
            skipped += 1
            continue
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--repo-root",
            str(repo_root),
            "--_render-one",
            "--_code-path",
            str(code_path),
            "--_output-path",
            str(target),
        ]
        try:
            result = subprocess.run(cmd, cwd=str(repo_root), timeout=args.timeout, check=False)
            if result.returncode == 0:
                rendered += 1
            else:
                failed += 1
                print(f"FAILED render: {code_path}")
        except subprocess.TimeoutExpired:
            failed += 1
            print(f"TIMEOUT render: {code_path}")

    print(f"Render complete. rendered={rendered}, skipped={skipped}, failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(parent_main())
