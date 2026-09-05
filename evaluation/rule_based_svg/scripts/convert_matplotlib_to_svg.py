#!/usr/bin/env python3
"""Render selected Matplotlib candidate/removed pairs as standalone SVG files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = next(
    parent
    for parent in SCRIPT_PATH.parents
    if (parent / "outputs" / "eval_assets").is_dir()
)
MPL_EVAL_ROOT = REPO_ROOT / "evaluation" / "rule_based"
if str(MPL_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(MPL_EVAL_ROOT))

from annotation_eval.extraction.runtime import load_chart_figures  # noqa: E402


DEFAULT_SAMPLES = (
    "Area/Area_1.py",
    "Area/Area_2.py",
    "Area/Area_3.py",
    "Line/Line_1.py",
    "Line/Line_2.py",
    "Line/Line_3.py",
    "Bar/Bar_1.py",
    "Bar/Bar_2.py",
    "Bar/Bar_3.py",
    "Bar/Bar_4.py",
)


def _convert(source: Path, output: Path) -> None:
    fig_extract, fig_render = load_chart_figures(str(source), str(REPO_ROOT))
    fig = fig_render or fig_extract
    if fig is None:
        raise RuntimeError(f"Matplotlib execution produced no figure: {source}")
    try:
        # Preserve SVG text as <text> nodes instead of converting glyphs to paths.
        mpl.rcParams["svg.fonttype"] = "none"
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, format="svg")
    finally:
        plt.close("all")


def main() -> int:
    assets = REPO_ROOT / "outputs" / "eval_assets"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", default=str(assets / "dataset_code"))
    parser.add_argument("--baseline-root", default=str(assets / "dataset_code_removed"))
    parser.add_argument("--candidate-output", default=str(assets / "dataset_code_MPL_SVG"))
    parser.add_argument("--baseline-output", default=str(assets / "dataset_removed_MPL_SVG"))
    parser.add_argument("--sample", action="append", dest="samples", help="Relative .py path; may be repeated")
    args = parser.parse_args()

    candidate_root = Path(args.candidate_root).resolve()
    baseline_root = Path(args.baseline_root).resolve()
    candidate_output = Path(args.candidate_output).resolve()
    baseline_output = Path(args.baseline_output).resolve()
    samples = tuple(args.samples or DEFAULT_SAMPLES)

    for relative_string in samples:
        relative = Path(relative_string)
        candidate_source = candidate_root / relative
        baseline_source = baseline_root / relative
        if not candidate_source.is_file() or not baseline_source.is_file():
            raise FileNotFoundError(f"Missing Matplotlib pair for {relative}")
        svg_relative = relative.with_suffix(".svg")
        candidate_svg = candidate_output / svg_relative
        baseline_svg = baseline_output / svg_relative
        _convert(candidate_source, candidate_svg)
        _convert(baseline_source, baseline_svg)
        print(f"{relative}: {candidate_svg} | {baseline_svg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
