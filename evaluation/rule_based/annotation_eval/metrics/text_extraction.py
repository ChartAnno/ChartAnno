"""Extract all visible text (title/legend/axis/ticks/annotations) to sidecar JSON files.

This script is isolated from annotation extraction outputs.
It writes to outputs/analysis/text_extraction by default.
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from annotation_eval.config import get_path
from annotation_eval.extraction.ast_call_filter import apply_ast_filters

REPO_ROOT = PROJECT_ROOT
OUTPUT_ROOT = REPO_ROOT / "outputs"
DEFAULT_OUT_ROOT = OUTPUT_ROOT / "analysis" / "text_extraction"
LEGACY_ROOT = os.environ.get("ANNOTATION_EVAL_LEGACY_ROOT", "")


def _round_color(color):
    if color is None:
        return "none"
    try:
        rgba = mcolors.to_rgba(color)
        return tuple(round(float(c), 3) for c in rgba)
    except Exception:
        return str(color)


def _canvas_size(fig, renderer):
    try:
        w = float(getattr(renderer, "width", 0.0))
        h = float(getattr(renderer, "height", 0.0))
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    try:
        w, h = fig.canvas.get_width_height()
        return float(max(w, 1)), float(max(h, 1))
    except Exception:
        w, h = fig.get_size_inches() * fig.dpi
        return float(max(w, 1)), float(max(h, 1))


def _bbox_to_norm(bbox, w_px, h_px):
    try:
        x0, y0, bw, bh = bbox.bounds
        return [
            round(float(x0) / w_px, 6),
            round(float(y0) / h_px, 6),
            round(float(bw) / w_px, 6),
            round(float(bh) / h_px, 6),
        ]
    except Exception:
        return None


def _point_bbox(t, w_px, h_px):
    try:
        x, y = t.get_position()
        disp = t.get_transform().transform((float(x), float(y)))
        return [
            round(float(disp[0]) / w_px, 6),
            round(float(disp[1]) / h_px, 6),
            0.0,
            0.0,
        ]
    except Exception:
        return None


def _text_only_bbox(t, renderer, w_px, h_px):
    try:
        if isinstance(t, mpl.text.Annotation):
            bbox_patch = t.get_bbox_patch()
            if bbox_patch is not None:
                patch_bbox = bbox_patch.get_window_extent(renderer)
                bbox_n = _bbox_to_norm(patch_bbox, w_px, h_px)
                if bbox_n is not None:
                    return bbox_n

            text_bbox = mpl.text.Text.get_window_extent(t, renderer)
            bbox_n = _bbox_to_norm(text_bbox, w_px, h_px)
            if bbox_n is not None:
                return bbox_n

        bbox = t.get_window_extent(renderer)
        return _bbox_to_norm(bbox, w_px, h_px)
    except Exception:
        return None


def _iter_axis_tick_text_artists(axis):
    try:
        lo, hi = axis.get_view_interval()
        lo, hi = sorted((float(lo), float(hi)))
    except Exception:
        lo = hi = None

    def _in_view(loc):
        if lo is None or hi is None:
            return True
        try:
            value = float(loc)
        except Exception:
            return True
        eps = max(abs(hi - lo) * 1e-9, 1e-12)
        return (lo - eps) <= value <= (hi + eps)

    ticks = []
    try:
        ticks.extend(axis.get_major_ticks())
    except Exception:
        pass
    try:
        ticks.extend(axis.get_minor_ticks())
    except Exception:
        pass

    for tick in ticks:
        try:
            if not _in_view(tick.get_loc()):
                continue
        except Exception:
            pass
        for label in (getattr(tick, "label1", None), getattr(tick, "label2", None)):
            if label is None:
                continue
            try:
                if not label.get_visible():
                    continue
            except Exception:
                pass
            yield label


def _iter_ax_text_artists(ax):
    for artist in ax.texts:
        yield artist

    yield ax.title
    yield getattr(ax, "_left_title", None)
    yield getattr(ax, "_right_title", None)
    yield ax.xaxis.label
    yield ax.yaxis.label

    for artist in _iter_axis_tick_text_artists(ax.xaxis):
        yield artist
    for artist in _iter_axis_tick_text_artists(ax.yaxis):
        yield artist

    legend = ax.get_legend()
    if legend is not None:
        yield legend.get_title()
        for artist in legend.get_texts():
            yield artist


def _collect_axes(fig):
    axes = []
    seen = set()

    def visit(ax):
        ax_id = id(ax)
        if ax_id in seen:
            return
        seen.add(ax_id)
        axes.append(ax)
        for child_ax in getattr(ax, "child_axes", []) or []:
            visit(child_ax)

    for ax in fig.axes:
        visit(ax)
    return axes


def _collect_all_text(fig):
    try:
        fig.canvas.draw()
    except Exception:
        pass

    try:
        renderer = fig.canvas.get_renderer()
    except Exception:
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()

    w_px, h_px = _canvas_size(fig, renderer)
    seen = set()
    out = []
    ax_texts = set()

    def _append_text(t):
        if t is None:
            return
        tid = id(t)
        if tid in seen:
            return
        seen.add(tid)
        try:
            content = t.get_text()
        except Exception:
            return
        if not (isinstance(content, str) and content.strip()):
            return

        bbox_n = _text_only_bbox(t, renderer, w_px, h_px)
        if bbox_n is None:
            bbox_n = _point_bbox(t, w_px, h_px)

        out.append(
            {
                "content": content,
                "bbox": bbox_n,
                "color": _round_color(getattr(t, "get_color", lambda: None)()),
            }
        )

    for ax in _collect_axes(fig):
        for t in _iter_ax_text_artists(ax):
            if t is not None:
                ax_texts.add(t)
            _append_text(t)

    for t in fig.texts:
        if t in ax_texts:
            continue
        _append_text(t)
    _append_text(getattr(fig, "_suptitle", None))

    return out


def _exec_source_get_figure(file_path: str, source_code: str):
    module_name = f"mod_all_text_{os.path.basename(file_path)}"
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = file_path
    module.__dict__["__name__"] = "__main__"

    with patch("matplotlib.pyplot.show"), patch(
        "matplotlib.pyplot.close", lambda *a, **k: None
    ), patch("matplotlib.pyplot.savefig", lambda *a, **k: None), patch(
        "matplotlib.figure.Figure.savefig", lambda *a, **k: None
    ):
        exec(compile(source_code, file_path, "exec"), module.__dict__)

    if len(plt.get_fignums()) == 0:
        return None
    return plt.gcf()


def _extract_one_file(
    file_path: Path,
    *,
    project_root: Path,
    ast_strip_grid_calls: bool,
    ast_remove_removed_draw_calls: bool,
    removed_reference_file: Optional[Path],
):
    src = file_path.read_text(encoding="utf-8")
    if LEGACY_ROOT and LEGACY_ROOT in src:
        src = src.replace(LEGACY_ROOT, str(project_root))

    use_src = src
    if ast_strip_grid_calls or ast_remove_removed_draw_calls:
        removed_src = None
        if ast_remove_removed_draw_calls and removed_reference_file and removed_reference_file.exists():
            removed_src = removed_reference_file.read_text(encoding="utf-8")
            if LEGACY_ROOT and LEGACY_ROOT in removed_src:
                removed_src = removed_src.replace(LEGACY_ROOT, str(project_root))
        use_src = apply_ast_filters(
            use_src,
            removed_source=removed_src,
            strip_grid_calls=ast_strip_grid_calls,
            remove_removed_drawing_calls=ast_remove_removed_draw_calls,
        )

    real_close = plt.close
    try:
        mpl.rcdefaults()
        real_close("all")
        fig = _exec_source_get_figure(str(file_path), use_src)
        if fig is None:
            return None
        return _collect_all_text(fig)
    except Exception:
        return None
    finally:
        real_close("all")


def _write_sidecar(out_path: Path, all_text):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"all_text": all_text or []}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _extract_group(
    *,
    source_tag: str,
    source_root: Path,
    out_root: Path,
    project_root: Path,
    ast_strip_grid_calls: bool,
    ast_remove_removed_draw_calls: bool = False,
    removed_reference_root: Optional[Path] = None,
):
    if not source_root.exists():
        print(f"[skip] {source_tag}: source not found -> {source_root}")
        return 0, 0

    total = 0
    ok = 0
    for py_path in sorted(source_root.rglob("*.py")):
        rel = py_path.relative_to(source_root)
        out_rel = rel.with_suffix(".json")
        removed_ref = None
        if ast_remove_removed_draw_calls and removed_reference_root is not None:
            # test_code/<Cat>/<Chart>/<Model>/<file>.py -> removed/<Cat>/<Chart>.py
            if len(rel.parts) >= 4:
                cat = rel.parts[0]
                stem = py_path.stem
                parts = stem.split("_")
                if len(parts) >= 2:
                    chart_id = f"{parts[0]}_{parts[1]}"
                    removed_ref = removed_reference_root / cat / f"{chart_id}.py"

        total += 1
        all_text = _extract_one_file(
            py_path,
            project_root=project_root,
            ast_strip_grid_calls=ast_strip_grid_calls,
            ast_remove_removed_draw_calls=ast_remove_removed_draw_calls,
            removed_reference_file=removed_ref,
        )
        if all_text is None:
            continue
        _write_sidecar(out_root / source_tag / out_rel, all_text)
        ok += 1

    print(f"[{source_tag}] {ok}/{total} -> {out_root / source_tag}")
    return ok, total


def build_cli():
    p = argparse.ArgumentParser(description="Extract sidecar text JSON files for text-relationship prompt inputs.")
    p.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT), help="Sidecar output root.")
    p.add_argument("--project-root", default=str(REPO_ROOT), help="Project root for path rewrite.")
    p.add_argument(
        "--ast-strip-grid-calls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Strip grid-related calls before execution.",
    )
    p.add_argument(
        "--ast-remove-removed-draw-calls",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For test_code extraction: remove draw calls shared with removed script.",
    )
    p.add_argument(
        "--run-all",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run all default groups.",
    )
    p.add_argument(
        "--only-tags",
        nargs="*",
        default=[],
        help=(
            "Optional source tags to run, e.g. artist_gt artist_removed "
            "artist_removed_test artist_test_llm artist_test_vlm."
        ),
    )
    return p


def _default_groups(project_root: Path):
    test_root = get_path(project_root, "test_code_dir", "test_code")
    dataset_code_root = get_path(project_root, "dataset_code_dir", "dataset_code")
    removed_code_root = get_path(project_root, "dataset_code_removed_dir", "dataset_code_removed")
    return [
        ("artist_gt", dataset_code_root, False),
        ("artist_removed", removed_code_root, False),
        ("artist_removed_test", removed_code_root, False),
        ("artist_test_llm", test_root, True),
        ("artist_test_vlm", test_root, True),
    ]


def main():
    args = build_cli().parse_args()
    project_root = Path(args.project_root).resolve()
    out_root = Path(args.out_root).resolve()

    if args.run_all:
        groups = _default_groups(project_root)
    else:
        groups = []
    if args.only_tags:
        wanted = set(args.only_tags)
        groups = [group for group in groups if group[0] in wanted]

    if not groups:
        print("No groups selected. Use --run-all.")
        return

    total_ok = 0
    total_n = 0
    for tag, src_root, is_test in groups:
        if is_test:
            # test_code has nested model folders; filter by tag model.
            allowed_dirs = {"code"} if tag.endswith("llm") else {"code+image"}
            tmp_root = src_root
            # We still traverse all .py and keep only matching model level.
            ok = 0
            n = 0
            for py_path in sorted(tmp_root.rglob("*.py")):
                rel = py_path.relative_to(tmp_root)
                if len(rel.parts) < 4 or rel.parts[2] not in allowed_dirs:
                    continue
                n += 1
                removed_ref = None
                if args.ast_remove_removed_draw_calls:
                    cat = rel.parts[0]
                    stem = py_path.stem
                    parts = stem.split("_")
                    if len(parts) >= 2:
                        chart_id = f"{parts[0]}_{parts[1]}"
                        removed_ref = get_path(project_root, "dataset_code_removed_dir", "dataset_code_removed") / cat / f"{chart_id}.py"

                all_text = _extract_one_file(
                    py_path,
                    project_root=project_root,
                    ast_strip_grid_calls=args.ast_strip_grid_calls,
                    ast_remove_removed_draw_calls=args.ast_remove_removed_draw_calls,
                    removed_reference_file=removed_ref,
                )
                if all_text is None:
                    continue
                out_rel = rel.with_suffix(".json")
                _write_sidecar(out_root / tag / out_rel.parts[0] / out_rel.parts[-1], all_text)
                ok += 1
            print(f"[{tag}] {ok}/{n} -> {out_root / tag}")
            total_ok += ok
            total_n += n
            continue

        ok, n = _extract_group(
            source_tag=tag,
            source_root=src_root,
            out_root=out_root,
            project_root=project_root,
            ast_strip_grid_calls=args.ast_strip_grid_calls,
        )
        total_ok += ok
        total_n += n

    print(f"Done. text extraction sidecar: {total_ok}/{total_n} -> {out_root}")


if __name__ == "__main__":
    main()
