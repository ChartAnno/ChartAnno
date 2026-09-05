"""Shared runtime helpers for executing chart scripts and extracting annotations."""

import datetime
import importlib.util
import json
import os
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import matplotlib as mpl
import matplotlib.pyplot as plt

from annotation_eval.extraction.ast_call_filter import apply_ast_filters
from annotation_eval.extraction.element_extractor import ChartAnnotationExtractor

LEGACY_ROOT = os.environ.get("ANNOTATION_EVAL_LEGACY_ROOT", "")


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        return super().default(obj)


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


def _capture_axes_state(fig):
    state = []
    for ax in _collect_axes(fig):
        try:
            state.append(
                {
                    "pos": ax.get_position().bounds,
                    "xlim": ax.get_xlim(),
                    "ylim": ax.get_ylim(),
                    "xscale": ax.get_xscale(),
                    "yscale": ax.get_yscale(),
                }
            )
        except Exception:
            state.append(None)
    return state


def _apply_axes_state(fig, state):
    if not state:
        return
    for idx, ax in enumerate(_collect_axes(fig)):
        if idx >= len(state):
            break
        st = state[idx]
        if not st:
            continue
        try:
            ax.set_position(st["pos"])
        except Exception:
            pass
        try:
            ax.set_xscale(st["xscale"])
            ax.set_yscale(st["yscale"])
        except Exception:
            pass
        try:
            ax.set_xlim(st["xlim"])
            ax.set_ylim(st["ylim"])
        except Exception:
            pass
    try:
        fig.canvas.draw()
    except Exception:
        pass


def _read_source(file_path: str, project_root: str):
    with open(file_path, "r", encoding="utf-8") as f:
        original_src = f.read()

    if LEGACY_ROOT and LEGACY_ROOT in original_src:
        original_src = original_src.replace(LEGACY_ROOT, project_root)
    return original_src


def _build_extraction_source(
    original_src: str,
    *,
    project_root: str,
    removed_reference_file: Optional[str] = None,
    ast_strip_grid_calls: bool = False,
    ast_remove_removed_draw_calls: bool = False,
):
    extraction_src = original_src
    if not (ast_strip_grid_calls or ast_remove_removed_draw_calls):
        return extraction_src

    removed_source = None
    if ast_remove_removed_draw_calls and removed_reference_file:
        try:
            with open(removed_reference_file, "r", encoding="utf-8") as f:
                removed_source = f.read()
            if LEGACY_ROOT and LEGACY_ROOT in removed_source:
                removed_source = removed_source.replace(LEGACY_ROOT, project_root)
        except Exception:
            removed_source = None

    return apply_ast_filters(
        extraction_src,
        removed_source=removed_source,
        strip_grid_calls=ast_strip_grid_calls,
        remove_removed_drawing_calls=ast_remove_removed_draw_calls,
    )


def _exec_source_get_figure(file_path: str, source_code: str):
    module_name = f"mod_{os.path.basename(file_path)}"
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = file_path
    # Many dataset/test scripts only build the figure inside a __main__ guard.
    # Run them with a script-like execution context so extraction sees the figure.
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


def load_chart_figures(
    file_path: str,
    project_root: str,
    *,
    removed_reference_file: Optional[str] = None,
    ast_strip_grid_calls: bool = False,
    ast_remove_removed_draw_calls: bool = False,
):
    real_close = plt.close
    try:
        original_src = _read_source(file_path, project_root)
        extraction_src = _build_extraction_source(
            original_src,
            project_root=project_root,
            removed_reference_file=removed_reference_file,
            ast_strip_grid_calls=ast_strip_grid_calls,
            ast_remove_removed_draw_calls=ast_remove_removed_draw_calls,
        )

        mpl.rcdefaults()
        real_close("all")

        needs_ast = extraction_src != original_src
        fig_render = None
        axes_state = None

        if needs_ast:
            fig_render = _exec_source_get_figure(file_path, original_src)
            if fig_render is None:
                return None, None
            axes_state = _capture_axes_state(fig_render)

        if needs_ast:
            real_close("all")
            mpl.rcdefaults()
            fig_extract = _exec_source_get_figure(file_path, extraction_src)
            if fig_extract is None:
                return None, None
            _apply_axes_state(fig_extract, axes_state)
        else:
            fig_extract = _exec_source_get_figure(file_path, original_src)
            if fig_extract is None:
                return None, None
            fig_render = fig_extract

        return fig_extract, fig_render
    except Exception as e:
        print(f"\n[Error] Extract failed: {file_path}: {e}")
        real_close("all")
        return None, None


def save_rendered_figure(fig, render_output_path: Optional[str]):
    if not render_output_path or fig is None:
        return
    try:
        render_path = Path(render_output_path)
        render_path.parent.mkdir(parents=True, exist_ok=True)
        fig.canvas.draw()
        fig.canvas.print_png(str(render_path))
    except Exception as e:
        print(f"[Warn] save render failed: {render_output_path}: {e}")


def run_extraction_on_file(
    file_path: str,
    project_root: str,
    render_output_path: Optional[str] = None,
    removed_reference_file: Optional[str] = None,
    ast_strip_grid_calls: bool = False,
    ast_remove_removed_draw_calls: bool = False,
):
    try:
        fig_extract, fig_render = load_chart_figures(
            file_path,
            project_root,
            removed_reference_file=removed_reference_file,
            ast_strip_grid_calls=ast_strip_grid_calls,
            ast_remove_removed_draw_calls=ast_remove_removed_draw_calls,
        )
        if fig_extract is None:
            return None

        extractor = ChartAnnotationExtractor(fig_extract)
        result = extractor.extract()
        save_rendered_figure(fig_render, render_output_path)
        return result
    finally:
        plt.close("all")


__all__ = [
    "NpEncoder",
    "load_chart_figures",
    "run_extraction_on_file",
    "save_rendered_figure",
]
