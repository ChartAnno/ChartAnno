"""Shared helpers for code-to-code raw diff pipelines."""

from __future__ import annotations

import json
from pathlib import Path

from annotation_eval.extraction.diff_extractor import extract_diffed_annotation_bundle
from annotation_eval.extraction.runtime import NpEncoder
from annotation_eval.extraction.raw_diff import raw_records_to_jsonable, to_jsonable_raw_diff
from annotation_eval.extraction.subtraction import build_structured_output_path, pre_dedupe_annotation_dict


def dump_json(path: str | Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, cls=NpEncoder)


def write_diff_bundle_outputs(
    bundle,
    *,
    category: str,
    filename: str,
    model_label: str,
    final_output_root: str | Path,
    raw_source_output_root: str | Path,
    raw_removed_output_root: str | Path,
    raw_diff_output_root: str | Path,
    raw_removed_filename: str | None = None,
) -> str:
    result = pre_dedupe_annotation_dict(bundle["diffed_semantic"])
    out_path = build_structured_output_path(
        str(final_output_root),
        category,
        filename,
        model_label,
    )

    dump_json(out_path, result)
    dump_json(
        Path(raw_source_output_root) / category / filename,
        raw_records_to_jsonable(bundle["gt_raw"]),
    )
    dump_json(
        Path(raw_removed_output_root) / category / (raw_removed_filename or filename),
        raw_records_to_jsonable(bundle["removed_raw"]),
    )
    dump_json(
        Path(raw_diff_output_root) / category / filename,
        to_jsonable_raw_diff(bundle["raw_diff"]),
    )
    return out_path


def run_diff_pipeline_for_code_pair(
    *,
    source_file: str,
    removed_file: str,
    project_root: str,
    category: str,
    filename: str,
    model_label: str,
    final_output_root: str | Path,
    raw_source_output_root: str | Path,
    raw_removed_output_root: str | Path,
    raw_diff_output_root: str | Path,
    raw_removed_filename: str | None = None,
    render_output_path: str | None = None,
    removed_reference_file: str | None = None,
    ast_strip_grid_calls: bool = False,
    ast_remove_removed_draw_calls: bool = False,
):
    bundle = extract_diffed_annotation_bundle(
        source_file,
        removed_file,
        project_root=project_root,
        render_output_path=render_output_path,
        removed_reference_file=removed_reference_file,
        ast_strip_grid_calls=ast_strip_grid_calls,
        ast_remove_removed_draw_calls=ast_remove_removed_draw_calls,
    )
    if bundle is None:
        return None

    out_path = write_diff_bundle_outputs(
        bundle,
        category=category,
        filename=filename,
        model_label=model_label,
        final_output_root=final_output_root,
        raw_source_output_root=raw_source_output_root,
        raw_removed_output_root=raw_removed_output_root,
        raw_diff_output_root=raw_diff_output_root,
        raw_removed_filename=raw_removed_filename,
    )
    return {"bundle": bundle, "output_path": out_path}
