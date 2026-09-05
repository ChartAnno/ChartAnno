#!/usr/bin/env python3
"""Generate semantic/design judge prompts from JSONL rows and text-relationship metrics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
LEVELS = ("intent", "operation", "implementation")


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def clean_template(text: str) -> str:
    text = text.strip()
    if text.startswith('"""') and text.endswith('"""'):
        text = text[3:-3].strip()
    return text


def relation_path(root: Path, category: str, sample_id: str, mode: str, level: str) -> Path:
    internal_dir = "LLM" if mode == "llm" else "VLM"
    public_tag = "code" if mode == "llm" else "code_image"
    candidates = [
        root / category / sample_id / internal_dir / f"{sample_id}_{public_tag}_{level}.json",
        root / category / sample_id / internal_dir / f"{sample_id}_{mode}_{level}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def relation_block(payload: dict[str, Any]) -> str:
    if not payload:
        return "\n".join(
            [
                "  - anno_text_n: 0",
                "  - anno_overlap_n: 0 (anno_overlap_pct=0%)",
                "  - overlap_severity_pct (%): mild=0, moderate=0, severe=0",
                "  - OverlapGlobal: 0%",
                "  - off_canvas_n: 0 (off_canvas_pct=0%)",
                "  - OutGlobal: 0%",
            ]
        )

    def first(*keys: str, default: Any = 0) -> Any:
        for key in keys:
            if key in payload:
                return payload[key]
        return default

    severity = first("overlap_severity_pct", "overlapping_added_text_severity_ratio_pct", default={})
    if not isinstance(severity, dict):
        severity = {}
    return "\n".join(
        [
            f"  - anno_text_n: {first('anno_text_n', 'added_text_count')}",
            (
                f"  - anno_overlap_n: {first('anno_overlap_n', 'overlapping_added_text_count')} "
                f"(anno_overlap_pct={first('anno_overlap_pct', 'overlapping_added_text_ratio_pct')}%)"
            ),
            (
                "  - overlap_severity_pct (%): "
                f"mild={severity.get('mild', severity.get('mild_overlap', 0))}, "
                f"moderate={severity.get('moderate', severity.get('moderate_overlap', 0))}, "
                f"severe={severity.get('severe', severity.get('severe_overlap', 0))}"
            ),
            f"  - OverlapGlobal: {first('overlap_global_pct')}%",
            (
                f"  - off_canvas_n: {first('off_canvas_n', 'out_of_canvas_added_text_count')} "
                f"(off_canvas_pct={first('off_canvas_pct', 'out_of_canvas_added_text_ratio_pct')}%)"
            ),
            f"  - OutGlobal: {first('out_global_pct')}%",
        ]
    )


def stage_constraints(level: str) -> str:
    if level == "intent":
        return (
            "- All metrics must be integers in [0, 5].\n"
            "- The GT image is provided as a reference only; exact alignment with GT is not required at intent level."
        )
    return (
        "- All metrics must be integers in [0, 5].\n"
        "- The GT image is provided as a reference for comparison.\n"
        "- For operation and implementation levels, substantial mismatch with GT under a metric should cap that metric at 3."
    )


def output_prompt_path(output_root: Path, source_model: str, category: str, sample_id: str, mode: str, level: str) -> Path:
    return output_root / source_model / category / sample_id / mode.upper() / f"{sample_id}_{mode}_{level}_prompt.txt"


def judge_output_path(output_root: Path, source_model: str, category: str, sample_id: str, mode: str, level: str) -> Path:
    return output_root / source_model / category / sample_id / mode.upper() / f"{sample_id}_{mode}_{level}_sd.json"


def image_path(image_root: Path, category: str, sample_id: str, mode: str, level: str) -> Path:
    mode_dir = "code" if mode == "llm" else "code+image"
    stem = f"{sample_id}_{'code' if mode == 'llm' else 'code_image'}_{level}"
    for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        candidate = image_root / category / sample_id / mode_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    legacy_stem = f"{sample_id}_{mode}_{level}"
    legacy_dir = mode.upper()
    for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        candidate = image_root / category / sample_id / legacy_dir / f"{legacy_stem}{ext}"
        if candidate.exists():
            return candidate
    return image_root / category / sample_id / mode_dir / f"{stem}.png"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate LLM-judge prompts.")
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--template-path", default="evaluation/llm_judged/templates/semantic_design.txt")
    parser.add_argument("--code-jsonl", default="data/input_code.jsonl")
    parser.add_argument("--code-image-jsonl", default="data/input_code_image.jsonl")
    parser.add_argument("--text-relation-root", default="outputs/analysis/text_relationship")
    parser.add_argument("--generated-image-root", default="outputs/test_images")
    parser.add_argument("--prompt-output-root", default="outputs/judge_prompts")
    parser.add_argument("--judge-output-root", default="outputs/api/semantic_design")
    parser.add_argument("--source-model", default="gpt54")
    parser.add_argument("--max-data-rows", type=int, default=0)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    template = clean_template(resolve(repo_root, args.template_path).read_text(encoding="utf-8"))
    rows_by_file = {
        "input_code.jsonl": read_jsonl(resolve(repo_root, args.code_jsonl)),
        "input_code_image.jsonl": read_jsonl(resolve(repo_root, args.code_image_jsonl)),
    }
    # Rebuild the metadata index from the inlined JSONL row fields
    # (category/sample_id/level/input_type used to live in data/metadata.jsonl).
    metadata = [
        {
            "jsonl": jsonl_name,
            "line_number": line_number,
            "category": row["category"],
            "sample_id": row["sample_id"],
            "level": row["level"],
            "input_type": row["input_type"],
        }
        for jsonl_name, rows in rows_by_file.items()
        for line_number, row in enumerate(rows, start=1)
    ]
    text_root = resolve(repo_root, args.text_relation_root)
    generated_image_root = resolve(repo_root, args.generated_image_root)
    prompt_root = resolve(repo_root, args.prompt_output_root)
    judge_root = resolve(repo_root, args.judge_output_root)
    manifest_rows: list[dict[str, Any]] = []
    mode_counts = {"llm": 0, "vlm": 0}

    for meta in metadata:
        mode = "llm" if meta["input_type"] == "Input: code" else "vlm"
        if args.max_data_rows and mode_counts[mode] >= args.max_data_rows:
            if all(v >= args.max_data_rows for v in mode_counts.values()):
                break
            continue
            
        jsonl_name = str(meta["jsonl"])
        line_number = int(meta["line_number"])
        row = rows_by_file[jsonl_name][line_number - 1]
        category = str(meta["category"])
        sample_id = str(meta["sample_id"])
        level = str(meta["level"])

        constraints = stage_constraints(level)
        rel_payload = read_json(relation_path(text_root, category, sample_id, mode, level))
        prompt = template.replace("{instruction}", compact_json(row["instruction"]))
        prompt = prompt.replace("{text_relation_results}", relation_block(rel_payload))
        prompt = prompt.replace("{stage_constraints}", constraints)

        prompt_path = output_prompt_path(prompt_root, args.source_model, category, sample_id, mode, level)
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt.strip() + "\n", encoding="utf-8")

        ai_image = image_path(generated_image_root, category, sample_id, mode, level)
        gt_image = resolve(repo_root, str(row["GT chart"]))
        out_path = judge_output_path(judge_root, args.source_model, category, sample_id, mode, level)
        manifest_rows.append(
            {
                "id": meta["id"],
                "source_model": args.source_model,
                "category": category,
                "chart_id": sample_id,
                "mode": mode.upper(),
                "stage": level,
                "prompt_path": prompt_path.relative_to(repo_root).as_posix(),
                "ai_image": ai_image.relative_to(repo_root).as_posix() if ai_image.exists() else "",
                "gt_image": gt_image.relative_to(repo_root).as_posix(),
                "output_path": out_path.relative_to(repo_root).as_posix(),
            }
        )
        mode_counts[mode] += 1
        
        if args.max_data_rows and all(v >= args.max_data_rows for v in mode_counts.values()):
            break

    manifest_path = prompt_root / args.source_model / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Generated {len(manifest_rows)} judge prompts")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
