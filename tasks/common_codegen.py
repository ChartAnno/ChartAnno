#!/usr/bin/env python3
"""Shared OpenAI Chat Completions code generation runner for ChartAnno_Eval."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from utils.openai_compat import (  # noqa: E402
    call_model_with_retries,
    extract_text,
    image_to_data_url,
)

LEVELS = ("intent", "operation", "implementation")
MODE_TO_INPUT_TYPE = {
    "llm": "Input: code",
    "vlm": "Input: code+Image",
}
MODE_TO_OUTPUT_DIR = {
    "llm": "code",
    "vlm": "code+image",
}
MODE_TO_FILE_TAG = {
    "llm": "code",
    "vlm": "code_image",
}


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def row_meta(row: dict[str, Any]) -> dict[str, Any]:
    """Per-row metadata (category/sample_id/level), now inlined in each JSONL row."""
    return {
        "category": str(row["category"]),
        "sample_id": str(row["sample_id"]),
        "level": str(row["level"]),
    }


def parse_levels(value: str) -> set[str]:
    levels = {item.strip().lower() for item in value.split(",") if item.strip()}
    invalid = levels - set(LEVELS)
    if invalid:
        raise ValueError(f"Invalid levels: {sorted(invalid)}; expected one of {LEVELS}")
    return levels or set(LEVELS)


def slugify_model(model: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model.strip())
    return slug.strip("_") or "model"


def output_path_for(output_root: Path, mode: str, meta: dict[str, Any]) -> Path:
    sample_id = str(meta["sample_id"])
    level = str(meta["level"])
    return (
        output_root
        / str(meta["category"])
        / sample_id
        / MODE_TO_OUTPUT_DIR[mode]
        / f"{sample_id}_{MODE_TO_FILE_TAG[mode]}_{level}.py"
    )


def response_path_for(response_root: Path, mode: str, meta: dict[str, Any]) -> Path:
    sample_id = str(meta["sample_id"])
    level = str(meta["level"])
    return (
        response_root
        / str(meta["category"])
        / sample_id
        / MODE_TO_OUTPUT_DIR[mode]
        / f"{sample_id}_{MODE_TO_FILE_TAG[mode]}_{level}.json"
    )


def build_messages(row: dict[str, Any], mode: str, repo_root: Path) -> list[dict[str, Any]]:
    prompt = str(row["instruction"])
    if mode == "llm":
        return [{"role": "user", "content": prompt}]

    image_value = row.get("GT w/o anno chart")
    if not image_value:
        raise ValueError("VLM row is missing `GT w/o anno chart`.")
    image_path = resolve_path(repo_root, str(image_value))
    if not image_path.exists():
        raise FileNotFoundError(f"VLM image not found: {image_path}")
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
            ],
        }
    ]


def strip_code_fence(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip() + "\n"
    return text + ("\n" if text and not text.endswith("\n") else "")


def call_with_retries(args: argparse.Namespace, messages: list[dict[str, Any]]) -> dict[str, Any]:
    return call_model_with_retries(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        messages=messages,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
    )


def build_parser(mode: str) -> argparse.ArgumentParser:
    input_type = MODE_TO_INPUT_TYPE[mode]
    default_data = "data/input_code.jsonl" if mode == "llm" else "data/input_code_image.jsonl"
    parser = argparse.ArgumentParser(description=f"Run {input_type} code generation.")
    parser.add_argument("--repo-root", default=str(PACKAGE_ROOT))
    parser.add_argument("--data-jsonl", default=default_data)
    parser.add_argument("--output-root", default="outputs/test_code")
    parser.add_argument("--response-root", default="")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--levels", default="intent,operation,implementation")
    parser.add_argument("--categories", default="", help="Optional comma-separated category filter.")
    parser.add_argument("--max-data-rows", type=int, default=0)
    parser.add_argument("--start-row", type=int, default=1, help="1-based row offset after filtering.")
    parser.add_argument("--max-qps", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run(mode: str) -> int:
    args = build_parser(mode).parse_args()
    repo_root = Path(args.repo_root).resolve()
    data_path = resolve_path(repo_root, args.data_jsonl)
    output_root = resolve_path(repo_root, args.output_root)
    response_root = (
        resolve_path(repo_root, args.response_root)
        if args.response_root
        else repo_root / "outputs" / "api_responses" / slugify_model(args.model)
    )
    args.api_key = args.api_key or os.environ.get(args.api_key_env, "")

    rows = read_jsonl(data_path)
    levels = parse_levels(args.levels)
    categories = {item.strip() for item in args.categories.split(",") if item.strip()}
    metas = [row_meta(row) for row in rows]
    indexed = [
        (idx, row, meta)
        for idx, (row, meta) in enumerate(zip(rows, metas), start=1)
        if str(meta["level"]) in levels and (not categories or str(meta["category"]) in categories)
    ]
    if args.start_row > 1:
        indexed = indexed[args.start_row - 1 :]
    if args.max_data_rows > 0:
        indexed = indexed[: args.max_data_rows]

    print(f"Selected {len(indexed)} {MODE_TO_INPUT_TYPE[mode]} rows from {data_path}")
    if args.dry_run:
        if indexed:
            _, _, first_meta = indexed[0]
            print(f"First output: {output_path_for(output_root, mode, first_meta)}")
        return 0
    if not args.api_key:
        raise SystemExit(f"Missing API key. Pass --api-key or set {args.api_key_env}.")

    min_interval = 1.0 / args.max_qps if args.max_qps > 0 else 0.0
    last_call = 0.0
    written = 0
    skipped = 0
    failed = 0

    for _, row, meta in indexed:
        out_path = output_path_for(output_root, mode, meta)
        resp_path = response_path_for(response_root, mode, meta)
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        elapsed = time.monotonic() - last_call
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        try:
            messages = build_messages(row, mode, repo_root)
            response = call_with_retries(args, messages)
            last_call = time.monotonic()
            code = strip_code_fence(extract_text(response))
            if not code.strip():
                raise RuntimeError("empty model response")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(code, encoding="utf-8")
            resp_path.parent.mkdir(parents=True, exist_ok=True)
            resp_path.write_text(json.dumps(response, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            written += 1
            print(f"Wrote {out_path}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR {meta.get('id')}: {exc}")

    print(f"Done. written={written}, skipped={skipped}, failed={failed}")
    return 1 if failed else 0
