#!/usr/bin/env python3
"""Run an OpenAI Chat Completions multimodal judge over generated prompt manifests."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


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


def image_to_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def build_messages(prompt: str, ai_image: Path | None, gt_image: Path | None) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if ai_image is not None and ai_image.exists():
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(ai_image)}})
    if gt_image is not None and gt_image.exists():
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(gt_image)}})
    return [{"role": "user", "content": content}]


def post_completion(args: argparse.Namespace, messages: list[dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "max_completion_tokens": args.max_tokens,
    }
    if args.temperature is not None:
        payload["temperature"] = args.temperature
    if args.top_p is not None:
        payload["top_p"] = args.top_p
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {args.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def format_http_error(exc: urllib.error.HTTPError) -> str:
    body = exc.read().decode("utf-8", errors="replace").strip()
    if body:
        try:
            parsed = json.loads(body)
            body = json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            pass
        return f"HTTP {exc.code} {exc.reason}: {body}"
    return f"HTTP {exc.code} {exc.reason}"


def response_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content", "") if isinstance(message, dict) else ""
    return content if isinstance(content, str) else ""


def parse_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None


def normalize_result(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None
    rows = result.get("results")
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
        return result
    row = dict(rows[0])
    if "org" not in row and "comp" in row:
        row["org"] = row["comp"]
    normalized = dict(result)
    normalized["results"] = [row]
    return normalized


def call_with_retries(args: argparse.Namespace, messages: list[dict[str, Any]]) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, args.max_retries + 1):
        try:
            return post_completion(args, messages)
        except urllib.error.HTTPError as exc:
            message = format_http_error(exc)
            if 400 <= exc.code < 500 and exc.code != 429:
                raise RuntimeError(message) from exc
            last_error = RuntimeError(message)
            if attempt == args.max_retries:
                break
            time.sleep(min(args.retry_sleep * attempt, 30.0))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == args.max_retries:
                break
            time.sleep(min(args.retry_sleep * attempt, 30.0))
    raise RuntimeError(f"Judge request failed after {args.max_retries} attempts: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run semantic/design LLM judge.")
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--max-data-rows", type=int, default=0)
    parser.add_argument("--max-qps", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    args.api_key = args.api_key or os.environ.get(args.api_key_env, "")
    rows = read_jsonl(resolve(repo_root, args.manifest))
    if args.max_data_rows > 0:
        mode_counts = {}
        filtered_rows = []
        for r in rows:
            m = r.get("mode", "").upper()
            if mode_counts.get(m, 0) < args.max_data_rows:
                filtered_rows.append(r)
                mode_counts[m] = mode_counts.get(m, 0) + 1
        rows = filtered_rows
    print(f"Selected {len(rows)} judge rows")
    if args.dry_run:
        if rows:
            print(f"First output: {resolve(repo_root, rows[0]['output_path'])}")
        return 0
    if not args.api_key:
        raise SystemExit(f"Missing API key. Pass --api-key or set {args.api_key_env}.")

    min_interval = 1.0 / args.max_qps if args.max_qps > 0 else 0.0
    last_call = 0.0
    written = skipped = failed = 0
    for row in rows:
        output_path = resolve(repo_root, str(row["output_path"]))
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        elapsed = time.monotonic() - last_call
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        try:
            prompt = resolve(repo_root, str(row["prompt_path"])).read_text(encoding="utf-8")
            ai_image = resolve(repo_root, row["ai_image"]) if row.get("ai_image") else None
            gt_image = resolve(repo_root, row["gt_image"]) if row.get("gt_image") else None
            response = call_with_retries(args, build_messages(prompt, ai_image, gt_image))
            last_call = time.monotonic()
            parsed = normalize_result(parse_json_object(response_text(response)))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(
                    {
                        "parse_ok": parsed is not None,
                        "source_model": row.get("source_model"),
                        "category": row.get("category"),
                        "chart_id": row.get("chart_id"),
                        "mode": row.get("mode"),
                        "stage": row.get("stage"),
                        "result": parsed,
                        "raw_response": response,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            written += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR {row.get('id')}: {exc}")

    print(f"Done. written={written}, skipped={skipped}, failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
