#!/usr/bin/env python3
"""Small HTTP client for OpenAI-compatible /chat/completions API."""

from __future__ import annotations

import base64
import json
import mimetypes
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def image_to_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def post_json(url: str, api_key: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def call_model(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    timeout: float,
    max_tokens: int,
    temperature: float | None,
    top_p: float | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    return post_json(base_url.rstrip("/") + "/chat/completions", api_key, payload, timeout)


def format_http_error(exc: urllib.error.HTTPError) -> str:
    body = exc.read().decode("utf-8", errors="replace").strip()
    if body:
        try:
            body = json.dumps(json.loads(body), ensure_ascii=False)
        except json.JSONDecodeError:
            pass
        return f"HTTP {exc.code} {exc.reason}: {body}"
    return f"HTTP {exc.code} {exc.reason}"


def call_model_with_retries(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    timeout: float,
    max_tokens: int,
    temperature: float | None,
    top_p: float | None,
    max_retries: int,
    retry_sleep: float,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return call_model(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=messages,
                timeout=timeout,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        except urllib.error.HTTPError as exc:
            message = format_http_error(exc)
            if 400 <= exc.code < 500 and exc.code != 429:
                raise RuntimeError(message) from exc
            last_error = RuntimeError(message)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc

        if attempt < max_retries:
            time.sleep(min(retry_sleep * attempt, 30.0))

    raise RuntimeError(f"Model request failed after {max_retries} attempts: {last_error}")


def extract_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content", "") if isinstance(message, dict) else ""
        return content if isinstance(content, str) else ""
    return ""
