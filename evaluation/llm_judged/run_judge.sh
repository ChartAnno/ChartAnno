#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_MODEL="${SOURCE_MODEL:-gpt54}"
PYTHON_BIN="${PYTHON:-python3}"

"${PYTHON_BIN}" "${ROOT}/evaluation/llm_judged/run_judge.py" \
  --repo-root "${ROOT}" \
  --manifest "${MANIFEST:-outputs/judge_prompts/${SOURCE_MODEL}/manifest.jsonl}" \
  --model "${MODEL:-gpt-5.4}" \
  --base-url "${BASE_URL:-https://api.openai.com/v1}" \
  --api-key "${API_KEY:-}" \
  --api-key-env "${API_KEY_ENV:-OPENAI_API_KEY}" \
  --max-data-rows "${MAX_DATA_ROWS:-0}" \
  --max-qps "${MAX_QPS:-1}" \
  "$@"
