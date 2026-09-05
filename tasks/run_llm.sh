#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

"${PYTHON_BIN}" "${ROOT}/tasks/evaluate_llm.py" \
  --repo-root "${ROOT}" \
  --model "${MODEL:-gpt-5.4}" \
  --base-url "${BASE_URL:-https://api.openai.com/v1}" \
  --api-key "${API_KEY:-}" \
  --api-key-env "${API_KEY_ENV:-OPENAI_API_KEY}" \
  --output-root "${OUTPUT_ROOT:-outputs/test_code}" \
  --max-data-rows "${MAX_DATA_ROWS:-0}" \
  --max-qps "${MAX_QPS:-1}" \
  --max-tokens "${MAX_TOKENS:-16384}" \
  --temperature "${TEMPERATURE:-0}" \
  --top-p "${TOP_P:-1}" \
  "$@"
