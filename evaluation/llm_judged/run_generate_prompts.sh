#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

"${PYTHON_BIN}" "${ROOT}/evaluation/llm_judged/generate_prompts.py" \
  --repo-root "${ROOT}" \
  --source-model "${SOURCE_MODEL:-gpt54}" \
  --max-data-rows "${MAX_DATA_ROWS:-0}" \
  "$@"
