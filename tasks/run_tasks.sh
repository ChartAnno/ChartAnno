#!/usr/bin/env bash
# run_tasks.sh — Unified LLM / VLM code generation runner
#
# Usage:
#   MODE=code       ./tasks/run_tasks.sh              # code input
#   MODE=code-image ./tasks/run_tasks.sh              # code+image input
#   MODE=both ./tasks/run_tasks.sh                    # run both sequentially
#
# Key environment variables (all optional):
#   MODE              code | code-image | both  (default: both)
#   MODEL             model API name    (default: gpt-5.4)
#   BASE_URL          API base URL      (default: https://api.openai.com/v1)
#   API_KEY           API key string    (takes precedence over API_KEY_ENV)
#   API_KEY_ENV       env-var name for the API key (default: OPENAI_API_KEY)
#   OUTPUT_ROOT       output dir        (default: outputs/test_code)
#   MAX_DATA_ROWS     0 = all rows      (default: 0)
#   MAX_QPS           requests/second   (default: 1)
#   LEVELS            comma-separated intent,operation,implementation (default: all)
#   CATEGORIES        optional comma-separated category filter
#   TIMEOUT           per-request timeout in seconds (default: 120)
#   MAX_RETRIES       max retry attempts (default: 5)
#   MAX_TOKENS        max output tokens (default: 16384)
#   TEMPERATURE       sampling temperature (default: 0)
#   TOP_P             nucleus sampling top-p (default: 1)
#   OVERWRITE         set to 1 to overwrite existing outputs
#   DRY_RUN           set to 1 to print commands without calling the API
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

MODE="${MODE:-both}"
MODEL="${MODEL:-gpt-5.4}"
BASE_URL="${BASE_URL:-https://api.openai.com/v1}"
API_KEY="${API_KEY:-}"
API_KEY_ENV="${API_KEY_ENV:-OPENAI_API_KEY}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/test_code}"
MAX_DATA_ROWS="${MAX_DATA_ROWS:-0}"
MAX_QPS="${MAX_QPS:-1}"
LEVELS="${LEVELS:-intent,operation,implementation}"
CATEGORIES="${CATEGORIES:-}"
TIMEOUT="${TIMEOUT:-120}"
MAX_RETRIES="${MAX_RETRIES:-5}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"

_extra_flags=()
if [ "${OVERWRITE}" = "1" ]; then _extra_flags+=(--overwrite); fi
if [ "${DRY_RUN}" = "1" ];   then _extra_flags+=(--dry-run);  fi

_run_mode() {
  local mode="$1"
  shift  # remove mode from positional args so "$@" = any extra flags
  local script
  if [ "${mode}" = "code" ] || [ "${mode}" = "llm" ]; then
    script="${ROOT}/tasks/evaluate_llm.py"
  else
    script="${ROOT}/tasks/evaluate_vlm.py"
  fi

  local cmd=(
    "${PYTHON_BIN}" "${script}"
    --repo-root   "${ROOT}"
    --model       "${MODEL}"
    --base-url    "${BASE_URL}"
    --api-key-env "${API_KEY_ENV}"
    --output-root "${OUTPUT_ROOT}"
    --levels      "${LEVELS}"
    --max-qps     "${MAX_QPS}"
    --timeout     "${TIMEOUT}"
    --max-retries "${MAX_RETRIES}"
    --max-tokens  "${MAX_TOKENS}"
    --temperature "${TEMPERATURE}"
    --top-p       "${TOP_P}"
    --max-data-rows "${MAX_DATA_ROWS}"
  )
  if [ -n "${API_KEY}" ];    then cmd+=(--api-key    "${API_KEY}");    fi
  if [ -n "${CATEGORIES}" ]; then cmd+=(--categories "${CATEGORIES}"); fi
  cmd+=("${_extra_flags[@]}")
  # forward any extra CLI args passed to this script
  if [ "$#" -gt 0 ]; then cmd+=("$@"); fi

  echo ""
  if [ "${mode}" = "code-image" ] || [ "${mode}" = "vlm" ]; then
    echo "=== Running code+image generation ==="
  else
    echo "=== Running code generation ==="
  fi
  "${cmd[@]}"
}

case "${MODE}" in
  code|llm)  _run_mode code  "$@" ;;
  code-image|code+image|vlm)  _run_mode code-image  "$@" ;;
  both)
    _run_mode code "$@"
    _run_mode code-image "$@"
    ;;
  *)
    echo "ERROR: MODE must be code, code-image, or both (got: ${MODE})" >&2
    exit 1
    ;;
esac
