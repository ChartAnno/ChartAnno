#!/usr/bin/env bash
set -euo pipefail

EVAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Edit this list before running.
models=( "gemini3flash" )

PROJECT_PATH="${PROJECT_PATH:-$(pwd)}"
TEST_CODE_PREFIX="${TEST_CODE_PREFIX:-test_code_}"
TEST_IMAGE_PREFIX="${TEST_IMAGE_PREFIX:-test_code_image_}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-outputs_}"
RUN_SCRIPT="${RUN_SCRIPT:-run_evaluation_all.sh}"

for model in "${models[@]}"; do
  echo "[model] ${model}"
  PROJECT_PATH="${PROJECT_PATH}" \
  TEST_CODE_DIR="${TEST_CODE_PREFIX}${model}" \
  TEST_IMAGE_DIR="${TEST_IMAGE_PREFIX}${model}" \
  ANNOTATIONS_DIR="${OUTPUT_PREFIX}${model}/annotations" \
  ANALYSIS_DIR="${OUTPUT_PREFIX}${model}/analysis" \
  RENDERED_IMAGES_DIR="${OUTPUT_PREFIX}${model}/rendered_images" \
  DISPLAY_NAME="${model}" \
  bash "${EVAL_ROOT}/scripts/${RUN_SCRIPT}"
done
