#!/usr/bin/env bash
set -euo pipefail

EVAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PROJECT_PATH="${PROJECT_PATH:-$(pwd)}"
MPLCONFIGDIR="${MPLCONFIGDIR:-outputs/.mplconfig}"
if [[ "${MPLCONFIGDIR}" != /* ]]; then
  MPLCONFIGDIR="${PROJECT_PATH}/${MPLCONFIGDIR}"
fi
export MPLCONFIGDIR
mkdir -p "${MPLCONFIGDIR}"

DATASET_CODE_DIR="${DATASET_CODE_DIR:-dataset_code}"
DATASET_CODE_REMOVED_DIR="${DATASET_CODE_REMOVED_DIR:-dataset_code_removed}"
DATASET_IMAGE_REMOVED_DIR="${DATASET_IMAGE_REMOVED_DIR:-dataset_image_removed}"
TEST_CODE_DIR="${TEST_CODE_DIR:-test_code}"
TEST_IMAGE_DIR="${TEST_IMAGE_DIR:-test_code_image}"
ANNOTATIONS_DIR="${ANNOTATIONS_DIR:-outputs/annotations}"
ANALYSIS_DIR="${ANALYSIS_DIR:-outputs/analysis}"
RENDERED_IMAGES_DIR="${RENDERED_IMAGES_DIR:-outputs/rendered_images}"
GT_ANNOTATIONS_DIR="${GT_ANNOTATIONS_DIR:-}"
SHARED_TEXT_EXTRACTION_DIR="${SHARED_TEXT_EXTRACTION_DIR:-}"
DISPLAY_NAME="${DISPLAY_NAME:-}"
REPORT_TITLE="${REPORT_TITLE:-}"
OUTPUT_CSV="${OUTPUT_CSV:-}"
SKIP_DIFF="${SKIP_DIFF:-1}"

args=(
  "--project-root" "${PROJECT_PATH}"
  "--dataset-code-dir" "${DATASET_CODE_DIR}"
  "--dataset-code-removed-dir" "${DATASET_CODE_REMOVED_DIR}"
  "--dataset-image-removed-dir" "${DATASET_IMAGE_REMOVED_DIR}"
  "--test-code-dir" "${TEST_CODE_DIR}"
  "--test-image-dir" "${TEST_IMAGE_DIR}"
  "--annotations-dir" "${ANNOTATIONS_DIR}"
  "--analysis-dir" "${ANALYSIS_DIR}"
  "--rendered-images-dir" "${RENDERED_IMAGES_DIR}"
)

if [[ -n "${GT_ANNOTATIONS_DIR}" ]]; then
  args+=("--gt-annotations-dir" "${GT_ANNOTATIONS_DIR}")
fi
if [[ -n "${SHARED_TEXT_EXTRACTION_DIR}" ]]; then
  args+=("--shared-text-extraction-dir" "${SHARED_TEXT_EXTRACTION_DIR}")
fi
if [[ -n "${DISPLAY_NAME}" ]]; then
  args+=("--display-name" "${DISPLAY_NAME}")
fi
if [[ -n "${REPORT_TITLE}" ]]; then
  args+=("--report-title" "${REPORT_TITLE}")
fi
if [[ -n "${OUTPUT_CSV}" ]]; then
  args+=("--output-csv" "${OUTPUT_CSV}")
fi
if [[ "${SKIP_DIFF}" == "1" ]]; then
  args+=("--skip-diff")
fi
if [[ "${SKIP_GT_DIFF:-0}" == "1" ]]; then
  args+=("--skip-gt-diff")
fi
if [[ "${SKIP_MODEL_DIFF:-0}" == "1" ]]; then
  args+=("--skip-model-diff")
fi
if [[ "${SKIP_RUN:-0}" == "1" ]]; then
  args+=("--skip-run")
fi
if [[ "${SKIP_EXISTING:-0}" == "1" ]]; then
  args+=("--skip-existing")
fi
if [[ "${ONLY_TEXT_REPORT:-0}" == "1" ]]; then
  args+=("--only-text-report")
fi
if [[ "${DIFF_ONLY:-0}" == "1" ]]; then
  args+=("--diff-only")
fi

python "${EVAL_ROOT}/scripts/run_evaluation_all.py" "${args[@]}"
