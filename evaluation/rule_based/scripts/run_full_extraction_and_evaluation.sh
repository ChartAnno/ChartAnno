#!/usr/bin/env bash
set -euo pipefail

EVAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SKIP_DIFF=0 bash "${EVAL_ROOT}/scripts/run_evaluation_all.sh"
