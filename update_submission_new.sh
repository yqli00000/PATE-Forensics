#!/usr/bin/env bash

set -euo pipefail
# Usage:
#   bash update_submission_new.sh
#
# Edit the paths below before running.

OUTPUT_DIR="outputs/test"

# Choose one image input source:
IMAGE_PATH=""
IMAGE_DIR="DATASET_DDL/DDL_X_test/image"

# DashScope / OpenAI-compatible explanation API settings.
EXPLAIN_API_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
EXPLAIN_API_KEY=""  ## fix it before run
EXPLAIN_MODEL="qwen3.5-flash"
EXPLAIN_TIMEOUT=60

cd "$(dirname "$0")"
CMD=(
  python update_json_traces.py
  --image-dir "${IMAGE_DIR}"
  --image-path "${IMAGE_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --explain-api-url "${EXPLAIN_API_URL}"
  --explain-api-key "${EXPLAIN_API_KEY}"
  --explain-model "${EXPLAIN_MODEL}"
  --explain-timeout "${EXPLAIN_TIMEOUT}"
  --explain-workers 4
)

"${CMD[@]}"
