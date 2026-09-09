#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
checkpoint="${1:?Usage: bash opensdi_test/test.sh CHECKPOINT [OUTPUT_DIR]}"
output="${2:-outputs/opensdi}"
data_root="${OPENSDI_ROOT:-datasets/OpenSDI}"
backbone="${BACKBONE_PATH:-weights/dinov3-l16}"
python -m scripts.evaluate_opensdi \
  --checkpoint "$checkpoint" --data-root "$data_root" --output-dir "$output" \
  --batch-size 16 --threshold 0.5 \
  model.backbone0="$backbone"
