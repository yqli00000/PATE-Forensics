#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
checkpoint="${1:-weights/synthscars.ckpt}"
output="${2:-outputs/synthscars_test}"
python -m scripts.evaluate_synthscars \
  --checkpoint "$checkpoint" --mask-threshold 0.5 \
  --data-root "${SYNTHSCARS_ROOT:-datasets/SynthScars}" \
  --backbone-path "${BACKBONE_PATH:-weights/dinov3-l16}" \
  --output-dir "$output" --limit 1000
