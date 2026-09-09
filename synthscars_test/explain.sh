#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
localization="${1:-outputs/synthscars_test/localization_per_sample.jsonl}"
output="${2:-outputs/synthscars_test/explanations}"
python -m scripts.generate_synthscars_explanations \
  --localization-jsonl "$localization" --output-jsonl "$output/predictions.jsonl" \
  --limit 1000 --resume
