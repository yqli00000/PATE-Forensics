#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
output="${1:-outputs/synthscars_test/explanations}"
python -m scripts.score_synthscars_explanations \
  --data-root "${SYNTHSCARS_ROOT:-datasets/SynthScars}" \
  --predictions-jsonl "$output/predictions.jsonl" --output-dir "$output" \
  --model "${CSS_MODEL:-sentence-transformers/paraphrase-MiniLM-L6-v2}" \
  --device "${CSS_DEVICE:-cpu}" --limit 1000
