#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -m scripts.train_opensdi --cfg cfgs/train/train_opensdi.yaml \
  --logdir logs/opensdi "$@"
