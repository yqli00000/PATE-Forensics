#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python train.py --cfg cfgs/train/train_synthscars.yaml \
  --logdir synthscars "$@"
