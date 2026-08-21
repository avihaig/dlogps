#!/usr/bin/env bash
# Re-score one run's final checkpoint on the unseen-cable test root, writing
# eval/model/unseen_test_step100000_{summary.json,windows.csv,per_cable.csv,curve.csv}
# into the run directory. This is what every training run already does at its
# end; use it to score a checkpoint fetched from the release
# (scripts/fetch_checkpoints.sh) without retraining.
#
#   scripts/evaluate_final.sh results/local_on/runs/<run_dir> [more run dirs...]
#
# Environment: PYTHON (default: python), DEVICE (default: cuda),
# DATA (default: data/unseen_cables_test).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# The checkout's own venv when it exists (uv sync), else whatever python is on PATH.
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then PYTHON="$ROOT/.venv/bin/python"; else PYTHON=python; fi
fi
DEVICE=${DEVICE:-cuda}
DATA=${DATA:-data/unseen_cables_test}
STEPS=100000

if (( $# < 1 )); then
  echo "usage: evaluate_final.sh <run_dir> [run_dir ...]" >&2
  exit 2
fi

for RUN in "$@"; do
  CKPT="$RUN/checkpoints/step_${STEPS}.pt"
  if [[ ! -f "$CKPT" ]]; then
    echo "no final checkpoint at $CKPT" >&2
    exit 1
  fi
  echo "=== $RUN"
  # Rollout 401 frames: frame 0 is the input state, frames 1..400 are scored.
  "$PYTHON" -m dlogps.harness.evaluate \
    --checkpoint "$CKPT" \
    --data "$DATA" \
    --stage unseen_test \
    --horizon 401 \
    --horizons 1,10,50,100,200,400 \
    --limit 400 \
    --device "$DEVICE" \
    --run-dir "$RUN"
done
