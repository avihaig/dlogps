#!/usr/bin/env bash
# The whole pipeline on the two bundled sample cables, in a few minutes on a
# CPU: statistics, 200 training steps, a checkpoint, autoregressive rollouts
# on the sample test cables, the scorecard, artifacts and figures. Its numbers
# mean nothing; it proves the wiring. Writes results/smoke/runs/<timestamp>_smoke/.
#
#   scripts/smoke.sh                  # CPU
#   DEVICE=cuda scripts/smoke.sh      # GPU
#   scripts/smoke.sh --variant F      # any of A B C D F, or chain_only
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# The checkout's own venv when it exists (uv sync), else whatever python is on PATH.
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then PYTHON="$ROOT/.venv/bin/python"; else PYTHON=python; fi
fi
DEVICE=${DEVICE:-cpu}

"$PYTHON" -m dlogps.harness.run_experiment \
  --config_path configs/experiments/smoke.yaml \
  --train.device "$DEVICE" \
  "$@"
