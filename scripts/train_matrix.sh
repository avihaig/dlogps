#!/usr/bin/env bash
# The bias-variant matrix: five variants x three seeds, with the local stream
# on (configs/experiments/local_on.yaml) or off (local_off.yaml).
#
#   scripts/train_matrix.sh --local on                 # all 15 local-on arms
#   scripts/train_matrix.sh --local off --variants "C F" --seeds "0"
#
# Variant letters (docs/method.md): A = Unbiased, B = Euclidean, C = Chain,
# D = Mixed, F = Euclidean+Chain only. Each arm trains 100k steps and scores
# its final checkpoint on data/unseen_cables_test (400 windows, 400 predicted
# frames); the run directory lands under results/<experiment>/runs/.
# Arms whose final unseen-test summary already exists are skipped, so an
# interrupted sweep resumes where it stopped.
#
# Environment: PYTHON (default: the checkout venv, else python), DEVICE (default: cuda).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# The checkout's own venv when it exists (uv sync), else whatever python is on PATH.
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then PYTHON="$ROOT/.venv/bin/python"; else PYTHON=python; fi
fi
DEVICE=${DEVICE:-cuda}
LOCAL=on
VARIANTS="A B C D F"
SEEDS="0 1 2"
STEPS=100000

while (( $# )); do
  case "$1" in
    --local) LOCAL=$2; shift ;;
    --variants) VARIANTS=$2; shift ;;
    --seeds) SEEDS=$2; shift ;;
    *) echo "usage: train_matrix.sh [--local on|off] [--variants \"A B C D F\"] [--seeds \"0 1 2\"]" >&2; exit 2 ;;
  esac
  shift
done
case "$LOCAL" in
  on) EXPERIMENT=local_on ;;
  off) EXPERIMENT=local_off ;;
  *) echo "--local must be on or off, got $LOCAL" >&2; exit 2 ;;
esac
CONFIG=configs/experiments/${EXPERIMENT}.yaml

if [[ ! -d data/release_train/cable_000 ]]; then
  echo "data/release_train is missing: run scripts/link_data.sh and scripts/merge_release_train.py (docs/data.md)" >&2
  exit 1
fi

arm_is_complete() {
  local label=$1 run
  for run in results/${EXPERIMENT}/runs/*_"${label}"; do
    [[ -f "$run/eval/model/unseen_test_step${STEPS}_summary.json" ]] && { echo "$run"; return 0; }
  done
  return 1
}

for SEED in $SEEDS; do
  for VARIANT in $VARIANTS; do
    LABEL="var${VARIANT}_s${SEED}"
    if RUN=$(arm_is_complete "$LABEL"); then
      echo "skip ${LABEL}: complete at ${RUN}"
      continue
    fi
    echo
    echo "================================================================"
    echo "  ${EXPERIMENT}  arm=${LABEL}  variant=${VARIANT}  seed=${SEED}  $(date -Is)"
    echo "================================================================"
    "$PYTHON" -m dlogps.harness.run_experiment \
      --config_path "$CONFIG" \
      --label "$LABEL" \
      --variant "$VARIANT" \
      --train.seed "$SEED" \
      --train.device "$DEVICE" || echo "ARM FAILED: ${LABEL} (continuing)"
  done
done
