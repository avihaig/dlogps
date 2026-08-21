#!/usr/bin/env bash
# The three baselines at three seeds each:
#   chain_only   GatedGCN message passing on the chain edges, no attention
#                (configs/experiments/chain_only.yaml)
#   mlp_pernode  per-node MLP, width 2048, three layers (structfree.yaml)
#   lstm_global  global LSTM, width 256, three layers   (structfree.yaml)
#
#   scripts/train_baselines.sh                       # all nine runs
#   scripts/train_baselines.sh --arms "lstm_global" --seeds "0"
#
# Same training protocol and evaluation as the matrix (scripts/train_matrix.sh).
# Completed arms are skipped. Environment: PYTHON (default: python), DEVICE
# (default: cuda).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# The checkout's own venv when it exists (uv sync), else whatever python is on PATH.
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then PYTHON="$ROOT/.venv/bin/python"; else PYTHON=python; fi
fi
DEVICE=${DEVICE:-cuda}
ARMS="chain_only mlp_pernode lstm_global"
SEEDS="0 1 2"
STEPS=100000

while (( $# )); do
  case "$1" in
    --arms) ARMS=$2; shift ;;
    --seeds) SEEDS=$2; shift ;;
    *) echo "usage: train_baselines.sh [--arms \"chain_only mlp_pernode lstm_global\"] [--seeds \"0 1 2\"]" >&2; exit 2 ;;
  esac
  shift
done

if [[ ! -d data/release_train/cable_000 ]]; then
  echo "data/release_train is missing: run scripts/link_data.sh and scripts/merge_release_train.py (docs/data.md)" >&2
  exit 1
fi

arm_is_complete() {
  local experiment=$1 label=$2 run
  for run in results/${experiment}/runs/*_"${label}"; do
    [[ -f "$run/eval/model/unseen_test_step${STEPS}_summary.json" ]] && { echo "$run"; return 0; }
  done
  return 1
}

run_arm() {
  local experiment=$1 label=$2; shift 2
  if RUN=$(arm_is_complete "$experiment" "$label"); then
    echo "skip ${label}: complete at ${RUN}"
    return 0
  fi
  echo
  echo "================================================================"
  echo "  ${experiment}  arm=${label}  $(date -Is)"
  echo "================================================================"
  "$PYTHON" -m dlogps.harness.run_experiment \
    --config_path "configs/experiments/${experiment}.yaml" \
    --label "$label" \
    --train.device "$DEVICE" \
    "$@" || echo "ARM FAILED: ${label} (continuing)"
}

for SEED in $SEEDS; do
  for ARM in $ARMS; do
    case "$ARM" in
      chain_only)
        run_arm chain_only "chain_s${SEED}" --train.seed "$SEED" ;;
      mlp_pernode)
        run_arm structfree "mlp_pernode_s${SEED}" --train.seed "$SEED" \
          --model.arch mlp_pernode --model.d_model 2048 ;;
      lstm_global)
        run_arm structfree "lstm_global_s${SEED}" --train.seed "$SEED" \
          --model.arch lstm_global --model.d_model 256 ;;
      *) echo "unknown arm ${ARM}" >&2; exit 2 ;;
    esac
  done
done
