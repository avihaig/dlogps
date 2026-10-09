#!/usr/bin/env bash
# Download the final checkpoints of the 39 reported runs from the Hugging Face
# Hub (avihaig/dlogps-checkpoints) and place each one at
# results/runs/<experiment>/<label>/checkpoints/step_100000.pt, beside the
# run's shipped records.
#
#   scripts/fetch_checkpoints.sh            # all 39 (~165 MB)
#   scripts/fetch_checkpoints.sh local_on   # one experiment
#
# Needs the Hugging Face CLI (`pip install -U huggingface_hub`). Files already
# downloaded are skipped.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
REPO=${REPO:-avihaig/dlogps-checkpoints}
FILTER=${1:-*}

hf download "$REPO" --include "${FILTER}/*/checkpoints/step_100000.pt" --local-dir results/runs
