#!/usr/bin/env bash
# Download the final checkpoints of the 39 reported runs from the GitHub
# release and place each one at results/runs/<experiment>/<label>/checkpoints/
# step_100000.pt, beside the run's shipped records.
#
#   scripts/fetch_checkpoints.sh            # all 39 (~3.5 MB each)
#   scripts/fetch_checkpoints.sh local_on   # one experiment
#
# Needs the GitHub CLI (`gh`) or curl. Release: v1.0 of avihaig/dlogps.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
REPO=${REPO:-avihaig/dlogps}
TAG=${TAG:-v1.0}
FILTER=${1:-}

for RUN in results/runs/*/*/; do
  RUN=${RUN%/}
  EXPERIMENT=$(basename "$(dirname "$RUN")")
  LABEL=$(basename "$RUN")
  if [[ -n "$FILTER" && "$EXPERIMENT" != "$FILTER" ]]; then
    continue
  fi
  ASSET="${EXPERIMENT}__${LABEL}__step_100000.pt"
  OUT="$RUN/checkpoints/step_100000.pt"
  if [[ -f "$OUT" ]]; then
    echo "have $OUT"
    continue
  fi
  mkdir -p "$(dirname "$OUT")"
  echo "fetch $ASSET -> $OUT"
  if command -v gh >/dev/null 2>&1; then
    gh release download "$TAG" --repo "$REPO" --pattern "$ASSET" --output "$OUT"
  else
    curl -fL "https://github.com/${REPO}/releases/download/${TAG}/${ASSET}" -o "$OUT"
  fi
done
