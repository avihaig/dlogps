#!/usr/bin/env bash
# Point this checkout's `data/` at the dataset on this machine.
#
# The dataset is ~16 GB and is kept outside the checkout at $DLOGPS_DATA
# (default ~/datasets/dlogps); `data/` is a symlink to it, which .gitignore
# covers by name. Idempotent, and it refuses to replace a real directory.
#
# Expected layout under $DLOGPS_DATA (docs/data.md):
#   release_v1/          46 cables x 50 episodes   (training root, half 1)
#   test_v1/             46 cables x 50 episodes   (training root, half 2)
#   release_train/       the two merged: 46 cables x 100 episodes
#                        (scripts/merge_release_train.py builds it)
#   unseen_cables_test/  40 held-out cables         (the reported test root)
set -euo pipefail

SHARED="${DLOGPS_DATA:-$HOME/datasets/dlogps}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LINK="$ROOT/data"

if [[ ! -d "$SHARED" ]]; then
    echo "no dataset at $SHARED" >&2
    echo "set DLOGPS_DATA to the directory holding release_v1/ test_v1/ unseen_cables_test/ (docs/data.md)" >&2
    exit 1
fi

if [[ -L "$LINK" ]]; then
    current="$(readlink -f "$LINK")"
    if [[ "$current" == "$(readlink -f "$SHARED")" ]]; then
        echo "already linked: data -> $current"
    else
        echo "relinking data: $current -> $SHARED"
        rm "$LINK"
        ln -s "$SHARED" "$LINK"
    fi
elif [[ -e "$LINK" ]]; then
    echo "refusing to replace $LINK: it is a real directory, not a link" >&2
    exit 1
else
    ln -s "$SHARED" "$LINK"
    echo "linked: data -> $SHARED"
fi

for split in release_v1 test_v1 release_train unseen_cables_test; do
    if [[ -d "$LINK/$split" ]]; then
        echo "  $split: $(find "$LINK/$split" -maxdepth 1 -name 'cable_*' | wc -l) cables"
    else
        echo "  $split: missing"
    fi
done
