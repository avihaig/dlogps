#!/usr/bin/env python
"""Build the training root ``release_train`` from ``release_v1`` and ``test_v1``.

The two released roots hold the same 46 cables, 50 episodes each, released from
different start poses (the generator seed is the only difference between
``configs/release.yaml`` and ``configs/release_test.yaml``). Every reported run
trains on their union: 46 cables x 100 episodes. This script writes that union
in the same ``cable_XXX/episodes.npz`` + ``params.yaml`` layout the loader
reads, concatenating the episodes of each cable along the episode axis and
re-padding to the longer of the two padded lengths.

    python scripts/merge_release_train.py \\
        --inputs data/release_v1 data/test_v1 --out data/release_train

Per-cable scalars (``cable_id``, ``length``, ``diameter``, ``bend_stiffness``,
``joint_damping``, ``effective_linear_density``, ``record_hz``) must agree
across inputs for the same cable, and the script refuses otherwise.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

PER_EPISODE = ("vertex_pos", "vertex_vel", "edge_quat", "edge_omega", "edge_wrench", "t")
"""Arrays with a leading episode axis and a padded time axis second."""

SCALARS = (
    "record_hz",
    "cable_id",
    "length",
    "diameter",
    "bend_stiffness",
    "joint_damping",
    "effective_linear_density",
)
"""Per-cable constants that must agree across the merged inputs."""

STOP_TOKEN = -1.0
"""Past ``episode_lengths[e]`` every padded array carries one stop token, then NaN."""


def _repad(array: np.ndarray, t_pad: int) -> np.ndarray:
    """Extend the padded time axis to ``t_pad`` with NaN, keeping the stop token."""
    if array.shape[1] == t_pad:
        return array
    shape = (array.shape[0], t_pad, *array.shape[2:])
    out = np.full(shape, np.nan, dtype=array.dtype)
    out[:, : array.shape[1]] = array
    return out


def merge_cable(inputs: list[Path], out_dir: Path) -> int:
    """Concatenate one cable's episodes across ``inputs``; return the episode count."""
    loaded = [np.load(path / "episodes.npz") for path in inputs]
    try:
        for key in SCALARS:
            values = [np.asarray(d[key]) for d in loaded]
            if any(not np.array_equal(values[0], v) for v in values[1:]):
                raise ValueError(f"{out_dir.name}: {key} differs across inputs: {values}")
        for key in PER_EPISODE:
            missing = [p for p, d in zip(inputs, loaded, strict=True) if key not in d.files]
            if missing:
                raise ValueError(f"{out_dir.name}: {key} missing under {missing}")
        t_pad = max(d["t"].shape[1] for d in loaded)
        merged = {
            key: np.concatenate([_repad(d[key], t_pad) for d in loaded]) for key in PER_EPISODE
        }
        merged["episode_lengths"] = np.concatenate([d["episode_lengths"] for d in loaded])
        for key in SCALARS:
            merged[key] = loaded[0][key]
    finally:
        for d in loaded:
            d.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "episodes.npz", **merged)
    # The knobs and provenance of the first input, kept beside the merged file;
    # the per-episode release records of the second input stay in its own root.
    shutil.copy2(inputs[0] / "params.yaml", out_dir / "params.yaml")
    return int(merged["episode_lengths"].shape[0])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--inputs", type=Path, nargs="+", required=True, help="roots to merge")
    parser.add_argument("--out", type=Path, required=True, help="the merged root to write")
    args = parser.parse_args(argv)

    cables = sorted(p.name for p in args.inputs[0].glob("cable_*"))
    if not cables:
        print(f"{args.inputs[0]}: no cable_* directories", file=sys.stderr)
        return 1
    for root in args.inputs[1:]:
        other = sorted(p.name for p in root.glob("cable_*"))
        if other != cables:
            print(f"{root}: cable set differs from {args.inputs[0]}", file=sys.stderr)
            return 1

    total = 0
    for name in cables:
        n = merge_cable([root / name for root in args.inputs], args.out / name)
        total += n
        print(f"{name}: {n} episodes")
    for extra in ("index.csv", "datagen.yaml"):
        src = args.inputs[0] / extra
        if src.is_file():
            shutil.copy2(src, args.out / extra)
    print(f"wrote {len(cables)} cables, {total} episodes to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
