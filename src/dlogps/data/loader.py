"""The public loader: a dataset directory to shuffled training batches.

**Nothing counts the cables.** How many there are is an attribute of the data
directory and of the population CSV, never of this code. The loader reads every
``cable_*/`` it finds and the split is a fraction, so a grid that grows or
shrinks needs no edit here.

**Shuffling is an index operation, not a file operation.** Every cable is loaded
once and held in memory with its padding stripped, then an index of
``(cable, episode, frame)`` is shuffled. Per-frame files would give O(C) opens
per sample and ~587k files for the release set; this gives none.

**Two evaluation stages, two datasets.** Generalization is asked in increasing
order of difficulty, and the split slices the *test* set into both:

1. **Seen cables, unseen drops.** The easier question. Same cables the model
   trained on, released from poses it never saw, because a test set generated at
   a different ``seed`` redraws every release pose while the cable is identical.
2. **Unseen cables, unseen drops.** The claim the paper makes: a stiffness,
   diameter and length combination the model has never met.

Both come off one generated test set. ``stratified_held_out`` is deterministic
from cable id and params, so it assigns a cable the same side in either dataset,
and the two stages are the two slices of the test set:
:func:`eval_seen_cables` and :func:`eval_unseen_cables`.

**Two entry points, because labels are not a training input.**
:func:`training_batches` yields ``Batch`` alone: the loss is one-step against
``target`` and needs no labels. :func:`eval_windows` yields ``(Batch, Labels)``
over an ``R``-frame ground-truth window, which is what the two decomposition
metrics score (``docs/metrics.md``).
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from dlogps.data.dataset import CableData, load_cable, make_batch, window_starts
from dlogps.data.labels import contact_labels
from dlogps.data.types import Batch, E, Labels

HELD_OUT_FRACTION = 0.2
"""Default share of cables held out for a within-root validation split. The
reported runs train on the whole root (``held_out_fraction: 0.0``) and score
dedicated test roots instead (``docs/experiments.md``)."""


@dataclass(frozen=True)
class SplitCfg:
    """How the population divides, and **the** label thresholds.

    The two thresholds are `[FIX ON PROBE]` and this is their one home: they are
    applied here, at label time, and they travel into the checkpoint so a
    scorecard reports what training committed to. The metric suite holds no copy
    (``harness/metrics.py``), because a second copy reaches nothing and turns the
    gate that the thresholds are fixed before results into a formality.

    Attributes:
        held_out_fraction: Share of cables held out, **a fraction and never a
            count**: the population size is the CSV's business. ``0`` means train
            on every cable (no within-grid holdout); use a separate test root for
            evaluation rather than slicing this population.
        d_close_diameters: Contact threshold in cable diameters, never absolute.
            A multiplier because the population spans 1.5 mm to 10 mm, so one
            fixed distance would be too strict for the thick cables and too loose
            for the thin ones, and the phase split would then track cable
            thickness rather than geometry.
        c_far_segments: Chain distance beyond which a pair counts as far. Four,
            which is past the two-edge stencil that carries bending in Discrete
            Elastic Rods, so a contact pair cannot be two segments merely bending
            against each other.

    Raises:
        ValueError: on a fraction outside ``[0, 1)`` or a threshold that would
            silently label nothing.
    """

    held_out_fraction: float = HELD_OUT_FRACTION
    d_close_diameters: float = 1.5
    c_far_segments: int = 4

    def __post_init__(self) -> None:
        if not 0.0 <= self.held_out_fraction < 1.0:
            raise ValueError(
                f"held_out_fraction is a fraction of the population and must be in [0, 1), "
                f"got {self.held_out_fraction}"
            )
        if self.d_close_diameters <= 0:
            raise ValueError(f"d_close_diameters must be positive, got {self.d_close_diameters}")
        if not 0 < self.c_far_segments < E:
            raise ValueError(f"c_far_segments must be in (0, {E}), got {self.c_far_segments}")


def _stratum_order(key: tuple[float, float], ids: list[int]) -> list[int]:
    """A stratum's cables in an order fixed by the stratum and blind to the id.

    **Taking the lowest ids was the rule as first written, and it confounded the
    whole stage-2 claim with diameter.** The population enumerates diameter as
    the inner axis, so within a stratum the id increases with thickness and
    ``sorted(ids)[:quota]`` always takes the thinnest cable. On the 46-row
    population that made all nine held-out cables 2.5 mm against a training span
    of 1.5 mm to 10 mm, at every fraction from 0.10 to 0.25: stage 2 then
    measures the thinnest corner rather than the population, and bending rigidity
    goes as the fourth power of diameter, so the held-out side is systematically
    the floppiest cable of each material.

    A permutation seeded from the stratum key alone breaks the correlation while
    keeping the split reproducible from the config: no RNG state crosses a call,
    and ``blake2b`` rather than ``hash`` because the built-in is salted per
    process and would move the split between runs.
    """
    seed = int.from_bytes(hashlib.blake2b(repr(key).encode(), digest_size=8).digest(), "big")
    order = sorted(ids)
    return [order[i] for i in np.random.default_rng(seed).permutation(len(order))]


def stratified_held_out(cables: list[CableData], cfg: SplitCfg) -> set[int]:
    """Cable ids held out, stratified so no stratum is missing from either side.

    Whole cable parameterizations, never trajectories from a training cable:
    episodes of one cable share material parameters and would leak.

    Strata are ``(bend_stiffness, length)``, the material and length axes.
    Diameter is not an axis because it is nested inside material rather than
    crossed with it (``docs/data.md``). Which
    member of a stratum is taken is :func:`_stratum_order`'s business, and it is
    deliberately not the lowest id.

    **Largest remainder, not rounding up per stratum.** Rounding each stratum up
    was the rule as first written and it does not hold the fraction: this
    population has 18 strata and six of them are singletons, so ``ceil`` takes
    the only cable and the split lands at 39% with six strata gone from training
    entirely. Floor each stratum, then hand the remaining places to the strata
    with the largest fractional parts until the global target is met. That hits
    the fraction exactly and cannot empty a stratum, because a stratum is only
    ever allocated its floor plus one.

    Deterministic: strata are ordered by their key and cables by id, so the split
    is reproducible from the config alone rather than from an RNG.
    """
    strata: dict[tuple[float, float], list[int]] = defaultdict(list)
    for cable in cables:
        strata[(cable.params["bend_stiffness"], cable.params["length"])].append(cable.cable_id)

    target = round(cfg.held_out_fraction * len(cables))
    quota = {key: math.floor(cfg.held_out_fraction * len(ids)) for key, ids in strata.items()}
    # Ties break on the stratum key, so the result does not depend on dict order.
    by_remainder = sorted(
        strata,
        key=lambda key: (-((cfg.held_out_fraction * len(strata[key])) % 1), key),
    )
    for key in by_remainder:
        if sum(quota.values()) >= target:
            break
        if quota[key] < len(strata[key]):
            quota[key] += 1

    held: set[int] = set()
    for key, ids in strata.items():
        held.update(_stratum_order(key, ids)[: quota[key]])
    return held


def load_dataset(root: str | Path) -> list[CableData]:
    """Load every ``cable_*/`` under ``root``, in id order.

    Raises:
        FileNotFoundError: if the directory holds no cable.
    """
    dirs = sorted(Path(root).glob("cable_*"))
    if not dirs:
        raise FileNotFoundError(f"{root}: no cable_* directories")
    return [load_cable(d) for d in dirs]


def sample_index(cables: list[CableData], history_frames: int) -> list[tuple[int, int, int]]:
    """Every ``(cable, episode, frame)`` a sample can be built from.

    This is the thing that gets shuffled. Building it costs one pass and no IO,
    and it is what makes per-frame files unnecessary.

    Args:
        cables: Loaded cables, from :func:`load_dataset`.
        history_frames: ``C``. A window is ``C`` frames of history plus a target,
            so the depth decides how many starts each episode offers.
    """
    return [
        (c, e, k)
        for c, cable in enumerate(cables)
        for e, (pos, _, _) in enumerate(cable.episodes)
        for k in window_starts(len(pos), history_frames)
    ]


def training_batches(
    cables: list[CableData],
    *,
    batch_size: int,
    history_frames: int,
    cfg: SplitCfg | None = None,
    held_out: bool = False,
    seed: int = 0,
    shuffle: bool = True,
) -> Iterator[Batch]:
    """Yield one-step training batches from one side of the split.

    **No ``Labels``.** The loss is one-step against ``target``; the labels exist
    for two evaluation metrics and are not a training input.

    Batches never mix cables, because ``BatchMeta`` is per sample but the loader
    builds it per cable; mixing would need a gather and buys nothing at this
    batch size.

    Args:
        cables: Loaded cables, from :func:`load_dataset`.
        batch_size: Samples per batch. A short final batch is yielded, not dropped.
        history_frames: ``C``, the run's velocity-history depth.
        cfg: Split and threshold config.
        held_out: Which side of the split to draw from.
        seed: Shuffle seed; the order is reproducible from it.
        shuffle: Off for a deterministic pass, e.g. in a test.
    """
    cfg = cfg or SplitCfg()
    held = stratified_held_out(cables, cfg)
    wanted = [c for c, cable in enumerate(cables) if (cable.cable_id in held) == held_out]

    index = [(c, e, k) for c, e, k in sample_index(cables, history_frames) if c in set(wanted)]
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(index)  # type: ignore[arg-type]

    by_cable: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for c, e, k in index:
        by_cable[c].append((e, k))
        if len(by_cable[c]) == batch_size:
            yield make_batch(
                cables[c], by_cable.pop(c), is_held_out=held_out, history_frames=history_frames
            )
    for c, windows in by_cable.items():
        yield make_batch(cables[c], windows, is_held_out=held_out, history_frames=history_frames)


def eval_windows(
    cables: list[CableData],
    *,
    rollout: int,
    history_frames: int,
    cfg: SplitCfg | None = None,
    held_out: bool = True,
    stride: int | None = None,
) -> Iterator[tuple[Batch, Labels]]:
    """Yield ``(Batch, Labels)`` for autoregressive evaluation.

    The batch is the rollout's **first** frame, the state the model is started
    from. The labels cover the whole ground-truth window, frame for frame with
    the tensors the metric suite scores.

    Args:
        cables: Loaded cables.
        rollout: Frames ``R`` in the evaluation window. Episodes too short for it
            are skipped rather than padded.
        history_frames: ``C``, the depth the model under evaluation was trained
            at. It sets the first startable frame, so the driver walking the
            ground truth beside this stream must use the same value.
        cfg: Split and threshold config.
        held_out: Which side of the split to evaluate.
        stride: Frames between window starts; defaults to ``rollout``, so windows
            do not overlap.
    """
    cfg = cfg or SplitCfg()
    held = stratified_held_out(cables, cfg)
    step = stride or rollout

    for cable in cables:
        if (cable.cable_id in held) != held_out:
            continue
        for e, (pos, _, _) in enumerate(cable.episodes):
            for k in range(history_frames - 1, len(pos) - rollout, step):
                batch = make_batch(
                    cable, [(e, k)], is_held_out=held_out, history_frames=history_frames
                )
                truth = pos[k : k + rollout].unsqueeze(0)
                labels = contact_labels(
                    truth,
                    batch.meta.diameter,
                    d_close_diameters=cfg.d_close_diameters,
                    c_far_segments=cfg.c_far_segments,
                )
                yield batch, labels


def eval_seen_cables(
    test_cables: list[CableData],
    *,
    rollout: int,
    history_frames: int,
    cfg: SplitCfg | None = None,
    **kwargs,
) -> Iterator[tuple[Batch, Labels]]:
    """Stage 1: cables the model trained on, released from poses it never saw.

    The easier generalization question, and the one to answer first: if a model
    cannot predict a fresh drop of a cable it knows, the harder question is moot.

    Args:
        test_cables: The **test** dataset, generated at a different ``seed`` from
            the training set so every release pose is redrawn.
        rollout: Frames in the evaluation window.
        history_frames: ``C``; must match the one training used.
        cfg: Split and threshold config; must match the one training used.
    """
    yield from eval_windows(
        test_cables,
        rollout=rollout,
        history_frames=history_frames,
        cfg=cfg,
        held_out=False,
        **kwargs,
    )


def eval_all_cables(
    test_cables: list[CableData],
    *,
    rollout: int,
    history_frames: int,
    cfg: SplitCfg | None = None,
    **kwargs,
) -> Iterator[tuple[Batch, Labels]]:
    """Every cable in ``test_cables``, ignoring the train split.

    For a dedicated test root (seen-test or interpolated unseen-test) where the
    whole population is the stage and nothing is held out of training by id.
    ``cfg`` still supplies the contact thresholds; its ``held_out_fraction`` is
    forced to zero so the stream and the truth walker stay aligned.
    """
    cfg = cfg or SplitCfg()
    all_cfg = SplitCfg(
        held_out_fraction=0.0,
        d_close_diameters=cfg.d_close_diameters,
        c_far_segments=cfg.c_far_segments,
    )
    yield from eval_windows(
        test_cables,
        rollout=rollout,
        history_frames=history_frames,
        cfg=all_cfg,
        held_out=False,
        **kwargs,
    )


def eval_unseen_cables(
    test_cables: list[CableData],
    *,
    rollout: int,
    history_frames: int,
    cfg: SplitCfg | None = None,
    **kwargs,
) -> Iterator[tuple[Batch, Labels]]:
    """Stage 2: cables the model never trained on, released from unseen poses.

    The claim the paper makes. These cables were held out of training entirely,
    so both their parameters and their drops are new.
    """
    yield from eval_windows(
        test_cables,
        rollout=rollout,
        history_frames=history_frames,
        cfg=cfg,
        held_out=True,
        **kwargs,
    )


def rollout_truth(cable: CableData, episode: int, start: int, rollout: int) -> torch.Tensor:
    """``[1, R, N, 3]`` ground-truth positions the rollout is scored against."""
    pos, _, _ = cable.episodes[episode]
    return pos[start : start + rollout].unsqueeze(0)
