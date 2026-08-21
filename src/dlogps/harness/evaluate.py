"""The evaluation driver: one stage of windows in, one scored result out.

**The two stages are reported separately and never merged.** Stage 1 is seen
cables released from unseen drops, stage 2 is cables the model never trained on.
A model can pass the first and fail the second, and that difference is the result
the paper reports; one averaged number would hide exactly the thing the
experiment asks.

**Windows are aggregated by count, not by episode and not by cable.** Episodes
differ in length by a factor of two in the sample data alone, so a per-episode
mean lets a 698-frame episode outvote a 1211-frame one, and the aggregate then
tracks how long a cable took to settle rather than how well the model predicted
it.

**The split arrives from the checkpoint.** Scoring against a different
:class:`SplitCfg` than training used puts held-out cables in stage 1, which
raises nothing and reports the easy question's answer under the hard question's
name. :func:`evaluate_checkpoint` is the path that cannot get it wrong.

**The ground-truth window is paired back onto the loader's stream, and the
pairing is checked rather than trusted.** ``eval_windows`` yields the rollout's
first frame and the labels it computed on the ground truth, but not the
``R``-frame window itself, so the driver walks the same enumeration and reads
each window off ``rollout_truth``. A second enumeration is a second thing that
can drift, so every pair is verified: frame 0 of the truth window is by
construction the frame the batch was built from, and the cable ids must agree.
Misalignment would shift every scored frame while leaving every metric finite.

**A window that defines a metric contributes to it; a window that does not is
dropped from it rather than counted as zero.** The metric suite returns NaN when
a mask selects no frames, because "no contact in this window" and "no error in
contact" are opposite findings, and a rollout that diverged returns NaN for the
same reason. Either way the window carries no evidence for that scalar, so it
leaves both the mean and the count alone. That is why :attr:`EvalResult.counts`
exists: a contact RMSE over 3 of 200 windows and one over all 200 are not the
same measurement, and the second number is the only thing that separates them.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn

from dlogps.data.dataset import CableData, feature_width
from dlogps.data.loader import (
    SplitCfg,
    eval_all_cables,
    eval_seen_cables,
    eval_unseen_cables,
    load_dataset,
    rollout_truth,
    stratified_held_out,
)
from dlogps.data.types import Batch, Labels
from dlogps.harness.metrics import RegimeCfg, per_frame_errors, score, unit_of
from dlogps.harness.normalize import Stats
from dlogps.harness.rollout import rollout
from dlogps.harness.runtime import batch_to, bound_intra_op_threads, labels_to, model_device
from dlogps.harness.train import Checkpoint, load_checkpoint


@dataclass(frozen=True, order=True)
class WindowKey:
    """Which rollout window a row of scores came from.

    Ordered and hashable so two arms' tables can be joined on it rather than
    zipped by position. Position holds only while both arms scored exactly the
    same windows in the same order, which a ``--limit``, a skipped short episode
    or a diverged rollout all break without raising.

    Attributes:
        cable_id: Population row id.
        episode: Index into that cable's kept episodes, **after** the
            diverged-episode filter, so it indexes ``CableData.episodes`` rather
            than the npz.
        start: First frame of the window, the one the rollout is seeded from.
    """

    cable_id: int
    episode: int
    start: int


@dataclass(frozen=True)
class WindowScore:
    """One window's scores, with the key that says which window it was."""

    key: WindowKey
    scores: dict[str, float]
    frame_errors: dict[str, tuple[float, ...]] = field(default_factory=dict)


WindowStream = Callable[..., Iterator[tuple[Batch, Labels]]]
"""A stage's window source: cables and a rollout length to ``(Batch, Labels)``."""

STAGES: dict[str, tuple[WindowStream, bool]] = {
    "seen_cables": (eval_seen_cables, False),
    "unseen_cables": (eval_unseen_cables, True),
    "all_cables": (eval_all_cables, False),
    "seen_test": (eval_all_cables, False),
    "unseen_test": (eval_all_cables, False),
}
"""Stages, each with the side of the split it draws from.

``seen_cables`` / ``unseen_cables`` slice one test root by the train split.
``all_cables`` / ``seen_test`` / ``unseen_test`` score every cable in the root
(for dedicated test populations). The named test stages exist so distinct roots
write distinct artifact prefixes in one experiment run; the paper reports
``unseen_test``, the root of cables never present in training.

The flag restates what the loader function already passes as ``held_out``, and
it is here because the ground-truth windows have to be walked down the same side
of the split. Reversing one of the two pairs a stage's rollouts against the
other stage's truth, which the alignment check in :func:`evaluate` turns into a
failure rather than a number."""

ALL_CABLE_STAGES = frozenset({"all_cables", "seen_test", "unseen_test"})
"""Stages that ignore the train holdout and score the whole root."""


def stratified_take[T](items: list[T], limit: int, *, cable_id: Callable[[T], int]) -> list[T]:
    """Cap a window list by round-robin across cables.

    A naive ``break`` after ``limit`` windows from an id-ordered stream scores
    only the first cables. Round-robin keeps every cable that has at least one
    window when ``limit >= n_cables``.
    """
    if limit < 1:
        raise ValueError(f"limit is a window count and must be at least 1, got {limit}")
    if len(items) <= limit:
        return list(items)

    buckets: dict[int, list[T]] = {}
    order: list[int] = []
    for item in items:
        cid = cable_id(item)
        if cid not in buckets:
            buckets[cid] = []
            order.append(cid)
        buckets[cid].append(item)

    out: list[T] = []
    cursors = {cid: 0 for cid in order}
    while len(out) < limit:
        progressed = False
        for cid in order:
            queue = buckets[cid]
            index = cursors[cid]
            if index >= len(queue):
                continue
            out.append(queue[index])
            cursors[cid] = index + 1
            progressed = True
            if len(out) >= limit:
                break
        if not progressed:
            break
    return out


@dataclass(frozen=True)
class EvalResult:
    """One stage's scorecard, and how much evidence is behind it.

    Attributes:
        stage: A key of :data:`STAGES`. Carried so two results cannot be added
            up by a runner that has forgotten which is which.
        n_windows: Rollout windows scored, the weight behind the aggregate.
        scores: Metric name to count-weighted mean. NaN for a metric no window
            defined, which is the honest answer when the class was always empty.
        limit: The cap this run was asked for, or ``None``. Recorded rather than
            applied silently: a smoke run's numbers and a full run's numbers are
            otherwise indistinguishable in a table.
        counts: Windows behind each scalar. Below ``n_windows`` when a
            decomposition's class was empty in some window, or when a rollout
            diverged there and the metric came back NaN. A count well under
            ``n_windows`` is the signal that the scalar above it describes a
            fraction of the run rather than the run.
        curve: Reduction name to mean error at each predicted frame, averaged
            over the windows scored. Index ``k`` is predicted frame ``k + 1``.
            **No scalar in ``scores`` can reconstruct this**: every metric
            collapses the frame axis, so whether a config is better everywhere or
            only early lives here alone.
        windows: One :class:`WindowScore` per scored window, in scoring order.
            **The aggregate above cannot be un-pooled**, so a paired comparison
            between two arms joins these on :class:`WindowKey`; pairing by
            position holds only while both arms scored the same windows, which a
            cap or a skipped episode breaks silently.
        stds: Spread of the per-batch values behind each mean, **across the
            windows of this stage and not across seeds**. It says whether the
            windows agree with each other; whether the run reproduces is a
            cross-seed question this object cannot answer. NaN where a single
            batch scored the metric, since one sample has no spread.

    **Every length here is metres**, as everywhere else in the project;
    ``metrics.unit_of`` gives the unit of any name and the reporters print it.
    """

    stage: str
    n_windows: int
    scores: dict[str, float]
    limit: int | None = None
    counts: dict[str, int] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    windows: tuple[WindowScore, ...] = ()
    curve: dict[str, tuple[float, ...]] = field(default_factory=dict)


class _Aggregate:
    """Count-weighted running mean per metric name, NaN read as no evidence.

    Names are seeded on first sight even when the value is NaN, so a metric that
    is undefined in every window is reported as NaN rather than vanishing from
    the result and taking its absence with it.
    """

    def __init__(self) -> None:
        self.windows: list[WindowScore] = []
        self.curve_total: dict[str, Tensor] = {}
        self.curve_windows = 0
        self.total: dict[str, float] = {}
        self.total_sq: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.n_windows = 0

    def add(
        self,
        scores: dict[str, float],
        windows: int,
        key: WindowKey | None = None,
        frame_errors: dict[str, tuple[float, ...]] | None = None,
    ) -> None:
        """Fold in one scored batch, weighted by the windows it covered.

        ``key`` is retained verbatim alongside the running mean, because a mean
        cannot be un-pooled afterwards and the paired statistic the config
        comparison rests on needs the same window under two arms.
        """
        self.n_windows += windows
        if key is not None:
            self.windows.append(
                WindowScore(key, dict(scores), frame_errors=dict(frame_errors or {}))
            )
        for name, value in scores.items():
            self.total.setdefault(name, 0.0)
            self.total_sq.setdefault(name, 0.0)
            self.counts.setdefault(name, 0)
            if math.isnan(value):
                continue
            self.total[name] += value * windows
            self.total_sq[name] += value * value * windows
            self.counts[name] += windows

    def scores(self) -> dict[str, float]:
        """The weighted means, in the order the registry produced the names."""
        return {
            name: total / self.counts[name] if self.counts[name] else math.nan
            for name, total in self.total.items()
        }

    def add_curve(self, per_frame: dict[str, Tensor]) -> None:
        """Fold one batch's per-frame errors into the running curve.

        Summed on the frame axis and divided at the end, so the memory is one
        vector per reduction rather than one per window, and a capped run and a
        full one build the same object.
        """
        for name, values in per_frame.items():
            batch_total = values.sum(dim=0).detach().cpu()
            existing = self.curve_total.get(name)
            self.curve_total[name] = batch_total if existing is None else existing + batch_total
        self.curve_windows += next(iter(per_frame.values())).shape[0]

    def curve(self) -> dict[str, tuple[float, ...]]:
        """Mean error at each predicted frame, one series per reduction."""
        if not self.curve_windows:
            return {}
        return {
            name: tuple((total / self.curve_windows).tolist())
            for name, total in self.curve_total.items()
        }

    def stds(self) -> dict[str, float]:
        """Spread of the per-batch values behind each mean, same weighting.

        **Across scored batches, not across seeds**, and the two answer different
        questions: this one says whether a stage's windows agree with each other,
        while a cross-seed spread says whether a run is reproducible. A variant
        comparison needs the second; a reader deciding how much to trust one
        stage's number needs this one. A single batch reports NaN rather than
        zero, because one sample has no spread to report.

        Computed from a running sum of squares, so it costs one float per metric
        and no second pass. The subtraction is clamped at zero: with values in the
        1e-5 range the cancellation can go slightly negative in float64 and a
        NaN from a negative sqrt would read as a missing metric.
        """
        out: dict[str, float] = {}
        for name, total in self.total.items():
            n = self.counts[name]
            if n <= 1:
                out[name] = math.nan
                continue
            mean = total / n
            variance = max(self.total_sq[name] / n - mean * mean, 0.0)
            out[name] = math.sqrt(variance)
        return out


def stage_cables(cables: list[CableData], stage: str, split: SplitCfg) -> set[int]:
    """Cable ids a stage covers, which the two stages must never share.

    Args:
        cables: The **test** dataset, as loaded by ``load_dataset``.
        stage: A key of :data:`STAGES`.
        split: The split training used.

    Returns:
        The cable ids on this stage's side of the split.

    Raises:
        ValueError: if the stage is not a known name.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; known: {sorted(STAGES)}")
    if stage in ALL_CABLE_STAGES:
        return {cable.cable_id for cable in cables}
    held = stratified_held_out(cables, split)
    held_out = STAGES[stage][1]
    return {cable.cable_id for cable in cables if (cable.cable_id in held) == held_out}


def _truth_windows(
    cables: list[CableData], *, held_out: bool, horizon: int, split: SplitCfg, history_frames: int
) -> Iterator[tuple[WindowKey, Tensor]]:
    """``(key, [1, R, N, 3])`` ground truth, in the loader's window order.

    Walks the enumeration ``eval_windows`` walks, at its default stride of one
    rollout, and reads each window through ``rollout_truth`` rather than slicing
    positions here. The order is what pairs it back onto the loader's stream, and
    ``history_frames`` is part of that order because it sets the first startable
    frame; :func:`evaluate` checks the pairing on every window rather than
    assuming it.

    **The episode and start frame travel with the cable id.** This loop is the
    only place in the project that knows them, and it used to drop them on the
    floor: without them a per-window table can only be joined by position, which
    is silently wrong the moment two arms skip different windows.
    """
    held = stratified_held_out(cables, split)
    for cable in cables:
        if (cable.cable_id in held) != held_out:
            continue
        for episode, (pos, _, _) in enumerate(cable.episodes):
            for start in range(history_frames - 1, len(pos) - horizon, horizon):
                yield (
                    WindowKey(cable.cable_id, episode, start),
                    rollout_truth(cable, episode, start, horizon),
                )


def _check_alignment(batch: Batch, cable_id: int, truth: Tensor, stage: str) -> None:
    """Fail loudly when the truth window is not the one the batch starts.

    Raises:
        RuntimeError: if the cable ids disagree, or frame 0 of the truth window
            is not the position the rollout is started from.
    """
    if int(batch.meta.cable_id[0]) != cable_id:
        raise RuntimeError(
            f"{stage}: window from cable {int(batch.meta.cable_id[0])} paired with ground truth "
            f"from cable {cable_id}"
        )
    if not torch.equal(truth[:, 0], batch.pos):
        raise RuntimeError(
            f"{stage}: frame 0 of the ground-truth window is not the frame the rollout starts "
            f"from, so the two enumerations have drifted apart"
        )


def evaluate(
    model: nn.Module,
    cables: list[CableData],
    *,
    stage: str,
    horizon: int,
    stats: Stats,
    regime: RegimeCfg,
    split: SplitCfg,
    history_frames: int,
    limit: int | None = None,
    progress: bool = False,
    desc: str | None = None,
    stop_flag: Callable[[], bool] | None = None,
) -> EvalResult:
    """Roll out and score one stage of the **test** dataset.

    **The driver follows the model's device**, because an autoregressive rollout
    at horizon 100 over many windows is the most compute-heavy step in the
    project and it is exactly what belongs on the GPU. The loader builds its
    batches on the CPU, so each window and its labels and truth are moved here;
    requiring the caller to do it instead is what made a CUDA model raise
    against this driver.

    Args:
        model: A trained block, callable on a ``Batch``. Left in the training
            mode it arrived in, and its device is the one the stage runs on.
        cables: The **test** dataset. The split then slices it into the two
            stages; handing this the training dataset scores the model on its
            own training drops.
        stage: A key of :data:`STAGES`.
        horizon: Frames ``R`` per window, including the input frame. One long
            rollout per window; ``regime.horizons`` slices it.
        stats: The statistics the checkpoint carries. Never refitted.
        regime: The `[FIX ON PROBE]` thresholds and reported horizons. An
            argument rather than a module constant, so no threshold can be
            picked here after a result is seen.
        split: The split **training** used.
        history_frames: ``C``, the depth **training** used. Windows are built at
            this depth, so a value other than the model's own scores it on
            features it was never trained to read.
        limit: Cap on windows scored. When set, windows are taken round-robin
            across cables rather than from the head of the id-ordered stream.
        progress: When true, wrap the window loop in a tqdm bar.
        desc: Optional tqdm description; defaults to ``eval/{stage}``.
        stop_flag: Optional callable polled between windows. When it returns
            true, scoring stops and a partial result is returned if any window
            completed.

    Returns:
        The stage's count-weighted scores, with the window count behind each.

    Raises:
        ValueError: on an unknown stage, a horizon or limit below 1, a stage
            that yields no window at all, or a model whose input width says it
            was trained at a different ``history_frames``.
        RuntimeError: if a ground-truth window and its batch disagree.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; known: {sorted(STAGES)}")
    if horizon < 1:
        raise ValueError(f"horizon is a frame count and must be at least 1, got {horizon}")
    if limit is not None and limit < 1:
        raise ValueError(f"limit is a window count and must be at least 1, got {limit}")

    # Refused here rather than left to the encoder's matmul, which raises with
    # two shapes and no mention of the history depth that produced them. Arms
    # that declare no width are checked instead by state_from_batch, on the
    # first window.
    declared = getattr(model, "feature_width", None)
    if isinstance(declared, int) and declared != feature_width(history_frames):
        raise ValueError(
            f"this model reads F={declared}, which is not the "
            f"{feature_width(history_frames)} a {history_frames}-frame history builds; "
            "it was trained at a different velocity-history depth"
        )

    stream, held_out = STAGES[stage]
    # Whole-root stages force an empty holdout so the stream and truth walker
    # agree even when training used a non-zero held_out_fraction on another root.
    eval_split = (
        SplitCfg(
            held_out_fraction=0.0,
            d_close_diameters=split.d_close_diameters,
            c_far_segments=split.c_far_segments,
        )
        if stage in ALL_CABLE_STAGES
        else split
    )
    device = model_device(model)
    stats = stats.to(device)
    aggregate = _Aggregate()
    paired = list(
        zip(
            stream(cables, rollout=horizon, cfg=eval_split, history_frames=history_frames),
            _truth_windows(
                cables,
                held_out=held_out,
                horizon=horizon,
                split=eval_split,
                history_frames=history_frames,
            ),
            strict=True,
        )
    )
    if limit is not None:
        paired = stratified_take(paired, limit, cable_id=lambda row: row[1][0].cable_id)

    windows_iter: object = paired
    if progress:
        from tqdm import tqdm

        windows_iter = tqdm(
            paired,
            total=len(paired),
            desc=desc or f"eval/{stage}",
            unit="window",
            leave=False,
        )

    for (batch, labels), (key, truth) in windows_iter:  # type: ignore[misc]
        if stop_flag is not None and stop_flag():
            break
        batch = batch_to(batch, device)
        truth = truth.to(device)
        labels = labels_to(labels, device)
        _check_alignment(batch, key.cable_id, truth, stage)
        predicted = rollout(
            model, batch, horizon=horizon, stats=stats, history_frames=history_frames
        )
        per_frame = per_frame_errors(predicted, truth, batch.meta)
        if predicted.shape[0] != 1:
            raise RuntimeError("keyed windows need B=1")
        frame_errors = {
            name: tuple(values[0].detach().cpu().tolist()) for name, values in per_frame.items()
        }
        aggregate.add_curve(per_frame)
        aggregate.add(
            score(predicted, truth, batch.meta, labels, regime),
            windows=predicted.shape[0],
            key=key,
            frame_errors=frame_errors,
        )

    if aggregate.n_windows == 0:
        raise ValueError(
            f"stage {stage!r} scored no window: it covers cables "
            f"{sorted(stage_cables(cables, stage, split))} at held_out_fraction "
            f"{split.held_out_fraction}, and no episode of those is longer than {horizon} frames"
        )
    return EvalResult(
        stage=stage,
        n_windows=aggregate.n_windows,
        scores=aggregate.scores(),
        limit=limit,
        counts=aggregate.counts,
        stds=aggregate.stds(),
        windows=tuple(aggregate.windows),
        curve=aggregate.curve(),
    )


def evaluate_checkpoint(
    checkpoint: Checkpoint,
    cables: list[CableData],
    *,
    stage: str,
    horizon: int,
    regime: RegimeCfg,
    limit: int | None = None,
    device: str | None = None,
) -> EvalResult:
    """Evaluate a saved run under the statistics, split **and depth** it was trained with.

    The one path that cannot score a held-out cable as seen, or roll a model out
    on a history it never saw, because none of the three is available to be
    passed in.

    Args:
        checkpoint: A run from ``harness.train``.
        cables: The test dataset.
        stage: A key of :data:`STAGES`.
        horizon: Frames per window.
        regime: Thresholds and reported horizons.
        limit: Cap on windows scored.
        device: Where to roll out. The checkpoint's own device if omitted, which
            is wherever ``load_checkpoint`` mapped it.

    Returns:
        The stage's scores.

    Raises:
        ValueError: if the checkpoint carries no split, which a run on a stub
            stream does not.
    """
    if checkpoint.split is None:
        raise ValueError(
            "this checkpoint carries no split, so the stage it was trained against is unknown; "
            "a guessed split scores held-out cables as seen"
        )
    model = checkpoint.restore_model()
    return evaluate(
        model if device is None else model.to(device),
        cables,
        stage=stage,
        horizon=horizon,
        stats=checkpoint.stats,
        regime=regime,
        split=checkpoint.split,
        history_frames=checkpoint.cfg.history_frames,
        limit=limit,
    )


def _report(result: EvalResult) -> None:
    """Print one stage: every scalar with its spread, its unit and its count."""
    cap = f", limit {result.limit}" if result.limit is not None else ""
    print(f"{result.stage}: {result.n_windows} windows{cap}")
    for name, value in result.scores.items():
        std = result.stds.get(name, math.nan)
        unit = unit_of(name)
        suffix = f" {unit}" if unit else ""
        print(f"  {name:<28} {value: .6g} +/- {std:.3g}{suffix}   [{result.counts[name]} windows]")


def _horizons(text: str) -> tuple[int, ...]:
    """Parse ``1,10,50,200`` into horizons, refusing anything unusable.

    A CLI-level parse rather than a bare string, so a typo fails at argument
    parsing instead of producing a table quietly missing the horizon the caller
    came for: ``rollout_rmse`` drops a horizon longer than the rollout without
    saying so, and until this existed the reported set could not be changed at
    all.
    """
    try:
        values = tuple(int(part) for part in text.split(",") if part.strip())
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"horizons must be integers, got {text!r}") from err
    if not values:
        raise argparse.ArgumentTypeError("horizons cannot be empty")
    return values


def main(argv: list[str] | None = None) -> int:
    """Score one checkpoint on one test dataset, both stages by default."""
    defaults = RegimeCfg()
    parser = argparse.ArgumentParser(prog="dlogps.harness.evaluate", description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="a run from harness.train")
    parser.add_argument(
        "--data", type=Path, required=True, help="the TEST dataset root holding cable_*/"
    )
    # max + 1, not max: frame 0 is the input state, so scoring h predicted frames
    # needs h + 1 rollout frames and the longest horizon would otherwise be
    # dropped from the table without saying so.
    parser.add_argument("--horizon", type=int, default=max(defaults.horizons) + 1)
    parser.add_argument(
        "--horizons",
        type=_horizons,
        default=defaults.horizons,
        help="comma-separated predicted-frame counts to report at, e.g. 1,10,50,100,200. Each needs a rollout of h+1, and any beyond it is dropped.",
    )
    parser.add_argument("--limit", type=int, default=None, help="cap windows, for a smoke run")
    parser.add_argument("--stage", choices=[*sorted(STAGES), "both"], default="both")
    parser.add_argument(
        "--device", default="cpu", help="where to roll out; the rollout is the expensive step"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="write eval artifacts and figures into an existing run directory",
    )
    args = parser.parse_args(argv)

    # An entry point rather than a library, so the process-wide thread count is
    # in scope here, and unbounded it costs 202x on this driver.
    bound_intra_op_threads()

    checkpoint = load_checkpoint(args.checkpoint)
    cables = load_dataset(args.data)
    regime = RegimeCfg(horizons=args.horizons)
    stages = sorted(STAGES) if args.stage == "both" else [args.stage]
    rollout_frames = args.horizon
    if 200 in args.horizons and rollout_frames < 201:
        raise ValueError(
            f"horizon 200 needs a rollout of at least 201 frames, got {rollout_frames}"
        )

    # The thresholds are the checkpoint's and cannot be set here, so the run
    # states which ones it scored under: a scorecard that does not name them
    # cannot be checked against the pre-registration.
    split = checkpoint.split
    print(
        f"variant {checkpoint.variant}  step {checkpoint.step}  horizon {args.horizon}  "
        f"C {checkpoint.cfg.history_frames}"
    )
    if split is not None:
        print(
            f"  thresholds from the checkpoint: d_close {split.d_close_diameters} diameters, "
            f"c_far {split.c_far_segments} segments, held-out {split.held_out_fraction}"
        )

    if args.run_dir is not None:
        from dlogps.harness.artifacts import write_eval_artifacts
        from dlogps.harness.plots import write_run_figures

        if split is None:
            raise ValueError(
                "cannot write run artifacts without the checkpoint split; "
                "a guessed split scores held-out cables as seen"
            )
        frame_dt = float(cables[0].dt(0))
        model = checkpoint.restore_model().to(args.device)
        for stage in stages:
            scored = evaluate(
                model,
                cables,
                stage=stage,
                horizon=rollout_frames,
                stats=checkpoint.stats,
                regime=regime,
                split=split,
                history_frames=checkpoint.cfg.history_frames,
                limit=args.limit,
            )
            write_eval_artifacts(
                args.run_dir / "eval" / "model",
                scored,
                step=checkpoint.step,
                horizon=rollout_frames,
                regime=regime,
                cables=cables,
                split=split,
                frame_dt=frame_dt,
            )
            _report(scored)
        write_run_figures(args.run_dir)
        return 0

    for stage in stages:
        _report(
            evaluate_checkpoint(
                checkpoint,
                cables,
                stage=stage,
                horizon=args.horizon,
                regime=regime,
                limit=args.limit,
                device=args.device,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
