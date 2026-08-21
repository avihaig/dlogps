"""The one-step training loop: a stream of batches in, a checkpoint out.

**The loss is on the normalized displacement, and the block is not modified to
get there.** The block already predicts a step and adds ``pos`` back, so the
trainer subtracts ``pos`` off both sides
rather than reaching into a merged interface. Taking the loss on absolute
positions instead would be the quietest defect available here: the numbers are
metres, they are dominated by where the cable happens to be rather than by where
it is going, and the gradient on the millimetre-scale step the model actually has
to predict is then a rounding error inside a metre-scale target.

**Three orderings carry the run, and each is silent when reversed.**

1. Statistics are fitted on a **clean** pass, before any noise, so they describe
   the data rather than the corruption.
2. Per step: draw, noise, **then** normalize. Normalizing before the noise scale
   would measure ``sigma`` in whitened units, which are not the metres per second
   the config key is written in.
3. Statistics are fitted on the **training side of the split only**, which is the
   invariant :mod:`dlogps.harness.normalize` cannot enforce for itself because it
   never opens a file. :func:`dataset_source` is the enforcement: it is the one
   path from a directory to a batch here, and it passes ``held_out=False``.

**One optimizer and one schedule for every variant.** A per-variant learning rate
would confound the structural prior under test with its tuning, the same
confound that applies to ``sigma``. That is why ``sigma`` has no default in this
module: it is one number for the whole matrix, set once and never per variant.

**Wall clock is a deliverable, not decoration.** Every run reports its total and
its per-step time.
"""

from __future__ import annotations

import argparse
import itertools
import math
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import torch
from torch import Tensor

from dlogps import PROJECT_ROOT
from dlogps.data.dataset import (
    DEFAULT_HISTORY_FRAMES,
    CableData,
    feature_width,
    history_frames_from_width,
)
from dlogps.data.loader import SplitCfg, load_dataset, stratified_held_out, training_batches
from dlogps.data.types import Batch
from dlogps.harness.noise import add_random_walk_noise
from dlogps.harness.normalize import PARAM_MODES, Stats, fit_stats, normalize, normalize_step
from dlogps.harness.runtime import batch_to, bound_intra_op_threads
from dlogps.model.bias import BiasPlugin, NoBias
from dlogps.model.gps import GPSCfg, GPSModel
from dlogps.model.structfree import TrainableModel, build_structfree
from dlogps.model.variants import (
    BCOnlyBias,
    ChainDistanceBias,
    MixedHeadBias,
    SpaceDistanceBias,
)

BatchSource = Callable[[int], Iterable[Batch]]
"""An epoch index to one pass over the training stream, on the CPU.

A factory rather than an iterable, for two reasons. A run asks for a fixed number
of *steps*, not epochs, so the loop has to be able to start the stream again; and
the shuffle order is then free to depend on the epoch, which is what keeps a long
run from seeing the same order every pass. The batches are CPU tensors because
the input noise is drawn from a CPU generator and added before the move to
``cfg.device``.
"""

STATS_EPOCH = 0
"""Epoch the statistics pass reads. Fitted before the loop, so the first training
epoch is drawn twice: once clean to fit, once noised to train."""

EVAL_EPOCH = 0
"""Epoch the held-out pass reads. Fixed, so two reports in one run score the same
windows and their difference is the model rather than the draw."""

NOISE_STREAM = 1
"""Offset onto ``cfg.seed`` for the input-noise generator, so the three RNGs a run
drives (weight init, the loader shuffle and the noise) are never handed the
identical integer."""

CHECKPOINT_FORMAT = 2
"""Payload version. A checkpoint that cannot be read is better than one that is
read as something it is not, so the loader refuses an unknown version.

Version 2 adds ``history_frames`` to the config. A version-1 payload is refused
rather than defaulted, because the depth would then be inferred by whichever
default the reading code happens to carry rather than recorded by the run."""

DEFAULT_VARIANT = "A"
"""Variant A, no bias: the control the whole experiment is read against."""


@dataclass(frozen=True)
class Variant:
    """One column of the experiment matrix: a bias plugin plus the block switches.

    Attributes:
        bias: Built with the head count; ``None`` means variant A's :class:`NoBias`.
        use_global: ``False`` is the chain-only message-passing baseline, which is
            a block flag rather than a plugin because the baseline removes the
            attention stream the bias lives in.
    """

    bias: Callable[[int], BiasPlugin] | None = None
    use_global: bool = True


VARIANTS: dict[str, Variant] = {
    "A": Variant(),
    "B": Variant(bias=SpaceDistanceBias),
    "C": Variant(bias=ChainDistanceBias),
    "D": Variant(bias=MixedHeadBias),
    "F": Variant(bias=BCOnlyBias),
    "chain_only": Variant(use_global=False),
}
"""The letters name the paper's variants in run labels and checkpoints
(A = Unbiased, B = Euclidean, C = Chain, D = Mixed, F = Euclidean+Chain only),
plus the one baseline that is reachable from the block alone. The name travels
in the checkpoint, so it is what a result table joins on."""

STANDALONE_VARIANTS = tuple(sorted(name for name in VARIANTS if name != "F"))
"""Variants safe under the standalone CLI's generic six-head defaults (F needs
``--n-heads 8``)."""


@dataclass(frozen=True)
class TrainCfg:
    """Everything a run needs that is not the data or the variant.

    Attributes:
        steps: Optimizer steps. The unit of a run, rather than epochs, because the
            stream length depends on how many cables were generated.
        batch_size: Samples per batch; a short final batch per epoch is trained
            on rather than dropped.
        sigma: Random-walk velocity-noise scale [m/s] at the newest history
            frame, **per axis**. The drift's Euclidean magnitude averages
            ``sqrt(8/pi)`` times larger, and that is the quantity a candidate
            value has to be judged against.
            **No default, here or anywhere in this module.** It is one number for
            the whole matrix, picked once on validation windows and then frozen,
            and a per-variant value would stop the experiment measuring the bias
            and start it measuring the tuning.
        lr: Initial Adam learning rate.
        lr_final: Learning rate at the last step; the decay is exponential
            between the two, GNS-style.
        weight_decay: Adam's, applied to every parameter including the bias rates.
        grad_clip: Global gradient-norm ceiling. On, because an early rollout-free
            model can take one very large step on a contact frame.
        seed: The **only** source of randomness in a run.
        eval_every: Steps between progress reports, and between held-out scores.
        eval_batches: Batches pooled per held-out score. Bounded because the
            report runs inside the training loop; the pooled figure over a few
            thousand windows is what a gate reads, and a per-batch loss is not
            (the spread across batches at initialization already runs 0.06 to
            4.2, so its minimum says nothing about the model).
        device: Torch device string for the model and the loss.
        history_frames: ``C``, frames of velocity history per node. It sets ``F``
            and therefore the width of the block's input encoder, so it is a
            structural hyperparameter rather than a loader detail, and it rides
            in the checkpoint for the same reason ``Stats`` does: a model cannot
            be evaluated except on windows of the depth it was trained at. A
            deeper history also costs
            samples, since the first ``C - 1`` frames of every episode stop being
            window starts.

    Raises:
        ValueError: on a non-positive count, a negative ``sigma``, or a decay
            that is not downward.
    """

    steps: int
    batch_size: int
    sigma: float
    lr: float = 1e-3
    lr_final: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    seed: int = 0
    eval_every: int = 100
    eval_batches: int = 32
    device: str = "cpu"
    history_frames: int = DEFAULT_HISTORY_FRAMES
    param_encoding: str = "designed"
    """How the four material channels are scaled: ``designed`` (the paper) maps
    them by ``log10`` and the fixed affine spans in
    :mod:`dlogps.harness.normalize`; ``fitted`` z-scores them by training
    statistics instead. Rides in the checkpoint via ``Stats.param_mode``."""

    def __post_init__(self) -> None:
        if self.param_encoding not in PARAM_MODES:
            raise ValueError(
                f"param_encoding must be one of {PARAM_MODES}, got {self.param_encoding!r}"
            )
        if self.eval_batches < 1:
            raise ValueError(
                f"eval_batches is a count and must be positive, got {self.eval_batches}"
            )
        if self.history_frames < 1:
            raise ValueError(
                f"history_frames is a frame count and must be at least 1, got {self.history_frames}"
            )
        if self.steps < 1 or self.batch_size < 1 or self.eval_every < 1:
            raise ValueError(
                f"steps, batch_size and eval_every are counts and must be positive, got "
                f"steps={self.steps}, batch_size={self.batch_size}, eval_every={self.eval_every}"
            )
        if self.sigma < 0:
            raise ValueError(
                f"sigma is a standard deviation and cannot be negative, got {self.sigma}"
            )
        if self.lr <= 0 or self.lr_final <= 0:
            raise ValueError(f"lr and lr_final must be positive, got {self.lr} and {self.lr_final}")
        if self.lr_final > self.lr:
            raise ValueError(
                f"the schedule decays, so lr_final <= lr; got {self.lr_final} > {self.lr}"
            )
        if self.grad_clip <= 0:
            raise ValueError(
                f"grad_clip is a norm ceiling and must be positive, got {self.grad_clip}"
            )
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay cannot be negative, got {self.weight_decay}")


@dataclass(frozen=True)
class TrainResult:
    """One finished run: the model, its input scaling, and what it cost.

    Attributes:
        model: The trained block, left on ``cfg.device``.
        stats: The statistics the run was trained under. The same object has to
            reach rollout, so it travels into the checkpoint rather than being
            refitted.
        cfg: The config the run used.
        variant: Its key in :data:`VARIANTS`.
        losses: The per-step loss trace, on the **noised** stream and one batch
            at a time. A progress signal, never a result: read ``eval_losses``.
        eval_losses: ``(step, pooled loss)`` on the held-out side of the training
            split, clean and in eval mode. The only generalization signal a run
            produces without a rollout, and the figure a gate reads.
        seconds: Wall clock over the optimizer loop, excluding the load and the
            statistics pass.
        split: The split the training side was taken from, or ``None`` for a run
            on a stub stream. The evaluation driver must use the same one or it
            scores held-out cables as seen.
    """

    model: TrainableModel
    stats: Stats
    cfg: TrainCfg
    variant: str
    losses: tuple[float, ...]
    seconds: float
    split: SplitCfg | None = None
    eval_losses: tuple[tuple[int, float], ...] = ()

    @property
    def final_eval_loss(self) -> float:
        """The last pooled held-out loss, or NaN if the run scored none.

        Against the 1.0 line: 1.0 is the constant predictor, below it is real
        prediction, above it is worse than predicting the mean step.
        """
        return self.eval_losses[-1][1] if self.eval_losses else float("nan")

    @property
    def steps(self) -> int:
        """Steps actually taken, which is the length of the trace."""
        return len(self.losses)

    @property
    def seconds_per_step(self) -> float:
        """The number the compute budget is set from."""
        return self.seconds / max(self.steps, 1)


@dataclass(frozen=True)
class Checkpoint:
    """A run, reloadable. Everything needed to reproduce one prediction.

    Attributes:
        state_dict: The block's parameters and buffers.
        stats: The six fitted tensors. A checkpoint that cannot reproduce its own
            input scaling is not a checkpoint.
        cfg: The :class:`TrainCfg` that produced it.
        model_cfg: The block config, which carries the chain-only baseline's
            ``use_global=False`` and the crossed local-stream ablation.
        variant: Its key in :data:`VARIANTS`.
        feature_width: ``F``. The loader sets it and the history depth moves
            it, so rebuilding the block needs it recorded rather than assumed.
        step: Steps trained.
        split: The split training used, or ``None``.
        git_sha: The code that produced it, or ``None`` when git cannot say.
        seconds: The run's wall clock.

    Raises:
        ValueError: if ``feature_width`` is not the width ``cfg.history_frames``
            produces, which is a payload whose two records of the input layout
            disagree.
    """

    state_dict: dict[str, Tensor]
    stats: Stats
    cfg: TrainCfg
    model_cfg: GPSCfg
    variant: str
    feature_width: int
    step: int
    split: SplitCfg | None = None
    git_sha: str | None = None
    seconds: float = 0.0

    def __post_init__(self) -> None:
        # Two independent records of the same layout, so they are checked against
        # each other rather than one being trusted: restore_model builds the
        # encoder from feature_width while every window is built from
        # history_frames, and a disagreement is a model fed the wrong features.
        expected = feature_width(self.cfg.history_frames)
        if self.feature_width != expected:
            raise ValueError(
                f"this checkpoint records F={self.feature_width} but a "
                f"{self.cfg.history_frames}-frame history makes F={expected}; "
                "it cannot be evaluated at either depth"
            )

    def restore_model(self) -> TrainableModel:
        """Rebuild the block and load the weights, in eval mode.

        Returns:
            A model that reproduces the trained one's prediction on any batch.
        """
        model = build_model(self.feature_width, self.variant, self.model_cfg)
        model.load_state_dict(self.state_dict)
        model.eval()
        return model


def build_model(feature_width: int, variant: str, cfg: GPSCfg | None = None) -> TrainableModel:
    """Build the model for a named variant, or a structure-free arm.

    The one place a name becomes a model, shared by the trainer, the CLI and the
    checkpoint loader, so a name cannot mean two different models.

    ``cfg.arch`` selects the family and is checked first: the structure-free
    floor arms have no bias slot, so ``variant`` is not theirs to honour and is
    ignored rather than silently pretended at. The run config still records it,
    which is why an arm's resolved config reads ``variant: A`` and means nothing
    by it.

    Args:
        feature_width: ``F``, read off ``node_feat`` by the caller. ``C`` is
            recovered from it for the arms that need a sequence length, rather
            than threaded through three call sites that do not.
        variant: A key of :data:`VARIANTS`. Ignored when ``cfg.arch`` is not
            ``gps``.
        cfg: Model hyper-parameters; the defaults if omitted. A variant may only
            switch a stream **off**, never back on, so a caller running the
            crossed local-stream ablation keeps it.

    Returns:
        An untrained model: a :class:`GPSModel` holding its bias plugin as a
        submodule, or a structure-free arm.

    Raises:
        ValueError: if the variant or the arch is not a known name.
    """
    base = cfg or GPSCfg()
    if base.arch != "gps":
        return build_structfree(
            base.arch, feature_width, base, history_frames_from_width(feature_width)
        )

    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; known: {sorted(VARIANTS)}")

    spec = VARIANTS[variant]
    block = replace(base, use_global=base.use_global and spec.use_global)
    bias = spec.bias(block.n_heads) if spec.bias is not None else NoBias()
    return GPSModel(feature_width, block, bias)


def step_loss(model: TrainableModel, batch: Batch, stats: Stats) -> Tensor:
    """Mean squared error on the **normalized displacement**.

    Both sides have ``pos`` subtracted off before scaling, so the quantity scored
    is the step the model was built to predict rather than the absolute position
    it adds that step to.

    ``batch.pos`` is the noised position and ``batch.target`` is ground truth, so
    the displacement the model has to produce includes the correction of its own
    drift. That asymmetry is the whole point of the noise regime.

    Args:
        model: The block.
        batch: A **noised and normalized** batch, on the model's device.
        stats: The run's statistics, on the same device.

    Returns:
        A scalar, in normalized units.
    """
    predicted = normalize_step(model(batch) - batch.pos, stats)
    observed = normalize_step(batch.target - batch.pos, stats)
    return torch.nn.functional.mse_loss(predicted, observed)


def dataset_source(
    cables: list[CableData], cfg: TrainCfg, split: SplitCfg | None = None
) -> BatchSource:
    """A :data:`BatchSource` over the **training side** of a loaded dataset.

    ``held_out=False`` is written here and nowhere else, which is what makes the
    statistics leak unreachable from this module: every path from a directory to
    a batch runs through this function, and the statistics pass reads the same
    source the loop trains on.

    Args:
        cables: Loaded cables, from ``load_dataset``.
        cfg: The run config; supplies the batch size and the shuffle seed.
        split: Split config; the loader's defaults if omitted.

    Returns:
        A callable from epoch index to one shuffled pass.
    """

    def source(epoch: int) -> Iterable[Batch]:
        return training_batches(
            cables,
            batch_size=cfg.batch_size,
            history_frames=cfg.history_frames,
            cfg=split,
            held_out=False,
            seed=cfg.seed + epoch,
        )

    return source


def held_out_source(
    cables: list[CableData], cfg: TrainCfg, split: SplitCfg | None = None
) -> BatchSource:
    """A :data:`BatchSource` over the **held-out** side, for scoring only.

    Deliberately a second function rather than a flag on :func:`dataset_source`:
    that function's single ``held_out=False`` is what makes a statistics leak
    unreachable, and a parameter there would put the leak one argument away.
    Nothing fitted is ever computed from this stream. :func:`train` consumes it
    under ``no_grad`` and in eval mode, and fits its statistics from the training
    source alone.

    Args:
        cables: Loaded cables, from ``load_dataset``.
        cfg: The run config; supplies the batch size and the shuffle seed.
        split: Split config; the loader's defaults if omitted.

    Returns:
        A callable from epoch index to one shuffled pass over the held-out side.
    """

    def source(epoch: int) -> Iterable[Batch]:
        return training_batches(
            cables,
            batch_size=cfg.batch_size,
            history_frames=cfg.history_frames,
            cfg=split,
            held_out=True,
            seed=cfg.seed + epoch,
        )

    return source


def pooled_loss(
    model: TrainableModel,
    source: BatchSource,
    cfg: TrainCfg,
    stats: Stats,
    *,
    epoch: int = EVAL_EPOCH,
) -> float:
    """Sample-weighted mean loss over a bounded pass, clean and in eval mode.

    **Pooled, because a per-batch loss is not a measurement of the model.** The
    spread across batches at initialization runs 0.06 to 4.2 on real cables, so a
    minimum over the trace sits far below the honest figure whatever the model
    has learned. Weighted by samples, so a short final batch does not count as
    much as a full one.

    No noise: the corruption is a training device, and the question here is how
    the model does on the data.

    Args:
        model: The block. Handed back in the training mode it arrived in.
        source: The stream to score, typically :func:`held_out_source`.
        cfg: Supplies ``eval_batches`` and the device.
        stats: The run's statistics. Never refitted here.
        epoch: Which pass to draw, fixed so two reports score the same windows.

    Returns:
        The pooled loss, against the 1.0 constant-predictor line.
    """
    device = torch.device(cfg.device)
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    try:
        with torch.no_grad():
            for batch in itertools.islice(source(epoch), cfg.eval_batches):
                scaled = normalize(batch_to(batch, device), stats)
                samples = batch.pos.shape[0]
                total += float(step_loss(model, scaled, stats)) * samples
                count += samples
    finally:
        model.train(was_training)
    return total / count if count else float("nan")


def _epochs(source: BatchSource) -> Iterator[Batch]:
    """Batches without end, restarting the stream at each epoch boundary.

    Raises:
        ValueError: if an epoch yields nothing, which would otherwise spin.
    """
    for epoch in itertools.count():
        empty = True
        for batch in source(epoch):
            empty = False
            yield batch
        if empty:
            raise ValueError(f"the batch source yielded no batches at epoch {epoch}")


def decay_gamma(cfg: TrainCfg) -> float:
    """Per-step factor taking ``lr`` to ``lr_final`` over the run.

    The exponent is ``steps - 1`` so the **last** step runs at exactly
    ``lr_final``; over ``steps`` it would stop one factor short and the schedule
    would silently depend on the run length.
    """
    if cfg.steps < 2:
        return 1.0
    return float((cfg.lr_final / cfg.lr) ** (1.0 / (cfg.steps - 1)))


def train(
    source: BatchSource,
    cfg: TrainCfg,
    *,
    variant: str = DEFAULT_VARIANT,
    model_cfg: GPSCfg | None = None,
    checkpoint_at: Iterable[int] = (),
    on_checkpoint: Callable[[int, TrainResult], None] | None = None,
    split: SplitCfg | None = None,
    eval_source: BatchSource | None = None,
    on_step: Callable[[int, float], None] | None = None,
    on_eval: Callable[[int, float], None] | None = None,
    on_best_checkpoint: Callable[[int, float, TrainResult], None] | None = None,
    progress: bool = False,
    interruptible: bool = False,
) -> TrainResult:
    """Train one model for ``cfg.steps`` optimizer steps.

    Deterministic from ``cfg.seed`` alone: it seeds torch globally before the
    weights are drawn, the noise generator, and the shuffle inside
    :func:`dataset_source`. Two runs at one seed produce bit-identical traces.

    Args:
        source: The training stream, on the CPU. Consumed once per epoch and
            restarted as many times as ``cfg.steps`` requires.
        cfg: The run config.
        variant: A key of :data:`VARIANTS`.
        model_cfg: Block hyper-parameters; the defaults if omitted.
        checkpoint_at: Steps at which to hand a snapshot to ``on_checkpoint``.
            **A set of explicit steps, not a period.** The grid wants 6000,
            20000 and 40000, and no single period emits exactly those: their
            greatest common divisor is 2000, so a period small enough to hit all
            three writes twenty files a run.
        on_checkpoint: Called with ``(step, snapshot)`` at each of those steps.
            The snapshot shares the live model rather than copying it, so the
            callback has to write it before returning.
        split: Recorded on the result and in the checkpoint, for the evaluation
            driver. ``None`` for a stub stream, which has no split.
        eval_source: The **held-out** stream, scored pooled every
            ``cfg.eval_every`` steps. Statistics are never fitted from it; it is
            read under ``no_grad`` in eval mode. ``None`` skips the scoring, which
            is what a stub stream with no split does.
        on_step: Called with ``(step, loss)`` every ``cfg.eval_every`` steps and
            on the last step. The progress hook the CLI prints from.
        on_eval: Called with ``(step, pooled held-out loss)`` at the same points,
            when ``eval_source`` is given.
        on_best_checkpoint: Called when the pooled held-out loss is strictly
            better than every previous report (and on the first report). Same
            live-model snapshot contract as ``on_checkpoint``.
        progress: When true, wrap the step loop in a tqdm bar (and print loss on
            each report). Off by default so library callers stay quiet.
        interruptible: When true, the first SIGINT / Ctrl+C finishes the current
            step, writes a checkpoint at that step, and returns. A second press
            aborts hard. Off by default so library callers still see a raw
            ``KeyboardInterrupt``.

    Returns:
        The trained model, its statistics, both loss traces and the wall clock.
        After an early stop the loss trace length is the step reached, which may
        be below ``cfg.steps``.

    Raises:
        ValueError: if the source yields no batches.
        KeyboardInterrupt: on a second interrupt while ``interruptible``, or on
            the first when ``interruptible`` is false.
    """
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)

    wanted = sorted(set(checkpoint_at))
    if wanted and on_checkpoint is None:
        raise ValueError("checkpoint_at was given with no on_checkpoint to receive the snapshots")
    beyond = [step for step in wanted if step > cfg.steps]
    if beyond:
        # Silently skipping these is how a grid ends up with a missing arm and a
        # summary table that still looks complete.
        raise ValueError(
            f"checkpoint_at asks for steps {beyond} but the run is {cfg.steps} steps long"
        )
    # A set, and materialised here rather than trusted: the loop tests membership
    # once per step, and a generator would be exhausted after the first hit.
    snapshot_steps = set(wanted)

    # Clean pass, before any noise: the statistics describe the data, not the
    # corruption.
    stats = fit_stats(source(STATS_EPOCH), param_mode=cfg.param_encoding).to(device)

    probe = next(iter(source(STATS_EPOCH)), None)
    if probe is None:
        raise ValueError("the batch source yielded no batches, so there is nothing to train on")

    # The stream's layout and the config's have to agree before a single step is
    # taken: the encoder is built from the stream's width and every later pass
    # reads the history back at cfg.history_frames, so a mismatch trains happily
    # and produces a checkpoint that cannot be loaded.
    width = probe.node_feat.shape[-1]
    if width != feature_width(cfg.history_frames):
        raise ValueError(
            f"the source yields node_feat of width {width}, but cfg.history_frames="
            f"{cfg.history_frames} makes it {feature_width(cfg.history_frames)}"
        )
    model = build_model(width, variant, model_cfg).to(device)

    # The head predicts the standardized step, so it needs the same scale the
    # loss divides by or its output means nothing (``gps.py``, the head buffers).
    # They ride in the state_dict, so a checkpoint carries the scaling it was
    # trained under and cannot be reloaded against another run's.
    with torch.no_grad():
        model.step_std.copy_(stats.step_std)
        model.step_mean.copy_(stats.step_mean)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_gamma(cfg))
    # The walk is drawn from a CPU generator, so noise is added before the move
    # to device rather than after.
    noise = torch.Generator().manual_seed(cfg.seed + NOISE_STREAM)

    model.train()
    losses: list[float] = []
    eval_losses: list[tuple[int, float]] = []
    started = time.perf_counter()
    step_stream = itertools.islice(_epochs(source), cfg.steps)
    progress_bar = None
    if progress:
        from tqdm import tqdm

        progress_bar = tqdm(step_stream, total=cfg.steps, desc="train", unit="step", leave=True)
        step_stream = progress_bar

    stop_requested = False
    previous_handler = signal.getsignal(signal.SIGINT)

    def _request_stop(signum: int, frame: object) -> None:
        nonlocal stop_requested
        if stop_requested:
            # Second Ctrl+C: hard abort rather than waiting on eval.
            raise KeyboardInterrupt
        stop_requested = True
        print(
            "\nCtrl+C: early stop after this step, then rollout eval on saved checkpoints. "
            "Press Ctrl+C again to abort hard.",
            flush=True,
        )

    if interruptible:
        signal.signal(signal.SIGINT, _request_stop)

    step = 0
    best_eval = math.inf
    try:
        for index, raw in enumerate(step_stream):
            noised = add_random_walk_noise(raw, cfg.sigma, noise, cfg.history_frames)
            batch = normalize(batch_to(noised, device), stats)
            loss = step_loss(model, batch, stats)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            schedule.step()

            losses.append(float(loss.detach()))
            step = index + 1
            if on_checkpoint is not None and step in snapshot_steps:
                # The same live model, stats and split the final checkpoint will
                # carry: a mid-run snapshot has to be scorable against the same
                # scaling and the same stage, or it measures a different thing from
                # the run it belongs to.
                on_checkpoint(
                    step,
                    TrainResult(
                        model=model,
                        stats=stats,
                        cfg=cfg,
                        variant=variant,
                        losses=tuple(losses),
                        seconds=time.perf_counter() - started,
                        split=split,
                        eval_losses=tuple(eval_losses),
                    ),
                )
            if step % cfg.eval_every == 0 or step == cfg.steps:
                if on_step is not None:
                    on_step(step, losses[-1])
                if eval_source is not None:
                    pooled = pooled_loss(model, eval_source, cfg, stats)
                    eval_losses.append((step, pooled))
                    if on_eval is not None:
                        on_eval(step, pooled)
                    if on_best_checkpoint is not None and pooled < best_eval:
                        best_eval = pooled
                        on_best_checkpoint(
                            step,
                            pooled,
                            TrainResult(
                                model=model,
                                stats=stats,
                                cfg=cfg,
                                variant=variant,
                                losses=tuple(losses),
                                seconds=time.perf_counter() - started,
                                split=split,
                                eval_losses=tuple(eval_losses),
                            ),
                        )
                if progress_bar is not None:
                    postfix: dict[str, object] = {"train": f"{losses[-1]:.4f}"}
                    if eval_losses and eval_losses[-1][0] == step:
                        postfix["eval"] = f"{eval_losses[-1][1]:.4f}"
                    progress_bar.set_postfix(postfix, refresh=False)
            if stop_requested:
                break
    finally:
        if interruptible:
            signal.signal(signal.SIGINT, previous_handler)

    if stop_requested and step >= 1 and on_checkpoint is not None and step not in snapshot_steps:
        # Mid-schedule early stop: write the weights at the step we reached so
        # the caller can still score a complete eval pass.
        print(f"early stop at step {step}; writing checkpoint", flush=True)
        on_checkpoint(
            step,
            TrainResult(
                model=model,
                stats=stats,
                cfg=cfg,
                variant=variant,
                losses=tuple(losses),
                seconds=time.perf_counter() - started,
                split=split,
                eval_losses=tuple(eval_losses),
            ),
        )
    elif stop_requested and step >= 1:
        print(f"early stop at step {step} (checkpoint already on disk)", flush=True)
    elif stop_requested:
        print("early stop before any step completed; nothing to evaluate", flush=True)

    return TrainResult(
        model=model,
        stats=stats,
        cfg=cfg,
        variant=variant,
        losses=tuple(losses),
        seconds=time.perf_counter() - started,
        split=split,
        eval_losses=tuple(eval_losses),
    )


def train_on_dataset(
    root: str | Path,
    cfg: TrainCfg,
    *,
    variant: str = DEFAULT_VARIANT,
    model_cfg: GPSCfg | None = None,
    checkpoint_at: Iterable[int] = (),
    on_checkpoint: Callable[[int, TrainResult], None] | None = None,
    split: SplitCfg | None = None,
    on_step: Callable[[int, float], None] | None = None,
    on_eval: Callable[[int, float], None] | None = None,
    on_best_checkpoint: Callable[[int, float, TrainResult], None] | None = None,
    progress: bool = False,
    interruptible: bool = False,
    eval_root: str | Path | None = None,
) -> TrainResult:
    """Train on a dataset directory, statistics fitted on the training side only.

    The held-out side of the same split is scored pooled at every report, which
    is the run's only generalization signal short of a rollout. When
    ``held_out_fraction`` is zero, pass ``eval_root`` (a dedicated test set) so
    one-step eval still has a stream; otherwise eval is skipped.

    Args:
        root: Dataset root holding ``cable_*/``.
        cfg: The run config.
        variant: A key of :data:`VARIANTS`.
        model_cfg: Block hyper-parameters; the defaults if omitted.
        checkpoint_at: Steps at which to snapshot, forwarded to :func:`train`.
        on_checkpoint: Receives ``(step, snapshot)`` at each of those steps.
        split: Split config; the loader's defaults if omitted.
        on_step: Progress hook, as in :func:`train`.
        on_eval: Held-out hook, as in :func:`train`.
        on_best_checkpoint: Best one-step held-out snapshot hook, as in :func:`train`.
        progress: Forwarded to :func:`train`; tqdm bar when true.
        interruptible: Forwarded to :func:`train`; soft Ctrl+C early stop.
        eval_root: Optional second root used for one-step eval when the train
            split holds no cables out.

    Returns:
        The finished run, carrying the split it used.
    """
    cables = load_dataset(root)
    split = split or SplitCfg()
    held = stratified_held_out(cables, split)
    if held:
        eval_source = held_out_source(cables, cfg, split)
    elif eval_root is not None:
        eval_cables = load_dataset(eval_root)
        eval_split = SplitCfg(
            held_out_fraction=0.0,
            d_close_diameters=split.d_close_diameters,
            c_far_segments=split.c_far_segments,
        )
        eval_source = dataset_source(eval_cables, cfg, eval_split)
    else:
        eval_source = None
    return train(
        dataset_source(cables, cfg, split),
        cfg,
        variant=variant,
        model_cfg=model_cfg,
        checkpoint_at=checkpoint_at,
        on_checkpoint=on_checkpoint,
        split=split,
        eval_source=eval_source,
        on_step=on_step,
        on_eval=on_eval,
        on_best_checkpoint=on_best_checkpoint,
        progress=progress,
        interruptible=interruptible,
    )


def git_sha(repo: Path = PROJECT_ROOT) -> str | None:
    """The current commit, or ``None`` if git cannot say.

    Provenance is worth a subprocess and is not worth a failed run, so a missing
    git, a missing repo and a failing call all degrade to ``None``.
    """
    try:
        done = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() or None


def save_checkpoint(path: str | Path, result: TrainResult) -> Path:
    """Write a run to disk as plain tensors and primitives.

    Dataclasses are flattened to dictionaries rather than pickled whole, so the
    file loads under ``weights_only=True`` and reading a checkpoint never
    executes what wrote it.

    Args:
        path: Destination file; parent directories are created.
        result: The finished run.

    Returns:
        The path written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "state_dict": result.model.state_dict(),
        "stats": {f.name: getattr(result.stats, f.name) for f in fields(Stats)},
        "cfg": asdict(result.cfg),
        "model_cfg": asdict(result.model.cfg),
        "split": asdict(result.split) if result.split is not None else None,
        "variant": result.variant,
        "feature_width": result.model.feature_width,
        "step": result.steps,
        "git_sha": git_sha(),
        "seconds": result.seconds,
    }
    torch.save(payload, destination)
    return destination


def load_checkpoint(path: str | Path, *, map_location: str = "cpu") -> Checkpoint:
    """Read a checkpoint written by :func:`save_checkpoint`.

    Args:
        path: The file.
        map_location: Device the tensors land on.

    Returns:
        The reloadable run.

    Raises:
        ValueError: if the payload version is not this module's, or if the
            payload's two records of the input layout disagree.
    """
    payload = torch.load(path, map_location=map_location, weights_only=True)
    version = payload.get("format") if isinstance(payload, dict) else None
    if version != CHECKPOINT_FORMAT:
        raise ValueError(f"{path}: checkpoint format {version!r}, expected {CHECKPOINT_FORMAT}")
    split = payload["split"]
    return Checkpoint(
        state_dict=payload["state_dict"],
        stats=Stats(**payload["stats"]),
        cfg=TrainCfg(**payload["cfg"]),
        model_cfg=GPSCfg(**payload["model_cfg"]),
        variant=payload["variant"],
        feature_width=payload["feature_width"],
        step=payload["step"],
        split=SplitCfg(**split) if split is not None else None,
        git_sha=payload["git_sha"],
        seconds=payload["seconds"],
    )


def main(argv: list[str] | None = None) -> int:
    """Train one variant on one dataset and write one checkpoint."""
    parser = argparse.ArgumentParser(prog="dlogps.harness.train", description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="dataset root holding cable_*/")
    parser.add_argument("--out", type=Path, required=True, help="checkpoint path to write")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument(
        "--sigma",
        type=float,
        required=True,
        help="velocity-noise scale [m/s]; one number for the whole matrix, so it has no default",
    )
    parser.add_argument("--variant", choices=STANDALONE_VARIANTS, default=DEFAULT_VARIANT)
    parser.add_argument(
        "--checkpoint-at",
        type=int,
        action="append",
        default=None,
        metavar="STEP",
        help=(
            "also write a checkpoint at this step; repeat for several. "
            "<out>.step{N}.pt beside the final file. Explicit steps rather than a "
            "period, because no period emits 6000, 20000 and 40000 without also "
            "emitting seventeen others."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-final", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--history-frames",
        type=int,
        default=DEFAULT_HISTORY_FRAMES,
        help="C, frames of velocity history per node; recorded in the checkpoint, and a "
        "checkpoint can only be evaluated at the depth it was trained at",
    )
    parser.add_argument(
        "--no-local-stream",
        action="store_true",
        help="the crossed ablation: run the attention stream without the chain MPNN",
    )
    # The split and its two thresholds are chosen here, once, and ride in the
    # checkpoint. Nothing downstream can change them, which is what makes the
    # gate that they are fixed before any result is seen enforceable.
    split = SplitCfg()
    parser.add_argument("--held-out-fraction", type=float, default=split.held_out_fraction)
    parser.add_argument(
        "--d-close-diameters",
        type=float,
        default=split.d_close_diameters,
        help="contact threshold in cable diameters; recorded in the checkpoint",
    )
    parser.add_argument(
        "--c-far-segments",
        type=int,
        default=split.c_far_segments,
        help="chain distance beyond which a pair counts as far; recorded in the checkpoint",
    )
    args = parser.parse_args(argv)

    # An entry point rather than a library, so the process-wide thread count is
    # in scope here, and unbounded it costs 23x per step at batch 512.
    threads = bound_intra_op_threads()

    cfg = TrainCfg(
        steps=args.steps,
        batch_size=args.batch_size,
        sigma=args.sigma,
        lr=args.lr,
        lr_final=args.lr_final,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        seed=args.seed,
        eval_every=args.eval_every,
        device=args.device,
        history_frames=args.history_frames,
    )
    split = SplitCfg(
        held_out_fraction=args.held_out_fraction,
        d_close_diameters=args.d_close_diameters,
        c_far_segments=args.c_far_segments,
    )
    print(
        f"variant {args.variant}  {cfg.steps} steps  sigma {cfg.sigma}  seed {cfg.seed}  "
        f"C {cfg.history_frames}"
    )
    print(
        f"  split: held-out {split.held_out_fraction}, d_close {split.d_close_diameters} "
        f"diameters, c_far {split.c_far_segments} segments"
    )

    def write_snapshot(step: int, snapshot: TrainResult) -> None:
        """Write a mid-run checkpoint beside the final one, named by its step."""
        path = args.out.with_suffix(f".step{step:06d}{args.out.suffix}")
        print(f"  step {step}/{cfg.steps}  checkpoint {save_checkpoint(path, snapshot)}")

    result = train_on_dataset(
        args.data,
        cfg,
        variant=args.variant,
        model_cfg=GPSCfg(use_local=not args.no_local_stream),
        split=split,
        checkpoint_at=args.checkpoint_at or (),
        on_checkpoint=write_snapshot,
        on_step=lambda step, loss: print(f"  step {step}/{cfg.steps}  batch loss {loss:.6e}"),
        on_eval=lambda step, loss: print(f"  step {step}/{cfg.steps}  held-out pooled {loss:.6e}"),
    )
    print(
        f"trained {result.steps} steps in {result.seconds:.1f} s "
        f"({result.seconds_per_step * 1e3:.1f} ms/step, {threads} intra-op threads)"
    )
    # The pooled held-out figure is the one a gate reads: 1.0 is the constant
    # predictor, and a per-batch training loss sits far below it whatever the
    # model has learned.
    print(f"held-out pooled loss {result.final_eval_loss:.6e}  (1.0 is the constant predictor)")
    print(f"checkpoint {save_checkpoint(args.out, result)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
