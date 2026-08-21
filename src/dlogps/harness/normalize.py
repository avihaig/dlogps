"""Feature statistics: fitted once on the training split, carried in the checkpoint.

**The two invariants this file exists to hold**:

1. Statistics are fitted on the **training side of the training split** and
   nowhere else. Fit them on everything and the held-out cables leak into the
   model's input scaling, and the leak shows up in no downstream number: every
   metric still runs, every table still prints, and the out-of-distribution
   column is quietly optimistic. The enforcement lives at the call site, which is
   why :func:`fit_stats` takes an iterable of batches rather than a path: it
   cannot open the wrong file because it cannot open a file.
2. ``pos`` is **never** scaled. The bias plugins read ``pos`` and compare real
   distances in metres against ``meta.diameter`` and ``meta.length``, so scaling
   it rewrites the 3D-distance variant's metric into arbitrary units. That
   failure does not crash; it reads as "the spatial prior did not help", which is
   one of the answers the experiment is trying to distinguish.

**Statistics are over the displacement, not the position.** Absolute positions
have no meaningful mean, since the cable is wherever it was dropped, while the
per-step displacement does. It is also what the model actually produces: the
block predicts a step and adds ``pos`` back.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor

from dlogps.data.types import Batch

UNIT_STD_BELOW = 1e-6
"""Standard deviations under this are replaced by exactly 1.0, not floored at
the epsilon. A near-constant channel carries no information, so the honest
transform is to leave it alone; dividing by 1e-6 would instead amplify its
float32 rounding noise into the largest feature the model sees."""

MATERIAL_CHANNELS = slice(-5, -1)
"""Where the four material parameters sit in ``node_feat``: the layout ends
``[..., material(4), floor(1)]`` (``dataset.features_from_state``), so negative
indexing finds them at any history depth."""

PARAM_MODES = ("designed", "fitted")
"""``designed`` (the paper, and the default): the material channels are encoded
by the fixed log10 spans below INSTEAD of z-scoring (``docs/method.md``); all
other channels stay fitted. ``fitted``: the material channels are z-scored by
training statistics like every other channel."""

PARAM_LOG_CENTERS = (7.0, -3.0, -2.301, -1.3495)
PARAM_LOG_HALVES = (3.0, 2.0, 1.0, 2.6505)
"""The designed encoding, ``(log10(x) - center) / half`` per channel, in
``dataset.MATERIAL_FIELDS`` order (bend_stiffness, joint_damping, diameter,
effective_linear_density). Spans map [1e4, 1e10] Pa, [1e-5, 1e-1], [0.5, 50] mm
and [1e-4, 20] kg/m onto [-1, +1]. Values outside a span encode past +/-1
linearly, never clamped: clamping would recreate the collapse the encoding
exists to remove. Constants of the spec, not knobs."""


def encode_params(material: Tensor) -> Tensor:
    """The designed encoding of a ``[..., 4]`` raw material slice.

    Inputs are positive by the population parser's contract, so ``log10`` is
    total on the domain this ever sees.
    """
    centers = torch.tensor(PARAM_LOG_CENTERS, dtype=material.dtype, device=material.device)
    halves = torch.tensor(PARAM_LOG_HALVES, dtype=material.dtype, device=material.device)
    return (material.log10() - centers) / halves


@dataclass(frozen=True)
class Stats:
    """Per-channel mean and standard deviation, fitted once and then frozen.

    Attributes:
        node_mean: ``[F]`` over ``node_feat``, pooled across batch and node.
        node_std: ``[F]``.
        edge_mean: ``[5]`` over ``edge_feat``, pooled across batch and edge.
        edge_std: ``[5]``.
        step_mean: ``[3]`` over ``target - pos``, the per-frame displacement.
        step_std: ``[3]``.

    Travels inside the checkpoint. A checkpoint that cannot reproduce its own
    input scaling cannot be evaluated, because the rollout loop must present the
    model with features scaled exactly as training did.
    """

    node_mean: Tensor
    node_std: Tensor
    edge_mean: Tensor
    edge_std: Tensor
    step_mean: Tensor
    step_std: Tensor
    param_mode: str = "designed"

    def __post_init__(self) -> None:
        if self.param_mode not in PARAM_MODES:
            raise ValueError(f"param_mode must be one of {PARAM_MODES}, got {self.param_mode!r}")

    def to(self, device: torch.device | str) -> Stats:
        """The same statistics on another device."""
        return Stats(*(t.to(device) for t in self.tensors()), param_mode=self.param_mode)

    def tensors(self) -> tuple[Tensor, ...]:
        """The six tensors in field order, for serialization and device moves."""
        return (
            self.node_mean,
            self.node_std,
            self.edge_mean,
            self.edge_std,
            self.step_mean,
            self.step_std,
        )


class _Accumulator:
    """Streaming count, sum and sum of squares over the trailing axis.

    ``float64`` throughout: the sum of squares over a few million frames of
    ``float32`` loses the low bits of the variance, and the variance is the thing
    being measured.
    """

    def __init__(self) -> None:
        self.count = 0
        self.total: Tensor | None = None
        self.total_sq: Tensor | None = None

    def add(self, values: Tensor) -> None:
        """Fold in ``[..., K]``, pooling every leading axis."""
        flat = values.reshape(-1, values.shape[-1]).double()
        self.count += flat.shape[0]
        self.total = flat.sum(0) if self.total is None else self.total + flat.sum(0)
        squares = (flat * flat).sum(0)
        self.total_sq = squares if self.total_sq is None else self.total_sq + squares

    def finish(self) -> tuple[Tensor, Tensor]:
        """Mean and standard deviation, with near-constant channels set to unit."""
        if self.total is None or self.total_sq is None or self.count == 0:
            raise ValueError("cannot fit statistics on an empty stream of batches")
        mean = self.total / self.count
        variance = (self.total_sq / self.count - mean * mean).clamp(min=0.0)
        std = variance.sqrt()
        return mean.float(), torch.where(std < UNIT_STD_BELOW, 1.0, std).float()


def fit_stats(batches: Iterable[Batch], *, param_mode: str = "designed") -> Stats:
    """Fit the six statistics in one streaming pass over a batch iterable.

    One pass, constant memory: sums and sums of squares are accumulated per
    channel, so fitting over the whole training split costs no more than holding
    one batch.

    **The caller chooses the split, and that choice is the invariant.** Pass
    ``training_batches(..., held_out=False)`` over the *training* dataset. Any
    other stream, whether the test dataset, the held-out cables or an evaluation
    window, leaks the thing the experiment is trying to measure.

    Fit this **before** noise is added, so the statistics describe the data
    rather than the corruption.

    Args:
        batches: Contract-shaped batches. Consumed once; a generator is fine.

    Returns:
        Statistics over ``node_feat``, ``edge_feat`` and the displacement
        ``target - pos``.

    Raises:
        ValueError: if the iterable is empty.
    """
    node, edge, step = _Accumulator(), _Accumulator(), _Accumulator()
    for batch in batches:
        node.add(batch.node_feat)
        edge.add(batch.edge_feat)
        step.add(batch.target - batch.pos)

    node_mean, node_std = node.finish()
    edge_mean, edge_std = edge.finish()
    step_mean, step_std = step.finish()
    if param_mode == "designed":
        # The designed encoding replaces z-scoring for the material channels:
        # pin them to identity so normalize() passes the encoded values through.
        node_mean = node_mean.clone()
        node_std = node_std.clone()
        node_mean[MATERIAL_CHANNELS] = 0.0
        node_std[MATERIAL_CHANNELS] = 1.0
    return Stats(
        node_mean, node_std, edge_mean, edge_std, step_mean, step_std, param_mode=param_mode
    )


def normalize(batch: Batch, stats: Stats) -> Batch:
    """Scale ``node_feat`` and ``edge_feat``. Everything else passes through.

    ``pos``, ``edge_index``, ``target`` and ``meta`` come out identical to what
    went in, by object identity where possible. ``pos`` in particular must not be
    scaled: see this module's docstring.

    Args:
        batch: The batch to scale.
        stats: Statistics from :func:`fit_stats`, on the batch's device.

    Returns:
        A new batch; the input is not mutated.
    """
    node_feat = batch.node_feat
    if stats.param_mode == "designed":
        node_feat = torch.cat(
            [
                node_feat[..., : MATERIAL_CHANNELS.start],
                encode_params(node_feat[..., MATERIAL_CHANNELS]),
                node_feat[..., MATERIAL_CHANNELS.stop :],
            ],
            dim=-1,
        )
    return Batch(
        pos=batch.pos,
        node_feat=(node_feat - stats.node_mean) / stats.node_std,
        edge_index=batch.edge_index,
        edge_feat=(batch.edge_feat - stats.edge_mean) / stats.edge_std,
        target=batch.target,
        meta=batch.meta,
    )


def normalize_step(step: Tensor, stats: Stats) -> Tensor:
    """Scale a displacement ``[..., 3]`` into the units the loss is taken in.

    Applied to **both** sides of the loss, so the mean cancels and only the
    scaling bites. It is subtracted anyway, because the inverse
    :func:`denormalize_step` has to be exact for a caller that does not difference
    two normalized quantities.
    """
    return (step - stats.step_mean) / stats.step_std


def denormalize_step(step: Tensor, stats: Stats) -> Tensor:
    """The inverse of :func:`normalize_step`."""
    return step * stats.step_std + stats.step_mean
