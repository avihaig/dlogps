"""``make_fake_batch``: contract-shaped tensors, no file on disk, no dataset.

This is the simulator-free fixture: the block and the harness are tested against
the **interface**, one stub batch with the contract's exact shapes, so the suite
needs neither the dataset nor a physics engine.

Written against the frozen contract, so the loader consumes it unchanged: what
it emits is fixed by the contract and not by this implementation.

The geometry is *plausible*, not physical: segment lengths are exactly the rest
length and the curve turns smoothly, so a link-length or self-intersection check
run against this batch measures the check rather than measuring noise. Nothing
here is a physics simulator and no test should read it as one.
"""

from __future__ import annotations

import torch
from torch import Tensor

from dlogps.data.types import Batch, BatchMeta, E, Labels, N

HISTORY_FRAMES = 5
"""``C`` a stub batch carries unless the caller asks for another depth.

The real depth is a run's own (``TrainCfg.history_frames``), so this is the
stub's fixture value rather than the project's constant. It matches the default a
run gets, which is what keeps a stub batch and a loader batch the same shape."""

FEATURE_WIDTH = HISTORY_FRAMES * 3 + 5
"""``F`` for this stub at :data:`HISTORY_FRAMES`: the velocity history (3 per
frame), the four material params broadcast per node, and the clipped floor
distance. The block must read ``F`` off the tensor anyway, which is exactly what
this width is here to exercise.

Written out rather than imported from the loader, because the whole point of this
module is to satisfy the contract independently: ``tests/test_fake_data.py``
checks the two agree, and that check is only worth anything while they are
computed separately."""

_L_REST_RANGE = (0.80, 1.60)
"""Rest length [m] spread of the release population (46 rows, 0.80 to 1.60 m)."""

_DIAMETER_RANGE = (0.0015, 0.0100)
"""Diameter [m] the stub draws from: all of the release population, 1.5 to 10 mm.

The population enumerates six diameters (1.5, 2.0, 2.5, 5.0, 7.5, 10.0), a third
of its 46 cables above 5 mm, measured off `release_v1` and `test_v1` on
2026-08-05. Both thresholds that read a diameter read it per cable — the contact
label at ``1.5 x D``, self-intersection at the bare ``D`` — so a stub that spans
only the thin half exercises them nowhere near their largest, which is where a
hard-coded distance or a diameter/length mix-up would show.

Continuous rather than the six grid values, as `_L_REST_RANGE` is over its four:
the stub stands in for the population's spread, not for its factorial design.

The thick end is affordable because the stub's geometry has room for it: over 800
stub cables the closest chain-far approach (``|i-j| > 2``, the metric's own mask)
is 46 mm, 4x the 10 mm diameter and 3x ``d_close`` there. So the whole range
keeps the property the self-intersection metric is controlled against — this
batch reads zero violation, and a metric that fires on it is measuring itself."""

_DT = 1.0 / 500.0
"""Frame interval [s] at the recorded 500 Hz, the rate settled on by the gate
that the recording rate is fast enough."""


def chain_edge_index(n: int = N) -> Tensor:
    """The static chain, both directions: ``[2, 2 * (n - 1)]`` int64.

    Columns ``0 .. n-2`` run forward (``i -> i + 1``) and the rest run backward,
    so column ``e`` and column ``e + n - 1`` are the two directions of segment
    ``e``. The order is an implementation detail and no consumer may rely on it;
    what the contract fixes is only that **both** directions are present.

    Args:
        n: Vertex count.

    Returns:
        Source row then destination row, as ``edge_index[0] -> edge_index[1]``.
    """
    fwd = torch.arange(n - 1, dtype=torch.int64)
    src = torch.cat([fwd, fwd + 1])
    dst = torch.cat([fwd + 1, fwd])
    return torch.stack([src, dst])


def _cable_positions(b: int, l_rest: Tensor, gen: torch.Generator) -> Tensor:
    """``[b, N, 3]`` vertex positions of a smoothly curving cable above the floor.

    Built by walking ``E`` unit tangents of length ``l_rest / E`` whose direction
    turns by a small random increment per step. Two consequences are deliberate:
    every segment is exactly at its rest length, so a link-length metric reads
    zero violation on this batch; and the curve is smooth rather than a random
    walk, so a chain-distance bias and a 3D-distance bias actually disagree,
    which a straight line or a white-noise cloud would not give.
    """
    step = (l_rest / E).view(b, 1, 1)

    # Small per-step turns: enough total curvature to fold, never a hairpin.
    turn = torch.randn(b, E, 2, generator=gen) * 0.25
    angles = torch.cumsum(turn, dim=1)
    theta, phi = angles[..., 0], angles[..., 1]
    tangents = torch.stack(
        [torch.cos(theta) * torch.cos(phi), torch.cos(theta) * torch.sin(phi), torch.sin(theta)],
        dim=-1,
    )

    origin = torch.zeros(b, 1, 3)
    pos = torch.cat([origin, torch.cumsum(tangents * step, dim=1)], dim=1)

    # The floor is z = 0 and the clipped floor-distance feature is defined
    # against it, so a cable that hangs below it would make that feature a lie.
    return pos - torch.stack(
        [torch.zeros(b), torch.zeros(b), pos[..., 2].amin(dim=1)], dim=-1
    ).unsqueeze(1)


def _edge_features(pos: Tensor, edge_index: Tensor, l_rest: Tensor) -> Tensor:
    """``[B, 2 * E, 5]``: displacement (3), distance (1), rest length (1).

    Derived from ``pos`` rather than sampled, so the edge channel and the
    positions cannot disagree — a stub that let them drift would let a bug in the
    local stream pass unnoticed.
    """
    src, dst = edge_index[0], edge_index[1]
    disp = pos[:, dst] - pos[:, src]
    dist = disp.norm(dim=-1, keepdim=True)
    seg_rest = (l_rest / E).view(-1, 1, 1).expand_as(dist)
    return torch.cat([disp, dist, seg_rest], dim=-1)


def make_fake_batch(
    b: int = 2, r: int = 1, seed: int = 0, history_frames: int = HISTORY_FRAMES
) -> tuple[Batch, Labels]:
    """One contract-shaped ``(Batch, Labels)`` pair. Touches no file on disk.

    Deterministic in ``seed`` and independent of global RNG state, so two suites
    that both call it get the same tensors and neither perturbs the other.

    Args:
        b: Batch size.
        r: Rollout length in frames, the ``R`` axis of ``Labels``. The batch
            itself is always one frame; ``r`` exists so a metric smoke test can
            run before any data exists.
        seed: Seeds a local generator.
        history_frames: ``C``. Another depth gives a batch of another ``F``,
            which is what a test of the depth being configurable needs.

    Returns:
        The batch, and the labels aligned to an ``r``-frame rollout of it.

    Raises:
        ValueError: if ``b``, ``r`` or ``history_frames`` is not positive.
    """
    if b < 1 or r < 1 or history_frames < 1:
        raise ValueError(
            f"b, r and history_frames must be positive, got b={b}, r={r}, "
            f"history_frames={history_frames}"
        )

    gen = torch.Generator().manual_seed(seed)

    def _uniform(lo: float, hi: float) -> Tensor:
        return torch.rand(b, generator=gen) * (hi - lo) + lo

    l_rest = _uniform(*_L_REST_RANGE)
    diameter = _uniform(*_DIAMETER_RANGE)
    meta = BatchMeta(
        cable_id=torch.arange(b, dtype=torch.int64),
        length=l_rest,
        diameter=diameter,
        bend_stiffness=_uniform(1e-4, 1e-1),
        joint_damping=_uniform(1e-5, 1e-2),
        effective_linear_density=_uniform(0.02, 0.35),
        dt=torch.full((b,), _DT),
        is_held_out=torch.zeros(b, dtype=torch.bool),
    )

    pos = _cable_positions(b, l_rest, gen)
    edge_index = chain_edge_index(N)

    # Velocity history is [B, C, N, 3] at the source and enters node_feat
    # flattened, newest first. Scale: metres per 2 ms frame, so ~1 m/s.
    vel_history = torch.randn(b, history_frames, N, 3, generator=gen) * 2e-3
    # Taken from meta, not sampled again. Before the three were added to
    # BatchMeta these were invented here and discarded, so the stub held numbers
    # in node_feat that existed nowhere else and nothing downstream could
    # recover — which is what a rollout needs to do every step.
    material = torch.stack(
        [meta.bend_stiffness, meta.joint_damping, meta.diameter, meta.effective_linear_density],
        dim=-1,
    )
    floor_distance = pos[..., 2].clamp(min=0.0, max=1.0).unsqueeze(-1)
    node_feat = torch.cat(
        [
            vel_history.permute(0, 2, 1, 3).reshape(b, N, history_frames * 3),
            material.unsqueeze(1).expand(b, N, 4),
            floor_distance,
        ],
        dim=-1,
    )

    batch = Batch(
        pos=pos,
        node_feat=node_feat,
        edge_index=edge_index,
        edge_feat=_edge_features(pos, edge_index, l_rest),
        target=pos + vel_history[:, 0] * _DT,
        meta=meta,
    )

    # Labels are the ground truth's, not the model's, so they are sampled rather
    # than derived: a stub cable has no contact to detect. Only the chain-far
    # constraint is honoured, since a decomposition that indexes pairs would
    # otherwise be handed adjacent segments it must never score as contact.
    seg = torch.arange(E)
    chain_far = (seg.view(-1, 1) - seg.view(1, -1)).abs() > 2
    close = torch.rand(b, r, E, E, generator=gen) > 0.98
    contact_pair = (close | close.transpose(-1, -2)) & chain_far
    labels = Labels(
        contact_frame=contact_pair.flatten(start_dim=2).any(dim=-1),
        contact_pair=contact_pair,
    )

    return batch, labels
