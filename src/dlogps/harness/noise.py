"""Random-walk input noise: the one thing that makes a one-step model roll out.

**The problem it solves.** Training predicts one frame from ground truth;
evaluation feeds the model its own output for hundreds of frames. A model trained
only on true inputs has never seen its own error, so the first small mistake puts
it in a state it was never trained on, the next mistake is larger, and the
trajectory leaves the data distribution entirely. Corrupting the training input
teaches it to correct drift instead of trusting what it is handed
(Sanchez-Gonzalez et al., 2020).

**Why a random walk and not independent noise per frame.** Rollout error
accumulates: the error at frame ``k + 1`` is the error at frame ``k`` plus
whatever is new. Independent per-frame noise has no such memory, so it trains the
model against a corruption pattern rollout never produces. The walk's increments
are independent; the walk itself is not.

**Sigma is one number for the whole matrix, and has no default here.** A per-variant sigma would stop the experiment measuring the structural
bias and start it measuring the tuning. The published GNS value is theirs, for
particles.
"""

from __future__ import annotations

import dataclasses

import torch
from torch import Tensor

from dlogps.data.dataset import feature_width, features_from_state, state_from_batch
from dlogps.data.types import Batch


def random_walk(
    shape: tuple[int, ...],
    sigma: float,
    generator: torch.Generator,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """A ``[B, C, N, 3]`` walk over the history axis, **newest first**.

    The ``depth`` increments, each of scale ``sigma / sqrt(depth)``, are
    accumulated from the oldest frame forward, so the newest frame carries the
    sum of all of them and has standard deviation ``sigma`` while the oldest
    carries one increment. Index 0 is the newest frame, which is why the
    cumulative sum runs over the reversed axis: accumulating in index order would
    give the *oldest* frame the largest corruption and quietly train the model on
    a history that gets more reliable as it gets older.

    The depth comes off ``shape`` rather than off a configured ``C``, so the
    increment scale and the number of terms summed cannot disagree. Reading a
    config instead would silently return a walk at ``sigma * sqrt(depth / C)``
    for any other history length.

    Args:
        shape: ``[B, depth, N, 3]``.
        sigma: Standard deviation at the newest frame, in the units of whatever
            the walk is added to.
        generator: Seeded, so a training run is reproducible. The draw happens on
            the generator's own device, which keeps a run on a CUDA model
            reproducible from a CPU generator.
        device: Where the walk is wanted. Defaults to the generator's device.

    Raises:
        ValueError: if ``shape`` has fewer than two axes to walk over.
    """
    if len(shape) < 2:
        raise ValueError(f"a walk needs a history axis, got shape {shape}")
    depth = shape[1]
    increments = torch.randn(shape, generator=generator, device=generator.device) * (
        sigma / depth**0.5
    )
    walk = increments.flip(1).cumsum(dim=1).flip(1)
    return walk if device is None else walk.to(device)


def add_random_walk_noise(
    batch: Batch, sigma: float, generator: torch.Generator, history_frames: int
) -> Batch:
    """Corrupt a batch's state with a random walk; leave its target at ground truth.

    **The target is not moved.** That is the whole mechanism: the model is shown
    a drifted state and asked for the true next frame, so the displacement it has
    to predict includes the correction. Move the target with the noise and the
    model learns to propagate drift instead of removing it.

    **The batch is rebuilt through ``features_from_state``, never patched.**
    ``edge_feat`` and the floor-distance feature are functions of ``pos``, so
    perturbing ``pos`` in place would leave a batch whose edges describe a cable
    that is no longer there. Rebuilding makes that class of bug unreachable.

    Args:
        batch: A contract-shaped batch.
        sigma: Velocity-noise scale [m/s] at the newest history frame. Zero
            returns the batch unchanged, which is what an ablation of the noise
            itself runs.
        generator: Seeded RNG.
        history_frames: ``C``, the run's velocity-history depth.

    Returns:
        A new batch. ``meta``, ``edge_index`` and ``target`` are carried over;
        noise corrupts the state, never the cable.

    Raises:
        ValueError: if ``sigma`` is negative, or if ``node_feat`` is not the
            width the hook rebuilds.
    """
    if sigma < 0:
        raise ValueError(f"sigma is a standard deviation and cannot be negative, got {sigma}")
    if sigma == 0:
        return batch

    # state_from_batch rejects this width too, but names the wrong cause for the
    # case that will actually hit it: features_from_state emits exactly these
    # columns, so a *wider* node_feat comes back narrower with no error. That is
    # reachable the day an extra feature block lands, since this pass runs on
    # training batches only: training would then see the narrow
    # features and evaluation the wide ones, and the ablation would measure
    # nothing. Loud here, because the silent version is invisible downstream.
    width, expected = batch.node_feat.shape[-1], feature_width(history_frames)
    if width != expected:
        raise ValueError(
            f"the rollout hook rebuilds {expected} node features at a "
            f"{history_frames}-frame history, so a batch carrying {width} would be "
            "silently narrowed; carry the extra channels across this pass before enabling them"
        )

    state = state_from_batch(batch, history_frames)
    walk = random_walk(state.vel_history.shape, sigma, generator, device=state.vel_history.device)

    # The current frame's velocity error, integrated over one frame, is the
    # position it puts the cable in. dt is per cable and comes off meta, never
    # from record_hz (docs/data.md).
    drift = walk[:, 0] * batch.meta.dt.view(-1, 1, 1)
    noised = dataclasses.replace(state, pos=state.pos + drift, vel_history=state.vel_history + walk)
    return dataclasses.replace(
        features_from_state(noised, batch.meta, history_frames), target=batch.target
    )
