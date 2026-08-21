"""Autoregressive rollout: the model's own prediction fed back as its next input.

**Frame 0 is the input state, not a prediction.** A rollout of ``R`` frames
returns ``[pos_0, pred_1, ..., pred_{R-1}]``, so it aligns index for index with
``rollout_truth``, which slices ``pos[k : k + R]``. Returning ``R`` predictions
instead shifts every scored frame one step into the future: nothing raises, every
metric still returns a number, and every RMSE in the project is wrong in the same
direction. That is the single most expensive mistake available in this file, so
it is pinned against ``rollout_truth`` itself rather than against a copy of its
convention.

**Velocity is differenced with ``meta.dt``, never with a record rate.** The
achieved rate is quantized to whole simulator steps and sits about 5% off the
requested one (``docs/data.md``); a 5% error in the velocity
history compounds over a hundred autoregressive frames into a systematic drift
the metric suite reads as model error.

**Features are rebuilt only through ``features_from_state``, and scaled only with
the ``Stats`` the checkpoint carries.** The harness writes no feature vector,
because the layout of ``node_feat`` is the loader's knowledge, and it refits no
statistics, because rollout-time scaling that differs from training-time scaling
evaluates the model on inputs it never saw.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from dlogps.data.dataset import State, features_from_state, state_from_batch
from dlogps.data.types import Batch
from dlogps.harness.normalize import Stats, normalize


def roll_in(history: Tensor, velocity: Tensor) -> Tensor:
    """Prepend one velocity frame and drop the oldest, keeping the length at ``C``.

    **Newest first**, which is the order the loader writes into ``node_feat``
    (:func:`dlogps.data.dataset.features_from_state`). Appending instead of
    prepending hands the model a time-reversed history, and the model runs on it
    perfectly happily.

    ``C`` comes off the tensor rather than from a config, for the same reason
    ``noise.random_walk`` takes its depth off ``shape``: the invariant this
    function owns is that the history it returns is as deep as the one it was
    handed, and reading a configured depth here would let the two disagree.
    Whether that depth is the *run's* is checked by ``features_from_state`` on
    the very next call of the rollout loop, which does know the config.

    Args:
        history: ``[B, C, N, 3]`` velocity history, newest first.
        velocity: ``[B, N, 3]`` velocity of the step just predicted.

    Returns:
        ``[B, C, N, 3]`` with ``velocity`` at index 0 and the oldest frame gone.

    Raises:
        ValueError: if the history is not a ``[B, C, N, 3]`` stack of at least
            one frame, or the velocity is not the shape of one frame of it.
    """
    if history.ndim != 4 or history.shape[1] < 1:
        raise ValueError(f"history must be [B, C, N, 3] with C >= 1, got {tuple(history.shape)}")
    if velocity.shape != history.shape[:1] + history.shape[2:]:
        raise ValueError(
            f"velocity must be one frame of the history, "
            f"{tuple(history.shape[:1] + history.shape[2:])}, got {tuple(velocity.shape)}"
        )
    return torch.cat([velocity.unsqueeze(1), history[:, : history.shape[1] - 1]], dim=1)


def rollout(
    model: nn.Module, batch: Batch, *, horizon: int, stats: Stats, history_frames: int
) -> Tensor:
    """Roll the model forward on its own predictions for ``horizon`` frames.

    Horizons are a truncation of one long rollout rather than separate runs: roll
    out once to the longest horizon the config asks for and let the metric suite
    slice, so the shorter horizons are prefixes of the same trajectory.

    Args:
        model: Anything callable on a ``Batch`` that returns ``[B, N, 3]``
            **absolute** next positions, which is what the block's head emits
            (``dlogps.model.gps``). Run in eval mode and handed back in the
            training mode it arrived in.
        batch: The rollout's first frame, **unnormalized**, as
            ``eval_windows`` yields it. Scaling is applied inside, so a batch
            that arrives pre-scaled makes the recovered velocity history a lie.
        horizon: Frames ``R`` in the returned trajectory, including the input
            frame. ``1`` runs no forward pass at all.
        stats: The statistics the checkpoint carries, on the batch's device.
            Never refitted here.
        history_frames: ``C``, the depth the checkpoint was trained at. The batch
            must have been built at the same depth; recovering the state is where
            that is checked, so a model rolled out against another ``C`` fails
            here rather than reading a reshaped history as its own.

    Returns:
        ``[B, R, N, 3]``: ``[pos_0, pred_1, ..., pred_{R-1}]``, frame for frame
        with ``rollout_truth``.

    Raises:
        ValueError: if ``horizon`` is not at least 1, or the batch was built at a
            different velocity-history depth.
    """
    if horizon < 1:
        raise ValueError(f"horizon is a frame count and must be at least 1, got {horizon}")

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            state = state_from_batch(batch, history_frames)
            dt = batch.meta.dt.view(-1, 1, 1)
            traj: list[Tensor] = [batch.pos]

            for _ in range(horizon - 1):
                pred: Tensor = model(
                    normalize(features_from_state(state, batch.meta, history_frames), stats)
                )
                velocity = (pred - state.pos) / dt
                state = State(pos=pred, vel_history=roll_in(state.vel_history, velocity))
                traj.append(pred)

            # Stacked inside the no-grad block so the trajectory carries no graph
            # even when the caller's input positions did.
            return torch.stack(traj, dim=1)
    finally:
        model.train(was_training)
