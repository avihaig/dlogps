"""The rollout loop's bookkeeping, checked without a trained model in sight.

One test carries this file:
:func:`test_ground_truth_fed_back_reproduces_rollout_truth`. A model that replays
the ground truth must return the ground truth, index for index, against the
loader's own ``rollout_truth``. That pins the frame-0 convention against the
function it has to align with rather than against a restatement of it, and it
proves the loop's bookkeeping independently of any model's quality. Every other
off-by-one in the project would be silent: the metrics still return numbers, and
every RMSE is wrong in the same direction.

The fake models are deliberately trivial and deliberately different from each
other. A fixed-step model makes the position feed-back readable, a history-echo
model makes the ``dt`` differencing observable, and a replay model makes the
alignment exact.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import pytest
import torch
from torch import Tensor, nn

from dlogps.data.dataset import CableData, make_batch, state_from_batch
from dlogps.data.loader import rollout_truth
from dlogps.data.types import Batch, N
from dlogps.harness.normalize import Stats, fit_stats, normalize
from dlogps.harness.rollout import roll_in, rollout
from dlogps.model.gps import GPSCfg, GPSModel
from tests.fake_data import HISTORY_FRAMES as C
from tests.fake_data import make_fake_batch


class _ConstantStep(nn.Module):
    """Adds a fixed displacement to whatever position it is shown.

    The displacement is a real parameter, so a rollout that forgot ``no_grad``
    would hand back a trajectory with a graph attached, which is what
    :func:`test_the_rollout_carries_no_autograd_graph` reads. The batches and the
    training flag are recorded because what the model was *shown* is half of what
    this suite is checking.
    """

    def __init__(self, step: float = 0.01) -> None:
        super().__init__()
        self.delta = nn.Parameter(torch.full((3,), step))
        self.seen: list[Batch] = []
        self.modes: list[bool] = []

    def forward(self, batch: Batch) -> Tensor:
        self.seen.append(batch)
        self.modes.append(self.training)
        return batch.pos + self.delta


class _VelocityEcho(nn.Module):
    """Steps along the newest velocity in the history it is handed.

    Reads the history back out of the scaled features it receives, which is what
    makes it sensitive to the differencing: the loop divides by ``dt`` to build
    that history, so a trajectory produced by this model depends on ``dt`` from
    its second prediction onward.
    """

    def __init__(self, stats: Stats, gain: float = 1.0, history_frames: int = C) -> None:
        super().__init__()
        self.stats = stats
        self.gain = gain
        self.history_frames = history_frames

    def forward(self, batch: Batch) -> Tensor:
        raw = dataclasses.replace(
            batch, node_feat=batch.node_feat * self.stats.node_std + self.stats.node_mean
        )
        return batch.pos + self.gain * state_from_batch(raw, self.history_frames).vel_history[:, 0]


class _Replay(nn.Module):
    """Replays a supplied trajectory frame by frame, ignoring its input.

    The oracle: whatever the loop feeds it, it answers with the true next frame.
    A rollout driven by it must reproduce the trajectory exactly, or the loop is
    mis-assembling frames.
    """

    def __init__(self, truth: Tensor) -> None:
        super().__init__()
        self.truth = truth
        self.calls = 0
        self.seen_pos: list[Tensor] = []

    def forward(self, batch: Batch) -> Tensor:
        self.seen_pos.append(batch.pos)
        self.calls += 1
        return self.truth[:, self.calls]


def _stub(b: int = 2, seed: int = 0) -> tuple[Batch, Stats]:
    """A contract-shaped batch, and statistics fitted on a stream containing it."""
    batch, _ = make_fake_batch(b=b, seed=seed)
    stats = fit_stats([make_fake_batch(b=b, seed=s)[0] for s in range(seed, seed + 3)])
    return batch, stats


def _synthetic_cable(frames: int = 20, seed: int = 3) -> CableData:
    """One in-memory episode, so the alignment test touches no file on disk.

    ``rollout_truth`` needs a ``CableData`` and nothing else, and the frame
    convention is a property of the loop rather than of any particular cable.
    """
    gen = torch.Generator().manual_seed(seed)
    return CableData(
        episodes=[
            (
                torch.cumsum(torch.randn(frames, N, 3, generator=gen) * 1e-3, dim=0),
                torch.randn(frames, N, 3, generator=gen) * 1e-2,
                torch.arange(frames, dtype=torch.float32) * 2e-3,
            )
        ],
        params={
            "length": 1.0,
            "diameter": 0.003,
            "bend_stiffness": 1e-2,
            "joint_damping": 1e-3,
            "effective_linear_density": 0.1,
        },
        cable_id=0,
    )


def _rollout_at(horizon: int) -> Tensor:
    """A rollout at one horizon, for the validation cases."""
    batch, stats = _stub()
    return rollout(_ConstantStep(), batch, horizon=horizon, stats=stats, history_frames=C)


@pytest.mark.contract
def test_roll_in_prepends_the_newest_frame_and_drops_the_oldest() -> None:
    """Newest first, and as deep afterwards as the history it was handed."""
    labelled = torch.arange(C, dtype=torch.float32).view(1, C, 1, 1).expand(2, C, N, 3).clone()
    velocity = torch.full((2, N, 3), -1.0)

    rolled = roll_in(labelled, velocity)

    assert rolled.shape == labelled.shape
    assert torch.equal(rolled[:, 0], velocity)
    assert torch.equal(rolled[:, 1:], labelled[:, : C - 1])


@pytest.mark.claim(
    "roll_in keeps the depth it was handed rather than a configured one, for the reason "
    "noise.random_walk takes its depth off the tensor: the two cannot then disagree "
    ""
)
@pytest.mark.parametrize("depth", [1, 3, 9])
def test_roll_in_keeps_whatever_depth_it_is_handed(depth: int) -> None:
    labelled = (
        torch.arange(depth, dtype=torch.float32).view(1, depth, 1, 1).expand(2, depth, N, 3).clone()
    )
    velocity = torch.full((2, N, 3), -1.0)

    rolled = roll_in(labelled, velocity)

    assert rolled.shape == labelled.shape, "the history neither grew nor shrank"
    assert torch.equal(rolled[:, 0], velocity)
    assert torch.equal(rolled[:, 1:], labelled[:, : depth - 1])


@pytest.mark.claim(
    "a rollout runs at whatever depth its checkpoint was trained at, which is what makes C "
    "testable as a lever on rollout stability (docs/method.md)"
)
@pytest.mark.parametrize("depth", [2, 8])
def test_a_rollout_runs_at_another_history_depth(depth: int) -> None:
    batch, _ = make_fake_batch(b=2, seed=4, history_frames=depth)
    stats = fit_stats([make_fake_batch(b=2, seed=s, history_frames=depth)[0] for s in range(3)])
    model = GPSModel.from_batch(batch, GPSCfg(d_model=12, n_heads=2, n_layers=1))

    traj = rollout(model, batch, horizon=5, stats=stats, history_frames=depth)

    assert traj.shape == (2, 5, N, 3)
    assert torch.isfinite(traj).all()
    assert torch.equal(traj[:, 0], batch.pos)


@pytest.mark.regression(
    "a rollout at a depth its batch was not built at would slice the history and roll a "
    "correctly-shaped lie forward"
)
def test_a_rollout_at_the_wrong_depth_is_refused() -> None:
    batch, stats = _stub()
    with pytest.raises(ValueError, match="different velocity-history depth"):
        rollout(_ConstantStep(), batch, horizon=3, stats=stats, history_frames=C + 2)


@pytest.mark.contract
def test_a_fixed_step_model_integrates_into_a_straight_trajectory() -> None:
    """The contracted shape, and the position actually fed back between frames."""
    batch, stats = _stub(b=3)
    model = _ConstantStep()

    traj = rollout(model, batch, horizon=4, stats=stats, history_frames=C)

    assert traj.shape == (3, 4, N, 3)
    step = model.delta.detach()
    for frame in range(4):
        assert torch.allclose(traj[:, frame], batch.pos + frame * step, atol=1e-6)


@pytest.mark.claim("frame 0 of a rollout is the input state")
def test_ground_truth_fed_back_reproduces_rollout_truth() -> None:
    """An oracle model scores exactly zero error, against the loader's own slice.

    ``rollout_truth`` slices ``pos[k : k + R]``, whose first entry is the frame
    the model was started from. A loop that returned ``R`` predictions instead
    would pass every shape check here and be wrong by one frame everywhere.
    """
    cable = _synthetic_cable()
    _, stats = _stub()
    k, r = C - 1, 8
    batch = make_batch(cable, [(0, k)], is_held_out=False, history_frames=C)
    truth = rollout_truth(cable, 0, k, r)
    model = _Replay(truth)

    traj = rollout(model, batch, horizon=r, stats=stats, history_frames=C)

    assert torch.equal(traj, truth)
    assert model.calls == r - 1, "R frames come from R - 1 forward passes"
    assert torch.equal(torch.stack(model.seen_pos, dim=1), truth[:, : r - 1])


@pytest.mark.claim("frame 0 of a rollout is the input state")
def test_frame_zero_is_the_input_position_not_a_prediction() -> None:
    batch, stats = _stub()

    traj = rollout(_ConstantStep(), batch, horizon=3, stats=stats, history_frames=C)

    assert torch.equal(traj[:, 0], batch.pos)
    assert not torch.allclose(traj[:, 1], batch.pos), "the model does move the cable"


@pytest.mark.contract
def test_horizon_one_runs_no_forward_pass() -> None:
    """Horizon 1 is the input frame alone, so nothing is predicted."""
    batch, stats = _stub()
    model = _ConstantStep()

    traj = rollout(model, batch, horizon=1, stats=stats, history_frames=C)

    assert traj.shape == (2, 1, N, 3)
    assert torch.equal(traj[:, 0], batch.pos)
    assert model.seen == []


@pytest.mark.contract
def test_frame_one_equals_a_single_forward_pass() -> None:
    """The first prediction is the model's one-step output on the input batch."""
    batch, stats = _stub()
    model = _VelocityEcho(stats)

    traj = rollout(model, batch, horizon=2, stats=stats, history_frames=C)

    assert torch.allclose(traj[:, 1], model(normalize(batch, stats)), atol=1e-6)


@pytest.mark.claim("velocity is differenced with meta.dt: docs/data.md")
def test_doubling_dt_changes_the_trajectory() -> None:
    """``dt`` enters at the differencing, so frame 1 is untouched and frame 2 is not."""
    batch, stats = _stub()
    slower = dataclasses.replace(
        batch, meta=dataclasses.replace(batch.meta, dt=batch.meta.dt * 2.0)
    )

    fast = rollout(_VelocityEcho(stats), batch, horizon=3, stats=stats, history_frames=C)
    slow = rollout(_VelocityEcho(stats), slower, horizon=3, stats=stats, history_frames=C)

    assert torch.allclose(fast[:, 1], slow[:, 1])
    assert not torch.allclose(fast[:, 2], slow[:, 2])


@pytest.mark.claim("rollout inputs are scaled by the checkpoint's Stats")
def test_the_model_is_shown_features_scaled_by_the_given_stats() -> None:
    """Scaled features, unscaled positions, and no statistics fitted here."""
    batch, stats = _stub()
    model = _ConstantStep()

    rollout(model, batch, horizon=3, stats=stats, history_frames=C)

    shown = model.seen[0]
    assert torch.allclose(shown.node_feat, normalize(batch, stats).node_feat, atol=1e-6)
    assert not torch.allclose(shown.node_feat, batch.node_feat)
    assert torch.equal(shown.pos, batch.pos)


@pytest.mark.claim("a rollout allocates no autograd graph")
def test_the_rollout_carries_no_autograd_graph() -> None:
    """At horizon 100 a retained graph does not fit, and it is retained silently."""
    batch, stats = _stub()
    model = _ConstantStep()
    assert model(normalize(batch, stats)).requires_grad, "the fake model has a real parameter"

    traj = rollout(model, batch, horizon=6, stats=stats, history_frames=C)

    assert not traj.requires_grad
    assert traj.grad_fn is None


@pytest.mark.contract
def test_rollout_runs_in_eval_and_restores_the_previous_mode() -> None:
    """Left in eval, the next training epoch runs with its norms frozen."""
    batch, stats = _stub()
    model = _ConstantStep()
    model.train()

    rollout(model, batch, horizon=4, stats=stats, history_frames=C)

    assert model.modes == [False, False, False]
    assert model.training

    model.eval()
    rollout(model, batch, horizon=2, stats=stats, history_frames=C)
    assert not model.training


@pytest.mark.contract
def test_the_real_block_rolls_out() -> None:
    """The one smoke test with a real model: the loop composes with the block."""
    batch, stats = _stub()
    model = GPSModel.from_batch(batch, GPSCfg(d_model=12, n_heads=2, n_layers=1))

    traj = rollout(model, batch, horizon=5, stats=stats, history_frames=C)

    assert traj.shape == (2, 5, N, 3)
    assert torch.isfinite(traj).all()
    assert torch.equal(traj[:, 0], batch.pos)


@pytest.mark.contract
@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: _rollout_at(0), "at least 1"),
        (lambda: _rollout_at(-3), "at least 1"),
        (lambda: roll_in(torch.zeros(2, 0, N, 3), torch.zeros(2, N, 3)), "history must be"),
        (lambda: roll_in(torch.zeros(2, N, 3), torch.zeros(2, N, 3)), "history must be"),
        (lambda: roll_in(torch.zeros(2, C, N, 3), torch.zeros(2, N)), "velocity must be"),
    ],
    ids=["horizon-zero", "horizon-negative", "empty-history", "flat-history", "velocity-shape"],
)
def test_invalid_arguments_raise(call: Callable[[], object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()
