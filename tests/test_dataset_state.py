"""``state_from_batch``: the inverse of the rollout hook, and the layout it must know.

The contract names round-tripping as its own test: ``features_from_state(
state_from_batch(b), b.meta)`` reproduces ``b`` field for field, ``target``
excepted. Two callers depend on it, the rollout loop and the input-noise pass,
and both feed what comes back straight into the model.

The round trip alone does not defend the layout. It pins the reader as the exact
inverse of the writer, which catches a transposition in either one alone, but a
transposition the two *share* composes back to the identity and passes: the
history reaching the model is then scrambled across frames and nodes while every
test in the suite stays green. Only a batch whose history says which frame and
which node each number came from separates the two cases, which is why
:func:`test_the_history_comes_back_newest_first_and_per_node` writes ``node_feat``
by hand from the contract instead of round-tripping.

Round-tripping the stub is also a cross-check rather than a tautology:
``tests/fake_data.py`` writes ``node_feat`` from the contract independently of
``features_from_state``, so the two implementations of one layout are compared.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from dlogps.data.dataset import features_from_state, state_from_batch
from dlogps.data.types import N
from tests.fake_data import HISTORY_FRAMES as C
from tests.fake_data import make_fake_batch


@pytest.mark.contract
def test_state_from_batch_returns_the_contracted_state() -> None:
    batch, _ = make_fake_batch(b=3, seed=2)

    state = state_from_batch(batch, C)

    assert state.pos is batch.pos, "position is not derived, so it is handed back untouched"
    assert state.vel_history.shape == (3, C, N, 3)
    assert state.vel_history.dtype == torch.float32


@pytest.mark.claim(
    "features_from_state(state_from_batch(b), b.meta) reproduces b field for field, target "
    "excepted, so the harness never needs to know the node_feat layout "
    ""
)
def test_the_round_trip_reproduces_every_field() -> None:
    """Asserted on values, not shapes.

    The failure it pins: a recovery that is shape-correct but value-wrong feeds
    the rollout loop a state that is not the one it was handed, and the first
    predicted frame is already wrong for a reason no metric attributes to the
    hook.
    """
    batch, _ = make_fake_batch(b=3, seed=1)

    rebuilt = features_from_state(state_from_batch(batch, C), batch.meta, C)

    assert torch.equal(rebuilt.pos, batch.pos)
    assert torch.equal(rebuilt.node_feat, batch.node_feat)
    assert torch.equal(rebuilt.edge_index, batch.edge_index)
    assert torch.equal(rebuilt.edge_feat, batch.edge_feat)
    assert rebuilt.meta is batch.meta

    # The documented exception: a rollout has no ground truth, so the hook leaves
    # target at the current position and only the dataset overwrites it.
    assert torch.equal(rebuilt.target, batch.pos)
    assert not torch.equal(rebuilt.target, batch.target)


@pytest.mark.claim(
    "the recovered history is newest first and node for node, so a reshape that transposes the "
    "frame and node axes is caught rather than undone by the round trip "
    ""
)
def test_the_history_comes_back_newest_first_and_per_node() -> None:
    """A ramp whose every value names the sample, frame, node and axis it belongs to.

    ``node_feat`` is written here straight from the contract's layout, per node
    with ``C`` frames newest first and xyz innermost, rather than by calling
    ``features_from_state``. That makes the assertion independent of the writer,
    so a regrouping the writer and the reader agree on is still a failure, and the
    values name the frame and the node they came from rather than relying on a
    shape check that any regrouping satisfies.
    """
    batch, _ = make_fake_batch(b=2, seed=3)
    b = batch.pos.shape[0]

    ramp = torch.zeros(b, C, N, 3)
    flat = torch.zeros(b, N, C * 3)
    for i in range(b):
        for c in range(C):
            for n in range(N):
                for k in range(3):
                    value = 1000.0 * i + 100.0 * c + n + k / 10.0
                    ramp[i, c, n, k] = value
                    flat[i, n, c * 3 + k] = value

    probe = dataclasses.replace(
        batch, node_feat=torch.cat([flat, batch.node_feat[..., C * 3 :]], dim=-1)
    )

    assert torch.equal(state_from_batch(probe, C).vel_history, ramp)
