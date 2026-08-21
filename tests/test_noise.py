"""Random-walk input noise, and the three substitutions that silently disable it.

The module has one job nothing downstream can check for it: corrupt the state a
training sample is read from while leaving its target at ground truth, so the
displacement the model learns to predict carries a drift correction as well as the
physics. Every failure here is silent, so three tests carry the file.

:func:`test_the_walk_grows_toward_the_newest_frame` pins the direction of the
cumulative sum. Reversing it changes no shape and no dtype, and it trains the
model on a history that gets *more* reliable as it gets older.
:func:`test_the_walk_is_correlated_across_the_history_axis` separates a walk from
iid noise of the same marginal scale, the substitution that reads as a
simplification and removes the one property that makes the corruption resemble
rollout error. :func:`test_the_target_does_not_move_with_the_noise` pins the
mechanism itself: a target that moves with the noise lowers the training loss and
teaches the model to propagate drift.
"""

from __future__ import annotations

import math

import pytest
import torch

from dlogps.data.dataset import FLOOR_CLIP_M
from dlogps.data.types import E, N
from dlogps.harness.noise import add_random_walk_noise, random_walk
from tests.fake_data import HISTORY_FRAMES as C
from tests.fake_data import make_fake_batch

SIGMA = 0.05
"""Velocity-noise scale [m/s] for the batch-level tests. Large enough against the
stub's ~2e-3 m/s history that every derived channel moves."""

SIGMA_UNIT = 1.0
"""Scale for the statistical tests, so a realized standard deviation reads as a
fraction of ``sigma`` directly."""

WALK_SAMPLES = 400
"""Batch axis of the statistical draws: ``400 * N * 3`` samples per history frame,
which puts the standard error of a realized standard deviation near 0.4%."""


def _gen(seed: int = 0) -> torch.Generator:
    """A seeded CPU generator, independent of global RNG state."""
    return torch.Generator().manual_seed(seed)


@pytest.mark.contract
def test_zero_sigma_leaves_the_batch_untouched() -> None:
    """The noise ablation runs at ``sigma = 0``, so it must be an identity, not a re-derivation."""
    batch, _ = make_fake_batch(b=2)

    assert add_random_walk_noise(batch, 0.0, _gen(), C) is batch


@pytest.mark.claim(
    "the walk accumulates toward the newest history frame, standard deviation sigma at index 0 "
    "and sigma/sqrt(C) at index C-1"
)
def test_the_walk_grows_toward_the_newest_frame() -> None:
    """Accumulating in index order swaps the two ends and changes nothing observable.

    The failure this pins: the *oldest* frame carries the largest corruption, so
    the model is trained on a history that grows more trustworthy with age, which
    is the opposite of what rollout hands it. Asserting the whole profile rather
    than the newest frame alone leaves no direction that satisfies both ends.
    """
    walk = random_walk((WALK_SAMPLES, C, N, 3), SIGMA_UNIT, _gen(0))

    realized = walk.permute(1, 0, 2, 3).reshape(C, -1).std(dim=1, unbiased=False)
    expected = torch.tensor([SIGMA_UNIT * math.sqrt((C - c) / C) for c in range(C)])

    assert torch.allclose(realized, expected, rtol=0.02)


@pytest.mark.claim(
    "the walk is correlated across the history axis rather than iid per frame, which is what "
    "makes the corruption resemble accumulated rollout error"
)
def test_the_walk_is_correlated_across_the_history_axis() -> None:
    """Both assertions are chosen so iid noise of the same marginals fails them.

    Independent frames scaled to the same per-frame standard deviations give a
    correlation of 0 rather than ``sqrt((C - 1) / C)``, and an adjacent difference
    of ``sigma * sqrt((2C - 1) / C)``, which is three times the walk's increment.
    """
    walk = random_walk((WALK_SAMPLES, C, N, 3), SIGMA_UNIT, _gen(1))
    frames = walk.permute(1, 0, 2, 3).reshape(C, -1)
    newest = frames[0] - frames[0].mean()
    second = frames[1] - frames[1].mean()

    correlation = float(
        (newest * second).mean() / (newest.std(unbiased=False) * second.std(unbiased=False))
    )
    increment = float((frames[0] - frames[1]).std(unbiased=False))

    assert correlation == pytest.approx(math.sqrt((C - 1) / C), rel=0.02)
    assert increment == pytest.approx(SIGMA_UNIT / math.sqrt(C), rel=0.02)


@pytest.mark.contract
def test_the_walk_lands_in_pos_and_in_the_velocity_history() -> None:
    """The newest increment reaches ``pos`` through ``dt``, the whole walk reaches the history.

    Reconstructed from a twin generator rather than from the output, so the
    expected tensors are independent of what the module returned.
    """
    batch, _ = make_fake_batch(b=3, seed=6)
    b = batch.pos.shape[0]

    out = add_random_walk_noise(batch, SIGMA, _gen(7), C)
    walk = random_walk((b, C, N, 3), SIGMA, _gen(7))

    assert torch.equal(out.pos, batch.pos + walk[:, 0] * batch.meta.dt.view(-1, 1, 1))
    assert torch.equal(
        out.node_feat[..., : C * 3],
        batch.node_feat[..., : C * 3] + walk.permute(0, 2, 1, 3).reshape(b, N, C * 3),
    )


@pytest.mark.contract
def test_the_noised_batch_still_satisfies_the_batch_contract() -> None:
    """Every channel derived from ``pos`` is rederived from the *perturbed* ``pos``.

    A batch patched in place would keep edges describing a cable that is no longer
    where the positions say it is, which no shape check and no loss curve reveals.
    """
    batch, _ = make_fake_batch(b=3, seed=4)
    before_pos, before_feat = batch.pos.clone(), batch.node_feat.clone()

    out = add_random_walk_noise(batch, SIGMA, _gen(5), C)

    src, dst = out.edge_index[0], out.edge_index[1]
    disp = out.pos[:, dst] - out.pos[:, src]
    dist = disp.norm(dim=-1, keepdim=True)
    rest = (out.meta.length / E).view(-1, 1, 1).expand_as(dist)

    assert not torch.equal(out.pos, batch.pos), "the state must move, or the assertions are vacuous"
    assert torch.equal(out.edge_feat, torch.cat([disp, dist, rest], dim=-1))
    assert torch.equal(out.node_feat[..., -1], out.pos[..., 2].clamp(0.0, FLOOR_CLIP_M))
    assert torch.equal(out.node_feat[..., C * 3 : -1], batch.node_feat[..., C * 3 : -1])
    assert torch.equal(batch.pos, before_pos)
    assert torch.equal(batch.node_feat, before_feat)


@pytest.mark.claim(
    "the target stays at ground truth while the input drifts, so the displacement the model "
    "learns to predict includes the drift correction"
)
def test_the_target_does_not_move_with_the_noise() -> None:
    """The mechanism of the whole module, and the one substitution that lowers the loss.

    A target moved with the noise leaves the displacement unchanged, which is the
    easier thing to predict, so the training curve looks *better* and the model
    learns to propagate drift instead of removing it. The damage appears only
    hundreds of frames into a rollout.
    """
    batch, _ = make_fake_batch(b=3, seed=8)

    out = add_random_walk_noise(batch, SIGMA, _gen(9), C)

    assert torch.equal(out.target, batch.target)
    assert not torch.equal(out.target - out.pos, batch.target - batch.pos)


@pytest.mark.claim(
    "noise corrupts the state and never the cable: meta and the chain topology are carried "
    "over untouched"
)
def test_meta_and_the_chain_are_carried_over() -> None:
    """A per-sample constant rewritten by the noise pass would change what cable is being trained on."""
    batch, _ = make_fake_batch(b=3, seed=10)

    out = add_random_walk_noise(batch, SIGMA, _gen(11), C)

    assert out.meta is batch.meta
    assert torch.equal(out.edge_index, batch.edge_index)


@pytest.mark.contract
def test_the_same_seed_gives_the_same_corruption() -> None:
    """Two runs at one seed must agree on bytes; the matrix is not reproducible otherwise."""
    batch, _ = make_fake_batch(b=3, seed=12)

    first = add_random_walk_noise(batch, SIGMA, _gen(13), C)
    again = add_random_walk_noise(batch, SIGMA, _gen(13), C)
    other = add_random_walk_noise(batch, SIGMA, _gen(14), C)

    assert torch.equal(first.pos, again.pos)
    assert torch.equal(first.node_feat, again.node_feat)
    assert not torch.equal(first.pos, other.pos)
    assert not torch.equal(first.node_feat, other.node_feat)


@pytest.mark.contract
def test_a_negative_sigma_is_refused() -> None:
    batch, _ = make_fake_batch(b=2)

    with pytest.raises(ValueError, match="cannot be negative"):
        add_random_walk_noise(batch, -1e-9, _gen(), C)
