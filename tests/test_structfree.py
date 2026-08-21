"""The structure-free floor arms against the properties the experiment claims.

The pair that earns its keep is
:func:`test_the_per_node_arm_cannot_communicate` with
:func:`test_the_global_arms_do_communicate`: the whole floor argument is that
these three arms differ from the block in what they are allowed to see, and an
arm that quietly mixed the node axis would read as a structure-free result while
being nothing of the kind.

**Translation invariance is asserted horizontally only.** ``node_feat`` carries a
clipped floor distance taken from ``pos[..., 2]``, so a vertical translation
genuinely changes the input and the arms are right to move with it. Asserting
invariance in ``z`` by mutating ``pos`` alone would pass against a stale
``node_feat`` while the real rollout path, which rebuilds features, is not
invariant at all.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from dlogps.data.dataset import feature_width
from dlogps.data.types import N
from dlogps.harness.train import build_model
from dlogps.model.gps import GPSCfg, GPSModel
from dlogps.model.structfree import (
    GlobalLSTM,
    PerNodeMLP,
    StructureFree,
    build_structfree,
    relative_positions,
)
from tests.fake_data import make_fake_batch

C = 4
"""Small history depth: every property here is independent of it, and a short
sequence keeps the LSTM arm fast."""

WIDTH = feature_width(C)


def _arms(hidden: int = 16) -> list[StructureFree]:
    """One of each arm, all at the same feature width."""
    return [
        PerNodeMLP(WIDTH, hidden),
        GlobalLSTM(WIDTH, hidden, C),
    ]


@pytest.mark.contract
def test_every_arm_returns_absolute_next_positions() -> None:
    """The block's contract: ``[B, N, 3]`` absolute positions, not deltas."""
    batch, _ = make_fake_batch(b=2, seed=1, history_frames=C)

    for arm in _arms():
        out = arm(batch)
        assert out.shape == batch.pos.shape
        assert torch.isfinite(out).all()


@pytest.mark.contract
def test_a_zero_head_predicts_the_mean_step() -> None:
    """The trained-model contract, the same one ``test_model`` pins for the
    block: a zero head is the constant predictor, which is ``pos + step_mean``
    once the trainer has copied the run's statistics in, **not** ``pos``."""
    batch, _ = make_fake_batch(b=2, seed=2, history_frames=C)
    step_mean = torch.tensor([0.1, -0.2, 0.3])

    for arm in _arms():
        with torch.no_grad():
            for parameter in arm.parameters():
                parameter.zero_()
            arm.step_mean.copy_(step_mean)

        assert torch.allclose(arm(batch) - batch.pos, step_mean.expand(batch.pos.shape))


@pytest.mark.contract
def test_a_batch_of_the_wrong_feature_width_is_refused() -> None:
    """Loud, not a silent reshape, exactly as the block refuses one."""
    batch, _ = make_fake_batch(b=2, seed=3, history_frames=C)
    wider = dataclasses.replace(
        batch, node_feat=torch.cat([batch.node_feat, torch.rand(2, N, 1)], dim=-1)
    )

    for arm in _arms():
        with pytest.raises(ValueError, match="arm built for F="):
            arm(wider)


def test_the_per_node_arm_cannot_communicate() -> None:
    """Perturbing one node leaves every other node's prediction untouched.

    The property that makes this arm the no-communication floor. Exact, not
    approximate: the node axis is never mixed, so the other rows are the same
    arithmetic on the same bits.
    """
    batch, _ = make_fake_batch(b=1, seed=4, history_frames=C)
    arm = PerNodeMLP(WIDTH, 16)
    perturbed_node = 7

    before = arm(batch)
    feat = batch.node_feat.clone()
    feat[0, perturbed_node] += 1.0
    after = arm(dataclasses.replace(batch, node_feat=feat))

    others = [i for i in range(N) if i != perturbed_node]
    assert torch.equal(before[0, others], after[0, others])
    assert not torch.equal(before[0, perturbed_node], after[0, perturbed_node])


def test_the_global_arm_does_communicate() -> None:
    """The mirror of the test above: one node's features reach the others.

    Without this the pair would pass on an arm that had accidentally been built
    per-node, and the floor's whole contrast would be silently empty.
    """
    batch, _ = make_fake_batch(b=1, seed=5, history_frames=C)
    perturbed_node = 7

    for arm in (GlobalLSTM(WIDTH, 16, C),):
        before = arm(batch)
        feat = batch.node_feat.clone()
        feat[0, perturbed_node] += 1.0
        after = arm(dataclasses.replace(batch, node_feat=feat))

        others = [i for i in range(N) if i != perturbed_node]
        assert not torch.allclose(before[0, others], after[0, others])


def test_the_per_node_arm_is_permutation_equivariant() -> None:
    """Shared weights and no mixing: permuting the nodes permutes the answer."""
    batch, _ = make_fake_batch(b=1, seed=6, history_frames=C)
    arm = PerNodeMLP(WIDTH, 16)
    order = torch.randperm(N, generator=torch.Generator().manual_seed(0))

    # Relabelling a node moves its position with its features; permuting the
    # features alone would compare a permuted step against an unpermuted origin.
    relabelled = dataclasses.replace(
        batch, node_feat=batch.node_feat[:, order], pos=batch.pos[:, order]
    )

    assert torch.allclose(arm(batch)[0, order], arm(relabelled)[0], atol=1e-6)


def test_relative_positions_are_translation_invariant_and_scaled_by_length() -> None:
    """The helper the global arms see geometry through.

    Invariant to any translation, because it is a difference from the batch's own
    centroid, and dimensionless, because it is divided by the cable's length.
    """
    batch, _ = make_fake_batch(b=2, seed=7, history_frames=C)
    shift = torch.tensor([1.5, -2.0, 0.75])

    moved = dataclasses.replace(batch, pos=batch.pos + shift)
    assert torch.allclose(relative_positions(batch), relative_positions(moved), atol=1e-6)

    longer = dataclasses.replace(
        batch, meta=dataclasses.replace(batch.meta, length=batch.meta.length * 2)
    )
    assert torch.allclose(relative_positions(longer), relative_positions(batch) / 2, atol=1e-6)


def test_the_global_arm_ignores_a_horizontal_translation() -> None:
    """A horizontal move changes no input, so it must change no prediction.

    Horizontal only: the floor-distance channel is ``pos[..., 2]`` clipped, so a
    vertical move does change ``node_feat`` under a consistent rebuild and the
    arms should follow it. Holding ``node_feat`` fixed while moving in ``x`` and
    ``y`` is the consistent version of this experiment, not a convenient one.
    """
    batch, _ = make_fake_batch(b=1, seed=8, history_frames=C)
    horizontal = torch.tensor([2.0, -3.0, 0.0])

    for arm in (GlobalLSTM(WIDTH, 16, C),):
        moved = dataclasses.replace(batch, pos=batch.pos + horizontal)
        before = arm(batch) - batch.pos
        after = arm(moved) - moved.pos
        assert torch.allclose(before, after, atol=1e-5)


def test_the_sequence_arm_reads_the_frames_in_order() -> None:
    """Reordering the history changes the answer.

    The property that separates a sequence model from a bag of frames. Stated as
    order-sensitivity rather than as "the newest frame matters most": at
    initialization an LSTM has no such recency preference, so a magnitude
    comparison would fail on a correct arm and could pass on a broken one that
    over-weighted the first index for unrelated reasons.
    """
    batch, _ = make_fake_batch(b=1, seed=9, history_frames=C)
    arm = GlobalLSTM(WIDTH, 16, C)

    velocity = batch.node_feat[..., : C * 3].reshape(1, N, C, 3)
    reversed_feat = torch.cat(
        [velocity.flip(2).reshape(1, N, C * 3), batch.node_feat[..., C * 3 :]], dim=-1
    )

    assert not torch.allclose(arm(batch), arm(dataclasses.replace(batch, node_feat=reversed_feat)))


def test_the_sequence_arm_refuses_a_depth_it_cannot_hold() -> None:
    """``C`` and ``F`` are two records of one layout; a mismatch is a defect."""
    with pytest.raises(ValueError, match="history makes F="):
        GlobalLSTM(WIDTH, 16, C + 1)


@pytest.mark.contract
def test_build_model_dispatches_on_arch() -> None:
    """The single choke point still builds a block by default, and an arm when
    asked. A typo must name the known arches rather than read as unimplemented."""
    assert isinstance(build_model(WIDTH, "A", GPSCfg(d_model=24, n_heads=6)), GPSModel)

    for arch, expected in (
        ("mlp_pernode", PerNodeMLP),
        ("lstm_global", GlobalLSTM),
    ):
        built = build_model(WIDTH, "A", GPSCfg(d_model=16, arch=arch))
        assert isinstance(built, expected)

    with pytest.raises(ValueError, match="unknown arch"):
        build_structfree("mlp_pernod", WIDTH, GPSCfg(d_model=16, arch="mlp_pernode"), C)


@pytest.mark.contract
def test_an_arm_carries_what_the_checkpoint_writer_reads() -> None:
    """``save_checkpoint`` reads ``model.cfg`` and ``model.feature_width`` off
    whatever the trainer built; an arm missing either cannot be checkpointed."""
    cfg = GPSCfg(d_model=16, arch="mlp_pernode")
    arm = build_model(WIDTH, "A", cfg)

    assert arm.cfg == cfg
    assert arm.feature_width == WIDTH


def test_the_arch_field_frees_an_arm_from_the_block_checks() -> None:
    """Head count and stream flags are the block's rules, not an arm's: an arm
    has neither, and a width that no head count divides is legal for it."""
    GPSCfg(d_model=100, n_heads=6, arch="mlp_pernode")

    with pytest.raises(ValueError, match="not divisible"):
        GPSCfg(d_model=100, n_heads=6)
