"""The four biased variants, and the gate that bias scales are comparable at the
start.

Every variant satisfies the one protocol, owns its own normalization, and runs
through the unchanged block. That is the claim the model block was built to make
cheap, and these are the tests of it.

:func:`test_bias_scales_agree_at_init` is the gate. It is the reason the
normalization is fixed a priori in the spec rather than tuned here: if it fails,
the normalization is wrong, not the constants.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

import dlogps.model as model_api
from dlogps.data.types import N
from dlogps.model import (
    VARIANTS,
    BiasPlugin,
    BiasSource,
    ChainDistanceBias,
    GPSCfg,
    GPSModel,
    MixedHeadBias,
    NoBias,
    SpaceDistanceBias,
    assign_sources,
)
from dlogps.model.variants import RATE_INIT, chain_distance, pairwise_distance
from tests.fake_data import make_fake_batch

CFG = GPSCfg(d_model=24, n_heads=6, n_layers=2)
H = CFG.n_heads

RatedBias = SpaceDistanceBias | ChainDistanceBias | MixedHeadBias
"""The three variants carrying a softplus'd per-head rate at any head count.
Variant F is the same construction fixed at eight heads and is tested below."""

BIASED = ["B", "C", "D"]
"""The three that produce a ``b`` at the generic head count. Variant A is tested
in ``test_model.py``, where its whole content is that it returns ``None``."""


def _plugin(letter: str) -> BiasPlugin:
    return VARIANTS[letter](H)


# --- the protocol, one shape for all of them --------------------------------


@pytest.mark.contract
@pytest.mark.parametrize("letter", ["A", *BIASED])
def test_every_variant_satisfies_the_protocol(letter: str) -> None:
    """The point of the model block: the variants are nearly free because none of
    them may change the interface."""
    assert isinstance(_plugin(letter), BiasPlugin)


@pytest.mark.contract
@pytest.mark.parametrize("letter", BIASED)
def test_every_variant_returns_a_broadcastable_bias(letter: str) -> None:
    """``[B, H, N, N]``, or ``[1, H, N, N]`` for the ones that do not depend on
    the sample."""
    batch, _ = make_fake_batch(b=3, seed=0)
    b = _plugin(letter)(batch.pos, batch.meta)

    assert b is not None
    assert b.shape[0] in (1, 3)
    assert b.shape[1:] == (H, N, N)


@pytest.mark.contract
@pytest.mark.parametrize("letter", BIASED)
def test_every_variant_runs_through_the_unchanged_block(letter: str) -> None:
    """No block change per variant. That is the whole design."""
    batch, _ = make_fake_batch(b=2, seed=1)
    model = GPSModel.from_batch(batch, CFG, _plugin(letter))

    assert model(batch).shape == batch.target.shape


@pytest.mark.regression(
    bug="the plugin was a forward argument, so its rates were not submodule parameters and the optimizer never saw them"
)
@pytest.mark.parametrize("letter", BIASED)
def test_the_rates_are_trained(letter: str) -> None:
    """The reason the plugin is a constructor argument and not a forward one: a
    plugin handed in at call time is not a submodule, its parameters never reach
    ``model.parameters()``, and the rates would sit at their init for the whole
    run while everything else trained."""
    batch, _ = make_fake_batch(b=2, seed=1)
    model = GPSModel.from_batch(batch, CFG, _plugin(letter))

    names = {name for name, _ in model.named_parameters()}
    assert any(name.startswith("bias.") for name in names)

    # Two steps, not one, and the rates' *values* rather than their gradients.
    # The head is zero-initialized, so the Jacobian through it is zero and every
    # parameter behind it sees a zero gradient on the first backward pass. The
    # head's own weights move on that step, so the body trains from the second
    # one. Asserting on the gradient alone would read that one-step delay as the
    # optimizer never seeing the rates, which is the opposite of the truth, and
    # the value change is the stronger statement anyway.
    rates = [(name, p) for name, p in model.named_parameters() if name.startswith("bias.")]
    before = [p.detach().clone() for _, p in rates]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(2):
        optimizer.zero_grad()
        model(batch).square().mean().backward()
        optimizer.step()

    assert rates, "the plugin contributed no parameters at all"
    for (name, p), start in zip(rates, before, strict=True):
        assert not torch.equal(p.detach(), start), f"{name} never moved"


@pytest.mark.contract
@pytest.mark.parametrize("letter", BIASED)
def test_bias_stats_reports_the_two_numbers_the_gate_reads(letter: str) -> None:
    batch, _ = make_fake_batch(b=2, seed=1)
    stats = _plugin(letter).bias_stats(batch.pos, batch.meta)

    assert set(stats) == {"bias_mean", "bias_std"}
    assert all(isinstance(v, float) for v in stats.values())


# --- the gate that bias scales are comparable at the start ------------------


@pytest.mark.claim(
    "both bias metrics are normalized to [0,1] inside the plugin, so B and C enter the softmax at comparable magnitude at init; docs/method.md, the gate that bias scales are comparable at the start"
)
def test_bias_scales_agree_at_init() -> None:
    """**The gate that bias scales are comparable at the start.** Raw, the two
    metrics sit roughly fifty times apart in the same logit, so variant C would
    start with a far stronger prior than variant B and the comparison would
    measure initialization instead of physics. Normalized to ``[0, 1]`` inside
    each plugin, they must agree to within a factor of two.

    A check, not a tuning knob: if this fails, the normalization is wrong.
    """
    batch, _ = make_fake_batch(b=8, seed=0)

    b_mean = SpaceDistanceBias(H).bias_stats(batch.pos, batch.meta)["bias_mean"]
    c_mean = ChainDistanceBias(H).bias_stats(batch.pos, batch.meta)["bias_mean"]

    ratio = b_mean / c_mean
    assert 0.5 <= ratio <= 2.0, f"B/C bias means differ by {ratio:.2f}x at init"


@pytest.mark.claim(
    "both bias metrics are normalized to [0,1] inside the plugin, so B and C enter the softmax at comparable magnitude at init; docs/method.md, the gate that bias scales are comparable at the start"
)
def test_the_gate_would_fail_without_the_normalization() -> None:
    """The gate has teeth. Un-normalized, the same two metrics are 31x apart on
    this population (the spec predicted ~50x from the raw ranges), which is the
    failure the gate exists to catch. Normalized, they land at 0.88x."""
    batch, _ = make_fake_batch(b=8, seed=0)

    raw_space = pairwise_distance(batch.pos).mean()
    raw_chain = chain_distance(N).float().mean()

    assert not 0.5 <= float(raw_chain / raw_space) <= 2.0


@pytest.mark.claim(
    "both bias metrics are normalized to [0,1] inside the plugin, so B and C enter the softmax at comparable magnitude at init; docs/method.md, the gate that bias scales are comparable at the start"
)
def test_both_rates_start_from_the_same_number() -> None:
    """A per-variant init would be exactly the tuning forbidden by the gate that
    bias scales are comparable at the start."""
    space, chain = SpaceDistanceBias(H), ChainDistanceBias(H)

    expected = torch.full((H,), RATE_INIT)
    assert torch.allclose(torch.nn.functional.softplus(space.rate), expected, atol=1e-6)
    assert torch.allclose(torch.nn.functional.softplus(chain.rate), expected, atol=1e-6)


# --- variant B: 3D distance, per step ---------------------------------------


@pytest.mark.contract
def test_variant_b_reads_positions_and_normalizes_by_cable_length() -> None:
    """``γ_h · ‖p_i − p_j‖ / length``, with the rate at one at init."""
    batch, _ = make_fake_batch(b=3, seed=2)
    b = SpaceDistanceBias(H)(batch.pos, batch.meta)

    expected = pairwise_distance(batch.pos) / batch.meta.length.view(-1, 1, 1)
    assert torch.allclose(b, RATE_INIT * expected.unsqueeze(1), atol=1e-6)


@pytest.mark.contract
def test_variant_b_moves_when_the_cable_moves() -> None:
    """State-dependent by construction, which is what makes it recomputed per
    step and what makes rollout drift the bias too."""
    batch, _ = make_fake_batch(b=2, seed=2)
    plugin = SpaceDistanceBias(H)
    folded = dataclasses.replace(batch, pos=batch.pos * 0.5)

    assert not torch.allclose(plugin(batch.pos, batch.meta), plugin(folded.pos, folded.meta))


@pytest.mark.regression(
    bug="torch.cdist default mm expansion made variant B read absolute position, 2.8e-3 on a 0.86 scale under a 5 m shift"
)
def test_variant_b_is_translation_invariant() -> None:
    """It reads a pairwise distance, so it must not care where the cable is:
    the same invariance that keeps positions out of the node features."""
    batch, _ = make_fake_batch(b=2, seed=2)
    plugin = SpaceDistanceBias(H)
    moved = dataclasses.replace(batch, pos=batch.pos + torch.tensor([5.0, -2.0, 1.0]))

    assert torch.allclose(plugin(batch.pos, batch.meta), plugin(moved.pos, moved.meta), atol=1e-6)


# --- variant C: chain distance, static --------------------------------------


@pytest.mark.contract
def test_variant_c_is_the_chain_metric_and_ignores_the_cable() -> None:
    """``β_h · |i − j| / (N − 1)``: fixed by topology, so two different cables get
    the same bias and it broadcasts over the batch."""
    batch, _ = make_fake_batch(b=4, seed=3)
    b = ChainDistanceBias(H)(batch.pos, batch.meta)

    assert b.shape == (1, H, N, N)
    expected = chain_distance(N).float() / (N - 1)
    assert torch.allclose(b[0, 0], RATE_INIT * expected, atol=1e-6)
    assert float(b.max().detach()) == pytest.approx(RATE_INIT, abs=1e-6)


@pytest.mark.contract
def test_variant_c_is_symmetric_and_zero_on_the_diagonal() -> None:
    """A node is at distance zero from itself, so it is never suppressed from
    attending to itself."""
    batch, _ = make_fake_batch(seed=3)
    b = ChainDistanceBias(H)(batch.pos, batch.meta)

    assert torch.allclose(b, b.transpose(-1, -2))
    assert torch.all(torch.diagonal(b, dim1=-2, dim2=-1) == 0)


# --- variant D: per-head mixture --------------------------------------------


@pytest.mark.contract
def test_variant_d_splits_the_heads_into_thirds() -> None:
    """Fixed, not learned: a learned gate would confound "does specialization
    help" with "can a gate find it"."""
    assert assign_sources(6) == [
        BiasSource.SPACE,
        BiasSource.SPACE,
        BiasSource.CHAIN,
        BiasSource.CHAIN,
        BiasSource.NONE,
        BiasSource.NONE,
    ]
    assert len(assign_sources(7)) == 7


@pytest.mark.contract
def test_variant_d_gives_each_head_the_metric_it_was_assigned() -> None:
    """Head for head against the single-metric variants, which is what makes D a
    mixture of them rather than a fourth thing."""
    batch, _ = make_fake_batch(b=3, seed=4)
    mixed = MixedHeadBias(H)(batch.pos, batch.meta)
    space = SpaceDistanceBias(H)(batch.pos, batch.meta)
    chain = ChainDistanceBias(H)(batch.pos, batch.meta).expand_as(mixed)

    for head, source in enumerate(assign_sources(H)):
        if source is BiasSource.SPACE:
            assert torch.allclose(mixed[:, head], space[:, head], atol=1e-6)
        elif source is BiasSource.CHAIN:
            assert torch.allclose(mixed[:, head], chain[:, head], atol=1e-6)
        else:
            assert torch.all(mixed[:, head] == 0)


@pytest.mark.regression(
    bug="adding the eight-head F arm could silently change D's historical 2B/2C/4A allocation"
)
def test_variant_d_keeps_its_historical_eight_head_allocation() -> None:
    """D keeps two B heads, two C heads, and four unbiased heads at H=8."""
    batch, _ = make_fake_batch(b=3, seed=4)
    n_heads = 8
    plugin = MixedHeadBias(n_heads)
    mixed = plugin(batch.pos, batch.meta)
    space = SpaceDistanceBias(n_heads)(batch.pos, batch.meta)
    chain = ChainDistanceBias(n_heads)(batch.pos, batch.meta).expand_as(mixed)

    assert plugin.sources == [
        BiasSource.SPACE,
        BiasSource.SPACE,
        BiasSource.CHAIN,
        BiasSource.CHAIN,
        BiasSource.NONE,
        BiasSource.NONE,
        BiasSource.NONE,
        BiasSource.NONE,
    ]
    assert torch.allclose(mixed[:, :2], space[:, :2], atol=1e-6)
    assert torch.allclose(mixed[:, 2:4], chain[:, 2:4], atol=1e-6)
    assert torch.count_nonzero(mixed[:, 4:]) == 0
    assert sum(parameter.numel() for parameter in plugin.parameters()) == n_heads


@pytest.mark.claim(
    "F assigns [B,B,C,C,B,B,C,C] at H=8, preserving D's first four heads while giving every head one normalized physical prior; docs/method.md"
)
def test_variant_f_uses_the_fixed_eight_head_physical_prior_allocation() -> None:
    """F is head-for-head equal to B or C and carries eight nonnegative rates."""
    plugin_type = getattr(model_api, "BCOnlyBias", None)
    assert plugin_type is not None, "the public model API must expose F's bias plugin"

    batch, _ = make_fake_batch(b=3, seed=8)
    n_heads = 8
    plugin = plugin_type(n_heads)
    bias = plugin(batch.pos, batch.meta)
    space = SpaceDistanceBias(n_heads)(batch.pos, batch.meta)
    chain = ChainDistanceBias(n_heads)(batch.pos, batch.meta).expand_as(bias)
    expected_sources = [
        BiasSource.SPACE,
        BiasSource.SPACE,
        BiasSource.CHAIN,
        BiasSource.CHAIN,
        BiasSource.SPACE,
        BiasSource.SPACE,
        BiasSource.CHAIN,
        BiasSource.CHAIN,
    ]

    assert plugin.sources == expected_sources
    for head, source in enumerate(expected_sources):
        expected = space[:, head] if source is BiasSource.SPACE else chain[:, head]
        assert torch.allclose(bias[:, head], expected, atol=1e-6)
    assert sum(parameter.numel() for parameter in plugin.parameters()) == n_heads
    assert plugin.rate.shape == (n_heads,)

    with torch.no_grad():
        plugin.rate.copy_(torch.randn(n_heads) * 5)
    assert torch.all(plugin(batch.pos, batch.meta) >= 0)


@pytest.mark.contract
@pytest.mark.parametrize("n_heads", [1, 6, 7, 9])
def test_variant_f_rejects_every_head_count_except_eight(n_heads: int) -> None:
    plugin_type = getattr(model_api, "BCOnlyBias", None)
    assert plugin_type is not None, "the public model API must expose F's bias plugin"

    with pytest.raises(ValueError, match="variant F requires n_heads=8"):
        plugin_type(n_heads)


@pytest.mark.claim(
    "variants A through D differ by at most H learnable scalars, so a gap is attributable to the prior and not to capacity; docs/method.md"
)
def test_variant_d_costs_the_same_parameters_as_b_and_c() -> None:
    """The matched-budget claim: A through D differ by at most ``H`` scalars, so
    a gap is attributable to the prior rather than to capacity."""
    counts = {
        letter: sum(p.numel() for p in getattr(_plugin(letter), "parameters", list)())
        for letter in ("B", "C", "D")
    }
    assert set(counts.values()) == {H}


# --- the nesting that makes the comparison mean anything --------------------


@pytest.mark.claim(
    "a zero rate recovers variant A exactly, so the variants are nested rather than merely different; docs/method.md"
)
@pytest.mark.parametrize("cls", [SpaceDistanceBias, ChainDistanceBias, MixedHeadBias])
def test_a_zero_rate_recovers_variant_a(cls: type[RatedBias]) -> None:
    """``γ_h = 0`` gives back plain attention exactly, so the variants are nested
    rather than merely different, which is what lets a gap be read as the prior
    helping rather than as two unrelated models."""
    batch, _ = make_fake_batch(b=2, seed=6)
    plugin = cls(H)
    with torch.no_grad():
        plugin.rate.fill_(-80.0)  # softplus(-80) is 1.8e-35, zero for any use

    b = plugin(batch.pos, batch.meta)
    assert torch.allclose(b, torch.zeros_like(b), atol=1e-20)


@pytest.mark.claim(
    "softplus keeps b >= 0, so a bias can only suppress a pair and never amplify it; docs/method.md"
)
@pytest.mark.parametrize("cls", [SpaceDistanceBias, ChainDistanceBias, MixedHeadBias])
def test_the_bias_can_only_suppress(cls: type[RatedBias]) -> None:
    """Softplus keeps ``b >= 0``, so a variant can make a pair invisible but never
    amplify it beyond what the content logits already say."""
    batch, _ = make_fake_batch(b=3, seed=7)
    plugin = cls(H)
    with torch.no_grad():
        plugin.rate.copy_(torch.randn(H) * 5)

    assert torch.all(plugin(batch.pos, batch.meta) >= 0)


@pytest.mark.contract
def test_the_registry_names_every_variant_the_write_up_names() -> None:
    """Explicit registration, and the letters are the ones the paper uses."""
    assert set(VARIANTS) == {"A", "B", "C", "D", "F"}
    assert isinstance(VARIANTS["A"](H), NoBias)
    assert VARIANTS["F"](8).__class__ is getattr(model_api, "BCOnlyBias", None)
