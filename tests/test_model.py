"""The block against the frozen interface: shapes, the two ablations, and Eq. 1.

These are the block's contract items written
as tests. The one that earns its keep is
:func:`test_bias_is_subtracted_before_the_softmax`: the sign of the bias is the
single assumption the whole experiment rests on, it trains perfectly well when
inverted, and an inverted prior would be discovered in the results table rather
than here.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from dlogps.data.types import Batch, BatchMeta, N
from dlogps.model import BiasPlugin, GPSCfg, GPSModel, NoBias
from dlogps.model.gps import BiasedAttention
from tests.fake_data import make_fake_batch

CFG = GPSCfg(d_model=24, n_heads=6, n_layers=2)
"""Small enough to be fast, wide enough to keep six heads with four channels
each. Nothing in this suite is sensitive to the width."""


class _ConstantBias:
    """A stub plugin returning a fixed ``b``, for shape and sign tests."""

    def __init__(self, value: torch.Tensor) -> None:
        self.value = value

    def __call__(self, pos: torch.Tensor, meta: BatchMeta) -> torch.Tensor:
        del pos, meta
        return self.value

    def bias_stats(self, pos: torch.Tensor, meta: BatchMeta) -> dict[str, float]:
        del pos, meta
        return {"bias_mean": float(self.value.mean()), "bias_std": float(self.value.std())}


# --- the interface ----------------------------------------------------------


@pytest.mark.contract
def test_variant_a_satisfies_the_protocol() -> None:
    """Variant A has no ``b``, so its stats are empty rather than zeroed."""
    batch, _ = make_fake_batch()
    plugin = NoBias()

    assert isinstance(plugin, BiasPlugin)
    assert plugin(batch.pos, batch.meta) is None
    assert plugin.bias_stats(batch.pos, batch.meta) == {}


# --- the end-to-end shape ---------------------------------------------------


@pytest.mark.contract
def test_variant_a_runs_end_to_end_and_returns_the_contract_shape() -> None:
    """Half of the gate that the harness tells two models apart: the no-bias
    block on the stub batch, no data involved."""
    batch, _ = make_fake_batch(b=3, seed=0)
    model = GPSModel.from_batch(batch, CFG)

    out = model(batch)

    assert out.shape == batch.target.shape
    assert torch.isfinite(out).all()


@pytest.mark.claim(
    "the block is translation invariant: pos reaches it only through edge_feat and the bias, never node_feat; docs/method.md"
)
def test_the_head_predicts_a_step_and_not_a_place() -> None:
    """``pos`` is not a node feature, so an absolute-position head would be asked
    for something the network cannot see. Translating the cable must translate
    the prediction by exactly the same amount."""
    batch, _ = make_fake_batch(b=2, seed=2)
    model = GPSModel.from_batch(batch, CFG).eval()
    shift = torch.tensor([3.0, -1.0, 0.5])
    moved = dataclasses.replace(batch, pos=batch.pos + shift)

    with torch.no_grad():
        base, after = model(batch), model(moved)

    assert torch.allclose(after, base + shift, atol=1e-5)


@pytest.mark.contract
def test_gradients_reach_both_streams() -> None:
    """A stream that is wired but never differentiated is a stream that is off."""
    batch, _ = make_fake_batch(b=2, seed=6)
    model = GPSModel.from_batch(batch, CFG)

    model(batch).square().mean().backward()

    named = dict(model.named_parameters())
    assert named["layers.0.local.lin_msg.weight"].grad is not None
    assert named["layers.0.attn.qkv.weight"].grad is not None
    assert named["edge_encoder.weight"].grad is not None


# --- the ablations ----------------------------------------------------------


@pytest.mark.contract
def test_chain_only_runs_the_same_path_with_the_global_stream_off() -> None:
    """The other half of the gate that the harness tells two models apart, and it
    is one flag rather than a second model: the chain-only baseline belongs to
    the block, not to the baselines."""
    batch, _ = make_fake_batch(b=2, seed=1)
    model = GPSModel.from_batch(batch, dataclasses.replace(CFG, use_global=False))

    out = model(batch)

    assert out.shape == batch.target.shape
    assert all(layer.attn is None for layer in model.layers)
    assert not any("attn.qkv" in name for name, _ in model.named_parameters())


@pytest.mark.contract
def test_the_local_stream_ablation_runs() -> None:
    """The crossed ablation: attention alone, no message passing."""
    batch, _ = make_fake_batch(b=2, seed=1)
    model = GPSModel.from_batch(batch, dataclasses.replace(CFG, use_local=False))

    assert model(batch).shape == batch.target.shape
    assert all(layer.local is None for layer in model.layers)


@pytest.mark.contract
def test_a_block_with_both_streams_off_is_rejected() -> None:
    with pytest.raises(ValueError, match="computes nothing"):
        GPSCfg(use_local=False, use_global=False)


@pytest.mark.contract
def test_the_bias_is_not_computed_when_the_global_stream_is_off() -> None:
    """Chain-only must not pay for a bias nothing consumes."""
    batch, _ = make_fake_batch(b=2, seed=1)
    calls: list[int] = []

    class _Counting(_ConstantBias):
        def __call__(self, pos: torch.Tensor, meta: BatchMeta) -> torch.Tensor:
            calls.append(1)
            return super().__call__(pos, meta)

    plugin = _Counting(torch.zeros(1, CFG.n_heads, N, N))
    GPSModel.from_batch(batch, dataclasses.replace(CFG, use_global=False), plugin)(batch)
    assert calls == []


@pytest.mark.contract
def test_the_bias_is_computed_once_and_shared_by_every_layer() -> None:
    """Once per forward pass is correct because the variants read *input*
    positions, which do not move within a step."""
    batch, _ = make_fake_batch(b=2, seed=1)
    calls: list[int] = []

    class _Counting(_ConstantBias):
        def __call__(self, pos: torch.Tensor, meta: BatchMeta) -> torch.Tensor:
            calls.append(1)
            return super().__call__(pos, meta)

    plugin = _Counting(torch.zeros(1, CFG.n_heads, N, N))
    GPSModel.from_batch(batch, CFG, plugin)(batch)
    assert len(calls) == 1
    assert CFG.n_layers > 1


# --- Eq. 1: the sign, and the pre-softmax placement -------------------------


@pytest.mark.claim(
    "the attention bias is subtracted pre-softmax, so it scales the weight by exp(-b); docs/method.md"
)
def test_bias_is_subtracted_before_the_softmax() -> None:
    """The whole experiment in one assertion.

    For a bias that is ``c`` on one key column and zero elsewhere, subtracting
    pre-softmax multiplies that column's odds against any other column by
    exactly ``exp(-c)``, and the normalizer cancels out of the ratio. Adding
    instead gives ``exp(+c)``; applying the bias after the softmax gives neither.
    """
    torch.manual_seed(0)
    attn = BiasedAttention(CFG).eval()
    x = torch.randn(2, N, CFG.d_model)
    c, target, other = 1.5, 7, 19

    bias = torch.zeros(1, CFG.n_heads, N, N)
    bias[..., target] = c

    with torch.no_grad():
        plain = attn.attention_weights(x, None)
        biased = attn.attention_weights(x, bias)

    odds = (biased[..., target] / biased[..., other]) / (plain[..., target] / plain[..., other])
    assert torch.allclose(odds, torch.full_like(odds, math.exp(-c)), atol=1e-4)


@pytest.mark.claim(
    "the attention bias is subtracted pre-softmax, so it scales the weight by exp(-b); docs/method.md"
)
def test_a_large_bias_makes_a_pair_invisible() -> None:
    """The reading of Eq. 1 in words: large ``b`` means ignore that pair."""
    torch.manual_seed(0)
    attn = BiasedAttention(CFG).eval()
    x = torch.randn(2, N, CFG.d_model)

    keep = 4
    bias = torch.full((1, CFG.n_heads, N, N), 30.0)
    bias[..., keep] = 0.0

    with torch.no_grad():
        weights = attn.attention_weights(x, bias)

    assert torch.allclose(weights[..., keep], torch.ones_like(weights[..., keep]), atol=1e-5)


@pytest.mark.contract
def test_a_zero_bias_leaves_attention_untouched() -> None:
    torch.manual_seed(0)
    attn = BiasedAttention(CFG).eval()
    x = torch.randn(2, N, CFG.d_model)

    with torch.no_grad():
        assert torch.allclose(
            attn.attention_weights(x, None),
            attn.attention_weights(x, torch.zeros(1, CFG.n_heads, N, N)),
            atol=1e-6,
        )


# --- the shapes the interface must refuse -----------------------------------


@pytest.mark.contract
def test_a_static_bias_broadcasts_over_the_batch() -> None:
    """``[1, H, N, N]`` is explicitly allowed, and a static variant will use it."""
    batch, _ = make_fake_batch(b=3, seed=1)

    plugin = _ConstantBias(torch.rand(1, CFG.n_heads, N, N))
    assert GPSModel.from_batch(batch, CFG, plugin)(batch).shape == batch.target.shape


@pytest.mark.contract
@pytest.mark.parametrize(
    "shape",
    [
        (3, N, N),  # no head axis: would broadcast a per-head prior into a shared one
        (3, 1, N, N),  # one head, broadcast: same failure, spelled differently
        (3, CFG.n_heads, N, N + 1),
        (2, CFG.n_heads, N, N),  # wrong batch, and not the broadcastable 1
    ],
)
def test_a_bias_of_the_wrong_rank_is_refused(shape: tuple[int, ...]) -> None:
    """Silent broadcasting is the failure mode: it turns variant D into variant C."""
    batch, _ = make_fake_batch(b=3, seed=1)
    model = GPSModel.from_batch(batch, CFG, _ConstantBias(torch.zeros(shape)))

    with pytest.raises(ValueError, match="bias must be"):
        model(batch)


# --- the feature width is the loader's, never this file's -------------------


@pytest.mark.contract
def test_the_block_reads_the_feature_width_off_the_batch() -> None:
    """``F`` moves with the history depth, so nothing here may
    assume it. Two widths, two blocks, both fine."""
    batch, _ = make_fake_batch(b=2, seed=1)
    wider = dataclasses.replace(
        batch, node_feat=torch.cat([batch.node_feat, torch.rand(2, N, 7)], dim=-1)
    )

    narrow = GPSModel.from_batch(batch, CFG)
    wide = GPSModel.from_batch(wider, CFG)

    assert narrow.feature_width != wide.feature_width
    assert narrow(batch).shape == wide(wider).shape


@pytest.mark.contract
def test_a_batch_of_the_wrong_feature_width_is_refused() -> None:
    """Loud, not a silent reshape: the loader changing ``F`` under a built block
    is a contract change and must read as one."""
    batch, _ = make_fake_batch(b=2, seed=1)
    model = GPSModel.from_batch(batch, CFG)
    wider = dataclasses.replace(
        batch, node_feat=torch.cat([batch.node_feat, torch.rand(2, N, 1)], dim=-1)
    )

    with pytest.raises(ValueError, match="block built for F="):
        model(wider)


@pytest.mark.contract
def test_an_edge_channel_the_contract_does_not_name_is_refused() -> None:
    """The raw recorded edge channels stay out in phase 1; the stress channel is
    an extension, not core."""
    batch, _ = make_fake_batch(b=2, seed=1)
    model = GPSModel.from_batch(batch, CFG)
    fattened: Batch = dataclasses.replace(
        batch,
        edge_feat=torch.cat([batch.edge_feat, torch.rand(2, batch.edge_feat.shape[1], 1)], -1),
    )

    with pytest.raises(ValueError, match="edge_feat must be"):
        model(fattened)


@pytest.mark.contract
def test_d_model_must_split_across_the_heads() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        GPSCfg(d_model=25, n_heads=6)


@pytest.mark.regression(bug="the default GPSCfg was unusable: d_model=128 against six heads")
def test_the_default_config_is_usable() -> None:
    """Every other test here passes an explicit small config, so the defaults are
    the one path the suite would otherwise never take — and they were briefly
    unusable, ``d_model=128`` against six heads."""
    batch, _ = make_fake_batch(b=2, seed=1)

    assert GPSModel.from_batch(batch)(batch).shape == batch.target.shape


@pytest.mark.regression(
    "the head predicts a standardized step and the two buffers put it back into metres; "
    "dropping the scale, or the offset, or the trainer's copy into them left the whole fast "
    "suite green while the training loss inflated by seven orders of magnitude"
)
def test_the_head_speaks_metres_through_the_step_buffers() -> None:
    """The one place the model's units are decided, and nothing read it.

    ``head.weight`` is zero-initialized, so the head emits its bias whatever the
    input and the buffers alone decide what reaches the cable. That makes the
    arithmetic exact rather than approximate: at a zero bias the block must
    predict exactly the mean step, and at a known bias exactly
    ``bias * step_std + step_mean``.
    """
    batch, _ = make_fake_batch(b=2)
    model = GPSModel.from_batch(batch, CFG)
    step_std = torch.tensor([2.0, 3.0, 4.0])
    step_mean = torch.tensor([0.1, 0.2, 0.3])
    with torch.no_grad():
        model.step_std.copy_(step_std)
        model.step_mean.copy_(step_mean)

    # A zero head is the constant predictor: the mean step and nothing else.
    assert torch.allclose(model(batch) - batch.pos, step_mean.expand(batch.pos.shape))

    emitted = torch.tensor([1.0, -2.0, 0.5])
    with torch.no_grad():
        model.head.bias.copy_(emitted)

    expected = emitted * step_std + step_mean
    assert torch.allclose(model(batch) - batch.pos, expected.expand(batch.pos.shape))
    # Both ride in the state_dict, or a checkpoint cannot reproduce its own units.
    assert {"step_std", "step_mean"} <= set(model.state_dict())
