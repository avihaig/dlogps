"""Feature statistics, and the two ways they can be silently wrong.

Most of this file is shape and round-trip checking. Two tests carry the module:
:func:`test_pos_survives_normalization_unscaled`, because scaling ``pos`` breaks
the 3D-distance variant specifically and reads as a null result rather than as a
crash; and :func:`test_a_constant_channel_gets_unit_scale`, because the material
params are constant within a cable and a variance of zero is reachable on any
single-cable stream.
"""

from __future__ import annotations

import pytest
import torch

from dlogps.harness.normalize import UNIT_STD_BELOW, Stats, denormalize_step, fit_stats, normalize
from tests.fake_data import FEATURE_WIDTH, make_fake_batch


def _stream(n: int = 4, b: int = 3) -> list:
    """``n`` independent stub batches, one per seed."""
    return [make_fake_batch(b=b, seed=s)[0] for s in range(n)]


@pytest.mark.contract
def test_fit_stats_returns_the_contracted_shapes() -> None:
    stats = fit_stats(_stream())

    assert stats.node_mean.shape == (FEATURE_WIDTH,)
    assert stats.node_std.shape == (FEATURE_WIDTH,)
    assert stats.edge_mean.shape == (5,)
    assert stats.edge_std.shape == (5,)
    assert stats.step_mean.shape == (3,)
    assert stats.step_std.shape == (3,)
    assert all(t.dtype == torch.float32 for t in stats.tensors())


@pytest.mark.contract
def test_statistics_match_a_direct_computation() -> None:
    """Streaming accumulation has to agree with the obvious one-shot version."""
    batches = _stream()
    stats = fit_stats(batches, param_mode="fitted")

    node = torch.cat([b.node_feat.reshape(-1, FEATURE_WIDTH) for b in batches])
    step = torch.cat([(b.target - b.pos).reshape(-1, 3) for b in batches])

    assert torch.allclose(stats.node_mean, node.mean(0), atol=1e-6)
    assert torch.allclose(stats.node_std, node.std(0, unbiased=False), atol=1e-6)
    assert torch.allclose(stats.step_mean, step.mean(0), atol=1e-9)
    assert torch.allclose(stats.step_std, step.std(0, unbiased=False), atol=1e-9)


@pytest.mark.contract
def test_fit_stats_rejects_an_empty_stream() -> None:
    with pytest.raises(ValueError, match="empty stream"):
        fit_stats([])


@pytest.mark.claim("statistics never rescale pos")
def test_pos_survives_normalization_unscaled() -> None:
    """``pos`` reaches the bias plugins in metres, or variant B measures nothing.

    Asserted by identity rather than by closeness: the plugins compare ``pos``
    against ``meta.diameter`` and ``meta.length``, both of which are in metres and
    neither of which is normalized, so *any* rescaling of ``pos`` is a defect.
    """
    batch, _ = make_fake_batch(b=3)
    out = normalize(batch, fit_stats(_stream()))

    assert torch.equal(out.pos, batch.pos)
    assert torch.equal(out.target, batch.target)
    assert torch.equal(out.edge_index, batch.edge_index)
    assert out.meta is batch.meta


@pytest.mark.contract
def test_normalize_whitens_the_two_feature_channels() -> None:
    """Fitted and applied on the same stream, the features come out standardized."""
    batches = _stream(n=6, b=4)
    stats = fit_stats(batches, param_mode="fitted")

    node = torch.cat([normalize(b, stats).node_feat.reshape(-1, FEATURE_WIDTH) for b in batches])
    edge = torch.cat([normalize(b, stats).edge_feat.reshape(-1, 5) for b in batches])

    assert torch.allclose(node.mean(0), torch.zeros(FEATURE_WIDTH), atol=1e-4)
    assert torch.allclose(node.std(0, unbiased=False), torch.ones(FEATURE_WIDTH), atol=1e-3)
    assert torch.allclose(edge.mean(0), torch.zeros(5), atol=1e-4)


@pytest.mark.contract
def test_normalize_does_not_mutate_its_input() -> None:
    batch, _ = make_fake_batch(b=2)
    before = batch.node_feat.clone()
    normalize(batch, fit_stats(_stream()))

    assert torch.equal(batch.node_feat, before)


@pytest.mark.regression("a zero-variance channel divided by eps and blew up the feature")
def test_a_constant_channel_gets_unit_scale() -> None:
    """One cable makes every material param constant, so the variance is exactly 0.

    Flooring at ``UNIT_STD_BELOW`` rather than replacing would divide the
    channel's float32 rounding noise by 1e-6 and hand the model its largest
    feature by a factor of a million.
    """
    batch, _ = make_fake_batch(b=1)
    stats = fit_stats([batch])

    constant = stats.node_std == 1.0
    assert constant.any(), "a single cable has constant material params"
    out = normalize(batch, stats)
    assert torch.isfinite(out.node_feat).all()
    assert out.node_feat.abs().max() < 1e3


@pytest.mark.contract
def test_step_normalization_round_trips() -> None:
    stats = fit_stats(_stream())
    step = torch.randn(4, 33, 3)

    assert torch.allclose(denormalize_step((step - stats.step_mean) / stats.step_std, stats), step)


@pytest.mark.contract
def test_unit_std_threshold_is_a_replacement_not_a_floor() -> None:
    """Below the threshold the scale is exactly one, not the threshold itself."""
    stats = fit_stats([make_fake_batch(b=1)[0]])

    assert not ((stats.node_std > 0) & (stats.node_std < UNIT_STD_BELOW)).any()


@pytest.mark.contract
def test_stats_move_between_devices() -> None:
    stats = fit_stats(_stream()).to("cpu")

    assert isinstance(stats, Stats)
    assert all(t.device.type == "cpu" for t in stats.tensors())


@pytest.mark.claim("designed encoding maps the declared spans to [-1, +1]")
def test_designed_encoding_hits_the_declared_span_endpoints() -> None:
    from dlogps.harness.normalize import encode_params

    # MATERIAL_FIELDS order: bend_stiffness, joint_damping, diameter, eff_linear_density.
    low = torch.tensor([[1e4, 1e-5, 5e-4, 1e-4]])
    mid = torch.tensor([[1e7, 1e-3, 10**-2.301, 10**-1.3495]])
    high = torch.tensor([[1e10, 1e-1, 5e-2, 20.0]])

    assert torch.allclose(encode_params(low), torch.full((1, 4), -1.0), atol=1e-3)
    assert torch.allclose(encode_params(mid), torch.zeros(1, 4), atol=1e-3)
    assert torch.allclose(encode_params(high), torch.full((1, 4), 1.0), atol=1e-3)


@pytest.mark.claim("values outside a designed span encode past +/-1, never clamped")
def test_designed_encoding_does_not_clamp() -> None:
    from dlogps.harness.normalize import encode_params

    beyond = torch.tensor([[1e13, 1e-5, 5e-4, 1e-4]])  # E three decades past the span
    assert encode_params(beyond)[0, 0] > 1.9


@pytest.mark.claim("designed mode replaces z-scoring for the material slice only")
def test_designed_mode_encodes_material_and_fits_the_rest() -> None:
    from dlogps.harness.normalize import MATERIAL_CHANNELS, encode_params

    batches = _stream()
    fitted = fit_stats(batches, param_mode="fitted")
    designed = fit_stats(batches)

    assert designed.param_mode == "designed"
    # Material channels are pinned to identity; every other channel matches fitted.
    assert torch.all(designed.node_mean[MATERIAL_CHANNELS] == 0.0)
    assert torch.all(designed.node_std[MATERIAL_CHANNELS] == 1.0)
    assert torch.allclose(designed.node_mean[:-5], fitted.node_mean[:-5])
    assert torch.allclose(designed.node_std[-1:], fitted.node_std[-1:])

    batch = batches[0]
    out = normalize(batch, designed)
    want = encode_params(batch.node_feat[..., MATERIAL_CHANNELS])
    assert torch.allclose(out.node_feat[..., MATERIAL_CHANNELS], want, atol=1e-6)
    # Non-material channels still whiten exactly as fitted stats dictate.
    ref = normalize(batch, fitted)
    assert torch.allclose(out.node_feat[..., :-5], ref.node_feat[..., :-5], atol=1e-6)


@pytest.mark.claim("the designed encoding is the default and survives device moves")
def test_stats_default_mode_is_designed_and_survives_device_moves() -> None:
    stats = fit_stats(_stream())
    assert stats.param_mode == "designed"
    assert stats.to("cpu").param_mode == "designed"

    fitted = fit_stats(_stream(), param_mode="fitted")
    assert fitted.to("cpu").param_mode == "fitted"
    with pytest.raises(ValueError, match="param_mode"):
        Stats(*stats.tensors(), param_mode="scaled")
