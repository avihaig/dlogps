"""The stub batch: contract shapes, no disk, and a cable that is not a straight line.

``make_fake_batch`` is what the block and the harness are built against before
any data exists, so a defect here is a defect in every downstream suite at once
and would be read as a model bug. These are the properties both consumers rely
on.
"""

from __future__ import annotations

import pytest
import torch

from dlogps.data.dataset import feature_width
from dlogps.data.types import E, N
from tests.fake_data import (
    FEATURE_WIDTH,
    HISTORY_FRAMES,
    chain_edge_index,
    make_fake_batch,
)


@pytest.mark.contract
def test_shapes_match_the_contract() -> None:
    """Every field at the rank and width the batch contract freezes."""
    b, r = 3, 4
    batch, labels = make_fake_batch(b=b, r=r, seed=0)

    assert batch.pos.shape == (b, N, 3)
    assert batch.node_feat.shape == (b, N, FEATURE_WIDTH)
    assert batch.edge_index.shape == (2, 2 * E)
    assert batch.edge_feat.shape == (b, 2 * E, 5)
    assert batch.target.shape == (b, N, 3)
    assert labels.contact_frame.shape == (b, r)
    assert labels.contact_pair.shape == (b, r, E, E)

    for name in ("cable_id", "length", "diameter", "dt", "is_held_out"):
        assert getattr(batch.meta, name).shape == (b,)


@pytest.mark.contract
def test_dtypes_are_float32_except_the_declared_ones() -> None:
    """``float64`` lives on disk only; ids are int64 and the tags are bool."""
    batch, labels = make_fake_batch()

    for name in ("pos", "node_feat", "edge_feat", "target"):
        assert getattr(batch, name).dtype == torch.float32
    assert batch.edge_index.dtype == torch.int64
    assert batch.meta.cable_id.dtype == torch.int64
    assert batch.meta.is_held_out.dtype == torch.bool
    assert labels.contact_frame.dtype == torch.bool
    assert labels.contact_pair.dtype == torch.bool


@pytest.mark.contract
def test_every_chain_edge_is_stored_in_both_directions() -> None:
    """A model that assumes one direction silently drops half the messages, so
    the stub must not be the thing that lets that pass."""
    edge_index = chain_edge_index(N)
    pairs = {(int(s), int(d)) for s, d in edge_index.t()}

    assert len(pairs) == 2 * E
    assert all((d, s) in pairs for s, d in pairs)
    assert all(abs(s - d) == 1 for s, d in pairs)


@pytest.mark.claim(
    "the stub batch is the simulator-free fixture: contract shapes, plausible geometry, no file on disk"
)
def test_segments_sit_at_their_rest_length() -> None:
    """Link-length constancy is a scored metric, so the stub must read as zero
    violation: a metric that fires on this batch is measuring the stub."""
    batch, _ = make_fake_batch(b=4, seed=7)
    seg_rest = batch.meta.length / E

    lengths = (batch.pos[:, 1:] - batch.pos[:, :-1]).norm(dim=-1)
    assert torch.allclose(lengths, seg_rest.unsqueeze(1).expand_as(lengths), atol=1e-6)


@pytest.mark.contract
def test_edge_features_agree_with_the_positions() -> None:
    """``edge_feat`` is derived, not sampled: displacement (3), distance (1),
    rest length (1), all consistent with ``pos``."""
    batch, _ = make_fake_batch(b=2, seed=3)
    src, dst = batch.edge_index[0], batch.edge_index[1]

    disp = batch.edge_feat[..., :3]
    assert torch.allclose(disp, batch.pos[:, dst] - batch.pos[:, src], atol=1e-6)
    assert torch.allclose(batch.edge_feat[..., 3], disp.norm(dim=-1), atol=1e-6)


@pytest.mark.claim(
    "the stub batch is the simulator-free fixture: contract shapes, plausible geometry, no file on disk"
)
def test_the_cable_is_curved_and_above_the_floor() -> None:
    """A straight line would make the chain-distance and 3D-distance biases
    identical, which is precisely the contrast the experiment turns on."""
    batch, _ = make_fake_batch(b=4, seed=11)

    end_to_end = (batch.pos[:, -1] - batch.pos[:, 0]).norm(dim=-1)
    assert torch.all(end_to_end < 0.95 * batch.meta.length)
    assert torch.all(batch.pos[..., 2] >= -1e-6)


@pytest.mark.contract
def test_node_features_carry_the_declared_channels() -> None:
    """``F`` is velocity history, four material params, floor distance."""
    batch, _ = make_fake_batch(b=2, seed=1)
    # The stub computes its width from the contract and the loader computes it
    # from the same contract; the check is worth having because the two are
    # written separately.
    assert FEATURE_WIDTH == HISTORY_FRAMES * 3 + 5 == feature_width(HISTORY_FRAMES)

    material = batch.node_feat[..., HISTORY_FRAMES * 3 : HISTORY_FRAMES * 3 + 4]
    assert torch.allclose(material, material[:, :1].expand_as(material))
    assert torch.all(batch.node_feat[..., -1] >= 0.0)

    # Every one of the four is reachable from meta, in the declared order. A
    # rollout is handed only State and BatchMeta, so a material param that lives
    # solely in node_feat cannot be rebuilt.
    for offset, expected in enumerate(
        [
            batch.meta.bend_stiffness,
            batch.meta.joint_damping,
            batch.meta.diameter,
            batch.meta.effective_linear_density,
        ]
    ):
        assert torch.allclose(material[..., offset], expected.unsqueeze(1).expand(2, N))


@pytest.mark.contract
def test_labels_never_mark_an_adjacent_pair() -> None:
    """Self-contact is 3D-close and **chain-far**; adjacent segments touching is
    the cable being a cable."""
    _, labels = make_fake_batch(b=2, r=6, seed=5)
    seg = torch.arange(E)
    adjacent = (seg.view(-1, 1) - seg.view(1, -1)).abs() <= 2

    assert not bool(labels.contact_pair[:, :, adjacent].any())
    assert torch.equal(labels.contact_pair, labels.contact_pair.transpose(-1, -2))


@pytest.mark.contract
def test_seeded_and_independent_of_global_rng() -> None:
    """Two suites call this; neither may perturb the other."""
    torch.manual_seed(0)
    first, _ = make_fake_batch(seed=4)
    torch.manual_seed(999)
    second, _ = make_fake_batch(seed=4)
    other, _ = make_fake_batch(seed=5)

    assert torch.equal(first.pos, second.pos)
    assert not torch.equal(first.pos, other.pos)


@pytest.mark.contract
@pytest.mark.parametrize(("b", "r"), [(0, 1), (1, 0)])
def test_rejects_an_empty_axis(b: int, r: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        make_fake_batch(b=b, r=r)
