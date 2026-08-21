"""Reading a cable off disk and windowing it into contract-shaped batches.

**This is the first non-circular test of the contract.** Every type in
the batch contract has so far been satisfied by ``make_fake_batch``, written by
reading the contract and producing what it asked for. These tests build the same
types from real recorded bytes, so a contract that cannot be implemented shows up
here rather than at training time.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dlogps.data.dataset import DEFAULT_HISTORY_FRAMES as C
from dlogps.data.dataset import (
    MATERIAL_FIELDS,
    MAX_STEP_M,
    CableData,
    State,
    chain_edge_index,
    feature_width,
    features_from_state,
    load_cable,
    make_batch,
    state_from_batch,
    window_starts,
)
from dlogps.data.types import E, N
from tests.fake_data import make_fake_batch

SOFT, STIFF = "assets/sample_v1/cable_000", "assets/sample_v1/cable_045"


@pytest.fixture(scope="module")
def soft() -> CableData:
    return load_cable(SOFT)


@pytest.mark.contract
def test_a_cable_loads_with_its_padding_stripped(soft: CableData):
    assert soft.cable_id == 0
    assert soft.n_episodes == 2

    with np.load(f"{SOFT}/episodes.npz") as raw:
        lengths = [int(n) for n in raw["episode_lengths"]]
        padded = raw["vertex_pos"].shape[1]

    assert lengths == [661, 648], "the fixture's measured episode lengths"
    assert padded == 2383, "on disk every episode is padded to the timeout bound"
    for (pos, vel, t), n in zip(soft.episodes, lengths, strict=True):
        # Real frames only: padding is free on disk and 3x the frames in RAM.
        assert pos.shape == (n, N, 3)
        assert vel.shape == (n, N, 3)
        assert t.shape == (n,)
        assert torch.isfinite(pos).all(), "a NaN here is padding that was not stripped"


@pytest.mark.claim(
    "dt comes from np.diff(t), never record_hz: the requested rate is quantized to whole "
    "sim steps, so the stored field is 5% off and biases every finite difference "
    "(docs/data.md)"
)
def test_dt_is_measured_not_taken_from_the_stored_rate(soft: CableData):
    with np.load(f"{SOFT}/episodes.npz") as raw:
        stored = float(raw["record_hz"])

    assert soft.dt(0) == pytest.approx(0.0021, abs=1e-6)
    assert 1.0 / stored == pytest.approx(0.0020), "the stored rate would give 0.0020"


@pytest.mark.claim(
    "a window never crosses an episode boundary: a pair spanning two episodes joins a "
    "settled pile to a fresh airborne cable (docs/data.md)"
)
def test_no_window_reaches_outside_its_episode(soft: CableData):
    for pos, _, _ in soft.episodes:
        n = len(pos)
        starts = window_starts(n, C)
        assert starts.start == C - 1, "the first C-1 frames have no history"
        assert starts.stop == n - 1, "the last frame has no target"
        assert len(starts) == n - C, f"{n} frames yield {n - C} samples"
        for k in (starts.start, starts.stop - 1):
            assert k - C + 1 >= 0 and k + 1 < n


@pytest.mark.contract
def test_a_batch_off_real_bytes_matches_the_contract(soft: CableData):
    windows = [(0, 10), (0, 300), (1, 5)]
    batch = make_batch(soft, windows, is_held_out=False, history_frames=C)

    assert batch.pos.shape == (3, N, 3)
    assert batch.node_feat.shape == (3, N, feature_width(C))
    assert batch.edge_index.shape == (2, 2 * E)
    assert batch.edge_feat.shape == (3, 2 * E, 5)
    assert batch.target.shape == (3, N, 3)
    for name in ("pos", "node_feat", "edge_feat", "target"):
        assert getattr(batch, name).dtype == torch.float32
    assert batch.meta.cable_id.dtype == torch.int64
    assert batch.meta.is_held_out.dtype == torch.bool
    assert not bool(batch.meta.is_held_out.any())


@pytest.mark.claim(
    "the four material params reach node_feat from BatchMeta in the declared order, so a "
    "rollout handed only State and BatchMeta can rebuild them "
    ""
)
def test_the_material_params_come_off_the_npz_and_reach_every_node(soft: CableData):
    batch = make_batch(soft, [(0, 10), (1, 5)], is_held_out=False, history_frames=C)

    with np.load(f"{SOFT}/episodes.npz") as raw:
        for name in MATERIAL_FIELDS:
            assert getattr(batch.meta, name)[0].item() == pytest.approx(float(raw[name]))

    material = batch.node_feat[..., C * 3 : C * 3 + 4]
    assert torch.allclose(material, material[:, :1].expand_as(material)), "constant across nodes"
    for offset, name in enumerate(MATERIAL_FIELDS):
        expected = getattr(batch.meta, name).unsqueeze(1).expand(2, N)
        assert torch.allclose(material[..., offset], expected)


@pytest.mark.contract
def test_target_is_the_next_recorded_frame(soft: CableData):
    batch = make_batch(soft, [(0, 10)], is_held_out=False, history_frames=C)
    pos, _, _ = soft.episodes[0]
    assert torch.equal(batch.pos[0], pos[10])
    assert torch.equal(batch.target[0], pos[11])


@pytest.mark.contract
def test_velocity_history_is_newest_first(soft: CableData):
    batch = make_batch(soft, [(0, 10)], is_held_out=False, history_frames=C)
    _, vel, _ = soft.episodes[0]
    history = batch.node_feat[0, :, : C * 3].reshape(N, C, 3)
    for age in range(C):
        assert torch.equal(history[:, age], vel[10 - age]), f"index {age} is frame k-{age}"


@pytest.mark.contract
def test_edge_features_are_derived_from_positions_not_stored(soft: CableData):
    batch = make_batch(soft, [(0, 10), (1, 5)], is_held_out=False, history_frames=C)
    src, dst = batch.edge_index[0], batch.edge_index[1]

    disp = batch.edge_feat[..., :3]
    assert torch.allclose(disp, batch.pos[:, dst] - batch.pos[:, src], atol=1e-5)
    assert torch.allclose(batch.edge_feat[..., 3], disp.norm(dim=-1), atol=1e-5)
    assert torch.allclose(batch.edge_feat[..., 4], batch.meta.length.view(-1, 1) / E)

    pairs = {(int(s), int(d)) for s, d in chain_edge_index().t()}
    assert len(pairs) == 2 * E
    assert all((d, s) in pairs for s, d in pairs), "both directions, or half the messages vanish"


@pytest.mark.claim(
    "features_from_state builds the identical feature set the dataset builds, so training "
    "features and rollout features cannot skew apart"
)
def test_the_rollout_hook_reproduces_a_dataset_row(soft: CableData):
    batch = make_batch(soft, [(0, 200)], is_held_out=False, history_frames=C)
    _, vel, _ = soft.episodes[0]
    state = State(pos=batch.pos, vel_history=vel[200 - C + 1 : 201].flip(0).unsqueeze(0))

    rebuilt = features_from_state(state, batch.meta, C)

    assert torch.equal(rebuilt.node_feat, batch.node_feat)
    assert torch.equal(rebuilt.edge_feat, batch.edge_feat)
    assert torch.equal(rebuilt.edge_index, batch.edge_index)


@pytest.mark.claim(
    "the loader and the stub emit the same type field for field, which is what stops the "
    "two fixtures drifting apart"
)
def test_the_loader_and_the_stub_agree(soft: CableData):
    real = make_batch(soft, [(0, 10), (1, 5)], is_held_out=False, history_frames=C)
    stub, _ = make_fake_batch(b=2, seed=0)

    for name in ("pos", "node_feat", "edge_index", "edge_feat", "target"):
        assert getattr(real, name).shape == getattr(stub, name).shape, name
        assert getattr(real, name).dtype == getattr(stub, name).dtype, name
    for name in vars(stub.meta):
        assert getattr(real.meta, name).shape == getattr(stub.meta, name).shape, name
        assert getattr(real.meta, name).dtype == getattr(stub.meta, name).dtype, name


@pytest.mark.contract
def test_a_batch_of_nothing_is_refused(soft: CableData):
    with pytest.raises(ValueError, match="zero windows"):
        make_batch(soft, [], is_held_out=False, history_frames=C)


@pytest.mark.contract
def test_a_history_of_the_wrong_depth_is_refused(soft: CableData):
    batch = make_batch(soft, [(0, 10)], is_held_out=False, history_frames=C)
    with pytest.raises(ValueError, match=f"must carry {C} frames"):
        features_from_state(
            State(pos=batch.pos, vel_history=torch.zeros(1, 3, N, 3)), batch.meta, C
        )


@pytest.mark.claim(
    "the velocity-history depth C is a hyperparameter rather than a module constant: the "
    "layout functions take it, so a run can be trained at another depth "
    "(docs/method.md)"
)
@pytest.mark.parametrize("depth", [1, 2, 8])
def test_the_pipeline_windows_and_builds_at_any_depth(soft: CableData, depth: int):
    """``F``, the history slice and the window starts all move together with ``C``.

    Asserted against the recorded velocities rather than against a shape, because
    a depth that changed ``F`` while reading the wrong frames would satisfy every
    shape check and hand the model a history that is not the cable's.
    """
    batch = make_batch(soft, [(0, 20)], is_held_out=False, history_frames=depth)
    _, vel, _ = soft.episodes[0]

    assert batch.node_feat.shape == (1, N, feature_width(depth))
    history = batch.node_feat[0, :, : depth * 3].reshape(N, depth, 3)
    for age in range(depth):
        assert torch.equal(history[:, age], vel[20 - age]), f"index {age} is frame k-{age}"

    starts = window_starts(len(soft.episodes[0][0]), depth)
    assert starts.start == depth - 1
    assert len(starts) == len(soft.episodes[0][0]) - depth


@pytest.mark.claim(
    "a deeper history costs samples at the front of every episode, so the window count is a "
    "function of C rather than a fixture (docs/data.md)"
)
def test_the_window_count_shrinks_as_the_history_deepens(soft: CableData):
    n = len(soft.episodes[0][0])
    counts = [len(window_starts(n, depth)) for depth in (2, 5, 9)]
    assert counts == [n - 2, n - 5, n - 9]
    assert counts == sorted(counts, reverse=True), "a deeper history cannot yield more windows"


@pytest.mark.regression(
    "slicing the leading C*3 columns of a batch built at another depth reads part of one "
    "frame as another and returns a correctly-shaped lie"
)
def test_a_batch_of_another_depth_is_refused_rather_than_truncated(soft: CableData):
    """The guard that makes a checkpoint unevaluable at the wrong ``C``.

    Both directions fail: a narrower batch cannot fill the history, and a wider
    one would silently be read as the first ``C`` frames of a deeper one.
    """
    deep = make_batch(soft, [(0, 20)], is_held_out=False, history_frames=8)
    shallow = make_batch(soft, [(0, 20)], is_held_out=False, history_frames=3)

    for batch, asked in ((deep, 5), (shallow, 5)):
        with pytest.raises(ValueError, match="different velocity-history depth"):
            state_from_batch(batch, asked)

    # The matching depth is the one that round-trips.
    assert state_from_batch(deep, 8).vel_history.shape == (1, 8, N, 3)


@pytest.mark.contract
def test_a_history_of_no_frames_has_no_width():
    with pytest.raises(ValueError, match="at least 1"):
        feature_width(0)


def _write_cable(directory, jumps: list[bool], frames: int = 12):
    """An ``episodes.npz`` whose flagged episodes each carry one metre-scale step."""
    directory.mkdir(parents=True, exist_ok=True)
    pos = np.zeros((len(jumps), frames, N, 3), dtype=np.float64)
    for e, jump in enumerate(jumps):
        # A millimetre per frame is physical; a flagged episode teleports.
        pos[e] = np.arange(frames).reshape(-1, 1, 1) * 1e-3
        if jump:
            pos[e, frames // 2 :] += 3.0
    np.savez_compressed(
        directory / "episodes.npz",
        episode_lengths=np.full(len(jumps), frames),
        vertex_pos=pos,
        vertex_vel=np.zeros_like(pos),
        t=np.tile(np.arange(frames) * 2e-3, (len(jumps), 1)),
        length=1.0,
        diameter=0.003,
        bend_stiffness=1e7,
        joint_damping=1e-3,
        effective_linear_density=0.1,
        cable_id=7,
    )
    return directory


@pytest.mark.claim(
    "an episode whose vertices move metres between two frames recorded 2 ms apart is a solver "
    "divergence rather than a whip: three of 4,600 are, they are 0.13% of the frames, and they "
    "set step_std, which sets the units the loss is taken in (docs/data.md)"
)
def test_a_diverged_episode_is_dropped_and_recorded(tmp_path):
    """Kept episodes are physical, and what was removed is on the record.

    Silence is the failure being ruled out: a filtered dataset and an unfiltered
    one produce the same table of finite numbers, and only the recorded indices
    say which one was scored. The numbering also stops matching the npz as soon
    as anything is dropped, so it has to be reported rather than inferred.
    """
    directory = _write_cable(tmp_path / "cable_007", jumps=[False, True, False])

    filtered = load_cable(directory)
    unfiltered = load_cable(directory, max_step_m=None)

    assert len(unfiltered.episodes) == 3 and unfiltered.dropped_episodes == ()
    assert len(filtered.episodes) == 2
    assert filtered.dropped_episodes == (1,), "the file's own index, not the kept one"
    for pos, _, _ in filtered.episodes:
        assert float((pos[1:] - pos[:-1]).norm(dim=-1).max()) < MAX_STEP_M
