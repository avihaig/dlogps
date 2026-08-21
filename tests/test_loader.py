"""Splits and the two public iterators.

The split is the one piece here whose failure is silent and fatal: episodes of
one cable share material parameters, so a cable on both sides leaks the test set
into training and every number afterwards is wrong.
"""

from __future__ import annotations

import pytest
import torch

from dlogps.data.dataset import DEFAULT_HISTORY_FRAMES as C
from dlogps.data.dataset import CableData, feature_width
from dlogps.data.loader import (
    SplitCfg,
    eval_seen_cables,
    eval_unseen_cables,
    eval_windows,
    load_dataset,
    sample_index,
    stratified_held_out,
    training_batches,
)
from dlogps.data.types import E, N

ROOT = "assets/sample_v1"


@pytest.fixture(scope="module")
def cables() -> list[CableData]:
    return load_dataset(ROOT)


def synthetic(
    cable_id: int, stiffness: float, length: float, diameter: float = 0.0025
) -> CableData:
    """A cable with no frames: the split reads only params, never geometry."""
    return CableData(
        episodes=[],
        params={
            "length": length,
            "diameter": diameter,
            "bend_stiffness": stiffness,
            "joint_damping": 1e-4,
            "effective_linear_density": 0.005,
        },
        cable_id=cable_id,
    )


@pytest.mark.contract
def test_the_dataset_loads_every_cable_it_finds(cables: list[CableData]):
    assert [c.cable_id for c in cables] == [0, 45]
    assert sum(c.n_episodes for c in cables) == 4


@pytest.mark.contract
def test_an_empty_directory_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="no cable_"):
        load_dataset(tmp_path)


@pytest.mark.claim(
    "the split holds out whole cable parameterizations, never trajectories from a training "
    "cable: episodes of one cable share material parameters and would leak "
    ""
)
def test_a_cable_is_never_on_both_sides(cables: list[CableData]):
    # Half, not the default: 20% of a two-cable fixture rounds to zero held out,
    # which is right for the fixture and vacuous for this assertion.
    cfg = SplitCfg(held_out_fraction=0.5)
    held = stratified_held_out(cables, cfg)
    train = {c.cable_id for c in cables} - held
    assert held and train, "both sides non-empty"
    assert held & train == set()

    train_ids = {
        int(b.meta.cable_id[0])
        for b in training_batches(cables, batch_size=64, history_frames=C, cfg=cfg)
    }
    held_ids = {
        int(b.meta.cable_id[0])
        for b in training_batches(cables, batch_size=64, history_frames=C, cfg=cfg, held_out=True)
    }
    assert train_ids & held_ids == set()
    assert train_ids | held_ids == {0, 45}


@pytest.mark.regression(
    "rounding each stratum up gave 39% held out on the 46-row population and emptied six "
    "singleton strata from training; largest remainder hits the fraction exactly"
)
def test_the_fraction_is_exact_and_no_stratum_is_emptied():
    # 18 strata, six of them singletons: the shape that broke ceil-per-stratum.
    grid = [
        synthetic(i, stiffness=10.0 ** (6 + i % 5), length=0.8 + 0.2 * (i % 4)) for i in range(46)
    ]
    held = stratified_held_out(grid, SplitCfg())
    assert len(held) == round(0.2 * 46) == 9
    by_stratum: dict[tuple[float, float], list[int]] = {}
    for c in grid:
        by_stratum.setdefault((c.params["bend_stiffness"], c.params["length"]), []).append(
            c.cable_id
        )
    for ids in by_stratum.values():
        assert not set(ids) <= held, "no stratum may lose every cable to the held-out side"


@pytest.mark.claim(
    "the split rule is a fraction of whatever the population is, never a count: how many "
    "cables exist is an attribute of the CSV and of nothing else "
    ""
)
def test_the_fraction_holds_on_a_population_of_a_different_size():
    # Five materials x four lengths, 40 cables: nothing in the loader knows that.
    grid = [
        synthetic(i, stiffness=10.0**s, length=lgth)
        for i, (s, lgth) in enumerate(
            (s, lgth) for s in range(6, 11) for lgth in (0.8, 1.1, 1.4, 1.6) for _ in range(2)
        )
    ]
    held = stratified_held_out(grid, SplitCfg(held_out_fraction=0.5))
    assert len(held) == 20, "half of 40"

    # And on a different size, with no code change.
    small = grid[:8]
    assert len(stratified_held_out(small, SplitCfg(held_out_fraction=0.5))) == 4


@pytest.mark.claim(
    "the split is stratified so no material and no length is missing from either side "
)
def test_every_stratum_appears_on_both_sides():
    grid = [
        synthetic(i, stiffness=10.0**s, length=lgth)
        for i, (s, lgth) in enumerate(
            (s, lgth) for s in range(6, 11) for lgth in (0.8, 1.6) for _ in range(4)
        )
    ]
    held = stratified_held_out(grid, SplitCfg(held_out_fraction=0.25))
    by_id = {c.cable_id: c for c in grid}

    for side in (held, {c.cable_id for c in grid} - held):
        stiffnesses = {by_id[i].params["bend_stiffness"] for i in side}
        lengths = {by_id[i].params["length"] for i in side}
        assert len(stiffnesses) == 5, "every material present"
        assert len(lengths) == 2, "every length present"


@pytest.mark.contract
def test_the_split_is_reproducible_from_the_config_alone(cables: list[CableData]):
    assert stratified_held_out(cables, SplitCfg()) == stratified_held_out(cables, SplitCfg())


@pytest.mark.regression(
    "the per-stratum quota was taken as sorted(ids)[:quota], and the population enumerates "
    "diameter as the inner axis, so every held-out cable was the thinnest of its stratum: on the "
    "46-row population all nine were 2.5 mm against a training span of 1.5 mm to 10 mm, and "
    "stage 2 measured one corner rather than the population"
)
def test_the_held_out_side_is_not_one_corner_of_the_population():
    """The stage-2 claim is about the population, so the split cannot pick a facet.

    The grid mirrors the real one: diameter nested inside ``(stiffness, length)``
    with the id increasing through it, which is the shape that made taking the
    lowest id equivalent to taking the thinnest cable.
    """
    diameters = [0.0025, 0.005, 0.0075, 0.01]
    grid = [
        synthetic(i, stiffness=10.0**s, length=lgth, diameter=d)
        for i, (s, lgth, d) in enumerate(
            (s, lgth, d) for s in (6, 7, 8) for lgth in (0.8, 1.6) for d in diameters
        )
    ]

    held = stratified_held_out(grid, SplitCfg(held_out_fraction=0.25))
    by_id = {c.cable_id: c for c in grid}

    assert len(held) == 6, "one per stratum"
    assert len({by_id[i].params["diameter"] for i in held}) > 1, (
        "the held-out side spans more than one diameter"
    )
    # Sharper than the count: the defect took exactly the lowest id per stratum.
    thinnest = {min(range(i, i + 4)) for i in range(0, 24, 4)}
    assert held != thinnest


@pytest.mark.contract
def test_a_split_config_that_would_label_nothing_is_refused():
    """The thresholds live here, so the validation does too.

    A non-positive ``d_close_diameters`` labels every frame free flight and a
    fraction outside ``(0, 1)`` empties a side of the split. Both produce a full
    table of finite numbers, and neither is a result.
    """
    with pytest.raises(ValueError, match="d_close_diameters must be positive"):
        SplitCfg(d_close_diameters=0.0)
    with pytest.raises(ValueError, match="c_far_segments must be in"):
        SplitCfg(c_far_segments=0)
    with pytest.raises(ValueError, match="c_far_segments must be in"):
        SplitCfg(c_far_segments=E)
    assert SplitCfg(held_out_fraction=0.0).held_out_fraction == 0.0
    with pytest.raises(ValueError, match="held_out_fraction"):
        SplitCfg(held_out_fraction=1.0)
    with pytest.raises(ValueError, match="held_out_fraction"):
        SplitCfg(held_out_fraction=-0.1)


@pytest.mark.contract
def test_every_sample_in_the_index_is_a_valid_window(cables: list[CableData]):
    index = sample_index(cables, C)
    # 661 + 648 + 1331 + 1397 frames, each episode losing C frames to the window.
    assert len(index) == sum(n - C for n in (661, 648, 1331, 1397))
    for c, e, k in index:
        assert C - 1 <= k < len(cables[c].episodes[e][0]) - 1


@pytest.mark.contract
def test_training_batches_cover_the_index_exactly_once(cables: list[CableData]):
    batches = list(training_batches(cables, batch_size=500, history_frames=C, seed=1))
    total = sum(b.pos.shape[0] for b in batches)
    train = {c.cable_id for c in cables} - stratified_held_out(cables, SplitCfg())
    assert train, "the default 20% of two cables holds out none, so all are training"
    expected = sum(len(pos) - C for c in cables if c.cable_id in train for pos, _, _ in c.episodes)
    assert total == expected, "a short final batch is yielded, not dropped"
    assert all(b.node_feat.shape[1:] == (N, feature_width(C)) for b in batches)


@pytest.mark.contract
def test_shuffling_changes_the_order_and_not_the_content(cables: list[CableData]):
    def first_positions(seed: int) -> torch.Tensor:
        return next(iter(training_batches(cables, batch_size=8, history_frames=C, seed=seed))).pos

    assert not torch.equal(first_positions(0), first_positions(1))
    a = list(training_batches(cables, batch_size=500, history_frames=C, seed=0))
    b = list(training_batches(cables, batch_size=500, history_frames=C, seed=1))
    assert sum(x.pos.shape[0] for x in a) == sum(x.pos.shape[0] for x in b)


@pytest.mark.claim(
    "training yields Batch alone and evaluation yields (Batch, Labels): labels are computed "
    "on the ground-truth rollout window, never the single training frame "
    ""
)
def test_evaluation_carries_labels_over_the_rollout_window(cables: list[CableData]):
    cfg = SplitCfg(held_out_fraction=0.5)
    batch, labels = next(iter(eval_windows(cables, rollout=32, history_frames=C, cfg=cfg)))

    assert batch.pos.shape == (1, N, 3), "the batch is the rollout's first frame"
    assert labels.contact_frame.shape == (1, 32), "labels span the whole window"
    assert labels.contact_pair.shape == (1, 32, E, E)
    assert bool(batch.meta.is_held_out.all())


@pytest.mark.claim(
    "the public iterators window at the depth they are handed, so a run at another C needs "
    "no edit here (docs/method.md)"
)
@pytest.mark.parametrize("depth", [3, 8])
def test_both_iterators_window_at_the_configured_depth(cables: list[CableData], depth: int):
    """``F`` follows ``C`` down both paths, and the sample count follows with it.

    Both are asserted because they fail apart: a loader that threaded the depth
    into the features but not into the index would build correct batches from
    windows whose first frames have no history behind them.
    """
    cfg = SplitCfg(held_out_fraction=0.5)
    width = feature_width(depth)

    trained = list(training_batches(cables, batch_size=500, history_frames=depth, cfg=cfg))
    assert all(b.node_feat.shape[1:] == (N, width) for b in trained)

    train_ids = {c.cable_id for c in cables} - stratified_held_out(cables, cfg)
    expected = sum(
        len(pos) - depth for c in cables if c.cable_id in train_ids for pos, _, _ in c.episodes
    )
    assert sum(b.pos.shape[0] for b in trained) == expected

    batch, _ = next(iter(eval_windows(cables, rollout=32, history_frames=depth, cfg=cfg)))
    assert batch.node_feat.shape == (1, N, width)


@pytest.mark.contract
def test_an_episode_shorter_than_the_rollout_is_skipped(cables: list[CableData]):
    cfg = SplitCfg(held_out_fraction=0.5)
    assert list(eval_windows(cables, rollout=10_000, history_frames=C, cfg=cfg)) == []


TEST_ROOT = "assets/sample_test_v1"


@pytest.fixture(scope="module")
def test_cables() -> list[CableData]:
    return load_dataset(TEST_ROOT)


@pytest.mark.claim(
    "the test set is the same cables released from poses training never used: per-cable RNG "
    "is seeded [seed, cable_id], so one seed change redraws every pose while the cable is "
    "identical"
)
def test_the_test_set_is_the_same_cables_from_different_drops(
    cables: list[CableData], test_cables: list[CableData]
):
    assert [c.cable_id for c in test_cables] == [c.cable_id for c in cables]
    for train, test in zip(cables, test_cables, strict=True):
        assert train.params == test.params, "same cable: stiffness, diameter, length"
        # A different drop, not a different cable. Both cables start metres apart.
        first_train = train.episodes[0][0][0]
        first_test = test.episodes[0][0][0]
        assert (first_train - first_test).abs().max() > 0.1


@pytest.mark.claim(
    "stage 1 is seen cables and unseen drops, stage 2 is unseen cables and unseen drops; "
    "both are slices of one test set because the split is deterministic from cable id "
    ""
)
def test_the_two_stages_slice_the_test_set_and_do_not_overlap(test_cables: list[CableData]):
    cfg = SplitCfg(held_out_fraction=0.5)
    seen = {
        int(b.meta.cable_id[0])
        for b, _ in eval_seen_cables(test_cables, rollout=16, history_frames=C, cfg=cfg)
    }
    unseen = {
        int(b.meta.cable_id[0])
        for b, _ in eval_unseen_cables(test_cables, rollout=16, history_frames=C, cfg=cfg)
    }

    assert seen and unseen, "both stages must have cables at this fraction"
    assert seen & unseen == set(), "a cable is in exactly one stage"
    assert seen | unseen == {0, 45}
    # The split assigns a cable the same side in either dataset, which is what
    # makes stage 2 genuinely unseen rather than accidentally overlapping.
    assert unseen == stratified_held_out(test_cables, cfg)
