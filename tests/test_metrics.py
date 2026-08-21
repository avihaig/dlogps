"""The metric suite, on rollouts built to have a known answer.

A scorecard is only worth having if a wrong rollout scores worse than a right
one, so most of these construct a specific failure and assert the metric sees
it. Scoring a random tensor and checking the number is finite proves nothing.

The two that earn their keep:
:func:`test_self_intersection_catches_a_crossing_the_vertices_miss`, because
vertex distance is the obvious implementation and it is wrong; and
:func:`test_the_contact_threshold_is_per_cable`, because a threshold stored as
one absolute distance would make the contact split track cable thickness rather
than geometry.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from dlogps.data.geometry import chain_far_mask, segment_distance, segment_midpoints
from dlogps.data.labels import contact_labels
from dlogps.data.types import BatchMeta, E, Labels, N
from dlogps.harness import METRIC_BUILDERS, RegimeCfg, build_metrics, score
from tests.fake_data import make_fake_batch

CFG = RegimeCfg(horizons=(1, 2, 4))
"""Short horizons so a smoke-sized rollout still reports every one of them."""


def _rollout(b: int = 2, r: int = 4, seed: int = 0) -> tuple[torch.Tensor, BatchMeta, Labels]:
    """A ground-truth rollout: the stub cable, repeated ``r`` frames."""
    batch, labels = make_fake_batch(b=b, r=r, seed=seed)
    truth = batch.pos.unsqueeze(1).expand(b, r, N, 3).clone()
    return truth, batch.meta, labels


def _folded(gap: float, length: float = 1.0) -> torch.Tensor:
    """``[1, 1, N, 3]`` hairpin whose two arms run ``gap`` apart.

    The return arm mirrors the outbound one, so segment ``i`` and segment
    ``31 - i`` share a span at a separation of exactly ``gap``. Those pairs are
    chain-far, which makes the frame a contact frame precisely when ``gap`` falls
    under the threshold: the geometry is fixed and the threshold is the variable.
    """
    node = torch.arange(N, dtype=torch.float32)
    outbound = node <= 16.0
    pos = torch.zeros(1, 1, N, 3)
    pos[..., 0] = torch.where(outbound, node, 32.0 - node) * (length / E)
    pos[..., 1] = torch.where(outbound, 0.0, 1.0) * gap
    return pos


# --- the registry -----------------------------------------------------------


@pytest.mark.claim("nothing enters the registry that docs/metrics.md does not name")
def test_the_registry_holds_exactly_the_documented_metrics() -> None:
    """Nothing enters the registry that ``docs/metrics.md`` does not name;
    inventing one is a defect, not a contribution."""
    assert set(METRIC_BUILDERS) == {
        "rollout_rmse",
        "link_length",
        "self_intersection",
        "error_by_phase",
        "error_by_floor_phase",
        "error_by_pair_regime",
        "error_by_chain_distance",
    }


@pytest.mark.contract
def test_every_metric_returns_named_scalars() -> None:
    truth, meta, labels = _rollout()

    for name, fn in build_metrics(CFG).items():
        out = fn(truth, truth, meta, labels)
        assert out, f"{name} returned nothing"
        assert all(isinstance(v, float) for v in out.values()), name


@pytest.mark.contract
def test_score_runs_before_any_data_exists() -> None:
    """The stub batch carries a rollout axis for exactly this: the scorecard is
    exercised end to end with no dataset and no simulator."""
    truth, meta, labels = _rollout(b=3, r=4)

    scores = score(truth, truth, meta, labels, CFG)

    assert scores["rmse_full"] == pytest.approx(0.0, abs=1e-6)
    assert set(scores) == {
        k for fn in build_metrics(CFG).values() for k in fn(truth, truth, meta, labels)
    }


# --- thresholds are configuration, never constants --------------------------


@pytest.mark.claim(
    "d_close is 1.5x the cable diameter so it tracks the geometry rather than a round number; docs/metrics.md"
)
def test_the_contact_threshold_is_per_cable() -> None:
    """**The trap this test exists for.** ``d_close`` is 1.5x the diameter and the
    population runs 1.5 mm to 10 mm, so one absolute distance would be at once too
    strict for the thick cables and too loose for the thin ones, and the contact
    split would then track thickness rather than geometry.

    Asserted on the path that labels a window, since that is the only place the
    threshold is applied. Two cables at identical geometry and a 5 mm gap: the
    thick one is in contact at 1.5 x 5 mm, the thin one is not at 1.5 x 1.5 mm.
    """
    pos = _folded(gap=0.005).expand(2, 1, N, 3)

    labels = contact_labels(
        pos, torch.tensor([0.0015, 0.0050]), d_close_diameters=1.5, c_far_segments=4
    )

    assert not bool(labels.contact_frame[0]), "1.5 mm cable: a 5 mm gap is free flight"
    assert bool(labels.contact_frame[1]), "5 mm cable: the same 5 mm gap is contact"


@pytest.mark.regression(
    "RegimeCfg carried a second copy of d_close and c_far that reached no metric, so the "
    "evaluate CLI's two threshold flags were silent no-ops and the [FIX ON PROBE] gate could be "
    "broken with every number still finite"
)
def test_the_metric_config_holds_no_copy_of_the_contact_thresholds() -> None:
    """One owner for a threshold, or a flag that sets nothing looks like one that does.

    The thresholds are ``SplitCfg``'s, applied by the loader at label time. A
    field of either name back on :class:`RegimeCfg` is unreachable from every
    metric here, because each one reads ``Labels`` and derives nothing.
    """
    assert {f.name for f in dataclasses.fields(RegimeCfg)} == {"horizons"}
    assert not hasattr(RegimeCfg(), "d_close")

    # The live owner does move the decomposition, so the numbers are not merely
    # unreachable from one object but reachable from the other.
    pos = _folded(gap=0.005).expand(1, 1, N, 3)
    diameter = torch.tensor([0.0050])
    loose = contact_labels(pos, diameter, d_close_diameters=1.5, c_far_segments=4)
    strict = contact_labels(pos, diameter, d_close_diameters=0.5, c_far_segments=4)

    assert bool(loose.contact_frame[0]) and not bool(strict.contact_frame[0])


@pytest.mark.contract
def test_the_horizons_are_rejected_when_nonsense() -> None:
    with pytest.raises(ValueError, match="horizons must be positive"):
        RegimeCfg(horizons=(0,))
    with pytest.raises(ValueError, match="horizons must be positive"):
        RegimeCfg(horizons=())


@pytest.mark.contract
def test_horizons_beyond_the_rollout_are_dropped_not_faked() -> None:
    """A 4-frame smoke rollout must not report a 100-frame horizon."""
    truth, meta, labels = _rollout(r=4)

    scores = build_metrics(RegimeCfg(horizons=(1, 100)))["rollout_rmse"](truth, truth, meta, labels)

    assert "rmse_h1" in scores
    assert "rmse_h100" not in scores


@pytest.mark.claim(
    "a reported horizon means h PREDICTED frames: frame 0 is the input state, so averaging "
    "0..h-1 made rmse_h1 identically zero and biased every other horizon low "
    ""
)
def test_a_horizon_averages_predicted_frames_only() -> None:
    # Frame 0 exact, every predicted frame off by a known amount. Under the old
    # convention rmse_h3 averaged in the free zero and read 2/3 of the truth.
    truth, meta, labels = _rollout(r=4)
    pred = truth.clone()
    pred[:, 1:] += 0.01

    scores = build_metrics(RegimeCfg(horizons=(1, 2, 3)))["rollout_rmse"](pred, truth, meta, labels)

    error = 0.01 * math.sqrt(3)
    for h in (1, 2, 3):
        assert scores[f"rmse_h{h}"] == pytest.approx(error, rel=1e-4), (
            f"rmse_h{h} is diluted, so frame 0 is still being averaged in"
        )
    assert scores["rmse_full"] == pytest.approx(error, rel=1e-4)


@pytest.mark.contract
def test_a_rollout_with_no_prediction_scores_nan_rather_than_zero() -> None:
    """Horizon 1 runs no forward pass, so there is nothing to score. Zero would
    read as a perfect model; NaN is what the aggregate treats as undefined."""
    truth, meta, labels = _rollout(r=1)

    scores = build_metrics(RegimeCfg(horizons=(1,)))["rollout_rmse"](truth, truth, meta, labels)

    assert math.isnan(scores["rmse_full"])
    assert "rmse_h1" not in scores


# --- primary error ----------------------------------------------------------


@pytest.mark.contract
def test_rmse_is_zero_on_a_perfect_rollout_and_grows_with_error() -> None:
    truth, meta, labels = _rollout()
    metric = build_metrics(CFG)["rollout_rmse"]

    perfect = metric(truth, truth, meta, labels)["rmse_full"]
    drifted = metric(truth + 0.01, truth, meta, labels)["rmse_full"]

    assert perfect == pytest.approx(0.0, abs=1e-6)
    assert drifted == pytest.approx(0.01 * math.sqrt(3), rel=1e-4)


@pytest.mark.contract
def test_a_later_horizon_sees_divergence_an_early_one_does_not() -> None:
    """Reporting one horizon would hide the difference between a model that is
    fine for ten frames and one that is fine throughout."""
    # Five frames, so h4 has the four predicted frames it now means; frame 0 is
    # the input state and is scored by nothing.
    truth, meta, labels = _rollout(r=5)
    pred = truth.clone()
    pred[:, 4:] += 1.0  # diverges only at the last frame

    scores = build_metrics(CFG)["rollout_rmse"](pred, truth, meta, labels)

    assert scores["rmse_h1"] == pytest.approx(0.0, abs=1e-6)
    assert scores["rmse_h4"] > 0.4


# --- physics violation, no ground truth needed ------------------------------


@pytest.mark.claim(
    "link-length constancy and self-intersection score the rollout with no ground truth, so the recorded data is the natural control; docs/metrics.md"
)
def test_link_length_reads_zero_on_a_cable_at_rest_length() -> None:
    """The stub cable's segments sit exactly at rest length, so a non-zero
    reading here would be the metric measuring itself."""
    truth, meta, labels = _rollout()

    scores = build_metrics(CFG)["link_length"](truth, truth, meta, labels)

    assert scores["link_length_mean_drift"] == pytest.approx(0.0, abs=1e-5)


@pytest.mark.claim(
    "link-length constancy and self-intersection score the rollout with no ground truth, so the recorded data is the natural control; docs/metrics.md"
)
def test_link_length_catches_a_stretched_cable() -> None:
    """Scale the cable up 10%: every link is 10% long, and it needs no reference
    trajectory to say so."""
    truth, meta, labels = _rollout()

    scores = build_metrics(CFG)["link_length"](truth * 1.1, truth, meta, labels)

    assert scores["link_length_mean_drift"] == pytest.approx(0.1, rel=1e-3)
    assert scores["link_length_max_drift"] == pytest.approx(0.1, rel=1e-3)


@pytest.mark.claim(
    "link-length constancy and self-intersection score the rollout with no ground truth, so the recorded data is the natural control; docs/metrics.md"
)
def test_self_intersection_is_clean_on_the_ground_truth() -> None:
    """The solver forbids interpenetration, so the truth is the natural control
    and any violation in a rollout is the model inventing physics."""
    truth, meta, labels = _rollout()

    scores = build_metrics(CFG)["self_intersection"](truth, truth, meta, labels)

    assert scores["selfint_frame_fraction"] == 0.0
    assert scores["selfint_worst_penetration"] == 0.0
    assert scores["selfint_persists"] == 0.0


@pytest.mark.claim(
    "self-intersection is segment-to-segment: for |i-j| > 2 the centreline distance must stay above the diameter D; docs/metrics.md Eq. 3"
)
def test_self_intersection_catches_a_crossing_the_vertices_miss() -> None:
    """**Segment-to-segment, not vertex-to-vertex.**

    Two straight strands crossing in an X, offset by a third of a diameter. Every
    one of the four endpoints is far away; the strands touch only at their
    middles. A vertex-distance implementation scores this cable as clean while it
    has passed straight through itself, which is the exact failure the metric
    exists to catch.
    """
    diameter = 0.01
    half = N // 2
    span = torch.linspace(-0.5, 0.5, half)

    cable = torch.zeros(1, 1, N, 3)
    cable[0, 0, :half, 0] = span  # one strand along x
    cable[0, 0, half:, 1] = torch.linspace(-0.5, 0.5, N - half)  # the other along y
    cable[0, 0, half:, 2] = diameter / 3  # passing just under it

    meta = dataclasses.replace(_rollout()[1], diameter=torch.tensor([diameter]))
    meta = dataclasses.replace(meta, length=torch.tensor([1.0]))
    labels = Labels(
        contact_frame=torch.zeros(1, 1, dtype=torch.bool),
        contact_pair=torch.zeros(1, 1, E, E, dtype=torch.bool),
    )

    vertex_gap = torch.cdist(cable[0, 0], cable[0, 0])
    far = chain_far_mask(2)
    node_far = torch.zeros(N, N, dtype=torch.bool)
    node_far[:-1, :-1] = far
    assert float(vertex_gap[node_far].min()) > diameter, "the vertices are not the crossing"

    scores = build_metrics(CFG)["self_intersection"](cable, cable, meta, labels)
    assert scores["selfint_frame_fraction"] == 1.0
    assert scores["selfint_worst_penetration"] > 0.0


@pytest.mark.claim(
    "self-intersection is segment-to-segment: for |i-j| > 2 the centreline distance must stay above the diameter D; docs/metrics.md Eq. 3"
)
def test_persistence_separates_an_excursion_from_a_crossed_cable() -> None:
    """One bad frame that recovers is a numerical excursion; two in a row is the
    cable genuinely on the wrong side of itself."""
    truth, meta, labels = _rollout(b=1, r=4)
    metric = build_metrics(CFG)["self_intersection"]

    collapsed = truth.clone()
    collapsed[:, 1] = 0.0  # one frame with every segment on top of every other

    both = truth.clone()
    both[:, 1:3] = 0.0

    assert metric(collapsed, truth, meta, labels)["selfint_persists"] == 0.0
    assert metric(both, truth, meta, labels)["selfint_persists"] == 1.0


# --- the decompositions -----------------------------------------------------


@pytest.mark.claim(
    "the two priors serve opposite regimes, so the phase decomposition is required and an aggregate averages them to nothing; docs/metrics.md"
)
def test_phase_split_reads_the_labels_and_never_recomputes_them() -> None:
    """Recomputing the phase label here would fork its definition, so the split
    must move when the labels move and the rollout does not."""
    truth, meta, _ = _rollout(b=1, r=4)
    pred = truth.clone()
    pred[:, :2] += 1.0  # error in the first two frames only

    def _labelled(flags: list[bool]) -> Labels:
        return Labels(
            contact_frame=torch.tensor([flags]),
            contact_pair=torch.zeros(1, 4, E, E, dtype=torch.bool),
        )

    metric = build_metrics(CFG)["error_by_phase"]
    first = metric(pred, truth, meta, _labelled([True, True, False, False]))
    second = metric(pred, truth, meta, _labelled([False, False, True, True]))

    assert first["rmse_contact"] > first["rmse_free_flight"]
    assert second["rmse_contact"] < second["rmse_free_flight"]


@pytest.mark.claim(
    "the paper's phase split labels a predicted frame floor contact when the recorded cable has min_i z_i <= D after the input frame; docs/metrics.md"
)
def test_floor_phase_split_reads_the_recorded_height_after_the_input_frame() -> None:
    """Floor contact comes off the ground truth, never the prediction, and the
    input frame (frame 0) takes no part in either class."""
    truth, meta, labels = _rollout(b=1, r=5)
    truth = truth.clone()
    truth[..., 2] = 1.0  # airborne everywhere ...
    truth[0, 3, 5, 2] = 0.5 * float(meta.diameter[0])  # ... except one vertex at frame 3
    truth[0, 0, :, 2] = 0.0  # frame 0 is the input state and must not count
    pred = truth.clone()
    pred[0, 3] += 0.02  # error only on the floor-contact frame
    pred[0, 1] += 0.01  # and a smaller one on a free-flight frame

    scores = build_metrics(CFG)["error_by_floor_phase"](pred, truth, meta, labels)

    length = float(meta.length[0])
    assert scores["rel_l2_floor_contact"] == pytest.approx(0.02 * math.sqrt(3) / length, rel=1e-5)
    assert scores["rel_l2_free_flight"] == pytest.approx(0.01 * math.sqrt(3) / 3 / length, rel=1e-5)
    assert scores["floor_contact_frame_fraction"] == pytest.approx(1 / 4)

    # Moving the prediction onto the floor changes nothing about the label.
    grounded = pred.clone()
    grounded[..., 2] = 0.0
    relabelled = build_metrics(CFG)["error_by_floor_phase"](grounded, truth, meta, labels)
    assert relabelled["floor_contact_frame_fraction"] == pytest.approx(1 / 4)


@pytest.mark.contract
def test_floor_phase_split_reports_nan_for_a_rollout_without_the_phase() -> None:
    """A rollout that never leaves the floor defines no free-flight value, and
    NaN is what lets the aggregate count eligible rollouts rather than zeros."""
    truth, meta, labels = _rollout(b=1, r=4)
    truth = truth.clone()
    truth[..., 2] = 0.0
    scores = build_metrics(CFG)["error_by_floor_phase"](truth + 0.1, truth, meta, labels)

    assert math.isnan(scores["rel_l2_free_flight"])
    assert not math.isnan(scores["rel_l2_floor_contact"])
    assert scores["floor_contact_frame_fraction"] == 1.0


@pytest.mark.contract
def test_a_decomposition_with_an_empty_class_reports_nan_not_zero() -> None:
    """ "No contact frames in this rollout" and "no error in the contact frames"
    are opposite findings, and zero would report the first as the second."""
    truth, meta, _ = _rollout(b=1, r=4)
    labels = Labels(
        contact_frame=torch.zeros(1, 4, dtype=torch.bool),
        contact_pair=torch.zeros(1, 4, E, E, dtype=torch.bool),
    )

    scores = build_metrics(CFG)["error_by_phase"](truth + 0.1, truth, meta, labels)

    assert math.isnan(scores["rmse_contact"])
    assert not math.isnan(scores["rmse_free_flight"])


@pytest.mark.claim(
    "the two priors serve opposite regimes, so the phase decomposition is required and an aggregate averages them to nothing; docs/metrics.md"
)
def test_pair_regime_attributes_a_segment_pair_to_its_two_nodes() -> None:
    """The labels are over segments and the error is per node, so a node counts
    as involved when either of its segments is in a contact pair."""
    truth, meta, _ = _rollout(b=1, r=1)
    pred = truth.clone()
    pred[:, :, 5] += 1.0  # error at vertex 5 only

    pair = torch.zeros(1, 1, E, E, dtype=torch.bool)
    pair[0, 0, 4, 20] = True  # segment 4 spans vertices 4 and 5
    labels = Labels(contact_frame=torch.zeros(1, 1, dtype=torch.bool), contact_pair=pair)

    scores = build_metrics(CFG)["error_by_pair_regime"](pred, truth, meta, labels)

    assert scores["rmse_contact_pair_nodes"] > scores["rmse_other_nodes"]


@pytest.mark.claim(
    "the two priors serve opposite regimes, so the phase decomposition is required and an aggregate averages them to nothing; docs/metrics.md"
)
def test_chain_distance_bands_localize_where_the_error_is() -> None:
    """The claim is a trend along the cable, so the bands must move with it."""
    truth, meta, labels = _rollout(b=1, r=2)
    pred = truth.clone()
    pred[:, :, -4:] += 1.0  # error at the far end only

    scores = build_metrics(CFG)["error_by_chain_distance"](pred, truth, meta, labels)

    assert scores["rmse_band3"] > scores["rmse_band0"]
    assert scores["rmse_band0"] == pytest.approx(0.0, abs=1e-6)


# --- geometry helpers -------------------------------------------------------


@pytest.mark.contract
def test_segment_distance_is_symmetric_and_zero_on_the_diagonal() -> None:
    truth, _, _ = _rollout(b=1, r=1)
    d = segment_distance(truth[:, 0])

    assert d.shape == (1, E, E)
    assert torch.allclose(d, d.transpose(-1, -2), atol=1e-6)
    assert torch.allclose(torch.diagonal(d, dim1=-2, dim2=-1), torch.zeros(1, E), atol=1e-6)


@pytest.mark.claim(
    "self-intersection is segment-to-segment: for |i-j| > 2 the centreline distance must stay above the diameter D; docs/metrics.md Eq. 3"
)
def test_segment_distance_never_exceeds_the_midpoint_distance() -> None:
    """A sanity bound with a closed form: the closest points on two segments are
    at least as close as their midpoints, which are one candidate pair."""
    truth, _, _ = _rollout(b=2, r=1)
    frame = truth[:, 0]

    exact = segment_distance(frame)
    mid = torch.cdist(segment_midpoints(frame), segment_midpoints(frame))

    assert torch.all(exact <= mid + 1e-6)


@pytest.mark.contract
def test_chain_far_mask_excludes_the_bending_stencil() -> None:
    """`|i-j| > 2` for the physics check: adjacent segments touch because the
    cable is a cable."""
    mask = chain_far_mask(2)

    assert not bool(mask.diagonal().any())
    assert not bool(mask[0, 1] or mask[0, 2])
    assert bool(mask[0, 3])


# --- the three reductions, cumulative and terminal ---------------------------


@pytest.mark.claim(
    "rel_l2 divides by each sample's OWN length before any average over the batch; dividing a "
    "pooled error by a pooled length would report the population's 2x size range as if it were "
    "model behaviour (docs/metrics.md)"
)
def test_rel_l2_is_per_sample() -> None:
    truth, meta, labels = _rollout(b=2, r=3)
    # Two cables of different length, identical absolute error.
    meta = dataclasses.replace(meta, length=torch.tensor([0.8, 1.6]))
    pred = truth + 0.01

    fn = build_metrics(RegimeCfg(horizons=(1,)))["rollout_rmse"]
    both = fn(pred, truth, meta, labels)["rel_l2_at_h1"]
    short = fn(pred[:1], truth[:1], dataclasses.replace(meta, length=meta.length[:1]), labels)
    long_ = fn(pred[1:], truth[1:], dataclasses.replace(meta, length=meta.length[1:]), labels)

    error = 0.01 * math.sqrt(3)
    assert short["rel_l2_at_h1"] == pytest.approx(error / 0.8, rel=1e-4)
    assert long_["rel_l2_at_h1"] == pytest.approx(error / 1.6, rel=1e-4)
    assert short["rel_l2_at_h1"] == pytest.approx(2 * long_["rel_l2_at_h1"], rel=1e-4)
    # The batch figure is the mean of the two per-sample ratios, not the pooled
    # error over the pooled length, which would be error / 1.2 here.
    assert both == pytest.approx(0.5 * (error / 0.8 + error / 1.6), rel=1e-4)
    assert both != pytest.approx(error / 1.2, rel=1e-3)


@pytest.mark.claim(
    "*_at_h{h} is the error AT frame h and *_h{h} the mean over frames 1..h; they differ by 3.8x "
    "at h100 on a real rollout, so a table sharing their prefix must not share their meaning"
)
def test_terminal_differs_from_cumulative_on_a_growing_error() -> None:
    truth, meta, labels = _rollout(b=1, r=5)
    pred = truth.clone()
    # Error grows linearly with frame: 0.01, 0.02, 0.03, 0.04 over frames 1..4.
    for frame in range(1, 5):
        pred[:, frame] += 0.01 * frame

    scores = build_metrics(RegimeCfg(horizons=(4,)))["rollout_rmse"](pred, truth, meta, labels)

    unit = 0.01 * math.sqrt(3)
    assert scores["mean_l2_at_h4"] == pytest.approx(4 * unit, rel=1e-4)
    assert scores["mean_l2_h4"] == pytest.approx(2.5 * unit, rel=1e-4)
    assert scores["mean_l2_at_h4"] > scores["mean_l2_h4"]


@pytest.mark.regression(
    "an unknown metric name returned '' and read as dimensionless, not as a typo"
)
def test_unit_of_raises_on_a_name_it_does_not_know() -> None:
    from dlogps.harness.metrics import unit_of

    assert unit_of("rel_l2_at_h200") == "fraction of cable length"
    assert unit_of("mean_l2_h10") == "m"
    with pytest.raises(KeyError, match="no unit registered"):
        unit_of("rmsee_h100")


@pytest.mark.contract
def test_every_metric_the_registry_emits_has_a_unit() -> None:
    """The guard that keeps :func:`unit_of` raising from breaking a report."""
    from dlogps.harness.metrics import unit_of

    truth, meta, labels = _rollout(b=2, r=6)
    for name in score(truth + 0.01, truth, meta, labels, RegimeCfg(horizons=(1, 5))):
        assert unit_of(name), name
