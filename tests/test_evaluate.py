"""The evaluation driver: the four ways a scorecard is wrong rather than red.

Every failure this file defends against produces a full table of finite numbers.
A stage that draws from the wrong side of the split scores held-out cables as
seen; a ground-truth window paired onto the wrong rollout shifts every frame; an
aggregate taken per cable lets a short episode outvote a long one; and a
decomposition whose class is empty reads as zero error rather than as no
evidence. One test each:
:func:`test_the_two_stages_cover_disjoint_cable_sets`,
:func:`test_a_misaligned_ground_truth_window_is_refused`,
:func:`test_windows_are_weighted_by_count_not_by_cable` and
:func:`test_a_metric_no_window_defines_is_nan_with_a_zero_count`.

The cables here are built in memory: a hairpin whose two arms breathe in and out
past the contact threshold, so both phases and both pair regimes occur inside
every six-frame window and every decomposition has a non-empty class. That is
what lets the registry-coverage test assert a *finite* value rather than merely
a present key. ``assets/sample_test_v1/`` carries the one test that runs the real
loader against the real block.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
import torch
from torch import Tensor, nn

import dlogps.harness.evaluate as evaluate_module
from dlogps.data.dataset import CableData, feature_width, make_batch
from dlogps.data.labels import contact_labels
from dlogps.data.loader import (
    SplitCfg,
    eval_seen_cables,
    eval_unseen_cables,
    load_dataset,
    rollout_truth,
    stratified_held_out,
    training_batches,
)
from dlogps.data.types import Batch, E, N
from dlogps.harness.evaluate import (
    STAGES,
    EvalResult,
    evaluate,
    evaluate_checkpoint,
    stage_cables,
)
from dlogps.harness.metrics import RegimeCfg, score
from dlogps.harness.normalize import Stats, fit_stats
from dlogps.harness.rollout import rollout
from dlogps.harness.train import Checkpoint, TrainCfg, build_model
from dlogps.model.gps import GPSCfg
from tests.fake_data import HISTORY_FRAMES as C
from tests.fake_data import make_fake_batch

# One thread. The block is a few thousand elements per op over 33 nodes, so
# torch's intra-op pool spends far more on dispatch than on arithmetic, and this
# file has to stay inside the fast suite's five-second budget.
torch.set_num_threads(1)

SPLIT = SplitCfg(held_out_fraction=0.5)
"""Half held out, the fraction the two-cable and four-cable fixtures need to put
a cable on each side of the split at all."""

REGIME = RegimeCfg(horizons=(1, 5))
"""Horizons inside the fixtures' six-frame windows, so both are reported."""

HORIZON = 6
"""Frames per window, the input frame included."""

DT = 2e-3
"""Frame interval [s] at the recorded 500 Hz."""

SAMPLE_TEST = "assets/sample_test_v1"
"""The **test**-side fixture. Stage 1 and stage 2 are both slices of this."""


class _Drift(nn.Module):
    """Adds a fixed displacement to whatever position it is shown.

    Trivial on purpose: the driver's job is bookkeeping, and a model with real
    error would make every expected number a function of training rather than of
    the aggregation under test. The drift is large enough that the rollout leaves
    the ground truth immediately, so no metric reads a degenerate zero.
    """

    def __init__(self, step: float = 2e-3) -> None:
        super().__init__()
        self.delta = torch.full((3,), step)

    def forward(self, batch: Batch) -> Tensor:
        return batch.pos + self.delta


def _hairpin(frames: int, *, length: float, phase: float, swing: bool) -> Tensor:
    """``[frames, N, 3]`` folded cable whose two arms breathe past contact.

    The return arm mirrors the outbound one, so segment ``i`` and segment
    ``31 - i`` occupy the same span at a separation of ``gap``. Those pairs are
    chain-far for ``i <= 13``, which makes the frame a contact frame exactly when
    ``gap`` drops under ``1.5 x diameter``. With ``swing`` the gap oscillates on a
    five-frame period, so both phases occur inside any six-frame window; without
    it the arms stay 5 cm apart and no frame is ever in contact.
    """
    segment = length / E
    node = torch.arange(N, dtype=torch.float32)
    outbound = node <= 16.0
    x = torch.where(outbound, node, 32.0 - node) * segment

    step = torch.arange(frames, dtype=torch.float32)
    swept = 0.0025 + 0.02 * (0.5 + 0.5 * torch.cos(2 * math.pi * (step + phase) / 5.0))
    gap = swept if swing else torch.full((frames,), 0.05)

    pos = torch.zeros(frames, N, 3)
    pos[..., 0] = x
    pos[..., 1] = torch.where(outbound, 0.0, 1.0) * gap.view(-1, 1)
    pos[..., 2] = 0.5
    return pos


def _cable(
    cable_id: int, *, stiffness: float, frames: int, swing: bool = True, length: float = 1.0
) -> CableData:
    """One in-memory cable, one episode, so no test here opens a file.

    ``bend_stiffness`` is the stratum key the split sorts on, so a distinct value
    per cable makes the held-out set predictable from the ids alone.
    """
    pos = _hairpin(frames, length=length, phase=float(cable_id), swing=swing)
    vel = torch.zeros_like(pos)
    vel[:-1] = (pos[1:] - pos[:-1]) / DT
    vel[-1] = vel[-2]
    return CableData(
        episodes=[(pos, vel, torch.arange(frames, dtype=torch.float32) * DT)],
        params={
            "length": length,
            "diameter": 0.003,
            "bend_stiffness": stiffness,
            "joint_damping": 1e-3,
            "effective_linear_density": 0.1,
        },
        cable_id=cable_id,
    )


def _cables(swing: bool = True) -> list[CableData]:
    """Four cables in four strata; at half held out, ids 0 and 1 are unseen.

    The two seen cables carry different episode lengths, and therefore different
    window counts, which is what makes a per-cable mean distinguishable from a
    per-window one.
    """
    return [
        _cable(0, stiffness=1.0, frames=20, swing=swing),
        _cable(1, stiffness=2.0, frames=32, swing=swing),
        _cable(2, stiffness=3.0, frames=20, swing=swing),
        _cable(3, stiffness=4.0, frames=32, swing=swing),
    ]


def _stats(cables: list[CableData]) -> Stats:
    """Statistics fitted on the training side, the way a run fits them."""
    return fit_stats(
        training_batches(cables, batch_size=4, history_frames=C, cfg=SPLIT, held_out=False)
    )


def _windows_of(cable: CableData, horizon: int = HORIZON) -> list[int]:
    """Window starts of a cable's one episode, enumerated for the test's own use."""
    return list(range(C - 1, len(cable.episodes[0][0]) - horizon, horizon))


def _registry_names(horizon: int = HORIZON) -> set[str]:
    """Every name the metric suite emits at this horizon, read off the suite."""
    batch, labels = make_fake_batch(b=1, r=horizon)
    truth = batch.pos.unsqueeze(1).expand(1, horizon, N, 3)
    return set(score(truth, truth, batch.meta, labels, REGIME))


def _scored(model: nn.Module, cables: list[CableData], stats: Stats) -> list[tuple[int, float]]:
    """``(cable_id, rmse_full)`` per window of the seen stage, computed here.

    An independent recomputation, deliberately not sharing a line with the
    driver: it enumerates the windows itself, labels the ground truth itself and
    calls the suite once per window.
    """
    seen = stage_cables(cables, "seen_cables", SPLIT)
    per_window: list[tuple[int, float]] = []
    for cable in cables:
        if cable.cable_id not in seen:
            continue
        for start in _windows_of(cable):
            batch = make_batch(cable, [(0, start)], is_held_out=False, history_frames=C)
            truth = rollout_truth(cable, 0, start, HORIZON)
            labels = contact_labels(
                truth,
                batch.meta.diameter,
                d_close_diameters=SPLIT.d_close_diameters,
                c_far_segments=SPLIT.c_far_segments,
            )
            predicted = rollout(model, batch, horizon=HORIZON, stats=stats, history_frames=C)
            scores = score(predicted, truth, batch.meta, labels, REGIME)
            per_window.append((cable.cable_id, scores["rmse_full"]))
    return per_window


def _run(
    stage: str,
    cables: list[CableData] | None = None,
    *,
    horizon: int = HORIZON,
    limit: int | None = None,
) -> EvalResult:
    """One stage of the standard fixture, under the drift model."""
    cables = cables if cables is not None else _cables()
    return evaluate(
        _Drift(),
        cables,
        stage=stage,
        horizon=horizon,
        stats=_stats(cables),
        regime=REGIME,
        split=SPLIT,
        history_frames=C,
        limit=limit,
    )


@pytest.mark.contract
def test_a_stage_reports_every_registry_name_with_a_finite_value() -> None:
    """The driver forwards the suite whole: every name, every one a real number.

    A metric missing from the result is a metric nobody notices is missing, and a
    NaN in a table is read as a formatting problem rather than as an empty class.
    """
    result = _run("seen_cables")

    assert set(result.scores) == _registry_names()
    # The floor-phase split is NaN by contract when a rollout has no frame of
    # one class, and this fixture's cables never touch the floor.
    phase_conditional = {"rel_l2_floor_contact", "rel_l2_free_flight"}
    assert all(
        math.isfinite(value)
        for name, value in result.scores.items()
        if name not in phase_conditional
    ), result.scores
    assert math.isfinite(result.scores["rel_l2_free_flight"])
    assert result.n_windows == 6, "two windows from the 20-frame cable, four from the 32-frame one"
    assert set(result.counts) == set(result.scores)
    assert all(
        0 < count <= result.n_windows
        for name, count in result.counts.items()
        if name not in phase_conditional
    )
    assert result.counts["rel_l2_floor_contact"] == 0


@pytest.mark.claim(
    "stage 1 is seen cables and stage 2 is unseen cables, and the two never share a cable: "
)
def test_the_two_stages_cover_disjoint_cable_sets() -> None:
    """Both stages score, and no cable is scored by both.

    A cable on both sides is the failure the whole split exists to prevent: the
    out-of-distribution column would then be reporting cables the model trained
    on, and every number in it would still be finite.
    """
    cables = _cables()
    seen = stage_cables(cables, "seen_cables", SPLIT)
    unseen = stage_cables(cables, "unseen_cables", SPLIT)

    assert seen and unseen
    assert seen & unseen == set()
    assert seen | unseen == {cable.cable_id for cable in cables}
    assert unseen == stratified_held_out(cables, SPLIT)

    # What the loader actually yields, rather than what the split says it should.
    streamed = {
        "seen_cables": eval_seen_cables(cables, rollout=HORIZON, history_frames=C, cfg=SPLIT),
        "unseen_cables": eval_unseen_cables(cables, rollout=HORIZON, history_frames=C, cfg=SPLIT),
    }
    for stage, stream in streamed.items():
        assert {int(batch.meta.cable_id[0]) for batch, _ in stream} == stage_cables(
            cables, stage, SPLIT
        )
        assert _run(stage, cables).n_windows > 0


@pytest.mark.claim(
    "windows are aggregated by count so a long episode does not outvote a short one: "
)
def test_windows_are_weighted_by_count_not_by_cable() -> None:
    """The aggregate is the mean over windows, and the two schemes disagree here."""
    cables = _cables()
    model = _Drift()
    per_window = _scored(model, cables, _stats(cables))

    result = _run("seen_cables", cables)

    by_cable: dict[int, list[float]] = {}
    for cable_id, value in per_window:
        by_cable.setdefault(cable_id, []).append(value)
    assert {len(values) for values in by_cable.values()} == {2, 4}, "unequal window counts"

    window_mean = sum(value for _, value in per_window) / len(per_window)
    cable_mean = sum(sum(v) / len(v) for v in by_cable.values()) / len(by_cable)

    assert result.n_windows == len(per_window)
    assert result.scores["rmse_full"] == pytest.approx(window_mean, rel=1e-9)
    assert result.scores["rmse_full"] != pytest.approx(cable_mean, rel=1e-6)


@pytest.mark.claim("scoring goes through metrics.score only")
def test_score_is_called_once_per_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """One call per window, and no metric computed inside the driver."""
    calls: list[tuple[int, ...]] = []

    def spy(pred: Tensor, target: Tensor, meta: object, labels: object, cfg: object) -> object:
        calls.append(tuple(pred.shape))
        return score(pred, target, meta, labels, cfg)  # type: ignore[arg-type]

    monkeypatch.setattr(evaluate_module, "score", spy)
    result = _run("seen_cables")

    assert len(calls) == result.n_windows
    assert calls == [(1, HORIZON, N, 3)] * result.n_windows


@pytest.mark.contract
def test_the_limit_caps_the_windows_and_is_recorded() -> None:
    """A smoke run is scored, and says on its face that it was capped."""
    full = _run("seen_cables")
    capped = _run("seen_cables", limit=2)

    assert full.limit is None
    assert capped.limit == 2
    assert capped.n_windows == 2 < full.n_windows
    assert set(capped.scores) == set(full.scores)


@pytest.mark.contract
def test_a_metric_no_window_defines_is_nan_with_a_zero_count() -> None:
    """An empty class is NaN over zero windows, never zero over all of them.

    Cables that never fold have no contact frame anywhere, so a contact RMSE has
    nothing to average. Reporting 0.0 would say the model is perfect in contact.
    """
    result = _run("seen_cables", _cables(swing=False))

    assert math.isnan(result.scores["rmse_contact"])
    assert result.counts["rmse_contact"] == 0
    assert math.isfinite(result.scores["rmse_free_flight"])
    assert result.counts["rmse_free_flight"] == result.n_windows
    assert result.scores["contact_frame_fraction"] == 0.0


@pytest.mark.claim(
    "the split used at evaluation is the one training used, read from the checkpoint: "
)
def test_the_split_comes_from_the_checkpoint_not_from_a_default() -> None:
    """A checkpoint's split decides the stage, and it is not the loader default.

    At the default fraction this population holds out one cable; at the
    checkpoint's it holds out two. Evaluating under the wrong one silently moves
    a cable from stage 2 into stage 1.
    """
    cables = _cables()
    stats = _stats(cables)
    model_cfg = GPSCfg(d_model=12, n_heads=2, n_layers=1)
    checkpoint = Checkpoint(
        state_dict=build_model(int(stats.node_mean.shape[0]), "A", model_cfg).state_dict(),
        stats=stats,
        cfg=TrainCfg(steps=1, batch_size=1, sigma=0.0),
        model_cfg=model_cfg,
        variant="A",
        feature_width=int(stats.node_mean.shape[0]),
        step=1,
        split=SPLIT,
    )

    result = evaluate_checkpoint(
        checkpoint, cables, stage="unseen_cables", horizon=HORIZON, regime=REGIME
    )

    held = stratified_held_out(cables, SPLIT)
    assert held == {0, 1} != stratified_held_out(cables, SplitCfg())
    assert result.n_windows == sum(len(_windows_of(c)) for c in cables if c.cable_id in held)
    assert math.isfinite(result.scores["rmse_full"])


def _checkpoint_at(cables: list[CableData], depth: int) -> Checkpoint:
    """A loadable checkpoint whose every layout record says ``depth``."""
    stats = fit_stats(
        training_batches(cables, batch_size=4, history_frames=depth, cfg=SPLIT, held_out=False)
    )
    model_cfg = GPSCfg(d_model=12, n_heads=2, n_layers=1)
    width = feature_width(depth)
    return Checkpoint(
        state_dict=build_model(width, "A", model_cfg).state_dict(),
        stats=stats,
        cfg=TrainCfg(steps=1, batch_size=1, sigma=0.0, history_frames=depth),
        model_cfg=model_cfg,
        variant="A",
        feature_width=width,
        step=1,
        split=SPLIT,
    )


@pytest.mark.claim(
    "the depth used at evaluation is the one training used, read from the checkpoint, and a "
    "model met at any other depth is refused rather than scored on reshaped features "
    "(docs/method.md)"
)
@pytest.mark.parametrize("depth", [3, 7])
def test_a_checkpoint_is_evaluated_at_its_own_history_depth(depth: int) -> None:
    """The driver reads ``C`` off the checkpoint, and a mismatch is loud.

    The failure this rules out is the quiet one: windows built at the loader's
    depth against a model trained at another still produce a full metric row, and
    nothing in the scorecard says which depth it describes.
    """
    cables = _cables()
    checkpoint = _checkpoint_at(cables, depth)

    result = evaluate_checkpoint(
        checkpoint, cables, stage="seen_cables", horizon=HORIZON, regime=REGIME
    )
    assert result.n_windows > 0
    assert math.isfinite(result.scores["rmse_full"])

    with pytest.raises(ValueError, match="different velocity-history depth"):
        evaluate(
            checkpoint.restore_model(),
            cables,
            stage="seen_cables",
            horizon=HORIZON,
            stats=checkpoint.stats,
            regime=REGIME,
            split=SPLIT,
            history_frames=depth + 1,
        )


@pytest.mark.contract
def test_a_misaligned_ground_truth_window_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """The truth window and the batch are paired, and the pairing is checked.

    The driver walks the loader's window enumeration a second time to recover the
    ground truth. Drift between the two would shift every scored frame and leave
    every metric finite, so the pairing raises instead of scoring.
    """
    cables = _cables()
    truth = evaluate_module._truth_windows(
        cables, held_out=True, horizon=HORIZON, split=SPLIT, history_frames=C
    )
    monkeypatch.setattr(evaluate_module, "_truth_windows", lambda *a, **k: truth)

    with pytest.raises(RuntimeError, match="cable"):
        _run("seen_cables", cables)


@pytest.mark.contract
@pytest.mark.parametrize(
    ("call", "error", "match"),
    [
        (lambda: _run("held_out"), ValueError, "unknown stage"),
        (lambda: stage_cables(_cables(), "held_out", SPLIT), ValueError, "unknown stage"),
        (lambda: _run("seen_cables", horizon=0), ValueError, "at least 1"),
        (lambda: _run("seen_cables", limit=0), ValueError, "at least 1"),
        (lambda: _run("unseen_cables", _cables()[:1]), ValueError, "scored no window"),
        (
            lambda: evaluate_checkpoint(
                Checkpoint(
                    state_dict={},
                    stats=_stats(_cables()),
                    cfg=TrainCfg(steps=1, batch_size=1, sigma=0.0),
                    model_cfg=GPSCfg(),
                    variant="A",
                    feature_width=20,
                    step=1,
                ),
                _cables(),
                stage="seen_cables",
                horizon=HORIZON,
                regime=REGIME,
            ),
            ValueError,
            "carries no split",
        ),
    ],
    ids=["stage", "stage-cables", "horizon", "limit", "empty-stage", "no-split"],
)
def test_invalid_arguments_raise(
    call: Callable[[], object], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        call()


@pytest.mark.claim("both stages run on the test dataset and every metric returns a finite scalar: ")
def test_both_stages_run_on_the_test_assets() -> None:
    """The one test on the real loader and the real recorded cables.

    ``assets/sample_test_v1/`` is the **test** side: the same two cables as the
    training fixture, released from poses no training run saw. The split then
    slices it into the two stages rather than the datasets being separate files.

    The model is the drift stub rather than the block, because an *untrained*
    block diverges within a few autoregressive frames and every metric then reads
    NaN for a reason that has nothing to do with the driver. The block's
    composition with the loop is pinned by
    :func:`test_the_split_comes_from_the_checkpoint_not_from_a_default`, which
    rolls out a restored :class:`GPSModel`.
    """
    cables = load_dataset(SAMPLE_TEST)
    stats = _stats(cables)
    horizon = 8

    results = {
        stage: evaluate(
            _Drift(),
            cables,
            stage=stage,
            horizon=horizon,
            stats=stats,
            regime=RegimeCfg(horizons=(1, 5)),
            split=SPLIT,
            history_frames=C,
            limit=3,
        )
        for stage in sorted(STAGES)
    }

    assert stage_cables(cables, "seen_cables", SPLIT) == {45}
    assert stage_cables(cables, "unseen_cables", SPLIT) == {0}
    for stage, result in results.items():
        assert result.stage == stage
        assert result.n_windows == 3
        assert set(result.scores) == _registry_names(horizon)
        assert math.isfinite(result.scores["rmse_full"])
        assert not math.isnan(result.scores["link_length_mean_drift"])
        # rmse_h1 is now the one-step error of a real prediction, and it must be
        # strictly positive even for a stub that barely moves: the cable moved
        # between frame 0 and frame 1, so predicting no movement is already
        # wrong. The zero this used to assert was frame 0 scoring itself, which
        # is why it could never separate two arms. Alignment between the truth
        # window and the batch is enforced by _check_alignment, which raises.
        assert result.scores["rmse_h1"] > 0.0
        assert math.isfinite(result.scores["rmse_h1"])
    assert results["seen_cables"].scores != results["unseen_cables"].scores


@pytest.mark.regression(
    "the driver pinned model, statistics and cables to the CPU and deferred the device move to a "
    "caller that did not exist, so a model trained on the GPU raised here instead of being scored"
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="the second device is a CUDA device")
def test_the_driver_follows_the_model_onto_another_device() -> None:
    """A rollout is the project's heaviest step, so it runs where the model is.

    The loader builds CPU batches whatever the model's device, so the driver has
    to move the window, its labels and its truth. The two devices must agree on
    every scalar: a move that dropped a tensor would still return finite numbers.
    """
    cables = _cables()
    stats = _stats(cables)
    model = build_model(int(stats.node_mean.shape[0]), "A", GPSCfg())

    def scored(device: str) -> EvalResult:
        return evaluate(
            model.to(device),
            cables,
            stage="seen_cables",
            horizon=HORIZON,
            stats=stats,
            regime=REGIME,
            split=SPLIT,
            history_frames=C,
            limit=2,
        )

    on_cpu = scored("cpu")
    on_cuda = scored("cuda")

    assert on_cuda.n_windows == on_cpu.n_windows
    assert set(on_cuda.scores) == set(on_cpu.scores)
    for name, value in on_cpu.scores.items():
        other = on_cuda.scores[name]
        if math.isnan(value):
            assert math.isnan(other), name
        else:
            assert other == pytest.approx(value, rel=1e-4, abs=1e-6), name


# --- the per-frame curve ----------------------------------------------------


@pytest.mark.claim(
    "the per-frame curve is retained because no scalar can reconstruct it: every metric collapses "
    "the frame axis, so whether a config is better everywhere or only early is visible here alone "
    "(docs/reproduction.md)"
)
def test_the_curve_records_error_at_every_predicted_frame() -> None:
    cables = _cables()
    stats = _stats(cables)
    horizon = 6

    result = evaluate(
        _Drift(),
        cables,
        stage="seen_cables",
        horizon=horizon,
        stats=stats,
        regime=REGIME,
        split=SPLIT,
        history_frames=C,
    )

    assert set(result.curve) == {"rmse", "mean_l2", "rel_l2"}
    for name, series in result.curve.items():
        assert len(series) == horizon - 1, f"{name}: one entry per PREDICTED frame"
        assert all(math.isfinite(v) for v in series)
    # _Drift adds a fixed displacement every step, so error grows monotonically
    # and the curve has to show it rather than reporting a flat mean.
    assert result.curve["mean_l2"] == tuple(sorted(result.curve["mean_l2"]))
    assert result.curve["mean_l2"][-1] > result.curve["mean_l2"][0]


@pytest.mark.contract
def test_the_curve_agrees_with_the_terminal_metric() -> None:
    """The last point of the curve is what ``*_at_h{h}`` reports at that frame."""
    cables = _cables()
    stats = _stats(cables)
    horizon = 6

    result = evaluate(
        _Drift(),
        cables,
        stage="seen_cables",
        horizon=horizon,
        stats=stats,
        regime=RegimeCfg(horizons=(horizon - 1,)),
        split=SPLIT,
        history_frames=C,
    )

    assert result.curve["mean_l2"][-1] == pytest.approx(
        result.scores[f"mean_l2_at_h{horizon - 1}"], rel=1e-5
    )


@pytest.mark.contract
def test_stratified_limit_covers_every_stage_cable() -> None:
    """A capped run round-robins so every stage cable appears when limit allows."""
    cables = _cables()
    expected = stage_cables(cables, "seen_cables", SPLIT)
    # Two cables on seen; limit=2 must hit both under round-robin.
    capped = _run("seen_cables", cables, limit=2)
    scored = {w.key.cable_id for w in capped.windows}
    assert scored == expected
    assert capped.n_windows == 2


@pytest.mark.contract
def test_evaluate_progress_true_still_returns_result() -> None:
    """tqdm on the window loop must not change the scorecard."""
    cables = _cables()
    quiet = _run("seen_cables", cables, limit=2)
    noisy = evaluate(
        _Drift(),
        cables,
        stage="seen_cables",
        horizon=HORIZON,
        stats=_stats(cables),
        regime=REGIME,
        split=SPLIT,
        history_frames=C,
        limit=2,
        progress=True,
        desc="unit",
    )
    assert noisy.n_windows == quiet.n_windows
    assert noisy.scores["rmse_full"] == pytest.approx(quiet.scores["rmse_full"], rel=1e-9)


@pytest.mark.contract
def test_evaluate_stop_flag_returns_partial_result() -> None:
    """Soft-stop between windows keeps whatever finished."""
    cables = _cables()
    calls = {"n": 0}

    def stop_after_one() -> bool:
        calls["n"] += 1
        return calls["n"] > 1

    result = evaluate(
        _Drift(),
        cables,
        stage="seen_cables",
        horizon=HORIZON,
        stats=_stats(cables),
        regime=REGIME,
        split=SPLIT,
        history_frames=C,
        stop_flag=stop_after_one,
    )
    assert result.n_windows == 1


@pytest.mark.claim(
    "a whole-root stage scores every cable in the root and ignores the train holdout: "
)
def test_a_whole_root_stage_scores_every_cable() -> None:
    """``seen_test`` / ``unseen_test`` / ``all_cables`` cover the whole root.

    The run's split held half this population out of training, and a dedicated
    test root ignores that: every cable is scored, under the same alignment
    check as the sliced stages, so the split cannot silently drop test cables.
    """
    cables = _cables()
    every = {cable.cable_id for cable in cables}
    for stage in ("all_cables", "seen_test", "unseen_test"):
        assert stage_cables(cables, stage, SPLIT) == every

    result = _run("seen_test")
    assert {window.key.cable_id for window in result.windows} == every
    assert result.n_windows == 12, (
        "two windows from each 20-frame cable, four from each 32-frame one"
    )


def test_a_whole_root_stage_matches_the_sliced_stages_window_for_window() -> None:
    """The whole-root enumeration is the union of the two sliced enumerations.

    Same fixture, same split: the windows ``seen_test`` scores must be exactly
    the windows ``seen_cables`` and ``unseen_cables`` score between them, so a
    whole-root number is comparable with a two-stage pair and no window is
    scored twice or skipped at the boundary.
    """
    whole = {window.key for window in _run("seen_test").windows}
    sliced = {window.key for window in _run("seen_cables").windows} | {
        window.key for window in _run("unseen_cables").windows
    }
    assert whole == sliced
