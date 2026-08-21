"""The training loop, and the four ways a run is silently wrong rather than red.

Most of this file pins orderings. A run that fits its statistics after the noise,
takes the loss on absolute positions, lets a held-out cable into the statistics,
or is not reproducible from its seed all train perfectly well and produce a
number nobody can tell is wrong. Each of those has one test here, and they are
the reason the module exists in this shape:
:func:`test_the_loss_is_on_the_normalized_displacement`,
:func:`test_statistics_are_fitted_before_the_noise`,
:func:`test_a_held_out_cable_cannot_move_the_statistics` and
:func:`test_two_runs_at_one_seed_agree_bit_for_bit`.

Nothing here is marked ``integration``: no test builds a scene, and pinning torch
to one thread (below) puts the whole file, ``assets/sample_v1/`` included, at
about two seconds.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import MISSING, fields
from pathlib import Path

import pytest
import torch

import dlogps.model as model_api
from dlogps.data.dataset import CableData, feature_width
from dlogps.data.loader import SplitCfg, stratified_held_out
from dlogps.data.types import Batch, N
from dlogps.harness.normalize import fit_stats, normalize
from dlogps.harness.train import (
    VARIANTS,
    BatchSource,
    Checkpoint,
    TrainCfg,
    TrainResult,
    build_model,
    dataset_source,
    decay_gamma,
    held_out_source,
    load_checkpoint,
    main,
    pooled_loss,
    save_checkpoint,
    step_loss,
    train,
    train_on_dataset,
)
from dlogps.model.bias import NoBias
from dlogps.model.gps import GPSCfg, GPSModel
from dlogps.model.variants import SpaceDistanceBias
from tests.fake_data import HISTORY_FRAMES as C
from tests.fake_data import make_fake_batch

# One thread. The block is tiny at these batch sizes, a few thousand elements per
# op over 33 nodes, so torch's default intra-op pool spends orders of magnitude
# more on dispatch than on arithmetic: measured here at ~18 ms per step against
# ~2 s per step on 24 threads. It is also what keeps the fast suite inside its
# five-second budget on a loaded machine.
torch.set_num_threads(1)

SAMPLE = Path(__file__).resolve().parents[1] / "assets" / "sample_v1"
"""The loader's fixture: two real cables, two complete episodes each."""

SMOKE = 20
"""Steps in the loss-trend run. Enough for Adam to move on two stub cables, short
enough that the fast suite stays under its budget."""


def stub_source(b: int = 2, batches: int = 2) -> tuple[Batch, ...]:
    """A tiny fixed training stream, no file on disk and no simulator."""
    return tuple(make_fake_batch(b=b, seed=s)[0] for s in range(batches))


def source_of(batches: Iterable[Batch]) -> BatchSource:
    """Wrap a fixed collection as a :data:`BatchSource`, ignoring the epoch."""
    held = tuple(batches)
    return lambda epoch: held


def small_cfg(**overrides: object) -> TrainCfg:
    """A short run on the stub. ``sigma`` is passed because it has no default."""
    settings: dict[str, object] = {"steps": 3, "batch_size": 2, "sigma": 0.01}
    settings.update(overrides)
    return TrainCfg(**settings)  # pyright: ignore[reportArgumentType]


def fake_cable(cable_id: int, bend_stiffness: float, scale: float) -> CableData:
    """One in-memory cable, long enough to window and cheap enough to fit.

    Built here rather than read from ``assets/`` so the statistics-leak test runs
    in the fast suite: the invariant under test is the split, not the bytes.
    """
    frames = C + 5
    pos = torch.arange(frames * N * 3, dtype=torch.float32).reshape(frames, N, 3) * scale
    vel = torch.ones(frames, N, 3) * scale
    t = torch.arange(frames, dtype=torch.float32) * 2e-3
    return CableData(
        episodes=[(pos, vel, t)],
        params={
            "length": 1.0,
            "diameter": 0.003,
            "bend_stiffness": bend_stiffness,
            "joint_damping": 1e-3,
            "effective_linear_density": 0.1,
        },
        cable_id=cable_id,
    )


@pytest.mark.claim("sigma is one number for the whole matrix, set once")
def test_sigma_has_no_default() -> None:
    """A run that silently picked a noise scale would measure the tuning.

    Asserted on the dataclass rather than by catching the ``TypeError``, because
    what is forbidden is a *value* being available, not the call failing.
    """
    sigma = next(f for f in fields(TrainCfg) if f.name == "sigma")

    assert sigma.default is MISSING
    assert sigma.default_factory is MISSING
    with pytest.raises(TypeError):
        TrainCfg(steps=1, batch_size=1)  # pyright: ignore[reportCallIssue]
    with pytest.raises(SystemExit):
        main(["--data", str(SAMPLE), "--out", "unwritten.pt", "--steps", "1"])


@pytest.mark.contract
@pytest.mark.parametrize(
    "override",
    [
        {"steps": 0},
        {"batch_size": 0},
        {"eval_every": 0},
        {"sigma": -1e-3},
        {"lr": 0.0},
        {"lr_final": 0.0},
        {"lr_final": 1.0, "lr": 1e-3},
        {"grad_clip": 0.0},
        {"weight_decay": -1.0},
        {"history_frames": 0},
    ],
)
def test_train_cfg_rejects_an_impossible_run(override: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        small_cfg(**override)


@pytest.mark.contract
def test_variant_names_map_to_the_plugins() -> None:
    """Every name in the registry builds, and the two special cases are honoured."""
    assert set(VARIANTS) == {"A", "B", "C", "D", "F", "chain_only"}
    for name in ("A", "B", "C", "D", "chain_only"):
        assert build_model(17, name).feature_width == 17

    assert isinstance(build_model(17, "B").bias, SpaceDistanceBias)
    assert isinstance(
        build_model(17, "F", GPSCfg(d_model=24, n_heads=8)).bias,
        model_api.BCOnlyBias,
    )
    assert isinstance(build_model(17, "chain_only").bias, NoBias)
    assert not build_model(17, "chain_only").cfg.use_global
    # A variant may switch a stream off, never back on: the crossed ablation is
    # the caller's and survives the variant.
    assert not build_model(17, "B", GPSCfg(use_local=False)).cfg.use_local
    with pytest.raises(ValueError, match="variant F requires n_heads=8"):
        build_model(17, "F")


@pytest.mark.claim("the loss is on the normalized displacement")
def test_the_loss_is_on_the_normalized_displacement() -> None:
    """Scored on the step, not the position, and in normalized units.

    Both failures are silent. Absolute positions are metres dominated by where
    the cable is rather than where it is going, and unnormalized displacements
    are millimetres whose gradient is a rounding error next to the features.
    """
    batch = stub_source()[0]
    stats = fit_stats([batch])
    model = build_model(batch.node_feat.shape[-1], "A")
    model.eval()

    with torch.no_grad():
        predicted = model(batch)
        expected = torch.nn.functional.mse_loss(
            (predicted - batch.pos - stats.step_mean) / stats.step_std,
            (batch.target - batch.pos - stats.step_mean) / stats.step_std,
        )
        on_positions = torch.nn.functional.mse_loss(predicted, batch.target)

        assert torch.allclose(step_loss(model, batch, stats), expected)
        assert not torch.allclose(step_loss(model, batch, stats), on_positions)


@pytest.mark.claim("loss falls on a smoke run")
def test_the_loss_falls_on_the_stub() -> None:
    """A short run on two stub cables, with no file on disk and no simulator."""
    result = train(source_of(stub_source()), small_cfg(steps=SMOKE))
    quarter = SMOKE // 4
    head = sum(result.losses[:quarter]) / quarter
    tail = sum(result.losses[-quarter:]) / quarter

    assert result.steps == SMOKE
    assert tail < head


@pytest.mark.contract
def test_the_result_reports_wall_clock() -> None:
    """Total and per step, because nothing else in the project costs a run."""
    result = train(source_of(stub_source()), small_cfg())

    assert result.seconds > 0.0
    assert result.seconds_per_step == pytest.approx(result.seconds / result.steps)


@pytest.mark.claim("determinism from the seed alone")
def test_two_runs_at_one_seed_agree_bit_for_bit() -> None:
    """Asserted on bytes, the standard the sweep was held to.

    Weight init, the input noise and the shuffle all derive from ``cfg.seed``, so
    a stream the seed does not reach shows up here and nowhere else.
    """
    batches = stub_source()
    first = train(source_of(batches), small_cfg(steps=5, seed=7))
    again = train(source_of(batches), small_cfg(steps=5, seed=7))
    other = train(source_of(batches), small_cfg(steps=5, seed=8))

    assert first.losses == again.losses
    assert first.losses != other.losses


@pytest.mark.claim("statistics are fitted before the noise")
def test_statistics_are_fitted_before_the_noise() -> None:
    """Fitted on the corruption, ``sigma`` would be measured in its own units."""
    batches = stub_source()
    result = train(source_of(batches), small_cfg(sigma=0.5))
    clean = fit_stats(batches)

    assert torch.equal(result.stats.node_mean, clean.node_mean)
    assert torch.equal(result.stats.node_std, clean.node_std)
    assert torch.equal(result.stats.step_std, clean.step_std)


@pytest.mark.claim("statistics see the training side only")
def test_a_held_out_cable_cannot_move_the_statistics() -> None:
    """The leak that no downstream number reveals.

    The held-out cable is replaced by one whose every channel is a thousand times
    larger. If it reached the statistics at all, the fitted scale would move by
    orders of magnitude; it must not move by a bit. The served cable ids are
    asserted alongside because they are the mechanism: ``held_out=False`` is
    written in :func:`dataset_source` and in no other path from a directory to a
    batch.
    """
    split = SplitCfg(held_out_fraction=0.5)
    cables = [fake_cable(0, 1e-3, scale=1.0), fake_cable(1, 1e-1, scale=1.0)]
    held = stratified_held_out(cables, split)
    assert len(held) == 1, "the fixture must put one cable on each side"

    loud = [
        fake_cable(c.cable_id, c.params["bend_stiffness"], scale=1e3) if c.cable_id in held else c
        for c in cables
    ]
    cfg = small_cfg(batch_size=2)
    served = {int(batch.meta.cable_id[0]) for batch in dataset_source(cables, cfg, split)(0)}
    quiet_stats = fit_stats(dataset_source(cables, cfg, split)(0))
    loud_stats = fit_stats(dataset_source(loud, cfg, split)(0))

    assert served and not (served & held)
    assert torch.equal(quiet_stats.node_mean, loud_stats.node_mean)
    assert torch.equal(quiet_stats.node_std, loud_stats.node_std)
    assert torch.equal(quiet_stats.step_std, loud_stats.step_std)


@pytest.mark.contract
def test_the_schedule_lands_on_lr_final() -> None:
    """Exponential decay over ``steps - 1``, so the last step runs at ``lr_final``."""
    cfg = small_cfg(steps=50, lr=1e-3, lr_final=1e-5)

    assert cfg.lr * decay_gamma(cfg) ** (cfg.steps - 1) == pytest.approx(cfg.lr_final)
    assert decay_gamma(small_cfg(steps=1)) == 1.0


@pytest.mark.claim("a checkpoint reproduces its own prediction")
def test_the_checkpoint_round_trips(tmp_path: Path) -> None:
    """Weights, statistics, config and variant, or it cannot be evaluated."""
    batches = stub_source()
    result = train(source_of(batches), small_cfg(), variant="C")
    saved = load_checkpoint(save_checkpoint(tmp_path / "run.pt", result))

    result.model.eval()
    probe = normalize(batches[0], saved.stats)
    with torch.no_grad():
        assert torch.equal(saved.restore_model()(probe), result.model(probe))

    assert isinstance(saved, Checkpoint)
    assert saved.variant == "C"
    assert saved.cfg == result.cfg
    assert saved.step == result.steps
    assert saved.feature_width == batches[0].node_feat.shape[-1]
    assert torch.equal(saved.stats.step_std, result.stats.step_std)


@pytest.mark.contract
def test_an_eight_head_f_checkpoint_round_trips(tmp_path: Path) -> None:
    """The existing checkpoint path restores F's plugin state and prediction."""
    batches = stub_source()
    model_cfg = GPSCfg(d_model=24, n_heads=8, n_layers=2)
    result = train(source_of(batches), small_cfg(), variant="F", model_cfg=model_cfg)
    saved = load_checkpoint(save_checkpoint(tmp_path / "run_f.pt", result))

    result.model.eval()
    restored = saved.restore_model()
    probe = normalize(batches[0], saved.stats)
    with torch.no_grad():
        assert torch.equal(restored(probe), result.model(probe))

    assert saved.variant == "F"
    assert isinstance(restored, GPSModel)
    assert isinstance(result.model, GPSModel)
    assert isinstance(restored.bias, model_api.BCOnlyBias)
    assert isinstance(result.model.bias, model_api.BCOnlyBias)
    assert torch.equal(restored.bias.rate, result.model.bias.rate)


@pytest.mark.claim(
    "the velocity-history depth rides in the checkpoint alongside Stats and SplitCfg, "
    "because a model cannot be evaluated except on windows of the depth it was trained at "
    "(docs/method.md)"
)
@pytest.mark.parametrize("depth", [3, 7])
def test_the_history_depth_rides_in_the_checkpoint(tmp_path: Path, depth: int) -> None:
    batches = tuple(make_fake_batch(b=2, seed=s, history_frames=depth)[0] for s in range(2))
    result = train(source_of(batches), small_cfg(history_frames=depth))

    saved = load_checkpoint(save_checkpoint(tmp_path / "run.pt", result))

    assert saved.cfg.history_frames == depth
    assert saved.feature_width == feature_width(depth)
    assert saved.restore_model().feature_width == feature_width(depth)


@pytest.mark.contract
@pytest.mark.parametrize("version", [-1, 1], ids=["foreign", "previous"])
def test_load_checkpoint_refuses_a_foreign_payload(tmp_path: Path, version: int) -> None:
    """Version 1 is refused as firmly as nonsense is.

    It carried no ``history_frames``, so reading one would mean supplying the
    depth from whatever default the reading code happens to hold, which is the
    inference this format bump exists to prevent.
    """
    path = tmp_path / "foreign.pt"
    torch.save({"format": version}, path)

    with pytest.raises(ValueError, match="checkpoint format"):
        load_checkpoint(path)


@pytest.mark.regression(
    "a payload whose feature_width and history_frames disagree rebuilds an encoder of one "
    "width and is then fed windows of another"
)
def test_a_checkpoint_whose_layout_records_disagree_is_refused(tmp_path: Path) -> None:
    path = save_checkpoint(tmp_path / "run.pt", train(source_of(stub_source()), small_cfg()))
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["cfg"]["history_frames"] += 1
    torch.save(payload, path)

    with pytest.raises(ValueError, match="cannot be evaluated at either depth"):
        load_checkpoint(path)


@pytest.mark.regression(
    "a stream of one depth trained under a config naming another produces a checkpoint whose "
    "two layout records disagree, and nothing said so until it failed to load"
)
def test_a_source_whose_depth_is_not_the_configs_is_refused() -> None:
    with pytest.raises(ValueError, match="history_frames"):
        train(source_of(stub_source()), small_cfg(history_frames=C + 3))


@pytest.mark.claim("the loop trains on assets/sample_v1")
def test_it_trains_on_the_sample_dataset() -> None:
    """End to end on real bytes: two cables, two episodes, the loader's fixture.

    The stub cannot catch a loader whose real batches disagree with the contract,
    which is the only failure this adds over the runs above.
    """
    result = train_on_dataset(SAMPLE, small_cfg(steps=5, batch_size=4), variant="B")

    assert result.steps == 5
    assert all(math.isfinite(loss) and loss > 0 for loss in result.losses)
    assert result.split == SplitCfg()


@pytest.mark.contract
def test_the_standalone_cli_excludes_f(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """F needs eight heads, so it is reachable only through the YAML-driven runner."""
    with pytest.raises(SystemExit) as help_exit:
        main(["--help"])
    assert help_exit.value.code == 0
    help_text = capsys.readouterr().out
    assert "{A,B,C,D,chain_only}" in help_text
    assert "{A,B,C,D,F,chain_only}" not in help_text

    with pytest.raises(SystemExit) as invalid_exit:
        main(
            [
                "--data",
                str(tmp_path / "missing-data"),
                "--out",
                str(tmp_path / "must-not-exist.pt"),
                "--steps",
                "1",
                "--sigma",
                "0.1",
                "--variant",
                "F",
            ]
        )
    assert invalid_exit.value.code != 0
    assert not (tmp_path / "must-not-exist.pt").exists()


@pytest.mark.contract
def test_the_cli_writes_a_checkpoint(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The wall clock is printed where a human reads it, and the run is saved."""
    out = tmp_path / "cli.pt"
    code = main(
        [
            "--data",
            str(SAMPLE),
            "--out",
            str(out),
            "--steps",
            "3",
            "--sigma",
            "0.01",
            "--seed",
            "1",
            "--variant",
            "chain_only",
            "--batch-size",
            "4",
            "--eval-every",
            "1",
        ]
    )

    assert code == 0
    assert load_checkpoint(out).variant == "chain_only"
    assert "ms/step" in capsys.readouterr().out


@pytest.mark.contract
def test_an_empty_source_is_loud() -> None:
    """The statistics pass is the first thing to touch the stream, so it is the
    one that complains. Loud either way; silently training on nothing is the
    failure being ruled out."""
    with pytest.raises(ValueError, match="empty"):
        train(source_of([]), small_cfg())


@pytest.mark.claim(
    "a run reports a pooled loss on the held-out side of its own split, which is the only "
    "generalization signal it produces short of a rollout"
)
def test_the_run_scores_the_held_out_side_pooled() -> None:
    """A per-batch training loss cannot say whether a run generalizes.

    Reported at each ``eval_every`` and on the last step, so a run that plateaus
    or turns over is visible while it is still running rather than afterwards.
    """
    split = SplitCfg(held_out_fraction=0.5)
    cables = [fake_cable(0, 1e-3, scale=1.0), fake_cable(1, 1e-1, scale=1.0)]
    cfg = small_cfg(steps=4, batch_size=2, eval_every=2)

    result = train(
        dataset_source(cables, cfg, split),
        cfg,
        split=split,
        eval_source=held_out_source(cables, cfg, split),
    )

    assert [step for step, _ in result.eval_losses] == [2, 4]
    assert all(math.isfinite(loss) for _, loss in result.eval_losses)
    assert result.final_eval_loss == result.eval_losses[-1][1]

    # And it is the held-out side that was scored, not the training side: the two
    # cables differ by two orders of magnitude in stiffness, so the figures do.
    on_training = pooled_loss(result.model, dataset_source(cables, cfg, split), cfg, result.stats)
    assert on_training != result.final_eval_loss


@pytest.mark.claim(
    "statistics are fitted on the training side only; scoring the held-out side is a second "
    "path to that stream and it must stay read-only"
)
def test_scoring_the_held_out_side_cannot_move_the_statistics() -> None:
    """The leak the held-out scorer could have reopened.

    ``held_out_source`` exists precisely so that ``dataset_source`` keeps its
    single ``held_out=False``. A run that scored the held-out side and fitted
    from it would show up nowhere downstream, so the held-out cable is made a
    thousand times louder and the fitted scale must not move by a bit.
    """
    split = SplitCfg(held_out_fraction=0.5)
    cables = [fake_cable(0, 1e-3, scale=1.0), fake_cable(1, 1e-1, scale=1.0)]
    held = stratified_held_out(cables, split)
    loud = [
        fake_cable(c.cable_id, c.params["bend_stiffness"], scale=1e3) if c.cable_id in held else c
        for c in cables
    ]
    cfg = small_cfg(steps=2, batch_size=2, eval_every=2)

    def run(population: list[CableData]) -> torch.Tensor:
        return train(
            dataset_source(population, cfg, split),
            cfg,
            split=split,
            eval_source=held_out_source(population, cfg, split),
        ).stats.step_std

    assert torch.equal(run(cables), run(loud))


@pytest.mark.regression(
    "the trainer copies the fitted step scale into the head's buffers, and nothing read them: "
    "deleting the copy left all 292 tests green while the head's output units stopped matching "
    "the units the loss is taken in"
)
def test_the_trainer_fills_the_heads_step_scale_from_the_fitted_statistics() -> None:
    """The head's units come from the run's own statistics, never from a default.

    Identity buffers are the constructor's default, so a trainer that forgets the
    copy leaves a model behaving as if one standardized unit were one metre.
    """
    batches = stub_source(batches=3)
    result = train(source_of(batches), small_cfg())

    assert torch.equal(result.model.step_std, result.stats.step_std)
    assert torch.equal(result.model.step_mean, result.stats.step_mean)
    assert not torch.allclose(result.model.step_std, torch.ones(3)), (
        "the fitted scale must not coincide with the constructor's identity, or this passes "
        "against a trainer that never copied it"
    )

    # And it survives the round trip, since rollout rebuilds the model from disk.
    restored = Checkpoint(
        state_dict=result.model.state_dict(),
        stats=result.stats,
        cfg=result.cfg,
        model_cfg=result.model.cfg,
        variant=result.variant,
        feature_width=result.model.feature_width,
        step=result.steps,
    ).restore_model()
    assert torch.equal(restored.step_std, result.stats.step_std)


@pytest.mark.claim(
    "the target is standardized, so a model predicting the mean per-frame displacement scores "
    "exactly 1.0: the reference line every loss in the project is read against "
    ""
)
def test_a_zero_head_scores_exactly_the_constant_predictor_line() -> None:
    """The anchor for every loss this project reports, asserted rather than assumed.

    Pooled over exactly the stream ``fit_stats`` saw, the constant predictor
    scores 1.0 by construction: the loss is the mean square of the standardized
    step and the standardization divides by that same spread. Below 1.0 is real
    prediction and above it is worse than a constant, so a run whose reference
    line has drifted cannot be read at all.
    """
    batches = stub_source(batches=4)
    stats = fit_stats(batches)
    model = build_model(batches[0].node_feat.shape[-1], "A")
    with torch.no_grad():
        model.step_std.copy_(stats.step_std)
        model.step_mean.copy_(stats.step_mean)

    # Clean batches: the line is a property of the data and its statistics, and
    # the input noise is not part of it.
    def pooled() -> float:
        with torch.no_grad():
            losses = [step_loss(model, normalize(batch, stats), stats) for batch in batches]
        return float(torch.stack(losses).mean())

    assert pooled() == pytest.approx(1.0, abs=1e-4)

    # Predicting no movement at all is worse, because the mean step is gravity
    # rather than zero.
    with torch.no_grad():
        model.step_mean.zero_()
    assert pooled() > 1.0


# --- mid-run checkpoints ----------------------------------------------------


@pytest.mark.claim(
    "a mid-run checkpoint carries the same Stats and SplitCfg as the final one, so a checkpoint "
    "scored at 6000 steps and one scored at 40000 are scored under the same scaling and the same "
    "stage"
)
def test_a_mid_run_checkpoint_shares_the_run_s_scaling_and_split() -> None:
    seen: list[tuple[int, TrainResult]] = []
    cfg = TrainCfg(steps=4, batch_size=2, sigma=0.0, eval_every=100, device="cpu")

    final = train(
        source_of(stub_source(batches=4)),
        cfg,
        split=SplitCfg(held_out_fraction=0.5),
        checkpoint_at=[2, 3],
        on_checkpoint=lambda step, snap: seen.append((step, snap)),
    )

    assert [step for step, _ in seen] == [2, 3]
    for step, snapshot in seen:
        assert snapshot.steps == step, "a snapshot reports the step it was taken at"
        assert snapshot.split == final.split
        for name in ("node_mean", "node_std", "step_mean", "step_std"):
            assert torch.equal(getattr(snapshot.stats, name), getattr(final.stats, name))


@pytest.mark.claim(
    "on_best_checkpoint fires on the first held-out report and again only when "
    "the pooled loss strictly improves, so a rolling best.pt tracks one-step MSE"
)
def test_on_best_checkpoint_tracks_strict_improvements() -> None:
    best_steps: list[tuple[int, float]] = []
    split = SplitCfg(held_out_fraction=0.5)
    cables = [fake_cable(0, 1e-3, scale=1.0), fake_cable(1, 1e-1, scale=1.0)]
    cfg = small_cfg(steps=6, batch_size=2, eval_every=2)

    train(
        dataset_source(cables, cfg, split),
        cfg,
        split=split,
        eval_source=held_out_source(cables, cfg, split),
        on_best_checkpoint=lambda step, loss, _snap: best_steps.append((step, loss)),
    )

    assert best_steps, "first eval report must seed the best tracker"
    assert best_steps[0][0] == 2
    losses = [loss for _, loss in best_steps]
    for earlier, later in zip(losses, losses[1:], strict=False):
        assert later < earlier


@pytest.mark.regression(
    "a checkpoint step past the end of the run was silently skipped, which is how a grid ends up "
    "with a missing arm and a summary table that still looks complete"
)
def test_an_unreachable_checkpoint_step_raises() -> None:
    cfg = TrainCfg(steps=3, batch_size=2, sigma=0.0, eval_every=100, device="cpu")

    with pytest.raises(ValueError, match="but the run is 3 steps long"):
        train(
            source_of(stub_source(batches=4)),
            cfg,
            checkpoint_at=[2, 9],
            on_checkpoint=lambda *_: None,
        )

    with pytest.raises(ValueError, match="no on_checkpoint"):
        train(source_of(stub_source(batches=4)), cfg, checkpoint_at=[2])


@pytest.mark.claim(
    "with nothing held out, one-step eval reads the dedicated test root and nothing is "
    "selected on it"
)
def test_the_eval_root_feeds_one_step_eval_when_nothing_is_held_out() -> None:
    """``held_out_fraction=0`` leaves no held-out stream; ``eval_root`` restores
    a monitoring signal from a second root, and omitting both skips eval."""
    cfg = small_cfg(steps=2, batch_size=4, eval_every=1)
    no_split = SplitCfg(held_out_fraction=0.0)

    silent = train_on_dataset(SAMPLE, cfg, split=no_split)
    assert silent.eval_losses == (), "no held-out cables and no eval_root: nothing to score"

    fed = train_on_dataset(SAMPLE, cfg, split=no_split, eval_root=SAMPLE.parent / "sample_test_v1")
    assert len(fed.eval_losses) == 2
    assert all(math.isfinite(loss) for _, loss in fed.eval_losses)
