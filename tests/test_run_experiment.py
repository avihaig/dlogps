"""Named experiment runner: config loading, validation, and the run layout."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import draccus
import pytest

from dlogps.data.loader import SplitCfg
from dlogps.harness.evaluate import EvalResult, WindowKey, WindowScore
from dlogps.harness.run_experiment import (
    EvalCfg,
    ExperimentCfg,
    run_experiment,
    validate_experiment,
)
from dlogps.harness.runlog import CONFIG_NAME, TRAIN_LOG_JSONL
from dlogps.harness.train import TrainCfg

PAPER_CONFIGS = ("local_on", "local_off", "chain_only", "structfree")
"""The four experiment configs behind the paper's tables."""


def _minimal_result(stage: str) -> EvalResult:
    key = WindowKey(0, 0, 4)
    frame = (0.01, 0.02)
    return EvalResult(
        stage=stage,
        n_windows=1,
        scores={"rel_l2_at_h200": 0.1, "mean_l2_at_h200": 0.05},
        counts={"rel_l2_at_h200": 1, "mean_l2_at_h200": 1},
        stds={"rel_l2_at_h200": 0.0, "mean_l2_at_h200": 0.0},
        windows=(
            WindowScore(
                key,
                {"rel_l2_at_h200": 0.1, "mean_l2_at_h200": 0.05},
                frame_errors={"mean_l2": frame, "rel_l2": frame},
            ),
        ),
        curve={"mean_l2": frame, "rel_l2": frame, "rmse": frame},
    )


@pytest.mark.contract
def test_draccus_loads_the_smoke_config() -> None:
    cfg = draccus.parse(ExperimentCfg, config_path=Path("configs/experiments/smoke.yaml"), args=[])
    assert cfg.experiment == "smoke"
    assert cfg.train.sigma == 0.1
    assert cfg.eval.arms == "model"
    assert cfg.data == Path("assets/sample_v1")
    assert validate_experiment(cfg) == max(cfg.eval.horizons) + 1


@pytest.mark.contract
@pytest.mark.parametrize("name", PAPER_CONFIGS)
def test_every_paper_config_loads_and_states_the_frozen_protocol(name: str) -> None:
    """The four shipped configs agree on everything the paper holds fixed."""
    cfg = draccus.parse(
        ExperimentCfg, config_path=Path(f"configs/experiments/{name}.yaml"), args=[]
    )
    assert cfg.experiment == name
    assert cfg.train.steps == 100_000
    assert cfg.train.batch_size == 512
    assert cfg.train.sigma == 0.1
    assert (cfg.train.lr, cfg.train.lr_final) == (0.001, 0.0001)
    assert cfg.train.weight_decay == 0.0
    assert cfg.train.grad_clip == 1.0
    assert cfg.train.history_frames == 70
    assert cfg.train.param_encoding == "designed"
    assert cfg.model.dropout == 0.0 and cfg.model.attn_dropout == 0.0
    assert cfg.split.held_out_fraction == 0.0
    assert cfg.data == Path("data/release_train")
    assert cfg.eval_data == Path("data/unseen_cables_test")
    assert cfg.eval.stage == "unseen_test"
    assert cfg.eval.limit == 400
    assert max(cfg.eval.horizons) == 400
    assert cfg.eval.arms == "model"
    assert validate_experiment(cfg) == 401


@pytest.mark.contract
def test_local_on_and_local_off_differ_only_in_the_local_stream() -> None:
    on = draccus.parse(
        ExperimentCfg, config_path=Path("configs/experiments/local_on.yaml"), args=[]
    )
    off = draccus.parse(
        ExperimentCfg, config_path=Path("configs/experiments/local_off.yaml"), args=[]
    )
    assert on.model.use_local is True and off.model.use_local is False
    assert on.model.use_global is True and off.model.use_global is True
    assert replace(on, experiment="x", model=replace(on.model, use_local=False)) == replace(
        off, experiment="x"
    )


@pytest.mark.contract
def test_chain_only_drops_the_attention_stream_and_nothing_else() -> None:
    on = draccus.parse(
        ExperimentCfg, config_path=Path("configs/experiments/local_on.yaml"), args=[]
    )
    chain = draccus.parse(
        ExperimentCfg, config_path=Path("configs/experiments/chain_only.yaml"), args=[]
    )
    assert chain.variant == "chain_only"
    assert replace(on, experiment="x", label="x", variant="x") == replace(
        chain, experiment="x", label="x", variant="x"
    )


@pytest.mark.contract
def test_validate_rejects_an_unknown_arm_and_stage() -> None:
    data = Path("data")
    base = ExperimentCfg(
        experiment="unit",
        label="x",
        data=data,
        eval_data=data,
        train=TrainCfg(steps=2, batch_size=4, sigma=0.01, eval_every=1),
        eval=EvalCfg(stage="unseen_cables", limit=1, horizons=(1, 10, 50, 100, 200)),
    )
    with pytest.raises(ValueError, match="eval.arms"):
        validate_experiment(replace(base, eval=replace(base.eval, arms="reference")))
    with pytest.raises(ValueError, match="eval.stage"):
        validate_experiment(replace(base, eval=replace(base.eval, stage="nowhere")))
    with pytest.raises(ValueError, match="horizon 200"):
        validate_experiment(replace(base, eval=replace(base.eval, horizons=(1, 10))))


def _fake_train(_root, _train_cfg, **kwargs) -> MagicMock:
    on_checkpoint = kwargs.get("on_checkpoint")
    on_eval = kwargs.get("on_eval")
    snapshot = MagicMock()
    if on_eval is not None:
        on_eval(1, 0.5)
        on_eval(2, 0.4)
    if on_checkpoint is not None:
        on_checkpoint(2, snapshot)
    result = MagicMock()
    result.losses = [0.1, 0.2]
    result.model.parameters.return_value = []
    return result


def _fake_save(path: Path, _result) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"ckpt")
    return path


def _fake_write_artifacts(eval_dir: Path, *_args, **_kwargs) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)


def _fake_write_figures(run_dir: Path) -> list[Path]:
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / "loss_mse.png"
    path.write_bytes(b"\x89PNG")
    return [path]


@pytest.mark.contract
def test_run_experiment_creates_layout(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    cfg = ExperimentCfg(
        experiment="unit",
        label="mock",
        data=data,
        eval_data=data,
        train=TrainCfg(steps=2, batch_size=4, sigma=0.01, eval_every=1, device="cpu"),
        checkpoint_at=(2,),
        eval=EvalCfg(stage="unseen_cables", limit=1, horizons=(1, 10, 50, 100, 200)),
        results_root=tmp_path / "results",
    )

    checkpoint = MagicMock()
    checkpoint.restore_model.return_value = MagicMock()
    checkpoint.stats = MagicMock()

    with (
        patch("dlogps.harness.run_experiment.train_on_dataset", side_effect=_fake_train),
        patch("dlogps.harness.run_experiment.load_checkpoint", return_value=checkpoint),
        patch("dlogps.harness.run_experiment.save_checkpoint", side_effect=_fake_save),
        patch(
            "dlogps.harness.run_experiment.evaluate",
            side_effect=lambda *_a, **kw: _minimal_result(kw["stage"]),
        ),
        patch(
            "dlogps.harness.run_experiment.load_dataset",
            return_value=[MagicMock(dt=lambda _i: 0.002)],
        ),
        patch(
            "dlogps.harness.run_experiment.write_eval_artifacts", side_effect=_fake_write_artifacts
        ),
        patch("dlogps.harness.run_experiment.write_run_figures", side_effect=_fake_write_figures),
        patch("dlogps.harness.run_experiment.assert_eval_device"),
    ):
        run_dir = run_experiment(cfg)

    assert (run_dir / CONFIG_NAME).is_file()
    assert (run_dir / "env.json").is_file()
    assert (run_dir / TRAIN_LOG_JSONL).is_file()
    assert list((run_dir / "checkpoints").glob("*.pt"))
    assert (run_dir / "eval" / "model").is_dir()
    assert [p.name for p in (run_dir / "eval").iterdir()] == ["model"]
    assert list((run_dir / "figures").glob("*.png"))


def _patched_run(cfg: ExperimentCfg, *, captured: dict, stages_seen: list[str]):
    """Run the pipeline with the heavy pieces mocked, recording the train kwargs
    and every stage the evaluator is asked for."""

    def fake_train(_root, _train_cfg, **kwargs) -> MagicMock:
        captured.update(kwargs)
        on_checkpoint = kwargs.get("on_checkpoint")
        if on_checkpoint is not None:
            on_checkpoint(2, MagicMock())
        result = MagicMock()
        result.losses = [0.1, 0.2]
        result.model.parameters.return_value = []
        return result

    def fake_evaluate(*_a, **kw) -> EvalResult:
        stages_seen.append(kw["stage"])
        return _minimal_result(kw["stage"])

    checkpoint = MagicMock()
    checkpoint.restore_model.return_value = MagicMock()
    checkpoint.stats = MagicMock()

    with (
        patch("dlogps.harness.run_experiment.train_on_dataset", side_effect=fake_train),
        patch("dlogps.harness.run_experiment.load_checkpoint", return_value=checkpoint),
        patch("dlogps.harness.run_experiment.save_checkpoint", side_effect=_fake_save),
        patch("dlogps.harness.run_experiment.evaluate", side_effect=fake_evaluate),
        patch(
            "dlogps.harness.run_experiment.load_dataset",
            return_value=[MagicMock(dt=lambda _i: 0.002)],
        ),
        patch(
            "dlogps.harness.run_experiment.write_eval_artifacts", side_effect=_fake_write_artifacts
        ),
        patch("dlogps.harness.run_experiment.write_run_figures", return_value=[]),
        patch("dlogps.harness.run_experiment.assert_eval_device"),
    ):
        run_experiment(cfg)


def _full_train_cfg(
    tmp_path: Path, *, held_out_fraction: float, unseen_root: bool
) -> ExperimentCfg:
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    return ExperimentCfg(
        experiment="unit_full",
        # Run dirs are timestamped to the second, so two configs built in one
        # test must not share a label or the second run collides with the first.
        label=f"ctl_h{int(held_out_fraction * 10)}_{'u' if unseen_root else 'n'}",
        data=data,
        eval_data=data,
        eval_unseen_data=(tmp_path / "unseen") if unseen_root else None,
        train=TrainCfg(steps=2, batch_size=4, sigma=0.01, eval_every=1, device="cpu"),
        split=SplitCfg(held_out_fraction=held_out_fraction),
        checkpoint_at=(2,),
        eval=EvalCfg(stage="seen_test", limit=1, horizons=(1, 10, 50, 100, 200), arms="model"),
        results_root=tmp_path / "results",
    )


@pytest.mark.claim("the test roots feed no selection decision")
def test_no_holdout_run_never_selects_best_on_the_test_root(tmp_path: Path) -> None:
    """With nothing held out, the one-step stream comes from the test root, so
    the best-checkpoint hook must not be armed; with a real holdout it is."""
    captured: dict = {}
    _patched_run(
        _full_train_cfg(tmp_path, held_out_fraction=0.0, unseen_root=False),
        captured=captured,
        stages_seen=[],
    )
    assert captured["on_best_checkpoint"] is None
    assert captured["eval_root"] == tmp_path / "data"

    captured.clear()
    _patched_run(
        _full_train_cfg(tmp_path, held_out_fraction=0.5, unseen_root=False),
        captured=captured,
        stages_seen=[],
    )
    assert captured["on_best_checkpoint"] is not None


@pytest.mark.contract
def test_the_unseen_test_root_is_scored_as_its_own_job(tmp_path: Path) -> None:
    """``eval_unseen_data`` appends an ``unseen_test`` job beside the configured
    stage, so both roots are scored in one run."""
    (tmp_path / "unseen").mkdir()
    stages_seen: list[str] = []
    _patched_run(
        _full_train_cfg(tmp_path, held_out_fraction=0.0, unseen_root=True),
        captured={},
        stages_seen=stages_seen,
    )
    assert set(stages_seen) == {"seen_test", "unseen_test"}
