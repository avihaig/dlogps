"""The run directory contract: config round-trip, provenance and partial logs.

Pins the three failures that are silent rather than red: a config that cannot
re-run the run, a dirty tree recorded as clean, and a killed run whose log is
unreadable because it was buffered.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch
import yaml

from dlogps.data.loader import SplitCfg
from dlogps.data.types import Batch
from dlogps.harness.runlog import (
    CONFIG_NAME,
    ENV_NAME,
    TRAIN_LOG_JSONL,
    TRAIN_LOG_TXT,
    RunCfg,
    RunLog,
    git_state,
    load_run_cfg,
    run_dir_name,
)
from dlogps.harness.train import BatchSource, TrainCfg, train
from dlogps.model.gps import GPSCfg
from tests.fake_data import make_fake_batch

torch.set_num_threads(1)

FIXED_START = datetime(2026, 8, 5, 16, 12, 3, tzinfo=UTC)


def stub_source(b: int = 2, batches: int = 2) -> tuple[Batch, ...]:
    """A tiny fixed training stream, no file on disk and no simulator."""
    return tuple(make_fake_batch(b=b, seed=s)[0] for s in range(batches))


def source_of(batches: Iterable[Batch]) -> BatchSource:
    """Wrap a fixed collection as a :data:`BatchSource`, ignoring the epoch."""
    held = tuple(batches)
    return lambda epoch: held


def small_cfg(**overrides: object) -> TrainCfg:
    """A short run on the stub. ``sigma`` is passed because it has no default."""
    settings: dict[str, object] = {"steps": 3, "batch_size": 2, "sigma": 0.01, "seed": 0}
    settings.update(overrides)
    return TrainCfg(**settings)  # pyright: ignore[reportArgumentType]


def sample_run_cfg(**train_overrides: object) -> RunCfg:
    return RunCfg(
        data=Path("assets/sample_v1"),
        variant="A",
        train=small_cfg(**train_overrides),
        model=GPSCfg(),
        split=SplitCfg(),
        checkpoint_at=(6000, 20000, 40000),
    )


def _init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "runlog@test"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "runlog"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    (path / "tracked.txt").write_text("clean\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=path,
        check=True,
        capture_output=True,
    )


@pytest.mark.contract
def test_run_dir_name_is_timestamp_plus_label() -> None:
    assert run_dir_name("ref", when=FIXED_START) == "2026-08-05T16-12-03_ref"
    assert run_dir_name("my run", when=FIXED_START) == "2026-08-05T16-12-03_my_run"


@pytest.mark.contract
def test_start_creates_the_runs_subdirectory(tmp_path: Path) -> None:
    cfg = sample_run_cfg()
    log = RunLog.start(tmp_path / "capacity_2x2", "ref", cfg, when=FIXED_START)
    assert log.run_dir == tmp_path / "capacity_2x2" / "runs" / "2026-08-05T16-12-03_ref"
    assert log.checkpoints_dir.is_dir()
    assert log.eval_dir.is_dir()
    assert log.checkpoint_path(6000) == log.checkpoints_dir / "step_06000.pt"


@pytest.mark.contract
def test_config_yaml_records_every_resolved_field(tmp_path: Path) -> None:
    # save_preds is RunCfg's, not TrainCfg's: it is a property of the run's
    # artifacts rather than of the optimizer, and routing it through the train
    # overrides is what the discarded first draft of this test did.
    cfg = RunCfg(
        data=Path("assets/sample_v1"),
        variant="A",
        train=small_cfg(seed=7),
        model=GPSCfg(),
        split=SplitCfg(),
        checkpoint_at=(6000, 20000, 40000),
        save_preds=True,
    )
    run_dir = tmp_path / "run"
    RunLog(run_dir, cfg, when=FIXED_START)

    raw = yaml.safe_load((run_dir / CONFIG_NAME).read_text())
    assert raw["data"] == str(cfg.data)
    assert raw["variant"] == "A"
    assert raw["train"]["seed"] == 7
    assert raw["train"]["sigma"] == pytest.approx(0.01)
    assert raw["model"]["d_model"] == 96
    assert raw["split"]["held_out_fraction"] == pytest.approx(0.2)
    assert raw["checkpoint_at"] == [6000, 20000, 40000]
    assert raw["save_preds"] is True


@pytest.mark.scaffolding("run layout")
def test_a_run_can_be_re_run_from_its_config_yaml(tmp_path: Path) -> None:
    """Verification item 1: config.yaml alone reproduces the first-step loss."""
    cfg = sample_run_cfg(steps=2, seed=42, eval_every=1)
    batches = stub_source()
    first = train(
        source_of(batches),
        cfg.train,
        variant=cfg.variant,
        model_cfg=cfg.model,
        split=cfg.split,
    )

    run_dir = tmp_path / "run"
    RunLog(run_dir, cfg, when=FIXED_START)
    loaded = load_run_cfg(run_dir)

    second = train(
        source_of(batches),
        loaded.train,
        variant=loaded.variant,
        model_cfg=loaded.model,
        split=loaded.split,
    )
    assert second.losses[0] == first.losses[0]


@pytest.mark.scaffolding("run layout")
def test_env_json_records_git_dirty_on_a_dirty_tree(tmp_path: Path) -> None:
    """Verification item 2: git_dirty is true when the tree is not clean."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    sha, dirty = git_state(repo)
    assert sha is not None
    assert dirty is False

    (repo / "tracked.txt").write_text("dirty\n")
    _, dirty_after = git_state(repo)
    assert dirty_after is True

    run_dir = tmp_path / "run"
    RunLog(run_dir, sample_run_cfg(), repo=repo, when=FIXED_START)
    env = json.loads((run_dir / ENV_NAME).read_text())
    assert env["git_dirty"] is True
    assert env["git_sha"] == sha
    assert env["python"]
    assert env["torch"]
    assert env["host"]


@pytest.mark.scaffolding("run layout")
def test_train_log_jsonl_stays_valid_when_the_run_is_killed(tmp_path: Path) -> None:
    """Verification item 3: partial runs keep readable JSONL."""
    run_dir = tmp_path / "run"
    log = RunLog(run_dir, sample_run_cfg(), when=FIXED_START)
    log.log_step(100, 1.83e-2, 9.4e-3, 9.9e-4, 1.9)
    log.log_step(200, 1.50e-2, 8.1e-3, 9.5e-4, 3.8)

    jsonl_lines = (run_dir / TRAIN_LOG_JSONL).read_text().strip().splitlines()
    assert len(jsonl_lines) == 2
    for line in jsonl_lines:
        parsed = json.loads(line)
        assert "step" in parsed
        assert "train_loss" in parsed
        assert "eval_loss" in parsed

    txt_lines = (run_dir / TRAIN_LOG_TXT).read_text().strip().splitlines()
    assert len(txt_lines) == 2
    assert txt_lines[0].startswith("step 100")


@pytest.mark.scaffolding("run layout")
def test_finish_records_elapsed_seconds(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    log = RunLog(run_dir, sample_run_cfg(), when=FIXED_START)
    log.finish(764.4)
    env = json.loads((run_dir / ENV_NAME).read_text())
    assert env["seconds"] == pytest.approx(764.4)
    assert "finished" in env
    assert "started" in env
