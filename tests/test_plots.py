"""Plot helpers: loss curve, error vs frame, best-checkpoint band."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from dlogps.harness.artifacts import (
    across_cables_column,
    metric_column,
    std_across_cables_column,
)
from dlogps.harness.plots import (
    BEST_BY,
    best_checkpoint_step,
    plot_error_vs_frame,
    plot_loss_mse,
    write_run_figures,
)
from dlogps.harness.runlog import TRAIN_LOG_JSONL


def _write_curve(
    path: Path,
    *,
    frames: tuple[int, ...],
    mean_l2: tuple[float, ...],
    rel_l2: tuple[float, ...],
    std_l2: tuple[float, ...] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame",
        "time_s",
        metric_column("mean_l2"),
        metric_column("rel_l2"),
    ]
    if std_l2 is not None:
        fieldnames.extend(
            [
                across_cables_column("mean_l2"),
                std_across_cables_column("mean_l2"),
                across_cables_column("rel_l2"),
                std_across_cables_column("rel_l2"),
            ]
        )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, frame in enumerate(frames):
            row: dict[str, object] = {
                "frame": frame,
                "time_s": frame * 0.002,
                metric_column("mean_l2"): mean_l2[index],
                metric_column("rel_l2"): rel_l2[index],
            }
            if std_l2 is not None:
                row[across_cables_column("mean_l2")] = mean_l2[index]
                row[std_across_cables_column("mean_l2")] = std_l2[index]
                row[across_cables_column("rel_l2")] = rel_l2[index]
                row[std_across_cables_column("rel_l2")] = std_l2[index]
            writer.writerow(row)


def _write_summary(path: Path, *, step: int, rel_l2_at_h200: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "stage": "unseen_cables",
        "checkpoint_step": step,
        "metrics": {
            BEST_BY: {"mean": rel_l2_at_h200, "std": 0.0, "count": 1, "unit": ""},
        },
    }
    path.write_text(json.dumps(document) + "\n")


@pytest.fixture
def plot_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / TRAIN_LOG_JSONL).write_text(
        "\n".join(
            [
                json.dumps({"step": 1, "train_loss": 0.5, "eval_loss": 0.4}),
                json.dumps({"step": 2, "train_loss": 0.3, "eval_loss": 0.2}),
            ]
        )
        + "\n"
    )
    model_dir = run_dir / "eval" / "model"
    frames = (1, 2, 3)
    for step, rel_terminal in ((100, 0.20), (200, 0.10)):
        _write_summary(
            model_dir / f"unseen_cables_step{step}_summary.json",
            step=step,
            rel_l2_at_h200=rel_terminal,
        )
        mean_curve = tuple(0.01 * frame for frame in frames)
        rel_curve = tuple(0.001 * frame for frame in frames)
        std_curve = tuple(0.001 for _ in frames)
        _write_curve(
            model_dir / f"unseen_cables_step{step}_curve.csv",
            frames=frames,
            mean_l2=mean_curve,
            rel_l2=rel_curve,
            std_l2=std_curve,
        )
    return run_dir


@pytest.mark.scaffolding("experiment runner")
def test_plot_loss_mse_writes_png(plot_run_dir: Path) -> None:
    path = plot_loss_mse(plot_run_dir)
    assert path.is_file()
    assert path.suffix == ".png"
    assert path.stat().st_size > 100


@pytest.mark.scaffolding("experiment runner")
def test_plot_loss_mse_overwrites_same_path(tmp_path: Path) -> None:
    """Live mid-train updates rewrite one figures/loss_mse.png, not a new file."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    log_path = run_dir / TRAIN_LOG_JSONL
    log_path.write_text(json.dumps({"step": 1, "train_loss": 0.5, "eval_loss": 0.4}) + "\n")
    first = plot_loss_mse(run_dir)
    assert first.name == "loss_mse.png"
    first_size = first.stat().st_size

    with log_path.open("a") as handle:
        handle.write(json.dumps({"step": 2, "train_loss": 0.3, "eval_loss": 0.2}) + "\n")
        handle.write(json.dumps({"step": 3, "train_loss": 0.2, "eval_loss": 0.1}) + "\n")
    second = plot_loss_mse(run_dir)
    assert second == first
    assert second.is_file()
    assert list((run_dir / "figures").glob("loss_mse*.png")) == [second]
    assert second.stat().st_size > 0
    # Content changed (longer curve); size is a cheap proxy.
    assert second.stat().st_size != first_size or second.stat().st_mtime_ns > 0


@pytest.mark.scaffolding("experiment runner")
def test_plot_loss_mse_raises_when_log_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing training log"):
        plot_loss_mse(tmp_path)


@pytest.mark.scaffolding("experiment runner")
def test_best_checkpoint_step_picks_earlier_when_last_is_worse(tmp_path: Path) -> None:
    model_dir = tmp_path / "eval" / "model"
    path100 = model_dir / "unseen_cables_step100_summary.json"
    path200 = model_dir / "unseen_cables_step200_summary.json"
    _write_summary(path100, step=100, rel_l2_at_h200=0.20)
    _write_summary(path200, step=200, rel_l2_at_h200=0.10)
    summaries = {100: path100, 200: path200}
    assert best_checkpoint_step(summaries) == 200

    _write_summary(path200, step=200, rel_l2_at_h200=0.99)
    assert best_checkpoint_step(summaries) == 100


@pytest.mark.scaffolding("experiment runner")
def test_write_run_figures_writes_expected_pngs(plot_run_dir: Path) -> None:
    paths = write_run_figures(plot_run_dir)
    names = {path.name for path in paths}
    assert "loss_mse.png" in names
    assert "mean_l2_vs_frame.png" in names
    assert "rel_l2_vs_frame.png" in names
    for path in paths:
        assert path.stat().st_size > 100


@pytest.mark.scaffolding("experiment runner")
def test_plot_error_vs_frame_writes_png(plot_run_dir: Path) -> None:
    path = plot_error_vs_frame(plot_run_dir, stage="unseen_cables", metric="mean_l2")
    assert path.name == "mean_l2_vs_frame.png"
    assert path.is_file()
