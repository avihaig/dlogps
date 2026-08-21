"""Cross-run summary table and curve overlay for an experiment directory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from dlogps.harness.artifacts import eval_artifact_stem, metric_column
from dlogps.harness.compare_runs import collect_arm_rows, compare_experiment
from dlogps.harness.plots import BEST_BY
from dlogps.harness.runlog import CONFIG_NAME


def _arm(
    experiment: Path,
    label: str,
    *,
    step: int,
    rel: float,
    d_model: int,
    n_layers: int,
    frames: tuple[int, ...] = (1, 2, 3),
    curve: tuple[float, ...] = (0.1, 0.2, 0.3),
) -> Path:
    run_dir = experiment / "runs" / f"2026-08-06T00-00-00_{label}"
    model_dir = run_dir / "eval" / "model"
    model_dir.mkdir(parents=True)
    (run_dir / CONFIG_NAME).write_text(
        yaml.safe_dump(
            {
                "label": label,
                "model": {"d_model": d_model, "n_layers": n_layers},
                "train": {"sigma": 0.01},
            }
        )
    )
    stem = eval_artifact_stem("unseen_cables", step)
    summary = {
        "stage": "unseen_cables",
        "checkpoint_step": step,
        "metrics": {BEST_BY: {"mean": rel, "std": 0.0, "count": 1, "unit": ""}},
    }
    (model_dir / f"{stem}_summary.json").write_text(json.dumps(summary) + "\n")
    col = metric_column("rel_l2")
    lines = ["frame,time_s," + col]
    for frame, value in zip(frames, curve, strict=True):
        lines.append(f"{frame},{frame * 0.002},{value}")
    (model_dir / f"{stem}_curve.csv").write_text("\n".join(lines) + "\n")
    return run_dir


def test_compare_experiment_writes_table_and_overlay(tmp_path: Path) -> None:
    experiment = tmp_path / "capacity_3x2"
    _arm(experiment, "w64_d4", step=40000, rel=0.20, d_model=64, n_layers=4, curve=(0.2, 0.3, 0.4))
    _arm(
        experiment,
        "w128_d4",
        step=40000,
        rel=0.10,
        d_model=128,
        n_layers=4,
        curve=(0.05, 0.08, 0.10),
    )

    csv_path, overlay = compare_experiment(experiment, stage="unseen_cables")
    assert csv_path.is_file()
    assert overlay is not None and overlay.is_file()
    rows = collect_arm_rows(experiment)
    assert [row.label for row in rows] == ["w128_d4", "w64_d4"]
    assert rows[0].primary == pytest.approx(0.10)


def test_compare_runs_skips_incomplete_arms(tmp_path: Path) -> None:
    experiment = tmp_path / "capacity_3x2"
    incomplete = experiment / "runs" / "2026-08-06T00-00-00_w160_d8"
    incomplete.mkdir(parents=True)
    (incomplete / CONFIG_NAME).write_text(yaml.safe_dump({"label": "w160_d8"}))
    _arm(experiment, "w64_d4", step=40000, rel=0.15, d_model=64, n_layers=4)
    rows = collect_arm_rows(experiment)
    assert [row.label for row in rows] == ["w64_d4"]
