"""Write the standard figures for a RunLog-shaped run directory.

Figures are produced from ``train_log.jsonl`` and the eval curve artifacts under
``eval/model/``. Every path that finishes a run
(``run_experiment``, ``evaluate --run-dir``) calls the same entry points here.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dlogps.harness.artifacts import (
    eval_artifact_stem,
    metric_column,
    std_across_cables_column,
)
from dlogps.harness.runlog import TRAIN_LOG_JSONL

BEST_BY = "rel_l2_at_h200"
"""Primary checkpoint selection metric: terminal relative L2 at frame 200."""

_CURVE_SUFFIX = "_curve.csv"
_SUMMARY_SUFFIX = "_summary.json"


def best_checkpoint_step(summaries: dict[int, Path], metric: str = BEST_BY) -> int:
    """Pick the checkpoint with the lowest aggregate ``metric``.

    Args:
        summaries: Optimizer step to ``*_summary.json`` path for one stage.
        metric: A scalar name present in each summary's ``metrics`` block.

    Returns:
        The winning optimizer step.

    Raises:
        ValueError: if no summary carries a finite value for ``metric``.
    """
    best_step: int | None = None
    best_value = math.inf
    for step, path in summaries.items():
        document = json.loads(path.read_text())
        raw = document["metrics"].get(metric, {}).get("mean")
        if raw is None:
            continue
        value = float(raw)
        if math.isnan(value):
            continue
        if value < best_value:
            best_value = value
            best_step = step
    if best_step is None:
        raise ValueError(f"no checkpoint has a finite {metric!r} among steps {sorted(summaries)}")
    return best_step


def plot_loss_mse(run_dir: Path) -> Path:
    """Plot normalized train and eval MSE vs optimizer step.

    Args:
        run_dir: A run directory holding ``train_log.jsonl``.

    Returns:
        Path to ``figures/loss_mse.png``.

    Raises:
        FileNotFoundError: if the training log is missing.
        ValueError: if the log exists but contains no records.
    """
    log_path = run_dir / TRAIN_LOG_JSONL
    if not log_path.is_file():
        raise FileNotFoundError(f"missing training log: {log_path}")

    steps: list[int] = []
    train_losses: list[float] = []
    eval_losses: list[float] = []
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        steps.append(int(record["step"]))
        train_losses.append(float(record["train_loss"]))
        eval_losses.append(float(record["eval_loss"]))

    if not steps:
        raise ValueError(f"training log is empty: {log_path}")

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output = figures_dir / "loss_mse.png"

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(steps, train_losses, label="train MSE", linewidth=1.5)
    axis.plot(steps, eval_losses, label="eval MSE", linewidth=1.5)
    axis.set_xlabel("optimizer step")
    axis.set_ylabel("normalized MSE (1.0 = constant predictor)")
    axis.set_title("Training and held-out one-step MSE")
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def plot_error_vs_frame(
    run_dir: Path,
    *,
    stage: str,
    metric: str,
    output: Path | None = None,
) -> Path:
    """Plot per-checkpoint error curves for one stage and reduction.

    The best checkpoint (lowest :data:`BEST_BY` on ``stage``) receives a shaded
    band from the std-across-cables column.

    Args:
        run_dir: A run directory with ``eval/model/`` curve artifacts.
        stage: ``seen_cables`` or ``unseen_cables``.
        metric: ``mean_l2`` or ``rel_l2``.

    Returns:
        Path to the PNG written under ``figures/``.

    Raises:
        ValueError: on an unsupported ``metric`` or when no model curves exist.
    """
    if metric not in {"mean_l2", "rel_l2"}:
        raise ValueError(f"metric must be mean_l2 or rel_l2, got {metric!r}")

    model_dir = run_dir / "eval" / "model"
    summaries = _summaries_for_stage(model_dir, stage)
    if not summaries:
        raise ValueError(f"no model summaries for stage {stage!r} under {model_dir}")

    best_step = best_checkpoint_step(summaries)
    mean_col = metric_column(metric)
    std_col = std_across_cables_column(metric)

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    destination = output or (figures_dir / f"{metric}_vs_frame.png")
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 5))
    for step in sorted(summaries):
        curve_path = model_dir / f"{eval_artifact_stem(stage, step)}{_CURVE_SUFFIX}"
        frames, means, stds = _read_curve(curve_path, mean_col, std_col)
        label = f"step {step}"
        if step == best_step:
            label += " (best)"
        line = axis.plot(frames, means, label=label, linewidth=1.5)[0]
        if step == best_step and stds is not None:
            # ±std can put the lower edge below 0 even when std>0; error metrics
            # are non-negative, so clip the band for display only.
            lower = [max(0.0, mean - std) for mean, std in zip(means, stds, strict=True)]
            upper = [mean + std for mean, std in zip(means, stds, strict=True)]
            axis.fill_between(
                frames,
                lower,
                upper,
                color=line.get_color(),
                alpha=0.2,
                label=f"± std across cables (step {step})",
            )

    unit = mean_col.split("[", 1)[1].rstrip("]")
    axis.set_xlabel("predicted frame")
    axis.set_ylabel(f"{metric} [{unit}]")
    axis.set_title(f"{metric} vs frame ({stage})")
    axis.legend(fontsize=8)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


def write_run_figures(run_dir: Path) -> list[Path]:
    """Write the standard figure set for one run directory.

    Args:
        run_dir: A RunLog-shaped directory.

    Returns:
        Paths to every PNG written. Loss is skipped when ``train_log.jsonl`` is
        absent; error-vs-frame plots require model eval artifacts.
    """
    written: list[Path] = []
    log_path = run_dir / TRAIN_LOG_JSONL
    if log_path.is_file():
        written.append(plot_loss_mse(run_dir))

    stages = _stages_with_model_curves(run_dir / "eval" / "model")
    if not stages:
        return written

    suffix_stage = len(stages) > 1
    for stage in stages:
        for metric in ("mean_l2", "rel_l2"):
            name = f"{metric}_vs_frame_{stage}.png" if suffix_stage else f"{metric}_vs_frame.png"
            written.append(
                plot_error_vs_frame(
                    run_dir,
                    stage=stage,
                    metric=metric,
                    output=run_dir / "figures" / name,
                )
            )
    return written


def _summaries_for_stage(eval_dir: Path, stage: str) -> dict[int, Path]:
    pattern = re.compile(rf"^{re.escape(stage)}_step(\d+){re.escape(_SUMMARY_SUFFIX)}$")
    summaries: dict[int, Path] = {}
    if not eval_dir.is_dir():
        return summaries
    for path in eval_dir.iterdir():
        match = pattern.match(path.name)
        if match is not None:
            summaries[int(match.group(1))] = path
    return summaries


def _stages_with_model_curves(model_dir: Path) -> tuple[str, ...]:
    if not model_dir.is_dir():
        return ()
    stages: set[str] = set()
    pattern = re.compile(r"^(seen_cables|unseen_cables)_step\d+_curve\.csv$")
    for path in model_dir.iterdir():
        if pattern.match(path.name):
            stages.add(path.name.split("_step", 1)[0])
    return tuple(sorted(stages))


def _read_curve(
    path: Path,
    mean_col: str,
    std_col: str | None,
) -> tuple[list[int], list[float], list[float] | None]:
    if not path.is_file():
        raise FileNotFoundError(f"missing curve artifact: {path}")
    frames: list[int] = []
    means: list[float] = []
    stds: list[float] | None = [] if std_col is not None else None
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        has_std = std_col is not None and std_col in fieldnames
        if std_col is not None and not has_std:
            stds = None
        for row in reader:
            frames.append(int(row["frame"]))
            means.append(float(row[mean_col]))
            if stds is not None and std_col is not None:
                raw = row.get(std_col, "")
                stds.append(float(raw) if raw else math.nan)
    return frames, means, stds
