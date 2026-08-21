"""Write the four evaluation artifacts a run commits under ``eval/``.

Layout and column contract: ``docs/reproduction.md``. Scoring lives in
:mod:`dlogps.harness.evaluate`; this module is the only writer of the CSV and
JSON those scores land in.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dlogps.data.dataset import CableData
from dlogps.data.loader import SplitCfg
from dlogps.harness.evaluate import EvalResult, WindowScore, stage_cables
from dlogps.harness.metrics import RegimeCfg, unit_of

KEY_COLUMNS = ("cable_id", "episode", "start_frame")
CABLE_COLUMNS = ("length", "diameter", "bend_stiffness")


@dataclass(frozen=True)
class EvalArtifactPaths:
    """The four files :func:`write_eval_artifacts` creates."""

    summary: Path
    windows: Path
    per_cable: Path
    curve: Path


def eval_artifact_stem(stage: str, step: int) -> str:
    """Filename prefix, e.g. ``unseen_cables_step40000``."""
    return f"{stage}_step{step}"


def metric_column(name: str) -> str:
    """CSV header for one metric, with its unit attached."""
    return f"{name} [{unit_of(name)}]"


def across_cables_column(name: str) -> str:
    """CSV header for the per-frame mean across held-out cables."""
    return f"{name}_across_cables [{unit_of(name)}]"


def std_across_cables_column(name: str) -> str:
    """CSV header for the per-frame sample std across held-out cables."""
    return f"{name}_std_across_cables [{unit_of(name)}]"


CABLE_CURVE_METRICS = ("mean_l2", "rel_l2")
"""Reductions that carry per-window frame curves for cable-wise aggregation."""


def cable_mean_std_curves(
    windows: tuple[WindowScore, ...] | list[WindowScore],
    metric: str,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Per-frame mean and sample std across cables.

    Within each cable, frame curves are averaged over windows. Across cables,
    the per-cable means yield a mean and a sample standard deviation (``ddof=1``).

    Args:
        windows: Scored windows with optional :attr:`WindowScore.frame_errors`.
        metric: A reduction name, e.g. ``mean_l2`` or ``rel_l2``.

    Returns:
        ``(mean_across_cables, std_across_cables)``. Empty tuples when no window
        carries ``metric`` in ``frame_errors``. Std entries are NaN when fewer
        than two cables contribute.
    """
    grouped: dict[int, list[tuple[float, ...]]] = {}
    for window in windows:
        curve = window.frame_errors.get(metric)
        if curve is None:
            continue
        grouped.setdefault(window.key.cable_id, []).append(curve)

    if not grouped:
        return (), ()

    cable_means: list[tuple[float, ...]] = []
    for curves in grouped.values():
        length = len(curves[0])
        averaged = tuple(
            sum(curve[index] for curve in curves) / len(curves) for index in range(length)
        )
        cable_means.append(averaged)

    length = len(cable_means[0])
    mean_across = tuple(
        sum(curve[index] for curve in cable_means) / len(cable_means) for index in range(length)
    )
    if len(cable_means) < 2:
        std_across = tuple(math.nan for _ in range(length))
    else:
        std_across = tuple(
            math.sqrt(
                sum((curve[index] - mean_across[index]) ** 2 for curve in cable_means)
                / (len(cable_means) - 1)
            )
            for index in range(length)
        )
    return mean_across, std_across


def write_eval_artifacts(
    eval_dir: Path,
    result: EvalResult,
    *,
    step: int,
    horizon: int,
    regime: RegimeCfg,
    cables: list[CableData],
    split: SplitCfg,
    frame_dt: float,
) -> EvalArtifactPaths:
    """Write ``summary.json``, ``windows.csv``, ``per_cable.csv`` and ``curve.csv``.

    Args:
        eval_dir: Run's ``eval/`` directory.
        result: One stage's scorecard from :func:`dlogps.harness.evaluate.evaluate`.
        step: Checkpoint optimizer step that produced the rollout.
        horizon: Rollout length ``R`` in frames, input frame included.
        regime: Horizons recorded in the summary for provenance.
        cables: The dataset the stage was scored on.
        split: The split training used.
        frame_dt: Simulation frame interval [s] for ``time_s`` in the curve file.

    Returns:
        Paths to the four files written.

    Raises:
        ValueError: if there are no per-window rows to write, or ``frame_dt`` is
            not positive.
    """
    if not result.windows:
        raise ValueError(
            "cannot write eval artifacts without per-window scores; "
            "evaluate must retain WindowScore rows keyed by (cable_id, episode, start)"
        )
    if frame_dt <= 0.0:
        raise ValueError(f"frame_dt is a positive duration in seconds, got {frame_dt}")
    if horizon < 1:
        raise ValueError(f"horizon is a frame count and must be at least 1, got {horizon}")

    eval_dir.mkdir(parents=True, exist_ok=True)
    stem = eval_artifact_stem(result.stage, step)
    paths = EvalArtifactPaths(
        summary=eval_dir / f"{stem}_summary.json",
        windows=eval_dir / f"{stem}_windows.csv",
        per_cable=eval_dir / f"{stem}_per_cable.csv",
        curve=eval_dir / f"{stem}_curve.csv",
    )

    _write_summary(
        paths.summary, result, step=step, horizon=horizon, regime=regime, frame_dt=frame_dt
    )
    _write_windows(paths.windows, result)
    _write_per_cable(paths.per_cable, result, cables=cables, split=split)
    _write_curve(paths.curve, result, frame_dt=frame_dt)
    return paths


def window_key_rows(path: Path) -> list[tuple[int, int, int]]:
    """Read ``(cable_id, episode, start_frame)`` from a windows artifact."""
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[tuple[int, int, int]] = []
        for raw in reader:
            rows.append(
                (
                    int(raw["cable_id"]),
                    int(raw["episode"]),
                    int(raw["start_frame"]),
                )
            )
        return rows


def _metric_names(result: EvalResult) -> tuple[str, ...]:
    names = {name for window in result.windows for name in window.scores}
    return tuple(sorted(names))


def _write_summary(
    path: Path,
    result: EvalResult,
    *,
    step: int,
    horizon: int,
    regime: RegimeCfg,
    frame_dt: float,
) -> None:
    metrics: dict[str, dict[str, Any]] = {}
    for name, mean in result.scores.items():
        metrics[name] = {
            "mean": _json_number(mean),
            "std": _json_number(result.stds.get(name, math.nan)),
            "count": result.counts.get(name, 0),
            "unit": unit_of(name),
        }
    document = {
        "stage": result.stage,
        "checkpoint_step": step,
        "horizon": horizon,
        "limit": result.limit,
        "n_windows": result.n_windows,
        "regime_horizons": list(regime.horizons),
        "frame_dt_s": frame_dt,
        "metrics": metrics,
    }
    path.write_text(json.dumps(document, indent=2) + "\n")


def _write_windows(path: Path, result: EvalResult) -> None:
    metric_names = _metric_names(result)
    fieldnames = list(KEY_COLUMNS) + [metric_column(name) for name in metric_names]
    rows: list[dict[str, object]] = []
    for window in result.windows:
        row: dict[str, object] = {
            "cable_id": window.key.cable_id,
            "episode": window.key.episode,
            "start_frame": window.key.start,
        }
        for name in metric_names:
            row[metric_column(name)] = window.scores.get(name, math.nan)
        rows.append(row)
    _write_csv(path, fieldnames, rows)


def _write_per_cable(
    path: Path,
    result: EvalResult,
    *,
    cables: list[CableData],
    split: SplitCfg,
) -> None:
    by_id = {cable.cable_id: cable for cable in cables}
    scored_ids = {window.key.cable_id for window in result.windows}
    stage_ids = stage_cables(cables, result.stage, split)
    extra = sorted(scored_ids - stage_ids)
    if extra:
        raise ValueError(f"per-cable table has cables not on stage {result.stage!r}: {extra}")

    metric_names = _metric_names(result)
    fieldnames = (
        list(KEY_COLUMNS[:1]) + list(CABLE_COLUMNS) + [metric_column(name) for name in metric_names]
    )

    grouped: dict[int, list[WindowScore]] = {}
    for window in result.windows:
        grouped.setdefault(window.key.cable_id, []).append(window)

    rows: list[dict[str, object]] = []
    for cable_id in sorted(grouped):
        cable = by_id[cable_id]
        row: dict[str, object] = {
            "cable_id": cable_id,
            "length": cable.params["length"],
            "diameter": cable.params["diameter"],
            "bend_stiffness": cable.params["bend_stiffness"],
        }
        windows = grouped[cable_id]
        for name in metric_names:
            values = [window.scores[name] for window in windows if name in window.scores]
            finite = [value for value in values if not math.isnan(value)]
            row[metric_column(name)] = sum(finite) / len(finite) if finite else math.nan
        rows.append(row)
    _write_csv(path, fieldnames, rows)


def _write_curve(path: Path, result: EvalResult, *, frame_dt: float) -> None:
    curve_names = tuple(sorted(result.curve))
    cable_curves: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {}
    extra_columns: list[str] = []
    for name in CABLE_CURVE_METRICS:
        if name not in curve_names:
            continue
        mean_across, std_across = cable_mean_std_curves(result.windows, name)
        if not mean_across:
            continue
        cable_curves[name] = (mean_across, std_across)
        extra_columns.extend([across_cables_column(name), std_across_cables_column(name)])

    fieldnames = ["frame", "time_s"] + [metric_column(name) for name in curve_names] + extra_columns
    rows: list[dict[str, object]] = []
    if curve_names:
        length = len(result.curve[curve_names[0]])
        for index in range(length):
            frame = index + 1
            row: dict[str, object] = {
                "frame": frame,
                "time_s": frame * frame_dt,
            }
            for name in curve_names:
                row[metric_column(name)] = result.curve[name][index]
            for name, (mean_across, std_across) in cable_curves.items():
                row[across_cables_column(name)] = mean_across[index]
                row[std_across_cables_column(name)] = std_across[index]
            rows.append(row)
    _write_csv(path, fieldnames, rows)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row[key]) for key in fieldnames})


def _csv_cell(value: object) -> str:
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _json_number(value: float) -> float | None:
    return None if math.isnan(value) else value
