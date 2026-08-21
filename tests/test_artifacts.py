"""The four eval artifacts: keys, coverage, curve agreement and units.

Pins the failures that stay finite: a window table that cannot be keyed for
pairing, a per-cable table that hides which cables failed, a curve whose terminal
point disagrees with the summary, and metric columns with no unit in the header.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from dlogps.harness.artifacts import (
    metric_column,
    window_key_rows,
    write_eval_artifacts,
)
from dlogps.harness.evaluate import EvalResult, WindowKey, WindowScore, evaluate, stage_cables
from dlogps.harness.metrics import RegimeCfg
from tests.test_evaluate import (
    DT,
    HORIZON,
    REGIME,
    SPLIT,
    C,
    _cables,
    _Drift,
    _stats,
)

torch.set_num_threads(1)


def _write_fixture(
    tmp_path: Path,
    *,
    stage: str = "seen_cables",
    horizon: int = HORIZON,
    regime: RegimeCfg = REGIME,
) -> tuple[EvalResult, Path]:
    cables = _cables()
    result = evaluate(
        _Drift(),
        cables,
        stage=stage,
        horizon=horizon,
        stats=_stats(cables),
        regime=regime,
        split=SPLIT,
        history_frames=C,
    )
    eval_dir = tmp_path / "eval"
    write_eval_artifacts(
        eval_dir,
        result,
        step=40000,
        horizon=horizon,
        regime=regime,
        cables=cables,
        split=SPLIT,
        frame_dt=DT,
    )
    return result, eval_dir


def _synthetic(
    *,
    stage: str = "seen_cables",
    horizon: int = 6,
    curve_mean_l2: tuple[float, ...] = (0.01, 0.02, 0.03, 0.04, 0.05),
    windows: tuple[WindowScore, ...],
) -> EvalResult:
    terminal = curve_mean_l2[-1]
    h = horizon - 1
    return EvalResult(
        stage=stage,
        n_windows=len(windows),
        scores={
            f"mean_l2_at_h{h}": terminal,
            "rmse_full": 0.1,
        },
        counts={"mean_l2_at_h{h}": len(windows), "rmse_full": len(windows)},
        stds={"mean_l2_at_h{h}": 0.0, "rmse_full": 0.0},
        windows=windows,
        curve={
            "mean_l2": curve_mean_l2,
            "rmse": tuple(v * 1.1 for v in curve_mean_l2),
            "rel_l2": tuple(v / 1.0 for v in curve_mean_l2),
        },
    )


@pytest.mark.scaffolding("run layout")
def test_windows_csv_has_one_row_per_window_with_unique_keys(tmp_path: Path) -> None:
    """Verification item 4: one row per scored window, unique key triples."""
    result, eval_dir = _write_fixture(tmp_path)

    windows_path = next(eval_dir.glob("*_windows.csv"))
    with windows_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == result.n_windows
    keys = window_key_rows(windows_path)
    assert len(keys) == len(set(keys))
    assert keys == [(w.key.cable_id, w.key.episode, w.key.start) for w in result.windows]


@pytest.mark.scaffolding("run layout")
def test_two_arms_on_the_same_stage_share_window_key_order(tmp_path: Path) -> None:
    """Verification item 5: pairable window tables share keys in the same order."""
    # Cables 0 and 1 are the unseen side of _cables() at half held out; cable 2
    # is on the seen side, and the writer refuses a per-cable table that covers a
    # cable the stage does not.
    keys = [
        WindowKey(0, 0, 4),
        WindowKey(0, 0, 10),
        WindowKey(1, 0, 4),
    ]
    metric = "rel_l2_at_h200"
    first = EvalResult(
        stage="unseen_cables",
        n_windows=3,
        scores={},
        windows=tuple(WindowScore(key, {metric: 1.0}) for key in keys),
    )
    second = EvalResult(
        stage="unseen_cables",
        n_windows=3,
        scores={},
        windows=tuple(WindowScore(key, {metric: 2.0}) for key in keys),
    )

    # One eval directory per arm, as the layout gives each run its own: a shared
    # one has both arms writing the same stage and step, so the second silently
    # overwrites the first and there is nothing left to pair.
    regime = RegimeCfg(horizons=(200,))
    cables = _cables()
    dirs = [tmp_path / "arm_a" / "eval", tmp_path / "arm_b" / "eval"]
    for eval_dir, result in zip(dirs, (first, second), strict=True):
        write_eval_artifacts(
            eval_dir,
            result,
            step=40000,
            horizon=201,
            regime=regime,
            cables=cables,
            split=SPLIT,
            frame_dt=DT,
        )

    tables = [next(d.glob("*_windows.csv")) for d in dirs]
    assert window_key_rows(tables[0]) == window_key_rows(tables[1])


@pytest.mark.scaffolding("run layout")
def test_per_cable_csv_covers_every_scored_cable_and_no_other(tmp_path: Path) -> None:
    """Verification item 6: per-cable rows match the stage's cable set."""
    cables = _cables()
    result, eval_dir = _write_fixture(tmp_path, stage="seen_cables")
    expected = stage_cables(cables, "seen_cables", SPLIT)

    per_cable_path = next(eval_dir.glob("*_per_cable.csv"))
    with per_cable_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    got = {int(row["cable_id"]) for row in rows}
    assert got == expected
    assert len(rows) == len(expected)
    for row in rows:
        cable = next(c for c in cables if c.cable_id == int(row["cable_id"]))
        assert float(row["length"]) == pytest.approx(cable.params["length"])
        assert float(row["diameter"]) == pytest.approx(cable.params["diameter"])
        assert float(row["bend_stiffness"]) == pytest.approx(cable.params["bend_stiffness"])


@pytest.mark.scaffolding("run layout")
def test_curve_terminal_mean_l2_matches_summary(tmp_path: Path) -> None:
    """Verification item 7: last curve point equals mean_l2_at_h{H} in the summary."""
    # Cables 2 and 3 are the seen side of _cables(), which is the stage
    # _synthetic writes under; the writer checks the two agree.
    keys = tuple(WindowScore(WindowKey(i, 0, 4), {"rmse_full": 0.1}) for i in (2, 3))
    horizon = 6
    curve = (0.01, 0.02, 0.03, 0.04, 0.05)
    result = _synthetic(horizon=horizon, curve_mean_l2=curve, windows=keys)
    terminal_h = horizon - 1

    eval_dir = tmp_path / "eval"
    write_eval_artifacts(
        eval_dir,
        result,
        step=40000,
        horizon=horizon,
        regime=RegimeCfg(horizons=(terminal_h,)),
        cables=_cables(),
        split=SPLIT,
        frame_dt=DT,
    )

    summary = json.loads(next(eval_dir.glob("*_summary.json")).read_text())
    metric_name = f"mean_l2_at_h{terminal_h}"
    summary_value = summary["metrics"][metric_name]["mean"]

    with next(eval_dir.glob("*_curve.csv")).open(newline="") as handle:
        curve_rows = list(csv.DictReader(handle))

    assert len(curve_rows) == terminal_h
    last_mean_l2 = float(curve_rows[-1][metric_column("mean_l2")])
    assert last_mean_l2 == pytest.approx(summary_value, rel=1e-5)
    assert last_mean_l2 == pytest.approx(curve[-1], rel=1e-5)


@pytest.mark.scaffolding("run layout")
def test_every_metric_column_carries_its_unit_in_the_header(tmp_path: Path) -> None:
    """Verification item 9: metric headers name the unit."""
    _, eval_dir = _write_fixture(tmp_path)

    for path in eval_dir.glob("*.csv"):
        with path.open(newline="") as handle:
            header = csv.DictReader(handle).fieldnames or []
        for column in header:
            if column in {"cable_id", "episode", "start_frame", "frame", "time_s"}:
                continue
            if column in {"length", "diameter", "bend_stiffness"}:
                continue
            assert "[" in column and "]" in column, f"{path.name}: {column!r} has no unit"


@pytest.mark.contract
def test_metric_column_raises_on_a_unknown_metric_name() -> None:
    with pytest.raises(KeyError, match="no unit registered"):
        metric_column("not_a_metric")
