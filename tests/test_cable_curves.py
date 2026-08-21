"""Cable-wise frame curve aggregation for eval artifacts."""

from __future__ import annotations

import math

import pytest

from dlogps.harness.artifacts import cable_mean_std_curves, std_across_cables_column
from dlogps.harness.evaluate import WindowKey, WindowScore


def _window(
    cable_id: int,
    *,
    start: int,
    mean_l2: tuple[float, ...],
    rel_l2: tuple[float, ...] | None = None,
) -> WindowScore:
    return WindowScore(
        WindowKey(cable_id, 0, start),
        scores={"mean_l2_at_h5": mean_l2[-1]},
        frame_errors={
            "mean_l2": mean_l2,
            "rel_l2": rel_l2 if rel_l2 is not None else tuple(v / 2.0 for v in mean_l2),
        },
    )


@pytest.mark.scaffolding("experiment runner")
def test_cable_mean_std_curves_averages_within_cable_then_across_cables() -> None:
    """Per-cable window means, then mean and ddof=1 std across cables."""
    windows = (
        _window(0, start=4, mean_l2=(0.0, 1.0)),
        _window(0, start=10, mean_l2=(2.0, 3.0)),
        _window(1, start=4, mean_l2=(4.0, 5.0)),
    )
    mean_across, std_across = cable_mean_std_curves(windows, "mean_l2")
    assert mean_across == pytest.approx((2.5, 3.5))
    assert std_across[0] == pytest.approx(math.sqrt(((1.0 - 2.5) ** 2 + (4.0 - 2.5) ** 2) / 1))
    assert std_across[1] == pytest.approx(math.sqrt(((2.0 - 3.5) ** 2 + (5.0 - 3.5) ** 2) / 1))


@pytest.mark.scaffolding("experiment runner")
def test_std_across_cables_differs_from_std_across_windows() -> None:
    """Unbalanced window counts must not be treated as extra cables."""
    # Cable 0 has two windows; cable 1 has one. Std across windows on the pooled
    # series would weight cable 0 twice; std across cables uses one mean per cable.
    windows = (
        _window(0, start=4, mean_l2=(0.0, 0.0)),
        _window(0, start=10, mean_l2=(2.0, 2.0)),
        _window(1, start=4, mean_l2=(10.0, 10.0)),
    )
    _, std_across_cables = cable_mean_std_curves(windows, "mean_l2")
    pooled = [0.0, 0.0, 2.0, 2.0, 10.0, 10.0]
    mean_pooled = sum(pooled) / len(pooled)
    var_windows = sum((value - mean_pooled) ** 2 for value in pooled) / (len(pooled) - 1)
    std_windows = math.sqrt(var_windows)
    assert std_across_cables[0] == pytest.approx(6.363961030678928)
    assert std_across_cables[0] != pytest.approx(std_windows)


@pytest.mark.scaffolding("experiment runner")
def test_cable_mean_std_curves_empty_when_metric_missing() -> None:
    windows = (_window(0, start=4, mean_l2=(1.0, 2.0)),)
    assert cable_mean_std_curves(windows, "rmse") == ((), ())


@pytest.mark.scaffolding("experiment runner")
def test_std_column_name_carries_unit() -> None:
    assert std_across_cables_column("mean_l2") == "mean_l2_std_across_cables [m]"
