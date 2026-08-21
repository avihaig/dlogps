"""``scripts/make_tables.py`` reproduces the paper's tables from the shipped records."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_tables.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("make_tables", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["make_tables"] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.claim(
    "every table cell recomputed from run records equals the paper at its printed precision"
)
def test_recomputed_cells_match_the_paper() -> None:
    mt = _load_module()
    paper = mt.load_paper_values()
    table1 = mt.build("1", ("local_on", "local_off", "baselines"), mt.TABLE1_METRICS, paper)
    table2 = mt.build("2", ("local_on",), mt.TABLE2_METRICS, paper)

    assert len(table1) == 13 * 3, "thirteen rows, three metrics"
    assert len(table2) == 5 * 2, "five rows, two phases"
    assert mt.check_against_paper(table1, "1", paper) == []
    assert mt.check_against_paper(table2, "2", paper) == []
    # The local-on block of Table 1 is recomputed from the shipped summaries.
    for arm in mt.VARIANT_NAMES:
        for metric in mt.TABLE1_METRICS:
            cell = table1[("local_on", arm, metric)]
            assert cell.source == "records", (arm, metric)
            assert cell.n_windows == (400, 400, 400)


@pytest.mark.contract
def test_the_committed_tables_are_what_the_script_writes(tmp_path: Path) -> None:
    """Regenerating into a scratch directory reproduces the committed CSVs byte for byte."""
    mt = _load_module()
    mt.TABLES = tmp_path
    (tmp_path / "paper_values.csv").write_bytes(
        (ROOT / "results/tables/paper_values.csv").read_bytes()
    )
    assert mt.main([]) == 0
    for name in (
        "table1_aggregate.csv",
        "table2_phase.csv",
        "table1_aggregate.md",
        "table2_phase.md",
    ):
        assert (tmp_path / name).read_text() == (ROOT / "results/tables" / name).read_text(), name


@pytest.mark.contract
def test_the_cell_statistic_is_mean_of_means_mean_of_sds_sd_of_means(tmp_path: Path) -> None:
    mt = _load_module()
    mt.RUNS = tmp_path
    import json

    for seed, (mean, sd) in enumerate(((0.10, 0.01), (0.20, 0.03), (0.30, 0.05))):
        path = (
            tmp_path
            / "local_on"
            / f"varA_s{seed}"
            / "eval"
            / "model"
            / "unseen_test_step100000_summary.json"
        )
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps({"metrics": {"rel_l2_h400": {"mean": mean, "std": sd, "count": 400}}})
        )

    cell = mt.cell_from_records("local_on", "varA_s{seed}", "rel_l2_h400")
    assert cell is not None
    assert cell.mean == pytest.approx(0.2)
    assert cell.window_sd == pytest.approx(0.03)
    assert cell.seed_sd == pytest.approx(0.1)
    assert cell.n_windows == (400, 400, 400)
    assert mt.cell_from_records("local_on", "varB_s{seed}", "rel_l2_h400") is None


def test_paper_values_cover_every_table_cell() -> None:
    with (ROOT / "results/tables/paper_values.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    keys = {(r["table"], r["block"], r["arm"], r["metric"]) for r in rows}
    assert len(keys) == len(rows) == 49
