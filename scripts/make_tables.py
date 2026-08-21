#!/usr/bin/env python
"""Regenerate the paper's two tables from the per-run records under ``results/runs``.

Table 1 (aggregate): rel l2, link-length drift and self-intersection on the
unseen-cable test set, final checkpoint, for the local-on and local-off variant
blocks and the three baselines. Table 2 (phase): local-on rel l2 on floor-contact
frames against free-flight frames.

Cell statistic, per metric and arm: each seed is reduced to its mean over the
400 evaluation windows together with that seed's window standard deviation; the
reported triple is **mean of the three seed means ± mean of the three window
sds ± sd (n−1) of the three seed means**. Bold marks the lowest mean per block
(Table 1) or per column (Table 2).

Inputs, one per run: ``results/runs/<experiment>/<label>/eval/model/
unseen_test_step100000_summary.json`` (written by ``dlogps.harness.evaluate``).
An arm whose three seed records are absent is filled from
``results/tables/paper_values.csv`` and marked ``transcribed`` in the CSV, so the
rendered tables always show the paper's numbers and the ``source`` column says
which of them this checkout can recompute.

    python scripts/make_tables.py            # writes results/tables/table{1,2}_*.{csv,md,tex}
    python scripts/make_tables.py --check    # also fail if a recomputed cell differs from the paper
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
TABLES = ROOT / "results" / "tables"
STEP = 100_000
SEEDS = (0, 1, 2)

VARIANT_NAMES = {
    "A": "Unbiased",
    "B": "Euclidean",
    "C": "Chain",
    "D": "Mixed",
    "F": "Euclidean+Chain only",
}
BASELINE_NAMES = {"chain_only": "chain only", "mlp_pernode": "MLP", "lstm_global": "LSTM"}

# (block, arm) -> (experiment directory, label pattern)
ARMS: dict[tuple[str, str], tuple[str, str]] = {
    **{("local_on", v): ("local_on", f"var{v}_s{{seed}}") for v in VARIANT_NAMES},
    **{("local_off", v): ("local_off", f"var{v}_s{{seed}}") for v in VARIANT_NAMES},
    ("baselines", "chain_only"): ("chain_only", "chain_s{seed}"),
    ("baselines", "mlp_pernode"): ("structfree", "mlp_pernode_s{seed}"),
    ("baselines", "lstm_global"): ("structfree", "lstm_global_s{seed}"),
}

TABLE1_METRICS = ("rel_l2_h400", "link_length_mean_drift", "selfint_frame_fraction")
TABLE1_HEADERS = ("rel l2 (frac. of L)", "drift δ (frac. of ℓ0)", "self-int. v (frac. of frames)")
TABLE2_METRICS = ("rel_l2_floor_contact", "rel_l2_free_flight")
TABLE2_HEADERS = ("floor contact rel l2", "free flight rel l2")
DECIMALS = {
    "rel_l2_h400": 4,
    "link_length_mean_drift": 4,
    "selfint_frame_fraction": 3,
    "rel_l2_floor_contact": 4,
    "rel_l2_free_flight": 4,
}
# The paper prints the MLP drift row with two decimals.
DECIMALS_OVERRIDE = {("baselines", "mlp_pernode", "link_length_mean_drift"): 2}


@dataclass(frozen=True)
class Cell:
    mean: float
    window_sd: float
    seed_sd: float
    source: str  # "records" or "transcribed"
    n_windows: tuple[int, ...] = ()

    def fmt(self, decimals: int) -> str:
        return f"{self.mean:.{decimals}f} ± {self.window_sd:.{decimals}f} ± {self.seed_sd:.{decimals}f}"

    def tex(self, decimals: int) -> str:
        return f"{self.mean:.{decimals}f}\\pm{self.window_sd:.{decimals}f}\\pm{self.seed_sd:.{decimals}f}"


def summary_path(experiment: str, label: str) -> Path:
    return RUNS / experiment / label / "eval" / "model" / f"unseen_test_step{STEP}_summary.json"


def cell_from_records(experiment: str, pattern: str, metric: str) -> Cell | None:
    """The triple from three seed summaries, or None if any seed record is missing."""
    means, sds, counts = [], [], []
    for seed in SEEDS:
        path = summary_path(experiment, pattern.format(seed=seed))
        if not path.is_file():
            return None
        payload = json.loads(path.read_text())
        entry = payload["metrics"].get(metric)
        if entry is None or entry.get("mean") is None or math.isnan(entry["mean"]):
            return None
        means.append(float(entry["mean"]))
        sds.append(float(entry["std"]))
        counts.append(int(entry["count"]))
    return Cell(
        mean=statistics.fmean(means),
        window_sd=statistics.fmean(sds),
        seed_sd=statistics.stdev(means),
        source="records",
        n_windows=tuple(counts),
    )


def load_paper_values() -> dict[tuple[str, str, str, str], Cell]:
    values: dict[tuple[str, str, str, str], Cell] = {}
    with (TABLES / "paper_values.csv").open() as fh:
        for row in csv.DictReader(fh):
            key = (row["table"], row["block"], row["arm"], row["metric"])
            values[key] = Cell(
                float(row["mean"]), float(row["window_sd"]), float(row["seed_sd"]), "transcribed"
            )
    return values


def decimals_for(block: str, arm: str, metric: str) -> int:
    return DECIMALS_OVERRIDE.get((block, arm, metric), DECIMALS[metric])


def build(
    table: str, blocks: tuple[str, ...], metrics: tuple[str, ...], paper: dict
) -> dict[tuple[str, str, str], Cell]:
    cells: dict[tuple[str, str, str], Cell] = {}
    for (block, arm), (experiment, pattern) in ARMS.items():
        if block not in blocks:
            continue
        for metric in metrics:
            cell = cell_from_records(experiment, pattern, metric)
            if cell is None:
                cell = paper.get((table, block, arm, metric))
            if cell is not None:
                cells[(block, arm, metric)] = cell
    return cells


def arm_name(block: str, arm: str) -> str:
    return BASELINE_NAMES.get(arm, VARIANT_NAMES.get(arm, arm))


def lowest(cells: dict, block: str, metric: str) -> str | None:
    candidates = [(c.mean, arm) for (b, arm, m), c in cells.items() if b == block and m == metric]
    return min(candidates)[1] if candidates else None


def write_csv(path: Path, cells: dict, table: str) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "table",
                "block",
                "arm",
                "name",
                "metric",
                "mean",
                "window_sd",
                "seed_sd",
                "source",
                "n_windows",
            ]
        )
        for (block, arm, metric), c in cells.items():
            d = decimals_for(block, arm, metric)
            writer.writerow(
                [
                    table,
                    block,
                    arm,
                    arm_name(block, arm),
                    metric,
                    f"{c.mean:.{d}f}",
                    f"{c.window_sd:.{d}f}",
                    f"{c.seed_sd:.{d}f}",
                    c.source,
                    "|".join(map(str, c.n_windows)),
                ]
            )


def render_markdown(
    cells: dict,
    blocks: tuple[str, ...],
    metrics: tuple[str, ...],
    headers: tuple[str, ...],
    *,
    bold_per: str,
) -> str:
    lines = ["| | " + " | ".join(headers) + " |", "|---|" + "---|" * len(headers)]
    block_titles = {
        "local_on": "local stream on",
        "local_off": "local stream off",
        "baselines": "baselines",
    }
    for block in blocks:
        arms = [
            arm for (b, arm, _m) in dict.fromkeys((b, a, None) for (b, a, _) in cells) if b == block
        ]
        if len(blocks) > 1:
            lines.append(f"| *{block_titles[block]}* |" + " |" * len(headers))
        for arm in arms:
            row = [arm_name(block, arm)]
            for metric in metrics:
                c = cells.get((block, arm, metric))
                if c is None:
                    row.append("—")
                    continue
                text = c.fmt(decimals_for(block, arm, metric))
                best = (
                    lowest(cells, block, metric)
                    if bold_per == "block"
                    else lowest(cells, block, metric)
                )
                if best == arm and block != "baselines":
                    text = f"**{text}**"
                row.append(text)
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def render_tex(cells: dict, blocks: tuple[str, ...], metrics: tuple[str, ...]) -> str:
    lines = []
    block_titles = {
        "local_on": "local stream on",
        "local_off": "local stream off",
        "baselines": "baselines",
    }
    for block in blocks:
        arms = [
            arm for (b, arm, _m) in dict.fromkeys((b, a, None) for (b, a, _) in cells) if b == block
        ]
        if len(blocks) > 1:
            lines.append(
                f"\\multicolumn{{{len(metrics) + 1}}}{{@{{}}l}}{{\\emph{{{block_titles[block]}}}}} \\\\"
            )
        for arm in arms:
            parts = [arm_name(block, arm)]
            for metric in metrics:
                c = cells.get((block, arm, metric))
                if c is None:
                    parts.append("--")
                    continue
                text = c.tex(decimals_for(block, arm, metric))
                if lowest(cells, block, metric) == arm and block != "baselines":
                    text = f"\\mathbf{{{text}}}"
                parts.append(f"${text}$")
            lines.append(" & ".join(parts) + " \\\\")
        if len(blocks) > 1 and block != blocks[-1]:
            lines.append("\\midrule")
    return "\n".join(lines) + "\n"


def check_against_paper(cells: dict, table: str, paper: dict) -> list[str]:
    """Every cell recomputed from records must equal the paper at its printed precision."""
    problems = []
    for (block, arm, metric), c in cells.items():
        if c.source != "records":
            continue
        ref = paper.get((table, block, arm, metric))
        if ref is None:
            problems.append(f"table {table} {block}/{arm}/{metric}: no paper value to compare")
            continue
        d = decimals_for(block, arm, metric)
        for name, got, want in (
            ("mean", c.mean, ref.mean),
            ("window_sd", c.window_sd, ref.window_sd),
            ("seed_sd", c.seed_sd, ref.seed_sd),
        ):
            if round(got, d) != round(want, d):
                problems.append(
                    f"table {table} {block}/{arm}/{metric} {name}: records give {got:.{d}f}, paper prints {want:.{d}f}"
                )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--check", action="store_true", help="fail if a recomputed cell differs from the paper"
    )
    args = parser.parse_args(argv)

    paper = load_paper_values()
    TABLES.mkdir(parents=True, exist_ok=True)

    table1 = build("1", ("local_on", "local_off", "baselines"), TABLE1_METRICS, paper)
    table2 = build("2", ("local_on",), TABLE2_METRICS, paper)

    write_csv(TABLES / "table1_aggregate.csv", table1, "1")
    write_csv(TABLES / "table2_phase.csv", table2, "2")
    (TABLES / "table1_aggregate.md").write_text(
        "Unseen-cable test, final checkpoint, 400 windows of 400-frame rollouts. "
        "Cell: mean of seed means ± mean of window sds ± sd of seed means; bold = lowest mean in the block.\n\n"
        + render_markdown(
            table1,
            ("local_on", "local_off", "baselines"),
            TABLE1_METRICS,
            TABLE1_HEADERS,
            bold_per="block",
        )
    )
    (TABLES / "table2_phase.md").write_text(
        "Local-on rows, unseen-cable test, final checkpoint. Floor contact: the recorded cable has a vertex at or "
        "below one diameter after the input frame; free flight otherwise. Cell format as Table 1; bold = lowest mean in the column.\n\n"
        + render_markdown(table2, ("local_on",), TABLE2_METRICS, TABLE2_HEADERS, bold_per="column")
    )
    (TABLES / "table1_aggregate.tex").write_text(
        render_tex(table1, ("local_on", "local_off", "baselines"), TABLE1_METRICS)
    )
    (TABLES / "table2_phase.tex").write_text(render_tex(table2, ("local_on",), TABLE2_METRICS))

    recomputed = sum(c.source == "records" for c in (*table1.values(), *table2.values()))
    total = len(table1) + len(table2)
    print(
        f"{recomputed}/{total} cells recomputed from records; the rest transcribed from the paper"
    )
    problems = check_against_paper(table1, "1", paper) + check_against_paper(table2, "2", paper)
    for line in problems:
        print("MISMATCH:", line)
    if problems:
        print(f"{len(problems)} recomputed cell(s) differ from the paper")
        return 1 if args.check else 0
    if recomputed:
        print("every recomputed cell matches the paper at its printed precision")
    return 0


if __name__ == "__main__":
    sys.exit(main())
