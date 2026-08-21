"""Compare finished ``run_experiment`` dirs under one experiment root.

Writes a primary-metric table and an overlay of best-checkpoint rollout curves
so a capacity / sigma screen can be reviewed without opening every figure.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from dlogps.harness.artifacts import eval_artifact_stem, metric_column
from dlogps.harness.metrics import unit_of
from dlogps.harness.plots import BEST_BY, best_checkpoint_step
from dlogps.harness.runlog import CONFIG_NAME

_CURVE_SUFFIX = "_curve.csv"
_SUMMARY_SUFFIX = "_summary.json"


def _read_mean_curve(path: Path, mean_col: str) -> tuple[list[int], list[float]]:
    frames: list[int] = []
    means: list[float] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            frames.append(int(row["frame"]))
            means.append(float(row[mean_col]))
    return frames, means


@dataclass(frozen=True)
class ArmRow:
    """One finished run's primary score at its best (or fixed) checkpoint."""

    label: str
    run_dir: Path
    step: int
    primary: float
    d_model: int | None
    n_layers: int | None
    sigma: float | None


def _run_dirs(experiment_dir: Path) -> list[Path]:
    runs = experiment_dir / "runs"
    if not runs.is_dir():
        raise FileNotFoundError(f"no runs/ under {experiment_dir}")
    return sorted(path for path in runs.iterdir() if path.is_dir())


def _load_config(run_dir: Path) -> dict:
    path = run_dir / CONFIG_NAME
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def _label_of(run_dir: Path, cfg: dict) -> str:
    if cfg.get("label"):
        return str(cfg["label"])
    # Directory is ``<timestamp>_<label>``.
    name = run_dir.name
    if "_" in name:
        return name.split("_", 1)[1]
    return name


def _summaries(model_dir: Path, stage: str) -> dict[int, Path]:
    out: dict[int, Path] = {}
    if not model_dir.is_dir():
        return out
    prefix = f"{stage}_step"
    for path in model_dir.iterdir():
        if not path.name.startswith(prefix) or not path.name.endswith(_SUMMARY_SUFFIX):
            continue
        mid = path.name[len(prefix) : -len(_SUMMARY_SUFFIX)]
        if mid.isdigit():
            out[int(mid)] = path
    return out


def _primary_from_summary(path: Path, metric: str = BEST_BY) -> float:
    document = json.loads(path.read_text())
    raw = document["metrics"].get(metric, {}).get("mean")
    if raw is None:
        return math.nan
    return float(raw)


def collect_arm_rows(
    experiment_dir: Path,
    *,
    stage: str = "unseen_cables",
    step: int | None = None,
    metric: str = BEST_BY,
    final: bool = False,
) -> list[ArmRow]:
    """One row per finished run that has model summaries for ``stage``.

    ``final`` quotes each run's **last written** checkpoint, which is the
    paper's reporting rule (``docs/experiments.md``): a
    per-arm minimum over checkpoints is a screen device, never a quoted number.
    """
    rows: list[ArmRow] = []
    for run_dir in _run_dirs(experiment_dir):
        cfg = _load_config(run_dir)
        label = _label_of(run_dir, cfg)
        model_dir = run_dir / "eval" / "model"
        summaries = _summaries(model_dir, stage)
        if not summaries:
            continue
        if final:
            chosen = max(summaries)
        elif step is not None and step in summaries:
            chosen = step
        else:
            chosen = best_checkpoint_step(summaries, metric=metric)
        primary = _primary_from_summary(summaries[chosen], metric=metric)
        model = cfg.get("model") or {}
        train = cfg.get("train") or {}
        rows.append(
            ArmRow(
                label=label,
                run_dir=run_dir,
                step=chosen,
                primary=primary,
                d_model=int(model["d_model"]) if "d_model" in model else None,
                n_layers=int(model["n_layers"]) if "n_layers" in model else None,
                sigma=float(train["sigma"]) if "sigma" in train else None,
            )
        )
    rows.sort(key=lambda row: (math.inf if math.isnan(row.primary) else row.primary, row.label))
    return rows


def write_summary_table(
    experiment_dir: Path,
    rows: list[ArmRow],
    *,
    metric: str = BEST_BY,
) -> Path:
    """Write ``summary_table.csv`` and ``summary_table.json`` under the experiment."""
    experiment_dir.mkdir(parents=True, exist_ok=True)
    csv_path = experiment_dir / "summary_table.csv"
    json_path = experiment_dir / "summary_table.json"
    fieldnames = ["label", "step", metric, "d_model", "n_layers", "sigma", "run_dir"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "label": row.label,
                    "step": row.step,
                    metric: "" if math.isnan(row.primary) else f"{row.primary:.6g}",
                    "d_model": row.d_model if row.d_model is not None else "",
                    "n_layers": row.n_layers if row.n_layers is not None else "",
                    "sigma": "" if row.sigma is None else f"{row.sigma:.6g}",
                    "run_dir": str(row.run_dir),
                }
            )
    payload = [
        {
            "label": row.label,
            "step": row.step,
            metric: None if math.isnan(row.primary) else row.primary,
            "d_model": row.d_model,
            "n_layers": row.n_layers,
            "sigma": row.sigma,
            "run_dir": str(row.run_dir),
        }
        for row in rows
    ]
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    return csv_path


def plot_overlay(
    experiment_dir: Path,
    rows: list[ArmRow],
    *,
    stage: str = "unseen_cables",
    metric: str = "rel_l2",
    output: Path | None = None,
) -> Path:
    """Overlay best-checkpoint curves for every arm."""
    if not rows:
        raise ValueError(f"no finished arms with {stage!r} model eval under {experiment_dir}")

    mean_col = metric_column(metric)
    figure, axis = plt.subplots(figsize=(9, 5.5))

    for row in rows:
        curve_path = (
            row.run_dir / "eval" / "model" / f"{eval_artifact_stem(stage, row.step)}{_CURVE_SUFFIX}"
        )
        if not curve_path.is_file():
            continue
        frames, means = _read_mean_curve(curve_path, mean_col)
        tag = row.label
        if row.d_model is not None and row.n_layers is not None:
            tag = f"{row.label} (d={row.d_model} L={row.n_layers})"
        axis.plot(frames, means, label=f"{tag} @ {row.step}", linewidth=1.5)

    axis.set_xlabel("predicted frame")
    axis.set_ylabel(f"{metric} [{unit_of(metric)}]")
    axis.set_title(f"{metric} vs frame ({stage}) — {experiment_dir.name}")
    axis.legend(fontsize=8, loc="best")
    axis.set_ylim(bottom=0.0)
    figure.tight_layout()

    out = output or (experiment_dir / "figures" / f"{metric}_overlay_{stage}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=150)
    plt.close(figure)
    return out


def compare_experiment(
    experiment_dir: Path,
    *,
    stage: str = "unseen_cables",
    step: int | None = None,
    primary: str = BEST_BY,
    curve_metric: str = "rel_l2",
    final: bool = False,
) -> tuple[Path, Path | None]:
    """Write summary table + overlay; return ``(csv_path, overlay_or_None)``."""
    rows = collect_arm_rows(experiment_dir, stage=stage, step=step, metric=primary, final=final)
    csv_path = write_summary_table(experiment_dir, rows, metric=primary)
    overlay: Path | None = None
    if rows:
        overlay = plot_overlay(experiment_dir, rows, stage=stage, metric=curve_metric)
    return csv_path, overlay


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dlogps.harness.compare_runs")
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="results/<experiment>/ containing runs/",
    )
    parser.add_argument("--stage", default="unseen_cables")
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="fixed checkpoint step; default = best by primary metric per arm",
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help="quote each run's last written checkpoint (the paper's reporting "
        "rule); overrides --step",
    )
    parser.add_argument("--primary", default=BEST_BY)
    parser.add_argument("--curve-metric", default="rel_l2", choices=("rel_l2", "mean_l2"))
    args = parser.parse_args(argv)

    rows = collect_arm_rows(
        args.experiment_dir,
        stage=args.stage,
        step=args.step,
        metric=args.primary,
        final=args.final,
    )
    if not rows:
        print(
            f"no finished arms yet under {args.experiment_dir / 'runs'} "
            f"(need eval/model/{args.stage}_step*_summary.json)",
            flush=True,
        )
        return 0

    csv_path, overlay = compare_experiment(
        args.experiment_dir,
        stage=args.stage,
        step=args.step,
        primary=args.primary,
        curve_metric=args.curve_metric,
        final=args.final,
    )
    print(f"{'label':<12} {'step':>7} {args.primary:>16}  d  L  sigma", flush=True)
    for row in rows:
        primary = "nan" if math.isnan(row.primary) else f"{row.primary:.4f}"
        d = "-" if row.d_model is None else str(row.d_model)
        layers = "-" if row.n_layers is None else str(row.n_layers)
        sigma = "-" if row.sigma is None else f"{row.sigma:g}"
        print(
            f"{row.label:<12} {row.step:>7} {primary:>16}  {d:>3} {layers:>2}  {sigma}",
            flush=True,
        )
    print(f"wrote {csv_path}", flush=True)
    if overlay is not None:
        print(f"wrote {overlay}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
