"""Named experiment runner: one yaml in, one complete run directory out.

Orchestrates training, rollout evaluation at each checkpoint, artifact writes and
the per-run figures. Multi-arm sweeps are shell scripts over this module
(``scripts/train_matrix.sh``, ``scripts/train_baselines.sh``).
"""

from __future__ import annotations

import json
import math
import re
import signal
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import draccus
import torch
import yaml

from dlogps.data.dataset import CableData
from dlogps.data.loader import SplitCfg, load_dataset
from dlogps.harness.artifacts import write_eval_artifacts
from dlogps.harness.evaluate import STAGES, evaluate
from dlogps.harness.metrics import RegimeCfg
from dlogps.harness.plots import best_checkpoint_step, plot_loss_mse, write_run_figures
from dlogps.harness.runlog import CONFIG_NAME, RunCfg, RunLog
from dlogps.harness.runtime import bound_intra_op_threads, model_device
from dlogps.harness.train import (
    TrainCfg,
    TrainResult,
    load_checkpoint,
    save_checkpoint,
    train_on_dataset,
)
from dlogps.model.gps import GPSCfg

PRIMARY_METRIC = "rel_l2_at_h200"
"""Fixed v1 rule for best-checkpoint selection in plots."""

EVAL_ARMS = ("model",)
"""Allowed values of :attr:`EvalCfg.arms`. Only the trained model is scored."""

BEST_ONE_STEP_NAME = "best.pt"
BEST_ONE_STEP_META = "best.json"
BEST_ROLLOUT_META = "best_rollout.json"


@dataclass
class EvalCfg:
    """Evaluation stage selection and reported horizons."""

    stage: str = "both"
    limit: int | None = None
    horizons: tuple[int, ...] = (1, 10, 50, 100, 200)
    arms: str = "model"


@dataclass
class PlotsCfg:
    """Figure generation options."""

    best_by: str = PRIMARY_METRIC


@dataclass
class ExperimentCfg:
    """Full configuration for one named experiment run."""

    experiment: str
    label: str
    data: Path
    eval_data: Path
    train: TrainCfg
    variant: str = "A"
    model: GPSCfg = field(default_factory=GPSCfg)
    split: SplitCfg = field(default_factory=SplitCfg)
    checkpoint_at: tuple[int, ...] = ()
    eval: EvalCfg = field(default_factory=EvalCfg)
    plots: PlotsCfg = field(default_factory=PlotsCfg)
    results_root: Path = Path("results")
    save_preds: bool = False
    eval_unseen_data: Path | None = None
    """Optional second test root scored as ``unseen_test`` (whole population)."""


def validate_experiment(cfg: ExperimentCfg) -> int:
    """Rollout length implied by ``eval.horizons``.

    Returns:
        Rollout frame count ``max(horizons) + 1``.

    Raises:
        ValueError: when v1 rules are violated.
    """
    if cfg.plots.best_by != PRIMARY_METRIC:
        raise ValueError(
            f"plots.best_by must be {PRIMARY_METRIC!r} in v1, got {cfg.plots.best_by!r}"
        )
    if PRIMARY_METRIC.endswith("h200") and 200 not in cfg.eval.horizons:
        raise ValueError(
            f"{PRIMARY_METRIC} requires horizon 200 in eval.horizons, got {cfg.eval.horizons}"
        )
    rollout_frames = max(cfg.eval.horizons) + 1
    if 200 in cfg.eval.horizons and rollout_frames < 201:
        raise ValueError(
            f"horizon 200 needs a rollout of at least 201 frames, got {rollout_frames}"
        )
    if cfg.eval.stage not in {*STAGES, "both"}:
        raise ValueError(
            f"eval.stage must be both or one of {sorted(STAGES)}, got {cfg.eval.stage!r}"
        )
    if cfg.eval.arms not in EVAL_ARMS:
        raise ValueError(f"eval.arms must be one of {EVAL_ARMS}, got {cfg.eval.arms!r}")
    return rollout_frames


def ensure_experiment_dir(experiment_dir: Path) -> None:
    """Create the experiment directory when it is new."""
    experiment_dir.mkdir(parents=True, exist_ok=True)


def write_experiment_config(run_dir: Path, cfg: ExperimentCfg) -> None:
    """Persist the resolved :class:`ExperimentCfg` as ``config.yaml``."""
    document = _cfg_document(cfg)
    (run_dir / CONFIG_NAME).write_text(yaml.safe_dump(document, sort_keys=False))


def assert_eval_device(requested: str, model: torch.nn.Module) -> None:
    """Fail fast when CUDA was requested but unavailable, and after ``.to``."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"train.device={requested!r} but torch.cuda.is_available() is false; "
            "refusing silent CPU fallback for rollout eval"
        )
    placed = model_device(model)
    want = torch.device(requested)
    if placed.type != want.type:
        raise RuntimeError(
            f"model landed on {placed} after .to({requested!r}); refusing mismatched eval device"
        )
    if want.index is not None and placed.index not in (None, want.index):
        raise RuntimeError(
            f"model landed on {placed} after .to({requested!r}); refusing mismatched eval device"
        )


def _arm_names(arms: str) -> tuple[str, ...]:
    return (arms,)


def run_experiment(cfg: ExperimentCfg) -> Path:
    """Execute the full train-eval-figures pipeline.

    Returns:
        The run directory path.
    """
    rollout_frames = validate_experiment(cfg)
    experiment_dir = cfg.results_root / cfg.experiment
    ensure_experiment_dir(experiment_dir)

    checkpoint_steps = sorted(set(cfg.checkpoint_at) | {cfg.train.steps})
    run_cfg = RunCfg(
        data=cfg.data,
        variant=cfg.variant,
        train=cfg.train,
        model=cfg.model,
        split=cfg.split,
        checkpoint_at=tuple(checkpoint_steps),
        save_preds=cfg.save_preds,
    )
    log = RunLog.start(experiment_dir, cfg.label, run_cfg)
    write_experiment_config(log.run_dir, cfg)
    print(
        f"=== {cfg.experiment}/{cfg.label} ===\n"
        f"run_dir={log.run_dir}\n"
        f"variant={cfg.variant}  sigma={cfg.train.sigma:.4f}  "
        f"steps={cfg.train.steps}  batch={cfg.train.batch_size}  "
        f"lr={cfg.train.lr:.4f}->{cfg.train.lr_final:.4f}  "
        f"seed={cfg.train.seed}  C={cfg.train.history_frames}  "
        f"device={cfg.train.device}\n"
        f"model d={cfg.model.d_model} L={cfg.model.n_layers} H={cfg.model.n_heads}  "
        f"global={cfg.model.use_global} local={cfg.model.use_local}\n"
        f"split held_out={cfg.split.held_out_fraction:.4f}  "
        f"d_close={cfg.split.d_close_diameters:.4f}  c_far={cfg.split.c_far_segments}\n"
        f"checkpoints={checkpoint_steps}  eval.stage={cfg.eval.stage}  "
        f"eval.arms={cfg.eval.arms}  limit={cfg.eval.limit}  horizons={cfg.eval.horizons}",
        flush=True,
    )

    print(f"loading eval data {cfg.eval_data} …", flush=True)
    eval_cables = load_dataset(cfg.eval_data)
    print(f"  {len(eval_cables)} cables", flush=True)
    unseen_cables = None
    if cfg.eval_unseen_data is not None:
        print(f"loading unseen eval data {cfg.eval_unseen_data} …", flush=True)
        unseen_cables = load_dataset(cfg.eval_unseen_data)
        print(f"  {len(unseen_cables)} cables", flush=True)
    regime = RegimeCfg(horizons=cfg.eval.horizons)
    if cfg.eval.stage == "both":
        stages = sorted(s for s in STAGES if s in ("seen_cables", "unseen_cables"))
    else:
        stages = [cfg.eval.stage]
    eval_jobs: list[tuple[str, list[CableData], Path]] = [
        (stage, eval_cables, cfg.eval_data) for stage in stages
    ]
    if unseen_cables is not None and cfg.eval_unseen_data is not None:
        eval_jobs.append(("unseen_test", unseen_cables, cfg.eval_unseen_data))
    arm_names = _arm_names(cfg.eval.arms)
    frame_dt = float(eval_cables[0].dt(0))

    snapshots: dict[int, Path] = {}
    best_one_step: dict[str, float | int] = {"step": -1, "eval_loss": math.inf}

    def keep(step: int, snapshot: TrainResult) -> None:
        path = save_checkpoint(log.checkpoint_path(step), snapshot)
        snapshots[step] = path
        print(f"  checkpoint step {step} → {path.name}", flush=True)

    def keep_best(step: int, eval_loss: float, snapshot: TrainResult) -> None:
        # Rolling pointer only. Do not add every improvement to the rollout set
        # (eval_every=500 would otherwise schedule dozens of rollouts).
        save_checkpoint(log.checkpoints_dir / BEST_ONE_STEP_NAME, snapshot)
        best_one_step["step"] = step
        best_one_step["eval_loss"] = eval_loss
        (log.checkpoints_dir / BEST_ONE_STEP_META).write_text(
            json.dumps(
                {
                    "step": step,
                    "eval_loss": eval_loss,
                    "criterion": "one_step_held_out_mse",
                    "path": BEST_ONE_STEP_NAME,
                },
                indent=2,
            )
            + "\n"
        )
        print(
            f"  new best one-step eval {eval_loss:.4f} at step {step} → {BEST_ONE_STEP_NAME}",
            flush=True,
        )

    last_train_loss = [float("nan")]
    started = time.perf_counter()

    def on_step(step: int, loss: float) -> None:
        last_train_loss[0] = loss

    def on_eval(step: int, loss: float) -> None:
        log.log_step(step=step, train_loss=last_train_loss[0], eval_loss=loss)
        elapsed = time.perf_counter() - started
        rate = step / elapsed if elapsed > 0 else 0.0
        remaining = (cfg.train.steps - step) / rate if rate > 0 else float("nan")
        print(
            f"  step {step}/{cfg.train.steps}  "
            f"train {last_train_loss[0]:.4f}  eval {loss:.4f}  "
            f"[{elapsed:.0f}s elapsed, ~{remaining:.0f}s left]",
            flush=True,
        )
        # One live file: overwrite figures/loss_mse.png from the growing JSONL.
        plot_loss_mse(log.run_dir)

    print(f"training on {cfg.data} …", flush=True)
    print(
        "Ctrl+C once = early-stop training then full eval on saved checkpoints; "
        "Ctrl+C twice = abort.",
        flush=True,
    )
    final = train_on_dataset(
        cfg.data,
        cfg.train,
        variant=cfg.variant,
        model_cfg=cfg.model,
        split=cfg.split,
        checkpoint_at=checkpoint_steps,
        on_checkpoint=keep,
        on_step=on_step,
        on_eval=on_eval,
        # With no within-grid holdout the one-step stream comes from the test
        # root (eval_root below): it may be watched, never selected on. Passing
        # the hook there would pick best.pt on a reporting dataset.
        on_best_checkpoint=keep_best if cfg.split.held_out_fraction > 0 else None,
        progress=True,
        interruptible=True,
        eval_root=cfg.eval_data,
    )
    train_seconds = time.perf_counter() - started
    log.finish(seconds=train_seconds)
    actual_steps = len(final.losses)
    early = actual_steps < cfg.train.steps
    print(
        f"training {'stopped early at' if early else 'done at'} step {actual_steps}"
        f"/{cfg.train.steps} in {train_seconds:.0f}s  "
        f"({train_seconds / max(actual_steps, 1) * 1e3:.2f} ms/step)",
        flush=True,
    )
    if int(best_one_step["step"]) > 0:
        best_step = int(best_one_step["step"])
        print(
            f"best one-step checkpoint: step {best_step}  "
            f"eval_loss={float(best_one_step['eval_loss']):.4f}  ({BEST_ONE_STEP_NAME})",
            flush=True,
        )
        # Include the one-step winner in rollout eval when it is not already a
        # scheduled snapshot. Copy off best.pt into a durable step_*.pt so the
        # eval loop cannot race with a later overwrite of the rolling pointer.
        if best_step not in snapshots:
            best_path = log.checkpoints_dir / BEST_ONE_STEP_NAME
            durable = log.checkpoint_path(best_step)
            durable.write_bytes(best_path.read_bytes())
            snapshots[best_step] = durable
            print(
                f"  also rolling out one-step best at step {best_step} → {durable.name}",
                flush=True,
            )

    eval_steps = sorted(snapshots)
    if not eval_steps:
        print("no checkpoints on disk; skipping rollout eval and figures", flush=True)
        return log.run_dir
    if early:
        print(
            f"evaluating available checkpoints only: {eval_steps} (planned {checkpoint_steps})",
            flush=True,
        )

    device_name = (
        torch.cuda.get_device_name(0)
        if cfg.train.device.startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    print(f"rollout eval on {cfg.train.device} ({device_name})", flush=True)

    n_evals = len(eval_steps) * len(eval_jobs) * len(arm_names)
    eval_i = 0
    stop_requested = False
    previous_handler = signal.getsignal(signal.SIGINT)

    def _request_stop(signum: int, frame: object) -> None:
        nonlocal stop_requested
        if stop_requested:
            raise KeyboardInterrupt
        stop_requested = True
        print(
            "\nCtrl+C: early-stop eval after current cell; Ctrl+C again to abort hard.",
            flush=True,
        )

    signal.signal(signal.SIGINT, _request_stop)
    try:
        for step in eval_steps:
            if stop_requested:
                break
            checkpoint = load_checkpoint(snapshots[step])
            model = checkpoint.restore_model()
            arms = {"model": model}
            for stage, cables_for_stage, _data_root in eval_jobs:
                if stop_requested:
                    break
                for arm_name in arm_names:
                    if stop_requested:
                        break
                    eval_i += 1
                    arm = arms[arm_name].to(cfg.train.device)
                    assert_eval_device(cfg.train.device, arm)
                    desc = f"{eval_i}/{n_evals} {stage}/{arm_name}"
                    print(
                        f"eval [{eval_i}/{n_evals}] step={step} stage={stage} arm={arm_name}",
                        flush=True,
                    )

                    scored = evaluate(
                        arm,
                        cables_for_stage,
                        stage=stage,
                        horizon=rollout_frames,
                        stats=checkpoint.stats,
                        regime=regime,
                        split=cfg.split,
                        history_frames=cfg.train.history_frames,
                        limit=cfg.eval.limit,
                        progress=True,
                        desc=desc,
                        stop_flag=lambda: stop_requested,
                    )

                    write_eval_artifacts(
                        log.eval_dir / arm_name,
                        scored,
                        step=step,
                        horizon=rollout_frames,
                        regime=regime,
                        cables=cables_for_stage,
                        split=cfg.split,
                        frame_dt=frame_dt,
                    )
                    primary = scored.scores.get(PRIMARY_METRIC, float("nan"))
                    print(
                        f"  {PRIMARY_METRIC}={primary:.4f}  windows={scored.n_windows}",
                        flush=True,
                    )
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        print("writing figures …", flush=True)
        try:
            figures = write_run_figures(log.run_dir)
            for path in figures:
                print(f"  {path.name}", flush=True)
        except Exception as err:  # noqa: BLE001 — figures are best-effort after soft-stop
            print(f"  figures failed: {err}", flush=True)
        _write_best_rollout_meta(log, stages=[job[0] for job in eval_jobs])

    if stop_requested:
        print("eval stopped early; figures written from artifacts on disk", flush=True)
    return log.run_dir


def _write_best_rollout_meta(log: RunLog, *, stages: list[str]) -> None:
    """Record the checkpoint with best ``PRIMARY_METRIC`` after rollout eval.

    Written only when a tuning-surface stage was scored. A whole-root test stage
    (``seen_test`` / ``unseen_test`` / ``all_cables``) may never feed a
    best-checkpoint pointer: the reported checkpoint there is the final one.
    """
    model_dir = log.eval_dir / "model"
    if not model_dir.is_dir():
        return
    # Prefer unseen when both tuning stages were scored; that is the selection
    # stage.
    tuning = [s for s in stages if s in ("unseen_cables", "seen_cables")]
    if not tuning:
        print(
            f"no tuning-surface stage scored; skipping {BEST_ROLLOUT_META} "
            "(the reported checkpoint is the final one)",
            flush=True,
        )
        return
    stage = "unseen_cables" if "unseen_cables" in tuning else tuning[0]
    pattern = re.compile(rf"^{re.escape(stage)}_step(\d+)_summary\.json$")
    summaries: dict[int, Path] = {}
    for path in model_dir.iterdir():
        match = pattern.match(path.name)
        if match is not None:
            summaries[int(match.group(1))] = path
    if not summaries:
        return
    try:
        best_step = best_checkpoint_step(summaries, PRIMARY_METRIC)
    except ValueError:
        return
    document = json.loads(summaries[best_step].read_text())
    value = float(document["metrics"][PRIMARY_METRIC]["mean"])
    meta = {
        "step": best_step,
        "metric": PRIMARY_METRIC,
        "value": value,
        "stage": stage,
        "checkpoint": log.checkpoint_path(best_step).name,
    }
    out = log.checkpoints_dir / BEST_ROLLOUT_META
    out.write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"best rollout checkpoint: step {best_step}  "
        f"{PRIMARY_METRIC}={value:.4f} on {stage} → {BEST_ROLLOUT_META}",
        flush=True,
    )


def _cfg_document(cfg: ExperimentCfg) -> dict[str, object]:
    raw = asdict(cfg)
    raw["data"] = str(cfg.data)
    raw["eval_data"] = str(cfg.eval_data)
    raw["results_root"] = str(cfg.results_root)
    if cfg.eval_unseen_data is not None:
        raw["eval_unseen_data"] = str(cfg.eval_unseen_data)
    return raw


def main(argv: list[str] | None = None) -> int:
    """Parse ``ExperimentCfg`` and run one experiment."""
    bound_intra_op_threads()
    cfg = draccus.parse(config_class=ExperimentCfg, args=argv)
    run_dir = run_experiment(cfg)
    print(f"run finished: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
