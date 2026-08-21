"""The on-disk record for one training run.

Every number in the paper has to trace back to the bytes that produced it months
later. This module is the only writer of ``config.yaml``, ``env.json`` and the
training logs under a run directory; checkpoints and evaluation artifacts land
in subdirectories it creates but are written by their respective callers.

Layout: ``docs/reproduction.md``.
"""

from __future__ import annotations

import json
import platform
import socket
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from dlogps import PROJECT_ROOT
from dlogps.data.loader import SplitCfg
from dlogps.harness.train import TrainCfg
from dlogps.model.gps import GPSCfg

CONFIG_NAME = "config.yaml"
ENV_NAME = "env.json"
TRAIN_LOG_JSONL = "train_log.jsonl"
TRAIN_LOG_TXT = "train_log.txt"
CHECKPOINTS_DIR = "checkpoints"
EVAL_DIR = "eval"


@dataclass(frozen=True)
class RunCfg:
    """The resolved config for one run: enough to re-run it with no CLI.

    Attributes:
        data: Dataset root holding ``cable_*/``.
        variant: A key of :data:`~dlogps.harness.train.VARIANTS`.
        train: The run's :class:`~dlogps.harness.train.TrainCfg`.
        model: Block hyper-parameters.
        split: Population split and label thresholds.
        checkpoint_at: Optimizer steps at which to write a checkpoint, in order.
        save_preds: Whether evaluation should write prediction dumps; recorded
            so its absence is visible when false.
    """

    data: Path
    variant: str
    train: TrainCfg
    model: GPSCfg
    split: SplitCfg
    checkpoint_at: tuple[int, ...] = ()
    save_preds: bool = False


def run_dir_name(label: str, when: datetime | None = None) -> str:
    """A run directory name: timestamp plus a short readable label.

    Sorting by name sorts by time, which is what anyone browsing wants. The
    encoded-hyperparameter style is deliberately not used here; the config file
    is the only place that has to be complete.

    Args:
        label: A short human label, sanitized to alphanumeric characters,
            hyphens and underscores.
        when: The start time; ``datetime.now()`` in the local timezone if omitted.

    Returns:
        A name of the form ``2026-08-05T16-12-03_ref``.

    Raises:
        ValueError: if the label has no alphanumeric characters after sanitizing.
    """
    started = when or datetime.now().astimezone()
    return f"{started.strftime('%Y-%m-%dT%H-%M-%S')}_{_label_slug(label)}"


def git_state(repo: Path = PROJECT_ROOT) -> tuple[str | None, bool]:
    """The current commit and whether the working tree is dirty.

    A dirty tree cannot be reproduced from committed code, so an unreadable
    repo is reported as dirty rather than clean.

    Args:
        repo: Git repository root.

    Returns:
        ``(sha, dirty)``. ``sha`` is ``None`` when git cannot say.
    """
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        return sha or None, bool(status)
    except (OSError, subprocess.SubprocessError):
        return None, True


def load_run_cfg(run_dir: str | Path) -> RunCfg:
    """Read the resolved config written at the start of a run.

    Args:
        run_dir: A run directory holding ``config.yaml``.

    Returns:
        The config sufficient to re-run the training job.

    Raises:
        ValueError: if the file is missing expected keys or has the wrong shape.
    """
    path = Path(run_dir) / CONFIG_NAME
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")

    for key in ("data", "variant", "train", "model", "split"):
        if key not in raw:
            raise ValueError(f"{path}: missing key {key!r}")

    train_raw = raw["train"]
    model_raw = raw["model"]
    split_raw = raw["split"]
    if not isinstance(train_raw, dict):
        raise ValueError(f"{path}: train must be a mapping")
    if not isinstance(model_raw, dict):
        raise ValueError(f"{path}: model must be a mapping")
    if not isinstance(split_raw, dict):
        raise ValueError(f"{path}: split must be a mapping")

    checkpoint_raw = raw.get("checkpoint_at") or ()
    if not isinstance(checkpoint_raw, list):
        raise ValueError(f"{path}: checkpoint_at must be a list")

    return RunCfg(
        data=Path(str(raw["data"])),
        variant=str(raw["variant"]),
        train=TrainCfg(**train_raw),
        model=GPSCfg(**model_raw),
        split=SplitCfg(**split_raw),
        checkpoint_at=tuple(int(step) for step in checkpoint_raw),
        save_preds=bool(raw.get("save_preds", False)),
    )


class RunLog:
    """One run's directory and the training log streams inside it.

    Creates ``checkpoints/`` and ``eval/`` at open time. Writes ``config.yaml``
    and the initial ``env.json`` once; appends ``train_log.jsonl`` and
    ``train_log.txt`` on every report step; rewrites ``env.json`` on finish.
    """

    def __init__(
        self,
        run_dir: str | Path,
        cfg: RunCfg,
        *,
        repo: Path = PROJECT_ROOT,
        when: datetime | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.cfg = cfg
        self._repo = repo
        self._started = when or datetime.now().astimezone()
        self.run_dir.mkdir(parents=True, exist_ok=False)
        (self.run_dir / CHECKPOINTS_DIR).mkdir()
        (self.run_dir / EVAL_DIR).mkdir()
        self._write_config()
        self._write_env()

    @classmethod
    def start(
        cls,
        experiment_dir: str | Path,
        label: str,
        cfg: RunCfg,
        *,
        repo: Path = PROJECT_ROOT,
        when: datetime | None = None,
    ) -> RunLog:
        """Open a new run under ``<experiment>/runs/<timestamp>_<label>/``.

        Args:
            experiment_dir: The experiment root, e.g. ``results/capacity_2x2``.
            label: Short readable label for the run.
            cfg: The resolved config to record.
            repo: Git repository for provenance in ``env.json``.
            when: Start time; ``datetime.now()`` if omitted.

        Returns:
            A :class:`RunLog` rooted at the new directory.
        """
        run_dir = Path(experiment_dir) / "runs" / run_dir_name(label, when=when)
        return cls(run_dir, cfg, repo=repo, when=when)

    @property
    def checkpoints_dir(self) -> Path:
        """Where ``step_NNNNN.pt`` files are written."""
        return self.run_dir / CHECKPOINTS_DIR

    @property
    def eval_dir(self) -> Path:
        """Where evaluation CSV and JSON artifacts are written."""
        return self.run_dir / EVAL_DIR

    def checkpoint_path(self, step: int) -> Path:
        """The path for one checkpoint file."""
        return self.checkpoints_dir / f"step_{step:05d}.pt"

    def log_step(
        self,
        step: int,
        train_loss: float,
        eval_loss: float,
        lr: float | None = None,
        seconds: float | None = None,
    ) -> None:
        """Append one report line to both training logs.

        Each line is flushed immediately so a killed run keeps what it had.

        Args:
            step: Optimizer step.
            train_loss: Noised single-batch training loss, normalized units.
            eval_loss: Pooled clean held-out loss at this step.
            lr: Learning rate after the step.
            seconds: Wall clock since the run started.
        """
        # A key the caller could not know is omitted rather than written as a
        # guessed zero: a JSONL line that always carries `lr` invites a reader to
        # plot one, and the learning rate belongs to the trainer's schedule.
        record: dict[str, float | int] = {
            "step": step,
            "train_loss": train_loss,
            "eval_loss": eval_loss,
        }
        if lr is not None:
            record["lr"] = lr
        if seconds is not None:
            record["seconds"] = seconds
        jsonl_path = self.run_dir / TRAIN_LOG_JSONL
        with jsonl_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()

        txt_path = self.run_dir / TRAIN_LOG_TXT
        parts = [
            f"step {step}",
            f"train_loss {train_loss:.6e}",
            f"eval_loss {eval_loss:.6e}",
        ]
        if lr is not None:
            parts.append(f"lr {lr:.6e}")
        if seconds is not None:
            parts.append(f"seconds {seconds:.1f}")
        line = "  ".join(parts) + "\n"
        with txt_path.open("a") as handle:
            handle.write(line)
            handle.flush()

    def finish(self, seconds: float) -> None:
        """Record the run's end time and total wall clock in ``env.json``."""
        finished = datetime.now().astimezone()
        self._write_env(finished=finished, seconds=seconds)

    def _write_config(self) -> None:
        document: dict[str, Any] = {
            "data": str(self.cfg.data),
            "variant": self.cfg.variant,
            "train": asdict(self.cfg.train),
            "model": asdict(self.cfg.model),
            "split": asdict(self.cfg.split),
            "checkpoint_at": list(self.cfg.checkpoint_at),
            "save_preds": self.cfg.save_preds,
        }
        (self.run_dir / CONFIG_NAME).write_text(yaml.safe_dump(document, sort_keys=False))

    def _write_env(self, *, finished: datetime | None = None, seconds: float | None = None) -> None:
        sha, dirty = git_state(self._repo)
        document: dict[str, Any] = {
            "git_sha": sha,
            "git_dirty": dirty,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": _device_name(self.cfg.train.device),
            "host": socket.gethostname(),
            "started": self._started.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if finished is not None:
            document["finished"] = finished.strftime("%Y-%m-%dT%H:%M:%S")
        if seconds is not None:
            document["seconds"] = seconds
        (self.run_dir / ENV_NAME).write_text(json.dumps(document, indent=2) + "\n")


def _label_slug(label: str) -> str:
    cleaned: list[str] = []
    for char in label.strip():
        if char.isalnum() or char in "-_":
            cleaned.append(char)
        elif cleaned and cleaned[-1] != "_":
            cleaned.append("_")
    slug = "".join(cleaned).strip("_")
    if not slug:
        raise ValueError(f"label {label!r} has no alphanumeric characters after sanitizing")
    return slug


def _device_name(device: str) -> str:
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(torch_device)
    return device
