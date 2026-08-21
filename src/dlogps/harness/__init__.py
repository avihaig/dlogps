"""Harness package: statistics, input noise, training, rollout, and the scorecard.

The whole path from a dataset directory to a scored trajectory lives here, in six
modules that each hold one link of it:

- :mod:`~dlogps.harness.normalize` fits the feature statistics, on the training
  side of the training split and nowhere else, and carries them into the
  checkpoint so rollout scales its inputs exactly as training did.
- :mod:`~dlogps.harness.noise` corrupts the training state with a random walk,
  which is the one thing that makes a one-step model survive being fed its own
  output.
- :mod:`~dlogps.harness.train` turns a batch stream into a checkpoint and reports
  what the run cost.
- :mod:`~dlogps.harness.rollout` feeds the model its own predictions, frame 0
  being the input state rather than a prediction.
- :mod:`~dlogps.harness.evaluate` rolls out a stage of the test set and scores it,
  reporting the two generalization stages separately and never merged.
- :mod:`~dlogps.harness.metrics` holds every number the write-up reports, and
  nothing the spec does not already name.

:mod:`~dlogps.harness.compare_arms` is the entry point that composes all six: it
trains no bias (A) and the chain-only baseline at one seed, evaluates both on
both stages, and prints what separates them. Run it with
``python -m dlogps.harness.compare_arms``.

**Four names are reached through their module and are deliberately not re-exported
here.** ``train``, ``normalize``, ``rollout`` and ``evaluate`` each name both a
module and that module's principal function, and binding the function in this
namespace replaces the submodule attribute: ``import dlogps.harness.evaluate as
m`` would then hand back a function, and the failure surfaces as an
``AttributeError`` somewhere else entirely. Import those four from their modules,
``from dlogps.harness.evaluate import evaluate``.
"""

from __future__ import annotations

from dlogps.harness.evaluate import (
    STAGES,
    EvalResult,
    WindowStream,
    evaluate_checkpoint,
    stage_cables,
)
from dlogps.harness.metrics import (
    METRIC_BUILDERS,
    MetricFn,
    RegimeCfg,
    build_metrics,
    score,
)
from dlogps.harness.noise import add_random_walk_noise, random_walk
from dlogps.harness.normalize import (
    Stats,
    denormalize_step,
    fit_stats,
    normalize_step,
)
from dlogps.harness.rollout import roll_in
from dlogps.harness.runtime import (
    INTRA_OP_THREADS,
    batch_to,
    bound_intra_op_threads,
    labels_to,
    model_device,
)
from dlogps.harness.train import (
    DEFAULT_VARIANT,
    VARIANTS,
    BatchSource,
    Checkpoint,
    TrainCfg,
    TrainResult,
    Variant,
    build_model,
    dataset_source,
    decay_gamma,
    git_sha,
    load_checkpoint,
    save_checkpoint,
    step_loss,
    train_on_dataset,
)

__all__ = [
    "DEFAULT_VARIANT",
    "INTRA_OP_THREADS",
    "METRIC_BUILDERS",
    "STAGES",
    "VARIANTS",
    "BatchSource",
    "Checkpoint",
    "EvalResult",
    "MetricFn",
    "RegimeCfg",
    "Stats",
    "TrainCfg",
    "TrainResult",
    "Variant",
    "WindowStream",
    "add_random_walk_noise",
    "batch_to",
    "bound_intra_op_threads",
    "build_metrics",
    "build_model",
    "dataset_source",
    "decay_gamma",
    "denormalize_step",
    "evaluate_checkpoint",
    "fit_stats",
    "git_sha",
    "labels_to",
    "load_checkpoint",
    "model_device",
    "normalize_step",
    "random_walk",
    "roll_in",
    "save_checkpoint",
    "score",
    "stage_cables",
    "step_loss",
    "train_on_dataset",
]
