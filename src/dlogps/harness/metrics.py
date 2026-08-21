"""The metric suite: every number the write-up reports, and nothing else.

The scorecard half of rollout and metrics. Scores a **rollout**, the model fed
its own predictions for hundreds of steps, against the ground truth and against
physics.

**The criteria are frozen, and this file implements them rather than choosing
them** (``docs/metrics.md``). Nothing enters the registry that the paper does
not already name. That is the whole point: thresholds picked after
seeing a result are not thresholds.

Two of the metrics need **no ground truth at all**. A cable's links should keep
their length and a cable cannot pass through itself, so a rollout that breaks
either has invented physics, and no reference trajectory is needed to say so.
On recorded data both read ~zero by construction, since the solver forbids
interpenetration — which makes the ground truth the natural control.

**The regime thresholds are the loader's, and this module holds no copy of
them.** ``d_close`` and ``c_far`` are marked `[FIX ON PROBE]`, and the place they
bite is :func:`dlogps.data.labels.contact_labels`, which the loader calls on the
ground-truth window; every decomposition here reads
:class:`~dlogps.data.types.Labels` and derives nothing. A second copy on
:class:`RegimeCfg` is therefore not a convenience but a trap: it reaches no
metric, so a flag that appears to set the thresholds sets nothing, and the gate
that the thresholds are fixed **before** any result is seen breaks with every
number still finite. One owner, ``SplitCfg``, chosen at training time and carried
in the checkpoint.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from dlogps.data.geometry import chain_far_mask, segment_distance
from dlogps.data.types import BatchMeta, E, Labels

UNITS: dict[str, str] = {
    "rmse": "m",
    "mean_l2": "m",
    "rel_l2": "fraction of cable length",
    "link_length_mean_drift": "fraction of rest length",
    "link_length_max_drift": "fraction of rest length",
    "selfint_frame_fraction": "fraction of frames",
    "selfint_pair_fraction": "fraction of chain-far pairs",
    "selfint_worst_penetration": "m",
    "selfint_persists": "fraction of windows",
    "contact_frame_fraction": "fraction of frames",
    "floor_contact_frame_fraction": "fraction of frames",
}
"""Physical unit per metric, looked up by :func:`unit_of`.

**Every length in this project is metres**, on disk and in every report; the
dataset is stored that way (``docs/data.md``) and nothing
converts. The map exists because a table of bare floats spanning 1e-5 to 1e0
invites a reader to assume millimetres somewhere, and because the two
dimensionless families -- drift as a share of rest length, and the several
fractions -- look identical to a length until something says otherwise."""


def unit_of(name: str) -> str:
    """The unit of a metric name. **Raises on a name it does not know.**

    Horizon-suffixed and decomposed names resolve to their family, so
    ``rmse_h100``, ``rmse_contact``, ``mean_l2_at_h200`` and ``rel_l2_full`` all
    resolve without the map listing every horizon a config might ask for.

    **An unknown name is an error, not a dimensionless quantity.** Returning the
    empty string for both would make a typo'd or newly-added metric print as a
    bare number beside metres, which is the reading a table cannot recover from.
    Every dimensionless family here carries an explicit descriptive unit instead.

    Raises:
        KeyError: naming the families, so a new metric fails at its first report
            rather than silently losing its unit.
    """
    if name in UNITS:
        return UNITS[name]
    for prefix, unit in UNITS.items():
        if name.startswith(f"{prefix}_"):
            return unit
    raise KeyError(
        f"no unit registered for metric {name!r}; add it to UNITS or name it after "
        f"an existing family ({', '.join(sorted(UNITS))})"
    )


MetricFn = Callable[[Tensor, Tensor, BatchMeta, Labels], dict[str, float]]
"""``(pred, target, meta, labels) -> named scalars``.

``pred`` and ``target`` are rollout tensors ``[B, R, N, 3]``, with ``R`` the
rollout length in frames — the same axis the labels carry, so a decomposition
indexes straight through."""


@dataclass(frozen=True)
class RegimeCfg:
    """The horizons to report at, and deliberately nothing else.

    Attributes:
        horizons: **Predicted** frame counts at which rollout error is reported.
            Frame 0 of a rollout is the input state, so ``h`` needs a rollout of
            ``h + 1`` and anything longer is truncated to the rollout actually
            supplied, which keeps a short smoke test scoring.

    **The contact thresholds are not here**, and their absence is the point: they
    are ``SplitCfg.d_close_diameters`` and ``SplitCfg.c_far_segments``, they are
    applied by the loader when it labels the ground-truth window, and they travel
    in the checkpoint. Holding a second copy on this object is what made two CLI
    flags appear to set them while setting nothing. See the module docstring.
    """

    horizons: tuple[int, ...] = (1, 10, 50, 100)

    def __post_init__(self) -> None:
        if not self.horizons or any(h < 1 for h in self.horizons):
            raise ValueError(f"horizons must be positive frame counts, got {self.horizons}")


# --- the metrics ------------------------------------------------------------


def _node_distance(pred: Tensor, target: Tensor) -> Tensor:
    """``[B, R, N]`` Euclidean distance between predicted and true node positions.

    **The axes reduce in this order and the order is the whole definition**: x, y
    and z collapse into one distance per node, and only then are nodes combined.
    Averaging over coordinates as if they were three independent scalars would
    divide by ``3N`` rather than ``N`` and report every error a factor of
    ``sqrt(3)`` too small, with nothing raising.
    """
    return (pred - target).norm(dim=-1)


def _rmse(pred: Tensor, target: Tensor) -> Tensor:
    """``[B, R]`` root-mean-square of the **per-node Euclidean distances**.

    Not an RMSE over coordinates; see :func:`_node_distance`. Squaring before the
    mean over nodes is what makes this weight the worst nodes hardest, which is
    the difference from :func:`_mean_l2` and is not a rescaling of it.
    """
    return _node_distance(pred, target).square().mean(-1).sqrt()


def _mean_l2(pred: Tensor, target: Tensor) -> Tensor:
    """``[B, R]`` mean of the per-node Euclidean distances.

    The DLO literature's per-node L2 convention.

    **Jensen puts this at or below** :func:`_rmse` **always, and the gap is not a
    constant.** It is how unevenly the error is spread across the cable: measured
    on real rollouts it runs 1.00 to 2.71. A model
    whose free end diverges while the rest tracks is punished far harder by the
    RMS, so the two reductions can rank two arms differently, and both are
    reported for that reason.
    """
    return _node_distance(pred, target).mean(-1)


def per_frame_errors(pred: Tensor, target: Tensor, meta: BatchMeta) -> dict[str, Tensor]:
    """``[B, R-1]`` per predicted frame, one entry per reduction.

    **The curve, which no scalar can reconstruct.** Every metric in the registry
    collapses the frame axis, so a table of horizons says where a rollout stood
    at four moments and nothing about the shape between them: whether a config is
    better everywhere or only early is visible in this and not in those.

    Frame 0 is dropped here as everywhere else, so index ``k`` is predicted frame
    ``k + 1`` and the vector aligns with the ``*_at_h{h}`` family.

    Args:
        pred: ``[B, R, N, 3]`` the rollout.
        target: ``[B, R, N, 3]`` ground truth over the same window.
        meta: Read for ``length``, which ``rel_l2`` divides by per sample.
    """
    l2 = _mean_l2(pred, target)[:, 1:]
    return {
        "rmse": _rmse(pred, target)[:, 1:],
        "mean_l2": l2,
        "rel_l2": l2 / meta.length.view(-1, 1),
    }


def rollout_rmse(cfg: RegimeCfg) -> MetricFn:
    """Primary metric: rollout position error at each reported horizon.

    Reported at several horizons rather than one, because a model that is
    excellent for ten frames and diverges by a hundred is a different result from
    one that is mediocre throughout, and a single number hides which.

    **``rmse_h{h}`` averages predicted frames ``1..h``, never frame 0.** Frame 0
    of a rollout is the input state the window starts from, not a prediction, so
    averaging ``0..h-1`` folds a guaranteed zero into every horizon: ``rmse_h1``
    came out identically 0.0 for every model, seed and dataset, and the rest were
    biased low by 17% at h=10 down to 3.4% at h=100. A reported
    horizon now means ``h`` predicted frames, which is what a reader assumes it
    means.

    **A rollout of ``R`` frames therefore scores horizons up to ``R - 1``**, and a
    horizon past that is dropped rather than faked. A caller wanting ``h`` wants a
    rollout of ``h + 1``.

    **Two reductions of the same per-node distances, both reported.**
    ``rmse_h{h}`` is their root-mean-square and ``mean_l2_h{h}`` their mean, the
    latter being what the DLO literature calls per-node L2. They are not a
    rescaling of one another: the ratio is how
    unevenly error is spread across the cable, measured at 1.00 to 2.71 on real
    rollouts, so an arm whose free end diverges ranks differently under each.
    Both are in metres.
    """

    def metric(pred: Tensor, target: Tensor, meta: BatchMeta, labels: Labels) -> dict[str, float]:
        del labels
        l2 = _mean_l2(pred, target)[:, 1:]
        families = {
            "rmse": _rmse(pred, target)[:, 1:],
            "mean_l2": l2,
            # Divided per sample by that sample's own length, before any average
            # over the batch. Dividing a pooled error by a pooled length would
            # report the population's size range as if it were model behaviour.
            "rel_l2": l2 / meta.length.view(-1, 1),
        }
        if l2.shape[1] == 0:
            # A one-frame rollout ran no forward pass, so there is no prediction
            # to score. NaN rather than 0.0: the aggregate reads NaN as "no
            # window defined this", while 0.0 would read as a perfect model.
            return {f"{name}_full": float("nan") for name in families}

        horizons = [h for h in cfg.horizons if h <= l2.shape[1]]
        scores: dict[str, float] = {}
        for name, values in families.items():
            for h in horizons:
                scores[f"{name}_h{h}"] = float(values[:, :h].mean())
                # Frame h is index h-1 here, because `values` already starts at
                # frame 1. The terminal reading is 3.8x the cumulative one at
                # h100 on a real rollout, which is why both are reported and why
                # they may never share a name.
                scores[f"{name}_at_h{h}"] = float(values[:, h - 1].mean())
            scores[f"{name}_full"] = float(values.mean())
        return scores

    return metric


def link_length(cfg: RegimeCfg) -> MetricFn:
    """Physics violation, no ground truth needed: do the links keep their length?

    The reference is the cable's own segment length, ``length / E``, not the true
    trajectory — which is what lets this score a rollout with no reference at all.
    Reported as a fraction of rest length, so cables of different length are
    comparable.

    **Reported only among the chain-edge models**, per the spec, so that it
    reflects the attention bias rather than the model's access to a chain edge
    set. That is a reporting rule for the runner; this function scores whatever
    it is handed.
    """
    del cfg

    def metric(pred: Tensor, target: Tensor, meta: BatchMeta, labels: Labels) -> dict[str, float]:
        del target, labels
        rest = (meta.length / E).view(-1, 1, 1)
        lengths = (pred[..., 1:, :] - pred[..., :-1, :]).norm(dim=-1)
        drift = (lengths - rest).abs() / rest
        return {
            "link_length_mean_drift": float(drift.mean()),
            "link_length_max_drift": float(drift.max()),
        }

    return metric


def self_intersection(cfg: RegimeCfg) -> MetricFn:
    """Physics violation, no ground truth needed: did the cable pass through itself?

    Eq. 3 of the spec. For non-adjacent segments, the centreline distance must
    stay above the cable diameter ``D``, read per cable from ``meta.diameter``.

    Three numbers, because they say different things. The **violating-frame
    fraction** is how often it happens. The **worst penetration** is how badly.
    And **persistence** separates a transient numerical excursion from the cable
    genuinely ending up on the wrong side of itself — a single bad frame that
    recovers is not the same failure as a cable that stays crossed.

    This is the sharpest test of the contact regime, and it gives variant B a
    directional prediction: a 3D-distance prior should *reduce* self-intersection
    relative to variant A and to the chain-metric variants, which are blind to 3D
    proximity by construction. A variant that wins on RMSE while producing a
    self-crossing cable has not learned the physics.
    """

    def metric(pred: Tensor, target: Tensor, meta: BatchMeta, labels: Labels) -> dict[str, float]:
        del target, labels
        distance = segment_distance(pred)
        far = chain_far_mask(2, pred.device)  # |i-j| > 2, per Eq. 3
        diameter = meta.diameter.view(-1, 1, 1, 1)

        penetration = (diameter - distance).where(far, torch.zeros_like(distance))
        violating = penetration > 0
        per_frame = violating.flatten(start_dim=2).any(-1)  # [B, R]

        # Persistence: two consecutive violating frames. One frame that recovers
        # is a numerical excursion; a cable that stays crossed is not.
        persists = (per_frame[:, 1:] & per_frame[:, :-1]).any() if per_frame.shape[1] > 1 else False

        return {
            "selfint_frame_fraction": float(per_frame.float().mean()),
            "selfint_pair_fraction": float(
                violating.sum() / far.sum().clamp_min(1) / per_frame.numel()
            ),
            "selfint_worst_penetration": float(penetration.max().clamp_min(0.0)),
            "selfint_persists": float(bool(persists)),
        }

    return metric


def error_by_phase(cfg: RegimeCfg) -> MetricFn:
    """Decomposition: error in contact frames against free-flight frames.

    The phase label arrives **inside the labels**, computed by the loader on the
    ground-truth window. Recomputing it here would fork the definition, so this
    reads and never derives.

    The point of the whole project lives in this split: the two priors under test
    serve opposite regimes, so an aggregate number averages them to nothing.
    """
    del cfg

    def metric(pred: Tensor, target: Tensor, meta: BatchMeta, labels: Labels) -> dict[str, float]:
        del meta
        per_frame = _rmse(pred, target)
        contact = labels.contact_frame
        return {
            "rmse_contact": _masked_mean(per_frame, contact),
            "rmse_free_flight": _masked_mean(per_frame, ~contact),
            "contact_frame_fraction": float(contact.float().mean()),
        }

    return metric


def error_by_floor_phase(cfg: RegimeCfg) -> MetricFn:
    """The paper's phase split: rel l2 on floor-contact frames against free flight.

    A predicted frame is *floor contact* when the **recorded** cable has any
    vertex at or below one diameter above the floor (``min_i z_i <= D``), and
    *free flight* otherwise. The label is read off the ground truth, never the
    prediction, and only frames after the input frame count. Each value is the
    mean of ``rel_l2(t)`` over that rollout's selected frames, NaN when the
    rollout has none, so the aggregate over windows is the equal-weight mean over
    rollouts with at least one selected frame and the count says how many there
    were.

    Distinct from :func:`error_by_phase`, whose ``contact`` is the loader's
    self-contact proximity label.
    """
    del cfg

    def metric(pred: Tensor, target: Tensor, meta: BatchMeta, labels: Labels) -> dict[str, float]:
        del labels
        rel = _mean_l2(pred, target)[:, 1:] / meta.length.view(-1, 1)  # [B, R-1]
        lowest = target[:, 1:, :, 2].amin(dim=-1)  # [B, R-1], recorded trajectory
        floor = lowest <= meta.diameter.view(-1, 1)
        return {
            "rel_l2_floor_contact": _masked_mean(rel, floor),
            "rel_l2_free_flight": _masked_mean(rel, ~floor),
            "floor_contact_frame_fraction": float(floor.float().mean())
            if floor.numel()
            else float("nan"),
        }

    return metric


def error_by_pair_regime(cfg: RegimeCfg) -> MetricFn:
    """Decomposition: error at nodes involved in a contact pair, against the rest.

    A *contact pair* is 3D-close and chain-far; chain neighbours are the contrast
    class. The pair labels arrive in ``labels.contact_pair`` as ``[B, R, E, E]``
    over **segments**, and error is per **node**, so a node counts as involved
    when either of its two segments is in a contact pair.

    **This is an interpretation.** The spec names the decomposition and fixes the
    labels but does not say how a pair-level regime attributes to a node-level
    error. Written down here rather than left implicit; if a different
    attribution is preferred, this is the one function to change.
    """
    del cfg

    def metric(pred: Tensor, target: Tensor, meta: BatchMeta, labels: Labels) -> dict[str, float]:
        del meta
        per_node = (pred - target).square().sum(-1).sqrt()  # [B, R, N]
        in_pair = labels.contact_pair.any(-1)  # [B, R, E] segment involved
        node = torch.zeros_like(per_node, dtype=torch.bool)
        node[..., :-1] |= in_pair
        node[..., 1:] |= in_pair
        return {
            "rmse_contact_pair_nodes": _masked_mean(per_node, node),
            "rmse_other_nodes": _masked_mean(per_node, ~node),
        }

    return metric


def error_by_chain_distance(cfg: RegimeCfg) -> MetricFn:
    """Decomposition: error against distance along the cable from vertex 0.

    Reported in four equal bands rather than 33 numbers, because the claim is
    about a trend — does error grow with distance from the reference end — and a
    per-vertex table is not readable in a five-page paper.

    **Vertex 0 is the reference end.** The spec says "from the released end"; both
    ends are released in this dataset, so a convention is needed and this is it.
    """
    del cfg

    def metric(pred: Tensor, target: Tensor, meta: BatchMeta, labels: Labels) -> dict[str, float]:
        del meta, labels
        per_node = (pred - target).square().sum(-1).sqrt()  # [B, R, N]
        bands = torch.chunk(per_node, 4, dim=-1)
        return {f"rmse_band{i}": float(band.mean()) for i, band in enumerate(bands)}

    return metric


def _masked_mean(values: Tensor, mask: Tensor) -> float:
    """Mean of ``values`` where ``mask``, or NaN when the mask selects nothing.

    NaN rather than zero on purpose: "no contact frames in this rollout" and "no
    error in the contact frames" are opposite findings, and zero would report the
    first as the second.
    """
    if not bool(mask.any()):
        return float("nan")
    return float(values[mask].mean())


METRIC_BUILDERS: dict[str, Callable[[RegimeCfg], MetricFn]] = {
    "rollout_rmse": rollout_rmse,
    "link_length": link_length,
    "self_intersection": self_intersection,
    "error_by_phase": error_by_phase,
    "error_by_floor_phase": error_by_floor_phase,
    "error_by_pair_regime": error_by_pair_regime,
    "error_by_chain_distance": error_by_chain_distance,
}
"""Explicit registration, one entry per metric the spec names.

Builders rather than functions, because ``horizons`` is bound at registration and
must never be a module constant. Adding an entry here that ``docs/metrics.md``
does not name is a defect."""


def build_metrics(cfg: RegimeCfg | None = None) -> dict[str, MetricFn]:
    """The registry the runner scores with: ``name -> fn``, thresholds bound in.

    Args:
        cfg: The horizons to report at; the spec's list if omitted. The contact
            thresholds are not here and cannot be passed in: they were applied by
            the loader when it built the labels this registry scores against.
    """
    resolved = cfg or RegimeCfg()
    return {name: build(resolved) for name, build in METRIC_BUILDERS.items()}


def score(
    pred: Tensor, target: Tensor, meta: BatchMeta, labels: Labels, cfg: RegimeCfg | None = None
) -> dict[str, float]:
    """Every metric, flattened into one dict. The one call a runner needs.

    Args:
        pred: ``[B, R, N, 3]`` the rollout.
        target: ``[B, R, N, 3]`` the ground truth over the same window.
        meta: Per-cable constants; ``diameter`` and ``length`` are read.
        labels: Phase and pair labels, aligned frame for frame with ``pred``.
        cfg: Thresholds and horizons.

    Raises:
        ValueError: if the rollout shapes or the label axis disagree.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must match, got {tuple(pred.shape)} vs {tuple(target.shape)}"
        )
    if pred.ndim != 4:
        raise ValueError(f"rollouts are [B, R, N, 3], got {tuple(pred.shape)}")
    if labels.contact_frame.shape != pred.shape[:2]:
        raise ValueError(
            f"labels carry R={tuple(labels.contact_frame.shape)}, rollout is {tuple(pred.shape[:2])}"
        )

    scores: dict[str, float] = {}
    for fn in build_metrics(cfg).values():
        scores |= fn(pred, target, meta, labels)
    return scores
