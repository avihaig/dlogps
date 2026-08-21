"""The frozen boundary types every module agrees on: ``Batch`` and ``Labels``.

The frozen boundary types every other module is written against: **code that
drifts from these shapes is the bug, not the shapes**,
and a change lands there first by PR.

These live here rather than in the model package because the contract assigns
them to the loader, and the block and the harness cannot even be type-annotated
without them. Written against the frozen contract, so the loader fills in the
pipeline that produces these and deletes no field.

Batch-first, ``float32``, on device. ``float64`` lives on disk only
(``docs/data.md``).
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

N = 33
"""Vertices per cable. Fixed by the discrete-elastic-rod discretization."""

E = 32
"""Segments per cable, i.e. ``N - 1``. Note ``edge_index`` carries ``2 * E``
columns, since every chain edge is stored in both directions."""

# The velocity-history depth C is deliberately absent. N and E are fixed by the
# DER discretization; C is a hyperparameter, so it lives on TrainCfg and reaches
# the layout functions as an argument (``docs/method.md``).


@dataclass(frozen=True)
class BatchMeta:
    """Per-sample constants that are not node features.

    Attributes:
        cable_id: ``[B]`` int64. Row of the population CSV this sample came from.
        length: ``[B]`` the DLO's length [m], a declared knob rather than a
            measurement. Normalizes the 3D-distance bias, so it is a contract
            field rather than a model constant.
        diameter: ``[B]`` diameter [m], the self-intersection threshold ``D``.
        bend_stiffness: ``[B]`` Young's modulus E [Pa].
        joint_damping: ``[B]`` chain joint damping [N*m*s/rad].
        effective_linear_density: ``[B]`` measured effective linear density [kg/m], **never**
            the nominal density.
        dt: ``[B]`` frame interval [s]. From ``np.diff(t)`` on the recorded time
            channel, **never** reconstructed from ``record_hz``.
        is_held_out: ``[B]`` bool, the split tag.

    The four material params — ``diameter``, ``bend_stiffness``, ``joint_damping``,
    ``effective_linear_density`` — are also broadcast per node into ``node_feat``. The
    duplication is what makes the rollout hook possible: ``features_from_state``
    is handed only a ``State`` and this, and must return a ``Batch`` whose
    ``node_feat`` holds all four, so any of them missing here is unreachable at
    rollout.
    """

    cable_id: Tensor
    length: Tensor
    diameter: Tensor
    bend_stiffness: Tensor
    joint_damping: Tensor
    effective_linear_density: Tensor
    dt: Tensor
    is_held_out: Tensor


@dataclass(frozen=True)
class Batch:
    """One training batch: ``B`` cables at one frame each.

    Attributes:
        pos: ``[B, N, 3]`` current vertex positions [m]. **Not a node feature.**
            Position reaches the model only through ``edge_feat`` and the bias
            plugin, which is what keeps the model translation invariant.
        node_feat: ``[B, N, F]``. ``F`` is the loader's business: velocity
            history, four material params and the clipped floor distance. The
            block reads ``F`` off this tensor and hard-codes nothing.
        edge_index: ``[2, 2 * E]`` int64, the static chain, **both directions**.
            One scatter then covers both message directions; a model that
            assumes a single direction silently drops half the messages.
        edge_feat: ``[B, 2 * E, 5]``: displacement (3), distance (1), rest
            length (1).
        target: ``[B, N, 3]`` next vertex position [m].
        meta: The per-sample constants above.
    """

    pos: Tensor
    node_feat: Tensor
    edge_index: Tensor
    edge_feat: Tensor
    target: Tensor
    meta: BatchMeta


@dataclass(frozen=True)
class Labels:
    """Phase and pair-regime labels, aligned frame for frame with a rollout.

    Computed on the ground-truth rollout window, never on the single training
    frame, and never recomputed by the harness.

    Attributes:
        contact_frame: ``[B, R]`` bool, the phase label per rollout frame.
        contact_pair: ``[B, R, E, E]`` bool, segment pairs that are 3D-close and
            chain-far.
    """

    contact_frame: Tensor
    contact_pair: Tensor
