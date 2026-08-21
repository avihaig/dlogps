"""One cable on disk to model-ready windows.

**Padding never reaches memory.** ``episodes.npz`` is rectangular and padded so
the episode boundary is the array's own shape, and the files are compressed so
that padding costs nothing on disk. Held in RAM it is ~3x the real frames, so
:func:`load_cable` slices to ``episode_lengths[e]`` and keeps only real frames
(``docs/data.md``).

**A sample is a ``C + 1``-frame window**, not a frame and not an episode: ``C``
frames of velocity history plus the next position as the target. At the default
``C = 5`` a 624-frame episode yields 619 of them. A window never crosses
``episode_lengths[e]``, because a pair spanning two episodes joins a settled pile
to a fresh airborne cable, a transition that cannot physically occur.

**``C`` is an argument here, never a module constant.** Every function that
writes or reads the ``node_feat`` layout takes ``history_frames`` explicitly and
none of them defaults it, so a caller that has forgotten to thread the run's
depth fails at the call rather than silently building features at somebody
else's ``C``. The one default in the project is :data:`DEFAULT_HISTORY_FRAMES`,
and it is only ever the default of ``TrainCfg.history_frames``.

**One function builds the features for both callers.** The dataset fills
``vel_history`` from the recorded channel and the rollout loop differences its
own predictions, but both go through :func:`features_from_state`, so training
features and rollout features cannot skew apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from dlogps.data.types import Batch, BatchMeta, E, N

DEFAULT_HISTORY_FRAMES = 5
"""Velocity-history depth ``C`` a run gets when its config does not say otherwise.

**A default, not a constant.** Nothing in the feature pipeline reads it: it is the
default of ``TrainCfg.history_frames`` and nothing else, so the depth a run used
is a property of that run's config and of the checkpoint it wrote. Five is the
value the whole experiment matrix was specified at
(``docs/method.md``), so leaving it alone reproduces every result
recorded before the depth became configurable.
"""

MATERIAL_FIELDS = ("bend_stiffness", "joint_damping", "diameter", "effective_linear_density")
"""The four, in the order ``node_feat`` carries them. Pinned by ``tests/test_fake_data.py``."""

FLOOR_CLIP_M = 1.0
"""Ceiling on the floor-distance feature: above this, how high stops mattering."""

MAX_STEP_M = 0.05
"""Per-frame node displacement above which an episode is a solver blow-up, dropped.

**A physics filter, not a length filter, and the gap it sits in is empty.** Over
all 4,600 episodes of ``release_v1`` and ``test_v1``, exactly three exceed this
and every other episode stays under 2.9 cm: a factor of sixty with nothing in it,
so no defensible threshold in that gap changes which episodes are kept. The three
reach 3.17 m, 3.37 m and 1.80 m in a single frame at 500 Hz, which is a cable
crossing the scene between frames rather than falling through it.

**Dropping them is a statistics decision before it is a hygiene one.** They are
0.13% of the frames and they set ``step_std``, which sets the units the loss is
taken in and therefore the 1.0 reference line itself: pooled training loss reads
3.21 with them and 1.2e-3 without. Length would be the tempting filter and it
does not work, because one of the three sits at 1309 frames, inside the normal
band.
"""


def feature_width(history_frames: int) -> int:
    """``F``: the ``node_feat`` width a ``C``-frame history produces.

    Velocity history (``C * 3``), the four material params, one clipped floor
    distance.

    Args:
        history_frames: ``C``, the depth of the velocity history.

    Raises:
        ValueError: if the depth is not at least one frame, which would leave the
            model no velocity at all and ``roll_in`` nothing to prepend to.
    """
    if history_frames < 1:
        raise ValueError(
            f"history_frames is a frame count and must be at least 1, got {history_frames}"
        )
    return history_frames * 3 + len(MATERIAL_FIELDS) + 1


def history_frames_from_width(width: int) -> int:
    """``C``: the depth that produced a ``node_feat`` of this width.

    The inverse of :func:`feature_width`, here rather than at the call site so
    the layout stays this file's knowledge. A caller handed only ``F`` needs it
    when the model it builds depends on the sequence length, which the
    structure-free sequence arm does.

    Args:
        width: ``F``, off ``node_feat``.

    Raises:
        ValueError: if no whole depth produces this width, which means the batch
            was not built by :func:`features_from_state` at all.
    """
    velocity = width - len(MATERIAL_FIELDS) - 1
    if velocity < 3 or velocity % 3:
        raise ValueError(
            f"F={width} is not a width any velocity-history depth produces; "
            f"the static tail is {len(MATERIAL_FIELDS) + 1} and the history is a multiple of 3"
        )
    return velocity // 3


@dataclass(frozen=True)
class State:
    """What moves. Everything else about a cable is in :class:`BatchMeta`.

    Attributes:
        pos: ``[B, N, 3]`` current vertex positions.
        vel_history: ``[B, C, N, 3]`` velocity history, **newest first**. ``C`` is
            the run's configured depth, not a fixed number.
    """

    pos: Tensor
    vel_history: Tensor


@dataclass(frozen=True)
class CableData:
    """One cable's episodes, padding stripped, ready to window.

    Attributes:
        episodes: Per episode, ``(vertex_pos, vertex_vel, t)`` at that episode's
            real length. A list rather than an array because lengths differ.
        params: The six per-cable constants, straight off the npz.
        cable_id: Population row id.
        dropped_episodes: Indices **in the file** of episodes :data:`MAX_STEP_M`
            rejected. Recorded rather than discarded silently, since a filtered
            dataset and an unfiltered one are otherwise indistinguishable in a
            result table, and the episode numbering here no longer matches the
            npz once anything is dropped.
    """

    episodes: list[tuple[Tensor, Tensor, Tensor]]
    params: dict[str, float]
    cable_id: int
    dropped_episodes: tuple[int, ...] = ()

    @property
    def n_episodes(self) -> int:
        """Episodes in this cable."""
        return len(self.episodes)

    def dt(self, episode: int) -> float:
        """Frame interval [s] from ``np.diff(t)``, never ``record_hz``.

        The requested rate is quantized to whole sim steps, so the stored
        ``record_hz`` is 5% off the achieved one and a loader trusting it biases
        every finite-difference velocity (``docs/data.md``).
        """
        return float(self.episodes[episode][2].diff().mean())


def load_cable(cable_dir: str | Path, *, max_step_m: float | None = MAX_STEP_M) -> CableData:
    """Read one ``cable_XXX/episodes.npz``, stripping padding and blow-ups.

    Args:
        cable_dir: Directory holding ``episodes.npz``.
        max_step_m: Per-frame node displacement above which an episode is
            rejected as a solver divergence, or ``None`` to keep everything. The
            default is :data:`MAX_STEP_M`; ``None`` is for the diagnostic that
            counts what the filter removes.

    Returns:
        The cable's real frames, its six constants, and which episodes were
        dropped.
    """
    with np.load(Path(cable_dir) / "episodes.npz") as loaded:
        lengths = loaded["episode_lengths"]
        pos, vel, t = loaded["vertex_pos"], loaded["vertex_vel"], loaded["t"]
        params = {
            "length": float(loaded["length"]),
            "diameter": float(loaded["diameter"]),
            "bend_stiffness": float(loaded["bend_stiffness"]),
            "joint_damping": float(loaded["joint_damping"]),
            "effective_linear_density": float(loaded["effective_linear_density"]),
        }
        cable_id = int(loaded["cable_id"])

    episodes: list[tuple[Tensor, Tensor, Tensor]] = []
    dropped: list[int] = []
    for e, n in enumerate(int(x) for x in lengths):
        frames = (
            torch.from_numpy(pos[e, :n]).float(),
            torch.from_numpy(vel[e, :n]).float(),
            torch.from_numpy(t[e, :n]).float(),
        )
        if max_step_m is not None and _worst_step(frames[0]) > max_step_m:
            dropped.append(e)
            continue
        episodes.append(frames)

    return CableData(
        episodes=episodes,
        params=params,
        cable_id=cable_id,
        dropped_episodes=tuple(dropped),
    )


def _worst_step(pos: Tensor) -> float:
    """Largest single-frame node displacement [m] in an episode.

    The divergence signature: a solver that has lost the cable moves a vertex
    metres between two frames recorded 2 ms apart, while a physical whip stays
    under 3 cm.
    """
    if pos.shape[0] < 2:
        return 0.0
    return float((pos[1:] - pos[:-1]).norm(dim=-1).max())


def window_starts(n_frames: int, history_frames: int) -> range:
    """Frames that can start a sample, given ``C`` history and one target.

    Needs ``k >= C - 1`` behind it and ``k + 1 < n_frames`` ahead, so the first
    ``C - 1`` frames and the last frame of every episode are not window starts.
    A deeper history therefore costs samples at both ends of every episode.

    Args:
        n_frames: Real frames in the episode, padding already stripped.
        history_frames: ``C``, the depth of the velocity history.
    """
    first = history_frames - 1
    return range(first, max(n_frames - 1, first))


def chain_edge_index(device: torch.device | None = None) -> Tensor:
    """``[2, 2 * E]`` static chain, **both directions**.

    One scatter then covers both message directions; a model that assumes a
    single direction silently drops half the messages.
    """
    seg = torch.arange(E, device=device)
    src = torch.cat([seg, seg + 1])
    dst = torch.cat([seg + 1, seg])
    return torch.stack([src, dst])


def features_from_state(state: State, meta: BatchMeta, history_frames: int) -> Batch:
    """Build a contract-shaped :class:`Batch` from state plus per-cable constants.

    The rollout hook, and the same code path the dataset uses. ``target`` is left
    as the current position: a rollout has no ground truth, and the dataset
    overwrites it with the real next frame.

    **Position never enters ``node_feat``.** It reaches the model only through
    the relative edge features and the bias plugin, which is what keeps the model
    translation invariant.

    Args:
        state: Positions and the velocity history, newest first.
        meta: The per-cable constants, four of which are broadcast per node.
        history_frames: ``C``. The state's history must be exactly this deep;
            this is the check that catches a rollout stepping at one depth
            against features built at another.

    Raises:
        ValueError: if the history is not ``C`` frames deep.
    """
    if state.vel_history.shape[1] != history_frames:
        raise ValueError(
            f"vel_history must carry {history_frames} frames, got {state.vel_history.shape[1]}"
        )

    b = state.pos.shape[0]
    material = torch.stack([getattr(meta, name) for name in MATERIAL_FIELDS], dim=-1)
    floor_distance = state.pos[..., 2].clamp(min=0.0, max=FLOOR_CLIP_M).unsqueeze(-1)
    node_feat = torch.cat(
        [
            state.vel_history.permute(0, 2, 1, 3).reshape(b, N, history_frames * 3),
            material.unsqueeze(1).expand(b, N, len(MATERIAL_FIELDS)),
            floor_distance,
        ],
        dim=-1,
    )

    edge_index = chain_edge_index(state.pos.device)
    src, dst = edge_index[0], edge_index[1]
    disp = state.pos[:, dst] - state.pos[:, src]
    dist = disp.norm(dim=-1, keepdim=True)
    seg_rest = (meta.length / E).view(-1, 1, 1).expand_as(dist)

    return Batch(
        pos=state.pos,
        node_feat=node_feat,
        edge_index=edge_index,
        edge_feat=torch.cat([disp, dist, seg_rest], dim=-1),
        target=state.pos,
        meta=meta,
    )


def state_from_batch(batch: Batch, history_frames: int) -> State:
    """Recover the :class:`State` inside a batch. The inverse of the hook above.

    **The loader's, not the harness's**, for the same reason
    :func:`features_from_state` is: recovering ``vel_history`` means knowing the
    layout that function wrote into ``node_feat``, and that layout is exactly the
    knowledge the harness is contracted not to hold. One file writes it, the same
    file reads it, and the two cannot drift.

    Used twice: the rollout loop is handed a ``Batch`` by ``eval_seen_cables``
    and has to start from the state in it, and the input-noise pass perturbs the
    state rather than patching derived features.

    Args:
        batch: Any contract-shaped batch.
        history_frames: ``C``, the depth the batch's ``node_feat`` was written at.

    Returns:
        The positions and the velocity history, still newest first.

    Raises:
        ValueError: if ``node_feat`` is not the width this depth produces.
    """
    # An exact width, not a lower bound. Slicing the leading C * 3 columns of a
    # batch built at a deeper C reads part of one frame as another and returns a
    # correctly-shaped lie, which is the failure a checkpoint evaluated at the
    # wrong depth would otherwise produce. The same reasoning as the width
    # check in harness/noise.py.
    width = batch.node_feat.shape[-1]
    if width != feature_width(history_frames):
        raise ValueError(
            f"a {history_frames}-frame history makes node_feat "
            f"{feature_width(history_frames)} wide, got {width}; "
            "the batch was built at a different velocity-history depth"
        )

    b = batch.pos.shape[0]
    history = batch.node_feat[..., : history_frames * 3].reshape(b, N, history_frames, 3)
    return State(pos=batch.pos, vel_history=history.permute(0, 2, 1, 3))


def make_batch(
    cable: CableData,
    windows: list[tuple[int, int]],
    *,
    is_held_out: bool,
    history_frames: int,
) -> Batch:
    """Assemble one batch from ``(episode, frame)`` window starts of one cable.

    Args:
        cable: The loaded cable.
        windows: ``(episode, k)`` pairs; each yields one sample. Every ``k`` must
            be a :func:`window_starts` value for the same ``history_frames``, or
            the slice behind it runs off the front of the episode.
        is_held_out: The split tag, the same for every sample of one cable
            because the split holds out whole cable parameterizations.
        history_frames: ``C``, the depth of the velocity history.

    Returns:
        A contract-shaped :class:`Batch` with the real next frame as ``target``.
    """
    if not windows:
        raise ValueError("cannot build a batch from zero windows")

    pos = torch.stack([cable.episodes[e][0][k] for e, k in windows])
    # Newest first: index 0 is frame k, index C-1 is frame k - C + 1.
    vel_history = torch.stack(
        [cable.episodes[e][1][k - history_frames + 1 : k + 1].flip(0) for e, k in windows]
    )
    target = torch.stack([cable.episodes[e][0][k + 1] for e, k in windows])
    dt = torch.tensor([cable.dt(e) for e, _ in windows])

    b = len(windows)
    meta = BatchMeta(
        cable_id=torch.full((b,), cable.cable_id, dtype=torch.int64),
        length=torch.full((b,), cable.params["length"]),
        diameter=torch.full((b,), cable.params["diameter"]),
        bend_stiffness=torch.full((b,), cable.params["bend_stiffness"]),
        joint_damping=torch.full((b,), cable.params["joint_damping"]),
        effective_linear_density=torch.full((b,), cable.params["effective_linear_density"]),
        dt=dt,
        is_held_out=torch.full((b,), is_held_out, dtype=torch.bool),
    )

    batch = features_from_state(State(pos=pos, vel_history=vel_history), meta, history_frames)
    return Batch(
        pos=batch.pos,
        node_feat=batch.node_feat,
        edge_index=batch.edge_index,
        edge_feat=batch.edge_feat,
        target=target,
        meta=meta,
    )
