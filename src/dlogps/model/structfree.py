"""Structure-free arms: the floor the variant table is read against.

**These are whole models, not bias plugins.** A variant changes one term inside
the attention softmax; these arms have no attention, no message passing and no
edges at all. They exist to answer what *any* relational computation buys, which
the variant matrix cannot ask of itself.

**The global arm is handed positions; the per-node arm is not.** The only
geometry in ``node_feat`` is a clipped floor distance, which is ``pos[..., 2]``
and so says how high a node is and nothing about which nodes are near each other;
relative geometry reaches the block through ``edge_feat`` and the bias plugin,
neither of which exists here. An arm fed ``node_feat`` alone can therefore
represent floor contact but not self-contact, which is the regime the whole
experiment is about, so its floor would measure a missing input rather than
missing structure. The global arm accordingly also receives ``pos - mean(pos)``,
divided by the cable's own length: translation invariant exactly as the block's
``edge_feat`` displacement is, and dimensionless so cables of different length
enter on one scale. The per-node arm receives no position term at all: it is the
no-communication floor, and a centroid is a global aggregate, so feeding it one
would smuggle communication back in.

**They predict the standardized step, like the block.** The trainer copies the
run's step statistics into ``step_std`` / ``step_mean`` on whatever model it
built, so those buffer names are an interface, not a detail, and the head's
output is scaled by them before it is added to ``pos``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn

from dlogps.data.dataset import feature_width, state_from_batch
from dlogps.data.types import Batch, N
from dlogps.model.gps import GPSCfg, GPSModel

STATIC_FEATURES = 5
"""Material params (4) and the clipped floor distance (1): the tail of
``node_feat`` that is not velocity history. The velocity block ahead of it is
``C * 3`` wide, which is what makes ``C`` recoverable from ``F``."""

POSITION_FEATURES = 3
"""Centroid-relative position channels the global arm receives per node."""


def _mlp(sizes: Sequence[int]) -> nn.Sequential:
    """A ReLU stack over the given layer widths, linear on the output.

    Args:
        sizes: Input width, hidden widths, output width. At least two entries.

    Returns:
        The stack, with no activation after the last linear so the head can
        produce a signed displacement.
    """
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class StructureFree(nn.Module):
    """What the three arms share: the step scaling and the width check.

    Subclasses implement :meth:`predict_step`, returning the **standardized**
    displacement. This class owns the conversion back to an absolute position,
    so the arms cannot disagree with the block about what a head predicts.
    """

    # Declared, because ``nn.Module.__getattr__`` types a registered buffer as
    # ``Tensor | Module`` and every use would otherwise need a cast.
    step_std: Tensor
    step_mean: Tensor

    def __init__(self, feature_width_in: int, cfg: GPSCfg) -> None:
        """
        Args:
            feature_width_in: ``F``, the width of ``batch.node_feat``. Off the
                data, never a constant here, for the same reason the block takes
                it as an argument: it moves with the history depth ``C``.
            cfg: The config this arm was built from. Held under the same name the
                block holds it, because ``save_checkpoint`` reads ``model.cfg``
                and ``model.feature_width`` off whatever the trainer built; an
                arm missing either cannot be checkpointed.
        """
        super().__init__()
        self.feature_width = feature_width_in
        self.cfg = cfg
        # Identity defaults, so an arm built without statistics behaves as an
        # unscaled one and the trainer's copy is what binds the run's scaling.
        self.register_buffer("step_std", torch.ones(3))
        self.register_buffer("step_mean", torch.zeros(3))

    def predict_step(self, batch: Batch) -> Tensor:
        """``[B, N, 3]`` standardized displacement. Implemented by each arm."""
        raise NotImplementedError

    def forward(self, batch: Batch) -> Tensor:
        """Predict the next vertex position for every node.

        Args:
            batch: A **normalized** contract-shaped batch, as the trainer and
                ``rollout`` build it. ``pos`` and ``meta`` are unscaled by
                construction (``harness/normalize.py``).

        Returns:
            ``[B, N, 3]`` absolute next positions, matching ``batch.target``.

        Raises:
            ValueError: if the batch's feature width is not the one this arm was
                built for.
        """
        if batch.node_feat.shape[-1] != self.feature_width:
            raise ValueError(
                f"arm built for F={self.feature_width}, got F={batch.node_feat.shape[-1]}"
            )
        return batch.pos + (self.predict_step(batch) * self.step_std + self.step_mean)


def relative_positions(batch: Batch) -> Tensor:
    """``[B, N, 3]`` positions about their own centroid, in cable lengths.

    Translation invariant, so the arm cannot learn the drop location, and
    dimensionless, so a 0.8 m cable and a 1.6 m one enter on one scale.

    Args:
        batch: Any contract-shaped batch. ``pos`` is unscaled even in a
            normalized batch, so no inverse is needed.
    """
    centred = batch.pos - batch.pos.mean(dim=1, keepdim=True)
    return centred / batch.meta.length.view(-1, 1, 1)


class PerNodeMLP(StructureFree):
    """Each node predicts its own step from its own features. No communication.

    **The floor under communication itself.** Weights are shared across nodes and
    the node axis is never mixed, so node ``i``'s output cannot depend on node
    ``j``. That is the property ``tests/test_structfree.py`` pins.

    It gets no position term, deliberately: see this module's docstring.
    """

    def __init__(
        self, feature_width_in: int, hidden: int, n_layers: int = 3, cfg: GPSCfg | None = None
    ) -> None:
        """
        Args:
            feature_width_in: ``F``.
            hidden: Width of every hidden layer.
            n_layers: Linear layers in total, so ``n_layers - 1`` hidden ones.
            cfg: The config this arm was built from, for the checkpoint writer.
                Reconstructed from ``hidden`` and ``n_layers`` if omitted, so a
                directly-built arm still carries a config that describes it.
        """
        super().__init__(
            feature_width_in,
            cfg or GPSCfg(d_model=hidden, n_layers=n_layers, arch="mlp_pernode"),
        )
        sizes = [feature_width_in] + [hidden] * (n_layers - 1) + [3]
        self.net = _mlp(sizes)

    def predict_step(self, batch: Batch) -> Tensor:
        """``[B, N, 3]``, each node from its own row of ``node_feat``."""
        return self.net(batch.node_feat)


class GlobalLSTM(StructureFree):
    """An LSTM over the frame axis of the whole cable. Time, no structure.

    **The history is a sequence here, not stacked channels.** The MLP arms see
    ``C`` velocity frames as ``C * 3`` independent inputs; this arm sees them in
    order, oldest step first so the final hidden state is the newest frame. The
    static channels and the positions are constant along the sequence and are
    repeated at every step rather than appended once, so no step is a different
    shape from another.

    **The velocity it reads is standardized and it never integrates it.** This
    arm consumes velocity only as a feature, so the scaled value is the right
    input and no inverse is wanted.
    """

    def __init__(
        self,
        feature_width_in: int,
        hidden: int,
        history_frames: int,
        n_layers: int = 2,
        n_nodes: int = N,
        cfg: GPSCfg | None = None,
    ) -> None:
        """
        Args:
            feature_width_in: ``F``.
            hidden: LSTM hidden width.
            history_frames: ``C``, the depth the batches it will be handed were
                built at. Needed at construction because it sets the sequence
                length, and checked against every batch so a checkpoint
                evaluated at another depth raises rather than reshaping one
                frame into another.
            n_layers: Stacked LSTM layers.
            n_nodes: ``N``. An argument so a test can build a small arm.
            cfg: The config this arm was built from, for the checkpoint writer.
                Reconstructed from ``hidden`` and ``n_layers`` if omitted.

        Raises:
            ValueError: if ``history_frames`` does not produce ``feature_width_in``.
        """
        super().__init__(
            feature_width_in,
            cfg or GPSCfg(d_model=hidden, n_layers=n_layers, arch="lstm_global"),
        )
        if feature_width(history_frames) != feature_width_in:
            raise ValueError(
                f"a {history_frames}-frame history makes F={feature_width(history_frames)}, "
                f"but this arm was asked for F={feature_width_in}"
            )
        self.n_nodes = n_nodes
        self.history_frames = history_frames
        per_step = n_nodes * (3 + STATIC_FEATURES + POSITION_FEATURES)
        self.lstm = nn.LSTM(per_step, hidden, num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(hidden, n_nodes * 3)

    def predict_step(self, batch: Batch) -> Tensor:
        """``[B, N, 3]`` from the last hidden state of the frame sequence."""
        b = batch.node_feat.shape[0]
        # The loader owns the node_feat layout; the velocity block comes back
        # through its inverse rather than being sliced a second time here.
        history = state_from_batch(batch, self.history_frames).vel_history
        # Newest first on the way in, oldest first on the way through the LSTM,
        # so the final hidden state is the frame the step is predicted from.
        sequence = history.flip(1).reshape(b, self.history_frames, -1)

        static = batch.node_feat[..., self.history_frames * 3 :]
        constant = torch.cat([static, relative_positions(batch)], dim=-1).reshape(b, 1, -1)
        joined = torch.cat([sequence, constant.expand(-1, self.history_frames, -1)], dim=-1)

        out, _ = self.lstm(joined)
        return self.head(out[:, -1]).reshape(b, self.n_nodes, 3)


ARCHS: dict[str, Callable[[int, GPSCfg, int], StructureFree]] = {
    "mlp_pernode": lambda width, cfg, c: PerNodeMLP(width, cfg.d_model, cfg.n_layers, cfg),
    "lstm_global": lambda width, cfg, c: GlobalLSTM(width, cfg.d_model, c, cfg.n_layers, N, cfg),
}
"""Arch name to a factory over ``(F, cfg, C)``.

Separate from the variant registry on purpose: a variant is a plugin inside the
block, and these are trained models with no graph."""


def build_structfree(
    arch: str, feature_width_in: int, cfg: GPSCfg, history_frames: int
) -> StructureFree:
    """One structure-free arm by name.

    ``d_model`` is reused as the hidden width and ``n_layers`` as the depth, so
    the speed-matched widths ride the fields every config already carries and no
    parallel width field can drift from them. ``n_heads``, ``use_global`` and
    ``use_local`` are meaningless here and are ignored.

    Args:
        arch: A key of :data:`ARCHS`.
        feature_width_in: ``F``, read off ``node_feat`` by the caller.
        cfg: The model config; ``d_model`` and ``n_layers`` are read from it.
        history_frames: ``C``, needed by the sequence arm.

    Returns:
        An untrained arm on the CPU.

    Raises:
        ValueError: naming the known arches, since a typo would otherwise read
            as an unimplemented arm rather than a misspelt one.
    """
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}; known: {sorted(ARCHS)}")
    return ARCHS[arch](feature_width_in, cfg, history_frames)


TrainableModel = GPSModel | StructureFree
"""What ``build_model`` can return, and what the trainer and the checkpoint
writer are typed against.

A union rather than ``nn.Module`` because both members declare ``cfg``,
``feature_width`` and the two step buffers as typed attributes, and
``nn.Module.__getattr__`` types every one of them as ``Tensor | Module``: widening
the annotation would push a cast onto every line that copies statistics in or
writes a checkpoint out."""
