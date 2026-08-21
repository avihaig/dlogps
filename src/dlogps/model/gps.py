"""The GraphGPS block: one local stream, one global stream, summed per layer.

The architecture is held **fixed** across the whole experiment; only the bias
plugin varies (``dlogps.model.bias``). Everything here is therefore written to be
boring and to stay boring.

Each layer runs two streams in parallel rather than in sequence, which is the
paper's own design argument: a two-stage MPNN-then-attention scheme lets the
early message-passing layers irreparably smooth away information before
attention ever sees it (Rampášek et al., 2022, §3.3).

- The **local** stream is GatedGCN over the chain. It is the only place edge
  features enter the layer, and the only GraphGPS MPNN flavour that updates edge
  attributes as well as node ones (Eq. 7).
- The **global** stream is dense multi-head attention over all ``N`` nodes and
  sees **node features only** — no edge features, no masking (Eq. 8, App. C.2).
  This is where the bias is subtracted.
- Each branch gets dropout, a residual add of the layer input, and layer norm
  (Eq. 9-10 put a normalizer here; GraphGPS leaves which one a config option and
  ``_node_norm`` records why LayerNorm settles it at this batch size); the two
  are then summed and mixed by a two-layer MLP (Eq. 11).

Two switches make the ablations one flag rather than a second model:
``use_global=False`` is the **chain-only baseline** (message passing on the chain
edges, no attention), and ``use_local=False`` is the **local-stream-off**
ablation (global attention alone).

Plain PyTorch, no PyG: the chain topology is static and identical in every
sample, so dense attention over a fixed adjacency needs no batching library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from dlogps.data.types import Batch
from dlogps.model.bias import BiasPlugin, NoBias

EDGE_FEATURE_WIDTH = 5
"""``edge_feat`` is displacement (3), distance (1), rest length (1). A contract
constant, asserted rather than trusted, because a loader that appends a channel
without a contract change is exactly the drift this catches."""


@dataclass(frozen=True)
class GPSCfg:
    """Block hyper-parameters. Internals, deliberately absent from the contract.

    Attributes:
        d_model: Hidden width. The default is 96 rather than a rounder 128
            because ``H = 6`` has to divide it, and the head count is the one
            fixed by the experiment.
        n_heads: Attention heads ``H``. Six, so variants A through D differ by at
            most six learnable scalars, under 0.01% of parameters: any gap is
            then attributable to the bias rather than to capacity
            (``docs/method.md``).
        n_layers: Stacked GPS layers.
        dropout: Applied to each branch output and inside the MLP.
        attn_dropout: Applied to the attention weights.
        use_global: ``False`` switches the global stream off, which is the
            chain-only message-passing baseline.
        use_local: ``False`` switches the local stream off, the crossed ablation.
        arch: Which model family the run builds. ``gps`` is the block the rest of
            these fields describe. The structure-free floor arms
            (``model/structfree.py``) name themselves here and read only
            ``d_model``, as their hidden width, and ``n_layers``; they have no
            heads and no streams, so the two checks below do not apply to them
            and the experiment's head-count rule is not theirs to satisfy.
    """

    d_model: int = 96
    n_heads: int = 6
    n_layers: int = 4
    dropout: float = 0.0
    attn_dropout: float = 0.0
    use_global: bool = True
    use_local: bool = True
    arch: str = "gps"

    def __post_init__(self) -> None:
        if self.arch != "gps":
            return
        if not self.use_local and not self.use_global:
            raise ValueError("a layer with both streams off computes nothing")
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model={self.d_model} is not divisible by n_heads={self.n_heads}")


def _node_norm(d_model: int) -> nn.LayerNorm:
    """Normalization over the feature axis of a ``[B, N, d]`` tensor.

    **LayerNorm rather than BatchNorm, and the choice is load-bearing.** GraphGPS
    treats the normalizer as a config option and its released code offers both,
    so this stays inside the cited architecture; what settles it here is the
    batch. A batch of 8 cables gives BatchNorm only ``8 * 33 = 264`` pooled rows
    to estimate from, and it makes each sample's prediction depend on the other
    cables it happened to be batched with. Measured on the overfit probe, where
    that coupling is exactly what has to disappear: pooled loss 0.0387 under
    BatchNorm against 0.0170 under LayerNorm on one cable, and 0.0697 against
    0.00302 on another, the difference between failing and passing.

    The dependence is on the pooled-sample count and it is monotone: the
    BatchNorm penalty is 84x at batch 8, 7.6x at 32, 2.2x at 256. A larger batch
    would shrink it without removing it, and the batch size is not free to
    choose, so the normalizer is the thing that gives.
    """
    return nn.LayerNorm(d_model)


def _apply_node_norm(norm: nn.LayerNorm, x: Tensor) -> Tensor:
    """LayerNorm handles ``[B, N, d]`` natively, so the flatten is a no-op here."""
    return norm(x)


class GatedGCN(nn.Module):
    """The local stream: gated message passing along the cable, edges updated.

    ``eta_ij = sigmoid(C e_ij + D h_i + E h_j)`` gates each message, and the
    aggregation is normalized by the gate sum so a node's update does not scale
    with its degree. Both directions of every chain edge are present in
    ``edge_index``, so one scatter covers both message directions.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.lin_self = nn.Linear(d_model, d_model)
        self.lin_msg = nn.Linear(d_model, d_model)
        self.lin_src = nn.Linear(d_model, d_model)
        self.lin_dst = nn.Linear(d_model, d_model)
        self.lin_edge = nn.Linear(d_model, d_model)
        self.edge_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, e: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        """One round of message passing.

        Args:
            x: ``[B, N, d]`` node embeddings.
            e: ``[B, 2E, d]`` edge embeddings.
            edge_index: ``[2, 2E]`` int64, source row then destination row.

        Returns:
            Updated ``(x, e)``, both at their input shapes. The node output
            carries no residual; the GPS layer adds it.
        """
        src, dst = edge_index[0], edge_index[1]

        e_hat = (
            self.lin_edge(e)
            + self.lin_src(x).index_select(1, src)
            + self.lin_dst(x).index_select(1, dst)
        )
        gate = torch.sigmoid(e_hat)

        msg = gate * self.lin_msg(x).index_select(1, src)
        num = torch.zeros_like(x).index_add(1, dst, msg)
        # The gate sum, not the degree: a saturated-closed gate must shrink the
        # update rather than be normalized back up to full size.
        den = torch.zeros_like(x).index_add(1, dst, gate) + 1e-6

        x_out = self.lin_self(x) + num / den
        e_out = e + torch.relu(self.edge_norm(e_hat))
        return x_out, e_out


class BiasedAttention(nn.Module):
    """The global stream: dense multi-head attention, bias subtracted pre-softmax.

    Dense and unmasked on purpose. ``N = 33``, so an ``[B, H, 33, 33]`` score
    matrix is cheap, and masking it would reintroduce exactly the locality the
    experiment is trying to supply as a *soft* prior instead.
    """

    def __init__(self, cfg: GPSCfg) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_drop = nn.Dropout(cfg.attn_dropout)

    def _qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Project to ``[B, H, N, d_head]`` queries, keys and values."""
        b, n, _ = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.n_heads, self.d_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        return q, k, v

    def _weights(self, q: Tensor, k: Tensor, bias: Tensor | None) -> Tensor:
        """Softmax over the biased scores. **The one place Eq. 1 is applied.**"""
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.d_head)
        if bias is not None:
            _check_bias_shape(bias, q.shape[0], self.n_heads, q.shape[2])
            logits = logits - bias
        return torch.softmax(logits, dim=-1)

    def attention_weights(self, x: Tensor, bias: Tensor | None) -> Tensor:
        """``[B, H, N, N]`` post-softmax weights, before dropout and the values.

        Not used by :meth:`forward`'s caller, but it shares :meth:`_weights` with
        it, so what this returns is what the layer actually attends with. It is
        how the sign of the bias is pinned by test, and it is what an attention
        map in the write-up would be drawn from.
        """
        q, k, _ = self._qkv(x)
        return self._weights(q, k, bias)

    def forward(self, x: Tensor, bias: Tensor | None) -> Tensor:
        """Attention over all nodes.

        Args:
            x: ``[B, N, d]`` node embeddings. Attention sees these and nothing
                else: no edge features, no positions.
            bias: ``b``, broadcastable to ``[B, H, N, N]``, or ``None``. It is
                **subtracted**, per Eq. 1, so it scales the resulting weight by
                ``exp(-b)``.

        Returns:
            ``[B, N, d]``, with no residual; the GPS layer adds it.

        Raises:
            ValueError: if the bias does not broadcast onto the score matrix.
        """
        b, n, _ = x.shape
        q, k, v = self._qkv(x)
        weights = self.attn_drop(self._weights(q, k, bias))
        merged = (weights @ v).transpose(1, 2).reshape(b, n, self.n_heads * self.d_head)
        return self.out(merged)


def _check_bias_shape(bias: Tensor, b: int, h: int, n: int) -> None:
    """Reject a bias that does not broadcast onto ``[B, H, N, N]``.

    Worth an explicit check rather than letting broadcasting sort it out: a
    plugin that returns ``[B, N, N]`` would broadcast silently across the head
    axis and quietly turn a per-head prior into a shared one, which is the
    difference between variant D and variant C.
    """
    if bias.ndim != 4 or bias.shape[0] not in (1, b) or bias.shape[1:] != (h, n, n):
        raise ValueError(
            f"bias must be [B, H, N, N] or [1, H, N, N] = {(b, h, n, n)}, got {tuple(bias.shape)}"
        )


class GPSLayer(nn.Module):
    """One GraphGPS layer: both streams in parallel, summed, then an MLP."""

    def __init__(self, cfg: GPSCfg) -> None:
        super().__init__()
        self.cfg = cfg
        # Built per stream, norms included: an ablated block must not carry
        # parameters it never uses, or the matched-budget claim stops holding and
        # some distributed backends refuse a parameter that gets no gradient.
        self.local = GatedGCN(cfg.d_model) if cfg.use_local else None
        self.local_norm = _node_norm(cfg.d_model) if cfg.use_local else None
        self.attn = BiasedAttention(cfg) if cfg.use_global else None
        self.attn_norm = _node_norm(cfg.d_model) if cfg.use_global else None
        self.ffn_norm = _node_norm(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, 2 * cfg.d_model),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(2 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(
        self, x: Tensor, e: Tensor, edge_index: Tensor, bias: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        """Run both streams on the same input and mix them.

        Args:
            x: ``[B, N, d]`` node embeddings.
            e: ``[B, 2E, d]`` edge embeddings, passed through untouched when the
                local stream is off.
            edge_index: ``[2, 2E]`` int64.
            bias: ``b``, or ``None``. Computed once per forward pass and shared
                by every layer.

        Returns:
            Updated ``(x, e)``.
        """
        streams: list[Tensor] = []
        if self.local is not None and self.local_norm is not None:
            local, e = self.local(x, e, edge_index)
            streams.append(_apply_node_norm(self.local_norm, self.drop(local) + x))
        if self.attn is not None and self.attn_norm is not None:
            glob = self.attn(x, bias)
            streams.append(_apply_node_norm(self.attn_norm, self.drop(glob) + x))

        mixed = streams[0] if len(streams) == 1 else streams[0] + streams[1]
        # Eq. 11 is written as MLP(X_M + X_T); the residual and norm around it
        # are the precise form of Fig. D.1, which the summary line compresses.
        return _apply_node_norm(self.ffn_norm, mixed + self.ffn(mixed)), e


class GPSModel(nn.Module):
    """The block, stacked: ``Batch`` and a bias plugin in, next positions out.

    **The head predicts a displacement, and the position is added back.** It has
    to: ``pos`` is not a node feature, so nothing in the network knows where the
    cable is in world coordinates and an absolute-position head would be being
    asked for something it cannot see. Predicting the step and adding ``pos`` is
    what makes the model translation invariant, and it is why the contract keeps
    position out of ``node_feat`` in the first place.

    Construct with :meth:`from_batch` so the feature width comes off the data.
    """

    # Declared, because ``nn.Module.__getattr__`` types a registered buffer as
    # ``Tensor | Module`` and every use would otherwise need a cast.
    step_std: Tensor
    step_mean: Tensor

    def __init__(
        self, feature_width: int, cfg: GPSCfg | None = None, bias: BiasPlugin | None = None
    ) -> None:
        """
        Args:
            feature_width: ``F``, the width of ``batch.node_feat``. The loader
                sets it and it moves with the history depth ``C``, so it is a
                constructor argument and never a constant in this file.
            cfg: Block hyper-parameters; the defaults if omitted.
            bias: The variant under test; variant A if omitted. **Held rather
                than passed to** :meth:`forward`, because variants B through E
                carry learnable per-head rates, and a plugin handed in at call
                time is not a submodule: its parameters never reach
                ``model.parameters()``, so the optimizer never sees them and the
                rates sit at their initial value for the whole run while
                everything else trains. That failure is silent and would be read
                as "the prior did not help".
        """
        super().__init__()
        self.cfg = cfg or GPSCfg()
        self.feature_width = feature_width
        self.bias = bias if bias is not None else NoBias()
        self.node_encoder = nn.Linear(feature_width, self.cfg.d_model)
        self.edge_encoder = nn.Linear(EDGE_FEATURE_WIDTH, self.cfg.d_model)
        self.layers = nn.ModuleList(GPSLayer(self.cfg) for _ in range(self.cfg.n_layers))
        self.head = nn.Linear(self.cfg.d_model, 3)

        # **The head predicts the STANDARDIZED step, and these buffers put it
        # back into metres.** The loss divides both sides by the fitted
        # per-channel standard deviation, roughly [0.4, 0.3, 3.0] mm, so a head
        # emitting metres directly has to reach a weight scale ~800x smaller
        # than every other layer in the block. Adam's first update is exactly
        # +/- lr per element whatever the gradient is, so one step at lr 1e-3
        # moved the head to an output of 44 mm, which is 110 standard deviations
        # of the target, and 110^2 is the ~1e4 loss that followed. The peak
        # scaled as lr^2 across two decades, which is the signature of a
        # constant scale factor rather than of an instability.
        #
        # Speaking the loss's own units makes the head's natural step size
        # commensurate with the target, so lr is correct rather than 110x too
        # large. It also stops a cable whose step scale differs from another's
        # from silently changing the effective learning rate, which would
        # confound the very variant comparison the experiment exists to make.
        #
        # The defaults are the identity, so a model built without statistics
        # behaves exactly as an un-scaled one; the trainer copies the run's
        # fitted values in. They are buffers, so a checkpoint carries its own
        # scaling and cannot be reloaded against someone else's.
        self.register_buffer("step_std", torch.ones(3))
        self.register_buffer("step_mean", torch.zeros(3))

        # A zero head therefore predicts the mean step, which scores exactly
        # 1.0: the constant predictor, and the floor any real prediction must
        # beat. Predicting no movement at all is worse, 1.146, because the mean
        # vertical step is gravity rather than zero.
        #
        # The body's gradient is zero on the first backward pass, since the
        # Jacobian through a zero weight is zero. The head's own weights take
        # gradient immediately, so the body trains from the second step; the
        # stall is one step, not a stall.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @classmethod
    def from_batch(
        cls, batch: Batch, cfg: GPSCfg | None = None, bias: BiasPlugin | None = None
    ) -> GPSModel:
        """Build a block sized to a batch, reading ``F`` off ``node_feat``.

        Args:
            batch: Any contract-shaped batch, including the stub batch.
            cfg: Block hyper-parameters; the defaults if omitted.
            bias: The variant under test; variant A if omitted.
        """
        return cls(batch.node_feat.shape[-1], cfg, bias)

    def forward(self, batch: Batch) -> Tensor:
        """Predict the next vertex position for every node.

        The bias plugin is called **once** here and shared by every layer, which
        is correct because the variants read input positions, and those do not
        move within a step.

        Args:
            batch: The contract's ``Batch``.

        Returns:
            ``[B, N, 3]``, matching ``batch.target``.

        Raises:
            ValueError: if the batch's feature width is not the one this block
                was built for, or its edge channel is not the contract's.
        """
        if batch.node_feat.shape[-1] != self.feature_width:
            raise ValueError(
                f"block built for F={self.feature_width}, got F={batch.node_feat.shape[-1]}"
            )
        if batch.edge_feat.shape[-1] != EDGE_FEATURE_WIDTH:
            raise ValueError(
                f"edge_feat must be {EDGE_FEATURE_WIDTH} wide, got {batch.edge_feat.shape[-1]}"
            )

        x = self.node_encoder(batch.node_feat)
        e = self.edge_encoder(batch.edge_feat)
        b = self.bias(batch.pos, batch.meta) if self.cfg.use_global else None

        for layer in self.layers:
            x, e = layer(x, e, batch.edge_index, b)

        return batch.pos + (self.head(x) * self.step_std + self.step_mean)
