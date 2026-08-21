"""The bias variants: Euclidean distance (B), chain distance (C), mixed heads
(D), and the Euclidean+Chain-only allocation (F).

Every variant is the same block with the same interface and one term swapped:
``b_h(i,j)``, subtracted from the attention logits before the softmax
(``docs/method.md``). Variant A, unbiased attention, lives in
:mod:`dlogps.model.bias` because it is the absence of all of this.

| | Paper name | Class | ``b_h(i,j)`` | Metric |
|---|---|---|---|---|
| B | Euclidean | :class:`SpaceDistanceBias` | ``γ_h · ‖p_i − p_j‖ / L`` | 3D, per step |
| C | Chain | :class:`ChainDistanceBias` | ``β_h · |i − j| / (N − 1)`` | chain, static |
| D | Mixed | :class:`MixedHeadBias` | per head: B's, C's, or nothing | both |
| F | Euclidean+Chain only | :class:`BCOnlyBias` | fixed per head: B or C | both |

**Both metrics are normalized to ``[0, 1]`` inside the plugin.** Raw, they sit
in the same logit roughly fifty times apart — ``|i − j| ∈ [0, 32]`` against
``‖p_i − p_j‖ ∈ [0, ~1.6]`` metres — so with the rates initialized alike variant
C would start with a far stronger prior than variant B and the comparison would
measure initialization rather than physics. Normalization is the *plugin's* job
because it is variant-specific, and it is fixed a priori rather than tuned. At
init, B's and C's ``bias_stats()`` agree to within a factor of two.

**The rates are softplus'd**, so ``b ≥ 0`` and the bias can only ever suppress a
pair, never amplify it. ``γ_h = 0`` recovers variant A exactly, which is what
makes these nested rather than merely different.

**Budget.** B, C, D and F each add exactly ``H`` learnable scalars (eight at
the reported ``H = 8``), under 0.01% of the model — so a gap between them is
attributable to the prior and not to capacity.
"""

from __future__ import annotations

from enum import IntEnum

import torch
from torch import Tensor, nn

from dlogps.data.types import BatchMeta, N

RATE_INIT = 1.0
"""Initial ``γ_h`` and ``β_h``, in units of "logits per normalized distance".

One, not a small number, and the same for B and C — which is the whole point.
Both metrics span ``[0, 1]`` after normalization, so a rate of one makes the
bias span roughly one logit: the same order as the content logits it competes
with, so the prior is neither inert nor overwhelming at step zero. Shared across
the variants because the gate that bias scales are comparable at the start
compares them against each other, and a per-variant init would be exactly the
tuning that gate exists to forbid.
"""


def _inverse_softplus(y: float) -> float:
    """The raw parameter whose softplus is ``y``. Rates are stored pre-softplus."""
    return float(torch.log(torch.expm1(torch.tensor(y))))


def chain_distance(n: int = N, device: torch.device | None = None) -> Tensor:
    """``[n, n]`` int64 chain distance ``|i − j|``, before normalization.

    Fixed by topology: unlike the 3D metric this never changes and is computed
    once. Returned raw; :func:`_normalized_chain_distance` divides by ``N − 1``.
    """
    idx = torch.arange(n, device=device)
    return (idx.view(-1, 1) - idx.view(1, -1)).abs()


def _normalized_chain_distance(n: int = N, device: torch.device | None = None) -> Tensor:
    """``[n, n]`` chain distance scaled to ``[0, 1]`` by ``n − 1``, per Eq. 2."""
    return chain_distance(n, device).float() / (n - 1)


def pairwise_distance(pos: Tensor) -> Tensor:
    """``[B, N, N]`` Euclidean distance between vertices, translation invariant.

    ``compute_mode`` is not a detail. ``torch.cdist`` defaults to the
    matmul-based expansion ``‖a‖² + ‖b‖² − 2a·b``, which loses the cancellation
    when the coordinates are large compared to the distances between them — the
    exact regime here, a 1 m cable that can be anywhere in the scene. Measured on
    the stub batch, translating by 5 m moves the default's answer by 2.8e-3 on a
    bias of scale 0.86, a third of a percent of pure artifact. The direct form is
    accurate to 4e-7 under the same shift.

    That would be a slow leak rather than a crash: the bias is meant to read a
    shape, and under the default it would also read *where the cable is*. Rollout
    is where it would bite, since the predicted cable drifts through the scene.
    At ``N = 33`` the direct form costs nothing.
    """
    return torch.cdist(pos, pos, compute_mode="donot_use_mm_for_euclid_dist")


def _stats(b: Tensor) -> dict[str, float]:
    """The two numbers read by the gate that bias scales are comparable at the
    start. Detached: this is a measurement."""
    with torch.no_grad():
        return {"bias_mean": float(b.mean()), "bias_std": float(b.std())}


class SpaceDistanceBias(nn.Module):
    """**Variant B**: close in 3D means relevant, which is the self-contact prior.

    ``b_h(i,j) = γ_h · ‖p_i − p_j‖ / length``. Transplants the radius-graph
    construction of GNS and MeshGraphNets' world edges out of the edge set and
    into the attention logits, as a soft learned kernel rather than a hard cutoff.

    ``length`` normalizes it, and it is a per-cable constant the contract already
    carries — so the rate is dimensionless and comparable across cables of
    different length, which is what makes the learned ``γ_h`` a readout rather
    than a number.

    State-dependent, and therefore recomputed every step: the cable moves. At
    rollout the positions are the model's *own predictions*, so rollout drift
    drifts the bias too. That is honest, and the rollout metrics measure it.

    Expect it to win in the landed phase, where contact pairs are 3D-close and
    chain-far, and to hurt when the cable is taut, where it suppresses exactly
    the 3D-far end-to-end pairs that tension couples.
    """

    def __init__(self, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.rate = nn.Parameter(torch.full((n_heads,), _inverse_softplus(RATE_INIT)))

    def forward(self, pos: Tensor, meta: BatchMeta) -> Tensor:
        """``[B, H, N, N]``. Batch-dependent, so no broadcasting shortcut."""
        distance = pairwise_distance(pos) / meta.length.view(-1, 1, 1)
        return nn.functional.softplus(self.rate).view(1, -1, 1, 1) * distance.unsqueeze(1)

    def bias_stats(self, pos: Tensor, meta: BatchMeta) -> dict[str, float]:
        return _stats(self(pos, meta))


class ChainDistanceBias(nn.Module):
    """**Variant C**: close along the cable means relevant, the bending prior.

    ``b_h(i,j) = β_h · |i − j| / (N − 1)``. The metric a rod's own mechanics uses:
    bending and twist energies in the discrete elastic rod are local in
    arc-length, and 3D distance scrambles that — a folded cable has 3D-close
    pairs that are mechanically unrelated.

    Static, so it is computed once as ``[1, H, N, N]`` and broadcast over the
    batch. The contract allows exactly this, and it is why the broadcasting
    clause exists.

    Blind to contact by construction, since landing pairs are chain-far and get
    suppressed. The subtle case is a fully taut cable, where influence does not
    decay with ``|i − j|`` at all because inextensibility couples the ends
    rigidly; there the best ``β_h`` is near zero, which the variant can reach.
    """

    def __init__(self, n_heads: int, n: int = N) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.rate = nn.Parameter(torch.full((n_heads,), _inverse_softplus(RATE_INIT)))
        self.register_buffer("metric", _normalized_chain_distance(n), persistent=False)

    def forward(self, pos: Tensor, meta: BatchMeta) -> Tensor:
        """``[1, H, N, N]``, broadcast over the batch: it does not depend on one."""
        del pos, meta
        metric = torch.as_tensor(self.metric)
        return nn.functional.softplus(self.rate).view(1, -1, 1, 1) * metric

    def bias_stats(self, pos: Tensor, meta: BatchMeta) -> dict[str, float]:
        return _stats(self(pos, meta))


class BiasSource(IntEnum):
    """Which metric a head is assigned in variant D."""

    SPACE = 0
    CHAIN = 1
    NONE = 2


def assign_sources(n_heads: int) -> list[BiasSource]:
    """Split the heads into thirds: space, chain, unbiased.

    **Fixed, not learned, and deliberately so.** A learned gate would confound
    "does specialization across conflicting metrics help" with "can a gate find
    it", and only the first question is open — HopFormer's Theorem 2 already
    settles that distinct per-head budgets beat one budget.

    A remainder goes to the unbiased heads, which is the choice that keeps the
    variant nested inside the others rather than inventing a fourth behaviour.
    At the reported ``H = 8`` that remainder is four, so D is 2/2/4.
    """
    third = n_heads // 3
    tail = [BiasSource.NONE] * (n_heads - 2 * third)
    return [BiasSource.SPACE] * third + [BiasSource.CHAIN] * third + tail


class MixedHeadBias(nn.Module):
    """**Variant D (Mixed)**: a different prior on each head.

    Thirds: ``⌊H/3⌋`` heads on Euclidean distance, ``⌊H/3⌋`` on chain distance,
    the rest unbiased (2/2/4 at ``H = 8``). Still exactly ``H`` learnable
    scalars, because the assignment costs nothing — which is what keeps D inside
    the matched-budget table alongside B and C.

    The unbiased heads keep a rate that multiplies zero. Inert by construction
    and left in on purpose: dropping it would make D's parameter count differ
    from B's and C's, and the matched budget is the claim.

    D should beat both single biases only if both neighbourhoods carry signal,
    and that conditionality is what makes it an ablation rather than a free win.
    """

    def __init__(self, n_heads: int, n: int = N) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.sources = assign_sources(n_heads)
        self.rate = nn.Parameter(torch.full((n_heads,), _inverse_softplus(RATE_INIT)))
        self.register_buffer("metric", _normalized_chain_distance(n), persistent=False)
        self.register_buffer(
            "source_ids", torch.tensor([int(s) for s in self.sources]), persistent=False
        )

    def forward(self, pos: Tensor, meta: BatchMeta) -> Tensor:
        """``[B, H, N, N]``, each head reading the metric it was assigned."""
        space = pairwise_distance(pos) / meta.length.view(-1, 1, 1)
        chain = torch.as_tensor(self.metric).expand_as(space)
        # Indexed rather than branched: one gather over a [3, B, N, N] stack
        # keeps every head on the same code path, so a head cannot silently take
        # a different one. Trivial at N = 33.
        stack = torch.stack([space, chain, torch.zeros_like(space)])
        per_head = stack[torch.as_tensor(self.source_ids)]
        return (nn.functional.softplus(self.rate).view(-1, 1, 1, 1) * per_head).permute(1, 0, 2, 3)

    def bias_stats(self, pos: Tensor, meta: BatchMeta) -> dict[str, float]:
        return _stats(self(pos, meta))


class BCOnlyBias(nn.Module):
    """**Variant F (Euclidean+Chain only)**: four Euclidean and four chain heads at ``H = 8``.

    The fixed order ``[B, B, C, C, B, B, C, C]`` preserves D's first four
    assignments while replacing its four unbiased heads with physical priors.
    Each head owns one softplus rate, so F keeps D's eight-scalar budget.

    Raises:
        ValueError: if ``n_heads`` is not eight, the head count the allocation
            is defined for.
    """

    def __init__(self, n_heads: int, n: int = N) -> None:
        if n_heads != 8:
            raise ValueError(f"variant F requires n_heads=8, got {n_heads}")
        super().__init__()
        self.n_heads = n_heads
        self.sources = [
            BiasSource.SPACE,
            BiasSource.SPACE,
            BiasSource.CHAIN,
            BiasSource.CHAIN,
            BiasSource.SPACE,
            BiasSource.SPACE,
            BiasSource.CHAIN,
            BiasSource.CHAIN,
        ]
        self.rate = nn.Parameter(torch.full((n_heads,), _inverse_softplus(RATE_INIT)))
        self.register_buffer("metric", _normalized_chain_distance(n), persistent=False)
        self.register_buffer(
            "source_ids", torch.tensor([int(source) for source in self.sources]), persistent=False
        )

    def forward(self, pos: Tensor, meta: BatchMeta) -> Tensor:
        """Return ``[B, 8, N, N]`` with each head reading its fixed metric."""
        space = pairwise_distance(pos) / meta.length.view(-1, 1, 1)
        chain = torch.as_tensor(self.metric).expand_as(space)
        metrics = torch.stack([space, chain])
        per_head = metrics[torch.as_tensor(self.source_ids)]
        rates = nn.functional.softplus(self.rate).view(-1, 1, 1, 1)
        return (rates * per_head).permute(1, 0, 2, 3)

    def bias_stats(self, pos: Tensor, meta: BatchMeta) -> dict[str, float]:
        """Return mean and standard deviation for the initialization gate."""
        return _stats(self(pos, meta))
