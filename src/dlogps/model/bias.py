"""The attention-bias interface, and variant A: no bias at all.

**This is the keystone of the project.** One GraphGPS block is held fixed and
exactly one term varies: a bias subtracted from the attention logits *before* the
softmax,

    A[h,i,j] = (q_i . k_j) / sqrt(d_k)  -  b_h(i,j)                        (Eq. 1)

Subtracting pre-softmax makes the bias scale the attention weight
multiplicatively, by ``exp(-b_h(i,j))``: a large bias makes a pair invisible to
attention, zero leaves it untouched. Get this interface right once and the five
variants are nearly free: the bias variants ride on the one model block.

The protocol is frozen. Four rules that
are easy to get wrong and are the reason the interface exists at all:

- **Sign.** A plugin returns ``b``, the quantity the block **subtracts**. Not the
  logit offset, not ``-b``. Returning the negated quantity is a bug that trains
  perfectly well and inverts the prior under test.
- **Rank.** ``[B, H, N, N]``, or ``[1, H, N, N]`` for a variant that does not
  depend on the sample, which then broadcasts.
- **When.** Called once per forward pass and shared by every layer. Correct
  because the variants read *input* positions, which do not move within a step.
- **Normalization is the plugin's**, because it is variant-specific. Both bias
  metrics reach the softmax normalized to ``[0, 1]``, or the chain-distance
  variant starts roughly fifty times stronger than the 3D-distance one and the
  comparison measures initialization instead of physics
  (``docs/method.md``).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor

from dlogps.data.types import BatchMeta


@runtime_checkable
class BiasPlugin(Protocol):
    """What the block needs from a bias variant. Frozen; see the module docstring.

    Inputs are ``pos`` and ``meta`` **only**, never node embeddings: the bias is
    a structural prior, and one that could read the embeddings would be a second
    attention mechanism rather than a fixed prior under test.
    """

    def __call__(self, pos: Tensor, meta: BatchMeta) -> Tensor | None:
        """The bias ``b`` the block subtracts.

        Args:
            pos: ``[B, N, 3]`` current vertex positions [m].
            meta: The batch's per-sample constants; ``length`` normalizes the
                3D-distance metric.

        Returns:
            ``[B, H, N, N]``, or ``[1, H, N, N]`` to broadcast, or ``None`` for a
            variant that applies no bias at all.
        """
        ...

    def bias_stats(self, pos: Tensor, meta: BatchMeta) -> dict[str, float]:
        """Mean and std of ``b`` over a batch, for the init-time bias check.

        Called at initialization, not during training. The gate that bias scales
        are comparable at the start compares these across variants B and C and
        requires agreement to within a factor of two. It is a check, not a
        tuning knob.

        Returns:
            ``{"bias_mean": ..., "bias_std": ...}``, or an empty dict for a
            variant with no ``b``.
        """
        ...


class NoBias:
    """Variant A: plain attention, no structural prior.

    The control the whole experiment is read against, and half of the first
    end-to-end gate, that the harness tells two models apart. Returning ``None``
    rather than a zero tensor is deliberate: it is what lets the block skip the
    subtraction entirely, so variant A costs no memory for an ``[B, H, N, N]``
    of zeros and cannot be accidentally "biased by zero" in a way that hides a
    shape bug.
    """

    def __call__(self, pos: Tensor, meta: BatchMeta) -> None:
        """Always ``None``. Signature fixed by the protocol; both args ignored."""
        del pos, meta
        return None

    def bias_stats(self, pos: Tensor, meta: BatchMeta) -> dict[str, float]:
        """Empty, since variant A has no ``b`` to summarize."""
        del pos, meta
        return {}
