"""Phase and pair-regime labels, computed on ground truth at load time.

**Ground truth only.** These are the loader's because they must never be
computed from a model's predictions: a model that hallucinates a self-crossing
would otherwise be scored as legitimately in contact, and the metric that was
supposed to catch it would agree with it instead. The harness receives them and
never recomputes them.

**Contact state is not recorded**, so the label is geometric: the solver forbids
interpenetration, which means a recorded contact channel would store a constant
(``docs/data.md``). A frame is *contact* when at least one chain-far segment
pair is within ``d_close``. That is a proximity criterion, not the solver's
contact set, and the paper says so.

Two thresholds, both tied to the same geometry and deliberately different:
self-intersection uses the bare diameter because interpenetration is a hard
violation, while a contact label uses ``1.5 x diameter``, a slightly generous
band so that segments about to touch count as contact rather than only those
already overlapping (``docs/metrics.md``).
"""

from __future__ import annotations

import torch
from torch import Tensor

from dlogps.data.geometry import chain_far_mask, segment_distance
from dlogps.data.types import Labels


def contact_labels(
    pos: Tensor, diameter: Tensor, *, d_close_diameters: float, c_far_segments: int
) -> Labels:
    """Label a ground-truth rollout window by contact phase and pair regime.

    Args:
        pos: ``[B, R, N, 3]`` ground-truth vertex positions over the window.
        diameter: ``[B]`` cable diameter [m], per sample.
        d_close_diameters: Contact threshold **in diameters**, never an absolute
            distance. The population spans 1.5 mm to 10 mm, so one fixed distance
            would be too strict for the thick cables and too loose for the thin
            ones, and the phase split would track cable thickness rather than
            geometry.
        c_far_segments: How far apart along the cable counts as far, in segments.

    Returns:
        :class:`Labels` aligned frame for frame with ``pos``.
    """
    if pos.ndim != 4:
        raise ValueError(f"pos must be [B, R, N, 3], got {tuple(pos.shape)}")
    if diameter.shape != pos.shape[:1]:
        raise ValueError(f"diameter must be [B]={pos.shape[0]}, got {tuple(diameter.shape)}")

    gap = segment_distance(pos)  # [B, R, E, E]
    far = chain_far_mask(c_far_segments, device=pos.device)
    d_close = (d_close_diameters * diameter).view(-1, 1, 1, 1)

    contact_pair = (gap < d_close) & far
    return Labels(
        contact_frame=contact_pair.any(dim=(-1, -2)),
        contact_pair=contact_pair,
    )


def phase_fraction(labels: Labels) -> Tensor:
    """``[B]`` fraction of frames in contact, the balance a split is judged on.

    Not a metric; a diagnostic. A cable that never folds reports 0.0 and its
    contact-phase decomposition has an empty class, which the metric suite
    reports as NaN rather than zero.
    """
    return labels.contact_frame.to(torch.float32).mean(dim=-1)
