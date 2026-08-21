"""Segment geometry shared by the loader's labels and the harness's metrics.

**The loader's, not the harness's.** ``contracts.md`` makes ``Labels`` the
loader's to compute and says the harness never recomputes them, and the loader
may never import the harness. Both need the same segment distance -- the loader
to label ground-truth contact, the harness to score self-intersection on
predictions -- so it lives here and the harness imports it. Two implementations
of segment distance is the defect that put the wrench oracle in two specs and
made it wrong in both.
"""

from __future__ import annotations

import torch
from torch import Tensor

from dlogps.data.types import E


def segment_midpoints(pos: Tensor) -> Tensor:
    """``[..., E, 3]`` midpoint of each segment, from ``[..., N, 3]`` vertices."""
    return 0.5 * (pos[..., :-1, :] + pos[..., 1:, :])


def segment_distance(pos: Tensor) -> Tensor:
    """``[..., E, E]`` shortest distance between every pair of segments.

    **Segment to segment, not vertex to vertex, and the difference is the whole
    metric.** Two segments crossing in an X touch at their middles while all four
    endpoints stay far apart, so a vertex-distance check would score a cable that
    has passed straight through itself as clean. That case is exactly what
    self-intersection is looking for.

    The standard clamped closest-point construction: solve for the closest points
    on the two infinite lines, clamp one parameter to its segment, re-solve the
    other, clamp again. Every segment here has positive length — it is a cable at
    rest length — so the degenerate branches reduce to an epsilon guard.
    """
    p0 = pos[..., :-1, :]
    d = pos[..., 1:, :] - p0

    # [..., E, E, 3]: segment i against segment j.
    d1 = d.unsqueeze(-2)
    d2 = d.unsqueeze(-3)
    r = p0.unsqueeze(-2) - p0.unsqueeze(-3)

    a = (d1 * d1).sum(-1).clamp_min(1e-12)
    e = (d2 * d2).sum(-1).clamp_min(1e-12)
    b = (d1 * d2).sum(-1)
    c = (d1 * r).sum(-1)
    f = (d2 * r).sum(-1)

    denom = (a * e - b * b).clamp_min(1e-12)
    s = ((b * f - c * e) / denom).clamp(0.0, 1.0)
    t = ((b * s + f) / e).clamp(0.0, 1.0)
    s = ((t * b - c) / a).clamp(0.0, 1.0)

    closest = r + s.unsqueeze(-1) * d1 - t.unsqueeze(-1) * d2
    return closest.norm(dim=-1)


def chain_far_mask(c_far: int, device: torch.device | None = None) -> Tensor:
    """``[E, E]`` bool, segment pairs at least ``c_far`` apart along the cable.

    Adjacent segments touch because the cable is a cable; only chain-far pairs
    can be a self-intersection or a contact.
    """
    seg = torch.arange(E, device=device)
    return (seg.view(-1, 1) - seg.view(1, -1)).abs() > c_far
