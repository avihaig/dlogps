"""Where the work runs: the intra-op thread bound, and moving a batch to a device.

Two placement concerns that every entry point needs and that no module owned, so
each entry point either repeated them or, more often, did not apply them at all.

**The thread bound is worth two orders of magnitude and is not optional.** The
block is 33 nodes wide, so every op in it is far too small to amortize a thread
fan-out, and torch's default of one thread per core turns that into contention.
Measured on a 24-core box: the evaluation driver at horizon 100 over 40 windows
costs 184.2 s at the default and 0.91 s bounded, and a training step at batch 512
costs 885 ms against 38.8 ms. Unbounded, a 2000-step training run prints nothing
for fifteen minutes and reads as a hang.

**A rollout belongs on the GPU**, since it is the most compute-heavy step in the
project: horizon times windows forward passes, autoregressive and therefore
serial. Following the model's own device is what lets the driver run where
training ran, and it is why the move lives here rather than in a caller that has
to be remembered.
"""

from __future__ import annotations

from dataclasses import fields

import torch
from torch import nn

from dlogps.data.types import Batch, BatchMeta, Labels

INTRA_OP_THREADS = 4
"""Ceiling on torch's intra-op threads, applied at every entry point.

A ceiling rather than a setting: a machine with fewer cores keeps its own count.
Printed wall clock is only reproducible next to the count it was measured at, so
every entry point that reports a time applies this one."""


def bound_intra_op_threads(ceiling: int = INTRA_OP_THREADS) -> int:
    """Cap torch's intra-op threads and return the count now in force.

    Process-wide, so it belongs to entry points and never to a library call.

    Args:
        ceiling: Maximum threads. The machine's own count wins when it is lower.

    Returns:
        The thread count after the cap, for printing beside a wall clock.
    """
    torch.set_num_threads(min(torch.get_num_threads(), ceiling))
    return torch.get_num_threads()


def model_device(model: nn.Module) -> torch.device:
    """The device a model's tensors live on.

    Parameters first, then buffers: a block whose parameters were all ablated
    away still carries the head's step-scale buffers, and a model with neither is
    on the CPU by construction.

    Args:
        model: Any module.

    Returns:
        The device of its first parameter or buffer, or CPU if it has neither.
    """
    for tensor in model.parameters():
        return tensor.device
    for tensor in model.buffers():
        return tensor.device
    return torch.device("cpu")


def batch_to(batch: Batch, device: torch.device | str) -> Batch:
    """The same batch on another device, ``meta`` included.

    ``meta`` moves because the bias plugins divide by ``meta.length`` and the
    noise integrates ``meta.dt``, so a batch whose meta stayed behind fails on the
    first variant that reads it rather than on the first that does not.

    Args:
        batch: A contract-shaped batch.
        device: Where it is wanted.

    Returns:
        A new batch; the input is not mutated.
    """
    meta = BatchMeta(**{f.name: getattr(batch.meta, f.name).to(device) for f in fields(BatchMeta)})
    return Batch(
        pos=batch.pos.to(device),
        node_feat=batch.node_feat.to(device),
        edge_index=batch.edge_index.to(device),
        edge_feat=batch.edge_feat.to(device),
        target=batch.target.to(device),
        meta=meta,
    )


def labels_to(labels: Labels, device: torch.device | str) -> Labels:
    """The same labels on another device.

    They index the rollout tensors inside the metric suite, so they have to
    travel with the window rather than being compared across devices.

    Args:
        labels: Phase and pair labels for one window.
        device: Where they are wanted.

    Returns:
        A new :class:`Labels`; the input is not mutated.
    """
    return Labels(
        contact_frame=labels.contact_frame.to(device),
        contact_pair=labels.contact_pair.to(device),
    )
