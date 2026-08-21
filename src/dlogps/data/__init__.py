"""Loader package: the dataset, the feature pipeline, and the boundary types.

Only :mod:`dlogps.data.types` exists so far, ahead of the loader itself, because
three of the four modules consume those types.
"""

from __future__ import annotations

from dlogps.data.types import Batch, BatchMeta, E, Labels, N

__all__ = ["E", "N", "Batch", "BatchMeta", "Labels"]
