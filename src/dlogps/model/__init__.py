"""Model package: the GraphGPS block and the pluggable attention bias.

It must **never** import the harness or read a file from disk: the block is
built and tested against the frozen interface, which is what lets it be written
before the dataset exists.

``VARIANTS`` maps the variant letters used in run labels to their plugins.
"""

from __future__ import annotations

from collections.abc import Callable

from dlogps.model.bias import BiasPlugin, NoBias
from dlogps.model.gps import GPSCfg, GPSModel
from dlogps.model.structfree import (
    ARCHS,
    GlobalLSTM,
    PerNodeMLP,
    StructureFree,
    TrainableModel,
    build_structfree,
)
from dlogps.model.variants import (
    RATE_INIT,
    BCOnlyBias,
    BiasSource,
    ChainDistanceBias,
    MixedHeadBias,
    SpaceDistanceBias,
    assign_sources,
)

VARIANTS: dict[str, Callable[[int], BiasPlugin]] = {
    "A": lambda n_heads: NoBias(),
    "B": SpaceDistanceBias,
    "C": ChainDistanceBias,
    "D": MixedHeadBias,
    "F": BCOnlyBias,
}
"""Variant letter -> a factory taking ``n_heads``.

The letters are the run labels' names for the paper's variants: A = Unbiased,
B = Euclidean, C = Chain, D = Mixed, F = Euclidean+Chain only. Explicit
registration, and every entry takes the same one argument, so the runner selects
a variant by letter rather than by importing a class name. Variant A ignores it,
having nothing per head."""

__all__ = [
    "ARCHS",
    "RATE_INIT",
    "VARIANTS",
    "BCOnlyBias",
    "BiasPlugin",
    "BiasSource",
    "ChainDistanceBias",
    "GPSCfg",
    "GPSModel",
    "GlobalLSTM",
    "MixedHeadBias",
    "NoBias",
    "PerNodeMLP",
    "SpaceDistanceBias",
    "StructureFree",
    "TrainableModel",
    "assign_sources",
    "build_structfree",
]
