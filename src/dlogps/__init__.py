"""dlogps: a GraphGPS-style learned simulator for falling cables, and the
attention-bias variants the paper compares.

:mod:`dlogps.data` reads the released dataset, :mod:`dlogps.model` holds the
block and the bias plugins, and :mod:`dlogps.harness` trains, rolls out and
scores. Everything depends on torch and nothing on a physics engine.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
"""The repository root, resolved once here rather than counted out per module."""
