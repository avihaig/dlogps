"""Shared test configuration.

The suite runs on the CPU against contract-shaped synthetic batches
(:mod:`tests.fake_data`) and the two bundled sample cables under ``assets/``; it
needs neither the released dataset nor a GPU.
"""

from __future__ import annotations

from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "assets"
"""The bundled fixtures: ``sample_v1`` (training-style) and ``sample_test_v1``."""
