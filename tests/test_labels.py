"""Contact labels, computed on ground truth at load time.

These are the labels two of the six metrics decompose by, so a defect here does
not fail loudly: it moves the contact/free-flight split and quietly changes what
"where the bias helps" means.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dlogps.data.labels import contact_labels, phase_fraction
from dlogps.data.types import E, N

SAMPLE = "assets/sample_v1"


def straight_cable(b: int = 1, r: int = 1, length: float = 1.6) -> torch.Tensor:
    """A taut cable along x: no pair is ever close, so nothing may be labelled."""
    x = torch.linspace(0.0, length, N)
    pos = torch.zeros(b, r, N, 3)
    pos[..., 0] = x
    return pos


@pytest.mark.contract
def test_a_straight_cable_has_no_contact():
    labels = contact_labels(
        straight_cable(), torch.tensor([0.0015]), d_close_diameters=1.5, c_far_segments=4
    )
    assert labels.contact_frame.shape == (1, 1)
    assert labels.contact_pair.shape == (1, 1, E, E)
    assert not bool(labels.contact_frame.any())
    assert not bool(labels.contact_pair.any())


@pytest.mark.claim(
    "a contact pair is 3D-close AND chain-far; adjacent segments touch because the cable "
    "is a cable, so labelling them would make every frame read as contact "
    "(docs/metrics.md)"
)
def test_chain_neighbours_are_never_labelled():
    # A hairpin: the cable folds back on itself, so both chain-near and
    # chain-far pairs are in 3D contact. Only the chain-far ones may be marked.
    pos = torch.zeros(1, 1, N, 3)
    half = N // 2
    pos[0, 0, :half, 0] = torch.linspace(0.0, 0.5, half)
    pos[0, 0, half:, 0] = torch.linspace(0.5, 0.0, N - half)
    pos[0, 0, half:, 1] = 0.001

    labels = contact_labels(pos, torch.tensor([0.005]), d_close_diameters=1.5, c_far_segments=4)

    assert bool(labels.contact_frame.any()), "the hairpin must register as contact at all"
    seg = torch.arange(E)
    near = (seg.view(-1, 1) - seg.view(1, -1)).abs() <= 4
    assert not bool(labels.contact_pair[..., near].any())
    # Symmetric: the pair (i, j) is the pair (j, i).
    assert torch.equal(labels.contact_pair, labels.contact_pair.transpose(-1, -2))


@pytest.mark.claim(
    "d_close is a multiplier on the diameter, never an absolute distance: the population "
    "spans 1.5 mm to 10 mm, so one distance would track cable thickness instead of geometry "
    ""
)
def test_the_threshold_is_per_cable():
    # One geometry, two cables. The thick one is in contact, the thin one is not,
    # and nothing about the positions differs.
    pos = torch.zeros(2, 1, N, 3)
    half = N // 2
    pos[:, 0, :half, 0] = torch.linspace(0.0, 0.5, half)
    pos[:, 0, half:, 0] = torch.linspace(0.5, 0.0, N - half)
    pos[:, 0, half:, 1] = 0.004

    labels = contact_labels(
        pos, torch.tensor([0.010, 0.001]), d_close_diameters=1.5, c_far_segments=4
    )

    assert bool(labels.contact_frame[0, 0]), "10 mm cable: 4 mm apart is contact"
    assert not bool(labels.contact_frame[1, 0]), "1 mm cable: 4 mm apart is not"


@pytest.mark.contract
@pytest.mark.parametrize(
    ("pos", "diameter", "match"),
    [
        (torch.zeros(1, N, 3), torch.tensor([0.001]), r"\[B, R, N, 3\]"),
        (torch.zeros(2, 1, N, 3), torch.tensor([0.001]), r"diameter must be"),
    ],
)
def test_a_shape_that_cannot_be_labelled_is_refused(pos, diameter, match):
    with pytest.raises(ValueError, match=match):
        contact_labels(pos, diameter, d_close_diameters=1.5, c_far_segments=4)


@pytest.mark.claim(
    "the committed sample carries both contact regimes: cable 000 folds and cable 045 never "
    "comes near itself, so a loader can be tested on each (assets/sample_v1/README.md)"
)
def test_the_sample_reproduces_both_regimes():
    fractions = {}
    for name in ("cable_000", "cable_045"):
        with np.load(f"{SAMPLE}/{name}/episodes.npz") as d:
            n = int(d["episode_lengths"][0])
            pos = torch.from_numpy(d["vertex_pos"][0, :n]).float().unsqueeze(0)
            diameter = torch.tensor([float(d["diameter"])])
        labels = contact_labels(pos, diameter, d_close_diameters=1.5, c_far_segments=4)
        fractions[name] = float(phase_fraction(labels)[0])

    # Measured on this fixture: the soft cable folds for most of its episode and
    # the stiff one never comes within 64 diameters of itself.
    assert fractions["cable_000"] > 0.4
    assert fractions["cable_045"] == 0.0
