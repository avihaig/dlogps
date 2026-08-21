# sample_v1

Two cables × two complete episodes from the released training data, generated
by `configs/release.yaml` unchanged: a true subset, so anything written against
it runs on `release_v1` with nothing changed.

| cable | material | L [m] | d [m] | E [Pa] | episodes |
|---|---|---|---|---|---|
| 000 | M1_very_soft | 0.80 | 0.0025 | 1e6 | 661, 648 frames |
| 045 | M5_very_stiff | 1.60 | 0.0015 | 1e9 | 1331, 1397 frames |

The two extremes of the grid: 1000× in stiffness, 2× in length, and one folds on
itself while the other does not. Format: docs/data.md.

```python
import numpy as np

d = np.load("assets/sample_v1/cable_045/episodes.npz")
for e, n in enumerate(d["episode_lengths"]):
    t = d["t"][e, :n]                # (n,)
    dt = np.diff(t).mean()           # never d["record_hz"]
    pos = d["vertex_pos"][e, :n]     # (n, 33, 3)
```

Shapes are `(episodes, T_pad, ...)` with `T_pad = 2383`; past
`episode_lengths[e]` sits one `-1` stop token, then NaN.
