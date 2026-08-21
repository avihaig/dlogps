# Metrics

`src/dlogps/harness/metrics.py` scores every rollout window; `harness/artifacts.py`
writes one row per window (`*_windows.csv`) and the aggregate with its
across-window standard deviation and count (`*_summary.json`). Every length is
in metres; the dimensionless families carry their unit in the column header.

Let `p̂_i(t)` and `p_i(t)` be the predicted and recorded positions of vertex `i`
at predicted frame `t` (`t = 1 … 400`; frame 0 is the input state and is never
scored), `N = 33`, and `L` the sample's rest length.

## Rollout error: `rel_l2_h400`

```
e_i(t)     = ‖p̂_i(t) − p_i(t)‖₂
ℓ2(t)      = (1/N) Σ_i e_i(t)
rel ℓ2(t)  = ℓ2(t) / L
```

The reported number per rollout is the mean of `rel ℓ2(t)` over the 400
predicted frames: column `rel_l2_h400` (`rollout_rmse` family; `rel_l2_at_h{h}`
is the value at frame `h` alone and `mean_l2_*`/`rmse_*` the unnormalized mean
and root-mean-square reductions of the same per-vertex distances — all written,
only `rel_l2_h400` is reported).

## Link-length drift: `link_length_mean_drift`

Uses the prediction and the cable geometry only. With rest length per segment
`ℓ0 = L / (N − 1)` and predicted segment length `ℓ̂_k(t) = ‖p̂_{k+1}(t) − p̂_k(t)‖₂`,

```
δ(t) = (1/(N − 1)) Σ_k |ℓ̂_k(t) − ℓ0| / ℓ0
```

reported as the mean of `δ(t)` over the rollout (`link_length` family; the
per-rollout maximum is also written as `link_length_max_drift`).

## Self-intersection: `selfint_frame_fraction`

Let `d_jk(t)` be the shortest distance between the centerlines of segments `j`
and `k` (`data/geometry.py::segment_distance`) and `D` the cable diameter. A pair
with `|j − k| > 2` — beyond the two-edge bending stencil — violates when
`d_jk(t) < D`. The reported value is the mean over the rollout of

```
v(t) = 1[ min_{|j−k|>2} d_jk(t) < D ]
```

(`self_intersection` family; the pair fraction, worst penetration and a
persistence flag are also written).

## Phase split: `rel_l2_floor_contact`, `rel_l2_free_flight`

`error_by_floor_phase`. A predicted frame is **floor contact** when the
**recorded** trajectory has `min_i z_i(t) ≤ D`, and **free flight** otherwise;
the label is read off the ground truth, never the prediction, and only frames
after the input frame count. Each rollout contributes the mean of `rel ℓ2(t)`
over its selected frames, NaN when it has none, and the table cell is the
equal-weight mean over rollouts with at least one selected frame. The summary's
`count` is the number of eligible rollouts (400 floor contact and 345 free flight
of the 400 test windows). `floor_contact_frame_fraction` records the share of
predicted frames in contact.

## Aggregation into the tables

Per seed: the mean over windows and the window standard deviation from
`*_summary.json`. Per table cell: mean of the three seed means ± mean of the
three window sds ± standard deviation (n − 1) of the three seed means
(`scripts/make_tables.py`).

## Also written, not reported

`error_by_phase` (`rmse_contact`, `rmse_free_flight`, `contact_frame_fraction`:
a self-contact proximity split using the loader's labels, `data/labels.py`),
`error_by_pair_regime` (`rmse_contact_pair_nodes`, `rmse_other_nodes`) and
`error_by_chain_distance` (`rmse_band0..3`) are diagnostics that every run
records alongside the reported columns. They do not appear in the paper.
