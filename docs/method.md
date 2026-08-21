# Method

The model is a learned simulator for a falling cable: a GraphGPS-style block
predicts every vertex's next displacement from a short velocity history, and
rollouts feed each prediction back as the next input. The study holds that block,
the training protocol and the evaluation windows fixed and changes only an
additive bias inside the global-attention softmax.

## Cable representation and prediction target

`src/dlogps/data/types.py`, `src/dlogps/data/dataset.py`

* A cable is `N = 33` vertices joined by `E = 32` segments; each chain edge is
  stored in both directions (`edge_index` has `2E` columns).
* **Node features** (`features_from_state`): the velocity history of the last
  `C = 70` frames (newest first, `3C` channels), the four material parameters
  (Young's modulus `bend_stiffness`, `joint_damping`, `diameter`,
  `effective_linear_density`), and the distance to the floor clipped at 1 m.
  `F = 3C + 5 = 215` at the reported depth.
* **Edge features**: relative displacement (3), distance (1), rest length (1).
* Absolute coordinates are never a feature: positions reach the model only
  through `edge_feat` and the attention bias, which makes the representation
  invariant to translations parallel to the floor while the floor-distance
  channel keeps height.
* **Material encoding** (`harness/normalize.py`, `param_encoding: designed`):
  the four material channels are mapped by `log10` and a fixed affine transform
  onto `[-1, +1]`, `(log10 x - c) / h` with `(c, h)` = `(7, 3)`, `(-3, 2)`,
  `(-2.301, 1)`, `(-1.3495, 2.6505)` in the order above; values outside a span
  are not clipped. Velocity history, floor distance, edge features and the
  displacement targets are standardized with statistics fitted on the training
  split before input noise is added; the statistics travel in the checkpoint
  (`Stats`) and are frozen for training and evaluation. Positions are not scaled.
* **Target**: the per-node displacement to the next frame, predicted in
  standardized units, inverse-standardized and added to the current positions;
  velocities are then updated with the recorded frame interval
  (`harness/rollout.py`).

## Hybrid local–global block

`src/dlogps/model/gps.py`

Two linear encoders lift node and edge features to width `d = 128`; `L = 4`
identical layers follow; a linear readout maps the final node embeddings to the
displacement. Each layer runs two streams on the **same** input, in parallel:

* the **local stream**, a GatedGCN over the chain edges, which also reads and
  updates the edge embeddings (nothing else touches them);
* the **global stream**, dense multi-head attention (`H = 8`) over all `N`
  vertices, seeing node embeddings only — no edge features, and in the unbiased
  variant no pairwise term and no mask.

Each stream output is added back to the layer input and layer-normalized over
the feature axis of a single node; the two are summed and mixed by a two-layer
ReLU MLP of hidden width `2d`, again with a post-residual LayerNorm. Dropout and
weight decay are zero in every reported run. The block has 891k parameters
(891,011 unbiased; each biased variant adds `H` scalars).

`GPSCfg.use_local = False` drops the GatedGCN module (the *local-off* rows);
variant `chain_only` drops the attention module instead (message passing on the
chain edges, no pairwise bias).

## Attention-bias intervention

`src/dlogps/model/bias.py`, `src/dlogps/model/variants.py`

For head `h`, the attention score and weight are

```
s_ij^h = q_i^h · k_j^h / sqrt(d_k) − b_h(i, j)
α_ij^h = exp(q_i^h · k_j^h / sqrt(d_k)) exp(−b_h(i, j)) / Σ_r exp(q_i^h · k_r^h / sqrt(d_k)) exp(−b_h(i, r))
```

so `b_h = 0` recovers standard attention and a positive bias attenuates the
contribution of vertex `j` to vertex `i`. The five variants differ only in `b_h`:

| letter | name | `b_h(i, j)` by head | class |
|---|---|---|---|
| A | Unbiased | 0 on every head | `NoBias` |
| B | Euclidean | `γ_h · D̃_ij` on every head | `SpaceDistanceBias` |
| C | Chain | `β_h · C̃_ij` on every head | `ChainDistanceBias` |
| D | Mixed | `⌊H/3⌋` heads Euclidean, `⌊H/3⌋` chain, the rest unbiased (2 / 2 / 4 at `H = 8`) | `MixedHeadBias` |
| F | Euclidean+Chain only | half the heads Euclidean, half chain, none unbiased (`[B,B,C,C,B,B,C,C]`) | `BCOnlyBias` |

with the normalized distances

```
C̃_ij = |i − j| / (N − 1),        D̃_ij = ‖p_i − p_j‖ / L
```

where `p_i` is the position of vertex `i` and `L` the cable's rest length.
`γ_h, β_h = softplus(θ_h) > 0`, one rate per biased head, initialized to one
(`RATE_INIT`) and shared across layers. The Euclidean bias is recomputed from the
**predicted** positions at every rollout step. The variants differ by at most
`H` learned scalars and are therefore effectively parameter matched; the letters
are the names used in run labels (`varA_s0`, …) and checkpoints.

## Baselines outside the block

`src/dlogps/model/structfree.py`

* **chain only** — the block with `use_global = False`: GatedGCN on the chain
  edges, no attention.
* **MLP** (`mlp_pernode`, width 2048, three layers) — maps each vertex's own
  feature row to that vertex's displacement and never mixes vertices; it receives
  no position term, since a centroid would mix vertices.
* **LSTM** (`lstm_global`, width 256, three layers) — reads the flattened
  `N`-vertex state as a `C`-step sequence, plus centroid-relative positions
  divided by rest length, and writes all `N` displacements.

Both structure-free arms predict the standardized step exactly as the block does.
