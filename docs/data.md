# Data

## Generation

Every trajectory is simulated in **MuJoCo 3.9.0**. The generator itself builds
on an unpublished MuJoCo scene package and is not part of this repository; the
complete protocol and every physical value it used are recorded in
`configs/release.yaml` (and the three files that extend it), and the populations
in `configs/release_cables.csv` and `configs/unseen_cables_test.csv`.

* Each cable is a 32-segment chain (`N = 33` vertices) under the MuJoCo cable
  elasticity plugin, with twist-to-bend ratio `G/E = 0.38`.
* Integrator `implicitfast` at `Δt = 4.2e-4 s`; kinematics recorded at 500 Hz
  (requested; the achieved rate is quantized to whole simulator steps, so
  readers take `dt` from `diff(t)`, never from the nominal rate).
* The scene is a floor (the plane `z = 0`) and two grippers welded to the cable
  ends. Each episode settles the cable, drives the grippers to a randomized pose
  (gripper separation between 0.10 m and a quarter of the length, a 0.5 m
  sampling box, a π/4 end-facing cone, a 25 s quasi-static reach and a fixed 5 s
  dwell), then releases both welds on the same tick. `t = 0` is that instant;
  the cable falls onto the floor and is recorded until its p95 vertex speed
  stays below 1 cm/s for 0.5 s, or 5 s elapse.

## Populations

| root | cables | episodes | role |
|---|---|---|---|
| `release_train` | 46 | 4,597 (100 per cable, five removed below) | **every reported run trains on it** |
| `unseen_cables_test` | 40 | 1,200 (30 per cable) | **the reported test set** (`configs/unseen_cables_test.yaml`, seed 3) |
| `seen_cables_test` | the 46 training cables | 1,378 (30 per cable) | new release poses of the training cables (`configs/seen_cables_test.yaml`, seed 2); not used in the paper |

`release_train` joins two generator runs over the same 46 cables, 50 release
poses per cable each: `configs/release.yaml` (seed 0) and
`configs/release_test.yaml` (seed 1, new poses). `scripts/merge_release_train.py`
concatenated them cable by cable, the seed-0 episodes first; the two halves are
not released separately.

The training population spans rest length 0.80–1.60 m, diameter 1.5–10 mm and
five Young's moduli from 10^6 to 10^9 Pa, organised as five material classes
(each with its own density and joint damping; see the CSV). The 40 test cables
lie inside those ranges and never appear in training. About two thirds of the
predicted frames in the test rollouts are floor contact, and the soft cables
fold onto themselves.

## On-disk format

```
<root>/
  cable_000/
    episodes.npz      vertex_pos, vertex_vel  (episodes, T_pad, 33, 3)   float64, metres
                      edge_quat (…, 32, 4)  edge_omega (…, 32, 3)  edge_wrench (…, 32, 6)
                      t (episodes, T_pad)    episode_lengths (episodes,)
                      record_hz, cable_id, length, diameter, bend_stiffness,
                      joint_damping, effective_linear_density   (scalars)
    params.yaml       the cable row, derived constants, the protocol values, and the
                      per-episode release record
  cable_001/ …
  index.csv           the population, with each cable's released episode count
  datagen.yaml        the generator's provenance (MERGED_FROM.txt in release_train)
```

Arrays are rectangular and padded: past `episode_lengths[e]` sits one `-1`
stop token, then NaN. The loader (`src/dlogps/data/dataset.py::load_cable`)
strips the padding, keeps positions, velocities and time, and drops any episode
in which a vertex moves more than 5 cm between two frames: a solver divergence,
where the cable jumps metres in a single frame. Five recorded episodes do, and
the released files leave them out: in `release_train`, cable 029 (episodes 45
and 78 of the merged file) and cable 037 (86); in `seen_cables_test`, cable 029
(22) and cable 037 (15). The loader always dropped them, so no reported number
changes, and each affected cable's `params.yaml` lists the original indices
under `release_filter`. The model reads only `vertex_pos`, `vertex_vel`, `t` and
the five scalar parameters.

Episode counts are read from `episode_lengths` or `index.csv`. A training
cable's `params.yaml` is the seed-0 half's, so its release record covers 50
episodes, except for cables 029 and 037, whose files were rebuilt from both
halves. In the test roots, `datagen.yaml` records the generation run (30
episodes per cable).

Two complete cables of each kind are bundled as fixtures: `assets/sample_v1`
(episodes 0 and 1 of cables 000 and 045, from the seed-0 half) and
`assets/sample_test_v1` (episodes 50 and 51 of the same cables, from the seed-1
half). They are exact copies of episodes in `release_train` and drive
`scripts/smoke.sh` and the test suite.

## Obtaining the data

The three roots are on the Hugging Face Hub as
[avihaig/dlogps-cables](https://huggingface.co/datasets/avihaig/dlogps-cables),
about 26 GB. Download them under one directory and point the checkout at it:

```bash
hf download avihaig/dlogps-cables --repo-type dataset --local-dir /path/to/dlogps-data   # pip install -U huggingface_hub
export DLOGPS_DATA=/path/to/dlogps-data
scripts/link_data.sh                         # creates the data/ symlink
```

`release_train` is 16.6 GB, `unseen_cables_test` 4.4 GB and `seen_cables_test`
5.0 GB. Scoring the released checkpoints needs only the test root: add
`--include "unseen_cables_test/*"` to the download.
