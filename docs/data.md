# Data

## Generation

Every trajectory is simulated in **MuJoCo 3.9.0**. The generator itself builds
on an unpublished MuJoCo scene package and is not part of this repository; the
complete protocol and every physical value it used are recorded in
`configs/release.yaml` (and the two files that extend it), and the populations
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

| root | cables | episodes / cable | role |
|---|---|---|---|
| `release_v1` | 46 | 50 | training, half 1 (`configs/release.yaml`, seed 0) |
| `test_v1` | 46 (the same) | 50 | training, half 2 (`configs/release_test.yaml`, seed 1: new release poses) |
| `release_train` | 46 | 100 | the union of the two, built by `scripts/merge_release_train.py`; **every reported run trains on it** |
| `unseen_cables_test` | 40 | 30 | **the reported test set** (`configs/unseen_cables_test.yaml`, seed 3) |

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
  index.csv, datagen.yaml   the population and the generator's provenance
```

Arrays are rectangular and padded: past `episode_lengths[e]` sits one `-1`
stop token, then NaN. The loader (`src/dlogps/data/dataset.py::load_cable`)
strips the padding, keeps positions, velocities and time, and drops any episode
in which a vertex moves more than 5 cm between two frames (a solver divergence;
none occurs in the released roots). The model reads only `vertex_pos`,
`vertex_vel`, `t` and the five scalar parameters.

Two complete cables of each kind are bundled as fixtures:
`assets/sample_v1` (two episodes each of cables 000 and 045, released with the
training seed) and `assets/sample_test_v1` (the same two cables, new poses).
They are real subsets of the released data and drive `scripts/smoke.sh` and the
test suite.

## Obtaining the data

The released roots total about 16 GB. *(Download link: to be added — the roots
are being uploaded; until then they are available from the authors on request.)*
Place them under one directory and point the checkout at it:

```bash
export DLOGPS_DATA=/path/to/dlogps-data      # holds release_v1/ test_v1/ unseen_cables_test/
scripts/link_data.sh                         # creates the data/ symlink
python scripts/merge_release_train.py --inputs data/release_v1 data/test_v1 --out data/release_train
```
