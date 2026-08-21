# Experiments

## Training protocol

`configs/experiments/*.yaml`, `src/dlogps/harness/train.py`

| | |
|---|---|
| loss | mean squared error on the standardized one-step displacement |
| input noise | random-walk velocity noise, `σ = 0.1` m/s per axis at the newest frame (`harness/noise.py`) |
| optimizer | Adam, gradient clipping at 1.0, exponential learning-rate decay from 1e-3 to 1e-4 over the run |
| schedule | 100k steps at batch 512; the reported checkpoint is the **last** one (no selection on the test root) |
| history | `C = 70` frames of velocity per vertex |
| regularization | weight decay 0, dropout 0 |
| backbone | `d = 128`, `H = 8`, `L = 4`; 891k parameters |
| encoding | `param_encoding: designed` (docs/method.md) |
| data | trained on `data/release_train` (46 cables × 100 episodes), no within-population holdout |

Width, noise scale, history depth and regularization were chosen on validation
windows drawn from the training population, never on the test cables; the
values above are the frozen result of that selection and the only ones this
repository runs.

## Rollouts and evaluation

`src/dlogps/harness/rollout.py`, `src/dlogps/harness/evaluate.py`

Evaluation is autoregressive: each predicted state is fed back as the next
input, and the Euclidean bias is recomputed from the predicted positions at every
step. Each rollout has 400 predicted frames (a 401-frame window whose frame 0 is
the observed input state). Windows start at a stride of the rollout length
inside every test episode, and a stratified cap of 400 windows per stage is
taken round-robin across cables, so every arm is scored on the same windows,
joined by `(cable, episode, start frame)`. The test root is
`data/unseen_cables_test`: 40 cables never present in training.

## Comparison design

The main matrix is the five bias variants at three seeds (0, 1, 2), with the
local stream on and with it off, plus three baselines at three seeds:
**39 training runs**.

| experiment | config | arms | runs | labels |
|---|---|---|---|---|
| local stream on | `local_on.yaml` | A, B, C, D, F | 15 | `varA_s0` … `varF_s2` |
| local stream off | `local_off.yaml` (`use_local: false`) | A, B, C, D, F | 15 | `varA_s0` … `varF_s2` |
| chain only | `chain_only.yaml` (`variant: chain_only`) | — | 3 | `chain_s0` … `chain_s2` |
| structure-free | `structfree.yaml` | `mlp_pernode` (width 2048), `lstm_global` (width 256) | 6 | `mlp_pernode_s0` …, `lstm_global_s0` … |

`scripts/train_matrix.sh` and `scripts/train_baselines.sh` launch them; each run
writes `results/<experiment>/runs/<timestamp>_<label>/` with the resolved
`config.yaml`, `env.json`, the training log, the final checkpoint and the
unseen-test scorecard (docs/reproduction.md).

Reporting: each seed is reduced to its mean over the 400 windows together with
that seed's window standard deviation; a table cell is the mean of the three
seed means ± the mean of the three window sds ± the standard deviation of the
three seed means. Bold marks the lowest mean in each block (Table 1) or column
(Table 2). No significance bar is applied; the paper reads the local-on ranking
as a three-seed table order, not a separated effect. `scripts/make_tables.py`
computes every cell from the per-run records under `results/runs/`
(results/README.md).

## Compute

One consumer GPU. On an RTX 5090 a local-on arm trains its 100k steps in about
40–50 minutes and scores its final checkpoint in a few more; the 39 runs are
roughly 30 GPU-hours in total.
