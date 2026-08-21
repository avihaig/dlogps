# Results

`runs/<experiment>/<label>/` holds the small record of each reported run —
the resolved `config.yaml`, `env.json`, the training log and the unseen-test
scorecard under `eval/model/` — and `tables/` holds the paper's two tables as
regenerated from those records by `scripts/make_tables.py` (docs/reproduction.md).
Checkpoints (`step_100000.pt`, ~3.5 MB each) are release assets;
`scripts/fetch_checkpoints.sh` places them beside the records.

## Run ↔ table row

| table row | experiment | labels (seeds 0, 1, 2) | config |
|---|---|---|---|
| local on · Unbiased | `local_on` | `varA_s0` `varA_s1` `varA_s2` | `local_on.yaml --variant A` |
| local on · Euclidean | `local_on` | `varB_s*` | `--variant B` |
| local on · Chain | `local_on` | `varC_s*` | `--variant C` |
| local on · Mixed | `local_on` | `varD_s*` | `--variant D` |
| local on · Euclidean+Chain only | `local_on` | `varF_s*` | `--variant F` |
| local off · (same five) | `local_off` | `varA_s*` … `varF_s*` | `local_off.yaml` |
| chain only | `chain_only` | `chain_s*` | `chain_only.yaml` |
| MLP | `structfree` | `mlp_pernode_s*` | `structfree.yaml --model.arch mlp_pernode --model.d_model 2048` |
| LSTM | `structfree` | `lstm_global_s*` | `structfree.yaml --model.arch lstm_global --model.d_model 256` |

Table 2 uses the `local_on` rows only.

## What this checkout contains

| experiment | records present | cells recomputed |
|---|---|---|
| `local_on` | 15 of 15 (`unseen_test_step100000_summary.json`) | Table 1 local-on block: all 15 cells match the paper |
| `local_off` | pending sync | transcribed from the paper |
| `chain_only` | pending sync | transcribed from the paper |
| `structfree` | pending sync | transcribed from the paper |
| Table 2 phase columns | pending (the local-on runs are re-scored with `error_by_floor_phase`) | transcribed from the paper |

`python scripts/make_tables.py` prints the current count and flags any
recomputed cell that disagrees with the paper; `tables/*.csv` carry the per-cell
`source` column.
