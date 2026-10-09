# Results

`runs/<experiment>/<label>/` holds the small record of each reported run —
the resolved `config.yaml`, `env.json`, the training log and the unseen-test
scorecard under `eval/model/` — and `tables/` holds the paper's two tables as
regenerated from those records by `scripts/make_tables.py` (docs/reproduction.md).
Checkpoints (`step_100000.pt`, 2 to 18 MB each) are on the Hugging Face Hub,
[avihaig/dlogps-checkpoints](https://huggingface.co/avihaig/dlogps-checkpoints),
with each run's per-window, per-cable and per-frame scores;
`scripts/fetch_checkpoints.sh` places them beside the records.

The records keep the names the runs had during development (`config.yaml`
says `experiment: b7_encoding`, `label: varF_enc_s0`, and so on).
`runs/provenance.csv` maps each folder to its source run, the development
commit it ran from, and whether that working tree had uncommitted changes.

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

The record of all 39 runs. The 15 `local_on` runs predate the phase split, so
each also carries `eval/rescore/unseen_test_step100000_summary.json`: its
checkpoint re-scored with `scripts/evaluate_final.sh` on the released test root.
Table 2 reads its two metrics from there; Table 1 reads each run's own
scorecard. The re-scoring reproduces the run's own Table 1 metrics to within
0.1%, the floating-point difference between the two GPUs.

`python scripts/make_tables.py --check` recomputes all 49 cells of Tables 1 and
2 from these records and checks each against the paper at its printed
precision; `tables/*.csv` carry the per-cell `source` column.
