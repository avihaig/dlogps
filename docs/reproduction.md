# Reproduction

Three levels, from minutes to GPU-days.

## 1. Regenerate the tables from the shipped records (seconds)

```bash
python scripts/make_tables.py --check
```

reads `results/runs/<experiment>/<label>/eval/model/unseen_test_step100000_summary.json`
for every run, recomputes each cell of Tables 1 and 2 and compares it with the
paper at its printed precision (`results/tables/paper_values.csv`). The outputs
are `results/tables/table{1,2}_*.{csv,md,tex}`; the CSVs' `source` column says
whether a cell was recomputed from records or transcribed from the paper because
the records are not yet in the checkout (results/README.md lists what is
there). `tests/test_tables.py` runs the same check.

## 2. The smoke run: the whole pipeline on the bundled sample (minutes, CPU)

```bash
uv sync                      # or: pip install -e .
scripts/smoke.sh             # DEVICE=cuda scripts/smoke.sh for a GPU
```

trains the block for 200 steps on `assets/sample_v1` (two cables, two episodes
each), takes a checkpoint, rolls it out on `assets/sample_test_v1`, scores the
windows and writes figures. The run lands in
`results/smoke/runs/<timestamp>_smoke/`:

```
config.yaml              the resolved configuration, complete
env.json                 git sha, python/torch/cuda versions, device, host, timings
train_log.jsonl / .txt   one record per report step (train loss, held-out loss, lr)
checkpoints/step_00200.pt
eval/model/unseen_test_step200_summary.json     every metric: mean, across-window sd, count, unit
eval/model/unseen_test_step200_windows.csv      one row per rollout window (cable, episode, start frame, metrics)
eval/model/unseen_test_step200_per_cable.csv    one row per cable
eval/model/unseen_test_step200_curve.csv        one row per predicted frame
figures/loss_mse.png
```

`scripts/smoke.sh --variant F`, `--variant chain_only`, or
`--model.arch lstm_global --model.d_model 32 --model.n_layers 3` exercise the
other arms. A full-size run has exactly this layout with `step_100000`.

## 3. The paper's 39 runs (about 30 GPU-hours on an RTX 5090)

1. **Data** (docs/data.md): place `release_v1/`, `test_v1/` and
   `unseen_cables_test/` under one directory, then
   ```bash
   export DLOGPS_DATA=/path/to/dlogps-data
   scripts/link_data.sh
   python scripts/merge_release_train.py --inputs data/release_v1 data/test_v1 --out data/release_train
   ```
2. **Train and score** (each arm ends by scoring its final checkpoint on the
   400 unseen-test windows; completed arms are skipped on restart):
   ```bash
   scripts/train_matrix.sh --local on       # A B C D F x seeds 0 1 2
   scripts/train_matrix.sh --local off
   scripts/train_baselines.sh               # chain_only, mlp_pernode, lstm_global x seeds 0 1 2
   ```
   Every option is a config key: `scripts/train_matrix.sh --variants "D F" --seeds "0"`,
   or directly `python -m dlogps.harness.run_experiment --config_path configs/experiments/local_on.yaml --variant D --train.seed 0 --label varD_s0`.
3. **Tables**: copy (or symlink) each finished run directory to
   `results/runs/<experiment>/<label>/` and run `python scripts/make_tables.py`.
   Seeds give bit-identical runs on the same hardware and library versions;
   across GPUs expect the third digit to move.

### Scoring a released checkpoint instead of training

```bash
scripts/fetch_checkpoints.sh                   # GitHub release assets -> results/runs/**/checkpoints/
scripts/evaluate_final.sh results/runs/local_on/varF_s0 [...]
```

`evaluate_final.sh` runs `python -m dlogps.harness.evaluate` with the paper's
protocol (`--stage unseen_test --horizon 401 --horizons 1,10,50,100,200,400
--limit 400`) and writes the scorecard into the run directory, including the
phase-split columns Table 2 is built from.

## Environment

Python ≥ 3.13, `torch == 2.11.0` (the version of every reported run; the stock
PyPI wheel includes CUDA), numpy, pyyaml, draccus, matplotlib, tqdm.
`uv sync` installs the locked set (`uv.lock`); `requirements.txt` is the same
set for pip. The test suite (`uv run pytest`, ~10 s on a CPU) needs neither
the dataset nor a GPU.
