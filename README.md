# Predicting Cable Dynamics with Physical Attention Bias

[![arXiv](https://img.shields.io/badge/arXiv-2610.11975-b31b1b.svg)](https://arxiv.org/abs/2610.11975)
[![Paper page](https://huggingface.co/datasets/huggingface/badges/resolve/main/paper-page-sm.svg)](https://huggingface.co/papers/2610.11975)
[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-sm.svg)](https://huggingface.co/datasets/avihaig/dlogps-cables)
[![Models on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/avihaig/dlogps-checkpoints)

Avihai Giuili\*, Rotem Atari\*, Avishai Sintov (Tel Aviv University),
Maya Bechler-Speicher (Meta AI). \*Equal contribution.

Extended abstract at the **NeurIPS 2026 Workshop on Symmetry and Geometry in
Neural Representations (NeurReps)**.
Paper: [arXiv:2610.11975](https://arxiv.org/abs/2610.11975) ·
[PDF](paper/predicting-cable-dynamics-with-physical-attention-bias.pdf) ·
[LaTeX source](paper/src/) · [BibTeX](#citation).
Data: [avihaig/dlogps-cables](https://huggingface.co/datasets/avihaig/dlogps-cables) ·
checkpoints: [avihaig/dlogps-checkpoints](https://huggingface.co/avihaig/dlogps-checkpoints).

A cable has two pairwise distances that agree only while it is straight: the
arc-length distance along the chain, which governs elastic forces, and the
Euclidean distance in space, which governs contact. We add a *physical attention
bias*, an additive term on the global-attention logits with a learned rate per
head, to a GraphGPS-style learned simulator of a falling cable (MuJoCo, 46
training cables, 40 unseen test cables), and ask which distance it should use:
**none, Euclidean, chain, or both on disjoint sets of heads**. The block, the
training protocol and the evaluation windows stay fixed; every variant runs with
the local message-passing stream on and off, against chain-only, MLP and LSTM
baselines. A physical bias improves prediction on unseen cables, most when
attention is the only path between distant segments: there the chain bias cuts
rel ℓ2 by 15% and more than halves link-length drift. Assigning both distances
across heads is best or near-best on every metric; with the local stream on,
that ranking is a three-seed ordering inside the window spread.

## Results

Unseen-cable test, final checkpoint, 400 windows of 400-frame rollouts. Cell:
mean of seed means ± mean of window sds ± sd of seed means; bold = lowest mean in
the block. Regenerated from the per-run records by `scripts/make_tables.py`
(see [results/](results/README.md)).

| | rel ℓ2 (frac. of L) | link-length drift δ (frac. of ℓ0) | self-intersection v (frac. of frames) |
|---|---|---|---|
| *local stream on* | | | |
| Unbiased | 0.0325 ± 0.0281 ± 0.0058 | 0.0300 ± 0.0159 ± 0.0032 | 0.171 ± 0.231 ± 0.009 |
| Euclidean | 0.0320 ± 0.0267 ± 0.0042 | 0.0303 ± 0.0135 ± 0.0021 | 0.174 ± 0.229 ± 0.032 |
| Chain | 0.0316 ± 0.0341 ± 0.0057 | 0.0271 ± 0.0124 ± 0.0023 | 0.167 ± 0.227 ± 0.030 |
| Mixed | 0.0302 ± 0.0251 ± 0.0067 | **0.0267 ± 0.0109 ± 0.0030** | 0.168 ± 0.229 ± 0.012 |
| Euclidean+Chain only | **0.0297 ± 0.0288 ± 0.0035** | 0.0275 ± 0.0129 ± 0.0007 | **0.165 ± 0.229 ± 0.012** |
| *local stream off* | | | |
| Unbiased | 0.0399 ± 0.0246 ± 0.0002 | 0.1693 ± 0.1891 ± 0.0030 | 0.216 ± 0.258 ± 0.006 |
| Euclidean | 0.0382 ± 0.0279 ± 0.0007 | 0.1470 ± 0.2232 ± 0.0178 | 0.208 ± 0.262 ± 0.020 |
| Chain | **0.0339 ± 0.0195 ± 0.0008** | 0.0783 ± 0.0382 ± 0.0040 | 0.174 ± 0.241 ± 0.004 |
| Mixed | 0.0352 ± 0.0211 ± 0.0016 | **0.0779 ± 0.0455 ± 0.0036** | 0.172 ± 0.242 ± 0.016 |
| Euclidean+Chain only | 0.0343 ± 0.0202 ± 0.0002 | 0.0784 ± 0.0409 ± 0.0013 | **0.161 ± 0.234 ± 0.006** |
| *baselines* | | | |
| chain only | 0.0925 ± 0.0838 ± 0.0720 | 0.0812 ± 0.0589 ± 0.0322 | 0.115 ± 0.191 ± 0.024 |
| MLP | 0.0868 ± 0.0785 ± 0.0163 | 1.35 ± 2.32 ± 0.73 | 0.435 ± 0.256 ± 0.010 |
| LSTM | 0.1070 ± 0.0911 ± 0.0397 | 0.2819 ± 0.3124 ± 0.0509 | 0.082 ± 0.150 ± 0.031 |

Local-on rel ℓ2 split by phase of the recorded trajectory (floor contact: a
vertex at or below one diameter; 400 and 345 eligible windows):

| | floor contact | free flight |
|---|---|---|
| Unbiased | 0.0511 ± 0.0536 ± 0.0105 | 0.0062 ± 0.0095 ± 0.0001 |
| Euclidean | 0.0510 ± 0.0615 ± 0.0080 | 0.0062 ± 0.0099 ± 0.0005 |
| Chain | 0.0529 ± 0.0871 ± 0.0122 | 0.0064 ± 0.0131 ± 0.0014 |
| Mixed | 0.0484 ± 0.0551 ± 0.0135 | **0.0058 ± 0.0097 ± 0.0011** |
| Euclidean+Chain only | **0.0479 ± 0.0643 ± 0.0068** | 0.0061 ± 0.0127 ± 0.0010 |

## Repository map

| | |
|---|---|
| [paper/](paper/README.md) | the camera-ready PDF and its LaTeX source |
| [docs/method.md](docs/method.md) | representation, the hybrid block, the bias equation and the five variants, the baselines |
| [docs/data.md](docs/data.md) | the MuJoCo protocol, the populations, the on-disk format, how to obtain the data |
| [docs/experiments.md](docs/experiments.md) | training protocol, rollouts, the 39-run matrix, reporting rules, compute |
| [docs/metrics.md](docs/metrics.md) | rel ℓ2, link-length drift, self-intersection, the phase split, aggregation |
| [docs/reproduction.md](docs/reproduction.md) | regenerate the tables · smoke run · the full matrix · scoring released checkpoints |
| [src/dlogps/](src/dlogps/) | `data/` loader and labels · `model/` the GPS block, bias plugins, structure-free arms · `harness/` training, rollout, metrics, artifacts, the experiment runner |
| [configs/](configs/) | `experiments/{local_on,local_off,chain_only,structfree,smoke}.yaml` and the data-generation records (`release*.yaml`, `*_cables_test.yaml`, cable populations) |
| [scripts/](scripts/) | `train_matrix.sh` · `train_baselines.sh` · `evaluate_final.sh` · `make_tables.py` · `merge_release_train.py` · `smoke.sh` · `link_data.sh` · `fetch_checkpoints.sh` |
| [results/](results/README.md) | per-run records behind every table row, and the regenerated tables |
| [assets/](assets/sample_v1/README.md) | two real cables of the released data, for the smoke run and the tests |
| [tests/](tests/) | the suite (`uv run pytest`; CPU, no dataset needed) |

Variant letters in run labels and checkpoints: **A** Unbiased · **B** Euclidean ·
**C** Chain · **D** Mixed · **F** Euclidean+Chain only.

## Quick start

```bash
git clone https://github.com/avihaig/dlogps && cd dlogps
uv sync                       # or: python -m venv .venv && . .venv/bin/activate && pip install -e .
uv run pytest -q              # ~10 s
scripts/smoke.sh              # the whole pipeline on the bundled sample, minutes on a CPU
python scripts/make_tables.py --check   # Tables 1-2 from the shipped run records
```

Training the 39 runs needs the dataset (≈26 GB on the Hugging Face Hub,
[docs/data.md](docs/data.md)) and about 30 GPU-hours on one RTX 5090:

```bash
hf download avihaig/dlogps-cables --repo-type dataset --local-dir /path/to/dlogps-data   # pip install -U huggingface_hub
export DLOGPS_DATA=/path/to/dlogps-data && scripts/link_data.sh
scripts/train_matrix.sh --local on && scripts/train_matrix.sh --local off && scripts/train_baselines.sh
```

The 39 trained checkpoints are on the Hub too: `scripts/fetch_checkpoints.sh`
places them beside their records and `scripts/evaluate_final.sh` re-scores them
([docs/reproduction.md](docs/reproduction.md)).

## Citation

```bibtex
@inproceedings{giuili2026predicting,
  title         = {Predicting Cable Dynamics with Physical Attention Bias},
  author        = {Giuili, Avihai and Atari, Rotem and Sintov, Avishai and Bechler-Speicher, Maya},
  booktitle     = {NeurIPS 2026 Workshop on Symmetry and Geometry in Neural Representations (NeurReps)},
  year          = {2026},
  eprint        = {2610.11975},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2610.11975}
}
```

## License

MIT (see [LICENSE](LICENSE)).
