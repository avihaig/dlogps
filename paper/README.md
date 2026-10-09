# Paper

**Predicting Cable Dynamics with Physical Attention Bias.** Avihai Giuili\*,
Rotem Atari\*, Avishai Sintov, Maya Bechler-Speicher (\*equal contribution).
Extended abstract, NeurIPS 2026 Workshop on Symmetry and Geometry in Neural
Representations (NeurReps). [arXiv:2610.11975](https://arxiv.org/abs/2610.11975)

| | |
|---|---|
| [`predicting-cable-dynamics-with-physical-attention-bias.pdf`](predicting-cable-dynamics-with-physical-attention-bias.pdf) | the camera-ready |
| [`src/`](src/) | its LaTeX source, identical to arXiv v1: `main.tex`, `references.bib`, and the PMLR template (`jmlr.cls`, `jmlrutils.sty`, v1.30) |

arXiv builds the same source with pdflatex:

```bash
cd paper/src && latexmk -pdf main.tex
```

Tables 1 and 2 are regenerated from the per-run records by
`scripts/make_tables.py` (`results/tables/table{1,2}_*.tex`), and `--check`
compares every recomputed cell with the paper.
