# NMF-Based Music Source Separation

EE269 Final Project — Stanford University, Winter 2026

Compares four NMF variants for separating musical stems (drums, bass, other, vocals) from mixture audio using the MUSDB18 dataset.

## Methods

| Method | Divergence |
|---|---|
| Euclidean NMF | Frobenius norm (Lee & Seung 2000) |
| IS-NMF | Itakura-Saito divergence (Févotte et al. 2009) |
| Sparse NMF | Frobenius + ℓ1 penalty on H (Hoyer 2004) |
| Convolutive NMF | Shift-invariant / KL divergence (Smaragdis 2004) |

All methods use supervised dictionary learning: source-specific dictionaries W are trained on MUSDB18 training stems, then fixed at test time while H is optimized on the mixture.

## Repo Structure

```
src/
  nmf.py              # NMF model classes
  audio.py            # audio I/O and STFT utilities
  train.py            # supervised dictionary learning
  evaluate.py         # BSS Eval scoring on test set
  run_experiments.py  # end-to-end pipeline runner
  make_figures.py     # SDR-vs-K, runtime, and poster figures
figures/              # generated plots (PDF + PNG)
results/              # weights, scores, and summary CSV
musdb18/              # MUSDB18 dataset (not tracked)
main.tex              # LaTeX report source
references.bib
```

## Setup

```bash
pip install numpy librosa stempeg museval matplotlib tqdm
```

MUSDB18 dataset: download from [sigsep.github.io/datasets/musdb.html](https://sigsep.github.io/datasets/musdb.html) and place as `musdb18/train/` and `musdb18/test/`.

## Usage

**Full pipeline** (train → evaluate → plot):
```bash
python src/run_experiments.py
```

**Smoke test** (fast, 2 tracks, skips CNMF):
```bash
python src/run_experiments.py --smoke-test
```

**Skip training** (reuse existing weights):
```bash
python src/run_experiments.py --skip-train
```

**Run specific methods only:**
```bash
python src/run_experiments.py --methods euclidean sparse
```

**Regenerate figures only** (requires `results/results.json`):
```bash
python src/make_figures.py
```

Individual scripts (`train.py`, `evaluate.py`) can also be run standalone — see `--help` for options.

## Results

Evaluation uses median SDR/SIR/SAR per source via `museval`. Summary saved to `results/summary.csv`. Trained dictionaries saved as `results/weights_{method}.npz`.
