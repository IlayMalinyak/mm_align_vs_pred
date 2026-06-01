# When to Align, When to Predict: A Phase Diagram for Multimodal Learning

This repository provides code for the paper **"When to Align, When to Predict: A Phase Diagram
for Multimodal Learning"**. We investigate when Cross-Alignment (CA, contrastive)
vs. Cross-Prediction (CP, predictive) objectives recover shared signal in multimodal data,
under the Spiked Covariance Model framework.

## Key idea

We characterize a **phase diagram** over (κ, ν) — signal strength and nuisance correlation —
with four regimes (Neither / CA only / CP only / Both). We validate our theoretical analysis
on different experiments: synthetic vision datasets, real image-caption dataset, and real astrophysical multimodal dataset.

## Interactive demo

The Colab notebook lets you explore CA vs CP recovery interactively with sliders:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IlayMalinyak/mm_align_vs_pred/blob/main/notebooks/interactive_linear_experiment_colab.ipynb)


## Install

```bash
pip install -e .
pip install datasets huggingface_hub   # for ImageNet experiments
pip install umap-learn                  # optional, for UMAP visualizations
```

## Running experiments

**Linear/theoretical** (fast, no GPU):
```bash
python -m src.linear.run_experiment
```

**dSprites sweep** (SLURM):
```bash
sbatch cluster/dsprites_sweep.sbatch
```


**Astrophysical** (LAMOST×Kepler):
```bash
python -m src.astro.cross_modal --mode all --use_cached_features
```

See `src/astro/README.md` for setup instructions for the astrophysical experiments.

## Phase diagram estimation

To estimate (κ, ν) for a new dataset and predict which method will work better:
```bash
python src/analyze_phase_diagram.py --modality_x /path/to/X.npy --modality_y /path/to/Y.npy
```

## Tests

```bash
pytest tests/
```
