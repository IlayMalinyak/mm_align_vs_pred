# When to Align, When to Predict?
### A Phase Diagram for Multimodal Self-Supervised Learning

<div align="center">

[![Paper](https://img.shields.io/badge/paper-arxiv-red)](https://arxiv.org/abs/2606.11190)
[![HuggingFace](https://img.shields.io/badge/🤗-dataset-yellow)](https://huggingface.co/datasets/Ilayk/mm_align_vs_pred)
[![Website](https://img.shields.io/badge/project-website-blue)](https://ilaymalinyak.github.io/mm_align_vs_pred/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IlayMalinyak/mm_align_vs_pred/blob/main/notebooks/interactive_linear_experiment_colab.ipynb)

</div>

<div align="center">
<img src="figs/phase_diagram_sweep.gif" width="480"/>
</div>

> As target-modality nuisance variance (γ̃_y) grows, the **CP region collapses** — Cross-Prediction gets trapped reconstructing high-variance noise instead of the shared signal. Cross-Alignment is immune.

---

## Overview

We characterize *when* contrastive (Cross-Alignment, CA) vs. predictive (Cross-Prediction, CP) self-supervised objectives recover shared signal in multimodal data, under the **Spiked Covariance Model** framework.

The answer is a **phase diagram** over (κ, ν) — signal strength vs. nuisance cross-modal correlation — with four regimes:

| Region | CA recovers? | CP recovers? |
|--------|:---:|:---:|
| **Both** | ✓ | ✓ |
| **CA only** | ✓ | ✗ |
| **CP only** | ✗ | ✓ |
| **Neither** | ✗ | ✗ |

We validate on synthetic vision datasets (dSprites, Shapes3D), image-caption data (COCO), real astrophysical data (LAMOST × Kepler/TESS spectra), and single-cell multi-omics (CITE-seq BMMC, RNA × surface protein).

---

## Datasets

| Dataset | How to get it |
|---------|--------------|
| **dSprites** | Bundled in `src/dsprites/` (2.5 MB `.npz`) |
| **Shapes3D** | Auto-downloaded from [Google](https://storage.googleapis.com/3d-shapes/3dshapes.h5) |
| **COCO** | `pip install pycocotools` + standard download |
| **Astro** | Pretrained encoders & cached features on [HuggingFace](https://huggingface.co/datasets/Ilayk/mm_align_vs_pred) |
| **BMMC (CITE-seq)** | NeurIPS 2021 multimodal single-cell benchmark, CITE-seq BMMC `.h5ad` ([GEO GSE194122](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194122)) + the [scGPT](https://github.com/bowang-lab/scGPT) whole-human checkpoint |

---

## Pretrained Models & Cached Features

```bash
# Download all (checkpoints + cached features + phase diagnostics data)
huggingface-cli download Ilayk/mm_align_vs_pred --repo-type dataset --local-dir hf_data
```

Or in Python:
```python
from huggingface_hub import snapshot_download
snapshot_download("Ilayk/mm_align_vs_pred", repo_type="dataset", local_dir="hf_data")
```

Point the astro experiment at the downloaded files:
```bash
export MULTIDESA_ROOT=hf_data
```

---

## Install

```bash
pip install -e .
pip install umap-learn   # optional, for UMAP visualizations
```

---

## Running Experiments

**Linear / theoretical** (fast, CPU-only):
```bash
python -m src.linear.run_experiment
```

**Synthetic vision** (dSprites / Shapes3D, SLURM):
```bash
sbatch cluster/dsprites_sweep.sbatch
sbatch cluster/shapes3d_sweep.sbatch
```

**COCO** (image-caption, SLURM):
```bash
sbatch cluster/coco_style_sweep.sbatch
```

**Astrophysical** (LAMOST × Kepler, requires HF data downloaded):
```bash
python -m src.astro.cross_modal --mode all --use_cached_features
```
See [`src/astro/README.md`](src/astro/README.md) for full setup.

**Single-cell** (CITE-seq BMMC: RNA × ADT protein, SLURM, needs `scgpt`):

~90k bone-marrow cells, RNA as frozen scGPT embeddings (512-d) × raw CLR ADT abundance (134 proteins).
The task is 44-class cell-type classification with a donor-based split (6 train / 1 val / 2 test donors).
```bash
sbatch cluster/bmmc_cross_modal.sbatch   # Stage 0 caches scGPT RNA embeddings, then trains CA / CP / CP← (5-seed array)
sbatch cluster/bmmc_phase.sbatch         # phase estimation (Δ̂_CA, Δ̂_CP) + diagnostics plot
```
The data paths (`.h5ad`, cached embeddings, scGPT checkpoint) are set at the top of
`src/bmmc/bmmc_dataset.py` and `src/bmmc/cross_modal_bmmc.py`; edit them for your machine.
An optional variant with a pretrained ADT encoder is in `src/bmmc/pretrain_adt.py` and
`cluster/bmmc_{pretrain_adt,cross_modal_pretrained,phase_pretrained}.sbatch`.

Expected result: the *Neither* regime (Δ̂_CA ≈ 0.67, Δ̂_CP ≈ 0.26). CA (0.606) > CP (0.572) > CP← (0.524)
in balanced accuracy, but no trained head beats concatenating the frozen unimodal embeddings (0.622).

---

## Phase Diagram Estimation

To estimate the CA and CP recovery regimes for your own dataset (paired embeddings plus labels for a small subsample):
```python
import sys; sys.path.insert(0, "src")
from analyze_phase_diagram import estimate_parameters

# X: (n, d_x) source embeddings, Y: (n, d_y) target embeddings (paired rows, centered)
# labels: (n, L) one-hot or continuous labels for the same samples
res = estimate_parameters(X, Y, labels, n_components=50)
print(res["delta_ca_direct"], res["delta_cp_direct"])   # > 1  =>  that objective recovers
print(res["r_hat_ca"], res["r_hat_cp"])                 # recovery counts (partial vs. complete failure)
```
`python src/analyze_phase_diagram.py <linear|dsprites|shapes3d|imagenet>` reruns the estimation for the paper's experiments.

---

## Tests

```bash
pytest tests/
```

---

## Citation

```bibtex
@article{,
  title   = {When to Align, When to Predict? A Phase Diagram for Multimodal Self-Supervised Learning},
  author  = {Kamai, Ilay and Van Assel, Hugues and Regev, Aviv and Perets, Hagai B. and Balestriero, Randall},
  journal = {arXiv preprint},
  year    = {2026}
}
```
