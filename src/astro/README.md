# Astrophysical Cross-Modal Experiment (LAMOST × Kepler / LAMOST × TESS)

This module contains the CA vs CP training and evaluation code for the astrophysical
multimodal experiment: LAMOST optical spectra (strong modality) paired with
Kepler or TESS photometric light curves (weak modality).

## Setup

### Required external resources

The frozen pretrained encoders and raw data are **not** included in this repository. The scripts expect frozen pretrained encoders
to be loaded from the paths specified by the environment variables below.

| Variable | Description | Example |
|---|---|---|
| `MULTIDESA_ROOT` | Root of the MultiDESA repo (for encoder configs + checkpoints) | `/path/to/MultiDESA` |
| `CROSSMATCH_PARQUET` | LAMOST×Kepler cross-match catalog | `/path/to/multimodal_df_min2.parquet` |
| `LAMOST_HDF5_DIR` | Directory of HEALPix-tiled LAMOST HDF5 files | `/path/to/lamost/hdf5_data` |
| `KEPLER_NPY_DIR` | Directory of per-star Kepler `.npy` files | `/path/to/kepler/raw_npy` |
| `AGE_CATALOG` | Gyrochronology age catalog CSV | `/path/to/ages_dataset_gyro.csv` |
| `NSS_CATALOG` | Non-single-star (binarity) catalog CSV | `/path/to/nss_dataset.csv` |

If `MULTIDESA_ROOT` is **not** set, the scripts fall back to loading the encoder
model code from `src/astro/nn/` (bundled in this repo).

### Install dependencies

```bash
pip install torch numpy pandas scikit-learn matplotlib astropy astropy_healpix h5py pyyaml einops
```

## Contents

| File | Description |
|---|---|
| `cross_modal.py` | Main training script: CA/CP/Joint heads on LAMOST×Kepler frozen features |
| `cross_modal_tess.py` | Same for LAMOST×TESS |
| `extract_features.py` | Extract and cache features from frozen encoders (run once) |
| `run_baselines.py` | Unimodal + supervised baselines on cached features |
| `nn/` | Encoder architectures (SpectralViT, DoubleInputRegressor) copied from MultiDESA |

## Usage

**Step 1: Extract features** (requires raw data + GPU, run once)

```bash
export MULTIDESA_ROOT=/path/to/MultiDESA
export CROSSMATCH_PARQUET=/path/to/multimodal_df_min2.parquet
export LAMOST_HDF5_DIR=/path/to/lamost/hdf5_data
export KEPLER_NPY_DIR=/path/to/kepler/raw_npy
python -m src.astro.extract_features
```

**Step 2: Train CA/CP/Joint heads on cached features**

```bash
python -m src.astro.cross_modal --mode all --use_cached_features
```

**Step 3: Run baselines**

```bash
python -m src.astro.run_baselines
```
