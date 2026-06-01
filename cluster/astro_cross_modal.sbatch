#!/bin/bash
#SBATCH --job-name=astro_cross_modal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=rtx6k-shared
#SBATCH --qos=12h_4g
#SBATCH -t 12:00:00

# Usage:
#   sbatch --export=ALL,MULTIDESA_ROOT=/path/to/MultiDESA,PAIR=kepler cluster/astro_cross_modal.sbatch
#
# Required env vars (set before submitting or pass via --export):
#   MULTIDESA_ROOT      — path to MultiDESA repo (contains configs/ and logs/ with checkpoints)
#   CROSSMATCH_PARQUET  — path to multimodal_df_min2.parquet
#   LAMOST_HDF5_DIR     — directory of HEALPix-tiled LAMOST HDF5 files
#   KEPLER_NPY_DIR      — directory of per-star Kepler .npy files (for PAIR=kepler)
#
# Optional:
#   PAIR                — "kepler" (default) or "tess"
#   AGE_CATALOG         — path to ages_dataset_gyro.csv
#   NSS_CATALOG         — path to nss_dataset.csv
#   ASTRO_CACHE_DIR     — where to read/write cached features

set -euo pipefail

PAIR=${PAIR:-kepler}

export APPTAINER_BIND="${PROJECT_ROOT},${MULTIDESA_ROOT}"

srun --ntasks=1 \
    --output=${HOME}/athena/logs/%x_%j_%N_%t.out \
    --error=${HOME}/athena/logs/%x_%j_%N_%t.err \
    --container=/apps/apptainer/images/pytorch/latest/torch2.7-tf2.20-cu128.sif \
    bash -lc "
        pip install einops astropy astropy_healpix h5py pyyaml
        cd ${PROJECT_ROOT}
        export PYTHONUNBUFFERED=1

        export MULTIDESA_ROOT=${MULTIDESA_ROOT}
        export CROSSMATCH_PARQUET=${CROSSMATCH_PARQUET:-/path/to/multimodal_df_min2.parquet}
        export LAMOST_HDF5_DIR=${LAMOST_HDF5_DIR:-/path/to/lamost/hdf5_data}
        export KEPLER_NPY_DIR=${KEPLER_NPY_DIR:-/path/to/kepler/raw_npy}
        export AGE_CATALOG=${AGE_CATALOG:-/path/to/ages_dataset_gyro.csv}
        export NSS_CATALOG=${NSS_CATALOG:-/path/to/nss_dataset.csv}
        export ASTRO_CACHE_DIR=${ASTRO_CACHE_DIR:-${MULTIDESA_ROOT}/logs/cross_modal/cached_features}

        if [ \"${PAIR}\" = \"tess\" ]; then
            python -m src.astro.cross_modal_tess \
                --mode all \
                --use_cached_features \
                --epochs 200 \
                --batch_size 256 \
                --lr 1e-3 \
                --patience 30 \
                --num_workers 0 \
                --print_every 5
        else
            python -m src.astro.cross_modal \
                --mode all \
                --use_cached_features \
                --epochs 200 \
                --batch_size 256 \
                --lr 1e-3 \
                --patience 30 \
                --num_workers 0 \
                --print_every 5
        fi
    "
