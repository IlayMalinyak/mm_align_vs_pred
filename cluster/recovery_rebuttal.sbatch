#!/bin/bash -l
#SBATCH --job-name=recovery_rebuttal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --partition=rtx6k-shared
#SBATCH --qos=12h_4g
#SBATCH -t 03:00:00
#SBATCH --output=/home/ilay.kamai/athena/logs/%x_%j.out
#SBATCH --error=/home/ilay.kamai/athena/logs/%x_%j.err

# TASK A (resampling-scheme diagnostic) + TASK B (sweep without replacement).
# CPU-only. Writes markdown tables to src/logs/recovery_rebuttal_output.md.
# Uses real /rg paths (the /home/ilay.kamai/work symlink is NOT visible inside the
# container) and binds /rg so MultiDESA/analysis + data are importable.

set -euo pipefail

RG=/rg/perets_prj/ilay.kamai
MD=$RG/MultiDESA
REPO=$RG/mm_align_vs_pred
OUT=$REPO/src/logs/recovery_rebuttal_output.md

export APPTAINER_BIND="$RG"

srun --ntasks=1 \
  --output=/home/ilay.kamai/athena/logs/%x_%j_%N_%t.out \
  --error=/home/ilay.kamai/athena/logs/%x_%j_%N_%t.err \
  --container=/apps/apptainer/images/pytorch/latest/torch2.7-tf2.20-cu128.sif \
  bash -lc "
    set -e
    pip install --quiet anndata scanpy scikit-misc astropy
    cd $REPO
    export PYTHONUNBUFFERED=1
    export MULTIDESA_ANALYSIS=$MD/analysis
    export MULTIMODAL_DF=$RG/multimodal/multimodal_df_min2.parquet
    export BMMC_RNA_EMB=$RG/citeseq/bmmc/rna_embeddings_scgpt.npy
    export BMMC_H5AD=$RG/citeseq/bmmc/neurips2021_cite_BMMC.h5ad
    python -u -m src.recovery_rebuttal --task AB --datasets kepler tess bmmc | tee $OUT
  "
