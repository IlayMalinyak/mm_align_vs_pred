#!/bin/bash -l
#SBATCH --job-name=recovery_rebuttal_bmmc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --partition=rtx6k-shared
#SBATCH --qos=12h_4g
#SBATCH -t 06:00:00
#SBATCH --output=/home/ilay.kamai/athena/logs/%x_%j.out
#SBATCH --error=/home/ilay.kamai/athena/logs/%x_%j.err

# RESCUE job: TASK B for BMMC only (skips TASK A + astro score-matrix load), so
# the full wall goes to the BMMC ladder incl. the heavy m=72000 x 50-rep block.
# Only needed if cluster/recovery_rebuttal.sbatch dies at its 3h wall before
# finishing BMMC m=72000. Writes to a SEPARATE file so it cannot clobber the
# main output; the BMMC tables are then merged into recovery_rebuttal_output.md.
# Real /rg paths (the /home/ilay.kamai/work symlink is NOT visible in-container).

set -euo pipefail

RG=/rg/perets_prj/ilay.kamai
MD=$RG/MultiDESA
REPO=$RG/mm_align_vs_pred
OUT=$REPO/src/logs/recovery_rebuttal_bmmc.md

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
    python -u -m src.recovery_rebuttal --task B --datasets bmmc | tee $OUT
  "
