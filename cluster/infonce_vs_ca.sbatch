#!/bin/bash -l
#SBATCH --job-name=infonce_vs_ca
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx6k-shared
#SBATCH --qos=12h_4g
#SBATCH -t 02:00:00
#SBATCH --output=/home/ilay.kamai/athena/logs/%x_%j.out
#SBATCH --error=/home/ilay.kamai/athena/logs/%x_%j.err

# InfoNCE vs CA/CP in the linear spiked model (Figure-3 / E4 setup).
# GPU (torch); ~30 min for the 15 ν × 20 seeds × 3 τ grid. Reuses the E4 data
# generator + CA/CP closed forms verbatim; only adds NT-Xent gradient training.
# Real /rg paths (the /home/ilay.kamai/work symlink is NOT visible in-container).

set -euo pipefail

RG=/rg/perets_prj/ilay.kamai
REPO=$RG/mm_align_vs_pred
OUT=$REPO/src/logs/infonce_vs_ca_table.md

export APPTAINER_BIND="$RG"

srun --ntasks=1 \
  --output=/home/ilay.kamai/athena/logs/%x_%j_%N_%t.out \
  --error=/home/ilay.kamai/athena/logs/%x_%j_%N_%t.err \
  --container=/apps/apptainer/images/pytorch/latest/torch2.7-tf2.20-cu128.sif \
  bash -lc "
    set -e
    cd $REPO
    export PYTHONUNBUFFERED=1
    python -u -m src.linear.infonce_vs_ca --device cuda --out $OUT
  "
