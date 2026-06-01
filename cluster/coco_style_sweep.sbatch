#!/bin/bash
#SBATCH --job-name=coco_style_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx6k:6
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=rtx6k-shared
#SBATCH --qos=72h_8g
#SBATCH -t 72:00:00

set -euo pipefail

# Binding current workspace
export APPTAINER_BIND="${PROJECT_ROOT},${PROJECT_ROOT},${HOME}/work/coco2017"

# COCO data root
export COCO_ROOT=${HOME}/work/coco2017

srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
    --output=${HOME}/athena/logs/%x_%j_%N_%t.out \
    --error=${HOME}/athena/logs/%x_%j_%N_%t.err \
    --container=/apps/apptainer/images/pytorch/latest/torch-tf-cu130.sif \
    bash -lc "cd ${PROJECT_ROOT} && NPROC=6 bash ${PROJECT_ROOT}/src/coco/run_coco_style_sweep.sh"
