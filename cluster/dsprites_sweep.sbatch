#!/bin/bash -l
#SBATCH --job-name=dsprites_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1                   
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=l40s-public
#SBATCH --qos=12h_4g
#SBATCH -t 12:00:00

set -euo pipefail

# Binding current workspace
export APPTAINER_BIND="${PROJECT_ROOT},${PROJECT_ROOT}"

srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
    --output=${HOME}/athena/logs/%x_%j_%N_%t.out \
    --error=${HOME}/athena/logs/%x_%j_%N_%t.err \
    --container=/apps/apptainer/images/pytorch/latest/torch2.7-tf2.20-cu128.sif \
    bash -lc "pip install datasets huggingface_hub && cd ${PROJECT_ROOT} && bash ${PROJECT_ROOT}/src/experiments/run_dsprites_sweep.sh"
