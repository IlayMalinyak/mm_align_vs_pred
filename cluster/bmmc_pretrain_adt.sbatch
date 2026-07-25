#!/bin/bash -l
#SBATCH --job-name=bmmc_pretrain_adt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s-shared
#SBATCH --qos=12h_4g
#SBATCH -t 2:00:00
#SBATCH --output=/home/ilay.kamai/athena/logs/%x_%j.out
#SBATCH --error=/home/ilay.kamai/athena/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

module load cuda/12.1.1-x86-nj6dyov
source /home/ilay.kamai/venvs/scgpt-env/bin/activate

cd /home/ilay.kamai/work/mm_align_vs_pred

python -m src.bmmc.pretrain_adt \
    --epochs 200 \
    --patience 10 \
    --batch_size 1024 \
    --embed_dim 512 \
    --hidden_dims 256 512 \
    --noise_std 0.1 \
    --dropout_p 0.1
