#!/bin/bash -l
#SBATCH --job-name=bmmc_cross_modal
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=l40s-shared
#SBATCH --qos=12h_4g
#SBATCH -t 12:00:00
#SBATCH --output=/home/ilay.kamai/athena/logs/%x_%A_%a.out
#SBATCH --error=/home/ilay.kamai/athena/logs/%x_%A_%a.err

set -euo pipefail

export PYTHONUNBUFFERED=1

module load cuda/12.1.1-x86-nj6dyov
source /home/ilay.kamai/venvs/scgpt-env/bin/activate

cd /home/ilay.kamai/work/mm_align_vs_pred

RNA_EMB=/home/ilay.kamai/work/citeseq/bmmc/rna_embeddings_scgpt.npy
if [ ! -f "$RNA_EMB" ]; then
    echo "Stage 0: extracting scGPT RNA embeddings..."
    python -m src.bmmc.cross_modal_bmmc --extract_rna_only
else
    echo "Stage 0: scGPT embeddings already cached, skipping."
fi

python -m src.bmmc.cross_modal_bmmc \
    --rna_model scgpt \
    --mode all \
    --seed $SLURM_ARRAY_TASK_ID \
    --epochs 200 \
    --batch_size 512 \
    --lr 1e-3 \
    --patience 30 \
    --proj_layers 2 \
    --hidden_dim 512 \
    --proj_dim 256 \
    --ca_loss_type vicreg \
    --val_donors 1 \
    --test_donors 2 \
    --num_workers 4 \
    --print_every 10
