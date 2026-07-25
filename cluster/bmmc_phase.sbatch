#!/bin/bash -l
#SBATCH --job-name=bmmc_phase
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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

python -m src.bmmc.analyze_phase_diagram_bmmc \
    --max_cells 5000 \
    --pca_components 100 \
    --n_components 100 \
    --n_permutations 30

python -m src.bmmc.plot_phase_diagnostics
