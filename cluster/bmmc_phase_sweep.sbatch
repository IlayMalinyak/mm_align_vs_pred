#!/bin/bash -l
#SBATCH --job-name=bmmc_phase_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --partition=l40s-shared
#SBATCH --qos=12h_4g
#SBATCH -t 4:00:00
#SBATCH --output=/home/ilay.kamai/athena/logs/%x_%j.out
#SBATCH --error=/home/ilay.kamai/athena/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

module load cuda/12.1.1-x86-nj6dyov
source /home/ilay.kamai/venvs/scgpt-env/bin/activate

cd /home/ilay.kamai/work/mm_align_vs_pred

python -m src.bmmc.run_phase_sweep
