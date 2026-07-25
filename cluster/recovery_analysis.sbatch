#!/bin/bash -l
#SBATCH --job-name=recovery_analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --partition=l40s-shared
#SBATCH --qos=12h_4g
#SBATCH -t 04:00:00
#SBATCH --output=/home/ilay.kamai/athena/logs/%x_%j.out
#SBATCH --error=/home/ilay.kamai/athena/logs/%x_%j.err

# Proposition 3.2 recovery count r̂ + ε×n sensitivity sweep.
# CPU-only. Needs: numpy scipy scikit-learn pandas anndata scanpy scikit-misc astropy.
# Reads astro features from MULTIDESA_ANALYSIS/../logs and BMMC h5ad + scGPT npy.

set -euo pipefail
source /home/ilay.kamai/venvs/scgpt-env/bin/activate 2>/dev/null || true
cd /home/ilay.kamai/work/mm_align_vs_pred
export PYTHONUNBUFFERED=1
export MULTIDESA_ANALYSIS=/home/ilay.kamai/work/MultiDESA/analysis
export MULTIMODAL_DF=/home/ilay.kamai/work/multimodal/multimodal_df_min2.parquet
export BMMC_RNA_EMB=/home/ilay.kamai/work/citeseq/bmmc/rna_embeddings_scgpt.npy
export BMMC_H5AD=/home/ilay.kamai/work/citeseq/bmmc/neurips2021_cite_BMMC.h5ad

python -m src.recovery_analysis --datasets kepler tess bmmc
