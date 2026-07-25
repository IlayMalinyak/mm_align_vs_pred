#!/bin/bash -l
#SBATCH --job-name=kepler_restore891
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --partition=rtx6k-shared
#SBATCH --qos=12h_4g
#SBATCH -t 04:00:00
#SBATCH --output=/home/ilay.kamai/athena/logs/%x_%j.out
#SBATCH --error=/home/ilay.kamai/athena/logs/%x_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# Re-extract the Kepler TEST-split features for the LAMOST×Kepler phase analysis,
# to restore the canonical n_matched=891 (Fig-12 / Table 1).
#
# WHY THIS IS NEEDED:
#   kepler.py forces  log_dir = <checkpoint dir>  when loading a checkpoint
#   (src/kepler.py:1103-1108). On 2026-05-06/05-13 `make_kepler_catalog.py`
#   loaded the model in logs/kepler/2026-02-12-07-02 and its test-set validation
#   REGENERATED features_output_space.npy / _preds.csv / svd_features/ there,
#   using an evolved data-pool assembly → test split shifted 891→879, changing
#   the CCA spectrum (Δ̂_CA 1.128→0.626, Both→CP only).
#
# NON-DESTRUCTIVE: writes into a FRESH dir (…_restore891); the current 879 dir is
# left untouched. Checkpoint (2026-02-14) and base catalog
# (kepler_predictions_clean_seg_0_1_2_median.csv, 2024-12-22) are unchanged, and
# kepler.py's get_dfs is a DIFFERENT pool path than make_kepler_catalog.py, so a
# direct inference is expected to reproduce the Feb test split (891).
#
# SUCCESS CRITERION: after the job, verify n_matched==891 (see VERIFY below).
# If it comes back 879, kepler.py's get_dfs also drifted since Feb → fall back to
# a git checkout of the Feb-era commit (≈0cd8e34 / 91e9af3, "SVD analysis").
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

MD=/home/ilay.kamai/work/MultiDESA
ORIG=$MD/logs/kepler/2026-02-12-07-02
NEW=$MD/logs/kepler/2026-02-12-07-02_restore891
CKPT=kepler_dualformer_dualvar_no_cov_DoubleInputRegressor.pth

# 1) Fresh output dir with the (unchanged) checkpoint copied in.
mkdir -p "$NEW"
cp -n "$ORIG/$CKPT" "$NEW/$CKPT"

# 2) Restore config = kepler.yaml with the ACTIVE checkpoint dir repointed to $NEW
#    (commented alternate checkpoints use other dirs and are unaffected).
RCFG=$MD/configs/kepler_restore891.yaml
sed 's#logs/kepler/2026-02-12-07-02/#logs/kepler/2026-02-12-07-02_restore891/#g' \
    "$MD/configs/kepler.yaml" > "$RCFG"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=$((29500 + ${SLURM_JOB_ID} % 1000))
export APPTAINER_BIND="$MD,/home/ilay.kamai/work/lightPred"

srun --ntasks=1 \
  --output=/home/ilay.kamai/athena/logs/%x_%j_%N_%t.out \
  --error=/home/ilay.kamai/athena/logs/%x_%j_%N_%t.err \
  --container=/apps/apptainer/images/pytorch/latest/torch2.7-tf2.20-cu128.sif \
  bash -lc "export PYTHONUNBUFFERED=1 && torchrun \
    --nnodes=1 --nproc_per_node=1 \
    --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    $MD/src/kepler.py $RCFG"

# ── VERIFY (run on a login node / CPU after the job finishes) ────────────────
#   python3 - <<'PY'
#   import sys; sys.path.insert(0,'/rg/perets_prj/ilay.kamai/MultiDESA/analysis')
#   import pandas as pd
#   from analyze_phase_diagram import load_data, match_samples
#   LAM='/rg/perets_prj/ilay.kamai/MultiDESA/logs/lamost/2026-02-12/44644'
#   KEP='/rg/perets_prj/ilay.kamai/MultiDESA/logs/kepler/2026-02-12-07-02_restore891'
#   mm=pd.read_parquet('/rg/perets_prj/ilay.kamai/multimodal/multimodal_df_min2.parquet')
#   _,dfa=load_data(LAM,'test'); _,dfb=load_data(KEP,'test')
#   ia,ib,_=match_samples(dfa,dfb,mm); print('n_matched =', len(ia))   # want 891
#   PY
