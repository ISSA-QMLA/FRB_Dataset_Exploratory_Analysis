#!/bin/bash
# Feature extraction over the corpus with frb_eda.pipeline (cluster/extract_features.py).
# Resumable: observations already done are skipped, so re-submitting is always safe.
#
# A few named observations (the test run):
#   cd ~/frb_eda_repo && sbatch --export=ALL,OBS="CUT_P_110 CUT_A_110 CUT_A_135 CUT_Q_62 CUT_CC_268" cluster/features.sbatch
# The whole corpus, 10 observations per array task, one task at a time (%1):
#   cd ~/frb_eda_repo && sbatch --array=0-94%1 --export=ALL,PER_TASK=10 cluster/features.sbatch
# Pool what is finished at any point (on the head node, takes seconds):
#   source ~/frb_venv/bin/activate && python ~/frb_eda_repo/cluster/pool_features.py ~/frb_features
#
#SBATCH --job-name=frb_feat
#SBATCH --output=frb_feat_%A_%a.txt
#SBATCH --nodelist=greed
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks=16
#SBATCH --time=04:00:00

set -uo pipefail

REPO="$HOME/frb_eda_repo"
SRC=/local_storage/nc_frb                                  # sshfs mount of betrayal's data. READ ONLY
OUT="$HOME/frb_features"                                   # small: ~100 kB per observation
STAGE="/local_storage/$USER/feat_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
trap 'rm -rf "$STAGE"' EXIT

echo "host $(hostname)  job ${SLURM_JOB_ID}  array task ${SLURM_ARRAY_TASK_ID:-none}  ntasks $SLURM_NTASKS"
echo "repo $(git -C "$REPO" rev-parse --short HEAD)   started $(date +%T)"

if ! mountpoint -q "$SRC"; then
    echo "$SRC is not mounted - trying to remount it from betrayal"
    sshfs "$USER@betrayal:/local_storage/nc_frb/updated_simualted_frbs" "$SRC" \
        -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 \
        || { echo "remount failed - ask the cluster admin to remount the data on greed"; exit 1; }
fi

echo "greed local disk before: $(df -h --output=used,avail /local_storage | tail -1)"

source "$HOME/frb_venv/bin/activate"
ARGS=(--src "$SRC" --out "$OUT" --stage "$STAGE")
if [ -n "${OBS:-}" ]; then
    read -r -a OBS_ARR <<< "$OBS"
    ARGS+=(--obs "${OBS_ARR[@]}")
elif [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    ARGS+=(--task-id "$SLURM_ARRAY_TASK_ID" --per-task "${PER_TASK:-10}")
fi
python "$REPO/cluster/extract_features.py" "${ARGS[@]}"
rc=$?
# staged copies must be gone AND their space freed (a deleted file stays allocated while mapped)
echo "greed local disk after:  $(df -h --output=used,avail /local_storage | tail -1)"
echo "left in /local_storage/$USER: $(ls -A "/local_storage/$USER" 2>/dev/null | grep -v "^feat_${SLURM_JOB_ID}" | tr '\n' ' ')"
echo "finished $(date +%T), exit $rc"
exit $rc
