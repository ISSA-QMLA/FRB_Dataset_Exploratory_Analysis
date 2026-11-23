#!/bin/bash
# Pilot: prove the whole cluster path end to end on one observation.
#   venv + packages on 3.10 -> sshfs mount -> stage to greed's disk -> 16 workers -> clean up
# Submit from the head node:  cd ~/frb_eda_repo && sbatch cluster/pilot_tests.sbatch
#
#SBATCH --job-name=frb_pilot
#SBATCH --output=frb_pilot_%j.txt
#SBATCH --nodelist=greed
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks=16
#SBATCH --time=00:30:00

set -uo pipefail

REPO="$HOME/frb_eda_repo"
OBS=CUT_P_110                                    # the observation the laptop EDA used
SRC=/local_storage/nc_frb                        # sshfs mount of betrayal's data. READ ONLY: never write here
STAGE="/local_storage/$USER/pilot_$SLURM_JOB_ID" # greed's own disk
sec() { printf '\n===== %s  [%s] =====\n' "$1" "$(date +%T)"; }
trap 'rm -rf "$STAGE"; echo "cleaned up $STAGE"' EXIT

sec "job"
echo "host $(hostname)  job $SLURM_JOB_ID  ntasks $SLURM_NTASKS"
echo "repo $(git -C "$REPO" rev-parse --short HEAD) ($(git -C "$REPO" rev-parse --abbrev-ref HEAD))"

sec "data mount"
if ! mountpoint -q "$SRC"; then
    echo "$SRC is not mounted - trying to remount it from betrayal"
    sshfs "$USER@betrayal:/local_storage/nc_frb/updated_simualted_frbs" "$SRC" \
        -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 \
        || { echo "remount failed - ask the cluster admin to remount the data on greed"; exit 1; }
fi
mountpoint "$SRC"

sec "stage $OBS to greed's disk"
CAND_SRC=$(ls -d "$SRC/${OBS}_inj_13_processed_"*/ 2>/dev/null | head -1)
FIL_SRC="$SRC/${OBS}_inj_13_processed.fil"
INJ_SRC="$SRC/${OBS}_inj_params.csv"
for f in "$FIL_SRC" "$INJ_SRC" "$CAND_SRC"; do
    [ -e "$f" ] || { echo "missing: $f"; exit 1; }
done
mkdir -p "$STAGE"
t0=$SECONDS
cp "$FIL_SRC" "$INJ_SRC" "$STAGE/"
cp -r "$CAND_SRC" "$STAGE/cands"
echo "copied in $((SECONDS - t0)) s:"
du -sh "$STAGE"/*
echo "fil bytes $(stat -c %s "$STAGE/${OBS}_inj_13_processed.fil") (laptop copy: 2074550599)"
echo "h5 files  $(ls "$STAGE/cands"/*.h5 | wc -l) (laptop copy: 102)"

sec "environment"
source "$HOME/frb_venv/bin/activate"
export FRB_FIL="$STAGE/${OBS}_inj_13_processed.fil"
export FRB_INJ_CSV="$STAGE/${OBS}_inj_params.csv"
export FRB_CAND_DIR="$STAGE/cands"
python -c "
import sys, numpy, scipy, pandas, h5py, joblib
print('python', sys.version.split()[0])
print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'pandas', pandas.__version__,
      'h5py', h5py.__version__, 'joblib', joblib.__version__)
sys.path.insert(0, '$REPO')
from frb_eda import config
for k, v in config.describe().items(): print(f'{k:14s}', v)
"

sec "tests"
cd "$REPO"
python tests/test_frb_eda.py
rc=$?

sec "done"
echo "test failures: $rc"
exit $rc
