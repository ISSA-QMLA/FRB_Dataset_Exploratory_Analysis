#!/bin/bash
# Step 5: re-run all four EDA notebooks from scratch on greed and check the cluster
# reproduces the laptop. Also measures why greed was slower in the pilot.
#   stage CUT_P_110 -> core benchmark -> nb01..nb04 into an EMPTY output folder
#   (so every cache is rebuilt) -> compare with committed outputs -> timings vs laptop
# Submit from the head node:  cd ~/frb_eda_repo && git pull && sbatch cluster/nbrun.sbatch
#
#SBATCH --job-name=frb_nbrun
#SBATCH --output=frb_nbrun_%j.txt
#SBATCH --nodelist=greed
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks=16
#SBATCH --time=02:00:00

set -uo pipefail

REPO="$HOME/frb_eda_repo"
OBS=CUT_P_110
SRC=/local_storage/nc_frb                         # sshfs mount of betrayal's data. READ ONLY: never write here
STAGE="/local_storage/$USER/nbrun_$SLURM_JOB_ID"  # greed's own disk
KEEP="$HOME/frb_runs/nbrun_$SLURM_JOB_ID"         # small results kept after the job (csv/json/notebooks)
sec() { printf '\n===== %s  [%s] =====\n' "$1" "$(date +%T)"; }

MON_PID=""
cleanup() {
    [ -n "$MON_PID" ] && kill "$MON_PID" 2>/dev/null
    rm -rf "$STAGE"
    echo "cleaned up $STAGE"
}
trap cleanup EXIT

sec "job"
echo "host $(hostname)  job $SLURM_JOB_ID  ntasks $SLURM_NTASKS"
echo "repo $(git -C "$REPO" rev-parse --short HEAD) ($(git -C "$REPO" rev-parse --abbrev-ref HEAD))"
lscpu | grep -E "Model name|^CPU\(s\)|Thread|Socket|MHz|NUMA node\(s\)"
scontrol show node greed | grep -oE "CPUAlloc=[0-9]+|CPULoad=[0-9.]+"

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
mkdir -p "$STAGE/out" "$STAGE/executed"
t0=$SECONDS
cp "$FIL_SRC" "$INJ_SRC" "$STAGE/"
cp -r "$CAND_SRC" "$STAGE/cands"
echo "copied in $((SECONDS - t0)) s; fil bytes $(stat -c %s "$STAGE/${OBS}_inj_13_processed.fil")," \
     "h5 files $(ls "$STAGE/cands"/*.h5 | wc -l)"

sec "environment"
source "$HOME/frb_venv/bin/activate"
export FRB_FIL="$STAGE/${OBS}_inj_13_processed.fil"
export FRB_INJ_CSV="$STAGE/${OBS}_inj_params.csv"
export FRB_CAND_DIR="$STAGE/cands"
export FRB_OUT="$STAGE/out"                      # empty: nb03/nb04 rebuild every cache
export MPLBACKEND=Agg
python -c "import sys; sys.path.insert(0, '$REPO'); from frb_eda import config; [print(f'{k:14s}', v) for k, v in config.describe().items()]"
jupyter kernelspec list

# load monitor: every 20 s, the node's load and how much CPU is ours vs everyone else's
(
    while true; do
        top -b -n 2 -d 1 -w 512 | awk -v me="$USER" -v t="$(date +%T)" -v la="$(cut -d' ' -f1 /proc/loadavg)" '
            /^top -/ {n++} n == 2 && $1 ~ /^[0-9]+$/ { if ($2 == me) m += $9; else o += $9 }
            END { printf "%s  load %5s   ours %5.0f%% CPU   others %5.0f%% CPU  (6400%% = all 64 threads)\n", t, la, m, o }'
        sleep 19
    done
) > "$STAGE/load.log" 2>&1 &
MON_PID=$!

sec "core benchmark"
python "$REPO/cluster/bench_core.py"

rc=0
for nb in 01_metadata_inventory 02_candidate_h5_anatomy 03_filterbank_exploration 04_feature_battery; do
    sec "run $nb"
    t0=$SECONDS
    jupyter nbconvert --to notebook --execute "$REPO/Data/$nb.ipynb" \
        --output-dir "$STAGE/executed" \
        --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 2>&1 | tail -3
    status=${PIPESTATUS[0]}
    echo "$nb: exit $status in $((SECONDS - t0)) s"
    if [ "$status" -ne 0 ]; then rc=1; echo "stopping: later notebooks need this one's outputs"; break; fi
done

sec "load during the run"
cat "$STAGE/load.log"

if [ "$rc" -eq 0 ]; then
    sec "compare with the committed laptop outputs"
    python "$REPO/cluster/run_report.py" compare "$STAGE/out" "$REPO/frb_analysis_out"
    rc=$?
    sec "timings vs laptop"
    python "$REPO/cluster/run_report.py" timings "$STAGE/executed" "$REPO/Data"
fi

sec "keep small results"
mkdir -p "$KEEP"
cp "$STAGE/load.log" "$KEEP/"
cp "$STAGE"/executed/*.ipynb "$KEEP/" 2>/dev/null
(cd "$STAGE/out" && find . \( -name '*.csv' -o -name '*.json' \) -exec cp --parents {} "$KEEP/" \;)
du -sh "$KEEP"

sec "done"
echo "result: $([ "$rc" -eq 0 ] && echo 'reproduced' || echo 'see DIFF / MISSING lines above')"
exit $rc
