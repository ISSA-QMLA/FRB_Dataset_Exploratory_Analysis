#!/bin/bash
#SBATCH --job-name=probe_greed
#SBATCH --nodelist=greed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=probe_greed_%j.txt

# Read-only: answers "what does a job on greed actually see?"
sec() { printf '\n===== %s =====\n' "$1"; }

sec "node";            hostname; nproc; free -g | head -2
sec "home shared?";    df -hT "$HOME"; ls "$HOME" | head -20
# On greed, betrayal's updated_simualted_frbs/ is sshfs-mounted straight at /local_storage/nc_frb
D=/local_storage/nc_frb
sec "node size";       scontrol show node greed 2>&1 | grep -E "CPUTot|CPUAlloc|RealMemory|AllocMem|State|Partitions"
sec "data mount";      mountpoint "$D" 2>&1; df -hT "$D" 2>&1
                       ls -la "$D" 2>&1 | head -40
                       echo "top-level entries: $(ls "$D" 2>/dev/null | wc -l)"
sec "fil files";       find "$D" -maxdepth 3 -name '*.fil' -printf '%s %p\n' 2>/dev/null | sort -n | awk '{n++; s+=$1; print} END {printf "count=%d total=%.1f GB\n", n, s/1e9}' | tail -15
sec "h5 / csv";        echo "h5 (depth<=3): $(find "$D" -maxdepth 3 -name '*.h5' 2>/dev/null | wc -l)"
                       find "$D" -maxdepth 2 -name '*inj_params.csv' 2>/dev/null | head -5
sec "read speed over sshfs (500 MB)"
                       f=$(find "$D" -maxdepth 3 -name '*.fil' -size +600M 2>/dev/null | head -1); echo "$f"
                       [ -n "$f" ] && dd if="$f" of=/dev/null bs=4M count=125 2>&1 | tail -1
sec "local disk";      df -hT /local_storage /tmp 2>&1
sec "ssh keys";        ls -la "$HOME/.ssh" 2>&1
sec "mounts";          df -hT -x tmpfs -x devtmpfs -x overlay -x squashfs 2>&1
                       mount | grep -iE "betrayal|nfs|fuse|sshfs|lustre|beegfs|gpfs" 2>&1
                       ls /mnt /media /net 2>&1
                       grep -v '^#' /etc/fstab 2>&1 | grep -v '^$'
sec "scratch?";        ls -ld /scratch /tmp "$TMPDIR" 2>&1; df -h /tmp 2>&1 | tail -1
sec "python";          which python3; python3 --version
                       ls -d "$HOME/frb_venv" 2>&1 && "$HOME/frb_venv/bin/python" --version
sec "slurm env";       env | grep -E "^SLURM_(JOB_ID|JOB_PARTITION|CPUS_PER_TASK|NODELIST|SUBMIT_DIR)"
