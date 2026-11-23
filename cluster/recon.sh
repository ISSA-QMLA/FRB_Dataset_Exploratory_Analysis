#!/usr/bin/env bash
# Read-only reconnaissance of the cluster. Run on the head node (gluttony):
#   bash recon.sh 2>&1 | tee recon_$(date +%F).txt
# It submits nothing and writes nothing except its own output.

sec() { printf '\n===== %s =====\n' "$1"; }
try() { printf '$ %s\n' "$*"; timeout 20 bash -c "$*" 2>&1 | head -n "${N:-40}"; }

sec "where am I"
try 'hostname; whoami; pwd; echo $HOME; id'
try 'cat /etc/os-release | head -3; uname -r'

sec "home directory: is it shared?"
try 'df -hT $HOME'
try 'mount | grep -E " /users| /home| /local_storage" '
try 'ls -la $HOME'

sec "slurm: partitions and nodes"
try 'sinfo'
N=80 try 'sinfo -N -o "%N %P %c %m %G %T %f"'
N=60 try 'scontrol show node greed'
N=30 try 'scontrol show node betrayal'
N=30 try 'scontrol show partition'
try 'squeue'
try 'squeue -u $USER'
try 'sacct -u $USER --starttime 2026-07-01 -o JobID,JobName,Partition,NodeList,State,Elapsed,AllocCPUS | tail -20'
try 'sacctmgr -n show assoc user=$USER format=account,partition,qos,maxjobs,maxsubmit 2>/dev/null'

sec "python / modules on the head node"
try 'which python3 python; python3 --version'
try 'command -v module && module avail 2>&1 | head -40'
try 'command -v conda mamba uv pip3'
try 'ls -d $HOME/frb_venv $HOME/*venv* 2>/dev/null; $HOME/frb_venv/bin/python --version 2>/dev/null'
try '$HOME/frb_venv/bin/pip list 2>/dev/null | grep -iE "numpy|scipy|h5py|joblib|pandas|matplotlib|your|sklearn|scikit|pennylane"'

sec "existing job scripts and outputs in home"
try 'find $HOME -maxdepth 3 \( -name "*.sh" -o -name "*.sbatch" -o -name "*.slurm" -o -name "slurm-*.out" \) -printf "%TY-%Tm-%Td %p\n" 2>/dev/null | sort | tail -30'

sec "data on betrayal (via ssh from head node)"
try 'ssh -o BatchMode=yes -o ConnectTimeout=8 betrayal "hostname; df -hT /local_storage; ls -la /local_storage; ls /local_storage/nc_frb | head; du -sh /local_storage/nc_frb/updated_simualted_frbs 2>/dev/null; ls /local_storage/nc_frb/updated_simualted_frbs | head -20; ls /local_storage/nc_frb/updated_simualted_frbs | wc -l"'

sec "can greed see the same paths? (ssh only, no job submitted)"
try 'ssh -o BatchMode=yes -o ConnectTimeout=8 greed "hostname; nproc; free -g | head -2; df -hT \$HOME /local_storage 2>&1; ls /local_storage/nc_frb 2>&1 | head; python3 --version"'

sec "load right now"
try 'ssh -o BatchMode=yes -o ConnectTimeout=8 greed "uptime; who"'
try 'ssh -o BatchMode=yes -o ConnectTimeout=8 betrayal "uptime; who"'

sec "done"
