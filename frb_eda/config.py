"""Paths, worker count and run switches for the pilot EDA.

Every notebook imports this instead of hardcoding paths. Each setting can be
overridden with an environment variable, so the same notebooks run on a laptop
and on a cluster node without edits:

| variable            | default                                   | meaning                              |
|---------------------|-------------------------------------------|--------------------------------------|
| FRB_ROOT            | the repository root (parent of frb_eda/)  | project root                         |
| FRB_CAND_DIR        | the single `*_processed_*` dir under root | candmaker output for one observation |
| FRB_FIL             | `Data/<basename>.fil`                     | the raw filterbank                   |
| FRB_INJ_CSV         | `Data/*_inj_params.csv`                   | injection ground truth               |
| FRB_OUT             | `frb_analysis_out/`                       | where notebook outputs go            |
| FRB_N_JOBS          | SLURM_CPUS_PER_TASK, else SLURM_NTASKS,   | worker processes                     |
|                     | else os.cpu_count()                       |                                      |
| FRB_CACHE_WINDOWS   | 0                                         | 1 = write nb03's raw windows to disk |
| FRB_SEED            | 0                                         | seed for every random draw           |
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("FRB_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = ROOT / "Data"


def _single(parent: Path, pattern: str, what: str) -> Path:
    hits = sorted(p for p in parent.glob(pattern))
    if len(hits) != 1:
        raise FileNotFoundError(
            f"{what}: expected exactly one match for {pattern!r} in {parent}, "
            f"found {len(hits)}: {[h.name for h in hits]}. Set the environment variable to choose one.")
    return hits[0]


def _env_path(var: str, fallback):
    v = os.environ.get(var)
    return Path(v) if v else fallback()


CAND_DIR = _env_path("FRB_CAND_DIR", lambda: _single(ROOT, "*_processed_*", "FRB_CAND_DIR"))
OUT_ROOT = _env_path("FRB_OUT", lambda: ROOT / "frb_analysis_out")


def _default_fil() -> Path:
    stem = CAND_DIR.name.split("_processed")[0] + "_processed"
    return DATA_DIR / f"{stem}.fil"


FIL_PATH = _env_path("FRB_FIL", _default_fil)


def _default_inj() -> Path:
    hits = sorted(DATA_DIR.glob("*_inj_params.csv"))
    # a missing file is allowed here; labelling raises a clear error when it needs one
    return hits[0] if len(hits) == 1 else DATA_DIR / "MISSING_inj_params.csv"


INJ_CSV = _env_path("FRB_INJ_CSV", _default_inj)


def _n_jobs() -> int:
    # SLURM_NTASKS covers jobs submitted with --ntasks=N (the cluster's house style),
    # where Slurm leaves SLURM_CPUS_PER_TASK unset; without it os.cpu_count() would
    # report the whole node (64 on greed) and oversubscribe the allocation
    for var in ("FRB_N_JOBS", "SLURM_CPUS_PER_TASK", "SLURM_NTASKS"):
        v = os.environ.get(var)
        if v:
            return max(1, int(v))
    return max(1, os.cpu_count() or 1)


N_JOBS = _n_jobs()
CACHE_WINDOWS = os.environ.get("FRB_CACHE_WINDOWS", "0") == "1"
SEED = int(os.environ.get("FRB_SEED", "0"))


def out_dir(name: str) -> Path:
    """Output folder for one notebook, created on first use."""
    d = OUT_ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def describe() -> dict:
    return {"ROOT": str(ROOT), "CAND_DIR": str(CAND_DIR), "FIL_PATH": str(FIL_PATH),
            "INJ_CSV": str(INJ_CSV), "OUT_ROOT": str(OUT_ROOT), "N_JOBS": N_JOBS,
            "CACHE_WINDOWS": CACHE_WINDOWS, "SEED": SEED}
