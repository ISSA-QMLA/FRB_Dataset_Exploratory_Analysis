"""Run frb_eda.pipeline over many observations, one at a time, resumably.

For each observation: copy its .fil, injection csv and candidate folder from SRC to
STAGE (greed's own disk), run the pipeline there with all workers, write
    OUT/<obs>.features.csv     one row per candidate
    OUT/<obs>.injections.csv   one row per injected burst
    OUT/<obs>.summary.json     counts, FETCH confusion, RFI, checks, timings, status
then delete the copy. An observation whose summary says "ok" is skipped next time,
so a stopped run picks up where it left off. A failure is recorded in its summary
(status "error" plus the traceback) and the run moves on to the next one.

    python cluster/extract_features.py --src /local_storage/nc_frb --out ~/frb_features \
        --stage /local_storage/$USER/feat_$SLURM_JOB_ID  [--obs CUT_P_110 CUT_A_1 ...]
        [--task-id N --per-task K]      # slice N of the sorted observation list
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from frb_eda import config, pipeline  # noqa: E402


def _git_rev() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _done(out: Path, name: str) -> bool:
    p = out / f"{name}.summary.json"
    try:
        return json.loads(p.read_text()).get("status") == "ok"
    except (OSError, ValueError):
        return False


def release_files() -> None:
    """Drop every open memmap of the staged filterbank, here and in the workers.

    Each process caches its memmaps (sigproc._memmap). On Linux a deleted file that is
    still mapped keeps its 2 GB on disk until the mapping goes, so without this the
    staging disk fills up a few observations at a time. Shutting the worker pool down
    is the only way to reach every worker; the next pmap starts a fresh one.
    """
    from joblib.externals.loky import get_reusable_executor
    from frb_eda import sigproc
    sigproc._memmap.cache_clear()
    get_reusable_executor().shutdown(wait=True)
    import gc
    gc.collect()


def stage(src_obs: pipeline.Observation, stage_dir: Path) -> pipeline.Observation:
    stage_dir.mkdir(parents=True, exist_ok=True)
    fil = Path(shutil.copy(src_obs.fil, stage_dir / src_obs.fil.name))
    inj = Path(shutil.copy(src_obs.inj_csv, stage_dir / src_obs.inj_csv.name))
    cands = stage_dir / "cands"
    shutil.copytree(src_obs.cand_dir, cands)
    return pipeline.Observation(src_obs.name, cands, fil, inj)


def run_one(name: str, src: Path, out: Path, stage_root: Path, rev: str) -> str:
    summary = {"obs": name, "status": "error", "pipeline_rev": rev, "n_jobs": config.N_JOBS,
               "host": os.uname().nodename if hasattr(os, "uname") else "", "started": time.strftime("%F %T")}
    sdir = stage_root / name
    t0 = time.perf_counter()
    try:
        obs = stage(pipeline.find_observation(src, name), sdir)
        summary["stage_s"] = time.perf_counter() - t0
        feats, inj, s = pipeline.run_observation(obs)
        summary.update(s)
        feats.to_csv(out / f"{name}.features.csv.tmp", index=False)
        os.replace(out / f"{name}.features.csv.tmp", out / f"{name}.features.csv")
        inj.to_csv(out / f"{name}.injections.csv.tmp", index=False)
        os.replace(out / f"{name}.injections.csv.tmp", out / f"{name}.injections.csv")
        summary["status"] = "ok"
    except Exception as e:  # noqa: BLE001 - one bad observation must not stop the corpus
        summary["error"] = f"{type(e).__name__}: {e}"
        summary["traceback"] = traceback.format_exc()
    finally:
        release_files()
        for _ in range(5):              # Windows can lag deleting a folder behind its files
            shutil.rmtree(sdir, ignore_errors=True)
            if not sdir.exists():
                break
            time.sleep(1)
        if sdir.exists():
            summary["stage_not_removed"] = str(sdir)
    summary["wall_s"] = time.perf_counter() - t0
    _write_atomic(out / f"{name}.summary.json", json.dumps(summary, indent=1, default=str))
    return summary.get("error", "") if summary["status"] != "ok" else (
        f"{summary['n_candidates']} cands, {summary['n_injections_recovered']}/{summary['n_injections']} inj recovered, "
        f"FETCH missed {summary['fetch_fn']}, {summary['n_bursty_channels']} bursty ch, "
        f"NaN {summary['n_feature_nan']}, stage {summary['stage_s']:.0f} s + run {summary['total_s']:.0f} s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stage", type=Path, required=True)
    ap.add_argument("--obs", nargs="*")
    ap.add_argument("--task-id", type=int)
    ap.add_argument("--per-task", type=int)
    a = ap.parse_args()

    names = a.obs or pipeline.list_observations(a.src)
    if a.task_id is not None and a.per_task:
        names = names[a.task_id * a.per_task:(a.task_id + 1) * a.per_task]
    a.out.mkdir(parents=True, exist_ok=True)
    rev = _git_rev()
    print(f"{len(names)} observations, {config.N_JOBS} workers, pipeline {rev}, out {a.out}", flush=True)

    n_err = 0
    try:
        for i, name in enumerate(names, 1):
            if _done(a.out, name):
                print(f"[{i}/{len(names)}] {name}: already done, skipped", flush=True)
                continue
            t0 = time.perf_counter()
            msg = run_one(name, a.src, a.out, a.stage, rev)
            ok = _done(a.out, name)
            n_err += not ok
            print(f"[{i}/{len(names)}] {name}: {'ok' if ok else 'ERROR'} in {time.perf_counter() - t0:.0f} s  {msg}",
                  flush=True)
    finally:
        shutil.rmtree(a.stage, ignore_errors=True)
    print(f"done: {len(names) - n_err} ok or skipped, {n_err} errors")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
