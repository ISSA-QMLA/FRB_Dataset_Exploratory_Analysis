"""How fast is one core here, and what happens when every worker is busy?

Two synthetic workloads (no input files needed), each run serially and then on
N_JOBS workers at once:
  cache    - small sorts that fit in a core's own cache: pure core speed
  memory   - three dedispersions of a 60 MB block, the shape of nb03's per-candidate
             work: limited by memory bandwidth once many run together
Run it on the laptop and on greed and compare. Same `cache serial` on both means
the cores match. A `memory` parallel/serial ratio far above the `cache` one means
the workers are queueing for memory bandwidth rather than for cores.

    python cluster/bench_core.py            # N_JOBS from frb_eda.config
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # before numpy: one thread, like a pmap worker

import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from frb_eda import config, parallel, physics  # noqa: E402

FREQS = 415.85445601851853 - 0.014467592592592591 * np.arange(1024)   # the pilot's band
TSAMP = 138.24e-6
_BLOCK = None


def memory_task(_: int) -> float:
    """Three dedispersions of a 60 MB uint16 block (made once per process)."""
    global _BLOCK
    if _BLOCK is None:
        _BLOCK = np.random.default_rng(0).integers(32000, 33500, size=(30000, 1024), dtype=np.uint16)
    t0 = time.perf_counter()
    for dm in (100.0, 500.0, 1500.0, 250.0, 800.0, 2000.0):
        physics.dedisperse(_BLOCK, 2000, 4000, dm, FREQS, TSAMP)
    return time.perf_counter() - t0


def cache_task(seed: int) -> float:
    """3000 sorts of a 16k float64 array (128 kB, stays in cache)."""
    x = np.random.default_rng(seed).random(16384)
    t0 = time.perf_counter()
    for _ in range(3000):
        np.sort(x)
    return time.perf_counter() - t0


def run(task, n: int) -> None:
    task(0)                                                         # warm up this process
    serial = statistics.median(task(i) for i in range(1, 6))
    parallel.pmap(task, range(2 * n), n_jobs=n)                     # start + warm the workers
    par = parallel.pmap(task, range(100, 100 + 3 * n), n_jobs=n)
    p = statistics.median(par)
    # speed-up from the tasks' own clocks: n workers each finishing a task every p seconds,
    # against one core finishing one every `serial` seconds (dispatch overhead left out)
    print(f"{task.__name__:12s} serial {serial:6.3f} s   parallel {p:6.3f} s/task   "
          f"slow-down {p / serial:5.2f}x   speed-up on {n} workers {n * serial / p:5.1f}x")


def main() -> None:
    n = config.N_JOBS
    print(f"host {platform.node()}  python {platform.python_version()}  numpy {np.__version__}  N_JOBS {n}")
    run(cache_task, n)
    run(memory_task, n)


if __name__ == "__main__":
    main()
