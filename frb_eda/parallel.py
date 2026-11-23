"""One parallel map for the whole project.

The heavy loops here are per-candidate or per-block work over a 2 GB memmap,
written in numpy with a Python loop over channels. Threads would mostly wait on
the GIL in that channel loop, so the work goes to separate processes (joblib's
loky backend). Each worker is capped at one BLAS thread, otherwise 8 workers x
8 BLAS threads oversubscribe the node.

Results come back in input order whatever the worker count, so a run with
`n_jobs=1` and a run with `n_jobs=32` produce the same arrays. The tests check
that.

Worker functions must live in a module (not a notebook cell) and must open the
memmap themselves rather than receive it: a pickled memmap is sent as a full
in-memory copy.
"""
from __future__ import annotations

import warnings
from typing import Callable, Iterable

from joblib import Parallel, delayed, parallel_config

from . import config


def pmap(func: Callable, items: Iterable, n_jobs: int | None = None,
         backend: str = "loky", **kwargs) -> list:
    """`[func(x, **kwargs) for x in items]`, spread over `n_jobs` workers.

    `n_jobs=None` uses `config.N_JOBS`. `n_jobs=1` runs serially in this
    process, which is the reference the parallel path is tested against.
    `backend="threading"` only helps work that releases the GIL and does not go
    through h5py: h5py holds a global lock around every HDF5 call, so threaded
    h5 reads run one at a time. Read h5 files with the default loky backend.
    """
    items = list(items)
    n = config.N_JOBS if n_jobs is None else max(1, int(n_jobs))
    n = min(n, max(1, len(items)))
    if n == 1:
        return [func(x, **kwargs) for x in items]
    with parallel_config(backend=backend, inner_max_num_threads=1 if backend == "loky" else None), \
            warnings.catch_warnings():
        # loky reuses its worker pool between calls and warns when an idle worker timed out
        # in the meantime; the jobs are resubmitted, so it is noise, not a lost result
        warnings.filterwarnings("ignore", message="A worker stopped while some jobs were given")
        # max_nbytes=None: no automatic memmapping of arguments. Workers get small arguments
        # and open the filterbank themselves, and the temp-folder cleanup is noisy on Windows.
        return Parallel(n_jobs=n, max_nbytes=None)(delayed(func)(x, **kwargs) for x in items)


def chunks(n_total: int, n_parts: int) -> list[tuple[int, int]]:
    """Split range(n_total) into about n_parts contiguous (start, stop) pieces."""
    n_parts = max(1, min(n_parts, n_total))
    edges = [round(i * n_total / n_parts) for i in range(n_parts + 1)]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]
