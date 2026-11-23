"""Environment capture, input hashes and ground-truth labelling."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

PACKAGES = ("numpy", "pandas", "scipy", "matplotlib", "scikit-learn", "h5py",
            "joblib", "threadpoolctl", "pennylane")


def environment() -> dict:
    """Python, platform, package versions and the run configuration."""
    vers = {}
    for p in PACKAGES:
        try:
            vers[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            vers[p] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "packages": vers, "config": config.describe()}


def sha256(path, chunk=1 << 24) -> str:
    """SHA-256 of a file, read in 16 MB pieces so a 2 GB file never sits in memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def write_environment(out_dir: Path, hashes: dict | None = None) -> dict:
    env = environment()
    if hashes:
        env["sha256"] = hashes
    (Path(out_dir) / "environment.json").write_text(json.dumps(env, indent=2))
    return env


def label_candidates(tcand, dm, inj_csv, tol_t_s=0.5, tol_dm_frac=0.10, sentinel=-999.0):
    """Ground truth from a TOA x DM cross-match against the injection list.

    A candidate is an injection when some injection lies within `tol_t_s` in time
    AND within `tol_dm_frac` fractional DM. Every candidate inside the window is
    positive, so a double detection gives two positives.

    Raises FileNotFoundError when the observation has no injection file: there is
    then no ground truth, and nothing downstream should pretend otherwise.
    """
    inj_csv = Path(inj_csv)
    if not inj_csv.exists():
        raise FileNotFoundError(
            f"no injection parameters at {inj_csv}. This observation has no ground truth; "
            f"set FRB_INJ_CSV, or treat its candidates as unlabelled.")
    inj = pd.read_csv(inj_csv).replace(sentinel, np.nan).sort_values("TOA").reset_index(drop=True)
    toa, idm = inj["TOA"].to_numpy(), inj["DM"].to_numpy()
    tcand, dm = np.asarray(tcand, float), np.asarray(dm, float)
    dt = np.abs(toa[None, :] - tcand[:, None])
    dfr = np.abs(idm[None, :] - dm[:, None]) / idm[None, :]
    ok = (dt <= tol_t_s) & (dfr <= tol_dm_frac)
    is_inj = ok.any(axis=1)
    j = np.where(is_inj, np.argmin(np.where(ok, dt, np.inf), axis=1), np.argmin(dt, axis=1))
    rows = np.arange(len(tcand))
    return pd.DataFrame({"is_injection": is_inj,
                         "matched_TOA": np.where(is_inj, toa[j], np.nan),
                         "dt_to_nearest_inj_s": toa[j] - tcand,
                         "dm_frac_err_nearest": dfr[rows, j]}), inj
