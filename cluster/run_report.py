"""Compare a fresh notebook run against the committed outputs, and time it cell by cell.

    python cluster/run_report.py compare  NEW_OUT  REF_OUT
    python cluster/run_report.py timings  NEW_EXECUTED_DIR  REF_NOTEBOOK_DIR

compare: every .csv, .json and .npz under REF_OUT is matched by relative path in NEW_OUT.
  Numbers are compared with a tolerance, strings exactly except file paths (basename
  only, since the two runs read from different folders). JSON keys that describe the
  run rather than the data (versions, paths, timings, worker counts) are listed apart.
timings: per-notebook wall time and the slowest cells, from the execution timestamps
  that Jupyter/nbclient write into each code cell's metadata, side by side for both runs.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RTOL, ATOL = 1e-6, 1e-9
RUN_KEYS = re.compile(r"version|path|dir|root|python|platform|host|n_jobs|workers|elapsed|seconds|"
                      r"time|date|config|packages|cache", re.I)


def _norm_str(v):
    s = str(v)
    return s.replace("\\", "/").rsplit("/", 1)[-1] if ("/" in s or "\\" in s) else s


def _cmp_arrays(a, b) -> str | None:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return f"shape {a.shape} vs {b.shape}"
    if a.dtype.kind in "fiucb" and b.dtype.kind in "fiucb":
        a, b = a.astype(np.float64), b.astype(np.float64)
        if np.array_equal(a, b, equal_nan=True):
            return None
        both = np.isfinite(a) & np.isfinite(b)
        if (np.isnan(a) != np.isnan(b)).any():
            return f"NaN pattern differs ({int((np.isnan(a) != np.isnan(b)).sum())} cells)"
        d = np.abs(a[both] - b[both])
        rel = d / np.maximum(np.abs(b[both]), ATOL)
        if np.allclose(a[both], b[both], rtol=RTOL, atol=ATOL):
            return f"~close (max rel {rel.max():.1e})"
        return f"DIFFERS max abs {d.max():.3g}, max rel {rel.max():.3g}, {int((~np.isclose(a[both], b[both], rtol=RTOL, atol=ATOL)).sum())} cells"
    av = np.vectorize(_norm_str)(a) if a.size else a
    bv = np.vectorize(_norm_str)(b) if b.size else b
    n = int((av != bv).sum())
    return None if n == 0 else f"DIFFERS {n} text cells"


def _flatten(o, pre=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _flatten(v, f"{pre}.{k}" if pre else str(k))
    elif isinstance(o, list) and o and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in o):
        yield pre, o
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _flatten(v, f"{pre}[{i}]")
    else:
        yield pre, o


def compare(new: Path, ref: Path) -> int:
    files = sorted(p for p in ref.rglob("*") if p.suffix in (".csv", ".json", ".npz") and "windows" not in p.parts)
    n_bad = 0
    for rp in files:
        rel = rp.relative_to(ref)
        np_ = new / rel
        if not np_.exists():
            print(f"MISSING  {rel}")
            n_bad += 1
            continue
        notes, run_notes = [], []
        if rp.suffix == ".csv":
            a, b = pd.read_csv(np_), pd.read_csv(rp)
            if list(a.columns) != list(b.columns):
                notes.append(f"columns differ: +{sorted(set(a.columns) - set(b.columns))} -{sorted(set(b.columns) - set(a.columns))}")
            for c in [c for c in b.columns if c in a.columns]:
                r = _cmp_arrays(a[c].to_numpy(), b[c].to_numpy())
                if r:
                    notes.append(f"{c}: {r}")
        elif rp.suffix == ".npz":
            za, zb = np.load(np_, allow_pickle=False), np.load(rp, allow_pickle=False)
            for k in zb.files:
                r = "missing" if k not in za.files else _cmp_arrays(za[k], zb[k])
                if r:
                    notes.append(f"{k}: {r}")
        else:
            fa = dict(_flatten(json.loads(np_.read_text())))
            fb = dict(_flatten(json.loads(rp.read_text())))
            for k in sorted(set(fa) | set(fb)):
                if k not in fa or k not in fb:
                    r = "only in " + ("new" if k in fa else "ref")
                elif isinstance(fb[k], (int, float, list)) and not isinstance(fb[k], bool) \
                        and isinstance(fa[k], (int, float, list)):
                    r = _cmp_arrays(fa[k], fb[k])
                else:
                    r = None if _norm_str(fa[k]) == _norm_str(fb[k]) else f"{fa[k]!r} vs {fb[k]!r}"[:120]
                if r:
                    (run_notes if RUN_KEYS.search(k) else notes).append(f"{k}: {r}")
        real = [x for x in notes if not x.split(": ", 1)[1].startswith("~close")]
        tag = "DIFF    " if real else ("CLOSE   " if notes else "SAME    ")
        n_bad += bool(real)
        print(f"{tag} {rel}")
        for x in notes[:8]:
            print(f"           {x}")
        if len(notes) > 8:
            print(f"           ... {len(notes) - 8} more")
        for x in run_notes[:4]:
            print(f"           (run detail) {x}")
    print(f"\n{len(files)} files compared, {n_bad} differ beyond rtol={RTOL:g} or are missing")
    return n_bad


def _cell_times(nb_path: Path):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    out = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        ex = c.get("metadata", {}).get("execution", {})
        t0, t1 = ex.get("iopub.execute_input"), ex.get("shell.execute_reply")
        if not (t0 and t1):
            continue
        dt = (datetime.fromisoformat(t1.replace("Z", "+00:00")) - datetime.fromisoformat(t0.replace("Z", "+00:00"))).total_seconds()
        first = next((l.strip() for l in "".join(c["source"]).splitlines() if l.strip() and not l.strip().startswith("#")), "")
        out.append((i, dt, first[:60]))
    return out


def timings(new_dir: Path, ref_dir: Path) -> None:
    for np_ in sorted(new_dir.glob("0*.ipynb")):
        new = {i: (t, s) for i, t, s in _cell_times(np_)}
        ref = {i: (t, s) for i, t, s in _cell_times(ref_dir / np_.name)} if (ref_dir / np_.name).exists() else {}
        tn, tr = sum(t for t, _ in new.values()), sum(t for t, _ in ref.values())
        print(f"\n{np_.stem}: this run {tn:7.1f} s   laptop {tr:7.1f} s   ratio {tn / tr if tr else float('nan'):.1f}x")
        for i, (t, s) in sorted(new.items(), key=lambda kv: -kv[1][0])[:6]:
            r = ref.get(i, (float("nan"), ""))[0]
            print(f"   cell {i:3d}  {t:7.1f} s  laptop {r:7.1f} s  {t / r if r else float('nan'):5.1f}x   {s}")


if __name__ == "__main__":
    cmd, a, b = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    if cmd == "compare":
        sys.exit(min(compare(a, b), 1))
    timings(a, b)
