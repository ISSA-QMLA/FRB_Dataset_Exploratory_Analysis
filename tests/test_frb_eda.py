"""Tests for frb_eda. Run with `python tests/test_frb_eda.py` (or pytest).

Tests that need the 2 GB filterbank skip themselves when it is absent.
`REFERENCE_DIR` can point at a copy of earlier serial outputs (nb03/nb04 .npz and
.csv files) so the parallel passes are checked against them value for value.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frb_eda import axes, config, parallel, physics, provenance, raw, sigproc  # noqa: E402

HAVE_FIL = config.FIL_PATH.exists()
REF = Path(os.environ["REFERENCE_DIR"]) if os.environ.get("REFERENCE_DIR") else None
NB01 = config.OUT_ROOT / "nb01"


class Skip(Exception):
    pass


def need_fil():
    if not HAVE_FIL:
        raise Skip("filterbank not present")


def master():
    m = pd.read_csv(NB01 / "candidates_labelled.csv")
    return [r._asdict() for r in m[["h5_name", "tcand", "dm_csv", "boxcar_samples",
                                    "is_injection"]].itertuples(index=False)]


# --- pure functions ---------------------------------------------------------------------

def test_dedisperse_synthetic_delta():
    """A delta laid along a DM-250 sweep must land all its power in one sample."""
    freqs = 415.85445601851853 - 0.014467592592592591 * np.arange(1024)
    tsamp, dm, t0, npad = 138.24e-6, 250.0, 5000, 200
    synth = np.full((20000, 1024), np.uint16(physics.PEDESTAL), np.uint16)
    synth[t0 + physics.delay_samples(dm, freqs, tsamp), np.arange(1024)] = 40000
    prof = physics.dedisperse(synth, t0 - npad, 2 * npad, dm, freqs, tsamp).sum(1)
    assert prof.argmax() == npad
    assert prof.sum() == prof.max()
    gap = physics.delay_samples(dm, freqs, tsamp, "top") - physics.delay_samples(dm, freqs, tsamp, "bottom")
    assert np.ptp(gap) <= 1


def test_boxcar_lag():
    p = np.zeros(200)
    p[100:108] = 1.0
    sm, lag = physics.boxcar(p, 8)
    assert sm.argmax() + lag == 103.5


def test_axis_model_basics():
    assert list(axes.time_decimation([1, 2, 3, 8, 1024])) == [1, 1, 1, 4, 512]
    t = axes.h5_time_axis_s(64, 138.24e-6)
    assert np.isclose(t[1] - t[0], 32 * 138.24e-6) and np.isclose(t.mean(), 0.0)
    # the log for the DM 47.47, boxcar 32 injection: nstart_read 70282, nsamp_read 4096
    f_hi, f_lo = 415.85445601851853, 415.85445601851853 - 0.014467592592592591 * 1023
    assert axes.chunk_geometry(9.99904, 47.4744, 32, 138.24e-6, f_hi, f_lo) == (70282, 4096)
    assert axes.chunk_geometry(30.005, 63.061, 16, 138.24e-6, f_hi, f_lo) == (216026, 2048)


def test_chunk_geometry_matches_every_log_line():
    """All 102 get_chunk lines in the candmaker log, not just the two above. The band
    edge instead of the last channel centre must break it (it shifts the sweep)."""
    log = next(config.CAND_DIR.glob("your_candmaker_*.log"))
    got = {(g[0], g[1]) for g in axes.parse_chunk_log(log)}
    rows = master()
    assert len(got) == len(rows)
    f_hi = 415.85445601851853
    f_lo = f_hi - 0.014467592592592591 * 1023
    pred = {axes.chunk_geometry(r["tcand"], r["dm_csv"], int(r["boxcar_samples"]), 138.24e-6, f_hi, f_lo)
            for r in rows}
    assert pred == got
    edge = f_hi - 0.014467592592592591 * 1024
    bad = {axes.chunk_geometry(r["tcand"], r["dm_csv"], int(r["boxcar_samples"]), 138.24e-6, f_hi, edge)
           for r in rows}
    assert bad != got


def test_exact_time_axis_steps_by_the_decimation():
    f_hi = 415.85445601851853
    f_lo = f_hi - 0.014467592592592591 * 1023
    for w in (1, 2, 16, 1024):
        t = axes.h5_time_axis_exact(30.005, 63.061, w, 138.24e-6, f_hi, f_lo)
        assert np.allclose(np.diff(t), axes.time_decimation(w) * 138.24e-6)
        # within a few raw samples (or half a column) of the idealised centred grid
        ideal = axes.h5_time_axis_s(w, 138.24e-6)
        assert abs(t[0] - ideal[0]) <= max(3, int(axes.time_decimation(w))) * 138.24e-6


def test_chunks_cover_range():
    c = parallel.chunks(989, 7)
    assert c[0][0] == 0 and c[-1][1] == 989
    assert all(a[1] == b[0] for a, b in zip(c[:-1], c[1:]))


def test_pmap_order_and_equality():
    xs = list(range(40))
    assert parallel.pmap(np.square, xs, n_jobs=1) == parallel.pmap(np.square, xs, n_jobs=4)


def test_n_jobs_follows_slurm():
    """--ntasks=16 sets only SLURM_NTASKS; that must win over os.cpu_count()."""
    keys = ("FRB_N_JOBS", "SLURM_CPUS_PER_TASK", "SLURM_NTASKS")
    saved = {k: os.environ.pop(k, None) for k in keys}
    try:
        os.environ["SLURM_NTASKS"] = "16"
        assert config._n_jobs() == 16
        os.environ["SLURM_CPUS_PER_TASK"] = "8"
        assert config._n_jobs() == 8
        os.environ["FRB_N_JOBS"] = "3"
        assert config._n_jobs() == 3
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


def test_labels_reproduce_nb01_and_fail_loudly():
    m = pd.read_csv(NB01 / "candidates_labelled.csv")
    lab, _ = provenance.label_candidates(m.tcand, m.dm_csv, config.INJ_CSV)
    assert (lab.is_injection.to_numpy() == m.is_injection.to_numpy()).all()
    try:
        provenance.label_candidates(m.tcand, m.dm_csv, config.DATA_DIR / "does_not_exist.csv")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing injection file did not raise")


def test_h5_curves_parallel_matches_serial():
    paths = [str(config.CAND_DIR / r["h5_name"]) for r in master()[:20]]
    a = parallel.pmap(raw.h5_curves, paths, n_jobs=1)
    b = parallel.pmap(raw.h5_curves, paths, n_jobs=4)      # loky, as nb04 runs it
    for x, z in zip(a, b):
        for u, v in zip(x, z):
            assert np.array_equal(u, v)


# --- raw-file passes --------------------------------------------------------------------

def test_block_stats_parallel_matches_serial():
    need_fil()
    fb = sigproc.open_filterbank(config.FIL_PATH)
    spans = parallel.chunks(40, 4)
    one = raw.combine_block_stats(parallel.pmap(raw.block_stats, spans, n_jobs=1, fb=fb, blk=1024))
    many = raw.combine_block_stats(parallel.pmap(raw.block_stats, spans, n_jobs=4, fb=fb, blk=1024))
    whole = raw.block_stats((0, 40), fb, 1024)
    for k in one:
        assert np.allclose(one[k], many[k], rtol=0, atol=1e-6), k
        assert np.allclose(one[k], whole[k], rtol=1e-12), k


def test_h5_time_axis_is_decimate_then_crop():
    """The h5 profile must match the raw data under candmaker's model, not nb02's."""
    need_fil()
    fb = sigproc.open_filterbank(config.FIL_PATH)
    rows = [r for r in master() if r["is_injection"]]
    res = parallel.pmap(raw.h5_reconstruction, rows, fb=fb, cand_dir=str(config.CAND_DIR))
    r_cm = np.array([x["r_time_candmaker"] for x in res])
    r_old = np.array([x["r_time_nb02"] for x in res])
    assert r_cm.min() > 0.9, r_cm
    assert (r_cm > r_old).all()


def test_raw_pass_parallel_matches_serial_and_reference():
    need_fil()
    fb = sigproc.open_filterbank(config.FIL_PATH)
    rows = master()
    pick = [rows[i] for i in (0, 5, 43, 60, 101)]      # includes a front-padded window
    a = parallel.pmap(raw.raw_pass_one, pick, n_jobs=1, fb=fb)
    b = parallel.pmap(raw.raw_pass_one, pick, n_jobs=4, fb=fb)
    for x, z in zip(a, b):
        for k in x:
            assert np.array_equal(np.asarray(x[k]), np.asarray(z[k]), equal_nan=True), k
    if REF is not None and (REF / "nb04" / "raw_pass.npz").exists():
        old = np.load(REF / "nb04" / "raw_pass.npz")
        for i, res in zip((0, 5, 43, 60, 101), a):
            for k in ("best_w", "peak_off_samp", "fwhm_samp"):
                assert np.isclose(old[k][i], res[k]), (k, i, old[k][i], res[k])
            assert np.allclose(old["spec_on"][i], res["spec_on"], atol=1e-3)


def test_pipeline_reproduces_nb04_feature_matrix():
    """The unattended corpus pipeline must give nb04's features and nb01's recovery table."""
    need_fil()
    from frb_eda import pipeline
    obs = pipeline.Observation("CUT_P_110", config.CAND_DIR, config.FIL_PATH, config.INJ_CSV)
    feats, inj, summ = pipeline.run_observation(obs)
    ref = pd.read_csv(config.OUT_ROOT / "nb04" / "feature_matrix.csv")
    assert list(ref.h5_name) == list(feats.h5_name)
    assert (ref.is_injection.to_numpy() == feats.is_injection.to_numpy()).all()
    for c in pipeline.FEATURES:     # 1e-6: spec_frac_bursty went through float32 and a csv
        assert np.allclose(feats[c].to_numpy(float), ref[c].to_numpy(float), rtol=1e-6, atol=1e-9), c
    mref = pd.read_csv(NB01 / "injection_recovery_match.csv")
    for c in ("TOA", "recovered", "snr_heimdall", "fetch_prob", "fetch_label"):
        assert np.allclose(inj[c].to_numpy(float), mref[c].to_numpy(float), equal_nan=True), c
    assert (summ["n_injections_recovered"], summ["fetch_fn"], summ["n_bursty_channels"]) == (13, 1, 8)
    assert summ["chunk_geometry_exact"] == summ["n_candidates"] == 102


def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        t0 = time.perf_counter()
        try:
            fn()
            print(f"PASS  {name}  ({time.perf_counter() - t0:.1f} s)")
        except Skip as e:
            print(f"SKIP  {name}  ({e})")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed (n_jobs={config.N_JOBS})")
    return failed


if __name__ == "__main__":
    sys.exit(main())
