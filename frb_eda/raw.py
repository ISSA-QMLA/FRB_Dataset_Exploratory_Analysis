"""Worker functions for passes over the raw filterbank.

Each function takes plain picklable arguments (a `Filterbank`, ints, dicts) and
opens the memmap inside the worker. They are called through `parallel.pmap`.
Reductions (sums, minima, counts) are returned per piece and combined by the
caller in input order, so results do not depend on the worker count beyond
float64 summation order.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from . import axes
from .physics import PEDESTAL, boxcar, dedisperse, delay_samples, mad_snr
from .sigproc import Filterbank


# --- nb03 section 2: bandpass and block statistics -------------------------------------

def block_stats(span, fb: Filterbank, blk: int):
    """Per-channel sums and block summaries for blocks span[0]..span[1]-1."""
    x_all = fb.memmap()
    b0, b1 = span
    nch = fb.nchans
    ch_sum = np.zeros(nch)
    ch_sq = np.zeros(nch)
    ch_min = np.full(nch, np.iinfo(np.uint16).max, np.uint16)
    ch_max = np.zeros(nch, np.uint16)
    ch_sat = np.zeros(nch, np.int64)
    blk_mean = np.zeros((b1 - b0, nch), np.float32)
    blk_max = np.zeros((b1 - b0, nch), np.uint16)
    for k, b in enumerate(range(b0, b1)):
        raw = np.asarray(x_all[b * blk:(b + 1) * blk])
        x = raw.astype(np.float64) - PEDESTAL       # offset kills the sum(x^2) cancellation
        ch_sum += x.sum(0)
        ch_sq += (x * x).sum(0)
        np.minimum(ch_min, raw.min(0), out=ch_min)
        np.maximum(ch_max, raw.max(0), out=ch_max)
        ch_sat += (raw == 65535).sum(0)
        blk_mean[k] = x.mean(0) + PEDESTAL
        blk_max[k] = raw.max(0)
    return dict(ch_sum=ch_sum, ch_sq=ch_sq, ch_min=ch_min, ch_max=ch_max, ch_sat=ch_sat,
                blk_mean=blk_mean, blk_max=blk_max)


def combine_block_stats(parts):
    out = dict(ch_sum=sum(p["ch_sum"] for p in parts), ch_sq=sum(p["ch_sq"] for p in parts),
               ch_sat=sum(p["ch_sat"] for p in parts),
               ch_min=np.minimum.reduce([p["ch_min"] for p in parts]),
               ch_max=np.maximum.reduce([p["ch_max"] for p in parts]),
               blk_mean=np.concatenate([p["blk_mean"] for p in parts]),
               blk_max=np.concatenate([p["blk_max"] for p in parts]))
    return out


# --- nb03 section 3: RFI occupancy ------------------------------------------------------

def block_counts(span, fb: Filterbank, blk: int, thr_lo):
    """Samples above `thr_lo` (per channel) and samples railed at 65535, per block."""
    x_all = fb.memmap()
    b0, b1 = span
    hi = np.zeros((b1 - b0, fb.nchans), np.int32)
    sat = np.zeros((b1 - b0, fb.nchans), np.int32)
    for k, b in enumerate(range(b0, b1)):
        raw = np.asarray(x_all[b * blk:(b + 1) * blk])
        hi[k] = (raw > thr_lo).sum(0)
        sat[k] = (raw == 65535).sum(0)
    return hi, sat


# --- raw windows -----------------------------------------------------------------------

def window_bounds(row, fb: Filterbank):
    return axes.raw_window(row["tcand"], row["dm_csv"], int(row["boxcar_samples"]),
                           fb.tsamp, fb.f_hi, fb.f_lo)


def load_window(row, fb: Filterbank, cache_dir=None):
    """(nout, nchan) uint16 raw window for one candidate, pedestal-padded off the file
    ends. Read from `cache_dir/<name>.npy` when that file exists, else from the memmap.
    Both give the same bytes."""
    start, nout = window_bounds(row, fb)
    if cache_dir is not None:
        p = Path(cache_dir) / (row["h5_name"][:-3] + ".npy")
        if p.exists():
            w = np.load(p)
            if w.shape == (nout, fb.nchans):
                return w
    x_all = fb.memmap()
    a, b = max(start, 0), min(start + nout, fb.nsamp)
    w = np.full((nout, fb.nchans), np.uint16(PEDESTAL), np.uint16)
    w[a - start:a - start + (b - a)] = x_all[a:b]
    return w


def window_record(row, fb: Filterbank, cache_dir=None, write=False):
    """Manifest row for one window. With `write=True` the window is saved to
    `cache_dir` unless a file of the right size is already there (resumable)."""
    start, nout = window_bounds(row, fb)
    a, b = max(start, 0), min(start + nout, fb.nsamp)
    path = None
    if write:
        path = Path(cache_dir) / (row["h5_name"][:-3] + ".npy")
        if not (path.exists() and path.stat().st_size > nout * fb.nchans * 2):
            np.save(path, load_window(row, fb, None))
    w = load_window(row, fb, cache_dir)
    return {"h5_name": row["h5_name"], "npy": row["h5_name"][:-3] + ".npy",
            "start_sample": start, "end_sample": start + nout, "nout": nout,
            "pad_front": a - start, "pad_back": (start + nout) - b,
            "n_clipped": int((w == 65535).sum()),
            "bytes": int(path.stat().st_size) if path is not None else 0}


# --- nb03 section 5: does the h5 image match the raw file? -----------------------------

def _dedispersed_mean_profile(x_all, fb, nstart, nsamp, dm):
    return dedisperse(x_all, nstart, nsamp, dm, fb.chan_freqs, fb.tsamp, "top").mean(axis=1)


def h5_reconstruction(row, fb: Filterbank, cand_dir, dm_rows=(0, 64, 192, 255)):
    """Correlate the h5 images with profiles rebuilt from the raw file under two
    time-axis models and two DM-axis models.

    time, candmaker model: chunk_geometry read, decimate by max(1, boxcar//2), crop 256.
    time, old nb02 model:  tcand +/- (sweep + boxcar*tsamp) resampled onto 256 columns.
    DM rows, linspace(0, 2*dm): row k of data_dm_time against the raw profile at that DM.
    DM rows, dm +/- dm/2:      the same rows under the narrower alternative.
    """
    x_all = fb.memmap()
    W = int(row["boxcar_samples"])
    dm, td = float(row["dm_csv"]), int(axes.time_decimation(W))
    with h5py.File(Path(cand_dir) / row["h5_name"], "r") as f:
        ft = f["data_freq_time"][:]
        dt = f["data_dm_time"][:]
    h5_prof = ft.mean(axis=1)

    nstart, nsamp = axes.chunk_geometry(row["tcand"], dm, W, fb.tsamp, fb.f_hi, fb.f_lo)
    p_cm = axes.decimate_crop(_dedispersed_mean_profile(x_all, fb, nstart, nsamp, dm), td)

    s0, nout = window_bounds(row, fb)
    p_old = _dedispersed_mean_profile(x_all, fb, s0, nout, dm)
    p_old = np.interp(np.linspace(0, nout - 1, axes.NCOL), np.arange(nout), p_old)

    out = {"h5_name": row["h5_name"], "r_time_candmaker": float(np.corrcoef(h5_prof, p_cm)[0, 1]),
           "r_time_nb02": float(np.corrcoef(h5_prof, p_old)[0, 1])}
    full = axes.h5_dm_axis(dm)
    narrow = np.linspace(0.5 * dm, 1.5 * dm, axes.NDM)
    for name, grid in (("linspace_0_2dm", full), ("dm_pm_half", narrow)):
        rs = []
        for k in dm_rows:
            pk = axes.decimate_crop(_dedispersed_mean_profile(x_all, fb, nstart, nsamp, grid[k]), td)
            rs.append(np.corrcoef(dt[k], pk)[0, 1])
        out[f"r_dm_{name}"] = float(np.mean(rs))
    return out


# --- nb03: DM-0 against candidate DM from raw counts -----------------------------------

def dm0_pair(row, fb: Filterbank):
    x_all = fb.memmap()
    start, nout = window_bounds(row, fb)
    w = int(row["boxcar_samples"])
    s0, _ = boxcar(dedisperse(x_all, start, nout, 0.0, fb.chan_freqs, fb.tsamp).sum(1), w)
    sd, _ = boxcar(dedisperse(x_all, start, nout, row["dm_csv"], fb.chan_freqs, fb.tsamp).sum(1), w)
    return {"h5_name": row["h5_name"], "snr_dm0_raw": mad_snr(s0), "snr_dm_raw": mad_snr(sd)}


# --- nb03 section 5: peak position against tcand, both band references ------------------

def peak_offsets(row, fb: Filterbank):
    x_all = fb.memmap()
    start, nout = window_bounds(row, fb)
    w = int(row["boxcar_samples"])
    out = {"h5_name": row["h5_name"], "nout": nout}
    for ref in ("top", "bottom"):
        sm, lag = boxcar(dedisperse(x_all, start, nout, row["dm_csv"], fb.chan_freqs,
                                    fb.tsamp, ref).sum(1), w)
        jpk = sm.argmax() + lag
        out[f"off_ms_{ref}"] = (jpk - nout // 2) * fb.tsamp * 1e3
        out[f"snr_{ref}"] = mad_snr(sm)
        if ref == "top":
            out["profile_top"] = sm.astype(np.float32)
            out["lag"] = lag
    return out


# --- nb04: the raw-window feature pass --------------------------------------------------

def raw_pass_one(row, fb: Filterbank, cache_dir=None, widths=tuple(2 ** np.arange(11)), nsub=8):
    """Widths, peak position, on-pulse spectrum and sub-band arrival times for one
    candidate, from its raw window. Mirrors nb04's original serial loop exactly."""
    w = load_window(row, fb, cache_dir)
    nch = fb.nchans
    d = delay_samples(row["dm_csv"], fb.chan_freqs, fb.tsamp)
    start, nout = window_bounds(row, fb)
    j0 = max(0, -start)                               # rows of synthetic pedestal at the front
    nv = nout - int(d.max()) - j0
    if nv <= 0:
        raise ValueError(f"{row['h5_name']}: no valid rows after dedispersion")

    valid = w[j0:]
    spec_mean = valid.mean(axis=0, dtype=np.float64) - PEDESTAL
    spec_std = valid.std(axis=0, dtype=np.float64)

    sub = np.zeros((nsub, nv), np.float64)
    per = nch // nsub
    for ch in range(nch):
        s = j0 + d[ch]
        sub[ch // per] += w[s:s + nv, ch]
    prof_dd = sub.sum(axis=0) - nch * PEDESTAL

    def smooth(p, k):
        return boxcar(p, int(k))[0]

    sn = np.array([mad_snr(smooth(prof_dd, x)) for x in widths if x < nv])
    bw = int(np.asarray(widths)[: len(sn)][sn.argmax()])
    k = int(smooth(prof_dd, bw).argmax())             # START of the best boxcar window
    peak_off = k + j0 - (int(round(row["tcand"] / fb.tsamp)) - start)

    sw = max(1, bw // 4)
    sm = smooth(prof_dd, sw)
    kk = int(sm.argmax())
    base, pk = np.median(sm), sm.max()
    half = base + (pk - base) / 2
    lo = kk
    while lo > 0 and sm[lo - 1] >= half:
        lo -= 1
    hi = kk
    while hi < sm.size - 1 and sm[hi + 1] >= half:
        hi += 1
    lo4, hi4 = max(0, kk - 4 * bw), min(sm.size, kk + 4 * bw + 1)
    weq = (sm[lo4:hi4] - base).sum() / (pk - base) if pk > base else np.nan

    spec_on = np.empty(nch, np.float32)
    for ch in range(nch):
        s = j0 + k + d[ch]
        spec_on[ch] = w[s:s + bw, ch].mean(dtype=np.float64) - PEDESTAL
    sub_off = np.array([int(smooth(sub[b], bw).argmax()) - k for b in range(nsub)], np.int64)

    return {"best_w": bw, "best_snr": float(sn.max()), "peak_row": k + j0,
            "peak_off_samp": int(peak_off), "fwhm_samp": float(hi - lo + 1), "weq_samp": float(weq),
            "spec_on": spec_on, "spec_mean": spec_mean.astype(np.float32),
            "spec_std": spec_std.astype(np.float32), "sub_off": sub_off}


# --- h5 curves (use processes: h5py serialises threads behind one lock) ------------------

def h5_curves(path):
    """Integrated profile (mean over frequency), DM-SNR curve (max over time), the
    mean-collapsed DM curve kept as evidence of the collapse trap, and the image mean."""
    with h5py.File(path, "r") as f:
        ft = f["data_freq_time"][:]
        dt = f["data_dm_time"][:]
    if ft.shape != (256, 256) or dt.shape != (256, 256):
        raise ValueError(f"{path}: unexpected shapes {ft.shape}, {dt.shape}")
    return (ft.mean(axis=1).astype(np.float64), dt.max(axis=1).astype(np.float64),
            dt.mean(axis=1).astype(np.float64), float(ft.mean()))
