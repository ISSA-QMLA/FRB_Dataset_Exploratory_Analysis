"""Geometry of the candmaker `.h5` images and of the raw windows cut around them.

Neither h5 array carries coordinates. Both axes are rebuilt here from what
`your`'s candmaker does (this run used the GPU path, `gpu_dedisp_and_dmt_crop`,
per the log), and nb02/nb03 check the result against the log and the raw file.

Time axis (both arrays). candmaker asks `Candidate.get_chunk` for
tcand +/- (sweep + boxcar*tsamp). If that is shorter than
`nchunk = boxcar * 256 // 2` samples (256 when boxcar < 3) the read is widened
symmetrically to `nchunk`. The block is dedispersed against the top channel,
summed in time over blocks of `max(1, boxcar // 2)` samples (and over 4 channels
in frequency), and the centre 256 columns are kept. So one column is
`max(1, boxcar // 2) * tsamp`, the image spans 256 of those, and every pulse
covers about two columns by design. The span scales with the boxcar, not with DM.
The log records `nstart_read`/`nsamp_read` for every candidate, and
`chunk_geometry` reproduces all 102 of them (`parse_chunk_log`).

DM axis of `data_dm_time`: 256 trial DMs `linspace(0, 2*dm, 256)`, so row 0 is
DM 0 and the candidate DM sits between rows 127 and 128.

Both images then go through `your.utils.math.normalise`: subtract the image
MEDIAN, divide by the image std. So the image std is 1 and the image mean is
(mean - median) / std of the image before normalising, a skewness-like number.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .physics import dm_delay_s

NCOL = 256      # time columns in both h5 arrays
NDM = 256       # DM rows in data_dm_time
MIN_SAMP = 256  # candmaker's time_size


def time_decimation(boxcar_samples) -> np.ndarray:
    """candmaker's time decimation factor, max(1, boxcar // 2). nb03 matches it
    against the candmaker log for all 102 candidates."""
    return np.maximum(1, np.asarray(boxcar_samples, int) // 2)


def h5_time_step_s(boxcar_samples, tsamp):
    """Width of one h5 time column in seconds."""
    return time_decimation(boxcar_samples) * tsamp


def h5_window_s(boxcar_samples, tsamp, ncol=NCOL):
    """Time spanned by the 256 h5 columns, first column start to last column end."""
    return ncol * h5_time_step_s(boxcar_samples, tsamp)


def h5_time_axis_s(boxcar_samples, tsamp, ncol=NCOL):
    """Idealised column centres in seconds relative to tcand: the centre of a
    256-column grid falls between columns 127 and 128. `h5_time_axis_exact`
    adds the few-sample offset that integer chunk arithmetic leaves."""
    return (np.arange(ncol) - (ncol - 1) / 2) * h5_time_step_s(boxcar_samples, tsamp)


def h5_time_axis_exact(tcand, dm, boxcar, tsamp, f_hi, f_lo, ncol=NCOL):
    """Column centres in seconds relative to tcand, from the exact chunk that
    candmaker read and the crop it applied (`int(n_dec / 2) - 128`)."""
    nstart, nsamp = chunk_geometry(tcand, dm, boxcar, tsamp, f_hi, f_lo)
    td = int(time_decimation(boxcar))
    s = (nsamp // td) // 2 - ncol // 2
    first = nstart + s * td + (td - 1) / 2.0           # mean sample index of column 0
    return (first + td * np.arange(ncol)) * tsamp - tcand


def h5_dm_axis(dm, ndm=NDM):
    """Trial DMs of the rows of data_dm_time. Row 0 is exactly DM 0."""
    return np.linspace(0.0, 2.0 * dm, ndm)


def freq_axis(f_hi, f_lo, nsub=256):
    """Decimated channel centres in MHz. Index 0 is the top of the band."""
    return np.linspace(f_hi, f_lo, nsub)


def chunk_geometry(tcand, dm, boxcar, tsamp, f_hi, f_lo, min_samp=MIN_SAMP):
    """(nstart, nsamp) of the raw samples candmaker read for one candidate,
    following `your.candidate.Candidate.get_chunk`."""
    sweep = float(dm_delay_s(dm, f_hi, f_lo))
    tstart = tcand - sweep - boxcar * tsamp
    tstop = tcand + sweep + boxcar * tsamp
    nstart = int(tstart / tsamp)
    nsamp = int((tstop - tstart) / tsamp)
    nchunk = min_samp if boxcar < 3 else boxcar * min_samp // 2
    if nsamp < nchunk:
        nstart -= (nchunk - nsamp) // 2
        nsamp = nchunk
    return nstart, nsamp


_CHUNK_RE = re.compile(r"nstart_read is (-?\d+), nsamp_read is (\d+),\s*nstart is (-?\d+), nsamp is (\d+)")


def parse_chunk_log(log_path) -> list[tuple[int, int, int, int]]:
    """Every (nstart_read, nsamp_read, nstart, nsamp) line in a candmaker log.
    Each line is self-contained, so worker interleaving does not matter: `nstart`
    (the unwidened start) identifies the candidate."""
    with open(Path(log_path), errors="replace") as f:
        return [tuple(map(int, m.groups())) for line in f if (m := _CHUNK_RE.search(line))]


def raw_window(tcand, dm, boxcar, tsamp, f_hi, f_lo):
    """(start, nout) of the raw window nb03 cuts: tcand +/- (sweep + boxcar*tsamp),
    the range `your` asks for before any widening. It holds the whole dispersed
    pulse, so dedispersing its centre needs no samples from outside it."""
    half = float(dm_delay_s(dm, f_hi, f_lo)) + boxcar * tsamp
    nout = int(round(2 * half / tsamp))
    start = int(round(tcand / tsamp)) - nout // 2
    return start, nout


def decimate_crop(p, factor, ncol=NCOL):
    """Mean over blocks of `factor` samples, then the centre `ncol` blocks."""
    p = np.asarray(p, float)
    m = len(p) // factor * factor
    q = p[:m].reshape(-1, factor).mean(axis=1)
    s = len(q) // 2 - ncol // 2
    return q[s:s + ncol]
