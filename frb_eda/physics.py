"""Cold-plasma dispersion, incoherent dedispersion and the boxcar S/N estimator."""
from __future__ import annotations

import numpy as np

K_DM = 4.148808e3          # s MHz^2 pc^-1 cm^3
PEDESTAL = 32768.0         # nb03: the gain-equalised data sit on 2**15


def dm_delay_s(dm, f_hi_mhz, f_lo_mhz):
    """Arrival delay (s) of f_lo relative to f_hi."""
    return K_DM * np.asarray(dm, float) * (f_lo_mhz ** -2 - f_hi_mhz ** -2)


def dm_smear_chan_s(dm, f_mhz, chan_bw_mhz):
    """Dispersion smearing inside one channel (s): the floor on observed width."""
    return 2.0 * K_DM * np.asarray(dm, float) * abs(chan_bw_mhz) / np.asarray(f_mhz, float) ** 3


def w_eff_s(w_int_s, dm, f_mhz, chan_bw_mhz, tsamp_s):
    """Observed width: intrinsic, intra-channel smearing and sampling in quadrature.
    Scattering and DM-step smearing are left out."""
    return np.sqrt(np.asarray(w_int_s, float) ** 2
                   + dm_smear_chan_s(dm, f_mhz, chan_bw_mhz) ** 2 + tsamp_s ** 2)


def delay_samples(dm, chan_freqs, tsamp, ref="top"):
    """Per-channel integer shift in samples, relative to the top or bottom of the band."""
    f_ref = chan_freqs.max() if ref == "top" else chan_freqs.min()
    dt = K_DM * dm * (chan_freqs ** -2 - f_ref ** -2)
    return np.round(dt / tsamp).astype(np.int64)


def dedisperse(src, start, nout, dm, chan_freqs, tsamp, ref="top", subbands=None):
    """Dedispersed (nout, nchan) float32 block, pedestal removed.

    `src` is the (nsamp, nchan) memmap or any array laid out the same way. Rows
    outside `src` are filled with the pedestal, so they add nothing once it comes
    off. Extra rows are read rather than wrapped (`your`'s kernel wraps modulo the
    chunk length). The cast to float happens one channel at a time on the way
    into `out`, because summing 1024 uint16 channels overflows.
    """
    d = delay_samples(dm, chan_freqs, tsamp, ref)
    lo, hi = start + d.min(), start + nout + d.max()
    n = len(src)
    buf = np.full((hi - lo, src.shape[1]), np.uint16(PEDESTAL), np.uint16)
    a, b = max(lo, 0), min(hi, n)
    if b > a:
        buf[a - lo:b - lo] = src[a:b]
    out = np.empty((nout, src.shape[1]), np.float32)
    for i, di in enumerate(d):
        j = start + di - lo
        out[:, i] = buf[j:j + nout, i]
    out -= PEDESTAL
    return out


def boxcar(p, w):
    """Running sum of w samples, divided by sqrt(w) so the noise level stays flat.

    Returns (smoothed, lag). Element k of the output sums p[k:k+w], so a pulse
    centred at c peaks at k = c - lag with lag = (w-1)/2. nb04 lost half a boxcar
    to this before the lag was returned explicitly.
    """
    if w <= 1:
        return np.asarray(p, float), 0.0
    c = np.cumsum(np.concatenate(([0.0], p)))
    return (c[w:] - c[:-w]) / np.sqrt(w), (w - 1) / 2.0


def mad_snr(p):
    """(max - median) / (1.4826 * MAD). Robust to the outliers we are hunting."""
    p = np.asarray(p, float)
    m = np.median(p)
    return (p.max() - m) / (1.4826 * np.median(np.abs(p - m)))


def robust(x):
    """(median, MAD-sigma)."""
    x = np.asarray(x, float)
    med = np.median(x)
    return med, 1.4826 * np.median(np.abs(x - med))
