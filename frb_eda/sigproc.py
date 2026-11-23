"""SIGPROC filterbank header parsing and the memmap onto the data.

Parsing by hand keeps the project on numpy, gives the exact data offset the
memmap needs, and lets nb01 check the header against the file size instead of
trusting `Search_parameters.txt`.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

# keyword -> type. 'd' float64, 'i' int32, 's' length-prefixed string, 'b' int8
SIGPROC_TYPES = {
    "az_start": "d", "za_start": "d", "src_raj": "d", "src_dej": "d",
    "tstart": "d", "tsamp": "d", "fch1": "d", "foff": "d", "refdm": "d",
    "period": "d", "fchannel": "d",
    "telescope_id": "i", "machine_id": "i", "data_type": "i", "nchans": "i",
    "nbits": "i", "nifs": "i", "nbeams": "i", "ibeam": "i", "nsamples": "i",
    "barycentric": "i", "pulsarcentric": "i",
    "source_name": "s", "rawdatafile": "s",
    "signed": "b",
}


def read_sigproc_header(path) -> dict:
    """Parse a SIGPROC header. The returned dict also holds `_header_bytes`,
    the offset of the first data byte."""
    hdr = {}
    with open(path, "rb") as f:
        def lenstr():
            n = struct.unpack("<i", f.read(4))[0]
            # desync tripwire: a misjudged field width makes the next length junk,
            # and without this check Python tries a multi-GB read
            if not 0 <= n <= 80:
                raise ValueError(f"implausible string length {n} at byte {f.tell() - 4}")
            return f.read(n).decode("ascii", errors="replace")

        first = lenstr()
        if first != "HEADER_START":
            raise ValueError(f"not a SIGPROC filterbank (got {first!r})")
        while True:
            key = lenstr()
            if key == "HEADER_END":
                break
            t = SIGPROC_TYPES.get(key)
            if t == "d":
                hdr[key] = struct.unpack("<d", f.read(8))[0]
            elif t == "i":
                hdr[key] = struct.unpack("<i", f.read(4))[0]
            elif t == "b":
                hdr[key] = struct.unpack("<b", f.read(1))[0]
            elif t == "s":
                hdr[key] = lenstr()
            else:
                raise KeyError(f"unhandled SIGPROC keyword {key!r} at byte {f.tell()}")
        hdr["_header_bytes"] = f.tell()
    return hdr


def sigproc_angle_to_deg(v: float, is_ra: bool) -> float:
    """SIGPROC packs angles as (+/-)HHMMSS.s or DDMMSS.s in one float64."""
    sign = -1.0 if v < 0 else 1.0
    v = abs(v)
    dd = int(v // 10000)
    mm = int((v - dd * 10000) // 100)
    ss = v - dd * 10000 - mm * 100
    deg = dd + mm / 60.0 + ss / 3600.0
    return sign * (deg * 15.0 if is_ra else deg)


@dataclass(frozen=True)
class Filterbank:
    """Instrument geometry of one filterbank file, derived from its header."""
    path: str
    header_bytes: int
    nchans: int
    nbits: int
    nsamp: int
    tsamp: float
    fch1: float
    foff: float

    @property
    def chan_freqs(self) -> np.ndarray:
        """Channel centres in MHz. Index 0 is the top of the band when foff < 0."""
        return self.fch1 + self.foff * np.arange(self.nchans)

    @property
    def f_hi(self) -> float:
        return float(self.chan_freqs.max())

    @property
    def f_lo(self) -> float:
        # the last channel centre, not the band edge: nb03 found this is the
        # convention `your` uses for the sweep (the edge is off by 5.2 ms at DM 2795)
        return float(self.chan_freqs.min())

    def memmap(self) -> np.memmap:
        return _memmap(self.path, self.header_bytes, self.nsamp, self.nchans, self.nbits)


@lru_cache(maxsize=4)
def _memmap(path, offset, nsamp, nchans, nbits):
    # cached per process, so each worker opens the file once and reuses the mapping
    return np.memmap(path, dtype=np.dtype(f"<u{nbits // 8}"), mode="r",
                     offset=offset, shape=(nsamp, nchans))


def open_filterbank(path) -> Filterbank:
    """Read the header and check that the file size closes exactly."""
    path = str(Path(path))
    hdr = read_sigproc_header(path)
    nchans, nbits = int(hdr["nchans"]), int(hdr["nbits"])
    data_bytes = Path(path).stat().st_size - hdr["_header_bytes"]
    spectrum = nchans * nbits // 8
    if data_bytes % spectrum:
        raise ValueError(f"{path}: {data_bytes} data bytes is not a whole number of "
                         f"{spectrum}-byte spectra (truncated file or wrong layout)")
    nsamp = data_bytes // spectrum
    if "nsamples" in hdr and int(hdr["nsamples"]) != nsamp:
        raise ValueError(f"header nsamples {hdr['nsamples']} != {nsamp} from file size")
    return Filterbank(path=path, header_bytes=int(hdr["_header_bytes"]), nchans=nchans,
                      nbits=nbits, nsamp=int(nsamp), tsamp=float(hdr["tsamp"]),
                      fch1=float(hdr["fch1"]), foff=float(hdr["foff"]))
