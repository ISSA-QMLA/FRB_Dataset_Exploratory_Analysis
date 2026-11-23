"""One observation in, one feature table and one diagnostics record out.

This is nb01-nb04 with the plots, prose and pilot-specific asserts taken out, so it
can run unattended over the whole corpus. Every formula is the notebooks' own; the
test `test_pipeline_reproduces_nb04_feature_matrix` holds it to nb04's committed
feature_matrix.csv on CUT_P_110. Instrument constants come from each file's own
header, never from the pilot's.

Per observation it produces
  features    one row per candidate: identity, ground truth, FETCH's verdict and the
              33 features of nb04 groups A, B_h5, B_raw, C, D and E
  injections  one row per injected burst: was it recovered by Heimdall, by FETCH
              (nb01's recovery table)
  summary     one dict: counts, FETCH confusion matrix, RFI, sanity checks, timings

Feature *selection* and transforms (nb04's second half) are corpus-level decisions
and are not done here: they run once on the pooled table.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import axes, config, parallel, physics, provenance, raw, sigproc
from .physics import PEDESTAL

# nb01's matching tolerances and sentinel
TOL_T_S, TOL_DM_FRAC, SENTINEL = 0.50, 0.10, -999.0
# nb03's RFI passes
BLK, K_LO, BURSTY_X = 1024, 4.0, 1.3
# nb04's raw pass and feature constants
WIDTHS = tuple(int(w) for w in 2 ** np.arange(11))
NSUB = 8
PULSE_HALFWIDTH = 4

GROUPS = {
    "A": ["prof_mean", "prof_std", "prof_skew", "prof_kurt", "dmc_mean", "dmc_std", "dmc_skew", "dmc_kurt"],
    "B_h5": ["prof_peak_snr", "prof_peak_off_abs", "prof_n_5sig", "prof_offpulse_kurt"],
    "B_raw": ["raw_boxcar_ms", "raw_peak_snr", "raw_fwhm_ms", "raw_weq_ms", "raw_peak_off_boxcar"],
    "C": ["dm0_ratio", "dmc_fwhm_rows", "dmc_asym", "dmc_sharpness", "dmc_peak_off_row",
          "dmc_peak_at_edge", "dmc_bowtie_corr"],
    "D": ["spec_frac_3sig", "spec_n_blocks", "spec_frac_bursty", "subband_coh"],
    "E": ["dm_csv", "boxcar_ms", "snr_csv", "members", "local_density"],
}
FEATURES = [f for g in GROUPS.values() for f in g]
ID_COLS = ["obs", "h5_name", "is_injection", "matched_TOA", "dt_to_nearest_inj_s", "dm_frac_err_nearest",
           "tcand", "boxcar_samples", "fetch_prob", "fetch_label", "fetch_prob_std", "fetch_n_models_pos"]


# --- locating an observation's files ----------------------------------------------------------

@dataclass(frozen=True)
class Observation:
    name: str          # e.g. "CUT_P_110"
    cand_dir: Path     # candmaker + FETCH output folder
    fil: Path          # raw filterbank
    inj_csv: Path      # injection ground truth


def _only(hits: list[Path], what: str) -> Path:
    if len(hits) != 1:
        raise FileNotFoundError(f"{what}: expected one match, found {len(hits)}: {[h.name for h in hits]}")
    return hits[0]


def find_observation(root: Path, name: str) -> Observation:
    """Files of observation `name` under `root` (the corpus folder, or a staging copy)."""
    root = Path(root)
    return Observation(
        name=name,
        cand_dir=_only([p for p in root.glob(f"{name}_*processed_*") if p.is_dir()], f"{name} candidate folder"),
        fil=_only(sorted(root.glob(f"{name}_*processed.fil")), f"{name} filterbank"),
        inj_csv=_only(sorted(root.glob(f"{name}_inj_params.csv")), f"{name} injection csv"),
    )


def list_observations(root: Path) -> list[str]:
    """Every observation in `root` that has an injection file, sorted."""
    return sorted(p.name[: -len("_inj_params.csv")] for p in Path(root).glob("*_inj_params.csv"))


# --- nb01: the labelled candidate table -------------------------------------------------------

_CAND_RE = re.compile(r"cand_tstart_(?P<tstart>[\d.]+)_tcand_(?P<tcand>[\d.]+)_dm_(?P<dm>[\d.]+)_snr_(?P<snr>[\d.]+)\.h5$")
_CAND_COLS = ["snr", "samp_idx", "time", "filter", "dm_trial", "dm", "members",
              "begin", "end", "nbeams", "beam_mask", "prim_beam", "max_snr", "beam"]


def _parse_cand_path(s) -> tuple[str, float]:
    name = str(s).replace("\\", "/").rsplit("/", 1)[-1]
    hit = _CAND_RE.match(name)
    if hit is None:
        raise ValueError(f"filename did not parse: {name}")
    return name, float(hit["tcand"])


def _read_cand(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, engine="python").dropna(axis=1, how="all")
    df.columns = _CAND_COLS[: df.shape[1]]
    return df


def master_table(obs: Observation, fb: sigproc.Filterbank) -> tuple[pd.DataFrame, dict]:
    """nb01's candidates_labelled.csv for one observation, plus what was dropped and why."""
    d = obs.cand_dir
    multi = pd.read_csv(d / "Results_MULTI.csv")
    fetch = pd.DataFrame([{"h5_name": _parse_cand_path(r["candidate"])[0],
                           "tcand": _parse_cand_path(r["candidate"])[1],
                           "fetch_prob": float(r["probability"]), "fetch_label": float(r["label"])}
                          for _, r in multi.iterrows()]).sort_values("tcand").reset_index(drop=True)

    model_prob = {}
    for p in sorted(d.glob("results_?.csv")):
        mdf = pd.read_csv(p, index_col=0)
        mdf["h5_name"] = [_parse_cand_path(c)[0] for c in mdf["candidate"]]
        model_prob[p.stem.split("_")[-1]] = mdf.set_index("h5_name")["probability"]
    M = pd.DataFrame(model_prob).reindex(fetch.h5_name)
    fetch["fetch_prob_std"] = M.std(axis=1).to_numpy()
    fetch["fetch_n_models_pos"] = (M > 0.5).sum(axis=1).to_numpy()

    cand_all = _read_cand(_only(sorted(d.glob("*_all.cand")), "*_all.cand"))
    allc = pd.read_csv(_only(sorted(d.glob("*_all_candidates.csv")), "*_all_candidates.csv"))
    allc.columns = [c.strip() for c in allc.columns]
    cand_all["_k"] = cand_all.time.round(5)
    allc["_k"] = allc.stime.round(5)
    heim = allc.merge(cand_all, on="_k", how="left", suffixes=("_csv", "_cand"))
    fetch["_k"] = fetch.tcand.round(5)
    c = fetch.merge(heim, on="_k", how="left").drop(columns=["_k"])
    c["boxcar_samples"] = 2 ** c["width"]
    c["boxcar_ms"] = c.boxcar_samples * fb.tsamp * 1e3

    n_fetch = len(c)
    no_heim = c.dm_csv.isna()
    no_h5 = ~np.array([(d / n).exists() for n in c.h5_name])
    c = c[~no_heim & ~no_h5].reset_index(drop=True)

    lab, inj = provenance.label_candidates(c.tcand, c.dm_csv, obs.inj_csv,
                                           tol_t_s=TOL_T_S, tol_dm_frac=TOL_DM_FRAC, sentinel=SENTINEL)
    for col in lab.columns:
        c[col] = lab[col].to_numpy()
    return c, {"n_fetch_candidates": int(n_fetch), "n_dropped_no_heimdall_row": int(no_heim.sum()),
               "n_dropped_no_h5": int((no_h5 & ~no_heim).sum()), "inj": inj}


def injection_table(c: pd.DataFrame, inj: pd.DataFrame) -> pd.DataFrame:
    """nb01's injection_recovery_match.csv: one row per injection, best candidate by S/N."""
    rows = []
    for _, q in inj.iterrows():
        sel = c[(np.abs(c.tcand - q.TOA) <= TOL_T_S) & (np.abs(c.dm_csv - q.DM) / q.DM <= TOL_DM_FRAC)]
        hit = len(sel) > 0
        bst = sel.loc[sel.snr_csv.idxmax()] if hit else None
        rows.append({"TOA": q.TOA, "dm_inj": q.DM, "snr_inj": q.get("SNR", np.nan),
                     "fwhm_inj_ms": q.get("FWHM_1", np.nan) * 1e3, "shape": q.get("SHAPE", ""),
                     "recovered": hit, "n_cands": len(sel),
                     "tcand": bst.tcand if hit else np.nan, "dm_heimdall": bst.dm_csv if hit else np.nan,
                     "snr_heimdall": bst.snr_csv if hit else np.nan,
                     "boxcar_ms": bst.boxcar_ms if hit else np.nan,
                     "fetch_prob": bst.fetch_prob if hit else np.nan,
                     "fetch_label": bst.fetch_label if hit else np.nan,
                     "members": bst.members if hit else np.nan})
    m = pd.DataFrame(rows)
    m["dt_s"] = m.tcand - m.TOA
    m["dm_err_%"] = 100 * (m.dm_heimdall - m.dm_inj) / m.dm_inj
    m["fetch_ok"] = m.fetch_label == 1
    return m


def chunk_geometry_check(c: pd.DataFrame, obs: Observation, fb: sigproc.Filterbank) -> dict:
    """nb01's check that the h5 axis model reproduces every candmaker log line."""
    logs = sorted(obs.cand_dir.glob("your_candmaker_*.log"))
    if len(logs) != 1:
        return {"chunk_log_found": False}
    log = pd.DataFrame(axes.parse_chunk_log(logs[0]), columns=["nstart_read", "nsamp_read", "nstart", "nsamp"])
    pred = pd.DataFrame([axes.chunk_geometry(r.tcand, r.dm_csv, int(r.boxcar_samples), fb.tsamp, fb.f_hi, fb.f_lo)
                         for r in c.itertuples()], columns=["nstart_read_pred", "nsamp_read_pred"])
    pred["nstart"] = [int((r.tcand - float(physics.dm_delay_s(r.dm_csv, fb.f_hi, fb.f_lo))
                           - r.boxcar_samples * fb.tsamp) / fb.tsamp) for r in c.itertuples()]
    chk = pred.merge(log.drop_duplicates("nstart"), on="nstart", how="left")
    ok = (chk.nstart_read == chk.nstart_read_pred) & (chk.nsamp_read == chk.nsamp_read_pred)
    return {"chunk_log_found": True, "chunk_geometry_exact": int(ok.sum()), "chunk_geometry_of": int(len(c))}


# --- nb03: which channels carry the bursty RFI ------------------------------------------------

def rfi_channels(fb: sigproc.Filterbank, n_jobs: int | None = None) -> dict:
    """nb03's two streaming passes: bandpass, then railed-sample bursts per channel."""
    n_jobs = n_jobs or config.N_JOBS
    nblk = int(np.ceil(fb.nsamp / BLK))
    agg = raw.combine_block_stats(parallel.pmap(raw.block_stats, parallel.chunks(nblk, 4 * n_jobs),
                                                n_jobs=n_jobs, fb=fb, blk=BLK))
    ch_mean = agg["ch_sum"] / fb.nsamp + PEDESTAL
    ch_std = np.sqrt(np.maximum(agg["ch_sq"] / fb.nsamp - (agg["ch_sum"] / fb.nsamp) ** 2, 0.0))
    nfull = fb.nsamp // BLK
    parts = parallel.pmap(raw.block_counts, parallel.chunks(nfull, 4 * n_jobs), n_jobs=n_jobs,
                          fb=fb, blk=BLK, thr_lo=ch_mean + K_LO * ch_std)
    blk_hi = np.concatenate([p[0] for p in parts])
    blk_sat = np.concatenate([p[1] for p in parts])
    occ_blocks = (blk_sat > 0).sum(0)
    n_clip = blk_sat.sum(0)
    lam = n_clip / nfull
    with np.errstate(divide="ignore", invalid="ignore"):
        poisson_e = np.where(lam > 0, lam / (1 - np.exp(-np.maximum(lam, 1e-12))), np.nan)
        burst_x = np.divide(n_clip, np.maximum(occ_blocks, 1), dtype=float) / poisson_e
    bursty = burst_x > BURSTY_X
    return {"bursty": bursty, "burstiness_x": burst_x,
            "frac_above_4sig": float(blk_hi.sum() / (nfull * BLK * fb.nchans)),
            "frac_railed": float(blk_sat.sum() / (nfull * BLK * fb.nchans)),
            "n_dead_channels": int((ch_std == 0).sum())}


# --- nb04: the features -----------------------------------------------------------------------

def _four_stats(x):
    return x.mean(axis=1), x.std(axis=1, ddof=0), stats.skew(x, axis=1), stats.kurtosis(x, axis=1)


def _n_runs(mask):
    return int(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))).clip(min=0).sum())


def features(c: pd.DataFrame, obs: Observation, fb: sigproc.Filterbank, bursty: np.ndarray,
             n_jobs: int | None = None) -> tuple[pd.DataFrame, dict]:
    """nb04 groups A-E for every row of the master table, in master order."""
    timings = {}
    names = c["h5_name"].to_numpy()
    tsamp, f_hi, f_lo = fb.tsamp, fb.f_hi, fb.f_lo

    t0 = time.perf_counter()
    res = parallel.pmap(raw.h5_curves, [str(obs.cand_dir / n) for n in names], n_jobs=n_jobs)
    prof = np.array([r[0] for r in res])
    dmc = np.array([r[1] for r in res])
    timings["h5_curves_s"] = time.perf_counter() - t0

    # nb02's axis table, the columns nb04 reads
    td = axes.time_decimation(c["boxcar_samples"].to_numpy())
    t_step_ms = td * tsamp * 1e3
    t0_ms = np.array([axes.h5_time_axis_exact(r.tcand, r.dm_csv, int(r.boxcar_samples), tsamp, f_hi, f_lo)[0] * 1e3
                      for r in c.itertuples()])
    tcand_col = -t0_ms / t_step_ms
    dm_step = 2 * c["dm_csv"].to_numpy() / (axes.NDM - 1)

    # group A
    dmc_base = dmc - np.median(dmc, axis=1, keepdims=True)
    X = pd.DataFrame({"h5_name": names})
    for tag, arr in (("prof", prof), ("dmc", dmc_base)):
        X[f"{tag}_mean"], X[f"{tag}_std"], X[f"{tag}_skew"], X[f"{tag}_kurt"] = _four_stats(arr)

    # group B, h5 profile
    off_col = prof.argmax(axis=1) - tcand_col
    base = np.median(prof, axis=1, keepdims=True)
    sigma = 1.4826 * np.median(np.abs(prof - base), axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        prof_z = (prof - base) / sigma
    off_kurt = np.empty(len(prof))
    for i, pz in enumerate(prof_z):
        k = int(prof[i].argmax())
        keep = np.ones(pz.size, bool)
        keep[max(0, k - PULSE_HALFWIDTH):k + PULSE_HALFWIDTH + 1] = False
        off_kurt[i] = stats.kurtosis(pz[keep])
    X["prof_peak_snr"] = [physics.mad_snr(physics.boxcar(p, 1)[0]) for p in prof]
    X["prof_peak_off_abs"] = np.abs(off_col)
    X["prof_n_5sig"] = [_n_runs(pz > 5) for pz in prof_z]
    X["prof_offpulse_kurt"] = off_kurt

    # group B, raw windows
    t0 = time.perf_counter()
    rows = [r._asdict() for r in c[["h5_name", "tcand", "dm_csv", "boxcar_samples"]].itertuples(index=False)]
    rp = parallel.pmap(raw.raw_pass_one, rows, n_jobs=n_jobs, fb=fb, widths=WIDTHS, nsub=NSUB)
    timings["raw_pass_s"] = time.perf_counter() - t0
    best_w = np.array([r["best_w"] for r in rp], np.int64)
    peak_off_samp = np.array([r["peak_off_samp"] for r in rp], np.int64)
    X["raw_boxcar_ms"] = best_w * tsamp * 1e3
    X["raw_peak_snr"] = np.array([r["best_snr"] for r in rp])
    X["raw_fwhm_ms"] = np.array([r["fwhm_samp"] for r in rp]) * tsamp * 1e3
    X["raw_weq_ms"] = np.array([r["weq_samp"] for r in rp]) * tsamp * 1e3
    X["raw_peak_off_boxcar"] = (peak_off_samp + (best_w - 1) / 2) / best_w

    # group C, DM-SNR curve
    k_sweep_ms = float(physics.dm_delay_s(1.0, f_hi, f_lo)) * 1e3
    dmc_pk = dmc_base.argmax(axis=1)
    rows_idx = np.arange(dmc_base.shape[1])
    last = dmc_base.shape[1] - 1
    fwhm_rows, asym, sharp, bowtie = (np.empty(len(dmc_base)) for _ in range(4))
    boxcar_ms = c["boxcar_ms"].to_numpy()
    for i, p in enumerate(dmc_base):
        k = int(dmc_pk[i])
        half = p.max() / 2
        lo = k
        while lo > 0 and p[lo - 1] >= half:
            lo -= 1
        hi = k
        while hi < last and p[hi + 1] >= half:
            hi += 1
        fwhm_rows[i] = hi - lo + 1
        left, right = p[max(0, k - 20):k].sum(), p[k + 1:k + 21].sum()
        asym[i] = (right - left) / (right + left) if (right + left) != 0 else np.nan
        sharp[i] = ((2 * p[k] - p[max(k - 1, 0)] - p[min(k + 1, last)]) / p[k]) if p[k] > 0 else np.nan
        ddm = (rows_idx - k) * dm_step[i]
        bowtie[i] = np.corrcoef(p, (1 + (ddm * k_sweep_ms / max(boxcar_ms[i], 1e-3)) ** 2) ** -0.25)[0, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        X["dm0_ratio"] = dmc_base[:, 0] / dmc_base.max(axis=1)
    X["dmc_fwhm_rows"] = fwhm_rows
    X["dmc_asym"] = asym
    X["dmc_sharpness"] = sharp
    X["dmc_peak_off_row"] = np.abs(dmc_pk - last / 2)
    X["dmc_peak_at_edge"] = ((dmc_pk <= 1) | (dmc_pk >= last - 1)).astype(int)
    X["dmc_bowtie_corr"] = bowtie

    # group D, on-pulse spectrum
    spec_on = np.stack([r["spec_on"] for r in rp])
    spec_mean = np.stack([r["spec_mean"] for r in rp])
    spec_std = np.stack([r["spec_std"] for r in rp])
    sub_off = np.stack([r["sub_off"] for r in rp])
    spec_amp = spec_on - spec_mean
    with np.errstate(divide="ignore", invalid="ignore"):
        spec_z = spec_amp / (spec_std / np.sqrt(best_w)[:, None])
        pos = np.clip(spec_amp, 0, None)
        tot = pos.sum(axis=1)
        X["spec_frac_3sig"] = (spec_z > 3).mean(axis=1)
        X["spec_n_blocks"] = [_n_runs(r > 3) for r in spec_z]
        X["spec_frac_bursty"] = pos[:, bursty].sum(axis=1) / np.where(tot > 0, tot, np.nan)
        X["subband_coh"] = sub_off.std(axis=1) / best_w

    # group E, Heimdall metadata and crowding
    t_arr, dm_arr = c["tcand"].to_numpy(), c["dm_csv"].to_numpy()
    X["dm_csv"] = dm_arr
    X["boxcar_ms"] = boxcar_ms
    X["snr_csv"] = c["snr_csv"].to_numpy()
    X["members"] = c["members"].to_numpy()
    X["local_density"] = [int(((np.abs(t_arr - t_arr[i]) <= 1.0) & (np.abs(dm_arr - dm_arr[i]) / dm_arr[i] <= 0.10)).sum() - 1)
                          for i in range(len(c))]

    out = c[[col for col in ID_COLS if col in c.columns]].copy()
    out = pd.concat([out.reset_index(drop=True), X[FEATURES].reset_index(drop=True)], axis=1)
    timings["k_sweep_ms_per_dm"] = k_sweep_ms
    return out, timings


# --- the whole observation --------------------------------------------------------------------

def run_observation(obs: Observation, n_jobs: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(features, injections, summary) for one observation. Raises on unreadable input."""
    t_all = time.perf_counter()
    fb = sigproc.open_filterbank(obs.fil)
    summary = {"obs": obs.name, "fil_bytes": obs.fil.stat().st_size, "nsamp": fb.nsamp, "nchans": fb.nchans,
               "tsamp_s": fb.tsamp, "fch1_mhz": fb.fch1, "foff_mhz": fb.foff, "duration_s": fb.nsamp * fb.tsamp}

    c, info = master_table(obs, fb)
    c.insert(0, "obs", obs.name)
    inj = info.pop("inj")
    summary.update(info)

    t0 = time.perf_counter()
    rfi = rfi_channels(fb, n_jobs)
    summary["rfi_s"] = time.perf_counter() - t0
    bursty = rfi.pop("bursty")
    summary.update({k: v for k, v in rfi.items() if k != "burstiness_x"})
    summary["n_bursty_channels"] = int(bursty.sum())
    summary["bursty_channels"] = " ".join(map(str, np.flatnonzero(bursty)))

    feats, timings = features(c, obs, fb, bursty, n_jobs)
    summary.update(timings)

    m = injection_table(c, inj)
    m.insert(0, "obs", obs.name)
    y = c["is_injection"].astype(bool).to_numpy()
    f = (c["fetch_label"] == 1).to_numpy()
    summary.update({
        "n_candidates": int(len(c)), "n_positive_candidates": int(y.sum()),
        "n_injections": int(len(inj)), "n_injections_recovered": int(m.recovered.sum()),
        "n_injections_fetch_ok": int(m.fetch_ok.sum()),
        "fetch_tp": int((y & f).sum()), "fetch_fn": int((y & ~f).sum()),
        "fetch_fp": int((~y & f).sum()), "fetch_tn": int((~y & ~f).sum()),
        "n_feature_nan": int(feats[FEATURES].isna().sum().sum()),
        "features_with_nan": " ".join(col for col in FEATURES if feats[col].isna().any()),
    })
    summary.update(chunk_geometry_check(c, obs, fb))
    summary["total_s"] = time.perf_counter() - t_all
    return feats, m, summary
