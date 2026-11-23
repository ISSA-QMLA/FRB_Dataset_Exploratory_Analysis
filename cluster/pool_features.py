"""Pool the per-observation outputs of extract_features.py into three tables.

    python cluster/pool_features.py ~/frb_features

writes OUT/pooled/features.csv.gz, injections.csv.gz and summaries.csv, and prints
a short report: how many observations ran, which failed and why, Heimdall recovery,
FETCH recall, RFI and NaN counts. When CUT_P_110 is among them its features are also
checked against nb04's committed feature_matrix.csv, which tests this machine and
these package versions against the laptop run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from frb_eda import pipeline  # noqa: E402


def main(out: Path) -> None:
    summ = pd.DataFrame([json.loads(p.read_text()) for p in sorted(out.glob("*.summary.json"))])
    if summ.empty:
        print(f"no summaries in {out}")
        return
    ok = summ[summ.status == "ok"]
    bad = summ[summ.status != "ok"]
    pooled = out / "pooled"
    pooled.mkdir(exist_ok=True)
    summ.drop(columns=["traceback"], errors="ignore").to_csv(pooled / "summaries.csv", index=False)

    feats = pd.concat([pd.read_csv(out / f"{o}.features.csv") for o in ok.obs], ignore_index=True)
    inj = pd.concat([pd.read_csv(out / f"{o}.injections.csv") for o in ok.obs], ignore_index=True)
    feats.to_csv(pooled / "features.csv.gz", index=False)
    inj.to_csv(pooled / "injections.csv.gz", index=False)

    print(f"observations: {len(ok)} ok, {len(bad)} failed")
    for _, r in bad.iterrows():
        print(f"  FAILED {r.obs}: {r.get('error', '?')}")
    if ok.empty:
        return
    y = feats.is_injection.astype(bool)
    print(f"candidates:   {len(feats):,}  ({int(y.sum()):,} injections, {int((~y).sum()):,} negatives)")
    print(f"injections:   {int(inj.recovered.sum()):,} of {len(inj):,} recovered by Heimdall "
          f"({inj.recovered.mean():.1%});  FETCH accepted {int(inj.fetch_ok.sum()):,} "
          f"({inj.fetch_ok.sum() / max(inj.recovered.sum(), 1):.1%} of recovered)")
    print(f"FETCH on candidates: TP {int(ok.fetch_tp.sum())}  FN {int(ok.fetch_fn.sum())}  "
          f"FP {int(ok.fetch_fp.sum())}  TN {int(ok.fetch_tn.sum())}")
    print(f"bursty channels per observation: median {ok.n_bursty_channels.median():.0f}, "
          f"range {int(ok.n_bursty_channels.min())}-{int(ok.n_bursty_channels.max())}")
    if "stage_not_removed" in summ.columns and summ.stage_not_removed.notna().any():
        print(f"staging copies left on disk: {int(summ.stage_not_removed.notna().sum())} - delete them by hand")
    geo = ok.chunk_geometry_exact / ok.chunk_geometry_of
    print(f"h5 axis model vs candmaker log: {int((geo == 1).sum())} of {len(ok)} observations exact"
          + ("" if (geo == 1).all() else f"  <- check: {', '.join(ok.obs[geo < 1])}"))
    nan = feats[pipeline.FEATURES].isna().sum()
    print(f"feature NaNs: {int(nan.sum())}" + (f"  ({', '.join(f'{k} {v}' for k, v in nan[nan > 0].items())})"
                                              if nan.sum() else ""))
    print(f"time per observation: stage median {ok.stage_s.median():.0f} s, run median {ok.total_s.median():.0f} s")
    print(f"wrote {pooled}/features.csv.gz, injections.csv.gz, summaries.csv")

    if "CUT_P_110" in set(ok.obs):
        ref = pd.read_csv(REPO / "frb_analysis_out/nb04/feature_matrix.csv")
        mine = feats[feats.obs == "CUT_P_110"].reset_index(drop=True)
        worst = max(float(np.nanmax(np.abs(mine[c].to_numpy(float) - ref[c].to_numpy(float))
                                    / np.maximum(np.abs(ref[c].to_numpy(float)), 1e-12)))
                    for c in pipeline.FEATURES)
        same = list(mine.h5_name) == list(ref.h5_name)
        print(f"\nCUT_P_110 vs laptop nb04 feature_matrix: rows {'match' if same else 'DIFFER'}, "
              f"worst relative difference {worst:.1e} ({'reproduced' if same and worst < 1e-6 else 'CHECK'})")


if __name__ == "__main__":
    main(Path(sys.argv[1]).expanduser())
