# FRB Candidate EDA

Four notebooks characterising one Medicina observation of SGR0501+4516 with 13 injected FRBs, and working out which features are worth encoding for the quantum arms. Nothing here trains a classifier or benchmarks one - it produces the inputs to an encoding decision and stops there.

The pipeline that produced the candidates is Heimdall for the search, `your`'s candmaker for the cutouts, and FETCH for the classification. Heimdall found all 13 injections. FETCH rejected one of them, and that single failure is the most useful object in the dataset.

## The notebooks

| notebook | what it does |
|---|---|
| `Data/01_metadata_inventory.ipynb` | Parses the SIGPROC header by hand, builds ground-truth labels off a TOA cross-match rather than off anything the pipeline wrote, and scores FETCH against them |
| `Data/02_candidate_h5_anatomy.ipynb` | Calibrates both axes of the two `.h5` images, since neither one carries coordinates, and pins the collapse conventions for the 1-D curves |
| `Data/03_filterbank_exploration.ipynb` | Goes to the raw 2 GB filterbank - dtype, bandpass, RFI occupancy, a dedisperser of my own, and an independent check of nb02's calibration against counts on disk |
| `Data/04_feature_battery.ipynb` | Builds the feature matrix, audits it for disguises and redundancy, and turns that into transforms, connectivity and a qubit budget |

Outputs land in `frb_analysis_out/nb01` through `nb04` - CSVs, summary JSON and the figures. All of that is committed, so the results are readable without re-running anything.

## What you need to get

**One file.** Everything else is in the repo.

| file | size | goes in | what needs it |
|---|---|---|---|
| `CUT_P_110_inj_13_processed.fil` | 2,074,550,599 bytes | `Data/` | nb01 (header only, 327 bytes of it) and nb03 (all of it) |

It's out of the repo because GitHub rejects any file over 100 MB at push, which is a hard limit rather than a warning. Git LFS doesn't help either - the free tier is 1 GB of storage and this is 2 GB.

Worth checking the size is exactly right before trusting anything downstream. The arithmetic closes with zero slack:

    2,074,550,599 - 327 = 2,074,550,272 = 1,012,964 * 1024 * 2

so if that division doesn't come out exact then you have a truncated or different file, and nb03 asserts on it in section 1 anyway.

The 3.3 GB of raw candidate windows under `frb_analysis_out/nb03/windows/` are also missing, but those aren't something to fetch - nb03 section 7 cuts them from the `.fil` and it's resumable, so a window that already exists at the right size gets skipped.

## What runs without the `.fil`

| notebook | runs? |
|---|---|
| nb01 | no - it needs the header, even though that's only 327 bytes |
| nb02 | **yes**, fully. The 102 `.h5` and nb01's `candidates_labelled.csv` are both committed |
| nb03 | section 6 only, which parses the candmaker log. Sections 1-5 and 7 need the raw file |
| nb04 | **yes**, fully. `raw_pass.npz` and `curves.npz` are committed, so it loads from cache instead of reading the 3.46 GB of windows |

So two of the four run end-to-end on a clean clone. The `.npz` caches are doing the work there - they're derived data that survives without the raw file, which is the whole reason they're committed rather than ignored.

## What's already here

| path | what it is |
|---|---|
| `CUT_P_110_inj_13_processed_2025-1-30-13-51/` | The candmaker output - 102 `.h5` candidates, the four Heimdall `.cand` files, `all_candidates.csv`, `Results_MULTI.csv`, the 11 per-model `results_a..k.csv`, and the 16 MB candmaker log |
| `Data/CUT_P_110_inj_params.csv` | The 13 injection parameters. This is the only ground truth in the dataset |
| `Data/*.h5` | Two reference candidates, the loudest injection and one of the smearing-limited ones |
| `Data/Search_parameters.txt` | Heimdall's search configuration, including `RFI Tool: NONE` |
| `frb_analysis_out/` | Everything the notebooks produce |

## Paths

All four notebooks hardcode the project root in their config cell:

    ROOT = Path(r"D:\QML\FRB Data")

Change that to wherever you cloned to. Everything else is relative to it, so it's one line per notebook.

## Environment

Python 3.14. Third-party packages are `numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-learn`, `h5py` and `pennylane` - the last one only in nb04, and only for measuring kernel geometry rather than for training.

Worth noting the notebooks avoid `your` and `sigpyproc` deliberately. Parsing the header by hand means staying on numpy and pandas, getting the exact data start offset that nb03's memmap needs, and verifying the header rather than trusting `Search_parameters.txt` - which turns out to matter, since four separate families of `.h5` attributes are stale placeholders.

## A standing caveat

13 strong injections into one clean file is too easy. `snr_csv` separates the classes with AUC 1.000 and disjoint support, `members` scores 0.999, and one feature reached a perfect 1.0000 purely by being `1/S/N` in disguise. So no separability ranking in these notebooks means anything, and they're all labelled as sanity checks where they appear. The label-free half - transforms, correlation structure, dimensionality, kernel geometry - is the part that transfers. A pull across the other 20 observations is what makes the ranking question answerable.
