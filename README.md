# FRB Candidate EDA

Four notebooks characterising one Medicina observation of SGR0501+4516 with 13 injected FRBs, and working out which features are worth encoding for the quantum arms. Nothing here trains a classifier or benchmarks one - it produces the inputs to an encoding decision and stops there.

The pipeline that produced the candidates is Heimdall for the search, `your`'s candmaker for the cutouts, and FETCH for the classification. Heimdall found all 13 injections. FETCH rejected one of them, and that single failure is the most useful object in the dataset.

## The notebooks

| notebook | what it does |
|---|---|
| `Data/01_metadata_inventory.ipynb` | Parses the SIGPROC header by hand, builds ground-truth labels off a TOA cross-match rather than off anything the pipeline wrote, scores FETCH against them, and matches the chunk each candidate was cut from against the candmaker log |
| `Data/02_candidate_h5_anatomy.ipynb` | Calibrates both axes of the two `.h5` images, since neither one carries coordinates, and pins the collapse conventions for the 1-D curves |
| `Data/03_filterbank_exploration.ipynb` | Goes to the raw 2 GB filterbank - dtype, bandpass, RFI occupancy, a dedisperser of my own, and a test of nb02's axes that rebuilds the h5 images from counts on disk |
| `Data/04_feature_battery.ipynb` | Builds the feature matrix, audits it for disguises and redundancy, and turns that into transforms, connectivity and a qubit budget |

Outputs land in `frb_analysis_out/nb01` through `nb04` - CSVs, summary JSON and the figures. All of that is committed, so the results are readable without re-running anything.

The code the notebooks share lives in `frb_eda/`:

| module | what's in it |
|---|---|
| `config.py` | paths, worker count, seed, all overridable by environment variable (below) |
| `sigproc.py` | the SIGPROC header parser and the size-checked memmap |
| `physics.py` | dispersion delay and smearing, the dedisperser, the boxcar and the MAD S/N |
| `axes.py` | the h5 time and DM axes, candmaker's chunk geometry, the log parser |
| `raw.py` | the worker functions for every pass over the raw file |
| `encoding.py` | the worker functions for nb04's Fourier-degree and kernel-geometry runs |
| `parallel.py` | `pmap`, the one parallel map everything goes through |
| `provenance.py` | ground-truth labelling, package versions, input hashes |

## Configuration

Nothing in the notebooks hardcodes a path. `frb_eda/config.py` reads these, with defaults that work on a clean clone:

| variable | default | meaning |
|---|---|---|
| `FRB_ROOT` | the repository root | project root |
| `FRB_CAND_DIR` | the single `*_processed_*` folder under the root | candmaker output for one observation |
| `FRB_FIL` | `Data/<observation>.fil` | the raw filterbank |
| `FRB_INJ_CSV` | the single `Data/*_inj_params.csv` | injection ground truth |
| `FRB_OUT` | `frb_analysis_out/` | where outputs go |
| `FRB_N_JOBS` | `SLURM_CPUS_PER_TASK`, else `SLURM_NTASKS`, else every CPU | worker processes |
| `FRB_CACHE_WINDOWS` | `0` | `1` writes nb03's 3.46 GB of raw windows to disk instead of cutting them off the memmap on demand |
| `FRB_SEED` | `0` | seed for every random draw |

## Parallelism

The heavy loops are per-candidate or per-block numpy work over a 2 GB memmap, so they run in separate processes (joblib's loky backend) rather than threads, one BLAS thread per worker. Each worker opens the memmap itself. Results come back in input order, so the output doesn't depend on `FRB_N_JOBS`: the two 2 GB streaming passes, the raw-window feature pass and the h5 curves were checked against the earlier serial run value for value.

What runs in parallel: nb03's bandpass and RFI passes, the 204 dedispersions, the per-injection peak checks and the h5 rebuild; nb04's h5 curve read (processes too, since h5py runs threads one at a time), the raw-window pass, the 264 Fourier-degree fits and the 63 kernel-geometry runs. On a 12-core laptop with the file in page cache nb03 takes about 5 minutes and nb04 well under one.

## What you need to get

**One file.** Everything else is in the repo.

| file | size | goes in | what needs it |
|---|---|---|---|
| `CUT_P_110_inj_13_processed.fil` | 2,074,550,599 bytes | `Data/` | nb01 (header and a hash), nb03 (all of it), nb04 (only if its `raw_pass.npz` cache is deleted) |

It's out of the repo because GitHub rejects any file over 100 MB at push, which is a hard limit rather than a warning. Git LFS doesn't help either - the free tier is 1 GB of storage and this is 2 GB.

Worth checking the size is exactly right before trusting anything downstream. The arithmetic closes with zero slack:

    2,074,550,599 - 327 = 2,074,550,272 = 1,012,964 * 1024 * 2

so if that division doesn't come out exact then you have a truncated or different file, and `frb_eda.sigproc.open_filterbank` refuses to open it.

## What runs without the `.fil`

| notebook | runs? |
|---|---|
| nb01 | no - it needs the header, even though that's only 327 bytes |
| nb02 | **yes**, fully. The 102 `.h5`, the candmaker log and nb01's `candidates_labelled.csv` are all committed |
| nb03 | no. Its caches (`bandpass_blocks.npz`, `rfi_blocks.npz`, `nb03_dm0_raw.csv`) cover §2, §3 and the dm0 rebuild, but §4, §5 and §7 read the raw file |
| nb04 | **yes**, fully. `raw_pass.npz` and `curves.npz` are committed, so it loads from cache instead of reading the raw windows |

## What's already here

| path | what it is |
|---|---|
| `CUT_P_110_inj_13_processed_2025-1-30-13-51/` | The candmaker output - 102 `.h5` candidates, the four Heimdall `.cand` files, `all_candidates.csv`, `Results_MULTI.csv`, the 11 per-model `results_a..k.csv`, and the 16 MB candmaker log |
| `Data/CUT_P_110_inj_params.csv` | The 13 injection parameters. This is the only ground truth in the dataset |
| `Data/*.h5` | Two reference candidates, the loudest injection and one of the smearing-limited ones |
| `Data/Search_parameters.txt` | Heimdall's search configuration, including `RFI Tool: NONE` |
| `frb_analysis_out/` | Everything the notebooks produce, plus `nb01/environment.json` with package versions and the SHA-256 of the inputs |
| `tests/test_frb_eda.py` | Tests for the package. Run `python tests/test_frb_eda.py`; the ones that need the `.fil` skip themselves without it |

## Environment

Python 3.14. Third-party packages are `numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-learn`, `h5py`, `joblib`, `threadpoolctl` and `pennylane` - the last one only in nb04, and only for measuring kernel geometry rather than for training. The notebooks find `frb_eda/` by walking up from their working directory, so run them from `Data/` (the default for Jupyter) or anywhere inside the repo.

Worth noting the notebooks avoid `your` and `sigpyproc` deliberately. Parsing the header by hand means staying on numpy and pandas, getting the exact data start offset the memmap needs, and verifying the header rather than trusting `Search_parameters.txt` - which turns out to matter, since four separate families of `.h5` attributes are stale placeholders.

## A standing caveat

13 strong injections into one clean file is too easy. `snr_csv` separates the classes with AUC 1.000 and disjoint support, `members` scores 0.999, and one feature reached a perfect 1.0000 purely by being `1/S/N` in disguise. So no separability ranking in these notebooks means anything, and they're all labelled as sanity checks where they appear. The label-free half - transforms, correlation structure, dimensionality, kernel geometry - is the part that transfers. A pull across the other 20 observations is what makes the ranking question answerable.
