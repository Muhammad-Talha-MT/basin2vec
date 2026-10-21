# Basin2Vec

Basin2Vec learns 64-dimensional representations of river basins from gridded
climate, land-cover, soil, terrain and population data, using a
hydrology-aware contrastive objective. This repository contains:

* the `basin2vec` Python package for looking up pretrained embeddings;
* the training code (`emb/src/`);
* the downstream evaluation code and the results reported in the paper
  (`emb/evaluation/`);
* an optional web app for exploring the embeddings (`app/`).

Pretrained embeddings cover **9,067 CONUS basins** (GAGES-II, keyed by USGS
site ID) for every year from **1985 to 2024**.

```python
from basin2vec import Basin2Vec

b2v = Basin2Vec.from_pretrained(data_dir="data/basin2vec-pretrained-v1.0.0")
z = b2v["04101500", 2018]      # basin-year embedding, shape (64,)
z_avg = b2v["04101500"]        # basin embedding averaged over all years
```

---

## Contents

1. [Repository layout](#repository-layout)
2. [Installation](#installation)
3. [Level 1: Use the pretrained embeddings](#level-1-use-the-pretrained-embeddings)
4. [Level 2: Reproduce the downstream evaluation](#level-2-reproduce-the-downstream-evaluation)
5. [Level 3: Retrain Basin2Vec](#level-3-retrain-basin2vec)
6. [Data layout and configuration](#data-layout-and-configuration)
7. [Exploration app](#exploration-app)
8. [Tests](#tests)
9. [Known gaps](#known-gaps)

---

## Repository layout

```text
.
├── src/basin2vec/                 # Pip-installable embedding lookup package
├── tests/                         # Package and script tests (pytest)
├── scripts/
│   ├── export_public_embeddings.py   # row-wise training output -> public archive
│   └── import_public_embeddings.py   # public archive -> row-wise NPZ for evaluation
├── emb/
│   ├── config/
│   │   ├── dataset_config.yaml       # inventory of gridded input variables
│   │   └── training_step5/           # normalisation stats and sample-index metadata
│   ├── src/                          # Basin2Vec model, loss, dataset, training, evaluation
│   │   └── evaluation_outputs/       # intrinsic metrics for the main model and ablations
│   └── evaluation/
│       ├── spatial_split.py          # ungauged-basin (6-fold spatial) streamflow benchmark
│       ├── temporal_split.py         # future-years streamflow benchmark
│       ├── extract_alphaearth.py     # AlphaEarth baseline embeddings (Earth Engine)
│       ├── extract_tessera.py        # TESSERA baseline embeddings (GeoTessera)
│       ├── streamflow_regime.py      # baseflow/recharge classes from USGS streamflow
│       ├── camels_loader.py          # CAMELS-US loading helpers
│       ├── train_eval_camels_lstm.py # single-model CAMELS benchmark script
│       ├── *.ipynb                   # analysis and figure notebooks
│       ├── alphaearth_embeddings/    # baseline embeddings (included)
│       ├── satclip_embeddings/       #   "
│       ├── tessera_embeddings/       #   "
│       ├── evaluation_outputs/       # benchmark results reported in the paper
│       └── result_figures/           # paper figures and tables
├── app/                           # FastAPI + Leaflet embedding explorer
├── environment.yml                # Full research environment (conda)
├── pyproject.toml                 # Package metadata
└── LICENSE
```

---

## Installation

### Package only (Level 1)

Requires Python 3.10 or later. The package depends only on NumPy and Pooch.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

### Full research environment (Levels 2 and 3)

```bash
conda env create -f environment.yml
conda activate basin2vec
```

This installs PyTorch (≥ 2.3) from PyPI. For a particular CUDA version, install
the matching build from <https://pytorch.org/get-started/locally/> afterwards.
The environment also installs this repository's package in editable mode.

---

## Level 1: Use the pretrained embeddings

### Get the data

Download `basin2vec-pretrained-v1.0.0.tar.gz` (88 MB) from the release page
and extract it into `data/`:

```bash
mkdir -p data/basin2vec-pretrained-v1.0.0
tar -xzf basin2vec-pretrained-v1.0.0.tar.gz -C data/basin2vec-pretrained-v1.0.0
```

Check the download first if you like:

```bash
sha256sum basin2vec-pretrained-v1.0.0.tar.gz
# 28eeb48bec4f7c35705995d2a11edfcfc652fc9525077df9c1478b0754262ee4
```

The archive contains:

| File | Shape | Contents |
| --- | --- | --- |
| `embeddings.npy` | (9067, 40, 64) float32 | basin-year embeddings, unit length |
| `avg_embeddings.npy` | (9067, 64) float32 | L2-normalised mean over the 40 years |
| `usgs_ids.npy` | (9067,) | USGS site IDs as strings (sorted) |
| `years.npy` | (40,) | 1985 to 2024 |
| `metadata.json` | | version, shapes, dtypes |

Instead of passing `data_dir` every time, you can set:

```bash
export BASIN2VEC_DATA_DIR="$PWD/data/basin2vec-pretrained-v1.0.0"
```

and then call `Basin2Vec.from_pretrained()` with no arguments.

### API

```python
from basin2vec import Basin2Vec

b2v = Basin2Vec.from_pretrained(data_dir="data/basin2vec-pretrained-v1.0.0")
print(b2v)
# Basin2Vec(basins=9,067, years=40, dimension=64, data_version='v1.0.0')

z = b2v["04101500", 2018]          # same as b2v.get("04101500", year=2018)
z_avg = b2v["04101500"]            # same as b2v.get("04101500")

"04101500" in b2v                  # True
b2v.available_years("04101500")    # (1985, ..., 2024)
b2v.basin_ids                      # tuple of all USGS IDs
b2v.years, b2v.embedding_dim       # (1985, ..., 2024), 64
```

Pass USGS IDs as strings so leading zeros are kept. Integers are accepted and
zero-padded to eight digits (`4101500` becomes `"04101500"`); IDs longer than
eight digits are kept as given. Every lookup returns a new `float32` array. The
embedding file is memory-mapped, so a single lookup doesn't load all of it.

### Smoke test

```bash
python - <<'PY'
import numpy as np
from basin2vec import Basin2Vec

b2v = Basin2Vec.from_pretrained(data_dir="data/basin2vec-pretrained-v1.0.0")
z = b2v["04101500", 2018]

assert len(b2v) == 9067 and b2v.embedding_dim == 64
assert (b2v.minimum_year, b2v.maximum_year) == (1985, 2024)
assert np.allclose(
    z[:5],
    [-0.13140203, -0.06685294, -0.11395856, 0.01976790, 0.04253904],
    atol=1e-6,
)
print("All Basin2Vec package checks passed.")
PY
```

---

## Level 2: Reproduce the downstream evaluation

The streamflow benchmarks need only CAMELS-US and the embedding files; they
don't need the training data. The AlphaEarth, SatCLIP and TESSERA baseline
embeddings are included in the repository.

### 1. Get CAMELS-US

Download the CAMELS-US time series (`basin_dataset_public_v1p2`) and the
CAMELS attributes (`camels_attributes_v2.0`) from NCAR
(<https://gdex.ucar.edu/datasets/d629000/>) and arrange them like this:

```text
data/camels_us/
├── basin_mean_forcing/daymet/<huc>/<gauge>_lump_cida_forcing_leap.txt
├── usgs_streamflow/<huc>/<gauge>_streamflow_qc.txt
└── camels_attributes_v2.0/camels_{name,topo,clim,geol,hydro,soil,vege}.txt
```

If CAMELS-US is stored elsewhere, set `CAMELS_ROOT=/path/to/camels_us`.

### 2. Convert the public embeddings to the evaluation format

The benchmark scripts read one row per basin-year. Recreate that file from the
public archive:

```bash
python scripts/import_public_embeddings.py \
  --source data/basin2vec-pretrained-v1.0.0 \
  --output emb/src/evaluation_outputs/HCLV4_grouped_monthly_static_meta_areaaux_64d/basin_embeddings_full.npz
```

This is the exact inverse of `scripts/export_public_embeddings.py`; the
round trip is lossless.

### 3. Run the benchmarks

Both scripts support multiple GPUs with `torchrun` and also run on a single
GPU or CPU with plain `python`. With no options, they run the configurations
in the paper, which are recorded in `benchmark_config_summary.json` in each
`final_*_results` folder. Output goes to a separate `reproduced_*_results`
folder so you can compare it with the shipped results.

```bash
# Ungauged-basin benchmark: 6-fold spatial cross-validation on 671 CAMELS basins,
# 5 models x 5 static descriptors, 365-day input sequences.
torchrun --nproc_per_node=4 emb/evaluation/spatial_split.py
# -> emb/evaluation/evaluation_outputs/reproduced_spatial_split_results/

# Temporal benchmark: train 1990-2009, evaluate 2010-2014 on the same basins,
# 5 models x 5 static descriptors x {365, 120}-day sequences.
torchrun --nproc_per_node=4 emb/evaluation/temporal_split.py
# -> emb/evaluation/evaluation_outputs/reproduced_temporal_split_results/
```

Options to run a subset or write somewhere else:

```bash
python emb/evaluation/temporal_split.py \
  --models lstm \
  --static-sources attributes basin2vec \
  --seq-lengths 365 \
  --out-dir /tmp/temporal_lstm_only
```

| Option | Default |
| --- | --- |
| `--models` | spatial: `lstm tft timexer timerxl iTransformer`; temporal: `lstm tft timexer timerxl itransformer` (also available: `ealstm timemixer moirai`) |
| `--static-sources` | `attributes basin2vec alphaearth tessera satclip` |
| `--seq-lengths` | spatial: `365`; temporal: `365 120` |
| `--basin2vec-npz` | the main model's row-wise NPZ (step 2) |
| `--out-dir` | `emb/evaluation/evaluation_outputs/reproduced_{spatial,temporal}_split_results` |

Each experiment's random seed is built from its model's and static source's
positions in these lists. Keep the default order to reproduce the paper's
numbers exactly.

An interrupted run resumes when restarted with the same `--out-dir`:
finished experiments are skipped and unfinished ones continue from their
last checkpoint.

### Analysis notebooks

`emb/evaluation/analysis_of_embeddings.ipynb` and
`emb/evaluation/basin2vec_results_metadata.ipynb` produce the paper's figures
and tables. Start Jupyter in `emb/evaluation/` and run the first (path setup)
cell first. Each later cell is self-contained. Some cells also need
GAGES-II polygons and the cached basin metadata and overlap pairs (see
[data layout](#data-layout-and-configuration)), plus the HUC-2 Watershed
Boundary Dataset for the overview maps.

---

## Level 3: Retrain Basin2Vec

Training reads a pre-stacked Zarr cache of gridded inputs per basin and year,
plus cached basin metadata. See [Known gaps](#known-gaps) for the parts of
the preprocessing pipeline that are not yet in this repository.

```bash
# 1. Build the grouped monthly/static Zarr from the step-4 patch caches.
python emb/src/build_grouped_monthly_static_zarr.py \
  --out-dtype float16 --batch-size 64 --num-workers 16 \
  --compressor lz4 --compression-level 1

# 2. Train (main model). Checkpoints are written every 5 epochs to
#    emb/src/checkpoints/<run-name>/, TensorBoard logs to emb/src/runs/<run-name>/.
torchrun --nproc_per_node=4 emb/src/train_basin2vec_monthly_hclv4.py

# 3. Evaluate and export embeddings to emb/src/evaluation_outputs/<run-name>/.
torchrun --nproc_per_node=4 emb/src/evaluate_embeddings_monthly.py

# 4. Convert to the public archive format.
python scripts/export_public_embeddings.py \
  --source emb/src/evaluation_outputs/HCLV4_grouped_monthly_static_meta_areaaux_64d/basin_embeddings_full.npz \
  --output data/basin2vec-pretrained-v1.0.0 \
  --data-version v1.0.0
```

Main training settings: 100 epochs, AdamW (lr 2e-4, weight decay 1e-4),
24 basins × 2 years per GPU batch, 64-dimensional embeddings, seed 42.

### Ablations

Each ablation turns off one part of the objective and uses its own run name:

```bash
torchrun --nproc_per_node=4 emb/src/train_basin2vec_monthly_hclv4.py \
  --run-name HCLV4_grouped_monthly_static_meta_areaaux_64d_ablation_no_temporal \
  --temporal-weight 0

torchrun --nproc_per_node=4 emb/src/train_basin2vec_monthly_hclv4.py \
  --run-name HCLV4_grouped_monthly_static_meta_areaaux_64d_ablation_no_overlap \
  --overlap-weight 0

torchrun --nproc_per_node=4 emb/src/train_basin2vec_monthly_hclv4.py \
  --run-name HCLV4_grouped_monthly_static_meta_areaaux_64d_ablation_no_area \
  --area-aux-weight 0

# Evaluate an ablation and run a downstream benchmark on it:
torchrun --nproc_per_node=4 emb/src/evaluate_embeddings_monthly.py \
  --run-name HCLV4_grouped_monthly_static_meta_areaaux_64d_ablation_no_temporal
torchrun --nproc_per_node=4 emb/evaluation/spatial_split.py \
  --models lstm --static-sources basin2vec \
  --basin2vec-npz emb/src/evaluation_outputs/HCLV4_grouped_monthly_static_meta_areaaux_64d_ablation_no_temporal/basin_embeddings_full.npz \
  --out-dir emb/evaluation/evaluation_outputs/reproduced_ablation_spatial_no_temporal
# Shipped result: emb/evaluation/evaluation_outputs/ablation/ablation_spatial_basin2vec_no_temporal_aggregated/
```

Every checkpoint saves its loss weights (`loss_config`, `area_aux_weight`) and
data paths, so you can always check which settings produced it.

---

## Data layout and configuration

All scripts find their data through two environment variables. Paths inside
the repository are resolved from each script's location, so scripts can be
run from any directory.

| Variable | Default | Used for |
| --- | --- | --- |
| `BASIN2VEC_DATA_ROOT` | `<repo>/data` | raw inputs and training caches |
| `CAMELS_ROOT` | `$BASIN2VEC_DATA_ROOT/camels_us` | CAMELS-US benchmarks |
| `BASIN2VEC_DATA_DIR` | (package cache) | extracted pretrained archive for the `basin2vec` package |
| `BASIN2VEC_APP_NPZ` | main model's row-wise NPZ | embeddings shown by the app |

Expected contents of `$BASIN2VEC_DATA_ROOT` (`data/` is git-ignored):

```text
data/
├── basin2vec-pretrained-v1.0.0/          # Level 1: public archive
├── camels_us/                            # Level 2: CAMELS-US
├── raw/
│   ├── gages-ii/geopackage/gages_basins.gpkg          # GAGES-II basin polygons
│   ├── gages-ii/us_selected_basins/us_selected_basins.shp
│   ├── daily_streamflow_usgs/<site_id>.parquet          # streamflow_regime.py
│   └── wbd/Watershed_Boundary_Dataset_HUC_2s_*.geojson  # notebook maps
└── cache/                                # Level 3: training inputs
    ├── patches_step4/                    # annual/static variable patches (Zarr)
    ├── patches_step4_monthly/            # monthly variable patches (Zarr)
    ├── patches_step4_grouped/grouped_monthly_static.zarr
    ├── static_masks.zarr
    ├── basin_overlap_pairs.parquet
    └── metadata/basin_metadata.parquet
```

Training also reads `emb/config/training_step5/sample_index.parquet` (one row
per basin-year, 362,680 rows). Its summary is in `sample_index_meta.json`
next to it.

---

## Exploration app

```bash
python app/geojson.py      # once: simplify GAGES-II polygons into basins.geojson
cd app && uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open <http://localhost:8000>. The app reads the row-wise NPZ (step 2 of
Level 2 creates it from the public archive). To share the app publicly, a
tunnel such as `cloudflared tunnel --url http://localhost:8000` works.
Download `cloudflared` separately; it is not part of this repository.

---

## Tests

```bash
python -m pip install ".[dev]"
python -m pytest
```

The tests use small synthetic archives and run in under a second. They also
check that exporting to the public format and importing back reproduces the
original arrays exactly. GitHub Actions runs them on Python 3.10 to 3.13.

---

## Known gaps

The following parts of the pipeline are not yet in this repository:

* **Preprocessing up to step 4.** The code that builds the gridded per-basin
  patch caches (`cache/patches_step4*`), `static_masks.zarr`,
  `basin_overlap_pairs.parquet`, `metadata/basin_metadata.parquet` and
  `emb/config/training_step5/sample_index.parquet` from raw sources.
* **`basin2vec_dataset_monthly.py`**, which `build_grouped_monthly_static_zarr.py`
  imports. Without it, the build script stops with an explicit error.
* **Converting baseline extraction output to NPZ.** `extract_alphaearth.py` and
  `extract_tessera.py` write per-batch parquet files; the step that merged
  them into the shipped `*_embeddings.npz` files is not included.

Levels 1 and 2 don't depend on any of these.

---

## License

MIT; see [LICENSE](LICENSE).
