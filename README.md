# Basin2Vec

This anonymous repository contains the implementation, evaluation code, and pretrained embedding interface for **Basin2Vec**.

Basin2Vec provides 64-dimensional basin representations indexed by a USGS site identifier and year.

The primary interface is:

```python
from basin2vec import Basin2Vec

b2v = Basin2Vec.from_pretrained(
    data_dir="review_data/basin2vec-pretrained-v1.0.0"
)

z = b2v["04101500", 2018]

print(z.shape)
# (64,)
```

## Repository structure

```text
.
├── app/
│   └── ...                         # Optional embedding exploration application
├── emb/
│   ├── src/                        # Basin2Vec training implementation
│   └── evaluation/                 # Downstream evaluation code
├── pip_release/
│   └── basin2vec/
│       ├── __init__.py             # Public package interface
│       ├── core.py                 # Basin2Vec lookup class
│       └── _download.py            # Pretrained-data loading utilities
├── scripts/
│   └── export_public_embeddings.py # Converts training outputs to public format
├── review_data/
│   └── basin2vec-pretrained-v1.0.0.tar.gz
├── pyproject.toml
├── environment.txt
└── README.md
```

## Pretrained embedding archive

The reviewer archive contains:

```text
embeddings.npy       # Shape: (9067, 40, 64)
usgs_ids.npy         # Shape: (9067,)
years.npy            # Shape: (40,)
avg_embeddings.npy   # Shape: (9067, 64)
metadata.json
```

The archive covers:

* 9,067 CONUS basins
* 40 years from 1985 through 2024
* 64-dimensional basin-year embeddings
* Basin-level embeddings averaged across all available years

The main embedding array is loaded using NumPy memory mapping. Querying one basin-year therefore does not require loading the entire array into memory.

---

# Quick start

## 1. Download the anonymous repository

Download the repository as a ZIP file or clone it using the anonymous repository link.

Enter the repository root:

```bash
cd basin2vec
```

Confirm that the expected files are present:

```bash
ls
```

The output should include:

```text
app
emb
pip_release
scripts
review_data
pyproject.toml
README.md
```

## 2. Create a clean Python environment

### Option A: Python `venv`

```bash
python -m venv .venv
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

### Option B: Conda

```bash
conda create -n basin2vec-review python=3.11 -y
conda activate basin2vec-review
```

Python 3.10 or later is recommended.

## 3. Install the Basin2Vec package

From the repository root, run:

```bash
python -m pip install .
```

For an editable installation:

```bash
python -m pip install -e .
```

Verify the installation:

```bash
python - <<'PY'
from basin2vec import Basin2Vec

print(Basin2Vec)
PY
```

Expected output:

```text
<class 'basin2vec.core.Basin2Vec'>
```

## 4. Extract the pretrained embeddings

Create the extraction directory:

```bash
mkdir -p review_data/basin2vec-pretrained-v1.0.0
```

Extract the archive:

```bash
tar -xzf review_data/basin2vec-pretrained-v1.0.0.tar.gz \
  -C review_data/basin2vec-pretrained-v1.0.0
```

Verify its contents:

```bash
ls -lh review_data/basin2vec-pretrained-v1.0.0
```

The extracted directory should contain:

```text
embeddings.npy
usgs_ids.npy
years.npy
avg_embeddings.npy
metadata.json
```

---

# Using the pretrained embeddings

## Load the embedding collection

```python
from basin2vec import Basin2Vec

b2v = Basin2Vec.from_pretrained(
    data_dir="review_data/basin2vec-pretrained-v1.0.0"
)

print(b2v)
```

Expected output:

```text
Basin2Vec(basins=9,067, years=40, dimension=64, data_version='v1.0.0')
```

## Retrieve a basin-year embedding

Use a USGS site identifier and year:

```python
z = b2v["04101500", 2018]

print(z.shape)
print(z.dtype)
```

Expected output:

```text
(64,)
float32
```

Complete command:

```bash
python - <<'PY'
from basin2vec import Basin2Vec

b2v = Basin2Vec.from_pretrained(
    data_dir="review_data/basin2vec-pretrained-v1.0.0"
)

z = b2v["04101500", 2018]

print("Collection:", b2v)
print("Embedding shape:", z.shape)
print("Embedding dtype:", z.dtype)
print("First five values:", z[:5])
PY
```

Expected first values:

```text
[-0.13140203 -0.06685294 -0.11395856  0.01976790  0.04253904]
```

Small formatting differences in the final printed decimal places are normal.

## Retrieve the basin-average embedding

When the year is omitted, Basin2Vec returns the embedding averaged across all years:

```python
z_avg = b2v["04101500"]

print(z_avg.shape)
# (64,)
```

Complete example:

```bash
python - <<'PY'
from basin2vec import Basin2Vec

b2v = Basin2Vec.from_pretrained(
    data_dir="review_data/basin2vec-pretrained-v1.0.0"
)

z_avg = b2v["04101500"]

print("Average embedding shape:", z_avg.shape)
print("Average embedding dtype:", z_avg.dtype)
print("First five values:", z_avg[:5])
PY
```

## Alternative explicit API

The indexed interface:

```python
z = b2v["04101500", 2018]
```

is equivalent to:

```python
z = b2v.get("04101500", year=2018)
```

The average embedding can be retrieved with:

```python
z_avg = b2v.get("04101500")
```

---

# Basin and year information

## Check whether a basin is available

```python
print("04101500" in b2v)
print(b2v.has_basin("04101500"))
```

Expected:

```text
True
True
```

## List available years

```python
years = b2v.available_years("04101500")

print(len(years))
print(years[0])
print(years[-1])
```

Expected:

```text
40
1985
2024
```

## Inspect all supported years

```python
print(b2v.years)
```

## Inspect all basin identifiers

```python
basin_ids = b2v.basin_ids

print("Number of basins:", len(basin_ids))
print("First five IDs:", basin_ids[:5])
```

## USGS identifier format

USGS identifiers should preferably be supplied as strings so that meaningful leading zeros are preserved:

```python
z = b2v["04101500", 2018]
```

Integer input is also supported for identifiers shorter than eight digits:

```python
z = b2v[4101500, 2018]
```

This is normalized internally to:

```text
04101500
```

Longer USGS identifiers are preserved:

```python
z = b2v["010735562", 2018]
```

---

# Environment-variable configuration

Instead of passing `data_dir` each time, configure the extracted data directory once:

```bash
export BASIN2VEC_DATA_DIR="$PWD/review_data/basin2vec-pretrained-v1.0.0"
```

The shorter interface can then be used:

```bash
python - <<'PY'
from basin2vec import Basin2Vec

b2v = Basin2Vec.from_pretrained()
z = b2v["04101500", 2018]

print(z.shape)
PY
```

Expected output:

```text
(64,)
```

To remove the environment-variable setting:

```bash
unset BASIN2VEC_DATA_DIR
```

---

# Complete reviewer smoke test

The following command checks installation, data loading, basin lookup, year lookup, embedding shape, data type, and numerical validity:

```bash
python - <<'PY'
import numpy as np

from basin2vec import Basin2Vec

data_directory = "review_data/basin2vec-pretrained-v1.0.0"

b2v = Basin2Vec.from_pretrained(
    data_dir=data_directory
)

assert len(b2v) == 9067
assert b2v.embedding_dim == 64
assert b2v.minimum_year == 1985
assert b2v.maximum_year == 2024

assert "04101500" in b2v
assert b2v.has_basin("04101500")

years = b2v.available_years("04101500")

assert len(years) == 40
assert years[0] == 1985
assert years[-1] == 2024

z = b2v["04101500", 2018]
z_avg = b2v["04101500"]

assert isinstance(z, np.ndarray)
assert isinstance(z_avg, np.ndarray)

assert z.shape == (64,)
assert z_avg.shape == (64,)

assert z.dtype == np.float32
assert z_avg.dtype == np.float32

assert np.isfinite(z).all()
assert np.isfinite(z_avg).all()

expected_prefix = np.array(
    [
        -0.13140203,
        -0.06685294,
        -0.11395856,
         0.01976790,
         0.04253904,
    ],
    dtype=np.float32,
)

assert np.allclose(
    z[:5],
    expected_prefix,
    atol=1e-6,
)

print(b2v)
print("Basin-year embedding:", z.shape, z.dtype)
print("Average embedding:", z_avg.shape, z_avg.dtype)
print("All Basin2Vec package checks passed.")
PY
```

Expected final output:

```text
Basin2Vec(basins=9,067, years=40, dimension=64, data_version='v1.0.0')
Basin-year embedding: (64,) float32
Average embedding: (64,) float32
All Basin2Vec package checks passed.
```

---

# Build the Python package

Install the build tools:

```bash
python -m pip install build twine
```

Remove previous builds:

```bash
rm -rf build dist
```

Build the source distribution and wheel:

```bash
python -m build
```

Validate the generated distributions:

```bash
python -m twine check dist/*
```

Expected output files:

```text
dist/basin2vec-0.1.0-py3-none-any.whl
dist/basin2vec-0.1.0.tar.gz
```

Inspect the wheel:

```bash
unzip -l dist/basin2vec-0.1.0-py3-none-any.whl
```

It should contain:

```text
basin2vec/__init__.py
basin2vec/core.py
basin2vec/_download.py
```

The pretrained embedding arrays should not be included inside the wheel.

## Test the built wheel

Create a clean environment:

```bash
python -m venv /tmp/basin2vec-wheel-test
source /tmp/basin2vec-wheel-test/bin/activate
```

Install the wheel:

```bash
python -m pip install \
  dist/basin2vec-0.1.0-py3-none-any.whl
```

Run the smoke test:

```bash
export BASIN2VEC_DATA_DIR="$PWD/review_data/basin2vec-pretrained-v1.0.0"

python - <<'PY'
from basin2vec import Basin2Vec

b2v = Basin2Vec.from_pretrained()
z = b2v["04101500", 2018]

print(b2v)
print(z.shape)
print(z.dtype)
PY
```

Deactivate the temporary environment afterward:

```bash
deactivate
```

---

# Exporting the public embedding format

The script:

```text
scripts/export_public_embeddings.py
```

converts the internal row-wise training archive into the public indexed format.

Expected internal keys:

```text
embeddings
labels
years
avg_embeddings
avg_labels
```

Example command:

```bash
python scripts/export_public_embeddings.py \
  --source emb/src/evaluation_outputs/HCLV4_grouped_monthly_static_meta_areaaux_64d/basin_embeddings_full.npz \
  --output review_data/basin2vec-pretrained-v1.0.0 \
  --data-version v1.0.0
```

The resulting files are:

```text
embeddings.npy
usgs_ids.npy
years.npy
avg_embeddings.npy
metadata.json
```

---

# Training and evaluation code

The main Basin2Vec training implementation is under:

```text
emb/src/
```

The downstream temporal and spatial evaluation implementation is under:

```text
emb/evaluation/
```

Generated model checkpoints, large prediction files, caches, and intermediate evaluation artifacts are intentionally excluded from the anonymous repository.

The included code provides the model definitions, dataset construction, relation-aware contrastive objective, training pipeline, embedding generation, and downstream evaluation procedures.

---

# Troubleshooting

## `ModuleNotFoundError: No module named 'basin2vec'`

Install the package from the repository root:

```bash
python -m pip install .
```

Confirm that `pyproject.toml` exists in the current directory:

```bash
ls pyproject.toml
```

## `metadata.json was not found`

Confirm that the pretrained archive was extracted:

```bash
ls review_data/basin2vec-pretrained-v1.0.0
```

The directory must contain:

```text
metadata.json
embeddings.npy
usgs_ids.npy
years.npy
avg_embeddings.npy
```

## Basin not available

Check the identifier:

```python
print(b2v.has_basin("04101500"))
```

USGS identifiers with leading zeros should be passed as strings.

## Year not available

Inspect the supported years:

```python
print(b2v.years)
```

This release contains embeddings from 1985 through 2024.

## Automatic download error

The anonymous review version uses locally supplied pretrained data. Load it explicitly:

```python
b2v = Basin2Vec.from_pretrained(
    data_dir="review_data/basin2vec-pretrained-v1.0.0"
)
```

or set:

```bash
export BASIN2VEC_DATA_DIR="$PWD/review_data/basin2vec-pretrained-v1.0.0"
```

---

# Anonymous-review note

The permanent public package index entry and non-anonymous data-hosting URL are intentionally withheld during anonymous review.

Reviewers can install the complete Python package directly from this repository:

```bash
python -m pip install .
```

and load the supplied local pretrained archive:

```python
from basin2vec import Basin2Vec

b2v = Basin2Vec.from_pretrained(
    data_dir="review_data/basin2vec-pretrained-v1.0.0"
)

z = b2v["04101500", 2018]
```

After the anonymous review period, the intended public interface will be:

```bash
pip install basin2vec
```

```python
from basin2vec import Basin2Vec

b2v = Basin2Vec.from_pretrained()
z = b2v["04101500", 2018]
```
