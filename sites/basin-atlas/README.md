# Basin2Vec Basin Atlas

A standalone, browser-based explorer for the real pretrained Basin2Vec dataset:
9,067 CONUS basins, 1985–2024, 64 dimensions, in both `embeddings` (projection z)
and `hydrologic_repr` (pre-projection h).

## Run locally

```bash
python -m http.server 4173 --directory dist
```

Open http://localhost:4173. There is no application server, API key, or build
step. Use an HTTP server rather than opening index.html directly: ES modules,
the web worker and fetched binary assets require HTTP. A current browser with
DecompressionStream support is required. The map and fonts have system fallbacks;
all map geometry, Leaflet code and scientific data are bundled locally.

## Features

- Every basin is drawn independently, with smaller basins above larger ones.
- Clicking checks all polygons, including holes and MultiPolygons, and offers
  every basin at the location, smallest first. Gauge points can also be enabled.
- Search by station name, USGS ID, state abbreviation, or HUC02 code.
- Full 64D cosine similarity runs in a web worker. Neighbors exclude the anchor.
- Both representation spaces have annual vectors and independently normalized
  40-year means; gzip-compressed float32 shards load on demand (about 2.1 MB each).
- Fixed PCA/RGB map, linked two-dimensional PCA plot, per-dimension inspector,
  annual similarity to 1985, and 1985–2024 representation change.
- Recorded strong-overlap relationships, optional neighbor exclusion, centroid
  distance filters, ranked CSV and vector JSON downloads, and shareable URL state.
- The basemap works without remote map tiles; no map account is needed.

## Regenerate the data

Requires Python with numpy, pandas, pyarrow and pyshp (`shapefile`).

```bash
OPENBLAS_NUM_THREADS=2 python scripts/export_data.py \
  --source ../../emb/src/evaluation_outputs/HCLV4_grouped_monthly_static_meta_areaaux_64d/basin_embeddings_full.npz \
  --data-root /data/basin2vec
```

The exporter checks the complete basin/year grid and alignment, writes all 40
annual float32 arrays for each space, recomputes means, and creates the catalog,
PCA basis, trend history, and provenance. Dataset payload is approximately 181 MB
on disk; the browser fetches only the active year/space and requested history.
No vectors are quantized or replaced with synthetic data.

## Scientific interpretation

Coverage is the contiguous US, not Alaska, Hawaii or territories. GAGES-II display
boundaries were simplified at 0.01 degrees and rounded to five decimals; map
hit-testing is consequently approximate near edges. Polygon holes are retained.

PCA is fitted separately on each representation's period means. The same axes
and 2nd–98th percentile RGB limits apply to all years within that space. Colors
are not comparable between representation spaces. Similarity rankings use all
64 dimensions and never the PCA projection. The similarity map uses a documented
5th–95th percentile color stretch; displayed/exported cosine values remain raw.

Overlap links are the original recorded strong-overlap pairs, with score
intersection area / minimum basin area, measured in EPSG:5070. The lowest recorded
score is 0.5012679346; the original generation filter is not established. An absent
link does not prove disjointness. Near-100% overlap is described as near containment,
not a strict hydrologic parent/child hierarchy. Small intersects and display-edge
differences can produce different map click candidates.

Change is 1 minus cosine between 1985 and 2024. It describes representation drift,
not a direct measurement of physical change or causation. The vector API, exports,
and map never claim hydrologic equivalence from embedding similarity alone.

## Verification

```bash
node --check dist/app.js
node --check dist/worker.js
node --test tests/math.test.js
# Optional browser suite (requires Playwright and Chromium, server running):
python tests/browser_smoke.py
```

Tests cover polygon holes, multipolygons, nested selection, geometry/ID alignment,
overlap symmetry, and independently computed real-data nearest-neighbor rankings
for both spaces and periods. `tests/browser_smoke.py` exercises the actual worker,
search, overlap chooser, filters, exports, history, representation switching,
rapid year changes, URL restore, and a narrow mobile viewport.

`dist/data/provenance.json` records the original archive SHA-256 and all data
interpretations. Leaflet 1.9.4 is BSD-2-Clause; license is in `dist/vendor/`.
State outlines are the PublicaMundi MappingAPI US states GeoJSON (source:
https://github.com/PublicaMundi/MappingAPI/blob/master/data/geojson/us-states.json).

Sites identity and deployment directory are stored in `.openai/hosting.json`.
The site is initially owner-private; publication does not change its audience.
