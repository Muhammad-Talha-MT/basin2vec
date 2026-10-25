#!/usr/bin/env python3
"""Build a self-contained, full-precision browser dataset from the research archive."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shapefile


def write_json(path, value, compressed=True):
    payload = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(payload, mtime=0) if compressed else payload)


def write_array(path, array):
    path.write_bytes(gzip.compress(np.asarray(array, dtype="<f4").tobytes(), compresslevel=6, mtime=0))


def rounded(value):
    if isinstance(value, (list, tuple)):
        return [rounded(v) for v in value]
    return round(float(value), 5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[3] / "emb/src/evaluation_outputs/HCLV4_grouped_monthly_static_meta_areaaux_64d/basin_embeddings_full.npz")
    parser.add_argument("--data-root", type=Path, default=Path("/data/basin2vec"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "dist/data")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(args.source, allow_pickle=False)
    labels = np.char.zfill(data["labels"].astype(str), 8)
    row_years = data["years"]
    ids, years = np.unique(labels), np.unique(row_years)
    n, ny = len(ids), len(years)
    order = np.lexsort((row_years, labels))
    assert len(labels) == n * ny
    assert np.array_equal(labels[order].reshape(n, ny), np.repeat(ids[:, None], ny, axis=1))
    assert np.array_equal(row_years[order].reshape(n, ny), np.repeat(years[None, :], n, axis=0))
    lookup = {s: i for i, s in enumerate(ids)}

    geometry = json.loads((args.data_root / "raw/gages-ii/us_selected_basins/basins.geojson").read_text())
    features = {}
    for f in geometry["features"]:
        bid = str(f["properties"]["GAGE_ID"]).zfill(8)
        if bid in lookup:
            g = f["geometry"]
            g["coordinates"] = rounded(g["coordinates"])
            features[bid] = {"type": "Feature", "id": lookup[bid], "properties": {"i": lookup[bid]}, "geometry": g}
    assert set(features) == set(ids), "All embedding basins must have a boundary."
    write_json(out / "basins.geojson.gz", {"type": "FeatureCollection", "features": [features[s] for s in ids]})

    reader = shapefile.Reader(str(args.data_root / "raw/gages-ii/gagesII_9322_point_shapefile/gagesII_9322_sept30_2011.shp"))
    records = {str(r.STAID).zfill(8): r.as_dict() for r in reader.iterRecords()}
    metadata = pd.read_parquet(args.data_root / "cache/metadata/basin_metadata.parquet").set_index("site_id")
    adjacency = [[] for _ in ids]
    pairs = pd.read_parquet(args.data_root / "cache/basin_overlap_pairs.parquet")
    for row in pairs.itertuples():
        a, b = lookup[str(row.site_id_a).zfill(8)], lookup[str(row.site_id_b).zfill(8)]
        score = round(float(row.overlap_score), 5)
        adjacency[a].append([b, score])
        adjacency[b].append([a, score])
    basins = []
    for i, bid in enumerate(ids):
        r, m = records[bid], metadata.loc[bid]
        basins.append({"id": bid, "name": str(r["STANAME"]).strip(), "state": r["STATE"],
                       "huc": str(r["HUC02"]).zfill(2), "area": round(float(m.area_km2), 2),
                       "lat": round(float(m.centroid_lat), 5), "lon": round(float(m.centroid_lon), 5),
                       "gauge": [float(r["LAT_GAGE"]), float(r["LNG_GAGE"])],
                       "class": r["CLASS"], "eco": r["AGGECOREGI"], "overlaps": adjacency[i]})
    spaces = {}
    for key in ["embeddings", "hydrologic_repr"]:
        print(f"Exporting {key}: {n} basins × {ny} years", flush=True)
        cube = data[key][order].reshape(n, ny, -1).astype(np.float32)
        assert np.isfinite(cube).all()
        assert np.allclose(np.linalg.norm(cube, axis=2), 1, atol=1e-5)
        dim = cube.shape[-1]
        avg = cube.mean(axis=1)
        avg /= np.linalg.norm(avg, axis=1, keepdims=True)
        directory = out / key
        directory.mkdir(exist_ok=True)
        write_array(directory / "avg.f32.gz", avg)
        for j, year in enumerate(years):
            write_array(directory / f"{year}.f32.gz", cube[:, j, :])
        # Fit one shared PCA basis per space; keep axes and colors consistent across years.
        mean = avg.mean(axis=0)
        _, singular, axes = np.linalg.svd(avg - mean, full_matrices=False)
        axes = axes[:3]
        for axis in axes:
            if axis[np.argmax(np.abs(axis))] < 0:
                axis *= -1
        pc = (avg - mean) @ axes.T
        limits = np.quantile(pc, [0.02, 0.98], axis=0).T
        trend = np.clip(np.einsum("nyd,nd->ny", cube, cube[:, 0]), -1, 1)
        change = np.clip(1 - trend[:, -1], 0, 2)
        write_array(directory / "trend.f32.gz", trend)
        spaces[key] = {"dim": dim, "mean": mean.tolist(), "axes": axes.tolist(), "limits": limits.tolist(),
                       "variance": ((singular[:3] ** 2) / (singular ** 2).sum()).tolist(),
                       "change": np.round(change, 6).tolist(), "changeMax": float(np.quantile(change, .98))}
    write_json(out / "catalog.json.gz", {"basins": basins, "years": years.tolist(), "spaces": spaces})
    provenance = {"model": "HCLV4_grouped_monthly_static_meta_areaaux_64d", "source_file": args.source.name,
                  "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(), "basins": n,
                  "years": years.tolist(), "dimensions": 64, "spaces": list(spaces), "dtype": "little-endian float32",
                  "coverage": "Contiguous United States (CONUS); Alaska, Hawaii and territories are not included.",
                  "mean": "L2-normalized mean of annual unit vectors, recomputed separately in each space.",
                  "similarity": "Cosine similarity in the selected full 64-dimensional space; no PCA used for ranking.",
                  "geometry": "GAGES-II basin boundaries, pre-simplified at 0.01 degrees with topology preservation; rounded to 5 decimal places for display. Click hit tests use these display boundaries.",
                  "overlap": "Recorded strong overlap pairs: intersection / minimum basin area in EPSG:5070. Minimum recorded score is " + str(float(pairs.overlap_score.min())) + ". Absence of a link does not establish disjointness.",
                  "pca": "PCA on period means; fixed axes across years; 2–98 percentile color stretch fitted to means.",
                  "change": "1 minus cosine between 1985 and 2024 annual vectors; representation change, not a measured physical trend."}
    write_json(out / "provenance.json", provenance, compressed=False)
    print(f"Done: {sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
