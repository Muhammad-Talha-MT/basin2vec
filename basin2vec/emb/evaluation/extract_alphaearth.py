#!/usr/bin/env python3
"""
Parallel AlphaEarth / Google Satellite Embedding basin aggregation.

This script aggregates annual pixel-level AlphaEarth embeddings to GAGES-II
basin polygons and saves one local parquet file per basin batch and year.

Example:
    python extract_alphaearth.py \
        --basins-gpkg /data/basin2vec/raw/gages-ii/geopackage/gages_basins.gpkg \
        --index-parquet ../config/training_step5/sample_index.parquet \
        --out-dir evaluation_outputs/alphaearth_baseline \
        --scale 500 \
        --batch-size 8 \
        --workers 6

Before running in the background, authenticate Earth Engine once:
    python -c "import ee; ee.Authenticate()"
"""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm


ALPHAEARTH_COLLECTION_ID = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
AE_BANDS = [f"A{i:02d}" for i in range(64)]


def normalize_site_id(x) -> Optional[str]:
    """Normalize USGS site IDs to 8-character strings with leading zeros."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    s = re.sub(r"\D", "", s)
    if len(s) == 0:
        return None
    if len(s) <= 8:
        s = s.zfill(8)
    return s


def find_site_id_column(df: pd.DataFrame) -> str:
    candidates = [
        "site_id", "SITE_ID", "STAID", "GAGE_ID", "gage_id",
        "SOURCE_FEA", "SOURCE_FEA_ID",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        "Could not infer site_id column. Available columns:\n"
        + str(df.columns.tolist())
    )


def parse_years(years_arg: str) -> List[int]:
    """Parse '2017-2024' or '2017,2018,2020'."""
    years_arg = years_arg.strip()
    if "-" in years_arg:
        a, b = years_arg.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x.strip()) for x in years_arg.split(",") if x.strip()]


def ensure_dirs(out_dir: Path, scale: int):
    emb_dir = out_dir / "embeddings"
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    batch_dir = emb_dir / f"annual_batches_scale{scale}m_parallel"
    for d in [out_dir, emb_dir, table_dir, fig_dir, batch_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return emb_dir, table_dir, fig_dir, batch_dir


def load_target_basins(
    basins_gpkg: Path,
    index_parquet: Path,
    years: List[int],
    simplify_tolerance: Optional[float] = None,
) -> gpd.GeoDataFrame:
    if not basins_gpkg.exists():
        raise FileNotFoundError(f"Could not find basin polygons: {basins_gpkg}")
    if not index_parquet.exists():
        raise FileNotFoundError(f"Could not find sample index: {index_parquet}")

    sample_index = pd.read_parquet(index_parquet).copy()
    if "site_id" not in sample_index.columns:
        raise KeyError("sample_index must contain 'site_id'.")
    if "year" not in sample_index.columns:
        raise KeyError("sample_index must contain 'year'.")

    sample_index["site_id"] = sample_index["site_id"].apply(normalize_site_id)
    sample_index["year"] = sample_index["year"].astype(int)
    sample_index_common = sample_index[sample_index["year"].isin(years)].copy()
    target_sites = set(sample_index_common["site_id"].dropna().unique())

    basins = gpd.read_file(basins_gpkg).copy()
    site_col = find_site_id_column(basins)
    basins["site_id"] = basins[site_col].apply(normalize_site_id)
    basins = basins[basins["site_id"].isin(target_sites)].copy()
    basins = basins.drop_duplicates(subset=["site_id"]).reset_index(drop=True)

    if basins.crs is None:
        raise ValueError("Basin GeoPackage has no CRS.")

    basins_wgs84 = basins.to_crs("EPSG:4326")[["site_id", "geometry"]].copy()

    if simplify_tolerance is not None and simplify_tolerance > 0:
        basins_wgs84["geometry"] = basins_wgs84.geometry.simplify(
            simplify_tolerance, preserve_topology=True
        )

    basins_wgs84 = basins_wgs84.dropna(subset=["site_id", "geometry"]).reset_index(drop=True)
    return basins_wgs84


def gdf_batch_to_ee_fc(gdf_batch: gpd.GeoDataFrame, site_col: str = "site_id") -> ee.FeatureCollection:
    features = []
    for _, row in gdf_batch.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom is None or geom.is_empty:
            continue
        ee_geom = ee.Geometry(geom.__geo_interface__, None, False)
        features.append(ee.Feature(ee_geom, {"site_id": str(row[site_col])}))
    if len(features) == 0:
        raise ValueError("No valid geometries found in batch.")
    return ee.FeatureCollection(features)


def alphaearth_image_for_year(year: int, region_fc: Optional[ee.FeatureCollection] = None) -> ee.Image:
    year = int(year)
    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, "year")
    col = ee.ImageCollection(ALPHAEARTH_COLLECTION_ID).filterDate(start, end)
    if region_fc is not None:
        col = col.filterBounds(region_fc.geometry())
    return col.mosaic().select(AE_BANDS)


def ee_featurecollection_to_dataframe(fc: ee.FeatureCollection) -> pd.DataFrame:
    info = fc.getInfo()
    rows = [feat.get("properties", {}) for feat in info.get("features", [])]
    return pd.DataFrame(rows)


def reduce_alphaearth_for_basin_batch(
    gdf_batch: gpd.GeoDataFrame,
    year: int,
    scale: int,
    tile_scale: int,
    max_pixels_per_region: float,
) -> pd.DataFrame:
    year = int(year)
    if len(gdf_batch) == 0:
        return pd.DataFrame(columns=["site_id", "year"] + AE_BANDS)

    fc = gdf_batch_to_ee_fc(gdf_batch, site_col="site_id")
    img = alphaearth_image_for_year(year=year, region_fc=fc)

    reduced = img.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.mean(),
        scale=scale,
        tileScale=tile_scale,
        maxPixelsPerRegion=max_pixels_per_region,
    )

    df = ee_featurecollection_to_dataframe(reduced)
    if len(df) == 0:
        return pd.DataFrame(columns=["site_id", "year"] + AE_BANDS)

    df["year"] = year
    for b in AE_BANDS:
        if b not in df.columns:
            df[b] = np.nan
    df = df[["site_id", "year"] + AE_BANDS].copy()
    for b in AE_BANDS:
        df[b] = pd.to_numeric(df[b], errors="coerce")
    return df


def build_tasks(
    basins_wgs84: gpd.GeoDataFrame,
    years: List[int],
    batch_size: int,
    batch_dir: Path,
    scale: int,
) -> List[Dict]:
    n_basins = len(basins_wgs84)
    batch_starts = list(range(0, n_basins, batch_size))
    tasks = []
    for year in years:
        for start in batch_starts:
            end = min(start + batch_size, n_basins)
            out_path = batch_dir / f"alphaearth_year{year}_scale{scale}m_batch{start:05d}_{end:05d}.parquet"
            tasks.append({"year": int(year), "start": start, "end": end, "out_path": out_path})
    return tasks


def process_alphaearth_task(
    task: Dict,
    basins_wgs84: gpd.GeoDataFrame,
    scale: int,
    tile_scale: int,
    max_retries: int,
    max_pixels_per_region: float,
) -> Dict:
    year = int(task["year"])
    start = int(task["start"])
    end = int(task["end"])
    out_path = Path(task["out_path"])

    if out_path.exists():
        return {
            "status": "skipped", "year": year, "start": start, "end": end,
            "n_total": np.nan, "n_valid_all": np.nan, "n_valid_any": np.nan,
            "elapsed_sec": 0.0, "path": str(out_path), "error": "",
        }

    gdf_batch = basins_wgs84.iloc[start:end].copy()
    last_error = ""

    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            df_batch = reduce_alphaearth_for_basin_batch(
                gdf_batch=gdf_batch,
                year=year,
                scale=scale,
                tile_scale=tile_scale,
                max_pixels_per_region=max_pixels_per_region,
            )
            elapsed = time.time() - t0

            for b in AE_BANDS:
                if b not in df_batch.columns:
                    df_batch[b] = np.nan
            df_batch = df_batch[["site_id", "year"] + AE_BANDS].copy()

            n_total = len(df_batch)
            n_valid_all = int(df_batch[AE_BANDS].notna().all(axis=1).sum())
            n_valid_any = int(df_batch[AE_BANDS].notna().any(axis=1).sum())

            tmp_path = out_path.with_suffix(".tmp.parquet")
            df_batch.to_parquet(tmp_path, index=False)
            tmp_path.replace(out_path)

            return {
                "status": "done", "year": year, "start": start, "end": end,
                "n_total": int(n_total), "n_valid_all": n_valid_all,
                "n_valid_any": n_valid_any, "elapsed_sec": float(elapsed),
                "path": str(out_path), "error": "",
            }
        except Exception as e:
            last_error = f"{type(e).__name__}: {str(e)}"
            time.sleep(5 * attempt)

    return {
        "status": "failed", "year": year, "start": start, "end": end,
        "n_total": np.nan, "n_valid_all": np.nan, "n_valid_any": np.nan,
        "elapsed_sec": np.nan, "path": str(out_path), "error": last_error,
    }


def run_parallel_extraction(
    basins_wgs84: gpd.GeoDataFrame,
    tasks: List[Dict],
    workers: int,
    scale: int,
    tile_scale: int,
    max_retries: int,
    max_pixels_per_region: float,
    progress_log: Path,
    failed_log: Path,
) -> None:
    remaining_tasks = [t for t in tasks if not Path(t["out_path"]).exists()]
    print(f"Total tasks: {len(tasks)}", flush=True)
    print(f"Already completed: {len(tasks) - len(remaining_tasks)}", flush=True)
    print(f"Remaining tasks: {len(remaining_tasks)}", flush=True)
    print(f"Workers: {workers}", flush=True)

    progress_rows = []
    failed_rows = []
    start_time_all = time.time()

    if len(remaining_tasks) == 0:
        print("All tasks already completed. Nothing to do.", flush=True)
        return

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_task = {
            executor.submit(
                process_alphaearth_task,
                task,
                basins_wgs84,
                scale,
                tile_scale,
                max_retries,
                max_pixels_per_region,
            ): task
            for task in remaining_tasks
        }

        for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="AlphaEarth extraction"):
            task = future_to_task[future]
            try:
                result = future.result()
            except Exception as e:
                result = {
                    "status": "failed", "year": task["year"], "start": task["start"],
                    "end": task["end"], "n_total": np.nan, "n_valid_all": np.nan,
                    "n_valid_any": np.nan, "elapsed_sec": np.nan,
                    "path": str(task["out_path"]), "error": f"{type(e).__name__}: {str(e)}",
                }

            progress_rows.append(result)
            if result["status"] == "failed":
                failed_rows.append(result)
                print(f"\nFAILED year={result['year']} batch={result['start']}:{result['end']} error={result['error']}", flush=True)
            elif result["status"] == "done":
                if pd.notna(result["n_valid_any"]) and pd.notna(result["n_total"]) and result["n_valid_any"] < result["n_total"]:
                    print(
                        f"\nWarning year={result['year']} batch={result['start']}:{result['end']} "
                        f"valid_any={result['n_valid_any']}/{result['n_total']}, "
                        f"valid_all={result['n_valid_all']}/{result['n_total']}",
                        flush=True,
                    )

            if len(progress_rows) % 25 == 0:
                pd.DataFrame(progress_rows).to_csv(progress_log, index=False)
                if failed_rows:
                    pd.DataFrame(failed_rows).to_csv(failed_log, index=False)

    progress_df = pd.DataFrame(progress_rows)
    progress_df.to_csv(progress_log, index=False)
    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(failed_log, index=False)

    elapsed_all = time.time() - start_time_all
    print("\nFinished parallel AlphaEarth extraction.", flush=True)
    print("Elapsed minutes:", round(elapsed_all / 60, 2), flush=True)
    print("Completed new tasks:", int((progress_df["status"] == "done").sum()), flush=True)
    print("Failed tasks:", len(failed_rows), flush=True)
    print("Progress log:", progress_log, flush=True)
    if failed_rows:
        print("Failed log:", failed_log, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Parallel AlphaEarth basin embedding aggregation.")
    parser.add_argument("--basins-gpkg", type=Path, default=Path("/data/basin2vec/raw/gages-ii/geopackage/gages_basins.gpkg"))
    parser.add_argument("--index-parquet", type=Path, default=Path("../config/training_step5/sample_index.parquet"))
    parser.add_argument("--out-dir", type=Path, default=Path("evaluation_outputs/alphaearth_baseline"))
    parser.add_argument("--years", type=str, default="2017-2024")
    parser.add_argument("--scale", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--tile-scale", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-pixels-per-region", type=float, default=1e13)
    parser.add_argument("--simplify-tolerance", type=float, default=0.00005)
    parser.add_argument("--ee-project", type=str, default=None)
    parser.add_argument("--test-only", action="store_true", help="Run only first 3 incomplete tasks as a smoke test.")
    args = parser.parse_args()

    years = parse_years(args.years)
    simplify_tolerance = None if args.simplify_tolerance <= 0 else args.simplify_tolerance

    _, table_dir, _, batch_dir = ensure_dirs(args.out_dir, args.scale)
    progress_log = table_dir / f"alphaearth_progress_scale{args.scale}m_parallel.csv"
    failed_log = table_dir / f"alphaearth_failed_batches_scale{args.scale}m_parallel.csv"
    metadata_path = args.out_dir / f"alphaearth_extraction_metadata_scale{args.scale}m.json"

    print("=" * 70, flush=True)
    print("AlphaEarth basin embedding extraction", flush=True)
    print("=" * 70, flush=True)
    print("Years:", years, flush=True)
    print("Scale:", args.scale, flush=True)
    print("Batch size:", args.batch_size, flush=True)
    print("Workers:", args.workers, flush=True)
    print("Tile scale:", args.tile_scale, flush=True)
    print("Output directory:", args.out_dir, flush=True)
    print("Batch directory:", batch_dir, flush=True)

    try:
        if args.ee_project:
            ee.Initialize(project=args.ee_project)
        else:
            ee.Initialize()
        print("Earth Engine initialized.", flush=True)
    except Exception as e:
        print("Earth Engine initialization failed.", flush=True)
        print('Run once before screen: python -c "import ee; ee.Authenticate()"', flush=True)
        raise e

    basins_wgs84 = load_target_basins(
        basins_gpkg=args.basins_gpkg,
        index_parquet=args.index_parquet,
        years=years,
        simplify_tolerance=simplify_tolerance,
    )
    print("Loaded target basins:", len(basins_wgs84), flush=True)
    print("CRS:", basins_wgs84.crs, flush=True)

    basin_geojson = args.out_dir / "gages_basins_for_alphaearth_extraction.geojson"
    basins_wgs84.to_file(basin_geojson, driver="GeoJSON")

    tasks = build_tasks(
        basins_wgs84=basins_wgs84,
        years=years,
        batch_size=args.batch_size,
        batch_dir=batch_dir,
        scale=args.scale,
    )

    all_tasks = tasks
    if args.test_only:
        print("TEST ONLY: processing first 3 incomplete tasks.", flush=True)
        incomplete = [t for t in tasks if not Path(t["out_path"]).exists()]
        tasks = incomplete[:3]

    metadata = {
        "alphaearth_collection": ALPHAEARTH_COLLECTION_ID,
        "bands": AE_BANDS,
        "years": years,
        "scale": args.scale,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "tile_scale": args.tile_scale,
        "max_retries": args.max_retries,
        "max_pixels_per_region": args.max_pixels_per_region,
        "simplify_tolerance": simplify_tolerance,
        "n_basins": int(len(basins_wgs84)),
        "expected_rows": int(len(years) * len(basins_wgs84)),
        "expected_files": int(len(all_tasks)),
        "basins_gpkg": str(args.basins_gpkg),
        "index_parquet": str(args.index_parquet),
        "batch_dir": str(batch_dir),
        "progress_log": str(progress_log),
        "failed_log": str(failed_log),
        "basin_geojson": str(basin_geojson),
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print("Metadata saved:", metadata_path, flush=True)

    run_parallel_extraction(
        basins_wgs84=basins_wgs84,
        tasks=tasks,
        workers=args.workers,
        scale=args.scale,
        tile_scale=args.tile_scale,
        max_retries=args.max_retries,
        max_pixels_per_region=args.max_pixels_per_region,
        progress_log=progress_log,
        failed_log=failed_log,
    )

    batch_files = sorted(batch_dir.glob("alphaearth_year*_batch*.parquet"))
    print("\nCompleteness check", flush=True)
    print("Batch files produced:", len(batch_files), flush=True)
    print("Expected files:", len(all_tasks), flush=True)
    if batch_files:
        example = pd.read_parquet(batch_files[0])
        print("Example file:", batch_files[0], flush=True)
        print("Example shape:", example.shape, flush=True)
        print(example.head(), flush=True)
        valid_all = int(example[AE_BANDS].notna().all(axis=1).sum())
        valid_any = int(example[AE_BANDS].notna().any(axis=1).sum())
        print(f"Example valid all bands: {valid_all}/{len(example)}", flush=True)
        print(f"Example valid any band: {valid_any}/{len(example)}", flush=True)


if __name__ == "__main__":
    main()
