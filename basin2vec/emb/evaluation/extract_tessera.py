#!/usr/bin/env python3
"""
Parallel TESSERA basin embedding aggregation using GeoTessera point sampling.

This script is designed as the TESSERA analogue of the AlphaEarth extraction
script used in Basin2Vec evaluation. It aggregates TESSERA point/pixel
embeddings to GAGES-II basin polygons and saves one parquet file per
basin batch and year.

Main difference from AlphaEarth:
    - AlphaEarth uses Google Earth Engine reduceRegions over image bands.
    - TESSERA is accessed through the public GeoTessera Python API, so we
      sample TESSERA embeddings at an equal-area grid of points inside each
      basin and average those 128-D vectors locally.

Recommended first smoke test:
    python extract_tessera_basins_fast.py \
        --basins-gpkg /data/basin2vec/raw/gages-ii/geopackage/gages_basins.gpkg \
        --index-parquet ../config/training_step5/sample_index.parquet \
        --out-dir evaluation_outputs/tessera_baseline \
        --years 2024 \
        --scale 1000 \
        --sampling-mode adaptive \
        --max-points-per-basin 512 \
        --batch-size 32 \
        --workers 2 \
        --test-only

Full run, once coverage is confirmed:
    python extract_tessera_basins_fast.py \
        --basins-gpkg /data/basin2vec/raw/gages-ii/geopackage/gages_basins.gpkg \
        --index-parquet ../config/training_step5/sample_index.parquet \
        --out-dir evaluation_outputs/tessera_baseline \
        --years 2017-2024 \
        --scale 1000 \
        --sampling-mode adaptive \
        --max-points-per-basin 512 \
        --batch-size 32 \
        --workers 2
python extract_tessera.py   --basins-gpkg /data/basin2vec/raw/gages-ii/geopackage/gages_basins.gpkg   --index-parquet /home/talhamuh/basin2vec/emb/config/training_step5/sample_index.parquet   --out-dir /data/tessera/basin_outputs/tessera_embeddings   --years 2024   --scale 3000   --sampling-mode adaptive   --max-points-per-basin 128   --min-points-per-basin 8   --batch-size 64   --workers 2   --query-chunk-size 32768   --dataset-version v1.1   --dataset-variant cambridge   --cache-dir /data/tessera/geotessera_cache   --embeddings-dir /data/tessera

Install:
    pip install geotessera geopandas pyarrow shapely pyproj tqdm pandas numpy

Notes:
    1. TESSERA availability changes as new years/regions are added. The script
       checks available years at runtime.
    2. --scale controls the approximate spacing used to set an adaptive
       point budget. Larger scale means fewer sampled points.
    3. The fast default uses adaptive/random sampling with a per-basin cap;
       this is much faster than a dense equal-area grid for all GAGES basins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point
from shapely.ops import transform as shapely_transform
from shapely.prepared import prep
from tqdm import tqdm


TESSERA_DIM = 128
TESSERA_BANDS = [f"T{i:03d}" for i in range(TESSERA_DIM)]

_THREAD_LOCAL = threading.local()


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


def ensure_dirs(
    out_dir: Path,
    scale: int,
    sampling_mode: str = "adaptive",
    max_points_per_basin: int = 512,
):
    emb_dir = out_dir / "embeddings"
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    tag = f"scale{scale}m_{sampling_mode}_max{max_points_per_basin}"
    batch_dir = emb_dir / f"annual_batches_{tag}"
    point_dir = out_dir / f"point_cache_{tag}"
    for d in [out_dir, emb_dir, table_dir, fig_dir, batch_dir, point_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return emb_dir, table_dir, fig_dir, batch_dir, point_dir


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


def _client_cache_key(
    dataset_version: Optional[str],
    dataset_variant: Optional[str],
    use_zarr: bool,
    cache_dir: Optional[str],
    embeddings_dir: Optional[str],
) -> str:
    return (
        f"gt__version={dataset_version}__variant={dataset_variant}__zarr={use_zarr}"
        f"__cache={cache_dir}__embdir={embeddings_dir}"
    )


def get_tessera_client(
    dataset_version: Optional[str] = None,
    dataset_variant: Optional[str] = None,
    use_zarr: bool = False,
    cache_dir: Optional[str] = None,
    embeddings_dir: Optional[str] = None,
):
    """
    Thread-local GeoTessera client.

    By default this uses the public high-level GeoTessera API because the
    documented method for point sampling is:

        GeoTessera(...).sample_embeddings_at_points(points, year=2024)

    The older/lower-level GeoTesseraZarr path can still be enabled with
    --use-zarr, but the high-level API is safer across GeoTessera releases.
    """
    key = _client_cache_key(dataset_version, dataset_variant, use_zarr, cache_dir, embeddings_dir)
    cache = getattr(_THREAD_LOCAL, "gt_cache", None)
    if cache is None:
        cache = {}
        _THREAD_LOCAL.gt_cache = cache
    if key in cache:
        return cache[key]

    kwargs = {}
    if dataset_version:
        kwargs["dataset_version"] = dataset_version
    if dataset_variant:
        kwargs["dataset_variant"] = dataset_variant
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if embeddings_dir:
        # Newer GeoTessera supports embeddings_dir for local tile reuse.
        kwargs["embeddings_dir"] = embeddings_dir

    if use_zarr:
        try:
            from geotessera.store import GeoTesseraZarr
        except Exception as e:
            raise ImportError(
                "Could not import GeoTesseraZarr. Install/upgrade with: pip install -U geotessera"
            ) from e
        cls = GeoTesseraZarr
    else:
        try:
            from geotessera import GeoTessera
        except Exception as e:
            raise ImportError(
                "Could not import GeoTessera. Install/upgrade with: pip install -U geotessera"
            ) from e
        cls = GeoTessera

    try:
        gt = cls(**kwargs)
    except TypeError:
        # Older GeoTessera releases may not support dataset_version/variant/cache_dir/embeddings_dir.
        # Fall back to defaults rather than crashing, but metadata will record the request.
        gt = cls()

    cache[key] = gt
    return gt


def get_available_tessera_years(
    dataset_version: Optional[str] = None,
    dataset_variant: Optional[str] = None,
    use_zarr: bool = False,
    cache_dir: Optional[str] = None,
    embeddings_dir: Optional[str] = None,
) -> List[int]:
    gt = get_tessera_client(
        dataset_version=dataset_version,
        dataset_variant=dataset_variant,
        use_zarr=use_zarr,
        cache_dir=cache_dir,
        embeddings_dir=embeddings_dir,
    )

    # High-level GeoTessera API
    registry = getattr(gt, "registry", None)
    if registry is not None and hasattr(registry, "get_available_years"):
        try:
            return sorted([int(y) for y in registry.get_available_years()])
        except Exception:
            pass

    # Lower-level / older APIs
    years = getattr(gt, "years", None)
    if years is not None:
        try:
            return sorted([int(y) for y in years])
        except Exception:
            pass

    return []


def stable_seed(*parts: object) -> int:
    text = "::".join(str(p) for p in parts)
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _contains_xy_fallback(geom, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    Vectorized point-in-polygon test when shapely.contains_xy is unavailable.
    """
    try:
        from shapely import contains_xy  # shapely >= 2
        return np.asarray(contains_xy(geom, xs, ys), dtype=bool)
    except Exception:
        pgeom = prep(geom)
        return np.asarray([pgeom.contains(Point(float(x), float(y))) for x, y in zip(xs, ys)], dtype=bool)


def _random_points_in_polygon(
    geom,
    rng: np.random.Generator,
    n_points: int,
    max_attempts: int = 80,
) -> np.ndarray:
    """Fast deterministic rejection sampling inside a projected polygon."""
    minx, miny, maxx, maxy = geom.bounds
    n_points = int(max(0, n_points))
    if n_points <= 0:
        return np.empty((0, 2), dtype=np.float64)

    pts_list = []
    pgeom = prep(geom)
    # Oversample because basin bounding boxes often include space outside polygon.
    batch = min(max(2048, n_points * 4), 100000)
    for _ in range(max_attempts):
        cand_x = rng.uniform(minx, maxx, size=batch)
        cand_y = rng.uniform(miny, maxy, size=batch)
        inside = np.asarray(
            [pgeom.contains(Point(float(x), float(y))) for x, y in zip(cand_x, cand_y)],
            dtype=bool,
        )
        if inside.any():
            pts_list.append(np.column_stack([cand_x[inside], cand_y[inside]]))
            if sum(len(p) for p in pts_list) >= n_points:
                break

    if not pts_list:
        return np.empty((0, 2), dtype=np.float64)
    pts = np.vstack(pts_list)
    return pts[:n_points].astype(np.float64, copy=False)


def points_in_polygon_sampled(
    geom,
    site_id: str,
    spacing_m: float,
    max_points: int,
    max_candidate_grid_points: int,
    sampling_mode: str = "adaptive",
    min_points: int = 8,
) -> np.ndarray:
    """
    Return Nx2 point coordinates in projected meter units.

    Modes:
      representative: 1 point inside polygon, fastest.
      random: exactly max_points sampled points, unless polygon is tiny.
      adaptive: min(max_points, area / spacing_m^2) random points.
      grid: original systematic grid, capped by max_points.
    """
    if geom is None or geom.is_empty:
        return np.empty((0, 2), dtype=np.float64)
    if not geom.is_valid:
        geom = geom.buffer(0)
    if geom is None or geom.is_empty:
        return np.empty((0, 2), dtype=np.float64)

    minx, miny, maxx, maxy = geom.bounds
    if not np.isfinite([minx, miny, maxx, maxy]).all():
        return np.empty((0, 2), dtype=np.float64)

    sampling_mode = str(sampling_mode).lower()
    spacing_m = float(spacing_m)
    max_points = int(max_points)
    min_points = int(max(1, min_points))
    max_candidate_grid_points = int(max_candidate_grid_points)

    rng = np.random.default_rng(stable_seed(site_id, spacing_m, max_points, sampling_mode))

    if sampling_mode in {"representative", "centroid"}:
        rp = geom.representative_point()
        return np.asarray([[rp.x, rp.y]], dtype=np.float64)

    area = float(getattr(geom, "area", 0.0) or 0.0)
    if sampling_mode == "adaptive":
        target = int(math.ceil(area / max(spacing_m * spacing_m, 1.0)))
        target = max(min_points, min(max_points, target))
        pts = _random_points_in_polygon(geom, rng, target)
    elif sampling_mode == "random":
        pts = _random_points_in_polygon(geom, rng, max_points)
    elif sampling_mode == "grid":
        nx = max(1, int(math.ceil((maxx - minx) / spacing_m)))
        ny = max(1, int(math.ceil((maxy - miny) / spacing_m)))
        n_candidates = nx * ny
        if n_candidates <= max_candidate_grid_points:
            xs = minx + (np.arange(nx, dtype=np.float64) + 0.5) * spacing_m
            ys = miny + (np.arange(ny, dtype=np.float64) + 0.5) * spacing_m
            xx, yy = np.meshgrid(xs, ys)
            flat_x = xx.ravel()
            flat_y = yy.ravel()
            inside = _contains_xy_fallback(geom, flat_x, flat_y)
            pts = np.column_stack([flat_x[inside], flat_y[inside]])
            if len(pts) > max_points:
                idx = rng.choice(len(pts), size=max_points, replace=False)
                pts = pts[idx]
        else:
            pts = _random_points_in_polygon(geom, rng, max_points)
    else:
        raise ValueError(f"Unknown sampling_mode={sampling_mode!r}")

    if len(pts) < min_points:
        rp = geom.representative_point()
        if len(pts) == 0:
            pts = np.asarray([[rp.x, rp.y]], dtype=np.float64)
        else:
            pts = np.vstack([pts, np.asarray([[rp.x, rp.y]], dtype=np.float64)])
    return pts.astype(np.float64, copy=False)


def generate_basin_points_wgs84(
    gdf_batch_wgs84: gpd.GeoDataFrame,
    year: int,
    scale_m: int,
    grid_crs: str,
    max_points_per_basin: int,
    max_candidate_grid_points: int,
    sampling_mode: str,
    min_points_per_basin: int,
    point_cache_dir: Optional[Path] = None,
    use_point_cache: bool = True,
) -> Tuple[List[Tuple[float, float]], List[Dict]]:
    """
    Generate WGS84 lon/lat sample points for each basin in the batch.

    Returns:
        all_points_lonlat: list of (lon, lat)
        point_meta: list of dicts with site_id, start, end, n_points
    """
    point_cache_path = None
    site_key = f"{gdf_batch_wgs84.iloc[0]['site_id']}_{gdf_batch_wgs84.iloc[-1]['site_id']}"
    if point_cache_dir is not None:
        point_cache_path = point_cache_dir / (
            f"points_scale{scale_m}m_{grid_crs.replace(':', '')}_{sampling_mode}_max{max_points_per_basin}_min{min_points_per_basin}_{site_key}.npz"
        )

    if use_point_cache and point_cache_path is not None and point_cache_path.exists():
        z = np.load(point_cache_path, allow_pickle=True)
        points_arr = z["points"]
        meta = z["meta"].tolist()
        all_points = [(float(x), float(y)) for x, y in points_arr]
        return all_points, meta

    gdf_eq = gdf_batch_wgs84.to_crs(grid_crs).copy()
    transformer = Transformer.from_crs(grid_crs, "EPSG:4326", always_xy=True)

    all_points: List[Tuple[float, float]] = []
    point_meta: List[Dict] = []

    for _, row in gdf_eq.iterrows():
        site_id = str(row["site_id"])
        pts_xy = points_in_polygon_sampled(
            geom=row.geometry,
            site_id=site_id,
            spacing_m=float(scale_m),
            max_points=max_points_per_basin,
            max_candidate_grid_points=max_candidate_grid_points,
            sampling_mode=sampling_mode,
            min_points=min_points_per_basin,
        )

        start = len(all_points)
        if len(pts_xy) > 0:
            lons, lats = transformer.transform(pts_xy[:, 0], pts_xy[:, 1])
            all_points.extend([(float(lon), float(lat)) for lon, lat in zip(lons, lats)])
        end = len(all_points)

        point_meta.append(
            {
                "site_id": site_id,
                "start": int(start),
                "end": int(end),
                "n_points": int(end - start),
            }
        )

    if use_point_cache and point_cache_path is not None:
        point_cache_path.parent.mkdir(parents=True, exist_ok=True)
        arr = np.asarray(all_points, dtype=np.float64)
        meta_arr = np.asarray(point_meta, dtype=object)
        tmp_path = point_cache_path.with_suffix(".tmp.npz")
        # Uncompressed cache is faster than compressed for repeated extraction runs.
        np.savez(tmp_path, points=arr, meta=meta_arr)
        tmp_path.replace(point_cache_path)

    return all_points, point_meta


def iter_chunks(seq: Sequence, chunk_size: int) -> Iterable[Sequence]:
    for i in range(0, len(seq), chunk_size):
        yield seq[i : i + chunk_size]


def _coerce_tessera_sample_output(x, n_expected: int) -> np.ndarray:
    """Convert GeoTessera point-sampling output to an (N, 128) float32 array."""
    if isinstance(x, pd.DataFrame):
        # GeoAI wrappers may return columns tessera_0 ... tessera_127.
        cols = [f"tessera_{i}" for i in range(TESSERA_DIM)]
        if all(c in x.columns for c in cols):
            arr = x[cols].to_numpy(dtype=np.float32)
        else:
            # Fallback: take the last 128 numeric columns.
            num = x.select_dtypes(include=[np.number])
            if num.shape[1] < TESSERA_DIM:
                raise ValueError(f"DataFrame output has only {num.shape[1]} numeric columns; expected at least {TESSERA_DIM}.")
            arr = num.iloc[:, -TESSERA_DIM:].to_numpy(dtype=np.float32)
    else:
        arr = np.asarray(x, dtype=np.float32)

    if arr.ndim != 2:
        raise ValueError(f"Expected point-sampling output with ndim=2, got shape={arr.shape}")
    if arr.shape[1] != TESSERA_DIM:
        raise ValueError(f"Expected {TESSERA_DIM} dims, got shape={arr.shape}")
    if arr.shape[0] != n_expected:
        raise ValueError(f"Expected {n_expected} rows, got shape={arr.shape}")
    return arr


def sample_tessera_points(
    points_lonlat: List[Tuple[float, float]],
    year: int,
    query_chunk_size: int,
    dataset_version: Optional[str],
    dataset_variant: Optional[str],
    use_zarr: bool,
    cache_dir: Optional[str],
    embeddings_dir: Optional[str],
    sort_points: bool,
) -> np.ndarray:
    """
    Sample TESSERA embeddings at WGS84 lon/lat points.

    Returns an array with shape (N, 128), filled with NaN for failed chunks.
    """
    n = len(points_lonlat)
    if n == 0:
        return np.empty((0, TESSERA_DIM), dtype=np.float32)

    gt = get_tessera_client(
        dataset_version=dataset_version,
        dataset_variant=dataset_variant,
        use_zarr=use_zarr,
        cache_dir=cache_dir,
        embeddings_dir=embeddings_dir,
    )
    out = np.full((n, TESSERA_DIM), np.nan, dtype=np.float32)

    # Prefer the documented high-level API. Keep lower-level fallbacks for
    # compatibility with different GeoTessera releases.
    if hasattr(gt, "sample_embeddings_at_points"):
        sampler_name = "sample_embeddings_at_points"
    elif hasattr(gt, "sample_points"):
        sampler_name = "sample_points"
    else:
        raise AttributeError(
            "GeoTessera client has neither sample_embeddings_at_points nor sample_points. "
            "Upgrade with: pip install -U geotessera"
        )
    sampler = getattr(gt, sampler_name)

    # Sorting improves cache locality and keeps chunks spatially coherent.
    # GeoTessera also groups by tile internally in newer releases, but sorting
    # still avoids sending scattered points in each chunk.
    if sort_points and n > 1:
        pts_arr = np.asarray(points_lonlat, dtype=np.float64)
        order = np.lexsort((pts_arr[:, 1], pts_arr[:, 0]))
        points_for_sampling = [points_lonlat[int(i)] for i in order]
        out_sorted = np.full((n, TESSERA_DIM), np.nan, dtype=np.float32)
    else:
        order = None
        points_for_sampling = points_lonlat
        out_sorted = out

    offset = 0
    for chunk in iter_chunks(points_for_sampling, query_chunk_size):
        chunk = list(chunk)
        try:
            try:
                # New GeoTessera supports overriding local embedding storage here.
                x = sampler(chunk, year=int(year), embeddings_dir=embeddings_dir) if embeddings_dir else sampler(chunk, year=int(year))
            except TypeError:
                x = sampler(chunk, year=int(year))
            x = _coerce_tessera_sample_output(x, n_expected=len(chunk))
            out_sorted[offset : offset + len(chunk), :] = x
        except Exception as e:
            print(
                f"Warning: TESSERA {sampler_name} failed for year={year}, "
                f"points {offset}:{offset+len(chunk)}. Error: {type(e).__name__}: {e}",
                flush=True,
            )
        offset += len(chunk)

    if order is not None:
        out[order, :] = out_sorted
    return out


def reduce_tessera_for_basin_batch(
    gdf_batch_wgs84: gpd.GeoDataFrame,
    year: int,
    scale_m: int,
    grid_crs: str,
    max_points_per_basin: int,
    max_candidate_grid_points: int,
    sampling_mode: str,
    min_points_per_basin: int,
    query_chunk_size: int,
    point_cache_dir: Path,
    use_point_cache: bool,
    dataset_version: Optional[str],
    dataset_variant: Optional[str],
    use_zarr: bool,
    cache_dir: Optional[str],
    embeddings_dir: Optional[str],
    sort_points: bool,
) -> pd.DataFrame:
    """
    Aggregate TESSERA point embeddings to one mean vector per basin.
    """
    if len(gdf_batch_wgs84) == 0:
        return pd.DataFrame(columns=["site_id", "year", "n_points", "n_valid_points"] + TESSERA_BANDS)

    points_lonlat, point_meta = generate_basin_points_wgs84(
        gdf_batch_wgs84=gdf_batch_wgs84,
        year=year,
        scale_m=scale_m,
        grid_crs=grid_crs,
        max_points_per_basin=max_points_per_basin,
        max_candidate_grid_points=max_candidate_grid_points,
        sampling_mode=sampling_mode,
        min_points_per_basin=min_points_per_basin,
        point_cache_dir=point_cache_dir,
        use_point_cache=use_point_cache,
    )

    sampled = sample_tessera_points(
        points_lonlat=points_lonlat,
        year=year,
        query_chunk_size=query_chunk_size,
        dataset_version=dataset_version,
        dataset_variant=dataset_variant,
        use_zarr=use_zarr,
        cache_dir=cache_dir,
        embeddings_dir=embeddings_dir,
        sort_points=sort_points,
    )

    rows = []
    for meta in point_meta:
        site_id = meta["site_id"]
        start = int(meta["start"])
        end = int(meta["end"])
        n_points = int(meta["n_points"])

        vec = np.full(TESSERA_DIM, np.nan, dtype=np.float32)
        n_valid_points = 0

        if end > start:
            vals = sampled[start:end, :]
            valid = np.isfinite(vals).all(axis=1)
            n_valid_points = int(valid.sum())
            if n_valid_points > 0:
                vec = np.nanmean(vals[valid], axis=0).astype(np.float32)

        row = {"site_id": site_id, "year": int(year), "n_points": n_points, "n_valid_points": n_valid_points}
        row.update({b: float(vec[i]) if np.isfinite(vec[i]) else np.nan for i, b in enumerate(TESSERA_BANDS)})
        rows.append(row)

    df = pd.DataFrame(rows)
    for b in TESSERA_BANDS:
        if b not in df.columns:
            df[b] = np.nan
        df[b] = pd.to_numeric(df[b], errors="coerce")
    return df[["site_id", "year", "n_points", "n_valid_points"] + TESSERA_BANDS].copy()


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
            out_path = batch_dir / f"tessera_year{year}_scale{scale}m_batch{start:05d}_{end:05d}.parquet"
            tasks.append({"year": int(year), "start": start, "end": end, "out_path": out_path})
    return tasks


def process_tessera_task(
    task: Dict,
    basins_wgs84: gpd.GeoDataFrame,
    scale: int,
    grid_crs: str,
    max_points_per_basin: int,
    max_candidate_grid_points: int,
    sampling_mode: str,
    min_points_per_basin: int,
    query_chunk_size: int,
    point_cache_dir: Path,
    use_point_cache: bool,
    max_retries: int,
    dataset_version: Optional[str],
    dataset_variant: Optional[str],
    use_zarr: bool,
    cache_dir: Optional[str],
    embeddings_dir: Optional[str],
    sort_points: bool,
) -> Dict:
    year = int(task["year"])
    start = int(task["start"])
    end = int(task["end"])
    out_path = Path(task["out_path"])

    if out_path.exists():
        return {
            "status": "skipped", "year": year, "start": start, "end": end,
            "n_total": np.nan, "n_valid_all": np.nan, "n_valid_any": np.nan,
            "n_points": np.nan, "n_valid_points": np.nan,
            "elapsed_sec": 0.0, "path": str(out_path), "error": "",
        }

    gdf_batch = basins_wgs84.iloc[start:end].copy()
    last_error = ""

    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            df_batch = reduce_tessera_for_basin_batch(
                gdf_batch_wgs84=gdf_batch,
                year=year,
                scale_m=scale,
                grid_crs=grid_crs,
                max_points_per_basin=max_points_per_basin,
                max_candidate_grid_points=max_candidate_grid_points,
                sampling_mode=sampling_mode,
                min_points_per_basin=min_points_per_basin,
                query_chunk_size=query_chunk_size,
                point_cache_dir=point_cache_dir,
                use_point_cache=use_point_cache,
                dataset_version=dataset_version,
                dataset_variant=dataset_variant,
                use_zarr=use_zarr,
                cache_dir=cache_dir,
                embeddings_dir=embeddings_dir,
                sort_points=sort_points,
            )
            elapsed = time.time() - t0

            for b in TESSERA_BANDS:
                if b not in df_batch.columns:
                    df_batch[b] = np.nan
            df_batch = df_batch[["site_id", "year", "n_points", "n_valid_points"] + TESSERA_BANDS].copy()

            n_total = len(df_batch)
            n_valid_all = int(df_batch[TESSERA_BANDS].notna().all(axis=1).sum())
            n_valid_any = int(df_batch[TESSERA_BANDS].notna().any(axis=1).sum())
            n_points = int(df_batch["n_points"].sum()) if "n_points" in df_batch.columns else 0
            n_valid_points = int(df_batch["n_valid_points"].sum()) if "n_valid_points" in df_batch.columns else 0

            # Fail fast instead of silently writing all-NaN parquet files. For
            # CONUS 2024, zero valid samples almost always means a GeoTessera
            # API/version/coverage problem rather than a basin geometry problem.
            if n_points > 0 and n_valid_points == 0:
                raise RuntimeError(
                    "GeoTessera returned zero valid embeddings for this batch. "
                    "Check sample_embeddings_at_points, dataset_version/dataset_variant, "
                    "coverage, and whether previously all-NaN parquet outputs should be deleted."
                )

            tmp_path = out_path.with_suffix(".tmp.parquet")
            df_batch.to_parquet(tmp_path, index=False)
            tmp_path.replace(out_path)

            return {
                "status": "done", "year": year, "start": start, "end": end,
                "n_total": int(n_total), "n_valid_all": n_valid_all,
                "n_valid_any": n_valid_any,
                "n_points": n_points, "n_valid_points": n_valid_points,
                "elapsed_sec": float(elapsed), "path": str(out_path), "error": "",
            }

        except Exception as e:
            last_error = f"{type(e).__name__}: {str(e)}"
            time.sleep(5 * attempt)

    return {
        "status": "failed", "year": year, "start": start, "end": end,
        "n_total": np.nan, "n_valid_all": np.nan, "n_valid_any": np.nan,
        "n_points": np.nan, "n_valid_points": np.nan,
        "elapsed_sec": np.nan, "path": str(out_path), "error": last_error,
    }


def run_parallel_extraction(
    basins_wgs84: gpd.GeoDataFrame,
    tasks: List[Dict],
    workers: int,
    scale: int,
    grid_crs: str,
    max_points_per_basin: int,
    max_candidate_grid_points: int,
    sampling_mode: str,
    min_points_per_basin: int,
    query_chunk_size: int,
    point_cache_dir: Path,
    use_point_cache: bool,
    max_retries: int,
    progress_log: Path,
    failed_log: Path,
    dataset_version: Optional[str],
    dataset_variant: Optional[str],
    use_zarr: bool,
    cache_dir: Optional[str],
    embeddings_dir: Optional[str],
    sort_points: bool,
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
                process_tessera_task,
                task,
                basins_wgs84,
                scale,
                grid_crs,
                max_points_per_basin,
                max_candidate_grid_points,
                sampling_mode,
                min_points_per_basin,
                query_chunk_size,
                point_cache_dir,
                use_point_cache,
                max_retries,
                dataset_version,
                dataset_variant,
                use_zarr,
                cache_dir,
                embeddings_dir,
                sort_points,
            ): task
            for task in remaining_tasks
        }

        for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="TESSERA extraction"):
            task = future_to_task[future]
            try:
                result = future.result()
            except Exception as e:
                result = {
                    "status": "failed", "year": task["year"], "start": task["start"],
                    "end": task["end"], "n_total": np.nan, "n_valid_all": np.nan,
                    "n_valid_any": np.nan, "n_points": np.nan, "n_valid_points": np.nan,
                    "elapsed_sec": np.nan,
                    "path": str(task["out_path"]), "error": f"{type(e).__name__}: {str(e)}",
                }

            progress_rows.append(result)
            if result["status"] == "failed":
                failed_rows.append(result)
                print(
                    f"\nFAILED year={result['year']} batch={result['start']}:{result['end']} "
                    f"error={result['error']}",
                    flush=True,
                )
            elif result["status"] == "done":
                if pd.notna(result["n_valid_any"]) and pd.notna(result["n_total"]) and result["n_valid_any"] < result["n_total"]:
                    print(
                        f"\nWarning year={result['year']} batch={result['start']}:{result['end']} "
                        f"valid_any={result['n_valid_any']}/{result['n_total']}, "
                        f"valid_all={result['n_valid_all']}/{result['n_total']}, "
                        f"valid_points={result['n_valid_points']}/{result['n_points']}",
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
    print("\nFinished parallel TESSERA extraction.", flush=True)
    print("Elapsed minutes:", round(elapsed_all / 60, 2), flush=True)
    print("Completed new tasks:", int((progress_df["status"] == "done").sum()), flush=True)
    print("Failed tasks:", len(failed_rows), flush=True)
    print("Progress log:", progress_log, flush=True)
    if failed_rows:
        print("Failed log:", failed_log, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Parallel TESSERA basin embedding aggregation.")
    parser.add_argument("--basins-gpkg", type=Path, default=Path("/data/basin2vec/raw/gages-ii/geopackage/gages_basins.gpkg"))
    parser.add_argument("--index-parquet", type=Path, default=Path("../config/training_step5/sample_index.parquet"))
    parser.add_argument("--out-dir", type=Path, default=Path("evaluation_outputs/tessera_baseline"))
    parser.add_argument("--years", type=str, default="2017-2024")
    parser.add_argument("--scale", type=int, default=1000, help="Approximate point spacing in meters for adaptive/grid sampling.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--simplify-tolerance", type=float, default=0.00005)
    parser.add_argument("--grid-crs", type=str, default="EPSG:5070", help="Projected CRS for equal-area point grid.")
    parser.add_argument("--max-points-per-basin", type=int, default=512)
    parser.add_argument("--max-candidate-grid-points", type=int, default=250000)
    parser.add_argument("--sampling-mode", type=str, default="adaptive", choices=["adaptive", "random", "grid", "representative", "centroid"])
    parser.add_argument("--min-points-per-basin", type=int, default=8)
    parser.add_argument("--query-chunk-size", type=int, default=16384)
    parser.add_argument("--skip-unavailable-years", action="store_true")
    parser.add_argument("--dataset-version", type=str, default="v1.1", help="GeoTessera dataset version, e.g. v1 or v1.1.")
    parser.add_argument("--dataset-variant", type=str, default="cambridge", help="GeoTessera dataset variant, e.g. vultr or cambridge.")
    parser.add_argument("--use-zarr", action="store_true", help="Use lower-level GeoTesseraZarr instead of high-level GeoTessera.")
    parser.add_argument("--cache-dir", type=str, default=None, help="Optional GeoTessera manifest/cache directory.")
    parser.add_argument("--embeddings-dir", type=str, default=None, help="Optional local GeoTessera tile directory for reuse/download cache.")
    parser.add_argument("--no-sort-points", action="store_true", help="Disable lon/lat sorting before GeoTessera sampling.")
    parser.add_argument("--no-point-cache", action="store_true")
    parser.add_argument("--test-only", action="store_true", help="Run only first 3 incomplete tasks as a smoke test.")
    args = parser.parse_args()

    requested_years = parse_years(args.years)
    simplify_tolerance = None if args.simplify_tolerance <= 0 else args.simplify_tolerance
    use_point_cache = not args.no_point_cache

    _, table_dir, _, batch_dir, point_dir = ensure_dirs(args.out_dir, args.scale, args.sampling_mode, args.max_points_per_basin)
    progress_log = table_dir / f"tessera_progress_scale{args.scale}m_parallel.csv"
    failed_log = table_dir / f"tessera_failed_batches_scale{args.scale}m_parallel.csv"
    metadata_path = args.out_dir / f"tessera_extraction_metadata_scale{args.scale}m.json"

    print("=" * 70, flush=True)
    print("TESSERA basin embedding extraction", flush=True)
    print("=" * 70, flush=True)
    print("Requested years:", requested_years, flush=True)
    print("Scale / point spacing:", args.scale, "m", flush=True)
    print("Batch size:", args.batch_size, flush=True)
    print("Workers:", args.workers, flush=True)
    print("Grid CRS:", args.grid_crs, flush=True)
    print("Sampling mode:", args.sampling_mode, flush=True)
    print("Max points per basin:", args.max_points_per_basin, flush=True)
    print("Min points per basin:", args.min_points_per_basin, flush=True)
    print("GeoTessera dataset version:", args.dataset_version, flush=True)
    print("GeoTessera dataset variant:", args.dataset_variant, flush=True)
    print("Use GeoTesseraZarr:", bool(args.use_zarr), flush=True)
    print("Output directory:", args.out_dir, flush=True)
    print("Batch directory:", batch_dir, flush=True)

    # Initialize once in the main thread to fail early and report availability.
    try:
        available_years = get_available_tessera_years(
            dataset_version=args.dataset_version,
            dataset_variant=args.dataset_variant,
            use_zarr=args.use_zarr,
            cache_dir=args.cache_dir,
            embeddings_dir=args.embeddings_dir,
        )
        print("GeoTessera initialized.", flush=True)
        print("Available TESSERA years reported by GeoTesseraZarr:", available_years, flush=True)
    except Exception as e:
        print("GeoTessera initialization failed.", flush=True)
        print("Install/upgrade with: pip install -U geotessera", flush=True)
        raise e

    if available_years:
        unavailable = [y for y in requested_years if y not in available_years]
        if unavailable and not args.skip_unavailable_years:
            raise ValueError(
                f"Requested years are not available in GeoTesseraZarr: {unavailable}. "
                f"Available years: {available_years}. Use --skip-unavailable-years to continue."
            )
        years = [y for y in requested_years if y in available_years]
    else:
        # If the client does not expose a years attribute, attempt requested years.
        years = requested_years

    if len(years) == 0:
        raise ValueError("No usable TESSERA years after availability filtering.")

    basins_wgs84 = load_target_basins(
        basins_gpkg=args.basins_gpkg,
        index_parquet=args.index_parquet,
        years=years,
        simplify_tolerance=simplify_tolerance,
    )
    print("Loaded target basins:", len(basins_wgs84), flush=True)
    print("CRS:", basins_wgs84.crs, flush=True)

    basin_geojson = args.out_dir / "gages_basins_for_tessera_extraction.geojson"
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
        "embedding_source": "TESSERA / GeoTesseraZarr",
        "bands": TESSERA_BANDS,
        "requested_years": requested_years,
        "years": years,
        "available_years_reported": available_years,
        "scale_m_point_spacing": args.scale,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "max_retries": args.max_retries,
        "simplify_tolerance": simplify_tolerance,
        "grid_crs": args.grid_crs,
        "max_points_per_basin": args.max_points_per_basin,
        "min_points_per_basin": args.min_points_per_basin,
        "sampling_mode": args.sampling_mode,
        "max_candidate_grid_points": args.max_candidate_grid_points,
        "query_chunk_size": args.query_chunk_size,
        "use_point_cache": use_point_cache,
        "dataset_version": args.dataset_version,
        "dataset_variant": args.dataset_variant,
        "use_zarr": bool(args.use_zarr),
        "cache_dir": args.cache_dir,
        "embeddings_dir": args.embeddings_dir,
        "sort_points": not args.no_sort_points,
        "n_basins": int(len(basins_wgs84)),
        "expected_rows": int(len(years) * len(basins_wgs84)),
        "expected_files": int(len(all_tasks)),
        "basins_gpkg": str(args.basins_gpkg),
        "index_parquet": str(args.index_parquet),
        "batch_dir": str(batch_dir),
        "point_cache_dir": str(point_dir),
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
        grid_crs=args.grid_crs,
        max_points_per_basin=args.max_points_per_basin,
        max_candidate_grid_points=args.max_candidate_grid_points,
        sampling_mode=args.sampling_mode,
        min_points_per_basin=args.min_points_per_basin,
        query_chunk_size=args.query_chunk_size,
        point_cache_dir=point_dir,
        use_point_cache=use_point_cache,
        max_retries=args.max_retries,
        progress_log=progress_log,
        failed_log=failed_log,
        dataset_version=args.dataset_version,
        dataset_variant=args.dataset_variant,
        use_zarr=args.use_zarr,
        cache_dir=args.cache_dir,
        embeddings_dir=args.embeddings_dir,
        sort_points=not args.no_sort_points,
    )

    batch_files = sorted(batch_dir.glob("tessera_year*_batch*.parquet"))
    print("\nCompleteness check", flush=True)
    print("Batch files produced:", len(batch_files), flush=True)
    print("Expected files:", len(all_tasks), flush=True)
    if batch_files:
        example = pd.read_parquet(batch_files[0])
        print("Example file:", batch_files[0], flush=True)
        print("Example shape:", example.shape, flush=True)
        print(example.head(), flush=True)
        valid_all = int(example[TESSERA_BANDS].notna().all(axis=1).sum())
        valid_any = int(example[TESSERA_BANDS].notna().any(axis=1).sum())
        print(f"Example valid all bands: {valid_all}/{len(example)}", flush=True)
        print(f"Example valid any band: {valid_any}/{len(example)}", flush=True)


if __name__ == "__main__":
    main()
