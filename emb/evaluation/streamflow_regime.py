#!/usr/bin/env python3
"""
Create recharge-style river classes from daily streamflow parquet files.

Inputs:
    1. One parquet file per basin/gauge
    2. Basin GeoJSON containing basin area

Outputs:
    1. annual_streamflow_recharge_metrics.csv
    2. basin_recharge_classes.csv
    3. basin_recharge_class_summary.csv

Main metrics:
    annual_runoff_mm
    annual_baseflow_mm
    BFI
    recharge_class based on annual_baseflow_mm
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm


# ==========================================================
# CONFIG
# ==========================================================

PARQUET_DIR = Path("/data/basin2vec/raw/daily_streamflow_usgs")
GEOJSON_FILE = Path("/data/basin2vec/raw/gages-ii/us_selected_basins/basins.geojson")

OUT_DIR = Path("recharge_classification_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_ANNUAL = OUT_DIR / "annual_streamflow_recharge_metrics.csv"
OUT_BASIN = OUT_DIR / "basin_recharge_classes.csv"
OUT_SUMMARY = OUT_DIR / "basin_recharge_class_summary.csv"

# Change this if your discharge is already in m3/s
DISCHARGE_UNITS = "cfs"   # options: "cfs" or "cms"

# Minimum number of valid daily observations required per year
MIN_DAYS_PER_YEAR = 300

# Baseflow filter parameter
# Higher alpha gives smoother baseflow.
# Typical values: 0.925 to 0.98
BASEFLOW_ALPHA = 0.925


# ==========================================================
# HELPERS
# ==========================================================

def normalize_site_id(x):
    """
    Normalize USGS site IDs as strings with leading zeros preserved.
    """
    if pd.isna(x):
        return None
    x = str(x).strip()
    if x.endswith(".0"):
        x = x[:-2]
    return x.zfill(8)


def find_site_id_column(gdf):
    """
    Try to detect the basin/gauge ID column in the GeoJSON.
    """
    candidate_cols = [
        "site_id",
        "gage_id",
        "STAID",
        "SOURCE_FEA",
        "SOURCE_FID",
        "GAGE_ID",
        "usgs_id",
        "USGS_ID",
        "huc_id",
    ]

    for col in candidate_cols:
        if col in gdf.columns:
            return col

    raise ValueError(
        "Could not find a site ID column in GeoJSON. "
        f"Available columns are: {list(gdf.columns)}"
    )


def find_area_column(gdf):
    """
    Try to detect an area column in the GeoJSON.
    Area may be in km2, m2, or acres depending on your file.
    """
    candidate_cols = [
        "area_km2",
        "AREA_KM2",
        "DRAIN_SQKM",
        "drain_sqkm",
        "area",
        "AREA",
        "AREA_SQKM",
        "basin_area_km2",
    ]

    for col in candidate_cols:
        if col in gdf.columns:
            return col

    return None


def get_area_km2(gdf):
    """
    Return GeoDataFrame with standardized area_km2 column.

    Priority:
        1. Existing area column if detected
        2. Geometry-derived area after projecting to equal-area CRS
    """
    gdf = gdf.copy()

    area_col = find_area_column(gdf)

    if area_col is not None:
        vals = pd.to_numeric(gdf[area_col], errors="coerce")

        # Heuristic:
        # If values are very large, they may be m2.
        # If moderate, likely km2.
        median_val = vals.median()

        if median_val > 1e6:
            gdf["area_km2"] = vals / 1e6
        else:
            gdf["area_km2"] = vals

    else:
        print("No area column found. Computing basin area from geometry.")

        if gdf.crs is None:
            raise ValueError(
                "GeoJSON has no CRS. Cannot safely compute area from geometry."
            )

        # EPSG:5070 is CONUS Albers Equal Area
        gdf_area = gdf.to_crs("EPSG:5070")
        gdf["area_km2"] = gdf_area.geometry.area / 1e6

    return gdf


def convert_discharge_to_cms(q):
    """
    Convert discharge to cubic meters per second.
    """
    q = pd.to_numeric(q, errors="coerce")

    if DISCHARGE_UNITS.lower() == "cfs":
        return q * 0.028316846592
    elif DISCHARGE_UNITS.lower() in ["cms", "m3s", "m3/s"]:
        return q
    else:
        raise ValueError("DISCHARGE_UNITS must be either 'cfs' or 'cms'.")


def lyne_hollick_baseflow(q, alpha=0.925, passes=3):
    """
    Estimate baseflow using a recursive digital filter.

    Input:
        q: 1D numpy array of daily discharge in m3/s

    Output:
        baseflow estimate in m3/s

    Notes:
        This is a practical hydrograph separation method.
        It estimates slow-flow/baseflow component but is not a direct
        physical measurement of groundwater recharge.
    """
    q = np.asarray(q, dtype=float)

    # Replace negative values with nan
    q[q < 0] = np.nan

    # Interpolate short gaps for filtering
    s = pd.Series(q)
    q_filled = s.interpolate(limit_direction="both").to_numpy()

    if np.all(np.isnan(q_filled)):
        return np.full_like(q, np.nan, dtype=float)

    def single_pass(flow):
        quickflow = np.zeros_like(flow, dtype=float)

        for i in range(1, len(flow)):
            quickflow[i] = (
                alpha * quickflow[i - 1]
                + ((1 + alpha) / 2.0) * (flow[i] - flow[i - 1])
            )

            if quickflow[i] < 0:
                quickflow[i] = 0

            if quickflow[i] > flow[i]:
                quickflow[i] = flow[i]

        baseflow = flow - quickflow
        baseflow[baseflow < 0] = 0
        baseflow[baseflow > flow] = flow[baseflow > flow]

        return baseflow

    bf = q_filled.copy()

    for _ in range(passes):
        bf = single_pass(bf)

    # Preserve original missing days
    bf[np.isnan(q)] = np.nan

    return bf


def classify_percentile(values):
    """
    Classify values into five percentile-based classes.

    Returns:
        class labels based on quintiles.
    """
    labels = [
        "Very low recharge",
        "Low recharge",
        "Moderate recharge",
        "High recharge",
        "Very high recharge",
    ]

    return pd.qcut(
        values,
        q=5,
        labels=labels,
        duplicates="drop"
    )


# ==========================================================
# LOAD BASIN AREAS
# ==========================================================

print("Loading basin GeoJSON...")
gdf = gpd.read_file(GEOJSON_FILE)

site_col = find_site_id_column(gdf)
gdf[site_col] = gdf[site_col].apply(normalize_site_id)

gdf = get_area_km2(gdf)

area_df = gdf[[site_col, "area_km2"]].copy()
area_df = area_df.rename(columns={site_col: "site_id"})
area_df = area_df.dropna(subset=["site_id", "area_km2"])
area_df = area_df.drop_duplicates(subset=["site_id"])

area_lookup = dict(zip(area_df["site_id"], area_df["area_km2"]))

print(f"Loaded area for {len(area_lookup):,} basins.")


# ==========================================================
# PROCESS STREAMFLOW FILES
# ==========================================================

annual_records = []
failed_records = []

parquet_files = sorted(PARQUET_DIR.glob("*.parquet"))

print(f"Found {len(parquet_files):,} streamflow parquet files.")

for fp in tqdm(parquet_files, desc="Processing basins"):

    site_id = normalize_site_id(fp.stem)

    if site_id not in area_lookup:
        failed_records.append({
            "site_id": site_id,
            "file": str(fp),
            "reason": "Missing basin area"
        })
        continue

    area_km2 = area_lookup[site_id]
    area_m2 = area_km2 * 1e6

    try:
        df = pd.read_parquet(fp)

        # Try to detect time and discharge columns
        time_col_candidates = ["time", "date", "datetime", "Date"]
        q_col_candidates = ["value", "discharge", "streamflow", "Q", "flow"]

        time_col = None
        q_col = None

        for col in time_col_candidates:
            if col in df.columns:
                time_col = col
                break

        for col in q_col_candidates:
            if col in df.columns:
                q_col = col
                break

        if time_col is None:
            raise ValueError(f"No time column found. Columns: {list(df.columns)}")

        if q_col is None:
            raise ValueError(f"No discharge column found. Columns: {list(df.columns)}")

        df = df[[time_col, q_col]].copy()
        df = df.rename(columns={time_col: "date", q_col: "q_raw"})

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

        df["q_cms"] = convert_discharge_to_cms(df["q_raw"])

        # Remove impossible negative flows
        df.loc[df["q_cms"] < 0, "q_cms"] = np.nan

        df = df.sort_values("date")
        df["year"] = df["date"].dt.year

        # Baseflow estimation on full time series
        df["baseflow_cms"] = lyne_hollick_baseflow(
            df["q_cms"].to_numpy(),
            alpha=BASEFLOW_ALPHA,
            passes=3
        )

        for year, g in df.groupby("year"):
            n_days = g["q_cms"].notna().sum()

            if n_days < MIN_DAYS_PER_YEAR:
                continue

            q_sum_volume_m3 = np.nansum(g["q_cms"].to_numpy() * 86400.0)
            bf_sum_volume_m3 = np.nansum(g["baseflow_cms"].to_numpy() * 86400.0)

            annual_runoff_mm = (q_sum_volume_m3 / area_m2) * 1000.0
            annual_baseflow_mm = (bf_sum_volume_m3 / area_m2) * 1000.0

            if q_sum_volume_m3 > 0:
                bfi = bf_sum_volume_m3 / q_sum_volume_m3
            else:
                bfi = np.nan

            annual_records.append({
                "site_id": site_id,
                "year": int(year),
                "area_km2": area_km2,
                "n_valid_days": int(n_days),
                "annual_runoff_mm": annual_runoff_mm,
                "annual_baseflow_mm": annual_baseflow_mm,
                "BFI": bfi,
                "mean_q_cms": np.nanmean(g["q_cms"]),
                "mean_baseflow_cms": np.nanmean(g["baseflow_cms"]),
            })

    except Exception as e:
        failed_records.append({
            "site_id": site_id,
            "file": str(fp),
            "reason": str(e)
        })


# ==========================================================
# SAVE ANNUAL METRICS
# ==========================================================

annual_df = pd.DataFrame(annual_records)

if annual_df.empty:
    raise RuntimeError("No annual records were created. Check input paths and columns.")

annual_df.to_csv(OUT_ANNUAL, index=False)

print(f"Saved annual metrics: {OUT_ANNUAL}")
print(f"Annual records: {len(annual_df):,}")


# ==========================================================
# AGGREGATE TO BASIN LEVEL
# ==========================================================

basin_df = (
    annual_df
    .groupby("site_id")
    .agg(
        area_km2=("area_km2", "first"),
        n_years=("year", "nunique"),
        mean_annual_runoff_mm=("annual_runoff_mm", "mean"),
        median_annual_runoff_mm=("annual_runoff_mm", "median"),
        mean_annual_baseflow_mm=("annual_baseflow_mm", "mean"),
        median_annual_baseflow_mm=("annual_baseflow_mm", "median"),
        mean_BFI=("BFI", "mean"),
        median_BFI=("BFI", "median"),
        runoff_cv=("annual_runoff_mm", lambda x: np.nanstd(x) / np.nanmean(x) if np.nanmean(x) > 0 else np.nan),
        baseflow_cv=("annual_baseflow_mm", lambda x: np.nanstd(x) / np.nanmean(x) if np.nanmean(x) > 0 else np.nan),
    )
    .reset_index()
)

# Keep only basins with enough years
# You can adjust this depending on your data period.
basin_df = basin_df[basin_df["n_years"] >= 5].copy()


# ==========================================================
# RECHARGE CLASSES
# ==========================================================

# Primary classification: based on mean annual baseflow depth
basin_df["recharge_class"] = classify_percentile(
    basin_df["mean_annual_baseflow_mm"]
)

# Optional classification based on total runoff depth
basin_df["runoff_class"] = classify_percentile(
    basin_df["mean_annual_runoff_mm"]
)

# Optional classification based on BFI
basin_df["baseflow_dominance_class"] = classify_percentile(
    basin_df["mean_BFI"]
)


# ==========================================================
# SAVE BASIN CLASSES
# ==========================================================

basin_df.to_csv(OUT_BASIN, index=False)

summary_df = (
    basin_df
    .groupby("recharge_class", observed=False)
    .agg(
        n_basins=("site_id", "count"),
        mean_baseflow_mm=("mean_annual_baseflow_mm", "mean"),
        median_baseflow_mm=("mean_annual_baseflow_mm", "median"),
        min_baseflow_mm=("mean_annual_baseflow_mm", "min"),
        max_baseflow_mm=("mean_annual_baseflow_mm", "max"),
        mean_runoff_mm=("mean_annual_runoff_mm", "mean"),
        mean_BFI=("mean_BFI", "mean"),
        median_BFI=("mean_BFI", "median"),
    )
    .reset_index()
)

summary_df.to_csv(OUT_SUMMARY, index=False)

print(f"Saved basin classes: {OUT_BASIN}")
print(f"Saved class summary: {OUT_SUMMARY}")


# ==========================================================
# SAVE FAILURES
# ==========================================================

if failed_records:
    failed_df = pd.DataFrame(failed_records)
    failed_path = OUT_DIR / "failed_streamflow_recharge_processing.csv"
    failed_df.to_csv(failed_path, index=False)
    print(f"Saved failed records: {failed_path}")


# ==========================================================
# PRINT SUMMARY
# ==========================================================

print("\nRecharge class summary:")
print(summary_df)

print("\nExample basin-level output:")
print(basin_df.head())