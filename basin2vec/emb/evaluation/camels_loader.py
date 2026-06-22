from pathlib import Path
import pandas as pd
import numpy as np


CAMELS_ROOT = Path("/data/camels_us")
DAYMET_DIR = CAMELS_ROOT / "basin_mean_forcing" / "daymet"
STREAMFLOW_DIR = CAMELS_ROOT / "usgs_streamflow"
ATTR_DIR = CAMELS_ROOT / "camels_attributes_v2.0"


CAMELS_PHYSIOGRAPHIC_COLUMNS = [
    # Topography
    "area_gages2",
    "elev_mean",
    "slope_mean",

    # Soil
    "sand_frac",
    "silt_frac",
    "clay_frac",
    "soil_depth_pelletier",
    "soil_depth_statsgo",
    "soil_porosity",
    "soil_conductivity",
    "max_water_content",

    # Geology
    "carbonate_rocks_frac",
    "geol_permeability",

    # Vegetation / land cover
    "frac_forest",
    "lai_max",
    "lai_diff",
    "gvf_max",
    "gvf_diff",
]

SELECTED_ATTRIBUTE_COLUMNS = CAMELS_PHYSIOGRAPHIC_COLUMNS
def read_camels_attributes(selected_only=True):
    files = [
        "camels_clim.txt",
        "camels_geol.txt",
        "camels_hydro.txt",
        "camels_soil.txt",
        "camels_topo.txt",
        "camels_vege.txt",
    ]

    dfs = []

    for f in files:
        path = ATTR_DIR / f
        df = pd.read_csv(path, sep=";")
        df["gauge_id"] = df["gauge_id"].astype(str).str.zfill(8)
        dfs.append(df)

    out = dfs[0]

    for df in dfs[1:]:
        out = out.merge(df, on="gauge_id", how="inner")

    if not selected_only:
        return out

    available = [c for c in SELECTED_ATTRIBUTE_COLUMNS if c in out.columns]
    missing = [c for c in SELECTED_ATTRIBUTE_COLUMNS if c not in out.columns]

    print(f"Requested selected attributes: {len(SELECTED_ATTRIBUTE_COLUMNS)}")
    print(f"Available selected attributes: {len(available)}")
    print(f"Missing selected attributes: {len(missing)}")

    if missing:
        print("\nMissing columns:")
        for c in missing:
            print(" -", c)

    if len(available) == 0:
        raise RuntimeError(
            "None of the selected columns exist in CAMELS attributes. "
            "These selected names are likely from HydroATLAS/GAGES attributes, not CAMELS."
        )

    out = out[["gauge_id"] + available].copy()

    for c in available:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c] = out[c].fillna(out[c].median(skipna=True))

    return out

def read_daymet_forcing(gauge_id):
    gauge_id = str(gauge_id).zfill(8)
    files = list(DAYMET_DIR.rglob(f"{gauge_id}*_forcing_leap.txt"))

    if len(files) == 0:
        raise FileNotFoundError(f"No Daymet forcing file found for {gauge_id}")

    path = files[0]

    df = pd.read_csv(path, sep=r"\s+", skiprows=4)
    df["date"] = pd.to_datetime(
        dict(year=df["Year"], month=df["Mnth"], day=df["Day"])
    )
    df["gauge_id"] = gauge_id

    return df


def read_streamflow(gauge_id):
    gauge_id = str(gauge_id).zfill(8)
    files = list(STREAMFLOW_DIR.rglob(f"{gauge_id}_streamflow_qc.txt"))

    if len(files) == 0:
        raise FileNotFoundError(f"No streamflow file found for {gauge_id}")

    path = files[0]

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["gauge_id", "Year", "Mnth", "Day", "QObs", "flag"],
    )

    df["gauge_id"] = df["gauge_id"].astype(str).str.zfill(8)
    df["date"] = pd.to_datetime(
        dict(year=df["Year"], month=df["Mnth"], day=df["Day"])
    )

    return df


def read_camels_basin(gauge_id):
    forcing = read_daymet_forcing(gauge_id)
    streamflow = read_streamflow(gauge_id)

    df = forcing.merge(
        streamflow[["date", "gauge_id", "QObs", "flag"]],
        on=["date", "gauge_id"],
        how="inner",
    )

    return df


def get_available_basins():
    forcing_ids = {
        p.name.split("_")[0]
        for p in DAYMET_DIR.rglob("*_forcing_leap.txt")
    }

    stream_ids = {
        p.name.split("_")[0]
        for p in STREAMFLOW_DIR.rglob("*_streamflow_qc.txt")
    }

    attrs = read_camels_attributes()
    attr_ids = set(attrs["gauge_id"].astype(str).str.zfill(8))

    common = sorted(forcing_ids & stream_ids & attr_ids)

    print("Daymet basins:", len(forcing_ids))
    print("Streamflow basins:", len(stream_ids))
    print("Attribute basins:", len(attr_ids))
    print("Common basins:", len(common))

    return common