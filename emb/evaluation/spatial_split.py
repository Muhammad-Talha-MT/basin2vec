#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import json
import time
import random
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import traceback
import gc
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


# ==========================================================
# CONFIG
# ==========================================================

CAMELS_ROOT = Path("/data/camels_us")
DAYMET_DIR = CAMELS_ROOT / "basin_mean_forcing" / "daymet"
STREAMFLOW_DIR = CAMELS_ROOT / "usgs_streamflow"
ATTR_DIR = CAMELS_ROOT / "camels_attributes_v2.0"

# This script now uses all available CAMELS-US basins rather than the
# older selected 531-basin XLSX list.  The constants below are kept only for
# backward compatibility with older helper functions and are not used for
# filtering the basin set.
CAMELS_531_XLSX = Path("michigan_data/Camels_531basins.xlsx")
CAMELS_531_XLSX_FALLBACK = Path("michigan_data/Camels_531basins.xlsx")

INDEX_PARQUET = Path("../config/training_step5/sample_index.parquet")

BASIN2VEC_NPZ = Path(
    "../src/evaluation_outputs/HCLV4_grouped_monthly_static_meta_areaaux_64d_ablation_no_temporal/basin_embeddings_full.npz"
)
BASIN2VEC_ARRAY_KEY = "embeddings"

# Basin2Vec temporal handling:
#   "aggregated" = one basin-level 64-d vector averaged over BASIN2VEC_AGG_YEARS.
#   "yearly"     = use the Basin2Vec vector from the sample target year.
# For LEAD=0 and H=1, the target year is the same as the issue/current-day year.
BASIN2VEC_TEMPORAL_MODE = "aggregated"
# BASIN2VEC_TEMPORAL_MODE = "yearly"

ALPHAEARTH_NPZ = Path(
    "alphaearth_embeddings/embeddings/alphaearth_annual_basin_embeddings_2017_2024_scale500m.npz"
)
SATCLIP_NPZ = Path("satclip_embeddings/satclip_8point_2024_embeddings.npz")
TESSERA_NPZ = Path("tessera_embeddings/tessera_8point_2024_embeddings.npz")

OUT_DIR = Path(
    f"evaluation_outputs/ablation_spatial_basin2vec_no_temporal_{BASIN2VEC_TEMPORAL_MODE}"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Keep the same model registry from your existing benchmark.
# MODEL_NAMES = ["lstm", "tft", "timexer", "timerxl", "iTransformer"]
MODEL_NAMES = ["lstm"]
DATE_COL = "date"
TARGET_COL = "discharge"

# No past discharge input. Dynamic inputs are meteorological forcing variables.
DYNAMIC_COLS = ["prcp", "srad", "tmax", "tmin", "vp"]

# All requested static sources. Basin2Vec is averaged over available years so it
# is a basin-level static descriptor, same as AlphaEarth/SatCLIP/TESSERA.
STATIC_SOURCES = ["basin2vec"]
# STATIC_SOURCES = ["basin2vec", "alphaearth", "tessera", "satclip"]
SEQ_LENGTHS = [365]
LEAD = 0
HORIZONS = [1]

# Kratzert-style ungauged-basin setup, Option A:
#   - test basins = one held-out spatial fold
#   - train/validation basins = all remaining folds
#   - train and validation are separated by time, not basin identity
#   - final metrics are reported only on held-out test basins
#
# We keep the CAMELS-US 2000--2014 period. The non-test basins are used for
# training over 2000--2010 and validation/early stopping over 2011--2014.
# The held-out test basins are evaluated over the full 2000--2014 period.


DATA_START = "1990-01-01"
DATA_END   = "2014-12-31"

TRAIN_START = "1990-01-01"
TRAIN_END   = "2004-12-31"

VAL_START   = "2005-01-01"
VAL_END     = "2014-12-31"

TEST_START  = "1990-01-01"
TEST_END    = "2014-12-31"

# Static embedding aggregation years.
# Basin2Vec is averaged over these years per basin, exactly like AlphaEarth is
# averaged over available annual embeddings.
BASIN2VEC_AGG_YEARS = list(range(1990, 2015))
ALPHAEARTH_AGG_YEARS = list(range(2017, 2025))
SATCLIP_AGG_YEARS = list(range(2024, 2025))
TESSERA_AGG_YEARS = list(range(2024, 2025))

# Kratzert-style ungauged-basin K-fold CV.
# Use 6 folds so each held-out test fold contains approximately 1/6 of the available CAMELS-US basins.
# Validation is temporal on the same non-test basins, so there is no inner
# validation-basin fraction.
K_FOLDS = 6
SPATIAL_SPLIT_SEED = 42
INNER_VAL_FRACTION = None
STRATIFY_FOLDS_BY_HUC = False

SEED = 42
BATCH_SIZE_PER_GPU = 128
NUM_WORKERS_PER_GPU = 4
PREFETCH_FACTOR = 2

EPOCHS = 30
PATIENCE = 8
LR = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
USE_AMP = True

USE_LR_SCHEDULER = True
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_MIN_LR = 1e-5

STANDARDIZE_DYNAMIC_AND_TARGET_FROM_TRAIN = True
CONVERT_STREAMFLOW_CFS_TO_MM_PER_DAY = False

D_MODEL = 128
N_HEADS = 8
N_LAYERS = 2
D_FF = 512
DROPOUT = 0.15
LSTM_HIDDEN = 256
LSTM_LAYERS = 2
PATCH_LEN = 16
PATCH_STRIDE = 8
MAX_SEQ_LEN = max(SEQ_LENGTHS) + 1
USE_LEARNED_POSITION = True

# Resume behavior
RESUME_SKIP_COMPLETED = True
RESUME_TRAINING_FROM_LAST_CHECKPOINT = True
SAVE_LAST_CHECKPOINT_EVERY_EPOCH = True

CAMELS_SELECTED_ATTRIBUTE_COLUMNS = [
    # Topography: DEM / elevation
    "elev_mean",

    # Soil texture: SoilGrids clay/silt/sand analogs
    "clay_frac",
    "silt_frac",
    "sand_frac",

    # Soil physical/hydraulic properties: partial analogs
    # CAMELS does not provide bulk density or soil organic carbon directly.
    "soil_depth_pelletier",
    "soil_depth_statsgo",
    "soil_porosity",
    "soil_conductivity",

    # Hydrogeology: GLHYMPS permeability/hydraulic-conductivity analog
    "geol_permeability",

    # Land cover: NLCD land-cover analogs
    "frac_forest",
    "dom_land_cover_frac",
    "dom_land_cover",
]


# CAMELS_SELECTED_ATTRIBUTE_COLUMNS = [
#     # Topography: DEM / elevation
#     "elev_mean",
#     "area_gages2",
#     # Soil texture: SoilGrids clay/silt/sand analogs
#     "clay_frac",
#     "silt_frac",
#     "sand_frac",

#     # Soil physical/hydraulic properties: partial analogs
#     # CAMELS does not provide bulk density or soil organic carbon directly.
#     "soil_depth_pelletier",
#     "soil_depth_statsgo",
#     "soil_porosity",
#     "soil_conductivity",

#     # Hydrogeology: GLHYMPS permeability/hydraulic-conductivity analog
#     "geol_permeability",

#     # Land cover: NLCD land-cover analogs
#     "frac_forest",
#     "dom_land_cover_frac",
#     "dom_land_cover",
# ]

# ALL_CAMELS_ATTRIBUTE_COLUMNS = [
#     # name / region
#     "gauge_name",
#     "huc_02",

#     # topography
#     "gauge_lat",
#     "gauge_lon",
#     "elev_mean",
#     "slope_mean",
#     "area_gages2",
#     "area_geospa_fabric",

#     # climate
#     "p_mean",
#     "pet_mean",
#     "p_seasonality",
#     "frac_snow",
#     "aridity",
#     "high_prec_freq",
#     "high_prec_dur",
#     "low_prec_freq",
#     "low_prec_dur",

#     # hydrologic signatures
#     "q_mean",
#     "runoff_ratio",
#     "slope_fdc",
#     "baseflow_index",
#     "stream_elas",
#     "q5",
#     "q95",
#     "high_q_freq",
#     "high_q_dur",
#     "low_q_freq",
#     "low_q_dur",
#     "zero_q_freq",
#     "hfd_mean",

#     # soil
#     "soil_depth_pelletier",
#     "soil_depth_statsgo",
#     "soil_porosity",
#     "soil_conductivity",
#     "max_water_content",
#     "sand_frac",
#     "silt_frac",
#     "clay_frac",
#     "water_frac",
#     "organic_frac",
#     "other_frac",

#     # vegetation
#     "frac_forest",
#     "lai_max",
#     "lai_diff",
#     "gvf_max",
#     "gvf_diff",
#     "dom_land_cover_frac",
#     "dom_land_cover",

#     # geology
#     "geol_1st_class",
#     "glim_1st_class_frac",
#     "geol_2nd_class",
#     "glim_2nd_class_frac",
#     "carbonate_rocks_frac",
#     "geol_porostiy",
#     "geol_permeability",
# ]

# CAMELS_SELECTED_ATTRIBUTE_COLUMNS = ALL_CAMELS_ATTRIBUTE_COLUMNS
# # DISTRIBUTED
# ==========================================================

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        distributed = world_size > 1
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        distributed = False

    if torch.cuda.is_available():
        if distributed:
            torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if distributed:
        dist.init_process_group(backend="nccl", init_method="env://")

    return distributed, rank, world_size, local_rank, device


DISTRIBUTED, RANK, WORLD_SIZE, LOCAL_RANK, DEVICE = setup_distributed()


def is_rank0():
    return RANK == 0


def rprint(*args, **kwargs):
    if is_rank0():
        print(*args, **kwargs)


def barrier():
    if DISTRIBUTED:
        dist.barrier()


def cleanup_distributed():
    if DISTRIBUTED and dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed=42):
    seed = seed + RANK
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ==========================================================
# HELPERS
# ==========================================================

def normalize_site_id(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    s = re.sub(r"\D", "", s)
    if not s:
        return None
    return s.zfill(8) if len(s) <= 8 else s


def l2_normalize(x, eps=1e-12):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), eps, None)


def to_builtin(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    return obj


def save_json(obj, path):
    if not is_rank0():
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(to_builtin(obj), f, indent=2)


def save_checkpoint(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(to_builtin(payload), path)


def load_checkpoint(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def autocast_context():
    return torch.amp.autocast("cuda", enabled=(USE_AMP and DEVICE.type == "cuda"))


def make_grad_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=(USE_AMP and DEVICE.type == "cuda"))
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE.type == "cuda"))


def make_lr_scheduler(optimizer):
    if not USE_LR_SCHEDULER:
        return None
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
        min_lr=LR_SCHEDULER_MIN_LR,
    )


def get_current_lr(optimizer):
    return float(optimizer.param_groups[0]["lr"])


# ==========================================================
# CAMELS DATA LOADING
# ==========================================================

def detect_camels_forcing_header(path: Path) -> int:
    with open(path, "r") as f:
        for i, line in enumerate(f):
            cols = line.strip().split()
            if len(cols) >= 3:
                if cols[0].lower() == "year" and cols[1].lower() in ["mnth", "month"] and cols[2].lower() == "day":
                    return i
    raise RuntimeError(f"Could not detect CAMELS forcing header: {path}")


def read_camels_daymet(site_id: str) -> pd.DataFrame:
    site_id = normalize_site_id(site_id)
    files = list(DAYMET_DIR.rglob(f"{site_id}*_forcing_leap.txt"))

    if not files:
        raise FileNotFoundError(f"No Daymet forcing found for {site_id}")

    path = files[0]
    header = detect_camels_forcing_header(path)

    df = pd.read_csv(path, sep=r"\s+", skiprows=header)

    rename = {
        "Year": "year",
        "Mnth": "month",
        "Month": "month",
        "Day": "day",
        "prcp(mm/day)": "prcp",
        "srad(W/m2)": "srad",
        "swe(mm)": "swe",
        "tmax(C)": "tmax",
        "tmin(C)": "tmin",
        "vp(Pa)": "vp",
    }
    df = df.rename(columns=rename)

    required = ["year", "month", "day"] + DYNAMIC_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing Daymet columns for {site_id}: {missing}. Available={df.columns.tolist()}")

    df[DATE_COL] = pd.to_datetime(
        dict(year=df["year"], month=df["month"], day=df["day"]),
        errors="coerce",
    )

    out = df[[DATE_COL] + DYNAMIC_COLS].copy()
    for c in DYNAMIC_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def read_camels_streamflow(site_id: str) -> pd.DataFrame:
    site_id = normalize_site_id(site_id)
    files = list(STREAMFLOW_DIR.rglob(f"{site_id}_streamflow_qc.txt"))

    if not files:
        raise FileNotFoundError(f"No streamflow found for {site_id}")

    df = pd.read_csv(
        files[0],
        sep=r"\s+",
        header=None,
        names=["site_id", "year", "month", "day", TARGET_COL, "flag"],
    )

    df["site_id"] = df["site_id"].astype(str).str.zfill(8)
    df[DATE_COL] = pd.to_datetime(
        dict(year=df["year"], month=df["month"], day=df["day"]),
        errors="coerce",
    )
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    return df[[DATE_COL, TARGET_COL]].copy()


def read_camels_attributes_static() -> Dict[str, np.ndarray]:
    files = [
        "camels_clim.txt",
        "camels_geol.txt",
        "camels_hydro.txt",
        "camels_soil.txt",
        "camels_topo.txt",
        "camels_vege.txt",
    ]

    dfs = []

    for fname in files:
        path = ATTR_DIR / fname
        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path, sep=";")
        df["site_id"] = df["gauge_id"].astype(str).str.zfill(8)
        df = df.drop(columns=["gauge_id"])
        dfs.append(df)

    out = dfs[0]

    for df in dfs[1:]:
        out = out.merge(df, on="site_id", how="inner")

    available = [c for c in CAMELS_SELECTED_ATTRIBUTE_COLUMNS if c in out.columns]
    missing = [c for c in CAMELS_SELECTED_ATTRIBUTE_COLUMNS if c not in out.columns]

    if len(available) == 0:
        raise RuntimeError(
            "None of the selected CAMELS attributes were found. "
            f"Available columns include: {out.columns.tolist()[:40]}"
        )

    rprint(f"Requested CAMELS attributes: {len(CAMELS_SELECTED_ATTRIBUTE_COLUMNS)}")
    rprint(f"Available CAMELS attributes used: {len(available)}")
    rprint(f"Missing CAMELS attributes: {len(missing)}")

    if is_rank0():
        pd.DataFrame({"attribute": available}).to_csv(
            OUT_DIR / "camels_attribute_columns_used.csv",
            index=False,
        )
        pd.DataFrame({"missing_attribute": missing}).to_csv(
            OUT_DIR / "camels_attribute_columns_missing.csv",
            index=False,
        )

    out = out[["site_id"] + available].copy()

    for c in available:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        med = out[c].median(skipna=True)
        out[c] = out[c].fillna(med)

    static = {}

    for _, row in out.iterrows():
        sid = normalize_site_id(row["site_id"])
        if sid is None:
            continue

        z = row[available].to_numpy(dtype=np.float32)
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        static[sid] = z

    rprint(
        f"Loaded selected CAMELS static attributes: {len(static)} basins, "
        f"dim={len(available)}"
    )

    return static


def resolve_camels_671_xlsx() -> Path:
    if CAMELS_531_XLSX.exists():
        return CAMELS_531_XLSX
    if CAMELS_531_XLSX_FALLBACK.exists():
        return CAMELS_531_XLSX_FALLBACK
    raise FileNotFoundError(
        f"Could not find 531-basin XLSX at {CAMELS_531_XLSX} or {CAMELS_531_XLSX_FALLBACK}. "
        "Place Camels_531basins.xlsx next to this script or update CAMELS_531_XLSX."
    )


def load_camels_671_metadata() -> pd.DataFrame:
    path = resolve_camels_671_xlsx()
    df = pd.read_excel(path)
    cols_lower = {str(c).strip().lower(): c for c in df.columns}

    if "gauge id" in cols_lower:
        gauge_col = cols_lower["gauge id"]
    elif "gauge_id" in cols_lower:
        gauge_col = cols_lower["gauge_id"]
    elif "site_id" in cols_lower:
        gauge_col = cols_lower["site_id"]
    else:
        # Fallback: use the second column, matching the uploaded XLSX.
        gauge_col = df.columns[1]

    huc_col = cols_lower.get("huc", None)

    out = pd.DataFrame({
        "site_id": [normalize_site_id(x) for x in df[gauge_col]],
    })
    if huc_col is not None:
        out["huc"] = pd.to_numeric(df[huc_col], errors="coerce").fillna(-1).astype(int)
    else:
        out["huc"] = -1

    out = out.dropna(subset=["site_id"]).drop_duplicates("site_id").reset_index(drop=True)

    if is_rank0():
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out.to_csv(OUT_DIR / "camels_671_basin_list_used.csv", index=False)

    rprint(f"Loaded CAMELS-671 basin list: {len(out)} basins from {path}")
    return out


def load_camels_huc_metadata() -> pd.DataFrame:
    """
    Load HUC metadata for all CAMELS-US basins from camels_name.txt.

    This replaces the 531-basin XLSX metadata for fold stratification, so the
    experiment can use all available CAMELS basins while still optionally
    stratifying folds by HUC-02.
    """
    name_path = ATTR_DIR / "camels_name.txt"
    if not name_path.exists():
        raise FileNotFoundError(name_path)

    df = pd.read_csv(name_path, sep=";")
    if "gauge_id" not in df.columns:
        raise RuntimeError("camels_name.txt must contain gauge_id for HUC stratification.")

    out = pd.DataFrame({
        "site_id": [normalize_site_id(x) for x in df["gauge_id"]],
    })

    if "huc_02" in df.columns:
        out["huc"] = pd.to_numeric(df["huc_02"], errors="coerce").fillna(-1).astype(int)
    else:
        out["huc"] = -1

    out = out.dropna(subset=["site_id"]).drop_duplicates("site_id").reset_index(drop=True)

    if is_rank0():
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out.to_csv(OUT_DIR / "camels_all_basin_huc_metadata_used.csv", index=False)

    rprint(f"Loaded CAMELS all-basin HUC metadata: {len(out)} basins from {name_path}")
    return out


def read_camels_area_km2() -> Dict[str, float]:
    topo_path = ATTR_DIR / "camels_topo.txt"
    if not topo_path.exists():
        raise FileNotFoundError(topo_path)
    topo = pd.read_csv(topo_path, sep=";")
    if "gauge_id" not in topo.columns or "area_gages2" not in topo.columns:
        raise RuntimeError("camels_topo.txt must contain gauge_id and area_gages2 for cfs -> mm/day conversion.")
    topo["site_id"] = topo["gauge_id"].astype(str).str.zfill(8)
    topo["area_gages2"] = pd.to_numeric(topo["area_gages2"], errors="coerce")
    return {
        normalize_site_id(row["site_id"]): float(row["area_gages2"])
        for _, row in topo.iterrows()
        if normalize_site_id(row["site_id"]) is not None and np.isfinite(row["area_gages2"]) and row["area_gages2"] > 0
    }


def cfs_to_mm_per_day(q_cfs: np.ndarray, area_km2: float) -> np.ndarray:
    # 1 cfs = 0.028316846592 m3/s; daily volume = q * factor * 86400.
    # depth_mm = volume_m3 / area_m2 * 1000.
    return q_cfs * 0.028316846592 * 86400.0 / (area_km2 * 1e6) * 1000.0


def get_camels_available_ids():
    """
    Return all CAMELS-US basins that have Daymet forcing, streamflow, and the
    selected CAMELS static attributes. This intentionally does NOT intersect
    with the 531-basin XLSX list.
    """
    forcing_ids = {
        normalize_site_id(p.name.split("_")[0])
        for p in DAYMET_DIR.rglob("*_forcing_leap.txt")
    }
    stream_ids = {
        normalize_site_id(p.name.split("_")[0])
        for p in STREAMFLOW_DIR.rglob("*_streamflow_qc.txt")
    }
    attr_ids = set(read_camels_attributes_static().keys())

    forcing_ids = {x for x in forcing_ids if x is not None}
    stream_ids = {x for x in stream_ids if x is not None}
    attr_ids = {x for x in attr_ids if x is not None}

    common = sorted(forcing_ids & stream_ids & attr_ids)

    rprint("CAMELS Daymet basins:", len(forcing_ids))
    rprint("CAMELS streamflow basins:", len(stream_ids))
    rprint("CAMELS attribute basins:", len(attr_ids))
    rprint("CAMELS common forcing/streamflow/attribute basins:", len(common))

    return common

def load_camels_data() -> Dict[str, pd.DataFrame]:
    common_ids = get_camels_available_ids()

    data_start = pd.Timestamp(DATA_START)
    data_end = pd.Timestamp(DATA_END)
    area_map = read_camels_area_km2() if CONVERT_STREAMFLOW_CFS_TO_MM_PER_DAY else {}

    basin_data = {}
    skipped = []

    iterator = tqdm(common_ids, desc="load CAMELS basins", disable=not is_rank0())

    for sid in iterator:
        try:
            forcing = read_camels_daymet(sid)
            flow = read_camels_streamflow(sid)

            df = forcing.merge(flow, on=DATE_COL, how="inner")
            df["site_id"] = sid

            if CONVERT_STREAMFLOW_CFS_TO_MM_PER_DAY:
                area_km2 = area_map.get(sid, None)
                if area_km2 is None or not np.isfinite(area_km2) or area_km2 <= 0:
                    raise RuntimeError(f"Missing/invalid area_gages2 for {sid}; cannot convert cfs to mm/day.")
                df[TARGET_COL] = cfs_to_mm_per_day(df[TARGET_COL].to_numpy(dtype=np.float32), area_km2)
            df = df.replace([-999, -999.0, -9999, -9999.0], np.nan)
            df = df[(df[DATE_COL] >= data_start) & (df[DATE_COL] <= data_end)]
            df = df.sort_values(DATE_COL).reset_index(drop=True)

            for c in DYNAMIC_COLS + [TARGET_COL]:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            if len(df) > 0:
                basin_data[sid] = df

        except Exception as e:
            skipped.append({"site_id": sid, "reason": str(e)})

    rprint("Loaded CAMELS basin time series:", len(basin_data))

    if is_rank0() and skipped:
        pd.DataFrame(skipped).to_csv(OUT_DIR / "skipped_camels_basins.csv", index=False)

    return basin_data


def check_year_coverage(basin_data):
    rows = []

    for sid, df in basin_data.items():
        d = df.copy()
        d["year"] = pd.to_datetime(d[DATE_COL]).dt.year

        for year, sub in d.groupby("year"):
            dyn = sub[DYNAMIC_COLS].to_numpy(dtype=np.float32)
            target = sub[TARGET_COL].to_numpy(dtype=np.float32)

            rows.append({
                "site_id": sid,
                "year": int(year),
                "n_days": int(len(sub)),
                "n_valid_dynamic": int(np.isfinite(dyn).all(axis=1).sum()),
                "n_valid_discharge": int(np.isfinite(target).sum()),
                "target_std": float(np.nanstd(target)),
            })

    coverage = pd.DataFrame(rows)
    if coverage.empty:
        raise RuntimeError("Coverage table is empty.")

    summary = (
        coverage.groupby("year")
        .agg(
            n_basins=("site_id", "nunique"),
            min_days=("n_days", "min"),
            median_days=("n_days", "median"),
            max_days=("n_days", "max"),
            median_target_std=("target_std", "median"),
        )
        .reset_index()
    )

    rprint("\nCAMELS year coverage summary:")
    rprint(summary)

    if is_rank0():
        coverage.to_csv(OUT_DIR / "camels_year_coverage_by_basin.csv", index=False)
        summary.to_csv(OUT_DIR / "camels_year_coverage_summary.csv", index=False)

    return coverage, summary


# ==========================================================
# STATIC SOURCES
# ==========================================================

def aggregate_annual_rows_to_static(
    meta: pd.DataFrame,
    emb: np.ndarray,
    source_name: str,
    years_keep: List[int],
    l2_normalize_rows: bool = True,
    l2_normalize_final: bool = True,
):
    """
    Average annual rows to one basin-level static vector.

    For Basin2Vec, l2_normalize_rows=False and l2_normalize_final=False because
    the exported Basin2Vec vectors are already normalized. The downstream
    fold-wise z-score standardization is still applied later using train basins
    only, which is separate from L2 normalization.
    """
    meta = meta.copy()
    valid_site = meta["site_id"].notna().to_numpy()
    meta = meta.loc[valid_site].reset_index(drop=True)
    emb = emb[valid_site]

    if years_keep is not None and "year" in meta.columns:
        keep = meta["year"].isin([int(y) for y in years_keep]).to_numpy()
        meta = meta.loc[keep].reset_index(drop=True)
        emb = emb[keep]

    if len(meta) == 0:
        raise RuntimeError(f"No {source_name} rows found after year/site filtering. years_keep={years_keep}")

    emb = np.nan_to_num(emb.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if l2_normalize_rows:
        emb = l2_normalize(emb)

    meta["row"] = np.arange(len(meta))
    static = {}
    counts = []
    for sid, sub in meta.groupby("site_id"):
        idx = sub["row"].to_numpy()
        z = emb[idx].mean(axis=0, keepdims=True)
        if l2_normalize_final:
            z = l2_normalize(z)
        static[sid] = z[0].astype(np.float32)
        years = sorted(set(int(y) for y in sub["year"].tolist())) if "year" in sub.columns else []
        counts.append({
            "site_id": sid,
            f"n_{source_name}_years": len(years),
            f"{source_name}_years": ",".join(map(str, years)),
        })

    if is_rank0():
        pd.DataFrame(counts).to_csv(OUT_DIR / f"{source_name}_aggregated_year_counts.csv", index=False)

    dim = len(next(iter(static.values())))
    loaded_years = sorted(set(int(y) for y in meta["year"].tolist())) if "year" in meta.columns else []
    rprint(
        f"Loaded aggregated {source_name} embeddings: {len(static)} basins, dim={dim}, "
        f"aggregation years={loaded_years if loaded_years else 'no year metadata'}, "
        f"l2_rows={l2_normalize_rows}, l2_final={l2_normalize_final}"
    )
    return static


def static_dict_is_yearly(static_dict) -> bool:
    """Return True if static_dict maps basin -> {year -> vector}."""
    if not static_dict:
        return False
    first_val = next(iter(static_dict.values()))
    return isinstance(first_val, dict)


def infer_static_dim(static_dict) -> int:
    """Infer static vector dimension for aggregated or yearly static dictionaries."""
    if not static_dict:
        raise RuntimeError("Cannot infer static dimension from empty static_dict.")
    first_val = next(iter(static_dict.values()))
    if isinstance(first_val, dict):
        if not first_val:
            raise RuntimeError("Cannot infer static dimension from empty yearly static entry.")
        first_vec = next(iter(first_val.values()))
        return int(np.asarray(first_vec).reshape(-1).shape[0])
    return int(np.asarray(first_val).reshape(-1).shape[0])


def get_train_years() -> List[int]:
    return list(range(pd.Timestamp(TRAIN_START).year, pd.Timestamp(TRAIN_END).year + 1))


def has_static_vector_for_sample(static_dict, sid: str, target_year: int) -> bool:
    if sid not in static_dict:
        return False
    val = static_dict[sid]
    if isinstance(val, dict):
        return int(target_year) in val
    return True


def get_static_vector_for_sample(static_dict, sid: str, target_year: int) -> np.ndarray:
    val = static_dict[sid]
    if isinstance(val, dict):
        return val[int(target_year)]
    return val


def annual_rows_to_yearly_static(
    meta: pd.DataFrame,
    emb: np.ndarray,
    source_name: str,
    years_keep: List[int],
    l2_normalize_rows: bool = False,
):
    """
    Convert annual embedding rows to a yearly static dictionary:
        static[site_id][year] = vector

    Each sample receives the Basin2Vec embedding corresponding to its target year.
    For LEAD=0 and H=1, the target year is the same as the issue/current-day year.
    """
    meta = meta.copy()
    valid_site = meta["site_id"].notna().to_numpy()
    meta = meta.loc[valid_site].reset_index(drop=True)
    emb = emb[valid_site]

    if years_keep is not None and "year" in meta.columns:
        keep = meta["year"].isin([int(y) for y in years_keep]).to_numpy()
        meta = meta.loc[keep].reset_index(drop=True)
        emb = emb[keep]

    if len(meta) == 0:
        raise RuntimeError(f"No {source_name} rows found after year/site filtering. years_keep={years_keep}")

    emb = np.nan_to_num(emb.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if l2_normalize_rows:
        emb = l2_normalize(emb)

    meta["row"] = np.arange(len(meta))
    yearly_static: Dict[str, Dict[int, np.ndarray]] = {}
    counts = []

    for (sid, year), sub in meta.groupby(["site_id", "year"]):
        idx = sub["row"].to_numpy()
        z = emb[idx].mean(axis=0).astype(np.float32)
        sid = str(sid)
        year = int(year)
        yearly_static.setdefault(sid, {})[year] = z
        counts.append({
            "site_id": sid,
            "year": year,
            f"n_{source_name}_rows": int(len(idx)),
            "dim": int(z.shape[0]),
        })

    if is_rank0():
        pd.DataFrame(counts).to_csv(OUT_DIR / f"{source_name}_yearly_embedding_counts.csv", index=False)

    n_basin_years = int(sum(len(v) for v in yearly_static.values()))
    dim = infer_static_dim(yearly_static)
    loaded_years = sorted({int(y) for d in yearly_static.values() for y in d.keys()})
    rprint(
        f"Loaded yearly {source_name} embeddings: {len(yearly_static)} basins, "
        f"{n_basin_years} basin-years, dim={dim}, years={loaded_years}, "
        f"l2_rows={l2_normalize_rows}"
    )
    return yearly_static


def load_basin2vec_static_embeddings(npz_path: Path, index_parquet: Path, years_keep: List[int]):
    data = np.load(npz_path, allow_pickle=True)
    rprint("Basin2Vec NPZ:", npz_path)
    rprint("Basin2Vec keys:", data.files)

    if BASIN2VEC_ARRAY_KEY in data.files:
        emb = np.asarray(data[BASIN2VEC_ARRAY_KEY], dtype=np.float32)
    elif "hydrologic_repr" in data.files:
        emb = np.asarray(data["hydrologic_repr"], dtype=np.float32)
    elif "embeddings" in data.files:
        emb = np.asarray(data["embeddings"], dtype=np.float32)
    else:
        raise KeyError(f"No usable Basin2Vec embedding array found in {npz_path}")

    if "labels" not in data.files or "years" not in data.files:
        raise KeyError("Basin2Vec NPZ must contain labels and years.")

    labels = np.asarray(data["labels"])
    years = np.asarray(data["years"]).astype(int)
    if len(labels) != len(emb) or len(years) != len(emb):
        raise ValueError("Basin2Vec metadata length mismatch.")

    meta = pd.DataFrame({
        "site_id": [normalize_site_id(x) for x in labels],
        "year": years,
    })
    mode = str(BASIN2VEC_TEMPORAL_MODE).strip().lower()
    if mode in {"aggregated", "avg", "average", "basin"}:
        return aggregate_annual_rows_to_static(
            meta, emb, "basin2vec", years_keep,
            l2_normalize_rows=False,
            l2_normalize_final=False,
        )
    if mode in {"yearly", "annual", "per_year", "target_year"}:
        return annual_rows_to_yearly_static(
            meta, emb, "basin2vec", years_keep,
            l2_normalize_rows=False,
        )
    raise ValueError(
        f"Unknown BASIN2VEC_TEMPORAL_MODE={BASIN2VEC_TEMPORAL_MODE!r}. "
        "Use 'aggregated' or 'yearly'."
    )


def load_alphaearth_aggregated_embeddings(npz_path: Path, years_keep: List[int]):
    data = np.load(npz_path, allow_pickle=True)
    rprint("AlphaEarth NPZ:", npz_path)
    rprint("AlphaEarth keys:", data.files)

    if "embeddings" in data.files:
        emb = np.asarray(data["embeddings"], dtype=np.float32)
    elif "raw_embeddings" in data.files:
        emb = np.asarray(data["raw_embeddings"], dtype=np.float32)
    else:
        raise KeyError(f"No AlphaEarth embedding key found in {npz_path}")

    if "site_ids" in data.files:
        site_ids = np.asarray(data["site_ids"])
    elif "labels" in data.files:
        site_ids = np.asarray(data["labels"])
    else:
        raise KeyError("AlphaEarth NPZ must contain site_ids or labels.")
    if "years" not in data.files:
        raise KeyError("AlphaEarth NPZ must contain years.")

    meta = pd.DataFrame({
        "site_id": [normalize_site_id(x) for x in site_ids],
        "year": np.asarray(data["years"]).astype(int),
    })
    return aggregate_annual_rows_to_static(meta, emb, "alphaearth", years_keep)


def load_static_npz_embeddings(npz_path: Path, source_name: str, years_keep: Optional[List[int]] = None):
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"{source_name} NPZ not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    rprint(f"{source_name} NPZ:", npz_path)
    rprint(f"{source_name} keys:", data.files)

    if "embeddings" in data.files:
        emb = np.asarray(data["embeddings"], dtype=np.float32)
    elif "X" in data.files:
        emb = np.asarray(data["X"], dtype=np.float32)
    elif "raw_embeddings" in data.files:
        emb = np.asarray(data["raw_embeddings"], dtype=np.float32)
    else:
        raise KeyError(f"No usable embedding array found in {npz_path}. Expected embeddings, X, or raw_embeddings.")

    if "site_ids" in data.files:
        site_ids = np.asarray(data["site_ids"])
    elif "basin_ids" in data.files:
        site_ids = np.asarray(data["basin_ids"])
    elif "labels" in data.files:
        site_ids = np.asarray(data["labels"])
    else:
        raise KeyError(f"{source_name} NPZ must contain site_ids, basin_ids, or labels.")

    if len(site_ids) != len(emb):
        raise ValueError(f"{source_name} metadata length mismatch: site_ids={len(site_ids)}, embeddings={len(emb)}")

    meta = pd.DataFrame({"site_id": [normalize_site_id(x) for x in site_ids]})
    if "years" in data.files:
        years = np.asarray(data["years"]).astype(int)
        if len(years) != len(emb):
            raise ValueError(f"{source_name} years length mismatch: years={len(years)}, embeddings={len(emb)}")
        meta["year"] = years
        return aggregate_annual_rows_to_static(meta, emb, source_name, years_keep)
    else:
        meta["year"] = -1
        return aggregate_annual_rows_to_static(meta, emb, source_name, None)

# COMMON BASINS / K-FOLD SPATIAL SPLIT / NORMALIZATION
# ==========================================================

def restrict_to_common_basins(basin_data, static_sources):
    csv_basins = set(basin_data.keys())
    source_sets = {name: set(obj.keys()) for name, obj in static_sources.items()}

    common = set(csv_basins)
    for ids in source_sets.values():
        common &= ids
    common_basins = sorted(common)

    rprint("\n" + "=" * 80)
    rprint("Common-basin filtering within all available CAMELS-US basins")
    rprint("=" * 80)
    rprint("CAMELS dynamic basins:", len(csv_basins))
    for name, ids in source_sets.items():
        rprint(f"{name} basins:", len(ids))
    rprint("Common basins used:", len(common_basins))

    if len(common_basins) == 0:
        raise RuntimeError("No common basins found across requested static sources.")

    if is_rank0():
        rows = [{"source": "camels_dynamic_all", "available_basins": len(csv_basins), "common_eval_basins": len(common_basins)}]
        for name, ids in source_sets.items():
            rows.append({"source": name, "available_basins": len(ids), "common_eval_basins": len(common_basins)})
        pd.DataFrame(rows).to_csv(OUT_DIR / "static_source_common_basin_coverage.csv", index=False)
        pd.DataFrame({"site_id": common_basins}).to_csv(OUT_DIR / "common_eval_basins_671_all_sources.csv", index=False)

    filtered_data = {sid: df for sid, df in basin_data.items() if sid in common_basins}
    filtered_static = {
        name: {sid: z for sid, z in obj.items() if sid in common_basins}
        for name, obj in static_sources.items()
    }
    return filtered_data, filtered_static, common_basins


def make_stratified_spatial_folds(common_basins: List[str], k: int, seed: int) -> List[List[str]]:
    """Make k basin folds, approximately stratified by CAMELS HUC-02 metadata."""
    meta = load_camels_huc_metadata()
    meta = meta[meta["site_id"].isin(common_basins)].copy()
    common_set = set(common_basins)
    missing = sorted(common_set - set(meta["site_id"].tolist()))
    if missing:
        extra = pd.DataFrame({"site_id": missing, "huc": -1})
        meta = pd.concat([meta[["site_id", "huc"]], extra], ignore_index=True)

    rng = np.random.default_rng(int(seed))
    folds = [[] for _ in range(k)]

    if STRATIFY_FOLDS_BY_HUC and "huc" in meta.columns:
        for _, sub in meta.groupby("huc"):
            ids = np.asarray(sorted(sub["site_id"].tolist()), dtype=object)
            rng.shuffle(ids)
            for i, sid in enumerate(ids):
                folds[i % k].append(str(sid))
    else:
        ids = np.asarray(sorted(common_basins), dtype=object)
        rng.shuffle(ids)
        for i, sid in enumerate(ids):
            folds[i % k].append(str(sid))

    return [sorted(f) for f in folds]


def date_windows_overlap(start_a, end_a, start_b, end_b) -> bool:
    """Return True when two inclusive date windows overlap."""
    start_a = pd.Timestamp(start_a)
    end_a = pd.Timestamp(end_a)
    start_b = pd.Timestamp(start_b)
    end_b = pd.Timestamp(end_b)
    return max(start_a, start_b) <= min(end_a, end_b)


def make_kfold_spatial_split(common_basins: List[str], fold_id: int, folds: List[List[str]]):
    """
    Kratzert-style ungauged-basin split, Option A.

    For each outer fold:
      - test_basins = one held-out basin fold
      - train_basins = all remaining basin folds
      - val_basins = all remaining basin folds

    Train and validation use the same non-test basins but different target
    periods. Test basins are fully held out from training and validation.
    """
    common_set = set(common_basins)
    test_basins = sorted([sid for sid in folds[fold_id] if sid in common_set])
    remaining = sorted(common_set - set(test_basins))

    train_basins = remaining
    val_basins = remaining
    excluded = sorted(common_set - set(train_basins) - set(test_basins))

    # Train/validation basin overlap is intentional in Option A. Test basins
    # must be completely unseen during training and validation.
    if set(train_basins) & set(test_basins):
        raise RuntimeError("Spatial split leakage: train and test basin sets overlap.")
    if set(val_basins) & set(test_basins):
        raise RuntimeError("Spatial split leakage: validation and test basin sets overlap.")
    if not train_basins or not val_basins or not test_basins:
        raise RuntimeError(
            f"Bad fold {fold_id}: train={len(train_basins)}, "
            f"val={len(val_basins)}, test={len(test_basins)}"
        )

    # Because train and validation use the same basins, the date windows must
    # be disjoint so validation is not simply training samples repeated.
    if date_windows_overlap(TRAIN_START, TRAIN_END, VAL_START, VAL_END):
        raise RuntimeError(
            "TRAIN and VAL date windows overlap. In Kratzert-style Option A, "
            "train/validation basins are the same non-test basins, but target "
            "periods must be disjoint for early stopping."
        )

    rprint("\n" + "=" * 80)
    rprint("CAMELS-671 Kratzert-style ungauged basin K-fold split, Option A")
    rprint("=" * 80)
    rprint("Fold:", fold_id)
    rprint("Total common basins:", len(common_basins))
    rprint("Train basins:", len(train_basins), "(all non-test basins)")
    rprint("Validation basins:", len(val_basins), "(same non-test basins; validation period)")
    rprint("Test basins:", len(test_basins), "(held-out unseen basin fold)")
    rprint("Excluded basins:", len(excluded))
    rprint("Train dates:", TRAIN_START, "to", TRAIN_END)
    rprint("Validation dates:", VAL_START, "to", VAL_END)
    rprint("Test dates:", TEST_START, "to", TEST_END)

    if is_rank0():
        split_dir = OUT_DIR / "splits"
        split_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        rows += [
            {
                "fold_id": int(fold_id),
                "site_id": sid,
                "spatial_split": "train_and_validation_non_test_basin",
                "train_period": f"{TRAIN_START}..{TRAIN_END}",
                "validation_period": f"{VAL_START}..{VAL_END}",
            }
            for sid in train_basins
        ]
        rows += [
            {
                "fold_id": int(fold_id),
                "site_id": sid,
                "spatial_split": "test_unseen_basin_fold",
                "test_period": f"{TEST_START}..{TEST_END}",
            }
            for sid in test_basins
        ]
        pd.DataFrame(rows).to_csv(
            split_dir / f"camels_671_kratzert_optionA_fold{fold_id}.csv",
            index=False,
        )

    return train_basins, val_basins, test_basins, excluded


def fit_train_scaler(basin_data, train_basin_ids):
    x_rows = []
    y_rows = []
    train_start = pd.Timestamp(TRAIN_START)
    train_end = pd.Timestamp(TRAIN_END)
    train_basin_ids = set(train_basin_ids)

    for sid, df in basin_data.items():
        if sid not in train_basin_ids:
            continue
        sub = df[(df[DATE_COL] >= train_start) & (df[DATE_COL] <= train_end)].copy()
        if len(sub) == 0:
            continue
        x_rows.append(sub[DYNAMIC_COLS].to_numpy(dtype=np.float32))
        y_rows.append(sub[[TARGET_COL]].to_numpy(dtype=np.float32))

    if not x_rows or not y_rows:
        raise RuntimeError("Cannot fit scaler: no training rows.")

    X = np.vstack(x_rows)
    Y = np.vstack(y_rows)
    scaler = {
        "x_mean": np.nanmean(X, axis=0).astype(np.float32),
        "x_std": np.nanstd(X, axis=0).astype(np.float32),
        "y_mean": np.nanmean(Y, axis=0).astype(np.float32),
        "y_std": np.nanstd(Y, axis=0).astype(np.float32),
    }
    scaler["x_std"] = np.where(scaler["x_std"] < 1e-6, 1.0, scaler["x_std"])
    scaler["y_std"] = np.where(scaler["y_std"] < 1e-6, 1.0, scaler["y_std"])
    return scaler


def apply_scaler(basin_data, scaler):
    out = {}
    for sid, df in basin_data.items():
        d = df.copy()
        d[DYNAMIC_COLS] = (d[DYNAMIC_COLS].to_numpy(dtype=np.float32) - scaler["x_mean"]) / scaler["x_std"]
        d[TARGET_COL] = (d[[TARGET_COL]].to_numpy(dtype=np.float32) - scaler["y_mean"]) / scaler["y_std"]
        out[sid] = d
    return out


def inverse_target(arr, scaler):
    arr = np.asarray(arr, dtype=np.float32)
    if scaler is None:
        return arr
    return arr * np.asarray(scaler["y_std"], dtype=np.float32).reshape(1, 1) + np.asarray(
        scaler["y_mean"], dtype=np.float32
    ).reshape(1, 1)


def standardize_static_source(static_dict, source_name, fit_basins, transform_basins, fit_years: Optional[List[int]] = None):
    """
    Z-score static features using training basins only.

    Supports both:
      1) aggregated static_dict: {site_id -> vector}
      2) yearly static_dict:     {site_id -> {year -> vector}}

    For yearly Basin2Vec, the scaler is fitted on training basins and training
    years only, then applied to all available basin-years.
    """
    fit_basins = sorted(set(fit_basins))
    transform_basins = sorted(set(transform_basins))
    fit_keys = [sid for sid in fit_basins if sid in static_dict]
    all_keys = [sid for sid in transform_basins if sid in static_dict]

    if len(fit_keys) == 0:
        raise RuntimeError(f"No fit keys for static source: {source_name}")
    if len(all_keys) == 0:
        raise RuntimeError(f"No transform keys for static source: {source_name}")

    yearly = static_dict_is_yearly(static_dict)

    if yearly:
        fit_years_set = set(int(y) for y in fit_years) if fit_years is not None else None
        fit_vectors = []
        for sid in fit_keys:
            for year, z in static_dict[sid].items():
                if fit_years_set is not None and int(year) not in fit_years_set:
                    continue
                fit_vectors.append(np.asarray(z, dtype=np.float32).reshape(-1))

        if len(fit_vectors) == 0:
            raise RuntimeError(
                f"No yearly fit vectors for static source={source_name}; "
                f"fit_years={sorted(fit_years_set) if fit_years_set is not None else 'all'}"
            )

        M = np.vstack(fit_vectors).astype(np.float32)
        mean = M.mean(axis=0, keepdims=True)
        std = M.std(axis=0, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)

        out = {}
        for sid in all_keys:
            out[sid] = {}
            for year, z in static_dict[sid].items():
                zz = (np.asarray(z, dtype=np.float32).reshape(1, -1) - mean) / std
                out[sid][int(year)] = zz[0].astype(np.float32)

        if is_rank0():
            save_json({
                "source_name": source_name,
                "static_temporal_mode": "yearly",
                "n_fit_basins": len(fit_keys),
                "n_transform_basins": len(all_keys),
                "n_fit_basin_year_vectors": len(fit_vectors),
                "fit_years": sorted(fit_years_set) if fit_years_set is not None else "all",
                "fit_scope": "training basins and training years only; validation/test basin-years transformed with train-fitted scaler",
            }, OUT_DIR / f"static_standardization_{source_name}.json")
        return out

    M = np.vstack([static_dict[k] for k in fit_keys]).astype(np.float32)
    mean = M.mean(axis=0, keepdims=True)
    std = M.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    out = {k: ((static_dict[k][None, :] - mean) / std)[0].astype(np.float32) for k in all_keys}

    if is_rank0():
        save_json({
            "source_name": source_name,
            "static_temporal_mode": "aggregated",
            "n_fit_basins": len(fit_keys),
            "n_transform_basins": len(all_keys),
            "fit_scope": "non-test train basins only; validation uses same basins but later time period",
        }, OUT_DIR / f"static_standardization_{source_name}.json")
    return out

# DATASET
# ==========================================================

class StreamflowWindowDataset(Dataset):
    def __init__(
        self,
        basin_data,
        static_dict,
        static_source,
        seq_len,
        horizon,
        split,
        start_date,
        end_date,
        basin_ids=None,
    ):
        self.basin_data = {}
        self.static_dict = static_dict
        self.static_source = static_source
        self.samples = []
        self.seq_len = int(seq_len)
        self.horizon = int(horizon)
        self.lead = int(LEAD)
        self.split = split

        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
        basin_ids = set(basin_ids) if basin_ids is not None else None

        for sid, df in basin_data.items():
            if basin_ids is not None and sid not in basin_ids:
                continue
            if sid not in static_dict:
                continue

            X = df[DYNAMIC_COLS].to_numpy(dtype=np.float32)
            y = df[TARGET_COL].to_numpy(dtype=np.float32)
            dates = pd.to_datetime(df[DATE_COL]).reset_index(drop=True)
            ord_days = dates.map(pd.Timestamp.toordinal).to_numpy(dtype=np.int64)

            valid_x = np.isfinite(X).all(axis=1)
            valid_y = np.isfinite(y)
            cx = np.concatenate([[0], np.cumsum(valid_x.astype(np.int32))])
            cy = np.concatenate([[0], np.cumsum(valid_y.astype(np.int32))])

            n = len(df)
            if n < self.seq_len + self.lead + self.horizon:
                continue

            self.basin_data[sid] = {
                "X": X,
                "y": y,
                "cx": cx,
                "cy": cy,
                "dates": dates,
                "ord_days": ord_days,
            }

            first_issue_t = self.seq_len - 1
            last_issue_t = n - 1 - self.lead - self.horizon + 1

            for issue_t in range(first_issue_t, last_issue_t + 1):
                target_start_t = issue_t + self.lead
                target_end_t = target_start_t + self.horizon - 1

                issue_date = pd.Timestamp(dates.iloc[issue_t])
                target_start_date = pd.Timestamp(dates.iloc[target_start_t])
                target_end_date = pd.Timestamp(dates.iloc[target_end_t])
                target_year = int(target_start_date.year)

                if not has_static_vector_for_sample(self.static_dict, sid, target_year):
                    continue

                if target_start_date < start_date or target_end_date > end_date:
                    continue
                if int(target_end_date.year) != target_year:
                    continue

                x0 = issue_t - self.seq_len + 1
                x1 = issue_t + 1
                y0 = target_start_t
                y1 = target_end_t + 1

                if cx[x1] - cx[x0] != self.seq_len:
                    continue
                if cy[y1] - cy[y0] != self.horizon:
                    continue
                if ord_days[x1 - 1] - ord_days[x0] != self.seq_len - 1:
                    continue
                if ord_days[y1 - 1] - ord_days[y0] != self.horizon - 1:
                    continue

                self.samples.append((
                    sid,
                    issue_t,
                    target_start_t,
                    target_end_t,
                    issue_date.strftime("%Y-%m-%d"),
                    target_start_date.strftime("%Y-%m-%d"),
                    target_end_date.strftime("%Y-%m-%d"),
                    target_year,
                ))

        rprint(
            f"{split:16s} | source={static_source:10s} | seq_len={seq_len:3d} | "
            f"horizon={horizon:2d} | samples={len(self.samples):8d} | "
            f"basins={len(self.basin_data):4d} | dates={start_date.date()}..{end_date.date()}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid, issue_t, target_start_t, target_end_t, issue_date, target_start_date, target_end_date, target_year = self.samples[idx]
        obj = self.basin_data[sid]

        x0 = issue_t - self.seq_len + 1
        x1 = issue_t + 1
        y0 = target_start_t
        y1 = target_end_t + 1

        x_dyn = obj["X"][x0:x1].astype(np.float32)
        y = obj["y"][y0:y1].astype(np.float32)
        x_static = get_static_vector_for_sample(self.static_dict, sid, target_year)

        return (
            torch.from_numpy(x_dyn).float(),
            torch.from_numpy(x_static).float(),
            torch.from_numpy(y).float(),
            sid,
            issue_date,
            target_start_date,
            target_end_date,
            target_year,
        )

# MODEL REGISTRY
# ==========================================================

def sinusoid_1d(L, D, device):
    assert D % 2 == 0
    pos = torch.arange(L, device=device).float()
    i = torch.arange(D // 2, device=device).float()
    angles = pos[:, None] / (10000 ** (2 * i / D))
    pe = torch.zeros(L, D, device=device)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe


def causal_mask(L, device):
    return torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()


class StaticProjector(nn.Module):
    def __init__(self, static_size, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(static_size),
            nn.Linear(static_size, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
    def forward(self, x_static):
        return self.net(torch.nan_to_num(x_static.float(), nan=0.0, posinf=0.0, neginf=0.0))


class LSTMStaticForecastModel(nn.Module):
    def __init__(self, dynamic_size, static_size, output_size, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.static_proj = StaticProjector(static_size, hidden_size, dropout)
        self.lstm = nn.LSTM(dynamic_size + hidden_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_size, output_size))
    def forward(self, x_dyn, x_static):
        x_dyn = torch.nan_to_num(x_dyn.float(), nan=0.0, posinf=0.0, neginf=0.0)
        s = self.static_proj(x_static).unsqueeze(1).expand(-1, x_dyn.shape[1], -1)
        z, _ = self.lstm(torch.cat([x_dyn, s], dim=-1))
        return self.head(z[:, -1])


class EALSTMStaticForecastModel(nn.Module):
    """Entity-aware LSTM: static vector controls an input gate over dynamic forcings."""
    def __init__(self, dynamic_size, static_size, output_size, hidden_size=128, num_layers=1, dropout=0.1):
        super().__init__()
        self.input_gate = nn.Sequential(nn.LayerNorm(static_size), nn.Linear(static_size, dynamic_size), nn.Sigmoid())
        self.static_proj = StaticProjector(static_size, hidden_size, dropout)
        self.lstm = nn.LSTM(dynamic_size, hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size * 2), nn.Linear(hidden_size * 2, hidden_size), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_size, output_size))
    def forward(self, x_dyn, x_static):
        x_dyn = torch.nan_to_num(x_dyn.float(), nan=0.0, posinf=0.0, neginf=0.0)
        gate = self.input_gate(torch.nan_to_num(x_static.float(), nan=0.0)).unsqueeze(1)
        z, _ = self.lstm(x_dyn * gate)
        s = self.static_proj(x_static)
        return self.head(torch.cat([z[:, -1], s], dim=-1))


class StaticTokenTransformerForecastModel(nn.Module):
    def __init__(self, dynamic_size, static_size, d_model, n_heads, n_layers, d_ff, output_size, dropout=0.1, max_seq_len=366, causal=True, use_learned_position=True):
        super().__init__()
        self.d_model = int(d_model); self.max_seq_len = int(max_seq_len); self.causal = bool(causal); self.use_learned_position = bool(use_learned_position)
        self.dynamic_proj = nn.Sequential(nn.Linear(dynamic_size, d_model), nn.LayerNorm(d_model))
        self.static_proj = StaticProjector(static_size, d_model, dropout)
        if self.use_learned_position:
            self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model)); nn.init.normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.backbone = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, output_size))
        self.reset_parameters()
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, x_dyn, x_static):
        x_dyn = torch.nan_to_num(x_dyn.float(), nan=0.0, posinf=0.0, neginf=0.0)
        dyn_tokens = self.dynamic_proj(x_dyn)
        static_token = self.static_proj(x_static).unsqueeze(1)
        seq = torch.cat([static_token, dyn_tokens], dim=1)
        L = seq.shape[1]
        if L > self.max_seq_len: raise RuntimeError(f"Sequence length {L} exceeds max_seq_len={self.max_seq_len}")
        seq = seq + (self.pos_embed[:, :L, :] if self.use_learned_position else sinusoid_1d(L, self.d_model, x_dyn.device).unsqueeze(0))
        mask = causal_mask(L, x_dyn.device) if self.causal else None
        z = self.backbone(seq, mask=mask)
        return self.head(self.final_norm(z[:, -1]))


class TFTStyleForecastModel(nn.Module):
    """Compact TFT-style baseline: static enrichment + LSTM encoder + multi-head temporal attention."""
    def __init__(self, dynamic_size, static_size, output_size, d_model=128, n_heads=8, dropout=0.1):
        super().__init__()
        self.static_proj = StaticProjector(static_size, d_model, dropout)
        self.dynamic_proj = nn.Linear(dynamic_size, d_model)
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True)
        self.enrich = nn.Sequential(nn.LayerNorm(d_model * 2), nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout))
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, output_size))
    def forward(self, x_dyn, x_static):
        x_dyn = torch.nan_to_num(x_dyn.float(), nan=0.0, posinf=0.0, neginf=0.0)
        s = self.static_proj(x_static)
        z = self.dynamic_proj(x_dyn)
        z, _ = self.lstm(z)
        s_rep = s.unsqueeze(1).expand(-1, z.shape[1], -1)
        z = self.enrich(torch.cat([z, s_rep], dim=-1))
        q = z[:, -1:, :]
        a, _ = self.attn(q, z, z, need_weights=False)
        feat = self.norm(q.squeeze(1) + self.gate(a.squeeze(1)) * a.squeeze(1))
        return self.head(feat)


class TimeXerStyleForecastModel(nn.Module):
    """TimeXer-style exogenous baseline: temporal tokens plus explicit static/exogenous token cross-attention."""
    def __init__(self, dynamic_size, static_size, output_size, d_model=128, n_heads=8, n_layers=3, d_ff=512, dropout=0.1, max_seq_len=366):
        super().__init__()
        self.temporal = StaticTokenTransformerForecastModel(dynamic_size, static_size, d_model, n_heads, n_layers, d_ff, d_model, dropout, max_seq_len, causal=False)
        self.out = nn.Linear(d_model, output_size)
    def forward(self, x_dyn, x_static):
        return self.out(self.temporal(x_dyn, x_static))


class ITransformerForecastModel(nn.Module):
    """iTransformer-style baseline: each variable is a token whose features are its full temporal history."""
    def __init__(self, seq_len, dynamic_size, static_size, output_size, d_model=128, n_heads=8, n_layers=3, d_ff=512, dropout=0.1):
        super().__init__()
        self.var_proj = nn.Linear(seq_len, d_model)
        self.var_embed = nn.Parameter(torch.zeros(1, dynamic_size, d_model)); nn.init.normal_(self.var_embed, std=0.02)
        self.static_proj = StaticProjector(static_size, d_model, dropout)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model * 2), nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, output_size))
    def forward(self, x_dyn, x_static):
        x_dyn = torch.nan_to_num(x_dyn.float(), nan=0.0, posinf=0.0, neginf=0.0)
        tokens = self.var_proj(x_dyn.transpose(1, 2)) + self.var_embed
        z = self.encoder(tokens).mean(dim=1)
        s = self.static_proj(x_static)
        return self.head(torch.cat([z, s], dim=-1))


class TimeMixerForecastModel(nn.Module):
    """TimeMixer-style baseline with multi-scale temporal mixing and static conditioning."""
    def __init__(self, dynamic_size, static_size, output_size, d_model=128, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(dynamic_size, d_model)
        self.static_proj = StaticProjector(static_size, d_model, dropout)
        self.mix = nn.Sequential(nn.LayerNorm(d_model * 4), nn.Linear(d_model * 4, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model), nn.GELU())
        self.head = nn.Sequential(nn.LayerNorm(d_model * 2), nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, output_size))
    def forward(self, x_dyn, x_static):
        x_dyn = torch.nan_to_num(x_dyn.float(), nan=0.0, posinf=0.0, neginf=0.0)
        z = self.proj(x_dyn).transpose(1, 2)  # B,D,T
        feats = [z[:, :, -1]]
        for k in [7, 30, 90]:
            kk = min(k, z.shape[-1])
            feats.append(torch.nn.functional.avg_pool1d(z, kernel_size=kk, stride=1)[:, :, -1])
        h = self.mix(torch.cat(feats, dim=-1))
        s = self.static_proj(x_static)
        return self.head(torch.cat([h, s], dim=-1))


class MoiraiStylePatchForecastModel(nn.Module):
    """Moirai-style supervised patch Transformer baseline. This is not the official pretrained Moirai checkpoint."""
    def __init__(self, seq_len, dynamic_size, static_size, output_size, d_model=128, n_heads=8, n_layers=3, d_ff=512, dropout=0.1, patch_len=16, patch_stride=8):
        super().__init__()
        self.patch_len = int(patch_len); self.patch_stride = int(patch_stride); self.dynamic_size = int(dynamic_size)
        self.patch_proj = nn.Linear(self.patch_len * dynamic_size, d_model)
        self.static_proj = StaticProjector(static_size, d_model, dropout)
        n_patches = max(1, (seq_len - self.patch_len) // self.patch_stride + 1)
        self.pos = nn.Parameter(torch.zeros(1, n_patches + 1, d_model)); nn.init.normal_(self.pos, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, output_size))
    def forward(self, x_dyn, x_static):
        x_dyn = torch.nan_to_num(x_dyn.float(), nan=0.0, posinf=0.0, neginf=0.0)
        B, T, C = x_dyn.shape
        if T < self.patch_len:
            pad = self.patch_len - T
            x_dyn = torch.nn.functional.pad(x_dyn, (0, 0, pad, 0))
        patches = x_dyn.unfold(dimension=1, size=self.patch_len, step=self.patch_stride)  # B,N,C,P
        patches = patches.permute(0, 1, 3, 2).reshape(B, patches.shape[1], self.patch_len * C)
        tokens = self.patch_proj(patches)
        static_token = self.static_proj(x_static).unsqueeze(1)
        seq = torch.cat([static_token, tokens], dim=1)
        seq = seq + self.pos[:, :seq.shape[1], :]
        z = self.encoder(seq)
        return self.head(z[:, 0])


def create_model(model_name, dynamic_size, static_size, output_size, seq_len):
    name = model_name.lower()
    if name == "lstm":
        return LSTMStaticForecastModel(dynamic_size, static_size, output_size, LSTM_HIDDEN, LSTM_LAYERS, DROPOUT)
    if name == "ealstm":
        return EALSTMStaticForecastModel(dynamic_size, static_size, output_size, LSTM_HIDDEN, 1, DROPOUT)
    if name == "tft":
        return TFTStyleForecastModel(dynamic_size, static_size, output_size, D_MODEL, N_HEADS, DROPOUT)
    if name == "timerxl":
        return StaticTokenTransformerForecastModel(dynamic_size, static_size, D_MODEL, N_HEADS, N_LAYERS, D_FF, output_size, DROPOUT, max(seq_len + 1, MAX_SEQ_LEN), causal=True, use_learned_position=USE_LEARNED_POSITION)
    if name == "timexer":
        return TimeXerStyleForecastModel(dynamic_size, static_size, output_size, D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT, max(seq_len + 1, MAX_SEQ_LEN))
    if name == "itransformer":
        return ITransformerForecastModel(seq_len, dynamic_size, static_size, output_size, D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT)
    if name == "timemixer":
        return TimeMixerForecastModel(dynamic_size, static_size, output_size, D_MODEL, DROPOUT)
    if name == "moirai":
        return MoiraiStylePatchForecastModel(seq_len, dynamic_size, static_size, output_size, D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT, PATCH_LEN, PATCH_STRIDE)
    raise ValueError(f"Unknown model_name={model_name}. Use one of {MODEL_NAMES}")


def get_model_core(model):
    return model.module if isinstance(model, DDP) else model

# ==========================================================
# METRICS
# ==========================================================


def nse(obs, pred):
    obs = np.asarray(obs, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    valid = np.isfinite(obs) & np.isfinite(pred)
    obs = obs[valid]
    pred = pred[valid]
    if len(obs) < 2:
        return np.nan
    denom = np.sum((obs - np.mean(obs)) ** 2)
    if denom <= 1e-12:
        return np.nan
    return float(1.0 - np.sum((obs - pred) ** 2) / denom)


def rmse(obs, pred):
    obs = np.asarray(obs, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    valid = np.isfinite(obs) & np.isfinite(pred)
    if valid.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((obs[valid] - pred[valid]) ** 2)))


def mae(obs, pred):
    obs = np.asarray(obs, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    valid = np.isfinite(obs) & np.isfinite(pred)
    if valid.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(obs[valid] - pred[valid])))


def pearson_r(obs, pred):
    obs = np.asarray(obs, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    valid = np.isfinite(obs) & np.isfinite(pred)
    obs = obs[valid]
    pred = pred[valid]
    if len(obs) < 2:
        return np.nan
    if np.std(obs) < 1e-12 or np.std(pred) < 1e-12:
        return np.nan
    return float(np.corrcoef(obs, pred)[0, 1])


def kge(obs, pred):
    obs = np.asarray(obs, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    valid = np.isfinite(obs) & np.isfinite(pred)
    obs = obs[valid]
    pred = pred[valid]
    if len(obs) < 2:
        return np.nan

    r = pearson_r(obs, pred)
    obs_std = np.std(obs)
    pred_std = np.std(pred)
    obs_mean = np.mean(obs)
    pred_mean = np.mean(pred)

    if obs_std < 1e-12 or abs(obs_mean) < 1e-12:
        return np.nan

    alpha = pred_std / obs_std
    beta = pred_mean / obs_mean

    if not np.isfinite(r) or not np.isfinite(alpha) or not np.isfinite(beta):
        return np.nan

    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def nanmean_safe(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else np.nan


def nanmedian_safe(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else np.nan


def compute_global_metrics(y_true, y_pred, horizon):
    rows = []
    for lead in range(horizon):
        obs = y_true[:, lead]
        pred = y_pred[:, lead]
        rows.append({
            "lead": int(lead),
            "horizon": int(horizon),
            "NSE": nse(obs, pred),
            "KGE": kge(obs, pred),
            "RMSE": rmse(obs, pred),
            "MAE": mae(obs, pred),
            "Pearson_r": pearson_r(obs, pred),
            "n_samples": int(np.isfinite(obs).sum()),
        })
    return pd.DataFrame(rows)


def compute_per_basin_metrics(y_true, y_pred, site_ids, horizon):
    site_ids = np.asarray(site_ids).astype(str)
    rows = []

    for sid in sorted(np.unique(site_ids)):
        idx = site_ids == sid
        for lead in range(horizon):
            obs = y_true[idx, lead]
            pred = y_pred[idx, lead]
            rows.append({
                "site_id": sid,
                "lead": int(lead),
                "horizon": int(horizon),
                "NSE": nse(obs, pred),
                "KGE": kge(obs, pred),
                "RMSE": rmse(obs, pred),
                "MAE": mae(obs, pred),
                "Pearson_r": pearson_r(obs, pred),
                "n_samples": int(idx.sum()),
                "obs_std": float(np.nanstd(obs)) if np.isfinite(obs).sum() >= 2 else np.nan,
                "obs_mean": float(np.nanmean(obs)),
            })

    return pd.DataFrame(rows)


def summarize_per_basin_metrics(per_basin_df, horizon):
    rows = []
    for lead, d in per_basin_df.groupby("lead"):
        rows.append({
            "lead": int(lead),
            "horizon": int(horizon),
            "median_NSE": nanmedian_safe(d["NSE"]),
            "mean_NSE": nanmean_safe(d["NSE"]),
            "median_KGE": nanmedian_safe(d["KGE"]),
            "mean_KGE": nanmean_safe(d["KGE"]),
            "median_RMSE": nanmedian_safe(d["RMSE"]),
            "mean_RMSE": nanmean_safe(d["RMSE"]),
            "median_MAE": nanmedian_safe(d["MAE"]),
            "mean_MAE": nanmean_safe(d["MAE"]),
            "median_Pearson_r": nanmedian_safe(d["Pearson_r"]),
            "mean_Pearson_r": nanmean_safe(d["Pearson_r"]),
            "n_basins": int(d["site_id"].nunique()),
            "n_basins_valid_NSE": int(np.isfinite(d["NSE"]).sum()),
            "n_basins_valid_KGE": int(np.isfinite(d["KGE"]).sum()),
        })
    return pd.DataFrame(rows)


# ==========================================================
# TRAIN / EVAL
# ==========================================================

def make_loader(ds, batch_size, shuffle, sampler):
    kwargs = dict(
        dataset=ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=NUM_WORKERS_PER_GPU,
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=False,
        persistent_workers=NUM_WORKERS_PER_GPU > 0,
    )
    if NUM_WORKERS_PER_GPU > 0:
        kwargs["prefetch_factor"] = PREFETCH_FACTOR
    return DataLoader(**kwargs)


def train_one_epoch(model, loader, optimizer, criterion, scaler, epoch):
    model.train()

    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)

    total_loss = 0.0
    total_batches = 0

    iterator = tqdm(loader, desc="train", leave=False, disable=not is_rank0())

    for x_dyn, x_static, y, *_ in iterator:
        x_dyn = x_dyn.to(DEVICE, non_blocking=True)
        x_static = x_static.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if USE_AMP and DEVICE.type == "cuda":
            with autocast_context():
                yhat = model(x_dyn, x_static)
                loss = criterion(yhat, y)
            scaler.scale(loss).backward()
            if GRAD_CLIP is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            yhat = model(x_dyn, x_static)
            loss = criterion(yhat, y)
            loss.backward()
            if GRAD_CLIP is not None:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1

    local = torch.tensor([total_loss, total_batches], dtype=torch.float64, device=DEVICE)
    if DISTRIBUTED:
        dist.all_reduce(local, op=dist.ReduceOp.SUM)

    return float(local[0].item() / max(local[1].item(), 1.0))


@torch.no_grad()
def evaluate_loss_distributed(model, loader):
    model.eval()

    mse_sum = 0.0
    n_elem = 0

    iterator = tqdm(loader, desc="val", leave=False, disable=not is_rank0())

    for x_dyn, x_static, y, *_ in iterator:
        x_dyn = x_dyn.to(DEVICE, non_blocking=True)
        x_static = x_static.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        if USE_AMP and DEVICE.type == "cuda":
            with autocast_context():
                yhat = model(x_dyn, x_static)
        else:
            yhat = model(x_dyn, x_static)

        mse_sum += float(torch.sum((yhat - y) ** 2).item())
        n_elem += int(y.numel())

    local = torch.tensor([mse_sum, n_elem], dtype=torch.float64, device=DEVICE)
    if DISTRIBUTED:
        dist.all_reduce(local, op=dist.ReduceOp.SUM)

    return float(local[0].item() / max(local[1].item(), 1.0))


@torch.no_grad()
def evaluate_full_rank0(model, loader, horizon, scaler_obj=None):
    model.eval()

    y_true_all = []
    y_pred_all = []
    site_all = []
    issue_date_all = []
    target_start_date_all = []
    target_end_date_all = []
    year_all = []

    iterator = tqdm(loader, desc=f"eval horizon {horizon}", leave=False)

    for x_dyn, x_static, y, site_ids, issue_dates, target_start_dates, target_end_dates, years in iterator:
        x_dyn = x_dyn.to(DEVICE, non_blocking=True)
        x_static = x_static.to(DEVICE, non_blocking=True)

        if USE_AMP and DEVICE.type == "cuda":
            with autocast_context():
                yhat = model(x_dyn, x_static)
        else:
            yhat = model(x_dyn, x_static)

        y_true_all.append(y.numpy())
        y_pred_all.append(yhat.detach().cpu().numpy())
        site_all.extend(list(site_ids))
        issue_date_all.extend(list(issue_dates))
        target_start_date_all.extend(list(target_start_dates))
        target_end_date_all.extend(list(target_end_dates))
        year_all.extend([int(y) for y in years])

    y_true_scaled = np.vstack(y_true_all)
    y_pred_scaled = np.vstack(y_pred_all)

    y_true = inverse_target(y_true_scaled, scaler_obj)
    y_pred = inverse_target(y_pred_scaled, scaler_obj)

    global_metrics = compute_global_metrics(y_true, y_pred, horizon)
    per_basin = compute_per_basin_metrics(y_true, y_pred, site_all, horizon)
    per_basin_summary = summarize_per_basin_metrics(per_basin, horizon)

    pred_df = pd.DataFrame({
        "site_id": site_all,
        "issue_date": issue_date_all,
        "target_start_date": target_start_date_all,
        "target_end_date": target_end_date_all,
        "target_year": year_all,
        "horizon": int(horizon),
    })

    for lead in range(horizon):
        pred_df[f"observed_lead{lead}"] = y_true[:, lead]
        pred_df[f"predicted_lead{lead}"] = y_pred[:, lead]
        pred_df[f"residual_lead{lead}"] = y_pred[:, lead] - y_true[:, lead]
        pred_df[f"observed_scaled_lead{lead}"] = y_true_scaled[:, lead]
        pred_df[f"predicted_scaled_lead{lead}"] = y_pred_scaled[:, lead]

    return global_metrics, per_basin, per_basin_summary, pred_df



def add_experiment_metadata(
    df,
    model_name,
    static_name,
    seq_len,
    horizon,
    best_epoch,
    best_val,
    static_dim,
    n_train,
    n_val,
    n_test,
    n_eval,
    n_basins,
    fold_id,
    n_train_basins,
    n_val_basins,
    n_test_basins,
):
    df = df.copy()
    df["fold_id"] = int(fold_id)
    df["model_name"] = model_name
    df["static_source"] = static_name
    df["seq_len"] = int(seq_len)
    df["lead_mode"] = "lead0_horizon1_same_day_simulation"
    df["horizon"] = int(horizon)
    df["eval_split"] = "test_unseen_basins"

    df["split_protocol"] = "kratzert_style_ungauged_optionA"
    df["validation_protocol"] = "same_non_test_basins_disjoint_time_period"

    df["best_epoch"] = int(best_epoch)
    df["best_val_loss"] = float(best_val)
    df["static_dim"] = int(static_dim)
    df["n_train_samples"] = int(n_train)
    df["n_val_samples_temporal_validation"] = int(n_val)
    df["n_test_samples_unseen_basins"] = int(n_test)
    df["n_eval_samples"] = int(n_eval)
    df["n_basins"] = int(n_basins)
    df["n_train_basins"] = int(n_train_basins)
    df["n_val_basins"] = int(n_val_basins)
    df["n_test_basins"] = int(n_test_basins)
    df["train_start"] = TRAIN_START
    df["train_end"] = TRAIN_END
    df["val_start"] = VAL_START
    df["val_end"] = VAL_END
    df["test_start"] = TEST_START
    df["test_end"] = TEST_END

    front = [
        "fold_id", "model_name", "static_source", "seq_len", "lead_mode", "horizon", "lead",
        "eval_split", "split_protocol", "validation_protocol", "best_epoch", "best_val_loss",
        "static_dim", "n_train_samples", "n_val_samples_temporal_validation",
        "n_test_samples_unseen_basins", "n_eval_samples", "n_basins",
        "n_train_basins", "n_val_basins", "n_test_basins",
        "train_start", "train_end", "val_start", "val_end", "test_start", "test_end",
    ]
    front = [c for c in front if c in df.columns]
    other = [c for c in df.columns if c not in front]
    return df[front + other]


def run_experiment(
    model_name,
    static_name,
    static_dict_raw,
    seq_len,
    horizon,
    basin_data,
    common_basins,
    train_basins,
    val_basins,
    test_basins,
    scaler_obj,
    fold_id,
):
    rprint("\n" + "=" * 80)
    rprint(f"Fold={fold_id}, model={model_name}, static={static_name}, seq_len={seq_len}, horizon={horizon}")
    rprint("=" * 80)

    model_idx = MODEL_NAMES.index(model_name) if model_name in MODEL_NAMES else 0
    static_idx = STATIC_SOURCES.index(static_name) if static_name in STATIC_SOURCES else 0
    exp_seed = int(SEED + 100000 * fold_id + 1000 * model_idx + 100 * static_idx + int(seq_len) + int(horizon))
    set_seed(exp_seed)

    exp_name = f"fold{fold_id:02d}_{model_name}_{static_name}_seq{seq_len}_lead0_horizon{horizon}_camels671_kratzert_optionA"
    exp_dir = OUT_DIR / exp_name
    ckpt_path = exp_dir / "best_model.pt"
    last_ckpt_path = exp_dir / "last_model.pt"
    done_file = exp_dir / "DONE.json"
    global_metrics_file = exp_dir / f"test_global_metrics_horizon{horizon}.csv"
    per_basin_summary_file = exp_dir / f"test_per_basin_summary_horizon{horizon}.csv"

    if RESUME_SKIP_COMPLETED and done_file.exists() and global_metrics_file.exists() and per_basin_summary_file.exists():
        rprint(f"[RESUME] Skipping completed experiment: {exp_name}")
        if is_rank0():
            eval_global = pd.read_csv(global_metrics_file)
            eval_per_basin_summary = pd.read_csv(per_basin_summary_file)
        else:
            eval_global = None
            eval_per_basin_summary = None
        barrier()
        return eval_global, eval_per_basin_summary

    static_dict = standardize_static_source(
        static_dict_raw,
        source_name=static_name,
        fit_basins=train_basins,
        transform_basins=common_basins,
        fit_years=get_train_years() if static_dict_is_yearly(static_dict_raw) else None,
    )

    train_ds = StreamflowWindowDataset(
        basin_data=basin_data,
        static_dict=static_dict,
        static_source=static_name,
        seq_len=seq_len,
        horizon=horizon,
        split=f"train_fold{fold_id}",
        start_date=TRAIN_START,
        end_date=TRAIN_END,
        basin_ids=train_basins,
    )
    val_ds = StreamflowWindowDataset(
        basin_data=basin_data,
        static_dict=static_dict,
        static_source=static_name,
        seq_len=seq_len,
        horizon=horizon,
        split=f"val_fold{fold_id}",
        start_date=VAL_START,
        end_date=VAL_END,
        basin_ids=val_basins,
    )
    test_ds = StreamflowWindowDataset(
        basin_data=basin_data,
        static_dict=static_dict,
        static_source=static_name,
        seq_len=seq_len,
        horizon=horizon,
        split=f"test_fold{fold_id}",
        start_date=TEST_START,
        end_date=TEST_END,
        basin_ids=test_basins,
    )

    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError(
            f"Empty split: fold={fold_id}, model={model_name}, source={static_name}, seq_len={seq_len}, "
            f"horizon={horizon}, train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
        )

    sampler_seed = int(SEED + 1000 * fold_id)
    train_sampler = DistributedSampler(train_ds, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True, seed=sampler_seed) if DISTRIBUTED else None
    val_sampler = DistributedSampler(val_ds, num_replicas=WORLD_SIZE, rank=RANK, shuffle=False, seed=sampler_seed) if DISTRIBUTED else None

    train_loader = make_loader(train_ds, BATCH_SIZE_PER_GPU, shuffle=(train_sampler is None), sampler=train_sampler)
    val_loader = make_loader(val_ds, BATCH_SIZE_PER_GPU, shuffle=False, sampler=val_sampler)

    static_dim = infer_static_dim(static_dict)
    model = create_model(
        model_name=model_name,
        dynamic_size=len(DYNAMIC_COLS),
        static_size=static_dim,
        output_size=horizon,
        seq_len=seq_len,
    ).to(DEVICE)

    if DISTRIBUTED:
        model = DDP(
            model,
            device_ids=[LOCAL_RANK] if DEVICE.type == "cuda" else None,
            output_device=LOCAL_RANK if DEVICE.type == "cuda" else None,
            find_unused_parameters=False,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = make_lr_scheduler(optimizer)
    criterion = nn.MSELoss()
    amp_scaler = make_grad_scaler()

    if is_rank0():
        exp_dir.mkdir(parents=True, exist_ok=True)
    barrier()

    best_val = np.inf
    best_epoch = -1
    bad_epochs = 0
    history = []
    start_epoch = 1

    if RESUME_TRAINING_FROM_LAST_CHECKPOINT and last_ckpt_path.exists() and not done_file.exists():
        try:
            last_ckpt = load_checkpoint(last_ckpt_path, map_location=DEVICE)
            get_model_core(model).load_state_dict(last_ckpt["model_state_dict"])
            if "optimizer_state_dict" in last_ckpt and last_ckpt["optimizer_state_dict"] is not None:
                optimizer.load_state_dict(last_ckpt["optimizer_state_dict"])
            if scheduler is not None and "scheduler_state_dict" in last_ckpt and last_ckpt["scheduler_state_dict"] is not None:
                scheduler.load_state_dict(last_ckpt["scheduler_state_dict"])
            if "amp_scaler_state_dict" in last_ckpt and last_ckpt["amp_scaler_state_dict"] is not None:
                try:
                    amp_scaler.load_state_dict(last_ckpt["amp_scaler_state_dict"])
                except Exception:
                    pass
            best_val = float(last_ckpt.get("best_val", np.inf))
            best_epoch = int(last_ckpt.get("best_epoch", -1))
            bad_epochs = int(last_ckpt.get("bad_epochs", 0))
            start_epoch = int(last_ckpt.get("epoch", 0)) + 1
            hist_path = exp_dir / "training_history.csv"
            if hist_path.exists():
                history = pd.read_csv(hist_path).to_dict("records")
            rprint(f"[RESUME] Resuming {exp_name} from epoch {start_epoch}; best_epoch={best_epoch}, best_val={best_val:.6f}")
        except Exception as e:
            rprint(f"[RESUME] Could not load last checkpoint for {exp_name}; restarting this config. Error: {e}")
            best_val = np.inf
            best_epoch = -1
            bad_epochs = 0
            history = []
            start_epoch = 1

    for epoch in range(start_epoch, EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, amp_scaler, epoch)
        val_loss = evaluate_loss_distributed(model, val_loader)

        if scheduler is not None:
            scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            if is_rank0():
                save_checkpoint({
                    "model_state_dict": get_model_core(model).state_dict(),
                    "model_name": model_name,
                    "static_name": static_name,
                    "seq_len": int(seq_len),
                    "horizon": int(horizon),
                    "fold_id": int(fold_id),
                    "static_dim": int(static_dim),
                    "dynamic_cols": DYNAMIC_COLS,
                    "target_col": TARGET_COL,
                    "scaler_obj": scaler_obj,
                    "train_basins": train_basins,
                    "val_basins": val_basins,
                    "test_basins": test_basins,
                    "config": {
                        "d_model": D_MODEL,
                        "n_heads": N_HEADS,
                        "n_layers": N_LAYERS,
                        "d_ff": D_FF,
                        "dropout": DROPOUT,
                        "lstm_hidden": LSTM_HIDDEN,
                        "lstm_layers": LSTM_LAYERS,
                        "patch_len": PATCH_LEN,
                        "patch_stride": PATCH_STRIDE,
                    },
                }, ckpt_path)
        else:
            bad_epochs += 1

        row = {
            "fold_id": int(fold_id),
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss_temporal_validation_non_test_basins": float(val_loss),
            "best_val_loss": float(best_val),
            "best_epoch": int(best_epoch),
            "lr": get_current_lr(optimizer),
            "elapsed_sec": float(time.time() - t0),
        }
        history.append(row)

        if is_rank0():
            pd.DataFrame(history).to_csv(exp_dir / "training_history.csv", index=False)
            if SAVE_LAST_CHECKPOINT_EVERY_EPOCH:
                save_checkpoint({
                    "model_state_dict": get_model_core(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    "amp_scaler_state_dict": amp_scaler.state_dict() if amp_scaler is not None else None,
                    "model_name": model_name,
                    "static_name": static_name,
                    "seq_len": int(seq_len),
                    "horizon": int(horizon),
                    "fold_id": int(fold_id),
                    "epoch": int(epoch),
                    "best_val": float(best_val),
                    "best_epoch": int(best_epoch),
                    "bad_epochs": int(bad_epochs),
                    "static_dim": int(static_dim),
                    "dynamic_cols": DYNAMIC_COLS,
                    "target_col": TARGET_COL,
                    "scaler_obj": scaler_obj,
                    "train_basins": train_basins,
                    "val_basins": val_basins,
                    "test_basins": test_basins,
                }, last_ckpt_path)
            rprint(
                f"fold={fold_id} | model={model_name} | static={static_name} | seq={seq_len} | H={horizon} | "
                f"epoch={epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                f"best={best_val:.6f} | best_epoch={best_epoch} | lr={get_current_lr(optimizer):.2e}"
            )

        stop = bad_epochs >= PATIENCE
        stop_tensor = torch.tensor([int(stop)], device=DEVICE)
        if DISTRIBUTED:
            dist.broadcast(stop_tensor, src=0)
        if bool(stop_tensor.item()):
            rprint(f"Early stopping at epoch {epoch}. Best epoch={best_epoch}")
            break

    barrier()

    eval_global = None
    eval_per_basin_summary = None
    if is_rank0():
        ckpt = load_checkpoint(ckpt_path, map_location=DEVICE)
        get_model_core(model).load_state_dict(ckpt["model_state_dict"])
        eval_loader = make_loader(test_ds, BATCH_SIZE_PER_GPU, shuffle=False, sampler=None)

        eval_global, eval_per_basin, eval_per_basin_summary, pred_df = evaluate_full_rank0(
            get_model_core(model), eval_loader, horizon=horizon, scaler_obj=scaler_obj
        )

        eval_global = add_experiment_metadata(
            eval_global, model_name, static_name, seq_len, horizon, best_epoch, best_val,
            static_dim, len(train_ds), len(val_ds), len(test_ds), len(test_ds), len(test_basins),
            fold_id=fold_id, n_train_basins=len(train_basins), n_val_basins=len(val_basins), n_test_basins=len(test_basins),
        )
        eval_per_basin = add_experiment_metadata(
            eval_per_basin, model_name, static_name, seq_len, horizon, best_epoch, best_val,
            static_dim, len(train_ds), len(val_ds), len(test_ds), len(test_ds), len(test_basins),
            fold_id=fold_id, n_train_basins=len(train_basins), n_val_basins=len(val_basins), n_test_basins=len(test_basins),
        )
        eval_per_basin_summary = add_experiment_metadata(
            eval_per_basin_summary, model_name, static_name, seq_len, horizon, best_epoch, best_val,
            static_dim, len(train_ds), len(val_ds), len(test_ds), len(test_ds), len(test_basins),
            fold_id=fold_id, n_train_basins=len(train_basins), n_val_basins=len(val_basins), n_test_basins=len(test_basins),
        )

        pred_df["fold_id"] = int(fold_id)
        pred_df["model_name"] = model_name
        pred_df["static_source"] = static_name
        pred_df["seq_len"] = int(seq_len)
        pred_df["lead_mode"] = "lead0_horizon1_same_day_simulation"
        pred_df["eval_split"] = "test_unseen_basins"

        eval_global.to_csv(exp_dir / f"test_global_metrics_horizon{horizon}.csv", index=False)
        eval_per_basin.to_csv(exp_dir / f"test_per_basin_metrics_horizon{horizon}.csv", index=False)
        eval_per_basin_summary.to_csv(exp_dir / f"test_per_basin_summary_horizon{horizon}.csv", index=False)
        pred_df.to_csv(exp_dir / f"test_predictions_horizon{horizon}.csv", index=False)

        save_json({
            "status": "done",
            "fold_id": int(fold_id),
            "model_name": model_name,
            "static_source": static_name,
            "seq_len": int(seq_len),
            "lead": int(LEAD),
            "horizon": int(horizon),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val),
            "finished_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "global_metrics_file": str(global_metrics_file),
            "per_basin_summary_file": str(per_basin_summary_file),
        }, done_file)

        save_json({
            "fold_id": int(fold_id),
            "model_name": model_name,
            "static_source": static_name,
            "seq_len": int(seq_len),
            "lead": int(LEAD),
            "horizon": int(horizon),
            "task": "NeuralHydrology-style daily streamflow simulation: meteorology through target day, target Q(t)",
            "split_type": "CAMELS-671 Kratzert-style ungauged basin K-fold CV, Option A: one held-out test fold; all remaining basins used for train/validation with disjoint time periods",
            "target_units": "mm/day" if CONVERT_STREAMFLOW_CFS_TO_MM_PER_DAY else "raw CAMELS streamflow units",
            "static_dim": int(static_dim),
            "n_train_samples": int(len(train_ds)),
            "n_val_samples_temporal_validation": int(len(val_ds)),
            "n_test_samples": int(len(test_ds)),
            "train_period": f"{TRAIN_START}..{TRAIN_END}",
            "validation_period": f"{VAL_START}..{VAL_END}",
            "test_period": f"{TEST_START}..{TEST_END}",
            "n_train_basins": int(len(train_basins)),
            "n_val_basins": int(len(val_basins)),
            "n_test_basins": int(len(test_basins)),
            "dynamic_cols": DYNAMIC_COLS,
            "target_col": TARGET_COL,
            "basin2vec_temporal_mode": BASIN2VEC_TEMPORAL_MODE,
            "no_past_discharge_input": True,
        }, exp_dir / "config_summary.json")

        rprint("\nTest global metrics:")
        rprint(eval_global)
        rprint("\nTest per-basin summary:")
        rprint(eval_per_basin_summary)

    try:
        del model, optimizer, scheduler, train_loader, val_loader, train_ds, val_ds, test_ds
    except Exception:
        pass
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return eval_global, eval_per_basin_summary


def append_failure_log(row: dict):
    if not is_rank0():
        return
    path = OUT_DIR / "FAILED_EXPERIMENTS.csv"
    row = to_builtin(row)
    df = pd.DataFrame([row])
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def run_experiment_safe(*args, **kwargs):
    fold_id = kwargs.get("fold_id", None)
    model_name = kwargs.get("model_name", args[0] if len(args) > 0 else None)
    static_name = kwargs.get("static_name", args[1] if len(args) > 1 else None)
    seq_len = kwargs.get("seq_len", args[3] if len(args) > 3 else None)
    horizon = kwargs.get("horizon", args[4] if len(args) > 4 else None)
    try:
        return run_experiment(*args, **kwargs)
    except Exception as e:
        err_text = traceback.format_exc()
        rprint("\n" + "!" * 80)
        rprint(f"Skipping failed experiment: fold={fold_id}, model={model_name}, static={static_name}, seq_len={seq_len}, horizon={horizon}")
        rprint(f"Error: {type(e).__name__}: {e}")
        rprint("!" * 80 + "\n")
        append_failure_log({
            "fold_id": fold_id,
            "model_name": model_name,
            "static_source": static_name,
            "seq_len": seq_len,
            "horizon": horizon,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": err_text,
        })
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        return None, None


# ==========================================================
# MAIN
# ==========================================================

def main():
    rprint("Device:", DEVICE)
    rprint("Distributed:", DISTRIBUTED)
    rprint("World size:", WORLD_SIZE)
    rprint("Output directory:", OUT_DIR)
    rprint("CAMELS root:", CAMELS_ROOT)
    rprint("Basin list: all available CAMELS-US basins; no CAMELS-531 XLSX filtering")
    rprint("Models:", MODEL_NAMES)
    rprint("Static sources:", STATIC_SOURCES)
    rprint("Basin2Vec temporal mode:", BASIN2VEC_TEMPORAL_MODE)
    rprint("No past discharge input: True")
    rprint("Target conversion cfs -> mm/day:", CONVERT_STREAMFLOW_CFS_TO_MM_PER_DAY)
    rprint("Sequence lengths:", SEQ_LENGTHS)
    rprint("Horizons:", HORIZONS)
    rprint("Lead:", LEAD, "meaning H=1 predicts Q(t) because meteorology is available through t")
    rprint("Spatial CV: Kratzert-style ungauged Option A, K_FOLDS=", K_FOLDS)
    rprint("Validation: same non-test basins as training, but disjoint target period")
    rprint("Train/val/test dates:", TRAIN_START, "to", TRAIN_END, "|", VAL_START, "to", VAL_END, "|", TEST_START, "to", TEST_END)

    basin_data_raw = load_camels_data()
    check_year_coverage(basin_data_raw)

    all_static_sources = {}
    if "basin2vec" in STATIC_SOURCES:
        all_static_sources["basin2vec"] = load_basin2vec_static_embeddings(
            BASIN2VEC_NPZ, INDEX_PARQUET, BASIN2VEC_AGG_YEARS
        )
    if "alphaearth" in STATIC_SOURCES:
        all_static_sources["alphaearth"] = load_alphaearth_aggregated_embeddings(
            ALPHAEARTH_NPZ, ALPHAEARTH_AGG_YEARS
        )
    if "tessera" in STATIC_SOURCES:
        all_static_sources["tessera"] = load_static_npz_embeddings(
            TESSERA_NPZ, source_name="tessera", years_keep=TESSERA_AGG_YEARS
        )
    if "satclip" in STATIC_SOURCES:
        all_static_sources["satclip"] = load_static_npz_embeddings(
            SATCLIP_NPZ, source_name="satclip", years_keep=SATCLIP_AGG_YEARS
        )
    if "attributes" in STATIC_SOURCES:
        all_static_sources["attributes"] = read_camels_attributes_static()

    basin_data_raw, static_sources, common_eval_basins = restrict_to_common_basins(
        basin_data=basin_data_raw,
        static_sources=all_static_sources,
    )

    folds = make_stratified_spatial_folds(common_eval_basins, K_FOLDS, SPATIAL_SPLIT_SEED)
    if is_rank0():
        split_dir = OUT_DIR / "splits"
        split_dir.mkdir(parents=True, exist_ok=True)
        fold_rows = []
        for i, fold in enumerate(folds):
            fold_rows += [{"fold_id": int(i), "site_id": sid, "role": "outer_test_fold"} for sid in fold]
        pd.DataFrame(fold_rows).to_csv(split_dir / "camels_671_outer_kfold_assignments.csv", index=False)
        save_json({
            "task": "NeuralHydrology-style daily streamflow simulation",
            "basin_list": "all available CAMELS-US basins",
            "split_type": "Kratzert-style ungauged basin K-fold CV, Option A",
            "validation_protocol": "train and validation basins are all non-test basins; validation uses a disjoint time period",
            "k_folds": int(K_FOLDS),
            "stratify_by_huc": bool(STRATIFY_FOLDS_BY_HUC),
            "inner_val_fraction": None,
            "input": "meteorology through target day t plus static descriptor",
            "target": "same-day discharge Q(t) for H=1, LEAD=0",
            "target_units": "mm/day" if CONVERT_STREAMFLOW_CFS_TO_MM_PER_DAY else "raw CAMELS streamflow units",
            "n_common_basins": int(len(common_eval_basins)),
            "static_sources": STATIC_SOURCES,
            "basin2vec_temporal_mode": BASIN2VEC_TEMPORAL_MODE,
            "models": MODEL_NAMES,
            "seq_lengths": SEQ_LENGTHS,
            "horizons": HORIZONS,
            "lead": int(LEAD),
            "data_start": DATA_START,
            "data_end": DATA_END,
            "train_start": TRAIN_START,
            "train_end": TRAIN_END,
            "val_start": VAL_START,
            "val_end": VAL_END,
            "test_start": TEST_START,
            "test_end": TEST_END,
            "basin2vec_agg_years": [int(y) for y in BASIN2VEC_AGG_YEARS],
            "alphaearth_agg_years": [int(y) for y in ALPHAEARTH_AGG_YEARS],
            "satclip_agg_years": [int(y) for y in SATCLIP_AGG_YEARS],
            "tessera_agg_years": [int(y) for y in TESSERA_AGG_YEARS],
        }, OUT_DIR / "benchmark_config_summary.json")

    all_global = []
    all_summary = []

    # ------------------------------------------------------------------
    # Run order:
    #   model -> sequence length -> horizon -> fold -> static source
    #
    # This finishes all outer folds for each (model, seq_len) pair before
    # moving to the next pair.  The experiment names and OUT_DIR are unchanged,
    # so previous completed runs are not lost: RESUME_SKIP_COMPLETED will skip
    # configs with DONE.json, and interrupted configs can resume from last_model.pt.
    # ------------------------------------------------------------------
    rprint("\nRun order: finish all folds for each (model, seq_len) pair.")

    # Build fold splits/scalers once, then reuse them for every model/static/seq.
    # In Option A, train/validation basins are the same non-test basins,
    # but train and validation samples use disjoint target periods.
    fold_contexts = []
    for fold_id in range(K_FOLDS):
        train_basins, val_basins, test_basins, excluded_basins = make_kfold_spatial_split(
            common_eval_basins, fold_id=fold_id, folds=folds
        )

        scaler_obj = None
        basin_data = basin_data_raw
        if STANDARDIZE_DYNAMIC_AND_TARGET_FROM_TRAIN:
            scaler_obj = fit_train_scaler(basin_data_raw, train_basins)
            basin_data = apply_scaler(basin_data_raw, scaler_obj)
            if is_rank0():
                scaler_dir = OUT_DIR / "scalers"
                scaler_dir.mkdir(parents=True, exist_ok=True)
                save_json(scaler_obj, scaler_dir / f"dynamic_target_scaler_fold{fold_id}_train_basins.json")

        fold_contexts.append({
            "fold_id": int(fold_id),
            "train_basins": train_basins,
            "val_basins": val_basins,
            "test_basins": test_basins,
            "excluded_basins": excluded_basins,
            "scaler_obj": scaler_obj,
            "basin_data": basin_data,
        })

    # New order: finish all folds for one (model, seq_len) pair first.
    for model_name in MODEL_NAMES:
        for seq_len in SEQ_LENGTHS:
            rprint("\n" + "#" * 80)
            rprint(f"Starting complete fold sweep for model={model_name}, seq_len={seq_len}")
            rprint("#" * 80)

            for horizon in HORIZONS:
                for fold_ctx in fold_contexts:
                    fold_id = int(fold_ctx["fold_id"])
                    train_basins = fold_ctx["train_basins"]
                    val_basins = fold_ctx["val_basins"]
                    test_basins = fold_ctx["test_basins"]
                    scaler_obj = fold_ctx["scaler_obj"]
                    basin_data = fold_ctx["basin_data"]

                    for static_name in STATIC_SOURCES:
                        if static_name not in static_sources:
                            append_failure_log({
                                "fold_id": int(fold_id),
                                "model_name": model_name,
                                "static_source": static_name,
                                "seq_len": int(seq_len),
                                "horizon": int(horizon),
                                "error_type": "MissingStaticSource",
                                "error_message": f"Requested static source missing after filtering: {static_name}",
                                "traceback": "",
                            })
                            continue

                        eval_global, eval_summary = run_experiment_safe(
                            model_name=model_name,
                            static_name=static_name,
                            static_dict_raw=static_sources[static_name],
                            seq_len=seq_len,
                            horizon=horizon,
                            basin_data=basin_data,
                            common_basins=common_eval_basins,
                            train_basins=train_basins,
                            val_basins=val_basins,
                            test_basins=test_basins,
                            scaler_obj=scaler_obj,
                            fold_id=int(fold_id),
                        )

                        if is_rank0():
                            if eval_global is not None:
                                all_global.append(eval_global)
                            if eval_summary is not None:
                                all_summary.append(eval_summary)

                            # Persist intermediate combined files after every
                            # completed/skipped config, so stopping the script
                            # does not lose the current aggregate CSVs.
                            if all_global:
                                pd.concat(all_global, ignore_index=True).to_csv(
                                    OUT_DIR / "ALL_CAMELS671_global_metrics_all_models_all_folds_partial.csv",
                                    index=False,
                                )
                            if all_summary:
                                pd.concat(all_summary, ignore_index=True).to_csv(
                                    OUT_DIR / "ALL_CAMELS671_per_basin_summary_all_models_all_folds_partial.csv",
                                    index=False,
                                )

    if is_rank0():
        if all_global:
            all_global_df = pd.concat(all_global, ignore_index=True)
            all_global_df.to_csv(OUT_DIR / "ALL_CAMELS671_global_metrics_all_models_all_folds.csv", index=False)
        else:
            all_global_df = pd.DataFrame()

        if all_summary:
            all_summary_df = pd.concat(all_summary, ignore_index=True)
            all_summary_df.to_csv(OUT_DIR / "ALL_CAMELS671_per_basin_summary_all_models_all_folds.csv", index=False)

            paper = all_summary_df.copy()
            paper = paper.sort_values(
                ["model_name", "seq_len", "horizon", "lead", "fold_id", "median_NSE"],
                ascending=[True, True, True, True, True, False],
            )
            paper.to_csv(OUT_DIR / "PAPER_TABLE_CAMELS671_all_models_per_basin_summary_all_folds.csv", index=False)

            metric_cols = [
                "median_NSE", "mean_NSE", "median_KGE", "mean_KGE",
                "median_RMSE", "mean_RMSE", "median_MAE", "mean_MAE",
                "median_Pearson_r", "mean_Pearson_r",
            ]
            group_cols = ["model_name", "static_source", "seq_len", "horizon", "lead"]
            agg_dict = {}
            for c in metric_cols:
                if c in all_summary_df.columns:
                    agg_dict[f"{c}_mean_across_folds"] = (c, "mean")
                    agg_dict[f"{c}_std_across_folds"] = (c, "std")
                    agg_dict[f"{c}_median_across_folds"] = (c, "median")
            fold_summary = (
                all_summary_df
                .groupby(group_cols, dropna=False)
                .agg(n_folds=("fold_id", "nunique"), n_rows=("fold_id", "size"), **agg_dict)
                .reset_index()
            )
            fold_summary.to_csv(OUT_DIR / "KFOLD_AGGREGATED_CAMELS671_per_basin_summary.csv", index=False)

            # Pair Basin2Vec against every baseline on same fold/model/seq/horizon/lead.
            if "basin2vec" in set(all_summary_df["static_source"].unique()):
                pair_keys = ["fold_id", "model_name", "seq_len", "horizon", "lead"]
                b2v = all_summary_df[all_summary_df["static_source"] == "basin2vec"].copy()
                keep_metrics = [c for c in metric_cols if c in all_summary_df.columns]
                paired_all = []
                for base in [s for s in STATIC_SOURCES if s != "basin2vec" and s in set(all_summary_df["static_source"].unique())]:
                    other = all_summary_df[all_summary_df["static_source"] == base].copy()
                    paired = b2v[pair_keys + keep_metrics].merge(
                        other[pair_keys + keep_metrics], on=pair_keys,
                        suffixes=("_basin2vec", f"_{base}"), how="inner"
                    )
                    paired["baseline_source"] = base
                    for c in keep_metrics:
                        paired[f"delta_{c}_basin2vec_minus_{base}"] = paired[f"{c}_basin2vec"] - paired[f"{c}_{base}"]
                    paired_all.append(paired)
                if paired_all:
                    paired_all_df = pd.concat(paired_all, ignore_index=True)
                    paired_all_df.to_csv(OUT_DIR / "PAIRED_BASIN2VEC_MINUS_BASELINES_by_fold.csv", index=False)
        else:
            all_summary_df = pd.DataFrame()

        rprint("\nSaved combined k-fold spatial results:")
        rprint(OUT_DIR / "ALL_CAMELS671_global_metrics_all_models_all_folds.csv")
        rprint(OUT_DIR / "ALL_CAMELS671_per_basin_summary_all_models_all_folds.csv")
        rprint(OUT_DIR / "PAPER_TABLE_CAMELS671_all_models_per_basin_summary_all_folds.csv")
        rprint(OUT_DIR / "KFOLD_AGGREGATED_CAMELS671_per_basin_summary.csv")
        rprint(OUT_DIR / "FAILED_EXPERIMENTS.csv")


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_distributed()
