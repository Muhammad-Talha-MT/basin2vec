#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import json
import time
import random
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

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

INDEX_PARQUET = Path("../config/training_step5/sample_index.parquet")

BASIN2VEC_NPZ = Path(
    "../src/evaluation_outputs/HCLV4_grouped_monthly_static_meta_areaaux_64d/basin_embeddings_full.npz"
)
BASIN2VEC_ARRAY_KEY = "embeddings"

ALPHAEARTH_NPZ = Path(
    "alphaearth_embeddings/embeddings/alphaearth_annual_basin_embeddings_2017_2024_scale500m.npz"
)

OUT_DIR = Path("evaluation_outputs/camels_timerxl_static_comparison_only_atttributes")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "timerxl_static_token"

DATE_COL = "date"
TARGET_COL = "discharge"

DYNAMIC_COLS = ["prcp", "srad", "swe", "tmax", "tmin", "vp"]

STATIC_SOURCES = ["attributes", "alphaearth", "basin2vec"]
# STATIC_SOURCES = ["attributes"]
# STATIC_SOURCES = ["basin2vec"]
# STATIC_SOURCES = ["alphaearth"]

# SEQ_LENGTHS = [120]
SEQ_LENGTHS = [365, 120]

LEAD = 0
# HORIZONS = [1]
HORIZONS = [1, 2]

DATA_START = "2000-01-01"
DATA_END = "2020-12-31"

TRAIN_START = "2000-01-01"
TRAIN_END = "2020-12-31"
VAL_START = "2000-01-01"
VAL_END = "2020-12-31"

BASIN2VEC_YEARS = list(range(2000, 2021))
ALPHAEARTH_AGG_YEARS = list(range(2024, 2025))

SPATIAL_SPLIT_SEED = 42
N_SPATIAL_TRAIN_BASINS = 470
N_SPATIAL_VAL_BASINS = None  # None means use all remaining common basins

MANUAL_TRAIN_BASINS = None
MANUAL_VAL_BASINS = None

SEED = 42
BATCH_SIZE_PER_GPU = 128
NUM_WORKERS_PER_GPU = 4
PREFETCH_FACTOR = 2

EPOCHS = 60
PATIENCE = 12
LR = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
USE_AMP = True

USE_LR_SCHEDULER = True
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_MIN_LR = 1e-5

STANDARDIZE_DYNAMIC_AND_TARGET_FROM_TRAIN = True

D_MODEL = 128
N_HEADS = 8
N_LAYERS = 4
D_FF = 512
DROPOUT = 0.15
MAX_SEQ_LEN = max(SEQ_LENGTHS) + 1
USE_LEARNED_POSITION = True

CAMELS_SELECTED_ATTRIBUTE_COLUMNS = [
    # Topography / scale
    # "area_gages2",
    "elev_mean",
    "slope_mean",

    # Soil texture / storage / hydraulic properties
    "sand_frac",
    "silt_frac",
    "clay_frac",
    "soil_depth_pelletier",
    "soil_depth_statsgo",
    "soil_porosity",
    "soil_conductivity",
    "max_water_content",

    # Hydrogeology / geology
    "geol_permeability",
    "carbonate_rocks_frac",

    # Land cover / vegetation
    "frac_forest",
    "lai_max",
    "lai_diff",
    "gvf_max",
    "gvf_diff",
]
# ==========================================================
# DISTRIBUTED
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

def get_camels_available_ids():
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

    basin_data = {}
    skipped = []

    iterator = tqdm(common_ids, desc="load CAMELS basins", disable=not is_rank0())

    for sid in iterator:
        try:
            forcing = read_camels_daymet(sid)
            flow = read_camels_streamflow(sid)

            df = forcing.merge(flow, on=DATE_COL, how="inner")
            df["site_id"] = sid
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

def collapse_duplicate_annual_rows(meta: pd.DataFrame, emb: np.ndarray):
    meta = meta.copy()
    meta["row"] = np.arange(len(meta))
    out = {}

    for (sid, yr), sub in meta.groupby(["site_id", "year"]):
        idx = sub["row"].to_numpy()
        z = emb[idx].mean(axis=0, keepdims=True)
        out[(sid, int(yr))] = l2_normalize(z)[0].astype(np.float32)

    return out


def load_basin2vec_annual_embeddings(npz_path: Path, index_parquet: Path, years_keep: List[int]):
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

    emb = l2_normalize(emb)

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

    keep = meta["year"].isin(years_keep).to_numpy()
    meta = meta.loc[keep].reset_index(drop=True)
    emb = emb[keep]

    if len(meta) == 0:
        raise RuntimeError(f"No Basin2Vec rows found for years_keep={years_keep}")

    annual = collapse_duplicate_annual_rows(meta, emb)

    dim = len(next(iter(annual.values())))
    basins = sorted({k[0] for k in annual.keys()})
    years_loaded = sorted({k[1] for k in annual.keys()})

    rprint(
        f"Loaded Basin2Vec annual embeddings: {len(annual)} basin-years, "
        f"{len(basins)} basins, dim={dim}, years={years_loaded[0]}--{years_loaded[-1]}"
    )

    return annual


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

    emb = l2_normalize(emb)

    if "site_ids" in data.files:
        site_ids = np.asarray(data["site_ids"])
    elif "labels" in data.files:
        site_ids = np.asarray(data["labels"])
    else:
        raise KeyError("AlphaEarth NPZ must contain site_ids or labels.")

    if "years" not in data.files:
        raise KeyError("AlphaEarth NPZ must contain years.")

    years = np.asarray(data["years"]).astype(int)

    meta = pd.DataFrame({
        "site_id": [normalize_site_id(x) for x in site_ids],
        "year": years,
    })

    keep = meta["year"].isin(years_keep).to_numpy()
    meta = meta.loc[keep].reset_index(drop=True)
    emb = emb[keep]

    if len(meta) == 0:
        raise RuntimeError(
            f"No AlphaEarth rows found for years_keep={years_keep}. "
            f"Check ALPHAEARTH_AGG_YEARS."
        )

    meta["row"] = np.arange(len(meta))
    static = {}
    year_counts = []

    for sid, sub in meta.groupby("site_id"):
        idx = sub["row"].to_numpy()
        z = emb[idx].mean(axis=0, keepdims=True)
        static[sid] = l2_normalize(z)[0].astype(np.float32)
        year_counts.append({"site_id": sid, "n_alphaearth_years": int(sub["year"].nunique())})

    if is_rank0():
        pd.DataFrame(year_counts).to_csv(OUT_DIR / "alphaearth_aggregated_year_counts.csv", index=False)

    dim = len(next(iter(static.values())))
    rprint(
        f"Loaded aggregated AlphaEarth embeddings: {len(static)} basins, dim={dim}, "
        f"aggregation years={sorted(set(meta['year'].astype(int)))}"
    )

    return static


# ==========================================================
# COMMON BASINS / SPLIT / NORMALIZATION
# ==========================================================

def basins_with_all_basin2vec_years(annual_dict, years_keep):
    year_set = set(int(y) for y in years_keep)
    basin_to_years = {}

    for sid, yr in annual_dict.keys():
        basin_to_years.setdefault(sid, set()).add(int(yr))

    return {sid for sid, yrs in basin_to_years.items() if year_set.issubset(yrs)}


def restrict_to_common_basins(basin_data, static_sources, basin2vec_years):
    csv_basins = set(basin_data.keys())

    source_sets = {}
    if "basin2vec" in static_sources:
        source_sets["basin2vec"] = basins_with_all_basin2vec_years(
            static_sources["basin2vec"], basin2vec_years
        )
    if "alphaearth" in static_sources:
        source_sets["alphaearth"] = set(static_sources["alphaearth"].keys())
    if "attributes" in static_sources:
        source_sets["attributes"] = set(static_sources["attributes"].keys())

    common = set(csv_basins)
    for s in source_sets.values():
        common = common & s

    common_basins = sorted(common)

    rprint("\n" + "=" * 80)
    rprint("Common-basin filtering")
    rprint("=" * 80)
    rprint("CAMELS dynamic basins:", len(csv_basins))
    for name, ids in source_sets.items():
        rprint(f"{name} basins:", len(ids))
    rprint("Common basins used:", len(common_basins))

    if len(common_basins) == 0:
        raise RuntimeError("No common basins found across requested static sources.")

    if is_rank0():
        rows = [{"source": "camels_dynamic", "available_basins": len(csv_basins), "common_eval_basins": len(common_basins)}]
        for name, ids in source_sets.items():
            rows.append({"source": name, "available_basins": len(ids), "common_eval_basins": len(common_basins)})
        pd.DataFrame(rows).to_csv(OUT_DIR / "static_source_common_basin_coverage.csv", index=False)
        pd.DataFrame({"site_id": common_basins}).to_csv(OUT_DIR / "common_eval_basins.csv", index=False)

    filtered_data = {sid: df for sid, df in basin_data.items() if sid in common_basins}

    filtered_static = {}
    if "basin2vec" in static_sources:
        filtered_static["basin2vec"] = {
            (sid, yr): z
            for (sid, yr), z in static_sources["basin2vec"].items()
            if sid in common_basins and int(yr) in basin2vec_years
        }
    if "alphaearth" in static_sources:
        filtered_static["alphaearth"] = {
            sid: z for sid, z in static_sources["alphaearth"].items()
            if sid in common_basins
        }
    if "attributes" in static_sources:
        filtered_static["attributes"] = {
            sid: z for sid, z in static_sources["attributes"].items()
            if sid in common_basins
        }

    return filtered_data, filtered_static, common_basins


def normalize_manual_basin_list(values):
    if values is None:
        return None
    out = [normalize_site_id(v) for v in values]
    out = [v for v in out if v is not None]
    if len(out) != len(set(out)):
        raise ValueError("Manual basin list contains duplicates after normalization.")
    return out


def make_spatial_basin_split(common_basins):
    common_basins = sorted([normalize_site_id(x) for x in common_basins])
    common_basins = [x for x in common_basins if x is not None]
    common_set = set(common_basins)

    manual_train = normalize_manual_basin_list(MANUAL_TRAIN_BASINS)
    manual_val = normalize_manual_basin_list(MANUAL_VAL_BASINS)

    if manual_train is not None:
        missing = sorted(set(manual_train) - common_set)
        if missing:
            raise ValueError(f"Manual train basins not in common basins: {missing}")

    if manual_val is not None:
        missing = sorted(set(manual_val) - common_set)
        if missing:
            raise ValueError(f"Manual val basins not in common basins: {missing}")

    fixed = []
    if manual_train is not None:
        fixed += manual_train
    if manual_val is not None:
        fixed += manual_val
    if len(fixed) != len(set(fixed)):
        raise ValueError("Manual train and val basins overlap.")

    rng = np.random.default_rng(SPATIAL_SPLIT_SEED)
    remaining = np.asarray(sorted(common_set - set(fixed)), dtype=object)
    rng.shuffle(remaining)

    if manual_train is not None:
        train_basins = sorted(manual_train)
        cursor = 0
    else:
        n_train = min(N_SPATIAL_TRAIN_BASINS, len(remaining) - 1)
        train_basins = sorted(remaining[:n_train].tolist())
        cursor = n_train

    if manual_val is not None:
        val_basins = sorted(manual_val)
    else:
        if N_SPATIAL_VAL_BASINS is None:
            val_basins = sorted(remaining[cursor:].tolist())
        else:
            val_basins = sorted(remaining[cursor:cursor + N_SPATIAL_VAL_BASINS].tolist())

    excluded = sorted(common_set - set(train_basins) - set(val_basins))

    if set(train_basins) & set(val_basins):
        raise RuntimeError("Spatial split leakage: train and val overlap.")

    if len(train_basins) == 0 or len(val_basins) == 0:
        raise RuntimeError(f"Bad split: train={len(train_basins)}, val={len(val_basins)}")

    rprint("\n" + "=" * 80)
    rprint("CAMELS spatial basin split")
    rprint("=" * 80)
    rprint("Common basins:", len(common_basins))
    rprint("Train basins:", len(train_basins))
    rprint("Validation/evaluation basins:", len(val_basins))
    rprint("Excluded basins:", len(excluded))
    rprint("Seed:", SPATIAL_SPLIT_SEED)

    if is_rank0():
        rows = []
        rows += [{"site_id": sid, "spatial_split": "train"} for sid in train_basins]
        rows += [{"site_id": sid, "spatial_split": "val_eval"} for sid in val_basins]
        rows += [{"site_id": sid, "spatial_split": "excluded"} for sid in excluded]
        pd.DataFrame(rows).to_csv(OUT_DIR / "camels_spatial_basin_split.csv", index=False)

    return train_basins, val_basins, excluded


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


def train_years_for_standardization():
    train_start = pd.Timestamp(TRAIN_START)
    train_end = pd.Timestamp(TRAIN_END)
    years = []
    for y in BASIN2VEC_YEARS:
        ys = pd.Timestamp(f"{y}-01-01")
        ye = pd.Timestamp(f"{y}-12-31")
        if ye >= train_start and ys <= train_end:
            years.append(y)
    return years


def standardize_static_source(static_dict, source_name, fit_basins, transform_basins):
    fit_basins = sorted(set(fit_basins))
    transform_basins = sorted(set(transform_basins))

    if source_name == "basin2vec":
        fit_years = train_years_for_standardization()
        fit_keys = [(sid, yr) for sid in fit_basins for yr in fit_years if (sid, yr) in static_dict]
        all_keys = [(sid, yr) for sid in transform_basins for yr in BASIN2VEC_YEARS if (sid, yr) in static_dict]
    else:
        fit_keys = [sid for sid in fit_basins if sid in static_dict]
        all_keys = [sid for sid in transform_basins if sid in static_dict]

    if len(fit_keys) == 0:
        raise RuntimeError(f"No fit keys for static source: {source_name}")

    M = np.vstack([static_dict[k] for k in fit_keys]).astype(np.float32)
    mean = M.mean(axis=0, keepdims=True)
    std = M.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    out = {}
    for k in all_keys:
        out[k] = ((static_dict[k][None, :] - mean) / std)[0].astype(np.float32)

    return out


# ==========================================================
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

        annual_static = static_source == "basin2vec"
        valid_b2v_years = set(BASIN2VEC_YEARS)

        for sid, df in basin_data.items():
            if basin_ids is not None and sid not in basin_ids:
                continue

            if annual_static:
                has_static = any((sid, y) in static_dict for y in valid_b2v_years)
            else:
                has_static = sid in static_dict

            if not has_static:
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

                if target_start_date < start_date or target_end_date > end_date:
                    continue
                if int(target_end_date.year) != target_year:
                    continue

                if annual_static:
                    if target_year not in valid_b2v_years:
                        continue
                    if (sid, target_year) not in static_dict:
                        continue
                else:
                    if sid not in static_dict:
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

        if self.static_source == "basin2vec":
            x_static = self.static_dict[(sid, target_year)]
        else:
            x_static = self.static_dict[sid]

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


# ==========================================================
# MODEL
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


class TimerXLStaticTokenForecastModel(nn.Module):
    def __init__(
        self,
        dynamic_size,
        static_size,
        d_model,
        n_heads,
        n_layers,
        d_ff,
        output_size,
        dropout=0.1,
        max_seq_len=366,
        use_learned_position=True,
    ):
        super().__init__()

        self.d_model = int(d_model)
        self.max_seq_len = int(max_seq_len)
        self.use_learned_position = bool(use_learned_position)

        self.dynamic_proj = nn.Sequential(
            nn.Linear(dynamic_size, d_model),
            nn.LayerNorm(d_model),
        )

        self.static_proj = nn.Sequential(
            nn.LayerNorm(static_size),
            nn.Linear(static_size, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        if self.use_learned_position:
            self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
            nn.init.normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.backbone = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_size),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_dyn, x_static):
        B, T, _ = x_dyn.shape
        device = x_dyn.device

        x_dyn = torch.nan_to_num(x_dyn.float(), nan=0.0, posinf=0.0, neginf=0.0)
        x_static = torch.nan_to_num(x_static.float(), nan=0.0, posinf=0.0, neginf=0.0)

        dyn_tokens = self.dynamic_proj(x_dyn)
        static_token = self.static_proj(x_static).unsqueeze(1)

        seq = torch.cat([static_token, dyn_tokens], dim=1)
        L = seq.shape[1]

        if L > self.max_seq_len:
            raise RuntimeError(f"Sequence length {L} exceeds max_seq_len={self.max_seq_len}")

        if self.use_learned_position:
            seq = seq + self.pos_embed[:, :L, :]
        else:
            seq = seq + sinusoid_1d(L, self.d_model, device).unsqueeze(0)

        z = self.backbone(seq, mask=causal_mask(L, device))
        feat = self.final_norm(z[:, -1, :])
        return self.head(feat)


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


def add_experiment_metadata(df, static_name, seq_len, horizon, best_epoch, best_val, static_dim, n_train, n_val, n_eval, n_basins):
    df = df.copy()
    df["model_name"] = MODEL_NAME
    df["static_source"] = static_name
    df["seq_len"] = int(seq_len)
    df["lead_mode"] = "lead0_multihorizon"
    df["horizon"] = int(horizon)
    df["eval_split"] = "val_unseen_basins"
    df["best_epoch"] = int(best_epoch)
    df["best_val_loss"] = float(best_val)
    df["static_dim"] = int(static_dim)
    df["n_train_samples"] = int(n_train)
    df["n_val_samples_unseen_basins"] = int(n_val)
    df["n_eval_samples"] = int(n_eval)
    df["n_basins"] = int(n_basins)

    front = [
        "model_name", "static_source", "seq_len", "lead_mode", "horizon",
        "lead", "eval_split", "best_epoch", "best_val_loss", "static_dim",
        "n_train_samples", "n_val_samples_unseen_basins", "n_eval_samples", "n_basins",
    ]
    front = [c for c in front if c in df.columns]
    other = [c for c in df.columns if c not in front]
    return df[front + other]


def run_experiment(static_name, static_dict_raw, seq_len, horizon, basin_data, common_basins, train_basins, val_basins, scaler_obj):
    rprint("\n" + "=" * 80)
    rprint(f"Experiment: static={static_name}, seq_len={seq_len}, horizon={horizon}")
    rprint("=" * 80)

    static_dict = standardize_static_source(
        static_dict_raw,
        source_name=static_name,
        fit_basins=train_basins,
        transform_basins=common_basins,
    )

    train_ds = StreamflowWindowDataset(
        basin_data=basin_data,
        static_dict=static_dict,
        static_source=static_name,
        seq_len=seq_len,
        horizon=horizon,
        split="train_basins",
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
        split="val_unseen_basins",
        start_date=VAL_START,
        end_date=VAL_END,
        basin_ids=val_basins,
    )

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(
            f"Empty split: source={static_name}, seq_len={seq_len}, horizon={horizon}, "
            f"train={len(train_ds)}, val={len(val_ds)}"
        )

    train_sampler = DistributedSampler(train_ds, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True, seed=SEED) if DISTRIBUTED else None
    val_sampler = DistributedSampler(val_ds, num_replicas=WORLD_SIZE, rank=RANK, shuffle=False, seed=SEED) if DISTRIBUTED else None

    train_loader = make_loader(train_ds, BATCH_SIZE_PER_GPU, shuffle=(train_sampler is None), sampler=train_sampler)
    val_loader = make_loader(val_ds, BATCH_SIZE_PER_GPU, shuffle=False, sampler=val_sampler)

    static_dim = len(next(iter(static_dict.values())))

    model = TimerXLStaticTokenForecastModel(
        dynamic_size=len(DYNAMIC_COLS),
        static_size=static_dim,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        output_size=horizon,
        dropout=DROPOUT,
        max_seq_len=max(seq_len + 1, MAX_SEQ_LEN),
        use_learned_position=USE_LEARNED_POSITION,
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

    exp_name = f"{MODEL_NAME}_{static_name}_seq{seq_len}_lead0_horizon{horizon}_camels_spatial"
    exp_dir = OUT_DIR / exp_name
    ckpt_path = exp_dir / "best_model.pt"

    if is_rank0():
        exp_dir.mkdir(parents=True, exist_ok=True)
    barrier()

    best_val = np.inf
    best_epoch = -1
    bad_epochs = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
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
                    "static_name": static_name,
                    "seq_len": seq_len,
                    "horizon": horizon,
                    "static_dim": static_dim,
                    "dynamic_cols": DYNAMIC_COLS,
                    "target_col": TARGET_COL,
                    "scaler_obj": scaler_obj,
                    "train_basins": train_basins,
                    "val_basins": val_basins,
                    "config": {
                        "d_model": D_MODEL,
                        "n_heads": N_HEADS,
                        "n_layers": N_LAYERS,
                        "d_ff": D_FF,
                        "dropout": DROPOUT,
                        "use_learned_position": USE_LEARNED_POSITION,
                    },
                }, ckpt_path)
        else:
            bad_epochs += 1

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss_unseen_basins": val_loss,
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "lr": get_current_lr(optimizer),
            "elapsed_sec": time.time() - t0,
        }
        history.append(row)

        if is_rank0():
            pd.DataFrame(history).to_csv(exp_dir / "training_history.csv", index=False)
            rprint(
                f"epoch={epoch:03d} | train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f} | best={best_val:.6f} | "
                f"best_epoch={best_epoch} | lr={get_current_lr(optimizer):.2e}"
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

        eval_loader = make_loader(val_ds, BATCH_SIZE_PER_GPU, shuffle=False, sampler=None)

        eval_global, eval_per_basin, eval_per_basin_summary, pred_df = evaluate_full_rank0(
            get_model_core(model),
            eval_loader,
            horizon=horizon,
            scaler_obj=scaler_obj,
        )

        eval_global = add_experiment_metadata(
            eval_global, static_name, seq_len, horizon, best_epoch, best_val,
            static_dim, len(train_ds), len(val_ds), len(val_ds), len(val_basins),
        )
        eval_per_basin = add_experiment_metadata(
            eval_per_basin, static_name, seq_len, horizon, best_epoch, best_val,
            static_dim, len(train_ds), len(val_ds), len(val_ds), len(val_basins),
        )
        eval_per_basin_summary = add_experiment_metadata(
            eval_per_basin_summary, static_name, seq_len, horizon, best_epoch, best_val,
            static_dim, len(train_ds), len(val_ds), len(val_ds), len(val_basins),
        )

        pred_df["model_name"] = MODEL_NAME
        pred_df["static_source"] = static_name
        pred_df["seq_len"] = int(seq_len)
        pred_df["lead_mode"] = "lead0_multihorizon"
        pred_df["eval_split"] = "val_unseen_basins"

        eval_global.to_csv(exp_dir / f"val_global_metrics_horizon{horizon}.csv", index=False)
        eval_per_basin.to_csv(exp_dir / f"val_per_basin_metrics_horizon{horizon}.csv", index=False)
        eval_per_basin_summary.to_csv(exp_dir / f"val_per_basin_summary_horizon{horizon}.csv", index=False)
        pred_df.to_csv(exp_dir / f"val_predictions_horizon{horizon}.csv", index=False)

        save_json({
            "model_name": MODEL_NAME,
            "static_source": static_name,
            "seq_len": seq_len,
            "horizon": horizon,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "static_dim": static_dim,
            "n_train_samples": len(train_ds),
            "n_val_samples": len(val_ds),
            "n_train_basins": len(train_basins),
            "n_val_basins": len(val_basins),
            "dynamic_cols": DYNAMIC_COLS,
            "target_col": TARGET_COL,
            "no_past_discharge_input": True,
            "camels_root": str(CAMELS_ROOT),
            "basin2vec_npz": str(BASIN2VEC_NPZ),
            "alphaearth_npz": str(ALPHAEARTH_NPZ),
        }, exp_dir / "config_summary.json")

        rprint("\nValidation global metrics:")
        rprint(eval_global)
        rprint("\nValidation per-basin summary:")
        rprint(eval_per_basin_summary)

    barrier()
    return eval_global, eval_per_basin_summary


# ==========================================================
# MAIN
# ==========================================================

def main():
    rprint("Device:", DEVICE)
    rprint("Distributed:", DISTRIBUTED)
    rprint("World size:", WORLD_SIZE)
    rprint("Output directory:", OUT_DIR)
    rprint("CAMELS root:", CAMELS_ROOT)
    rprint("Model:", MODEL_NAME)
    rprint("No past discharge input: True")
    rprint("Static fusion: prepend static token")
    rprint("Static sources:", STATIC_SOURCES)
    rprint("Sequence lengths:", SEQ_LENGTHS)
    rprint("Horizons:", HORIZONS)

    basin_data = load_camels_data()
    check_year_coverage(basin_data)

    all_static_sources = {}

    if "basin2vec" in STATIC_SOURCES:
        all_static_sources["basin2vec"] = load_basin2vec_annual_embeddings(
            BASIN2VEC_NPZ,
            INDEX_PARQUET,
            BASIN2VEC_YEARS,
        )

    if "alphaearth" in STATIC_SOURCES:
        all_static_sources["alphaearth"] = load_alphaearth_aggregated_embeddings(
            ALPHAEARTH_NPZ,
            ALPHAEARTH_AGG_YEARS,
        )

    if "attributes" in STATIC_SOURCES:
        all_static_sources["attributes"] = read_camels_attributes_static()

    basin_data, static_sources, common_eval_basins = restrict_to_common_basins(
        basin_data=basin_data,
        static_sources=all_static_sources,
        basin2vec_years=BASIN2VEC_YEARS,
    )

    train_basins, val_basins, excluded_basins = make_spatial_basin_split(common_eval_basins)

    scaler_obj = None
    if STANDARDIZE_DYNAMIC_AND_TARGET_FROM_TRAIN:
        scaler_obj = fit_train_scaler(basin_data, train_basins)
        basin_data = apply_scaler(basin_data, scaler_obj)

        if is_rank0():
            save_json(scaler_obj, OUT_DIR / "dynamic_target_scaler_train_basins.json")

    all_global = []
    all_summary = []

    for static_name in STATIC_SOURCES:
        if static_name not in static_sources:
            raise KeyError(f"Requested static source missing after filtering: {static_name}")

        for seq_len in SEQ_LENGTHS:
            for horizon in HORIZONS:
                eval_global, eval_summary = run_experiment(
                    static_name=static_name,
                    static_dict_raw=static_sources[static_name],
                    seq_len=seq_len,
                    horizon=horizon,
                    basin_data=basin_data,
                    common_basins=common_eval_basins,
                    train_basins=train_basins,
                    val_basins=val_basins,
                    scaler_obj=scaler_obj,
                )

                if is_rank0():
                    if eval_global is not None:
                        all_global.append(eval_global)
                    if eval_summary is not None:
                        all_summary.append(eval_summary)

    if is_rank0():
        if all_global:
            pd.concat(all_global, ignore_index=True).to_csv(
                OUT_DIR / "ALL_CAMELS_global_metrics.csv",
                index=False,
            )

        if all_summary:
            pd.concat(all_summary, ignore_index=True).to_csv(
                OUT_DIR / "ALL_CAMELS_per_basin_summary.csv",
                index=False,
            )

        rprint("\nSaved combined results:")
        rprint(OUT_DIR / "ALL_CAMELS_global_metrics.csv")
        rprint(OUT_DIR / "ALL_CAMELS_per_basin_summary.csv")


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_distributed()