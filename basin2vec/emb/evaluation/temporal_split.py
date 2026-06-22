#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import json
import time
import random
import traceback
import gc
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
    "../src/evaluation_outputs/HCLV4_grouped_monthly_static_meta_areaaux_64d_ablation_no_temporal/basin_embeddings_full.npz"
)
BASIN2VEC_ARRAY_KEY = "embeddings"

ALPHAEARTH_NPZ = Path(
    "alphaearth_embeddings/embeddings/alphaearth_annual_basin_embeddings_2017_202ce4_scale500m.npz"
)

SATCLIP_NPZ = Path(
    "satclip_embeddings/satclip_8point_2024_embeddings.npz"
)

TESSERA_NPZ = Path(
    "tessera_embeddings/tessera_8point_2024_embeddings.npz"
)

OUT_DIR = Path("evaluation_outputs/temporal_split_HCLV4_grouped_monthly_static_meta_areaaux_64d_ablation_no_temporal")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# MODEL_NAMES = ["lstm", "tft", "timexer", "timerxl", "itransformer"]
MODEL_NAMES = ["lstm"]

DATE_COL = "date"
TARGET_COL = "discharge"

DYNAMIC_COLS = ["prcp", "srad", "tmax", "tmin", "vp"]

# STATIC_SOURCES = ["basin2vec", "alphaearth"]
# STATIC_SOURCES = ["attributes"]
STATIC_SOURCES = ["basin2vec"]
# STATIC_SOURCES = ["alphaearth"]
# STATIC_SOURCES = ["tessera"]
# STATIC_SOURCES = ["satclip"]
# STATIC_SOURCES = ["attributes", "basin2vec", "alphaearth", "tessera", "satclip"]

SEQ_LENGTHS = [365]
# SEQ_LENGTHS = [365, 120]

LEAD = 0
HORIZONS = [1]

DATA_START = "1990-01-01"
DATA_END = "2014-12-31"

TRAIN_START = "1990-01-01"
TRAIN_END = "2009-12-31"
VAL_START = "2010-01-01"
VAL_END = "2014-12-31"

BASIN2VEC_YEARS = list(range(1990, 2015))
ALPHAEARTH_AGG_YEARS = list(range(2017, 2025))
TESSERA_AGG_YEARS = [2024]
SATCLIP_AGG_YEARS = [2024]

# Temporal-split static descriptor usage
# Basin2Vec is averaged over BASIN2VEC_YEARS to create one basin-level
# static descriptor per basin, analogous to the AlphaEarth temporal average.
# This avoids using target-year Basin2Vec vectors in the temporal benchmark.
BASIN2VEC_USAGE_MODE = "aggregated_static"
ALPHAEARTH_USAGE_MODE = "aggregated_static"


SEED = 42
BATCH_SIZE_PER_GPU = 128
NUM_WORKERS_PER_GPU = 4
PREFETCH_FACTOR = 2

EPOCHS = 40
PATIENCE = 5
LR = 1e-3
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
LSTM_HIDDEN = 128
LSTM_LAYERS = 2
PATCH_LEN = 16
PATCH_STRIDE = 8
MAX_SEQ_LEN = max(SEQ_LENGTHS) + 1
USE_LEARNED_POSITION = True

# Resume behavior
# - If DONE.json and metric CSVs exist, the whole configuration is skipped.
# - If a configuration was interrupted mid-training, training resumes from last_model.pt.
RESUME_SKIP_COMPLETED = True
RESUME_TRAINING_FROM_LAST_CHECKPOINT = True
SAVE_LAST_CHECKPOINT_EVERY_EPOCH = True

CAMELS_SELECTED_ATTRIBUTE_COLUMNS = [
    # Topography / scale
    "area_gages2",
    "elev_mean",
    # "slope_mean",

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


def load_basin2vec_aggregated_embeddings(npz_path: Path, index_parquet: Path, years_keep: List[int]):
    """
    Load Basin2Vec annual embeddings and average them into one static vector per basin.

    This mirrors the AlphaEarth baseline: all selected annual vectors for a basin
    are averaged, then L2-normalized. The resulting dictionary is keyed only by
    site_id, so downstream code treats Basin2Vec as a basin-level static descriptor.
    """
    annual = load_basin2vec_annual_embeddings(npz_path, index_parquet, years_keep)

    rows = []
    for (sid, yr), z in annual.items():
        rows.append((sid, int(yr), z))

    if not rows:
        raise RuntimeError(f"No Basin2Vec annual rows found for years_keep={years_keep}")

    static = {}
    year_counts = []
    by_basin = {}
    for sid, yr, z in rows:
        by_basin.setdefault(sid, []).append((yr, z))

    for sid, yz in by_basin.items():
        yrs = sorted({int(yr) for yr, _ in yz})
        Z = np.vstack([z for _, z in yz]).astype(np.float32)
        z_mean = Z.mean(axis=0, keepdims=True)
        static[sid] = l2_normalize(z_mean)[0].astype(np.float32)
        year_counts.append({
            "site_id": sid,
            "n_basin2vec_years": int(len(yrs)),
            "min_basin2vec_year": int(min(yrs)),
            "max_basin2vec_year": int(max(yrs)),
        })

    if is_rank0():
        pd.DataFrame(year_counts).to_csv(
            OUT_DIR / "basin2vec_aggregated_year_counts.csv",
            index=False,
        )

    dim = len(next(iter(static.values())))
    rprint(
        f"Loaded aggregated Basin2Vec embeddings: {len(static)} basins, dim={dim}, "
        f"aggregation years={sorted(set(yr for _, yr, _ in rows))}"
    )

    return static


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



def load_generic_static_npz_embeddings(
    npz_path: Path,
    source_name: str,
    years_keep: Optional[List[int]] = None,
):
    """
    Load a basin-level static embedding NPZ and return {site_id: vector}.

    Expected NPZ variants supported:
      - embedding array key: embeddings, X, raw_embeddings, hydrologic_repr
      - basin id key: site_ids, basin_ids, labels
      - optional years key: years

    If years are present and years_keep is not None, rows are filtered to those
    years and duplicate rows per basin are averaged. This lets one-year products
    such as SatCLIP-8PointMean/TESSERA-8PointMean and multi-year static products
    share the same downstream interface.
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"{source_name} NPZ not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    rprint(f"{source_name} NPZ:", npz_path)
    rprint(f"{source_name} keys:", data.files)

    emb_key = None
    for k in ["embeddings", "X", "raw_embeddings", "hydrologic_repr"]:
        if k in data.files:
            emb_key = k
            break
    if emb_key is None:
        raise KeyError(
            f"No usable embedding array key found in {npz_path}. "
            "Expected one of: embeddings, X, raw_embeddings, hydrologic_repr."
        )

    id_key = None
    for k in ["site_ids", "basin_ids", "labels"]:
        if k in data.files:
            id_key = k
            break
    if id_key is None:
        raise KeyError(
            f"{source_name} NPZ must contain one of: site_ids, basin_ids, labels."
        )

    emb = np.asarray(data[emb_key], dtype=np.float32)
    site_ids = np.asarray(data[id_key])

    if emb.ndim != 2:
        raise ValueError(f"{source_name} embeddings must be 2D, got shape={emb.shape}.")
    if len(site_ids) != len(emb):
        raise ValueError(
            f"{source_name} metadata length mismatch: "
            f"len({id_key})={len(site_ids)} vs embeddings={len(emb)}."
        )

    if "years" in data.files:
        years = np.asarray(data["years"]).astype(int)
        if len(years) != len(emb):
            raise ValueError(
                f"{source_name} years length mismatch: len(years)={len(years)} "
                f"vs embeddings={len(emb)}."
            )
    else:
        years = np.full(len(emb), -1, dtype=np.int32)

    meta = pd.DataFrame({
        "site_id": [normalize_site_id(x) for x in site_ids],
        "year": years,
    })

    valid = meta["site_id"].notna().to_numpy()
    meta = meta.loc[valid].reset_index(drop=True)
    emb = emb[valid]

    if years_keep is not None and "years" in data.files:
        keep = meta["year"].isin([int(y) for y in years_keep]).to_numpy()
        meta = meta.loc[keep].reset_index(drop=True)
        emb = emb[keep]

    if len(meta) == 0:
        raise RuntimeError(
            f"No {source_name} rows found after filtering. "
            f"years_keep={years_keep}, path={npz_path}"
        )

    emb = np.nan_to_num(emb.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    emb = l2_normalize(emb)

    meta["row"] = np.arange(len(meta))
    static = {}
    year_counts = []

    for sid, sub in meta.groupby("site_id"):
        idx = sub["row"].to_numpy()
        z = emb[idx].mean(axis=0, keepdims=True)
        static[sid] = l2_normalize(z)[0].astype(np.float32)

        valid_years = sorted({int(y) for y in sub["year"].tolist() if int(y) >= 0})
        year_counts.append({
            "site_id": sid,
            f"n_{source_name}_rows": int(len(sub)),
            f"n_{source_name}_years": int(len(valid_years)),
            f"min_{source_name}_year": int(min(valid_years)) if valid_years else None,
            f"max_{source_name}_year": int(max(valid_years)) if valid_years else None,
        })

    if is_rank0():
        pd.DataFrame(year_counts).to_csv(
            OUT_DIR / f"{source_name}_aggregated_year_counts.csv",
            index=False,
        )

    dim = len(next(iter(static.values())))
    year_msg = (
        sorted(set(meta["year"].astype(int)))
        if "years" in data.files
        else "no years key"
    )
    rprint(
        f"Loaded {source_name} static embeddings: {len(static)} basins, "
        f"dim={dim}, years={year_msg}"
    )

    return static


# ==========================================================
# COMMON BASINS / TEMPORAL SPLIT / NORMALIZATION
# ==========================================================

def years_intersect_period(years_keep: List[int], start_date: str, end_date: str) -> List[int]:
    """Return calendar years from years_keep whose year interval intersects [start_date, end_date]."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    out = []
    for y in years_keep:
        ys = pd.Timestamp(f"{int(y)}-01-01")
        ye = pd.Timestamp(f"{int(y)}-12-31")
        if ye >= start and ys <= end:
            out.append(int(y))
    return sorted(set(out))


def train_years_for_standardization() -> List[int]:
    return years_intersect_period(BASIN2VEC_YEARS, TRAIN_START, TRAIN_END)


def val_years_for_evaluation() -> List[int]:
    return years_intersect_period(BASIN2VEC_YEARS, VAL_START, VAL_END)


def target_years_required_for_basin2vec() -> List[int]:
    # With H=1 and LEAD=0 this is exactly the union of target years in train and validation.
    return sorted(set(train_years_for_standardization()) | set(val_years_for_evaluation()))


def basins_with_all_basin2vec_years(annual_dict, years_keep):
    year_set = set(int(y) for y in years_keep)
    basin_to_years = {}
    for sid, yr in annual_dict.keys():
        basin_to_years.setdefault(sid, set()).add(int(yr))
    return {sid for sid, yrs in basin_to_years.items() if year_set.issubset(yrs)}


def write_basin2vec_coverage(annual_dict, common_basins=None):
    if not is_rank0():
        return
    rows = []
    common_set = set(common_basins) if common_basins is not None else None
    for (sid, yr), _ in annual_dict.items():
        if common_set is not None and sid not in common_set:
            continue
        rows.append({"site_id": sid, "year": int(yr)})
    if not rows:
        return
    cov = pd.DataFrame(rows)
    by_year = cov.groupby("year").agg(n_basins=("site_id", "nunique")).reset_index()
    by_basin = cov.groupby("site_id").agg(
        n_years=("year", "nunique"),
        min_year=("year", "min"),
        max_year=("year", "max"),
    ).reset_index()
    cov.to_csv(OUT_DIR / "basin2vec_available_basin_years.csv", index=False)
    by_year.to_csv(OUT_DIR / "basin2vec_coverage_by_year.csv", index=False)
    by_basin.to_csv(OUT_DIR / "basin2vec_coverage_by_basin.csv", index=False)


def restrict_to_common_basins(basin_data, static_sources, basin2vec_years):
    """
    Temporal split filtering.

    All common basins are used in both train and validation. Basin2Vec can be
    used either as an annual target-year descriptor or as an aggregated static
    basin descriptor. In the aggregated mode, Basin2Vec is keyed by site_id,
    just like AlphaEarth and CAMELS attributes.
    """
    csv_basins = set(basin_data.keys())
    required_b2v_years = target_years_required_for_basin2vec()

    source_sets = {}
    if "basin2vec" in static_sources:
        if BASIN2VEC_USAGE_MODE == "annual_target_year":
            source_sets["basin2vec"] = basins_with_all_basin2vec_years(
                static_sources["basin2vec"], required_b2v_years
            )
        elif BASIN2VEC_USAGE_MODE == "aggregated_static":
            source_sets["basin2vec"] = set(static_sources["basin2vec"].keys())
        else:
            raise ValueError(f"Unknown BASIN2VEC_USAGE_MODE={BASIN2VEC_USAGE_MODE}")

    if "alphaearth" in static_sources:
        source_sets["alphaearth"] = set(static_sources["alphaearth"].keys())
    if "tessera" in static_sources:
        source_sets["tessera"] = set(static_sources["tessera"].keys())
    if "satclip" in static_sources:
        source_sets["satclip"] = set(static_sources["satclip"].keys())
    if "attributes" in static_sources:
        source_sets["attributes"] = set(static_sources["attributes"].keys())

    common = set(csv_basins)
    for ids in source_sets.values():
        common &= ids
    common_basins = sorted(common)

    rprint("\n" + "=" * 80)
    rprint("Temporal common-basin filtering")
    rprint("=" * 80)
    rprint("CAMELS dynamic basins:", len(csv_basins))
    rprint("Basin2Vec usage mode:", BASIN2VEC_USAGE_MODE)
    if BASIN2VEC_USAGE_MODE == "annual_target_year":
        rprint("Required Basin2Vec target years:", required_b2v_years)
    else:
        rprint("Basin2Vec aggregation years:", basin2vec_years)
    for name, ids in source_sets.items():
        rprint(f"{name} basins:", len(ids))
    rprint("Common basins used for both train and validation:", len(common_basins))

    if len(common_basins) == 0:
        raise RuntimeError("No common basins found across requested static sources.")

    if is_rank0():
        rows = [{"source": "camels_dynamic", "available_basins": len(csv_basins), "common_eval_basins": len(common_basins)}]
        for name, ids in source_sets.items():
            rows.append({"source": name, "available_basins": len(ids), "common_eval_basins": len(common_basins)})
        pd.DataFrame(rows).to_csv(OUT_DIR / "static_source_common_basin_coverage.csv", index=False)
        pd.DataFrame({"site_id": common_basins}).to_csv(OUT_DIR / "common_temporal_basins.csv", index=False)

    filtered_data = {sid: df for sid, df in basin_data.items() if sid in common_basins}
    filtered_static = {}

    if "basin2vec" in static_sources:
        if BASIN2VEC_USAGE_MODE == "annual_target_year":
            filtered_static["basin2vec"] = {
                (sid, yr): z
                for (sid, yr), z in static_sources["basin2vec"].items()
                if sid in common_basins and int(yr) in basin2vec_years
            }
            write_basin2vec_coverage(filtered_static["basin2vec"], common_basins=common_basins)

            missing = []
            for sid in common_basins:
                for yr in required_b2v_years:
                    if (sid, int(yr)) not in filtered_static["basin2vec"]:
                        missing.append({"site_id": sid, "missing_year": int(yr)})
            if missing:
                if is_rank0():
                    pd.DataFrame(missing).to_csv(OUT_DIR / "missing_required_basin2vec_years.csv", index=False)
                raise RuntimeError(
                    f"Basin2Vec is missing {len(missing)} required basin-year vectors after filtering. "
                    f"See {OUT_DIR / 'missing_required_basin2vec_years.csv'}"
                )
        else:
            filtered_static["basin2vec"] = {
                sid: z for sid, z in static_sources["basin2vec"].items()
                if sid in common_basins
            }

    if "alphaearth" in static_sources:
        filtered_static["alphaearth"] = {
            sid: z for sid, z in static_sources["alphaearth"].items()
            if sid in common_basins
        }
    if "tessera" in static_sources:
        filtered_static["tessera"] = {
            sid: z for sid, z in static_sources["tessera"].items()
            if sid in common_basins
        }
    if "satclip" in static_sources:
        filtered_static["satclip"] = {
            sid: z for sid, z in static_sources["satclip"].items()
            if sid in common_basins
        }
    if "attributes" in static_sources:
        filtered_static["attributes"] = {
            sid: z for sid, z in static_sources["attributes"].items()
            if sid in common_basins
        }

    return filtered_data, filtered_static, common_basins


def fit_train_scaler(basin_data, basin_ids):
    """Fit dynamic and target scalers using only training years from all temporal basins."""
    x_rows = []
    y_rows = []

    train_start = pd.Timestamp(TRAIN_START)
    train_end = pd.Timestamp(TRAIN_END)
    basin_ids = set(basin_ids)

    for sid, df in basin_data.items():
        if sid not in basin_ids:
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


def standardize_static_source(static_dict, source_name, fit_basins, transform_basins):
    """
    Standardize static descriptors without validation-target leakage.

    In aggregated_static mode, Basin2Vec is already one vector per basin and is
    standardized exactly like AlphaEarth/attributes. In annual_target_year mode,
    Basin2Vec normalization is fit on training target years and applied to all
    required annual keys.
    """
    fit_basins = sorted(set(fit_basins))
    transform_basins = sorted(set(transform_basins))

    if source_name == "basin2vec" and BASIN2VEC_USAGE_MODE == "annual_target_year":
        fit_years = train_years_for_standardization()
        transform_years = target_years_required_for_basin2vec()
        fit_keys = [(sid, yr) for sid in fit_basins for yr in fit_years if (sid, yr) in static_dict]
        all_keys = [(sid, yr) for sid in transform_basins for yr in transform_years if (sid, yr) in static_dict]
    else:
        fit_keys = [sid for sid in fit_basins if sid in static_dict]
        all_keys = [sid for sid in transform_basins if sid in static_dict]

    if len(fit_keys) == 0:
        raise RuntimeError(f"No fit keys for static source: {source_name}")
    if len(all_keys) == 0:
        raise RuntimeError(f"No transform keys for static source: {source_name}")

    M = np.vstack([static_dict[k] for k in fit_keys]).astype(np.float32)
    mean = M.mean(axis=0, keepdims=True)
    std = M.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    out = {}
    for k in all_keys:
        out[k] = ((static_dict[k][None, :] - mean) / std)[0].astype(np.float32)

    if is_rank0():
        save_json({
            "source_name": source_name,
            "basin2vec_usage_mode": BASIN2VEC_USAGE_MODE if source_name == "basin2vec" else None,
            "n_fit_keys": len(fit_keys),
            "n_transform_keys": len(all_keys),
            "fit_years_for_basin2vec": train_years_for_standardization() if (source_name == "basin2vec" and BASIN2VEC_USAGE_MODE == "annual_target_year") else None,
            "transform_years_for_basin2vec": target_years_required_for_basin2vec() if (source_name == "basin2vec" and BASIN2VEC_USAGE_MODE == "annual_target_year") else None,
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

        annual_static = (static_source == "basin2vec" and BASIN2VEC_USAGE_MODE == "annual_target_year")
        # In aggregated_static mode Basin2Vec is keyed by site_id, exactly like
        # AlphaEarth. In annual_target_year mode it is keyed by (site_id, year).
        valid_b2v_years = set(years_intersect_period(BASIN2VEC_YEARS, start_date, end_date)) if annual_static else set()

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

        if self.static_source == "basin2vec" and BASIN2VEC_USAGE_MODE == "annual_target_year":
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
    n_eval,
    n_basins,
):
    df = df.copy()
    df["model_name"] = model_name
    df["static_source"] = static_name
    df["seq_len"] = int(seq_len)
    df["lead_mode"] = "lead0_horizon1_same_day_temporal_prediction"
    df["horizon"] = int(horizon)
    df["eval_split"] = "future_years_same_basins"
    df["best_epoch"] = int(best_epoch)
    df["best_val_loss"] = float(best_val)
    df["static_dim"] = int(static_dim)
    df["n_train_samples"] = int(n_train)
    df["n_val_samples_future_years"] = int(n_val)
    df["n_eval_samples"] = int(n_eval)
    df["n_basins"] = int(n_basins)

    front = [
        "model_name", "static_source", "seq_len", "lead_mode", "horizon", "lead",
        "eval_split", "best_epoch", "best_val_loss", "static_dim",
        "n_train_samples", "n_val_samples_future_years", "n_eval_samples", "n_basins",
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
    scaler_obj,
):
    rprint("\n" + "=" * 80)
    rprint(f"Temporal experiment: model={model_name}, static={static_name}, seq_len={seq_len}, horizon={horizon}")
    rprint("=" * 80)

    model_idx = MODEL_NAMES.index(model_name) if model_name in MODEL_NAMES else 0
    static_idx = STATIC_SOURCES.index(static_name) if static_name in STATIC_SOURCES else 0
    exp_seed = int(SEED + 1000 * model_idx + 100 * static_idx + int(seq_len) + int(horizon))
    set_seed(exp_seed)

    exp_name = f"temporal_{model_name}_{static_name}_seq{seq_len}_lead0_horizon{horizon}"
    exp_dir = OUT_DIR / exp_name
    ckpt_path = exp_dir / "best_model.pt"
    last_ckpt_path = exp_dir / "last_model.pt"
    done_file = exp_dir / "DONE.json"
    global_metrics_file = exp_dir / f"val_global_metrics_horizon{horizon}.csv"
    per_basin_summary_file = exp_dir / f"val_per_basin_summary_horizon{horizon}.csv"

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
        fit_basins=common_basins,
        transform_basins=common_basins,
    )

    train_ds = StreamflowWindowDataset(
        basin_data=basin_data,
        static_dict=static_dict,
        static_source=static_name,
        seq_len=seq_len,
        horizon=horizon,
        split="train_temporal",
        start_date=TRAIN_START,
        end_date=TRAIN_END,
        basin_ids=common_basins,
    )

    val_ds = StreamflowWindowDataset(
        basin_data=basin_data,
        static_dict=static_dict,
        static_source=static_name,
        seq_len=seq_len,
        horizon=horizon,
        split="val_temporal_future_years",
        start_date=VAL_START,
        end_date=VAL_END,
        basin_ids=common_basins,
    )

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(
            f"Empty temporal split: model={model_name}, source={static_name}, seq_len={seq_len}, "
            f"horizon={horizon}, train={len(train_ds)}, val={len(val_ds)}. "
            "For Basin2Vec, check missing_required_basin2vec_years.csv and basin2vec_coverage_by_year.csv."
        )

    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=WORLD_SIZE,
        rank=RANK,
        shuffle=True,
        seed=SEED,
    ) if DISTRIBUTED else None
    val_sampler = DistributedSampler(
        val_ds,
        num_replicas=WORLD_SIZE,
        rank=RANK,
        shuffle=False,
        seed=SEED,
    ) if DISTRIBUTED else None

    train_loader = make_loader(train_ds, BATCH_SIZE_PER_GPU, shuffle=(train_sampler is None), sampler=train_sampler)
    val_loader = make_loader(val_ds, BATCH_SIZE_PER_GPU, shuffle=False, sampler=val_sampler)

    static_dim = len(next(iter(static_dict.values())))

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
            rprint(
                f"[RESUME] Resuming {exp_name} from epoch {start_epoch}; "
                f"best_epoch={best_epoch}, best_val={best_val:.6f}"
            )
        except Exception as e:
            rprint(f"[RESUME] Could not load last checkpoint for {exp_name}; restarting. Error: {e}")
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
                    "static_dim": int(static_dim),
                    "dynamic_cols": DYNAMIC_COLS,
                    "target_col": TARGET_COL,
                    "scaler_obj": scaler_obj,
                    "common_basins": common_basins,
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
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss_future_years": float(val_loss),
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
                    "epoch": int(epoch),
                    "best_val": float(best_val),
                    "best_epoch": int(best_epoch),
                    "bad_epochs": int(bad_epochs),
                    "static_dim": int(static_dim),
                    "dynamic_cols": DYNAMIC_COLS,
                    "target_col": TARGET_COL,
                    "scaler_obj": scaler_obj,
                    "common_basins": common_basins,
                }, last_ckpt_path)
            rprint(
                f"model={model_name} | static={static_name} | seq={seq_len} | H={horizon} | "
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
            eval_global, model_name, static_name, seq_len, horizon, best_epoch, best_val,
            static_dim, len(train_ds), len(val_ds), len(val_ds), len(common_basins),
        )
        eval_per_basin = add_experiment_metadata(
            eval_per_basin, model_name, static_name, seq_len, horizon, best_epoch, best_val,
            static_dim, len(train_ds), len(val_ds), len(val_ds), len(common_basins),
        )
        eval_per_basin_summary = add_experiment_metadata(
            eval_per_basin_summary, model_name, static_name, seq_len, horizon, best_epoch, best_val,
            static_dim, len(train_ds), len(val_ds), len(val_ds), len(common_basins),
        )

        pred_df["model_name"] = model_name
        pred_df["static_source"] = static_name
        pred_df["seq_len"] = int(seq_len)
        pred_df["lead_mode"] = "lead0_horizon1_same_day_temporal_prediction"
        pred_df["eval_split"] = "future_years_same_basins"

        eval_global.to_csv(exp_dir / f"val_global_metrics_horizon{horizon}.csv", index=False)
        eval_per_basin.to_csv(exp_dir / f"val_per_basin_metrics_horizon{horizon}.csv", index=False)
        eval_per_basin_summary.to_csv(exp_dir / f"val_per_basin_summary_horizon{horizon}.csv", index=False)
        pred_df.to_csv(exp_dir / f"val_predictions_horizon{horizon}.csv", index=False)

        save_json({
            "status": "done",
            "model_name": model_name,
            "static_source": static_name,
            "seq_len": int(seq_len),
            "lead": int(LEAD),
            "horizon": int(horizon),
            "split_type": "temporal_future_years_same_basins",
            "train_start": TRAIN_START,
            "train_end": TRAIN_END,
            "val_start": VAL_START,
            "val_end": VAL_END,
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val),
            "finished_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "global_metrics_file": str(global_metrics_file),
            "per_basin_summary_file": str(per_basin_summary_file),
        }, done_file)

        save_json({
            "model_name": model_name,
            "static_source": static_name,
            "seq_len": int(seq_len),
            "lead": int(LEAD),
            "horizon": int(horizon),
            "task": "daily streamflow prediction / rainfall-runoff simulation; input meteorology through target day; target Q(t)",
            "split_type": "temporal_future_years_same_basins",
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val),
            "static_dim": int(static_dim),
            "n_train_samples": int(len(train_ds)),
            "n_val_samples": int(len(val_ds)),
            "n_basins": int(len(common_basins)),
            "dynamic_cols": DYNAMIC_COLS,
            "target_col": TARGET_COL,
            "no_past_discharge_input": True,
            "camels_root": str(CAMELS_ROOT),
            "basin2vec_npz": str(BASIN2VEC_NPZ),
            "alphaearth_npz": str(ALPHAEARTH_NPZ),
            "tessera_npz": str(TESSERA_NPZ),
            "satclip_npz": str(SATCLIP_NPZ),
            "tessera_agg_years": [int(y) for y in TESSERA_AGG_YEARS],
            "satclip_agg_years": [int(y) for y in SATCLIP_AGG_YEARS],
            "note": "Basin2Vec is averaged across BASIN2VEC_YEARS to create one static vector per basin. TESSERA and SatCLIP are treated as static basin-level descriptors from their 8-point mean NPZ files. Moirai entry is a supervised Moirai-style patch Transformer, not an official pretrained Moirai checkpoint.",
        }, exp_dir / "config_summary.json")

        rprint("\nValidation global metrics:")
        rprint(eval_global)
        rprint("\nValidation per-basin summary:")
        rprint(eval_per_basin_summary)

    barrier()

    try:
        del model, optimizer, scheduler, train_loader, val_loader, train_ds, val_ds
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
    model_name = kwargs.get("model_name", args[0] if len(args) > 0 else None)
    static_name = kwargs.get("static_name", args[1] if len(args) > 1 else None)
    seq_len = kwargs.get("seq_len", args[3] if len(args) > 3 else None)
    horizon = kwargs.get("horizon", args[4] if len(args) > 4 else None)

    try:
        return run_experiment(*args, **kwargs)
    except Exception as e:
        err_text = traceback.format_exc()
        rprint("\n" + "!" * 80)
        rprint(
            f"Skipping failed temporal experiment: model={model_name}, static={static_name}, "
            f"seq_len={seq_len}, horizon={horizon}"
        )
        rprint(f"Error: {type(e).__name__}: {e}")
        rprint("!" * 80 + "\n")
        append_failure_log({
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


def check_temporal_split_dates():
    train_end = pd.Timestamp(TRAIN_END)
    val_start = pd.Timestamp(VAL_START)
    if val_start <= train_end:
        rprint("\n" + "!" * 80)
        rprint("WARNING: TRAIN_END and VAL_START overlap in target dates.")
        rprint(f"TRAIN_END={TRAIN_END}, VAL_START={VAL_START}.")
        rprint("For strict future-year validation, use TRAIN_END before VAL_START.")
        rprint("!" * 80 + "\n")


def main():
    rprint("Device:", DEVICE)
    rprint("Distributed:", DISTRIBUTED)
    rprint("World size:", WORLD_SIZE)
    rprint("Output directory:", OUT_DIR)
    rprint("CAMELS root:", CAMELS_ROOT)
    rprint("Models:", MODEL_NAMES)
    rprint("No past discharge input: True")
    rprint("Static sources:", STATIC_SOURCES)
    rprint("Sequence lengths:", SEQ_LENGTHS)
    rprint("Horizons:", HORIZONS)
    rprint("Lead:", LEAD, "meaning H=1 predicts Q(t) because meteorology is available through t")
    rprint("Temporal split: all common basins are used in both train and validation; dates differ.")
    rprint("Resume skip completed:", RESUME_SKIP_COMPLETED)
    rprint("Resume interrupted training:", RESUME_TRAINING_FROM_LAST_CHECKPOINT)
    rprint("Train/val dates:", TRAIN_START, "to", TRAIN_END, "|", VAL_START, "to", VAL_END)
    rprint("Basin2Vec years:", BASIN2VEC_YEARS[0], "to", BASIN2VEC_YEARS[-1])
    rprint("Basin2Vec usage mode:", BASIN2VEC_USAGE_MODE)
    rprint("AlphaEarth aggregation years:", ALPHAEARTH_AGG_YEARS)
    rprint("TESSERA aggregation years:", TESSERA_AGG_YEARS)
    rprint("SatCLIP aggregation years:", SATCLIP_AGG_YEARS)
    rprint("TESSERA NPZ:", TESSERA_NPZ)
    rprint("SatCLIP NPZ:", SATCLIP_NPZ)

    check_temporal_split_dates()

    basin_data_raw = load_camels_data()
    check_year_coverage(basin_data_raw)

    all_static_sources = {}

    if "basin2vec" in STATIC_SOURCES:
        if BASIN2VEC_USAGE_MODE == "aggregated_static":
            all_static_sources["basin2vec"] = load_basin2vec_aggregated_embeddings(
                BASIN2VEC_NPZ,
                INDEX_PARQUET,
                BASIN2VEC_YEARS,
            )
        elif BASIN2VEC_USAGE_MODE == "annual_target_year":
            all_static_sources["basin2vec"] = load_basin2vec_annual_embeddings(
                BASIN2VEC_NPZ,
                INDEX_PARQUET,
                BASIN2VEC_YEARS,
            )
        else:
            raise ValueError(f"Unknown BASIN2VEC_USAGE_MODE={BASIN2VEC_USAGE_MODE}")

    if "alphaearth" in STATIC_SOURCES:
        all_static_sources["alphaearth"] = load_alphaearth_aggregated_embeddings(
            ALPHAEARTH_NPZ,
            ALPHAEARTH_AGG_YEARS,
        )

    if "tessera" in STATIC_SOURCES:
        all_static_sources["tessera"] = load_generic_static_npz_embeddings(
            TESSERA_NPZ,
            source_name="tessera",
            years_keep=TESSERA_AGG_YEARS,
        )

    if "satclip" in STATIC_SOURCES:
        all_static_sources["satclip"] = load_generic_static_npz_embeddings(
            SATCLIP_NPZ,
            source_name="satclip",
            years_keep=SATCLIP_AGG_YEARS,
        )

    if "attributes" in STATIC_SOURCES:
        all_static_sources["attributes"] = read_camels_attributes_static()

    basin_data_raw, static_sources, common_basins = restrict_to_common_basins(
        basin_data=basin_data_raw,
        static_sources=all_static_sources,
        basin2vec_years=BASIN2VEC_YEARS,
    )

    if is_rank0():
        split_dir = OUT_DIR / "splits"
        split_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"site_id": common_basins, "temporal_split": "train_and_val_same_basin"}).to_csv(
            split_dir / "camels_temporal_common_basins.csv",
            index=False,
        )
        save_json({
            "task": "daily streamflow prediction / rainfall-runoff simulation",
            "split_type": "temporal_future_years_same_basins",
            "input": "meteorology through target day t plus descriptor",
            "target": "same-day discharge Q(t) for H=1, LEAD=0",
            "train_basins": "all common basins",
            "validation_basins": "same common basins",
            "n_common_basins": int(len(common_basins)),
            "static_sources": STATIC_SOURCES,
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
            "basin2vec_years": [int(y) for y in BASIN2VEC_YEARS],
            "basin2vec_agg_years": [int(y) for y in BASIN2VEC_YEARS],
            "alphaearth_agg_years": [int(y) for y in ALPHAEARTH_AGG_YEARS],
            "tessera_agg_years": [int(y) for y in TESSERA_AGG_YEARS],
            "satclip_agg_years": [int(y) for y in SATCLIP_AGG_YEARS],
            "basin2vec_npz": str(BASIN2VEC_NPZ),
            "alphaearth_npz": str(ALPHAEARTH_NPZ),
            "tessera_npz": str(TESSERA_NPZ),
            "satclip_npz": str(SATCLIP_NPZ),
            "basin2vec_usage": "aggregated static Basin2Vec vector: average over BASIN2VEC_YEARS per basin",
            "tessera_usage": "static TESSERA basin descriptor: average over TESSERA_AGG_YEARS per basin",
            "satclip_usage": "static SatCLIP basin descriptor: average over SATCLIP_AGG_YEARS per basin",
        }, OUT_DIR / "benchmark_config_summary.json")

    scaler_obj = None
    basin_data = basin_data_raw
    if STANDARDIZE_DYNAMIC_AND_TARGET_FROM_TRAIN:
        scaler_obj = fit_train_scaler(basin_data_raw, common_basins)
        basin_data = apply_scaler(basin_data_raw, scaler_obj)
        if is_rank0():
            scaler_dir = OUT_DIR / "scalers"
            scaler_dir.mkdir(parents=True, exist_ok=True)
            save_json(scaler_obj, scaler_dir / "dynamic_target_scaler_temporal_train_years_all_basins.json")

    all_global = []
    all_summary = []

    for model_name in MODEL_NAMES:
        for static_name in STATIC_SOURCES:
            if static_name not in static_sources:
                append_failure_log({
                    "model_name": model_name,
                    "static_source": static_name,
                    "seq_len": None,
                    "horizon": None,
                    "error_type": "MissingStaticSource",
                    "error_message": f"Requested static source missing after filtering: {static_name}",
                    "traceback": "",
                })
                continue

            for seq_len in SEQ_LENGTHS:
                for horizon in HORIZONS:
                    eval_global, eval_summary = run_experiment_safe(
                        model_name=model_name,
                        static_name=static_name,
                        static_dict_raw=static_sources[static_name],
                        seq_len=seq_len,
                        horizon=horizon,
                        basin_data=basin_data,
                        common_basins=common_basins,
                        scaler_obj=scaler_obj,
                    )

                    if is_rank0():
                        if eval_global is not None:
                            all_global.append(eval_global)
                        if eval_summary is not None:
                            all_summary.append(eval_summary)

                        if all_global:
                            pd.concat(all_global, ignore_index=True).to_csv(
                                OUT_DIR / "ALL_CAMELS_temporal_global_metrics_partial.csv",
                                index=False,
                            )
                        if all_summary:
                            pd.concat(all_summary, ignore_index=True).to_csv(
                                OUT_DIR / "ALL_CAMELS_temporal_per_basin_summary_partial.csv",
                                index=False,
                            )

    if is_rank0():
        if all_global:
            all_global_df = pd.concat(all_global, ignore_index=True)
            all_global_df.to_csv(OUT_DIR / "ALL_CAMELS_temporal_global_metrics.csv", index=False)
        else:
            all_global_df = pd.DataFrame()

        if all_summary:
            all_summary_df = pd.concat(all_summary, ignore_index=True)
            all_summary_df.to_csv(OUT_DIR / "ALL_CAMELS_temporal_per_basin_summary.csv", index=False)

            paper = all_summary_df.copy()
            paper = paper.sort_values(
                ["model_name", "seq_len", "horizon", "lead", "median_NSE"],
                ascending=[True, True, True, True, False],
            )
            paper.to_csv(OUT_DIR / "PAPER_TABLE_CAMELS_temporal_per_basin_summary.csv", index=False)

            metric_cols = [
                "median_NSE", "mean_NSE", "median_KGE", "mean_KGE",
                "median_RMSE", "mean_RMSE", "median_MAE", "mean_MAE",
                "median_Pearson_r", "mean_Pearson_r",
            ]
            metric_cols = [c for c in metric_cols if c in all_summary_df.columns]
            group_cols = ["model_name", "static_source", "seq_len", "horizon", "lead"]
            agg_dict = {}
            for c in metric_cols:
                agg_dict[f"{c}_mean"] = (c, "mean")
                agg_dict[f"{c}_std"] = (c, "std")
                agg_dict[f"{c}_median"] = (c, "median")
            compact_summary = (
                all_summary_df
                .groupby(group_cols, dropna=False)
                .agg(n_rows=("model_name", "size"), **agg_dict)
                .reset_index()
            )
            compact_summary.to_csv(OUT_DIR / "TEMPORAL_AGGREGATED_CAMELS_per_basin_summary.csv", index=False)

            if {"basin2vec", "alphaearth"}.issubset(set(all_summary_df["static_source"].unique())):
                pair_keys = ["model_name", "seq_len", "horizon", "lead"]
                b2v = all_summary_df[all_summary_df["static_source"] == "basin2vec"].copy()
                ae = all_summary_df[all_summary_df["static_source"] == "alphaearth"].copy()
                keep_metrics = [
                    "median_NSE", "mean_NSE", "median_KGE", "mean_KGE",
                    "median_RMSE", "mean_RMSE", "median_MAE", "mean_MAE",
                    "median_Pearson_r", "mean_Pearson_r",
                ]
                keep_metrics = [c for c in keep_metrics if c in all_summary_df.columns]
                paired = b2v[pair_keys + keep_metrics].merge(
                    ae[pair_keys + keep_metrics],
                    on=pair_keys,
                    suffixes=("_basin2vec", "_alphaearth"),
                    how="inner",
                )
                for c in keep_metrics:
                    paired[f"delta_{c}_basin2vec_minus_alphaearth"] = paired[f"{c}_basin2vec"] - paired[f"{c}_alphaearth"]
                paired.to_csv(OUT_DIR / "PAIRED_BASIN2VEC_MINUS_ALPHAEARTH_temporal.csv", index=False)

            # General paired comparisons: Basin2Vec minus every other static source
            # on the same model, sequence length, horizon, and lead.
            if "basin2vec" in set(all_summary_df["static_source"].unique()):
                pair_keys = ["model_name", "seq_len", "horizon", "lead"]
                keep_metrics = [
                    "median_NSE", "mean_NSE", "median_KGE", "mean_KGE",
                    "median_RMSE", "mean_RMSE", "median_MAE", "mean_MAE",
                    "median_Pearson_r", "mean_Pearson_r",
                ]
                keep_metrics = [c for c in keep_metrics if c in all_summary_df.columns]
                b2v_base = all_summary_df[all_summary_df["static_source"] == "basin2vec"].copy()
                paired_rows = []
                for other_source in sorted(set(all_summary_df["static_source"].unique()) - {"basin2vec"}):
                    other = all_summary_df[all_summary_df["static_source"] == other_source].copy()
                    pp = b2v_base[pair_keys + keep_metrics].merge(
                        other[pair_keys + keep_metrics],
                        on=pair_keys,
                        suffixes=("_basin2vec", f"_{other_source}"),
                        how="inner",
                    )
                    if len(pp) == 0:
                        continue
                    pp["comparison_source"] = other_source
                    for c in keep_metrics:
                        pp[f"delta_{c}_basin2vec_minus_{other_source}"] = pp[f"{c}_basin2vec"] - pp[f"{c}_{other_source}"]
                    paired_rows.append(pp)
                if paired_rows:
                    pd.concat(paired_rows, ignore_index=True).to_csv(
                        OUT_DIR / "PAIRED_BASIN2VEC_MINUS_ALL_STATIC_SOURCES_temporal.csv",
                        index=False,
                    )

        rprint("\nSaved temporal-split results:")
        rprint(OUT_DIR / "ALL_CAMELS_temporal_global_metrics.csv")
        rprint(OUT_DIR / "ALL_CAMELS_temporal_per_basin_summary.csv")
        rprint(OUT_DIR / "PAPER_TABLE_CAMELS_temporal_per_basin_summary.csv")
        rprint(OUT_DIR / "TEMPORAL_AGGREGATED_CAMELS_per_basin_summary.csv")
        rprint(OUT_DIR / "PAIRED_BASIN2VEC_MINUS_ALPHAEARTH_temporal.csv")
        rprint(OUT_DIR / "FAILED_EXPERIMENTS.csv")


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_distributed()
