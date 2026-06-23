#!/usr/bin/env python3

from pathlib import Path
import re

import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Paths
# --------------------------------------------------
NPZ_PATH = Path(
    "../emb/src/evaluation_outputs/"
    "HCLV4_grouped_monthly_static_meta_areaaux_64d/basin_embeddings_full.npz"
)

INDEX_HTML = Path("static/index.html")

GEOJSON_PATH = Path(
    "/data/basin2vec/raw/gages-ii/us_selected_basins/basins.geojson"
)

# --------------------------------------------------
# Representation choice
# --------------------------------------------------
# Use "hydrologic_repr" for downstream hydrologic similarity.
# Use "embeddings" for final contrastive projection-space similarity.
REPRESENTATION_KEY = "hydrologic_repr"


# --------------------------------------------------
# Utilities
# --------------------------------------------------
def normalize_site_id(x) -> str:
    """
    Normalize USGS/GAGES site IDs.

    Examples:
        1013500   -> 01013500
        01013500  -> 01013500
        01013500.0 -> 01013500
    """
    s = str(x).strip()

    if s.endswith(".0"):
        s = s[:-2]

    s = re.sub(r"\D", "", s)

    if len(s) == 0:
        return ""

    if len(s) <= 8:
        s = s.zfill(8)

    return s


def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def normalize_vector(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x), eps, None)


def minmax_scale(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x_min = np.min(x)
    x_max = np.max(x)
    return (x - x_min) / (x_max - x_min + eps)


def select_representation(data: np.lib.npyio.NpzFile, key: str) -> tuple[np.ndarray, str]:
    """
    Select representation from Basin2Vec NPZ.

    New HCLV4 NPZ usually contains:
        embeddings      : final contrastive projection z
        hydrologic_repr : pre-projection hydrologic representation h
    """
    if key in data.files:
        return data[key].astype(np.float32), key

    if "hydrologic_repr" in data.files:
        return data["hydrologic_repr"].astype(np.float32), "hydrologic_repr"

    if "embeddings" in data.files:
        return data["embeddings"].astype(np.float32), "embeddings"

    raise KeyError(
        "NPZ must contain either 'hydrologic_repr' or 'embeddings'. "
        f"Available keys: {data.files}"
    )


# --------------------------------------------------
# Load embedding file
# --------------------------------------------------
if not NPZ_PATH.exists():
    raise FileNotFoundError(f"NPZ file not found: {NPZ_PATH}")

data = np.load(NPZ_PATH, allow_pickle=True)

sample_embeddings, loaded_representation_key = select_representation(
    data,
    REPRESENTATION_KEY,
)

if "labels" not in data.files:
    raise KeyError(f"'labels' not found in {NPZ_PATH}. Available keys: {data.files}")

if "years" not in data.files:
    raise KeyError(f"'years' not found in {NPZ_PATH}. Available keys: {data.files}")

sample_labels = np.array([normalize_site_id(x) for x in data["labels"]], dtype=str)
sample_years = data["years"].astype(int)

if len(sample_embeddings) != len(sample_labels) or len(sample_embeddings) != len(sample_years):
    raise ValueError(
        "NPZ length mismatch: "
        f"embeddings={len(sample_embeddings)}, "
        f"labels={len(sample_labels)}, "
        f"years={len(sample_years)}"
    )

# optional metadata
year_counts = data["year_counts"].astype(int) if "year_counts" in data.files else None
area_target = data["area_target"].astype(np.float32) if "area_target" in data.files else None
area_pred = data["area_pred"].astype(np.float32) if "area_pred" in data.files else None
overlap_degree = data["overlap_degree"].astype(np.float32) if "overlap_degree" in data.files else None

# normalize selected sample-level representation
sample_embeddings = normalize_rows(sample_embeddings)

# --------------------------------------------------
# Basin-level average embeddings
# Important:
#   Recompute basin averages from selected representation.
#   Do not use data["avg_embeddings"] if REPRESENTATION_KEY = "hydrologic_repr",
#   because avg_embeddings may have been computed from "embeddings".
# --------------------------------------------------
basin_to_vecs: dict[str, list[np.ndarray]] = {}

for basin_id, emb in zip(sample_labels, sample_embeddings):
    if basin_id == "":
        continue
    basin_to_vecs.setdefault(basin_id, []).append(emb)

all_basin_ids = np.array(sorted(basin_to_vecs.keys()), dtype=str)
avg_list = []

for basin_id in all_basin_ids:
    vecs = np.stack(basin_to_vecs[basin_id], axis=0)
    mean_vec = vecs.mean(axis=0)
    mean_vec = normalize_vector(mean_vec)
    avg_list.append(mean_vec)

avg_embeddings = np.stack(avg_list, axis=0).astype(np.float32)
avg_embeddings = normalize_rows(avg_embeddings)

all_id_to_index = {bid: i for i, bid in enumerate(all_basin_ids)}

# --------------------------------------------------
# Optional basin-level metadata summaries
# --------------------------------------------------
basin_year_count_map = None
if year_counts is not None and len(year_counts) == len(all_basin_ids):
    basin_year_count_map = {
        str(bid): int(c)
        for bid, c in zip(all_basin_ids, year_counts)
    }
else:
    basin_year_count_map = {
        bid: len(set(sample_years[sample_labels == bid].astype(int).tolist()))
        for bid in all_basin_ids
    }

basin_overlap_degree_map = {}
if overlap_degree is not None and len(overlap_degree) == len(sample_labels):
    for basin_id in all_basin_ids:
        vals = overlap_degree[sample_labels == basin_id]
        if len(vals) > 0:
            basin_overlap_degree_map[basin_id] = float(np.nanmean(vals))

basin_area_target_map = {}
basin_area_pred_map = {}
if (
    area_target is not None
    and area_pred is not None
    and len(area_target) == len(sample_labels)
    and len(area_pred) == len(sample_labels)
):
    for basin_id in all_basin_ids:
        idx = sample_labels == basin_id
        if np.any(idx):
            basin_area_target_map[basin_id] = float(np.nanmean(area_target[idx]))
            basin_area_pred_map[basin_id] = float(np.nanmean(area_pred[idx]))


# --------------------------------------------------
# Build year-specific embedding tables from sample-level storage
#
# year_data[year] = {
#     "basin_ids": np.array([...]),
#     "embeddings": np.ndarray [num_basins_that_year, dim],
#     "id_to_index": {...}
# }
#
# If a basin appears multiple times in the same year, average and renormalize.
# --------------------------------------------------
year_data = {}
unique_years = np.unique(sample_years)

for yr in unique_years:
    idx = np.where(sample_years == yr)[0]

    yr_labels = sample_labels[idx]
    yr_emb = sample_embeddings[idx]

    basin_to_vecs_year: dict[str, list[np.ndarray]] = {}

    for basin_id, emb in zip(yr_labels, yr_emb):
        if basin_id == "":
            continue
        basin_to_vecs_year.setdefault(basin_id, []).append(emb)

    basin_ids_this_year = []
    emb_list_this_year = []

    for basin_id in sorted(basin_to_vecs_year.keys()):
        vecs = np.stack(basin_to_vecs_year[basin_id], axis=0)
        mean_vec = vecs.mean(axis=0)
        mean_vec = normalize_vector(mean_vec)

        basin_ids_this_year.append(basin_id)
        emb_list_this_year.append(mean_vec)

    if len(emb_list_this_year) == 0:
        continue

    basin_ids_this_year = np.array(basin_ids_this_year, dtype=str)
    emb_matrix_this_year = np.stack(emb_list_this_year, axis=0).astype(np.float32)
    emb_matrix_this_year = normalize_rows(emb_matrix_this_year)

    year_data[int(yr)] = {
        "basin_ids": basin_ids_this_year,
        "embeddings": emb_matrix_this_year,
        "id_to_index": {
            bid: i for i, bid in enumerate(basin_ids_this_year)
        },
    }

year_list = sorted(year_data.keys())

print(f"[INFO] NPZ: {NPZ_PATH}")
print(f"[INFO] Requested representation: {REPRESENTATION_KEY}")
print(f"[INFO] Loaded representation: {loaded_representation_key}")
print(f"[INFO] Loaded averaged basins: {len(all_basin_ids)}")
print(f"[INFO] Loaded sample embeddings: {len(sample_embeddings)}")
print(f"[INFO] Embedding dim: {avg_embeddings.shape[1]}")
print(
    f"[INFO] Available years: "
    f"{year_list[:5]} ... {year_list[-5:] if len(year_list) > 5 else year_list}"
)


# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.get("/")
def root():
    return FileResponse(INDEX_HTML)


@app.get("/geojson")
def geojson():
    return FileResponse(GEOJSON_PATH)


@app.get("/years")
def get_years():
    return {"years": year_list}


@app.get("/basins")
def get_basins():
    return {"basin_ids": all_basin_ids.tolist()}


@app.get("/metadata")
def metadata():
    out = {
        "npz_path": str(NPZ_PATH),
        "requested_representation_key": REPRESENTATION_KEY,
        "loaded_representation_key": loaded_representation_key,
        "num_unique_basins": int(len(all_basin_ids)),
        "num_samples": int(len(sample_embeddings)),
        "num_years": int(len(year_list)),
        "years": year_list,
        "embedding_dim": int(avg_embeddings.shape[1]),
    }

    if basin_year_count_map:
        counts = np.array(list(basin_year_count_map.values()), dtype=float)
        out["year_count_min"] = int(np.nanmin(counts))
        out["year_count_max"] = int(np.nanmax(counts))
        out["year_count_mean"] = float(np.nanmean(counts))

    if basin_overlap_degree_map:
        vals = np.array(list(basin_overlap_degree_map.values()), dtype=float)
        out["overlap_degree_min"] = float(np.nanmin(vals))
        out["overlap_degree_max"] = float(np.nanmax(vals))
        out["overlap_degree_mean"] = float(np.nanmean(vals))

    if basin_area_target_map and basin_area_pred_map:
        target = np.array(
            [basin_area_target_map[b] for b in all_basin_ids if b in basin_area_target_map],
            dtype=float,
        )
        pred = np.array(
            [basin_area_pred_map[b] for b in all_basin_ids if b in basin_area_pred_map],
            dtype=float,
        )
        if len(target) == len(pred) and len(target) > 1:
            out["area_aux_corr"] = float(np.corrcoef(target, pred)[0, 1])
            out["area_aux_mae"] = float(np.nanmean(np.abs(target - pred)))

    return out


@app.get("/basin-info/{gage_id}")
def basin_info(gage_id: str):
    gage_id = normalize_site_id(gage_id)

    if gage_id not in all_id_to_index:
        return {"error": "Basin not found", "gage_id": gage_id}

    out = {
        "gage_id": gage_id,
        "index": int(all_id_to_index[gage_id]),
        "years_available": sorted(
            set(sample_years[sample_labels == gage_id].astype(int).tolist())
        ),
        "year_count": int(basin_year_count_map.get(gage_id, 0))
        if basin_year_count_map else None,
    }

    if gage_id in basin_overlap_degree_map:
        out["overlap_degree"] = float(basin_overlap_degree_map[gage_id])

    if gage_id in basin_area_target_map:
        out["area_target_z"] = float(basin_area_target_map[gage_id])

    if gage_id in basin_area_pred_map:
        out["area_pred_z"] = float(basin_area_pred_map[gage_id])

    return out


@app.get("/similarity/{gage_id}")
def similarity(
    gage_id: str,
    year: int | None = Query(default=None),
    raw: bool = Query(default=False),
):
    """
    Return similarity map for a selected basin.

    Parameters
    ----------
    gage_id:
        USGS/GAGES basin ID.

    year:
        If None, use basin-level average representation.
        If provided, use year-specific representation.

    raw:
        If False, return min-max scaled similarity in [0, 1].
        If True, return raw cosine similarity.
    """
    gage_id = normalize_site_id(gage_id)

    # ----------------------------------------------
    # Basin-level averaged similarity
    # ----------------------------------------------
    if year is None:
        if gage_id not in all_id_to_index:
            return {
                "error": "Basin not found",
                "gage_id": gage_id,
            }

        idx = all_id_to_index[gage_id]
        query = avg_embeddings[idx:idx + 1]
        sims = (query @ avg_embeddings.T)[0]

        if not raw:
            sims = minmax_scale(sims)

        return {
            str(all_basin_ids[i]): float(sims[i])
            for i in range(len(all_basin_ids))
        }

    # ----------------------------------------------
    # Year-specific similarity
    # Only basins present in that year get scores.
    # Others return None so frontend can style missing values.
    # ----------------------------------------------
    if year not in year_data:
        return {
            "error": "Year not available",
            "year": int(year),
            "available_years": year_list,
        }

    yinfo = year_data[year]

    if gage_id not in yinfo["id_to_index"]:
        return {
            "error": f"Basin {gage_id} not available in year {year}",
            "gage_id": gage_id,
            "year": int(year),
        }

    idx = yinfo["id_to_index"][gage_id]
    emb_matrix = yinfo["embeddings"]
    basin_ids_this_year = yinfo["basin_ids"]

    query = emb_matrix[idx:idx + 1]
    sims = (query @ emb_matrix.T)[0]

    if not raw:
        sims = minmax_scale(sims)

    sim_map = {str(bid): None for bid in all_basin_ids}

    for i, bid in enumerate(basin_ids_this_year):
        sim_map[str(bid)] = float(sims[i])

    return sim_map


@app.get("/topk/{gage_id}")
def topk(
    gage_id: str,
    year: int | None = Query(default=None),
    k: int = Query(default=20, ge=1, le=200),
):
    """
    Return top-k most similar basins.

    This is useful for debugging the map output.
    """
    gage_id = normalize_site_id(gage_id)

    if year is None:
        if gage_id not in all_id_to_index:
            return {
                "error": "Basin not found",
                "gage_id": gage_id,
            }

        idx = all_id_to_index[gage_id]
        query = avg_embeddings[idx:idx + 1]
        sims = (query @ avg_embeddings.T)[0]

        order = np.argsort(-sims)
        rows = []

        for rank, i in enumerate(order[:k], start=1):
            rows.append({
                "rank": rank,
                "gage_id": str(all_basin_ids[i]),
                "similarity": float(sims[i]),
            })

        return {
            "query_gage_id": gage_id,
            "year": None,
            "representation_key": loaded_representation_key,
            "topk": rows,
        }

    if year not in year_data:
        return {
            "error": "Year not available",
            "year": int(year),
            "available_years": year_list,
        }

    yinfo = year_data[year]

    if gage_id not in yinfo["id_to_index"]:
        return {
            "error": f"Basin {gage_id} not available in year {year}",
            "gage_id": gage_id,
            "year": int(year),
        }

    idx = yinfo["id_to_index"][gage_id]
    emb_matrix = yinfo["embeddings"]
    basin_ids_this_year = yinfo["basin_ids"]

    query = emb_matrix[idx:idx + 1]
    sims = (query @ emb_matrix.T)[0]

    order = np.argsort(-sims)
    rows = []

    for rank, i in enumerate(order[:k], start=1):
        rows.append({
            "rank": rank,
            "gage_id": str(basin_ids_this_year[i]),
            "similarity": float(sims[i]),
        })

    return {
        "query_gage_id": gage_id,
        "year": int(year),
        "representation_key": loaded_representation_key,
        "topk": rows,
    }
