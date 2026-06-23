#!/usr/bin/env python3
# ==========================================================
# Build pre-stacked grouped monthly/static Basin2Vec Zarr
# ==========================================================
#
# Output:
#
#   /data/basin2vec/cache/patches_step4_grouped/grouped_monthly_static.zarr
#
# Arrays:
#
#   monthly_x : [N_basin, N_year, 12, C_monthly, H, W]
#   static_x  : [N_basin, N_year, C_static, H, W]
#   done      : [N_basin, N_year]
#
# This script reuses Basin2VecMixedMonthlyDataset, so the output is already:
#   - normalized
#   - log1p-transformed for prcp/swe/we where configured
#   - land_cover one-hot encoded
#
# Important fixes:
#   - clips values before float16 casting to prevent overflow
#   - uses lz4 compression by default for faster training reads
#   - supports --resume
#
# Example:
#
#   python build_grouped_monthly_static_zarr.py \
#     --out-dtype float16 \
#     --batch-size 64 \
#     --num-workers 16 \
#     --compressor lz4 \
#     --compression-level 1 \
#     --overwrite
#
# Resume:
#
#   python build_grouped_monthly_static_zarr.py \
#     --out-dtype float16 \
#     --batch-size 64 \
#     --num-workers 16 \
#     --resume
#
# ==========================================================

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import zarr
from numcodecs import Blosc

from basin2vec_dataset_monthly import Basin2VecMixedMonthlyDataset


# ------------------------------------------------------------
# Default paths
# ------------------------------------------------------------
INDEX_PARQUET = "../config/training_step5/sample_index.parquet"
INDEX_META_JSON = "../config/training_step5/sample_index_meta.json"
NORM_STATS_JSON = "../config/training_step5/norm_stats.json"

MONTHLY_ZARR_ROOT = "/data/basin2vec/cache/patches_step4_monthly"
ANNUAL_ZARR_ROOT = "/data/basin2vec/cache/patches_step4"
MASK_ZARR_PATH = "/data/basin2vec/cache/static_masks.zarr"

OUT_ZARR = "/data/basin2vec/cache/patches_step4_grouped/grouped_monthly_static.zarr"

MONTHLY_VARIABLES = [
    "prcp",
    "tmax",
    "tmin",
    "vp",
    "swe",
]

NONMONTHLY_VARIABLES = [
    "bdod",
    "clay",
    "dem",
    "hydraulic_conductivity",
    "land_cover",
    "permeability",
    "pet",
    "population",
    "sand",
    "slit",
    "soc",
    "wind",
]

VARIABLE_FILE_ALIASES = {
    "slit": "silt",
    "we": "swe",
}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def safe_clip_and_cast(
    x: np.ndarray,
    out_dtype: np.dtype,
    clip_value: float,
) -> np.ndarray:
    """
    Prevent float16 overflow and remove NaN/Inf before writing.

    The encoder also clamps to [-10, 10], so clipping here is consistent
    with the training-time numerical range.
    """
    x = np.asarray(x)

    x = np.nan_to_num(
        x,
        nan=0.0,
        posinf=float(clip_value),
        neginf=-float(clip_value),
    )

    if clip_value is not None and clip_value > 0:
        x = np.clip(x, -float(clip_value), float(clip_value))

    return x.astype(out_dtype, copy=False)


def infer_channel_slices(dataset: Basin2VecMixedMonthlyDataset) -> dict[str, Any]:
    """
    Infer channel slices from one dataset sample.

    Monthly sample:
        [M, C, H, W]

    Nonmonthly sample:
        [C, H, W]
    """
    example = dataset[0]

    monthly_slices: dict[str, list[int]] = {}
    static_slices: dict[str, list[int]] = {}

    start = 0
    for v in dataset.monthly_variables:
        x = example[v]

        if x.dim() != 4:
            raise ValueError(
                f"Monthly variable {v} expected [M,C,H,W], got {tuple(x.shape)}"
            )

        c = int(x.shape[1])
        monthly_slices[v] = [start, start + c]
        start += c

    monthly_channels = start

    start = 0
    for v in dataset.nonmonthly_variables:
        x = example[v]

        if x.dim() != 3:
            raise ValueError(
                f"Non-monthly variable {v} expected [C,H,W], got {tuple(x.shape)}"
            )

        c = int(x.shape[0])
        static_slices[v] = [start, start + c]
        start += c

    static_channels = start

    if len(dataset.monthly_variables) == 0:
        raise ValueError("No monthly variables found in dataset.")

    sample_monthly = example[dataset.monthly_variables[0]]
    months = int(sample_monthly.shape[0])
    H = int(sample_monthly.shape[-2])
    W = int(sample_monthly.shape[-1])

    return {
        "monthly_slices": monthly_slices,
        "static_slices": static_slices,
        "monthly_channels": monthly_channels,
        "static_channels": static_channels,
        "months": months,
        "H": H,
        "W": W,
    }


def create_or_open_group(out_path: Path, overwrite: bool):
    if out_path.exists() and overwrite:
        print(f"[INFO] Removing existing output: {out_path}")
        shutil.rmtree(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    return zarr.open_group(str(out_path), mode="a")


def create_or_open_array(
    root,
    name: str,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype: str,
    compressor,
    overwrite_array: bool,
):
    if name in root and overwrite_array:
        del root[name]

    if name in root:
        arr = root[name]

        if tuple(arr.shape) != tuple(shape):
            raise ValueError(
                f"Existing array '{name}' has shape {arr.shape}, expected {shape}. "
                "Use --overwrite to rebuild."
            )

        return arr

    arr = root.create_dataset(
        name,
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        compressor=compressor,
        fill_value=0,
        overwrite=False,
    )

    return arr


def estimate_gb(shape: tuple[int, ...], dtype: str) -> float:
    return float(np.prod(shape) * np.dtype(dtype).itemsize / 1e9)


def validate_output_sample(root, dataset, years: list[int], n_checks: int = 5):
    """
    Light sanity check after writing.
    """
    monthly_arr = root["monthly_x"]
    static_arr = root["static_x"]
    done_arr = root["done"]

    rng = np.random.default_rng(42)
    n = len(dataset)

    print("[INFO] Running light output validation")

    for _ in range(min(n_checks, n)):
        idx = int(rng.integers(0, n))
        row = dataset.index.iloc[idx]

        b = int(row["basin_index"])
        y = int(row["year"])
        y_pos = years.index(y)

        if not bool(done_arr[b, y_pos]):
            raise RuntimeError(f"Validation failed: done[{b},{y_pos}] is False")

        mx = np.asarray(monthly_arr[b, y_pos])
        sx = np.asarray(static_arr[b, y_pos])

        if not np.isfinite(mx).all():
            raise RuntimeError(f"Validation failed: monthly_x has non-finite values at basin={b}, year={y}")

        if not np.isfinite(sx).all():
            raise RuntimeError(f"Validation failed: static_x has non-finite values at basin={b}, year={y}")

    print("[INFO] Validation passed")


# ------------------------------------------------------------
# Main builder
# ------------------------------------------------------------
def build_grouped_zarr(args):
    torch.set_num_threads(max(1, int(args.torch_threads)))

    out_path = Path(args.out_zarr)
    out_dtype = np.dtype(args.out_dtype)

    print("[INFO] Loading source mixed monthly dataset")

    dataset = Basin2VecMixedMonthlyDataset(
        index_parquet=args.index_parquet,
        index_meta_json=args.index_meta_json,
        norm_stats_json=args.norm_stats_json,
        monthly_zarr_root=args.monthly_zarr_root,
        annual_zarr_root=args.annual_zarr_root,
        monthly_variables=MONTHLY_VARIABLES,
        nonmonthly_variables=NONMONTHLY_VARIABLES,
        basin_metadata_path=None,
        basin_meta_cols=None,
        mask_zarr_path=args.mask_zarr_path,
        months_per_year=12,
        landcover_var_name="land_cover",
        variable_file_aliases=VARIABLE_FILE_ALIASES,
        strict_variables=True,
    )

    info = infer_channel_slices(dataset)

    monthly_slices = info["monthly_slices"]
    static_slices = info["static_slices"]

    C_monthly = int(info["monthly_channels"])
    C_static = int(info["static_channels"])
    M = int(info["months"])
    H = int(info["H"])
    W = int(info["W"])

    index = dataset.index.copy()
    years = sorted(index["year"].astype(int).unique().tolist())
    year_to_pos = {int(y): i for i, y in enumerate(years)}

    n_basins = int(dataset.mask_z.shape[0])
    n_years = len(years)

    monthly_shape = (n_basins, n_years, M, C_monthly, H, W)
    static_shape = (n_basins, n_years, C_static, H, W)
    done_shape = (n_basins, n_years)

    print("[INFO] Output grouped arrays")
    print(f"  n_basins          : {n_basins}")
    print(f"  n_years           : {n_years}")
    print(f"  years             : {years[0]} ... {years[-1]}")
    print(f"  monthly variables : {dataset.monthly_variables}")
    print(f"  nonmonthly vars   : {dataset.nonmonthly_variables}")
    print(f"  monthly channels  : {C_monthly}")
    print(f"  static channels   : {C_static}")
    print(f"  patch             : {H} x {W}")
    print(f"  out dtype         : {args.out_dtype}")
    print(f"  clip value        : {args.clip_value}")
    print(f"  compressor        : {args.compressor}")
    print(f"  compression level : {args.compression_level}")

    print(f"[INFO] Estimated uncompressed monthly_x size: {estimate_gb(monthly_shape, args.out_dtype):.1f} GB")
    print(f"[INFO] Estimated uncompressed static_x size : {estimate_gb(static_shape, args.out_dtype):.1f} GB")
    print("[INFO] Compression will reduce this depending on sparsity/repetition.")

    compressor = Blosc(
        cname=args.compressor,
        clevel=int(args.compression_level),
        shuffle=Blosc.BITSHUFFLE,
    )

    root = create_or_open_group(
        out_path=out_path,
        overwrite=bool(args.overwrite),
    )

    # Chunking optimized for basin-aware batches:
    # one basin, several years, full channel/month/spatial tile.
    monthly_chunks = (
        1,
        min(int(args.year_chunk), n_years),
        M,
        C_monthly,
        H,
        W,
    )

    static_chunks = (
        1,
        min(int(args.year_chunk), n_years),
        C_static,
        H,
        W,
    )

    monthly_arr = create_or_open_array(
        root=root,
        name="monthly_x",
        shape=monthly_shape,
        chunks=monthly_chunks,
        dtype=args.out_dtype,
        compressor=compressor,
        overwrite_array=bool(args.overwrite),
    )

    static_arr = create_or_open_array(
        root=root,
        name="static_x",
        shape=static_shape,
        chunks=static_chunks,
        dtype=args.out_dtype,
        compressor=compressor,
        overwrite_array=bool(args.overwrite),
    )

    done_arr = create_or_open_array(
        root=root,
        name="done",
        shape=done_shape,
        chunks=(min(1024, n_basins), min(40, n_years)),
        dtype="bool",
        compressor=compressor,
        overwrite_array=bool(args.overwrite),
    )

    root.attrs["format"] = "basin2vec_grouped_monthly_static_v1"
    root.attrs["description"] = (
        "Pre-stacked normalized monthly and non-monthly inputs for fast Basin2Vec training"
    )
    root.attrs["created_or_updated_at"] = _now()
    root.attrs["source_monthly_zarr_root"] = str(args.monthly_zarr_root)
    root.attrs["source_annual_zarr_root"] = str(args.annual_zarr_root)
    root.attrs["source_index_parquet"] = str(args.index_parquet)
    root.attrs["source_norm_stats_json"] = str(args.norm_stats_json)
    root.attrs["years"] = [int(y) for y in years]
    root.attrs["months"] = list(range(1, M + 1))
    root.attrs["monthly_variables"] = list(dataset.monthly_variables)
    root.attrs["nonmonthly_variables"] = list(dataset.nonmonthly_variables)
    root.attrs["monthly_channel_slices_json"] = _json_dumps(monthly_slices)
    root.attrs["static_channel_slices_json"] = _json_dumps(static_slices)
    root.attrs["monthly_channels"] = int(C_monthly)
    root.attrs["static_channels"] = int(C_static)
    root.attrs["patch_size"] = int(H)
    root.attrs["patch_width"] = int(W)
    root.attrs["out_dtype"] = str(np.dtype(args.out_dtype))
    root.attrs["already_normalized"] = True
    root.attrs["land_cover_one_hot"] = True
    root.attrs["clip_value"] = float(args.clip_value)
    root.attrs["monthly_chunks"] = list(monthly_chunks)
    root.attrs["static_chunks"] = list(static_chunks)

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": int(args.batch_size),
        "shuffle": False,
        "num_workers": int(args.num_workers),
        "pin_memory": False,
        "drop_last": False,
    }

    if int(args.num_workers) > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)

    loader = DataLoader(**loader_kwargs)

    total_written = 0
    total_skipped = 0
    total_nonfinite_before_clip = 0
    total_clipped_monthly = 0
    total_clipped_static = 0

    print("[INFO] Building grouped Zarr")

    for batch in tqdm(loader, total=len(loader)):
        basin_idx = batch["basin_index"].cpu().numpy().astype(np.int64)
        batch_years = batch["year"].cpu().numpy().astype(np.int64)
        year_idx = np.array([year_to_pos[int(y)] for y in batch_years], dtype=np.int64)

        if args.resume:
            keep = np.array(
                [
                    not bool(done_arr[int(b), int(y)])
                    for b, y in zip(basin_idx, year_idx)
                ],
                dtype=bool,
            )

            if not keep.any():
                total_skipped += len(basin_idx)
                continue
        else:
            keep = np.ones(len(basin_idx), dtype=bool)

        monthly_parts = []
        for v in dataset.monthly_variables:
            x = batch[v]  # [B, M, C, H, W]

            if x.dim() != 5:
                raise ValueError(f"{v} expected [B,M,C,H,W], got {tuple(x.shape)}")

            monthly_parts.append(x)

        static_parts = []
        for v in dataset.nonmonthly_variables:
            x = batch[v]  # [B, C, H, W]

            if x.dim() != 4:
                raise ValueError(f"{v} expected [B,C,H,W], got {tuple(x.shape)}")

            static_parts.append(x)

        monthly_stack_f32 = torch.cat(monthly_parts, dim=2).numpy()
        static_stack_f32 = torch.cat(static_parts, dim=1).numpy()

        # Track non-finite before sanitizing.
        total_nonfinite_before_clip += int((~np.isfinite(monthly_stack_f32)).sum())
        total_nonfinite_before_clip += int((~np.isfinite(static_stack_f32)).sum())

        # Track clipping counts before clipping.
        if args.clip_value is not None and args.clip_value > 0:
            cv = float(args.clip_value)
            total_clipped_monthly += int((np.abs(monthly_stack_f32) > cv).sum())
            total_clipped_static += int((np.abs(static_stack_f32) > cv).sum())

        monthly_stack = safe_clip_and_cast(
            monthly_stack_f32,
            out_dtype=out_dtype,
            clip_value=float(args.clip_value),
        )

        static_stack = safe_clip_and_cast(
            static_stack_f32,
            out_dtype=out_dtype,
            clip_value=float(args.clip_value),
        )

        # Per-sample write is slower than a single contiguous write, but robust for
        # arbitrary basin/year ordering. This is a one-time offline build.
        for k in range(len(basin_idx)):
            if not keep[k]:
                total_skipped += 1
                continue

            b = int(basin_idx[k])
            y = int(year_idx[k])

            monthly_arr[b, y, :, :, :, :] = monthly_stack[k]
            static_arr[b, y, :, :, :] = static_stack[k]
            done_arr[b, y] = True

            total_written += 1

    # Save the exact index used during the build.
    index_out = out_path.parent / "grouped_index.parquet"
    index.to_parquet(index_out, index=False)

    # Validation after build.
    if args.validate:
        validate_output_sample(root, dataset, years, n_checks=int(args.validation_checks))

    done_count = int(done_arr[:].sum())
    expected_count = int(len(index))

    print("[DONE]")
    print(f"  output zarr                 : {out_path}")
    print(f"  grouped index               : {index_out}")
    print(f"  written this run            : {total_written}")
    print(f"  skipped this run            : {total_skipped}")
    print(f"  done count                  : {done_count} / {expected_count} training samples")
    print(f"  done count full grid         : {done_count} / {n_basins * n_years}")
    print(f"  non-finite values sanitized : {total_nonfinite_before_clip}")
    print(f"  monthly values clipped      : {total_clipped_monthly}")
    print(f"  static values clipped       : {total_clipped_static}")

    if done_count < expected_count:
        print("[WARN] Not all training-index samples are marked done.")
        print("       Re-run with --resume to continue.")


# ------------------------------------------------------------
# Args
# ------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--index-parquet", default=INDEX_PARQUET)
    p.add_argument("--index-meta-json", default=INDEX_META_JSON)
    p.add_argument("--norm-stats-json", default=NORM_STATS_JSON)

    p.add_argument("--monthly-zarr-root", default=MONTHLY_ZARR_ROOT)
    p.add_argument("--annual-zarr-root", default=ANNUAL_ZARR_ROOT)
    p.add_argument("--mask-zarr-path", default=MASK_ZARR_PATH)
    p.add_argument("--out-zarr", default=OUT_ZARR)

    p.add_argument("--out-dtype", default="float16", choices=["float16", "float32"])

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--torch-threads", type=int, default=1)

    # lz4 is faster for training reads. zstd compresses more but is slower.
    p.add_argument(
        "--compressor",
        default="lz4",
        choices=["lz4", "zstd", "blosclz", "zlib"],
    )
    p.add_argument("--compression-level", type=int, default=1)

    p.add_argument("--year-chunk", type=int, default=4)

    # Critical fix for float16 overflow.
    p.add_argument("--clip-value", type=float, default=10.0)

    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--resume", action="store_true")

    p.add_argument("--validate", action="store_true", default=True)
    p.add_argument("--no-validate", dest="validate", action="store_false")
    p.add_argument("--validation-checks", type=int, default=8)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.overwrite and args.resume:
        raise ValueError("Use either --overwrite or --resume, not both.")

    build_grouped_zarr(args)