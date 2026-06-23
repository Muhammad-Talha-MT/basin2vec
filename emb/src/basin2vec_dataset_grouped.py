# emb/src/basin2vec_dataset_grouped.py

from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import zarr


DEFAULT_BASIN_META_COLS = [
    "log_area_km2_z",
    "log_bbox_width_km_z",
    "log_bbox_height_km_z",
    "log_patch_pixel_width_km_z",
    "log_patch_pixel_height_km_z",
    "log_patch_pixel_area_km2_z",
    "bbox_fill_fraction_z",
    "compactness_z",
    "solidity_z",
    "log_shape_index_z",
    "log_overlap_degree_z",
]


class Basin2VecGroupedZarrDataset(Dataset):
    """
    Fast grouped Basin2Vec dataset.

    Reads pre-stacked arrays:

        monthly_x : [N_basin, N_year, 12, C_monthly, H, W]
        static_x  : [N_basin, N_year, C_static, H, W]

    Returns:

        {
            "site_id"        : str,
            "site_id_int"    : int,
            "basin_index"    : int,
            "year"           : int,
            "mask"           : Tensor [1, H, W],
            "monthly_x"      : Tensor [12, C_monthly, H, W],
            "static_x"       : Tensor [C_static, H, W],
            "basin_meta"     : Tensor [M],
            "log_area_km2_z" : scalar Tensor,
            "overlap_degree" : scalar Tensor,
        }

    This dataset avoids reading many separate variable Zarr files per sample.
    """

    def __init__(
        self,
        index_parquet: str,
        grouped_zarr_path: str,
        basin_metadata_path: str | None = None,
        basin_meta_cols: list[str] | None = None,
        mask_zarr_path: str = "/data/basin2vec/cache/static_masks.zarr",
        mask_dataset: str = "mask",
        require_done: bool = True,
    ):
        self.index = pd.read_parquet(index_parquet).reset_index(drop=True)
        self.index["site_id"] = self.index["site_id"].astype(str).str.strip()
        self.index_years = sorted(self.index["year"].astype(int).unique().tolist())

        self.grouped_zarr_path = Path(grouped_zarr_path)
        if not self.grouped_zarr_path.exists():
            raise FileNotFoundError(f"Grouped Zarr not found: {self.grouped_zarr_path}")

        self.root = zarr.open_group(str(self.grouped_zarr_path), mode="r")

        self.monthly_z = self.root["monthly_x"]
        self.static_z = self.root["static_x"]
        self.done_z = self.root["done"] if "done" in self.root else None

        self.years = [int(y) for y in self.root.attrs.get("years", self.index_years)]
        self.year_to_pos = {int(y): i for i, y in enumerate(self.years)}

        self.months = [int(m) for m in self.root.attrs.get("months", list(range(1, 13)))]

        self.monthly_variables = list(self.root.attrs.get("monthly_variables", []))
        self.nonmonthly_variables = list(self.root.attrs.get("nonmonthly_variables", []))

        monthly_slices_json = self.root.attrs.get("monthly_channel_slices_json", "{}")
        static_slices_json = self.root.attrs.get("static_channel_slices_json", "{}")

        self.monthly_channel_slices = json.loads(monthly_slices_json)
        self.static_channel_slices = json.loads(static_slices_json)

        self.monthly_in_channels = int(self.monthly_z.shape[3])
        self.static_in_channels = int(self.static_z.shape[2])
        self.months_per_year = int(self.monthly_z.shape[2])
        self.patch_size = int(self.monthly_z.shape[-2])
        self.patch_width = int(self.monthly_z.shape[-1])

        self.require_done = bool(require_done)

        unique_site_ids = sorted(self.index["site_id"].astype(str).unique().tolist())
        self.site_to_int = {sid: i for i, sid in enumerate(unique_site_ids)}
        self.int_to_site = {i: sid for sid, i in self.site_to_int.items()}

        self.basin_metadata_path = basin_metadata_path
        self.basin_meta_cols = basin_meta_cols or DEFAULT_BASIN_META_COLS
        self.has_basin_metadata = basin_metadata_path is not None

        self.meta_by_site: dict[str, np.ndarray] = {}
        self.area_z_by_site: dict[str, float] = {}
        self.overlap_degree_by_site: dict[str, float] = {}

        if self.has_basin_metadata:
            self._load_basin_metadata(basin_metadata_path)

        self.mask_zarr_path = Path(mask_zarr_path)
        self.mask_z = zarr.open(self.mask_zarr_path, mode="r")[mask_dataset]

        self.basin_to_indices = self.index.groupby("site_id").indices

    def _read_metadata_table(self, path: str) -> pd.DataFrame:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Basin metadata file not found: {path}")

        if p.suffix.lower() in [".parquet", ".pq"]:
            return pd.read_parquet(p)

        if p.suffix.lower() in [".csv", ".txt"]:
            return pd.read_csv(p, dtype={"site_id": str})

        raise ValueError(f"Unsupported basin metadata format: {p.suffix}")

    def _normalize_site_ids_for_join(self, meta_df: pd.DataFrame) -> pd.DataFrame:
        meta_df = meta_df.copy()

        if "site_id" not in meta_df.columns and "GAGE_ID" in meta_df.columns:
            meta_df = meta_df.rename(columns={"GAGE_ID": "site_id"})

        if "site_id" not in meta_df.columns:
            raise ValueError("Basin metadata must contain 'site_id' or 'GAGE_ID'.")

        meta_df["site_id"] = meta_df["site_id"].astype(str).str.strip()

        target_len = int(self.index["site_id"].astype(str).str.len().mode().iloc[0])
        meta_df["site_id"] = meta_df["site_id"].str.zfill(target_len)

        return meta_df

    def _load_basin_metadata(self, path: str):
        meta_df = self._read_metadata_table(path)
        meta_df = self._normalize_site_ids_for_join(meta_df)

        missing_cols = [c for c in self.basin_meta_cols if c not in meta_df.columns]
        if missing_cols:
            raise ValueError(
                "Basin metadata is missing required columns: "
                f"{missing_cols}. Available columns: {list(meta_df.columns)}"
            )

        for c in self.basin_meta_cols:
            meta_df[c] = pd.to_numeric(meta_df[c], errors="coerce")

        meta_df[self.basin_meta_cols] = meta_df[self.basin_meta_cols].replace(
            [np.inf, -np.inf],
            np.nan,
        ).fillna(0.0)

        if "log_area_km2_z" in meta_df.columns:
            meta_df["log_area_km2_z"] = pd.to_numeric(
                meta_df["log_area_km2_z"],
                errors="coerce",
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        else:
            meta_df["log_area_km2_z"] = 0.0

        if "overlap_degree" in meta_df.columns:
            meta_df["overlap_degree"] = pd.to_numeric(
                meta_df["overlap_degree"],
                errors="coerce",
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        else:
            meta_df["overlap_degree"] = 0.0

        meta_df = meta_df.drop_duplicates(subset=["site_id"], keep="first")

        valid_sites = set(self.index["site_id"].astype(str).unique())
        missing_sites = valid_sites - set(meta_df["site_id"].astype(str).unique())

        if missing_sites:
            raise ValueError(
                f"Basin metadata is missing {len(missing_sites)} sites from training index. "
                f"Example missing site IDs: {sorted(list(missing_sites))[:10]}"
            )

        meta_indexed = meta_df.set_index("site_id", drop=False)

        self.meta_by_site = {
            str(site): row[self.basin_meta_cols].to_numpy(dtype=np.float32)
            for site, row in meta_indexed.iterrows()
        }

        self.area_z_by_site = {
            str(site): float(row["log_area_km2_z"])
            for site, row in meta_indexed.iterrows()
        }

        self.overlap_degree_by_site = {
            str(site): float(row["overlap_degree"])
            for site, row in meta_indexed.iterrows()
        }

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        row = self.index.iloc[idx]

        basin_idx = int(row["basin_index"])
        site_id = str(row["site_id"]).strip()
        site_id_int = int(self.site_to_int[site_id])
        year = int(row["year"])

        if year not in self.year_to_pos:
            raise KeyError(f"Year {year} not found in grouped Zarr years: {self.years}")

        year_idx = int(self.year_to_pos[year])

        if self.done_z is not None and self.require_done:
            if not bool(self.done_z[basin_idx, year_idx]):
                raise RuntimeError(
                    f"Grouped Zarr sample not built yet: basin_index={basin_idx}, year={year}"
                )

        monthly_x = np.asarray(
            self.monthly_z[basin_idx, year_idx],
            dtype=np.float32,
        )

        static_x = np.asarray(
            self.static_z[basin_idx, year_idx],
            dtype=np.float32,
        )

        mask = self.mask_z[basin_idx, 0].astype(np.float32, copy=False)

        sample: dict[str, Any] = {
            "site_id": site_id,
            "site_id_int": site_id_int,
            "basin_index": basin_idx,
            "year": year,
            "monthly_x": torch.from_numpy(monthly_x).float(),
            "static_x": torch.from_numpy(static_x).float(),
            "mask": torch.from_numpy(mask).unsqueeze(0).float(),
        }

        if self.has_basin_metadata:
            sample["basin_meta"] = torch.from_numpy(self.meta_by_site[site_id]).float()
            sample["log_area_km2_z"] = torch.tensor(
                self.area_z_by_site[site_id],
                dtype=torch.float32,
            )
            sample["overlap_degree"] = torch.tensor(
                self.overlap_degree_by_site[site_id],
                dtype=torch.float32,
            )

        return sample