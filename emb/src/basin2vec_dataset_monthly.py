# emb/src/basin2vec_dataset_monthly.py

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


DEFAULT_LANDCOVER_CLASSES = sorted([
    11, 12,          # Water / Snow-Ice
    21, 22, 23, 24,  # Developed
    31,              # Barren
    41, 42, 43,      # Forest
    52,              # Shrub/Scrub
    71, 72, 73, 74,  # Herbaceous
    81, 82,          # Agriculture
    90, 95,          # Wetlands
])


class Basin2VecMixedMonthlyDataset(Dataset):
    """
    Basin2Vec dataset that reads monthly meteorological variables from one
    Zarr root and annual/static/non-meteorological variables from another root.

    Monthly variables are returned as:
        [12, C, H, W]

    Non-monthly variables are returned as:
        [C, H, W]

    This lets the encoder treat true monthly meteorology as seasonal sequences,
    while keeping DEM/soil/permeability/population/land_cover/etc. as static or
    annual spatial variables.

    Expected roots for your current setup
    ------------------------------------
    monthly_zarr_root:
        $BASIN2VEC_DATA_ROOT/cache/patches_step4_monthly
        e.g., prcp.zarr, tmin.zarr, tmax.zarr, vp.zarr, we.zarr
        array shape: [N, Y, 12, C, H, W]

    annual_zarr_root:
        $BASIN2VEC_DATA_ROOT/cache/patches_step4
        e.g., bdod.zarr, clay.zarr, dem.zarr, land_cover.zarr, pet.zarr, wind.zarr
        common array shape: [N, Y, C, H, W] or [N, C, H, W]

    Returns one sample as:
        {
            "site_id"          : str,
            "site_id_int"      : int,
            "basin_index"      : int,
            "year"             : int,
            "mask"             : Tensor [1, H, W],
            "basin_meta"       : Tensor [M], optional,
            "log_area_km2_z"   : scalar Tensor, optional,
            "overlap_degree"   : scalar Tensor, optional,
            monthly_var         : Tensor [12, C, H, W],
            nonmonthly_var      : Tensor [C, H, W],
            "land_cover"       : Tensor [K, H, W] if enabled,
        }
    """

    def __init__(
        self,
        index_parquet: str,
        index_meta_json: str | None,
        norm_stats_json: str | None,
        monthly_zarr_root: str,
        annual_zarr_root: str,
        monthly_variables: list[str],
        nonmonthly_variables: list[str],
        basin_metadata_path: str | None = None,
        basin_meta_cols: list[str] | None = None,
        mask_zarr_path: str | None = None,
        mask_dataset: str = "mask",
        months_per_year: int = 12,
        landcover_var_name: str = "land_cover",
        landcover_classes: list[int] | None = None,
        log1p_vars: list[str] | None = None,
        variable_file_aliases: dict[str, str] | None = None,
        strict_variables: bool = True,
    ):
        self.index = pd.read_parquet(index_parquet).reset_index(drop=True)
        self.index["site_id"] = self.index["site_id"].astype(str).str.strip()
        self.index_years = sorted(self.index["year"].astype(int).unique().tolist())

        self.meta: dict[str, Any] = {}
        if index_meta_json is not None and Path(index_meta_json).exists():
            with open(index_meta_json, "r") as f:
                self.meta = json.load(f)

        self.norm_stats: dict[str, Any] = {}
        if norm_stats_json is not None and Path(norm_stats_json).exists():
            with open(norm_stats_json, "r") as f:
                self.norm_stats = json.load(f)

        self.monthly_zarr_root = Path(monthly_zarr_root)
        self.annual_zarr_root = Path(annual_zarr_root)
        self.months_per_year = int(months_per_year)
        self.strict_variables = bool(strict_variables)
        self.landcover_var_name = str(landcover_var_name)
        self.landcover_classes = sorted([int(c) for c in (landcover_classes or DEFAULT_LANDCOVER_CLASSES)])
        self.landcover_class_to_index = {cls: i for i, cls in enumerate(self.landcover_classes)}

        self.variable_file_aliases = dict(variable_file_aliases or {})
        # Common typo-safe alias. Output key can remain "slit" if the config uses it,
        # but the loader will read silt.zarr when slit.zarr does not exist.
        self.variable_file_aliases.setdefault("slit", "silt")

        if log1p_vars is None:
            self.log1p_vars = {"prcp", "precip", "precipitation", "swe", "we"}
        else:
            self.log1p_vars = {str(v) for v in log1p_vars}

        self.monthly_variables = [str(v) for v in monthly_variables]
        self.nonmonthly_variables = [str(v) for v in nonmonthly_variables]

        dup = sorted(set(self.monthly_variables).intersection(self.nonmonthly_variables))
        if dup:
            raise ValueError(f"Variables cannot be both monthly and non-monthly: {dup}")

        # Keep order deterministic and explicit: monthly first, then non-monthly.
        self.variables = self.monthly_variables + self.nonmonthly_variables
        self.variable_kind = {v: "monthly" for v in self.monthly_variables}
        self.variable_kind.update({v: "nonmonthly" for v in self.nonmonthly_variables})

        # Deterministic site-id mapping used by current training code.
        unique_site_ids = sorted(self.index["site_id"].astype(str).unique().tolist())
        self.site_to_int = {sid: i for i, sid in enumerate(unique_site_ids)}
        self.int_to_site = {i: sid for sid, i in self.site_to_int.items()}

        # Optional metadata.
        self.basin_metadata_path = basin_metadata_path
        self.basin_meta_cols = basin_meta_cols or DEFAULT_BASIN_META_COLS
        self.has_basin_metadata = basin_metadata_path is not None
        self.meta_by_site: dict[str, np.ndarray] = {}
        self.area_z_by_site: dict[str, float] = {}
        self.overlap_degree_by_site: dict[str, float] = {}

        if self.has_basin_metadata:
            self._load_basin_metadata(basin_metadata_path)

        # Open both groups of variables.
        self.zarr: dict[str, dict[str, Any]] = {}
        for v in self.variables:
            kind = self.variable_kind[v]
            root_dir = self.monthly_zarr_root if kind == "monthly" else self.annual_zarr_root
            zarr_path, file_var = self._resolve_variable_path(root_dir, v)

            if not zarr_path.exists():
                msg = f"Variable '{v}' not found at {root_dir / (v + '.zarr')}"
                alias = self.variable_file_aliases.get(v)
                if alias:
                    msg += f" or alias {root_dir / (alias + '.zarr')}"
                if strict_variables:
                    raise FileNotFoundError(msg)
                print(f"[WARN] {msg}; skipping variable.")
                continue

            root, arr = self._open_zarr_array(zarr_path, file_var)
            years = self._read_years(root, arr, kind)
            months = self._read_months(root)

            self.zarr[v] = {
                "root": root,
                "array": arr,
                "years": years,
                "months": months,
                "kind": kind,
                "file_var": file_var,
                "path": str(zarr_path),
            }

        # Remove skipped variables if strict_variables=False.
        self.variables = [v for v in self.variables if v in self.zarr]
        self.monthly_variables = [v for v in self.monthly_variables if v in self.zarr]
        self.nonmonthly_variables = [v for v in self.nonmonthly_variables if v in self.zarr]

        if len(self.variables) == 0:
            raise ValueError("No usable variables were opened.")

        # Patch size. Prefer first monthly variable; otherwise first variable.
        sample_var = self.monthly_variables[0] if self.monthly_variables else self.variables[0]
        sample_shape = self.zarr[sample_var]["array"].shape
        if len(sample_shape) == 6:
            _, _, M, _, H, W = sample_shape
            self.months_per_year = int(M)
        elif len(sample_shape) == 5:
            _, _, _, H, W = sample_shape
        elif len(sample_shape) == 4:
            _, _, H, W = sample_shape
        else:
            raise ValueError(
                f"Unsupported array shape for {sample_var}: {sample_shape}. "
                "Expected [N,Y,12,C,H,W], [N,Y,C,H,W], or [N,C,H,W]."
            )
        self.patch_size = int(H)
        self.patch_width = int(W)

        if mask_zarr_path is None:
            mask_zarr_path = str(self.annual_zarr_root.parent / "static_masks.zarr")
        self.mask_zarr_path = Path(mask_zarr_path)
        self.mask_z = zarr.open(self.mask_zarr_path, mode="r")[mask_dataset]

        # Used by the existing basin-aware sampler.
        self.basin_to_indices = self.index.groupby("site_id").indices

    # ------------------------------------------------------------------
    # Zarr helpers
    # ------------------------------------------------------------------
    def _resolve_variable_path(self, root_dir: Path, var: str) -> tuple[Path, str]:
        direct = root_dir / f"{var}.zarr"
        if direct.exists():
            return direct, var

        alias = self.variable_file_aliases.get(var)
        if alias:
            alias_path = root_dir / f"{alias}.zarr"
            if alias_path.exists():
                return alias_path, alias

        return direct, var

    def _open_zarr_array(self, zarr_path: Path, file_var: str):
        root = zarr.open(zarr_path, mode="r")

        # Array stored directly at <var>.zarr.
        if hasattr(root, "shape"):
            return root, root

        # Group containing an array named after the variable or file alias.
        try:
            if file_var in root:
                return root, root[file_var]
        except TypeError:
            pass

        # Group containing exactly one array.
        if hasattr(root, "array_keys"):
            keys = list(root.array_keys())
        else:
            keys = [k for k in root.keys()]

        if len(keys) == 1:
            return root, root[keys[0]]

        raise KeyError(
            f"Could not identify array inside {zarr_path}. "
            f"Expected key '{file_var}' or exactly one array key. Found keys={keys}"
        )

    def _read_years(self, root, arr, kind: str) -> list[int] | list[str] | None:
        years = root.attrs.get("years", None)
        if years is None:
            years = self.meta.get("years", None)

        if years == ["static"] or years == "static":
            return ["static"]

        # If the array is [N,C,H,W], it is static regardless of attrs.
        if len(arr.shape) == 4:
            return ["static"]

        if years is None:
            # For dynamic arrays with no attrs, assume the index years are aligned
            # with the Y axis. This avoids silently using year index 0 for all years.
            return list(self.index_years)

        return [int(y) for y in years]

    def _read_months(self, root) -> list[int]:
        months = root.attrs.get("months", None)
        if months is None:
            return list(range(1, self.months_per_year + 1))
        return [int(m) for m in months]

    # ------------------------------------------------------------------
    # Metadata loading
    # ------------------------------------------------------------------
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
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)

        if "log_area_km2_z" in meta_df.columns:
            meta_df["log_area_km2_z"] = pd.to_numeric(
                meta_df["log_area_km2_z"], errors="coerce"
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        else:
            meta_df["log_area_km2_z"] = 0.0

        if "overlap_degree" in meta_df.columns:
            meta_df["overlap_degree"] = pd.to_numeric(
                meta_df["overlap_degree"], errors="coerce"
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

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.index)

    def _get_year_index(self, years: list[int] | list[str] | None, year: int) -> int:
        if years is None or years == ["static"]:
            return 0
        if year in years:
            return list(years).index(year)
        years_arr = np.array(years, dtype=np.int32)
        return int(np.abs(years_arr - int(year)).argmin())

    def _stats_for_var(self, var: str) -> dict[str, Any] | None:
        if var in self.norm_stats:
            return self.norm_stats[var]
        alias = self.variable_file_aliases.get(var)
        if alias and alias in self.norm_stats:
            return self.norm_stats[alias]
        entry = self.zarr.get(var, {})
        file_var = entry.get("file_var")
        if file_var and file_var in self.norm_stats:
            return self.norm_stats[file_var]
        return None

    def _stats_vector(self, stats: dict[str, Any], names: list[str]) -> np.ndarray | None:
        for name in names:
            if name in stats:
                return np.asarray(stats[name], dtype=np.float32)
        return None

    def _normalize_continuous(self, x: np.ndarray, var: str, monthly: bool) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if var in self.log1p_vars:
            x = np.log1p(np.clip(x, a_min=0.0, a_max=None)).astype(np.float32, copy=False)

        stats = self._stats_for_var(var)
        if stats is None:
            return x

        if monthly:
            monthly_mean = self._stats_vector(
                stats,
                ["monthly_mean", "mean_by_month", "month_mean", "means_by_month"],
            )
            monthly_std = self._stats_vector(
                stats,
                ["monthly_std", "std_by_month", "month_std", "stds_by_month"],
            )

            if monthly_mean is not None and monthly_std is not None:
                if len(monthly_mean) != x.shape[0] or len(monthly_std) != x.shape[0]:
                    raise ValueError(
                        f"Monthly stats for {var} have lengths "
                        f"mean={len(monthly_mean)}, std={len(monthly_std)}, "
                        f"but data has {x.shape[0]} months."
                    )
                mean = monthly_mean.reshape(-1, 1, 1, 1)
                std = monthly_std.reshape(-1, 1, 1, 1)
                std = np.where(std > 0, std, 1.0).astype(np.float32)
                return ((x - mean) / std).astype(np.float32, copy=False)

        # Scalar fallback for annual/static or monthly variables without monthly stats.
        mean = float(stats.get("mean", 0.0))
        std = float(stats.get("std", 0.0))
        if std > 0:
            x = (x - mean) / std
        return x.astype(np.float32, copy=False)

    def _one_hot_landcover(self, x: np.ndarray) -> np.ndarray:
        # Accept [H,W] or [C,H,W]; use the first channel if C exists.
        x = np.asarray(x)
        if x.ndim == 3:
            x = x[0]
        if x.ndim != 2:
            raise ValueError(f"land_cover expected [H,W] or [C,H,W], got {x.shape}")

        x = x.astype(np.int32, copy=False)
        H, W = x.shape
        K = len(self.landcover_classes)
        out = np.zeros((K, H, W), dtype=np.float32)
        for class_value, class_idx in self.landcover_class_to_index.items():
            out[class_idx] = (x == class_value).astype(np.float32)
        return out

    def _read_monthly_variable(self, var: str, basin_idx: int, year: int) -> np.ndarray:
        entry = self.zarr[var]
        arr = entry["array"]
        years = entry["years"]
        rank = len(arr.shape)
        year_idx = self._get_year_index(years, year)

        if rank == 6:
            x = arr[basin_idx, year_idx]  # [M, C, H, W]
        elif rank == 5:
            # Annual fallback for a variable accidentally listed as monthly.
            x = arr[basin_idx, year_idx]  # [C, H, W]
            x = np.repeat(np.asarray(x)[None, ...], self.months_per_year, axis=0)
        else:
            raise ValueError(
                f"Monthly variable {var} has unsupported shape {arr.shape}. "
                "Expected [N,Y,12,C,H,W] or [N,Y,C,H,W]."
            )

        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 3:
            x = x[:, None, :, :]  # [M,H,W] -> [M,1,H,W]
        if x.ndim != 4:
            raise ValueError(f"Monthly variable {var} read with unsupported shape: {x.shape}")
        return self._normalize_continuous(x, var, monthly=True)

    def _read_nonmonthly_variable(self, var: str, basin_idx: int, year: int) -> np.ndarray:
        entry = self.zarr[var]
        arr = entry["array"]
        years = entry["years"]
        rank = len(arr.shape)

        if years == ["static"] or rank == 4:
            if rank == 4:
                x = arr[basin_idx]  # [C,H,W]
            elif rank == 5:
                x = arr[basin_idx, 0]  # [C,H,W]
            elif rank == 6:
                x = arr[basin_idx, 0, 0]  # [C,H,W]
            else:
                raise ValueError(f"Static variable {var} has unsupported shape {arr.shape}")
        else:
            year_idx = self._get_year_index(years, year)
            if rank == 5:
                x = arr[basin_idx, year_idx]  # [C,H,W]
            elif rank == 6:
                # If an annual-root variable is unexpectedly monthly, average it.
                x = np.asarray(arr[basin_idx, year_idx]).mean(axis=0)  # [C,H,W]
            else:
                raise ValueError(f"Dynamic nonmonthly variable {var} has unsupported shape {arr.shape}")

        x = np.asarray(x)
        if var == self.landcover_var_name:
            return self._one_hot_landcover(x)

        x = x.astype(np.float32, copy=False)
        if x.ndim == 2:
            x = x[None, :, :]
        if x.ndim != 3:
            raise ValueError(f"Nonmonthly variable {var} read with unsupported shape: {x.shape}")
        return self._normalize_continuous(x, var, monthly=False)

    def __getitem__(self, idx: int):
        row = self.index.iloc[idx]

        basin_idx = int(row["basin_index"])
        site_id = str(row["site_id"]).strip()
        site_id_int = int(self.site_to_int[site_id])
        year = int(row["year"])

        sample: dict[str, Any] = {
            "site_id": site_id,
            "site_id_int": site_id_int,
            "basin_index": basin_idx,
            "year": year,
        }

        if self.has_basin_metadata:
            sample["basin_meta"] = torch.from_numpy(self.meta_by_site[site_id]).float()
            sample["log_area_km2_z"] = torch.tensor(
                self.area_z_by_site[site_id], dtype=torch.float32
            )
            sample["overlap_degree"] = torch.tensor(
                self.overlap_degree_by_site[site_id], dtype=torch.float32
            )

        mask = self.mask_z[basin_idx, 0].astype(np.float32, copy=False)
        sample["mask"] = torch.from_numpy(mask).unsqueeze(0).float()

        for v in self.variables:
            if self.variable_kind[v] == "monthly":
                x = self._read_monthly_variable(v, basin_idx, year)      # [M,C,H,W]
            else:
                x = self._read_nonmonthly_variable(v, basin_idx, year)  # [C,H,W]
            sample[v] = torch.from_numpy(x).float()

        return sample


# Backward-compatible class name for older scripts.
Basin2VecMonthlyDataset = Basin2VecMixedMonthlyDataset
