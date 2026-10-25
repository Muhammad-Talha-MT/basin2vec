#!/usr/bin/env python3
"""
Basin2Vec Round-2 ablation embedding extraction.

Purpose
-------
For every completed ablation checkpoint and seed, this script:

1. Reconstructs the exact ablation encoder from the checkpoint metadata.
2. Runs the encoder in eval/inference mode on UN-AUGMENTED basin-year inputs.
3. Saves one L2-normalized embedding for every basin-year sample.
4. Computes the paper's Eq. 14 basin descriptor:

       z_bar_b = Normalize( mean_y z_{b,y} )

5. Validates counts, uniqueness, NaNs, and embedding norms.
6. Supports single-GPU or multi-GPU extraction with torchrun.
7. Automatically skips already-completed checkpoint/seed extractions unless
   --overwrite is supplied.

Expected project files in the same directory
--------------------------------------------
    attention_encoder_monthly_ablation.py
    basin2vec_dataset_grouped.py

Typical 4-GPU command (reads emb/src/checkpoints/ablation_round2/,
writes emb/src/evaluation_outputs/ablation_round2/)
---------------------
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
    extract_basin2vec_ablation_embeddings.py \
    --base-config paper \
    --epoch 100 \
    --variants all \
    --seeds 42,123,456 \
    --batch-size 64 \
    --num-workers 8

Notes
-----
- No augmentation is applied during extraction.
- model.eval() disables dropout, including metadata dropout.
- Encoder outputs are already L2 normalized per basin-year.
- Multi-GPU extraction uses independent rank shards; the model is NOT wrapped
  in DDP because inference requires no gradient synchronization.
- Rank 0 merges per-rank Parquet shards and computes basin-level embeddings.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.amp import autocast
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from attention_encoder_monthly_ablation import GroupedBasinEncoderMonthlyV2
from basin2vec_dataset_grouped import Basin2VecGroupedZarrDataset


# Data locations are resolved from BASIN2VEC_DATA_ROOT (default: <repo>/data).
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DATA_ROOT = Path(os.environ.get("BASIN2VEC_DATA_ROOT", REPO_ROOT / "data"))
CACHE_DIR = DATA_ROOT / "cache"

# Used when neither a CLI override nor an existing checkpoint-recorded path is available.
DEFAULT_DATA_PATHS = {
    "index_parquet": REPO_ROOT / "emb" / "config" / "training_step5" / "sample_index.parquet",
    "grouped_zarr_path": CACHE_DIR / "patches_step4_grouped" / "grouped_monthly_static.zarr",
    "mask_zarr_path": CACHE_DIR / "static_masks.zarr",
    "basin_metadata": CACHE_DIR / "metadata" / "basin_metadata.parquet",
}


# -----------------------------------------------------------------------------
# Finalized Round-2 ablation order
# -----------------------------------------------------------------------------
DEFAULT_VARIANTS = [
    # "full",
    # "no_temporal_positive",
    # "no_overlap_positive",
    "no_area_regularization",
    "no_monthly_branch",
    "no_static_branch",
    "no_metadata_encoder",
    "metadata_concat",
    "no_meta_denominator",
    "meta_as_positive",
    "no_negative_margin",
    "instance_contrastive",
]

DEFAULT_SEEDS = [42, 123, 456]

# Architecture settings that were fixed in the Round-2 trainer.
DEFAULT_TEMPORAL_LAYERS = 1
DEFAULT_TEMPORAL_HEADS = 4
DEFAULT_DROPOUT = 0.05
DEFAULT_METADATA_DROPOUT = 0.15


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_csv_strings(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract basin-year and Eq.14 basin embeddings from all Basin2Vec ablation checkpoints."
    )

    parser.add_argument(
        "--checkpoint-root",
        type=str,
        default=str(SCRIPT_DIR / "checkpoints" / "ablation_round2"),
        help="Root containing Basin2Vec_<base>_<variant>/seed_<seed>/epoch_*.pt",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(SCRIPT_DIR / "evaluation_outputs" / "ablation_round2"),
        help="Root directory for extracted embeddings.",
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default="paper",
        choices=["paper", "uploaded"],
        help="Checkpoint family to extract.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="all",
        help=(
            "Comma-separated variant names, or 'all'. Default: all finalized variants."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(x) for x in DEFAULT_SEEDS),
        help="Comma-separated checkpoint seeds.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=100,
        help=(
            "Checkpoint epoch to use. Set to 0 to automatically use the highest available epoch."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Optional explicit single checkpoint. If supplied, --variants/--seeds/--epoch discovery is ignored."
        ),
    )

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA autocast during inference.",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="zstd",
        choices=["zstd", "snappy", "gzip", "brotli", "none"],
        help="Parquet compression codec.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract even when final outputs already exist.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip missing requested checkpoints instead of failing.",
    )
    parser.add_argument(
        "--keep-temp-shards",
        action="store_true",
        help="Keep per-rank temporary Parquet shards after successful merge.",
    )

    # Optional path overrides. By default paths are read from each checkpoint,
    # falling back to the BASIN2VEC_DATA_ROOT layout if those paths do not exist.
    parser.add_argument("--index-parquet", type=str, default=None)
    parser.add_argument("--grouped-zarr-path", type=str, default=None)
    parser.add_argument("--mask-zarr-path", type=str, default=None)
    parser.add_argument("--basin-metadata", type=str, default=None)

    # Architecture override only if a future checkpoint used a different fixed
    # Transformer head count. Current Round-2 checkpoints used 4 heads.
    parser.add_argument("--temporal-heads", type=int, default=DEFAULT_TEMPORAL_HEADS)

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Distributed setup
# -----------------------------------------------------------------------------
def setup_distributed():
    """Return (device, local_rank, global_rank, world_size, use_dist)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])

        if not torch.cuda.is_available():
            raise RuntimeError("torchrun was used but CUDA is not available.")

        visible = torch.cuda.device_count()
        if local_rank >= visible:
            raise RuntimeError(
                "LOCAL_RANK exceeds the number of visible CUDA devices: "
                f"LOCAL_RANK={local_rank}, visible={visible}, "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}"
            )

        # Select device before NCCL initialization.
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        dist.init_process_group(backend="nccl")
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
        return device, local_rank, global_rank, world_size, True

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        return torch.device("cuda:0"), 0, 0, 1, False

    return torch.device("cpu"), 0, 0, 1, False


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Checkpoint helpers
# -----------------------------------------------------------------------------
def safe_torch_load(path: Path) -> dict[str, Any]:
    """Load checkpoints across both pre/post PyTorch weights_only-default APIs."""
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")

    if not isinstance(obj, dict):
        raise TypeError(f"Checkpoint must contain a dict, got {type(obj)}: {path}")
    return obj


def clean_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip common wrappers while leaving normal keys unchanged."""
    cleaned = {}
    for key, value in state.items():
        new_key = key
        for prefix in ("module.", "_orig_mod."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def highest_epoch_checkpoint(seed_dir: Path) -> Optional[Path]:
    candidates: list[tuple[int, Path]] = []
    for p in seed_dir.glob("epoch_*.pt"):
        m = re.fullmatch(r"epoch_(\d+)\.pt", p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def discover_checkpoints(args: argparse.Namespace) -> list[Path]:
    if args.checkpoint is not None:
        p = Path(args.checkpoint)
        if not p.exists():
            raise FileNotFoundError(f"Explicit checkpoint not found: {p}")
        return [p]

    variants = DEFAULT_VARIANTS if args.variants.lower() == "all" else parse_csv_strings(args.variants)
    unknown = sorted(set(variants) - set(DEFAULT_VARIANTS))
    if unknown:
        raise ValueError(
            f"Unknown variant(s): {unknown}. Allowed: {DEFAULT_VARIANTS}"
        )

    seeds = parse_csv_ints(args.seeds)
    root = Path(args.checkpoint_root)
    found: list[Path] = []
    missing: list[str] = []

    for variant in variants:
        experiment = f"Basin2Vec_{args.base_config}_{variant}"
        for seed in seeds:
            seed_dir = root / experiment / f"seed_{seed}"
            if args.epoch > 0:
                ckpt = seed_dir / f"epoch_{args.epoch}.pt"
                if not ckpt.exists():
                    ckpt = None
            else:
                ckpt = highest_epoch_checkpoint(seed_dir)

            if ckpt is None:
                missing.append(f"{experiment}/seed_{seed}")
            else:
                found.append(ckpt)

    if missing and not args.allow_missing:
        raise FileNotFoundError(
            "Missing requested checkpoint(s):\n  - " + "\n  - ".join(missing)
        )

    if missing:
        print(
            "[WARN] Skipping missing checkpoint(s):\n  - " + "\n  - ".join(missing),
            file=sys.stderr,
        )

    if not found:
        raise FileNotFoundError("No checkpoints discovered.")

    return found


def infer_temporal_layers(state: dict[str, torch.Tensor]) -> int:
    """Infer number of TransformerEncoder layers from state_dict keys."""
    layer_ids = []
    pattern = re.compile(r"^monthly_branch\.temporal_encoder\.layers\.(\d+)\.")
    for key in state:
        m = pattern.match(key)
        if m:
            layer_ids.append(int(m.group(1)))
    return max(layer_ids) + 1 if layer_ids else DEFAULT_TEMPORAL_LAYERS


def fallback_variant_config(variant: str) -> dict[str, Any]:
    """Compatibility fallback for older ablation checkpoints."""
    cfg = {
        "name": variant,
        "use_monthly_branch": True,
        "use_static_branch": True,
        "metadata_fusion": "gated",
        "use_area_head": True,
    }

    if variant == "no_area_regularization":
        cfg["use_area_head"] = False
    elif variant == "no_monthly_branch":
        cfg["use_monthly_branch"] = False
    elif variant == "no_static_branch":
        cfg["use_static_branch"] = False
    elif variant == "no_metadata_encoder":
        cfg["metadata_fusion"] = "none"
    elif variant == "metadata_concat":
        cfg["metadata_fusion"] = "concat"

    return cfg


def checkpoint_identity(ckpt_path: Path, ckpt: dict[str, Any]) -> dict[str, Any]:
    variant = str(ckpt.get("ablation_variant", "unknown"))
    seed = int(ckpt.get("seed", -1))
    epoch = int(ckpt.get("epoch", -1))
    base_config = str(ckpt.get("base_config", "unknown"))

    # Prefer the directory naming when older checkpoints lack metadata.
    if variant == "unknown":
        experiment_name = ckpt_path.parent.parent.name
        prefix = f"Basin2Vec_{base_config}_" if base_config != "unknown" else ""
        if prefix and experiment_name.startswith(prefix):
            variant = experiment_name[len(prefix):]

    if seed < 0:
        m = re.fullmatch(r"seed_(\d+)", ckpt_path.parent.name)
        if m:
            seed = int(m.group(1))

    return {
        "variant": variant,
        "seed": seed,
        "epoch": epoch,
        "base_config": base_config,
        "experiment_name": ckpt_path.parent.parent.name,
    }


# -----------------------------------------------------------------------------
# Selective extraction dataset
# -----------------------------------------------------------------------------
class SelectiveBasin2VecGroupedZarrDataset(Basin2VecGroupedZarrDataset):
    """
    Same data/indexing semantics as Basin2VecGroupedZarrDataset, but avoids
    reading a disabled branch from Zarr during ablation extraction.
    """

    def __init__(self, *args, load_monthly: bool, load_static: bool, **kwargs):
        self.load_monthly = bool(load_monthly)
        self.load_static = bool(load_static)
        if not self.load_monthly and not self.load_static:
            raise ValueError("At least one branch must be loaded.")
        super().__init__(*args, **kwargs)

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
                    f"Grouped Zarr sample not built: basin_index={basin_idx}, year={year}"
                )

        mask = self.mask_z[basin_idx, 0].astype(np.float32, copy=False)

        sample: dict[str, Any] = {
            "dataset_index": int(idx),
            "site_id": site_id,
            "site_id_int": site_id_int,
            "basin_index": basin_idx,
            "year": year,
            "mask": torch.from_numpy(mask).unsqueeze(0).float(),
        }

        if self.load_monthly:
            monthly_x = np.asarray(
                self.monthly_z[basin_idx, year_idx], dtype=np.float32
            )
            sample["monthly_x"] = torch.from_numpy(monthly_x).float()

        if self.load_static:
            static_x = np.asarray(
                self.static_z[basin_idx, year_idx], dtype=np.float32
            )
            sample["static_x"] = torch.from_numpy(static_x).float()

        if self.has_basin_metadata:
            sample["basin_meta"] = torch.from_numpy(self.meta_by_site[site_id]).float()

        return sample


# -----------------------------------------------------------------------------
# Path/model reconstruction
# -----------------------------------------------------------------------------
def choose_path(
    cli_override: Optional[str],
    checkpoint_paths: dict[str, Any],
    key: str,
) -> str:
    value = cli_override if cli_override is not None else checkpoint_paths.get(key)
    if cli_override is None and (value is None or not Path(value).exists()):
        value = DEFAULT_DATA_PATHS.get(key, value)
    if value is None:
        raise ValueError(
            f"No path available for {key!r}. Supply the corresponding CLI override."
        )
    return str(value)


def build_model_from_checkpoint(
    ckpt: dict[str, Any],
    temporal_heads: int,
    device: torch.device,
) -> tuple[GroupedBasinEncoderMonthlyV2, dict[str, Any]]:
    state = clean_state_dict(ckpt["model_state_dict"])

    variant = str(ckpt.get("ablation_variant", "full"))
    cfg = dict(ckpt.get("variant_config") or fallback_variant_config(variant))

    use_monthly = bool(cfg.get("use_monthly_branch", True))
    use_static = bool(cfg.get("use_static_branch", True))
    metadata_fusion = str(cfg.get("metadata_fusion", "gated")).lower()
    use_area_head = bool(cfg.get("use_area_head", True))

    monthly_in_channels = int(ckpt["monthly_in_channels"])
    static_in_channels = int(ckpt["static_in_channels"])
    emb_dim = int(ckpt.get("emb_dim", 64))
    months_per_year = int(ckpt.get("months_per_year", 12))
    basin_meta_cols = list(ckpt.get("basin_meta_cols", []))

    encoder_meta_dim = len(basin_meta_cols) if metadata_fusion != "none" else 0
    temporal_layers = infer_temporal_layers(state) if use_monthly else DEFAULT_TEMPORAL_LAYERS

    model = GroupedBasinEncoderMonthlyV2(
        monthly_in_channels=monthly_in_channels,
        static_in_channels=static_in_channels,
        emb_dim=emb_dim,
        meta_dim=encoder_meta_dim,
        max_months=months_per_year,
        temporal_layers=temporal_layers,
        temporal_heads=temporal_heads,
        dropout=DEFAULT_DROPOUT,
        metadata_dropout=DEFAULT_METADATA_DROPOUT,
        predict_month_aux=False,
        use_monthly_branch=use_monthly,
        use_static_branch=use_static,
        metadata_fusion=metadata_fusion,
        use_area_head=use_area_head,
    )

    missing, unexpected = model.load_state_dict(state, strict=False)

    # Strict semantic validation. The only accepted missing/unexpected keys are
    # none; strict=False is used solely to provide a cleaner diagnostic.
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/model architecture mismatch.\n"
            f"Missing keys ({len(missing)}): {missing[:30]}\n"
            f"Unexpected keys ({len(unexpected)}): {unexpected[:30]}"
        )

    model.to(device)
    model.eval()

    model_info = {
        "variant_config": cfg,
        "use_monthly_branch": use_monthly,
        "use_static_branch": use_static,
        "metadata_fusion": metadata_fusion,
        "use_area_head": use_area_head,
        "monthly_in_channels": monthly_in_channels,
        "static_in_channels": static_in_channels,
        "emb_dim": emb_dim,
        "months_per_year": months_per_year,
        "basin_meta_cols": basin_meta_cols,
        "temporal_layers": temporal_layers,
        "temporal_heads": temporal_heads,
    }
    return model, model_info


# -----------------------------------------------------------------------------
# Output/validation helpers
# -----------------------------------------------------------------------------
def embedding_columns(dim: int) -> list[str]:
    width = max(3, len(str(dim - 1)))
    return [f"z_{i:0{width}d}" for i in range(dim)]


def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def validate_basin_year_df(df: pd.DataFrame, emb_cols: list[str], expected_n: int):
    if len(df) != expected_n:
        raise RuntimeError(
            f"Merged extraction has {len(df)} rows; expected dataset length {expected_n}."
        )

    if df["dataset_index"].duplicated().any():
        dup = df.loc[df["dataset_index"].duplicated(), "dataset_index"].head().tolist()
        raise RuntimeError(f"Duplicate dataset_index values after rank merge: {dup}")

    expected_idx = np.arange(expected_n, dtype=np.int64)
    actual_idx = df["dataset_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual_idx, expected_idx):
        raise RuntimeError("dataset_index coverage/order is incomplete after merge.")

    duplicated_site_year = df.duplicated(subset=["site_id", "year"], keep=False)
    if duplicated_site_year.any():
        examples = (
            df.loc[duplicated_site_year, ["site_id", "year"]]
            .head(10)
            .to_dict("records")
        )
        raise RuntimeError(
            "Expected one embedding per basin-year, but duplicates were found. "
            f"Examples: {examples}"
        )

    z = df[emb_cols].to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(z).all():
        raise RuntimeError("Non-finite values found in basin-year embeddings.")

    norms = np.linalg.norm(z, axis=1)
    if np.any(norms < 1e-6):
        raise RuntimeError("Near-zero basin-year embedding(s) found.")

    return {
        "basin_year_norm_mean": float(norms.mean()),
        "basin_year_norm_std": float(norms.std()),
        "basin_year_norm_min": float(norms.min()),
        "basin_year_norm_max": float(norms.max()),
    }


def aggregate_eq14(df: pd.DataFrame, emb_cols: list[str]) -> pd.DataFrame:
    """Mean basin-year embeddings over years, then L2 normalize (paper Eq. 14)."""
    records: list[dict[str, Any]] = []

    for site_id, group in df.groupby("site_id", sort=True):
        group = group.sort_values("year")
        z = group[emb_cols].to_numpy(dtype=np.float32)
        mean_z = z.mean(axis=0, dtype=np.float32)
        norm = float(np.linalg.norm(mean_z))
        if not np.isfinite(norm) or norm <= 1e-12:
            raise RuntimeError(f"Invalid mean embedding norm for basin {site_id}: {norm}")
        z_bar = (mean_z / norm).astype(np.float32)

        rec: dict[str, Any] = {
            "site_id": str(site_id),
            "site_id_int": int(group["site_id_int"].iloc[0]),
            "basin_index": int(group["basin_index"].iloc[0]),
            "n_years": int(len(group)),
            "first_year": int(group["year"].min()),
            "last_year": int(group["year"].max()),
            "pre_normalization_mean_norm": norm,
        }
        rec.update({c: float(v) for c, v in zip(emb_cols, z_bar)})
        records.append(rec)

    out = pd.DataFrame.from_records(records)
    z = out[emb_cols].to_numpy(dtype=np.float32, copy=False)
    norms = np.linalg.norm(z, axis=1)
    if not np.isfinite(z).all():
        raise RuntimeError("Non-finite values found in Eq.14 basin embeddings.")
    if not np.allclose(norms, 1.0, atol=2e-4, rtol=2e-4):
        raise RuntimeError(
            f"Eq.14 basin embeddings are not unit-normalized: min={norms.min()}, max={norms.max()}"
        )
    return out


def parquet_compression(value: str) -> Optional[str]:
    return None if value == "none" else value


# -----------------------------------------------------------------------------
# Rank-local extraction
# -----------------------------------------------------------------------------
def extract_rank_shard(
    model: GroupedBasinEncoderMonthlyV2,
    dataset: SelectiveBasin2VecGroupedZarrDataset,
    indices: list[int],
    output_path: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    use_amp: bool,
    compression: Optional[str],
    is_main: bool,
) -> int:
    subset = Subset(dataset, indices)

    loader_kwargs = {
        "dataset": subset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor

    loader = DataLoader(**loader_kwargs)

    emb_dim = int(model.emb_dim)
    emb_cols = embedding_columns(emb_dim)

    all_dataset_index: list[np.ndarray] = []
    all_site_id: list[str] = []
    all_site_id_int: list[np.ndarray] = []
    all_basin_index: list[np.ndarray] = []
    all_year: list[np.ndarray] = []
    all_z: list[np.ndarray] = []

    progress = tqdm(loader, disable=not is_main, desc="extract", dynamic_ncols=True)

    with torch.inference_mode():
        for batch in progress:
            mask = batch["mask"].to(device, non_blocking=True).float()
            if not bool((mask.sum(dim=(1, 2, 3)) > 0).all()):
                raise RuntimeError("Encountered an empty basin mask during extraction.")

            monthly_x = None
            if model.use_monthly_branch:
                monthly_x = batch["monthly_x"].to(device, non_blocking=True).float()

            static_x = None
            if model.use_static_branch:
                static_x = batch["static_x"].to(device, non_blocking=True).float()

            basin_meta = None
            if model.metadata_fusion != "none":
                basin_meta = batch["basin_meta"].to(device, non_blocking=True).float()

            with autocast(
                device_type="cuda",
                enabled=(use_amp and device.type == "cuda"),
            ):
                z, _, _, _, _ = model(monthly_x, static_x, mask, basin_meta)

            z_np = z.float().cpu().numpy().astype(np.float32, copy=False)

            all_dataset_index.append(batch["dataset_index"].numpy().astype(np.int64))
            all_site_id.extend([str(x) for x in batch["site_id"]])
            all_site_id_int.append(batch["site_id_int"].numpy().astype(np.int64))
            all_basin_index.append(batch["basin_index"].numpy().astype(np.int64))
            all_year.append(batch["year"].numpy().astype(np.int64))
            all_z.append(z_np)

    if all_z:
        dataset_index = np.concatenate(all_dataset_index)
        site_id_int = np.concatenate(all_site_id_int)
        basin_index = np.concatenate(all_basin_index)
        years = np.concatenate(all_year)
        z_mat = np.concatenate(all_z, axis=0)
    else:
        dataset_index = np.empty((0,), dtype=np.int64)
        site_id_int = np.empty((0,), dtype=np.int64)
        basin_index = np.empty((0,), dtype=np.int64)
        years = np.empty((0,), dtype=np.int64)
        z_mat = np.empty((0, emb_dim), dtype=np.float32)

    if len(all_site_id) != len(z_mat):
        raise RuntimeError("site_id count does not match extracted embedding count.")

    local = pd.DataFrame(
        {
            "dataset_index": dataset_index,
            "site_id": all_site_id,
            "site_id_int": site_id_int,
            "basin_index": basin_index,
            "year": years,
        }
    )
    for j, col in enumerate(emb_cols):
        local[col] = z_mat[:, j]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    local.to_parquet(output_path, index=False, compression=compression)
    return len(local)


# -----------------------------------------------------------------------------
# One checkpoint
# -----------------------------------------------------------------------------
def process_checkpoint(
    ckpt_path: Path,
    args: argparse.Namespace,
    device: torch.device,
    global_rank: int,
    world_size: int,
):
    is_main = global_rank == 0

    ckpt = safe_torch_load(ckpt_path)
    identity = checkpoint_identity(ckpt_path, ckpt)

    experiment_name = identity["experiment_name"]
    seed = identity["seed"]
    variant = identity["variant"]
    epoch = identity["epoch"]

    output_dir = Path(args.output_root) / experiment_name / f"seed_{seed}"
    basin_year_path = output_dir / "basin_year_embeddings.parquet"
    basin_path = output_dir / "basin_embeddings.parquet"
    manifest_path = output_dir / "extraction_manifest.json"
    temp_dir = output_dir / "_rank_shards"

    if is_main:
        print("\n" + "=" * 88)
        print("Basin2Vec embedding extraction")
        print("=" * 88)
        print(f"checkpoint       : {ckpt_path}")
        print(f"experiment       : {experiment_name}")
        print(f"variant          : {variant}")
        print(f"seed             : {seed}")
        print(f"checkpoint epoch : {epoch}")
        print(f"world size       : {world_size}")
        print(f"output           : {output_dir}")
        print("=" * 88)

    # Skip only when all final artifacts are present.
    already_complete = basin_year_path.exists() and basin_path.exists() and manifest_path.exists()

    if dist.is_available() and dist.is_initialized():
        flag = torch.tensor([1 if already_complete else 0], device=device, dtype=torch.int32)
        # Rank 0 is the authority because all ranks share the same filesystem.
        dist.broadcast(flag, src=0)
        already_complete = bool(flag.item())

    if already_complete and not args.overwrite:
        if is_main:
            print("[SKIP] Final embedding outputs already exist.")
        barrier()
        return

    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.overwrite and temp_dir.exists():
            shutil.rmtree(temp_dir)
    barrier()

    model, model_info = build_model_from_checkpoint(
        ckpt=ckpt,
        temporal_heads=args.temporal_heads,
        device=device,
    )

    ckpt_paths = dict(ckpt.get("paths", {}))
    index_parquet = choose_path(args.index_parquet, ckpt_paths, "index_parquet")
    grouped_zarr = choose_path(args.grouped_zarr_path, ckpt_paths, "grouped_zarr_path")
    mask_zarr = choose_path(args.mask_zarr_path, ckpt_paths, "mask_zarr_path")
    basin_metadata = choose_path(args.basin_metadata, ckpt_paths, "basin_metadata")

    basin_meta_cols = list(model_info["basin_meta_cols"])
    if not basin_meta_cols:
        raise ValueError(
            "Checkpoint does not contain basin_meta_cols. They are needed to recreate the dataset."
        )

    dataset = SelectiveBasin2VecGroupedZarrDataset(
        index_parquet=index_parquet,
        grouped_zarr_path=grouped_zarr,
        basin_metadata_path=basin_metadata,
        basin_meta_cols=basin_meta_cols,
        mask_zarr_path=mask_zarr,
        require_done=True,
        load_monthly=model_info["use_monthly_branch"],
        load_static=model_info["use_static_branch"],
    )

    # Sanity-check checkpoint/data dimensions before a long extraction.
    if int(dataset.monthly_in_channels) != int(model_info["monthly_in_channels"]):
        raise RuntimeError(
            f"Monthly channel mismatch: dataset={dataset.monthly_in_channels}, "
            f"checkpoint={model_info['monthly_in_channels']}"
        )
    if int(dataset.static_in_channels) != int(model_info["static_in_channels"]):
        raise RuntimeError(
            f"Static channel mismatch: dataset={dataset.static_in_channels}, "
            f"checkpoint={model_info['static_in_channels']}"
        )
    if int(dataset.months_per_year) != int(model_info["months_per_year"]):
        raise RuntimeError(
            f"Months/year mismatch: dataset={dataset.months_per_year}, "
            f"checkpoint={model_info['months_per_year']}"
        )

    # Non-padding deterministic shard: rank r processes r, r+W, r+2W, ...
    local_indices = list(range(global_rank, len(dataset), world_size))
    local_shard = temp_dir / f"rank_{global_rank:03d}.parquet"

    if is_main:
        print(f"[INFO] Dataset basin-year samples: {len(dataset):,}")
        print(f"[INFO] Encoder config: {json.dumps(model_info, indent=2)}")
        print(f"[INFO] Rank-0 samples: {len(local_indices):,}")
        print(f"[INFO] AMP: {not args.no_amp and device.type == 'cuda'}")

    extract_rank_shard(
        model=model,
        dataset=dataset,
        indices=local_indices,
        output_path=local_shard,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        use_amp=not args.no_amp,
        compression=parquet_compression(args.compression),
        is_main=is_main,
    )

    barrier()

    if is_main:
        shard_paths = [temp_dir / f"rank_{r:03d}.parquet" for r in range(world_size)]
        missing_shards = [str(p) for p in shard_paths if not p.exists()]
        if missing_shards:
            raise RuntimeError(f"Missing rank shard(s): {missing_shards}")

        print("[INFO] Merging rank shards...")
        pieces = [pd.read_parquet(p) for p in shard_paths]
        merged = pd.concat(pieces, axis=0, ignore_index=True)
        merged = merged.sort_values("dataset_index").reset_index(drop=True)

        emb_cols = embedding_columns(int(model_info["emb_dim"]))
        norm_stats = validate_basin_year_df(
            merged,
            emb_cols=emb_cols,
            expected_n=len(dataset),
        )

        # Force compact numeric dtypes before final write.
        merged["dataset_index"] = merged["dataset_index"].astype(np.int64)
        merged["site_id_int"] = merged["site_id_int"].astype(np.int64)
        merged["basin_index"] = merged["basin_index"].astype(np.int64)
        merged["year"] = merged["year"].astype(np.int32)
        merged[emb_cols] = merged[emb_cols].astype(np.float32)

        merged.to_parquet(
            basin_year_path,
            index=False,
            compression=parquet_compression(args.compression),
        )

        print("[INFO] Computing Eq.14 basin-level descriptors...")
        basin_df = aggregate_eq14(merged, emb_cols)
        basin_df[emb_cols] = basin_df[emb_cols].astype(np.float32)
        basin_df.to_parquet(
            basin_path,
            index=False,
            compression=parquet_compression(args.compression),
        )

        basin_norms = np.linalg.norm(
            basin_df[emb_cols].to_numpy(dtype=np.float32), axis=1
        )

        manifest = {
            "checkpoint": str(ckpt_path.resolve()),
            "checkpoint_epoch": int(epoch),
            "variant": variant,
            "seed": int(seed),
            "base_config": identity["base_config"],
            "experiment_name": experiment_name,
            "extraction": {
                "unaugmented": True,
                "model_eval": True,
                "torch_inference_mode": True,
                "amp": bool(not args.no_amp and device.type == "cuda"),
                "world_size": int(world_size),
                "batch_size_per_rank": int(args.batch_size),
                "num_workers_per_rank": int(args.num_workers),
            },
            "model": model_info,
            "data_paths": {
                "index_parquet": index_parquet,
                "grouped_zarr_path": grouped_zarr,
                "mask_zarr_path": mask_zarr,
                "basin_metadata": basin_metadata,
            },
            "outputs": {
                "basin_year_embeddings": str(basin_year_path),
                "basin_embeddings_eq14": str(basin_path),
            },
            "counts": {
                "basin_year_rows": int(len(merged)),
                "basins": int(len(basin_df)),
                "embedding_dim": int(len(emb_cols)),
                "min_year": int(merged["year"].min()),
                "max_year": int(merged["year"].max()),
                "min_years_per_basin": int(basin_df["n_years"].min()),
                "max_years_per_basin": int(basin_df["n_years"].max()),
            },
            "validation": {
                **norm_stats,
                "eq14_norm_mean": float(basin_norms.mean()),
                "eq14_norm_std": float(basin_norms.std()),
                "eq14_norm_min": float(basin_norms.min()),
                "eq14_norm_max": float(basin_norms.max()),
                "nan_count_basin_year": int(
                    np.isnan(merged[emb_cols].to_numpy(dtype=np.float32)).sum()
                ),
                "nan_count_basin": int(
                    np.isnan(basin_df[emb_cols].to_numpy(dtype=np.float32)).sum()
                ),
            },
            "aggregation": {
                "equation": "Normalize(mean_y(z_b,y))",
                "description": (
                    "Basin-year embeddings are generated from unaugmented inputs, "
                    "averaged over all available pretraining years for each basin, "
                    "then L2 normalized."
                ),
            },
        }

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(
            f"[DONE] {variant} seed={seed}: "
            f"{len(merged):,} basin-years -> {len(basin_df):,} basins"
        )
        print(
            "[CHECK] annual norm mean/std="
            f"{norm_stats['basin_year_norm_mean']:.6f}/"
            f"{norm_stats['basin_year_norm_std']:.6f}; "
            f"Eq14 norm mean={basin_norms.mean():.6f}"
        )
        print(f"[WRITE] {basin_year_path}")
        print(f"[WRITE] {basin_path}")
        print(f"[WRITE] {manifest_path}")

        if not args.keep_temp_shards:
            shutil.rmtree(temp_dir, ignore_errors=True)

    barrier()

    # Release checkpoint/model/data references before moving to next run.
    del model
    del dataset
    if device.type == "cuda":
        torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def run():
    args = parse_args()
    device, local_rank, global_rank, world_size, use_dist = setup_distributed()
    is_main = global_rank == 0

    try:
        checkpoints = discover_checkpoints(args)

        if is_main:
            print("\n" + "#" * 88)
            print("Basin2Vec Round-2 embedding extraction suite")
            print("#" * 88)
            print(f"device on rank 0   : {device}")
            print(f"world size         : {world_size}")
            print(f"checkpoints queued : {len(checkpoints)}")
            print(f"checkpoint root    : {args.checkpoint_root}")
            print(f"output root        : {args.output_root}")
            print(f"base config        : {args.base_config}")
            print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
            print("#" * 88 + "\n")

        for i, ckpt_path in enumerate(checkpoints, start=1):
            if is_main:
                print(f"\n[QUEUE] {i}/{len(checkpoints)}")
            process_checkpoint(
                ckpt_path=ckpt_path,
                args=args,
                device=device,
                global_rank=global_rank,
                world_size=world_size,
            )

        if is_main:
            print("\n[COMPLETE] All requested checkpoint embeddings have been extracted.")

    finally:
        cleanup_distributed()


@record
def main():
    try:
        run()
    except Exception:
        rank = os.environ.get("RANK", "0")
        local_rank = os.environ.get("LOCAL_RANK", "0")
        print(
            f"\n[FATAL][rank={rank} local_rank={local_rank}] "
            "Embedding extraction failed. Full traceback follows:",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()