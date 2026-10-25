#!/usr/bin/env python3
"""
Unified Basin2Vec rebuttal ablation trainer.

This script implements the finalized core ablations:

  full
  no_temporal_positive
  no_overlap_positive
  no_area_regularization
  no_monthly_branch
  no_static_branch
  no_metadata_encoder
  metadata_concat
  no_meta_denominator
  meta_as_positive
  no_negative_margin
  instance_contrastive

The same encoder/data/sampler/training setup is kept fixed unless the named
ablation explicitly changes one component.

Important configuration note
----------------------------
The submitted manuscript reports (lambda_t, lambda_o, lambda_a)
= (0.35, 0.12, 0.05), whereas the uploaded trainer used
(0.30, 0.12, 0.03). To avoid silently mixing settings, this script provides:

  --base-config paper      -> 0.35 / 0.12 / 0.05
  --base-config uploaded   -> 0.30 / 0.12 / 0.03

The released model (train_basin2vec_monthly_hclv4.py) uses the 'uploaded'
weights. The shipped Round-2 ablation results used 'paper'.

Use ONE base configuration for the full model and every ablation in the final
Round-2 table.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import traceback
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from basin2vec_dataset_grouped import (
    Basin2VecGroupedZarrDataset,
    DEFAULT_BASIN_META_COLS,
)
from basin_batch_sampler import (
    DistributedBasinBatchSampler,
    build_sampler_metadata,
)
from attention_encoder_monthly_ablation import GroupedBasinEncoderMonthlyV2
from hydrology_aware_contrastive_loss_ablation import (
    HydrologyAwareContrastiveLossV4,
    load_overlap_adjacency_weighted,
)


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Online softmax is disabled.*")
np.set_printoptions(threshold=np.inf)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


# ============================================================
# DATA / TRAINING PATHS: keep identical across all variants
# ============================================================
# Data locations are resolved from BASIN2VEC_DATA_ROOT (default: <repo>/data).
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DATA_ROOT = Path(os.environ.get("BASIN2VEC_DATA_ROOT", REPO_ROOT / "data"))
CACHE_DIR = DATA_ROOT / "cache"
CONFIG_DIR = REPO_ROOT / "emb" / "config" / "training_step5"

INDEX_PARQUET = str(CONFIG_DIR / "sample_index.parquet")
GROUPED_ZARR_PATH = str(CACHE_DIR / "patches_step4_grouped" / "grouped_monthly_static.zarr")
MASK_ZARR_PATH = str(CACHE_DIR / "static_masks.zarr")
OVERLAP_PAIRS_PARQUET = str(CACHE_DIR / "basin_overlap_pairs.parquet")
BASIN_METADATA = str(CACHE_DIR / "metadata" / "basin_metadata.parquet")
BASIN_META_COLS = DEFAULT_BASIN_META_COLS

CHECKPOINT_ROOT = SCRIPT_DIR / "checkpoints" / "ablation_round2"
RUN_ROOT = SCRIPT_DIR / "runs" / "ablation_round2"


# ============================================================
# FIXED TRAINING SETTINGS
# ============================================================
NUM_EPOCHS = 100
DEFAULT_SEED = 42
BASINS_PER_BATCH = 24
YEARS_PER_BASIN = 2
BATCH_SIZE = BASINS_PER_BATCH * YEARS_PER_BASIN

LR = 2e-4
WEIGHT_DECAY = 1e-4
EMB_DIM = 64
DROPOUT = 0.05
METADATA_DROPOUT = 0.15
TEMPORAL_LAYERS = 1
TEMPORAL_HEADS = 4

COMPILE_MODEL = False
PREDICT_MONTH_AUX = False
MONTH_AUX_WEIGHT = 0.0

NUM_WORKERS = 8
PREFETCH_FACTOR = 2
PIN_MEMORY = True
GRAD_CLIP_NORM = 1.0


# ============================================================
# BASE CONFIGURATIONS
# ============================================================
BASE_CONFIGS = {
    "paper": {
        "temporal_weight": 0.35,
        "overlap_weight": 0.12,
        "area_aux_weight": 0.05,
    },
    "uploaded": {
        "temporal_weight": 0.30,
        "overlap_weight": 0.12,
        "area_aux_weight": 0.03,
    },
}

# Other full-model loss settings from the uploaded implementation.
FULL_TEMPERATURE = 0.15
FULL_YEAR_TAU = 8.0
FULL_META_TAU = 1.5
FULL_MIN_NEGATIVE_WEIGHT = 0.20
FULL_NEG_WEIGHT = 0.05
FULL_NEG_MARGIN_LOW = 0.20
FULL_NEG_MARGIN_HIGH = 0.60

# Operationalization of the reviewer-requested "metadata as positive" variant.
# It replaces metadata denominator relief with direct metadata-positive weights.
# The values can be changed through CLI for the sensitivity study.
DEFAULT_META_POSITIVE_WEIGHT = 0.12
DEFAULT_META_POSITIVE_THRESHOLD = 0.80


@dataclass(frozen=True)
class VariantConfig:
    name: str
    description: str

    # Encoder switches
    use_monthly_branch: bool = True
    use_static_branch: bool = True
    metadata_fusion: str = "gated"  # gated | concat | none
    use_area_head: bool = True

    # Contrastive relation switches / weights
    temporal_weight: float = 0.35
    overlap_weight: float = 0.12
    use_meta_denominator: bool = True
    meta_positive_weight: float = 0.0
    meta_positive_threshold: Optional[float] = None
    neg_weight: float = FULL_NEG_WEIGHT

    # Auxiliary loss
    area_aux_weight: float = 0.05

    # Shared loss hyperparameters
    temperature: float = FULL_TEMPERATURE
    year_tau: float = FULL_YEAR_TAU
    meta_tau: float = FULL_META_TAU
    min_negative_weight: float = FULL_MIN_NEGATIVE_WEIGHT
    neg_margin_low: float = FULL_NEG_MARGIN_LOW
    neg_margin_high: float = FULL_NEG_MARGIN_HIGH


def build_variant_config(
    variant: str,
    base_config: str,
    meta_positive_weight: float,
    meta_positive_threshold: Optional[float],
) -> VariantConfig:
    if base_config not in BASE_CONFIGS:
        raise ValueError(f"Unknown base_config={base_config}")

    base = BASE_CONFIGS[base_config]
    full = VariantConfig(
        name="full",
        description="Full Basin2Vec objective and encoder",
        temporal_weight=float(base["temporal_weight"]),
        overlap_weight=float(base["overlap_weight"]),
        area_aux_weight=float(base["area_aux_weight"]),
    )

    if variant == "full":
        return full

    if variant == "no_temporal_positive":
        return replace(
            full,
            name=variant,
            description="Remove same-basin different-year temporal positives",
            temporal_weight=0.0,
        )

    if variant == "no_overlap_positive":
        return replace(
            full,
            name=variant,
            description="Remove overlap/nested-basin positive relations",
            overlap_weight=0.0,
        )

    if variant == "no_area_regularization":
        return replace(
            full,
            name=variant,
            description="Remove auxiliary drainage-area regularization",
            use_area_head=False,
            area_aux_weight=0.0,
        )

    if variant == "no_monthly_branch":
        return replace(
            full,
            name=variant,
            description="Remove monthly/dynamic hydroclimatic encoder branch",
            use_monthly_branch=False,
        )

    if variant == "no_static_branch":
        return replace(
            full,
            name=variant,
            description="Remove quasi-static physiographic encoder branch",
            use_static_branch=False,
        )

    if variant == "no_metadata_encoder":
        return replace(
            full,
            name=variant,
            description=(
                "Remove metadata encoder/fusion from representation; retain metadata "
                "only inside relation-aware SSL loss to isolate encoder contribution"
            ),
            metadata_fusion="none",
        )

    if variant == "metadata_concat":
        return replace(
            full,
            name=variant,
            description=(
                "Replace sigmoid-gated metadata fusion with concatenation + 2D-to-D projection"
            ),
            metadata_fusion="concat",
        )

    if variant == "no_meta_denominator":
        return replace(
            full,
            name=variant,
            description="Remove metadata-aware false-negative relief in contrastive denominator",
            use_meta_denominator=False,
        )

    if variant == "meta_as_positive":
        return replace(
            full,
            name=variant,
            description=(
                "Reviewer alternative: replace metadata denominator relief with direct "
                "metadata-similarity positive relation weights"
            ),
            use_meta_denominator=False,
            meta_positive_weight=float(meta_positive_weight),
            meta_positive_threshold=meta_positive_threshold,
        )

    if variant == "no_negative_margin":
        return replace(
            full,
            name=variant,
            description="Remove explicit metadata-aware true-negative margin penalty",
            neg_weight=0.0,
        )

    if variant == "instance_contrastive":
        return replace(
            full,
            name=variant,
            description=(
                "Standard instance-level contrastive objective: only paired augmented "
                "views are positives; relation-aware temporal/overlap/metadata/negative-margin "
                "mechanisms are disabled; encoder and area regularization remain unchanged"
            ),
            temporal_weight=0.0,
            overlap_weight=0.0,
            use_meta_denominator=False,
            meta_positive_weight=0.0,
            meta_positive_threshold=None,
            neg_weight=0.0,
        )

    raise ValueError(f"Unknown variant={variant}")


VARIANTS = [
    "full",
    "no_temporal_positive",
    "no_overlap_positive",
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


def optional_float(text: str) -> Optional[float]:
    if str(text).lower() in {"none", "null", "na", "off"}:
        return None
    return float(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified Basin2Vec Round-2 ablation trainer"
    )
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument(
        "--base-config",
        default="paper",
        choices=sorted(BASE_CONFIGS),
        help=(
            "Use 'paper' for reported 0.35/0.12/0.05 or 'uploaded' for the "
            "uploaded trainer's 0.30/0.12/0.03. Keep one choice across all runs."
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--checkpoint-root", type=str, default=str(CHECKPOINT_ROOT))
    parser.add_argument("--run-root", type=str, default=str(RUN_ROOT))
    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help="DataLoader workers per rank. Use 0 while debugging worker/Zarr issues.",
    )
    parser.add_argument(
        "--find-unused-parameters",
        action="store_true",
        help=(
            "Enable DDP unused-parameter detection. Normally unnecessary because "
            "ablation-disabled modules are not constructed, but useful for diagnosis."
        ),
    )
    parser.add_argument(
        "--detect-anomaly",
        action="store_true",
        help="Enable torch autograd anomaly detection (slow; debugging only).",
    )

    # Reviewer-alternative parameters.
    parser.add_argument(
        "--meta-positive-weight",
        type=float,
        default=DEFAULT_META_POSITIVE_WEIGHT,
    )
    parser.add_argument(
        "--meta-positive-threshold",
        type=optional_float,
        default=DEFAULT_META_POSITIVE_THRESHOLD,
        help="Similarity threshold in [0,1], or 'none' for continuous weighting of all non-overlap pairs.",
    )

    # Optional hyperparameter overrides. These also make this same script useful
    # for the reviewer-requested one-factor-at-a-time sensitivity study.
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--temporal-weight", type=float, default=None)
    parser.add_argument("--overlap-weight", type=float, default=None)
    parser.add_argument("--year-tau", type=float, default=None)
    parser.add_argument("--meta-tau", type=float, default=None)
    parser.add_argument("--min-negative-weight", type=float, default=None)
    parser.add_argument("--neg-weight", type=float, default=None)
    parser.add_argument("--area-aux-weight", type=float, default=None)

    return parser.parse_args()


def apply_cli_overrides(cfg: VariantConfig, args: argparse.Namespace) -> VariantConfig:
    updates = {}
    for arg_name, field_name in [
        ("temperature", "temperature"),
        ("temporal_weight", "temporal_weight"),
        ("overlap_weight", "overlap_weight"),
        ("year_tau", "year_tau"),
        ("meta_tau", "meta_tau"),
        ("min_negative_weight", "min_negative_weight"),
        ("neg_weight", "neg_weight"),
        ("area_aux_weight", "area_aux_weight"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            updates[field_name] = value

    if updates:
        cfg = replace(cfg, **updates)

    # Keep area head structurally consistent with area loss after overrides.
    if cfg.area_aux_weight <= 0 and cfg.use_area_head:
        cfg = replace(cfg, use_area_head=False)

    return cfg


# ============================================================
# DDP
# ============================================================
def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not torch.cuda.is_available():
            raise RuntimeError("torchrun/DDP was requested but CUDA is not available.")

        # Set the CUDA device before creating the NCCL process group.
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        dist.init_process_group(backend="nccl", init_method="env://")
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
        return device, local_rank, global_rank, world_size, True

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
    else:
        device = torch.device("cpu")
    return device, 0, 0, 1, False


def cleanup_ddp():
    """Best-effort distributed cleanup without hiding the original exception."""
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception as exc:
            print(
                f"[WARN] destroy_process_group() failed during cleanup: {exc}",
                file=sys.stderr,
                flush=True,
            )


def ddp_barrier(local_rank: int | None = None):
    if not dist.is_initialized():
        return
    try:
        if local_rank is not None:
            dist.barrier(device_ids=[local_rank])
        else:
            dist.barrier()
    except TypeError:
        dist.barrier()


# ============================================================
# DDP GATHER HELPERS
# ============================================================
def gather_with_local_grad(x: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return x
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    gathered = [torch.zeros_like(x) for _ in range(world_size)]
    dist.all_gather(gathered, x.detach())
    gathered[rank] = x
    return torch.cat(gathered, dim=0)


def gather_long_tensor(x: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return x
    world_size = dist.get_world_size()
    gathered = [torch.zeros_like(x) for _ in range(world_size)]
    dist.all_gather(gathered, x)
    return torch.cat(gathered, dim=0)


def gather_float_tensor(x: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return x
    world_size = dist.get_world_size()
    gathered = [torch.zeros_like(x) for _ in range(world_size)]
    dist.all_gather(gathered, x.detach())
    return torch.cat(gathered, dim=0)


# ============================================================
# HYDROLOGICALLY SAFE AUGMENTATION
# ============================================================
def augment_grouped_inputs(
    monthly_x: torch.Tensor | None,
    static_x: torch.Tensor | None,
    mask: torch.Tensor,
    static_landcover_slice: tuple[int, int] | None = None,
    monthly_noise_std: float = 0.03,
    static_noise_std: float = 0.02,
    spatial_dropout_p: float = 0.03,
    month_dropout_p: float = 0.08,
    variable_dropout_p: float = 0.02,
    monthly_scale_std: float = 0.04,
    clamp_value: float = 10.0,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    Same augmentation family as the uploaded trainer, but each branch is optional
    so branch-removal ablations do not waste GPU compute.
    """
    mask = mask.float()
    mask5 = mask.unsqueeze(1)

    mx = None
    if monthly_x is not None:
        if monthly_x.dim() != 5:
            raise ValueError(
                f"monthly_x expected [B,M,C,H,W], got {tuple(monthly_x.shape)}"
            )
        mx = monthly_x.clone()
        B, M, C, H, W = mx.shape

        if monthly_scale_std > 0:
            scale = 1.0 + monthly_scale_std * torch.randn(
                B, 1, C, 1, 1, device=mx.device, dtype=mx.dtype
            )
            mx = mx * scale

        if monthly_noise_std > 0:
            mx = mx + monthly_noise_std * torch.randn_like(mx) * mask5

        if variable_dropout_p > 0:
            channel_keep = (
                torch.rand(B, 1, C, 1, 1, device=mx.device, dtype=mx.dtype)
                > variable_dropout_p
            ).to(mx.dtype)
            mx = mx * channel_keep

        if spatial_dropout_p > 0:
            keep = (
                torch.rand(B, 1, 1, H, W, device=mx.device, dtype=mx.dtype)
                > spatial_dropout_p
            ).to(mx.dtype)
            mx = mx * keep * mask5

        if M > 1 and month_dropout_p > 0:
            month_keep = (
                torch.rand(B, M, 1, 1, 1, device=mx.device, dtype=mx.dtype)
                > month_dropout_p
            ).to(mx.dtype)
            all_dropped = month_keep.sum(dim=1, keepdim=True) == 0
            if all_dropped.any():
                restore_idx = torch.randint(0, M, (B,), device=mx.device)
                restore = torch.zeros_like(month_keep)
                restore[torch.arange(B, device=mx.device), restore_idx, 0, 0, 0] = 1.0
                month_keep = torch.where(all_dropped, restore, month_keep)
            mx = mx * month_keep

        mx = torch.nan_to_num(
            mx, nan=0.0, posinf=clamp_value, neginf=-clamp_value
        )
        mx = torch.clamp(mx, -clamp_value, clamp_value) * mask5

    sx = None
    if static_x is not None:
        if static_x.dim() != 4:
            raise ValueError(
                f"static_x expected [B,C,H,W], got {tuple(static_x.shape)}"
            )
        sx = static_x.clone()

        if static_noise_std > 0 and sx.numel() > 0:
            noise = static_noise_std * torch.randn_like(sx)
            if static_landcover_slice is not None:
                s, e = static_landcover_slice
                noise[:, s:e] = 0.0
            sx = sx + noise * mask

        sx = torch.nan_to_num(
            sx, nan=0.0, posinf=clamp_value, neginf=-clamp_value
        )
        sx = torch.clamp(sx, -clamp_value, clamp_value) * mask

    return mx, sx


# ============================================================
# MAIN
# ============================================================
def run_training():
    args = parse_args()

    cfg = build_variant_config(
        variant=args.variant,
        base_config=args.base_config,
        meta_positive_weight=args.meta_positive_weight,
        meta_positive_threshold=args.meta_positive_threshold,
    )
    cfg = apply_cli_overrides(cfg, args)

    # If a user turns area loss back on through an override for the explicit
    # no-area variant, fail loudly instead of silently changing the ablation.
    if args.variant == "no_area_regularization" and cfg.area_aux_weight != 0:
        raise ValueError(
            "no_area_regularization must keep --area-aux-weight 0. "
            "Use --variant full for sensitivity experiments."
        )

    default_experiment_name = f"Basin2Vec_{args.base_config}_{cfg.name}"
    experiment_name = args.experiment_name or default_experiment_name

    # Seed-specific subfolders prevent accidental mixing/overwrite across runs.
    checkpoint_dir = (
        Path(args.checkpoint_root) / experiment_name / f"seed_{args.seed}"
    )
    run_dir = Path(args.run_root) / experiment_name / f"seed_{args.seed}"

    device, local_rank, global_rank, world_size, use_ddp = setup_ddp()
    is_main = global_rank == 0

    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        if is_main:
            print("[DEBUG] Autograd anomaly detection ENABLED (training will be slower).")

    torch.manual_seed(args.seed + global_rank)
    np.random.seed(args.seed + global_rank)
    random.seed(args.seed + global_rank)

    use_cuda = device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_cuda)

    if is_main:
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        run_dir.mkdir(exist_ok=True, parents=True)
        print("\n" + "=" * 80)
        print("Basin2Vec Round-2 Ablation")
        print("=" * 80)
        print(f"variant           : {cfg.name}")
        print(f"description       : {cfg.description}")
        print(f"base_config       : {args.base_config}")
        print(f"seed              : {args.seed}")
        print(f"device            : {device}")
        print(f"world_size        : {world_size}")
        print(f"checkpoint_dir    : {checkpoint_dir}")
        print(json.dumps(asdict(cfg), indent=2))
        print("=" * 80 + "\n")

        with open(checkpoint_dir / "run_config.json", "w") as f:
            json.dump(
                {
                    "variant": asdict(cfg),
                    "base_config": args.base_config,
                    "seed": args.seed,
                    "epochs": args.epochs,
                    "batch": {
                        "basins_per_batch": BASINS_PER_BATCH,
                        "years_per_basin": YEARS_PER_BASIN,
                        "local_batch_size": BATCH_SIZE,
                        "world_size": world_size,
                    },
                    "optimizer": {
                        "lr": LR,
                        "weight_decay": WEIGHT_DECAY,
                        "grad_clip_norm": GRAD_CLIP_NORM,
                    },
                    "runtime": {
                        "num_workers": args.num_workers,
                        "find_unused_parameters": args.find_unused_parameters,
                        "detect_anomaly": args.detect_anomaly,
                    },
                    "augmentation": {
                        "monthly_noise_std": 0.03,
                        "static_noise_std": 0.02,
                        "spatial_dropout_p": 0.03,
                        "month_dropout_p": 0.08,
                        "variable_dropout_p": 0.02,
                        "monthly_scale_std": 0.04,
                        "cropping": False,
                        "rotation": False,
                        "month_permutation": False,
                    },
                },
                f,
                indent=2,
            )

    writer = SummaryWriter(str(run_dir)) if is_main else None

    overlap_adjacency, overlap_pair_count = load_overlap_adjacency_weighted(
        OVERLAP_PAIRS_PARQUET
    )
    if is_main:
        print(f"[INFO] Loaded weighted overlap-aware basin pairs: {overlap_pair_count}")

    dataset = Basin2VecGroupedZarrDataset(
        index_parquet=INDEX_PARQUET,
        grouped_zarr_path=GROUPED_ZARR_PATH,
        basin_metadata_path=BASIN_METADATA,
        basin_meta_cols=BASIN_META_COLS,
        mask_zarr_path=MASK_ZARR_PATH,
        require_done=True,
    )

    metadata_df = build_sampler_metadata(dataset, INDEX_PARQUET)
    batches_per_epoch = len(dataset) // (BATCH_SIZE * world_size)
    if batches_per_epoch <= 0:
        raise ValueError("batches_per_epoch <= 0; reduce BATCH_SIZE or world_size")

    batch_sampler = DistributedBasinBatchSampler(
        metadata_df=metadata_df,
        basins_per_batch=BASINS_PER_BATCH,
        years_per_basin=YEARS_PER_BASIN,
        batches_per_epoch=batches_per_epoch,
        seed=args.seed,
        rank=global_rank,
        world_size=world_size,
    )

    loader_kwargs = {
        "dataset": dataset,
        "batch_sampler": batch_sampler,
        "num_workers": args.num_workers,
        "pin_memory": PIN_MEMORY and use_cuda,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR
    loader = DataLoader(**loader_kwargs)

    example = dataset[0]
    monthly_var_names = list(dataset.monthly_variables)
    nonmonthly_var_names = list(dataset.nonmonthly_variables)
    monthly_in_channels = int(dataset.monthly_in_channels)
    static_in_channels = int(dataset.static_in_channels)
    months_per_year = int(dataset.months_per_year)

    static_landcover_slice = None
    if "land_cover" in dataset.static_channel_slices:
        s, e = dataset.static_channel_slices["land_cover"]
        static_landcover_slice = (int(s), int(e))

    if is_main:
        print(f"[INFO] Example sample keys: {list(example.keys())}")
        print(f"[INFO] Monthly variables: {monthly_var_names}")
        print(f"[INFO] Static/annual variables: {nonmonthly_var_names}")
        print(f"[INFO] Monthly channels: {monthly_in_channels}")
        print(f"[INFO] Static channels: {static_in_channels}")
        print(f"[INFO] Months/year: {months_per_year}")
        print(f"[INFO] Local batches/epoch: {batches_per_epoch}")
        print(f"[INFO] Global gathered base batch: {BATCH_SIZE * world_size}")

    # For no_metadata_encoder, metadata is removed ONLY from the encoder. It can
    # still be used by the loss, which isolates the representation-module effect.
    encoder_meta_dim = (
        len(BASIN_META_COLS) if cfg.metadata_fusion != "none" else 0
    )

    model = GroupedBasinEncoderMonthlyV2(
        monthly_in_channels=monthly_in_channels,
        static_in_channels=static_in_channels,
        emb_dim=EMB_DIM,
        meta_dim=encoder_meta_dim,
        max_months=months_per_year,
        temporal_layers=TEMPORAL_LAYERS,
        temporal_heads=TEMPORAL_HEADS,
        dropout=DROPOUT,
        metadata_dropout=METADATA_DROPOUT,
        predict_month_aux=PREDICT_MONTH_AUX and cfg.use_monthly_branch,
        use_monthly_branch=cfg.use_monthly_branch,
        use_static_branch=cfg.use_static_branch,
        metadata_fusion=cfg.metadata_fusion,
        use_area_head=cfg.use_area_head,
    ).to(device)

    if COMPILE_MODEL:
        model = torch.compile(model)

    if use_ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=args.find_unused_parameters,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    criterion = HydrologyAwareContrastiveLossV4(
        temperature=cfg.temperature,
        temporal_weight=cfg.temporal_weight,
        overlap_weight=cfg.overlap_weight,
        year_tau=cfg.year_tau,
        meta_tau=cfg.meta_tau,
        min_negative_weight=cfg.min_negative_weight,
        use_meta_denominator=cfg.use_meta_denominator,
        meta_positive_weight=cfg.meta_positive_weight,
        meta_positive_threshold=cfg.meta_positive_threshold,
        neg_weight=cfg.neg_weight,
        neg_margin_low=cfg.neg_margin_low,
        neg_margin_high=cfg.neg_margin_high,
        return_debug=False,
    )

    # Metadata is sent to the SSL loss only if an active loss component needs it.
    loss_uses_metadata = (
        cfg.use_meta_denominator
        or cfg.meta_positive_weight > 0
        or cfg.neg_weight > 0
    )

    global_step = 0

    for epoch in range(args.epochs):
        batch_sampler.set_epoch(epoch)
        model.train()

        epoch_loss_sum = 0.0
        epoch_con_sum = 0.0
        epoch_area_sum = 0.0
        epoch_n = 0

        if is_main:
            print(f"\n[INFO] Epoch {epoch + 1}/{args.epochs}")

        pbar = tqdm(loader, disable=not is_main)

        for batch in pbar:
            global_step += 1

            mask_cpu = batch["mask"]
            valid = mask_cpu.sum(dim=(1, 2, 3)) > 0

            # In DDP every rank must execute the same collectives with the same
            # tensor shapes. Never silently skip/filter a local DDP batch.
            if use_ddp:
                if not bool(valid.all()):
                    bad = int((~valid).sum().item())
                    raise RuntimeError(
                        f"Rank {global_rank} found {bad} invalid basin mask(s) in a "
                        "DDP batch. Filtering/skipping would desynchronize ranks."
                    )
            else:
                if int(valid.sum().item()) < 2:
                    continue

            mask = mask_cpu[valid].to(device, non_blocking=True).float()

            monthly_x = None
            if cfg.use_monthly_branch:
                monthly_x = (
                    batch["monthly_x"][valid]
                    .to(device, non_blocking=True)
                    .float()
                )

            static_x = None
            if cfg.use_static_branch:
                static_x = (
                    batch["static_x"][valid]
                    .to(device, non_blocking=True)
                    .float()
                )

            basin_meta = (
                batch["basin_meta"][valid]
                .to(device, non_blocking=True)
                .float()
            )

            area_target = None
            if cfg.area_aux_weight > 0:
                area_target = (
                    batch["log_area_km2_z"][valid]
                    .to(device, non_blocking=True)
                    .float()
                )

            site_ids_int = batch["site_id_int"][valid]
            if not torch.is_tensor(site_ids_int):
                site_ids_int = torch.tensor(site_ids_int, dtype=torch.long)
            site_ids_int = site_ids_int.to(device, non_blocking=True)

            years = batch["year"][valid]
            if not torch.is_tensor(years):
                years = torch.tensor(years, dtype=torch.long)
            years = years.to(device, non_blocking=True)

            monthly_x1, static_x1 = augment_grouped_inputs(
                monthly_x=monthly_x,
                static_x=static_x,
                mask=mask,
                static_landcover_slice=static_landcover_slice,
            )
            monthly_x2, static_x2 = augment_grouped_inputs(
                monthly_x=monthly_x,
                static_x=static_x,
                mask=mask,
                static_landcover_slice=static_landcover_slice,
            )

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=use_cuda):
                # Run both augmented views in ONE DDP forward pass. This keeps
                # DDP reducer state simple and still gives independent dropout/
                # augmentation realizations for the two views.
                view_batch = mask.shape[0]
                monthly_cat = (
                    torch.cat([monthly_x1, monthly_x2], dim=0)
                    if monthly_x1 is not None
                    else None
                )
                static_cat = (
                    torch.cat([static_x1, static_x2], dim=0)
                    if static_x1 is not None
                    else None
                )
                mask_cat = torch.cat([mask, mask], dim=0)
                basin_meta_cat = torch.cat([basin_meta, basin_meta], dim=0)

                z_cat, branch_attn_cat, _, _, aux_cat = model(
                    monthly_cat, static_cat, mask_cat, basin_meta_cat
                )

                if z_cat.shape[0] != 2 * view_batch:
                    raise RuntimeError(
                        f"Unexpected encoder batch size: got {z_cat.shape[0]}, "
                        f"expected {2 * view_batch}."
                    )

                z1 = z_cat[:view_batch]
                z2 = z_cat[view_batch:]
                branch_attn1 = branch_attn_cat[:view_batch]

                z1_all = gather_with_local_grad(z1)
                z2_all = gather_with_local_grad(z2)
                site_ids_all = gather_long_tensor(site_ids_int)
                years_all = gather_long_tensor(years)

                if loss_uses_metadata:
                    basin_meta_all = gather_float_tensor(basin_meta)
                else:
                    basin_meta_all = None

                contrastive_loss = criterion(
                    z1_all,
                    z2_all,
                    site_ids_all,
                    years_all,
                    basin_meta=basin_meta_all,
                    overlap_adjacency=overlap_adjacency,
                )

                if cfg.area_aux_weight > 0:
                    area_pred_cat = aux_cat["area_pred"]
                    if area_pred_cat is None:
                        raise RuntimeError(
                            "area_aux_weight > 0 but encoder has no area head"
                        )
                    area_pred1 = area_pred_cat[:view_batch]
                    area_pred2 = area_pred_cat[view_batch:]
                    area_loss = 0.5 * (
                        F.smooth_l1_loss(area_pred1, area_target)
                        + F.smooth_l1_loss(area_pred2, area_target)
                    )
                else:
                    area_loss = torch.zeros(
                        (), device=device, dtype=contrastive_loss.dtype
                    )

                loss = contrastive_loss + cfg.area_aux_weight * area_loss

            scaler.scale(loss).backward()

            if GRAD_CLIP_NORM is not None and GRAD_CLIP_NORM > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=GRAD_CLIP_NORM
                )

            scaler.step(optimizer)
            scaler.update()

            epoch_loss_sum += float(loss.item())
            epoch_con_sum += float(contrastive_loss.item())
            epoch_area_sum += float(area_loss.item())
            epoch_n += 1

            if is_main:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    con=f"{contrastive_loss.item():.4f}",
                    area=f"{area_loss.item():.4f}",
                )

                if writer is not None:
                    writer.add_scalar("train/loss", loss.item(), global_step)
                    writer.add_scalar(
                        "train/contrastive_loss",
                        contrastive_loss.item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "train/area_loss", area_loss.item(), global_step
                    )

                    # Branch attention labels adapt to branch-removal variants.
                    bw = branch_attn1.detach().mean(dim=0).squeeze(-1)
                    cursor = 0
                    if cfg.use_monthly_branch:
                        writer.add_scalar(
                            "attention_branch/monthly", bw[cursor].item(), global_step
                        )
                        cursor += 1
                    if cfg.use_static_branch:
                        writer.add_scalar(
                            "attention_branch/static", bw[cursor].item(), global_step
                        )

                    month_attention_cat = aux_cat.get("month_attention")
                    month_attention = (
                        month_attention_cat[:view_batch]
                        if month_attention_cat is not None
                        else None
                    )
                    if month_attention is not None and global_step % 100 == 0:
                        mw = month_attention.detach().mean(dim=0).squeeze(-1)
                        for m_idx, val in enumerate(mw, start=1):
                            writer.add_scalar(
                                f"attention_month/month_{m_idx:02d}",
                                val.item(),
                                global_step,
                            )

        if use_ddp:
            ddp_barrier(local_rank)

        if is_main and epoch_n > 0:
            epoch_loss_mean = epoch_loss_sum / epoch_n
            epoch_con_mean = epoch_con_sum / epoch_n
            epoch_area_mean = epoch_area_sum / epoch_n

            print(
                f"[EPOCH SUMMARY] epoch={epoch + 1} "
                f"loss={epoch_loss_mean:.4f} "
                f"con={epoch_con_mean:.4f} "
                f"area={epoch_area_mean:.4f}"
            )

            if writer is not None:
                writer.add_scalar("epoch/loss", epoch_loss_mean, epoch + 1)
                writer.add_scalar(
                    "epoch/contrastive_loss", epoch_con_mean, epoch + 1
                )
                writer.add_scalar("epoch/area_loss", epoch_area_mean, epoch + 1)

        if is_main and ((epoch + 1) % args.save_every == 0):
            ckpt_path = checkpoint_dir / f"epoch_{epoch + 1}.pt"
            raw_model = model.module if isinstance(model, DDP) else model

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": raw_model.state_dict(),
                    "ablation_variant": cfg.name,
                    "ablation_description": cfg.description,
                    "variant_config": asdict(cfg),
                    "base_config": args.base_config,
                    "seed": args.seed,
                    "encoder_name": "GroupedBasinEncoderMonthlyV2",
                    "dataset_name": "Basin2VecGroupedZarrDataset",
                    "monthly_variables": monthly_var_names,
                    "nonmonthly_variables": nonmonthly_var_names,
                    "monthly_in_channels": monthly_in_channels,
                    "static_in_channels": static_in_channels,
                    "static_channel_slices": dataset.static_channel_slices,
                    "monthly_channel_slices": dataset.monthly_channel_slices,
                    "emb_dim": EMB_DIM,
                    "months_per_year": months_per_year,
                    "basins_per_batch": BASINS_PER_BATCH,
                    "years_per_basin": YEARS_PER_BASIN,
                    "num_workers": args.num_workers,
                    "find_unused_parameters": args.find_unused_parameters,
                    "basin_meta_cols": BASIN_META_COLS,
                    "augmentation_config": {
                        "monthly_noise_std": 0.03,
                        "static_noise_std": 0.02,
                        "spatial_dropout_p": 0.03,
                        "month_dropout_p": 0.08,
                        "variable_dropout_p": 0.02,
                        "monthly_scale_std": 0.04,
                    },
                    "paths": {
                        "index_parquet": INDEX_PARQUET,
                        "grouped_zarr_path": GROUPED_ZARR_PATH,
                        "mask_zarr_path": MASK_ZARR_PATH,
                        "basin_metadata": BASIN_METADATA,
                        "overlap_pairs_parquet": OVERLAP_PAIRS_PARQUET,
                    },
                },
                ckpt_path,
            )
            print(f"[INFO] Saved checkpoint: {ckpt_path}")

    if is_main and writer is not None:
        writer.close()


@record
def main():
    """TorchElastic-aware entrypoint that preserves the real worker traceback."""
    try:
        run_training()
    except Exception:
        rank = os.environ.get("RANK", "0")
        local_rank = os.environ.get("LOCAL_RANK", "0")
        print(
            f"\n[FATAL][rank={rank} local_rank={local_rank}] "
            "Basin2Vec training failed. Full traceback follows:",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        raise
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()
