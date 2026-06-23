#!/usr/bin/env python3
# ==========================================================
# Fast Basin2Vec grouped monthly/static HCL-V4 trainer
# Uses pre-stacked grouped Zarr:
#
#   monthly_x : [N_basin, N_year, 12, C_monthly, H, W]
#   static_x  : [N_basin, N_year, C_static, H, W]
#
# This avoids reading 17 separate Zarr files per sample.
# ==========================================================

from __future__ import annotations

import os
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from torch.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from basin2vec_dataset_grouped import (
    Basin2VecGroupedZarrDataset,
    DEFAULT_BASIN_META_COLS,
)
from attention_encoder_monthly import GroupedBasinEncoderMonthlyV2
from hydrology_aware_contrastive_loss import (
    HydrologyAwareContrastiveLossV4,
    load_overlap_adjacency_weighted,
    build_overlap_strength_matrix,
)
from basin_batch_sampler import (
    DistributedBasinBatchSampler,
    build_sampler_metadata,
)


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Online softmax is disabled.*")

np.set_printoptions(threshold=np.inf)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
NUM_EPOCHS = 100
SEED = 42

# Start with the old safer batch size.
# Once speed is improved and GPU utilization is good, try BASINS_PER_BATCH=48.
BASINS_PER_BATCH = 24
YEARS_PER_BASIN = 2
BATCH_SIZE = BASINS_PER_BATCH * YEARS_PER_BASIN

LR = 2e-4
WEIGHT_DECAY = 1e-4
EMB_DIM = 64

COMPILE_MODEL = False

INDEX_PARQUET = "../config/training_step5/sample_index.parquet"

GROUPED_ZARR_PATH = "/data/basin2vec/cache/patches_step4_grouped/grouped_monthly_static.zarr"
MASK_ZARR_PATH = "/data/basin2vec/cache/static_masks.zarr"

OVERLAP_PAIRS_PARQUET = "/data/basin2vec/cache/basin_overlap_pairs.parquet"
BASIN_METADATA = "/data/basin2vec/cache/metadata/basin_metadata.parquet"

BASIN_META_COLS = DEFAULT_BASIN_META_COLS

AREA_AUX_WEIGHT = 0.03

# Keep month auxiliary off for speed/stability.
# The objective is HCL-V4 + metadata + overlap + area auxiliary.
PREDICT_MONTH_AUX = False
MONTH_AUX_WEIGHT = 0.0

CHECKPOINT_DIR = Path("checkpoints/HCLV4_grouped_monthly_static_meta_areaaux_64d")
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

RUN_DIR = "runs/HCLV4_grouped_monthly_static_meta_areaaux_64d"

NUM_WORKERS = 8
PREFETCH_FACTOR = 2
PIN_MEMORY = True

GRAD_CLIP_NORM = 1.0


# ------------------------------------------------------------
# DDP SETUP
# ------------------------------------------------------------
def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")

        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()

        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        return device, local_rank, global_rank, world_size, True

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
    else:
        device = torch.device("cpu")

    return device, 0, 0, 1, False


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


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


# ------------------------------------------------------------
# GATHER HELPERS
# ------------------------------------------------------------
def gather_with_local_grad(x: torch.Tensor) -> torch.Tensor:
    """
    Gather tensor across ranks while preserving gradients for local slice.
    """
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


# ------------------------------------------------------------
# GROUPED AUGMENTATION
# ------------------------------------------------------------
def augment_grouped_inputs(
    monthly_x: torch.Tensor,
    static_x: torch.Tensor,
    mask: torch.Tensor,
    static_landcover_slice: tuple[int, int] | None = None,
    monthly_noise_std: float = 0.03,
    static_noise_std: float = 0.02,
    spatial_dropout_p: float = 0.03,
    month_dropout_p: float = 0.08,
    variable_dropout_p: float = 0.02,
    monthly_scale_std: float = 0.04,
    clamp_value: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Hydrologically safe augmentation for pre-stacked grouped inputs.

    monthly_x:
        [B, M, C_monthly, H, W]

    static_x:
        [B, C_static, H, W]

    mask:
        [B, 1, H, W]

    Notes:
        - static land_cover one-hot channels should not receive noise.
        - month dropout is applied at the month level.
        - variable dropout is applied at monthly-channel level.
    """
    if monthly_x.dim() != 5:
        raise ValueError(f"monthly_x expected [B,M,C,H,W], got {tuple(monthly_x.shape)}")

    if static_x.dim() != 4:
        raise ValueError(f"static_x expected [B,C,H,W], got {tuple(static_x.shape)}")

    mask = mask.float()
    mask5 = mask.unsqueeze(1)

    mx = monthly_x.clone()
    sx = static_x.clone()

    B, M, C, H, W = mx.shape

    # Mild multiplicative scaling across the whole monthly sequence.
    if monthly_scale_std > 0:
        scale = 1.0 + monthly_scale_std * torch.randn(
            B,
            1,
            C,
            1,
            1,
            device=mx.device,
            dtype=mx.dtype,
        )
        mx = mx * scale

    # Monthly noise.
    if monthly_noise_std > 0:
        mx = mx + monthly_noise_std * torch.randn_like(mx) * mask5

    # Monthly channel dropout.
    if variable_dropout_p > 0:
        channel_keep = (
            torch.rand(B, 1, C, 1, 1, device=mx.device, dtype=mx.dtype)
            > variable_dropout_p
        ).to(mx.dtype)
        mx = mx * channel_keep

    # Monthly spatial dropout.
    if spatial_dropout_p > 0:
        keep = (
            torch.rand(B, 1, 1, H, W, device=mx.device, dtype=mx.dtype)
            > spatial_dropout_p
        ).to(mx.dtype)
        mx = mx * keep * mask5

    # Month dropout.
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

    # Static noise, excluding land_cover one-hot channels.
    if static_noise_std > 0 and sx.numel() > 0:
        noise = static_noise_std * torch.randn_like(sx)

        if static_landcover_slice is not None:
            s, e = static_landcover_slice
            noise[:, s:e] = 0.0

        sx = sx + noise * mask

    mx = torch.nan_to_num(
        mx,
        nan=0.0,
        posinf=clamp_value,
        neginf=-clamp_value,
    )
    sx = torch.nan_to_num(
        sx,
        nan=0.0,
        posinf=clamp_value,
        neginf=-clamp_value,
    )

    mx = torch.clamp(mx, -clamp_value, clamp_value) * mask5
    sx = torch.clamp(sx, -clamp_value, clamp_value) * mask

    return mx, sx


# ------------------------------------------------------------
# OPTIONAL MONTH AUXILIARY TARGET
# ------------------------------------------------------------
def compute_monthly_stack_target(
    monthly_x: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    monthly_x:
        [B, M, C, H, W]

    mask:
        [B, 1, H, W]

    returns:
        [B, M]
    """
    if monthly_x.dim() != 5:
        raise ValueError(
            f"Expected monthly_x [B,M,C,H,W], got {tuple(monthly_x.shape)}"
        )

    mask = mask.float()
    mask5 = mask.unsqueeze(1)

    denom = mask.sum(dim=(2, 3)).clamp(min=1.0)

    target = (monthly_x * mask5).sum(dim=(-2, -1)) / denom[:, None, :]
    target = target.mean(dim=2)

    return target


def compute_month_aux_loss(
    month_pred: torch.Tensor | None,
    month_target: torch.Tensor | None,
) -> torch.Tensor:
    if month_pred is None or month_target is None:
        device = None

        if month_pred is not None:
            device = month_pred.device
        elif month_target is not None:
            device = month_target.device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        return torch.zeros((), device=device)

    if month_pred.shape != month_target.shape:
        raise ValueError(
            f"Month prediction/target shape mismatch: "
            f"pred={tuple(month_pred.shape)}, target={tuple(month_target.shape)}"
        )

    return F.smooth_l1_loss(month_pred, month_target)


# ------------------------------------------------------------
# DEBUG SUMMARY
# ------------------------------------------------------------
def relation_debug_summary(
    z1: torch.Tensor,
    z2: torch.Tensor,
    site_ids_int: torch.Tensor,
    years: torch.Tensor,
    overlap_adjacency,
    year_tau: float,
) -> dict[str, float | int]:
    device = z1.device

    z1 = F.normalize(z1.float(), dim=1)
    z2 = F.normalize(z2.float(), dim=1)

    z = torch.cat([z1, z2], dim=0)

    site_ids_int = site_ids_int.to(device=device, dtype=torch.long)
    years = years.to(device=device, dtype=torch.long)

    site_tensor = torch.cat([site_ids_int, site_ids_int], dim=0)
    year_tensor = torch.cat([years, years], dim=0)

    N = z.shape[0]
    eye = torch.eye(N, device=device, dtype=torch.bool)

    same_basin = site_tensor.unsqueeze(0) == site_tensor.unsqueeze(1)
    same_year = year_tensor.unsqueeze(0) == year_tensor.unsqueeze(1)

    overlap_strength = build_overlap_strength_matrix(
        site_tensor=site_tensor,
        overlap_adjacency=overlap_adjacency,
        eye=eye,
    )

    overlap_mask = overlap_strength > 0

    hard_pos = same_basin & same_year & (~eye)
    temporal_pos = same_basin & (~same_year) & (~eye)
    overlap_pos = (~same_basin) & overlap_mask & (~eye)
    neg_mask = (~same_basin) & (~overlap_mask) & (~eye)

    year_gap = (year_tensor.unsqueeze(0) - year_tensor.unsqueeze(1)).abs().float()
    temporal_decay = torch.exp(-year_gap / float(year_tau))

    ov_values = overlap_strength[overlap_pos]
    ov_mean = float(ov_values.mean().item()) if ov_values.numel() > 0 else 0.0
    ov_max = float(ov_values.max().item()) if ov_values.numel() > 0 else 0.0

    temp_values = temporal_decay[temporal_pos]
    temp_decay_mean = float(temp_values.mean().item()) if temp_values.numel() > 0 else 0.0

    return {
        "anchors": int(N),
        "base_batch": int(N // 2),
        "hard_total": int(hard_pos.sum().item()),
        "temporal_total": int(temporal_pos.sum().item()),
        "overlap_total": int(overlap_pos.sum().item()),
        "negative_total": int(neg_mask.sum().item()),
        "hard_per_anchor_mean": float(hard_pos.sum(dim=1).float().mean().item()),
        "temporal_per_anchor_mean": float(temporal_pos.sum(dim=1).float().mean().item()),
        "overlap_per_anchor_mean": float(overlap_pos.sum(dim=1).float().mean().item()),
        "negative_per_anchor_mean": float(neg_mask.sum(dim=1).float().mean().item()),
        "overlap_strength_mean": ov_mean,
        "overlap_strength_max": ov_max,
        "temporal_decay_mean": temp_decay_mean,
    }


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    device, local_rank, global_rank, world_size, use_ddp = setup_ddp()
    is_main = global_rank == 0

    torch.manual_seed(SEED + global_rank)
    np.random.seed(SEED + global_rank)
    random.seed(SEED + global_rank)

    use_cuda = device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_cuda)

    writer = None

    if is_main:
        print(f"[INFO] Device: {device}")
        print(f"[INFO] World size: {world_size}")
        print(f"[INFO] COMPILE_MODEL: {COMPILE_MODEL}")
        print(f"[INFO] PREDICT_MONTH_AUX: {PREDICT_MONTH_AUX}")
        print(f"[INFO] GROUPED_ZARR_PATH: {GROUPED_ZARR_PATH}")
        writer = SummaryWriter(RUN_DIR)

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
        seed=SEED,
        rank=global_rank,
        world_size=world_size,
    )

    loader_kwargs = {
        "dataset": dataset,
        "batch_sampler": batch_sampler,
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY and use_cuda,
    }

    if NUM_WORKERS > 0:
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
        print(f"[INFO] Non-monthly variables: {nonmonthly_var_names}")
        print(f"[INFO] monthly_x shape: {tuple(example['monthly_x'].shape)}")
        print(f"[INFO] static_x shape: {tuple(example['static_x'].shape)}")
        print(f"[INFO] Monthly grouped channels: {monthly_in_channels}")
        print(f"[INFO] Static grouped channels: {static_in_channels}")
        print(f"[INFO] Months per year: {months_per_year}")
        print(f"[INFO] Static channel slices: {dataset.static_channel_slices}")
        print(f"[INFO] Static land_cover slice: {static_landcover_slice}")
        print(f"[INFO] Basin metadata columns: {BASIN_META_COLS}")
        print(f"[INFO] Local batch size: {BATCH_SIZE}")
        print(f"[INFO] Basins per batch: {BASINS_PER_BATCH}")
        print(f"[INFO] Years per basin: {YEARS_PER_BASIN}")
        print(f"[INFO] Local batches per epoch: {batches_per_epoch}")
        print(f"[INFO] Global gathered base batch size: {BATCH_SIZE * world_size}")

    model = GroupedBasinEncoderMonthlyV2(
        monthly_in_channels=monthly_in_channels,
        static_in_channels=static_in_channels,
        emb_dim=EMB_DIM,
        meta_dim=len(BASIN_META_COLS),
        max_months=months_per_year,
        temporal_layers=1,
        temporal_heads=4,
        dropout=0.05,
        metadata_dropout=0.15,
        predict_month_aux=PREDICT_MONTH_AUX,
    ).to(device)

    if COMPILE_MODEL:
        model = torch.compile(model)

    if use_ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = HydrologyAwareContrastiveLossV4(
        temperature=0.15,
        temporal_weight=0.30,
        overlap_weight=0.12,
        year_tau=8.0,
        meta_tau=1.5,
        min_negative_weight=0.20,
        neg_weight=0.05,
        neg_margin_low=0.20,
        neg_margin_high=0.60,
        return_debug=False,
    )

    global_step = 0
    logged_relation_debug = False

    for epoch in range(NUM_EPOCHS):
        batch_sampler.set_epoch(epoch)
        model.train()

        epoch_loss_sum = 0.0
        epoch_con_sum = 0.0
        epoch_area_sum = 0.0
        epoch_month_sum = 0.0
        epoch_n = 0

        if is_main:
            print(f"\n[INFO] Epoch {epoch + 1}/{NUM_EPOCHS}")

        pbar = tqdm(loader, disable=not is_main)

        for batch in pbar:
            global_step += 1

            mask_cpu = batch["mask"]
            valid = mask_cpu.sum(dim=(1, 2, 3)) > 0

            if valid.sum() < 2:
                continue

            if use_ddp and not bool(valid.all()):
                raise RuntimeError(
                    "Found invalid basin masks in a DDP batch. "
                    "Filtering would cause different local batch sizes across ranks."
                )

            mask = mask_cpu[valid].to(device, non_blocking=True).float()

            monthly_x = batch["monthly_x"][valid].to(device, non_blocking=True).float()
            static_x = batch["static_x"][valid].to(device, non_blocking=True).float()

            basin_meta = batch["basin_meta"][valid].to(device, non_blocking=True).float()
            area_target = batch["log_area_km2_z"][valid].to(device, non_blocking=True).float()

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

            if PREDICT_MONTH_AUX and MONTH_AUX_WEIGHT > 0:
                month_target = compute_monthly_stack_target(monthly_x, mask)
            else:
                month_target = None

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=use_cuda):
                z1, branch_attn1, _, _, aux1 = model(
                    monthly_x1,
                    static_x1,
                    mask,
                    basin_meta,
                )

                z2, branch_attn2, _, _, aux2 = model(
                    monthly_x2,
                    static_x2,
                    mask,
                    basin_meta,
                )

                z1_all = gather_with_local_grad(z1)
                z2_all = gather_with_local_grad(z2)

                site_ids_all = gather_long_tensor(site_ids_int)
                years_all = gather_long_tensor(years)
                basin_meta_all = gather_float_tensor(basin_meta)

                contrastive_loss = criterion(
                    z1_all,
                    z2_all,
                    site_ids_all,
                    years_all,
                    basin_meta=basin_meta_all,
                    overlap_adjacency=overlap_adjacency,
                )

                area_loss = 0.5 * (
                    F.smooth_l1_loss(aux1["area_pred"], area_target)
                    + F.smooth_l1_loss(aux2["area_pred"], area_target)
                )

                if PREDICT_MONTH_AUX and MONTH_AUX_WEIGHT > 0:
                    month_loss = 0.5 * (
                        compute_month_aux_loss(aux1["month_pred"], month_target)
                        + compute_month_aux_loss(aux2["month_pred"], month_target)
                    )
                else:
                    month_loss = torch.zeros(
                        (),
                        device=device,
                        dtype=contrastive_loss.dtype,
                    )

                loss = (
                    contrastive_loss
                    + AREA_AUX_WEIGHT * area_loss
                    + MONTH_AUX_WEIGHT * month_loss
                )

            if is_main and not logged_relation_debug:
                with torch.no_grad():
                    summary = relation_debug_summary(
                        z1=z1_all.detach(),
                        z2=z2_all.detach(),
                        site_ids_int=site_ids_all.detach(),
                        years=years_all.detach(),
                        overlap_adjacency=overlap_adjacency,
                        year_tau=criterion.year_tau,
                    )

                print("\n[DEBUG] One gathered pre-stacked grouped batch relation summary")
                for k, v in summary.items():
                    print(f"  {k}: {v}")

                if writer is not None:
                    for k, v in summary.items():
                        writer.add_scalar(f"debug_loss_space/{k}", v, global_step)

                logged_relation_debug = True

            scaler.scale(loss).backward()

            if GRAD_CLIP_NORM is not None and GRAD_CLIP_NORM > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)

            scaler.step(optimizer)
            scaler.update()

            epoch_loss_sum += float(loss.item())
            epoch_con_sum += float(contrastive_loss.item())
            epoch_area_sum += float(area_loss.item())
            epoch_month_sum += float(month_loss.item())
            epoch_n += 1

            if is_main:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    con=f"{contrastive_loss.item():.4f}",
                    area=f"{area_loss.item():.4f}",
                    month=f"{month_loss.item():.4f}",
                )

                if writer is not None:
                    writer.add_scalar("train/loss", loss.item(), global_step)
                    writer.add_scalar("train/contrastive_loss", contrastive_loss.item(), global_step)
                    writer.add_scalar("train/area_loss", area_loss.item(), global_step)
                    writer.add_scalar("train/month_loss", month_loss.item(), global_step)

                    bw = branch_attn1.detach().mean(dim=0).squeeze(-1)
                    writer.add_scalar("attention_branch/monthly", bw[0].item(), global_step)

                    if bw.numel() > 1:
                        writer.add_scalar("attention_branch/static", bw[1].item(), global_step)

                    if global_step % 100 == 0:
                        mw = aux1["month_attention"].detach().mean(dim=0).squeeze(-1)

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
            epoch_month_mean = epoch_month_sum / epoch_n

            print(
                f"[EPOCH SUMMARY] epoch={epoch + 1} "
                f"loss={epoch_loss_mean:.4f} "
                f"con={epoch_con_mean:.4f} "
                f"area={epoch_area_mean:.4f} "
                f"month={epoch_month_mean:.4f}"
            )

            if writer is not None:
                writer.add_scalar("epoch/loss", epoch_loss_mean, epoch + 1)
                writer.add_scalar("epoch/contrastive_loss", epoch_con_mean, epoch + 1)
                writer.add_scalar("epoch/area_loss", epoch_area_mean, epoch + 1)
                writer.add_scalar("epoch/month_loss", epoch_month_mean, epoch + 1)

        if is_main and ((epoch + 1) % 5 == 0):
            ckpt_path = CHECKPOINT_DIR / f"epoch_{epoch + 1}.pt"

            raw_model = model.module if isinstance(model, DDP) else model

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": raw_model.state_dict(),

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

                    "basin_meta_cols": BASIN_META_COLS,
                    "area_aux_weight": AREA_AUX_WEIGHT,
                    "month_aux_weight": MONTH_AUX_WEIGHT,
                    "predict_month_aux": PREDICT_MONTH_AUX,

                    "loss_name": "HydrologyAwareContrastiveLossV4",
                    "loss_config": {
                        "temperature": criterion.temperature,
                        "temporal_weight": criterion.temporal_weight,
                        "overlap_weight": criterion.overlap_weight,
                        "year_tau": criterion.year_tau,
                        "meta_tau": criterion.meta_tau,
                        "min_negative_weight": criterion.min_negative_weight,
                        "neg_weight": criterion.neg_weight,
                        "neg_margin_low": criterion.neg_margin_low,
                        "neg_margin_high": criterion.neg_margin_high,
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

    cleanup_ddp()


if __name__ == "__main__":
    main()