#!/usr/bin/env python3
"""
Distributed evaluation for grouped/pre-stacked monthly-static Basin2Vec HCL-V4.

This evaluator is for checkpoints trained with:

    Basin2VecGroupedZarrDataset
    GroupedBasinEncoderMonthlyV2

Example checkpoint:

    checkpoints/HCLV4_grouped_monthly_static_meta_areaaux_64d/epoch_10.pt

Run:

    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
        --master_port=29518 evaluate_grouped_prestacked_hclv4.py

Outputs:

    evaluation_outputs/HCLV4_grouped_monthly_static_meta_areaaux_64d/
        basin_embeddings_full.npz
        metrics_summary.txt
        relation_hierarchy_arrays.npz
        attention_arrays.npz
        plots/*.png
"""

from __future__ import annotations

import os
import math
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from basin2vec_dataset_grouped import (
    Basin2VecGroupedZarrDataset,
    DEFAULT_BASIN_META_COLS,
)
from attention_encoder_monthly import GroupedBasinEncoderMonthlyV2
from hydrology_aware_contrastive_loss import load_overlap_adjacency_weighted


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Online softmax is disabled.*")

np.set_printoptions(threshold=np.inf)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
CHECKPOINT = "checkpoints/HCLV4_grouped_monthly_static_meta_areaaux_64d_ablation_no_temporal/epoch_100.pt"

INDEX_PARQUET = "../config/training_step5/sample_index.parquet"
GROUPED_ZARR_PATH = "/data/basin2vec/cache/patches_step4_grouped/grouped_monthly_static.zarr"
MASK_ZARR_PATH = "/data/basin2vec/cache/static_masks.zarr"

BASIN_METADATA = "/data/basin2vec/cache/metadata/basin_metadata.parquet"
OVERLAP_PAIRS_PARQUET = "/data/basin2vec/cache/basin_overlap_pairs.parquet"

OUT_DIR = Path("evaluation_outputs/HCLV4_grouped_monthly_static_meta_areaaux_64d_ablation_no_temporal")
PLOT_DIR = OUT_DIR / "plots"

BATCH_SIZE = 256
NUM_WORKERS = 8
PREFETCH_FACTOR = 2
SEED = 42

MAX_QUERIES = 5000
MAX_RELATION_PAIRS = 100000
MAX_PROJECTION_POINTS = 5000
MAX_INTER_PAIRS = 100000
MAX_NEGATIVE_TEMPORAL_PAIRS = 50000

USE_AMP = True
STRICT_LOAD = True

RNG = np.random.default_rng(SEED)


# ----------------------------------------------------------
# DDP SETUP
# ----------------------------------------------------------
def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        return local_rank, global_rank, world_size, True

    local_rank = 0
    global_rank = 0
    world_size = 1
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    return local_rank, global_rank, world_size, False


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_dist() -> bool:
    return dist.is_initialized()


def barrier(local_rank: int | None = None):
    if not is_dist():
        return

    try:
        if local_rank is not None:
            dist.barrier(device_ids=[local_rank])
        else:
            dist.barrier()
    except TypeError:
        dist.barrier()


# ----------------------------------------------------------
# Distributed non-padding eval sampler
# ----------------------------------------------------------
class DistributedEvalSampler(Sampler):
    """
    Shards dataset indices across ranks without padding.
    Each sample is evaluated exactly once globally.
    """

    def __init__(self, dataset, rank: int, world_size: int):
        self.dataset = dataset
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.total_size = len(dataset)

    def __iter__(self):
        indices = list(range(self.total_size))
        shard = indices[self.rank:self.total_size:self.world_size]
        return iter(shard)

    def __len__(self):
        return math.ceil((self.total_size - self.rank) / self.world_size)


# ----------------------------------------------------------
# Utilities
# ----------------------------------------------------------
def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), eps, None)


def save_json_like_txt(path: Path, data: dict):
    with open(path, "w") as f:
        for k, v in data.items():
            f.write(f"{k}: {v}\n")


def safe_mean(x):
    if x is None or len(x) == 0:
        return float("nan")
    return float(np.mean(x))


def safe_std(x):
    if x is None or len(x) == 0:
        return float("nan")
    return float(np.std(x))


def safe_median(x):
    if x is None or len(x) == 0:
        return float("nan")
    return float(np.median(x))


def finite_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return float("nan")

    x = x[valid]
    y = y[valid]

    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")

    return float(np.corrcoef(x, y)[0, 1])


def clean_state_dict_keys(state_dict: dict) -> dict:
    cleaned = {}

    for k, v in state_dict.items():
        nk = k

        if nk.startswith("module._orig_mod."):
            nk = nk[len("module._orig_mod."):]
        elif nk.startswith("module."):
            nk = nk[len("module."):]
        elif nk.startswith("_orig_mod."):
            nk = nk[len("_orig_mod."):]

        cleaned[nk] = v

    return cleaned


def set_plot_style():
    plt.rcParams.update({
        "figure.figsize": (9, 6),
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 12,
        "figure.dpi": 150,
        "savefig.dpi": 200,
    })


def pair_summary(prefix: str, sims: np.ndarray) -> dict:
    sims = np.asarray(sims, dtype=np.float32)

    return {
        f"{prefix}_n": int(len(sims)),
        f"{prefix}_cosine_mean": safe_mean(sims),
        f"{prefix}_cosine_std": safe_std(sims),
        f"{prefix}_cosine_median": safe_median(sims),
        f"{prefix}_cosine_p05": float(np.percentile(sims, 5)) if len(sims) > 0 else float("nan"),
        f"{prefix}_cosine_p95": float(np.percentile(sims, 95)) if len(sims) > 0 else float("nan"),
    }


# ----------------------------------------------------------
# Distributed gather helpers
# ----------------------------------------------------------
def gather_numpy_array(local_array: np.ndarray, dtype=torch.float32, device=None):
    """
    Gather numpy array with variable first dimension across ranks.

    Returns:
        rank 0: concatenated numpy array
        other ranks: None
    """
    if not is_dist():
        return local_array

    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())

    local_tensor = torch.as_tensor(local_array, device=device, dtype=dtype)
    local_n = torch.tensor([local_tensor.shape[0]], device=device, dtype=torch.long)

    size_list = [torch.zeros_like(local_n) for _ in range(dist.get_world_size())]
    dist.all_gather(size_list, local_n)

    sizes = [int(x.item()) for x in size_list]
    max_n = max(sizes)

    if local_tensor.ndim == 1:
        padded = torch.zeros((max_n,), device=device, dtype=local_tensor.dtype)
        padded[: local_tensor.shape[0]] = local_tensor
    else:
        feat_shape = tuple(local_tensor.shape[1:])
        padded = torch.zeros(
            (max_n, *feat_shape),
            device=device,
            dtype=local_tensor.dtype,
        )
        padded[: local_tensor.shape[0]] = local_tensor

    gathered = [torch.zeros_like(padded) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, padded)

    if dist.get_rank() != 0:
        return None

    parts = []
    for g, n in zip(gathered, sizes):
        parts.append(g[:n].cpu().numpy())

    return np.concatenate(parts, axis=0)


def gather_object_list(local_obj):
    if not is_dist():
        return local_obj

    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local_obj)

    if dist.get_rank() != 0:
        return None

    flat = []
    for part in gathered:
        if isinstance(part, list):
            flat.extend(part)
        else:
            flat.append(part)

    return flat


def gather_small_dict(local_dict):
    if not is_dist():
        return [local_dict]

    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local_dict)

    return gathered if dist.get_rank() == 0 else None


# ----------------------------------------------------------
# Embedding extraction
# ----------------------------------------------------------
@torch.no_grad()
def compute_local_embeddings(
    dataset,
    model,
    rank,
    world_size,
    device,
):
    sampler = DistributedEvalSampler(dataset, rank=rank, world_size=world_size)

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": BATCH_SIZE,
        "sampler": sampler,
        "shuffle": False,
        "num_workers": NUM_WORKERS,
        "pin_memory": True,
        "drop_last": False,
    }

    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR

    loader = DataLoader(**loader_kwargs)

    model.eval()

    embeddings = []
    hyd_repr = []
    z_multisource_all = []
    z_monthly_all = []
    z_static_all = []

    labels = []
    label_ints = []
    years = []

    basin_meta_all = []
    area_targets = []
    area_preds = []
    overlap_degrees = []

    branch_attention_all = []
    month_attention_sum = None
    month_attention_count = 0

    pbar = tqdm(
        loader,
        desc=f"Rank {rank} embedding",
        disable=(rank != 0),
    )

    for batch in pbar:
        mask = batch["mask"].to(device, non_blocking=True).float()
        monthly_x = batch["monthly_x"].to(device, non_blocking=True).float()
        static_x = batch["static_x"].to(device, non_blocking=True).float()
        basin_meta = batch["basin_meta"].to(device, non_blocking=True).float()

        with torch.amp.autocast(
            device_type="cuda",
            enabled=(USE_AMP and device.type == "cuda"),
        ):
            z, branch_attn, _, _, aux = model(
                monthly_x,
                static_x,
                mask,
                basin_meta,
            )

        z = F.normalize(z.float(), dim=1)

        h = aux.get("h", None)
        if h is not None:
            h = F.normalize(h.float(), dim=1)

        z_multisource = aux.get("z_multisource", None)
        z_monthly = aux.get("z_monthly", None)
        z_static = aux.get("z_static", None)

        if z_multisource is not None:
            z_multisource = F.normalize(z_multisource.float(), dim=1)

        if z_monthly is not None:
            z_monthly = F.normalize(z_monthly.float(), dim=1)

        if z_static is not None:
            z_static = F.normalize(z_static.float(), dim=1)

        embeddings.append(z.detach().cpu().numpy().astype(np.float32))

        if h is not None:
            hyd_repr.append(h.detach().cpu().numpy().astype(np.float32))

        if z_multisource is not None:
            z_multisource_all.append(z_multisource.detach().cpu().numpy().astype(np.float32))

        if z_monthly is not None:
            z_monthly_all.append(z_monthly.detach().cpu().numpy().astype(np.float32))

        if z_static is not None:
            z_static_all.append(z_static.detach().cpu().numpy().astype(np.float32))

        if branch_attn is not None:
            branch_attention_all.append(
                branch_attn.detach().float().squeeze(-1).cpu().numpy().astype(np.float32)
            )

        month_attention = aux.get("month_attention", None)
        if month_attention is not None:
            mw = month_attention.detach().float().squeeze(-1)  # [B, M]
            mw_sum = mw.sum(dim=0).cpu().numpy().astype(np.float64)

            if month_attention_sum is None:
                month_attention_sum = mw_sum
            else:
                month_attention_sum += mw_sum

            month_attention_count += int(mw.shape[0])

        labels.extend(list(batch["site_id"]))
        label_ints.extend(batch["site_id_int"].cpu().numpy().tolist())
        years.extend(batch["year"].cpu().numpy().tolist())

        basin_meta_all.append(batch["basin_meta"].cpu().numpy().astype(np.float32))
        area_targets.extend(batch["log_area_km2_z"].cpu().numpy().tolist())
        area_preds.extend(aux["area_pred"].detach().float().cpu().numpy().tolist())
        overlap_degrees.extend(batch["overlap_degree"].cpu().numpy().tolist())

    emb_dim = int(model.emb_dim)

    if len(embeddings) == 0:
        local_emb = np.zeros((0, emb_dim), dtype=np.float32)
        local_h = np.zeros((0, emb_dim), dtype=np.float32)
        local_z_multisource = np.zeros((0, emb_dim), dtype=np.float32)
        local_z_monthly = np.zeros((0, emb_dim), dtype=np.float32)
        local_z_static = np.zeros((0, emb_dim), dtype=np.float32)
        local_meta = np.zeros((0, len(getattr(dataset, "basin_meta_cols", []))), dtype=np.float32)
        local_branch_attn = np.zeros((0, 2), dtype=np.float32)
    else:
        local_emb = np.concatenate(embeddings, axis=0).astype(np.float32)

        local_h = (
            np.concatenate(hyd_repr, axis=0).astype(np.float32)
            if len(hyd_repr) > 0 else np.zeros_like(local_emb)
        )

        local_z_multisource = (
            np.concatenate(z_multisource_all, axis=0).astype(np.float32)
            if len(z_multisource_all) > 0 else np.zeros_like(local_emb)
        )

        local_z_monthly = (
            np.concatenate(z_monthly_all, axis=0).astype(np.float32)
            if len(z_monthly_all) > 0 else np.zeros_like(local_emb)
        )

        local_z_static = (
            np.concatenate(z_static_all, axis=0).astype(np.float32)
            if len(z_static_all) > 0 else np.zeros_like(local_emb)
        )

        local_meta = np.concatenate(basin_meta_all, axis=0).astype(np.float32)

        if len(branch_attention_all) > 0:
            local_branch_attn = np.concatenate(branch_attention_all, axis=0).astype(np.float32)
        else:
            local_branch_attn = np.zeros((local_emb.shape[0], 2), dtype=np.float32)

    local_month_attention = {
        "sum": month_attention_sum,
        "count": month_attention_count,
    }

    return {
        "embeddings": local_emb,
        "hyd_repr": local_h,
        "z_multisource": local_z_multisource,
        "z_monthly": local_z_monthly,
        "z_static": local_z_static,
        "labels": labels,
        "label_ints": np.array(label_ints, dtype=np.int64),
        "years": np.array(years, dtype=np.int64),
        "basin_meta": local_meta,
        "area_targets": np.array(area_targets, dtype=np.float32),
        "area_preds": np.array(area_preds, dtype=np.float32),
        "overlap_degrees": np.array(overlap_degrees, dtype=np.float32),
        "branch_attention": local_branch_attn,
        "month_attention": local_month_attention,
    }


# ----------------------------------------------------------
# Basin-level averages
# ----------------------------------------------------------
def compute_basin_average_embeddings(embeddings, labels, years):
    basin_to_vecs = defaultdict(list)
    basin_to_years = defaultdict(list)

    for z, lab, yr in zip(embeddings, labels, years):
        basin_to_vecs[lab].append(z)
        basin_to_years[lab].append(int(yr))

    avg_embeddings = []
    avg_labels = []
    year_counts = []

    for lab in sorted(basin_to_vecs.keys()):
        z = np.stack(basin_to_vecs[lab], axis=0)
        z_mean = z.mean(axis=0)
        z_mean = z_mean / np.clip(np.linalg.norm(z_mean), 1e-12, None)

        avg_embeddings.append(z_mean)
        avg_labels.append(lab)
        year_counts.append(len(basin_to_vecs[lab]))

    avg_embeddings = np.stack(avg_embeddings, axis=0).astype(np.float32)
    avg_labels = np.array(avg_labels)
    year_counts = np.array(year_counts)

    return avg_embeddings, avg_labels, year_counts


# ----------------------------------------------------------
# Metrics
# ----------------------------------------------------------
def embedding_stats(emb, prefix="embedding"):
    stats = {
        f"{prefix}_mean": float(emb.mean()),
        f"{prefix}_std": float(emb.std()),
        f"{prefix}_abs_mean": float(np.abs(emb).mean()),
        f"{prefix}_num_samples": int(len(emb)),
        f"{prefix}_dim": int(emb.shape[1]),
    }

    print(f"\n[STATS: {prefix}]")
    for k, v in stats.items():
        print(f"{k}: {v}")

    return stats


def build_positive_pairs(labels, years):
    basin_to_idx = defaultdict(list)

    for i, (lab, yr) in enumerate(zip(labels, years)):
        basin_to_idx[lab].append((i, int(yr)))

    pairs = []

    for _, items in basin_to_idx.items():
        items = sorted(items, key=lambda x: x[1])

        if len(items) < 2:
            continue

        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                ia, ya = items[a]
                ib, yb = items[b]

                if ya != yb:
                    pairs.append((ia, ib))

    return pairs


def alignment_score(embeddings, pos_pairs, alpha=2.0, max_pairs=50000):
    if len(pos_pairs) == 0:
        return float("nan")

    if len(pos_pairs) > max_pairs:
        chosen = RNG.choice(len(pos_pairs), size=max_pairs, replace=False)
        pos_pairs = [pos_pairs[i] for i in chosen]

    i = np.array([p[0] for p in pos_pairs], dtype=np.int64)
    j = np.array([p[1] for p in pos_pairs], dtype=np.int64)

    d = np.linalg.norm(embeddings[i] - embeddings[j], axis=1)
    return float(np.mean(d ** alpha))


def uniformity_score(embeddings, t=2.0, max_points=4000):
    n = len(embeddings)

    if n < 2:
        return float("nan")

    if n > max_points:
        idx = RNG.choice(n, size=max_points, replace=False)
        x = embeddings[idx]
    else:
        x = embeddings

    x = l2_normalize(x)
    sqdist = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)

    iu = np.triu_indices(len(x), k=1)
    vals = np.exp(-t * sqdist[iu])

    return float(np.log(np.mean(vals) + 1e-12))


def distance_analysis(embeddings, labels, max_inter_pairs=MAX_INTER_PAIRS):
    print("\n[INFO] Computing intra/inter distances")

    unique = np.unique(labels)
    intra = []

    for b in unique:
        idx = np.where(labels == b)[0]

        if len(idx) < 2:
            continue

        e = embeddings[idx]
        sim = e @ e.T
        dist = 1.0 - sim

        mask = ~np.eye(len(idx), dtype=bool)
        intra.extend(dist[mask])

    intra = np.array(intra, dtype=np.float32)

    rng = np.random.default_rng(42)
    i = rng.integers(0, len(labels), max_inter_pairs)
    j = rng.integers(0, len(labels), max_inter_pairs)

    valid = labels[i] != labels[j]

    sims = np.sum(embeddings[i[valid]] * embeddings[j[valid]], axis=1)
    inter = 1.0 - sims
    inter = np.array(inter, dtype=np.float32)

    mean_intra = safe_mean(intra)
    mean_inter = safe_mean(inter)

    print("Mean intra distance:", mean_intra)
    print("Mean inter distance:", mean_inter)

    if np.isfinite(mean_intra) and np.isfinite(mean_inter) and mean_intra < mean_inter:
        print("✅ Identity separation working")

    return {
        "mean_intra_distance": mean_intra,
        "std_intra_distance": safe_std(intra),
        "median_intra_distance": safe_median(intra),
        "mean_inter_distance": mean_inter,
        "std_inter_distance": safe_std(inter),
        "median_inter_distance": safe_median(inter),
        "intra_distances": intra,
        "inter_distances": inter,
    }


def retrieval_metrics_chunked(
    embeddings,
    labels,
    ks=(1, 5, 10),
    max_queries=MAX_QUERIES,
    query_chunk_size=128,
):
    print("\n[INFO] Computing same-basin retrieval metrics")

    n = len(embeddings)
    qn = min(max_queries, n)

    query_idx = RNG.choice(n, size=qn, replace=False)

    recalls = {k: [] for k in ks}
    reciprocal_ranks = []

    labels_arr = np.asarray(labels)

    for start in tqdm(range(0, qn, query_chunk_size), desc="retrieval"):
        end = min(start + query_chunk_size, qn)
        qidx = query_idx[start:end]

        sims = embeddings[qidx] @ embeddings.T

        for row, idx in enumerate(qidx):
            sims[row, idx] = -np.inf

        top_order = np.argsort(-sims, axis=1)

        for row, idx in enumerate(qidx):
            target = labels_arr[idx]
            ranked = top_order[row]
            ranked_labels = labels_arr[ranked]

            matches = np.where(ranked_labels == target)[0]

            if len(matches) == 0:
                reciprocal_ranks.append(0.0)
            else:
                reciprocal_ranks.append(1.0 / float(matches[0] + 1))

            for k in ks:
                recalls[k].append(float(np.any(ranked_labels[:k] == target)))

    metrics = {f"Recall@{k}": float(np.mean(recalls[k])) for k in ks}
    metrics["MRR"] = float(np.mean(reciprocal_ranks))

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return metrics


def temporal_consistency(
    embeddings,
    labels,
    years,
    max_negative_pairs=MAX_NEGATIVE_TEMPORAL_PAIRS,
):
    print("\n[INFO] Computing temporal consistency")

    basin_to_idx = defaultdict(list)

    for i, (lab, yr) in enumerate(zip(labels, years)):
        basin_to_idx[lab].append((i, int(yr)))

    same_basin_sims = []
    year_gaps = []

    for _, items in basin_to_idx.items():
        items = sorted(items, key=lambda x: x[1])

        if len(items) < 2:
            continue

        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                i, yi = items[a]
                j, yj = items[b]

                sim = float(np.dot(embeddings[i], embeddings[j]))
                same_basin_sims.append(sim)
                year_gaps.append(abs(yi - yj))

    same_basin_sims = np.array(same_basin_sims, dtype=np.float32)
    year_gaps = np.array(year_gaps, dtype=np.int32)

    rng = np.random.default_rng(0)
    i = rng.integers(0, len(labels), max_negative_pairs)
    j = rng.integers(0, len(labels), max_negative_pairs)

    valid = labels[i] != labels[j]

    diff_basin_sims = np.sum(
        embeddings[i[valid]] * embeddings[j[valid]],
        axis=1,
    )
    diff_basin_sims = np.array(diff_basin_sims, dtype=np.float32)

    metrics = {
        "same_basin_cosine_mean": safe_mean(same_basin_sims),
        "same_basin_cosine_std": safe_std(same_basin_sims),
        "same_basin_cosine_median": safe_median(same_basin_sims),
        "diff_basin_cosine_mean": safe_mean(diff_basin_sims),
        "diff_basin_cosine_std": safe_std(diff_basin_sims),
        "diff_basin_cosine_median": safe_median(diff_basin_sims),
    }

    print("same_basin_cosine_mean:", metrics["same_basin_cosine_mean"])
    print("diff_basin_cosine_mean:", metrics["diff_basin_cosine_mean"])

    return metrics, same_basin_sims, diff_basin_sims, year_gaps


# ----------------------------------------------------------
# Hydrology-aware relation hierarchy
# ----------------------------------------------------------
def build_overlap_pair_list(overlap_adjacency: dict[int, dict[int, float]]):
    pairs = []
    seen = set()

    for a, nbrs in overlap_adjacency.items():
        for b, w in nbrs.items():
            aa = int(a)
            bb = int(b)

            if aa == bb:
                continue

            key = (min(aa, bb), max(aa, bb))

            if key in seen:
                continue

            seen.add(key)
            pairs.append((key[0], key[1], float(w)))

    return pairs


def has_overlap(a: int, b: int, overlap_adjacency) -> bool:
    nbrs = overlap_adjacency.get(int(a), {}) if overlap_adjacency else {}

    if isinstance(nbrs, dict):
        return int(b) in nbrs

    return int(b) in nbrs


def metadata_similarity_numpy(meta_a, meta_b, meta_tau=1.5):
    meta_a = np.asarray(meta_a, dtype=np.float32)
    meta_b = np.asarray(meta_b, dtype=np.float32)

    d = np.linalg.norm(meta_a - meta_b, axis=-1)
    meta_dim = meta_a.shape[-1]
    d = d / max(float(meta_dim) ** 0.5, 1.0)

    sim = np.exp(-(d ** 2) / (2.0 * meta_tau ** 2))
    return np.clip(sim, 0.0, 1.0)


def sample_temporal_pairs(labels_int, years, max_pairs=MAX_RELATION_PAIRS):
    basin_to_idx = defaultdict(list)

    for i, (sid, yr) in enumerate(zip(labels_int, years)):
        basin_to_idx[int(sid)].append((i, int(yr)))

    pairs = []

    for _, items in basin_to_idx.items():
        if len(items) < 2:
            continue

        items = sorted(items, key=lambda x: x[1])

        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                ia, ya = items[a]
                ib, yb = items[b]

                if ya != yb:
                    pairs.append((ia, ib))

    if len(pairs) > max_pairs:
        idx = RNG.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[i] for i in idx]

    return pairs


def sample_overlap_pairs(
    labels_int,
    years,
    overlap_adjacency,
    max_pairs=MAX_RELATION_PAIRS,
):
    if not overlap_adjacency:
        return []

    basin_to_idx = defaultdict(list)

    for i, sid in enumerate(labels_int):
        basin_to_idx[int(sid)].append(i)

    overlap_basin_pairs = build_overlap_pair_list(overlap_adjacency)

    valid_basin_pairs = [
        (a, b, w)
        for a, b, w in overlap_basin_pairs
        if a in basin_to_idx and b in basin_to_idx
    ]

    if len(valid_basin_pairs) == 0:
        return []

    pairs = []
    attempts = 0
    max_attempts = max_pairs * 20

    while len(pairs) < max_pairs and attempts < max_attempts:
        attempts += 1

        a, b, w = valid_basin_pairs[RNG.integers(0, len(valid_basin_pairs))]

        ia = int(RNG.choice(basin_to_idx[a]))
        ib = int(RNG.choice(basin_to_idx[b]))

        pairs.append((ia, ib, w, abs(int(years[ia]) - int(years[ib]))))

    return pairs


def sample_nonoverlap_pairs_by_metadata(
    labels_int,
    basin_meta,
    overlap_adjacency,
    n_candidate_pairs=250000,
    max_pairs=MAX_RELATION_PAIRS,
    meta_tau=1.5,
):
    n = len(labels_int)

    i = RNG.integers(0, n, n_candidate_pairs)
    j = RNG.integers(0, n, n_candidate_pairs)

    valid = (i != j) & (labels_int[i] != labels_int[j])

    if not valid.any():
        return [], [], []

    i = i[valid]
    j = j[valid]

    non_overlap_mask = []

    for a, b in zip(labels_int[i], labels_int[j]):
        non_overlap_mask.append(not has_overlap(int(a), int(b), overlap_adjacency))

    non_overlap_mask = np.array(non_overlap_mask, dtype=bool)

    i = i[non_overlap_mask]
    j = j[non_overlap_mask]

    if len(i) == 0:
        return [], [], []

    meta_sim = metadata_similarity_numpy(
        basin_meta[i],
        basin_meta[j],
        meta_tau=meta_tau,
    )

    rand_n = min(max_pairs, len(i))
    rand_idx = RNG.choice(len(i), size=rand_n, replace=False)

    random_pairs = [(int(i[k]), int(j[k]), float(meta_sim[k])) for k in rand_idx]

    q_hi = np.quantile(meta_sim, 0.90)
    q_lo = np.quantile(meta_sim, 0.10)

    hi_idx = np.where(meta_sim >= q_hi)[0]
    lo_idx = np.where(meta_sim <= q_lo)[0]

    if len(hi_idx) > max_pairs:
        hi_idx = RNG.choice(hi_idx, size=max_pairs, replace=False)

    if len(lo_idx) > max_pairs:
        lo_idx = RNG.choice(lo_idx, size=max_pairs, replace=False)

    similar_pairs = [(int(i[k]), int(j[k]), float(meta_sim[k])) for k in hi_idx]
    dissimilar_pairs = [(int(i[k]), int(j[k]), float(meta_sim[k])) for k in lo_idx]

    return similar_pairs, dissimilar_pairs, random_pairs


def cosine_for_pairs(embeddings, pairs):
    if len(pairs) == 0:
        return np.array([], dtype=np.float32)

    i = np.array([p[0] for p in pairs], dtype=np.int64)
    j = np.array([p[1] for p in pairs], dtype=np.int64)

    return np.sum(embeddings[i] * embeddings[j], axis=1).astype(np.float32)


def relation_hierarchy_analysis(
    embeddings,
    labels_int,
    years,
    basin_meta,
    overlap_adjacency,
    meta_tau=1.5,
):
    print("\n[INFO] Computing hydrology-aware relation hierarchy")

    labels_int = np.asarray(labels_int, dtype=np.int64)
    years = np.asarray(years, dtype=np.int64)
    basin_meta = np.asarray(basin_meta, dtype=np.float32)

    temporal_pairs = sample_temporal_pairs(
        labels_int,
        years,
        max_pairs=MAX_RELATION_PAIRS,
    )

    overlap_pairs = sample_overlap_pairs(
        labels_int,
        years,
        overlap_adjacency,
        max_pairs=MAX_RELATION_PAIRS,
    )

    meta_similar_pairs, meta_dissimilar_pairs, random_nonoverlap_pairs = (
        sample_nonoverlap_pairs_by_metadata(
            labels_int=labels_int,
            basin_meta=basin_meta,
            overlap_adjacency=overlap_adjacency,
            max_pairs=MAX_RELATION_PAIRS,
            meta_tau=meta_tau,
        )
    )

    temporal_sims = cosine_for_pairs(embeddings, temporal_pairs)
    overlap_sims = cosine_for_pairs(embeddings, overlap_pairs)
    meta_similar_sims = cosine_for_pairs(embeddings, meta_similar_pairs)
    meta_dissimilar_sims = cosine_for_pairs(embeddings, meta_dissimilar_pairs)
    random_nonoverlap_sims = cosine_for_pairs(embeddings, random_nonoverlap_pairs)

    metrics = {}
    metrics.update(pair_summary("relation_temporal_positive", temporal_sims))
    metrics.update(pair_summary("relation_overlap_positive", overlap_sims))
    metrics.update(pair_summary("relation_meta_similar_nonoverlap", meta_similar_sims))
    metrics.update(pair_summary("relation_meta_dissimilar_nonoverlap", meta_dissimilar_sims))
    metrics.update(pair_summary("relation_random_nonoverlap_negative", random_nonoverlap_sims))

    if len(overlap_pairs) > 0:
        overlap_strengths = np.array([p[2] for p in overlap_pairs], dtype=np.float32)
        overlap_year_gaps = np.array([p[3] for p in overlap_pairs], dtype=np.float32)

        metrics["relation_overlap_strength_mean"] = safe_mean(overlap_strengths)
        metrics["relation_overlap_strength_std"] = safe_std(overlap_strengths)
        metrics["relation_overlap_year_gap_mean"] = safe_mean(overlap_year_gaps)
        metrics["relation_overlap_cosine_vs_strength_corr"] = finite_corr(
            overlap_sims,
            overlap_strengths,
        )
    else:
        metrics["relation_overlap_strength_mean"] = float("nan")
        metrics["relation_overlap_strength_std"] = float("nan")
        metrics["relation_overlap_year_gap_mean"] = float("nan")
        metrics["relation_overlap_cosine_vs_strength_corr"] = float("nan")

    if len(meta_similar_pairs) > 0:
        meta_sim_hi = np.array([p[2] for p in meta_similar_pairs], dtype=np.float32)
        metrics["relation_meta_similar_metadata_similarity_mean"] = safe_mean(meta_sim_hi)

    if len(meta_dissimilar_pairs) > 0:
        meta_sim_lo = np.array([p[2] for p in meta_dissimilar_pairs], dtype=np.float32)
        metrics["relation_meta_dissimilar_metadata_similarity_mean"] = safe_mean(meta_sim_lo)

    temporal_mean = metrics["relation_temporal_positive_cosine_mean"]
    overlap_mean = metrics["relation_overlap_positive_cosine_mean"]
    meta_sim_mean = metrics["relation_meta_similar_nonoverlap_cosine_mean"]
    random_mean = metrics["relation_random_nonoverlap_negative_cosine_mean"]
    meta_dis_mean = metrics["relation_meta_dissimilar_nonoverlap_cosine_mean"]

    metrics["hierarchy_temporal_gt_overlap"] = bool(
        np.isfinite(temporal_mean)
        and np.isfinite(overlap_mean)
        and temporal_mean > overlap_mean
    )

    metrics["hierarchy_overlap_gt_random_negative"] = bool(
        np.isfinite(overlap_mean)
        and np.isfinite(random_mean)
        and overlap_mean > random_mean
    )

    metrics["hierarchy_meta_similar_gt_random_negative"] = bool(
        np.isfinite(meta_sim_mean)
        and np.isfinite(random_mean)
        and meta_sim_mean > random_mean
    )

    metrics["hierarchy_meta_similar_gt_meta_dissimilar"] = bool(
        np.isfinite(meta_sim_mean)
        and np.isfinite(meta_dis_mean)
        and meta_sim_mean > meta_dis_mean
    )

    print("Temporal+ cosine mean          :", temporal_mean)
    print("Overlap+ cosine mean           :", overlap_mean)
    print("Metadata-similar non-overlap   :", meta_sim_mean)
    print("Random non-overlap negative    :", random_mean)
    print("Metadata-dissimilar non-overlap:", meta_dis_mean)

    arrays = {
        "temporal_sims": temporal_sims,
        "overlap_sims": overlap_sims,
        "meta_similar_sims": meta_similar_sims,
        "meta_dissimilar_sims": meta_dissimilar_sims,
        "random_nonoverlap_sims": random_nonoverlap_sims,
    }

    return metrics, arrays


def area_prediction_metrics(area_pred, area_target):
    area_pred = np.asarray(area_pred, dtype=np.float32)
    area_target = np.asarray(area_target, dtype=np.float32)

    diff = area_pred - area_target

    metrics = {
        "area_pred_mean": safe_mean(area_pred),
        "area_target_mean": safe_mean(area_target),
        "area_aux_mae": safe_mean(np.abs(diff)),
        "area_aux_rmse": float(np.sqrt(np.mean(diff ** 2))) if len(diff) > 0 else float("nan"),
        "area_aux_corr": finite_corr(area_pred, area_target),
    }

    print("\n[INFO] Area auxiliary diagnostics")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    return metrics


# ----------------------------------------------------------
# Attention summaries
# ----------------------------------------------------------
def summarize_branch_attention(branch_attention):
    metrics = {}

    if branch_attention is None or len(branch_attention) == 0:
        return metrics

    mean_attn = np.mean(branch_attention, axis=0)
    std_attn = np.std(branch_attention, axis=0)

    names = ["monthly", "static"]

    print("\n[INFO] Branch attention summary")

    for i in range(branch_attention.shape[1]):
        name = names[i] if i < len(names) else f"branch_{i}"
        metrics[f"attention_branch_{name}_mean"] = float(mean_attn[i])
        metrics[f"attention_branch_{name}_std"] = float(std_attn[i])
        print(f"{name:12s} mean={mean_attn[i]:.4f} std={std_attn[i]:.4f}")

    return metrics


def summarize_month_attention(month_attention_gathered):
    metrics = {}

    if month_attention_gathered is None:
        return metrics

    total_sum = None
    total_count = 0

    for rank_dict in month_attention_gathered:
        if rank_dict is None:
            continue

        s = rank_dict.get("sum")
        c = int(rank_dict.get("count", 0))

        if s is None or c == 0:
            continue

        s = np.asarray(s, dtype=np.float64)

        if total_sum is None:
            total_sum = s.copy()
        else:
            total_sum += s

        total_count += c

    if total_sum is None or total_count == 0:
        return metrics

    mean_month = total_sum / max(total_count, 1)

    print("\n[INFO] Month attention summary")
    formatted = " ".join([f"{m + 1}:{mean_month[m]:.3f}" for m in range(len(mean_month))])
    print(formatted)

    for m_idx, val in enumerate(mean_month, start=1):
        metrics[f"attention_month_{m_idx:02d}_mean"] = float(val)

    return metrics


# ----------------------------------------------------------
# Projection and clustering
# ----------------------------------------------------------
def compute_2d_projection(embeddings, max_points=MAX_PROJECTION_POINTS):
    n = len(embeddings)

    if n > max_points:
        idx = RNG.choice(n, size=max_points, replace=False)
        x = embeddings[idx]
    else:
        idx = np.arange(n)
        x = embeddings

    x = x.astype(np.float32)

    try:
        import umap

        reducer = umap.UMAP(
            n_neighbors=25,
            min_dist=0.15,
            metric="cosine",
            random_state=42,
        )
        xy = reducer.fit_transform(x)
        method = "UMAP"
        return idx, xy, method
    except Exception:
        pass

    x_centered = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x_centered, full_matrices=False)
    xy = x_centered @ vt[:2].T

    return idx, xy, "PCA"


def kmeans_numpy(x, k=8, n_iter=50):
    n = len(x)

    if n < k:
        return np.zeros(n, dtype=np.int64), x.copy()

    init_idx = RNG.choice(n, size=k, replace=False)
    centers = x[init_idx].copy()

    for _ in range(n_iter):
        d2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(d2, axis=1)

        new_centers = []

        for c in range(k):
            pts = x[labels == c]

            if len(pts) == 0:
                new_centers.append(centers[c])
            else:
                new_centers.append(pts.mean(axis=0))

        new_centers = np.stack(new_centers, axis=0)

        if np.allclose(new_centers, centers):
            break

        centers = new_centers

    return labels, centers


# ----------------------------------------------------------
# Plotting
# ----------------------------------------------------------
def plot_distance_histograms(intra, inter, out_path):
    fig, ax = plt.subplots(constrained_layout=True)

    if len(intra) > 0:
        ax.hist(intra, bins=60, alpha=0.7, density=True, label="Same basin")

    if len(inter) > 0:
        ax.hist(inter, bins=60, alpha=0.7, density=True, label="Different basin")

    ax.set_title("Cosine Distance Distribution")
    ax.set_xlabel("Cosine distance = 1 - cosine similarity")
    ax.set_ylabel("Density")
    ax.legend()

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_recall_bar(metrics, out_path):
    recall_keys = [k for k in metrics.keys() if k.startswith("Recall@")]
    recall_keys = sorted(recall_keys, key=lambda x: int(x.split("@")[1]))

    vals = [metrics[k] for k in recall_keys]

    fig, ax = plt.subplots(constrained_layout=True)
    ax.bar(recall_keys, vals)

    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center")

    ax.set_ylim(0, min(1.05, max(vals) + 0.12 if len(vals) else 1.0))
    ax.set_title("Same-basin Retrieval Performance")
    ax.set_ylabel("Score")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_similarity_histograms(same_sims, diff_sims, out_path):
    fig, ax = plt.subplots(constrained_layout=True)

    if len(same_sims) > 0:
        ax.hist(
            same_sims,
            bins=60,
            alpha=0.7,
            density=True,
            label="Same basin across years",
        )

    if len(diff_sims) > 0:
        ax.hist(
            diff_sims,
            bins=60,
            alpha=0.7,
            density=True,
            label="Different basins",
        )

    ax.set_title("Cosine Similarity Distribution")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.legend()

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_similarity_vs_year_gap(same_sims, year_gaps, out_path):
    if len(same_sims) == 0 or len(year_gaps) == 0:
        return

    unique_gaps = np.unique(year_gaps)
    mean_vals = []
    std_vals = []

    for g in unique_gaps:
        vals = same_sims[year_gaps == g]
        mean_vals.append(np.mean(vals))
        std_vals.append(np.std(vals))

    mean_vals = np.array(mean_vals)
    std_vals = np.array(std_vals)

    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(unique_gaps, mean_vals, marker="o")
    ax.fill_between(
        unique_gaps,
        mean_vals - std_vals,
        mean_vals + std_vals,
        alpha=0.2,
    )

    ax.set_title("Temporal Consistency by Year Gap")
    ax.set_xlabel("Absolute year gap")
    ax.set_ylabel("Mean cosine similarity")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_relation_hierarchy(relation_arrays, out_path):
    data = [
        relation_arrays.get("temporal_sims", np.array([])),
        relation_arrays.get("overlap_sims", np.array([])),
        relation_arrays.get("meta_similar_sims", np.array([])),
        relation_arrays.get("random_nonoverlap_sims", np.array([])),
        relation_arrays.get("meta_dissimilar_sims", np.array([])),
    ]

    labels = [
        "Temporal+",
        "Overlap+",
        "Meta-similar\nnon-overlap",
        "Random\nnon-overlap",
        "Meta-dissimilar\nnon-overlap",
    ]

    data = [d[np.isfinite(d)] for d in data]

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_title("Hydrology-aware relation hierarchy")
    ax.set_ylabel("Cosine similarity")
    ax.axhline(0.0, linewidth=1, alpha=0.5)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_area_prediction(area_target, area_pred, out_path):
    fig, ax = plt.subplots(constrained_layout=True)

    ax.scatter(area_target, area_pred, s=6, alpha=0.35)

    lo = float(np.nanmin([area_target.min(), area_pred.min()]))
    hi = float(np.nanmax([area_target.max(), area_pred.max()]))

    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    ax.set_title("Auxiliary area prediction")
    ax.set_xlabel("Target log_area_km2_z")
    ax.set_ylabel("Predicted log_area_km2_z")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_projection(xy, cluster_labels, method_name, out_path, title_suffix):
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)

    sc = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=cluster_labels,
        s=10,
        alpha=0.8,
    )

    ax.set_title(f"{method_name} projection ({title_suffix})")
    ax.set_xlabel(f"{method_name}-1")
    ax.set_ylabel(f"{method_name}-2")

    fig.colorbar(sc, ax=ax, label="Cluster")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_year_count_hist(year_counts, out_path):
    fig, ax = plt.subplots(constrained_layout=True)

    ax.hist(
        year_counts,
        bins=min(40, len(np.unique(year_counts))),
        alpha=0.8,
    )

    ax.set_title("Number of yearly samples per basin")
    ax.set_xlabel("Years available per basin")
    ax.set_ylabel("Count")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_branch_attention(branch_attention, out_path):
    if branch_attention is None or len(branch_attention) == 0:
        return

    mean_attn = np.mean(branch_attention, axis=0)
    labels = ["monthly", "static"][: len(mean_attn)]

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.bar(labels, mean_attn)
    ax.set_ylabel("Mean attention")
    ax.set_title("Branch attention")

    for i, v in enumerate(mean_attn):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center")

    ax.set_ylim(0, min(1.05, max(mean_attn) + 0.15))

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_month_attention(month_attention_gathered, out_path):
    if month_attention_gathered is None:
        return

    total_sum = None
    total_count = 0

    for rank_dict in month_attention_gathered:
        if rank_dict is None:
            continue

        s = rank_dict.get("sum")
        c = int(rank_dict.get("count", 0))

        if s is None or c == 0:
            continue

        s = np.asarray(s, dtype=np.float64)

        if total_sum is None:
            total_sum = s.copy()
        else:
            total_sum += s

        total_count += c

    if total_sum is None or total_count == 0:
        return

    mean_month = total_sum / max(total_count, 1)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    months = np.arange(1, len(mean_month) + 1)
    ax.plot(months, mean_month, marker="o")
    ax.set_xticks(months)
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean attention")
    ax.set_title("Month attention")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------
# Save outputs
# ----------------------------------------------------------
def save_embeddings(
    emb,
    h,
    z_multisource,
    z_monthly,
    z_static,
    labels,
    label_ints,
    years,
    basin_meta,
    area_target,
    area_pred,
    overlap_degree,
    branch_attention,
    avg_emb,
    avg_labels,
    year_counts,
):
    np.savez_compressed(
        OUT_DIR / "basin_embeddings_full.npz",
        embeddings=emb,
        hydrologic_repr=h,
        z_multisource=z_multisource,
        z_monthly=z_monthly,
        z_static=z_static,
        labels=np.array(labels),
        label_ints=label_ints,
        years=years,
        basin_meta=basin_meta,
        area_target=area_target,
        area_pred=area_pred,
        overlap_degree=overlap_degree,
        branch_attention=branch_attention,
        avg_embeddings=avg_emb,
        avg_labels=avg_labels,
        year_counts=year_counts,
    )

    print(f"\nSaved embeddings -> {OUT_DIR / 'basin_embeddings_full.npz'}")


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    local_rank, global_rank, world_size, use_ddp = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = global_rank == 0

    torch.manual_seed(SEED + global_rank)
    np.random.seed(SEED + global_rank)

    if is_main:
        OUT_DIR.mkdir(exist_ok=True, parents=True)
        PLOT_DIR.mkdir(exist_ok=True, parents=True)
        set_plot_style()
        print(f"[INFO] Running grouped/pre-stacked evaluation on {world_size} GPU(s)")
        print(f"[INFO] CHECKPOINT: {CHECKPOINT}")

    print(f"[Rank {global_rank}] Loading checkpoint")
    checkpoint = torch.load(CHECKPOINT, map_location=device)

    if is_main:
        print("checkpoint epoch:", checkpoint.get("epoch"))
        print("checkpoint encoder_name:", checkpoint.get("encoder_name", "unknown"))
        print("checkpoint dataset_name:", checkpoint.get("dataset_name", "unknown"))
        print("checkpoint loss_name:", checkpoint.get("loss_name", "unknown"))

    basin_meta_cols = checkpoint.get("basin_meta_cols", DEFAULT_BASIN_META_COLS)

    paths = checkpoint.get("paths", {})
    grouped_zarr_path = paths.get("grouped_zarr_path", GROUPED_ZARR_PATH)
    index_parquet = paths.get("index_parquet", INDEX_PARQUET)
    mask_zarr_path = paths.get("mask_zarr_path", MASK_ZARR_PATH)
    basin_metadata = paths.get("basin_metadata", BASIN_METADATA)
    overlap_pairs_parquet = paths.get("overlap_pairs_parquet", OVERLAP_PAIRS_PARQUET)

    emb_dim = int(checkpoint.get("emb_dim", 64))
    months_per_year = int(checkpoint.get("months_per_year", 12))
    monthly_in_channels = int(checkpoint.get("monthly_in_channels"))
    static_in_channels = int(checkpoint.get("static_in_channels"))
    predict_month_aux = bool(checkpoint.get("predict_month_aux", False))

    monthly_variables = checkpoint.get("monthly_variables", [])
    nonmonthly_variables = checkpoint.get("nonmonthly_variables", [])

    if is_main:
        print(f"[INFO] grouped_zarr_path: {grouped_zarr_path}")
        print(f"[INFO] emb_dim: {emb_dim}")
        print(f"[INFO] months_per_year: {months_per_year}")
        print(f"[INFO] monthly_in_channels: {monthly_in_channels}")
        print(f"[INFO] static_in_channels: {static_in_channels}")
        print(f"[INFO] predict_month_aux: {predict_month_aux}")
        print(f"[INFO] monthly_variables: {monthly_variables}")
        print(f"[INFO] nonmonthly_variables: {nonmonthly_variables}")
        print(f"[INFO] basin_meta_cols: {basin_meta_cols}")

    print(f"[Rank {global_rank}] Loading grouped/pre-stacked dataset")

    dataset = Basin2VecGroupedZarrDataset(
        index_parquet=index_parquet,
        grouped_zarr_path=grouped_zarr_path,
        basin_metadata_path=basin_metadata,
        basin_meta_cols=basin_meta_cols,
        mask_zarr_path=mask_zarr_path,
        require_done=True,
    )

    if is_main:
        example = dataset[0]
        print(f"[INFO] Dataset monthly variables: {dataset.monthly_variables}")
        print(f"[INFO] Dataset nonmonthly variables: {dataset.nonmonthly_variables}")
        print(f"[INFO] monthly_x example shape: {tuple(example['monthly_x'].shape)}")
        print(f"[INFO] static_x example shape: {tuple(example['static_x'].shape)}")
        print(f"[INFO] Dataset monthly_in_channels: {dataset.monthly_in_channels}")
        print(f"[INFO] Dataset static_in_channels: {dataset.static_in_channels}")

    if int(dataset.monthly_in_channels) != monthly_in_channels:
        raise ValueError(
            f"Checkpoint monthly_in_channels={monthly_in_channels}, "
            f"dataset monthly_in_channels={dataset.monthly_in_channels}"
        )

    if int(dataset.static_in_channels) != static_in_channels:
        raise ValueError(
            f"Checkpoint static_in_channels={static_in_channels}, "
            f"dataset static_in_channels={dataset.static_in_channels}"
        )

    model = GroupedBasinEncoderMonthlyV2(
        monthly_in_channels=monthly_in_channels,
        static_in_channels=static_in_channels,
        emb_dim=emb_dim,
        meta_dim=len(basin_meta_cols),
        max_months=months_per_year,
        temporal_layers=1,
        temporal_heads=4,
        dropout=0.05,
        metadata_dropout=0.15,
        predict_month_aux=predict_month_aux,
    ).to(device)

    state_dict = clean_state_dict_keys(checkpoint["model_state_dict"])

    try:
        missing, unexpected = model.load_state_dict(
            state_dict,
            strict=STRICT_LOAD,
        )
    except RuntimeError as e:
        if is_main:
            print("\n[ERROR] Could not load checkpoint strictly.")
            print("This usually means attention_encoder_monthly.py does not match")
            print("GroupedBasinEncoderMonthlyV2 used during training.")
            print("\nOriginal error:")
            print(e)
        cleanup_ddp()
        raise

    if is_main:
        print(f"[INFO] Missing state dict keys: {len(missing)}")
        print(f"[INFO] Unexpected state dict keys: {len(unexpected)}")
        if len(missing) > 0:
            print("[WARN] First missing keys:", missing[:10])
        if len(unexpected) > 0:
            print("[WARN] First unexpected keys:", unexpected[:10])

    model.eval()

    overlap_adjacency, overlap_pair_count = load_overlap_adjacency_weighted(
        overlap_pairs_parquet
    )

    if is_main:
        print(f"[INFO] Loaded weighted overlap pairs: {overlap_pair_count}")

    barrier(local_rank)

    local = compute_local_embeddings(
        dataset=dataset,
        model=model,
        rank=global_rank,
        world_size=world_size,
        device=device,
    )

    barrier(local_rank)

    if is_main:
        print("[INFO] Gathering embeddings and metadata to rank 0")

    emb = gather_numpy_array(local["embeddings"], dtype=torch.float32, device=device)
    h = gather_numpy_array(local["hyd_repr"], dtype=torch.float32, device=device)
    z_multisource = gather_numpy_array(local["z_multisource"], dtype=torch.float32, device=device)
    z_monthly = gather_numpy_array(local["z_monthly"], dtype=torch.float32, device=device)
    z_static = gather_numpy_array(local["z_static"], dtype=torch.float32, device=device)

    label_ints = gather_numpy_array(local["label_ints"], dtype=torch.long, device=device)
    years = gather_numpy_array(local["years"], dtype=torch.long, device=device)
    basin_meta = gather_numpy_array(local["basin_meta"], dtype=torch.float32, device=device)
    area_target = gather_numpy_array(local["area_targets"], dtype=torch.float32, device=device)
    area_pred = gather_numpy_array(local["area_preds"], dtype=torch.float32, device=device)
    overlap_degree = gather_numpy_array(local["overlap_degrees"], dtype=torch.float32, device=device)
    branch_attention = gather_numpy_array(local["branch_attention"], dtype=torch.float32, device=device)

    labels = gather_object_list(local["labels"])
    month_attention_gathered = gather_small_dict(local["month_attention"])

    barrier(local_rank)

    if not is_main:
        cleanup_ddp()
        return

    emb = l2_normalize(emb.astype(np.float32))
    h = l2_normalize(h.astype(np.float32))
    z_multisource = l2_normalize(z_multisource.astype(np.float32))
    z_monthly = l2_normalize(z_monthly.astype(np.float32))
    z_static = l2_normalize(z_static.astype(np.float32))

    label_ints = label_ints.astype(np.int64)
    years = years.astype(np.int64)
    basin_meta = basin_meta.astype(np.float32)
    area_target = area_target.astype(np.float32)
    area_pred = area_pred.astype(np.float32)
    overlap_degree = overlap_degree.astype(np.float32)
    branch_attention = branch_attention.astype(np.float32)

    labels = np.array(labels)

    print(f"[INFO] Final gathered samples: {len(emb)}")
    print(f"[INFO] Unique basins: {len(np.unique(labels))}")
    print(f"[INFO] Basin metadata matrix: {basin_meta.shape}")
    print(f"[INFO] Embedding matrix: {emb.shape}")

    avg_emb, avg_labels, year_counts = compute_basin_average_embeddings(
        emb,
        labels,
        years,
    )

    # ------------------------------------------------------
    # Metrics
    # ------------------------------------------------------
    stats = embedding_stats(emb, prefix="embedding")
    h_stats = embedding_stats(h, prefix="hydrologic_repr")
    multisource_stats = embedding_stats(z_multisource, prefix="z_multisource")
    monthly_stats = embedding_stats(z_monthly, prefix="z_monthly")
    static_stats = embedding_stats(z_static, prefix="z_static")

    dist_metrics = distance_analysis(emb, labels)

    retrieval = retrieval_metrics_chunked(
        emb,
        labels,
        ks=(1, 5, 10),
        max_queries=MAX_QUERIES,
    )

    pos_pairs = build_positive_pairs(labels, years)

    align = alignment_score(
        emb,
        pos_pairs,
        alpha=2.0,
        max_pairs=50000,
    )

    uni = uniformity_score(
        emb,
        t=2.0,
        max_points=4000,
    )

    print("\n[INFO] Alignment / Uniformity")
    print("alignment:", align)
    print("uniformity:", uni)

    temp_metrics, same_sims, diff_sims, year_gaps = temporal_consistency(
        emb,
        labels,
        years,
        max_negative_pairs=MAX_NEGATIVE_TEMPORAL_PAIRS,
    )

    relation_metrics, relation_arrays = relation_hierarchy_analysis(
        embeddings=emb,
        labels_int=label_ints,
        years=years,
        basin_meta=basin_meta,
        overlap_adjacency=overlap_adjacency,
        meta_tau=checkpoint.get("loss_config", {}).get("meta_tau", 1.5),
    )

    area_metrics = area_prediction_metrics(area_pred, area_target)

    branch_attention_metrics = summarize_branch_attention(branch_attention)
    month_attention_metrics = summarize_month_attention(month_attention_gathered)

    summary = {}
    summary.update(stats)
    summary.update(h_stats)
    summary.update(multisource_stats)
    summary.update(monthly_stats)
    summary.update(static_stats)

    summary.update({
        "alignment": align,
        "uniformity": uni,
    })

    summary.update({
        k: v
        for k, v in dist_metrics.items()
        if not isinstance(v, np.ndarray)
    })

    summary.update(retrieval)
    summary.update(temp_metrics)
    summary.update(relation_metrics)
    summary.update(area_metrics)
    summary.update(branch_attention_metrics)
    summary.update(month_attention_metrics)

    summary["num_unique_basins"] = int(len(np.unique(labels)))
    summary["num_samples"] = int(len(emb))
    summary["num_positive_pairs_used_for_alignment"] = int(len(pos_pairs))
    summary["num_overlap_pairs_loaded"] = int(overlap_pair_count)
    summary["basin_metadata_num_columns"] = int(len(basin_meta_cols))
    summary["basin_metadata_columns"] = ",".join(basin_meta_cols)
    summary["checkpoint"] = str(CHECKPOINT)
    summary["checkpoint_epoch"] = int(checkpoint.get("epoch", -1))
    summary["encoder_name"] = str(checkpoint.get("encoder_name", "unknown"))
    summary["dataset_name"] = str(checkpoint.get("dataset_name", "unknown"))
    summary["monthly_variables"] = ",".join([str(v) for v in monthly_variables])
    summary["nonmonthly_variables"] = ",".join([str(v) for v in nonmonthly_variables])
    summary["monthly_in_channels"] = int(monthly_in_channels)
    summary["static_in_channels"] = int(static_in_channels)

    # ------------------------------------------------------
    # Projections and plots
    # ------------------------------------------------------
    print("\n[INFO] Building 2D projections")

    proj_idx, xy, proj_name = compute_2d_projection(
        emb,
        max_points=MAX_PROJECTION_POINTS,
    )

    proj_cluster_labels, _ = kmeans_numpy(
        emb[proj_idx],
        k=8,
        n_iter=60,
    )

    plot_projection(
        xy,
        proj_cluster_labels,
        proj_name,
        PLOT_DIR / "projection_sample_level_z.png",
        title_suffix="sample-level z",
    )

    avg_proj_idx, avg_xy, avg_proj_name = compute_2d_projection(
        avg_emb,
        max_points=MAX_PROJECTION_POINTS,
    )

    avg_cluster_labels, _ = kmeans_numpy(
        avg_emb[avg_proj_idx],
        k=8,
        n_iter=60,
    )

    plot_projection(
        avg_xy,
        avg_cluster_labels,
        avg_proj_name,
        PLOT_DIR / "projection_basin_level_avg_z.png",
        title_suffix="basin-level averaged z",
    )

    h_proj_idx, h_xy, h_proj_name = compute_2d_projection(
        h,
        max_points=MAX_PROJECTION_POINTS,
    )

    h_cluster_labels, _ = kmeans_numpy(
        h[h_proj_idx],
        k=8,
        n_iter=60,
    )

    plot_projection(
        h_xy,
        h_cluster_labels,
        h_proj_name,
        PLOT_DIR / "projection_hydrologic_repr_h.png",
        title_suffix="hydrologic representation h",
    )

    plot_distance_histograms(
        dist_metrics["intra_distances"],
        dist_metrics["inter_distances"],
        PLOT_DIR / "distance_histograms.png",
    )

    plot_recall_bar(
        retrieval,
        PLOT_DIR / "retrieval_recall_bar.png",
    )

    plot_similarity_histograms(
        same_sims,
        diff_sims,
        PLOT_DIR / "temporal_similarity_histograms.png",
    )

    plot_similarity_vs_year_gap(
        same_sims,
        year_gaps,
        PLOT_DIR / "temporal_similarity_vs_year_gap.png",
    )

    plot_relation_hierarchy(
        relation_arrays,
        PLOT_DIR / "relation_hierarchy_boxplot.png",
    )

    plot_area_prediction(
        area_target,
        area_pred,
        PLOT_DIR / "area_aux_prediction.png",
    )

    plot_year_count_hist(
        year_counts,
        PLOT_DIR / "year_count_per_basin.png",
    )

    plot_branch_attention(
        branch_attention,
        PLOT_DIR / "branch_attention_mean.png",
    )

    plot_month_attention(
        month_attention_gathered,
        PLOT_DIR / "month_attention_mean.png",
    )

    # ------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------
    save_embeddings(
        emb=emb,
        h=h,
        z_multisource=z_multisource,
        z_monthly=z_monthly,
        z_static=z_static,
        labels=labels,
        label_ints=label_ints,
        years=years,
        basin_meta=basin_meta,
        area_target=area_target,
        area_pred=area_pred,
        overlap_degree=overlap_degree,
        branch_attention=branch_attention,
        avg_emb=avg_emb,
        avg_labels=avg_labels,
        year_counts=year_counts,
    )

    np.savez_compressed(
        OUT_DIR / "relation_hierarchy_arrays.npz",
        **relation_arrays,
    )

    np.savez_compressed(
        OUT_DIR / "attention_arrays.npz",
        branch_attention=branch_attention,
        month_attention_sum=np.array(
            month_attention_gathered[0]["sum"]
            if month_attention_gathered and month_attention_gathered[0].get("sum") is not None
            else [],
            dtype=np.float32,
        ),
        monthly_variables=np.array(monthly_variables),
        nonmonthly_variables=np.array(nonmonthly_variables),
    )

    save_json_like_txt(
        OUT_DIR / "metrics_summary.txt",
        summary,
    )

    print("\n[INFO] Saved metrics ->", OUT_DIR / "metrics_summary.txt")
    print("[INFO] Saved embeddings ->", OUT_DIR / "basin_embeddings_full.npz")
    print("[INFO] Saved relation arrays ->", OUT_DIR / "relation_hierarchy_arrays.npz")
    print("[INFO] Saved attention arrays ->", OUT_DIR / "attention_arrays.npz")
    print("[INFO] Saved plots ->", PLOT_DIR)
    print("[DONE]")

    cleanup_ddp()


if __name__ == "__main__":
    main()