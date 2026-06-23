# emb/src/contrastive_loss.py

from __future__ import annotations

from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# Overlap helpers
# ------------------------------------------------------------
def load_overlap_adjacency_weighted(
    path: str,
    weight_col_candidates=("overlap_score", "overlap_strength", "score", "weight"),
) -> tuple[dict[int, dict[int, float]], int]:
    """
    Read overlap pair parquet/CSV and build weighted adjacency:
        a -> {b1: w1, b2: w2, ...}
        b -> {a1: w1, a2: w2, ...}

    Expected columns:
        site_id_int_a, site_id_int_b
    Optional weight column:
        overlap_score, overlap_strength, score, or weight

    If no weight column exists, all pair weights are set to 1.0.
    """
    p = Path(path)
    if not p.exists():
        return {}, 0

    if p.suffix.lower() in [".parquet", ".pq"]:
        df = pd.read_parquet(p)
    elif p.suffix.lower() in [".csv", ".txt"]:
        df = pd.read_csv(p)
    else:
        raise ValueError(f"Unsupported overlap file format: {p.suffix}")

    required = {"site_id_int_a", "site_id_int_b"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    weight_col = None
    for c in weight_col_candidates:
        if c in df.columns:
            weight_col = c
            break

    adjacency: dict[int, dict[int, float]] = {}
    pair_count = 0

    for _, row in df.iterrows():
        a = int(row["site_id_int_a"])
        b = int(row["site_id_int_b"])
        if a == b:
            continue

        if weight_col is None:
            w = 1.0
        else:
            w = float(row[weight_col])
            if not np.isfinite(w):
                w = 0.0

        w = float(np.clip(w, 0.0, 1.0))
        if w <= 0:
            continue

        adjacency.setdefault(a, {})[b] = max(adjacency.setdefault(a, {}).get(b, 0.0), w)
        adjacency.setdefault(b, {})[a] = max(adjacency.setdefault(b, {}).get(a, 0.0), w)
        pair_count += 1

    return adjacency, pair_count


def build_overlap_strength_matrix(
    site_tensor: torch.Tensor,
    overlap_adjacency: dict[int, dict[int, float]] | dict[int, set[int]] | None,
    eye: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Build [N, N] overlap-strength matrix from a weighted or binary adjacency.
    """
    device = site_tensor.device
    N = site_tensor.numel()
    strength = torch.zeros((N, N), device=device, dtype=torch.float32)

    if not overlap_adjacency:
        return strength

    site_list = [int(x) for x in site_tensor.detach().cpu().tolist()]

    positions: dict[int, list[int]] = defaultdict(list)
    for i, sid in enumerate(site_list):
        positions[sid].append(i)

    for a, nbrs in overlap_adjacency.items():
        a = int(a)
        if a not in positions:
            continue

        if isinstance(nbrs, dict):
            items = nbrs.items()
        else:
            items = [(b, 1.0) for b in nbrs]

        for b, w in items:
            b = int(b)
            if b not in positions:
                continue

            rows = torch.tensor(positions[a], device=device, dtype=torch.long)
            cols = torch.tensor(positions[b], device=device, dtype=torch.long)
            strength[rows[:, None], cols[None, :]] = float(w)

    if eye is not None:
        strength = strength.masked_fill(eye, 0.0)

    return strength.clamp(0.0, 1.0)


# ------------------------------------------------------------
# Loss
# ------------------------------------------------------------
class HydrologyAwareContrastiveLossV4(nn.Module):
    """
    Metadata-aware hierarchical contrastive loss.

    Positive relations:
        1. same basin + same year                     -> hard positive
        2. same basin + different year                -> temporal positive
        3. different basin + spatial overlap          -> overlap positive

    Metadata use:
        1. Reduces false-negative pressure between metadata-similar basins.
        2. Uses metadata-aware negative margin, so physically dissimilar basins
           are penalized if they become too similar.

    This loss expects z1 and z2 from two augmented views of the same batch.
    """

    def __init__(
        self,
        temperature: float = 0.15,
        temporal_weight: float = 0.35,
        overlap_weight: float = 0.12,
        year_tau: float = 8.0,
        meta_tau: float = 1.5,
        min_negative_weight: float = 0.20,
        neg_weight: float = 0.05,
        neg_margin_low: float = 0.20,
        neg_margin_high: float = 0.60,
        eps: float = 1e-8,
        return_debug: bool = False,
    ):
        super().__init__()

        self.temperature = float(temperature)
        self.temporal_weight = float(temporal_weight)
        self.overlap_weight = float(overlap_weight)
        self.year_tau = float(year_tau)
        self.meta_tau = float(meta_tau)
        self.min_negative_weight = float(min_negative_weight)
        self.neg_weight = float(neg_weight)
        self.neg_margin_low = float(neg_margin_low)
        self.neg_margin_high = float(neg_margin_high)
        self.eps = float(eps)
        self.return_debug = bool(return_debug)

    def _to_long(self, x, device):
        if not torch.is_tensor(x):
            return torch.tensor(x, device=device, dtype=torch.long)
        return x.to(device=device, dtype=torch.long)

    def _metadata_similarity(self, basin_meta: torch.Tensor) -> torch.Tensor:
        """
        Convert standardized metadata vectors to [N, N] similarity in [0, 1].
        """
        basin_meta = torch.nan_to_num(basin_meta.float())
        if basin_meta.ndim != 2:
            raise ValueError(f"basin_meta must be [B, M], got {tuple(basin_meta.shape)}")

        # Normalized Euclidean distance so meta_dim does not dominate scale.
        d = torch.cdist(basin_meta, basin_meta, p=2)
        d = d / max(float(basin_meta.size(1)) ** 0.5, 1.0)
        sim = torch.exp(-(d ** 2) / (2.0 * self.meta_tau ** 2))
        return sim.clamp(0.0, 1.0)

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        site_ids_int,
        years,
        basin_meta: torch.Tensor | None = None,
        overlap_adjacency: dict[int, dict[int, float]] | dict[int, set[int]] | None = None,
    ):
        device = z1.device

        z1 = F.normalize(z1.float(), dim=1)
        z2 = F.normalize(z2.float(), dim=1)
        z = torch.cat([z1, z2], dim=0)  # [2B, D]

        site_ids_int = self._to_long(site_ids_int, device)
        years = self._to_long(years, device)

        site_tensor = torch.cat([site_ids_int, site_ids_int], dim=0)  # [2B]
        year_tensor = torch.cat([years, years], dim=0)                # [2B]

        N = z.size(0)
        eye = torch.eye(N, device=device, dtype=torch.bool)

        cosine_sim = z @ z.T
        logits = (cosine_sim / self.temperature).float()

        # Do NOT use -1e9 here under autocast/torch.compile.
        # fp16 cannot represent -1e9, which can trigger compiler overflow.
        mask_value = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(eye, mask_value)

        same_basin = site_tensor[:, None] == site_tensor[None, :]
        same_year = year_tensor[:, None] == year_tensor[None, :]
        year_gap = (year_tensor[:, None] - year_tensor[None, :]).abs().float()
        temporal_decay = torch.exp(-year_gap / self.year_tau)

        overlap_strength = build_overlap_strength_matrix(
            site_tensor=site_tensor,
            overlap_adjacency=overlap_adjacency,
            eye=eye,
        )

        hard_pos = same_basin & same_year & (~eye)
        temporal_pos = same_basin & (~same_year) & (~eye)
        overlap_pos = (~same_basin) & (overlap_strength > 0) & (~eye)

        # --------------------------------------------------
        # Positive relation weights
        # --------------------------------------------------
        pos_w = torch.zeros((N, N), device=device, dtype=torch.float32)
        pos_w = pos_w + hard_pos.float()
        pos_w = pos_w + self.temporal_weight * temporal_decay * temporal_pos.float()
        pos_w = pos_w + self.overlap_weight * overlap_strength * temporal_decay * overlap_pos.float()

        pos_sum = pos_w.sum(dim=1, keepdim=True)
        valid_anchor = pos_sum.squeeze(1) > self.eps
        pos_prob = pos_w / pos_sum.clamp(min=self.eps)

        # --------------------------------------------------
        # Metadata-aware denominator weights
        # --------------------------------------------------
        denom_w = (~eye).float()
        meta_sim = None

        if basin_meta is not None:
            basin_meta = basin_meta.to(device=device, dtype=torch.float32)
            basin_meta = torch.cat([basin_meta, basin_meta], dim=0)
            meta_sim = self._metadata_similarity(basin_meta)

            # Do not treat metadata-similar non-positive basins as equally hard
            # negatives. This reduces false-negative pressure.
            non_positive = (pos_w <= 0) & (~eye)
            relief = self.min_negative_weight + (1.0 - self.min_negative_weight) * (1.0 - meta_sim)
            relief = relief.clamp(min=self.min_negative_weight, max=1.0)
            denom_w = torch.where(non_positive, relief, denom_w)

        denom_w = denom_w.clamp(min=self.eps)
        weighted_logits = logits + torch.log(denom_w)
        log_denom = torch.logsumexp(weighted_logits, dim=1, keepdim=True)
        log_prob = logits - log_denom

        contrastive_per_anchor = -(pos_prob * log_prob).sum(dim=1)
        contrastive_loss = contrastive_per_anchor[valid_anchor].mean()

        # --------------------------------------------------
        # Metadata-aware true-negative margin penalty
        # --------------------------------------------------
        true_neg = (~same_basin) & (overlap_strength <= 0) & (~eye)
        neg_loss = torch.tensor(0.0, device=device)

        if true_neg.any():
            if meta_sim is None:
                margin = torch.full_like(cosine_sim, self.neg_margin_low)
            else:
                # Metadata-similar basins are allowed to be closer.
                # Metadata-dissimilar basins should not have very high cosine sim.
                margin = self.neg_margin_low + (
                    self.neg_margin_high - self.neg_margin_low
                ) * meta_sim

            neg_penalty = F.relu(cosine_sim - margin).pow(2)
            neg_loss = neg_penalty[true_neg].mean()

        total_loss = contrastive_loss + self.neg_weight * neg_loss

        if self.return_debug:
            debug = {
                "loss_total": total_loss.detach(),
                "loss_contrastive": contrastive_loss.detach(),
                "loss_negative": neg_loss.detach(),
                "hard_pos_total": hard_pos.sum().detach(),
                "temporal_pos_total": temporal_pos.sum().detach(),
                "overlap_pos_total": overlap_pos.sum().detach(),
                "true_neg_total": true_neg.sum().detach(),
                "pos_weight_mean": pos_w[pos_w > 0].mean().detach() if (pos_w > 0).any() else torch.tensor(0.0, device=device),
                "denom_weight_mean": denom_w[(~eye)].mean().detach(),
            }
            return total_loss, debug

        return total_loss
