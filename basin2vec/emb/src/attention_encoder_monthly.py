# emb/src/attention_encoder_monthly.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------
# Masked global average pooling
# -----------------------------------------------------------
def masked_global_avg_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    x    : [B, C, H, W]
    mask : [B, 1, H, W]
    """
    mask = mask.float()
    masked = x * mask

    denom = mask.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
    pooled = masked.sum(dim=(2, 3), keepdim=True) / denom

    return pooled.squeeze(-1).squeeze(-1)


# -----------------------------------------------------------
# Spatial attention
# -----------------------------------------------------------
class SpatialAttention(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.attn = nn.Conv2d(
            in_channels,
            1,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x    : [B, C, H, W]
        mask : [B, 1, H, W]
        """
        mask_bool = mask.bool()

        # Keep attention logits in fp32 under autocast.
        attn_logits = self.attn(x).float()

        # Safe for fp16/autocast.
        mask_value = torch.finfo(attn_logits.dtype).min
        attn_logits = attn_logits.masked_fill(~mask_bool, mask_value)

        flat = attn_logits.flatten(1)
        attn_map = torch.softmax(flat, dim=1).view_as(attn_logits)

        attn_map = attn_map * mask.float()
        attn_map = attn_map / attn_map.sum(dim=(2, 3), keepdim=True).clamp(min=1e-8)
        attn_map = torch.nan_to_num(attn_map, nan=0.0, posinf=0.0, neginf=0.0)

        return attn_map.to(dtype=x.dtype)


# -----------------------------------------------------------
# Spatial CNN token encoder
# -----------------------------------------------------------
class SpatialStackEncoder(nn.Module):
    """
    Encodes a multi-channel spatial stack into one token.

    Input:
        x    : [B, C, H, W]
        mask : [B, 1, H, W]

    Output:
        token    : [B, D]
        attn_map : [B, 1, H, W]
    """

    def __init__(
        self,
        in_channels: int,
        emb_dim: int = 128,
        dropout: float = 0.05,
    ):
        super().__init__()

        self.in_channels = int(in_channels)
        self.emb_dim = int(emb_dim)

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, padding=1, bias=False),
            nn.GroupNorm(6, 48),
            nn.ReLU(inplace=True),

            nn.Conv2d(48, 96, 3, padding=1, bias=False),
            nn.GroupNorm(12, 96),
            nn.ReLU(inplace=True),

            nn.Conv2d(96, 128, 3, padding=1, bias=False),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
        )

        self.spatial_attn = SpatialAttention(128)

        # 128 masked-mean + 128 attention-pooled
        self.fc = nn.Sequential(
            nn.Linear(256, emb_dim, bias=False),
            nn.LayerNorm(emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim, bias=False),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        x = torch.nan_to_num(x)
        x = torch.clamp(x, -10, 10)

        feat = self.backbone(x)

        mask_ds = F.interpolate(
            mask.float(),
            size=feat.shape[-2:],
            mode="nearest",
        )

        attn_map = self.spatial_attn(feat, mask_ds)

        mean_pooled = masked_global_avg_pool(feat, mask_ds)
        attn_pooled = (feat * attn_map).sum(dim=(2, 3))

        pooled = torch.cat([mean_pooled, attn_pooled], dim=1)
        token = self.fc(pooled)

        return token, attn_map


# -----------------------------------------------------------
# Monthly meteorology branch
# -----------------------------------------------------------
class MonthlyMeteorologyBranch(nn.Module):
    """
    Encodes grouped monthly meteorological input.

    Input:
        x_monthly : [B, M, C_monthly, H, W]
                    e.g. C_monthly=5 for prcp/tmax/tmin/vp/swe

    Output:
        z_monthly     : [B, D]
        month_weights : [B, M, 1]
        spatial_maps  : [B, M, 1, H, W]
        month_pred    : [B, M] or None
    """

    def __init__(
        self,
        in_channels: int,
        emb_dim: int = 128,
        max_months: int = 12,
        temporal_layers: int = 1,
        temporal_heads: int = 4,
        dropout: float = 0.05,
        predict_month_aux: bool = False,
    ):
        super().__init__()

        self.in_channels = int(in_channels)
        self.emb_dim = int(emb_dim)
        self.max_months = int(max_months)
        self.predict_month_aux = bool(predict_month_aux)

        self.spatial_encoder = SpatialStackEncoder(
            in_channels=in_channels,
            emb_dim=emb_dim,
            dropout=dropout,
        )

        self.month_pos = nn.Parameter(torch.zeros(1, max_months, emb_dim))
        nn.init.normal_(self.month_pos, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=temporal_heads,
            dim_feedforward=emb_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=temporal_layers,
            enable_nested_tensor=False,
        )

        self.month_attention = nn.Sequential(
            nn.Linear(emb_dim, max(emb_dim // 2, 1)),
            nn.Tanh(),
            nn.Linear(max(emb_dim // 2, 1), 1),
        )

        self.out_fc = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim, bias=False),
        )

        if self.predict_month_aux:
            self.month_head = nn.Sequential(
                nn.LayerNorm(emb_dim),
                nn.Linear(emb_dim, max(emb_dim // 2, 1)),
                nn.ReLU(inplace=True),
                nn.Linear(max(emb_dim // 2, 1), 1),
            )
        else:
            self.month_head = None

    def forward(self, x_monthly: torch.Tensor, mask: torch.Tensor):
        if x_monthly.dim() != 5:
            raise ValueError(
                f"MonthlyMeteorologyBranch expected [B,M,C,H,W], "
                f"got {tuple(x_monthly.shape)}"
            )

        B, M, C, H, W = x_monthly.shape

        if M > self.max_months:
            raise ValueError(f"Input has {M} months, but max_months={self.max_months}")

        if C != self.in_channels:
            raise ValueError(
                f"Expected monthly in_channels={self.in_channels}, got {C}"
            )

        # One spatial CNN pass per month, with all meteorological variables as channels.
        x_flat = x_monthly.reshape(B * M, C, H, W)

        mask_flat = (
            mask.unsqueeze(1)
            .expand(B, M, 1, H, W)
            .reshape(B * M, 1, H, W)
        )

        token_flat, attn_flat = self.spatial_encoder(x_flat, mask_flat)

        monthly_tokens = token_flat.view(B, M, self.emb_dim)
        monthly_tokens = monthly_tokens + self.month_pos[:, :M, :].to(
            dtype=monthly_tokens.dtype
        )

        temporal_tokens = self.temporal_encoder(monthly_tokens)

        month_scores = self.month_attention(temporal_tokens)
        month_weights = torch.softmax(month_scores, dim=1)

        z_monthly = (temporal_tokens * month_weights).sum(dim=1)
        z_monthly = self.out_fc(z_monthly)

        spatial_maps = attn_flat.view(
            B,
            M,
            1,
            attn_flat.shape[-2],
            attn_flat.shape[-1],
        )

        if self.month_head is not None:
            month_pred = self.month_head(temporal_tokens).squeeze(-1)
        else:
            month_pred = None

        return z_monthly, month_weights, spatial_maps, month_pred


# -----------------------------------------------------------
# Static / annual grouped branch
# -----------------------------------------------------------
class StaticGroupedBranch(nn.Module):
    """
    Encodes all non-monthly/static/annual variables as one spatial stack.

    Input:
        x_static : [B, C_static, H, W]

    Output:
        z_static : [B, D]
        attn_map : [B, 1, H, W]
    """

    def __init__(
        self,
        in_channels: int,
        emb_dim: int = 128,
        dropout: float = 0.05,
    ):
        super().__init__()

        self.in_channels = int(in_channels)

        self.spatial_encoder = SpatialStackEncoder(
            in_channels=in_channels,
            emb_dim=emb_dim,
            dropout=dropout,
        )

        self.out_fc = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim, bias=False),
        )

    def forward(self, x_static: torch.Tensor, mask: torch.Tensor):
        if x_static.dim() != 4:
            raise ValueError(
                f"StaticGroupedBranch expected [B,C,H,W], got {tuple(x_static.shape)}"
            )

        if x_static.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected static in_channels={self.in_channels}, got {x_static.shape[1]}"
            )

        token, attn_map = self.spatial_encoder(x_static, mask)
        z_static = self.out_fc(token)

        return z_static, attn_map


# -----------------------------------------------------------
# Fast grouped Basin2Vec encoder
# -----------------------------------------------------------
class GroupedBasinEncoderMonthlyV2(nn.Module):
    """
    Fast grouped encoder for mixed monthly + non-monthly Basin2Vec training.

    Inputs
    ------
    monthly_x:
        [B, 12, C_monthly, H, W]

    static_x:
        [B, C_static, H, W]

    mask:
        [B, 1, H, W]

    basin_meta:
        [B, M_meta]

    Output
    ------
    z:
        [B, D] basin-year embedding
    """

    def __init__(
        self,
        monthly_in_channels: int,
        static_in_channels: int,
        emb_dim: int = 128,
        meta_dim: int = 0,
        max_months: int = 12,
        temporal_layers: int = 1,
        temporal_heads: int = 4,
        dropout: float = 0.05,
        metadata_dropout: float = 0.15,
        predict_month_aux: bool = False,
    ):
        super().__init__()

        self.monthly_in_channels = int(monthly_in_channels)
        self.static_in_channels = int(static_in_channels)
        self.emb_dim = int(emb_dim)
        self.meta_dim = int(meta_dim or 0)
        self.metadata_dropout = float(metadata_dropout)

        if self.monthly_in_channels <= 0:
            raise ValueError("monthly_in_channels must be > 0")

        self.monthly_branch = MonthlyMeteorologyBranch(
            in_channels=monthly_in_channels,
            emb_dim=emb_dim,
            max_months=max_months,
            temporal_layers=temporal_layers,
            temporal_heads=temporal_heads,
            dropout=dropout,
            predict_month_aux=predict_month_aux,
        )

        if self.static_in_channels > 0:
            self.static_branch = StaticGroupedBranch(
                in_channels=static_in_channels,
                emb_dim=emb_dim,
                dropout=dropout,
            )
            self.num_branches = 2
        else:
            self.static_branch = None
            self.num_branches = 1

        self.branch_attention = nn.Linear(emb_dim, 1, bias=False)

        if self.meta_dim > 0:
            self.meta_encoder = nn.Sequential(
                nn.Linear(self.meta_dim, emb_dim),
                nn.LayerNorm(emb_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(emb_dim, emb_dim),
                nn.LayerNorm(emb_dim),
                nn.ReLU(inplace=True),
            )

            self.meta_gate = nn.Sequential(
                nn.Linear(emb_dim * 2, emb_dim),
                nn.Sigmoid(),
            )

            self.meta_norm = nn.LayerNorm(emb_dim)
        else:
            self.meta_encoder = None
            self.meta_gate = None
            self.meta_norm = None

        self.fusion = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
        )

        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, emb_dim, bias=False),
        )

        # Area prediction from pre-metadata representation.
        self.area_head = nn.Linear(emb_dim, 1)

    def forward(
        self,
        monthly_x: torch.Tensor,
        static_x: torch.Tensor | None,
        mask: torch.Tensor,
        basin_meta: torch.Tensor | None = None,
    ):
        z_monthly, month_weights, monthly_spatial_maps, month_pred = self.monthly_branch(
            monthly_x,
            mask,
        )

        branch_embeddings = [z_monthly]
        spatial_maps = {
            "monthly": monthly_spatial_maps,
        }

        if self.static_branch is not None:
            if static_x is None:
                raise ValueError("static_x is required because static_branch exists.")

            z_static, static_attn = self.static_branch(static_x, mask)
            branch_embeddings.append(z_static)
            spatial_maps["static"] = static_attn
        else:
            z_static = None

        H = torch.stack(branch_embeddings, dim=1)  # [B, num_branches, D]

        branch_scores = self.branch_attention(H)   # [B, num_branches, 1]
        branch_weights = torch.softmax(branch_scores, dim=1)

        z_multisource = (H * branch_weights).sum(dim=1)

        area_pred = self.area_head(z_multisource).squeeze(-1)

        if self.meta_dim > 0:
            if basin_meta is None:
                basin_meta = torch.zeros(
                    z_multisource.size(0),
                    self.meta_dim,
                    device=z_multisource.device,
                    dtype=z_multisource.dtype,
                )

            basin_meta = torch.nan_to_num(basin_meta.float())

            if self.training and self.metadata_dropout > 0:
                basin_meta = F.dropout(
                    basin_meta,
                    p=self.metadata_dropout,
                    training=True,
                )

            z_meta = self.meta_encoder(basin_meta)
            gate = self.meta_gate(torch.cat([z_multisource, z_meta], dim=1))
            h_in = self.meta_norm(z_multisource + gate * z_meta)
            h = self.fusion(h_in)
        else:
            z_meta = None
            h = self.fusion(z_multisource)

        z = self.proj(h)
        z = F.normalize(z, dim=1)

        aux = {
            "h": h,
            "z_multisource": z_multisource,
            "z_monthly": z_monthly,
            "z_static": z_static,
            "z_meta": z_meta,
            "area_pred": area_pred,
            "month_pred": month_pred,
            "month_attention": month_weights,
            "branch_attention": branch_weights,
        }

        recon_maps = {}

        return z, branch_weights, spatial_maps, recon_maps, aux


# Backward-compatible alias
BasinEncoderMonthlyV1 = GroupedBasinEncoderMonthlyV2