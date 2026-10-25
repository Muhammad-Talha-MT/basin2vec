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
        self.attn = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_bool = mask.bool()
        attn_logits = self.attn(x).float()
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
    def __init__(self, in_channels: int, emb_dim: int = 128, dropout: float = 0.05):
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
        mask_ds = F.interpolate(mask.float(), size=feat.shape[-2:], mode="nearest")
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

        hidden = max(emb_dim // 2, 1)
        self.month_attention = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
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
                nn.Linear(emb_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, 1),
            )
        else:
            self.month_head = None

    def forward(self, x_monthly: torch.Tensor, mask: torch.Tensor):
        if x_monthly.dim() != 5:
            raise ValueError(
                f"MonthlyMeteorologyBranch expected [B,M,C,H,W], got {tuple(x_monthly.shape)}"
            )

        B, M, C, H, W = x_monthly.shape
        if M > self.max_months:
            raise ValueError(f"Input has {M} months, but max_months={self.max_months}")
        if C != self.in_channels:
            raise ValueError(f"Expected monthly in_channels={self.in_channels}, got {C}")

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
        spatial_maps = attn_flat.view(B, M, 1, attn_flat.shape[-2], attn_flat.shape[-1])
        month_pred = (
            self.month_head(temporal_tokens).squeeze(-1)
            if self.month_head is not None
            else None
        )
        return z_monthly, month_weights, spatial_maps, month_pred


# -----------------------------------------------------------
# Static / annual grouped branch
# -----------------------------------------------------------
class StaticGroupedBranch(nn.Module):
    def __init__(self, in_channels: int, emb_dim: int = 128, dropout: float = 0.05):
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
# Ablation-ready grouped Basin2Vec encoder
# -----------------------------------------------------------
class GroupedBasinEncoderMonthlyV2(nn.Module):
    """
    Ablation-ready Basin2Vec encoder.

    Key switches
    ------------
    use_monthly_branch:
        If False, the monthly/dynamic branch is removed.

    use_static_branch:
        If False, the quasi-static branch is removed.

    metadata_fusion:
        "gated" : original dimension-wise sigmoid-gated metadata fusion.
        "concat": concatenate gridded + metadata embeddings, project 2D -> D,
                  then use the same refinement block.
        "none"  : no metadata encoder/fusion in the representation encoder.
                  Metadata may still be passed separately to the SSL loss.

    use_area_head:
        If False, the auxiliary area prediction head is not constructed.
    """

    VALID_METADATA_FUSIONS = {"gated", "concat", "none"}

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
        use_monthly_branch: bool = True,
        use_static_branch: bool = True,
        metadata_fusion: str = "gated",
        use_area_head: bool = True,
    ):
        super().__init__()

        self.monthly_in_channels = int(monthly_in_channels)
        self.static_in_channels = int(static_in_channels)
        self.emb_dim = int(emb_dim)
        self.meta_dim = int(meta_dim or 0)
        self.metadata_dropout = float(metadata_dropout)
        self.use_monthly_branch = bool(use_monthly_branch)
        self.use_static_branch = bool(use_static_branch)
        self.metadata_fusion = str(metadata_fusion).lower()
        self.use_area_head = bool(use_area_head)

        if not self.use_monthly_branch and not self.use_static_branch:
            raise ValueError("At least one of monthly or static branches must be enabled.")
        if self.use_monthly_branch and self.monthly_in_channels <= 0:
            raise ValueError("monthly_in_channels must be > 0 when monthly branch is enabled")
        if self.use_static_branch and self.static_in_channels <= 0:
            raise ValueError("static_in_channels must be > 0 when static branch is enabled")
        if self.metadata_fusion not in self.VALID_METADATA_FUSIONS:
            raise ValueError(
                f"metadata_fusion must be one of {sorted(self.VALID_METADATA_FUSIONS)}, "
                f"got {self.metadata_fusion!r}"
            )
        if self.metadata_fusion != "none" and self.meta_dim <= 0:
            raise ValueError("meta_dim must be > 0 when metadata fusion is enabled")

        if self.use_monthly_branch:
            self.monthly_branch = MonthlyMeteorologyBranch(
                in_channels=self.monthly_in_channels,
                emb_dim=emb_dim,
                max_months=max_months,
                temporal_layers=temporal_layers,
                temporal_heads=temporal_heads,
                dropout=dropout,
                predict_month_aux=predict_month_aux,
            )
        else:
            self.monthly_branch = None

        if self.use_static_branch:
            self.static_branch = StaticGroupedBranch(
                in_channels=self.static_in_channels,
                emb_dim=emb_dim,
                dropout=dropout,
            )
        else:
            self.static_branch = None

        self.num_branches = int(self.use_monthly_branch) + int(self.use_static_branch)
        self.branch_attention = nn.Linear(emb_dim, 1, bias=False)

        if self.metadata_fusion != "none":
            self.meta_encoder = nn.Sequential(
                nn.Linear(self.meta_dim, emb_dim),
                nn.LayerNorm(emb_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(emb_dim, emb_dim),
                nn.LayerNorm(emb_dim),
                nn.ReLU(inplace=True),
            )
            self.meta_norm = nn.LayerNorm(emb_dim)

            if self.metadata_fusion == "gated":
                self.meta_gate = nn.Sequential(
                    nn.Linear(emb_dim * 2, emb_dim),
                    nn.Sigmoid(),
                )
                self.meta_concat_project = None
            else:  # concat
                self.meta_gate = None
                self.meta_concat_project = nn.Linear(emb_dim * 2, emb_dim)
        else:
            self.meta_encoder = None
            self.meta_norm = None
            self.meta_gate = None
            self.meta_concat_project = None

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

        self.area_head = nn.Linear(emb_dim, 1) if self.use_area_head else None

    def forward(
        self,
        monthly_x: torch.Tensor | None,
        static_x: torch.Tensor | None,
        mask: torch.Tensor,
        basin_meta: torch.Tensor | None = None,
    ):
        branch_embeddings: list[torch.Tensor] = []
        spatial_maps: dict[str, torch.Tensor] = {}

        z_monthly = None
        z_static = None
        month_weights = None
        month_pred = None

        if self.monthly_branch is not None:
            if monthly_x is None:
                raise ValueError("monthly_x is required because monthly branch is enabled")
            z_monthly, month_weights, monthly_spatial_maps, month_pred = self.monthly_branch(
                monthly_x, mask
            )
            branch_embeddings.append(z_monthly)
            spatial_maps["monthly"] = monthly_spatial_maps

        if self.static_branch is not None:
            if static_x is None:
                raise ValueError("static_x is required because static branch is enabled")
            z_static, static_attn = self.static_branch(static_x, mask)
            branch_embeddings.append(z_static)
            spatial_maps["static"] = static_attn

        H = torch.stack(branch_embeddings, dim=1)  # [B, n_branch, D]
        branch_scores = self.branch_attention(H)
        branch_weights = torch.softmax(branch_scores, dim=1)
        z_multisource = (H * branch_weights).sum(dim=1)

        area_pred = (
            self.area_head(z_multisource).squeeze(-1)
            if self.area_head is not None
            else None
        )

        z_meta = None
        if self.metadata_fusion != "none":
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

            if self.metadata_fusion == "gated":
                gate = self.meta_gate(torch.cat([z_multisource, z_meta], dim=1))
                h_in = self.meta_norm(z_multisource + gate * z_meta)
            elif self.metadata_fusion == "concat":
                h_in = self.meta_concat_project(
                    torch.cat([z_multisource, z_meta], dim=1)
                )
                h_in = self.meta_norm(h_in)
            else:  # defensive
                raise RuntimeError(f"Unexpected metadata_fusion={self.metadata_fusion}")
        else:
            h_in = z_multisource

        h = self.fusion(h_in)
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
            "metadata_fusion": self.metadata_fusion,
        }

        recon_maps = {}
        return z, branch_weights, spatial_maps, recon_maps, aux


# Backward-compatible alias
BasinEncoderMonthlyV1 = GroupedBasinEncoderMonthlyV2
