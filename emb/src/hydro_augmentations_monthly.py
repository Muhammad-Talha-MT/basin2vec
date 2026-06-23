# emb/src/hydro_augmentations_monthly.py

from __future__ import annotations

import torch


DEFAULT_TEMPERATURE_VARS = {"tmin", "tmax", "tmean", "temperature"}
DEFAULT_CATEGORICAL_VARS = {"land_cover", "landcover", "nlcd"}


def _as_sequence(x: torch.Tensor):
    """
    Convert variable tensor to [B, M, C, H, W].

    Input can be:
        [B, M, C, H, W] monthly
        [B, C, H, W]    non-monthly/static/annual
    """
    if x.dim() == 5:
        return x, True
    if x.dim() == 4:
        return x.unsqueeze(1), False
    raise ValueError(f"Expected variable tensor [B,M,C,H,W] or [B,C,H,W], got {tuple(x.shape)}")


def _restore_shape(x: torch.Tensor, was_monthly: bool) -> torch.Tensor:
    return x if was_monthly else x[:, 0]


def hydro_augment_monthly(
    batch_dict: dict[str, torch.Tensor],
    mask: torch.Tensor,
    variable_names: list[str] | None = None,
    monthly_variable_names: list[str] | None = None,
    categorical_variable_names: list[str] | None = None,
    temperature_variable_names: list[str] | None = None,
    noise_std: float = 0.03,
    spatial_dropout_p: float = 0.03,
    month_dropout_p: float = 0.08,
    seasonal_block_dropout_p: float = 0.08,
    variable_dropout_p: float = 0.03,
    multiplicative_scale_std: float = 0.04,
    temperature_shift_std: float = 0.03,
    clamp_value: float = 10.0,
) -> dict[str, torch.Tensor]:
    """
    Hydrologically safe augmentations for mixed monthly + non-monthly inputs.

    Monthly variables receive weak noise, optional spatial dropout, month dropout,
    short seasonal-block dropout, and weak variable-level dropout.

    Non-monthly continuous variables receive weak noise, spatial dropout, and weak
    variable-level dropout, but no fake month dropout.

    Categorical variables such as land_cover are not noised or pixel-dropped. They
    may only be dropped as a whole variable for a small fraction of samples.

    Notes
    -----
    - No month permutation is used.
    - No aggressive crop/rotation is used because basin geometry is meaningful.
    - All augmentations are mask-aware.
    """
    if variable_names is None:
        variable_names = list(batch_dict.keys())

    monthly_set = set(monthly_variable_names or [])
    categorical_set = set(categorical_variable_names or DEFAULT_CATEGORICAL_VARS)
    temperature_set = set(temperature_variable_names or DEFAULT_TEMPERATURE_VARS)

    mask = mask.float()
    mask5 = mask.unsqueeze(1)  # [B,1,1,H,W]

    out: dict[str, torch.Tensor] = {}

    for v in variable_names:
        x = batch_dict[v]
        y, was_monthly = _as_sequence(x)
        y = y.clone()

        B, M, C, H, W = y.shape
        is_monthly_var = v in monthly_set or was_monthly
        is_categorical = v in categorical_set
        is_temperature = v in temperature_set

        if is_categorical:
            # Preserve one-hot/categorical semantics. Only whole-variable dropout
            # is allowed, and even that is weak.
            if variable_dropout_p > 0:
                keep = (
                    torch.rand(B, 1, 1, 1, 1, device=y.device, dtype=y.dtype)
                    > variable_dropout_p
                ).to(y.dtype)
                y = y * keep
            y = y * mask5
            out[v] = _restore_shape(y, was_monthly)
            continue

        # Weak variable-wise perturbation. For temperature-like normalized maps,
        # additive shifts are more natural than multiplicative scaling.
        if is_temperature and temperature_shift_std > 0:
            shift = temperature_shift_std * torch.randn(
                B, 1, 1, 1, 1, device=y.device, dtype=y.dtype
            )
            y = y + shift
        elif multiplicative_scale_std > 0:
            scale = 1.0 + multiplicative_scale_std * torch.randn(
                B, 1, 1, 1, 1, device=y.device, dtype=y.dtype
            )
            y = y * scale

        # Weak pixel noise inside basin.
        if noise_std > 0:
            y = y + noise_std * torch.randn_like(y) * mask5

        # Spatial dropout. One dropout map per sample, shared across months so it
        # does not create artificial month-specific spatial holes.
        if spatial_dropout_p > 0:
            keep = (
                torch.rand(B, 1, 1, H, W, device=y.device, dtype=y.dtype)
                > spatial_dropout_p
            ).to(y.dtype)
            y = y * keep * mask5

        # Month dropout only for true monthly variables.
        if is_monthly_var and M > 1 and month_dropout_p > 0:
            month_keep = (
                torch.rand(B, M, 1, 1, 1, device=y.device, dtype=y.dtype)
                > month_dropout_p
            ).to(y.dtype)

            all_dropped = month_keep.sum(dim=1, keepdim=True) == 0
            if all_dropped.any():
                restore_idx = torch.randint(0, M, (B,), device=y.device)
                restore = torch.zeros_like(month_keep)
                restore[torch.arange(B, device=y.device), restore_idx, 0, 0, 0] = 1.0
                month_keep = torch.where(all_dropped, restore, month_keep)

            y = y * month_keep

        # Short seasonal block dropout only for true monthly variables.
        if is_monthly_var and M > 2 and seasonal_block_dropout_p > 0:
            do_block = torch.rand(B, device=y.device) < seasonal_block_dropout_p
            if do_block.any():
                block_keep = torch.ones(B, M, 1, 1, 1, device=y.device, dtype=y.dtype)
                starts = torch.randint(0, M, (B,), device=y.device)
                lengths = torch.randint(1, min(3, M) + 1, (B,), device=y.device)
                for i in torch.nonzero(do_block, as_tuple=False).flatten():
                    s = int(starts[i].item())
                    L = int(lengths[i].item())
                    for k in range(L):
                        block_keep[i, (s + k) % M, 0, 0, 0] = 0.0
                y = y * block_keep

        # Whole-variable dropout.
        if variable_dropout_p > 0:
            keep = (
                torch.rand(B, 1, 1, 1, 1, device=y.device, dtype=y.dtype)
                > variable_dropout_p
            ).to(y.dtype)
            y = y * keep

        y = torch.nan_to_num(y, nan=0.0, posinf=clamp_value, neginf=-clamp_value)
        y = torch.clamp(y, -clamp_value, clamp_value)
        y = y * mask5
        out[v] = _restore_shape(y, was_monthly)

    return out


# Backward-compatible name if older scripts import hydro_augment.
def hydro_augment(batch_dict: dict[str, torch.Tensor], mask: torch.Tensor):
    return hydro_augment_monthly(batch_dict, mask)
