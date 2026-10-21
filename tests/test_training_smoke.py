"""
Smoke test for the Basin2Vec model and loss on synthetic tensors.

Checks that the research environment (PyTorch etc.) can run a training
step without any data. Skipped when PyTorch is not installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("zarr")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "emb" / "src"))

from attention_encoder_monthly import GroupedBasinEncoderMonthlyV2  # noqa: E402
from hydrology_aware_contrastive_loss import HydrologyAwareContrastiveLossV4  # noqa: E402


def test_training_step_on_synthetic_batch():
    torch.manual_seed(0)

    basins, years_per_basin, months = 4, 2, 12
    batch = basins * years_per_basin
    monthly_channels, static_channels, meta_dim, size = 5, 7, 3, 16

    model = GroupedBasinEncoderMonthlyV2(
        monthly_in_channels=monthly_channels,
        static_in_channels=static_channels,
        emb_dim=64,
        meta_dim=meta_dim,
        max_months=months,
        temporal_layers=1,
        temporal_heads=4,
        dropout=0.05,
        metadata_dropout=0.15,
        predict_month_aux=False,
    )
    criterion = HydrologyAwareContrastiveLossV4(temperature=0.15, temporal_weight=0.30, overlap_weight=0.12)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    def view():
        return dict(
            monthly_x=torch.randn(batch, months, monthly_channels, size, size),
            static_x=torch.randn(batch, static_channels, size, size),
            mask=torch.ones(batch, 1, size, size),
            basin_meta=torch.randn(batch, meta_dim),
        )

    z1, *_ = model(**view())
    z2, *_ = model(**view())

    assert z1.shape == (batch, 64)
    torch.testing.assert_close(z1.norm(dim=1), torch.ones(batch))

    site_ids = torch.arange(basins).repeat_interleave(years_per_basin)
    years = torch.tensor([2000, 2001] * basins)
    overlap = {0: {1: 0.5}, 1: {0: 0.5}}

    loss = criterion(z1, z2, site_ids, years, basin_meta=torch.randn(batch, meta_dim), overlap_adjacency=overlap)

    assert torch.isfinite(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
