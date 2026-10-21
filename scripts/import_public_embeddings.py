#!/usr/bin/env python3
"""
Convert the public Basin2Vec archive back into the row-wise NPZ format.

This is the inverse of ``export_public_embeddings.py``. The downstream
evaluation scripts (``emb/evaluation/spatial_split.py``,
``emb/evaluation/temporal_split.py``) and the exploration app read a
row-wise archive, which is normally produced by
``emb/src/evaluate_embeddings_monthly.py`` after training. This script
recreates it from the public release so the evaluation can be reproduced
without retraining.

Input directory
---------------
    embeddings.npy       (num_basins, num_years, embedding_dim)
    usgs_ids.npy         (num_basins,)
    years.npy            (num_years,)
    avg_embeddings.npy   (num_basins, embedding_dim)
    metadata.json

Output NPZ keys
---------------
    embeddings       (num_basins * num_years, embedding_dim)
    labels           (num_basins * num_years,)   USGS IDs
    years            (num_basins * num_years,)
    avg_embeddings   (num_basins, embedding_dim)
    avg_labels       (num_basins,)

Example
-------
python scripts/import_public_embeddings.py \
    --source data/basin2vec-pretrained-v1.0.0 \
    --output emb/src/evaluation_outputs/HCLV4_grouped_monthly_static_meta_areaaux_64d/basin_embeddings_full.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


REQUIRED_FILES = (
    "embeddings.npy",
    "usgs_ids.npy",
    "years.npy",
    "avg_embeddings.npy",
)


def load_public_archive(source: Path) -> dict[str, np.ndarray]:
    """Load and validate the public pretrained arrays."""
    missing = [name for name in REQUIRED_FILES if not (source / name).is_file()]

    if missing:
        raise FileNotFoundError(
            f"Missing files in {source}: {missing}"
        )

    arrays = {
        name.removesuffix(".npy"): np.load(source / name, allow_pickle=False)
        for name in REQUIRED_FILES
    }

    num_basins, num_years, embedding_dim = arrays["embeddings"].shape

    if arrays["usgs_ids"].shape != (num_basins,):
        raise ValueError(
            f"usgs_ids.npy has shape {arrays['usgs_ids'].shape}; "
            f"expected ({num_basins},)."
        )

    if arrays["years"].shape != (num_years,):
        raise ValueError(
            f"years.npy has shape {arrays['years'].shape}; "
            f"expected ({num_years},)."
        )

    if arrays["avg_embeddings"].shape != (num_basins, embedding_dim):
        raise ValueError(
            f"avg_embeddings.npy has shape {arrays['avg_embeddings'].shape}; "
            f"expected ({num_basins}, {embedding_dim})."
        )

    return arrays


def to_row_wise(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Flatten the (basin, year) grid into one row per basin-year."""
    grid = arrays["embeddings"]
    usgs_ids = arrays["usgs_ids"].astype(str)
    years = arrays["years"].astype(np.int64)

    num_basins, num_years, embedding_dim = grid.shape

    return {
        "embeddings": grid.reshape(num_basins * num_years, embedding_dim).astype(np.float32),
        "labels": np.repeat(usgs_ids, num_years),
        "years": np.tile(years, num_basins),
        "avg_embeddings": arrays["avg_embeddings"].astype(np.float32),
        "avg_labels": usgs_ids,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the public Basin2Vec archive into the row-wise NPZ format."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Extracted public archive directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination .npz file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output} already exists. Use --overwrite to replace it."
        )

    rows = to_row_wise(load_public_archive(source))

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **rows)

    print(f"Wrote {output}")
    print(f"  rows:   {rows['embeddings'].shape[0]:,}")
    print(f"  basins: {rows['avg_embeddings'].shape[0]:,}")
    print(f"  years:  {rows['years'].min()}-{rows['years'].max()}")
    print(f"  dim:    {rows['embeddings'].shape[1]}")


if __name__ == "__main__":
    main()
