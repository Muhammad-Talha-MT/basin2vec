#!/usr/bin/env python3

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
from torch.utils.data import Sampler


def normalize_sampler_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize metadata to exactly:
        - site_id_int
        - year

    Accepts either:
        - site_id_int + year
        - site_id + year  -> deterministic factorized site_id_int
    """
    df = df.reset_index(drop=True).copy()

    if "year" not in df.columns:
        raise ValueError(
            f"Metadata must contain 'year'. Available columns: {list(df.columns)}"
        )

    if "site_id_int" in df.columns:
        out = df[["site_id_int", "year"]].copy()
        out["site_id_int"] = out["site_id_int"].astype(np.int64)
        out["year"] = out["year"].astype(np.int64)
        return out

    if "site_id" in df.columns:
        codes, _ = pd.factorize(df["site_id"], sort=True)
        out = pd.DataFrame(
            {
                "site_id_int": codes.astype(np.int64),
                "year": df["year"].astype(np.int64).values,
            }
        )
        return out

    raise ValueError(
        "Metadata must contain either 'site_id_int' or 'site_id', plus 'year'. "
        f"Available columns: {list(df.columns)}"
    )


def build_sampler_metadata(dataset, index_parquet_path: str) -> pd.DataFrame:
    """
    Try to recover metadata aligned with dataset indexing.

    Preferred:
        dataset.index_df / dataset.df / dataset.sample_index

    Fallback:
        read parquet directly
    """
    candidate_attrs = ["index_df", "df", "sample_index"]

    for attr in candidate_attrs:
        if hasattr(dataset, attr):
            obj = getattr(dataset, attr)
            if isinstance(obj, pd.DataFrame):
                df = normalize_sampler_metadata(obj)
                if len(df) != len(dataset):
                    raise ValueError(
                        f"Dataset metadata attr '{attr}' length {len(df)} "
                        f"does not match dataset length {len(dataset)}"
                    )
                return df

    df = pd.read_parquet(index_parquet_path)
    df = normalize_sampler_metadata(df)

    if len(df) != len(dataset):
        raise ValueError(
            f"Metadata length ({len(df)}) does not match dataset length ({len(dataset)})."
        )

    return df


class DistributedBasinBatchSampler(Sampler):
    """
    DDP-compatible basin-aware batch sampler.

    Each local batch contains:
        basins_per_batch * years_per_basin samples

    Construction:
        - choose `basins_per_batch` distinct basins
        - for each basin, choose `years_per_basin` years
        - for each (basin, year), choose one dataset index

    This increases same-basin/different-year samples in a batch and therefore
    improves soft positives for the hierarchical contrastive loss.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        basins_per_batch: int,
        years_per_basin: int,
        batches_per_epoch: int,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()

        required_cols = {"site_id_int", "year"}
        missing = required_cols - set(metadata_df.columns)
        if missing:
            raise ValueError(
                f"metadata_df missing required columns: {sorted(missing)}"
            )

        self.metadata_df = metadata_df.reset_index(drop=True).copy()
        self.basins_per_batch = int(basins_per_batch)
        self.years_per_basin = int(years_per_basin)
        self.local_batch_size = self.basins_per_batch * self.years_per_basin
        self.batches_per_epoch = int(batches_per_epoch)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0

        if self.basins_per_batch <= 0:
            raise ValueError("basins_per_batch must be positive")
        if self.years_per_basin <= 0:
            raise ValueError("years_per_basin must be positive")
        if self.batches_per_epoch <= 0:
            raise ValueError("batches_per_epoch must be positive")
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        if not (0 <= self.rank < self.world_size):
            raise ValueError(
                f"rank must be in [0, {self.world_size - 1}], got {self.rank}"
            )

        # basin -> year -> [dataset indices]
        self.basin_to_year_to_indices: Dict[int, Dict[int, List[int]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for idx, row in self.metadata_df.iterrows():
            basin = int(row["site_id_int"])
            year = int(row["year"])
            self.basin_to_year_to_indices[basin][year].append(idx)

        self.basins = sorted(self.basin_to_year_to_indices.keys())

        if len(self.basins) == 0:
            raise ValueError("No basins found in metadata_df")

        if len(self.basins) < self.basins_per_batch:
            raise ValueError(
                f"Only {len(self.basins)} basins available, but "
                f"basins_per_batch={self.basins_per_batch}"
            )

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return self.batches_per_epoch

    def _sample_years_for_basin(self, rng: random.Random, basin: int) -> List[int]:
        """
        If fewer unique years are available than requested, sample with replacement.
        """
        years = list(self.basin_to_year_to_indices[basin].keys())

        if len(years) >= self.years_per_basin:
            return rng.sample(years, self.years_per_basin)

        return [rng.choice(years) for _ in range(self.years_per_basin)]

    def _sample_index_for_basin_year(
        self,
        rng: random.Random,
        basin: int,
        year: int,
    ) -> int:
        candidates = self.basin_to_year_to_indices[basin][year]
        return rng.choice(candidates)

    def _make_one_batch(self, rng: random.Random) -> List[int]:
        chosen_basins = rng.sample(self.basins, self.basins_per_batch)
        batch: List[int] = []

        for basin in chosen_basins:
            chosen_years = self._sample_years_for_basin(rng, basin)
            for year in chosen_years:
                idx = self._sample_index_for_basin_year(rng, basin, year)
                batch.append(idx)

        if len(batch) != self.local_batch_size:
            raise RuntimeError(
                f"Constructed batch size {len(batch)} != expected {self.local_batch_size}"
            )

        return batch

    def _make_all_global_batches(self) -> List[List[int]]:
        """
        Generate all batches deterministically for all ranks, then shard.
        """
        rng = random.Random(self.seed + self.epoch)
        total_batches = self.batches_per_epoch * self.world_size
        return [self._make_one_batch(rng) for _ in range(total_batches)]

    def __iter__(self):
        all_batches = self._make_all_global_batches()
        rank_batches = all_batches[self.rank :: self.world_size]

        if len(rank_batches) != self.batches_per_epoch:
            raise RuntimeError(
                f"Rank {self.rank} got {len(rank_batches)} batches, "
                f"expected {self.batches_per_epoch}"
            )

        for batch in rank_batches:
            yield batch