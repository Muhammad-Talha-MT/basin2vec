from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ._download import fetch_pretrained_data


def normalize_usgs_id(
    value: str | int | np.integer[Any],
) -> str:
    """
    Normalize a USGS site identifier.

    Identifiers shorter than eight digits are left-padded with zeros.
    Longer identifiers are preserved.

    Notes
    -----
    Pass IDs containing meaningful leading zeros as strings.

    Examples
    --------
    >>> normalize_usgs_id("04101500")
    '04101500'
    >>> normalize_usgs_id(4101500)
    '04101500'
    >>> normalize_usgs_id("010735562")
    '010735562'
    """
    if isinstance(value, (int, np.integer)):
        text = str(int(value))
    elif isinstance(value, str):
        text = value.strip()
    else:
        raise TypeError(
            "USGS ID must be a string or integer, "
            f"not {type(value).__name__}."
        )

    if not text:
        raise ValueError("USGS ID cannot be empty.")

    if not text.isdigit():
        raise ValueError(
            f"USGS ID must contain digits only; received {value!r}."
        )

    if len(text) > 15:
        raise ValueError(
            f"Unexpectedly long USGS ID: {text!r}."
        )

    return text.zfill(8) if len(text) < 8 else text


class Basin2Vec:
    """
    Indexed access to pretrained Basin2Vec embeddings.

    Create an instance using:

    >>> b2v = Basin2Vec.from_pretrained()

    Retrieve a basin-year embedding using:

    >>> z = b2v["04101500", 2018]

    Retrieve the basin-level average using:

    >>> z_avg = b2v["04101500"]
    """

    def __init__(self, data_directory: str | Path) -> None:
        self.data_directory = (
            Path(data_directory).expanduser().resolve()
        )

        metadata_path = self.data_directory / "metadata.json"

        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"metadata.json was not found in {self.data_directory}"
            )

        with metadata_path.open("r", encoding="utf-8") as stream:
            self.metadata: dict[str, Any] = json.load(stream)

        self._embeddings = np.load(
            self.data_directory / "embeddings.npy",
            mmap_mode="r",
            allow_pickle=False,
        )

        self._usgs_ids = np.load(
            self.data_directory / "usgs_ids.npy",
            allow_pickle=False,
        )

        self._years = np.load(
            self.data_directory / "years.npy",
            allow_pickle=False,
        )

        self._avg_embeddings = np.load(
            self.data_directory / "avg_embeddings.npy",
            mmap_mode="r",
            allow_pickle=False,
        )

        self._validate()

        self._basin_to_index = {
            str(basin_id): index
            for index, basin_id in enumerate(self._usgs_ids.tolist())
        }

        self._year_to_index = {
            int(year): index
            for index, year in enumerate(self._years.tolist())
        }

    @classmethod
    def from_pretrained(
        cls,
        *,
        cache_dir: str | Path | None = None,
        force_download: bool = False,
        data_dir: str | Path | None = None,
    ) -> "Basin2Vec":
        """
        Load pretrained Basin2Vec embeddings.

        On first use, the archive is downloaded from the versioned GitHub
        Release and cached locally.

        Parameters
        ----------
        cache_dir
            Optional custom cache location.
        force_download
            Download and extract the data again.
        data_dir
            Load an already extracted local public archive. This is useful
            for testing and offline use.
        """
        if data_dir is not None:
            return cls(data_dir)

        data_directory = fetch_pretrained_data(
            cache_dir=cache_dir,
            force_download=force_download,
        )

        return cls(data_directory)

    def _validate(self) -> None:
        """Validate the public pretrained arrays."""
        if self._usgs_ids.ndim != 1:
            raise RuntimeError(
                f"Invalid usgs_ids.npy shape: {self._usgs_ids.shape}"
            )

        if self._years.ndim != 1:
            raise RuntimeError(
                f"Invalid years.npy shape: {self._years.shape}"
            )

        embedding_dimension = int(
            self.metadata["embedding_dimension"]
        )

        expected_embedding_shape = (
            len(self._usgs_ids),
            len(self._years),
            embedding_dimension,
        )

        if self._embeddings.shape != expected_embedding_shape:
            raise RuntimeError(
                f"Invalid embeddings shape {self._embeddings.shape}; "
                f"expected {expected_embedding_shape}."
            )

        expected_average_shape = (
            len(self._usgs_ids),
            embedding_dimension,
        )

        if self._avg_embeddings.shape != expected_average_shape:
            raise RuntimeError(
                f"Invalid average-embedding shape "
                f"{self._avg_embeddings.shape}; "
                f"expected {expected_average_shape}."
            )

        basin_ids = [
            str(value)
            for value in self._usgs_ids.tolist()
        ]

        if len(set(basin_ids)) != len(basin_ids):
            raise RuntimeError(
                "Duplicate USGS IDs were found in usgs_ids.npy."
            )

        years = [
            int(value)
            for value in self._years.tolist()
        ]

        if len(set(years)) != len(years):
            raise RuntimeError(
                "Duplicate years were found in years.npy."
            )

    def __getitem__(
        self,
        key: tuple[str | int, int] | str | int,
    ) -> np.ndarray:
        """
        Retrieve a basin-year or basin-average embedding.

        Examples
        --------
        >>> b2v["04101500", 2018]
        >>> b2v["04101500"]
        """
        if isinstance(key, tuple):
            if len(key) != 2:
                raise KeyError(
                    "Use b2v[usgs_id, year] to retrieve "
                    "a basin-year embedding."
                )

            usgs_id, year = key
            return self.get(usgs_id, year=int(year))

        return self.get(key)

    def get(
        self,
        usgs_id: str | int,
        year: int | None = None,
    ) -> np.ndarray:
        """
        Retrieve one 64-dimensional embedding.

        When ``year`` is omitted, the basin-level averaged embedding is
        returned.
        """
        basin_id = normalize_usgs_id(usgs_id)

        try:
            basin_index = self._basin_to_index[basin_id]
        except KeyError as error:
            raise KeyError(
                f"USGS basin {basin_id} is not available in Basin2Vec."
            ) from error

        if year is None:
            return np.asarray(
                self._avg_embeddings[basin_index],
                dtype=np.float32,
            ).copy()

        try:
            year_index = self._year_to_index[int(year)]
        except KeyError as error:
            raise KeyError(
                f"Year {year} is not available. "
                f"Available range: {self.minimum_year}–"
                f"{self.maximum_year}."
            ) from error

        return np.asarray(
            self._embeddings[basin_index, year_index],
            dtype=np.float32,
        ).copy()

    def available_years(
        self,
        usgs_id: str | int,
    ) -> tuple[int, ...]:
        """Return all years available for a basin."""
        basin_id = normalize_usgs_id(usgs_id)

        if basin_id not in self._basin_to_index:
            raise KeyError(
                f"USGS basin {basin_id} is not available in Basin2Vec."
            )

        return self.years

    def has_basin(self, usgs_id: str | int) -> bool:
        """Return whether a USGS basin is available."""
        basin_id = normalize_usgs_id(usgs_id)
        return basin_id in self._basin_to_index

    @property
    def basin_ids(self) -> tuple[str, ...]:
        """Return all available USGS basin identifiers."""
        return tuple(
            str(value)
            for value in self._usgs_ids.tolist()
        )

    @property
    def years(self) -> tuple[int, ...]:
        """Return all available embedding years."""
        return tuple(
            int(value)
            for value in self._years.tolist()
        )

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return int(self._embeddings.shape[-1])

    @property
    def minimum_year(self) -> int:
        return int(self._years.min())

    @property
    def maximum_year(self) -> int:
        return int(self._years.max())

    def __len__(self) -> int:
        """Return the number of basins."""
        return len(self._usgs_ids)

    def __contains__(self, usgs_id: object) -> bool:
        try:
            return self.has_basin(usgs_id)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False

    def __repr__(self) -> str:
        return (
            "Basin2Vec("
            f"basins={len(self):,}, "
            f"years={len(self._years)}, "
            f"dimension={self.embedding_dim}, "
            f"data_version={self.metadata.get('data_version')!r}"
            ")"
        )
