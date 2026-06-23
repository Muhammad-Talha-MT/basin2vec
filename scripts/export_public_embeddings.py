#!/usr/bin/env python3
"""
Export the internal Basin2Vec NPZ archive into a compact public format.

Input archive
-------------
Expected NPZ keys:

    embeddings
    labels
    years
    avg_embeddings
    avg_labels

Output directory
----------------
The script creates:

    embeddings.npy       (num_basins, num_years, embedding_dim)
    usgs_ids.npy         (num_basins,)
    years.npy            (num_years,)
    avg_embeddings.npy   (num_basins, embedding_dim)
    metadata.json

Example
-------
python scripts/export_public_embeddings.py \
    --source emb/src/evaluation_outputs/HCLV4_grouped_monthly_static_meta_areaaux_64d/basin_embeddings_full.npz \
    --output release_assets/basin2vec-pretrained-v1.0.0 \
    --data-version v1.0.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_KEYS = {
    "embeddings",
    "labels",
    "years",
    "avg_embeddings",
    "avg_labels",
}


def normalize_usgs_id(value: Any) -> str:
    """
    Normalize a USGS site identifier.

    Rules
    -----
    - Byte strings are decoded.
    - Integer-like values are converted to digit strings.
    - IDs shorter than eight digits are left-padded with zeros.
    - IDs with eight or more digits are preserved.
    - IDs must contain only digits.

    Examples
    --------
    4101500       -> "04101500"
    "04101500"    -> "04101500"
    "010735562"   -> "010735562"
    """
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode("utf-8")

    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)

        if not math.isfinite(numeric_value):
            raise ValueError(f"Non-finite USGS ID: {value!r}")

        if not numeric_value.is_integer():
            raise ValueError(
                f"Floating-point USGS ID is not integer-valued: {value!r}"
            )

        text = str(int(numeric_value))

    elif isinstance(value, (int, np.integer)):
        text = str(int(value))

    else:
        text = str(value).strip()

        # Handle values that were stored as strings such as "04101500.0".
        if text.endswith(".0"):
            candidate = text[:-2]

            if candidate.isdigit():
                text = candidate

    if not text:
        raise ValueError("Encountered an empty USGS ID.")

    if not text.isdigit():
        raise ValueError(
            f"USGS ID must contain digits only; received {value!r}"
        )

    # Sanity check only. Longer USGS identifiers are valid and preserved.
    if len(text) > 15:
        raise ValueError(
            f"Unexpectedly long USGS ID with {len(text)} digits: {text!r}"
        )

    if len(text) < 8:
        text = text.zfill(8)

    return text


def normalize_year(value: Any) -> int:
    """Convert a year value to a validated integer."""
    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)

        if not math.isfinite(numeric_value):
            raise ValueError(f"Non-finite year value: {value!r}")

        if not numeric_value.is_integer():
            raise ValueError(
                f"Year is not integer-valued: {value!r}"
            )

        year = int(numeric_value)
    else:
        year = int(value)

    if year < 1800 or year > 2200:
        raise ValueError(f"Unexpected year value: {year}")

    return year


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 checksum of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        for block in iter(
            lambda: file_handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def validate_source_shapes(
    row_embeddings: np.ndarray,
    row_ids: np.ndarray,
    row_years: np.ndarray,
    avg_embeddings: np.ndarray,
    avg_ids: np.ndarray,
) -> None:
    """Validate shapes of arrays loaded from the internal NPZ archive."""
    if row_embeddings.ndim != 2:
        raise ValueError(
            "Expected `embeddings` to have shape (rows, dimension), "
            f"but received {row_embeddings.shape}."
        )

    if row_ids.ndim != 1:
        raise ValueError(
            f"Expected `labels` to be one-dimensional; got {row_ids.shape}."
        )

    if row_years.ndim != 1:
        raise ValueError(
            f"Expected `years` to be one-dimensional; got {row_years.shape}."
        )

    if avg_embeddings.ndim != 2:
        raise ValueError(
            "Expected `avg_embeddings` to have shape "
            f"(basins, dimension); got {avg_embeddings.shape}."
        )

    if avg_ids.ndim != 1:
        raise ValueError(
            f"Expected `avg_labels` to be one-dimensional; got {avg_ids.shape}."
        )

    number_of_rows = row_embeddings.shape[0]

    if len(row_ids) != number_of_rows:
        raise ValueError(
            "`embeddings` and `labels` have different row counts: "
            f"{number_of_rows:,} versus {len(row_ids):,}."
        )

    if len(row_years) != number_of_rows:
        raise ValueError(
            "`embeddings` and `years` have different row counts: "
            f"{number_of_rows:,} versus {len(row_years):,}."
        )

    if len(avg_ids) != avg_embeddings.shape[0]:
        raise ValueError(
            "`avg_embeddings` and `avg_labels` have different row counts: "
            f"{avg_embeddings.shape[0]:,} versus {len(avg_ids):,}."
        )

    if row_embeddings.shape[1] != avg_embeddings.shape[1]:
        raise ValueError(
            "`embeddings` and `avg_embeddings` use different dimensions: "
            f"{row_embeddings.shape[1]} versus {avg_embeddings.shape[1]}."
        )


def load_source_archive(
    source: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
]:
    """Load and normalize arrays from the internal Basin2Vec archive."""
    print(f"Loading source archive: {source}")

    with np.load(source, allow_pickle=True) as archive:
        missing_keys = REQUIRED_KEYS.difference(archive.files)

        if missing_keys:
            raise KeyError(
                "Source archive is missing required keys: "
                f"{sorted(missing_keys)}"
            )

        row_embeddings = np.asarray(
            archive["embeddings"],
            dtype=np.float32,
        )

        normalized_row_ids = [
            normalize_usgs_id(value)
            for value in archive["labels"]
        ]

        row_years = np.asarray(
            [
                normalize_year(value)
                for value in archive["years"]
            ],
            dtype=np.int16,
        )

        avg_embeddings = np.asarray(
            archive["avg_embeddings"],
            dtype=np.float32,
        )

        normalized_avg_ids = [
            normalize_usgs_id(value)
            for value in archive["avg_labels"]
        ]

    maximum_id_length = max(
        max(len(value) for value in normalized_row_ids),
        max(len(value) for value in normalized_avg_ids),
    )

    id_dtype = f"<U{maximum_id_length}"

    row_ids = np.asarray(
        normalized_row_ids,
        dtype=id_dtype,
    )

    avg_ids = np.asarray(
        normalized_avg_ids,
        dtype=id_dtype,
    )

    validate_source_shapes(
        row_embeddings=row_embeddings,
        row_ids=row_ids,
        row_years=row_years,
        avg_embeddings=avg_embeddings,
        avg_ids=avg_ids,
    )

    return (
        row_embeddings,
        row_ids,
        row_years,
        avg_embeddings,
        avg_ids,
        maximum_id_length,
    )


def build_basin_year_grid(
    row_embeddings: np.ndarray,
    row_ids: np.ndarray,
    row_years: np.ndarray,
    maximum_id_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert row-wise embeddings into a basin × year × dimension array.
    """
    id_dtype = f"<U{maximum_id_length}"

    usgs_ids = np.asarray(
        sorted(set(str(value) for value in row_ids.tolist())),
        dtype=id_dtype,
    )

    years = np.asarray(
        sorted(set(int(value) for value in row_years.tolist())),
        dtype=np.int16,
    )

    number_of_basins = len(usgs_ids)
    number_of_years = len(years)
    embedding_dimension = row_embeddings.shape[1]

    expected_rows = number_of_basins * number_of_years
    actual_rows = len(row_embeddings)

    print()
    print("Source archive summary")
    print("----------------------")
    print(f"Rows:                 {actual_rows:,}")
    print(f"Unique basins:        {number_of_basins:,}")
    print(f"Unique years:         {number_of_years:,}")
    print(f"Year range:           {int(years.min())}–{int(years.max())}")
    print(f"Embedding dimension:  {embedding_dimension}")
    print(f"Maximum ID length:    {maximum_id_length}")
    print(f"Expected grid rows:   {expected_rows:,}")

    if actual_rows != expected_rows:
        raise ValueError(
            "The source archive does not form a complete rectangular "
            "basin-year grid.\n"
            f"Expected {expected_rows:,} rows from "
            f"{number_of_basins:,} basins × {number_of_years:,} years, "
            f"but found {actual_rows:,} rows.\n"
            "An availability mask would be required for an incomplete grid."
        )

    basin_to_index = {
        str(basin_id): index
        for index, basin_id in enumerate(usgs_ids.tolist())
    }

    year_to_index = {
        int(year): index
        for index, year in enumerate(years.tolist())
    }

    embeddings = np.empty(
        (
            number_of_basins,
            number_of_years,
            embedding_dimension,
        ),
        dtype=np.float32,
    )

    seen = np.zeros(
        (number_of_basins, number_of_years),
        dtype=bool,
    )

    for source_index in range(actual_rows):
        basin_id = str(row_ids[source_index])
        year = int(row_years[source_index])

        basin_index = basin_to_index[basin_id]
        year_index = year_to_index[year]

        if seen[basin_index, year_index]:
            raise ValueError(
                "Duplicate basin-year entry found for "
                f"USGS ID {basin_id}, year {year}."
            )

        embeddings[basin_index, year_index] = (
            row_embeddings[source_index]
        )

        seen[basin_index, year_index] = True

    if not seen.all():
        missing_positions = np.argwhere(~seen)
        missing_count = len(missing_positions)

        examples = []

        for basin_index, year_index in missing_positions[:10]:
            examples.append(
                f"{usgs_ids[basin_index]}:{int(years[year_index])}"
            )

        raise ValueError(
            f"Output grid contains {missing_count:,} missing basin-year "
            f"entries. Examples: {', '.join(examples)}"
        )

    return embeddings, usgs_ids, years


def align_average_embeddings(
    usgs_ids: np.ndarray,
    avg_embeddings: np.ndarray,
    avg_ids: np.ndarray,
) -> np.ndarray:
    """
    Align source averaged embeddings with the sorted public basin ID order.
    """
    average_lookup: dict[str, np.ndarray] = {}

    for basin_id, embedding in zip(
        avg_ids.tolist(),
        avg_embeddings,
        strict=True,
    ):
        basin_id_string = str(basin_id)

        if basin_id_string in average_lookup:
            raise ValueError(
                f"Duplicate averaged embedding for USGS ID "
                f"{basin_id_string}."
            )

        average_lookup[basin_id_string] = embedding

    public_ids = [str(value) for value in usgs_ids.tolist()]

    missing_ids = [
        basin_id
        for basin_id in public_ids
        if basin_id not in average_lookup
    ]

    extra_ids = sorted(
        set(average_lookup).difference(public_ids)
    )

    if missing_ids:
        raise ValueError(
            f"Missing averaged embeddings for {len(missing_ids):,} basins. "
            f"Examples: {missing_ids[:10]}"
        )

    if extra_ids:
        raise ValueError(
            f"Found {len(extra_ids):,} averaged embeddings with no "
            f"basin-year rows. Examples: {extra_ids[:10]}"
        )

    aligned = np.stack(
        [
            average_lookup[basin_id]
            for basin_id in public_ids
        ],
        axis=0,
    )

    return np.asarray(aligned, dtype=np.float32)


def verify_average_embeddings(
    embeddings: np.ndarray,
    avg_embeddings: np.ndarray,
) -> None:
    """
    Compare released averages with averages recomputed from basin-year rows.
    """
    recomputed = embeddings.mean(axis=1, dtype=np.float64).astype(
        np.float32
    )

    maximum_difference = float(
        np.max(np.abs(recomputed - avg_embeddings))
    )

    mean_difference = float(
        np.mean(np.abs(recomputed - avg_embeddings))
    )

    print()
    print("Average-embedding verification")
    print("------------------------------")
    print(f"Maximum absolute difference: {maximum_difference:.8f}")
    print(f"Mean absolute difference:    {mean_difference:.8f}")

    # Do not fail for small differences caused by source precision,
    # normalization, or the averaging method used during generation.
    if maximum_difference > 1e-3:
        print(
            "Warning: source avg_embeddings differ from simple means of "
            "the basin-year embeddings. The source-provided averages will "
            "be retained."
        )


def save_array(
    output_directory: Path,
    filename: str,
    array: np.ndarray,
) -> Path:
    """Save one NumPy array without pickle support."""
    path = output_directory / filename

    np.save(
        path,
        array,
        allow_pickle=False,
    )

    size_mib = path.stat().st_size / (1024**2)

    print(
        f"Saved {filename:<20} "
        f"shape={str(array.shape):<22} "
        f"dtype={str(array.dtype):<8} "
        f"size={size_mib:,.2f} MiB"
    )

    return path


def export_archive(
    source: Path,
    output_directory: Path,
    data_version: str,
) -> None:
    """Create the complete public Basin2Vec data directory."""
    if not source.is_file():
        raise FileNotFoundError(
            f"Source archive was not found: {source}"
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        row_embeddings,
        row_ids,
        row_years,
        source_avg_embeddings,
        source_avg_ids,
        maximum_id_length,
    ) = load_source_archive(source)

    embeddings, usgs_ids, years = build_basin_year_grid(
        row_embeddings=row_embeddings,
        row_ids=row_ids,
        row_years=row_years,
        maximum_id_length=maximum_id_length,
    )

    avg_embeddings = align_average_embeddings(
        usgs_ids=usgs_ids,
        avg_embeddings=source_avg_embeddings,
        avg_ids=source_avg_ids,
    )

    verify_average_embeddings(
        embeddings=embeddings,
        avg_embeddings=avg_embeddings,
    )

    print()
    print("Writing public files")
    print("--------------------")

    output_paths = {
        "embeddings.npy": save_array(
            output_directory,
            "embeddings.npy",
            embeddings,
        ),
        "usgs_ids.npy": save_array(
            output_directory,
            "usgs_ids.npy",
            usgs_ids,
        ),
        "years.npy": save_array(
            output_directory,
            "years.npy",
            years,
        ),
        "avg_embeddings.npy": save_array(
            output_directory,
            "avg_embeddings.npy",
            avg_embeddings,
        ),
    }

    metadata: dict[str, Any] = {
        "model": "Basin2Vec",
        "data_version": data_version,
        "description": (
            "Pretrained basin-year embeddings generated by Basin2Vec."
        ),
        "num_basins": int(len(usgs_ids)),
        "num_years": int(len(years)),
        "embedding_dimension": int(embeddings.shape[2]),
        "minimum_year": int(years.min()),
        "maximum_year": int(years.max()),
        "embedding_dtype": str(embeddings.dtype),
        "average_embedding_dtype": str(avg_embeddings.dtype),
        "usgs_id_dtype": str(usgs_ids.dtype),
        "usgs_id_format": (
            "USGS site identifier stored as a digit string. "
            "Identifiers shorter than eight digits are left-padded "
            "with zeros; longer identifiers are preserved."
        ),
        "maximum_usgs_id_length": int(maximum_id_length),
        "embedding_shape": [
            int(value)
            for value in embeddings.shape
        ],
        "average_embedding_shape": [
            int(value)
            for value in avg_embeddings.shape
        ],
        "complete_basin_year_grid": True,
        "year_values": [
            int(value)
            for value in years.tolist()
        ],
        "files": {},
    }

    for filename, path in output_paths.items():
        metadata["files"][filename] = {
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }

    metadata_path = output_directory / "metadata.json"

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file_handle:
        json.dump(
            metadata,
            file_handle,
            indent=2,
            sort_keys=False,
        )
        file_handle.write("\n")

    metadata_size_kib = metadata_path.stat().st_size / 1024

    print(
        f"Saved {'metadata.json':<20} "
        f"size={metadata_size_kib:,.2f} KiB"
    )

    total_size = sum(
        path.stat().st_size
        for path in output_directory.iterdir()
        if path.is_file()
    )

    print()
    print("Export completed successfully")
    print("-----------------------------")
    print(f"Output directory: {output_directory}")
    print(f"Total size:       {total_size / (1024**2):,.2f} MiB")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Average shape:    {avg_embeddings.shape}")
    print(f"ID shape:         {usgs_ids.shape}")
    print(f"Year shape:       {years.shape}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert an internal Basin2Vec NPZ archive into the "
            "public pretrained embedding format."
        )
    )

    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to basin_embeddings_full.npz.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "release_assets/basin2vec-pretrained-v1.0.0"
        ),
        help="Output directory for public embedding files.",
    )

    parser.add_argument(
        "--data-version",
        type=str,
        default="v1.0.0",
        help="Data-release version stored in metadata.json.",
    )

    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    arguments = parse_arguments()

    export_archive(
        source=arguments.source,
        output_directory=arguments.output,
        data_version=arguments.data_version,
    )


if __name__ == "__main__":
    main()