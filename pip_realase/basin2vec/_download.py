from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
from pathlib import Path

import pooch


DATA_VERSION = "v1.0.0"

DATA_DIRECTORY_NAME = f"basin2vec-pretrained-{DATA_VERSION}"
ASSET_NAME = f"{DATA_DIRECTORY_NAME}.tar.gz"

DATA_URL = (
    "https://github.com/<user_name>/basin2vec/"
    f"releases/download/{DATA_VERSION}/{ASSET_NAME}"
)

# Replaced below using the generated .sha256 file.
DATA_SHA256 = "sha256:28eeb48bec4f7c35705995d2a11edfcfc652fc9525077df9c1478b0754262ee4"

REQUIRED_FILES = (
    "embeddings.npy",
    "usgs_ids.npy",
    "years.npy",
    "avg_embeddings.npy",
    "metadata.json",
)


def _validate_data_directory(path: Path) -> None:
    """Check that all required pretrained-data files exist."""
    if not path.is_dir():
        raise FileNotFoundError(
            f"Basin2Vec data directory does not exist: {path}"
        )

    missing = [
        filename
        for filename in REQUIRED_FILES
        if not (path / filename).is_file()
    ]

    if missing:
        raise RuntimeError(
            f"Incomplete Basin2Vec data directory: {path}. "
            f"Missing files: {missing}"
        )


def _safe_extract_tar(archive_path: Path, destination: Path) -> None:
    """
    Safely extract the Basin2Vec release archive.

    Archive members are checked to prevent paths from escaping the
    destination directory. Symbolic and hard links are rejected.
    """
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()

    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = archive.getmembers()

        for member in members:
            if member.issym() or member.islnk():
                raise RuntimeError(
                    f"Archive contains an unsupported link: {member.name}"
                )

            member_path = (destination / member.name).resolve()

            if not member_path.is_relative_to(destination_root):
                raise RuntimeError(
                    f"Unsafe archive path detected: {member.name}"
                )

        archive.extractall(destination, members=members)


def fetch_pretrained_data(
    cache_dir: str | Path | None = None,
    *,
    force_download: bool = False,
) -> Path:
    """
    Download, verify, extract, and cache pretrained Basin2Vec embeddings.

    Parameters
    ----------
    cache_dir
        Optional custom cache directory.
    force_download
        Remove the existing cached copy and download it again.

    Returns
    -------
    pathlib.Path
        Directory containing the extracted pretrained files.

    Environment variables
    ---------------------
    BASIN2VEC_DATA_DIR
        Use an already extracted local data directory.
    BASIN2VEC_CACHE
        Override the default cache directory.
    """
    local_data_directory = os.environ.get("BASIN2VEC_DATA_DIR")

    if local_data_directory:
        path = Path(local_data_directory).expanduser().resolve()
        _validate_data_directory(path)
        return path

    if cache_dir is not None:
        cache_root = Path(cache_dir).expanduser().resolve()
    else:
        cache_override = os.environ.get("BASIN2VEC_CACHE")

        if cache_override:
            cache_root = Path(cache_override).expanduser().resolve()
        else:
            cache_root = Path(pooch.os_cache("basin2vec"))

    cache_root.mkdir(parents=True, exist_ok=True)

    download_directory = cache_root / "downloads"
    download_directory.mkdir(parents=True, exist_ok=True)

    data_directory = cache_root / DATA_DIRECTORY_NAME
    downloaded_archive = download_directory / ASSET_NAME

    if force_download:
        if data_directory.exists():
            shutil.rmtree(data_directory)

        downloaded_archive.unlink(missing_ok=True)

    if data_directory.exists():
        try:
            _validate_data_directory(data_directory)
            return data_directory
        except (FileNotFoundError, RuntimeError):
            shutil.rmtree(data_directory)

    archive_path = Path(
        pooch.retrieve(
            url=DATA_URL,
            known_hash=DATA_SHA256,
            fname=ASSET_NAME,
            path=download_directory,
            progressbar=True,
        )
    )

    with tempfile.TemporaryDirectory(
        prefix="basin2vec-extract-",
        dir=cache_root,
    ) as temporary_directory:
        temporary_path = Path(temporary_directory)
        extracted_directory = temporary_path / DATA_DIRECTORY_NAME

        _safe_extract_tar(
            archive_path=archive_path,
            destination=extracted_directory,
        )

        _validate_data_directory(extracted_directory)

        if data_directory.exists():
            shutil.rmtree(data_directory)

        shutil.move(
            str(extracted_directory),
            str(data_directory),
        )

    _validate_data_directory(data_directory)
    return data_directory
