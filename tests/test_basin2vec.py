"""Tests for the public Basin2Vec package using a small synthetic archive."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from basin2vec import Basin2Vec, normalize_usgs_id
from basin2vec._download import fetch_pretrained_data


USGS_IDS = np.array(["01013500", "04101500", "010735562"])
YEARS = np.arange(2000, 2005, dtype=np.int32)
DIM = 8


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    embeddings = rng.normal(
        size=(len(USGS_IDS), len(YEARS), DIM)
    ).astype(np.float32)

    np.save(tmp_path / "embeddings.npy", embeddings)
    np.save(tmp_path / "usgs_ids.npy", USGS_IDS)
    np.save(tmp_path / "years.npy", YEARS)
    np.save(tmp_path / "avg_embeddings.npy", embeddings.mean(axis=1))

    metadata = {"embedding_dimension": DIM, "data_version": "test"}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))

    return tmp_path


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("04101500", "04101500"),
        (4101500, "04101500"),
        (np.int64(4101500), "04101500"),
        (" 04101500 ", "04101500"),
        ("010735562", "010735562"),
    ],
)
def test_normalize_usgs_id(value, expected):
    assert normalize_usgs_id(value) == expected


@pytest.mark.parametrize("value", ["", "04A01500", "1" * 16])
def test_normalize_usgs_id_rejects_invalid(value):
    with pytest.raises(ValueError):
        normalize_usgs_id(value)


def test_normalize_usgs_id_rejects_wrong_type():
    with pytest.raises(TypeError):
        normalize_usgs_id(4101500.0)


def test_lookup(data_dir: Path):
    b2v = Basin2Vec.from_pretrained(data_dir=data_dir)
    embeddings = np.load(data_dir / "embeddings.npy")

    assert len(b2v) == 3
    assert b2v.embedding_dim == DIM
    assert b2v.years == tuple(YEARS.tolist())
    assert (b2v.minimum_year, b2v.maximum_year) == (2000, 2004)
    assert repr(b2v) == (
        "Basin2Vec(basins=3, years=5, dimension=8, data_version='test')"
    )

    z = b2v["04101500", 2002]
    assert z.shape == (DIM,)
    assert z.dtype == np.float32
    np.testing.assert_array_equal(z, embeddings[1, 2])
    np.testing.assert_array_equal(b2v[4101500, 2002], z)
    np.testing.assert_array_equal(b2v.get("04101500", year=2002), z)

    np.testing.assert_allclose(
        b2v["04101500"], embeddings[1].mean(axis=0), rtol=1e-6
    )


def test_returned_arrays_are_copies(data_dir: Path):
    b2v = Basin2Vec.from_pretrained(data_dir=data_dir)
    z = b2v["04101500", 2002]
    z[:] = 0
    assert not np.all(b2v["04101500", 2002] == 0)


def test_membership(data_dir: Path):
    b2v = Basin2Vec.from_pretrained(data_dir=data_dir)

    assert "04101500" in b2v
    assert 4101500 in b2v
    assert "99999999" not in b2v
    assert "not-an-id" not in b2v
    assert b2v.available_years("010735562") == tuple(YEARS.tolist())


def test_missing_basin_and_year(data_dir: Path):
    b2v = Basin2Vec.from_pretrained(data_dir=data_dir)

    with pytest.raises(KeyError):
        b2v["99999999", 2002]
    with pytest.raises(KeyError):
        b2v["04101500", 1999]
    with pytest.raises(KeyError):
        b2v.available_years("99999999")


def test_invalid_shape_is_rejected(data_dir: Path):
    np.save(data_dir / "avg_embeddings.npy", np.zeros((2, DIM), np.float32))
    with pytest.raises(RuntimeError):
        Basin2Vec(data_dir)


def test_missing_metadata(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        Basin2Vec(tmp_path)


def test_env_data_dir(data_dir: Path, monkeypatch):
    monkeypatch.setenv("BASIN2VEC_DATA_DIR", str(data_dir))
    assert fetch_pretrained_data() == data_dir.resolve()
    assert len(Basin2Vec.from_pretrained()) == 3


def test_env_data_dir_incomplete(data_dir: Path, monkeypatch):
    (data_dir / "years.npy").unlink()
    monkeypatch.setenv("BASIN2VEC_DATA_DIR", str(data_dir))
    with pytest.raises(RuntimeError):
        fetch_pretrained_data()
