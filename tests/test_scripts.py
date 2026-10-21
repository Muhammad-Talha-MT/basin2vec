"""Round-trip test for the public-archive export/import scripts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def run(script: str, *args: str | Path) -> None:
    subprocess.run(
        [sys.executable, str(SCRIPTS / script), *map(str, args)],
        check=True,
        capture_output=True,
    )


def test_export_import_round_trip(tmp_path: Path):
    rng = np.random.default_rng(0)
    usgs_ids = np.array(["01013500", "010735562", "04101500"])  # exporter sorts IDs
    years = np.arange(2000, 2004)

    grid = rng.normal(size=(len(usgs_ids), len(years), 8)).astype(np.float32)
    grid /= np.linalg.norm(grid, axis=-1, keepdims=True)
    avg = grid.mean(axis=1)
    avg /= np.linalg.norm(avg, axis=-1, keepdims=True)

    source = tmp_path / "rows.npz"
    np.savez(
        source,
        embeddings=grid.reshape(-1, 8),
        labels=np.repeat(usgs_ids, len(years)),
        years=np.tile(years, len(usgs_ids)),
        avg_embeddings=avg,
        avg_labels=usgs_ids,
    )

    public = tmp_path / "public"
    run("export_public_embeddings.py", "--source", source, "--output", public)

    np.testing.assert_array_equal(np.load(public / "embeddings.npy"), grid)
    np.testing.assert_array_equal(np.load(public / "usgs_ids.npy"), usgs_ids)

    restored = tmp_path / "restored.npz"
    run("import_public_embeddings.py", "--source", public, "--output", restored)

    with np.load(restored) as rows, np.load(source) as original:
        for key in original.files:
            np.testing.assert_array_equal(rows[key], original[key])
