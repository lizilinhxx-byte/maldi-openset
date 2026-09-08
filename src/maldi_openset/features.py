from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .util import atomic_write_json, sha256_file


REQUIRED_PEAK_COLUMNS = {"spectrum_id", "mass", "intensity"}


def peaks_to_sparse_matrix(
    peaks: pd.DataFrame,
    spectrum_order: list[str],
    mass_min: float,
    mass_max: float,
    bin_width: float,
) -> sparse.csr_matrix:
    missing = REQUIRED_PEAK_COLUMNS.difference(peaks.columns)
    if missing:
        raise KeyError(f"peak table is missing columns: {sorted(missing)}")
    if mass_min >= mass_max or bin_width <= 0:
        raise ValueError("invalid mass range or bin width")
    n_bins = int(np.ceil((mass_max - mass_min) / bin_width))
    row_lookup = {spectrum_id: idx for idx, spectrum_id in enumerate(spectrum_order)}
    work = peaks.loc[
        peaks["spectrum_id"].isin(row_lookup)
        & peaks["mass"].ge(mass_min)
        & peaks["mass"].lt(mass_max)
        & np.isfinite(peaks["intensity"])
        & peaks["intensity"].gt(0),
        ["spectrum_id", "mass", "intensity"],
    ].copy()
    work["row"] = work["spectrum_id"].map(row_lookup).astype(int)
    work["col"] = np.floor((work["mass"] - mass_min) / bin_width).astype(int)
    grouped = work.groupby(["row", "col"], sort=False, as_index=False)["intensity"].sum()
    matrix = sparse.coo_matrix(
        (
            grouped["intensity"].to_numpy(dtype=np.float32),
            (grouped["row"].to_numpy(), grouped["col"].to_numpy()),
        ),
        shape=(len(spectrum_order), n_bins),
        dtype=np.float32,
    ).tocsr()
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    nonzero = row_sums > 0
    matrix[nonzero] = sparse.diags(1.0 / row_sums[nonzero]) @ matrix[nonzero]
    return matrix


def write_feature_bundle(
    matrix,
    manifest: pd.DataFrame,
    output_dir: str | Path,
    parameters: dict,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if matrix.shape[0] != len(manifest):
        raise ValueError("feature rows must match manifest rows")
    is_sparse = sparse.issparse(matrix)
    matrix_path = output_dir / ("features.npz" if is_sparse else "features.npy")
    manifest_path = output_dir / "manifest.parquet"
    if is_sparse:
        sparse.save_npz(matrix_path, matrix.tocsr(), compressed=True)
        nnz = int(matrix.nnz)
    else:
        matrix = np.asarray(matrix, dtype=np.float32)
        np.save(matrix_path, matrix, allow_pickle=False)
        nnz = int(np.count_nonzero(matrix))
    manifest.to_parquet(manifest_path, index=False)
    metadata = {
        "shape": list(matrix.shape),
        "storage": "sparse_npz" if is_sparse else "dense_npy",
        "nnz": nnz,
        "density": float(nnz / np.prod(matrix.shape)),
        "parameters": parameters,
        "features_sha256": sha256_file(matrix_path),
        "manifest_sha256": sha256_file(manifest_path),
    }
    atomic_write_json(output_dir / "feature_metadata.json", metadata)
    return metadata


def load_feature_bundle(path: str | Path) -> tuple[sparse.csr_matrix, pd.DataFrame, dict]:
    path = Path(path)
    if (path / "features.npy").exists():
        matrix = np.load(path / "features.npy", mmap_mode="r", allow_pickle=False)
    else:
        matrix = sparse.load_npz(path / "features.npz").tocsr()
    manifest = pd.read_parquet(path / "manifest.parquet")
    with (path / "feature_metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if matrix.shape[0] != len(manifest):
        raise ValueError("corrupt feature bundle: row mismatch")
    return matrix, manifest, metadata


def read_float32_profile_matrix(path: str | Path, n_rows: int, n_bins: int) -> np.memmap:
    path = Path(path)
    expected = int(n_rows) * int(n_bins) * np.dtype("<f4").itemsize
    if path.stat().st_size != expected:
        raise ValueError(f"profile matrix size mismatch: {path.stat().st_size} != {expected}")
    return np.memmap(path, mode="r", dtype="<f4", shape=(int(n_rows), int(n_bins)))
