from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .config import StudyConfig
from .features import peaks_to_sparse_matrix, read_float32_profile_matrix, write_feature_bundle
from .manifest import manifest_from_export
from .taxonomy import assign_analysis_sets, load_taxonomy_workbook, taxonomy_summary
from .util import atomic_write_json, sha256_file, utc_now


def locate_single(root: str | Path, pattern: str) -> Path:
    matches = sorted(Path(root).rglob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {pattern!r} below {root}, found {len(matches)}")
    return matches[0]


def run_r_exporter(
    extracted_root: str | Path,
    peak_csv: str | Path,
    exporter_manifest: str | Path,
    config_path: str | Path,
    script_path: str | Path,
    rscript: str | Path,
    profile_matrix: str | Path,
) -> None:
    command = [
        str(rscript),
        str(script_path),
        str(extracted_root),
        str(peak_csv),
        str(exporter_manifest),
        str(config_path),
        str(profile_matrix),
    ]
    environment = os.environ.copy()
    for key in ("LC_ALL", "LC_COLLATE", "LC_CTYPE", "LC_MONETARY", "LC_TIME"):
        environment[key] = "English_United States.utf8"
    completed = subprocess.run(
        command, check=False, text=True, capture_output=True, env=environment
    )
    log_path = Path(exporter_manifest).with_suffix(".log")
    log_path.write_text(completed.stdout + "\n--- STDERR ---\n" + completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"R importer failed with exit code {completed.returncode}; see {log_path}")


def prepare_taxonomy_only(
    taxonomy_workbook: str | Path,
    output_path: str | Path,
    config: StudyConfig,
) -> dict:
    spectra, species = load_taxonomy_workbook(taxonomy_workbook)
    manifest = assign_analysis_sets(
        spectra,
        known_min_strains=config.known_min_strains,
        sensitivity_min_strains=config.sensitivity_min_strains,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(output_path, index=False)
    species.to_csv(output_path.with_name("species_counts.csv"), index=False)
    summary = taxonomy_summary(manifest)
    summary.update(
        {
            "created_at": utc_now(),
            "taxonomy_workbook": str(Path(taxonomy_workbook).resolve()),
            "taxonomy_sha256": sha256_file(taxonomy_workbook),
        }
    )
    atomic_write_json(output_path.with_name("taxonomy_summary.json"), summary)
    return summary


def prepare_feature_bundle(
    peak_table_path: str | Path,
    exporter_manifest_path: str | Path,
    taxonomy_workbook: str | Path,
    output_dir: str | Path,
    config: StudyConfig,
    extracted_root: str | Path | None = None,
    acquisition_metadata_path: str | Path | None = None,
    species_aliases_path: str | Path | None = None,
    profile_matrix_path: str | Path | None = None,
) -> dict:
    peak_table_path = Path(peak_table_path)
    if peak_table_path.suffix.lower() == ".parquet":
        peaks = pd.read_parquet(peak_table_path)
    else:
        peaks = pd.read_csv(peak_table_path)
    manifest = manifest_from_export(
        exporter_manifest_path,
        taxonomy_workbook,
        extracted_root=extracted_root,
        acquisition_metadata_path=acquisition_metadata_path,
        species_aliases_path=species_aliases_path,
        known_min_strains=config.known_min_strains,
        sensitivity_min_strains=config.sensitivity_min_strains,
    )
    included = manifest.loc[manifest["included"]].copy().reset_index(drop=True)
    available = set(peaks["spectrum_id"].astype(str))
    included = included.loc[included["spectrum_id"].astype(str).isin(available)].reset_index(drop=True)
    profile_path = Path(profile_matrix_path) if profile_matrix_path else None
    if profile_path is not None and profile_path.is_file():
        full_matrix = read_float32_profile_matrix(
            profile_path,
            n_rows=len(manifest),
            n_bins=int((config.mass_max - config.mass_min) / config.bin_width),
        )
        matrix = np.asarray(
            full_matrix[included["profile_row"].to_numpy(dtype=int)], dtype=np.float32
        )
        row_sums = matrix.sum(axis=1)
        if not np.all(np.isfinite(matrix)) or not np.allclose(row_sums, 1.0, atol=1e-6):
            raise ValueError("profile matrix contains nonfinite values or violates TIC row-sum invariant")
    else:
        matrix = peaks_to_sparse_matrix(
            peaks,
            included["spectrum_id"].astype(str).tolist(),
            config.mass_min,
            config.mass_max,
            config.bin_width,
        )
    parameters = {
        "mass_min": config.mass_min,
        "mass_max": config.mass_max,
        "bin_width": config.bin_width,
        "preprocessing": config.preprocessing,
        "source_peak_table": str(peak_table_path.resolve()),
        "source_peak_table_sha256": sha256_file(peak_table_path),
        "exporter_manifest_sha256": sha256_file(exporter_manifest_path),
        "taxonomy_workbook_sha256": sha256_file(taxonomy_workbook),
        "acquisition_metadata_sha256": (
            sha256_file(acquisition_metadata_path)
            if acquisition_metadata_path and Path(acquisition_metadata_path).is_file()
            else None
        ),
        "species_aliases_sha256": (
            sha256_file(species_aliases_path)
            if species_aliases_path and Path(species_aliases_path).is_file()
            else None
        ),
        "profile_matrix_sha256": (
            sha256_file(profile_path) if profile_path is not None and profile_path.is_file() else None
        ),
        "feature_representation": (
            "full-profile-fixed-grid"
            if profile_path is not None and profile_path.is_file()
            else "detected-peaks-fixed-grid"
        ),
    }
    metadata = write_feature_bundle(matrix, included, output_dir, parameters)
    summary = taxonomy_summary(included)
    summary.update(metadata)
    atomic_write_json(Path(output_dir) / "dataset_summary.json", summary)
    manifest.to_parquet(Path(output_dir) / "manifest_all_records.parquet", index=False)
    return summary
