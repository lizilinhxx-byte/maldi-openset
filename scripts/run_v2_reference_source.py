#!/usr/bin/env python3
"""Run the v2 fit-only reference library and source transport analyses."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
from pathlib import Path

# Fix numerical-library thread ceilings before importing NumPy/sklearn.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from maldi_openset_v2.reference_source import (
    DEFAULT_SEEDS,
    EXPECTED_FEATURE_SHA256,
    ReferenceLibrary,
    atomic_json,
    build_source_cohorts,
    build_source_splits,
    canonical_sha256,
    classification_metrics,
    fit_extra_trees,
    known_calibration_threshold,
    runtime_record,
    sha256_file,
    utc_now,
)


# Analytical checkpoints were generated with this runner revision. Subsequent
# changes below affect provenance/consolidation only; pinning that exact digest
# allows validated predictions to be re-consolidated without refitting models.
CHECKPOINT_RUNNER_SHA256 = "747a1db531de925e55743dcf63705568bfc286f1ec81c6b2e9ddd9df1af28aed"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--phase", choices=("main", "source", "all"), default="all")
    parser.add_argument("--backend", choices=("auto", "numpy", "torch"), default="auto")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cell-workers", type=int, default=1)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--folds", type=int, nargs="*", default=None)
    return parser.parse_args()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _indices(manifest: pd.DataFrame, spectrum_ids: pd.Series) -> np.ndarray:
    lookup = pd.Series(
        np.arange(len(manifest), dtype=np.int64),
        index=manifest["spectrum_id"].astype(str),
    )
    requested = pd.Index(spectrum_ids.astype(str))
    missing = requested.difference(lookup.index)
    if len(missing):
        raise KeyError(f"{len(missing)} spectrum IDs absent from feature manifest")
    return lookup.loc[requested].to_numpy(dtype=np.int64)


def _compact_rows(rows: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "spectrum_id",
        "strain_id",
        "species",
        "genus",
        "instrum",
        "acquisition_date",
        "source_lab",
        "cmt_identity_conflict",
        "ood_distance",
    ]
    return rows[[column for column in columns if column in rows]].copy().reset_index(drop=True)


def _source_hash(root: Path) -> str:
    """Hash the analytical implementation that created reusable checkpoints."""

    module = root / "src/maldi_openset_v2/reference_source.py"
    digest_parts = [
        {"path": module.relative_to(root).as_posix(), "sha256": sha256_file(module)},
        {
            "path": "scripts/run_v2_reference_source.py",
            "sha256": CHECKPOINT_RUNNER_SHA256,
        },
    ]
    return canonical_sha256(digest_parts)


def _current_git_provenance(root: Path) -> dict[str, object]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1"], cwd=root, text=True, encoding="utf-8"
    )
    return {
        "git_commit": head,
        "git_worktree_dirty": bool(status.strip()),
        "git_status_sha256": canonical_sha256(status.splitlines()),
        "runner_sha256": sha256_file(root / "scripts/run_v2_reference_source.py"),
    }


def _verify_frozen_input_manifest(root: Path, spec: dict[str, object]) -> dict[str, object]:
    """Verify the frozen v1 feature and split identities before any analysis."""

    path = root / "output/v2/input_manifest.json"
    frozen = json.loads(path.read_text(encoding="utf-8"))
    expected_commit = str(spec["legacy_analysis_commit"])
    if frozen.get("schema_version") != "2.0.0":
        raise RuntimeError("unexpected frozen-input manifest schema")
    if frozen.get("legacy_analysis_commit") != expected_commit:
        raise RuntimeError("frozen-input analysis commit differs from the v2 spec")
    if frozen.get("feature_sha256") != EXPECTED_FEATURE_SHA256:
        raise RuntimeError("frozen-input feature identity differs")
    feature_path = root / str(frozen.get("feature_matrix_path"))
    if feature_path != (root / "data/processed/production/features.npy"):
        raise RuntimeError("frozen-input feature path differs")
    if feature_path.stat().st_size != int(frozen.get("feature_matrix_bytes", -1)):
        raise RuntimeError("frozen feature byte count differs")
    feature_actual = sha256_file(feature_path)
    if feature_actual != EXPECTED_FEATURE_SHA256:
        raise RuntimeError("locked feature matrix byte hash mismatch")

    file_index = {str(row["path"]): row for row in frozen.get("files", [])}
    required = (
        "config/v2/analysis_spec.json",
        "data/processed/production/manifest.parquet",
        "data/processed/production/feature_metadata.json",
        "data/processed/splits.json",
        "data/processed/splits.parquet",
    )
    verified: dict[str, str] = {}
    for relative in required:
        if relative not in file_index:
            raise RuntimeError(f"frozen-input manifest lacks {relative}")
        target = root / relative
        record = file_index[relative]
        if target.stat().st_size != int(record["bytes"]):
            raise RuntimeError(f"frozen-input byte count differs: {relative}")
        actual = sha256_file(target)
        if actual != record["sha256"]:
            raise RuntimeError(f"frozen-input hash differs: {relative}")
        verified[relative] = actual
    metadata = json.loads(
        (root / "data/processed/production/feature_metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("features_sha256") != feature_actual:
        raise RuntimeError("feature metadata and frozen matrix identity differ")
    return {
        "frozen_input_manifest_sha256": sha256_file(path),
        "frozen_input_analysis_commit": expected_commit,
        "feature_hash_verified": True,
        "frozen_files_verified": verified,
    }


def _cell_complete(run_dir: Path, fingerprint: str) -> bool:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return False
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    if record.get("status") != "complete" or record.get("input_fingerprint") != fingerprint:
        return False
    for field, filename in (
        ("predictions_sha256", "predictions.parquet"),
        ("metrics_sha256", "metrics.json"),
        ("classes_sha256", "classes.json"),
    ):
        path = run_dir / filename
        if not path.is_file() or record.get(field) != sha256_file(path):
            return False
    return True


def _refresh_cell_provenance(
    run_dir: Path,
    *,
    provenance: dict[str, object],
    frozen_provenance: dict[str, object],
) -> None:
    path = run_dir / "run_manifest.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    legacy = record.pop("legacy_analysis_commit", None)
    expected_legacy = frozen_provenance["frozen_input_analysis_commit"]
    if legacy is not None and legacy != expected_legacy:
        raise RuntimeError(f"legacy input commit mismatch in {run_dir.name}")
    record.update(provenance)
    record.update(frozen_provenance)
    atomic_json(path, record)


def _upgrade_source_checkpoint_reporting(run_dir: Path, holdout_label: str) -> None:
    """Promote target-label spectra to the primary transport estimand.

    Existing model predictions remain untouched except for an explicit analysis
    population label. The 955 non-MALDIMESS spectra from quarantined mixed
    strains are retained as auxiliary rows rather than entering the main result.
    """

    metric_path = run_dir / "metrics.json"
    prediction_path = run_dir / "predictions.parquet"
    run_manifest_path = run_dir / "run_manifest.json"
    metrics = json.loads(metric_path.read_text(encoding="utf-8"))
    predictions = pd.read_parquet(prediction_path)
    target = predictions["role"].eq("test_source") & predictions["instrum"].astype(str).eq(
        holdout_label
    )
    auxiliary = predictions["role"].eq("test_source") & ~predictions["instrum"].astype(str).eq(
        holdout_label
    )
    predictions["source_analysis_population"] = np.select(
        [target, auxiliary],
        ["target-label-primary", "other-label-quarantine-auxiliary"],
        default="calibration",
    )
    if "test_source_target_label_spectra" in metrics:
        whole = metrics.pop("test_source")
        primary = metrics.pop("test_source_target_label_spectra")
        metrics["test_source"] = primary
        metrics["test_source_whole_strain_quarantine_auxiliary"] = whole
    elif metrics.get("source_primary_population") != "target-label-only":
        raise RuntimeError(f"source checkpoint lacks target-label metrics: {run_dir.name}")
    metrics["source_primary_population"] = "target-label-only"
    metrics["source_primary_test_spectra"] = int(target.sum())
    metrics["source_quarantine_auxiliary_spectra"] = int(auxiliary.sum())
    metrics["interpretation"] = (
        "within-dataset source/instrument-era transport on target-label spectra; "
        "other-label spectra from quarantined strains are auxiliary only"
    )
    _atomic_parquet(predictions, prediction_path)
    atomic_json(metric_path, metrics)
    record = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    record["predictions_sha256"] = sha256_file(prediction_path)
    record["metrics_sha256"] = sha256_file(metric_path)
    record["source_primary_population"] = "target-label-only"
    record["source_primary_test_spectra"] = int(target.sum())
    record["source_quarantine_auxiliary_spectra"] = int(auxiliary.sum())
    atomic_json(run_manifest_path, record)


def _write_cell(
    run_dir: Path,
    predictions: pd.DataFrame,
    metrics: dict[str, object],
    classes: np.ndarray,
    manifest: dict[str, object],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = run_dir / "predictions.parquet"
    metric_path = run_dir / "metrics.json"
    classes_path = run_dir / "classes.json"
    _atomic_parquet(predictions, prediction_path)
    atomic_json(metric_path, metrics)
    atomic_json(classes_path, np.asarray(classes, dtype=str).tolist())
    manifest = dict(manifest)
    manifest.update(
        {
            "status": "complete",
            "completed_at": utc_now(),
            "predictions_sha256": sha256_file(prediction_path),
            "metrics_sha256": sha256_file(metric_path),
            "classes_sha256": sha256_file(classes_path),
        }
    )
    atomic_json(run_dir / "run_manifest.json", manifest)


def _apply_reference_predictions(
    matrix: np.ndarray,
    manifest: pd.DataFrame,
    assignment: pd.DataFrame,
    *,
    backend: str,
    block_size: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, float, int, ReferenceLibrary]:
    role_indices: dict[str, np.ndarray] = {}
    role_rows: dict[str, pd.DataFrame] = {}
    for role, frame in assignment.groupby("role", sort=False):
        idx = _indices(manifest, frame["spectrum_id"])
        role_indices[str(role)] = idx
        role_rows[str(role)] = manifest.iloc[idx].reset_index(drop=True)
    for role in ("train", "calibration"):
        if role not in role_indices:
            raise ValueError(f"missing required role: {role}")
    prediction_roles = [role for role in ("calibration", "test_known", "test_ood", "test_source") if role in role_indices]

    library = ReferenceLibrary.fit(matrix[role_indices["train"]], role_rows["train"])
    query_indices = np.concatenate([role_indices[role] for role in prediction_roles])
    query_rows = pd.concat([role_rows[role] for role in prediction_roles], ignore_index=True)
    role_vector = np.concatenate(
        [np.repeat(role, len(role_indices[role])) for role in prediction_roles]
    )
    compact = _compact_rows(query_rows)
    scored, scores = library.score(
        matrix[query_indices], backend=backend, block_size=block_size
    )
    predictions = pd.concat([compact, scored], axis=1)
    predictions["role"] = role_vector
    cal_mask = predictions["role"].eq("calibration").to_numpy()
    threshold, calibration_strains = known_calibration_threshold(
        predictions.loc[cal_mask, "score"],
        predictions.loc[cal_mask, "strain_id"],
        target=0.95,
    )
    predictions["acceptance_threshold"] = float(threshold)
    predictions["threshold_source"] = "known-calibration-strain-median-max-score"
    predictions["accepted"] = predictions["score"].ge(threshold)
    predictions["true_is_fitted"] = predictions["species"].astype(str).isin(
        set(library.species_classes)
    )
    predictions["correct_forced_choice"] = predictions["predicted_species"].astype(str).eq(
        predictions["species"].astype(str)
    )
    return predictions, scores, query_indices, threshold, calibration_strains, library


def _metrics_for_reference(
    predictions: pd.DataFrame,
    scores: np.ndarray,
    classes: np.ndarray,
) -> dict[str, object]:
    output: dict[str, object] = {}
    for role in ("calibration", "test_known", "test_source"):
        mask = predictions["role"].eq(role).to_numpy()
        if mask.any():
            output[role] = classification_metrics(
                predictions.loc[mask].reset_index(drop=True),
                score_columns=scores[mask],
                classes=classes,
            )
    ood_mask = predictions["role"].eq("test_ood")
    if ood_mask.any():
        ood = predictions.loc[ood_mask].copy()
        output["test_ood"] = {
            "n_spectra": int(len(ood)),
            "n_strains": int(ood["strain_id"].nunique()),
            "n_species": int(ood["species"].nunique()),
            "false_acceptance_count": int(ood["accepted"].sum()),
            "false_acceptance_rate": float(ood["accepted"].mean()),
        }
        distance = ood["ood_distance"].replace({"near": "same-genus", "far": "different-genus"})
        for label in ("same-genus", "different-genus"):
            part = ood.loc[distance.eq(label)]
            output[f"test_ood_{label}"] = {
                "n_spectra": int(len(part)),
                "n_strains": int(part["strain_id"].nunique()),
                "n_species": int(part["species"].nunique()),
                "false_acceptance_count": int(part["accepted"].sum()),
                "false_acceptance_rate": float(part["accepted"].mean()) if len(part) else None,
            }
    return output


def _run_main_cells(
    root: Path,
    matrix: np.ndarray,
    feature_manifest: pd.DataFrame,
    splits: pd.DataFrame,
    seeds: list[int],
    folds: list[int] | None,
    *,
    backend: str,
    block_size: int,
    resume: bool,
    base_inputs: dict[str, object],
    manifest_provenance: dict[str, object],
) -> dict[str, str]:
    run_root = root / "output/v2/reference_source/runs"
    completed: dict[str, str] = {}
    for design in ("spectrum_random", "strain_grouped"):
        for seed in seeds:
            available = sorted(
                splits.loc[
                    splits["design"].eq(design) & splits["seed"].astype(int).eq(seed),
                    "fold",
                ].astype(int).unique()
            )
            chosen_folds = available if folds is None else [fold for fold in folds if fold in available]
            for fold in chosen_folds:
                run_id = f"main__reference_library__{design}__seed-{seed}__fold-{fold}"
                run_dir = run_root / run_id
                selected = splits.loc[
                    splits["design"].eq(design)
                    & splits["seed"].astype(int).eq(seed)
                    & splits["fold"].astype(int).eq(fold)
                ].copy()
                fingerprint = canonical_sha256(
                    {**base_inputs, "run_id": run_id, "assignment": canonical_sha256(selected.to_dict("records"))}
                )
                if resume and _cell_complete(run_dir, fingerprint):
                    _refresh_cell_provenance(
                        run_dir,
                        provenance=manifest_provenance["current"],
                        frozen_provenance=manifest_provenance["frozen"],
                    )
                    completed[run_id] = fingerprint
                    continue
                predictions, scores, _query, threshold, n_cal, library = _apply_reference_predictions(
                    matrix,
                    feature_manifest,
                    selected,
                    backend=backend,
                    block_size=block_size,
                )
                predictions["analysis"] = "main_reference_library"
                predictions["design"] = design
                predictions["model"] = "reference_library"
                predictions["seed"] = int(seed)
                predictions["fold"] = int(fold)
                metrics = _metrics_for_reference(predictions, scores, library.species_classes)
                metrics.update(
                    {
                        "run_id": run_id,
                        "analysis": "main_reference_library",
                        "design": design,
                        "model": "reference_library",
                        "seed": int(seed),
                        "fold": int(fold),
                        "acceptance_threshold": float(threshold),
                        "calibration_strains": int(n_cal),
                        "template_strains": int(len(library.strain_ids)),
                        "fitted_species": int(len(library.species_classes)),
                    }
                )
                _write_cell(
                    run_dir,
                    predictions,
                    metrics,
                    library.species_classes,
                    {
                        **runtime_record(),
                        "run_id": run_id,
                        "analysis": "main_reference_library",
                        "design": design,
                        "model": "reference_library",
                        "seed": int(seed),
                        "fold": int(fold),
                        "input_fingerprint": fingerprint,
                        **{
                            key: value
                            for key, value in base_inputs.items()
                            if key != "legacy_analysis_commit"
                        },
                        **manifest_provenance["current"],
                        **manifest_provenance["frozen"],
                        "backend_requested": backend,
                        "block_size": int(block_size),
                        "threshold_data_role": "known calibration only",
                        "n_templates": int(len(library.strain_ids)),
                        "n_classes": int(len(library.species_classes)),
                    },
                )
                completed[run_id] = fingerprint
                del predictions, scores, library
                gc.collect()
    return completed


def _extra_trees_predictions(
    matrix: np.ndarray,
    manifest: pd.DataFrame,
    assignment: pd.DataFrame,
    *,
    seed: int,
    threads: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, float, int]:
    role_indices = {
        role: _indices(manifest, frame["spectrum_id"])
        for role, frame in assignment.groupby("role", sort=False)
    }
    model = fit_extra_trees(
        matrix[role_indices["train"]],
        manifest.iloc[role_indices["train"]].reset_index(drop=True),
        seed=seed,
        n_jobs=threads,
    )
    roles = ["calibration", "test_source"]
    query_idx = np.concatenate([role_indices[role] for role in roles])
    query_rows = pd.concat(
        [manifest.iloc[role_indices[role]] for role in roles], ignore_index=True
    )
    role_vector = np.concatenate([np.repeat(role, len(role_indices[role])) for role in roles])
    probabilities = np.asarray(model.predict_proba(matrix[query_idx]), dtype=np.float32)
    classes = model.classes_.astype(str)
    order = np.argsort(-probabilities, axis=1, kind="stable")
    winner = order[:, 0]
    runner = order[:, 1] if len(classes) > 1 else order[:, 0]
    row = np.arange(len(probabilities))
    predictions = _compact_rows(query_rows)
    predictions["predicted_species"] = classes[winner]
    predictions["score"] = probabilities[row, winner].astype(float)
    predictions["winning_reference_strain_id"] = None
    predictions["runner_up_species"] = classes[runner]
    predictions["runner_up_score"] = probabilities[row, runner].astype(float)
    predictions["runner_up_margin"] = (
        probabilities[row, winner] - probabilities[row, runner]
    ).astype(float)
    predictions["top3_species"] = [
        json.dumps(classes[index[: min(3, len(classes))]].tolist(), ensure_ascii=False)
        for index in order
    ]
    predictions["role"] = role_vector
    cal = predictions["role"].eq("calibration")
    threshold, calibration_strains = known_calibration_threshold(
        predictions.loc[cal, "score"], predictions.loc[cal, "strain_id"], target=0.95
    )
    predictions["acceptance_threshold"] = float(threshold)
    predictions["threshold_source"] = "known-calibration-strain-median-max-probability"
    predictions["accepted"] = predictions["score"].ge(threshold)
    predictions["true_is_fitted"] = predictions["species"].astype(str).isin(set(classes))
    predictions["correct_forced_choice"] = predictions["predicted_species"].astype(str).eq(
        predictions["species"].astype(str)
    )
    return predictions, probabilities, classes, threshold, calibration_strains


def _run_source_cells(
    root: Path,
    matrix: np.ndarray,
    feature_manifest: pd.DataFrame,
    source_splits: pd.DataFrame,
    seeds: list[int],
    *,
    backend: str,
    block_size: int,
    threads: int,
    resume: bool,
    base_inputs: dict[str, object],
    manifest_provenance: dict[str, object],
) -> dict[str, str]:
    run_root = root / "output/v2/reference_source/runs"
    completed: dict[str, str] = {}
    for label in sorted(source_splits["holdout_label"].astype(str).unique()):
        for seed in seeds:
            selected = source_splits.loc[
                source_splits["holdout_label"].astype(str).eq(label)
                & source_splits["seed"].astype(int).eq(seed)
            ].copy()
            for model_name in ("reference_library", "extra_trees"):
                run_id = f"source_transport__{model_name}__{label}__seed-{seed}"
                run_dir = run_root / run_id
                fingerprint = canonical_sha256(
                    {
                        **base_inputs,
                        "run_id": run_id,
                        "assignment": canonical_sha256(selected.to_dict("records")),
                    }
                )
                if resume and _cell_complete(run_dir, fingerprint):
                    _upgrade_source_checkpoint_reporting(run_dir, label)
                    _refresh_cell_provenance(
                        run_dir,
                        provenance=manifest_provenance["current"],
                        frozen_provenance=manifest_provenance["frozen"],
                    )
                    if not _cell_complete(run_dir, fingerprint):
                        raise RuntimeError(f"checkpoint failed after reporting upgrade: {run_id}")
                    completed[run_id] = fingerprint
                    continue
                if model_name == "reference_library":
                    predictions, scores, _query, threshold, n_cal, library = _apply_reference_predictions(
                        matrix,
                        feature_manifest,
                        selected,
                        backend=backend,
                        block_size=block_size,
                    )
                    classes = library.species_classes
                    template_count = len(library.strain_ids)
                    del library
                else:
                    predictions, scores, classes, threshold, n_cal = _extra_trees_predictions(
                        matrix,
                        feature_manifest,
                        selected,
                        seed=seed,
                        threads=threads,
                    )
                    template_count = None
                predictions["analysis"] = "source_instrument_era_transport"
                predictions["design"] = "source_holdout"
                predictions["model"] = model_name
                predictions["holdout_label"] = label
                predictions["query_is_holdout_label"] = predictions["instrum"].astype(str).eq(label)
                target_test = predictions["role"].eq("test_source") & predictions[
                    "query_is_holdout_label"
                ]
                auxiliary_test = predictions["role"].eq("test_source") & ~predictions[
                    "query_is_holdout_label"
                ]
                predictions["source_analysis_population"] = np.select(
                    [target_test, auxiliary_test],
                    ["target-label-primary", "other-label-quarantine-auxiliary"],
                    default="calibration",
                )
                predictions["seed"] = int(seed)
                predictions["fold"] = -1
                metrics = _metrics_for_reference(predictions, scores, classes)
                whole_strain_auxiliary = metrics.pop("test_source")
                if not target_test.any():
                    raise RuntimeError(f"source holdout has no target-label test spectra: {label}")
                metrics["test_source"] = classification_metrics(
                    predictions.loc[target_test].reset_index(drop=True),
                    score_columns=scores[target_test.to_numpy()],
                    classes=classes,
                )
                metrics["test_source_whole_strain_quarantine_auxiliary"] = whole_strain_auxiliary
                metrics.update(
                    {
                        "run_id": run_id,
                        "analysis": "source_instrument_era_transport",
                        "design": "source_holdout",
                        "model": model_name,
                        "holdout_label": label,
                        "seed": int(seed),
                        "fold": -1,
                        "acceptance_threshold": float(threshold),
                        "calibration_strains": int(n_cal),
                        "template_strains": None if template_count is None else int(template_count),
                        "fitted_species": int(len(classes)),
                        "source_primary_population": "target-label-only",
                        "source_primary_test_spectra": int(target_test.sum()),
                        "source_quarantine_auxiliary_spectra": int(auxiliary_test.sum()),
                        "interpretation": "within-dataset source/instrument-era transport on target-label spectra; other-label spectra from quarantined strains are auxiliary only",
                    }
                )
                _write_cell(
                    run_dir,
                    predictions,
                    metrics,
                    classes,
                    {
                        **runtime_record(),
                        "run_id": run_id,
                        "analysis": "source_instrument_era_transport",
                        "input_fingerprint": fingerprint,
                        **{
                            key: value
                            for key, value in base_inputs.items()
                            if key != "legacy_analysis_commit"
                        },
                        **manifest_provenance["current"],
                        **manifest_provenance["frozen"],
                        "holdout_label": label,
                        "model": model_name,
                        "seed": int(seed),
                        "backend_requested": backend if model_name == "reference_library" else None,
                        "threads": int(threads),
                        "threshold_data_role": "known calibration only",
                        "whole_strain_source_quarantine": True,
                        "source_primary_population": "target-label-only",
                        "source_primary_test_spectra": int(target_test.sum()),
                        "source_quarantine_auxiliary_spectra": int(auxiliary_test.sum()),
                        "interpretation": "within-dataset source/instrument-era transport on target-label spectra; other-label spectra from quarantined strains are auxiliary only",
                    },
                )
                completed[run_id] = fingerprint
                del predictions, scores
                gc.collect()
    return completed


def _flatten_metrics(record: dict[str, object]) -> dict[str, object]:
    flat: dict[str, object] = {}
    for key, value in record.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                flat[f"{key}__{nested_key}"] = nested_value
        else:
            flat[key] = value
    return flat


def _consolidate(
    root: Path,
    expected_cells: dict[str, str],
    base_inputs: dict[str, object],
    manifest_provenance: dict[str, object],
    *,
    expected_count: int = 70,
) -> None:
    output = root / "output/v2/reference_source"
    run_dirs = sorted(
        [path for path in (output / "runs").glob("*") if path.is_dir()],
        key=lambda path: path.name,
    )
    inventory: dict[str, Path] = {}
    for run_dir in run_dirs:
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"unmanifested run directory requires quarantine: {run_dir.name}")
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_id = str(record.get("run_id", ""))
        if run_id != run_dir.name:
            raise RuntimeError(f"run-id/directory mismatch: {run_dir.name}")
        inventory[run_id] = run_dir

    expected_run_ids = set(expected_cells)
    if len(expected_cells) != expected_count:
        raise RuntimeError(
            f"final consolidation requires exactly {expected_count} cells, got {len(expected_cells)}"
        )
    extras = sorted(set(inventory).difference(expected_run_ids))
    if extras:
        raise RuntimeError(
            f"{len(extras)} unexpected run directories require quarantine: {extras[:5]}"
        )
    missing = sorted(expected_run_ids.difference(inventory))
    if missing:
        raise RuntimeError(f"cannot consolidate; {len(missing)} expected runs missing")

    valid: list[Path] = []
    for run_id in sorted(expected_run_ids):
        run_dir = inventory[run_id]
        if not _cell_complete(run_dir, expected_cells[run_id]):
            raise RuntimeError(f"cell hash/fingerprint validation failed: {run_id}")
        valid.append(run_dir)
    if len(valid) != expected_count:
        raise AssertionError("validated run count differs from exact expected count")

    prediction_parts = [pd.read_parquet(path / "predictions.parquet") for path in valid]
    metric_records = [
        _flatten_metrics(json.loads((path / "metrics.json").read_text(encoding="utf-8")))
        for path in valid
    ]
    predictions = pd.concat(prediction_parts, ignore_index=True)
    metrics = pd.DataFrame(metric_records)
    main_prediction = predictions.loc[predictions["analysis"].eq("main_reference_library")]
    source_prediction = predictions.loc[
        predictions["analysis"].eq("source_instrument_era_transport")
    ]
    main_metrics = metrics.loc[metrics["analysis"].eq("main_reference_library")]
    source_metrics = metrics.loc[metrics["analysis"].eq("source_instrument_era_transport")]
    _atomic_parquet(main_prediction, output / "predictions/main_reference_predictions.parquet")
    _atomic_parquet(source_prediction, output / "predictions/source_transport_predictions.parquet")
    _atomic_csv(main_metrics, output / "metrics/main_reference_metrics.csv")
    _atomic_csv(source_metrics, output / "metrics/source_transport_metrics.csv")

    grouping = ["analysis", "design", "model"]
    if "holdout_label" in metrics:
        metrics["holdout_label"] = metrics["holdout_label"].fillna("")
        grouping.append("holdout_label")
    numeric_columns = [
        column
        for column in metrics.select_dtypes(include=np.number).columns
        if column not in {"seed", "fold"}
    ]
    summary = metrics.groupby(grouping, dropna=False)[numeric_columns].agg(["mean", "std"])
    summary.columns = ["__".join(column) for column in summary.columns]
    summary = summary.reset_index()
    _atomic_csv(summary, output / "metrics/summary.csv")

    artifact_paths = [
        output / "cohorts/source_cohorts.parquet",
        output / "cohorts/source_species_counts.csv",
        output / "splits/main_reference_splits.parquet",
        output / "splits/source_splits.parquet",
        output / "predictions/main_reference_predictions.parquet",
        output / "predictions/source_transport_predictions.parquet",
        output / "metrics/main_reference_metrics.csv",
        output / "metrics/source_transport_metrics.csv",
        output / "metrics/summary.csv",
    ]
    artifacts = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in artifact_paths
        if path.is_file()
    ]
    atomic_json(
        output / "manifests/reference_source_manifest.json",
        {
            **runtime_record(),
            "status": "complete",
            "schema_version": "2.0.0",
            "analysis": "reference_library_and_source_instrument_era_transport",
            **{
                key: value
                for key, value in base_inputs.items()
                if key != "legacy_analysis_commit"
            },
            **manifest_provenance["current"],
            **manifest_provenance["frozen"],
            "run_count": int(len(valid)),
            "run_ids": sorted(expected_run_ids),
            "all_cells_hash_and_fingerprint_verified": True,
            "unexpected_run_count": 0,
            "artifacts": artifacts,
            "source_primary_population": "target acquisition-label spectra only",
            "clinical_interpretation": "source analyses are within-dataset source/instrument-era transport, not external clinical validation; other-label spectra from quarantined strains are auxiliary only",
        },
    )


def main() -> int:
    args = _parse_args()
    if not 1 <= args.threads <= 4:
        raise ValueError("--threads must be between 1 and 4")
    if not 1 <= args.cell_workers <= 2:
        raise ValueError("--cell-workers must be 1 or 2")
    if args.cell_workers != 1:
        # GPU/reference and ExtraTrees cells have intentionally different
        # resource profiles. Keep this orchestrator sequential rather than
        # silently oversubscribing; the declared ceiling remains auditable.
        raise ValueError("this deterministic runner currently requires --cell-workers 1")

    root = args.project.resolve()
    spec_path = root / "config/v2/analysis_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    feature_path = root / "data/processed/production/features.npy"
    manifest_path = root / "data/processed/production/manifest.parquet"
    split_path = root / "data/processed/splits.parquet"
    metadata_path = root / "data/processed/production/feature_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("features_sha256") != EXPECTED_FEATURE_SHA256:
        raise RuntimeError("feature metadata does not identify the locked v2 input")
    if spec.get("feature_sha256") != EXPECTED_FEATURE_SHA256:
        raise RuntimeError("analysis spec does not identify the locked v2 feature matrix")
    frozen_provenance = _verify_frozen_input_manifest(root, spec)
    current_provenance = _current_git_provenance(root)
    manifest_provenance = {
        "current": current_provenance,
        "frozen": frozen_provenance,
    }

    matrix = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    feature_manifest = pd.read_parquet(manifest_path)
    if len(feature_manifest) != len(matrix):
        raise RuntimeError("feature/manifest row mismatch")
    specified_seeds = [int(value) for value in spec.get("seeds", DEFAULT_SEEDS)]
    seeds = [int(value) for value in (args.seeds or specified_seeds)]
    unknown_seeds = sorted(set(seeds).difference(specified_seeds))
    if unknown_seeds:
        raise ValueError(f"requested seeds are absent from the frozen v2 spec: {unknown_seeds}")
    folds = None if args.folds is None else [int(value) for value in args.folds]
    source_hash = _source_hash(root)
    base_inputs = {
        "schema_version": "2.0.0",
        "legacy_analysis_commit": spec["legacy_analysis_commit"],
        "feature_sha256": EXPECTED_FEATURE_SHA256,
        "feature_manifest_sha256": sha256_file(manifest_path),
        "legacy_split_sha256": sha256_file(split_path),
        "analysis_spec_sha256": sha256_file(spec_path),
        "analysis_source_sha256": source_hash,
    }
    output = root / "output/v2/reference_source"
    output.mkdir(parents=True, exist_ok=True)
    cohorts, cohort_counts = build_source_cohorts(
        feature_manifest,
        spec.get("source_holdouts", ("FLI-RIE-PC032", "MALDIMESS")),
        min_nonheld_strains=int(spec.get("source_cohort_min_nonheld_strains", 5)),
        min_held_strains=int(spec.get("source_cohort_min_held_strains", 2)),
    )
    # The frozen split artifact always contains every specified seed. CLI seed
    # filtering controls execution only and therefore cannot mutate its hash.
    source_splits = build_source_splits(feature_manifest, cohorts, specified_seeds)
    _atomic_parquet(cohorts, output / "cohorts/source_cohorts.parquet")
    _atomic_csv(cohort_counts, output / "cohorts/source_species_counts.csv")
    _atomic_parquet(source_splits, output / "splits/source_splits.parquet")
    base_inputs["source_cohort_sha256"] = sha256_file(output / "cohorts/source_cohorts.parquet")
    base_inputs["source_split_sha256"] = sha256_file(output / "splits/source_splits.parquet")

    completed: dict[str, str] = {}
    with threadpool_limits(limits=args.threads):
        if args.phase in {"main", "all"}:
            legacy_splits = pd.read_parquet(split_path)
            frozen_main_splits = legacy_splits.loc[
                legacy_splits["design"].isin(["spectrum_random", "strain_grouped"])
                & legacy_splits["seed"].astype(int).isin(specified_seeds)
            ].copy()
            _atomic_parquet(
                frozen_main_splits,
                output / "splits/main_reference_splits.parquet",
            )
            completed.update(
                _run_main_cells(
                    root,
                    matrix,
                    feature_manifest,
                    legacy_splits,
                    seeds,
                    folds,
                    backend=args.backend,
                    block_size=args.block_size,
                    resume=args.resume,
                    base_inputs=base_inputs,
                    manifest_provenance=manifest_provenance,
                )
            )
        if args.phase in {"source", "all"}:
            completed.update(
                _run_source_cells(
                    root,
                    matrix,
                    feature_manifest,
                    source_splits,
                    seeds,
                    backend=args.backend,
                    block_size=args.block_size,
                    threads=args.threads,
                    resume=args.resume,
                    base_inputs=base_inputs,
                    manifest_provenance=manifest_provenance,
                )
            )

    full_execution = (
        args.phase == "all"
        and set(seeds) == set(specified_seeds)
        and len(seeds) == len(specified_seeds)
        and folds is None
    )
    if full_execution:
        _consolidate(
            root,
            completed,
            base_inputs,
            manifest_provenance,
            expected_count=70,
        )
    print(
        json.dumps(
            {
                "status": "complete",
                "phase": args.phase,
                "completed_runs": len(completed),
                "consolidated": full_execution,
                "output": str(output),
                "manifest": (
                    str(output / "manifests/reference_source_manifest.json")
                    if full_execution
                    else None
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
