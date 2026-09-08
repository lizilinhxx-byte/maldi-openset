from __future__ import annotations

import json
import hashlib
import os
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import StudyConfig
from .conformal import (
    ClassConditionalConformal,
    TemperatureScaler,
    aggregate_probabilities_by_group,
    known_acceptance_threshold,
)
from .features import load_feature_bundle
from .metrics import equal_strain_weights
from .models import (
    CNN1DClassifier,
    cross_validated_logits,
    fit_sklearn_model,
    model_spec,
    raw_logits,
)
from .splits import read_splits
from .util import atomic_write_json, runtime_manifest, sha256_file, stable_id, utc_now


def _indices_for_ids(manifest: pd.DataFrame, ids: Iterable[str]) -> np.ndarray:
    lookup = pd.Series(np.arange(len(manifest), dtype=int), index=manifest["spectrum_id"].astype(str))
    requested = pd.Index([str(value) for value in ids])
    missing = requested.difference(lookup.index)
    if len(missing):
        raise KeyError(f"{len(missing)} split spectrum IDs are absent from feature manifest")
    return lookup.loc[requested].to_numpy(dtype=int)


def _aligned_manifest(manifest: pd.DataFrame, indices: np.ndarray) -> pd.DataFrame:
    return manifest.iloc[indices].reset_index(drop=True)


def _fit_temperature(
    model,
    model_name: str,
    X_train,
    y_train: np.ndarray,
    groups: np.ndarray,
    sample_weight: np.ndarray,
    seed: int,
) -> tuple[TemperatureScaler, str]:
    class_lookup = {str(label): idx for idx, label in enumerate(model.classes_.astype(str))}
    seed = int(seed)
    if model_name == "cnn1d":
        raise ValueError("CNN temperature must be supplied from its grouped internal holdout")
    oof_logits, oof_classes = cross_validated_logits(
        model, X_train, y_train, groups, sample_weight, seed
    )
    if not np.array_equal(oof_classes.astype(str), model.classes_.astype(str)):
        raise AssertionError("OOF and final model classes differ")
    labels = np.array([class_lookup[str(label)] for label in y_train], dtype=int)
    return (
        TemperatureScaler().fit(
            oof_logits,
            labels,
            sample_weight=equal_strain_weights(groups),
        ),
        "grouped-out-of-fold-training",
    )


def _cnn_grouped_fit_and_temperature(
    X, y: np.ndarray, groups: np.ndarray, sample_weight: np.ndarray, seed: int
):
    """Keep the outer calibration set untouched by CNN early stopping and scaling."""
    minimum = min(len(np.unique(groups[y == label])) for label in np.unique(y))
    n_splits = min(3, minimum)
    if n_splits < 2:
        raise ValueError("CNN grouped validation requires at least two strains per class")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fit_idx, validation_idx = next(splitter.split(X, y, groups))
    model = CNN1DClassifier(seed=seed).fit(
        X[fit_idx],
        y[fit_idx],
        X[validation_idx],
        y[validation_idx],
        sample_weight=sample_weight[fit_idx],
    )
    lookup = {str(label): idx for idx, label in enumerate(model.classes_.astype(str))}
    validation_logits = raw_logits(model, X[validation_idx])
    validation_labels = np.array([lookup[str(label)] for label in y[validation_idx]], dtype=int)
    validation_weight = equal_strain_weights(groups[validation_idx])
    temperature = TemperatureScaler().fit(
        validation_logits,
        validation_labels,
        sample_weight=validation_weight,
    )
    return model, temperature, {
        "tuned": False,
        "architecture": "locked-3-block-1d-cnn",
        "internal_fit_spectra": int(len(fit_idx)),
        "internal_validation_spectra": int(len(validation_idx)),
        "internal_fit_strains": int(len(np.unique(groups[fit_idx]))),
        "internal_validation_strains": int(len(np.unique(groups[validation_idx]))),
        "internal_validation_folds": n_splits,
    }


def strain_balanced_weights(y: np.ndarray, strain_ids: np.ndarray) -> np.ndarray:
    """Give each class equal total mass and each strain equal mass within class."""
    y = np.asarray(y).astype(str)
    strain_ids = np.asarray(strain_ids).astype(str)
    weights = np.zeros(len(y), dtype=float)
    for label in np.unique(y):
        class_mask = y == label
        class_strains = np.unique(strain_ids[class_mask])
        for strain in class_strains:
            mask = class_mask & (strain_ids == strain)
            weights[mask] = 1.0 / (len(class_strains) * int(mask.sum()))
    weights *= len(weights) / weights.sum()
    return weights


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_tree_sha256() -> str:
    """Hash executable Python sources so uncommitted code cannot pass resume gates."""
    package_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package_root.glob("*.py"), key=lambda value: value.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_head_commit() -> str | None:
    """Resolve a normal repository HEAD without invoking a potentially unavailable git CLI."""
    current = Path(__file__).resolve().parent
    git_dir = next((path / ".git" for path in (current, *current.parents) if (path / ".git").is_dir()), None)
    if git_dir is None:
        return None
    head_path = git_dir / "HEAD"
    if not head_path.is_file():
        return None
    head = head_path.read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head or None
    ref = head[5:]
    loose = git_dir / ref
    if loose.is_file():
        return loose.read_text(encoding="utf-8").strip() or None
    packed = git_dir / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith(("#", "^")):
                commit, name = line.split(" ", 1)
                if name == ref:
                    return commit
    return None


def _split_artifact(path: str | Path) -> Path:
    path = Path(path)
    parquet = path.with_suffix(".parquet")
    if parquet.is_file():
        return parquet
    if path.is_file():
        return path
    raise FileNotFoundError(f"split artifact not found: {path}")


def _resume_inputs(
    config: StudyConfig,
    feature_metadata: dict,
    splits_path: str | Path,
    runtime: dict,
) -> dict[str, str | None]:
    environment = {key: value for key, value in runtime.items() if key != "created_at"}
    return {
        "feature_sha256": feature_metadata.get("features_sha256"),
        "feature_manifest_sha256": feature_metadata.get("manifest_sha256"),
        "split_sha256": sha256_file(_split_artifact(splits_path)),
        "config_sha256": _canonical_sha256(asdict(config)),
        "environment_sha256": _canonical_sha256(environment),
        "source_tree_sha256": _source_tree_sha256(),
        "git_commit": _git_head_commit(),
    }


def _completed_run_is_reusable(run_dir: Path, prior: dict, expected: dict) -> bool:
    if prior.get("status") != "complete":
        return False
    if any(prior.get(key) != value for key, value in expected.items()):
        return False
    artifact_fields = {
        "predictions_sha256": run_dir / "predictions.parquet",
        "probabilities_sha256": run_dir / "probabilities.npy",
        "classes_sha256": run_dir / "classes.json",
    }
    for field, path in artifact_fields.items():
        if not path.is_file() or prior.get(field) != sha256_file(path):
            return False
    model_path = run_dir / ("model.pt" if prior.get("model") == "cnn1d" else "model.joblib")
    return (
        model_path.is_file()
        and prior.get("model_sha256") == sha256_file(model_path)
    )


def _make_predictions(
    model,
    temperature: TemperatureScaler,
    conformal: ClassConditionalConformal,
    genus_conformal: ClassConditionalConformal,
    confidence_threshold: float,
    X,
    rows: pd.DataFrame,
    role: str,
    class_to_genus: dict[str, str],
) -> tuple[pd.DataFrame, np.ndarray]:
    logits = raw_logits(model, X)
    probability = temperature.transform(logits)
    classes = model.classes_.astype(str)
    predicted_index = np.argmax(probability, axis=1)
    predicted = classes[predicted_index]
    confidence = probability[np.arange(len(probability)), predicted_index]
    species_sets = conformal.prediction_sets(probability)
    genus_probability, genera = aggregate_probabilities_by_group(probability, classes, class_to_genus)
    genus_sets = genus_conformal.prediction_sets(genus_probability)

    accepted = np.array(
        [len(values) == 1 and conf >= confidence_threshold for values, conf in zip(species_sets, confidence)],
        dtype=bool,
    )
    reported_label: list[str] = []
    reported_level: list[str] = []
    for keep, species_set, genus_set in zip(accepted, species_sets, genus_sets):
        if keep:
            reported_label.append(species_set[0])
            reported_level.append("species")
        elif len(genus_set) == 1:
            reported_label.append(genus_set[0])
            reported_level.append("genus")
        else:
            reported_label.append("unidentified")
            reported_level.append("unidentified")

    output = rows.copy().reset_index(drop=True)
    output["role"] = role
    output["predicted_species"] = predicted
    output["predicted_genus"] = [class_to_genus[value] for value in predicted]
    output["confidence"] = confidence.astype(float)
    output["known_acceptance_threshold"] = float(confidence_threshold)
    output["threshold_source"] = "calibration-strain-median-lower-5th-percentile"
    output["conformal_threshold_source"] = [
        "class-conditional"
        if (conformal.class_counts or {}).get(str(label), 0) >= conformal.min_class_calibration
        else "pooled-fallback"
        for label in predicted
    ]
    output["conformal_species_set"] = [json.dumps(values, ensure_ascii=False) for values in species_sets]
    output["conformal_species_set_size"] = [len(values) for values in species_sets]
    output["conformal_genus_set"] = [json.dumps(values, ensure_ascii=False) for values in genus_sets]
    output["conformal_genus_set_size"] = [len(values) for values in genus_sets]
    output["accepted_species"] = accepted
    output["reported_label"] = reported_label
    output["reported_level"] = reported_level
    output["true_is_known"] = output["species"].astype(str).isin(classes)
    return output, probability.astype(np.float32)


def _make_closed_predictions(
    model,
    temperature: TemperatureScaler,
    X,
    rows: pd.DataFrame,
    role: str,
    class_to_genus: dict[str, str],
) -> tuple[pd.DataFrame, np.ndarray]:
    probability = temperature.transform(raw_logits(model, X))
    classes = model.classes_.astype(str)
    predicted = classes[np.argmax(probability, axis=1)]
    confidence = probability.max(axis=1)
    output = rows.copy().reset_index(drop=True)
    output["role"] = role
    output["predicted_species"] = predicted
    output["predicted_genus"] = [class_to_genus[value] for value in predicted]
    output["confidence"] = confidence
    output["known_acceptance_threshold"] = np.nan
    output["threshold_source"] = "not-applicable-closed-set-sensitivity"
    output["conformal_threshold_source"] = "not-applied"
    output["conformal_species_set"] = [json.dumps([value]) for value in predicted]
    output["conformal_species_set_size"] = 1
    output["conformal_genus_set"] = [json.dumps([class_to_genus[value]]) for value in predicted]
    output["conformal_genus_set_size"] = 1
    output["accepted_species"] = True
    output["reported_label"] = predicted
    output["reported_level"] = "species"
    output["true_is_known"] = True
    return output, probability.astype(np.float32)


def _fit_fixed_sensitivity_model(
    model_name: str,
    X,
    y: np.ndarray,
    groups: np.ndarray,
    sample_weight: np.ndarray,
    seed: int,
    n_jobs: int,
):
    if model_name == "cnn1d":
        model, temperature, info = _cnn_grouped_fit_and_temperature(
            X, y, groups, sample_weight, seed
        )
        info["sensitivity_fixed"] = True
        return model, temperature, info, "grouped-training-holdout"
    model, _ = model_spec(model_name, seed, n_jobs=n_jobs)
    anchors = {
        "linear_svm": {"C": 1.0},
        "rbf_svm": {"C": 1.0, "gamma": "scale"},
        "extra_trees": {"max_features": "sqrt", "min_samples_leaf": 1},
        "xgboost": {"max_depth": 4, "learning_rate": 0.05},
    }
    if model_name in anchors:
        model.set_params(**anchors[model_name])
    model.fit(X, y, sample_weight=sample_weight)
    temperature, source = _fit_temperature(
        model, model_name, X, y, groups, sample_weight, seed
    )
    return model, temperature, {
        "tuned": False,
        "sensitivity_fixed": True,
        "fixed_params": anchors.get(model_name, {}),
    }, source


def train_one(
    feature_dir: str | Path,
    splits_path: str | Path,
    output_root: str | Path,
    config: StudyConfig,
    design: str,
    model_name: str,
    seed: int,
    fold: int,
    n_jobs: int = -1,
    resume: bool = True,
) -> Path:
    seed = int(seed)
    fold = int(fold)
    matrix, manifest, feature_metadata = load_feature_bundle(feature_dir)
    assignments = read_splits(splits_path)
    current_runtime = runtime_manifest()
    resume_inputs = _resume_inputs(config, feature_metadata, splits_path, current_runtime)
    selected = assignments.loc[
        (assignments["design"] == design)
        & (assignments["seed"].astype(int) == int(seed))
        & (assignments["fold"].astype(int) == int(fold))
    ].copy()
    if selected.empty:
        raise KeyError(f"no split for design={design}, seed={seed}, fold={fold}")
    run_id = f"{design}__{model_name}__seed-{seed}__fold-{fold}"
    run_dir = Path(output_root) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    completion_path = run_dir / "run_manifest.json"
    if resume and completion_path.exists():
        with completion_path.open("r", encoding="utf-8") as handle:
            prior = json.load(handle)
        if _completed_run_is_reusable(run_dir, prior, resume_inputs):
            return run_dir

    role_indices: dict[str, np.ndarray] = {}
    role_rows: dict[str, pd.DataFrame] = {}
    for role, part in selected.groupby("role", sort=False):
        idx = _indices_for_ids(manifest, part["spectrum_id"].astype(str))
        role_indices[role] = idx
        role_rows[role] = _aligned_manifest(manifest, idx)
    sensitivity_design = design == "strain_grouped_sensitivity"
    required_roles = ("train", "test_known") if sensitivity_design else (
        "train",
        "calibration",
        "test_known",
    )
    for required in required_roles:
        if required not in role_indices:
            raise ValueError(f"split lacks required role: {required}")

    # The authoritative OOD distance is fold-specific: sensitivity analyses can
    # fit genera that are absent from the primary-known set stored in manifest.
    if "test_ood" in role_rows:
        fitted_genera = set(role_rows["train"]["genus"].astype(str))
        role_rows["test_ood"]["ood_distance"] = np.where(
            role_rows["test_ood"]["genus"].astype(str).isin(fitted_genera),
            "near",
            "far",
        )

    train_idx = role_indices["train"]
    cal_idx = role_indices.get("calibration", np.array([], dtype=int))
    X_train = matrix[train_idx]
    X_cal = matrix[cal_idx] if len(cal_idx) else None
    y_train = manifest.iloc[train_idx]["species"].astype(str).to_numpy()
    y_cal = manifest.iloc[cal_idx]["species"].astype(str).to_numpy() if len(cal_idx) else None
    groups = manifest.iloc[train_idx]["strain_id"].astype(str).to_numpy()
    sample_weight = strain_balanced_weights(y_train, groups)

    if sensitivity_design:
        model, temperature, tuning, calibration_source = _fit_fixed_sensitivity_model(
            model_name,
            X_train,
            y_train,
            groups,
            sample_weight,
            seed + fold,
            n_jobs,
        )
    elif model_name == "cnn1d":
        model, temperature, tuning = _cnn_grouped_fit_and_temperature(
            X_train, y_train, groups, sample_weight, seed + fold
        )
        calibration_source = "grouped-training-holdout"
    else:
        model, tuning = fit_sklearn_model(
            model_name,
            X_train,
            y_train,
            groups,
            sample_weight,
            seed=seed + fold,
            n_jobs=n_jobs,
        )
        temperature, calibration_source = _fit_temperature(
            model, model_name, X_train, y_train, groups, sample_weight, seed + fold
        )
    species_genus = (
        manifest.loc[manifest["species"].astype(str).isin(model.classes_.astype(str)), ["species", "genus"]]
        .drop_duplicates("species")
        .set_index("species")["genus"]
        .astype(str)
        .to_dict()
    )
    if not sensitivity_design:
        cal_probability = temperature.transform(raw_logits(model, X_cal))
        conformal = ClassConditionalConformal(
            classes=model.classes_.astype(str), coverage=config.known_acceptance_target
        ).fit(cal_probability, y_cal)
        confidence_threshold = known_acceptance_threshold(
            cal_probability,
            manifest.iloc[cal_idx]["strain_id"].astype(str).to_numpy(),
            target=config.known_acceptance_target,
        )
        cal_genus_probability, genus_classes = aggregate_probabilities_by_group(
            cal_probability, model.classes_.astype(str), species_genus
        )
        genus_conformal = ClassConditionalConformal(
            classes=genus_classes.astype(str), coverage=config.known_acceptance_target
        ).fit(cal_genus_probability, manifest.iloc[cal_idx]["genus"].astype(str).to_numpy())
    else:
        conformal = None
        genus_conformal = None
        confidence_threshold = float("nan")

    prediction_parts: list[pd.DataFrame] = []
    probability_parts: list[np.ndarray] = []
    prediction_roles = ("test_known",) if sensitivity_design else (
        "calibration",
        "test_known",
        "test_ood",
    )
    for role in prediction_roles:
        if role not in role_indices:
            continue
        if sensitivity_design:
            frame, probability = _make_closed_predictions(
                model,
                temperature,
                matrix[role_indices[role]],
                role_rows[role],
                role,
                species_genus,
            )
        else:
            frame, probability = _make_predictions(
                model,
                temperature,
                conformal,
                genus_conformal,
                confidence_threshold,
                matrix[role_indices[role]],
                role_rows[role],
                role,
                species_genus,
            )
        frame["probability_row"] = np.arange(
            sum(len(x) for x in probability_parts),
            sum(len(x) for x in probability_parts) + len(frame),
        )
        prediction_parts.append(frame)
        probability_parts.append(probability)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    probabilities = np.vstack(probability_parts)
    predictions["design"] = design
    predictions["model"] = model_name
    predictions["seed"] = int(seed)
    predictions["fold"] = int(fold)

    predictions_path = run_dir / "predictions.parquet"
    probabilities_path = run_dir / "probabilities.npy"
    predictions.to_parquet(predictions_path, index=False)
    np.save(probabilities_path, probabilities, allow_pickle=False)
    classes_path = run_dir / "classes.json"
    atomic_write_json(classes_path, model.classes_.astype(str).tolist())
    if model_name == "cnn1d":
        import torch

        torch.save(
            {"state_dict": model.network_.state_dict(), "classes": model.classes_.tolist()},
            run_dir / "model.pt",
        )
        model_path = run_dir / "model.pt"
    else:
        model_path = run_dir / "model.joblib"
        joblib.dump(model, model_path, compress=3)
    run_manifest = current_runtime
    run_manifest.update(
        {
            "status": "complete",
            "run_id": run_id,
            "design": design,
            "model": model_name,
            "seed": int(seed),
            "fold": int(fold),
            **resume_inputs,
            "n_train": int(len(train_idx)),
            "n_calibration": int(len(cal_idx)),
            "n_test_known": int(len(role_indices["test_known"])),
            "n_test_ood": int(len(role_indices.get("test_ood", []))),
            "class_count": int(len(model.classes_)),
            "tuning": tuning,
            "temperature": temperature.temperature,
            "temperature_source": calibration_source,
            "known_acceptance_threshold": None if sensitivity_design else confidence_threshold,
            "conformal_global_threshold": conformal.global_threshold if conformal else None,
            "conformal_class_counts": conformal.class_counts if conformal else None,
            "predictions_sha256": sha256_file(predictions_path),
            "probabilities_sha256": sha256_file(probabilities_path),
            "classes_sha256": sha256_file(classes_path),
            "model_sha256": sha256_file(model_path),
            "completed_at": utc_now(),
        }
    )
    atomic_write_json(completion_path, run_manifest)
    return run_dir


def train_grid(
    feature_dir: str | Path,
    splits_path: str | Path,
    output_root: str | Path,
    config: StudyConfig,
    designs: Iterable[str] | None = None,
    models: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
    folds: Iterable[int] | None = None,
    n_jobs: int = -1,
    resume: bool = True,
    cell_workers: int = 1,
) -> list[Path]:
    designs = list(designs or ["strain_grouped", "spectrum_random"])
    models = list(models or config.models)
    seeds = [int(value) for value in (seeds or config.seeds)]
    requested_folds = None if folds is None else [int(value) for value in folds]
    available = read_splits(splits_path)[["design", "seed", "fold"]].drop_duplicates()
    tasks = []
    for design in designs:
        for model_name in models:
            for seed in seeds:
                cell_folds = (
                    requested_folds
                    if requested_folds is not None
                    else sorted(
                        available.loc[
                            (available["design"] == design)
                            & (available["seed"].astype(int) == int(seed)),
                            "fold",
                        ].astype(int).unique()
                    )
                )
                for fold in cell_folds:
                    tasks.append((design, model_name, seed, fold))

    def execute(cell):
        design, model_name, seed, fold = cell
        return train_one(
            feature_dir,
            splits_path,
            output_root,
            config,
            design,
            model_name,
            seed,
            fold,
            n_jobs=n_jobs,
            resume=resume,
        )

    if cell_workers <= 1:
        return [execute(cell) for cell in tasks]
    results: list[Path] = []
    with ThreadPoolExecutor(max_workers=cell_workers) as executor:
        future_map = {executor.submit(execute, cell): cell for cell in tasks}
        for future in as_completed(future_map):
            results.append(future.result())
    return sorted(results)
