from __future__ import annotations

import hashlib
import json
import math
import subprocess
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score


ANALYSIS_COMMIT = "2293ce52a07e24e507e48814dcd3303bf512c0f3"
COMPLETE_MODELS = ("cnn1d", "cosine_centroid", "extra_trees", "rbf_svm")
DESIGNS = ("spectrum_random", "strain_grouped")
OUTCOME_COLUMNS = (
    "confidence_only_accept",
    "saved_singleton",
    "corrected_singleton_argmax",
    "legacy_species_accept",
    "corrected_species_accept",
    "legacy_genus_report",
    "corrected_genus_report",
    "legacy_unidentified",
    "corrected_unidentified",
    "saved_species_set_contains_true",
    "legacy_any_report_correct",
    "corrected_any_report_correct",
)


def sha256_file(path: str | Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def finite_sample_quantile(scores: Sequence[float] | np.ndarray, coverage: float = 0.95) -> float:
    """Conservative split-conformal order statistic for scores bounded by one."""
    values = np.sort(np.asarray(scores, dtype=float))
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("scores must be a nonempty finite vector")
    if not 0 < coverage < 1:
        raise ValueError("coverage must lie in (0, 1)")
    rank = int(math.ceil((len(values) + 1) * coverage))
    return 1.0 if rank > len(values) else float(values[max(rank, 1) - 1])


def empirical_acceptance_threshold(
    confidence: Sequence[float] | np.ndarray, target: float = 0.95
) -> float:
    """Lower empirical confidence cutoff giving a nominal target acceptance rate."""
    values = np.asarray(confidence, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("confidence must be a nonempty finite vector")
    if not 0 < target < 1:
        raise ValueError("target must lie in (0, 1)")
    return float(np.quantile(values, 1.0 - target, method="lower"))


def is_explicit_binomial_label(label: object) -> bool:
    """Return true for a frozen `Genus epithet` label and false for `Genus sp.`."""
    parts = str(label).strip().split()
    return len(parts) == 2 and parts[1].lower() != "sp."


def _parse_set(value: object) -> list[str]:
    if isinstance(value, list):
        result = value
    else:
        result = json.loads(str(value))
    if not isinstance(result, list) or any(not isinstance(item, str) for item in result):
        raise ValueError(f"invalid saved prediction set: {value!r}")
    if len(result) != len(set(result)):
        raise ValueError("saved prediction set contains duplicate labels")
    return result


def _taxonomy_map(frame: pd.DataFrame, classes: np.ndarray) -> dict[str, str]:
    pairs = frame[["species", "genus"]].drop_duplicates()
    conflicts = pairs.groupby("species")["genus"].nunique()
    if (conflicts > 1).any():
        raise ValueError(f"ambiguous species-to-genus mapping: {conflicts[conflicts > 1].index.tolist()}")
    mapping = pairs.set_index("species")["genus"].astype(str).to_dict()
    missing = sorted(set(classes.astype(str)) - set(mapping))
    if missing:
        raise ValueError(f"model classes absent from frozen taxonomy: {missing}")
    return {str(label): mapping[str(label)] for label in classes.astype(str)}


def aggregate_genus_probability(
    probability: np.ndarray,
    classes: np.ndarray,
    class_to_genus: dict[str, str],
) -> tuple[np.ndarray, np.ndarray]:
    probability = np.asarray(probability)
    genera = np.array(sorted({class_to_genus[str(label)] for label in classes}), dtype=object)
    lookup = {str(label): index for index, label in enumerate(genera)}
    membership = np.zeros((len(classes), len(genera)), dtype=np.float64)
    for class_index, label in enumerate(classes.astype(str)):
        membership[class_index, lookup[class_to_genus[label]]] = 1.0
    return probability.astype(np.float64, copy=False) @ membership, genera


@dataclass(frozen=True)
class RunBundle:
    run_dir: Path
    frame: pd.DataFrame
    probability: np.ndarray
    classes: np.ndarray
    manifest: dict
    input_record: dict


def load_locked_run(
    run_dir: str | Path,
    spec: dict,
    project_root: str | Path,
    frozen_record: dict | None = None,
) -> RunBundle:
    run_dir = Path(run_dir)
    project_root = Path(project_root)
    paths = {
        "predictions": run_dir / "predictions.parquet",
        "probabilities": run_dir / "probabilities.npy",
        "classes": run_dir / "classes.json",
        "run_manifest": run_dir / "run_manifest.json",
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError(f"incomplete locked run: {run_dir}")
    manifest = json.loads(paths["run_manifest"].read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"run is not complete: {run_dir.name}")
    if manifest.get("git_commit") != spec["legacy_analysis_commit"] or manifest.get("git_commit") != ANALYSIS_COMMIT:
        raise ValueError(f"analysis commit mismatch: {run_dir.name}")
    if manifest.get("feature_sha256") != spec["feature_sha256"]:
        raise ValueError(f"feature hash mismatch: {run_dir.name}")
    actual = {key: sha256_file(path) for key, path in paths.items()}
    declared = {
        "predictions": manifest.get("predictions_sha256"),
        "probabilities": manifest.get("probabilities_sha256"),
        "classes": manifest.get("classes_sha256"),
    }
    for key in declared:
        if actual[key] != declared[key]:
            raise ValueError(f"locked {key} hash mismatch: {run_dir.name}")
    if frozen_record is not None:
        if frozen_record.get("run_id") != manifest.get("run_id"):
            raise ValueError(f"frozen run identity mismatch: {run_dir.name}")
        if actual["run_manifest"] != frozen_record.get("manifest_sha256"):
            raise ValueError(f"run manifest differs from frozen v2 input: {run_dir.name}")
        for key in ("predictions", "probabilities", "classes"):
            if actual[key] != frozen_record.get(f"{key}_sha256"):
                raise ValueError(f"{key} differs from frozen v2 input: {run_dir.name}")
        if manifest.get("split_sha256") != frozen_record.get("split_sha256"):
            raise ValueError(f"run split identity differs from frozen v2 input: {run_dir.name}")
    frame = pd.read_parquet(paths["predictions"])
    probability = np.load(paths["probabilities"], allow_pickle=False)
    classes = np.asarray(json.loads(paths["classes"].read_text(encoding="utf-8")), dtype=object)
    if probability.dtype != np.float32:
        raise ValueError(f"saved probability matrix is not float32: {run_dir.name}")
    if probability.shape != (len(frame), len(classes)):
        raise ValueError(f"prediction/probability/class dimensions disagree: {run_dir.name}")
    if not np.array_equal(frame["probability_row"].to_numpy(), np.arange(len(frame))):
        raise ValueError(f"probability row alignment failure: {run_dir.name}")
    if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
        raise ValueError(f"invalid saved probabilities: {run_dir.name}")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-6, rtol=0):
        raise ValueError(f"saved probabilities do not sum to one: {run_dir.name}")
    record = {
        "run_id": manifest["run_id"],
        "design": manifest["design"],
        "model": manifest["model"],
        "seed": int(manifest["seed"]),
        "fold": int(manifest["fold"]),
        "analysis_commit": manifest["git_commit"],
        "feature_sha256": manifest["feature_sha256"],
        "files": {
            key: {
                "path": path.resolve().relative_to(project_root.resolve()).as_posix(),
                "sha256": actual[key],
                "bytes": path.stat().st_size,
            }
            for key, path in paths.items()
        },
    }
    return RunBundle(run_dir, frame, probability, classes, manifest, record)


def load_and_verify_frozen_inputs(
    project_root: str | Path,
    frozen_manifest_path: str | Path,
    spec: dict,
) -> tuple[dict, str, dict[str, dict]]:
    """Verify the immutable v1 bytes against the separately frozen v2 inventory."""
    project_root = Path(project_root).resolve()
    frozen_manifest_path = Path(frozen_manifest_path).resolve()
    if frozen_manifest_path != (project_root / "output" / "v2" / "input_manifest.json").resolve():
        raise ValueError("frozen input manifest must be output/v2/input_manifest.json")
    frozen_hash = sha256_file(frozen_manifest_path)
    frozen = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
    if frozen.get("schema_version") != "2.0.0":
        raise ValueError("unexpected frozen input manifest schema")
    if frozen.get("legacy_analysis_commit") != ANALYSIS_COMMIT:
        raise ValueError("frozen input manifest is not bound to the legacy analysis commit")
    if frozen.get("feature_sha256") != spec.get("feature_sha256"):
        raise ValueError("frozen input feature identity differs from the v2 specification")
    fixed_records = frozen.get("files")
    if not isinstance(fixed_records, list) or not fixed_records:
        raise ValueError("frozen input manifest has no fixed-file inventory")
    for record in fixed_records:
        path = (project_root / str(record["path"])).resolve()
        if project_root not in path.parents or not path.is_file():
            raise ValueError(f"frozen fixed input is missing or outside project: {record.get('path')}")
        if path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
            raise ValueError(f"frozen fixed input hash mismatch: {record['path']}")
    feature_path = (project_root / str(frozen["feature_matrix_path"])).resolve()
    if not feature_path.is_file() or feature_path.stat().st_size != int(frozen["feature_matrix_bytes"]):
        raise ValueError("frozen feature matrix size mismatch")
    if sha256_file(feature_path) != frozen["feature_sha256"]:
        raise ValueError("frozen feature matrix hash mismatch")
    run_records = frozen.get("runs")
    if not isinstance(run_records, list) or len(run_records) != int(frozen.get("run_count", -1)):
        raise ValueError("frozen run inventory count mismatch")
    by_id: dict[str, dict] = {}
    for record in run_records:
        run_id = str(record["run_id"])
        if run_id in by_id:
            raise ValueError(f"duplicate run in frozen input manifest: {run_id}")
        by_id[run_id] = record
    return frozen, frozen_hash, by_id


def discover_locked_runs(
    production_root: str | Path,
    spec: dict,
    frozen_runs: dict[str, dict] | None = None,
) -> list[Path]:
    root = Path(production_root) / "runs"
    expected = {
        (design, model, int(seed), fold)
        for design in DESIGNS
        for model in COMPLETE_MODELS
        for seed in spec["seeds"]
        for fold in range(5)
    }
    found: dict[tuple[str, str, int, int], Path] = {}
    for path in sorted(root.glob("*/run_manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        key = (
            str(manifest.get("design")),
            str(manifest.get("model")),
            int(manifest.get("seed", -1)),
            int(manifest.get("fold", -1)),
        )
        if key in expected and manifest.get("status") == "complete" and manifest.get("git_commit") == ANALYSIS_COMMIT:
            if key in found:
                raise ValueError(f"duplicate locked run for {key}")
            found[key] = path.parent
    missing = sorted(expected - set(found))
    if missing:
        raise ValueError(f"missing complete v1 model instances: {missing}")
    selected = [found[key] for key in sorted(expected)]
    if frozen_runs is not None:
        selected_ids = {path.name for path in selected}
        missing_frozen = sorted(selected_ids - set(frozen_runs))
        if missing_frozen:
            raise ValueError(f"complete model instances absent from frozen inventory: {missing_frozen}")
        for path in selected:
            expected_manifest = (Path(production_root).parent.parent / frozen_runs[path.name]["manifest_path"]).resolve()
            if expected_manifest != (path / "run_manifest.json").resolve():
                raise ValueError(f"frozen run path mismatch: {path.name}")
    return selected


def build_decision_frame(
    frame: pd.DataFrame,
    probability: np.ndarray,
    classes: np.ndarray,
    run_id: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Apply the saved legacy rule and the post-result v2 correction without retuning."""
    frame = frame.reset_index(drop=True)
    probability = np.asarray(probability)
    if probability.shape != (len(frame), len(classes)):
        raise ValueError("probability dimensions do not match rows/classes")
    species_sets = frame["conformal_species_set"].map(_parse_set)
    genus_sets = frame["conformal_genus_set"].map(_parse_set)
    if not np.array_equal(species_sets.map(len).to_numpy(), frame["conformal_species_set_size"].to_numpy()):
        raise ValueError("saved species set sizes disagree with serialized sets")
    if not np.array_equal(genus_sets.map(len).to_numpy(), frame["conformal_genus_set_size"].to_numpy()):
        raise ValueError("saved genus set sizes disagree with serialized sets")

    saved_argmax = frame["predicted_species"].astype(str).to_numpy()
    saved_confidence = frame["confidence"].astype(float).to_numpy()
    tau = frame["known_acceptance_threshold"].astype(float).to_numpy()
    float32_index = probability.argmax(axis=1)
    float32_argmax = classes[float32_index].astype(str)
    float32_confidence = probability[np.arange(len(probability)), float32_index].astype(float)

    class_to_genus = _taxonomy_map(frame, classes)
    genus_probability, genera = aggregate_genus_probability(probability, classes, class_to_genus)
    genus_argmax = genera[genus_probability.argmax(axis=1)].astype(str)

    species_singleton = species_sets.map(lambda values: values[0] if len(values) == 1 else None)
    genus_singleton = genus_sets.map(lambda values: values[0] if len(values) == 1 else None)
    singleton = species_singleton.notna().to_numpy()
    confidence_only = saved_confidence >= tau
    legacy_accept = singleton & confidence_only
    corrected_singleton = singleton & (species_singleton.fillna("").astype(str).to_numpy() == saved_argmax)
    corrected_accept = corrected_singleton & confidence_only
    legacy_saved = frame["accepted_species"].astype(bool).to_numpy()
    if not np.array_equal(legacy_accept, legacy_saved):
        raise ValueError("saved legacy decisions cannot be reproduced from saved set/confidence/tau")

    legacy_genus_report = frame["reported_level"].astype(str).eq("genus").to_numpy()
    legacy_unidentified = frame["reported_level"].astype(str).eq("unidentified").to_numpy()
    corrected_genus_report = (
        (~corrected_accept)
        & genus_singleton.notna().to_numpy()
        & (genus_singleton.fillna("").astype(str).to_numpy() == genus_argmax)
    )
    corrected_level = np.where(
        corrected_accept, "species", np.where(corrected_genus_report, "genus", "unidentified")
    )
    corrected_label = np.where(
        corrected_accept,
        species_singleton.fillna("").astype(str).to_numpy(),
        np.where(corrected_genus_report, genus_singleton.fillna("").astype(str).to_numpy(), "unidentified"),
    )
    true_species = frame["species"].astype(str).to_numpy()
    true_genus = frame["genus"].astype(str).to_numpy()
    legacy_level = frame["reported_level"].astype(str).to_numpy()
    legacy_label = frame["reported_label"].astype(str).to_numpy()
    legacy_correct = np.where(
        legacy_level == "species",
        legacy_label == true_species,
        np.where(legacy_level == "genus", legacy_label == true_genus, False),
    )
    corrected_correct = np.where(
        corrected_level == "species",
        corrected_label == true_species,
        np.where(corrected_level == "genus", corrected_label == true_genus, False),
    )
    species_contains_true = np.fromiter(
        (truth in values for truth, values in zip(true_species, species_sets)), dtype=bool, count=len(frame)
    )
    genus_contains_true = np.fromiter(
        (truth in values for truth, values in zip(true_genus, genus_sets)), dtype=bool, count=len(frame)
    )

    keep = [
        "spectrum_id", "strain_id", "species", "genus", "role", "analysis_set",
        "ood_distance", "ood_singleton", "design", "model", "seed", "fold",
        "conformal_species_set", "conformal_species_set_size", "conformal_genus_set",
        "conformal_genus_set_size", "threshold_source", "conformal_threshold_source",
    ]
    output = frame[[column for column in keep if column in frame]].copy()
    output.insert(0, "run_id", run_id or "")
    output["saved_argmax_species"] = saved_argmax
    output["float32_argmax_species"] = float32_argmax
    output["saved_confidence"] = saved_confidence
    output["float32_confidence"] = float32_confidence
    output["saved_tau"] = tau
    output["saved_species_singleton_label"] = species_singleton
    output["saved_genus_singleton_label"] = genus_singleton
    output["float32_aggregate_genus_argmax"] = genus_argmax
    output["confidence_only_accept"] = confidence_only
    output["saved_singleton"] = singleton
    output["corrected_singleton_argmax"] = corrected_singleton
    output["legacy_species_accept"] = legacy_accept
    output["corrected_species_accept"] = corrected_accept
    output["legacy_reported_label"] = legacy_label
    output["legacy_reported_level"] = legacy_level
    output["corrected_reported_label"] = corrected_label
    output["corrected_reported_level"] = corrected_level
    output["legacy_genus_report"] = legacy_genus_report
    output["corrected_genus_report"] = corrected_genus_report
    output["legacy_unidentified"] = legacy_unidentified
    output["corrected_unidentified"] = corrected_level == "unidentified"
    output["saved_species_set_contains_true"] = species_contains_true
    output["saved_genus_set_contains_true"] = genus_contains_true
    output["legacy_any_report_correct"] = legacy_correct
    output["corrected_any_report_correct"] = corrected_correct
    output["legacy_species_report_correct"] = legacy_accept & (legacy_label == true_species)
    output["corrected_species_report_correct"] = corrected_accept & (corrected_label == true_species)
    output["legacy_genus_report_correct"] = legacy_genus_report & (legacy_label == true_genus)
    output["corrected_genus_report_correct"] = corrected_genus_report & (corrected_label == true_genus)
    output["decision_status"] = "post-result-correction"

    precision = {
        "run_id": run_id or "",
        "n_rows": int(len(frame)),
        "stored_vs_float32_argmax_mismatch_n": int(np.sum(saved_argmax != float32_argmax)),
        "stored_vs_float32_confidence_max_abs_difference": float(
            np.max(np.abs(saved_confidence - float32_confidence), initial=0.0)
        ),
        "stored_vs_float32_threshold_decision_mismatch_n": int(
            np.sum((saved_confidence >= tau) != (float32_confidence >= tau))
        ),
        "minimum_abs_saved_confidence_minus_tau": float(np.min(np.abs(saved_confidence - tau))),
        "probability_sum_max_abs_error": float(np.max(np.abs(probability.sum(axis=1, dtype=np.float64) - 1.0))),
        "legacy_species_accept_n": int(legacy_accept.sum()),
        "accepted_singleton_not_argmax_n": int((legacy_accept & ~corrected_singleton).sum()),
        "legacy_genus_singleton_not_aggregate_argmax_n": int(
            (legacy_genus_report & (genus_singleton.fillna("").astype(str).to_numpy() != genus_argmax)).sum()
        ),
    }
    return output, precision


def _selector(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    role = frame["role"].astype(str)
    if name == "calibration":
        return frame.loc[role == "calibration"]
    if name == "known":
        return frame.loc[role == "test_known"]
    ood = role == "test_ood"
    if name == "all_ood":
        return frame.loc[ood]
    if name == "same_genus_ood":
        return frame.loc[ood & frame["ood_distance"].astype(str).eq("near")]
    if name == "different_genus_ood":
        return frame.loc[ood & frame["ood_distance"].astype(str).eq("far")]
    if name == "singleton_ood":
        return frame.loc[ood & frame["ood_singleton"].astype(bool)]
    explicit = frame["species"].map(is_explicit_binomial_label).to_numpy(bool)
    if name == "known_explicit_binomial":
        return frame.loc[role.eq("test_known").to_numpy() & explicit]
    if name == "known_sp_only":
        return frame.loc[role.eq("test_known").to_numpy() & ~explicit]
    if name == "same_genus_ood_explicit_binomial":
        return frame.loc[
            ood.to_numpy() & frame["ood_distance"].astype(str).eq("near").to_numpy() & explicit
        ]
    if name == "same_genus_ood_sp_only":
        return frame.loc[
            ood.to_numpy() & frame["ood_distance"].astype(str).eq("near").to_numpy() & ~explicit
        ]
    if name == "different_genus_ood_explicit_binomial":
        return frame.loc[
            ood.to_numpy() & frame["ood_distance"].astype(str).eq("far").to_numpy() & explicit
        ]
    if name == "different_genus_ood_sp_only":
        return frame.loc[
            ood.to_numpy() & frame["ood_distance"].astype(str).eq("far").to_numpy() & ~explicit
        ]
    raise KeyError(name)


def _rate(frame: pd.DataFrame, column: str, aggregation: str) -> float:
    if frame.empty:
        return float("nan")
    values = frame[column].astype(float)
    if aggregation == "spectrum":
        return float(values.mean())
    temp = frame[["species", "strain_id"]].copy()
    temp["value"] = values.to_numpy()
    by_strain = temp.groupby(["species", "strain_id"], sort=False)["value"].mean()
    if aggregation == "strain":
        return float(by_strain.mean())
    if aggregation == "species_macro_strain":
        return float(by_strain.groupby(level="species").mean().mean())
    raise KeyError(aggregation)


def summarize_decisions_per_run(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    selectors = (
        "calibration", "known", "all_ood", "same_genus_ood", "different_genus_ood",
        "singleton_ood", "known_explicit_binomial", "known_sp_only",
        "same_genus_ood_explicit_binomial", "same_genus_ood_sp_only",
        "different_genus_ood_explicit_binomial", "different_genus_ood_sp_only",
    )
    for selector in selectors:
        part = _selector(frame, selector)
        if part.empty:
            continue
        base = {
            "run_id": str(frame["run_id"].iloc[0]),
            "design": str(frame["design"].iloc[0]),
            "model": str(frame["model"].iloc[0]),
            "seed": int(frame["seed"].iloc[0]),
            "fold": int(frame["fold"].iloc[0]),
            "selector": selector,
            "n_prediction_rows": int(len(part)),
            "n_unique_spectra": int(part["spectrum_id"].nunique()),
            "n_strains": int(part["strain_id"].nunique()),
            "n_species": int(part["species"].nunique()),
        }
        for column in OUTCOME_COLUMNS:
            if column == "saved_species_set_contains_true" and selector not in ("calibration", "known"):
                continue
            for aggregation in ("spectrum", "strain", "species_macro_strain"):
                base[f"{column}__{aggregation}"] = _rate(part, column, aggregation)
        for prefix in ("legacy", "corrected"):
            species_n = int(part[f"{prefix}_species_accept"].sum())
            species_correct = int(part[f"{prefix}_species_report_correct"].sum())
            genus_n = int(part[f"{prefix}_genus_report"].sum())
            genus_correct = int(part[f"{prefix}_genus_report_correct"].sum())
            identified_n = species_n + genus_n
            any_correct = int(part[f"{prefix}_any_report_correct"].sum())
            base[f"{prefix}_species_report_n"] = species_n
            base[f"{prefix}_species_report_correct_n"] = species_correct
            base[f"{prefix}_species_report_accuracy"] = species_correct / species_n if species_n else float("nan")
            base[f"{prefix}_species_report_error_rate"] = 1.0 - species_correct / species_n if species_n else float("nan")
            base[f"{prefix}_genus_report_n"] = genus_n
            base[f"{prefix}_genus_report_correct_n"] = genus_correct
            base[f"{prefix}_genus_fallback_accuracy"] = genus_correct / genus_n if genus_n else float("nan")
            base[f"{prefix}_identified_n"] = identified_n
            base[f"{prefix}_identified_correct_n"] = any_correct
            base[f"{prefix}_identified_accuracy"] = any_correct / identified_n if identified_n else float("nan")
        base["mean_saved_species_set_size__spectrum"] = float(part["conformal_species_set_size"].mean())
        base["mean_saved_species_set_size__strain"] = _rate(
            part.assign(_set_size=part["conformal_species_set_size"]), "_set_size", "strain"
        )
        base["mean_saved_species_set_size__species_macro_strain"] = _rate(
            part.assign(_set_size=part["conformal_species_set_size"]), "_set_size", "species_macro_strain"
        )
        rows.append(base)
    return pd.DataFrame(rows)


def calibration_support(
    frame: pd.DataFrame,
    probability: np.ndarray,
    classes: np.ndarray,
    manifest: dict,
) -> pd.DataFrame:
    calibration_mask = frame["role"].astype(str).eq("calibration").to_numpy()
    cal = frame.loc[calibration_mask].reset_index(drop=True)
    prob = probability[calibration_mask].astype(np.float64)
    lookup = {str(label): index for index, label in enumerate(classes.astype(str))}
    y_index = np.array([lookup[str(label)] for label in cal["species"]], dtype=int)
    scores = 1.0 - prob[np.arange(len(cal)), y_index]
    global_q = finite_sample_quantile(scores, 0.95)
    declared_q = float(manifest["conformal_global_threshold"])
    class_to_genus = _taxonomy_map(frame, classes)
    genus_probability, genera = aggregate_genus_probability(prob, classes, class_to_genus)
    genus_lookup = {str(label): index for index, label in enumerate(genera.astype(str))}
    genus_index = np.array([genus_lookup[str(label)] for label in cal["genus"]], dtype=int)
    genus_scores = 1.0 - genus_probability[np.arange(len(cal)), genus_index]
    genus_global_q = finite_sample_quantile(genus_scores, 0.95)
    rows: list[dict] = []
    for level, labels, score_vector, all_labels, global_threshold in (
        ("species", cal["species"].astype(str).to_numpy(), scores, classes.astype(str), global_q),
        ("genus", cal["genus"].astype(str).to_numpy(), genus_scores, genera.astype(str), genus_global_q),
    ):
        for label in all_labels:
            selected = labels == str(label)
            n_spectra = int(selected.sum())
            n_strains = int(cal.loc[selected, "strain_id"].nunique())
            conditional = n_spectra >= 20
            threshold = finite_sample_quantile(score_vector[selected], 0.95) if conditional else global_threshold
            rows.append(
                {
                    "run_id": manifest["run_id"], "design": manifest["design"], "model": manifest["model"],
                    "seed": int(manifest["seed"]), "fold": int(manifest["fold"]), "level": level,
                    "label": str(label), "n_calibration_spectra": n_spectra,
                    "n_calibration_strains": n_strains,
                    "threshold_source": "class-conditional" if conditional else "pooled-fallback",
                    "threshold": float(threshold), "global_threshold": float(global_threshold),
                    "coverage_target": 0.95,
                    "declared_species_global_threshold": declared_q if level == "species" else float("nan"),
                    "declared_vs_recomputed_global_abs_difference": abs(declared_q - global_q) if level == "species" else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _strain_mean_table(frame: pd.DataFrame, probability: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[pd.Series] = []
    vectors: list[np.ndarray] = []
    for (_, _), indices in frame.groupby(["role", "strain_id"], sort=True).indices.items():
        idx = np.asarray(indices, dtype=int)
        part = frame.iloc[idx]
        for column in ("species", "genus", "role"):
            if part[column].astype(str).nunique() != 1:
                raise ValueError(f"strain spans multiple {column} values")
        rows.append(part.iloc[0])
        vectors.append(probability[idx].astype(np.float64).mean(axis=0))
    return pd.DataFrame(rows).reset_index(drop=True), np.vstack(vectors)


def strain_mean_conformal_sensitivity(
    frame: pd.DataFrame,
    probability: np.ndarray,
    classes: np.ndarray,
    manifest: dict,
) -> tuple[pd.DataFrame, dict]:
    means, mean_probability = _strain_mean_table(frame, probability)
    class_to_genus = _taxonomy_map(frame, classes)
    genus_probability, genera = aggregate_genus_probability(mean_probability, classes, class_to_genus)
    class_lookup = {str(label): index for index, label in enumerate(classes.astype(str))}
    genus_lookup = {str(label): index for index, label in enumerate(genera.astype(str))}
    cal = means["role"].astype(str).eq("calibration").to_numpy()
    if not cal.any():
        raise ValueError("pooled strain-mean calibration requires calibration strains")
    y_cal = np.array([class_lookup[str(label)] for label in means.loc[cal, "species"]], dtype=int)
    species_scores = 1.0 - mean_probability[cal][np.arange(cal.sum()), y_cal]
    species_q = finite_sample_quantile(species_scores, 0.95)
    genus_cal = np.array([genus_lookup[str(label)] for label in means.loc[cal, "genus"]], dtype=int)
    genus_scores = 1.0 - genus_probability[cal][np.arange(cal.sum()), genus_cal]
    genus_q = finite_sample_quantile(genus_scores, 0.95)
    calibration_confidence = mean_probability[cal].max(axis=1)
    strain_confidence_tau = empirical_acceptance_threshold(calibration_confidence, 0.95)

    species_included = (1.0 - mean_probability) <= species_q
    genus_included = (1.0 - genus_probability) <= genus_q
    species_sets = [classes[row].astype(str).tolist() for row in species_included]
    genus_sets = [genera[row].astype(str).tolist() for row in genus_included]
    argmax_species = classes[mean_probability.argmax(axis=1)].astype(str)
    argmax_genus = genera[genus_probability.argmax(axis=1)].astype(str)
    confidence = mean_probability.max(axis=1)
    saved_tau_reference = frame.groupby(
        ["role", "strain_id"], sort=True
    )["known_acceptance_threshold"].first().to_numpy(float)
    tau_values = np.full(len(means), strain_confidence_tau, dtype=float)
    species_singleton = np.array([values[0] if len(values) == 1 else "" for values in species_sets], dtype=object)
    genus_singleton = np.array([values[0] if len(values) == 1 else "" for values in genus_sets], dtype=object)
    species_accept = (
        np.array([len(values) == 1 for values in species_sets])
        & (species_singleton == argmax_species)
        & (confidence >= tau_values)
    )
    genus_report = (
        ~species_accept
        & np.array([len(values) == 1 for values in genus_sets])
        & (genus_singleton == argmax_genus)
    )
    reported_level = np.where(species_accept, "species", np.where(genus_report, "genus", "unidentified"))
    reported_label = np.where(species_accept, species_singleton, np.where(genus_report, genus_singleton, "unidentified"))
    true_species = means["species"].astype(str).to_numpy()
    true_genus = means["genus"].astype(str).to_numpy()

    output = means[[
        "spectrum_id", "strain_id", "species", "genus", "role", "analysis_set", "ood_distance",
        "ood_singleton", "design", "model", "seed", "fold",
    ]].copy()
    output.insert(0, "run_id", manifest["run_id"])
    output = output.rename(columns={"spectrum_id": "representative_spectrum_id"})
    output["n_spectra_aggregated"] = frame.groupby(["role", "strain_id"], sort=True).size().to_numpy(int)
    output["strain_mean_argmax_species"] = argmax_species
    output["strain_mean_argmax_genus"] = argmax_genus
    output["strain_mean_confidence"] = confidence
    output["strain_confidence_tau"] = tau_values
    output["v1_saved_tau_reference"] = saved_tau_reference
    output["confidence_threshold_source"] = (
        "calibration-strain-mean-max-probability-lower-5th-percentile"
    )
    output["pooled_species_threshold"] = species_q
    output["pooled_genus_threshold"] = genus_q
    output["species_prediction_set"] = [json.dumps(values, ensure_ascii=False) for values in species_sets]
    output["species_prediction_set_size"] = [len(values) for values in species_sets]
    output["genus_prediction_set"] = [json.dumps(values, ensure_ascii=False) for values in genus_sets]
    output["genus_prediction_set_size"] = [len(values) for values in genus_sets]
    output["confidence_only_accept"] = confidence >= tau_values
    output["set_only_singleton_argmax"] = np.array([len(values) == 1 for values in species_sets]) & (species_singleton == argmax_species)
    output["species_accept"] = species_accept
    output["genus_report"] = genus_report
    output["unidentified"] = reported_level == "unidentified"
    output["reported_level"] = reported_level
    output["reported_label"] = reported_label
    output["species_set_contains_true"] = np.fromiter(
        (truth in values for truth, values in zip(true_species, species_sets)), dtype=bool, count=len(output)
    )
    output["species_report_correct"] = species_accept & (reported_label == true_species)
    output["genus_report_correct"] = genus_report & (reported_label == true_genus)
    output["any_report_correct"] = output["species_report_correct"] | output["genus_report_correct"]
    output["analysis_status"] = "exploratory-nominal-95pct-empirical-strain-mean-calibration"
    support = {
        "run_id": manifest["run_id"], "design": manifest["design"], "model": manifest["model"],
        "seed": int(manifest["seed"]), "fold": int(manifest["fold"]),
        "n_calibration_strains": int(cal.sum()), "n_calibration_species": int(means.loc[cal, "species"].nunique()),
        "species_threshold": float(species_q), "genus_threshold": float(genus_q),
        "strain_confidence_threshold": float(strain_confidence_tau),
        "strain_confidence_threshold_source": "calibration-strain-mean-max-probability-lower-5th-percentile",
        "calibration_confidence_only_acceptance": float(
            np.mean(calibration_confidence >= strain_confidence_tau)
        ),
        "v1_saved_tau": float(manifest["known_acceptance_threshold"]),
        "coverage_target": 0.95,
        "species_upper_bound_used": bool(species_q == 1.0), "genus_upper_bound_used": bool(genus_q == 1.0),
        "interpretation": "nominal empirical sensitivity conditional on fitted model; no formal class-conditional or OOD guarantee",
    }
    return output, support


def _strain_mean_metric_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for selector in (
        "calibration", "known", "all_ood", "same_genus_ood", "different_genus_ood",
        "singleton_ood", "known_explicit_binomial", "known_sp_only",
        "same_genus_ood_explicit_binomial", "same_genus_ood_sp_only",
        "different_genus_ood_explicit_binomial", "different_genus_ood_sp_only",
    ):
        part = _selector(frame.rename(columns={
            "species_accept": "corrected_species_accept", "genus_report": "corrected_genus_report",
            "unidentified": "corrected_unidentified", "species_set_contains_true": "saved_species_set_contains_true",
        }), selector)
        if part.empty:
            continue
        row = {
            "run_id": frame["run_id"].iloc[0], "design": frame["design"].iloc[0], "model": frame["model"].iloc[0],
            "seed": int(frame["seed"].iloc[0]), "fold": int(frame["fold"].iloc[0]), "selector": selector,
            "n_strains": int(len(part)), "n_species": int(part["species"].nunique()),
        }
        for column in ("confidence_only_accept", "set_only_singleton_argmax", "corrected_species_accept", "corrected_genus_report", "corrected_unidentified"):
            row[column] = float(part[column].astype(float).mean())
        if selector in ("calibration", "known"):
            row["species_set_coverage"] = float(part["saved_species_set_contains_true"].astype(float).mean())
        species_n = int(part["corrected_species_accept"].sum())
        genus_n = int(part["corrected_genus_report"].sum())
        row["species_report_n"] = species_n
        row["species_report_accuracy"] = float(part.loc[part["corrected_species_accept"], "species_report_correct"].mean()) if species_n else float("nan")
        row["genus_report_n"] = genus_n
        row["genus_fallback_accuracy"] = float(part.loc[part["corrected_genus_report"], "genus_report_correct"].mean()) if genus_n else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def hierarchical_occurrence_bootstrap(
    frame: pd.DataFrame,
    value_columns: Sequence[str],
    replicates: int,
    seed: int,
    instance_columns: Sequence[str] = ("seed", "fold"),
) -> tuple[np.ndarray, np.ndarray]:
    """Species-occurrence bootstrap with an independent strain draw per occurrence.

    All available model-instance observations for a sampled strain remain linked.
    The point estimator is: mean over instances of the species-macro mean of
    within-species strain means.
    """
    required = {"species", "strain_id", *instance_columns, *value_columns}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"bootstrap columns missing: {sorted(missing)}")
    if frame.empty or replicates < 1:
        raise ValueError("bootstrap needs observations and at least one replicate")
    work = frame[["species", "strain_id", *instance_columns, *value_columns]].copy()
    for column in value_columns:
        work[column] = pd.to_numeric(work[column], errors="raise").astype(float)
    grouped = work.groupby(["species", "strain_id", *instance_columns], sort=True, as_index=False)[list(value_columns)].mean()
    instance_keys = list(grouped[list(instance_columns)].drop_duplicates().itertuples(index=False, name=None))
    instance_lookup = {key: index for index, key in enumerate(instance_keys)}
    species_arrays: list[np.ndarray] = []
    for _, species_part in grouped.groupby("species", sort=True):
        strains = sorted(species_part["strain_id"].astype(str).unique())
        strain_lookup = {label: index for index, label in enumerate(strains)}
        array = np.full((len(strains), len(instance_keys), len(value_columns)), np.nan, dtype=float)
        for row in species_part.itertuples(index=False):
            key = tuple(getattr(row, column) for column in instance_columns)
            values = [float(getattr(row, column)) for column in value_columns]
            array[strain_lookup[str(row.strain_id)], instance_lookup[key], :] = values
        species_arrays.append(array)

    def nanmean(values: np.ndarray, axis: int) -> np.ndarray:
        # A resampled species occurrence can legitimately contain no strain from
        # one held-out fold. That cell remains missing; other occurrences still
        # contribute to the model-instance mean.
        with warnings.catch_warnings(), np.errstate(invalid="ignore"):
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(values, axis=axis)

    def aggregate(species_means: list[np.ndarray]) -> np.ndarray:
        per_instance = nanmean(np.stack(species_means, axis=0), axis=0)
        return nanmean(per_instance, axis=0)

    point = aggregate([nanmean(array, axis=0) for array in species_arrays])
    rng = np.random.default_rng(seed)
    draws = np.empty((replicates, len(value_columns)), dtype=float)
    for replicate in range(replicates):
        occurrence_indices = rng.integers(0, len(species_arrays), size=len(species_arrays))
        occurrence_means: list[np.ndarray] = []
        for species_index in occurrence_indices:
            array = species_arrays[int(species_index)]
            sampled_strains = rng.integers(0, len(array), size=len(array))
            occurrence_means.append(nanmean(array[sampled_strains], axis=0))
        draws[replicate] = aggregate(occurrence_means)
    return point, draws


def hierarchical_occurrence_bootstrap_overall_strain(
    frame: pd.DataFrame,
    value_columns: Sequence[str],
    replicates: int,
    seed: int,
    instance_columns: Sequence[str] = ("seed", "fold"),
) -> tuple[np.ndarray, np.ndarray]:
    """Hierarchical bootstrap for the overall strain-balanced companion.

    Species occurrences and strains are sampled exactly as in the OOD
    hierarchical bootstrap, but each model instance then gives every sampled
    strain one vote instead of giving every species one vote.
    """
    required = {"species", "strain_id", *instance_columns, *value_columns}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"bootstrap columns missing: {sorted(missing)}")
    if frame.empty or replicates < 1:
        raise ValueError("bootstrap needs observations and at least one replicate")
    work = frame[["species", "strain_id", *instance_columns, *value_columns]].copy()
    for column in value_columns:
        work[column] = pd.to_numeric(work[column], errors="raise").astype(float)
    grouped = work.groupby(
        ["species", "strain_id", *instance_columns], sort=True, as_index=False
    )[list(value_columns)].mean()
    instance_keys = list(
        grouped[list(instance_columns)].drop_duplicates().itertuples(index=False, name=None)
    )
    instance_lookup = {key: index for index, key in enumerate(instance_keys)}
    species_arrays: list[np.ndarray] = []
    for _, species_part in grouped.groupby("species", sort=True):
        strains = sorted(species_part["strain_id"].astype(str).unique())
        strain_lookup = {label: index for index, label in enumerate(strains)}
        array = np.full(
            (len(strains), len(instance_keys), len(value_columns)), np.nan, dtype=float
        )
        for row in species_part.itertuples(index=False):
            key = tuple(getattr(row, column) for column in instance_columns)
            values = [float(getattr(row, column)) for column in value_columns]
            array[strain_lookup[str(row.strain_id)], instance_lookup[key], :] = values
        species_arrays.append(array)

    def aggregate(arrays: list[np.ndarray]) -> np.ndarray:
        total = np.zeros((len(instance_keys), len(value_columns)), dtype=float)
        count = np.zeros_like(total)
        for array in arrays:
            finite = np.isfinite(array)
            total += np.where(finite, array, 0.0).sum(axis=0)
            count += finite.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            per_instance = total / count
            return np.nanmean(per_instance, axis=0)

    point = aggregate(species_arrays)
    rng = np.random.default_rng(seed)
    draws = np.empty((replicates, len(value_columns)), dtype=float)
    for replicate in range(replicates):
        occurrence_indices = rng.integers(0, len(species_arrays), size=len(species_arrays))
        sampled_occurrences: list[np.ndarray] = []
        for species_index in occurrence_indices:
            array = species_arrays[int(species_index)]
            sampled_strains = rng.integers(0, len(array), size=len(array))
            sampled_occurrences.append(array[sampled_strains])
        draws[replicate] = aggregate(sampled_occurrences)
    return point, draws


def _bootstrap_tables(decisions: pd.DataFrame, replicates: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict] = []
    replicate_rows: list[pd.DataFrame] = []
    bootstrap_columns = [
        "confidence_only_accept", "saved_singleton", "corrected_singleton_argmax",
        "legacy_species_accept", "corrected_species_accept", "legacy_genus_report",
        "corrected_genus_report", "legacy_unidentified", "corrected_unidentified",
        "legacy_any_report_correct", "corrected_any_report_correct",
    ]

    def record_bootstrap(
        *,
        model: str,
        selector: str,
        part: pd.DataFrame,
        columns: list[str],
        point: np.ndarray,
        draws: np.ndarray,
        draw_seed: int,
        label_scope: str,
        estimand: str,
        analysis_role: str,
        bootstrap_unit: str,
    ) -> None:
        rep = pd.DataFrame(draws, columns=columns)
        rep.insert(0, "replicate", np.arange(1, replicates + 1))
        rep.insert(0, "analysis_role", analysis_role)
        rep.insert(0, "estimand", estimand)
        rep.insert(0, "label_scope", label_scope)
        rep.insert(0, "selector", selector)
        rep.insert(0, "model", model)
        replicate_rows.append(rep)
        counts = {
            "n_prediction_rows": int(len(part)),
            "n_spectra": int(part["spectrum_id"].nunique()),
            "n_strains": int(part["strain_id"].nunique()),
            "n_taxonomic_labels": int(part["species"].nunique()),
            "n_model_instances": int(part[["seed", "fold"]].drop_duplicates().shape[0]),
        }
        for index, column in enumerate(columns):
            low, high = np.quantile(draws[:, index], [0.025, 0.975])
            summary_rows.append(
                {
                    "design": "strain_grouped", "model": model, "selector": selector,
                    "label_scope": label_scope, "estimand": estimand,
                    "analysis_role": analysis_role, "metric": column,
                    "estimate": float(point[index]), "ci_low": float(low), "ci_high": float(high),
                    "bootstrap_replicates": int(replicates), "bootstrap_seed": int(draw_seed),
                    "bootstrap_unit": bootstrap_unit,
                    "inference_scope": "conditional on fitted models and fixed calibration/reporting rules",
                    **counts,
                }
            )

    ood_specs = (
        (
            "all_ood_taxonomic_label_macro", "all_ood", "all_taxonomic_labels",
            "taxonomic-label-macro", "descriptive_all_ood",
        ),
        (
            "same_genus_ood_taxonomic_label_macro", "same_genus_ood", "all_taxonomic_labels",
            "taxonomic-label-macro", "primary_all_label",
        ),
        (
            "different_genus_ood_taxonomic_label_macro", "different_genus_ood", "all_taxonomic_labels",
            "taxonomic-label-macro", "primary_all_label",
        ),
        (
            "singleton_ood_taxonomic_label_macro", "singleton_ood", "all_taxonomic_labels",
            "taxonomic-label-macro", "descriptive_singleton_ood",
        ),
        (
            "same_genus_ood_explicit_binomial", "same_genus_ood_explicit_binomial", "explicit_binomial_only",
            "explicit-binomial-label-macro", "sensitivity_explicit_binomial",
        ),
        (
            "different_genus_ood_explicit_binomial", "different_genus_ood_explicit_binomial", "explicit_binomial_only",
            "explicit-binomial-label-macro", "sensitivity_explicit_binomial",
        ),
        (
            "same_genus_ood_sp_only", "same_genus_ood_sp_only", "genus_sp_label_only",
            "sp-label-macro", "descriptive_sp_label_only",
        ),
        (
            "different_genus_ood_sp_only", "different_genus_ood_sp_only", "genus_sp_label_only",
            "sp-label-macro", "descriptive_sp_label_only",
        ),
    )
    for model_index, model in enumerate(COMPLETE_MODELS):
        model_frame = decisions.loc[
            decisions["design"].eq("strain_grouped") & decisions["model"].eq(model)
        ]
        for selector_index, (output_selector, input_selector, label_scope, estimand, role) in enumerate(ood_specs):
            part = _selector(model_frame, input_selector)
            if part.empty:
                continue
            columns = list(bootstrap_columns)
            part = part.copy()
            part["legacy_minus_corrected_species_accept"] = (
                part["legacy_species_accept"].astype(float) - part["corrected_species_accept"].astype(float)
            )
            columns.append("legacy_minus_corrected_species_accept")
            draw_seed = seed + model_index * 1009 + selector_index * 37
            point, draws = hierarchical_occurrence_bootstrap(
                part, columns, replicates, draw_seed
            )
            record_bootstrap(
                model=model, selector=output_selector, part=part, columns=columns,
                point=point, draws=draws, draw_seed=draw_seed, label_scope=label_scope,
                estimand=estimand, analysis_role=role,
                bootstrap_unit=(
                    "species occurrence then independently resampled strains; "
                    "taxonomic-label-macro strain rate within model instance, then instances averaged"
                ),
            )

        known_specs = (
            ("known", "known", "all_taxonomic_labels", "companion_all_label"),
            (
                "known_explicit_binomial", "known_explicit_binomial",
                "explicit_binomial_only", "sensitivity_explicit_binomial",
            ),
            ("known_sp_only", "known_sp_only", "genus_sp_label_only", "descriptive_sp_label_only"),
        )
        for known_index, (output_selector, input_selector, label_scope, role) in enumerate(known_specs):
            known = _selector(model_frame, input_selector).copy()
            known["legacy_minus_corrected_species_accept"] = (
                known["legacy_species_accept"].astype(float)
                - known["corrected_species_accept"].astype(float)
            )
            columns = [
                *bootstrap_columns, "saved_species_set_contains_true",
                "legacy_minus_corrected_species_accept",
            ]
            draw_seed = seed + model_index * 1009 + 701 + known_index * 41
            point, draws = hierarchical_occurrence_bootstrap_overall_strain(
                known, columns, replicates, draw_seed
            )
            record_bootstrap(
                model=model, selector=output_selector, part=known, columns=columns,
                point=point, draws=draws, draw_seed=draw_seed, label_scope=label_scope,
                estimand="overall-strain-balanced", analysis_role=role,
                bootstrap_unit=(
                    "species occurrences resampled, then strains independently resampled; "
                    "overall strain-balanced rate within model instance, then instances averaged"
                ),
            )

        known = _selector(model_frame, "known").copy()
        known["legacy_minus_corrected_species_accept"] = (
            known["legacy_species_accept"].astype(float)
            - known["corrected_species_accept"].astype(float)
        )
        known_columns = [
            *bootstrap_columns, "saved_species_set_contains_true",
            "legacy_minus_corrected_species_accept",
        ]
        macro_seed = seed + model_index * 1009 + 809
        macro_point, macro_draws = hierarchical_occurrence_bootstrap(
            known, known_columns, replicates, macro_seed
        )
        record_bootstrap(
            model=model, selector="known_taxonomic_label_macro", part=known,
            columns=known_columns, point=macro_point, draws=macro_draws,
            draw_seed=macro_seed, label_scope="all_taxonomic_labels",
            estimand="taxonomic-label-macro", analysis_role="sensitivity_known_label_macro",
            bootstrap_unit=(
                "species occurrences resampled, then strains independently resampled; "
                "taxonomic-label-macro strain rate within model instance, then instances averaged"
            ),
        )
    return pd.DataFrame(summary_rows), pd.concat(replicate_rows, ignore_index=True)


def _leakage_metrics(leakage: dict[tuple[str, str, int], list[tuple[pd.DataFrame, np.ndarray, np.ndarray]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict] = []
    strain_prediction_rows: list[pd.DataFrame] = []
    for (design, model, seed), parts in sorted(leakage.items()):
        frames = pd.concat([part[0] for part in parts], ignore_index=True)
        probability = np.vstack([part[1] for part in parts])
        classes = parts[0][2].astype(str)
        if any(not np.array_equal(classes, part[2].astype(str)) for part in parts):
            raise ValueError(f"class order changed across folds: {(design, model, seed)}")
        if frames["spectrum_id"].duplicated().any():
            raise ValueError(f"known OOF spectra are duplicated: {(design, model, seed)}")
        truth = frames["species"].astype(str).to_numpy()
        predicted = frames["predicted_species"].astype(str).to_numpy()
        counts = frames.groupby("strain_id")["spectrum_id"].transform("size").to_numpy(float)
        weights = 1.0 / counts
        grouped_indices = frames.groupby("strain_id", sort=True).indices
        strain_meta: list[pd.Series] = []
        strain_probability: list[np.ndarray] = []
        for _, indices in grouped_indices.items():
            idx = np.asarray(indices, dtype=int)
            part = frames.iloc[idx]
            if part["species"].astype(str).nunique() != 1:
                raise ValueError("strain has inconsistent species")
            strain_meta.append(part.iloc[0])
            strain_probability.append(probability[idx].astype(np.float64).mean(axis=0))
        strain_frame = pd.DataFrame(strain_meta).reset_index(drop=True)
        strain_prob = np.vstack(strain_probability)
        strain_pred = classes[strain_prob.argmax(axis=1)]
        strain_truth = strain_frame["species"].astype(str).to_numpy()
        metric_rows.append(
            {
                "design": design, "model": model, "seed": int(seed),
                "n_spectra": int(len(frames)), "n_strains": int(len(strain_frame)),
                "n_species": int(frames["species"].nunique()),
                "pooled_spectrum_macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
                "equal_strain_weight_macro_f1": float(
                    f1_score(truth, predicted, average="macro", sample_weight=weights, zero_division=0)
                ),
                "strain_mean_probability_macro_f1": float(
                    f1_score(strain_truth, strain_pred, average="macro", zero_division=0)
                ),
                "strain_mean_probability_balanced_accuracy": float(balanced_accuracy_score(strain_truth, strain_pred)),
            }
        )
        strain_prediction_rows.append(
            pd.DataFrame(
                {
                    "design": design, "model": model, "seed": int(seed),
                    "strain_id": strain_frame["strain_id"].astype(str),
                    "species": strain_truth, "predicted_species": strain_pred,
                    "n_spectra": [len(grouped_indices[str(label)]) for label in strain_frame["strain_id"].astype(str)],
                    "confidence": strain_prob.max(axis=1),
                }
            )
        )
    return pd.DataFrame(metric_rows), pd.concat(strain_prediction_rows, ignore_index=True)


def _leakage_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    paired = metrics.pivot(index=["model", "seed"], columns="design", values=[
        "pooled_spectrum_macro_f1", "equal_strain_weight_macro_f1", "strain_mean_probability_macro_f1"
    ])
    rows = []
    for model in COMPLETE_MODELS:
        part = paired.loc[model]
        for metric in ("pooled_spectrum_macro_f1", "equal_strain_weight_macro_f1", "strain_mean_probability_macro_f1"):
            random_values = part[(metric, "spectrum_random")].to_numpy(float)
            grouped_values = part[(metric, "strain_grouped")].to_numpy(float)
            differences = random_values - grouped_values
            rows.append(
                {
                    "model": model, "metric": metric, "n_seeds": int(len(part)),
                    "spectrum_random_mean": float(random_values.mean()),
                    "strain_grouped_mean": float(grouped_values.mean()),
                    "random_minus_grouped_mean": float(differences.mean()),
                    "random_minus_grouped_sd": float(differences.std(ddof=1)),
                    "random_minus_grouped_min": float(differences.min()),
                    "random_minus_grouped_max": float(differences.max()),
                    "status": "sensitivity; no model refitting",
                }
            )
    return pd.DataFrame(rows)


def _aggregate_per_run_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["design", "model", "selector"]
    excluded = {*keys, "run_id", "seed", "fold"}
    numeric = [column for column in metrics.select_dtypes(include=[np.number]).columns if column not in excluded]
    rows: list[dict] = []
    for key, part in metrics.groupby(keys, sort=True):
        for column in numeric:
            values = part[column].dropna().astype(float)
            if values.empty:
                continue
            rows.append(
                {
                    **dict(zip(keys, key)), "metric": column, "n_model_instances": int(len(values)),
                    "mean": float(values.mean()), "sd": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
                    "minimum": float(values.min()), "maximum": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def _git_head(project_root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()


def run_analysis(
    project_root: str | Path,
    production_root: str | Path,
    output_dir: str | Path,
    spec_path: str | Path,
    bootstrap_replicates: int | None = None,
    frozen_input_manifest_path: str | Path | None = None,
) -> dict:
    project_root = Path(project_root).resolve()
    production_root = Path(production_root).resolve()
    output_dir = Path(output_dir).resolve()
    spec_path = Path(spec_path).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec["legacy_analysis_commit"] != ANALYSIS_COMMIT:
        raise ValueError("v2 spec is not bound to the locked analysis commit")
    if production_root != (project_root / "output" / "production").resolve():
        raise ValueError("production input root must be the immutable output/production directory")
    expected_output_parent = (project_root / "output" / "v2").resolve()
    if expected_output_parent not in output_dir.parents:
        raise ValueError("v2 output must be inside output/v2")
    output_dir.mkdir(parents=True, exist_ok=True)
    n_bootstrap = int(bootstrap_replicates or spec["bootstrap_replicates"])

    frozen_input_manifest_path = Path(
        frozen_input_manifest_path or project_root / "output" / "v2" / "input_manifest.json"
    ).resolve()
    frozen, frozen_manifest_sha256, frozen_runs = load_and_verify_frozen_inputs(
        project_root, frozen_input_manifest_path, spec
    )

    run_dirs = discover_locked_runs(production_root, spec, frozen_runs)
    decisions: list[pd.DataFrame] = []
    precision_rows: list[dict] = []
    metric_rows: list[pd.DataFrame] = []
    calibration_rows: list[pd.DataFrame] = []
    strain_conformal: list[pd.DataFrame] = []
    strain_support: list[dict] = []
    strain_metric_rows: list[pd.DataFrame] = []
    input_records: list[dict] = []
    leakage: dict[tuple[str, str, int], list[tuple[pd.DataFrame, np.ndarray, np.ndarray]]] = defaultdict(list)

    for run_dir in run_dirs:
        bundle = load_locked_run(
            run_dir, spec, project_root, frozen_record=frozen_runs[run_dir.name]
        )
        decision, precision = build_decision_frame(
            bundle.frame, bundle.probability, bundle.classes, bundle.manifest["run_id"]
        )
        decisions.append(decision)
        precision.update(
            {
                "design": bundle.manifest["design"], "model": bundle.manifest["model"],
                "seed": int(bundle.manifest["seed"]), "fold": int(bundle.manifest["fold"]),
            }
        )
        precision_rows.append(precision)
        metric_rows.append(summarize_decisions_per_run(decision))
        calibration_rows.append(calibration_support(bundle.frame, bundle.probability, bundle.classes, bundle.manifest))
        strain_decision, support = strain_mean_conformal_sensitivity(
            bundle.frame, bundle.probability, bundle.classes, bundle.manifest
        )
        strain_conformal.append(strain_decision)
        strain_support.append(support)
        strain_metric_rows.append(_strain_mean_metric_rows(strain_decision))
        known = bundle.frame["role"].astype(str).eq("test_known").to_numpy()
        leakage[(bundle.manifest["design"], bundle.manifest["model"], int(bundle.manifest["seed"]))].append(
            (bundle.frame.loc[known].reset_index(drop=True), bundle.probability[known], bundle.classes)
        )
        input_records.append(bundle.input_record)

    all_decisions = pd.concat(decisions, ignore_index=True)
    all_metrics = pd.concat(metric_rows, ignore_index=True)
    all_calibration = pd.concat(calibration_rows, ignore_index=True)
    all_strain = pd.concat(strain_conformal, ignore_index=True)
    all_strain_metrics = pd.concat(strain_metric_rows, ignore_index=True)
    leakage_metrics, strain_leakage_predictions = _leakage_metrics(leakage)
    bootstrap_summary, bootstrap_draws = _bootstrap_tables(
        all_decisions, n_bootstrap, int(spec["bootstrap_seed"])
    )

    outputs: dict[str, Path] = {
        "decisions": output_dir / "decisions.parquet",
        "per_run_rule_metrics": output_dir / "per_run_rule_metrics.csv",
        "aggregate_rule_metrics": output_dir / "aggregate_rule_metrics.csv",
        "calibration_support": output_dir / "calibration_support.csv",
        "precision_audit": output_dir / "precision_audit.csv",
        "hierarchical_bootstrap": output_dir / "hierarchical_bootstrap.csv",
        "hierarchical_bootstrap_replicates": output_dir / "hierarchical_bootstrap_replicates.parquet",
        "label_type_bootstrap": output_dir / "label_type_bootstrap.csv",
        "strain_mean_conformal_decisions": output_dir / "strain_mean_conformal_decisions.parquet",
        "strain_mean_conformal_support": output_dir / "strain_mean_conformal_support.csv",
        "strain_mean_conformal_metrics": output_dir / "strain_mean_conformal_metrics.csv",
        "strain_balanced_leakage": output_dir / "strain_balanced_leakage.csv",
        "strain_balanced_leakage_summary": output_dir / "strain_balanced_leakage_summary.csv",
        "strain_level_leakage_predictions": output_dir / "strain_level_leakage_predictions.parquet",
    }
    all_decisions.to_parquet(outputs["decisions"], index=False)
    all_metrics.to_csv(outputs["per_run_rule_metrics"], index=False)
    _aggregate_per_run_metrics(all_metrics).to_csv(outputs["aggregate_rule_metrics"], index=False)
    all_calibration.to_csv(outputs["calibration_support"], index=False)
    pd.DataFrame(precision_rows).to_csv(outputs["precision_audit"], index=False)
    bootstrap_summary.to_csv(outputs["hierarchical_bootstrap"], index=False)
    bootstrap_draws.to_parquet(outputs["hierarchical_bootstrap_replicates"], index=False)
    bootstrap_summary.loc[
        bootstrap_summary["label_scope"].isin(
            ["explicit_binomial_only", "genus_sp_label_only"]
        )
    ].to_csv(outputs["label_type_bootstrap"], index=False)
    all_strain.to_parquet(outputs["strain_mean_conformal_decisions"], index=False)
    pd.DataFrame(strain_support).to_csv(outputs["strain_mean_conformal_support"], index=False)
    all_strain_metrics.to_csv(outputs["strain_mean_conformal_metrics"], index=False)
    leakage_metrics.to_csv(outputs["strain_balanced_leakage"], index=False)
    _leakage_summary(leakage_metrics).to_csv(outputs["strain_balanced_leakage_summary"], index=False)
    strain_leakage_predictions.to_parquet(outputs["strain_level_leakage_predictions"], index=False)

    input_manifest = {
        "schema_version": "2.0.0", "analysis_commit": ANALYSIS_COMMIT,
        "current_git_head": _git_head(project_root), "feature_sha256": spec["feature_sha256"],
        "frozen_input_manifest": {
            "path": frozen_input_manifest_path.relative_to(project_root).as_posix(),
            "sha256": frozen_manifest_sha256,
            "declared_run_count": int(frozen["run_count"]),
        },
        "analysis_spec": {
            "path": spec_path.relative_to(project_root).as_posix(), "sha256": sha256_file(spec_path),
        },
        "n_locked_runs": len(input_records), "runs": input_records,
    }
    _json_dump(output_dir / "input_manifest.json", input_manifest)

    primary = bootstrap_summary.loc[
        bootstrap_summary["model"].eq("extra_trees")
        & bootstrap_summary["selector"].isin(
            [
                "known",
                "same_genus_ood_taxonomic_label_macro",
                "different_genus_ood_taxonomic_label_macro",
                "known_explicit_binomial",
                "same_genus_ood_explicit_binomial",
                "different_genus_ood_explicit_binomial",
            ]
        )
        & bootstrap_summary["metric"].isin([
            "legacy_species_accept", "corrected_species_accept", "legacy_minus_corrected_species_accept"
        ])
    ].to_dict(orient="records")
    summary = {
        "schema_version": "2.0.0", "created_at": datetime.now(timezone.utc).isoformat(),
        "analysis_commit": ANALYSIS_COMMIT, "current_git_head_provenance": _git_head(project_root),
        "frozen_input_manifest_sha256": frozen_manifest_sha256,
        "status": "post-result correction and exploratory sensitivity",
        "complete_models": list(COMPLETE_MODELS), "designs": list(DESIGNS), "n_locked_runs": len(run_dirs),
        "bootstrap_replicates": n_bootstrap,
        "corrected_species_rule": spec["corrected_species_rule"],
        "corrected_genus_rule": spec["corrected_genus_rule"],
        "primary_corrected_results": primary,
        "interpretation": {
            "correction": "post-result; saved thresholds were not retuned",
            "bootstrap": "conditional on fitted models and fixed calibration/reporting rules",
            "strain_conformal": "nominal 95% empirical strain-mean sensitivity, not a formal class-conditional or OOD guarantee",
        },
    }
    _json_dump(output_dir / "summary.json", summary)
    output_hashes = {
        name: {"path": path.relative_to(project_root).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for name, path in outputs.items()
    }
    run_manifest = {
        **summary,
        "inputs_manifest_sha256": sha256_file(output_dir / "input_manifest.json"),
        "source_files": {
            "module": {
                "path": Path(__file__).resolve().relative_to(project_root).as_posix(),
                "sha256": sha256_file(Path(__file__).resolve()),
            }
        },
        "outputs": output_hashes,
    }
    _json_dump(output_dir / "run_manifest.json", run_manifest)
    return run_manifest
