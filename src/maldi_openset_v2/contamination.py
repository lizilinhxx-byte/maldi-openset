"""Paired technical-replicate contamination experiments for MALDI-OpenSet v2.

The module deliberately separates two mechanisms that the legacy dose-response
analysis conflated:

* prevalence: the fraction of evaluation strains with one technical replicate
  leaked into training; and
* intensity: the number of leaked technical replicates per exposed strain.

Every leakage condition is paired with a same-species, different-strain donor
condition containing exactly the same number and species composition of added
spectra.  Evaluation and donor strains are quarantined from the baseline
training set.  All selections use SHA-256 ordering and are therefore invariant
to the input table's row order.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


SCHEMA_VERSION = "2.0.0"
PREVALENCE_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
INTENSITY_LEVELS = (0, 1, 2, 4)
EXPECTED_FEATURE_SHA256 = "7a57e01a7ed0adf6b76bf1403192f7988802e230f94a13424f35ce78bfbb4f45"
# The 400 production fits were generated before the P1 aggregation hardening.
# Scientific fit/assignment code did not change in that review.  Keep its hash
# frozen so metadata-only aggregation changes do not force a model refit.
FROZEN_FIT_ENGINE_SOURCE_SHA256 = "ef45177d9c225508c276ad87b63a4872235be96673b8cfc1bfece22e081db131"
LEGACY_ANALYSIS_COMMIT = "2293ce52a07e24e507e48814dcd3303bf512c0f3"


@dataclass(frozen=True)
class ExperimentDefinition:
    name: str
    minimum_spectra_per_strain: int
    spectrum_cap: int
    minimum_strains_per_species: int
    levels: tuple[float | int, ...]


EXPERIMENTS: Mapping[str, ExperimentDefinition] = {
    "prevalence": ExperimentDefinition(
        name="prevalence",
        minimum_spectra_per_strain=2,
        spectrum_cap=2,
        minimum_strains_per_species=5,
        levels=PREVALENCE_LEVELS,
    ),
    "intensity": ExperimentDefinition(
        name="intensity",
        minimum_spectra_per_strain=5,
        spectrum_cap=5,
        minimum_strains_per_species=5,
        levels=INTENSITY_LEVELS,
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stable_token(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stable_seed(*parts: object) -> int:
    return int(_stable_token(*parts)[:8], 16) & 0x7FFFFFFF


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def _source_sha256() -> str:
    return sha256_file(Path(__file__))


def validate_frozen_inputs(project_root: str | Path) -> dict[str, Any]:
    """Verify the v2 freeze ledger and the exact feature/manifest/split inputs.

    The current Git HEAD is provenance only.  It is intentionally not an
    execution gate because v2 code and publication artifacts live after the
    immutable legacy analysis commit.
    """

    root = Path(project_root).resolve()
    ledger_path = root / "output/v2/input_manifest.json"
    ledger_sha = sha256_file(ledger_path)
    # The global freeze ledger can be regenerated when its complete run-key
    # inventory is strengthened.  Bind this invocation to the bytes actually
    # validated below instead of pinning an obsolete ledger-file hash.
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("frozen input manifest schema differs")
    if ledger.get("legacy_analysis_commit") != LEGACY_ANALYSIS_COMMIT:
        raise RuntimeError("legacy analysis commit identity differs")
    if ledger.get("run_count") != 210 or not ledger.get("v1_write_protected_by_contract"):
        raise RuntimeError("frozen v1 run inventory is incomplete or not write-protected")
    if ledger.get("feature_sha256") != EXPECTED_FEATURE_SHA256:
        raise RuntimeError("frozen feature identity differs")

    file_entries = {str(item["path"]): item for item in ledger.get("files", [])}
    required_paths = (
        "config/v2/analysis_spec.json",
        "data/processed/production/manifest.parquet",
        "data/processed/production/feature_metadata.json",
        "data/processed/splits.json",
        "data/processed/splits.parquet",
    )
    verified_files: dict[str, dict[str, Any]] = {}
    for relative in required_paths:
        if relative not in file_entries:
            raise RuntimeError(f"frozen input manifest lacks {relative}")
        path = root / relative
        entry = file_entries[relative]
        actual_sha = sha256_file(path)
        if actual_sha != entry.get("sha256") or path.stat().st_size != int(entry.get("bytes", -1)):
            raise RuntimeError(f"frozen input identity differs for {relative}")
        verified_files[relative] = {
            "bytes": path.stat().st_size,
            "sha256": actual_sha,
        }

    feature_path = root / str(ledger.get("feature_matrix_path"))
    feature_sha = sha256_file(feature_path)
    if (
        feature_sha != EXPECTED_FEATURE_SHA256
        or feature_path.stat().st_size != int(ledger.get("feature_matrix_bytes", -1))
    ):
        raise RuntimeError("frozen feature matrix bytes differ")
    feature_metadata = json.loads(
        (root / "data/processed/production/feature_metadata.json").read_text(encoding="utf-8")
    )
    if feature_metadata.get("features_sha256") != feature_sha:
        raise RuntimeError("feature metadata and frozen matrix identity differ")
    analysis_spec = json.loads(
        (root / "config/v2/analysis_spec.json").read_text(encoding="utf-8")
    )
    if analysis_spec.get("feature_sha256") != feature_sha:
        raise RuntimeError("v2 analysis specification and frozen feature identity differ")

    try:
        current_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        current_head = None
    return {
        "input_manifest_path": ledger_path.relative_to(root).as_posix(),
        "input_manifest_sha256": ledger_sha,
        "legacy_analysis_commit": LEGACY_ANALYSIS_COMMIT,
        "current_head_at_execution": current_head,
        "feature_matrix": {
            "path": feature_path.relative_to(root).as_posix(),
            "bytes": feature_path.stat().st_size,
            "sha256": feature_sha,
        },
        "verified_files": verified_files,
        "split_identity_verified": True,
        "current_head_is_provenance_only": True,
    }


def strain_balanced_weights(labels: Sequence[str], strain_ids: Sequence[str]) -> np.ndarray:
    """Equal total class mass and equal strain mass within each class."""

    labels_array = np.asarray(labels).astype(str)
    strain_array = np.asarray(strain_ids).astype(str)
    weights = np.zeros(len(labels_array), dtype=np.float64)
    for label in np.unique(labels_array):
        class_mask = labels_array == label
        class_strains = np.unique(strain_array[class_mask])
        for strain_id in class_strains:
            mask = class_mask & (strain_array == strain_id)
            weights[mask] = 1.0 / (len(class_strains) * int(mask.sum()))
    if not len(weights) or weights.sum() <= 0:
        raise ValueError("training weights are empty")
    weights *= len(weights) / weights.sum()
    return weights


def build_cohort(
    manifest: pd.DataFrame,
    definition: ExperimentDefinition,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Freeze an eligible, capped spectrum cohort without relying on row order.

    Returns a spectrum-level cohort and a strain-level inventory.  Only the
    primary-known set is eligible, matching the closed-set study population.
    """

    required = {"spectrum_id", "strain_id", "species", "primary_known"}
    missing = required.difference(manifest.columns)
    if missing:
        raise KeyError(f"manifest missing columns: {sorted(missing)}")
    work = manifest.loc[:, list(required)].copy()
    work["feature_row"] = np.arange(len(manifest), dtype=np.int64)
    work = work.loc[work["primary_known"].fillna(False).astype(bool)].copy()
    for column in ("spectrum_id", "strain_id", "species"):
        work[column] = work[column].astype(str)
    if work["spectrum_id"].duplicated().any():
        raise ValueError("spectrum_id must be unique")
    identity_species = work.groupby("strain_id", sort=False)["species"].nunique()
    if int(identity_species.max()) != 1:
        raise ValueError("a strain_id maps to more than one species")

    strain_counts = (
        work.groupby(["species", "strain_id"], sort=False)
        .size()
        .rename("available_spectra")
        .reset_index()
    )
    eligible = strain_counts.loc[
        strain_counts["available_spectra"] >= definition.minimum_spectra_per_strain
    ].copy()
    species_counts = eligible.groupby("species", sort=False)["strain_id"].nunique()
    included_species = set(
        species_counts.loc[species_counts >= definition.minimum_strains_per_species].index.astype(str)
    )
    eligible = eligible.loc[eligible["species"].isin(included_species)].copy()
    eligible["eligible_strains_in_species"] = eligible["species"].map(species_counts).astype(int)
    eligible["q_per_rotation"] = (eligible["eligible_strains_in_species"] // 5).astype(int)
    eligible = eligible.sort_values(["species", "strain_id"], kind="mergesort").reset_index(drop=True)

    cohort = work.merge(
        eligible[["species", "strain_id", "available_spectra"]],
        on=["species", "strain_id"],
        how="inner",
        validate="many_to_one",
    )
    cohort["spectrum_hash"] = [
        _stable_token("spectrum", definition.name, strain, spectrum)
        for strain, spectrum in zip(cohort["strain_id"], cohort["spectrum_id"])
    ]
    cohort = cohort.sort_values(
        ["species", "strain_id", "spectrum_hash", "spectrum_id"], kind="mergesort"
    )
    cohort["selected_rank"] = cohort.groupby("strain_id", sort=False).cumcount()
    cohort = cohort.loc[cohort["selected_rank"] < definition.spectrum_cap].copy()
    selected_counts = cohort.groupby("strain_id")["spectrum_id"].size()
    if not (selected_counts == definition.spectrum_cap).all():
        raise AssertionError("cohort cap was not satisfied for every eligible strain")
    cohort = cohort.sort_values("feature_row", kind="mergesort").reset_index(drop=True)
    return cohort, eligible


def build_assignments(
    cohort: pd.DataFrame,
    inventory: pd.DataFrame,
    definition: ExperimentDefinition,
    seed: int,
    rotation: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign whole strains to baseline train, evaluation, or donor roles.

    Within each species, five equally sized blocks of q=floor(n/5) strains are
    available for evaluation.  Rotation r evaluates block r and quarantines
    block r+1 as its paired donor block.  Remaining strains train the baseline.
    """

    if rotation not in range(5):
        raise ValueError("rotation must be one of 0, 1, 2, 3, 4")
    strain_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for species, part in inventory.groupby("species", sort=True):
        ordered = part.copy()
        ordered["strain_order_hash"] = [
            _stable_token("strain", definition.name, int(seed), species, strain)
            for strain in ordered["strain_id"].astype(str)
        ]
        ordered = ordered.sort_values(
            ["strain_order_hash", "strain_id"], kind="mergesort"
        ).reset_index(drop=True)
        n_strains = len(ordered)
        q = n_strains // 5
        if q < 1:
            raise AssertionError(f"species below five eligible strains: {species}")
        ordered["strain_rank"] = np.arange(n_strains, dtype=int)
        ordered["rotation_block"] = np.where(
            ordered["strain_rank"] < 5 * q,
            ordered["strain_rank"] // q,
            -1,
        )
        eval_block = rotation
        donor_block = (rotation + 1) % 5
        eval_rows = ordered.loc[ordered["rotation_block"] == eval_block].copy()
        donor_rows = ordered.loc[ordered["rotation_block"] == donor_block].copy()
        if len(eval_rows) != q or len(donor_rows) != q:
            raise AssertionError("rotation blocks are not q-sized")

        eval_ids = eval_rows["strain_id"].astype(str).tolist()
        donor_ids = donor_rows["strain_id"].astype(str).tolist()
        pair_map: dict[str, tuple[str, str]] = {}
        for pair_rank, (eval_id, donor_id) in enumerate(zip(eval_ids, donor_ids)):
            pair_id = _stable_token(
                "pair", definition.name, int(seed), int(rotation), species, eval_id, donor_id
            )[:20]
            pair_map[eval_id] = (pair_id, donor_id)
            pair_map[donor_id] = (pair_id, eval_id)
            pair_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "experiment": definition.name,
                    "seed": int(seed),
                    "rotation": int(rotation),
                    "species": str(species),
                    "pair_rank_within_species": int(pair_rank),
                    "pair_id": pair_id,
                    "evaluation_strain_id": eval_id,
                    "donor_strain_id": donor_id,
                    "exposure_hash": _stable_token(
                        "exposure", definition.name, int(seed), int(rotation), eval_id
                    ),
                }
            )

        eval_set, donor_set = set(eval_ids), set(donor_ids)
        for row in ordered.itertuples(index=False):
            strain_id = str(row.strain_id)
            if strain_id in eval_set:
                role = "evaluation"
                pair_id, paired_id = pair_map[strain_id]
            elif strain_id in donor_set:
                role = "donor"
                pair_id, paired_id = pair_map[strain_id]
            else:
                role = "baseline_train"
                pair_id, paired_id = None, None
            strain_rows.append(
                {
                    "experiment": definition.name,
                    "seed": int(seed),
                    "rotation": int(rotation),
                    "species": str(species),
                    "strain_id": strain_id,
                    "available_spectra": int(row.available_spectra),
                    "eligible_strains_in_species": int(n_strains),
                    "q_per_rotation": int(q),
                    "strain_order_hash": str(row.strain_order_hash),
                    "strain_rank": int(row.strain_rank),
                    "rotation_block": int(row.rotation_block),
                    "is_rotation_remainder": bool(row.rotation_block == -1),
                    "strain_role": role,
                    "pair_id": pair_id,
                    "paired_strain_id": paired_id,
                }
            )

    strain_frame = pd.DataFrame(strain_rows)
    pairs = pd.DataFrame(pair_rows).sort_values(
        ["species", "pair_rank_within_species"], kind="mergesort"
    ).reset_index(drop=True)
    assignments = cohort.merge(
        strain_frame,
        on=["species", "strain_id"],
        how="inner",
        validate="many_to_one",
    )

    def spectrum_role(row: Any) -> str:
        rank = int(row.selected_rank)
        if row.strain_role == "baseline_train":
            return "baseline_train"
        if row.strain_role == "evaluation":
            return "evaluation_query" if rank == 0 else "same_strain_candidate"
        if row.strain_role == "donor":
            return "donor_candidate" if rank < definition.spectrum_cap - 1 else "donor_unused"
        raise AssertionError(f"unknown strain role: {row.strain_role}")

    assignments["spectrum_role"] = [spectrum_role(row) for row in assignments.itertuples()]
    assignments.insert(0, "schema_version", SCHEMA_VERSION)
    assignments = assignments.sort_values("feature_row", kind="mergesort").reset_index(drop=True)
    validate_assignments(assignments, pairs, definition)
    return assignments, pairs


def validate_assignments(
    assignments: pd.DataFrame,
    pairs: pd.DataFrame,
    definition: ExperimentDefinition,
) -> None:
    if assignments["spectrum_id"].duplicated().any():
        raise AssertionError("selected spectra are not unique")
    strain_roles = assignments[["strain_id", "strain_role"]].drop_duplicates()
    if strain_roles["strain_id"].duplicated().any():
        raise AssertionError("a strain was assigned to multiple roles")
    role_sets = {
        role: set(strain_roles.loc[strain_roles["strain_role"] == role, "strain_id"].astype(str))
        for role in ("baseline_train", "evaluation", "donor")
    }
    if role_sets["baseline_train"] & role_sets["evaluation"]:
        raise AssertionError("evaluation strain entered baseline training")
    if role_sets["baseline_train"] & role_sets["donor"]:
        raise AssertionError("donor strain entered baseline training")
    if role_sets["evaluation"] & role_sets["donor"]:
        raise AssertionError("evaluation and donor strains overlap")
    if set(pairs["evaluation_strain_id"].astype(str)) != role_sets["evaluation"]:
        raise AssertionError("evaluation pair mapping is incomplete")
    if set(pairs["donor_strain_id"].astype(str)) != role_sets["donor"]:
        raise AssertionError("donor pair mapping is incomplete")
    if not (pairs["evaluation_strain_id"].astype(str) != pairs["donor_strain_id"].astype(str)).all():
        raise AssertionError("a technical replicate control used the evaluation strain")
    species_lookup = assignments.drop_duplicates("strain_id").set_index("strain_id")["species"]
    if not all(
        species_lookup.loc[evaluation] == species_lookup.loc[donor]
        for evaluation, donor in zip(pairs["evaluation_strain_id"], pairs["donor_strain_id"])
    ):
        raise AssertionError("donor is not from the evaluation species")
    selected_per_strain = assignments.groupby("strain_id")["spectrum_id"].size()
    if not (selected_per_strain == definition.spectrum_cap).all():
        raise AssertionError("spectrum cap differs by strain")
    query_counts = assignments.loc[
        assignments["spectrum_role"] == "evaluation_query"
    ].groupby("strain_id").size()
    if not (query_counts == 1).all():
        raise AssertionError("each evaluation strain must contribute exactly one query")
    leak_counts = assignments.loc[
        assignments["spectrum_role"] == "same_strain_candidate"
    ].groupby("strain_id").size()
    donor_counts = assignments.loc[
        assignments["spectrum_role"] == "donor_candidate"
    ].groupby("strain_id").size()
    expected = definition.spectrum_cap - 1
    if not (leak_counts == expected).all() or not (donor_counts == expected).all():
        raise AssertionError("candidate replicate count does not match the experiment cap")


def _prevalence_exposed_pairs(pairs: pd.DataFrame, level: float) -> pd.DataFrame:
    level = float(level)
    if level < 0 or level > 1:
        raise ValueError("prevalence must be in [0, 1]")
    ordered = pairs.sort_values(["exposure_hash", "pair_id"], kind="mergesort").reset_index(drop=True)
    exact = level * len(ordered)
    count = int(round(exact))
    if not math.isclose(exact, count, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError(
            f"prevalence {level} is not exactly representable for {len(ordered)} evaluation strains"
        )
    return ordered.iloc[:count].copy()


def _candidate_rows(
    assignments: pd.DataFrame,
    exposed_pairs: pd.DataFrame,
    arm: str,
    count_per_pair: int,
) -> pd.DataFrame:
    if arm == "leakage":
        strain_column = "evaluation_strain_id"
        role = "same_strain_candidate"
        rank_offset = 1
    elif arm == "matched_control":
        strain_column = "donor_strain_id"
        role = "donor_candidate"
        rank_offset = 0
    else:
        raise ValueError(f"unknown paired arm: {arm}")
    if count_per_pair < 0:
        raise ValueError("count_per_pair cannot be negative")
    if count_per_pair == 0 or exposed_pairs.empty:
        return assignments.iloc[0:0].copy().assign(pair_order=pd.Series(dtype=int))
    selected = exposed_pairs.reset_index(drop=True).copy()
    selected["pair_order"] = np.arange(len(selected), dtype=int)
    mapping = selected[["pair_id", "pair_order", strain_column]].rename(
        columns={strain_column: "strain_id"}
    )
    candidates = assignments.loc[assignments["spectrum_role"] == role].merge(
        mapping,
        on=["pair_id", "strain_id"],
        how="inner",
        validate="many_to_one",
    )
    candidates["candidate_rank"] = candidates["selected_rank"].astype(int) - rank_offset
    candidates = candidates.loc[candidates["candidate_rank"] < count_per_pair].copy()
    candidates = candidates.sort_values(
        ["pair_order", "candidate_rank", "spectrum_id"], kind="mergesort"
    ).reset_index(drop=True)
    expected = len(exposed_pairs) * count_per_pair
    if len(candidates) != expected:
        raise AssertionError(f"expected {expected} injected spectra, selected {len(candidates)}")
    return candidates


def build_condition_pair(
    assignments: pd.DataFrame,
    pairs: pd.DataFrame,
    definition: ExperimentDefinition,
    level: float | int,
) -> dict[str, Any]:
    """Build paired leakage/control injections for one prevalence or intensity."""

    if definition.name == "prevalence":
        exposed = _prevalence_exposed_pairs(pairs, float(level))
        count_per_pair = 1 if float(level) > 0 else 0
    elif definition.name == "intensity":
        count_per_pair = int(level)
        if float(level) != count_per_pair or count_per_pair not in INTENSITY_LEVELS:
            raise ValueError(f"invalid intensity: {level}")
        exposed = pairs.sort_values(["exposure_hash", "pair_id"], kind="mergesort").reset_index(drop=True)
        if count_per_pair == 0:
            exposed = exposed.iloc[0:0].copy()
    else:
        raise KeyError(definition.name)

    leakage = _candidate_rows(assignments, exposed, "leakage", count_per_pair)
    control = _candidate_rows(assignments, exposed, "matched_control", count_per_pair)
    if len(leakage) != len(control):
        raise AssertionError("paired arms add a different number of spectra")
    leak_composition = leakage.groupby("species").size().sort_index()
    control_composition = control.groupby("species").size().sort_index()
    if not leak_composition.equals(control_composition):
        raise AssertionError("paired arms do not have identical species composition")
    evaluation_spectra = set(
        assignments.loc[assignments["spectrum_role"] == "evaluation_query", "spectrum_id"].astype(str)
    )
    if evaluation_spectra.intersection(leakage["spectrum_id"].astype(str)):
        raise AssertionError("an evaluation query was injected in the leakage arm")
    if evaluation_spectra.intersection(control["spectrum_id"].astype(str)):
        raise AssertionError("an evaluation query was injected in the control arm")
    donor_strains = set(
        assignments.loc[assignments["strain_role"] == "donor", "strain_id"].astype(str)
    )
    baseline_strains = set(
        assignments.loc[assignments["strain_role"] == "baseline_train", "strain_id"].astype(str)
    )
    evaluation_strains = set(
        assignments.loc[assignments["strain_role"] == "evaluation", "strain_id"].astype(str)
    )
    if donor_strains & (baseline_strains | evaluation_strains):
        raise AssertionError("donor strain is not quarantined")
    return {
        "level": level,
        "exposed_pairs": exposed,
        "leakage": leakage,
        "matched_control": control,
        "n_evaluation_strains": int(len(pairs)),
        "n_exposed_strains": int(len(exposed)),
        "n_injected_spectra_per_arm": int(len(leakage)),
    }


def compose_training_rows(assignments: pd.DataFrame, injected: pd.DataFrame) -> np.ndarray:
    """Append injections after the unchanged, original-order baseline rows."""

    baseline = assignments.loc[
        assignments["spectrum_role"] == "baseline_train"
    ].sort_values("feature_row", kind="mergesort")
    if baseline["feature_row"].duplicated().any():
        raise AssertionError("baseline feature rows are duplicated")
    injected_ordered = injected.sort_values(
        ["pair_order", "candidate_rank", "spectrum_id"], kind="mergesort"
    ) if len(injected) else injected
    rows = np.concatenate(
        [
            baseline["feature_row"].to_numpy(dtype=np.int64),
            injected_ordered["feature_row"].to_numpy(dtype=np.int64),
        ]
    )
    if len(np.unique(rows)) != len(rows):
        raise AssertionError("a spectrum appears twice in the composed training data")
    return rows


def _level_token(experiment: str, level: float | int) -> str:
    if experiment == "prevalence":
        return f"p-{float(level):.2f}".replace(".", "p")
    return f"r-{int(level)}"


def _condition_arms(level: float | int) -> tuple[str, ...]:
    return ("baseline",) if float(level) == 0 else ("leakage", "matched_control")


def expected_fit_count(definition: ExperimentDefinition) -> int:
    return sum(len(_condition_arms(level)) for level in definition.levels)


def _assignment_signature(
    definition: ExperimentDefinition,
    seed: int,
    rotation: int,
    manifest_sha256: str,
    source_sha256: str,
) -> str:
    return canonical_sha256(
        {
            "schema": SCHEMA_VERSION,
            "definition": asdict(definition),
            "seed": int(seed),
            "rotation": int(rotation),
            "manifest_sha256": manifest_sha256,
            "source_sha256": source_sha256,
        }
    )


def _fit_signature(
    assignment_signature: str,
    feature_sha256: str,
    experiment: str,
    seed: int,
    rotation: int,
    level: float | int,
    arm: str,
    n_estimators: int,
) -> str:
    return canonical_sha256(
        {
            "schema": SCHEMA_VERSION,
            "assignment_signature": assignment_signature,
            "feature_sha256": feature_sha256,
            "experiment": experiment,
            "seed": int(seed),
            "rotation": int(rotation),
            "level": level,
            "arm": arm,
            "model": "ExtraTreesClassifier",
            "n_estimators": int(n_estimators),
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "weighting": "equal class mass; equal strain mass within class",
        }
    )


def build_expected_checkpoint_universe(
    output_root: str | Path,
    experiments: Sequence[str],
    seeds: Sequence[int],
    rotations: Sequence[int],
    manifest_sha256: str,
    feature_sha256: str,
    n_estimators: int,
    fit_engine_source_sha256: str = FROZEN_FIT_ENGINE_SOURCE_SHA256,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Precompute every permitted assignment and fit checkpoint path/signature."""

    root = Path(output_root).resolve()
    expected_assignments: dict[str, dict[str, Any]] = {}
    expected_fits: dict[str, dict[str, Any]] = {}
    for experiment in experiments:
        definition = EXPERIMENTS[experiment]
        for seed in seeds:
            for rotation in rotations:
                assignment_dir, cell_dir, _ = _cell_paths(
                    root, experiment, int(seed), int(rotation)
                )
                assignment_signature = _assignment_signature(
                    definition,
                    int(seed),
                    int(rotation),
                    manifest_sha256,
                    fit_engine_source_sha256,
                )
                assignment_manifest = assignment_dir / "assignment_manifest.json"
                assignment_relative = assignment_manifest.relative_to(root).as_posix()
                expected_assignments[assignment_relative] = {
                    "schema_version": SCHEMA_VERSION,
                    "experiment": experiment,
                    "seed": int(seed),
                    "rotation": int(rotation),
                    "assignment_signature": assignment_signature,
                    "fit_engine_source_sha256": fit_engine_source_sha256,
                    "manifest_sha256": manifest_sha256,
                    "assignments_path": (assignment_dir / "assignments.parquet").relative_to(root).as_posix(),
                    "pairings_path": (assignment_dir / "pairings.parquet").relative_to(root).as_posix(),
                }
                for level in definition.levels:
                    for arm in _condition_arms(level):
                        fit_dir = cell_dir / _level_token(experiment, level) / arm
                        fit_manifest = fit_dir / "fit_manifest.json"
                        fit_relative = fit_manifest.relative_to(root).as_posix()
                        expected_fits[fit_relative] = {
                            "schema_version": SCHEMA_VERSION,
                            "experiment": experiment,
                            "seed": int(seed),
                            "rotation": int(rotation),
                            "level": float(level),
                            "level_token": _level_token(experiment, level),
                            "arm": arm,
                            "model": "ExtraTrees",
                            "n_estimators": int(n_estimators),
                            "assignment_signature": assignment_signature,
                            "fit_signature": _fit_signature(
                                assignment_signature,
                                feature_sha256,
                                experiment,
                                int(seed),
                                int(rotation),
                                level,
                                arm,
                                int(n_estimators),
                            ),
                            "fit_engine_source_sha256": fit_engine_source_sha256,
                            "feature_sha256": feature_sha256,
                            "predictions_path": (fit_dir / "predictions.parquet").relative_to(root).as_posix(),
                            "injections_path": (fit_dir / "injections.parquet").relative_to(root).as_posix(),
                        }
    return {"assignments": expected_assignments, "fits": expected_fits}


def _record_matches(record: Mapping[str, Any], expected: Mapping[str, Any], fields: Sequence[str]) -> None:
    for field in fields:
        observed = record.get(field)
        target = expected[field]
        if field == "level":
            equal = observed is not None and float(observed) == float(target)
        else:
            equal = observed == target
        if not equal:
            raise RuntimeError(
                f"checkpoint field mismatch for {field}: observed={observed!r}, expected={target!r}"
            )


def validate_checkpoint_universe(
    output_root: str | Path,
    universe: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    require_complete: bool,
    upgrade_missing_provenance: bool = False,
) -> dict[str, Any]:
    """Validate an exact, closed checkpoint universe; never silently skip stale cells.

    Partial states may contain a strict subset of the expected paths.  Any extra
    path, wrong signature, wrong source/feature/model identity, incomplete
    artifact trio, or artifact hash corruption is fatal.  With
    ``require_complete=True``, observed paths must equal expected paths.
    """

    root = Path(output_root).resolve()
    expected_assignments = dict(universe["assignments"])
    expected_fits = dict(universe["fits"])

    assignment_names = {"assignment_manifest.json", "assignments.parquet", "pairings.parquet"}
    observed_assignment_dirs = {
        path.parent.relative_to(root).as_posix()
        for name in assignment_names
        for path in (root / "assignments").glob(f"*/{name}")
    } if (root / "assignments").exists() else set()
    expected_assignment_dirs = {
        str(Path(relative).parent).replace("\\", "/")
        for relative in expected_assignments
    }
    extra_assignment_dirs = sorted(observed_assignment_dirs - expected_assignment_dirs)
    if extra_assignment_dirs:
        raise RuntimeError(f"unexpected assignment checkpoint directories: {extra_assignment_dirs}")

    fit_names = {"fit_manifest.json", "predictions.parquet", "injections.parquet"}
    observed_fit_dirs = {
        path.parent.relative_to(root).as_posix()
        for name in fit_names
        for path in (root / "cells").glob(f"**/{name}")
    } if (root / "cells").exists() else set()
    expected_fit_dirs = {
        str(Path(relative).parent).replace("\\", "/")
        for relative in expected_fits
    }
    extra_fit_dirs = sorted(observed_fit_dirs - expected_fit_dirs)
    if extra_fit_dirs:
        raise RuntimeError(f"unexpected fit checkpoint directories: {extra_fit_dirs}")

    validated_assignment_paths: list[str] = []
    for directory in sorted(observed_assignment_dirs):
        files = {name: root / directory / name for name in assignment_names}
        missing_artifacts = [name for name, path in files.items() if not path.exists()]
        if missing_artifacts:
            raise RuntimeError(
                f"incomplete assignment checkpoint {directory}: missing {missing_artifacts}"
            )
        relative = (Path(directory) / "assignment_manifest.json").as_posix()
        expected = expected_assignments[relative]
        record = json.loads(files["assignment_manifest.json"].read_text(encoding="utf-8"))
        _record_matches(
            record,
            expected,
            ("schema_version", "experiment", "seed", "rotation", "assignment_signature"),
        )
        assignments_sha = sha256_file(files["assignments.parquet"])
        pairings_sha = sha256_file(files["pairings.parquet"])
        if record.get("assignments_sha256") != assignments_sha:
            raise RuntimeError(f"assignment artifact hash corruption: {directory}/assignments.parquet")
        if record.get("pairings_sha256") != pairings_sha:
            raise RuntimeError(f"assignment artifact hash corruption: {directory}/pairings.parquet")
        provenance = {
            "fit_engine_source_sha256": expected["fit_engine_source_sha256"],
            "manifest_sha256": expected["manifest_sha256"],
        }
        for field, target in provenance.items():
            if field in record and record[field] != target:
                raise RuntimeError(f"assignment provenance mismatch for {directory}/{field}")
        if upgrade_missing_provenance and any(field not in record for field in provenance):
            record.update(provenance)
            _atomic_json(files["assignment_manifest.json"], record)
        elif any(field not in record for field in provenance):
            raise RuntimeError(f"assignment provenance missing in {directory}")
        validated_assignment_paths.append(relative)

    validated_fit_paths: list[str] = []
    for directory in sorted(observed_fit_dirs):
        files = {name: root / directory / name for name in fit_names}
        missing_artifacts = [name for name, path in files.items() if not path.exists()]
        if missing_artifacts:
            raise RuntimeError(f"incomplete fit checkpoint {directory}: missing {missing_artifacts}")
        relative = (Path(directory) / "fit_manifest.json").as_posix()
        expected = expected_fits[relative]
        record = json.loads(files["fit_manifest.json"].read_text(encoding="utf-8"))
        _record_matches(
            record,
            expected,
            (
                "schema_version",
                "experiment",
                "seed",
                "rotation",
                "level",
                "level_token",
                "arm",
                "model",
                "n_estimators",
                "assignment_signature",
                "fit_signature",
                "feature_sha256",
            ),
        )
        predictions_sha = sha256_file(files["predictions.parquet"])
        injections_sha = sha256_file(files["injections.parquet"])
        if record.get("predictions_sha256") != predictions_sha:
            raise RuntimeError(f"fit artifact hash corruption: {directory}/predictions.parquet")
        if record.get("injections_sha256") != injections_sha:
            raise RuntimeError(f"fit artifact hash corruption: {directory}/injections.parquet")
        source_target = expected["fit_engine_source_sha256"]
        if "fit_engine_source_sha256" in record and record["fit_engine_source_sha256"] != source_target:
            raise RuntimeError(f"fit source mismatch in {directory}")
        if upgrade_missing_provenance and "fit_engine_source_sha256" not in record:
            record["fit_engine_source_sha256"] = source_target
            _atomic_json(files["fit_manifest.json"], record)
        elif "fit_engine_source_sha256" not in record:
            raise RuntimeError(f"fit source provenance missing in {directory}")
        validated_fit_paths.append(relative)

    expected_assignment_paths = set(expected_assignments)
    expected_fit_paths = set(expected_fits)
    observed_assignment_paths = set(validated_assignment_paths)
    observed_fit_paths = set(validated_fit_paths)
    missing_assignments = sorted(expected_assignment_paths - observed_assignment_paths)
    missing_fits = sorted(expected_fit_paths - observed_fit_paths)
    if require_complete and (missing_assignments or missing_fits):
        raise RuntimeError(
            f"checkpoint universe incomplete: {len(missing_assignments)} assignments and "
            f"{len(missing_fits)} fits missing"
        )
    return {
        "expected_assignment_paths": len(expected_assignment_paths),
        "observed_assignment_paths": len(observed_assignment_paths),
        "expected_fit_paths": len(expected_fit_paths),
        "observed_fit_paths": len(observed_fit_paths),
        "missing_assignment_paths": missing_assignments,
        "missing_fit_paths": missing_fits,
        "extra_assignment_paths": [],
        "extra_fit_paths": [],
        "observed_equals_expected": (
            not missing_assignments
            and not missing_fits
            and observed_assignment_paths == expected_assignment_paths
            and observed_fit_paths == expected_fit_paths
        ),
        "strict_signature_and_artifact_hash_validation": True,
    }


def _checkpoint_valid(
    path: Path,
    fit_signature: str,
    *,
    fit_engine_source_sha256: str = FROZEN_FIT_ENGINE_SOURCE_SHA256,
    feature_sha256: str | None = None,
    n_estimators: int | None = None,
) -> bool:
    metadata_path = path / "fit_manifest.json"
    prediction_path = path / "predictions.parquet"
    injection_path = path / "injections.parquet"
    if not metadata_path.exists() or not prediction_path.exists() or not injection_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        valid = (
            metadata.get("fit_signature") == fit_signature
            and metadata.get("predictions_sha256") == sha256_file(prediction_path)
            and metadata.get("injections_sha256") == sha256_file(injection_path)
            and metadata.get("fit_engine_source_sha256") == fit_engine_source_sha256
        )
        if feature_sha256 is not None:
            valid = valid and metadata.get("feature_sha256") == feature_sha256
        if n_estimators is not None:
            valid = valid and int(metadata.get("n_estimators", -1)) == int(n_estimators)
        return valid
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def _fit_condition(
    matrix: np.ndarray,
    manifest: pd.DataFrame,
    assignments: pd.DataFrame,
    pairs: pd.DataFrame,
    definition: ExperimentDefinition,
    level: float | int,
    arm: str,
    seed: int,
    rotation: int,
    output_dir: Path,
    assignment_signature: str,
    feature_sha256: str,
    fit_engine_source_sha256: str,
    n_estimators: int,
    n_jobs: int,
    force: bool,
) -> dict[str, Any]:
    fit_signature = _fit_signature(
        assignment_signature,
        feature_sha256,
        definition.name,
        seed,
        rotation,
        level,
        arm,
        n_estimators,
    )
    if not force and _checkpoint_valid(
        output_dir,
        fit_signature,
        fit_engine_source_sha256=fit_engine_source_sha256,
        feature_sha256=feature_sha256,
        n_estimators=n_estimators,
    ):
        return json.loads((output_dir / "fit_manifest.json").read_text(encoding="utf-8"))

    condition = build_condition_pair(assignments, pairs, definition, level)
    if arm == "baseline":
        injected = assignments.iloc[0:0].copy().assign(
            pair_order=pd.Series(dtype=int), candidate_rank=pd.Series(dtype=int)
        )
    else:
        injected = condition[arm]
    train_rows = compose_training_rows(assignments, injected)
    evaluation = assignments.loc[
        assignments["spectrum_role"] == "evaluation_query"
    ].sort_values("feature_row", kind="mergesort").copy()
    evaluation_rows = evaluation["feature_row"].to_numpy(dtype=np.int64)
    if set(train_rows).intersection(set(evaluation_rows)):
        raise AssertionError("evaluation rows entered model fitting")

    train_meta = manifest.iloc[train_rows]
    labels = train_meta["species"].astype(str).to_numpy()
    strain_ids = train_meta["strain_id"].astype(str).to_numpy()
    weights = strain_balanced_weights(labels, strain_ids)
    # Keep the forest RNG fixed across every level and both arms within a cell.
    # This removes avoidable Monte-Carlo variation from dose/intensity contrasts.
    fit_seed = _stable_seed("fit", definition.name, int(seed), int(rotation))
    estimator = ExtraTreesClassifier(
        n_estimators=int(n_estimators),
        max_features="sqrt",
        min_samples_leaf=1,
        class_weight=None,
        random_state=fit_seed,
        n_jobs=int(n_jobs),
    )
    started = time.perf_counter()
    estimator.fit(matrix[train_rows], labels, sample_weight=weights)
    probabilities = np.asarray(estimator.predict_proba(matrix[evaluation_rows]), dtype=np.float32)
    predicted = estimator.classes_[np.argmax(probabilities, axis=1)].astype(str)
    elapsed = time.perf_counter() - started
    truth = evaluation["species"].astype(str).to_numpy()
    confidence = probabilities.max(axis=1)

    predictions = pd.DataFrame(
        {
            "schema_version": SCHEMA_VERSION,
            "experiment": definition.name,
            "seed": int(seed),
            "rotation": int(rotation),
            "level": float(level),
            "level_token": _level_token(definition.name, level),
            "arm": arm,
            "pair_id": evaluation["pair_id"].astype(str).to_numpy(),
            "spectrum_id": evaluation["spectrum_id"].astype(str).to_numpy(),
            "strain_id": evaluation["strain_id"].astype(str).to_numpy(),
            "species": truth,
            "predicted_species": predicted,
            "confidence": confidence,
            "correct": predicted == truth,
            "feature_row": evaluation_rows,
            "evaluation_exposed": evaluation["pair_id"].astype(str).isin(
                set(condition["exposed_pairs"]["pair_id"].astype(str))
            ).to_numpy(),
            "n_exposed_eval_strains": int(condition["n_exposed_strains"]),
            "n_injected_spectra": int(len(injected)),
        }
    )
    if len(injected):
        pair_lookup = pairs.set_index("pair_id")
        injections = pd.DataFrame(
            {
                "schema_version": SCHEMA_VERSION,
                "experiment": definition.name,
                "seed": int(seed),
                "rotation": int(rotation),
                "level": float(level),
                "level_token": _level_token(definition.name, level),
                "arm": arm,
                "pair_id": injected["pair_id"].astype(str).to_numpy(),
                "evaluation_strain_id": [
                    str(pair_lookup.loc[pair_id, "evaluation_strain_id"])
                    for pair_id in injected["pair_id"].astype(str)
                ],
                "source_strain_id": injected["strain_id"].astype(str).to_numpy(),
                "source_spectrum_id": injected["spectrum_id"].astype(str).to_numpy(),
                "source_feature_row": injected["feature_row"].astype(np.int64).to_numpy(),
                "species": injected["species"].astype(str).to_numpy(),
                "candidate_rank": injected["candidate_rank"].astype(int).to_numpy(),
                "pair_order": injected["pair_order"].astype(int).to_numpy(),
            }
        )
    else:
        injections = pd.DataFrame(
            {
                "schema_version": pd.Series(dtype=str),
                "experiment": pd.Series(dtype=str),
                "seed": pd.Series(dtype="int64"),
                "rotation": pd.Series(dtype="int64"),
                "level": pd.Series(dtype=float),
                "level_token": pd.Series(dtype=str),
                "arm": pd.Series(dtype=str),
                "pair_id": pd.Series(dtype=str),
                "evaluation_strain_id": pd.Series(dtype=str),
                "source_strain_id": pd.Series(dtype=str),
                "source_spectrum_id": pd.Series(dtype=str),
                "source_feature_row": pd.Series(dtype="int64"),
                "species": pd.Series(dtype=str),
                "candidate_rank": pd.Series(dtype="int64"),
                "pair_order": pd.Series(dtype="int64"),
            }
        )
    macro_f1 = float(f1_score(truth, predicted, average="macro", zero_division=0))
    accuracy = float(accuracy_score(truth, predicted))
    balanced_accuracy = float(balanced_accuracy_score(truth, predicted))
    baseline_rows = assignments.loc[assignments["spectrum_role"] == "baseline_train"]
    metric = {
        "schema_version": SCHEMA_VERSION,
        "experiment": definition.name,
        "seed": int(seed),
        "rotation": int(rotation),
        "level": float(level),
        "level_token": _level_token(definition.name, level),
        "arm": arm,
        "model": "ExtraTrees",
        "n_estimators": int(n_estimators),
        "n_jobs": int(n_jobs),
        "fit_random_state": int(fit_seed),
        "n_species": int(evaluation["species"].nunique()),
        "n_baseline_training_strains": int(baseline_rows["strain_id"].nunique()),
        "n_baseline_training_spectra": int(len(baseline_rows)),
        "n_evaluation_strains": int(len(evaluation)),
        "n_evaluation_spectra": int(len(evaluation)),
        "n_exposed_eval_strains": int(condition["n_exposed_strains"]),
        "exposure_fraction": float(condition["n_exposed_strains"] / len(evaluation)),
        "n_injected_spectra": int(len(injected)),
        "n_training_spectra": int(len(train_rows)),
        "n_training_strains": int(train_meta["strain_id"].nunique()),
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "accuracy": accuracy,
        "fit_seconds": float(elapsed),
        "estimand_unit": "one fixed-hash query spectrum per evaluation strain",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.parquet"
    injection_path = output_dir / "injections.parquet"
    _atomic_parquet(predictions, prediction_path)
    _atomic_parquet(injections, injection_path)
    fit_manifest = {
        **metric,
        "created_at": utc_now(),
        "fit_signature": fit_signature,
        "assignment_signature": assignment_signature,
        "feature_sha256": feature_sha256,
        "fit_engine_source_sha256": fit_engine_source_sha256,
        "predictions_path": prediction_path.as_posix(),
        "predictions_sha256": sha256_file(prediction_path),
        "injections_path": injection_path.as_posix(),
        "injections_sha256": sha256_file(injection_path),
    }
    _atomic_json(output_dir / "fit_manifest.json", fit_manifest)
    return fit_manifest


def _cell_paths(output_root: Path, experiment: str, seed: int, rotation: int) -> tuple[Path, Path, Path]:
    token = f"{experiment}__seed-{int(seed)}__rotation-{int(rotation)}"
    assignment_dir = output_root / "assignments" / token
    cell_dir = output_root / "cells" / experiment / f"seed-{int(seed)}" / f"rotation-{int(rotation)}"
    return assignment_dir, cell_dir, output_root / "cells" / token


def _process_cell(
    project_root: Path,
    output_root: Path,
    matrix: np.ndarray,
    manifest: pd.DataFrame,
    cohort: pd.DataFrame,
    inventory: pd.DataFrame,
    definition: ExperimentDefinition,
    seed: int,
    rotation: int,
    feature_sha256: str,
    manifest_sha256: str,
    fit_engine_source_sha256: str,
    n_estimators: int,
    n_jobs: int,
    force: bool,
) -> dict[str, Any]:
    del project_root  # included in the signature at the run level; no cell-local path dependence
    assignment_dir, cell_dir, _ = _cell_paths(output_root, definition.name, seed, rotation)
    assignment_signature = _assignment_signature(
        definition, seed, rotation, manifest_sha256, fit_engine_source_sha256
    )
    assignment_path = assignment_dir / "assignments.parquet"
    pair_path = assignment_dir / "pairings.parquet"
    assignment_manifest_path = assignment_dir / "assignment_manifest.json"
    use_saved = False
    if not force and assignment_path.exists() and pair_path.exists() and assignment_manifest_path.exists():
        saved = json.loads(assignment_manifest_path.read_text(encoding="utf-8"))
        use_saved = (
            saved.get("assignment_signature") == assignment_signature
            and saved.get("assignments_sha256") == sha256_file(assignment_path)
            and saved.get("pairings_sha256") == sha256_file(pair_path)
            and saved.get("fit_engine_source_sha256") == fit_engine_source_sha256
            and saved.get("manifest_sha256") == manifest_sha256
        )
        if not use_saved:
            raise RuntimeError(
                f"stale assignment checkpoint at {assignment_dir}; rerun with --force"
            )
    if use_saved:
        assignments = pd.read_parquet(assignment_path)
        pairs = pd.read_parquet(pair_path)
        validate_assignments(assignments, pairs, definition)
    else:
        assignments, pairs = build_assignments(
            cohort, inventory, definition, seed=int(seed), rotation=int(rotation)
        )
        _atomic_parquet(assignments, assignment_path)
        _atomic_parquet(pairs, pair_path)
        _atomic_json(
            assignment_manifest_path,
            {
                "schema_version": SCHEMA_VERSION,
                "created_at": utc_now(),
                "experiment": definition.name,
                "seed": int(seed),
                "rotation": int(rotation),
                "assignment_signature": assignment_signature,
                "fit_engine_source_sha256": fit_engine_source_sha256,
                "manifest_sha256": manifest_sha256,
                "assignments_sha256": sha256_file(assignment_path),
                "pairings_sha256": sha256_file(pair_path),
                "n_species": int(assignments["species"].nunique()),
                "n_eligible_strains": int(assignments["strain_id"].nunique()),
                "n_evaluation_strains": int(len(pairs)),
                "n_donor_strains": int(len(pairs)),
                "n_baseline_training_strains": int(
                    assignments.loc[
                        assignments["strain_role"] == "baseline_train", "strain_id"
                    ].nunique()
                ),
            },
        )

    fits: list[dict[str, Any]] = []
    for level in definition.levels:
        for arm in _condition_arms(level):
            condition_dir = cell_dir / _level_token(definition.name, level) / arm
            fits.append(
                _fit_condition(
                    matrix=matrix,
                    manifest=manifest,
                    assignments=assignments,
                    pairs=pairs,
                    definition=definition,
                    level=level,
                    arm=arm,
                    seed=int(seed),
                    rotation=int(rotation),
                    output_dir=condition_dir,
                    assignment_signature=assignment_signature,
                    feature_sha256=feature_sha256,
                    fit_engine_source_sha256=fit_engine_source_sha256,
                    n_estimators=n_estimators,
                    n_jobs=n_jobs,
                    force=force,
                )
            )
    return {
        "experiment": definition.name,
        "seed": int(seed),
        "rotation": int(rotation),
        "fit_count": len(fits),
        "assignment_signature": assignment_signature,
    }


def _metric_from_confusion(confusion: np.ndarray) -> tuple[float, float]:
    confusion = np.asarray(confusion, dtype=np.float64)
    true_total = confusion.sum(axis=1)
    predicted_total = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    denominator = true_total + predicted_total
    f1 = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    total = confusion.sum()
    accuracy = float(true_positive.sum() / total) if total else math.nan
    return float(f1.mean()), accuracy


def _paired_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    positive = predictions.loc[predictions["arm"].isin(["leakage", "matched_control"])].copy()
    keys = [
        "experiment",
        "seed",
        "rotation",
        "level",
        "level_token",
        "pair_id",
        "spectrum_id",
        "strain_id",
        "species",
    ]
    leakage = positive.loc[positive["arm"] == "leakage", keys + ["predicted_species"]].rename(
        columns={"predicted_species": "predicted_leakage"}
    )
    control = positive.loc[
        positive["arm"] == "matched_control", keys + ["predicted_species"]
    ].rename(columns={"predicted_species": "predicted_control"})
    return leakage.merge(control, on=keys, how="inner", validate="one_to_one")


def paired_cell_contrasts(predictions: pd.DataFrame) -> pd.DataFrame:
    paired = _paired_predictions(predictions)
    rows: list[dict[str, Any]] = []
    group_columns = ["experiment", "seed", "rotation", "level", "level_token"]
    for key, part in paired.groupby(group_columns, sort=True):
        experiment, seed, rotation, level, level_token = key
        truth = part["species"].astype(str).to_numpy()
        leak = part["predicted_leakage"].astype(str).to_numpy()
        control = part["predicted_control"].astype(str).to_numpy()
        f1_leak = float(f1_score(truth, leak, average="macro", zero_division=0))
        f1_control = float(f1_score(truth, control, average="macro", zero_division=0))
        acc_leak = float(accuracy_score(truth, leak))
        acc_control = float(accuracy_score(truth, control))
        rows.append(
            {
                "experiment": experiment,
                "seed": int(seed),
                "rotation": int(rotation),
                "level": float(level),
                "level_token": level_token,
                "n_paired_evaluation_strains": int(len(part)),
                "macro_f1_leakage": f1_leak,
                "macro_f1_matched_control": f1_control,
                "macro_f1_delta_leakage_minus_control": f1_leak - f1_control,
                "accuracy_leakage": acc_leak,
                "accuracy_matched_control": acc_control,
                "accuracy_delta_leakage_minus_control": acc_leak - acc_control,
            }
        )
    return pd.DataFrame(rows)


def paired_cluster_bootstrap(
    predictions: pd.DataFrame,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Species-stratified, paired strain-cluster bootstrap.

    A physical strain is the cluster.  All of its prediction occurrences across
    seeds and rotations are retained together.  The same resample is applied to
    leakage and donor-control predictions.
    """

    paired = _paired_predictions(predictions)
    replicate_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    for (experiment, level, level_token), part in paired.groupby(
        ["experiment", "level", "level_token"], sort=True
    ):
        labels = sorted(part["species"].astype(str).unique())
        lookup = {label: index for index, label in enumerate(labels)}
        cluster_matrices: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        for (species, strain_id), cluster in part.groupby(["species", "strain_id"], sort=True):
            leak_matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
            control_matrix = np.zeros_like(leak_matrix)
            true_index = lookup[str(species)]
            for predicted in cluster["predicted_leakage"].astype(str):
                leak_matrix[true_index, lookup[predicted]] += 1
            for predicted in cluster["predicted_control"].astype(str):
                control_matrix[true_index, lookup[predicted]] += 1
            cluster_matrices.setdefault(str(species), []).append((leak_matrix, control_matrix))

        point_leak = sum((item[0] for values in cluster_matrices.values() for item in values), np.zeros((len(labels), len(labels)), dtype=np.int64))
        point_control = sum((item[1] for values in cluster_matrices.values() for item in values), np.zeros((len(labels), len(labels)), dtype=np.int64))
        point_f1_leak, point_acc_leak = _metric_from_confusion(point_leak)
        point_f1_control, point_acc_control = _metric_from_confusion(point_control)
        local_rng = np.random.default_rng(
            _stable_seed("bootstrap", int(seed), experiment, str(level))
        )
        for replicate in range(int(replicates)):
            leak_total = np.zeros_like(point_leak)
            control_total = np.zeros_like(point_control)
            for species in labels:
                clusters = cluster_matrices[species]
                sampled = local_rng.integers(0, len(clusters), size=len(clusters))
                for index in sampled:
                    leak_total += clusters[int(index)][0]
                    control_total += clusters[int(index)][1]
            f1_leak, acc_leak = _metric_from_confusion(leak_total)
            f1_control, acc_control = _metric_from_confusion(control_total)
            replicate_rows.append(
                {
                    "experiment": experiment,
                    "level": float(level),
                    "level_token": level_token,
                    "replicate": int(replicate),
                    "macro_f1_leakage": f1_leak,
                    "macro_f1_matched_control": f1_control,
                    "macro_f1_delta_leakage_minus_control": f1_leak - f1_control,
                    "accuracy_leakage": acc_leak,
                    "accuracy_matched_control": acc_control,
                    "accuracy_delta_leakage_minus_control": acc_leak - acc_control,
                }
            )
        replicate_part = pd.DataFrame(replicate_rows).loc[
            lambda x: (x["experiment"] == experiment) & (x["level"] == float(level))
        ]
        for metric, point in (
            ("macro_f1_leakage", point_f1_leak),
            ("macro_f1_matched_control", point_f1_control),
            ("macro_f1_delta_leakage_minus_control", point_f1_leak - point_f1_control),
            ("accuracy_leakage", point_acc_leak),
            ("accuracy_matched_control", point_acc_control),
            ("accuracy_delta_leakage_minus_control", point_acc_leak - point_acc_control),
        ):
            values = replicate_part[metric].to_numpy(dtype=float)
            interval_rows.append(
                {
                    "experiment": experiment,
                    "level": float(level),
                    "level_token": level_token,
                    "metric": metric,
                    "point_estimate": float(point),
                    "ci_lower": float(np.quantile(values, 0.025)),
                    "ci_upper": float(np.quantile(values, 0.975)),
                    "bootstrap_replicates": int(replicates),
                    "bootstrap_method": "species-stratified paired strain-cluster bootstrap",
                    "n_unique_strains": int(part["strain_id"].nunique()),
                    "n_prediction_instances": int(len(part)),
                }
            )
    return pd.DataFrame(replicate_rows), pd.DataFrame(interval_rows)


def aggregate_outputs(
    output_root: Path,
    checkpoint_universe: Mapping[str, Mapping[str, Mapping[str, Any]]],
    aggregation_source_sha256: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    expected_experiments: Sequence[str],
    expected_seeds: Sequence[int],
    expected_rotations: Sequence[int],
    run_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_audit = validate_checkpoint_universe(
        output_root,
        checkpoint_universe,
        require_complete=False,
        upgrade_missing_provenance=False,
    )
    fit_records: list[tuple[Path, dict[str, Any]]] = []
    for relative in sorted(checkpoint_universe["fits"]):
        path = output_root / relative
        if path.exists():
            fit_records.append((path, json.loads(path.read_text(encoding="utf-8"))))
    prediction_frames = [pd.read_parquet(path.parent / "predictions.parquet") for path, _ in fit_records]
    injection_frames = [pd.read_parquet(path.parent / "injections.parquet") for path, _ in fit_records]
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    injections = (
        pd.concat(injection_frames, ignore_index=True)
        if injection_frames and any(len(frame) for frame in injection_frames)
        else pd.DataFrame()
    )
    metrics = pd.DataFrame([record for _, record in fit_records])

    assignment_manifest_paths = [
        output_root / relative
        for relative in sorted(checkpoint_universe["assignments"])
        if (output_root / relative).exists()
    ]
    assignment_paths = [path.parent / "assignments.parquet" for path in assignment_manifest_paths]
    pair_paths = [path.parent / "pairings.parquet" for path in assignment_manifest_paths]
    assignments = (
        pd.concat([pd.read_parquet(path) for path in assignment_paths], ignore_index=True)
        if assignment_paths
        else pd.DataFrame()
    )
    pairings = (
        pd.concat([pd.read_parquet(path) for path in pair_paths], ignore_index=True)
        if pair_paths
        else pd.DataFrame()
    )
    paired = paired_cell_contrasts(predictions) if len(predictions) else pd.DataFrame()
    if len(paired) and bootstrap_replicates > 0:
        bootstrap, intervals = paired_cluster_bootstrap(
            predictions, replicates=int(bootstrap_replicates), seed=int(bootstrap_seed)
        )
    else:
        bootstrap, intervals = pd.DataFrame(), pd.DataFrame()

    output_files: dict[str, Path] = {}
    for name, frame, kind in (
        ("assignments", assignments, "parquet"),
        ("pairings", pairings, "parquet"),
        ("predictions", predictions, "parquet"),
        ("injections", injections, "parquet"),
        ("metrics", metrics, "parquet"),
        ("paired_contrasts", paired, "csv"),
        ("bootstrap_replicates", bootstrap, "parquet"),
        ("bootstrap_intervals", intervals, "csv"),
    ):
        if frame.empty:
            continue
        path = output_root / f"{name}.{kind}"
        if kind == "parquet":
            _atomic_parquet(frame, path)
        else:
            _atomic_csv(frame, path)
        output_files[name] = path
    if not metrics.empty:
        metrics_csv = output_root / "metrics.csv"
        _atomic_csv(metrics, metrics_csv)
        output_files["metrics_csv"] = metrics_csv

    expected_fits = len(checkpoint_universe["fits"])
    completed_fits = int(len(metrics))
    completed_cells = int(
        metrics[["experiment", "seed", "rotation"]].drop_duplicates().shape[0]
    ) if not metrics.empty else 0
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "status": "complete" if checkpoint_audit["observed_equals_expected"] else "partial",
        "methodological_status": "post-result exploratory v2 enhancement",
        "fit_engine_source_sha256": FROZEN_FIT_ENGINE_SOURCE_SHA256,
        "aggregation_source_sha256": aggregation_source_sha256,
        "expected_experiments": list(expected_experiments),
        "expected_seeds": [int(seed) for seed in expected_seeds],
        "expected_rotations": [int(rotation) for rotation in expected_rotations],
        "expected_cells": int(len(checkpoint_universe["assignments"])),
        "completed_cells_with_at_least_one_fit": completed_cells,
        "expected_fits": int(expected_fits),
        "completed_fits": completed_fits,
        "bootstrap_replicates": int(bootstrap_replicates),
        "bootstrap_seed": int(bootstrap_seed),
        "model": {
            "name": "ExtraTreesClassifier",
            "n_estimators": int(run_metadata["n_estimators"]),
            "n_jobs_per_fit": int(run_metadata["n_jobs"]),
            "weighting": "strain-balanced within equal-mass species",
            "endpoint": "closed-set species classification",
        },
        "cohorts": run_metadata["cohorts"],
        "inputs": run_metadata["inputs"],
        "runtime": run_metadata["runtime"],
        "checkpoint_universe_audit": checkpoint_audit,
        "output_files": {
            name: {
                "path": path.relative_to(output_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in output_files.items()
        },
        "invariants": {
            "evaluation_query_never_injected": True,
            "donor_whole_strain_quarantine": True,
            "paired_increment_count_and_species_composition_equal": True,
            "prevalence_exposure_nested_by_fixed_hash": True,
            "intensity_candidates_nested_by_fixed_hash": True,
            "baseline_row_order_preserved_before_appended_injections": True,
        },
    }
    _atomic_json(output_root / "run_manifest.json", manifest)
    return manifest


def plan_contamination(
    project_root: str | Path,
    experiments: Sequence[str] = ("prevalence", "intensity"),
    seeds: Sequence[int] | None = None,
    rotations: Sequence[int] = (0, 1, 2, 3, 4),
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    spec = json.loads((root / "config/v2/analysis_spec.json").read_text(encoding="utf-8"))
    seeds = tuple(int(value) for value in (seeds or spec["seeds"]))
    manifest = pd.read_parquet(
        root / "data/processed/production/manifest.parquet",
        columns=["spectrum_id", "strain_id", "species", "primary_known"],
    )
    cohorts: dict[str, Any] = {}
    for name in experiments:
        definition = EXPERIMENTS[name]
        cohort, inventory = build_cohort(manifest, definition)
        per_species = inventory.groupby("species")["strain_id"].nunique()
        cohorts[name] = {
            "minimum_spectra_per_strain": definition.minimum_spectra_per_strain,
            "spectrum_cap": definition.spectrum_cap,
            "species": int(inventory["species"].nunique()),
            "eligible_strains": int(len(inventory)),
            "selected_spectra": int(len(cohort)),
            "evaluation_strains_per_cell": int((per_species // 5).sum()),
            "donor_strains_per_cell": int((per_species // 5).sum()),
            "fits_per_cell": expected_fit_count(definition),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "experiments": list(experiments),
        "seeds": list(seeds),
        "rotations": [int(value) for value in rotations],
        "cells": int(len(experiments) * len(seeds) * len(rotations)),
        "fits": int(
            sum(
                expected_fit_count(EXPERIMENTS[name])
                for name in experiments
                for _seed in seeds
                for _rotation in rotations
            )
        ),
        "cohorts": cohorts,
    }


def run_contamination(
    project_root: str | Path,
    output_root: str | Path | None = None,
    experiments: Sequence[str] = ("prevalence", "intensity"),
    seeds: Sequence[int] | None = None,
    rotations: Sequence[int] = (0, 1, 2, 3, 4),
    max_cells: int | None = None,
    parallel_cells: int = 2,
    n_jobs: int = 4,
    n_estimators: int = 300,
    bootstrap_replicates: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    canonical_destination = (root / "output/v2/contamination").resolve()
    destination = (
        Path(output_root).resolve()
        if output_root is not None
        else canonical_destination
    )
    if not 1 <= int(parallel_cells) <= 2:
        raise ValueError("parallel_cells must be 1 or 2")
    if not 1 <= int(n_jobs) <= 4:
        raise ValueError("n_jobs must be between 1 and 4")
    if int(n_estimators) < 1:
        raise ValueError("n_estimators must be positive")
    if destination == canonical_destination and int(n_estimators) != 300:
        raise ValueError("the production contamination checkpoint universe requires 300 trees")
    unknown = set(experiments).difference(EXPERIMENTS)
    if unknown:
        raise KeyError(f"unknown experiments: {sorted(unknown)}")

    frozen_inputs = validate_frozen_inputs(root)
    spec_path = root / "config/v2/analysis_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    seeds = tuple(int(value) for value in (seeds or spec["seeds"]))
    rotations = tuple(int(value) for value in rotations)
    bootstrap_replicates = int(
        spec["bootstrap_replicates"] if bootstrap_replicates is None else bootstrap_replicates
    )
    feature_path = root / "data/processed/production/features.npy"
    expected_feature_sha = str(spec.get("feature_sha256", EXPECTED_FEATURE_SHA256))
    actual_feature_sha = str(frozen_inputs["feature_matrix"]["sha256"])
    if actual_feature_sha != expected_feature_sha:
        raise RuntimeError("validated frozen feature identity differs from the v2 specification")
    manifest_path = root / "data/processed/production/manifest.parquet"
    manifest_sha = str(
        frozen_inputs["verified_files"]["data/processed/production/manifest.parquet"]["sha256"]
    )
    manifest = pd.read_parquet(manifest_path)
    matrix = np.load(feature_path, mmap_mode="r")
    if len(manifest) != matrix.shape[0]:
        raise RuntimeError("feature matrix and manifest row counts differ")
    aggregation_source_sha = _source_sha256()

    # Production is always reconciled against the full 50-cell / 400-fit
    # universe, even when this invocation schedules only a requested subset.
    if destination == canonical_destination:
        universe_experiments = ("prevalence", "intensity")
        universe_seeds = tuple(int(value) for value in spec["seeds"])
        universe_rotations = (0, 1, 2, 3, 4)
    else:
        universe_experiments = tuple(experiments)
        universe_seeds = seeds
        universe_rotations = rotations
    if not set(experiments).issubset(universe_experiments):
        raise ValueError("scheduled experiments are outside the checkpoint universe")
    if not set(seeds).issubset(universe_seeds) or not set(rotations).issubset(universe_rotations):
        raise ValueError("scheduled seeds/rotations are outside the checkpoint universe")

    cohorts: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    cohort_metadata: dict[str, Any] = {}
    for name in universe_experiments:
        definition = EXPERIMENTS[name]
        cohort, inventory = build_cohort(manifest, definition)
        cohorts[name] = (cohort, inventory)
        per_species = inventory.groupby("species")["strain_id"].nunique()
        cohort_metadata[name] = {
            "minimum_spectra_per_strain": definition.minimum_spectra_per_strain,
            "spectrum_cap": definition.spectrum_cap,
            "minimum_strains_per_species": definition.minimum_strains_per_species,
            "species": int(inventory["species"].nunique()),
            "eligible_strains": int(len(inventory)),
            "selected_spectra": int(len(cohort)),
            "evaluation_strains_per_cell": int((per_species // 5).sum()),
            "donor_strains_per_cell": int((per_species // 5).sum()),
            "rotation_remainder_strains": int((per_species % 5).sum()),
            "levels": list(definition.levels),
            "fits_per_cell": expected_fit_count(definition),
        }

    checkpoint_universe = build_expected_checkpoint_universe(
        destination,
        experiments=universe_experiments,
        seeds=universe_seeds,
        rotations=universe_rotations,
        manifest_sha256=manifest_sha,
        feature_sha256=actual_feature_sha,
        n_estimators=int(n_estimators),
        fit_engine_source_sha256=FROZEN_FIT_ENGINE_SOURCE_SHA256,
    )
    destination.mkdir(parents=True, exist_ok=True)
    # Existing scientific checkpoints predate this P1 metadata hardening.  Add
    # explicit source/manifest provenance only after their legacy signatures
    # and artifact hashes have passed validation; prediction bytes are untouched.
    initial_audit = validate_checkpoint_universe(
        destination,
        checkpoint_universe,
        require_complete=False,
        upgrade_missing_provenance=True,
    )

    all_cells = [
        (name, int(seed), int(rotation))
        for name in experiments
        for seed in seeds
        for rotation in rotations
    ]
    missing_assignments = set(initial_audit["missing_assignment_paths"])
    missing_fits = set(initial_audit["missing_fit_paths"])
    incomplete: list[tuple[str, int, int]] = []
    for name, seed, rotation in all_cells:
        definition = EXPERIMENTS[name]
        assignment_dir, cell_dir, _ = _cell_paths(destination, name, seed, rotation)
        assignment_signature = _assignment_signature(
            definition, seed, rotation, manifest_sha, FROZEN_FIT_ENGINE_SOURCE_SHA256
        )
        assignment_relative = (assignment_dir / "assignment_manifest.json").relative_to(
            destination
        ).as_posix()
        valid = 0
        for level in definition.levels:
            for arm in _condition_arms(level):
                fit_signature = _fit_signature(
                    assignment_signature,
                    actual_feature_sha,
                    name,
                    seed,
                    rotation,
                    level,
                    arm,
                    n_estimators,
                )
                if _checkpoint_valid(
                    cell_dir / _level_token(name, level) / arm,
                    fit_signature,
                    fit_engine_source_sha256=FROZEN_FIT_ENGINE_SOURCE_SHA256,
                    feature_sha256=actual_feature_sha,
                    n_estimators=int(n_estimators),
                ):
                    valid += 1
        cell_prefix = cell_dir.relative_to(destination).as_posix() + "/"
        cell_has_missing_fit = any(path.startswith(cell_prefix) for path in missing_fits)
        if (
            force
            or assignment_relative in missing_assignments
            or cell_has_missing_fit
            or valid != expected_fit_count(definition)
        ):
            incomplete.append((name, seed, rotation))
    scheduled = incomplete[: int(max_cells)] if max_cells is not None else incomplete

    results: list[dict[str, Any]] = []
    if scheduled:
        with ThreadPoolExecutor(max_workers=int(parallel_cells)) as executor:
            futures = {
                executor.submit(
                    _process_cell,
                    root,
                    destination,
                    matrix,
                    manifest,
                    cohorts[name][0],
                    cohorts[name][1],
                    EXPERIMENTS[name],
                    seed,
                    rotation,
                    actual_feature_sha,
                    manifest_sha,
                    FROZEN_FIT_ENGINE_SOURCE_SHA256,
                    int(n_estimators),
                    int(n_jobs),
                    bool(force),
                ): (name, seed, rotation)
                for name, seed, rotation in scheduled
            }
            for future in as_completed(futures):
                results.append(future.result())

    run_metadata = {
        "n_estimators": int(n_estimators),
        "n_jobs": int(n_jobs),
        "cohorts": cohort_metadata,
        "inputs": {
            "frozen_input_manifest": frozen_inputs,
            "analysis_spec": {
                "path": spec_path.relative_to(root).as_posix(),
                "sha256": sha256_file(spec_path),
            },
            "feature_matrix": {
                "path": feature_path.relative_to(root).as_posix(),
                "sha256": actual_feature_sha,
                "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
                "dtype": str(matrix.dtype),
            },
            "manifest": {
                "path": manifest_path.relative_to(root).as_posix(),
                "sha256": manifest_sha,
                "rows": int(len(manifest)),
            },
            "fit_engine_source": {
                "path": Path(__file__).name,
                "sha256": FROZEN_FIT_ENGINE_SOURCE_SHA256,
                "status": "frozen source identity of the completed scientific fits",
            },
            "aggregation_source": {
                "path": Path(__file__).name,
                "sha256": aggregation_source_sha,
                "status": "current P1 checkpoint-validation and aggregation source",
            },
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    manifest_record = aggregate_outputs(
        destination,
        checkpoint_universe=checkpoint_universe,
        aggregation_source_sha256=aggregation_source_sha,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=int(spec["bootstrap_seed"]),
        expected_experiments=universe_experiments,
        expected_seeds=universe_seeds,
        expected_rotations=universe_rotations,
        run_metadata=run_metadata,
    )
    manifest_record["scheduled_cells_this_invocation"] = [
        {"experiment": name, "seed": seed, "rotation": rotation}
        for name, seed, rotation in scheduled
    ]
    manifest_record["completed_this_invocation"] = results
    _atomic_json(destination / "run_manifest.json", manifest_record)
    return manifest_record
