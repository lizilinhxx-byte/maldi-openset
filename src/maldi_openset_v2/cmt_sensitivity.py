"""Conservative CMT-identity sensitivity analysis for MALDI-OpenSet v2.

The RKI export contains both directory-derived strain groups and raw
``(GEN, SPE, STR)`` CMT identity strings.  Directory groups are the locked v1
unit.  This module constructs a bipartite graph between directory groups and
non-empty CMT identities, excludes every directory in a component that either
contains a heterogeneous directory or spans more than one directory, and then
recomputes the requested no-refit sensitivity analyses.

Only frozen v1 predictions and already-derived v2 decisions are consumed.  No
model is refit and no threshold is retuned here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score


SCHEMA_VERSION = "2.0.0"
ANALYSIS_COMMIT = "2293ce52a07e24e507e48814dcd3303bf512c0f3"
DESIGNS = ("spectrum_random", "strain_grouped")
MODEL = "extra_trees"
EXPLICIT_BINOMIAL = re.compile(r"^[A-Z][A-Za-z-]+ [a-z][A-Za-z-]+$")
COMPLETE_MODELS = ("cnn1d", "cosine_centroid", "extra_trees", "rbf_svm")
LOCKED_SEEDS = (20260907, 20260919, 20261001, 20261013, 20261025)
EXPLORATORY_RUNS = (("xgboost", 20260907),)
DIRECT_FROZEN_INPUTS = (
    "data/processed/production/manifest.parquet",
    "data/processed/splits.parquet",
    "output/production/artifacts/predictions.parquet",
)
OPEN_SET_OUTPUTS = {
    "decisions": "output/v2/open_set/decisions.parquet",
    "strain_level_leakage_predictions": (
        "output/v2/open_set/strain_level_leakage_predictions.parquet"
    ),
}


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 digest of *path* without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(values: Iterable[object]) -> str:
    return json.dumps(sorted({str(value) for value in values}), ensure_ascii=False)


def _clean_identity(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    result = str(value).strip()
    return result or None


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def add(self, value: str) -> None:
        if value not in self.parent:
            self.parent[value] = value
            self.rank[value] = 0

    def find(self, value: str) -> str:
        self.add(value)
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def classify_species_label(label: object) -> str:
    """Classify a canonical label as an explicit binomial or an ``sp.`` label."""

    text = str(label).strip()
    if re.fullmatch(r"[A-Z][A-Za-z-]+ sp\.", text):
        return "sp_unspecified"
    if EXPLICIT_BINOMIAL.fullmatch(text):
        return "explicit_binomial"
    return "other"


def build_cmt_components(
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build deterministic bipartite CMT components and an exclusion table.

    A component is conservatively excluded when at least one member directory
    has ``cmt_identity_conflict`` set or when the component contains more than
    one directory-derived ``strain_id``.  The latter captures identities reused
    across directory groups and propagates exclusion through heterogeneous
    bridge directories.
    """

    required = {
        "spectrum_id",
        "strain_id",
        "cmt_identity",
        "cmt_identity_conflict",
        "species",
        "genus",
        "analysis_set",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise KeyError(f"manifest lacks required columns: {sorted(missing)}")
    if manifest.empty:
        raise ValueError("manifest is empty")

    data = manifest.copy()
    data["strain_id"] = data["strain_id"].astype(str)
    if data["strain_id"].str.strip().eq("").any():
        raise ValueError("blank strain_id in manifest")
    data["_cmt"] = data["cmt_identity"].map(_clean_identity)

    # Every strain receives a graph node.  Strains with missing CMT data would
    # therefore remain singleton components rather than silently disappearing.
    graph = _UnionFind()
    for strain_id in sorted(data["strain_id"].unique()):
        graph.add(f"strain::{strain_id}")
    edges = (
        data.loc[data["_cmt"].notna(), ["strain_id", "_cmt"]]
        .drop_duplicates()
        .sort_values(["strain_id", "_cmt"], kind="stable")
    )
    for strain_id, identity in edges.itertuples(index=False, name=None):
        graph.union(f"strain::{strain_id}", f"identity::{identity}")

    strain_roots = {
        strain_id: graph.find(f"strain::{strain_id}")
        for strain_id in sorted(data["strain_id"].unique())
    }
    root_strains: dict[str, set[str]] = defaultdict(set)
    root_identities: dict[str, set[str]] = defaultdict(set)
    for strain_id, root in strain_roots.items():
        root_strains[root].add(strain_id)
    for strain_id, identity in edges.itertuples(index=False, name=None):
        root_identities[strain_roots[strain_id]].add(str(identity))

    conflict = (
        data.groupby("strain_id", sort=True)["cmt_identity_conflict"]
        .any()
        .astype(bool)
        .to_dict()
    )
    strain_metadata = (
        data.groupby("strain_id", sort=True)
        .agg(
            n_spectra=("spectrum_id", "nunique"),
            species=("species", lambda x: _json(x)),
            genera=("genus", lambda x: _json(x)),
            analysis_sets=("analysis_set", lambda x: _json(x)),
        )
        .reset_index()
        .set_index("strain_id")
    )
    component_rows: list[dict[str, object]] = []
    exclusion_rows: list[dict[str, object]] = []
    for root in sorted(root_strains, key=lambda value: sorted(root_strains[value])):
        strains = sorted(root_strains[root])
        identities = sorted(root_identities.get(root, set()))
        conflict_strains = sorted(s for s in strains if conflict.get(s, False))
        spans_directories = len(strains) > 1
        has_conflict = bool(conflict_strains)
        excluded = bool(spans_directories or has_conflict)
        component_id = "cmtcc_" + canonical_sha256(
            {"strain_ids": strains, "cmt_identities": identities}
        )[:16]
        spectra = int(sum(int(strain_metadata.loc[s, "n_spectra"]) for s in strains))
        species: set[str] = set()
        genera: set[str] = set()
        analysis_sets: set[str] = set()
        for strain_id in strains:
            species.update(json.loads(strain_metadata.loc[strain_id, "species"]))
            genera.update(json.loads(strain_metadata.loc[strain_id, "genera"]))
            analysis_sets.update(
                json.loads(strain_metadata.loc[strain_id, "analysis_sets"])
            )
        component_rows.append(
            {
                "component_id": component_id,
                "n_strain_directories": len(strains),
                "n_cmt_identities": len(identities),
                "n_spectra": spectra,
                "n_conflict_directories": len(conflict_strains),
                "has_within_directory_conflict": has_conflict,
                "spans_multiple_directories": spans_directories,
                "conservative_exclude": excluded,
                "strain_ids": _json(strains),
                "cmt_identities": _json(identities),
                "conflict_strain_ids": _json(conflict_strains),
                "species": _json(species),
                "genera": _json(genera),
                "analysis_sets": _json(analysis_sets),
            }
        )
        if not excluded:
            continue
        component_reasons = []
        if has_conflict:
            component_reasons.append("component_contains_conflict_directory")
        if spans_directories:
            component_reasons.append("cmt_component_spans_multiple_directories")
        for strain_id in strains:
            own_identities = edges.loc[
                edges["strain_id"].eq(strain_id), "_cmt"
            ].astype(str)
            own_reasons = list(component_reasons)
            if conflict.get(strain_id, False):
                own_reasons.insert(0, "within_directory_cmt_conflict")
            exclusion_rows.append(
                {
                    "strain_id": strain_id,
                    "component_id": component_id,
                    "exclude_reason": ";".join(dict.fromkeys(own_reasons)),
                    "own_cmt_identity_conflict": bool(conflict.get(strain_id, False)),
                    "component_spans_multiple_directories": spans_directories,
                    "component_contains_conflict_directory": has_conflict,
                    "n_component_strain_directories": len(strains),
                    "n_component_cmt_identities": len(identities),
                    "n_spectra": int(strain_metadata.loc[strain_id, "n_spectra"]),
                    "species": strain_metadata.loc[strain_id, "species"],
                    "genera": strain_metadata.loc[strain_id, "genera"],
                    "analysis_sets": strain_metadata.loc[strain_id, "analysis_sets"],
                    "cmt_identities": _json(own_identities),
                }
            )

    components = pd.DataFrame(component_rows).sort_values(
        ["conservative_exclude", "component_id"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    exclusions = pd.DataFrame(exclusion_rows).sort_values(
        ["component_id", "strain_id"], kind="stable"
    ).reset_index(drop=True)
    return components, exclusions


def validate_grouped_cmt_isolation(
    manifest: pd.DataFrame,
    splits: pd.DataFrame,
    excluded_strains: Iterable[str],
) -> pd.DataFrame:
    """Audit known non-empty CMT identities across grouped split roles."""

    required_manifest = {"strain_id", "cmt_identity"}
    required_splits = {"spectrum_id", "strain_id", "seed", "design", "fold", "role"}
    if missing := required_manifest.difference(manifest.columns):
        raise KeyError(f"manifest lacks split-audit columns: {sorted(missing)}")
    if missing := required_splits.difference(splits.columns):
        raise KeyError(f"split table lacks columns: {sorted(missing)}")

    excluded = {str(value) for value in excluded_strains}
    identity = manifest[["strain_id", "cmt_identity"]].copy()
    identity["strain_id"] = identity["strain_id"].astype(str)
    identity["cmt_identity"] = identity["cmt_identity"].map(_clean_identity)
    identity = identity.loc[
        identity["cmt_identity"].notna()
        & ~identity["strain_id"].isin(excluded)
    ].drop_duplicates()
    clean_counts = identity.groupby("cmt_identity")["strain_id"].nunique()
    duplicate_clean = sorted(clean_counts[clean_counts > 1].index.astype(str))

    selected = splits.loc[
        splits["design"].astype(str).eq("strain_grouped")
        & ~splits["strain_id"].astype(str).isin(excluded)
    ].copy()
    selected["strain_id"] = selected["strain_id"].astype(str)
    joined = selected.merge(identity, on="strain_id", how="inner", validate="many_to_many")
    joined = joined[
        ["seed", "fold", "role", "spectrum_id", "strain_id", "cmt_identity"]
    ].drop_duplicates()
    rows: list[dict[str, object]] = []
    role_pairs = (
        ("train", "test_known"),
        ("train", "calibration"),
        ("calibration", "test_known"),
        ("train", "test_ood"),
        ("calibration", "test_ood"),
        ("test_known", "test_ood"),
    )
    for (seed, fold), part in joined.groupby(["seed", "fold"], sort=True):
        identity_by_role = {
            role: set(part.loc[part["role"].astype(str).eq(role), "cmt_identity"].astype(str))
            for role in sorted(part["role"].astype(str).unique())
        }
        strain_by_role = {
            role: set(part.loc[part["role"].astype(str).eq(role), "strain_id"].astype(str))
            for role in sorted(part["role"].astype(str).unique())
        }
        row: dict[str, object] = {
            "design": "strain_grouped",
            "seed": int(seed),
            "fold": int(fold),
            "n_clean_split_rows": int(len(part)),
            "n_clean_strains": int(part["strain_id"].nunique()),
            "n_known_cmt_identities": int(part["cmt_identity"].nunique()),
            "duplicate_identity_across_clean_directories_n": len(duplicate_clean),
            "duplicate_identity_across_clean_directories": _json(duplicate_clean),
        }
        forbidden_total: set[str] = set()
        for left, right in role_pairs:
            overlap = identity_by_role.get(left, set()) & identity_by_role.get(right, set())
            strain_overlap = strain_by_role.get(left, set()) & strain_by_role.get(right, set())
            name = f"{left}__{right}"
            row[f"{name}__cmt_overlap_n"] = len(overlap)
            row[f"{name}__cmt_overlap"] = _json(overlap)
            row[f"{name}__strain_overlap_n"] = len(strain_overlap)
            forbidden_total.update(overlap)
        row["any_cross_role_cmt_overlap_n"] = len(forbidden_total)
        row["validation_pass"] = len(duplicate_clean) == 0 and not forbidden_total
        rows.append(row)
    if not rows:
        raise ValueError("no strain_grouped split cells were available for CMT audit")
    return pd.DataFrame(rows)


def _validate_oof_frame(frame: pd.DataFrame, key: tuple[str, int]) -> None:
    if frame.empty:
        raise ValueError(f"empty OOF frame: {key}")
    if frame["spectrum_id"].duplicated().any():
        raise ValueError(f"duplicated OOF spectrum IDs: {key}")
    species_per_spectrum = frame.groupby("spectrum_id")["species"].nunique()
    if (species_per_spectrum > 1).any():
        raise ValueError(f"inconsistent OOF truth labels: {key}")


def clean_leakage_metrics(
    predictions: pd.DataFrame,
    strain_predictions: pd.DataFrame,
    excluded_strains: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Recompute no-refit ExtraTrees leakage metrics on identical clean spectra."""

    required = {
        "spectrum_id",
        "strain_id",
        "species",
        "role",
        "predicted_species",
        "design",
        "model",
        "seed",
        "fold",
    }
    if missing := required.difference(predictions.columns):
        raise KeyError(f"prediction table lacks columns: {sorted(missing)}")
    required_strain = {"strain_id", "species", "predicted_species", "design", "model", "seed", "n_spectra"}
    if missing := required_strain.difference(strain_predictions.columns):
        raise KeyError(f"strain prediction table lacks columns: {sorted(missing)}")

    excluded = {str(value) for value in excluded_strains}
    spectra = predictions.loc[
        predictions["model"].astype(str).eq(MODEL)
        & predictions["design"].astype(str).isin(DESIGNS)
        & predictions["role"].astype(str).eq("test_known")
        & ~predictions["strain_id"].astype(str).isin(excluded)
    ].copy()
    strains = strain_predictions.loc[
        strain_predictions["model"].astype(str).eq(MODEL)
        & strain_predictions["design"].astype(str).isin(DESIGNS)
        & ~strain_predictions["strain_id"].astype(str).isin(excluded)
    ].copy()
    if spectra.empty or strains.empty:
        raise ValueError("missing clean ExtraTrees OOF predictions")

    metrics: list[dict[str, object]] = []
    cohort_checks: list[dict[str, object]] = []
    seeds = sorted(set(spectra["seed"].astype(int)))
    for seed in seeds:
        by_design: dict[str, pd.DataFrame] = {}
        strain_by_design: dict[str, pd.DataFrame] = {}
        for design in DESIGNS:
            part = spectra.loc[
                spectra["seed"].astype(int).eq(seed)
                & spectra["design"].astype(str).eq(design)
            ].copy()
            _validate_oof_frame(part, (design, seed))
            strain_part = strains.loc[
                strains["seed"].astype(int).eq(seed)
                & strains["design"].astype(str).eq(design)
            ].copy()
            if strain_part["strain_id"].duplicated().any():
                raise ValueError(f"duplicated strain-level rows: {(design, seed)}")
            by_design[design] = part
            strain_by_design[design] = strain_part

        random_ids = set(by_design["spectrum_random"]["spectrum_id"].astype(str))
        grouped_ids = set(by_design["strain_grouped"]["spectrum_id"].astype(str))
        random_strains = set(strain_by_design["spectrum_random"]["strain_id"].astype(str))
        grouped_strains = set(strain_by_design["strain_grouped"]["strain_id"].astype(str))
        spectra_match = random_ids == grouped_ids
        strains_match = random_strains == grouped_strains
        truth_random = by_design["spectrum_random"].set_index("spectrum_id")["species"].astype(str)
        truth_grouped = by_design["strain_grouped"].set_index("spectrum_id")["species"].astype(str)
        truth_match = spectra_match and truth_random.sort_index().equals(truth_grouped.sort_index())
        cohort_checks.append(
            {
                "seed": int(seed),
                "random_n_spectra": len(random_ids),
                "grouped_n_spectra": len(grouped_ids),
                "spectrum_sets_identical": spectra_match,
                "random_n_strains": len(random_strains),
                "grouped_n_strains": len(grouped_strains),
                "strain_sets_identical": strains_match,
                "truth_labels_identical": truth_match,
                "random_only_spectra": _json(random_ids - grouped_ids),
                "grouped_only_spectra": _json(grouped_ids - random_ids),
                "validation_pass": spectra_match and strains_match and truth_match,
            }
        )
        if not (spectra_match and strains_match and truth_match):
            raise ValueError(f"random/grouped clean cohorts differ for seed {seed}")

        for design in DESIGNS:
            part = by_design[design]
            strain_part = strain_by_design[design]
            if set(part["strain_id"].astype(str)) != set(strain_part["strain_id"].astype(str)):
                raise ValueError(f"spectrum/strain prediction cohorts differ: {(design, seed)}")
            observed_counts = part.groupby("strain_id")["spectrum_id"].size().astype(int)
            declared_counts = strain_part.set_index("strain_id")["n_spectra"].astype(int)
            if not observed_counts.sort_index().equals(declared_counts.sort_index()):
                raise ValueError(f"strain spectrum counts disagree: {(design, seed)}")

            truth = part["species"].astype(str).to_numpy()
            predicted = part["predicted_species"].astype(str).to_numpy()
            count = part.groupby("strain_id")["spectrum_id"].transform("size").to_numpy(float)
            weight = 1.0 / count
            strain_truth = strain_part["species"].astype(str).to_numpy()
            strain_predicted = strain_part["predicted_species"].astype(str).to_numpy()
            metrics.append(
                {
                    "population": "cmt_component_clean",
                    "design": design,
                    "model": MODEL,
                    "seed": int(seed),
                    "n_spectra": int(len(part)),
                    "n_strains": int(part["strain_id"].nunique()),
                    "n_species": int(part["species"].nunique()),
                    "pooled_spectrum_macro_f1": float(
                        f1_score(truth, predicted, average="macro", zero_division=0)
                    ),
                    "equal_strain_weight_macro_f1": float(
                        f1_score(
                            truth,
                            predicted,
                            average="macro",
                            sample_weight=weight,
                            zero_division=0,
                        )
                    ),
                    "strain_mean_probability_macro_f1": float(
                        f1_score(
                            strain_truth,
                            strain_predicted,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    "strain_mean_probability_balanced_accuracy": float(
                        balanced_accuracy_score(strain_truth, strain_predicted)
                    ),
                }
            )

    metric_frame = pd.DataFrame(metrics).sort_values(
        ["seed", "design"], kind="stable"
    ).reset_index(drop=True)
    cohort_frame = pd.DataFrame(cohort_checks)
    metric_names = (
        "pooled_spectrum_macro_f1",
        "equal_strain_weight_macro_f1",
        "strain_mean_probability_macro_f1",
    )
    delta_rows: list[dict[str, object]] = []
    for seed, part in metric_frame.groupby("seed", sort=True):
        indexed = part.set_index("design")
        for metric in metric_names:
            random_value = float(indexed.loc["spectrum_random", metric])
            grouped_value = float(indexed.loc["strain_grouped", metric])
            delta_rows.append(
                {
                    "population": "cmt_component_clean",
                    "model": MODEL,
                    "seed": int(seed),
                    "metric": metric,
                    "spectrum_random": random_value,
                    "strain_grouped": grouped_value,
                    "random_minus_grouped": random_value - grouped_value,
                }
            )
    deltas = pd.DataFrame(delta_rows)
    summary_rows: list[dict[str, object]] = []
    for metric, part in deltas.groupby("metric", sort=True):
        values = part["random_minus_grouped"].to_numpy(float)
        summary_rows.append(
            {
                "population": "cmt_component_clean",
                "model": MODEL,
                "metric": metric,
                "n_seeds": len(values),
                "spectrum_random_mean": float(part["spectrum_random"].mean()),
                "strain_grouped_mean": float(part["strain_grouped"].mean()),
                "random_minus_grouped_mean": float(values.mean()),
                "random_minus_grouped_sd": float(values.std(ddof=1)) if len(values) > 1 else math.nan,
                "random_minus_grouped_min": float(values.min()),
                "random_minus_grouped_max": float(values.max()),
                "status": "CMT-component exclusion sensitivity; no model refitting",
            }
        )
    return metric_frame, deltas, pd.DataFrame(summary_rows), cohort_frame


def _rate(frame: pd.DataFrame, column: str, equal_strain: bool = False) -> float:
    if frame.empty:
        return math.nan
    values = frame[column].astype(float)
    if not equal_strain:
        return float(values.mean())
    temporary = frame[["strain_id"]].copy()
    temporary["value"] = values.to_numpy()
    return float(temporary.groupby("strain_id", sort=False)["value"].mean().mean())


def _conditional_accuracy(
    frame: pd.DataFrame,
    accepted_column: str,
    correct_column: str,
    *,
    equal_strain: bool = False,
) -> float:
    accepted = frame[accepted_column].astype(bool)
    if not equal_strain:
        denominator = int(accepted.sum())
        return (
            float(frame.loc[accepted, correct_column].astype(bool).sum() / denominator)
            if denominator
            else math.nan
        )
    temporary = frame[["strain_id"]].copy()
    temporary["accepted"] = accepted.astype(float).to_numpy()
    temporary["correct"] = (
        accepted & frame[correct_column].astype(bool)
    ).astype(float).to_numpy()
    by_strain = temporary.groupby("strain_id", sort=False)[["accepted", "correct"]].mean()
    denominator = float(by_strain["accepted"].sum())
    return float(by_strain["correct"].sum() / denominator) if denominator else math.nan


def _open_metric_row(
    part: pd.DataFrame,
    *,
    selector: str,
    label_type: str,
    evaluation_instance: str,
    seed: int,
    fold: int | None,
) -> dict[str, object]:
    identified = ~part["corrected_unidentified"].astype(bool)
    work = part.assign(_corrected_identified=identified)
    row: dict[str, object] = {
        "population": "cmt_component_clean",
        "design": "strain_grouped",
        "model": MODEL,
        "selector": selector,
        "label_type": label_type,
        "evaluation_instance": evaluation_instance,
        "seed": int(seed),
        "fold": int(fold) if fold is not None else pd.NA,
        "n_prediction_rows": int(len(part)),
        "n_unique_spectra": int(part["spectrum_id"].nunique()),
        "n_strains": int(part["strain_id"].nunique()),
        "n_species_labels": int(part["species"].nunique()),
        "species_labels": _json(part["species"]),
    }
    rate_columns = (
        "confidence_only_accept",
        "saved_singleton",
        "legacy_species_accept",
        "corrected_species_accept",
        "corrected_genus_report",
        "corrected_unidentified",
        "corrected_any_report_correct",
        "corrected_species_report_correct",
        "saved_species_set_contains_true",
    )
    for column in rate_columns:
        row[f"{column}__spectrum"] = _rate(work, column)
        row[f"{column}__equal_strain"] = _rate(work, column, True)
    row["corrected_identified__spectrum"] = _rate(work, "_corrected_identified")
    row["corrected_identified__equal_strain"] = _rate(
        work, "_corrected_identified", True
    )
    row["corrected_species_report_accuracy__spectrum"] = _conditional_accuracy(
        work,
        "corrected_species_accept",
        "corrected_species_report_correct",
    )
    row["corrected_species_report_accuracy__equal_strain"] = _conditional_accuracy(
        work,
        "corrected_species_accept",
        "corrected_species_report_correct",
        equal_strain=True,
    )
    row["corrected_identified_accuracy__spectrum"] = _conditional_accuracy(
        work,
        "_corrected_identified",
        "corrected_any_report_correct",
    )
    row["corrected_identified_accuracy__equal_strain"] = _conditional_accuracy(
        work,
        "_corrected_identified",
        "corrected_any_report_correct",
        equal_strain=True,
    )
    row["mean_saved_species_set_size__spectrum"] = float(
        work["conformal_species_set_size"].astype(float).mean()
    )
    row["mean_saved_species_set_size__equal_strain"] = _rate(
        work.assign(_set_size=work["conformal_species_set_size"]),
        "_set_size",
        True,
    )
    return row


def open_set_label_sensitivity(
    decisions: pd.DataFrame,
    excluded_strains: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Describe corrected v2 reporting separately for binomial and ``sp.`` labels.

    Known spectra are pooled across their five OOF folds within each seed.  OOD
    spectra are evaluated separately for every seed/fold fitted model because
    each OOD spectrum is intentionally scored by all five fold models.
    """

    required = {
        "spectrum_id",
        "strain_id",
        "species",
        "role",
        "ood_distance",
        "design",
        "model",
        "seed",
        "fold",
        "confidence_only_accept",
        "saved_singleton",
        "legacy_species_accept",
        "corrected_species_accept",
        "corrected_genus_report",
        "corrected_unidentified",
        "corrected_any_report_correct",
        "corrected_species_report_correct",
        "saved_species_set_contains_true",
        "conformal_species_set_size",
    }
    if missing := required.difference(decisions.columns):
        raise KeyError(f"v2 decision table lacks columns: {sorted(missing)}")
    excluded = {str(value) for value in excluded_strains}
    data = decisions.loc[
        decisions["model"].astype(str).eq(MODEL)
        & decisions["design"].astype(str).eq("strain_grouped")
        & decisions["role"].astype(str).isin(["test_known", "test_ood"])
        & ~decisions["strain_id"].astype(str).isin(excluded)
    ].copy()
    data["label_type"] = data["species"].map(classify_species_label)
    unexpected = sorted(data.loc[data["label_type"].eq("other"), "species"].astype(str).unique())
    if unexpected:
        raise ValueError(f"unclassified canonical species labels: {unexpected}")

    rows: list[dict[str, object]] = []
    # Known: one OOF prediction per spectrum per seed after concatenating folds.
    known = data.loc[data["role"].astype(str).eq("test_known")]
    for seed, seed_part in known.groupby("seed", sort=True):
        if seed_part["spectrum_id"].duplicated().any():
            raise ValueError(f"known OOF decision rows duplicated for seed {seed}")
        for label_type, part in seed_part.groupby("label_type", sort=True):
            rows.append(
                _open_metric_row(
                    part,
                    selector="known",
                    label_type=str(label_type),
                    evaluation_instance=f"known_oof__seed-{int(seed)}",
                    seed=int(seed),
                    fold=None,
                )
            )

    # OOD: each fold is a distinct fitted-model evaluation instance.
    ood = data.loc[data["role"].astype(str).eq("test_ood")]
    selectors: Mapping[str, pd.Series] = {
        "all_ood": pd.Series(True, index=ood.index),
        "same_genus_ood": ood["ood_distance"].astype(str).eq("near"),
        "different_genus_ood": ood["ood_distance"].astype(str).eq("far"),
    }
    for (seed, fold), instance in ood.groupby(["seed", "fold"], sort=True):
        for selector, mask in selectors.items():
            selected = instance.loc[mask.reindex(instance.index, fill_value=False)]
            for label_type, part in selected.groupby("label_type", sort=True):
                rows.append(
                    _open_metric_row(
                        part,
                        selector=selector,
                        label_type=str(label_type),
                        evaluation_instance=(
                            f"ood__seed-{int(seed)}__fold-{int(fold)}"
                        ),
                        seed=int(seed),
                        fold=int(fold),
                    )
                )
    per_instance = pd.DataFrame(rows).sort_values(
        ["selector", "label_type", "seed", "fold"], kind="stable"
    ).reset_index(drop=True)

    key_columns = {
        "population",
        "design",
        "model",
        "selector",
        "label_type",
        "evaluation_instance",
        "seed",
        "fold",
        "species_labels",
    }
    numeric = [
        column
        for column in per_instance.select_dtypes(include=[np.number, "boolean"]).columns
        if column not in key_columns
    ]
    summary_rows: list[dict[str, object]] = []
    for (selector, label_type), part in per_instance.groupby(
        ["selector", "label_type"], sort=True
    ):
        for metric in numeric:
            values = pd.to_numeric(part[metric], errors="coerce").dropna().to_numpy(float)
            if not len(values):
                continue
            summary_rows.append(
                {
                    "population": "cmt_component_clean",
                    "design": "strain_grouped",
                    "model": MODEL,
                    "selector": selector,
                    "label_type": label_type,
                    "metric": metric,
                    "n_evaluation_instances": int(len(values)),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)) if len(values) > 1 else math.nan,
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                    "aggregation_note": (
                        "known: five seed-pooled OOF instances; OOD: 25 seed-fold fitted-model instances"
                    ),
                }
            )
    return per_instance, pd.DataFrame(summary_rows)


def expected_frozen_run_keys() -> set[tuple[str, str, int, int]]:
    """Return the exact 210-run grid declared by the strengthened v2 freeze."""

    keys = {
        (design, model, int(seed), int(fold))
        for design in DESIGNS
        for model in COMPLETE_MODELS
        for seed in LOCKED_SEEDS
        for fold in range(5)
    }
    keys.update(
        (design, model, int(seed), int(fold))
        for model, seed in EXPLORATORY_RUNS
        for design in DESIGNS
        for fold in range(5)
    )
    return keys


def _safe_project_path(project_root: Path, relative: object) -> Path:
    text = str(relative)
    path = (project_root / text).resolve()
    if path == project_root or project_root not in path.parents:
        raise RuntimeError(f"provenance path is outside project: {text}")
    return path


def _verify_declared_file(
    project_root: Path,
    record: Mapping[str, object],
    *,
    expected_relative: str | None = None,
) -> dict[str, object]:
    required = {"path", "bytes", "sha256"}
    if missing := required.difference(record):
        raise RuntimeError(f"provenance file record lacks fields: {sorted(missing)}")
    relative = str(record["path"])
    if expected_relative is not None and relative != expected_relative:
        raise RuntimeError(
            f"provenance path mismatch: expected {expected_relative}, found {relative}"
        )
    path = _safe_project_path(project_root, relative)
    if not path.is_file():
        raise RuntimeError(f"provenance input is missing: {relative}")
    actual_bytes = path.stat().st_size
    actual_sha = sha256_file(path)
    if actual_bytes != int(record["bytes"]) or actual_sha != str(record["sha256"]):
        raise RuntimeError(f"provenance file identity differs: {relative}")
    return {"path": relative, "bytes": actual_bytes, "sha256": actual_sha}


def validate_common_freeze_ledger(
    project_root: str | Path,
    frozen_manifest_path: str | Path | None = None,
) -> dict[str, object]:
    """Verify the common frozen-v1 ledger before reading analysis inputs.

    The ledger-file digest is discovered dynamically.  Git HEAD is recorded as
    provenance only and is deliberately not an execution gate.  Every fixed
    file listed by the ledger is byte/hash checked, the separately listed
    feature matrix is checked, and both the declared key grid and run inventory
    must equal the exact 210-run design.
    """

    root = Path(project_root).resolve()
    ledger_path = Path(
        frozen_manifest_path or root / "output" / "v2" / "input_manifest.json"
    ).resolve()
    expected_path = (root / "output" / "v2" / "input_manifest.json").resolve()
    if ledger_path != expected_path:
        raise RuntimeError("common freeze ledger must be output/v2/input_manifest.json")
    if not ledger_path.is_file():
        raise FileNotFoundError(ledger_path)
    ledger_sha = sha256_file(ledger_path)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("common freeze ledger schema differs")
    if ledger.get("legacy_analysis_commit") != ANALYSIS_COMMIT:
        raise RuntimeError("common freeze ledger legacy commit differs")
    if not ledger.get("v1_write_protected_by_contract"):
        raise RuntimeError("common freeze ledger does not declare v1 write protection")

    file_records = ledger.get("files")
    if not isinstance(file_records, list) or not file_records:
        raise RuntimeError("common freeze ledger has no fixed-file inventory")
    by_path: dict[str, Mapping[str, object]] = {}
    for record in file_records:
        if not isinstance(record, Mapping) or "path" not in record:
            raise RuntimeError("invalid fixed-file record in common freeze ledger")
        relative = str(record["path"])
        if relative in by_path:
            raise RuntimeError(f"duplicate fixed-file record: {relative}")
        by_path[relative] = record
    if missing := set(DIRECT_FROZEN_INPUTS).difference(by_path):
        raise RuntimeError(f"common freeze ledger lacks direct CMT inputs: {sorted(missing)}")
    verified_files = {
        relative: _verify_declared_file(root, record, expected_relative=relative)
        for relative, record in sorted(by_path.items())
    }

    feature_relative = str(ledger.get("feature_matrix_path", ""))
    feature_record = {
        "path": feature_relative,
        "bytes": ledger.get("feature_matrix_bytes", -1),
        "sha256": ledger.get("feature_sha256", ""),
    }
    verified_feature = _verify_declared_file(
        root, feature_record, expected_relative=feature_relative
    )

    expected = expected_frozen_run_keys()
    declared_keys_raw = ledger.get("expected_run_keys")
    if not isinstance(declared_keys_raw, list):
        raise RuntimeError("common freeze ledger lacks expected_run_keys")
    declared_keys: list[tuple[str, str, int, int]] = []
    for item in declared_keys_raw:
        try:
            declared_keys.append(
                (
                    str(item["design"]),
                    str(item["model"]),
                    int(item["seed"]),
                    int(item["fold"]),
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("invalid expected_run_keys record") from error
    if len(declared_keys) != len(set(declared_keys)):
        raise RuntimeError("duplicate key in expected_run_keys")
    if set(declared_keys) != expected:
        raise RuntimeError("common freeze ledger expected run grid differs")

    run_records_raw = ledger.get("runs")
    if (
        not isinstance(run_records_raw, list)
        or int(ledger.get("run_count", -1)) != len(expected)
        or len(run_records_raw) != len(expected)
    ):
        raise RuntimeError("common freeze ledger run inventory count differs")
    inventory: dict[tuple[str, str, int, int], Mapping[str, object]] = {}
    verified_run_manifests: list[dict[str, object]] = []
    for record in run_records_raw:
        try:
            key = (
                str(record["design"]),
                str(record["model"]),
                int(record["seed"]),
                int(record["fold"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("invalid frozen run inventory record") from error
        if key in inventory:
            raise RuntimeError(f"duplicate frozen run inventory key: {key}")
        inventory[key] = record
        design, model, seed, fold = key
        expected_run_id = f"{design}__{model}__seed-{seed}__fold-{fold}"
        if str(record.get("run_id")) != expected_run_id:
            raise RuntimeError(f"frozen run_id differs for {key}")
        required_hashes = {
            "manifest_sha256",
            "predictions_sha256",
            "probabilities_sha256",
            "classes_sha256",
            "split_sha256",
        }
        if any(not str(record.get(field, "")) for field in required_hashes):
            raise RuntimeError(f"frozen run record lacks hashes: {expected_run_id}")
        manifest_path = _safe_project_path(root, record.get("manifest_path", ""))
        if not manifest_path.is_file() or sha256_file(manifest_path) != str(
            record["manifest_sha256"]
        ):
            raise RuntimeError(f"frozen run manifest identity differs: {expected_run_id}")
        run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            run_manifest.get("run_id") != expected_run_id
            or run_manifest.get("status") != "complete"
            or run_manifest.get("git_commit") != ANALYSIS_COMMIT
            or run_manifest.get("feature_sha256") != ledger.get("feature_sha256")
            or str(run_manifest.get("split_sha256")) != str(record["split_sha256"])
        ):
            raise RuntimeError(f"frozen run manifest metadata differs: {expected_run_id}")
        for artifact in ("predictions", "probabilities", "classes"):
            if str(run_manifest.get(f"{artifact}_sha256")) != str(
                record[f"{artifact}_sha256"]
            ):
                raise RuntimeError(
                    f"frozen run manifest {artifact} binding differs: {expected_run_id}"
                )
        verified_run_manifests.append(
            {
                "run_id": expected_run_id,
                "path": manifest_path.relative_to(root).as_posix(),
                "sha256": str(record["manifest_sha256"]),
            }
        )
    if set(inventory) != expected:
        raise RuntimeError("common freeze ledger run inventory grid differs")

    return {
        "schema_version": SCHEMA_VERSION,
        "path": ledger_path.relative_to(root).as_posix(),
        "sha256": ledger_sha,
        "legacy_analysis_commit": ANALYSIS_COMMIT,
        "current_head_at_execution": _git_head(root),
        "current_head_is_provenance_only": True,
        "declared_run_count": int(ledger["run_count"]),
        "expected_run_grid_sha256": canonical_sha256(
            [
                {"design": d, "model": m, "seed": s, "fold": f}
                for d, m, s, f in sorted(expected)
            ]
        ),
        "verified_files": verified_files,
        "feature_matrix": verified_feature,
        "verified_run_manifest_count": len(verified_run_manifests),
        "ledger": ledger,
    }


def validate_open_set_dependency(
    project_root: str | Path,
    common_freeze: Mapping[str, object],
    open_set_dir: str | Path | None = None,
) -> dict[str, object]:
    """Verify open-set outputs and their transitive common-ledger binding."""

    root = Path(project_root).resolve()
    directory = Path(open_set_dir or root / "output" / "v2" / "open_set").resolve()
    expected_directory = (root / "output" / "v2" / "open_set").resolve()
    if directory != expected_directory:
        raise RuntimeError("open-set dependency must be output/v2/open_set")
    run_manifest_path = directory / "run_manifest.json"
    analysis_input_path = directory / "input_manifest.json"
    if not run_manifest_path.is_file() or not analysis_input_path.is_file():
        raise FileNotFoundError("open-set provenance manifests are incomplete")
    run_manifest_sha = sha256_file(run_manifest_path)
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("analysis_commit") != ANALYSIS_COMMIT:
        raise RuntimeError("open-set analysis commit binding differs")
    common_sha = str(common_freeze.get("sha256", ""))
    if str(run_manifest.get("frozen_input_manifest_sha256", "")) != common_sha:
        raise RuntimeError("open-set run manifest common freeze binding differs")

    input_manifest_sha = sha256_file(analysis_input_path)
    if str(run_manifest.get("inputs_manifest_sha256", "")) != input_manifest_sha:
        raise RuntimeError("open-set analysis input-manifest hash differs")
    analysis_input = json.loads(analysis_input_path.read_text(encoding="utf-8"))
    frozen_binding = analysis_input.get("frozen_input_manifest")
    if not isinstance(frozen_binding, Mapping):
        raise RuntimeError("open-set analysis input manifest lacks frozen binding")
    if (
        str(frozen_binding.get("path")) != str(common_freeze.get("path"))
        or str(frozen_binding.get("sha256")) != common_sha
        or int(frozen_binding.get("declared_run_count", -1))
        != int(common_freeze.get("declared_run_count", -2))
    ):
        raise RuntimeError("open-set analysis input binds a different common ledger")
    if (
        analysis_input.get("analysis_commit") != ANALYSIS_COMMIT
        or int(analysis_input.get("n_locked_runs", -1)) != 200
    ):
        raise RuntimeError("open-set analysis input run binding differs")

    declared_outputs = run_manifest.get("outputs")
    if not isinstance(declared_outputs, Mapping):
        raise RuntimeError("open-set run manifest has no output inventory")
    verified_outputs: dict[str, dict[str, object]] = {}
    for name, expected_relative in OPEN_SET_OUTPUTS.items():
        record = declared_outputs.get(name)
        if not isinstance(record, Mapping):
            raise RuntimeError(f"open-set run manifest lacks output record: {name}")
        verified_outputs[name] = _verify_declared_file(
            root, record, expected_relative=expected_relative
        )

    return {
        "directory": directory.relative_to(root).as_posix(),
        "run_manifest": {
            "path": run_manifest_path.relative_to(root).as_posix(),
            "bytes": run_manifest_path.stat().st_size,
            "sha256": run_manifest_sha,
        },
        "analysis_input_manifest": {
            "path": analysis_input_path.relative_to(root).as_posix(),
            "bytes": analysis_input_path.stat().st_size,
            "sha256": input_manifest_sha,
        },
        "common_freeze_sha256": common_sha,
        "transitive_common_freeze_binding_verified": True,
        "outputs": verified_outputs,
    }


def _git_head(project_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _file_record(path: Path, project_root: Path) -> dict[str, object]:
    return {
        "path": path.resolve().relative_to(project_root.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def run_analysis(project_root: str | Path, output_dir: str | Path) -> dict[str, object]:
    """Run the complete CMT sensitivity and write an auditable result bundle."""

    project_root = Path(project_root).resolve()
    output_dir = Path(output_dir).resolve()
    # Provenance gates run before any result is read or written.  Paths to v2
    # derivatives come from their producing run manifest, not from a fresh hash
    # captured after the fact.
    common_freeze = validate_common_freeze_ledger(project_root)
    open_set_dependency = validate_open_set_dependency(project_root, common_freeze)
    verified_files = common_freeze["verified_files"]
    paths = {
        "manifest": project_root
        / verified_files["data/processed/production/manifest.parquet"]["path"],
        "splits": project_root / verified_files["data/processed/splits.parquet"]["path"],
        "predictions": project_root
        / verified_files["output/production/artifacts/predictions.parquet"]["path"],
        "strain_predictions": project_root
        / open_set_dependency["outputs"]["strain_level_leakage_predictions"]["path"],
        "decisions": project_root
        / open_set_dependency["outputs"]["decisions"]["path"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_columns = [
        "spectrum_id",
        "strain_id",
        "cmt_identity",
        "cmt_identity_conflict",
        "species",
        "genus",
        "analysis_set",
        "primary_known",
        "primary_ood",
    ]
    manifest = pd.read_parquet(paths["manifest"], columns=manifest_columns)
    components, exclusions = build_cmt_components(manifest)
    excluded = set(exclusions["strain_id"].astype(str))

    splits = pd.read_parquet(paths["splits"])
    split_validation = validate_grouped_cmt_isolation(manifest, splits, excluded)

    output_paths = {
        "components": output_dir / "cmt_components.parquet",
        "exclusions": output_dir / "excluded_strains.csv",
        "split_validation": output_dir / "split_validation.csv",
        "leakage_metrics": output_dir / "leakage_metrics.csv",
        "leakage_deltas": output_dir / "leakage_deltas.csv",
        "leakage_summary": output_dir / "leakage_summary.csv",
        "cohort_validation": output_dir / "cohort_validation.csv",
        "open_set_label_metrics": output_dir / "open_set_label_metrics.csv",
        "open_set_label_summary": output_dir / "open_set_label_summary.csv",
        "summary": output_dir / "summary.json",
        "run_manifest": output_dir / "run_manifest.json",
    }
    components.to_parquet(output_paths["components"], index=False)
    exclusions.to_csv(output_paths["exclusions"], index=False)
    split_validation.to_csv(output_paths["split_validation"], index=False)

    common_record = {
        key: value for key, value in common_freeze.items() if key != "ledger"
    }
    base_manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "analysis_commit": ANALYSIS_COMMIT,
        "current_git_head_provenance": _git_head(project_root),
        "current_head_is_provenance_only": True,
        "status": "running",
        "method": {
            "graph": "undirected bipartite connected components between directory strain_id and each non-empty CMT identity",
            "exclusion": "all directories in any component containing a cmt_identity_conflict directory or spanning multiple directories",
            "model_refitting": False,
            "threshold_retuning": False,
            "leakage_population": "identical remaining test_known spectra after component exclusion",
            "open_set_population": "component-clean ExtraTrees strain-grouped v2 decisions",
            "label_types": {
                "explicit_binomial": "canonical Genus epithet label matching a strict two-token binomial",
                "sp_unspecified": "canonical Genus sp. label",
            },
        },
        "provenance": {
            "common_freeze": common_record,
            "open_set_dependency": open_set_dependency,
            "all_inputs_verified_before_analysis": True,
        },
        "component_counts": {
            "components": int(len(components)),
            "excluded_components": int(components["conservative_exclude"].sum()),
            "excluded_strain_directories": int(len(excluded)),
            "original_conflict_directories": int(
                manifest.loc[manifest["cmt_identity_conflict"], "strain_id"].nunique()
            ),
            "additional_cross_directory_component_members": int(
                len(
                    excluded
                    - set(
                        manifest.loc[
                            manifest["cmt_identity_conflict"], "strain_id"
                        ].astype(str)
                    )
                )
            ),
            "excluded_spectra": int(
                manifest.loc[manifest["strain_id"].astype(str).isin(excluded), "spectrum_id"].nunique()
            ),
            "excluded_primary_known_strains": int(
                manifest.loc[
                    manifest["primary_known"].astype(bool)
                    & manifest["strain_id"].astype(str).isin(excluded),
                    "strain_id",
                ].nunique()
            ),
            "excluded_primary_known_spectra": int(
                manifest.loc[
                    manifest["primary_known"].astype(bool)
                    & manifest["strain_id"].astype(str).isin(excluded),
                    "spectrum_id",
                ].nunique()
            ),
            "excluded_ood_strains": int(
                manifest.loc[
                    manifest["primary_ood"].astype(bool)
                    & manifest["strain_id"].astype(str).isin(excluded),
                    "strain_id",
                ].nunique()
            ),
            "excluded_ood_spectra": int(
                manifest.loc[
                    manifest["primary_ood"].astype(bool)
                    & manifest["strain_id"].astype(str).isin(excluded),
                    "spectrum_id",
                ].nunique()
            ),
        },
    }
    if not split_validation["validation_pass"].astype(bool).all():
        base_manifest["status"] = "failed_cmt_split_isolation"
        base_manifest["failure"] = (
            "Known CMT identity remained across grouped roles after conservative exclusion; "
            "no clean-cohort performance result was inferred."
        )
        base_manifest["outputs"] = {
            key: _file_record(path, project_root)
            for key, path in output_paths.items()
            if key not in {"run_manifest"} and path.is_file()
        }
        atomic_json(output_paths["run_manifest"], base_manifest)
        raise RuntimeError(str(base_manifest["failure"]))

    predictions = pd.read_parquet(
        paths["predictions"],
        columns=[
            "spectrum_id",
            "strain_id",
            "species",
            "role",
            "predicted_species",
            "design",
            "model",
            "seed",
            "fold",
        ],
    )
    strain_predictions = pd.read_parquet(paths["strain_predictions"])
    leakage, deltas, leakage_summary, cohort_validation = clean_leakage_metrics(
        predictions, strain_predictions, excluded
    )
    leakage.to_csv(output_paths["leakage_metrics"], index=False)
    deltas.to_csv(output_paths["leakage_deltas"], index=False)
    leakage_summary.to_csv(output_paths["leakage_summary"], index=False)
    cohort_validation.to_csv(output_paths["cohort_validation"], index=False)

    decisions = pd.read_parquet(paths["decisions"])
    label_metrics, label_summary = open_set_label_sensitivity(decisions, excluded)
    label_metrics.to_csv(output_paths["open_set_label_metrics"], index=False)
    label_summary.to_csv(output_paths["open_set_label_summary"], index=False)

    summary = {
        "analysis": "CMT identity component exclusion sensitivity",
        "status": "complete",
        "component_counts": base_manifest["component_counts"],
        "split_validation_pass": bool(
            split_validation["validation_pass"].astype(bool).all()
        ),
        "cohort_validation_pass": bool(
            cohort_validation["validation_pass"].astype(bool).all()
        ),
        "clean_known_cohort": {
            "n_spectra": int(leakage["n_spectra"].iloc[0]),
            "n_strains": int(leakage["n_strains"].iloc[0]),
            "n_species": int(leakage["n_species"].iloc[0]),
        },
        "leakage_delta_summary": leakage_summary.to_dict(orient="records"),
        "interpretation": (
            "Descriptive post-result CMT grouping sensitivity conditional on frozen "
            "ExtraTrees fits and v2 reporting decisions."
        ),
    }
    atomic_json(output_paths["summary"], summary)

    base_manifest["status"] = "complete"
    base_manifest["validation"] = {
        "all_grouped_split_cells_clean": True,
        "random_grouped_remaining_spectrum_sets_identical": bool(
            cohort_validation["spectrum_sets_identical"].astype(bool).all()
        ),
        "random_grouped_remaining_strain_sets_identical": bool(
            cohort_validation["strain_sets_identical"].astype(bool).all()
        ),
        "random_grouped_truth_labels_identical": bool(
            cohort_validation["truth_labels_identical"].astype(bool).all()
        ),
        "canonical_label_types_complete": True,
    }
    base_manifest["outputs"] = {
        key: _file_record(path, project_root)
        for key, path in output_paths.items()
        if key != "run_manifest" and path.is_file()
    }
    module_path = Path(__file__).resolve()
    base_manifest["source_files"] = {
        "module": _file_record(module_path, project_root),
    }
    atomic_json(output_paths["run_manifest"], base_manifest)
    return base_manifest
