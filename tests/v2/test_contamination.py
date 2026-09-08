from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from maldi_openset_v2.contamination import (
    EXPERIMENTS,
    build_assignments,
    build_cohort,
    build_condition_pair,
    compose_training_rows,
    paired_cluster_bootstrap,
    paired_cell_contrasts,
    sha256_file,
    validate_checkpoint_universe,
)


def _manifest(spectra_per_strain: int = 5, strains_per_species: int = 10) -> pd.DataFrame:
    rows = []
    for species_index, species in enumerate(("Species alpha", "Species beta")):
        for strain_index in range(strains_per_species):
            strain = f"s{species_index}-{strain_index}"
            for spectrum_index in range(spectra_per_strain):
                rows.append(
                    {
                        "spectrum_id": f"{strain}-p{spectrum_index}",
                        "strain_id": strain,
                        "species": species,
                        "primary_known": True,
                    }
                )
    # The implementation must not depend on input order for selections.
    return pd.DataFrame(rows).sample(frac=1, random_state=17).reset_index(drop=True)


def _setup(experiment: str, seed: int = 11, rotation: int = 0):
    definition = EXPERIMENTS[experiment]
    cohort, inventory = build_cohort(_manifest(), definition)
    assignments, pairs = build_assignments(
        cohort, inventory, definition, seed=seed, rotation=rotation
    )
    return definition, assignments, pairs


def test_cohorts_apply_threshold_and_fixed_cap():
    manifest = _manifest()
    prevalence, prevalence_inventory = build_cohort(manifest, EXPERIMENTS["prevalence"])
    intensity, intensity_inventory = build_cohort(manifest, EXPERIMENTS["intensity"])
    assert len(prevalence_inventory) == 20
    assert len(intensity_inventory) == 20
    assert prevalence.groupby("strain_id").size().eq(2).all()
    assert intensity.groupby("strain_id").size().eq(5).all()


def test_hash_selection_is_invariant_to_manifest_row_order():
    definition = EXPERIMENTS["prevalence"]
    original = _manifest().sort_values("spectrum_id").reset_index(drop=True)
    reversed_rows = original.iloc[::-1].reset_index(drop=True)
    first_cohort, first_inventory = build_cohort(original, definition)
    second_cohort, second_inventory = build_cohort(reversed_rows, definition)
    first, first_pairs = build_assignments(
        first_cohort, first_inventory, definition, seed=23, rotation=2
    )
    second, second_pairs = build_assignments(
        second_cohort, second_inventory, definition, seed=23, rotation=2
    )
    columns = ["spectrum_id", "strain_id", "strain_role", "spectrum_role", "pair_id"]
    pd.testing.assert_frame_equal(
        first[columns].sort_values("spectrum_id").reset_index(drop=True),
        second[columns].sort_values("spectrum_id").reset_index(drop=True),
    )
    pair_columns = ["pair_id", "evaluation_strain_id", "donor_strain_id"]
    pd.testing.assert_frame_equal(first_pairs[pair_columns], second_pairs[pair_columns])


def test_rotations_are_whole_strain_disjoint_and_cover_each_q_block_once():
    definition = EXPERIMENTS["prevalence"]
    cohort, inventory = build_cohort(_manifest(), definition)
    evaluation_sets = []
    for rotation in range(5):
        assignments, pairs = build_assignments(
            cohort, inventory, definition, seed=19, rotation=rotation
        )
        train = set(assignments.loc[assignments.strain_role == "baseline_train", "strain_id"])
        evaluation = set(assignments.loc[assignments.strain_role == "evaluation", "strain_id"])
        donor = set(assignments.loc[assignments.strain_role == "donor", "strain_id"])
        assert not train & evaluation
        assert not train & donor
        assert not evaluation & donor
        assert len(evaluation) == len(donor) == len(pairs) == 4
        evaluation_sets.append(evaluation)
    assert len(set.union(*evaluation_sets)) == 20
    assert sum(len(values) for values in evaluation_sets) == 20


def test_prevalence_zero_nested_exposure_and_queries_never_injected():
    definition, assignments, pairs = _setup("prevalence")
    conditions = {
        level: build_condition_pair(assignments, pairs, definition, level)
        for level in definition.levels
    }
    assert conditions[0.0]["n_injected_spectra_per_arm"] == 0
    exposed_sets = [
        set(conditions[level]["exposed_pairs"]["evaluation_strain_id"])
        for level in definition.levels
    ]
    assert all(left <= right for left, right in zip(exposed_sets, exposed_sets[1:]))
    assert [len(values) for values in exposed_sets] == [0, 1, 2, 3, 4]
    query_ids = set(assignments.loc[assignments.spectrum_role == "evaluation_query", "spectrum_id"])
    for condition in conditions.values():
        assert not query_ids & set(condition["leakage"]["spectrum_id"])
        assert not query_ids & set(condition["matched_control"]["spectrum_id"])


def test_paired_arms_have_equal_counts_and_species_composition_with_donor_quarantine():
    definition, assignments, pairs = _setup("prevalence")
    condition = build_condition_pair(assignments, pairs, definition, 1.0)
    leakage = condition["leakage"]
    control = condition["matched_control"]
    assert len(leakage) == len(control) == len(pairs)
    pd.testing.assert_series_equal(
        leakage.groupby("species").size().sort_index(),
        control.groupby("species").size().sort_index(),
    )
    baseline_strains = set(
        assignments.loc[assignments.strain_role == "baseline_train", "strain_id"]
    )
    donor_strains = set(assignments.loc[assignments.strain_role == "donor", "strain_id"])
    evaluation_strains = set(assignments.loc[assignments.strain_role == "evaluation", "strain_id"])
    assert not donor_strains & baseline_strains
    assert not donor_strains & evaluation_strains
    assert set(control["strain_id"]) == donor_strains
    assert set(leakage["strain_id"]) == evaluation_strains


def test_intensity_candidates_are_nested_at_0_1_2_4():
    definition, assignments, pairs = _setup("intensity")
    conditions = {
        level: build_condition_pair(assignments, pairs, definition, level)
        for level in definition.levels
    }
    assert [conditions[level]["n_injected_spectra_per_arm"] for level in definition.levels] == [
        0,
        4,
        8,
        16,
    ]
    for arm in ("leakage", "matched_control"):
        spectra = [set(conditions[level][arm]["spectrum_id"]) for level in definition.levels]
        assert all(left <= right for left, right in zip(spectra, spectra[1:]))


def test_composed_training_keeps_baseline_prefix_and_original_order():
    definition, assignments, pairs = _setup("intensity")
    condition = build_condition_pair(assignments, pairs, definition, 2)
    baseline = assignments.loc[assignments.spectrum_role == "baseline_train"].sort_values(
        "feature_row", kind="mergesort"
    )["feature_row"].to_numpy()
    rows = compose_training_rows(assignments, condition["leakage"])
    np.testing.assert_array_equal(rows[: len(baseline)], baseline)
    assert len(np.unique(rows)) == len(rows)


def test_paired_contrasts_and_bootstrap_are_reproducible():
    rows = []
    for species, truth, leak, control in (
        ("A", ["A", "A"], ["A", "A"], ["A", "B"]),
        ("B", ["B", "B"], ["B", "A"], ["A", "A"]),
    ):
        for index, (actual, pred_l, pred_c) in enumerate(zip(truth, leak, control)):
            for arm, predicted in (("leakage", pred_l), ("matched_control", pred_c)):
                rows.append(
                    {
                        "experiment": "prevalence",
                        "seed": 1,
                        "rotation": 0,
                        "level": 1.0,
                        "level_token": "p-1p00",
                        "pair_id": f"{species}-{index}",
                        "spectrum_id": f"p-{species}-{index}",
                        "strain_id": f"s-{species}-{index}",
                        "species": actual,
                        "arm": arm,
                        "predicted_species": predicted,
                    }
                )
    predictions = pd.DataFrame(rows)
    contrasts = paired_cell_contrasts(predictions)
    assert len(contrasts) == 1
    assert contrasts.iloc[0]["macro_f1_delta_leakage_minus_control"] > 0
    first, first_intervals = paired_cluster_bootstrap(predictions, replicates=25, seed=7)
    second, second_intervals = paired_cluster_bootstrap(predictions, replicates=25, seed=7)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first_intervals, second_intervals)


def _checkpoint_fixture(tmp_path: Path):
    root = tmp_path / "output"
    assignment_relative = "assignments/prevalence__seed-1__rotation-0/assignment_manifest.json"
    fit_relative = "cells/prevalence/seed-1/rotation-0/p-0p00/baseline/fit_manifest.json"
    assignment_expected = {
        "schema_version": "2.0.0",
        "experiment": "prevalence",
        "seed": 1,
        "rotation": 0,
        "assignment_signature": "assignment-ok",
        "fit_engine_source_sha256": "source-ok",
        "manifest_sha256": "manifest-ok",
        "assignments_path": "assignments/prevalence__seed-1__rotation-0/assignments.parquet",
        "pairings_path": "assignments/prevalence__seed-1__rotation-0/pairings.parquet",
    }
    fit_expected = {
        "schema_version": "2.0.0",
        "experiment": "prevalence",
        "seed": 1,
        "rotation": 0,
        "level": 0.0,
        "level_token": "p-0p00",
        "arm": "baseline",
        "model": "ExtraTrees",
        "n_estimators": 300,
        "assignment_signature": "assignment-ok",
        "fit_signature": "fit-ok",
        "fit_engine_source_sha256": "source-ok",
        "feature_sha256": "feature-ok",
        "predictions_path": "cells/prevalence/seed-1/rotation-0/p-0p00/baseline/predictions.parquet",
        "injections_path": "cells/prevalence/seed-1/rotation-0/p-0p00/baseline/injections.parquet",
    }
    assignment_dir = root / Path(assignment_relative).parent
    assignment_dir.mkdir(parents=True)
    pd.DataFrame({"x": [1]}).to_parquet(assignment_dir / "assignments.parquet", index=False)
    pd.DataFrame({"x": [2]}).to_parquet(assignment_dir / "pairings.parquet", index=False)
    assignment_record = {
        **{key: assignment_expected[key] for key in (
            "schema_version", "experiment", "seed", "rotation", "assignment_signature",
            "fit_engine_source_sha256", "manifest_sha256",
        )},
        "assignments_sha256": sha256_file(assignment_dir / "assignments.parquet"),
        "pairings_sha256": sha256_file(assignment_dir / "pairings.parquet"),
    }
    (assignment_dir / "assignment_manifest.json").write_text(
        json.dumps(assignment_record), encoding="utf-8"
    )

    fit_dir = root / Path(fit_relative).parent
    fit_dir.mkdir(parents=True)
    pd.DataFrame({"prediction": ["A"]}).to_parquet(fit_dir / "predictions.parquet", index=False)
    pd.DataFrame({"injection": pd.Series(dtype=str)}).to_parquet(
        fit_dir / "injections.parquet", index=False
    )
    fit_record = {
        **{key: fit_expected[key] for key in (
            "schema_version", "experiment", "seed", "rotation", "level", "level_token",
            "arm", "model", "n_estimators", "assignment_signature", "fit_signature",
            "fit_engine_source_sha256", "feature_sha256",
        )},
        "predictions_sha256": sha256_file(fit_dir / "predictions.parquet"),
        "injections_sha256": sha256_file(fit_dir / "injections.parquet"),
    }
    (fit_dir / "fit_manifest.json").write_text(json.dumps(fit_record), encoding="utf-8")
    universe = {
        "assignments": {assignment_relative: assignment_expected},
        "fits": {fit_relative: fit_expected},
    }
    return root, universe, fit_dir


def test_checkpoint_universe_rejects_missing_checkpoint(tmp_path):
    root, universe, fit_dir = _checkpoint_fixture(tmp_path)
    for name in ("fit_manifest.json", "predictions.parquet", "injections.parquet"):
        (fit_dir / name).unlink()
    audit = validate_checkpoint_universe(root, universe, require_complete=False)
    assert audit["observed_fit_paths"] == 0
    assert len(audit["missing_fit_paths"]) == 1
    with pytest.raises(RuntimeError, match="universe incomplete"):
        validate_checkpoint_universe(root, universe, require_complete=True)


def test_checkpoint_universe_rejects_extra_checkpoint(tmp_path):
    root, universe, _ = _checkpoint_fixture(tmp_path)
    extra = root / "cells/extra"
    extra.mkdir(parents=True)
    for name in ("fit_manifest.json", "predictions.parquet", "injections.parquet"):
        (extra / name).write_text("extra", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unexpected fit checkpoint"):
        validate_checkpoint_universe(root, universe, require_complete=True)


def test_checkpoint_universe_rejects_wrong_signature(tmp_path):
    root, universe, fit_dir = _checkpoint_fixture(tmp_path)
    path = fit_dir / "fit_manifest.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["fit_signature"] = "wrong"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(RuntimeError, match="fit_signature"):
        validate_checkpoint_universe(root, universe, require_complete=True)


def test_checkpoint_universe_rejects_artifact_hash_corruption(tmp_path):
    root, universe, fit_dir = _checkpoint_fixture(tmp_path)
    with (fit_dir / "predictions.parquet").open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(RuntimeError, match="artifact hash corruption"):
        validate_checkpoint_universe(root, universe, require_complete=True)
