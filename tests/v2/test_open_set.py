from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maldi_openset_v2.open_set import (  # noqa: E402
    build_decision_frame,
    empirical_acceptance_threshold,
    finite_sample_quantile,
    hierarchical_occurrence_bootstrap,
    hierarchical_occurrence_bootstrap_overall_strain,
    is_explicit_binomial_label,
    strain_mean_conformal_sensitivity,
)


def _fixture_frame() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    classes = np.array(["Alpha one", "Alpha two", "Beta one"], dtype=object)
    probability = np.array(
        [
            [0.60, 0.39, 0.01],
            [0.20, 0.70, 0.10],
            [0.20, 0.10, 0.70],
            [0.50, 0.50, 0.00],
        ],
        dtype=np.float32,
    )
    species_sets = [["Alpha two"], ["Alpha two"], [], ["Alpha one"]]
    genus_sets = [["Alpha"], ["Alpha"], ["Alpha", "Beta"], ["Beta"]]
    confidence = [0.60, 0.70, 0.70, 0.50]
    tau = [0.50] * 4
    legacy_accept = [True, True, False, True]
    legacy_level = ["species", "species", "unidentified", "species"]
    legacy_label = ["Alpha two", "Alpha two", "unidentified", "Alpha one"]
    return (
        pd.DataFrame(
            {
                "spectrum_id": [f"s{i}" for i in range(4)], "strain_id": [f"x{i}" for i in range(4)],
                "species": ["Alpha one", "Alpha two", "Beta one", "Alpha one"],
                "genus": ["Alpha", "Alpha", "Beta", "Alpha"], "role": ["test_known"] * 4,
                "analysis_set": ["primary_known"] * 4, "ood_distance": [None] * 4,
                "ood_singleton": [False] * 4, "design": ["strain_grouped"] * 4,
                "model": ["test"] * 4, "seed": [1] * 4, "fold": [0] * 4,
                "predicted_species": ["Alpha one", "Alpha two", "Beta one", "Alpha one"],
                "confidence": confidence, "known_acceptance_threshold": tau,
                "threshold_source": ["saved"] * 4, "conformal_threshold_source": ["saved"] * 4,
                "conformal_species_set": [json.dumps(value) for value in species_sets],
                "conformal_species_set_size": [len(value) for value in species_sets],
                "conformal_genus_set": [json.dumps(value) for value in genus_sets],
                "conformal_genus_set_size": [len(value) for value in genus_sets],
                "accepted_species": legacy_accept, "reported_level": legacy_level, "reported_label": legacy_label,
            }
        ),
        probability,
        classes,
    )


def test_corrected_rule_requires_singleton_argmax_and_includes_threshold_boundary() -> None:
    frame, probability, classes = _fixture_frame()
    result, audit = build_decision_frame(frame, probability, classes, "fixture")
    assert result["legacy_species_accept"].tolist() == [True, True, False, True]
    assert result["corrected_species_accept"].tolist() == [False, True, False, True]
    assert result.loc[0, "corrected_reported_level"] == "genus"
    assert result.loc[0, "corrected_reported_label"] == "Alpha"
    assert result.loc[3, "saved_confidence"] == result.loc[3, "saved_tau"]
    assert result.loc[3, "corrected_species_accept"]
    assert audit["accepted_singleton_not_argmax_n"] == 1


def test_corrected_genus_fallback_requires_aggregate_genus_argmax() -> None:
    frame, probability, classes = _fixture_frame()
    result, _ = build_decision_frame(frame, probability, classes)
    # Row 3's saved genus singleton is Beta, while the summed genus argmax is Alpha.
    # Species is accepted at the exact threshold, so mutate only the saved set to force fallback.
    frame.loc[3, "conformal_species_set"] = "[]"
    frame.loc[3, "conformal_species_set_size"] = 0
    frame.loc[3, "accepted_species"] = False
    frame.loc[3, "reported_level"] = "genus"
    frame.loc[3, "reported_label"] = "Beta"
    corrected, _ = build_decision_frame(frame, probability, classes)
    assert result.loc[0, "float32_aggregate_genus_argmax"] == "Alpha"
    assert corrected.loc[3, "corrected_reported_level"] == "unidentified"
    assert corrected.loc[3, "corrected_reported_label"] == "unidentified"


def test_corrected_acceptance_is_subset_of_legacy() -> None:
    frame, probability, classes = _fixture_frame()
    result, _ = build_decision_frame(frame, probability, classes)
    assert (~result["corrected_species_accept"] | result["legacy_species_accept"]).all()


def test_finite_sample_quantile_small_n_boundaries() -> None:
    scores18 = np.linspace(0.01, 0.90, 18)
    scores19 = np.linspace(0.01, 0.95, 19)
    scores20 = np.linspace(0.01, 0.99, 20)
    assert finite_sample_quantile(scores18, 0.95) == 1.0
    assert finite_sample_quantile(scores19, 0.95) == pytest.approx(scores19[-1])
    assert finite_sample_quantile(scores20, 0.95) == pytest.approx(scores20[-1])


def test_empirical_acceptance_threshold_uses_lower_calibration_quantile() -> None:
    confidence = np.array([0.51, 0.72, 0.81, 0.93, 0.99])
    assert empirical_acceptance_threshold(confidence, 0.80) == pytest.approx(0.51)


def test_explicit_binomial_label_excludes_genus_sp_labels() -> None:
    assert is_explicit_binomial_label("Bacillus anthracis")
    assert not is_explicit_binomial_label("Bacillus sp.")
    assert not is_explicit_binomial_label("Bacillus cereus group")


def test_strain_mean_conformal_calibrates_confidence_gate_independently() -> None:
    classes = np.array(["Alpha one", "Beta one"], dtype=object)
    rows = []
    probabilities = []
    for index in range(20):
        is_alpha = index < 10
        species = "Alpha one" if is_alpha else "Beta one"
        genus = "Alpha" if is_alpha else "Beta"
        true_probability = 0.60 if index == 0 else 0.90
        probabilities.append(
            [true_probability, 1 - true_probability]
            if is_alpha
            else [1 - true_probability, true_probability]
        )
        rows.append(
            {
                "spectrum_id": f"cal-{index}", "strain_id": f"cal-strain-{index}",
                "species": species, "genus": genus, "role": "calibration",
                "analysis_set": "primary_known", "ood_distance": None, "ood_singleton": False,
                "design": "strain_grouped", "model": "fixture", "seed": 1, "fold": 0,
                "known_acceptance_threshold": 0.99,
            }
        )
    rows.append(
        {
            "spectrum_id": "test-0", "strain_id": "test-strain-0", "species": "Alpha one",
            "genus": "Alpha", "role": "test_known", "analysis_set": "primary_known",
            "ood_distance": None, "ood_singleton": False, "design": "strain_grouped",
            "model": "fixture", "seed": 1, "fold": 0, "known_acceptance_threshold": 0.99,
        }
    )
    probabilities.append([0.80, 0.20])
    manifest = {
        "run_id": "fixture", "design": "strain_grouped", "model": "fixture",
        "seed": 1, "fold": 0, "known_acceptance_threshold": 0.99,
    }
    decisions, support = strain_mean_conformal_sensitivity(
        pd.DataFrame(rows), np.asarray(probabilities, dtype=np.float32), classes, manifest
    )
    test = decisions.loc[decisions["role"] == "test_known"].iloc[0]
    assert support["strain_confidence_threshold"] == pytest.approx(0.60)
    assert test["strain_confidence_tau"] == pytest.approx(0.60)
    assert test["v1_saved_tau_reference"] == pytest.approx(0.99)
    assert bool(test["species_accept"])


def _naive_bootstrap(frame: pd.DataFrame, replicates: int, seed: int) -> np.ndarray:
    grouped = frame.groupby(["species", "strain_id", "seed", "fold"], as_index=False, sort=True)["value"].mean()
    instances = list(grouped[["seed", "fold"]].drop_duplicates().itertuples(index=False, name=None))
    species = sorted(grouped["species"].unique())
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        species_occurrences = rng.integers(0, len(species), len(species))
        occurrence_values = []
        for species_index in species_occurrences:
            part = grouped[grouped["species"] == species[species_index]]
            strains = sorted(part["strain_id"].unique())
            sampled = rng.integers(0, len(strains), len(strains))
            values = []
            for instance in instances:
                picked = []
                for strain_index in sampled:
                    row = part[(part["strain_id"] == strains[strain_index]) & (part["seed"] == instance[0]) & (part["fold"] == instance[1])]
                    picked.extend(row["value"].tolist())
                values.append(np.mean(picked) if picked else np.nan)
            occurrence_values.append(values)
        draws.append(np.nanmean(np.asarray(occurrence_values), axis=0).mean())
    return np.asarray(draws)


def test_occurrence_bootstrap_matches_independent_naive_implementation() -> None:
    rows = []
    for species, strains in (("A", ("a1", "a2")), ("B", ("b1", "b2", "b3"))):
        for strain_index, strain in enumerate(strains):
            for fold in (0, 1):
                rows.append({"species": species, "strain_id": strain, "seed": 7, "fold": fold, "value": 10 * (species == "B") + strain_index + fold / 10})
    frame = pd.DataFrame(rows)
    _, actual = hierarchical_occurrence_bootstrap(frame, ["value"], 40, 123)
    expected = _naive_bootstrap(frame, 40, 123)
    np.testing.assert_allclose(actual[:, 0], expected)


def test_duplicate_species_occurrences_receive_independent_strain_draws() -> None:
    frame = pd.DataFrame(
        {
            "species": ["A", "A", "B"], "strain_id": ["a1", "a2", "b1"],
            "seed": [1, 1, 1], "fold": [0, 0, 0], "value": [0.0, 1.0, 10.0],
        }
    )
    _, draws = hierarchical_occurrence_bootstrap(frame, ["value"], 200, 44)
    # Independent redrawing permits intermediate values that a reused A draw cannot produce
    # when A is sampled twice in the same species-level draw.
    assert len(np.unique(np.round(draws[:, 0], 6))) > 6


def test_overall_strain_bootstrap_point_is_not_species_macro() -> None:
    frame = pd.DataFrame(
        {
            "species": ["A", "A", "B"], "strain_id": ["a1", "a2", "b1"],
            "seed": [1, 1, 1], "fold": [0, 0, 0], "value": [0.0, 0.0, 1.0],
        }
    )
    overall, _ = hierarchical_occurrence_bootstrap_overall_strain(
        frame, ["value"], 20, 17
    )
    species_macro, _ = hierarchical_occurrence_bootstrap(frame, ["value"], 20, 17)
    assert overall[0] == pytest.approx(1.0 / 3.0)
    assert species_macro[0] == pytest.approx(0.5)
