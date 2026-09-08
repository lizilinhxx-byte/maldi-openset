from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from maldi_openset_v2.reference_source import (
    ReferenceLibrary,
    build_source_cohorts,
    build_source_splits,
    classification_metrics,
    known_calibration_threshold,
)


def _runner_module():
    path = Path(__file__).resolve().parents[2] / "scripts/run_v2_reference_source.py"
    spec = importlib.util.spec_from_file_location("run_v2_reference_source_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_library_uses_strain_means_and_stable_ties() -> None:
    # z-strain has two technical replicates whose mean points along x; a-strain
    # is an exact same-species tie and must win lexically.
    X = np.array(
        [
            [2.0, 0.0],
            [4.0, 0.0],
            [1.0, 0.0],
            [0.0, 2.0],
        ],
        dtype=np.float32,
    )
    rows = pd.DataFrame(
        {
            "strain_id": ["z", "z", "a", "b"],
            "species": ["Alpha", "Alpha", "Alpha", "Beta"],
        }
    )
    library = ReferenceLibrary.fit(X, rows)
    assert library.strain_ids.tolist() == ["a", "z", "b"]
    np.testing.assert_allclose(np.linalg.norm(library.templates, axis=1), 1.0)

    prediction, scores = library.score(
        np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        backend="numpy",
        block_size=1,
    )
    assert prediction.loc[0, "predicted_species"] == "Alpha"
    assert prediction.loc[0, "winning_reference_strain_id"] == "a"
    # Alpha and Beta tie for [1, 1]; lexical species tie is stable.
    assert prediction.loc[1, "predicted_species"] == "Alpha"
    assert json.loads(prediction.loc[1, "top3_species"]) == ["Alpha", "Beta"]
    np.testing.assert_allclose(scores[1], [2**-0.5, 2**-0.5], atol=1e-6)


def test_calibration_threshold_uses_one_median_per_strain() -> None:
    # Ten low technical replicates from one strain do not outweigh two strains.
    scores = [0.1] * 10 + [0.8, 0.9]
    strains = ["many"] * 10 + ["one", "two"]
    threshold, n = known_calibration_threshold(scores, strains, target=0.95)
    assert n == 3
    assert threshold == pytest.approx(0.1)

    scores = np.arange(20, dtype=float)
    strains = [f"s{i:02d}" for i in range(20)]
    threshold, n = known_calibration_threshold(scores, strains, target=0.95)
    assert n == 20
    assert threshold == pytest.approx(1.0)  # exactly 19/20 retained


def _synthetic_source_manifest() -> pd.DataFrame:
    records = []
    ordinal = 0
    for species, genus in (("Alpha one", "Alpha"), ("Beta two", "Beta")):
        # Five nonheld strains plus two held strains per species.
        for strain_number in range(7):
            strain = f"{species}|{strain_number}"
            held = strain_number >= 5
            instruments = ["TARGET"] if held else ["BASE"]
            if strain_number == 5:
                instruments.append("BASE")  # mixed label; all rows still test.
            for instrument in instruments:
                records.append(
                    {
                        "spectrum_id": f"q{ordinal}",
                        "strain_id": strain,
                        "species": species,
                        "genus": genus,
                        "instrum": instrument,
                    }
                )
                ordinal += 1
    return pd.DataFrame(records)


def test_source_cohort_and_split_quarantine_whole_strains() -> None:
    manifest = _synthetic_source_manifest()
    cohorts, summary = build_source_cohorts(
        manifest, ["TARGET"], min_nonheld_strains=5, min_held_strains=2
    )
    assert cohorts["species"].nunique() == 2
    included = summary.loc[summary["included_species"]]
    assert set(included["held_strains"]) == {2}
    assert set(included["nonheld_strains"]) == {5}

    split_a = build_source_splits(manifest, cohorts, [7])
    split_b = build_source_splits(manifest, cohorts, [7])
    pd.testing.assert_frame_equal(split_a, split_b)
    assert set(split_a["role"]) == {"train", "calibration", "test_source"}
    assert split_a.groupby("strain_id")["role"].nunique().eq(1).all()
    mixed = split_a.loc[split_a["strain_id"].str.endswith("|5")]
    assert set(mixed["role"]) == {"test_source"}
    assert set(mixed["instrum"]) == {"TARGET", "BASE"}
    assert not split_a.loc[split_a["role"].ne("test_source"), "instrum"].eq("TARGET").any()


def test_source_cohort_excludes_underpowered_species() -> None:
    manifest = _synthetic_source_manifest()
    # Remove one held strain for Beta; it should no longer qualify.
    manifest = manifest.loc[~manifest["strain_id"].eq("Beta two|6")].reset_index(drop=True)
    cohorts, summary = build_source_cohorts(manifest, ["TARGET"])
    assert set(cohorts["species"]) == {"Alpha one"}
    beta = summary.loc[summary["species"].eq("Beta two")].iloc[0]
    assert not bool(beta["included_species"])


def test_metrics_include_spectrum_and_strain_aggregate_estimands() -> None:
    predictions = pd.DataFrame(
        {
            "species": ["A", "A", "B"],
            "predicted_species": ["A", "B", "B"],
            "strain_id": ["a", "a", "b"],
            "accepted": [True, True, False],
        }
    )
    scores = np.array([[0.9, 0.1], [0.2, 0.8], [0.1, 0.9]])
    metrics = classification_metrics(predictions, score_columns=scores, classes=["A", "B"])
    assert metrics["n_spectra"] == 3
    assert metrics["n_strains"] == 2
    assert metrics["accepted_count"] == 2
    assert metrics["strain_aggregate_accuracy"] == pytest.approx(1.0)


def _write_minimal_cell(runner, root: Path, run_id: str, fingerprint: str) -> Path:
    run_dir = root / "output/v2/reference_source/runs" / run_id
    run_dir.mkdir(parents=True)
    predictions = pd.DataFrame(
        {
            "analysis": ["main_reference_library"],
            "design": ["strain_grouped"],
            "model": ["reference_library"],
            "species": ["A"],
            "predicted_species": ["A"],
            "strain_id": ["s"],
            "accepted": [True],
        }
    )
    predictions.to_parquet(run_dir / "predictions.parquet", index=False)
    runner.atomic_json(
        run_dir / "metrics.json",
        {
            "run_id": run_id,
            "analysis": "main_reference_library",
            "design": "strain_grouped",
            "model": "reference_library",
            "seed": 1,
            "fold": 0,
        },
    )
    runner.atomic_json(run_dir / "classes.json", ["A"])
    runner.atomic_json(
        run_dir / "run_manifest.json",
        {
            "status": "complete",
            "run_id": run_id,
            "input_fingerprint": fingerprint,
            "predictions_sha256": runner.sha256_file(run_dir / "predictions.parquet"),
            "metrics_sha256": runner.sha256_file(run_dir / "metrics.json"),
            "classes_sha256": runner.sha256_file(run_dir / "classes.json"),
        },
    )
    return run_dir


def test_consolidation_rejects_unexpected_run_directories(tmp_path: Path) -> None:
    runner = _runner_module()
    _write_minimal_cell(runner, tmp_path, "expected", "fp")
    _write_minimal_cell(runner, tmp_path, "unexpected", "other")
    with pytest.raises(RuntimeError, match="unexpected run directories"):
        runner._consolidate(
            tmp_path,
            {"expected": "fp"},
            {},
            {"current": {}, "frozen": {}},
            expected_count=1,
        )


def test_consolidation_revalidates_every_cell_hash(tmp_path: Path) -> None:
    runner = _runner_module()
    run_dir = _write_minimal_cell(runner, tmp_path, "expected", "fp")
    # Preserve a parseable parquet but make the recorded digest stale.
    predictions = pd.read_parquet(run_dir / "predictions.parquet")
    predictions["tampered"] = True
    predictions.to_parquet(run_dir / "predictions.parquet", index=False)
    with pytest.raises(RuntimeError, match="hash/fingerprint validation failed"):
        runner._consolidate(
            tmp_path,
            {"expected": "fp"},
            {},
            {"current": {}, "frozen": {}},
            expected_count=1,
        )
