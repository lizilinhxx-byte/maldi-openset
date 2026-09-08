from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maldi_openset_v2.cmt_sensitivity import (  # noqa: E402
    build_cmt_components,
    classify_species_label,
    clean_leakage_metrics,
    expected_frozen_run_keys,
    open_set_label_sensitivity,
    validate_common_freeze_ledger,
    validate_grouped_cmt_isolation,
    validate_open_set_dependency,
)


def _manifest() -> pd.DataFrame:
    rows = []
    specification = {
        "s1": (["Alpha|one|A"], False, "Alpha one"),
        "s2": (["Alpha|one|A"], False, "Alpha one"),
        "s3": (["Beta|one|B", "Beta|one|C"], True, "Beta one"),
        # s4 is reached through s3's second identity and must also be excluded.
        "s4": (["Beta|one|C"], False, "Beta one"),
        "s5": (["Gamma|one|D"], False, "Gamma one"),
    }
    ordinal = 0
    for strain, (identities, conflict, species) in specification.items():
        for identity in identities:
            ordinal += 1
            rows.append(
                {
                    "spectrum_id": f"x{ordinal}",
                    "strain_id": strain,
                    "cmt_identity": identity,
                    "cmt_identity_conflict": conflict,
                    "species": species,
                    "genus": species.split()[0],
                    "analysis_set": "primary_known",
                }
            )
    return pd.DataFrame(rows)


def test_components_exclude_conflicts_shared_identities_and_bridge_members() -> None:
    components, exclusions = build_cmt_components(_manifest())
    assert set(exclusions["strain_id"]) == {"s1", "s2", "s3", "s4"}
    assert set(components.loc[~components["conservative_exclude"], "strain_ids"]) == {
        '["s5"]'
    }
    beta = components.loc[components["strain_ids"].str.contains('"s3"')].iloc[0]
    assert beta["n_strain_directories"] == 2
    assert beta["n_cmt_identities"] == 2
    assert beta["has_within_directory_conflict"]
    assert beta["spans_multiple_directories"]


def test_canonical_label_classifier_is_explicit() -> None:
    assert classify_species_label("Bacillus cereus") == "explicit_binomial"
    assert classify_species_label("Bacillus sp.") == "sp_unspecified"
    assert classify_species_label("Bacillus cereus group") == "other"


def test_grouped_split_validator_detects_identity_cross_role() -> None:
    manifest = pd.DataFrame(
        {
            "strain_id": ["a", "b"],
            "cmt_identity": ["Alpha|one|A", "Alpha|one|A"],
        }
    )
    splits = pd.DataFrame(
        {
            "spectrum_id": ["x1", "x2"],
            "strain_id": ["a", "b"],
            "seed": [1, 1],
            "design": ["strain_grouped", "strain_grouped"],
            "fold": [0, 0],
            "role": ["train", "test_known"],
        }
    )
    failed = validate_grouped_cmt_isolation(manifest, splits, set())
    assert failed.loc[0, "train__test_known__cmt_overlap_n"] == 1
    assert not failed.loc[0, "validation_pass"]
    passed = validate_grouped_cmt_isolation(manifest, splits, {"b"})
    assert passed.loc[0, "train__test_known__cmt_overlap_n"] == 0
    assert passed.loc[0, "validation_pass"]


def _prediction_fixtures() -> tuple[pd.DataFrame, pd.DataFrame]:
    spectra_rows = []
    strain_rows = []
    truth = {"a": "Alpha one", "b": "Beta one", "bad": "Beta one"}
    spectrum_ids = {"a": ["a1", "a2"], "b": ["b1"], "bad": ["z1"]}
    for design in ("spectrum_random", "strain_grouped"):
        for strain, species in truth.items():
            predicted = species
            if design == "strain_grouped" and strain == "b":
                predicted = "Alpha one"
            for spectrum in spectrum_ids[strain]:
                spectra_rows.append(
                    {
                        "spectrum_id": spectrum,
                        "strain_id": strain,
                        "species": species,
                        "role": "test_known",
                        "predicted_species": predicted,
                        "design": design,
                        "model": "extra_trees",
                        "seed": 7,
                        "fold": 0,
                    }
                )
            strain_rows.append(
                {
                    "strain_id": strain,
                    "species": species,
                    "predicted_species": predicted,
                    "design": design,
                    "model": "extra_trees",
                    "seed": 7,
                    "n_spectra": len(spectrum_ids[strain]),
                }
            )
    return pd.DataFrame(spectra_rows), pd.DataFrame(strain_rows)


def test_clean_leakage_uses_identical_cohort_and_all_three_estimands() -> None:
    spectra, strains = _prediction_fixtures()
    metrics, deltas, summary, validation = clean_leakage_metrics(
        spectra, strains, {"bad"}
    )
    assert validation["validation_pass"].all()
    assert set(metrics["n_spectra"]) == {3}
    assert set(metrics["n_strains"]) == {2}
    assert set(deltas["metric"]) == {
        "pooled_spectrum_macro_f1",
        "equal_strain_weight_macro_f1",
        "strain_mean_probability_macro_f1",
    }
    assert (deltas["random_minus_grouped"] > 0).all()
    assert len(summary) == 3


def test_clean_leakage_fails_if_design_cohorts_are_not_identical() -> None:
    spectra, strains = _prediction_fixtures()
    remove = (
        spectra["design"].eq("strain_grouped")
        & spectra["spectrum_id"].eq("b1")
    )
    with pytest.raises(ValueError, match="cohorts differ"):
        clean_leakage_metrics(spectra.loc[~remove], strains, {"bad"})


def test_open_set_label_sensitivity_separates_oof_known_from_foldwise_ood() -> None:
    base = {
        "design": "strain_grouped",
        "model": "extra_trees",
        "confidence_only_accept": True,
        "saved_singleton": True,
        "legacy_species_accept": True,
        "corrected_species_accept": True,
        "corrected_genus_report": False,
        "corrected_unidentified": False,
        "corrected_any_report_correct": True,
        "corrected_species_report_correct": True,
        "saved_species_set_contains_true": True,
        "conformal_species_set_size": 1,
    }
    rows = [
        dict(base, spectrum_id="k1", strain_id="k", species="Alpha one", role="test_known", ood_distance=None, seed=1, fold=0),
        dict(base, spectrum_id="k2", strain_id="ks", species="Alpha sp.", role="test_known", ood_distance=None, seed=1, fold=1),
        dict(base, spectrum_id="u1", strain_id="u", species="Beta sp.", role="test_ood", ood_distance="near", seed=1, fold=0),
        dict(base, spectrum_id="u1", strain_id="u", species="Beta sp.", role="test_ood", ood_distance="near", seed=1, fold=1),
    ]
    metrics, summary = open_set_label_sensitivity(pd.DataFrame(rows), set())
    known = metrics[metrics["selector"].eq("known")]
    ood = metrics[metrics["selector"].eq("all_ood")]
    assert len(known) == 2  # two label types, one seed-pooled OOF instance
    assert len(ood) == 2  # one sp. row for each fitted fold
    assert set(metrics["label_type"]) == {"explicit_binomial", "sp_unspecified"}
    assert not summary.empty


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, content: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {"bytes": len(content), "sha256": _sha(path)}


def _provenance_fixture(root: Path) -> tuple[Path, Path]:
    direct = {
        "data/processed/production/manifest.parquet": b"manifest",
        "data/processed/splits.parquet": b"splits",
        "output/production/artifacts/predictions.parquet": b"predictions",
    }
    files = []
    for relative, content in direct.items():
        record = _write(root / relative, content)
        files.append({"path": relative, **record})
    feature_relative = "data/processed/production/features.npy"
    feature = _write(root / feature_relative, b"feature-matrix")

    expected = sorted(expected_frozen_run_keys())
    run_records = []
    expected_records = []
    for design, model, seed, fold in expected:
        run_id = f"{design}__{model}__seed-{seed}__fold-{fold}"
        expected_records.append(
            {"design": design, "model": model, "seed": seed, "fold": fold}
        )
        artifact_hashes = {
            "predictions_sha256": "1" * 64,
            "probabilities_sha256": "2" * 64,
            "classes_sha256": "3" * 64,
        }
        run_manifest = {
            "run_id": run_id,
            "design": design,
            "model": model,
            "seed": seed,
            "fold": fold,
            "status": "complete",
            "git_commit": "2293ce52a07e24e507e48814dcd3303bf512c0f3",
            "feature_sha256": feature["sha256"],
            "split_sha256": "4" * 64,
            **artifact_hashes,
        }
        manifest_relative = f"output/production/runs/{run_id}/run_manifest.json"
        manifest_path = root / manifest_relative
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(run_manifest), encoding="utf-8")
        run_records.append(
            {
                "run_id": run_id,
                "design": design,
                "model": model,
                "seed": seed,
                "fold": fold,
                "manifest_path": manifest_relative,
                "manifest_sha256": _sha(manifest_path),
                "split_sha256": "4" * 64,
                **artifact_hashes,
            }
        )
    ledger = {
        "schema_version": "2.0.0",
        "legacy_analysis_commit": "2293ce52a07e24e507e48814dcd3303bf512c0f3",
        "v1_write_protected_by_contract": True,
        "files": files,
        "feature_matrix_path": feature_relative,
        "feature_matrix_bytes": feature["bytes"],
        "feature_sha256": feature["sha256"],
        "expected_run_keys": expected_records,
        "run_count": len(run_records),
        "runs": run_records,
    }
    ledger_path = root / "output/v2/input_manifest.json"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    ledger_sha = _sha(ledger_path)

    open_dir = root / "output/v2/open_set"
    decisions = _write(open_dir / "decisions.parquet", b"decisions")
    strain = _write(
        open_dir / "strain_level_leakage_predictions.parquet", b"strain predictions"
    )
    open_input = {
        "schema_version": "2.0.0",
        "analysis_commit": "2293ce52a07e24e507e48814dcd3303bf512c0f3",
        "n_locked_runs": 200,
        "frozen_input_manifest": {
            "path": "output/v2/input_manifest.json",
            "sha256": ledger_sha,
            "declared_run_count": 210,
        },
    }
    open_input_path = open_dir / "input_manifest.json"
    open_input_path.write_text(json.dumps(open_input), encoding="utf-8")
    open_run = {
        "analysis_commit": "2293ce52a07e24e507e48814dcd3303bf512c0f3",
        "frozen_input_manifest_sha256": ledger_sha,
        "inputs_manifest_sha256": _sha(open_input_path),
        "outputs": {
            "decisions": {
                "path": "output/v2/open_set/decisions.parquet",
                **decisions,
            },
            "strain_level_leakage_predictions": {
                "path": "output/v2/open_set/strain_level_leakage_predictions.parquet",
                **strain,
            },
        },
    }
    (open_dir / "run_manifest.json").write_text(json.dumps(open_run), encoding="utf-8")
    return ledger_path, open_dir


def test_wrong_frozen_file_hash_is_rejected(tmp_path: Path) -> None:
    _provenance_fixture(tmp_path)
    (tmp_path / "output/production/artifacts/predictions.parquet").write_bytes(
        b"tampered predictions"
    )
    with pytest.raises(RuntimeError, match="provenance file identity differs"):
        validate_common_freeze_ledger(tmp_path)


def test_wrong_open_set_output_hash_is_rejected(tmp_path: Path) -> None:
    _, open_dir = _provenance_fixture(tmp_path)
    common = validate_common_freeze_ledger(tmp_path)
    (open_dir / "decisions.parquet").write_bytes(b"tampered decisions")
    with pytest.raises(RuntimeError, match="provenance file identity differs"):
        validate_open_set_dependency(tmp_path, common)


def test_open_set_must_bind_same_common_ledger(tmp_path: Path) -> None:
    _, open_dir = _provenance_fixture(tmp_path)
    common = validate_common_freeze_ledger(tmp_path)
    run_path = open_dir / "run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["frozen_input_manifest_sha256"] = "0" * 64
    run_path.write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(RuntimeError, match="common freeze binding differs"):
        validate_open_set_dependency(tmp_path, common)
