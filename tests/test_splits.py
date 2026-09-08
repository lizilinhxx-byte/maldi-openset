from maldi_openset.demo import make_demo_bundle
from maldi_openset.features import load_feature_bundle
from maldi_openset.splits import build_splits, validate_splits
import pytest


def test_grouped_splits_are_strain_disjoint_and_reproducible(tmp_path):
    make_demo_bundle(tmp_path / "bundle", 7)
    _, manifest, _ = load_feature_bundle(tmp_path / "bundle")
    first = build_splits(manifest, [11], 5)
    second = build_splits(manifest, [11], 5)
    validate_splits(first)
    assert first.equals(second)
    frame = first[(first["design"] == "strain_grouped") & (first["fold"] == 0)]
    train = set(frame.loc[frame["role"] == "train", "strain_id"])
    test = set(frame.loc[frame["role"] == "test_known", "strain_id"])
    assert not train.intersection(test)
    assert set(frame.loc[frame["role"] == "test_ood", "species"]).isdisjoint(
        set(frame.loc[frame["role"] == "train", "species"])
    )


def test_validation_rejects_ood_species_in_calibration(tmp_path):
    make_demo_bundle(tmp_path / "bundle", 7)
    _, manifest, _ = load_feature_bundle(tmp_path / "bundle")
    assignments = build_splits(manifest, [11], 5)
    frame = assignments[
        (assignments["design"] == "strain_grouped")
        & (assignments["seed"] == 11)
        & (assignments["fold"] == 0)
    ]
    ood_species = frame.loc[frame["role"] == "test_ood", "species"].iloc[0]
    calibration_index = frame.loc[frame["role"] == "calibration"].index[0]
    assignments.loc[calibration_index, "species"] = ood_species
    with pytest.raises(AssertionError, match="training/calibration"):
        validate_splits(assignments)
