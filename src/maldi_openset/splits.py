from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

from .util import atomic_write_json


REQUIRED_COLUMNS = {"spectrum_id", "strain_id", "species", "genus", "analysis_set"}


def _select_grouped_calibration(
    rows: pd.DataFrame, train_indices: np.ndarray, seed: int, fraction: float = 0.2
) -> tuple[set[str], set[str]]:
    rng = np.random.default_rng(seed)
    training = rows.iloc[train_indices]
    calibration: set[str] = set()
    for _, subset in training.drop_duplicates("strain_id").groupby("species", sort=True):
        strains = np.array(sorted(subset["strain_id"].astype(str).unique()))
        rng.shuffle(strains)
        n_cal = max(1, int(math.ceil(len(strains) * fraction)))
        if n_cal >= len(strains):
            n_cal = max(1, len(strains) - 1)
        calibration.update(strains[:n_cal].tolist())
    all_train = set(training["strain_id"].astype(str))
    fitting = all_train.difference(calibration)
    if not fitting or not calibration:
        raise ValueError("failed to create nonempty fitting and calibration strain sets")
    return fitting, calibration


def _grouped_assignments(
    known: pd.DataFrame,
    ood: pd.DataFrame,
    seed: int,
    n_splits: int,
    design: str,
) -> pd.DataFrame:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    output: list[pd.DataFrame] = []
    for fold, (outer_train, outer_test) in enumerate(
        splitter.split(known, y=known["species"], groups=known["strain_id"])
    ):
        fitting, calibration = _select_grouped_calibration(
            known, outer_train, seed=seed + 1009 * (fold + 1)
        )
        train_rows = known[known["strain_id"].astype(str).isin(fitting)].copy()
        cal_rows = known[known["strain_id"].astype(str).isin(calibration)].copy()
        test_rows = known.iloc[outer_test].copy()
        for frame, role in (
            (train_rows, "train"),
            (cal_rows, "calibration"),
            (test_rows, "test_known"),
        ):
            frame = frame[["spectrum_id", "strain_id", "species", "genus"]].copy()
            frame["seed"] = seed
            frame["design"] = design
            frame["fold"] = fold
            frame["role"] = role
            output.append(frame)
        if not ood.empty:
            ood_rows = ood[["spectrum_id", "strain_id", "species", "genus"]].copy()
            ood_rows["seed"] = seed
            ood_rows["design"] = design
            ood_rows["fold"] = fold
            ood_rows["role"] = "test_ood"
            output.append(ood_rows)
    return pd.concat(output, ignore_index=True)


def _random_assignments(known: pd.DataFrame, seed: int, n_splits: int) -> pd.DataFrame:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    output: list[pd.DataFrame] = []
    for fold, (outer_train, outer_test) in enumerate(splitter.split(known, known["species"])):
        train_idx, cal_idx = train_test_split(
            outer_train,
            test_size=0.2,
            random_state=seed + 1009 * (fold + 1),
            stratify=known.iloc[outer_train]["species"],
        )
        for indices, role in (
            (train_idx, "train"),
            (cal_idx, "calibration"),
            (outer_test, "test_known"),
        ):
            frame = known.iloc[indices][["spectrum_id", "strain_id", "species", "genus"]].copy()
            frame["seed"] = seed
            frame["design"] = "spectrum_random"
            frame["fold"] = fold
            frame["role"] = role
            output.append(frame)
    return pd.concat(output, ignore_index=True)


def _sensitivity_assignments(
    known: pd.DataFrame, seed: int, n_splits: int = 3
) -> pd.DataFrame:
    """Closed-set >=3-strain stress test without an outer calibration role."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    output = []
    for fold, (train, test) in enumerate(
        splitter.split(known, y=known["species"], groups=known["strain_id"])
    ):
        for indices, role in ((train, "train"), (test, "test_known")):
            frame = known.iloc[indices][["spectrum_id", "strain_id", "species", "genus"]].copy()
            frame["seed"] = seed
            frame["design"] = "strain_grouped_sensitivity"
            frame["fold"] = fold
            frame["role"] = role
            output.append(frame)
    return pd.concat(output, ignore_index=True)


def build_splits(
    manifest: pd.DataFrame,
    seeds: list[int] | tuple[int, ...],
    n_splits: int = 5,
) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(manifest.columns)
    if missing:
        raise KeyError(f"manifest is missing columns: {sorted(missing)}")
    if manifest["spectrum_id"].duplicated().any():
        raise ValueError("spectrum_id values must be unique")
    known = manifest.loc[manifest["analysis_set"] == "primary_known"].reset_index(drop=True)
    ood = manifest.loc[manifest["analysis_set"] == "ood"].reset_index(drop=True)
    if known["species"].nunique() < 2:
        raise ValueError("at least two known species are required")
    min_strains = known.drop_duplicates(["species", "strain_id"]).groupby("species").size().min()
    if min_strains < n_splits:
        raise ValueError(f"known species has only {min_strains} strains for {n_splits} folds")
    outputs: list[pd.DataFrame] = []
    for seed in seeds:
        outputs.append(_grouped_assignments(known, ood, int(seed), n_splits, "strain_grouped"))
        outputs.append(_random_assignments(known, int(seed), n_splits))
        if "sensitivity_known" in manifest.columns:
            sensitivity_known = manifest.loc[manifest["sensitivity_known"]].reset_index(drop=True)
            sensitivity_folds = int(
                min(
                    n_splits,
                    sensitivity_known.drop_duplicates(["species", "strain_id"])
                    .groupby("species")
                    .size()
                    .min(),
                )
            )
            outputs.append(_sensitivity_assignments(sensitivity_known, int(seed), sensitivity_folds))
    assignments = pd.concat(outputs, ignore_index=True)
    validate_splits(assignments)
    return assignments


def validate_splits(assignments: pd.DataFrame) -> None:
    keys = ["seed", "design", "fold"]
    for key, frame in assignments.groupby(keys, sort=False):
        if frame["spectrum_id"].duplicated().any():
            duplicate = frame.loc[frame["spectrum_id"].duplicated(), "spectrum_id"].iloc[0]
            raise AssertionError(f"duplicate assignment in {key}: {duplicate}")
        if str(key[1]).startswith("strain_grouped"):
            role_groups = {
                role: set(part["strain_id"].astype(str))
                for role, part in frame.groupby("role")
                if role != "test_ood"
            }
            roles = sorted(role_groups)
            for i, left in enumerate(roles):
                for right in roles[i + 1 :]:
                    overlap = role_groups[left].intersection(role_groups[right])
                    if overlap:
                        raise AssertionError(f"strain leakage in {key}: {left}/{right}")
            train_species = set(frame.loc[frame["role"] == "train", "species"])
            calibration_species = set(frame.loc[frame["role"] == "calibration", "species"])
            ood_species = set(frame.loc[frame["role"] == "test_ood", "species"])
            if (train_species | calibration_species).intersection(ood_species):
                raise AssertionError(f"OOD species leaked into training/calibration in {key}")


def write_splits(assignments: pd.DataFrame, output_path: str | Path) -> dict:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_parquet(output_path.with_suffix(".parquet"), index=False)
    compact = {
        "schema_version": "1.0",
        "assignment_count": int(len(assignments)),
        "seeds": sorted(int(x) for x in assignments["seed"].unique()),
        "designs": sorted(str(x) for x in assignments["design"].unique()),
        "folds": sorted(int(x) for x in assignments["fold"].unique()),
        "assignments": assignments.to_dict(orient="records"),
    }
    atomic_write_json(output_path, compact)
    return {k: v for k, v in compact.items() if k != "assignments"}


def read_splits(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    parquet = path.with_suffix(".parquet")
    if parquet.exists():
        return pd.read_parquet(parquet)
    with path.open("r", encoding="utf-8") as handle:
        return pd.DataFrame(json.load(handle)["assignments"])
