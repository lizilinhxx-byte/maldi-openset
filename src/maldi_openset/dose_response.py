from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score

from .config import StudyConfig
from .features import load_feature_bundle
from .splits import read_splits
from .training import strain_balanced_weights
from .util import atomic_write_json, sha256_file, utc_now


DOSES = (0.0, 0.25, 0.5, 0.75, 1.0)


def _hash_order(seed: int, fold: int, spectrum_ids: Iterable[str]) -> list[str]:
    return sorted(
        [str(value) for value in spectrum_ids],
        key=lambda value: hashlib.sha256(f"{seed}|{fold}|{value}".encode()).hexdigest(),
    )


def run_dose_response(
    feature_dir: str | Path,
    splits_path: str | Path,
    output_root: str | Path,
    config: StudyConfig,
    seeds: Iterable[int] | None = None,
    folds: Iterable[int] | None = None,
) -> dict:
    matrix, manifest, feature_meta = load_feature_bundle(feature_dir)
    assignments = read_splits(splits_path)
    index_lookup = pd.Series(np.arange(len(manifest)), index=manifest["spectrum_id"].astype(str))
    seeds = [int(x) for x in (seeds or config.seeds)]
    folds = [int(x) for x in (folds or range(config.outer_folds))]
    prediction_rows: list[dict] = []
    metric_rows: list[dict] = []
    for seed in seeds:
        for fold in folds:
            selected = assignments.loc[
                (assignments["design"] == "strain_grouped")
                & (assignments["seed"].astype(int) == seed)
                & (assignments["fold"].astype(int) == fold)
            ]
            train_ids = selected.loc[selected["role"] == "train", "spectrum_id"].astype(str).tolist()
            test_ids = selected.loc[selected["role"] == "test_known", "spectrum_id"].astype(str).tolist()
            train_idx = index_lookup.loc[train_ids].to_numpy(dtype=int)
            test_frame = manifest.iloc[index_lookup.loc[test_ids].to_numpy(dtype=int)].copy()

            evaluation_ids: list[str] = []
            candidates: dict[str, list[str]] = {}
            for strain, part in test_frame.groupby("strain_id"):
                ordered = _hash_order(seed, fold, part["spectrum_id"])
                if len(ordered) < 2:
                    continue
                evaluation_ids.append(ordered[0])
                candidates[str(strain)] = ordered[1:]
            evaluation_idx = index_lookup.loc[evaluation_ids].to_numpy(dtype=int)
            evaluation = manifest.iloc[evaluation_idx]
            base_path = (
                Path(output_root)
                / "runs"
                / f"strain_grouped__extra_trees__seed-{seed}__fold-{fold}"
                / "model.joblib"
            )
            if not base_path.exists():
                raise FileNotFoundError(f"base ExtraTrees model missing: {base_path}")
            base_model = joblib.load(base_path)
            for dose in DOSES:
                injected_ids: list[str] = []
                for values in candidates.values():
                    count = int(np.ceil(len(values) * dose)) if dose > 0 else 0
                    injected_ids.extend(values[:count])
                combined_ids = train_ids + injected_ids
                combined_idx = index_lookup.loc[combined_ids].to_numpy(dtype=int)
                y_train = manifest.iloc[combined_idx]["species"].astype(str).to_numpy()
                groups = manifest.iloc[combined_idx]["strain_id"].astype(str).to_numpy()
                weight = strain_balanced_weights(y_train, groups)
                model = clone(base_model).fit(matrix[combined_idx], y_train, sample_weight=weight)
                predicted = model.predict(matrix[evaluation_idx]).astype(str)
                true = evaluation["species"].astype(str).to_numpy()
                metric_rows.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "dose": dose,
                        "n_injected_spectra": len(injected_ids),
                        "n_evaluation_strains": len(evaluation),
                        "macro_f1": float(f1_score(true, predicted, average="macro", zero_division=0)),
                        "accuracy": float(accuracy_score(true, predicted)),
                    }
                )
                for row, label in zip(evaluation.itertuples(index=False), predicted):
                    prediction_rows.append(
                        {
                            "seed": seed,
                            "fold": fold,
                            "dose": dose,
                            "spectrum_id": row.spectrum_id,
                            "strain_id": row.strain_id,
                            "species": row.species,
                            "predicted_species": label,
                            "correct": label == str(row.species),
                        }
                    )
    destination = Path(output_root) / "dose_response"
    destination.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    metrics.to_csv(destination / "dose_response_metrics.csv", index=False)
    predictions.to_parquet(destination / "dose_response_predictions.parquet", index=False)
    summary = {
        "created_at": utc_now(),
        "doses": list(DOSES),
        "cells": int(len(metrics)),
        "feature_sha256": feature_meta.get("features_sha256"),
        "metrics_sha256": sha256_file(destination / "dose_response_metrics.csv"),
        "predictions_sha256": sha256_file(destination / "dose_response_predictions.parquet"),
    }
    atomic_write_json(destination / "dose_response_manifest.json", summary)
    return summary

