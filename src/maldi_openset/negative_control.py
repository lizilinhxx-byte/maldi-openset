from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from .config import StudyConfig
from .features import load_feature_bundle
from .models import model_spec
from .training import strain_balanced_weights
from .util import atomic_write_json, sha256_file, utc_now


CONTROL_MODELS = ("cosine_centroid", "linear_svm", "extra_trees")


def run_label_permutation_control(
    feature_dir: str | Path,
    output_root: str | Path,
    config: StudyConfig,
    permutations: int = 100,
    seed: int = 20260907,
) -> dict:
    X, manifest, feature_meta = load_feature_bundle(feature_dir)
    rows = manifest.loc[manifest["primary_known"]].copy().reset_index()
    strains = rows[["strain_id", "species"]].drop_duplicates("strain_id").sort_values("strain_id")
    original_labels = strains["species"].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    results: list[dict] = []
    for permutation in range(permutations):
        permuted = rng.permutation(original_labels)
        label_map = dict(zip(strains["strain_id"].astype(str), permuted))
        y = rows["strain_id"].astype(str).map(label_map).to_numpy()
        groups = rows["strain_id"].astype(str).to_numpy()
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + permutation)
        for model_name in CONTROL_MODELS:
            estimator, _ = model_spec(model_name, seed + permutation, n_jobs=-1)
            if model_name == "extra_trees":
                estimator.set_params(n_estimators=200, max_features="sqrt", min_samples_leaf=1)
            predicted = np.empty(len(rows), dtype=object)
            for train, test in splitter.split(rows, y, groups):
                y_train = y[train]
                weight = strain_balanced_weights(y_train, groups[train])
                model = clone(estimator).fit(X[rows.iloc[train]["index"].to_numpy()], y_train, sample_weight=weight)
                predicted[test] = model.predict(X[rows.iloc[test]["index"].to_numpy()])
            results.append(
                {
                    "permutation": permutation,
                    "seed": seed,
                    "model": model_name,
                    "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
                }
            )
    destination = Path(output_root) / "negative_control"
    destination.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(results)
    frame.to_csv(destination / "label_permutation_metrics.csv", index=False)
    envelope = (
        frame.groupby("model")["macro_f1"]
        .agg(
            mean="mean",
            median="median",
            lower=lambda values: values.quantile(0.025),
            upper=lambda values: values.quantile(0.975),
            maximum="max",
        )
        .reset_index()
    )
    envelope.to_csv(destination / "chance_envelope.csv", index=False)
    summary = {
        "created_at": utc_now(),
        "permutations": permutations,
        "seed": seed,
        "models": list(CONTROL_MODELS),
        "extra_trees_control_estimators": 200,
        "feature_sha256": feature_meta.get("features_sha256"),
        "metrics_sha256": sha256_file(destination / "label_permutation_metrics.csv"),
        "envelope": envelope.to_dict(orient="records"),
    }
    atomic_write_json(destination / "negative_control_manifest.json", summary)
    return summary

