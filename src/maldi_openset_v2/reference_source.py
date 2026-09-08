"""Reference-library and source/instrument-era transport analyses for v2.

The module is intentionally independent of :mod:`maldi_openset`: v1 is a
frozen historical analysis.  All split construction, template fitting, score
calibration, and result writing implemented here operate on v2-only paths.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


EXPECTED_FEATURE_SHA256 = (
    "7a57e01a7ed0adf6b76bf1403192f7988802e230f94a13424f35ce78bfbb4f45"
)
DEFAULT_SEEDS = (20260907, 20260919, 20261001, 20261013, 20261025)
DEFAULT_SOURCE_HOLDOUTS = ("FLI-RIE-PC032", "MALDIMESS")
SCHEMA_VERSION = "2.0.0"


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.divide(matrix, norms, out=matrix, where=norms > 0)
    return matrix


@dataclass(frozen=True)
class ReferenceLibrary:
    """Fit-role strain-average templates sorted for deterministic ties."""

    templates: np.ndarray
    strain_ids: np.ndarray
    species: np.ndarray
    species_classes: np.ndarray
    species_template_indices: tuple[np.ndarray, ...]

    @classmethod
    def fit(cls, X: np.ndarray, rows: pd.DataFrame) -> "ReferenceLibrary":
        required = {"strain_id", "species"}
        missing = required.difference(rows.columns)
        if missing:
            raise KeyError(f"template rows lack columns: {sorted(missing)}")
        if len(rows) != len(X):
            raise ValueError("feature rows and template metadata differ")
        if rows.empty:
            raise ValueError("cannot fit an empty reference library")

        metadata = rows[["strain_id", "species"]].astype(str).reset_index(drop=True)
        conflicts = metadata.groupby("strain_id", sort=False)["species"].nunique()
        if (conflicts > 1).any():
            bad = conflicts[conflicts > 1].index.tolist()[:5]
            raise ValueError(f"strain IDs map to multiple species: {bad}")

        # Stable lexical ordering makes np.argmax's first-maximum rule explicit.
        strain_table = (
            metadata.drop_duplicates("strain_id")
            .sort_values(["species", "strain_id"], kind="stable")
            .reset_index(drop=True)
        )
        strain_ids = strain_table["strain_id"].to_numpy(dtype=str)
        species = strain_table["species"].to_numpy(dtype=str)
        row_groups = metadata.groupby("strain_id", sort=False).indices
        template_rows = []
        for strain_id in strain_ids:
            idx = np.asarray(row_groups[str(strain_id)], dtype=np.int64)
            # Accumulate in float64, then freeze float32 templates. This avoids
            # replicate-count-dependent float32 summation drift.
            mean = np.asarray(X[idx], dtype=np.float64).mean(axis=0)
            template_rows.append(mean.astype(np.float32))
        templates = _l2_normalize(np.stack(template_rows))
        classes = np.unique(species).astype(str)  # np.unique is lexical.
        by_species = tuple(np.flatnonzero(species == label) for label in classes)
        return cls(templates, strain_ids, species, classes, by_species)

    def score(
        self,
        X: np.ndarray,
        *,
        backend: str = "auto",
        block_size: int = 512,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Score query spectra against strain templates.

        Returns a compact prediction frame and the full species-score matrix.
        Species scores are the maximum cosine similarity over fit-role strain
        templates. Ties resolve lexically first at both strain and species level.
        """

        if block_size < 1:
            raise ValueError("block_size must be positive")
        n_query = int(len(X))
        n_species = int(len(self.species_classes))
        species_scores = np.empty((n_query, n_species), dtype=np.float32)
        winning_template = np.empty((n_query, n_species), dtype=np.int32)

        template_scores = _matrix_cosine(
            X, self.templates, backend=backend, block_size=block_size
        )
        for class_index, indices in enumerate(self.species_template_indices):
            local = template_scores[:, indices]
            # indices preserve lexically sorted strain IDs; first maximum wins.
            local_winner = np.argmax(local, axis=1)
            species_scores[:, class_index] = local[
                np.arange(n_query), local_winner
            ]
            winning_template[:, class_index] = indices[local_winner]
        del template_scores

        predicted_index = np.argmax(species_scores, axis=1)
        row_number = np.arange(n_query)
        predicted = self.species_classes[predicted_index]
        score = species_scores[row_number, predicted_index]
        reference_index = winning_template[row_number, predicted_index]
        winning_strain = self.strain_ids[reference_index]

        if n_species > 1:
            runner_matrix = species_scores.copy()
            runner_matrix[row_number, predicted_index] = -np.inf
            runner_index = np.argmax(runner_matrix, axis=1)
            runner_score = runner_matrix[row_number, runner_index]
            runner_species = self.species_classes[runner_index]
        else:
            runner_score = np.full(n_query, np.nan, dtype=np.float32)
            runner_species = np.full(n_query, "", dtype=str)

        # Stable descending sort because species_classes is already lexical.
        k = min(3, n_species)
        top = np.argsort(-species_scores, axis=1, kind="stable")[:, :k]
        top3 = [
            json.dumps(self.species_classes[index].tolist(), ensure_ascii=False)
            for index in top
        ]
        prediction = pd.DataFrame(
            {
                "predicted_species": predicted,
                "score": score.astype(float),
                "winning_reference_strain_id": winning_strain,
                "runner_up_species": runner_species,
                "runner_up_score": runner_score.astype(float),
                "runner_up_margin": (score - runner_score).astype(float),
                "top3_species": top3,
            }
        )
        return prediction, species_scores


def _matrix_cosine(
    X: np.ndarray,
    templates: np.ndarray,
    *,
    backend: str = "auto",
    block_size: int = 512,
) -> np.ndarray:
    """Blockwise query-to-template cosine scores with optional CUDA."""

    backend = str(backend).lower()
    if backend not in {"auto", "numpy", "torch"}:
        raise ValueError("backend must be auto, numpy, or torch")
    use_torch = False
    torch = None
    if backend in {"auto", "torch"}:
        try:
            import torch as torch_module

            torch = torch_module
            use_torch = bool(torch.cuda.is_available())
        except ImportError:
            use_torch = False
        if backend == "torch" and not use_torch:
            raise RuntimeError("torch CUDA backend requested but unavailable")

    templates = np.ascontiguousarray(templates, dtype=np.float32)
    output = np.empty((len(X), len(templates)), dtype=np.float32)
    if use_torch:
        assert torch is not None
        torch.use_deterministic_algorithms(True, warn_only=True)
        device = torch.device("cuda")
        template_tensor = torch.from_numpy(templates).to(device=device)
        template_tensor = template_tensor.transpose(0, 1).contiguous()
        for start in range(0, len(X), block_size):
            stop = min(start + block_size, len(X))
            block = _l2_normalize(np.array(X[start:stop], dtype=np.float32, copy=True))
            with torch.no_grad():
                result = torch.from_numpy(block).to(device=device) @ template_tensor
            output[start:stop] = result.cpu().numpy()
        del template_tensor
        torch.cuda.empty_cache()
        return output

    template_transpose = templates.T
    for start in range(0, len(X), block_size):
        stop = min(start + block_size, len(X))
        block = _l2_normalize(np.array(X[start:stop], dtype=np.float32, copy=True))
        output[start:stop] = block @ template_transpose
    return output


def known_calibration_threshold(
    scores: Sequence[float],
    strain_ids: Sequence[str],
    target: float = 0.95,
) -> tuple[float, int]:
    """Lower-tail threshold based only on strain-median known calibration scores.

    The largest empirical lower-tail order statistic retaining at least the
    requested fraction is used. Repeated technical spectra therefore cannot
    increase the effective calibration sample size.
    """

    scores = np.asarray(scores, dtype=float)
    strain_ids = np.asarray(strain_ids, dtype=str)
    if len(scores) != len(strain_ids) or not len(scores):
        raise ValueError("nonempty, aligned scores and strain_ids are required")
    if not 0 < target <= 1:
        raise ValueError("target must lie in (0, 1]")
    if not np.isfinite(scores).all():
        raise ValueError("calibration scores must be finite")
    frame = pd.DataFrame({"score": scores, "strain_id": strain_ids})
    strain_scores = (
        frame.groupby("strain_id", sort=True)["score"].median().sort_values().to_numpy()
    )
    lower_count = int(math.floor((1.0 - target) * len(strain_scores) + 1e-12))
    lower_count = min(max(lower_count, 0), len(strain_scores) - 1)
    return float(strain_scores[lower_count]), int(len(strain_scores))


def build_source_cohorts(
    manifest: pd.DataFrame,
    holdouts: Iterable[str] = DEFAULT_SOURCE_HOLDOUTS,
    *,
    min_nonheld_strains: int = 5,
    min_held_strains: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Freeze whole-strain source/instrument-era holdout cohorts.

    A strain is held out if *any* of its spectra carries the target acquisition
    label; every spectrum from that strain is then quarantined into the test
    role. This is deliberate for the 300 strains spanning multiple labels.
    """

    required = {"spectrum_id", "strain_id", "species", "instrum"}
    missing = required.difference(manifest.columns)
    if missing:
        raise KeyError(f"manifest lacks source-cohort columns: {sorted(missing)}")
    if min_nonheld_strains < 1 or min_held_strains < 1:
        raise ValueError("minimum strain counts must be positive")

    all_rows: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    base = manifest.copy()
    base["strain_id"] = base["strain_id"].astype(str)
    base["species"] = base["species"].astype(str)
    base["instrum"] = base["instrum"].astype(str)
    strain_species = base[["strain_id", "species"]].drop_duplicates()
    if strain_species["strain_id"].duplicated().any():
        raise ValueError("a strain_id maps to multiple species")

    for label in holdouts:
        label = str(label)
        aggregation = (
            base.assign(_held=base["instrum"].eq(label))
            .groupby("strain_id", sort=True)
            .agg(
                species=("species", "first"),
                total_spectra=("spectrum_id", "size"),
                held_label_spectra=("_held", "sum"),
                instrument_label_count=("instrum", "nunique"),
            )
            .reset_index()
        )
        aggregation["is_held_strain"] = aggregation["held_label_spectra"].gt(0)
        aggregation["other_label_spectra"] = (
            aggregation["total_spectra"] - aggregation["held_label_spectra"]
        )
        aggregation["mixed_instrument_strain"] = aggregation[
            "instrument_label_count"
        ].gt(1)
        counts = (
            aggregation.groupby("species", sort=True)["is_held_strain"]
            .agg(held_strains="sum", total_strains="size")
            .reset_index()
        )
        counts["nonheld_strains"] = counts["total_strains"] - counts["held_strains"]
        counts["included_species"] = counts["held_strains"].ge(min_held_strains) & counts[
            "nonheld_strains"
        ].ge(min_nonheld_strains)
        included = set(counts.loc[counts["included_species"], "species"].astype(str))
        cohort = aggregation.loc[aggregation["species"].isin(included)].copy()
        cohort.insert(0, "holdout_label", label)
        cohort["source_role"] = np.where(
            cohort["is_held_strain"], "test_source", "nonheld_pool"
        )
        cohort["min_nonheld_strains"] = int(min_nonheld_strains)
        cohort["min_held_strains"] = int(min_held_strains)
        all_rows.append(cohort)
        for row in counts.itertuples(index=False):
            summaries.append(
                {
                    "holdout_label": label,
                    "species": str(row.species),
                    "held_strains": int(row.held_strains),
                    "nonheld_strains": int(row.nonheld_strains),
                    "total_strains": int(row.total_strains),
                    "included_species": bool(row.included_species),
                }
            )
    return (
        pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(),
        pd.DataFrame(summaries),
    )


def build_source_splits(
    manifest: pd.DataFrame,
    cohorts: pd.DataFrame,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    *,
    calibration_fraction: float = 0.20,
) -> pd.DataFrame:
    """Assign nonheld strains to fit/calibration and all held strains to test."""

    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must lie in (0, 1)")
    needed = {"holdout_label", "strain_id", "species", "is_held_strain"}
    missing = needed.difference(cohorts.columns)
    if missing:
        raise KeyError(f"cohorts lack split columns: {sorted(missing)}")
    spectrum_rows = manifest[["spectrum_id", "strain_id", "species", "genus", "instrum"]].copy()
    spectrum_rows["strain_id"] = spectrum_rows["strain_id"].astype(str)
    parts: list[pd.DataFrame] = []
    for label, cohort in cohorts.groupby("holdout_label", sort=True):
        cohort = cohort.copy()
        for seed in [int(value) for value in seeds]:
            role_by_strain: dict[str, str] = {}
            held = cohort.loc[cohort["is_held_strain"].astype(bool), "strain_id"].astype(str)
            role_by_strain.update({value: "test_source" for value in held})
            pool = cohort.loc[~cohort["is_held_strain"].astype(bool)]
            for species, species_rows in pool.groupby("species", sort=True):
                strains = np.array(sorted(species_rows["strain_id"].astype(str).unique()))
                derived_seed = int.from_bytes(
                    hashlib.sha256(f"{seed}|{label}|{species}".encode()).digest()[:8],
                    "little",
                )
                order = np.random.default_rng(derived_seed).permutation(strains)
                n_calibration = max(1, int(math.floor(len(strains) * calibration_fraction)))
                n_calibration = min(n_calibration, len(strains) - 1)
                calibration = set(order[:n_calibration])
                for strain in strains:
                    role_by_strain[str(strain)] = (
                        "calibration" if strain in calibration else "train"
                    )
            selected = spectrum_rows.loc[
                spectrum_rows["strain_id"].isin(role_by_strain)
            ].copy()
            selected["role"] = selected["strain_id"].map(role_by_strain)
            selected["holdout_label"] = str(label)
            selected["seed"] = int(seed)
            parts.append(selected)
    result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    validate_source_splits(result)
    return result


def validate_source_splits(splits: pd.DataFrame) -> None:
    if splits.empty:
        raise ValueError("source splits are empty")
    for key, frame in splits.groupby(["holdout_label", "seed"], sort=False):
        if frame["spectrum_id"].duplicated().any():
            raise AssertionError(f"duplicate spectrum assignment in {key}")
        per_strain = frame.groupby("strain_id")["role"].nunique()
        if (per_strain != 1).any():
            raise AssertionError(f"whole-strain role violation in {key}")
        roles = set(frame["role"])
        if roles != {"train", "calibration", "test_source"}:
            raise AssertionError(f"incomplete source roles in {key}: {roles}")
        label = str(key[0])
        test = frame.loc[frame["role"].eq("test_source")]
        non_test = frame.loc[~frame["role"].eq("test_source")]
        if not test.groupby("strain_id")["instrum"].apply(lambda x: (x == label).any()).all():
            raise AssertionError(f"test strain without target label in {key}")
        if non_test["instrum"].eq(label).any():
            raise AssertionError(f"target-label spectrum entered fit/calibration in {key}")
        test_strains = set(test["strain_id"].astype(str))
        other_strains = set(non_test["strain_id"].astype(str))
        if test_strains & other_strains:
            raise AssertionError(f"test/source strain leakage in {key}")


def strain_balanced_weights(species: Sequence[str], strain_ids: Sequence[str]) -> np.ndarray:
    """Equal total weight per species and per strain within species."""

    species = np.asarray(species, dtype=str)
    strain_ids = np.asarray(strain_ids, dtype=str)
    if len(species) != len(strain_ids):
        raise ValueError("species and strain_ids must align")
    weights = np.zeros(len(species), dtype=float)
    for label in np.unique(species):
        class_mask = species == label
        strains = np.unique(strain_ids[class_mask])
        for strain in strains:
            mask = class_mask & (strain_ids == strain)
            weights[mask] = 1.0 / (len(strains) * int(mask.sum()))
    if len(weights):
        weights *= len(weights) / weights.sum()
    return weights


def classification_metrics(
    predictions: pd.DataFrame,
    *,
    score_columns: np.ndarray | None = None,
    classes: Sequence[str] | None = None,
) -> dict[str, object]:
    """Return spectrum, equal-strain-weight, and strain-aggregate endpoints."""

    if predictions.empty:
        return {"n_spectra": 0, "n_strains": 0}
    true = predictions["species"].astype(str).to_numpy()
    predicted = predictions["predicted_species"].astype(str).to_numpy()
    strains = predictions["strain_id"].astype(str).to_numpy()
    weights = np.zeros(len(predictions), dtype=float)
    for strain in np.unique(strains):
        mask = strains == strain
        weights[mask] = 1.0 / int(mask.sum())
    labels = np.unique(true)
    result: dict[str, object] = {
        "n_spectra": int(len(predictions)),
        "n_strains": int(len(np.unique(strains))),
        "n_species": int(len(labels)),
        "spectrum_accuracy": float(accuracy_score(true, predicted)),
        "spectrum_macro_f1": float(f1_score(true, predicted, labels=labels, average="macro", zero_division=0)),
        "spectrum_balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "equal_strain_weight_accuracy": float(accuracy_score(true, predicted, sample_weight=weights)),
        "equal_strain_weight_macro_f1": float(
            f1_score(
                true,
                predicted,
                labels=labels,
                average="macro",
                sample_weight=weights,
                zero_division=0,
            )
        ),
        "accepted_count": int(predictions["accepted"].astype(bool).sum()),
        "accepted_rate": float(predictions["accepted"].astype(bool).mean()),
    }
    accepted = predictions["accepted"].astype(bool).to_numpy()
    result["accepted_accuracy"] = (
        float(accuracy_score(true[accepted], predicted[accepted])) if accepted.any() else None
    )

    if score_columns is not None and classes is not None:
        classes = np.asarray(classes, dtype=str)
        if score_columns.shape != (len(predictions), len(classes)):
            raise ValueError("score matrix shape does not align with predictions/classes")
        top = np.argsort(-score_columns, axis=1, kind="stable")[:, : min(3, len(classes))]
        result["spectrum_top3_accuracy"] = float(
            np.mean([label in classes[row] for label, row in zip(true, top)])
        )
        grouped_scores: list[np.ndarray] = []
        grouped_true: list[str] = []
        grouped_predicted: list[str] = []
        for strain in sorted(np.unique(strains)):
            mask = strains == strain
            mean_score = np.asarray(score_columns[mask], dtype=np.float64).mean(axis=0)
            grouped_scores.append(mean_score)
            grouped_true.append(str(true[np.flatnonzero(mask)[0]]))
            grouped_predicted.append(str(classes[int(np.argmax(mean_score))]))
        result["strain_aggregate_accuracy"] = float(
            accuracy_score(grouped_true, grouped_predicted)
        )
        result["strain_aggregate_macro_f1"] = float(
            f1_score(
                grouped_true,
                grouped_predicted,
                labels=np.unique(grouped_true),
                average="macro",
                zero_division=0,
            )
        )
    return result


def fit_extra_trees(
    X: np.ndarray,
    rows: pd.DataFrame,
    *,
    seed: int,
    n_jobs: int = 4,
) -> ExtraTreesClassifier:
    y = rows["species"].astype(str).to_numpy()
    groups = rows["strain_id"].astype(str).to_numpy()
    weights = strain_balanced_weights(y, groups)
    model = ExtraTreesClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=1,
        class_weight=None,
        random_state=int(seed),
        n_jobs=int(n_jobs),
    )
    model.fit(X, y, sample_weight=weights)
    return model


def runtime_record() -> dict[str, object]:
    try:
        import sklearn

        sklearn_version = sklearn.__version__
    except Exception:
        sklearn_version = None
    try:
        import torch

        torch_version = torch.__version__
        cuda = bool(torch.cuda.is_available())
        gpu = torch.cuda.get_device_name(0) if cuda else None
    except Exception:
        torch_version = None
        cuda = False
        gpu = None
    return {
        "created_at": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn_version,
        "torch": torch_version,
        "cuda_available": cuda,
        "gpu": gpu,
    }

