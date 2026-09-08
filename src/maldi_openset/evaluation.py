from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score, f1_score

from .config import StudyConfig
from .metrics import (
    classification_metrics,
    clustered_rate_bootstrap,
    equal_strain_weights,
    ood_metrics,
    paired_macro_f1_bootstrap,
    risk_coverage_curve,
)
from .util import atomic_write_json, sha256_file, utc_now


def _strain_balanced_rate(frame: pd.DataFrame, column: str) -> float:
    if frame.empty:
        return float("nan")
    return float(frame.groupby("strain_id")[column].mean().mean())


def _species_macro_strain_rate(frame: pd.DataFrame, column: str) -> float:
    if frame.empty:
        return float("nan")
    by_strain = frame.groupby(["species", "strain_id"])[column].mean()
    return float(by_strain.groupby(level="species").mean().mean())


def _require_model_instances(rows: pd.DataFrame, config: StudyConfig, label: str) -> None:
    expected = {(int(seed), fold) for seed in config.seeds for fold in range(config.outer_folds)}
    observed = set(
        rows[["seed", "fold"]]
        .drop_duplicates()
        .astype({"seed": int, "fold": int})
        .itertuples(index=False, name=None)
    )
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"{label} is incomplete: missing model instances={missing}, unexpected={extra}"
        )


def discover_runs(output_root: str | Path) -> list[Path]:
    output_root = Path(output_root)
    return sorted(
        path.parent
        for path in (output_root / "runs").glob("*/run_manifest.json")
        if json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"
    )


def load_run(run_dir: str | Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    run_dir = Path(run_dir)
    predictions_path = run_dir / "predictions.parquet"
    probabilities_path = run_dir / "probabilities.npy"
    classes_path = run_dir / "classes.json"
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    for field, path in (
        ("predictions_sha256", predictions_path),
        ("probabilities_sha256", probabilities_path),
        ("classes_sha256", classes_path),
    ):
        if not path.is_file() or manifest.get(field) != sha256_file(path):
            raise ValueError(f"locked artifact hash mismatch for {field} in {run_dir.name}")
    predictions = pd.read_parquet(predictions_path)
    probability = np.load(probabilities_path, allow_pickle=False)
    classes = np.array(json.loads(classes_path.read_text(encoding="utf-8")), dtype=object)
    if len(predictions) != len(probability):
        raise ValueError(f"prediction/probability row mismatch in {run_dir.name}")
    if probability.ndim != 2 or probability.shape[1] != len(classes):
        raise ValueError(f"probability/class dimension mismatch in {run_dir.name}")
    if (
        not np.isfinite(probability).all()
        or (probability < 0).any()
        or (probability > 1).any()
        or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-6, rtol=0)
    ):
        raise ValueError(f"invalid probability matrix in {run_dir.name}")
    if "probability_row" not in predictions or not np.array_equal(
        predictions["probability_row"].to_numpy(), np.arange(len(predictions))
    ):
        raise ValueError(f"prediction rows are not aligned to probabilities in {run_dir.name}")
    return predictions, probability, classes, manifest


def per_run_metrics(run_dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict] = []
    all_predictions: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        predictions, probability, classes, run_manifest = load_run(run_dir)
        known = predictions["role"] == "test_known"
        known_probability = probability[known.to_numpy()]
        class_lookup = {str(label): idx for idx, label in enumerate(classes.astype(str))}
        y_known = predictions.loc[known, "species"].astype(str).to_numpy()
        y_index = np.array([class_lookup[label] for label in y_known], dtype=int)
        top3 = np.argsort(-known_probability, axis=1, kind="stable")[:, : min(3, len(classes))]
        predictions["correct_species"] = False
        predictions.loc[known, "correct_species"] = (
            predictions.loc[known, "predicted_species"].astype(str).to_numpy() == y_known
        )
        predictions["true_probability"] = np.nan
        predictions.loc[known, "true_probability"] = known_probability[np.arange(len(y_index)), y_index]
        target = np.zeros_like(known_probability)
        target[np.arange(len(y_index)), y_index] = 1.0
        predictions["brier_row"] = np.nan
        predictions.loc[known, "brier_row"] = np.sum((known_probability - target) ** 2, axis=1)
        predictions["top3_correct"] = False
        predictions.loc[known, "top3_correct"] = [label in values for label, values in zip(y_index, top3)]
        metrics = classification_metrics(
            predictions.loc[known, "species"].astype(str).to_numpy(),
            classes,
            known_probability,
            predictions.loc[known, "strain_id"].astype(str).to_numpy(),
        )
        metrics.update(
            {
                "run_id": run_manifest["run_id"],
                "design": run_manifest["design"],
                "model": run_manifest["model"],
                "seed": run_manifest["seed"],
                "fold": run_manifest["fold"],
                "n_test_known": int(known.sum()),
            }
        )
        if (predictions["role"] == "test_ood").any():
            combined = predictions["role"].isin(["test_known", "test_ood"])
            strain_scores = (
                predictions.loc[combined]
                .assign(is_ood=lambda frame: frame["role"].eq("test_ood"))
                .groupby(["strain_id", "is_ood"], as_index=False)["confidence"]
                .mean()
            )
            metrics.update(ood_metrics(strain_scores["is_ood"].to_numpy(), strain_scores["confidence"].to_numpy()))
            ood = predictions["role"] == "test_ood"
            metrics["ood_false_accept_rate"] = _species_macro_strain_rate(
                predictions.loc[ood], "accepted_species"
            )
            for distance in ("near", "far"):
                selected = ood & predictions["ood_distance"].eq(distance)
                metrics[f"{distance}_ood_false_accept_rate"] = (
                    _species_macro_strain_rate(predictions.loc[selected], "accepted_species")
                    if selected.any()
                    else float("nan")
                )
        metrics["heldout_known_acceptance_rate"] = _strain_balanced_rate(
            predictions.loc[known], "accepted_species"
        )
        correct = predictions.loc[known, "predicted_species"].astype(str).to_numpy() == predictions.loc[
            known, "species"
        ].astype(str).to_numpy()
        known_strains = predictions.loc[known, "strain_id"].astype(str).to_numpy()
        _, _, aurc = risk_coverage_curve(
            correct,
            predictions.loc[known, "confidence"].to_numpy(),
            equal_strain_weights(known_strains),
        )
        metrics["aurc"] = aurc
        fallback = known & predictions["reported_level"].eq("genus")
        metrics["genus_fallback_accuracy"] = (
            float(
                (
                    predictions.loc[fallback, "reported_label"].astype(str)
                    == predictions.loc[fallback, "genus"].astype(str)
                ).mean()
            )
            if fallback.any()
            else float("nan")
        )
        metric_rows.append(metrics)
        all_predictions.append(predictions)
    return pd.DataFrame(metric_rows), pd.concat(all_predictions, ignore_index=True)


def seed_level_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for key, frame in predictions.groupby(["design", "model", "seed"], sort=True):
        design, model, seed = key
        known = frame.loc[frame["role"] == "test_known"].copy()
        if known["spectrum_id"].duplicated().any():
            raise ValueError(f"duplicate OOF known prediction for {key}")
        strain_weight = equal_strain_weights(known["strain_id"].astype(str).to_numpy())
        row = {
            "design": design,
            "model": model,
            "seed": int(seed),
            "n_spectra": int(len(known)),
            "n_strains": int(known["strain_id"].nunique()),
            "n_species": int(known["species"].nunique()),
            "macro_f1": float(
                f1_score(known["species"], known["predicted_species"], average="macro", zero_division=0)
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(known["species"], known["predicted_species"])
            ),
            "top3_accuracy": float(np.average(known["top3_correct"].astype(float), weights=strain_weight)),
            "log_loss": float(
                np.average(-np.log(np.clip(known["true_probability"].astype(float), 1e-12, 1)), weights=strain_weight)
            ),
            "brier": float(np.average(known["brier_row"].astype(float), weights=strain_weight)),
            "heldout_known_acceptance_rate": float(
                np.average(known["accepted_species"].astype(float), weights=strain_weight)
            ),
        }
        ood = frame.loc[frame["role"] == "test_ood"].copy()
        if not ood.empty:
            # Each OOD spectrum is scored by five outer models. Average those
            # model-instance predictions before forming strain/species summaries.
            ood_unique = (
                ood.groupby(["spectrum_id", "strain_id", "species", "genus", "ood_distance"], as_index=False)
                .agg(confidence=("confidence", "mean"), accepted_species=("accepted_species", "mean"))
            )
            known_scores = known.groupby("strain_id", as_index=False)["confidence"].mean().assign(is_ood=False)
            ood_scores = ood_unique.groupby("strain_id", as_index=False)["confidence"].mean().assign(is_ood=True)
            score_frame = pd.concat([known_scores, ood_scores], ignore_index=True)
            row.update(ood_metrics(score_frame["is_ood"].to_numpy(), score_frame["confidence"].to_numpy()))
            row["ood_false_accept_rate"] = _species_macro_strain_rate(ood_unique, "accepted_species")
            for distance in ("near", "far"):
                part = ood_unique.loc[ood_unique["ood_distance"] == distance]
                row[f"{distance}_ood_false_accept_rate"] = _species_macro_strain_rate(
                    part, "accepted_species"
                )
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_primary_endpoint(
    predictions: pd.DataFrame, config: StudyConfig, bootstrap_seed: int = 20260907
) -> dict:
    primary = predictions.loc[
        (predictions["model"] == config.primary_model)
        & (predictions["role"] == "test_known")
    ].copy()
    grouped = primary[primary["design"] == "strain_grouped"]
    random = primary[primary["design"] == "spectrum_random"]
    _require_model_instances(grouped, config, "grouped leakage endpoint")
    _require_model_instances(random, config, "spectrum-random leakage endpoint")
    for seed in config.seeds:
        grouped_ids = set(grouped.loc[grouped["seed"].astype(int) == int(seed), "spectrum_id"].astype(str))
        random_ids = set(random.loc[random["seed"].astype(int) == int(seed), "spectrum_id"].astype(str))
        if grouped_ids != random_ids:
            raise ValueError(f"paired leakage endpoint spectrum sets differ for seed {seed}")
    paired = random.merge(
        grouped,
        on=["seed", "spectrum_id"],
        suffixes=("_random", "_grouped"),
        validate="one_to_one",
    )
    if not (
        paired["species_random"].astype(str).equals(paired["species_grouped"].astype(str))
        and paired["strain_id_random"].astype(str).equals(paired["strain_id_grouped"].astype(str))
    ):
        raise ValueError("paired leakage endpoint labels or strain identifiers disagree")
    from sklearn.metrics import f1_score

    def effect(frame: pd.DataFrame) -> float:
        seed_effects = []
        for _, part in frame.groupby("seed"):
            seed_effects.append(
                f1_score(
                    part["species_random"], part["predicted_species_random"], average="macro", zero_division=0
                )
                - f1_score(
                    part["species_random"], part["predicted_species_grouped"], average="macro", zero_division=0
                )
            )
        return float(np.mean(seed_effects))

    estimate = effect(paired)
    rng = np.random.default_rng(bootstrap_seed)
    by_species = {
        species: sorted(part["strain_id_random"].unique())
        for species, part in paired.groupby("species_random")
    }
    by_strain = {strain: part for strain, part in paired.groupby("strain_id_random")}
    draws = np.empty(config.bootstrap_replicates, dtype=float)
    for index in range(config.bootstrap_replicates):
        parts = []
        for strains in by_species.values():
            sampled = rng.choice(strains, size=len(strains), replace=True)
            parts.extend(by_strain[strain] for strain in sampled)
        draws[index] = effect(pd.concat(parts, ignore_index=True))
    ci_low, ci_high = np.quantile(draws, [0.025, 0.975])
    p_value = float((1 + np.sum(draws <= 0)) / (len(draws) + 1))
    return {
        "endpoint": "paired_macro_f1_random_minus_strain_grouped",
        "model": config.primary_model,
        "estimate": estimate,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": min(p_value, 1.0),
        "n_spectrum_seed_pairs": int(len(paired)),
        "n_strains": int(paired["strain_id_random"].nunique()),
    }


def _near_ood_endpoint(
    predictions: pd.DataFrame, config: StudyConfig, bootstrap_seed: int = 20260919
) -> dict:
    all_ood = predictions.loc[
        (predictions["model"] == config.primary_model)
        & (predictions["design"] == "strain_grouped")
        & (predictions["role"] == "test_ood")
    ].copy()
    _require_model_instances(all_ood, config, "near-OOD endpoint")
    rows = all_ood.loc[all_ood["ood_distance"] == "near"].copy()
    if rows.empty:
        raise ValueError("near-OOD primary endpoint has no observations")
    def effect(frame: pd.DataFrame) -> float:
        by_instance = []
        for _, instance in frame.groupby(["seed", "fold"]):
            by_strain = instance.groupby(["species", "strain_id"])["accepted_species"].mean()
            by_species = by_strain.groupby(level="species").mean()
            by_instance.append(float(by_species.mean()))
        return float(np.mean(by_instance))

    estimate = effect(rows)
    rng = np.random.default_rng(bootstrap_seed)
    species_groups = {species: part for species, part in rows.groupby("species")}
    species_names = np.array(sorted(species_groups), dtype=object)
    draws = np.empty(config.bootstrap_replicates, dtype=float)
    for index in range(config.bootstrap_replicates):
        sampled_species = rng.choice(species_names, size=len(species_names), replace=True)
        sampled_parts = []
        for draw_index, species in enumerate(sampled_species):
            species_frame = species_groups[str(species)]
            strain_groups = {strain: part for strain, part in species_frame.groupby("strain_id")}
            strain_names = np.array(sorted(strain_groups), dtype=object)
            sampled_strains = rng.choice(strain_names, size=len(strain_names), replace=True)
            for strain_index, strain in enumerate(sampled_strains):
                part = strain_groups[str(strain)].copy()
                # Synthetic identifiers preserve multiplicity when species or strains are sampled twice.
                part["species"] = f"draw-{draw_index}:{species}"
                part["strain_id"] = f"draw-{draw_index}-{strain_index}:{strain}"
                sampled_parts.append(part)
        draws[index] = effect(pd.concat(sampled_parts, ignore_index=True))
    ci_low, ci_high = np.quantile(draws, [0.025, 0.975])
    p_value = float((1 + np.sum(draws <= 0.05)) / (len(draws) + 1))
    known = predictions.loc[
        (predictions["model"] == config.primary_model)
        & (predictions["design"] == "strain_grouped")
        & (predictions["role"] == "test_known")
    ]
    _require_model_instances(known, config, "known-acceptance companion")
    instance_known_acceptance = [
        _strain_balanced_rate(instance, "accepted_species")
        for _, instance in known.groupby(["seed", "fold"])
    ]
    return {
        "endpoint": "near_ood_false_accept_rate_at_95pct_known_target",
        "model": config.primary_model,
        "estimate": estimate,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": min(p_value, 1.0),
        "null_rate": 0.05,
        "heldout_known_acceptance": float(np.mean(instance_known_acceptance)),
        "n_spectra": int(rows["spectrum_id"].nunique()),
        "n_strains": int(rows["strain_id"].nunique()),
        "n_species": int(rows["species"].nunique()),
    }


def _holm_two(primary: list[dict]) -> None:
    p_items = [(idx, row.get("p_value")) for idx, row in enumerate(primary) if row.get("p_value") is not None]
    if not p_items:
        return
    ordered = sorted(p_items, key=lambda item: item[1])
    adjusted = {}
    running = 0.0
    m = len(ordered)
    for rank, (idx, value) in enumerate(ordered):
        candidate = min(1.0, (m - rank) * float(value))
        running = max(running, candidate)
        adjusted[idx] = running
    for idx, value in adjusted.items():
        primary[idx]["holm_adjusted_p"] = value


def build_figures(
    metrics: pd.DataFrame, predictions: pd.DataFrame, figure_dir: Path, config: StudyConfig
) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    paths: list[Path] = []

    primary = predictions.loc[
        (predictions["model"] == config.primary_model)
        & (predictions["design"] == "strain_grouped")
        & (predictions["seed"].astype(int) == int(config.seeds[0]))
    ]
    known = primary.loc[primary["role"] == "test_known"].drop_duplicates("spectrum_id")
    ood = primary.loc[primary["role"] == "test_ood"].drop_duplicates("spectrum_id")
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.axis("off")
    boxes = [
        (0.05, 0.62, 0.25, 0.22, "RKI v4.2\n" f"{len(known)+len(ood):,} spectra"),
        (0.40, 0.68, 0.25, 0.22, "Primary known\n" f"{known['species'].nunique()} species / {known['strain_id'].nunique():,} strains"),
        (0.40, 0.25, 0.25, 0.22, "Primary OOD\n" f"{ood['species'].nunique()} species / {ood['strain_id'].nunique():,} strains"),
        (0.73, 0.68, 0.22, 0.22, "5 x 5 OOF\nrandom vs grouped"),
        (0.73, 0.25, 0.22, 0.22, "Near / far OOD\nselective reporting"),
    ]
    for x, y, w, h, label in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor="#E7F0F7", edgecolor="#275D7A", lw=1.5))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10)
    for start, end in [((0.30, 0.73), (0.40, 0.76)), ((0.30, 0.73), (0.40, 0.36)), ((0.65, 0.79), (0.73, 0.79)), ((0.65, 0.36), (0.73, 0.36))]:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "#275D7A"})
    ax.set_title("Study cohort and evaluation design", fontsize=12, weight="bold")
    fig.tight_layout()
    path = figure_dir / "figure1_study_flow.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    sns.pointplot(data=metrics, x="model", y="macro_f1", hue="design", dodge=0.35, errorbar="sd", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Macro-F1")
    ax.tick_params(axis="x", rotation=30)
    ax.set_title("Spectrum-random versus strain-grouped evaluation")
    fig.tight_layout()
    path = figure_dir / "figure2_split_performance.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths.append(path)

    grouped = predictions.loc[
        (predictions["design"] == "strain_grouped")
        & predictions["role"].isin(["test_known", "test_ood"])
    ].copy()
    if not grouped.empty:
        grouped["evaluation_group"] = np.where(
            grouped["role"].eq("test_known"), "Known strain", grouped["ood_distance"].str.title() + " OOD"
        )
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        sns.boxenplot(data=grouped, x="evaluation_group", y="confidence", hue="model", ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("Maximum calibrated probability")
        ax.set_title("Confidence on known and unseen taxa")
        fig.tight_layout()
        path = figure_dir / "figure3_confidence_distributions.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        paths.append(path)

    confusion = predictions.loc[
        (predictions["model"] == config.primary_model)
        & (predictions["design"] == "strain_grouped")
        & (predictions["role"] == "test_known")
        & (predictions["predicted_species"] != predictions["species"])
    ].copy()
    if not confusion.empty:
        pairs = (
            confusion.groupby(["species", "predicted_species"], as_index=False)
            .size()
            .sort_values("size", ascending=False)
            .head(20)
        )
        pairs["pair"] = pairs["species"] + " -> " + pairs["predicted_species"]
        fig, ax = plt.subplots(figsize=(8.0, 6.2))
        sns.barplot(data=pairs, y="pair", x="size", color="#3C78A8", ax=ax)
        ax.set_xlabel("Misclassified spectra across seeds")
        ax.set_ylabel("")
        ax.set_title("Most frequent strain-disjoint species confusions")
        fig.tight_layout()
        path = figure_dir / "figure5_confusion_pairs.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        paths.append(path)

    if not grouped.empty:
        summary = (
            grouped.groupby(["model", "evaluation_group"], as_index=False)["accepted_species"]
            .mean()
            .rename(columns={"accepted_species": "acceptance_rate"})
        )
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        sns.barplot(data=summary, x="model", y="acceptance_rate", hue="evaluation_group", ax=ax)
        ax.set_ylim(0, 1)
        ax.set_xlabel("")
        ax.set_ylabel("Species-level acceptance rate")
        ax.tick_params(axis="x", rotation=30)
        ax.set_title("Selective reporting after calibration")
        fig.tight_layout()
        path = figure_dir / "figure4_selective_acceptance.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        paths.append(path)

    return sorted(paths)


def evaluate_all(output_root: str | Path, config: StudyConfig) -> dict:
    output_root = Path(output_root)
    run_dirs = discover_runs(output_root)
    if not run_dirs:
        raise FileNotFoundError("no completed model runs found")
    metrics, predictions = per_run_metrics(run_dirs)
    seed_metrics = seed_level_metrics(predictions)
    artifacts = output_root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(artifacts / "metrics.csv", index=False)
    seed_metrics.to_csv(artifacts / "seed_level_metrics.csv", index=False)
    predictions.to_parquet(artifacts / "predictions.parquet", index=False)
    primary = [_paired_primary_endpoint(predictions, config)]
    if (predictions["role"] == "test_ood").any():
        primary.append(_near_ood_endpoint(predictions, config))
    _holm_two(primary)
    pd.DataFrame(primary).to_csv(artifacts / "primary_endpoints.csv", index=False)
    figure_paths = build_figures(seed_metrics, predictions, artifacts / "figures", config)
    summary = {
        "created_at": utc_now(),
        "completed_runs": len(run_dirs),
        "models": sorted(metrics["model"].unique().tolist()),
        "designs": sorted(metrics["design"].unique().tolist()),
        "primary_endpoints": primary,
        "figures": [str(path) for path in figure_paths],
        "metrics_sha256": sha256_file(artifacts / "metrics.csv"),
        "seed_level_metrics_sha256": sha256_file(artifacts / "seed_level_metrics.csv"),
        "predictions_sha256": sha256_file(artifacts / "predictions.parquet"),
    }
    atomic_write_json(artifacts / "metrics.json", summary)
    return summary
