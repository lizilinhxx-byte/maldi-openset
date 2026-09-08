from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
    roc_curve,
)


def multiclass_brier(
    y_index: np.ndarray, probability: np.ndarray, sample_weight: np.ndarray | None = None
) -> float:
    target = np.zeros_like(probability, dtype=float)
    target[np.arange(len(y_index)), y_index] = 1.0
    loss = np.sum((probability - target) ** 2, axis=1)
    return float(np.average(loss, weights=sample_weight))


def equal_frequency_ece(
    y_index: np.ndarray,
    probability: np.ndarray,
    bins: int = 15,
    sample_weight: np.ndarray | None = None,
) -> float:
    confidence = probability.max(axis=1)
    correct = probability.argmax(axis=1) == y_index
    sample_weight = np.ones(len(y_index), dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    order = np.argsort(confidence, kind="stable")
    chunks = np.array_split(order, min(bins, len(order)))
    total_weight = sample_weight.sum()
    result = 0.0
    for chunk in chunks:
        if not len(chunk):
            continue
        weights = sample_weight[chunk]
        result += weights.sum() / total_weight * abs(
            float(np.average(correct[chunk], weights=weights))
            - float(np.average(confidence[chunk], weights=weights))
        )
    return float(result)


def strain_class_weights(y: np.ndarray, strain_ids: np.ndarray) -> np.ndarray:
    """Deprecated alias retained for callers; metrics weight every strain equally."""
    return equal_strain_weights(strain_ids)


def equal_strain_weights(strain_ids: np.ndarray) -> np.ndarray:
    """Give each strain equal total weight and split it among its spectra."""
    strain_ids = np.asarray(strain_ids).astype(str)
    if not len(strain_ids):
        raise ValueError("at least one strain is required")
    strains, counts = np.unique(strain_ids, return_counts=True)
    count_lookup = dict(zip(strains, counts))
    weights = np.array([1.0 / count_lookup[strain] for strain in strain_ids], dtype=float)
    return weights / weights.sum()


def top_k_accuracy(y_index: np.ndarray, probability: np.ndarray, k: int = 3) -> float:
    k = min(k, probability.shape[1])
    # Stable sorting implements the prespecified fitted-class-order tie break.
    top = np.argsort(-probability, axis=1, kind="stable")[:, :k]
    return float(np.mean([label in row for label, row in zip(y_index, top)]))


def classification_metrics(
    y: np.ndarray,
    classes: np.ndarray,
    probability: np.ndarray,
    strain_ids: np.ndarray | None = None,
) -> dict[str, float]:
    y = np.asarray(y).astype(str)
    classes = np.asarray(classes).astype(str)
    lookup = {label: idx for idx, label in enumerate(classes)}
    y_index = np.array([lookup[label] for label in y], dtype=int)
    predicted_index = probability.argmax(axis=1)
    predicted = classes[predicted_index]
    weights = None if strain_ids is None else equal_strain_weights(strain_ids)
    return {
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "top3_accuracy": top_k_accuracy(y_index, probability, 3),
        "log_loss": float(
            log_loss(y_index, probability, labels=np.arange(len(classes)), sample_weight=weights)
        ),
        "brier": multiclass_brier(y_index, probability, weights),
        "ece_15_equal_frequency": equal_frequency_ece(y_index, probability, 15, weights),
    }


def fpr_at_tpr(y_ood: np.ndarray, ood_score: np.ndarray, target_tpr: float = 0.95) -> float:
    fpr, tpr, _ = roc_curve(y_ood.astype(int), ood_score)
    eligible = np.where(tpr >= target_tpr)[0]
    return float(fpr[eligible[0]]) if len(eligible) else float("nan")


def ood_metrics(y_ood: np.ndarray, confidence: np.ndarray) -> dict[str, float]:
    y_ood = np.asarray(y_ood, dtype=bool)
    score = 1.0 - np.asarray(confidence, dtype=float)
    if len(np.unique(y_ood)) < 2:
        return {"ood_auroc": float("nan"), "ood_auprc": float("nan"), "fpr_at_95_tpr": float("nan")}
    return {
        "ood_auroc": float(roc_auc_score(y_ood, score)),
        "ood_auprc": float(average_precision_score(y_ood, score)),
        "fpr_at_95_tpr": fpr_at_tpr(y_ood, score, 0.95),
    }


def risk_coverage_curve(
    correct: np.ndarray,
    confidence: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    correct = np.asarray(correct, dtype=bool)
    confidence = np.asarray(confidence, dtype=float)
    if not len(correct) or len(correct) != len(confidence):
        raise ValueError("correct and confidence must be nonempty and aligned")
    if sample_weight is None:
        sample_weight = np.ones(len(correct), dtype=float)
    else:
        sample_weight = np.asarray(sample_weight, dtype=float)
    if (
        sample_weight.ndim != 1
        or len(sample_weight) != len(correct)
        or not np.isfinite(sample_weight).all()
        or (sample_weight < 0).any()
        or sample_weight.sum() <= 0
    ):
        raise ValueError("invalid sample weights")
    order = np.argsort(-confidence, kind="stable")
    ordered_confidence = confidence[order]
    weights = sample_weight[order]
    weighted_errors = (~correct[order]).astype(float) * weights
    cumulative_weight = np.cumsum(weights)
    cumulative_error = np.cumsum(weighted_errors)
    # A threshold accepts an entire confidence tie, never an arbitrary prefix.
    tie_ends = np.r_[ordered_confidence[1:] != ordered_confidence[:-1], True]
    coverage = cumulative_weight[tie_ends] / cumulative_weight[-1]
    risk = cumulative_error[tie_ends] / cumulative_weight[tie_ends]
    coverage = np.r_[0.0, coverage]
    risk = np.r_[0.0, risk]
    aurc = float(np.trapezoid(risk, coverage))
    return coverage, risk, aurc


def percentile_interval(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    alpha = (1.0 - level) / 2.0
    return tuple(float(x) for x in np.quantile(values, [alpha, 1.0 - alpha]))


@dataclass
class BootstrapResult:
    estimate: float
    ci_low: float
    ci_high: float
    replicates: np.ndarray


def clustered_rate_bootstrap(
    value: np.ndarray,
    cluster: np.ndarray,
    replicates: int,
    seed: int,
) -> BootstrapResult:
    value = np.asarray(value, dtype=float)
    cluster = np.asarray(cluster).astype(str)
    unique = np.unique(cluster)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    by_cluster = {key: value[cluster == key] for key in unique}
    for i in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        draws[i] = np.concatenate([by_cluster[key] for key in sampled]).mean()
    low, high = percentile_interval(draws)
    return BootstrapResult(float(value.mean()), low, high, draws)


def paired_macro_f1_bootstrap(
    y: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    cluster: np.ndarray,
    replicates: int,
    seed: int,
) -> BootstrapResult:
    y = np.asarray(y).astype(str)
    prediction_a = np.asarray(prediction_a).astype(str)
    prediction_b = np.asarray(prediction_b).astype(str)
    cluster = np.asarray(cluster).astype(str)
    unique = np.unique(cluster)
    rng = np.random.default_rng(seed)

    def effect(indices: np.ndarray) -> float:
        return float(
            f1_score(y[indices], prediction_a[indices], average="macro", zero_division=0)
            - f1_score(y[indices], prediction_b[indices], average="macro", zero_division=0)
        )

    all_indices = np.arange(len(y))
    estimate = effect(all_indices)
    by_cluster = {key: np.where(cluster == key)[0] for key in unique}
    draws = np.empty(replicates, dtype=float)
    for i in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([by_cluster[key] for key in sampled])
        draws[i] = effect(indices)
    low, high = percentile_interval(draws)
    return BootstrapResult(estimate, low, high, draws)
