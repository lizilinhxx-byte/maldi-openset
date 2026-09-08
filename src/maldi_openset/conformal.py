from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def probability_to_logits(probability: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=float), epsilon, 1.0)
    return np.log(probability)


@dataclass
class TemperatureScaler:
    temperature: float = 1.0

    def fit(
        self,
        logits: np.ndarray,
        y_index: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "TemperatureScaler":
        logits = np.asarray(logits, dtype=float)
        y_index = np.asarray(y_index, dtype=int)
        if logits.ndim != 2 or len(logits) != len(y_index):
            raise ValueError("invalid logits or labels")
        if sample_weight is None:
            sample_weight = np.ones(len(y_index), dtype=float)
        else:
            sample_weight = np.asarray(sample_weight, dtype=float)
        if (
            sample_weight.ndim != 1
            or len(sample_weight) != len(y_index)
            or not np.isfinite(sample_weight).all()
            or (sample_weight < 0).any()
            or sample_weight.sum() <= 0
        ):
            raise ValueError("sample_weight must be finite, nonnegative, and match labels")

        def objective(log_temperature: float) -> float:
            temperature = float(np.exp(log_temperature))
            scaled = logits / temperature
            nll = logsumexp(scaled, axis=1) - scaled[np.arange(len(y_index)), y_index]
            return float(np.average(nll, weights=sample_weight))

        result = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded")
        if not result.success:
            raise RuntimeError(f"temperature optimization failed: {result.message}")
        self.temperature = float(np.exp(result.x))
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return softmax(np.asarray(logits, dtype=float) / self.temperature)


def finite_sample_quantile(scores: np.ndarray, coverage: float) -> float:
    scores = np.sort(np.asarray(scores, dtype=float))
    if not len(scores):
        raise ValueError("at least one calibration score is required")
    rank = int(np.ceil((len(scores) + 1) * coverage))
    if rank > len(scores):
        # Nonconformity scores are bounded by one. Returning the upper bound is
        # the conservative finite-sample rule; clipping to the largest observed
        # score would overstate the requested coverage when n is too small.
        return 1.0
    rank = max(rank, 1)
    return float(scores[rank - 1])


@dataclass
class ClassConditionalConformal:
    classes: np.ndarray
    coverage: float = 0.95
    min_class_calibration: int = 20
    global_threshold: float | None = None
    class_thresholds: dict[str, float] | None = None
    class_counts: dict[str, int] | None = None

    def fit(self, probability: np.ndarray, y: np.ndarray) -> "ClassConditionalConformal":
        probability = np.asarray(probability, dtype=float)
        y = np.asarray(y).astype(str)
        lookup = {str(label): idx for idx, label in enumerate(self.classes)}
        valid = np.array([label in lookup for label in y])
        if not valid.all():
            raise ValueError("calibration labels are absent from model classes")
        indices = np.array([lookup[label] for label in y], dtype=int)
        scores = 1.0 - probability[np.arange(len(y)), indices]
        self.global_threshold = finite_sample_quantile(scores, self.coverage)
        self.class_thresholds = {}
        self.class_counts = {}
        for label in self.classes.astype(str):
            selected = scores[y == label]
            self.class_counts[label] = int(len(selected))
            if len(selected) >= self.min_class_calibration:
                self.class_thresholds[label] = finite_sample_quantile(selected, self.coverage)
            else:
                self.class_thresholds[label] = self.global_threshold
        return self

    def prediction_sets(self, probability: np.ndarray) -> list[list[str]]:
        if self.class_thresholds is None:
            raise RuntimeError("conformal calibrator is not fitted")
        probability = np.asarray(probability, dtype=float)
        thresholds = np.array([self.class_thresholds[str(x)] for x in self.classes])
        included = (1.0 - probability) <= thresholds[None, :]
        return [self.classes[row].astype(str).tolist() for row in included]


def known_acceptance_threshold(
    probability: np.ndarray,
    strain_ids: np.ndarray,
    target: float = 0.95,
) -> float:
    """Choose a threshold using one median max-probability value per calibration strain."""
    probability = np.asarray(probability, dtype=float)
    strain_ids = np.asarray(strain_ids).astype(str)
    confidence = probability.max(axis=1)
    strain_confidence = []
    for strain in np.unique(strain_ids):
        strain_confidence.append(float(np.median(confidence[strain_ids == strain])))
    return float(np.quantile(strain_confidence, 1.0 - target, method="lower"))


def aggregate_probabilities_by_group(
    probability: np.ndarray, classes: np.ndarray, class_to_group: dict[str, str]
) -> tuple[np.ndarray, np.ndarray]:
    groups = np.array(sorted({class_to_group[str(label)] for label in classes}), dtype=object)
    group_lookup = {label: idx for idx, label in enumerate(groups)}
    output = np.zeros((len(probability), len(groups)), dtype=float)
    for class_index, label in enumerate(classes.astype(str)):
        output[:, group_lookup[class_to_group[label]]] += probability[:, class_index]
    return output, groups
