import numpy as np

from maldi_openset.conformal import (
    ClassConditionalConformal,
    TemperatureScaler,
    finite_sample_quantile,
)


def test_finite_sample_quantile_is_conservative_when_n_is_too_small():
    assert finite_sample_quantile(np.array([0.1, 0.2, 0.3]), 0.95) == 1.0


def test_temperature_scaling_returns_probabilities():
    logits = np.array([[4.0, 0.0], [0.0, 4.0], [2.0, 1.0], [1.0, 2.0]])
    labels = np.array([0, 1, 0, 1])
    scaler = TemperatureScaler().fit(logits, labels)
    probability = scaler.transform(logits)
    np.testing.assert_allclose(probability.sum(axis=1), 1.0)
    assert scaler.temperature > 0


def test_temperature_scaling_honors_sample_weights():
    logits = np.array([[8.0, 0.0], [8.0, 0.0], [0.0, 8.0]])
    labels = np.array([0, 1, 1])
    unweighted = TemperatureScaler().fit(logits, labels)
    weighted = TemperatureScaler().fit(logits, labels, sample_weight=np.array([1.0, 0.01, 1.0]))
    assert weighted.temperature < unweighted.temperature


def test_conformal_sets_use_global_fallback_for_sparse_classes():
    classes = np.array(["A", "B"])
    probability = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    labels = np.array(["A", "A", "B", "B"])
    fitted = ClassConditionalConformal(classes, coverage=0.8, min_class_calibration=10).fit(
        probability, labels
    )
    assert fitted.class_thresholds["A"] == fitted.global_threshold
    assert len(fitted.prediction_sets(probability)) == 4
