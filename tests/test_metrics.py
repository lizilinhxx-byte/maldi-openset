import numpy as np

from maldi_openset.metrics import (
    classification_metrics,
    equal_strain_weights,
    ood_metrics,
    risk_coverage_curve,
)


def test_metric_suite_on_perfect_predictions():
    classes = np.array(["A", "B"])
    y = np.array(["A", "B", "A", "B"])
    probability = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    metrics = classification_metrics(y, classes, probability)
    assert metrics["macro_f1"] == 1.0
    assert metrics["top3_accuracy"] == 1.0
    ood = ood_metrics(np.array([False, False, True, True]), np.array([0.9, 0.8, 0.2, 0.1]))
    assert ood["ood_auroc"] == 1.0
    coverage, risk, aurc = risk_coverage_curve(np.ones(4, dtype=bool), probability.max(axis=1))
    assert aurc == 0.0


def test_equal_strain_weights_do_not_overweight_technical_replicates():
    weights = equal_strain_weights(np.array(["many", "many", "many", "single"]))
    np.testing.assert_allclose(weights[:3].sum(), weights[3:].sum())


def test_risk_coverage_accepts_confidence_ties_as_a_group():
    coverage, risk, _ = risk_coverage_curve(
        np.array([True, False, True]),
        np.array([0.9, 0.9, 0.2]),
        np.array([0.25, 0.25, 0.5]),
    )
    np.testing.assert_allclose(coverage, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(risk, [0.0, 0.5, 0.25])
