import numpy as np
from scipy import sparse
from sklearn.metrics import f1_score

from maldi_openset.models import CosineCentroidClassifier, fit_sklearn_model, raw_logits


def _simple_data(seed=4):
    rng = np.random.default_rng(seed)
    X = np.vstack(
        [rng.normal([2, 0, 0], 0.1, size=(15, 3)), rng.normal([0, 2, 0], 0.1, size=(15, 3))]
    )
    y = np.array(["A"] * 15 + ["B"] * 15)
    groups = np.array([f"A{i // 3}" for i in range(15)] + [f"B{i // 3}" for i in range(15)])
    return sparse.csr_matrix(np.clip(X, 0, None)), y, groups


def test_cosine_centroid_classifier():
    X, y, _ = _simple_data()
    model = CosineCentroidClassifier().fit(X, y)
    assert f1_score(y, model.predict(X), average="macro") > 0.95
    assert raw_logits(model, X).shape == (len(y), 2)


def test_extra_trees_tuning_uses_groups():
    X, y, groups = _simple_data()
    model, info = fit_sklearn_model(
        "extra_trees", X, y, groups, np.ones(len(y)), seed=3, n_jobs=1
    )
    assert set(model.classes_) == {"A", "B"}
    assert not info["tuned"]
    assert model.n_estimators == 300
