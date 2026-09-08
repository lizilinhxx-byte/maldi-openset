from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
try:
    from sklearnex import patch_sklearn

    patch_sklearn(["sklearn.svm.SVC"], verbose=False)
except (ImportError, RuntimeError, ValueError):
    pass
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC, SVC
from sklearn.utils.validation import check_is_fitted

from .conformal import probability_to_logits


class CosineCentroidClassifier(ClassifierMixin, BaseEstimator):
    """Nearest class centroid classifier with cosine similarity."""

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y).astype(str)
        self.classes_ = np.unique(y)
        sample_weight = np.ones(len(y), dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        centroids = []
        for label in self.classes_:
            subset = X[y == label]
            weights = sample_weight[y == label]
            if sparse.issparse(subset):
                centroid = np.asarray(subset.multiply(weights[:, None]).sum(axis=0)).ravel() / weights.sum()
            else:
                centroid = np.average(np.asarray(subset), axis=0, weights=weights)
            norm = np.linalg.norm(centroid)
            centroids.append(centroid / norm if norm else centroid)
        self.centroids_ = np.asarray(centroids, dtype=np.float32)
        return self

    def decision_function(self, X):
        check_is_fitted(self, ["classes_", "centroids_"])
        if sparse.issparse(X):
            norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
            similarity = X @ self.centroids_.T
            similarity = np.asarray(similarity)
        else:
            X = np.asarray(X)
            norms = np.linalg.norm(X, axis=1)
            similarity = X @ self.centroids_.T
        norms[norms == 0] = 1.0
        return similarity / norms[:, None]

    def predict_proba(self, X):
        scores = self.decision_function(X)
        scores -= scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]


class XGBoostStringClassifier(ClassifierMixin, BaseEstimator):
    """XGBoost adapter that preserves string class labels for shared evaluation code."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 0,
        n_jobs: int = -1,
        device: str = "cuda",
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.device = device

    def fit(self, X, y, sample_weight=None):
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("xgboost is required for this model") from exc
        self.encoder_ = LabelEncoder().fit(np.asarray(y).astype(str))
        self.classes_ = self.encoder_.classes_
        self.model_ = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            tree_method="hist",
            num_class=len(self.classes_),
            device=self.device,
        )
        self.model_.fit(
            X,
            self.encoder_.transform(np.asarray(y).astype(str)),
            sample_weight=sample_weight,
        )
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ["model_", "encoder_"])
        return np.asarray(self.model_.predict_proba(X), dtype=float)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def model_spec(name: str, seed: int, n_jobs: int = -1) -> tuple[BaseEstimator, dict[str, list]]:
    if name == "cosine_centroid":
        return CosineCentroidClassifier(), {}
    if name == "linear_svm":
        return (
            LinearSVC(C=1.0, class_weight=None, dual="auto", random_state=seed, max_iter=10000),
            {},
        )
    if name == "rbf_svm":
        return (
            SVC(C=1.0, gamma="scale", kernel="rbf", class_weight=None, random_state=seed),
            {},
        )
    if name == "extra_trees":
        return (
            ExtraTreesClassifier(
                n_estimators=300,
                class_weight=None,
                max_features="sqrt",
                min_samples_leaf=1,
                random_state=seed,
                n_jobs=n_jobs,
            ),
            {},
        )
    if name == "xgboost":
        return (
            XGBoostStringClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                n_jobs=n_jobs,
            ),
            {},
        )
    raise KeyError(f"unknown non-CNN model: {name}")


def fit_sklearn_model(
    name: str,
    X,
    y: np.ndarray,
    groups: np.ndarray,
    sample_weight: np.ndarray | None,
    seed: int,
    n_jobs: int = -1,
) -> tuple[BaseEstimator, dict[str, Any]]:
    # Parallelize across hyperparameter/fold cells. Tree estimators inside the
    # search stay single-threaded to avoid n_jobs x n_jobs oversubscription.
    search_jobs = 1 if n_jobs == 1 else n_jobs
    estimator, grid = model_spec(name, seed, n_jobs=1 if grid_capable_name(name) else n_jobs)
    final_template = clone(estimator)
    # Tree count is fixed, not tuned. Screen structural parameters with smaller
    # deterministic ensembles, then refit the winner at the production size.
    if grid and name == "extra_trees":
        estimator.set_params(n_estimators=200)
    elif grid and name == "xgboost":
        estimator.set_params(n_estimators=100)
    y = np.asarray(y).astype(str)
    groups = np.asarray(groups).astype(str)
    if not grid:
        return estimator.fit(X, y, sample_weight=sample_weight), {"tuned": False, "best_params": {}}
    per_class_groups = [len(np.unique(groups[y == label])) for label in np.unique(y)]
    n_splits = min(3, min(per_class_groups))
    if n_splits < 2:
        fitted = estimator.fit(X, y, sample_weight=sample_weight)
        return fitted, {"tuned": False, "best_params": {}, "reason": "insufficient groups"}
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    search = GridSearchCV(
        estimator,
        grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=search_jobs,
        refit=False,
        error_score="raise",
    )
    search.fit(X, y, groups=groups, sample_weight=sample_weight)
    best = clone(final_template).set_params(**search.best_params_)
    if "n_jobs" in best.get_params(deep=False):
        best.set_params(n_jobs=n_jobs)
    best.fit(X, y, sample_weight=sample_weight)
    return best, {
        "tuned": True,
        "best_params": search.best_params_,
        "best_cv_macro_f1": float(search.best_score_),
        "n_splits": n_splits,
    }


def grid_capable_name(name: str) -> bool:
    return name in {"extra_trees", "xgboost"}


def raw_logits(model: BaseEstimator, X) -> np.ndarray:
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=float)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        return scores
    if hasattr(model, "predict_proba"):
        return probability_to_logits(np.asarray(model.predict_proba(X), dtype=float))
    raise TypeError(f"model {type(model).__name__} exposes neither scores nor probabilities")


def _align_logits(logits: np.ndarray, local_classes: np.ndarray, classes: np.ndarray) -> np.ndarray:
    output = np.full((len(logits), len(classes)), -50.0, dtype=float)
    lookup = {str(label): idx for idx, label in enumerate(classes.astype(str))}
    for source, label in enumerate(local_classes.astype(str)):
        output[:, lookup[label]] = logits[:, source]
    return output


def cross_validated_logits(
    estimator: BaseEstimator,
    X,
    y: np.ndarray,
    groups: np.ndarray,
    sample_weight: np.ndarray | None,
    seed: int,
    n_splits: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y).astype(str)
    groups = np.asarray(groups).astype(str)
    classes = np.unique(y)
    minimum = min(len(np.unique(groups[y == label])) for label in classes)
    n_splits = min(n_splits, minimum)
    if n_splits < 2:
        fitted = clone(estimator).fit(X, y, sample_weight=sample_weight)
        return raw_logits(fitted, X), classes
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    output = np.empty((len(y), len(classes)), dtype=float)
    for train, test in splitter.split(X, y, groups):
        fold_weight = None if sample_weight is None else np.asarray(sample_weight)[train]
        fold_model = clone(estimator).fit(X[train], y[train], sample_weight=fold_weight)
        fold_logits = raw_logits(fold_model, X[test])
        output[test] = _align_logits(fold_logits, fold_model.classes_, classes)
    return output, classes


class _CNNNetwork:
    @staticmethod
    def build(n_classes: int):
        import torch.nn as nn

        return nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # Preserve coarse absolute m/z position. Global pooling to one bin
            # makes a mass spectrum nearly translation invariant and cannot
            # distinguish peaks by location.
            nn.AdaptiveAvgPool1d(64),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128 * 64, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )


@dataclass
class CNN1DClassifier:
    seed: int
    learning_rate: float = 3e-4
    batch_size: int = 64
    max_epochs: int = 100
    patience: int = 10
    device: str | None = None

    def _set_seed(self) -> None:
        import torch

        seed = int(self.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

    @staticmethod
    def _dense_rows(X, indices: np.ndarray) -> np.ndarray:
        part = X[indices]
        dense = part.toarray().astype(np.float32) if sparse.issparse(part) else np.asarray(part, np.float32)
        # TIC-normalized 11k-bin intensities are O(1e-4). This fixed monotone
        # scaling avoids vanishing first-layer activations without fitting any
        # statistic on validation/test data.
        return np.log1p(dense * dense.shape[1]).astype(np.float32)

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None):
        try:
            import torch
            import torch.nn.functional as functional
        except ImportError as exc:
            raise RuntimeError("install the 'cnn' optional dependency to train cnn1d") from exc
        self._set_seed()
        self.classes_ = np.unique(np.asarray(y).astype(str))
        lookup = {label: idx for idx, label in enumerate(self.classes_)}
        y_idx = np.array([lookup[str(label)] for label in y], dtype=np.int64)
        sample_weight = np.ones(len(y_idx), dtype=np.float32) if sample_weight is None else np.asarray(sample_weight, dtype=np.float32)
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.network_ = _CNNNetwork.build(len(self.classes_)).to(self.device)
        optimizer = torch.optim.AdamW(self.network_.parameters(), lr=self.learning_rate)
        rng = np.random.default_rng(self.seed)
        best_state = None
        best_loss = math.inf
        stalled = 0
        for _epoch in range(self.max_epochs):
            self.network_.train()
            order = rng.permutation(len(y_idx))
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                xb = torch.from_numpy(self._dense_rows(X, idx)[:, None, :]).to(self.device)
                yb = torch.from_numpy(y_idx[idx]).to(self.device)
                wb = torch.from_numpy(sample_weight[idx]).to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = (functional.cross_entropy(self.network_(xb), yb, reduction="none") * wb).sum() / wb.sum()
                loss.backward()
                optimizer.step()
            if X_val is None or y_val is None:
                continue
            logits = self.decision_function(X_val)
            yv = np.array([lookup[str(label)] for label in y_val], dtype=int)
            shifted = logits - logits.max(axis=1, keepdims=True)
            probability = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
            validation_loss = float(-np.log(np.clip(probability[np.arange(len(yv)), yv], 1e-12, 1)).mean())
            if validation_loss < best_loss - 1e-5:
                best_loss = validation_loss
                best_state = copy.deepcopy(self.network_.state_dict())
                stalled = 0
            else:
                stalled += 1
                if stalled >= self.patience:
                    break
        if best_state is not None:
            self.network_.load_state_dict(best_state)
        self.validation_loss_ = best_loss if np.isfinite(best_loss) else None
        return self

    def decision_function(self, X):
        import torch

        if not hasattr(self, "network_"):
            raise RuntimeError("CNN model is not fitted")
        self.network_.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, X.shape[0], self.batch_size):
                idx = np.arange(start, min(start + self.batch_size, X.shape[0]))
                xb = torch.from_numpy(self._dense_rows(X, idx)[:, None, :]).to(self.device)
                chunks.append(self.network_(xb).cpu().numpy())
        return np.vstack(chunks)

    def predict_proba(self, X):
        logits = self.decision_function(X)
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]
