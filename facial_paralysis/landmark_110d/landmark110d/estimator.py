"""Fixed standardized L2 logistic estimator used with the 110D vector."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES


FIXED_C = 0.01
FIXED_THRESHOLD = 0.5
FIXED_SOLVER = "liblinear"
FIXED_MAX_ITER = 2000
FIXED_RANDOM_STATE = 0
_ARTIFACT_FIELDS = {
    "schema_version",
    "feature_names",
    "classifier",
    "standardizer",
}


def _validated_groups(group_ids: Sequence[object]) -> np.ndarray:
    groups = np.asarray(group_ids, dtype=object)
    if groups.ndim != 1 or groups.size == 0:
        raise ValueError("group_ids must be a nonempty one-dimensional array")
    values = groups.tolist()
    if any(
        not isinstance(group, str)
        or not group
        or group != group.strip()
        for group in values
    ):
        raise ValueError("group_ids must contain nonempty, trimmed strings")
    return groups


def equal_group_weights(group_ids: Sequence[object]) -> np.ndarray:
    """Give each training group total weight one across its recordings."""
    groups = _validated_groups(group_ids)
    values = groups.tolist()
    counts = {group: values.count(group) for group in set(values)}
    return np.asarray([1.0 / counts[group] for group in values], dtype=np.float64)


def _matrix(rows: np.ndarray) -> np.ndarray:
    checked = np.asarray(rows, dtype=np.float64)
    if checked.ndim != 2 or checked.shape[1] != 110:
        raise ValueError("features must have shape (N, 110)")
    if checked.shape[0] == 0 or not np.isfinite(checked).all():
        raise ValueError("features must be nonempty and finite")
    return checked


class Landmark110DEstimator:
    """Small research estimator with transparent JSON-safe serialization."""

    def __init__(self) -> None:
        self.c = FIXED_C
        self.threshold = FIXED_THRESHOLD
        self._fitted = False

    def fit(
        self,
        features: np.ndarray,
        labels: Sequence[int],
        group_ids: Sequence[object],
    ) -> "Landmark110DEstimator":
        matrix = _matrix(features)
        target = np.asarray(labels)
        raw_groups = np.asarray(group_ids, dtype=object)
        if target.shape != (matrix.shape[0],) or raw_groups.shape != target.shape:
            raise ValueError("labels and group_ids must align with feature rows")
        groups = _validated_groups(raw_groups)
        if target.dtype.kind not in {"b", "i", "u"} or not np.isin(
            target, (0, 1)
        ).all():
            raise ValueError("labels must contain binary integers")
        if np.unique(target).size != 2:
            raise ValueError("both binary classes are required")
        group_labels: dict[str, int] = {}
        for group, label in zip(groups.tolist(), target.astype(int).tolist()):
            previous = group_labels.setdefault(group, label)
            if previous != label:
                raise ValueError("a group cannot cross binary labels")

        scaler = StandardScaler().fit(matrix)
        standardized = scaler.transform(matrix)
        classifier = LogisticRegression(
            C=FIXED_C,
            penalty="l2",
            solver=FIXED_SOLVER,
            max_iter=FIXED_MAX_ITER,
            random_state=FIXED_RANDOM_STATE,
        )
        classifier.fit(
            standardized,
            target.astype(np.int64, copy=False),
            sample_weight=equal_group_weights(groups),
        )
        self.mean_ = scaler.mean_.astype(np.float64, copy=True)
        self.scale_ = scaler.scale_.astype(np.float64, copy=True)
        self.coef_ = classifier.coef_[0].astype(np.float64, copy=True)
        self.intercept_ = float(classifier.intercept_[0])
        self._fitted = True
        return self

    def _decision_function(self, features: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise ValueError("estimator must be fitted before prediction")
        matrix = _matrix(features)
        standardized = (matrix - self.mean_) / self.scale_
        return standardized @ self.coef_ + self.intercept_

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        logits = self._decision_function(features)
        output = np.empty_like(logits, dtype=np.float64)
        positive = logits >= 0
        output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exponent = np.exp(logits[~positive])
        output[~positive] = exponent / (1.0 + exponent)
        return output

    def predict(self, features: np.ndarray) -> np.ndarray:
        return (self.predict_proba(features) >= self.threshold).astype(np.int64)

    def to_dict(self) -> dict[str, object]:
        if not self._fitted:
            raise ValueError("estimator must be fitted before serialization")
        return {
            "schema_version": "landmark110d_estimator_v1",
            "feature_names": list(FEATURE_NAMES),
            "classifier": {
                "type": "standardized_l2_logistic_regression",
                "c": FIXED_C,
                "solver": FIXED_SOLVER,
                "max_iter": FIXED_MAX_ITER,
                "random_state": FIXED_RANDOM_STATE,
                "threshold": FIXED_THRESHOLD,
                "coef": self.coef_.tolist(),
                "intercept": self.intercept_,
            },
            "standardizer": {
                "mean": self.mean_.tolist(),
                "scale": self.scale_.tolist(),
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "Landmark110DEstimator":
        if not isinstance(payload, Mapping) or set(payload) != _ARTIFACT_FIELDS:
            raise ValueError("estimator artifact fields differ from v1")
        if payload["schema_version"] != "landmark110d_estimator_v1":
            raise ValueError("estimator artifact schema is unsupported")
        if tuple(payload["feature_names"]) != FEATURE_NAMES:
            raise ValueError("estimator feature-name contract has drifted")
        classifier = payload["classifier"]
        standardizer = payload["standardizer"]
        if not isinstance(classifier, Mapping) or not isinstance(
            standardizer, Mapping
        ):
            raise ValueError("estimator artifact components must be objects")
        expected_classifier = {
            "type",
            "c",
            "solver",
            "max_iter",
            "random_state",
            "threshold",
            "coef",
            "intercept",
        }
        if set(classifier) != expected_classifier or set(standardizer) != {
            "mean", "scale"
        }:
            raise ValueError("estimator artifact component fields differ from v1")
        if (
            classifier["type"] != "standardized_l2_logistic_regression"
            or classifier["c"] != FIXED_C
            or classifier["solver"] != FIXED_SOLVER
            or classifier["max_iter"] != FIXED_MAX_ITER
            or classifier["random_state"] != FIXED_RANDOM_STATE
            or classifier["threshold"] != FIXED_THRESHOLD
        ):
            raise ValueError("estimator fixed protocol has drifted")

        estimator = cls()
        estimator.mean_ = np.asarray(standardizer["mean"], dtype=np.float64)
        estimator.scale_ = np.asarray(standardizer["scale"], dtype=np.float64)
        estimator.coef_ = np.asarray(classifier["coef"], dtype=np.float64)
        try:
            estimator.intercept_ = float(classifier["intercept"])
        except (TypeError, ValueError) as exc:
            raise ValueError("estimator intercept must be numeric") from exc
        vectors = (estimator.mean_, estimator.scale_, estimator.coef_)
        if any(vector.shape != (110,) for vector in vectors):
            raise ValueError("estimator vectors must have length 110")
        if not all(np.isfinite(vector).all() for vector in vectors):
            raise ValueError("estimator vectors must be finite")
        if np.any(estimator.scale_ <= 0) or not np.isfinite(estimator.intercept_):
            raise ValueError("estimator standardizer or intercept is invalid")
        estimator._fitted = True
        return estimator


__all__ = [
    "FIXED_C",
    "FIXED_MAX_ITER",
    "FIXED_RANDOM_STATE",
    "FIXED_SOLVER",
    "FIXED_THRESHOLD",
    "Landmark110DEstimator",
    "equal_group_weights",
]
