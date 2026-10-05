"""Four fixed reference models for the final matching comparison.

Reference fits consume fit labels only. Predict consumes label-free feature
rows. Heavy sklearn dependencies are imported only for an explicit tree fit.
"""
from __future__ import annotations
import hashlib
import io
import math
import pickle
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from .errors import AdmissionError

WIDTHS = {"B22": 22, "L30": 30, "S33": 33, "L42": 42}
HGB_PARAMETERS = {
    "loss": "log_loss", "learning_rate": 0.05, "max_iter": 150,
    "max_leaf_nodes": 7, "max_depth": 3, "min_samples_leaf": 20,
    "l2_regularization": 1.0, "max_bins": 255, "early_stopping": False,
    "max_features": 1.0, "class_weight": None, "warm_start": False,
}
REFERENCE_SCHEMA = "product-matching-reference-1"
LEGACY_REFERENCES = {
    "a5dd9847576f457f1fac4fa9c41542a0dae8094fbd98c5d8e31f205f18c5c59c": "B22",
    "db07492ba10bd1d4546bb638e438b1cbd95416137ff37f5f96ad4335c852c579": "L30",
    "a93997e3bd6014ccfb7c0cf69b8feb839560d9a170599fb39e5a626ec4f1603d": "S33",
    "4a0c6838b691f8e3801c6b5d6c4d1009ee7659129b99180b2c70b71c777ea3c2": "L42",
}


class _CurrentReferenceUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "product_entity_matching" or module.startswith("product_entity_matching."):
            raise AdmissionError("An unpinned old reference is not supported")
        return super().find_class(module, name)


class _LegacyReferenceUnpickler(_CurrentReferenceUnpickler):
    def find_class(self, module, name):
        if (module, name) == ("product_entity_matching.experiments", "LogisticRegressionGD"):
            return LogisticRegressionGD
        if module == "product_entity_matching" or module.startswith("product_entity_matching."):
            raise AdmissionError("Unsupported legacy reference class")
        return super().find_class(module, name)

def _validate_matrix(rows: Sequence[Sequence[float]], width: int | None = None) -> int:
    if not rows:
        if width is None:
            raise AdmissionError("Feature matrix must not be empty")
        return width
    actual_width = len(rows[0])
    if actual_width == 0:
        raise AdmissionError("Feature rows must not be empty")
    if width is not None and actual_width != width:
        raise AdmissionError(f"Expected {width} features, got {actual_width}")
    for row in rows:
        if len(row) != actual_width:
            raise AdmissionError("Feature rows have inconsistent widths")
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise AdmissionError("Feature matrix contains a non-finite or non-numeric value")
    return actual_width


def _validate_labels(labels: Sequence[int], expected_length: int | None = None) -> None:
    if expected_length is not None and len(labels) != expected_length:
        raise AdmissionError("Labels and feature rows are not aligned")
    if not labels:
        raise AdmissionError("Labels must not be empty")
    if any(type(label) is not int or label not in (0, 1) for label in labels):
        raise AdmissionError("Labels must be integer 0/1 values")


def _sigmoid(value: float) -> float:
    if not math.isfinite(value):
        raise AdmissionError("Model score is non-finite")
    if value >= 0:
        z = math.exp(-min(value, 60.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -60.0))
    return z / (1.0 + z)


class LogisticRegressionGD:
    """Deterministic full-batch logistic regression for the CPU baseline."""

    def __init__(self, learning_rate: float = 0.3, l2: float = 0.01, epochs: int = 800):
        if not math.isfinite(float(learning_rate)) or learning_rate <= 0:
            raise AdmissionError("learning_rate must be finite and positive")
        if not math.isfinite(float(l2)) or l2 < 0:
            raise AdmissionError("l2 must be finite and non-negative")
        if type(epochs) is not int or epochs <= 0:
            raise AdmissionError("epochs must be a positive integer")
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.epochs = epochs
        self.weights: list[float] = []
        self.intercept = 0.0
        self.loss_history: list[float] = []

    def loss_and_gradients(
        self, rows: Sequence[Sequence[float]], labels: Sequence[int]
    ) -> tuple[float, list[float], float]:
        """Return average regularized BCE loss and gradients in fixed order."""

        width = _validate_matrix(rows)
        _validate_labels(labels, len(rows))
        if not self.weights:
            self.weights = [0.0] * width
        if len(self.weights) != width:
            raise AdmissionError("Model weights do not match feature width")
        if not math.isfinite(self.intercept) or any(not math.isfinite(value) for value in self.weights):
            raise AdmissionError("Model parameters are non-finite")

        gradient_weights = [0.0] * width
        gradient_intercept = 0.0
        loss = 0.0
        for row, label in zip(rows, labels):
            score = self.intercept + sum(weight * value for weight, value in zip(self.weights, row))
            probability = _sigmoid(score)
            # Stable BCE: log(1 + exp(score)) - label * score.
            loss += max(score, 0.0) - label * score + math.log1p(math.exp(-abs(score)))
            error = probability - label
            gradient_intercept += error
            for index, value in enumerate(row):
                gradient_weights[index] += error * value
        scale = 1.0 / len(rows)
        loss = loss * scale + 0.5 * self.l2 * sum(weight * weight for weight in self.weights)
        gradient_intercept *= scale
        gradient_weights = [value * scale + self.l2 * weight for value, weight in zip(gradient_weights, self.weights)]
        values = [loss, gradient_intercept, *gradient_weights]
        if any(not math.isfinite(value) for value in values):
            raise AdmissionError("Loss or gradient became non-finite")
        return loss, gradient_weights, gradient_intercept

    def fit(
        self,
        rows: Sequence[Sequence[float]],
        labels: Sequence[int],
        deadline: float | None = None,
    ) -> "LogisticRegressionGD":
        width = _validate_matrix(rows)
        _validate_labels(labels, len(rows))
        positives = sum(labels)
        rate = min(max(positives / len(labels), 1e-6), 1.0 - 1e-6)
        self.intercept = math.log(rate / (1.0 - rate))
        self.weights = [0.0] * width
        self.loss_history = []
        for epoch in range(self.epochs):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Model training exceeded the registered CPU budget")
            loss, gradient_weights, gradient_intercept = self.loss_and_gradients(rows, labels)
            self.loss_history.append(loss)
            self.intercept -= self.learning_rate * gradient_intercept
            self.weights = [
                weight - self.learning_rate * gradient
                for weight, gradient in zip(self.weights, gradient_weights)
            ]
            if not math.isfinite(self.intercept) or any(not math.isfinite(value) for value in self.weights):
                raise AdmissionError(f"Model parameters became non-finite at epoch {epoch}")
        return self

    def predict_scores(self, rows: Sequence[Sequence[float]]) -> list[float]:
        if not self.weights:
            raise AdmissionError("Model has not been fitted")
        _validate_matrix(rows, len(self.weights))
        scores = [
            _sigmoid(self.intercept + sum(weight * value for weight, value in zip(self.weights, row)))
            for row in rows
        ]
        if any(not math.isfinite(score) for score in scores):
            raise AdmissionError("Model produced a non-finite score")
        return scores

def feature_matrix(method, rows):
    if method not in WIDTHS or not isinstance(rows, list) or not rows:
        raise AdmissionError("reference input invalid")
    matrix = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise AdmissionError("reference rows must be objects")
        x = row.get("x42")
        if (not isinstance(x, list) or len(x) != 42
                or any(type(v) not in (int, float) or not math.isfinite(v)
                       or not 0 <= v <= 1 for v in x)):
            raise AdmissionError("lexical vector must contain 42 finite values in [0,1]")
        if method == "S33":
            semantic = row.get("semantic3")
            if (not isinstance(semantic, list) or len(semantic) != 3
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in semantic)
                    or any(not -1 <= v <= 1 for v in semantic[:2])
                    or semantic[2] not in (0, 1)):
                raise AdmissionError("S33 needs the three fixed MiniLM features")
            vector = list(x[:30]) + semantic
        else:
            vector = list(x[:WIDTHS[method]])
        matrix.append(vector)
    return matrix


def fit_reference(method, rows, *, deadline=None):
    matrix = feature_matrix(method, rows)
    labels = [row.get("label") for row in rows]
    if any(type(v) is not int or v not in (0, 1) for v in labels) or set(labels) != {0, 1}:
        raise AdmissionError("reference fits require integer labels and both classes")
    if method == "B22":
        model = LogisticRegressionGD(learning_rate=0.3, l2=0.01, epochs=800)
        model.fit(matrix, labels, deadline=time.monotonic() + 180 if deadline is None else deadline)
    else:
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
        except ImportError as exc:
            raise AdmissionError("an explicit tree fit requires the pinned sklearn environment") from exc
        model = HistGradientBoostingClassifier(random_state=42, **HGB_PARAMETERS)
        model.fit(matrix, labels)
    return {"schema": REFERENCE_SCHEMA, "method": method, "estimator": model}


def predict_reference(bundle, rows):
    if (not isinstance(bundle, Mapping) or bundle.get("schema") != REFERENCE_SCHEMA
            or not isinstance(rows, list)
            or any(not isinstance(row, Mapping) or "label" in row for row in rows)):
        raise AdmissionError("reference prediction requires label-free rows and a valid bundle")
    method, model = bundle["method"], bundle["estimator"]
    matrix = feature_matrix(method, rows)
    if method == "B22":
        values = model.predict_scores(matrix)
    else:
        if list(model.classes_) != [0, 1]:
            raise AdmissionError("HGB class order differs from [0,1]")
        values = [float(row[1]) for row in model.predict_proba(matrix)]
    if len(values) != len(rows) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise AdmissionError("reference probability invalid")
    ids = [row["pair_id"] for row in rows]
    if any(not isinstance(v, str) or not v for v in ids) or len(set(ids)) != len(ids):
        raise AdmissionError("reference pair IDs invalid")
    return [{"pair_id": pid, "score": float(value)} for pid, value in zip(ids, values)]


def save_reference(path, bundle):
    """Write a new local checkpoint; never overwrite an existing output."""
    if bundle.get("schema") != REFERENCE_SCHEMA or bundle.get("method") not in WIDTHS:
        raise AdmissionError("reference checkpoint schema invalid")
    payload = pickle.dumps(bundle, protocol=5)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(payload)
    return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def load_reference(path, *, expected_sha256):
    """Load only trusted locally produced pickle, with an explicit pinned hash.

    A hash checks integrity, not trust: do not use checkpoints from strangers.
    """
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise AdmissionError("reference checkpoint missing or linked")
    payload = target.read_bytes()
    if (not isinstance(expected_sha256, str)
            or hashlib.sha256(payload).hexdigest() != expected_sha256):
        raise AdmissionError("reference checkpoint SHA mismatch")
    if expected_sha256 in LEGACY_REFERENCES:
        # Four locally produced originals only. Never broaden old-schema loading
        # to arbitrary caller-supplied SHA values. Estimator state is unchanged;
        # B22's module relocation does not train or update its parameters.
        bundle = _LegacyReferenceUnpickler(io.BytesIO(payload)).load()
        if (not isinstance(bundle, Mapping) or set(bundle) != {"schema", "method", "estimator"}
                or bundle["schema"] != "pem-v10-reference-v1"
                or bundle["method"] != LEGACY_REFERENCES[expected_sha256]):
            raise AdmissionError("Pinned original reference identity differs")
        bundle = {**bundle, "schema": REFERENCE_SCHEMA}
    else:
        bundle = _CurrentReferenceUnpickler(io.BytesIO(payload)).load()
    if not isinstance(bundle, Mapping) or bundle.get("schema") != REFERENCE_SCHEMA or bundle.get("method") not in WIDTHS:
        raise AdmissionError("reference checkpoint schema invalid")
    return bundle
