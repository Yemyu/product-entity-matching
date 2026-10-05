"""Synthetic checks of fixed references and frozen semantic preprocessing."""
import hashlib
import math
import sys
import tempfile
import types
import unittest
from typing import get_type_hints
from pathlib import Path
from unittest.mock import patch
from product_matching.errors import AdmissionError
from product_matching.references import (
    HGB_PARAMETERS, LogisticRegressionGD, feature_matrix, fit_reference,
    load_reference, predict_reference, save_reference,
)
from product_matching.semantic import (
    DIMENSION, MODEL_REVISION, _unit_vectors, mask_code_spans, semantic3_for_role,
)


def business(title):
    return {"normalized": {"brand": None, "title": title, "description": None,
                           "price": None, "priceCurrency": None},
            "quality": {"price": "missing", "priceCurrency": "missing"}}


def feature_rows():
    return [{"pair_id": f"p{i}", "x42": [float(i % 2)] * 42, "label": i % 2}
            for i in range(4)]


class FinalReferenceTests(unittest.TestCase):
    def test_reference_annotations_resolve(self):
        self.assertIn("rows", get_type_hints(LogisticRegressionGD.loss_and_gradients))

    def test_matrix_dimensions_and_s33_order(self):
        row = {"pair_id": "a", "x42": [i / 42 for i in range(42)], "semantic3": [.3, -.2, 0.]}
        for method, width in [("B22", 22), ("L30", 30), ("S33", 33), ("L42", 42)]:
            matrix = feature_matrix(method, [row])
            self.assertEqual(len(matrix[0]), width)
        self.assertEqual(feature_matrix("S33", [row])[0], row["x42"][:30] + row["semantic3"])
        for bad in [[], [True] * 42, [float("nan")] * 42, [1.1] * 42]:
            with self.assertRaises(AdmissionError):
                feature_matrix("L42", [{"x42": bad}])
        with self.assertRaises(AdmissionError):
            feature_matrix("S33", [{"x42": [0.] * 42}])

    def test_logistic_gradients_match_finite_differences(self):
        model = LogisticRegressionGD()
        model.weights = [.2, -.3]
        model.intercept = .1
        rows, labels = [[0., 1.], [1., .5]], [0, 1]
        _, grad, bias_grad = model.loss_and_gradients(rows, labels)
        epsilon = 1e-6
        for index in range(2):
            value = model.weights[index]
            model.weights[index] = value + epsilon
            upper = model.loss_and_gradients(rows, labels)[0]
            model.weights[index] = value - epsilon
            lower = model.loss_and_gradients(rows, labels)[0]
            model.weights[index] = value
            self.assertAlmostEqual(grad[index], (upper - lower) / (2 * epsilon), places=7)
        model.intercept += epsilon
        upper = model.loss_and_gradients(rows, labels)[0]
        model.intercept -= 2 * epsilon
        lower = model.loss_and_gradients(rows, labels)[0]
        self.assertAlmostEqual(bias_grad, (upper - lower) / (2 * epsilon), places=7)

    def test_b22_fit_label_boundary_and_trusted_roundtrip(self):
        rows = feature_rows()
        bundle = fit_reference("B22", rows)
        plain = [{key: value for key, value in row.items() if key != "label"} for row in rows]
        predicted = predict_reference(bundle, plain)
        self.assertLess(predicted[0]["score"], predicted[1]["score"])
        with self.assertRaises(AdmissionError):
            predict_reference(bundle, rows)
        with self.assertRaises(AdmissionError):
            fit_reference("B22", [{**row, "label": 0} for row in rows])
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "reference.pkl"
            info = save_reference(target, bundle)
            self.assertEqual(info["sha256"], hashlib.sha256(target.read_bytes()).hexdigest())
            loaded = load_reference(target, expected_sha256=info["sha256"])
            self.assertEqual(predict_reference(loaded, plain), predicted)
            with self.assertRaises(AdmissionError):
                load_reference(target, expected_sha256="0" * 64)
            with self.assertRaises(FileExistsError):
                save_reference(target, bundle)

    def test_tree_factory_uses_fixed_hyperparameters_and_class_order(self):
        captured = {}
        class Tree:
            classes_ = [0, 1]
            def __init__(self, **kwargs):
                captured.update(kwargs)
            def fit(self, x, y):
                captured["width"] = len(x[0])
            def predict_proba(self, x):
                return [[.75, .25] for _ in x]
        module = types.ModuleType("sklearn.ensemble")
        module.HistGradientBoostingClassifier = Tree
        with patch.dict(sys.modules, {"sklearn.ensemble": module}):
            bundle = fit_reference("L42", feature_rows())
        self.assertEqual({key: captured[key] for key in HGB_PARAMETERS}, HGB_PARAMETERS)
        self.assertEqual(captured["random_state"], 42)
        self.assertEqual(captured["width"], 42)
        plain = [{"pair_id": "a", "x42": [0.] * 42}]
        self.assertEqual(predict_reference(bundle, plain), [{"pair_id": "a", "score": .25}])
        bundle["estimator"].classes_ = [1, 0]
        with self.assertRaises(AdmissionError):
            predict_reference(bundle, plain)

    def test_code_mask_preserves_non_code_words(self):
        self.assertEqual(mask_code_spans("  ACME ZX-100 16 GB  "), "acme 16 gb")
        self.assertEqual(mask_code_spans("AB 12 . CAT"), "ab 12 . cat")

    def test_semantic_role_local_vectors_and_label_boundary(self):
        class Encoder:
            revision, dimension = MODEL_REVISION, DIMENSION
            last_truncated_count = 0
            def encode(self, texts):
                self.texts = list(texts)
                return [[1.] + [0.] * (DIMENSION - 1) for _ in texts]
        encoder = Encoder()
        rows = [{"pair_id": "a", "left": business("ZX-100"), "right": business("zx-100")}]
        result, summary = semantic3_for_role(rows, role="dev", encoder=encoder)
        self.assertEqual(result, [{"pair_id": "a", "semantic3": [1., 0., 1.]}])
        self.assertEqual(summary["unique_encoded_texts"], 1)
        self.assertEqual(encoder.texts, ["zx-100"])
        with self.assertRaises(AdmissionError):
            semantic3_for_role([{**rows[0], "label": 1}], role="dev", encoder=encoder)
        fitted, _ = semantic3_for_role([{**rows[0], "label": 1}], role="fit", encoder=encoder)
        self.assertEqual(fitted, result)

    def test_semantic_vectors_reject_wrong_dimension_nan_and_norm(self):
        for vector in [[1.], [0.] * DIMENSION, [float("nan")] * DIMENSION]:
            with self.assertRaises(AdmissionError):
                _unit_vectors([vector], 1)


if __name__ == "__main__":
    unittest.main()
