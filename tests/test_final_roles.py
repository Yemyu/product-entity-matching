"""Membership and label-boundary checks for the frozen role adapter."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from product_matching.cli import main
from product_matching.errors import ContractError
from product_matching.roles import project_role, project_roles, ROLES
from product_matching.references import _LegacyReferenceUnpickler, LogisticRegressionGD
from product_matching.references import _CurrentReferenceUnpickler, load_reference
import hashlib
import pickle


def fixture(prefix="p", fit=False):
    business = {"normalized": {"brand": None, "title": "example a-1", "description": None,
                               "price": None, "priceCurrency": None},
                "quality": {"price": "missing", "priceCurrency": "missing"}}
    views = [{"pair_id": prefix + str(i), "left": business, "right": business,
              **({"label": i % 2} if fit else {})} for i in range(3)]
    ids = [views[i]["pair_id"] for i in (0, 2)]
    native = [{"pair_id": pid, "left_modelno": " A-1 ", "right_modelno": "a1",
               "left_category": " CAT ", "right_category": None} for pid in ids]
    return views, native, ids


class FinalRoleTests(unittest.TestCase):
    def test_ordered_subsequence_and_native_normalization(self):
        views, native, ids = fixture(fit=True)
        before = copy.deepcopy((views, native, ids))
        rows = project_role(views, native, ids, role="fit")
        self.assertEqual([r["pair_id"] for r in rows], ids)
        self.assertEqual(rows[0]["left_native"], {"modelno": "a-1", "category": "cat"})
        self.assertEqual((views, native, ids), before)

    def test_nonfit_label_is_rejected(self):
        for role in ROLES[1:]:
            with self.assertRaises(ContractError):
                project_role(*fixture(fit=True), role=role)

    def test_invalid_membership_or_order_is_rejected(self):
        views, native, ids = fixture()
        for bad in ([], ids + ids, ["absent"], ids[::-1], [True]):
            with self.assertRaises(ContractError):
                project_role(views, native, bad, role="dev")
        with self.assertRaises(ContractError):
            project_role(views + [views[0]], native, ids, role="dev")
        with self.assertRaises(ContractError):
            project_role(views, native[::-1], ids, role="dev")

    def test_native_or_business_unknown_fields_are_rejected(self):
        views, native, ids = fixture()
        native[0]["label"] = 1
        with self.assertRaises(ContractError):
            project_role(views, native, ids, role="dev")
        views, native, ids = fixture()
        views[0]["left"] = {"title": "bad"}
        with self.assertRaises(ContractError):
            project_role(views, native, ids, role="dev")

    def test_all_roles_and_overlap(self):
        fixtures = {r: fixture(prefix=r, fit=r == "fit") for r in ROLES}
        v, n, ids = ({r: fixtures[r][index] for r in ROLES} for index in range(3))
        projected = project_roles(v, n, ids)
        self.assertEqual(set(projected), set(ROLES))
        for r in ROLES[1:]:
            self.assertNotIn("label", projected[r][0])
        v["evaluation"], n["evaluation"], ids["evaluation"] = fixture(prefix="dev")
        with self.assertRaises(ContractError):
            project_roles(v, n, ids)

    def test_cli_json_output_and_no_overwrite(self):
        views, native, ids = fixture()
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            for name, value in (("views", views), ("sidecars", native), ("ids", ids)):
                (d / (name + ".json")).write_text(json.dumps(value))
            args = ["project-role", "--role", "dev", "--views", str(d / "views.json"),
                    "--sidecars", str(d / "sidecars.json"), "--kept-ids", str(d / "ids.json"),
                    "--output", str(d / "result.json")]
            self.assertEqual(main(args), 0)
            self.assertEqual(json.loads((d / "result.json").read_text()),
                             project_role(views, native, ids, role="dev"))
            with self.assertRaises(SystemExit) as raised:
                main(args)
            self.assertEqual(raised.exception.code, 2)

    def test_legacy_class_relocation_without_old_package(self):
        payload = b"cproduct_entity_matching.experiments\nLogisticRegressionGD\n."
        self.assertIs(_LegacyReferenceUnpickler(io.BytesIO(payload)).load(), LogisticRegressionGD)
        with self.assertRaises(ContractError):
            _LegacyReferenceUnpickler(io.BytesIO(b"cproduct_entity_matching.unknown\nAnything\n.")).load()

    def test_unpinned_old_class_is_rejected_before_import(self):
        payload = b"cproduct_entity_matching.experiments\nLogisticRegressionGD\n."
        with self.assertRaises(ContractError):
            _CurrentReferenceUnpickler(io.BytesIO(payload)).load()

    def test_old_schema_with_arbitrary_matching_hash_is_not_promoted(self):
        payload = pickle.dumps({"schema": "pem-v10-reference-v1", "method": "B22",
                                "estimator": LogisticRegressionGD()})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.pkl"
            path.write_bytes(payload)
            with self.assertRaises(ContractError):
                load_reference(path, expected_sha256=hashlib.sha256(payload).hexdigest())


if __name__ == "__main__":
    unittest.main()
