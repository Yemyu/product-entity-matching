"""Frozen raw-CSV reconstruction checks using fictional products only."""
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
import warnings
import zipfile

from product_matching.data import business_record, native_record
from product_matching.errors import ContractError
from product_matching.source import (MEMBERS, PAIR_COLUMNS, ROLES, SCHEMA,
                                     TABLE_COLUMNS, canonical_bytes, reconstruct_roles)


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def csv_bytes(columns, rows):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(columns)
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def fixture_files():
    a = [["a1", "Acme A-1", "Device", "Acme", "A-1", "12.00"],
         ["a\n2", 'Tower\nSpecial "édition"', "Device", "Acme", "A-2", "13"],
         ["a3", "Acme A-3", "Device", "Acme", "A-3", "14"],
         ["a4", "Acme A-4", "Device", "Acme", "A-4", "15"]]
    b = [["b1", "Acme A1", "Device", "Acme", "A1", "12"],
         ["b2", "Acme Other", "Device", "Acme", "OTHER-2", "13"],
         ["b3", "Acme A3", "Device", "Acme", "A3", "14"],
         ["b4", "Acme A4", "Device", "Acme", "A4", "15"]]
    files = {"tableA.csv": csv_bytes(TABLE_COLUMNS, a),
             "tableB.csv": csv_bytes(TABLE_COLUMNS, b),
             "train.csv": csv_bytes(PAIR_COLUMNS, [["a1", "b1", "1"], ["a\n2", "b2", "0"]]),
             "valid.csv": csv_bytes(PAIR_COLUMNS, [["a3", "b3", "1"]]),
             "test.csv": csv_bytes(PAIR_COLUMNS, [["a4", "b4", "1"]])}
    return files, a, b


def write_archive(path, files, extras=()):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("exp_data/", b"")
        for name in MEMBERS:
            bundle.writestr("exp_data/" + name, files[name])
        for name, payload in extras:
            bundle.writestr(name, payload)


def recipe_for(archive, files, a, b):
    source = {"archive": {"bytes": archive.stat().st_size, "sha256": sha(archive.read_bytes())},
              "members": {name: {"path": "exp_data/" + name, "bytes": len(files[name]),
                                  "sha256": sha(files[name])} for name in MEMBERS},
              "table_columns": TABLE_COLUMNS.copy(), "pair_columns": PAIR_COLUMNS.copy()}
    specifications = {"fit": ("train.csv", 3, 1, 0, ["modelno"], []),
                      "dev": ("train.csv", 2, 0, 1, [], []),
                      "calibration": ("valid.csv", 2, 2, 1, [], ["category"]),
                      "evaluation": ("test.csv", 2, 3, 1, [], [])}
    roles = {}
    for role, (member, record, index, label, left_missing, right_missing) in specifications.items():
        pid = f"walmart-amazon:structured:{member}:{record}"
        left, right = dict(zip(TABLE_COLUMNS, a[index])), dict(zip(TABLE_COLUMNS, b[index]))
        left_native, right_native = native_record(left), native_record(right)
        for key in left_missing:
            left_native[key] = None
        for key in right_missing:
            right_native[key] = None
        row = {"pair_id": pid, "left": business_record(left), "right": business_record(right),
               "left_native": left_native, "right_native": right_native}
        if role == "fit":
            row["label"] = label
        roles[role] = {"entries": [{"member": member, "record": record,
                                   "left_missing": left_missing, "right_missing": right_missing}],
                       "count": 1, "expected_business_sha256": sha(canonical_bytes([row])),
                       "expected_pair_ids_sha256": sha(canonical_bytes([pid]))}
    return {"schema": SCHEMA, "source": source, "roles": roles}


class SourceReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.archive = self.root / "source.zip"
        self.files, self.a, self.b = fixture_files()
        write_archive(self.archive, self.files)
        self.recipe = recipe_for(self.archive, self.files, self.a, self.b)
        self.output = self.root / "reconstructed"

    def reject(self, *, recipe=None):
        with self.assertRaises(ContractError):
            reconstruct_roles(self.archive, recipe or self.recipe, self.output)
        self.assertFalse(self.output.exists(), "Invalid input must not create an output directory")

    def repack(self, *, extras=()):
        write_archive(self.archive, self.files, extras)
        self.recipe["source"] = recipe_for(self.archive, self.files, self.a, self.b)["source"]

    def test_multiline_csv_uses_parsed_record_ordinals_and_preserves_unicode(self):
        before = copy.deepcopy(self.recipe)
        receipt = reconstruct_roles(self.archive, self.recipe, self.output)
        row = json.loads((self.output / "roles/fit-business.json").read_bytes())[0]
        self.assertEqual(row["pair_id"], "walmart-amazon:structured:train.csv:3")
        self.assertEqual(row["left"]["normalized"]["title"], 'tower special "édition"')
        self.assertEqual(row["left"]["normalized"]["price"], "13")
        self.assertIsNone(row["left"]["normalized"]["description"])
        self.assertIsNone(row["left"]["normalized"]["priceCurrency"])
        self.assertEqual(self.recipe, before)
        for relative, identity in receipt["outputs"].items():
            payload = (self.output / relative).read_bytes()
            self.assertEqual(identity, {"bytes": len(payload), "sha256": sha(payload)})
            self.assertTrue(payload.endswith(b"\n"))
        self.assertEqual(json.loads((self.output / "RECONSTRUCTION_RECEIPT.json").read_bytes()), receipt)
        self.assertTrue(receipt["no_new_split"] and receipt["no_training"] and receipt["no_model_inference"])
        self.assertFalse(receipt["historical_exposure_certified"])

    def test_native_missing_recipe_only_removes_observed_fields(self):
        reconstruct_roles(self.archive, self.recipe, self.output)
        fit = json.loads((self.output / "roles/fit-business.json").read_bytes())[0]
        cal = json.loads((self.output / "roles/calibration-business.json").read_bytes())[0]
        self.assertEqual(fit["left_native"], {"modelno": None, "category": "device"})
        self.assertEqual(cal["right_native"], {"modelno": "a3", "category": None})

    def test_nonfit_labels_stay_private_and_fit_labels_are_preserved(self):
        reconstruct_roles(self.archive, self.recipe, self.output)
        for role in ROLES:
            rows = json.loads((self.output / f"roles/{role}-business.json").read_bytes())
            labels = json.loads((self.output / f"private/{role}-labels.json").read_bytes())
            self.assertEqual(labels[0]["pair_id"], rows[0]["pair_id"])
            self.assertEqual(set(labels[0]), {"pair_id", "label"})
            if role == "fit":
                self.assertEqual(rows[0]["label"], labels[0]["label"])
            else:
                self.assertNotIn("label", rows[0])

    def test_membership_order_is_bound_by_digest_and_duplicate_roles_are_rejected(self):
        changed = copy.deepcopy(self.recipe)
        spec = changed["roles"]["fit"]
        spec["entries"].append(changed["roles"]["dev"]["entries"][0])
        spec["count"] = 2
        del changed["roles"]["dev"]["entries"][:]
        spec["expected_pair_ids_sha256"] = sha(canonical_bytes([
            "walmart-amazon:structured:train.csv:3", "walmart-amazon:structured:train.csv:2"]))
        spec["entries"].reverse()
        self.reject(recipe=changed)
        changed = copy.deepcopy(self.recipe)
        changed["roles"]["dev"]["entries"] = copy.deepcopy(changed["roles"]["fit"]["entries"])
        changed["roles"]["dev"]["expected_pair_ids_sha256"] = changed["roles"]["fit"]["expected_pair_ids_sha256"]
        self.reject(recipe=changed)

    def test_archive_member_business_and_membership_hashes_are_all_required(self):
        variants = []
        for target, key in (("archive", "sha256"), ("archive", "bytes")):
            value = copy.deepcopy(self.recipe)
            value["source"][target][key] = "0" * 64 if key == "sha256" else 1
            variants.append(value)
        value = copy.deepcopy(self.recipe)
        value["source"]["members"]["tableA.csv"]["sha256"] = "0" * 64
        variants.append(value)
        for key in ("expected_business_sha256", "expected_pair_ids_sha256"):
            value = copy.deepcopy(self.recipe)
            value["roles"]["evaluation"][key] = "0" * 64
            variants.append(value)
        for value in variants:
            with self.subTest(value=value):
                self.reject(recipe=value)

    def test_headers_width_and_utf8_are_checked(self):
        bad_payloads = [csv_bytes(TABLE_COLUMNS[::-1], self.a),
                        csv_bytes(TABLE_COLUMNS, [self.a[0][:-1]]), b"\xff\n"]
        for payload in bad_payloads:
            with self.subTest(payload=payload):
                self.files["tableA.csv"] = payload
                self.repack()
                self.reject()

    def test_labels_missing_endpoints_and_duplicate_endpoint_pairs_are_rejected(self):
        variants = [[["a1", "b1", "2"], ["a\n2", "b2", "0"]],
                    [["a1", "missing", "1"], ["a\n2", "b2", "0"]],
                    [["a1", "b1", "1"], ["a1", "b1", "0"]]]
        for rows in variants:
            with self.subTest(rows=rows):
                self.files["train.csv"] = csv_bytes(PAIR_COLUMNS, rows)
                self.repack()
                self.reject()

    def test_missing_and_duplicate_table_ids_are_rejected(self):
        for rows in ([self.a[0], self.a[0]], [["", *self.a[0][1:]]]):
            with self.subTest(rows=rows):
                self.files["tableA.csv"] = csv_bytes(TABLE_COLUMNS, rows)
                self.repack()
                self.reject()

    def test_additional_members_and_path_traversal_are_rejected(self):
        for name in ("exp_data/extra.csv", "../outside.csv", "/absolute.csv", "exp_data/nested/"):
            with self.subTest(name=name):
                self.repack(extras=[(name, b"x")])
                self.reject()
        changed = copy.deepcopy(self.recipe)
        changed["source"]["members"]["tableA.csv"]["path"] = "../tableA.csv"
        self.reject(recipe=changed)

    def test_duplicate_zip_members_symlinks_and_expansion_limits_are_rejected(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.repack(extras=[("exp_data/tableA.csv", self.files["tableA.csv"])])
        self.reject()
        self.repack()
        with zipfile.ZipFile(self.archive, "a") as bundle:
            member = zipfile.ZipInfo("exp_data/link")
            member.create_system = 3
            member.external_attr = (stat.S_IFLNK | 0o777) << 16
            bundle.writestr(member, "tableA.csv")
        self.recipe["source"]["archive"] = {"bytes": self.archive.stat().st_size,
                                                "sha256": sha(self.archive.read_bytes())}
        self.reject()
        changed = copy.deepcopy(self.recipe)
        changed["source"]["members"]["tableA.csv"]["bytes"] = 100 * 1024 * 1024
        self.reject(recipe=changed)

    def test_bad_membership_ordinals_masks_and_unlinked_source_are_rejected(self):
        for updates in ({"record": True}, {"record": 1}, {"record": 99},
                        {"left_missing": ["brand"]}, {"left_missing": ["modelno", "modelno"]}):
            with self.subTest(updates=updates):
                changed = copy.deepcopy(self.recipe)
                changed["roles"]["fit"]["entries"][0].update(updates)
                if updates == {"record": 99}:
                    changed["roles"]["fit"]["expected_pair_ids_sha256"] = sha(canonical_bytes([
                        "walmart-amazon:structured:train.csv:99"]))
                self.reject(recipe=changed)
        link = self.root / "linked.zip"
        link.symlink_to(self.archive)
        with self.assertRaises(ContractError):
            reconstruct_roles(link, self.recipe, self.output)
        self.assertFalse(self.output.exists())

    def test_existing_output_is_never_overwritten(self):
        reconstruct_roles(self.archive, self.recipe, self.output)
        receipt = (self.output / "RECONSTRUCTION_RECEIPT.json").read_bytes()
        with self.assertRaises(ContractError):
            reconstruct_roles(self.archive, self.recipe, self.output)
        self.assertEqual((self.output / "RECONSTRUCTION_RECEIPT.json").read_bytes(), receipt)


if __name__ == "__main__":
    unittest.main()
