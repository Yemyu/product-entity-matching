"""Synthetic checks for final features, role boundaries and serialization."""
import copy
import json
import unittest

from product_matching.data import business_record, native_record, normalize_native
from product_matching.errors import AdmissionError
from product_matching.features import (
    FEATURE_NAMES_42, fit_title_evidence, local_code_features, prepare_rows,
)
from product_matching.lexical import TitleEvidence
from product_matching.normalization import normalize_text, parse_price
from product_matching.serialization import (
    encode_rows, pair_tokens, record_tokens, reference_rows,
)


class CharacterTokenizer:
    """A deterministic synthetic tokenizer, never a substitute model result."""
    pad_token_id = 1
    cls_token_id = 0
    sep_token_id = 2
    cls_token = "<s>"
    sep_token = "</s>"

    def encode(self, text, *, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("field encoding must not add special tokens")
        return [3 + ord(char) for char in text]

    def num_special_tokens_to_add(self, *, pair):
        return 4 if pair else 2

    def build_inputs_with_special_tokens(self, left, right):
        return [0, *left, 2, 2, *right, 2]


class FallbackTokenizer(CharacterTokenizer):
    build_inputs_with_special_tokens = None


def sample(pair_id="synthetic:01", *, labelled=True):
    row = {
        "pair_id": pair_id,
        "left": business_record({"title": "Brand ZX-100", "brand": "Brand", "price": "0", "priceCurrency": "USD"}),
        "right": business_record({"title": "Brand ZX100", "brand": "Brand", "price": "0", "priceCurrency": "USD"}),
        "left_native": native_record({"modelno": "ZX-100", "category": "electronics"}),
        "right_native": native_record({"modelno": "ZX100", "category": "electronics"}),
    }
    if labelled:
        row["label"] = 1
    return row


class FinalFeatureTests(unittest.TestCase):
    def setUp(self):
        self.fit = [sample()]
        self.idf = fit_title_evidence(self.fit)

    def test_conservative_normalization_and_price(self):
        self.assertEqual(normalize_text("  ＺＸ-100\u200e  "), "zx-100")
        self.assertIsNone(normalize_native(" N/A "))
        self.assertEqual(parse_price("0").normalized, "0")
        self.assertEqual(parse_price("$1,000").status, "ambiguous_format")
        with self.assertRaises(AdmissionError):
            normalize_native(100)

    def test_feature_width_finiteness_and_native_compaction(self):
        row = prepare_rows(self.fit, role="fit", fit_idf=self.idf)[0]
        self.assertEqual(len(FEATURE_NAMES_42), 42)
        self.assertEqual(len(row["x42"]), 42)
        self.assertTrue(all(0 <= value <= 1 for value in row["x42"]))
        values = dict(zip(FEATURE_NAMES_42, row["x42"]))
        self.assertEqual(values["native_model_compact_exact"], 1)
        self.assertEqual(values["price_similarity"], 1)

    def test_features_are_symmetric(self):
        row = sample()
        reverse = {**row, "pair_id": "synthetic:reverse", "left": row["right"], "right": row["left"],
                   "left_native": row["right_native"], "right_native": row["left_native"]}
        values = prepare_rows([row, reverse], role="fit", fit_idf=self.idf)
        self.assertEqual(values[0]["x42"], values[1]["x42"])

    def test_fit_idf_deduplicates_business_records_and_ignores_labels(self):
        duplicate = sample("synthetic:02")
        duplicate["label"] = 0
        self.assertEqual(fit_title_evidence(self.fit + [duplicate]).payload(), self.idf.payload())
        self.assertEqual(self.idf.document_count, 2)

    def test_idf_state_json_roundtrip_and_role_rejection(self):
        state = json.loads(json.dumps(self.idf.payload()))
        restored = TitleEvidence.from_payload(state)
        row = sample()
        self.assertEqual(restored.pair(row["left"], row["right"]), self.idf.pair(row["left"], row["right"]))
        state["fit_role"] = "evaluation"
        with self.assertRaises(AdmissionError):
            TitleEvidence.from_payload(state)
        with self.assertRaises(ValueError):
            TitleEvidence.fit([{"offer_id": "x", "split": "dev", **row["left"]}])

    def test_nonfit_labels_rejected_and_idf_is_not_refitted(self):
        before = copy.deepcopy(self.idf.payload())
        for role in ("dev", "calibration", "evaluation"):
            with self.subTest(role=role):
                with self.assertRaises(AdmissionError):
                    prepare_rows(self.fit, role=role, fit_idf=self.idf)
                row = sample(labelled=False)
                row["left"]["normalized"]["title"] = "previously unseen ABC-900"
                prepared = prepare_rows([row], role=role, fit_idf=self.idf)
                self.assertNotIn("label", prepared[0])
        self.assertEqual(self.idf.payload(), before)

    def test_duplicate_ids_extra_fields_and_invalid_labels_rejected(self):
        with self.assertRaises(AdmissionError):
            prepare_rows(self.fit * 2, role="fit", fit_idf=self.idf)
        for patch in ({"label": True}, {"leaked_target": 1}, {"left_native": {"modelno": None}}):
            with self.subTest(patch=patch), self.assertRaises(AdmissionError):
                prepare_rows([{**sample(), **patch}], role="fit", fit_idf=self.idf)

    def test_fixed_field_truncation_and_pair_template(self):
        raw = {"title": "x" * 180, "brand": "b" * 80}
        record = business_record(raw)
        native = native_record({"modelno": "m" * 80, "category": "c" * 80})
        ids, cut = record_tokens(CharacterTokenizer(), record, native)
        self.assertEqual(len(ids), 126)
        self.assertTrue(all(cut.values()))
        self.assertEqual(ids[-62:], [3 + ord(c) for c in ("\ntitle: " + "x" * 180)][:40] + [3 + ord(c) for c in ("\ntitle: " + "x" * 180)][-22:])
        ab, ba = pair_tokens(CharacterTokenizer(), ids, ids)
        self.assertEqual(len(ab), 256)
        self.assertEqual(ab, ba)
        self.assertEqual((ab, ba), pair_tokens(FallbackTokenizer(), ids, ids))

    def test_encode_rows_preserves_features_labels_and_direction(self):
        prepared = prepare_rows(self.fit, role="fit", fit_idf=self.idf)
        encoded, receipt = encode_rows(prepared, role="fit", tokenizer=CharacterTokenizer())
        self.assertEqual(encoded[0]["label"], 1)
        self.assertEqual(encoded[0]["x42"], prepared[0]["x42"])
        self.assertEqual(receipt["pairs"], 1)
        self.assertNotEqual(encoded[0]["ab"], encoded[0]["ba"])
        with self.assertRaises(AdmissionError):
            encode_rows(prepared, role="evaluation", tokenizer=CharacterTokenizer())

    def test_reference_rows_are_label_free_except_fit(self):
        prepared = prepare_rows([sample(labelled=False)], role="evaluation", fit_idf=self.idf)
        value = reference_rows(prepared, role="evaluation", method="L42")
        self.assertEqual(set(value[0]), {"pair_id", "x42"})
        semantic = [{"pair_id": "synthetic:01", "semantic3": [0.9, 0.8, 0.0]}]
        value = reference_rows(prepared, role="evaluation", method="S33", semantic_features=semantic)
        self.assertEqual(value[0]["semantic3"], semantic[0]["semantic3"])
        with self.assertRaises(AdmissionError):
            reference_rows(prepared, role="evaluation", method="S33")

    def test_model_suffix_conflict_is_preserved(self):
        values = local_code_features("ZX100", "ZX101")
        self.assertEqual(values["code_exact_overlap"], 0)
        self.assertEqual(values["code_same_letters_different_digits"], 1)
        self.assertEqual(values["code_one_edit_difference"], 1)


if __name__ == "__main__":
    unittest.main()
