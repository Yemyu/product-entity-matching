"""Synthetic, data-free checks of the final ranking/selection contracts."""
import math
import unittest
from product_matching.errors import AdmissionError
from product_matching.metrics import (
    align_scores, average_precision, fixed_threshold_metrics, join_labels,
    mean_scores, precision_at_k, select_common_epoch, threshold_selection,
)


class FinalMetricsTests(unittest.TestCase):
    def test_tied_ap_is_grouped_and_order_independent(self):
        rows = [{"pair_id": "a", "label": 1, "score": .9},
                {"pair_id": "b", "label": 0, "score": .9},
                {"pair_id": "c", "label": 1, "score": .5}]
        self.assertAlmostEqual(average_precision(rows), (.5 + 2 / 3) / 2)
        self.assertEqual(average_precision(rows), average_precision(rows[::-1]))

    def test_alignment_uses_ids_and_rejects_duplicates_missing_extra(self):
        rows = [{"pair_id": "b", "score": .1}, {"pair_id": "a", "score": .8}]
        self.assertEqual(align_scores(["a", "b"], rows), [("a", .8), ("b", .1)])
        for bad in [rows[:1], rows + [rows[0]], rows + [{"pair_id": "c", "score": .2}]]:
            with self.assertRaises(AdmissionError):
                align_scores(["a", "b"], bad)

    def test_invalid_probabilities_and_labels_are_rejected(self):
        for value in [True, float("nan"), float("inf"), -.1, 1.1]:
            with self.assertRaises(AdmissionError):
                average_precision([{"pair_id": "a", "label": 1, "score": value}])
        with self.assertRaises(AdmissionError):
            average_precision([{"pair_id": "a", "label": True, "score": .5}])
        with self.assertRaises(AdmissionError):
            average_precision([{"pair_id": "a", "label": 0, "score": .5}])

    def test_threshold_ties_and_all_negative_cutoff(self):
        rows = [{"pair_id": "a", "label": 1, "score": 1.0},
                {"pair_id": "b", "label": 0, "score": .5}]
        selected = threshold_selection(rows)
        self.assertEqual(selected["selected"]["threshold"], 1.0)
        self.assertEqual(selected["selected"]["confusion"], {"tp": 1, "fp": 0, "fn": 0, "tn": 1})
        self.assertEqual(selected["candidates"], 3)
        self.assertTrue(selected["includes_all_negative_cutoff"])

    def test_top100_ties_use_ids_and_confusion_uses_ge(self):
        rows = [{"pair_id": f"p{i:03}", "label": int(i in (0, 100)), "score": .5}
                for i in range(101)]
        result = fixed_threshold_metrics(rows[::-1], .5)
        self.assertEqual(result["top_100_pair_ids"], [f"p{i:03}" for i in range(100)])
        self.assertEqual(result["p_at_100"], .01)
        self.assertEqual(result["confusion"], {"tp": 2, "fp": 99, "fn": 0, "tn": 0})
        self.assertEqual(result["top_100_boundary_tie_count"], 101)
        self.assertEqual(precision_at_k(rows, 1), 1.0)
        all_negative = fixed_threshold_metrics(rows, math.nextafter(1.0, math.inf))
        self.assertEqual(all_negative["confusion"], {"tp": 0, "fp": 0, "fn": 2, "tn": 99})

    def test_seed_mean_is_probability_mean_and_requires_all_three(self):
        seeds = {42: [{"pair_id": "a", "score": .9}],
                 43: [{"pair_id": "a", "score": .3}],
                 44: [{"pair_id": "a", "score": .3}]}
        self.assertEqual(mean_scores(["a"], seeds), [{"pair_id": "a", "score": .5}])
        with self.assertRaises(AdmissionError):
            mean_scores(["a"], {42: seeds[42], 44: seeds[44]})

    def test_join_never_relies_on_row_position(self):
        labels = [{"pair_id": "a", "label": 1}, {"pair_id": "b", "label": 0}]
        scores = [{"pair_id": "b", "score": .2}, {"pair_id": "a", "score": .8}]
        self.assertEqual(join_labels(labels, scores)[0], {"pair_id": "a", "label": 1, "score": .8})

    def test_shared_epoch_selected_from_ensemble_and_earlier_tie(self):
        labels = [{"pair_id": f"p{i:03}", "label": int(i < 10)} for i in range(100)]
        good = [{"pair_id": row["pair_id"], "score": .9 if row["label"] else .1} for row in labels]
        bad = [{"pair_id": row["pair_id"], "score": .1 if row["label"] else .9} for row in labels]
        refs = {name: bad for name in ("B22", "L30", "S33", "L42")}
        scores = {seed: {epoch: good if epoch in (3, 4) else bad for epoch in range(1, 7)}
                  for seed in (42, 43, 44)}
        result = select_common_epoch(labels, refs, scores)
        self.assertEqual(result["selected_epoch"], 3)
        self.assertEqual(len(result["six_common_epoch_candidates"]), 6)
        self.assertNotIn("adoption", result)
        with self.assertRaises(AdmissionError):
            select_common_epoch(labels, refs, {42: scores[42], 43: scores[43]})


if __name__ == "__main__":
    unittest.main()
