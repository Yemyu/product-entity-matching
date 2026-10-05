"""Label-aligned ranking, calibration and development selection arithmetic.

Scores are probabilities. AP groups equal scores; Top-k uses pair IDs to break
ties. Calibration chooses F1, then precision, then the higher score cutoff.
These functions do not open files, train models, or make adoption decisions.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from .errors import AdmissionError

REFERENCES = ("B22", "L30", "S33", "L42")
SEEDS = (42, 43, 44)
EPOCHS = tuple(range(1, 7))
MAX_CUTOFF = math.nextafter(1.0, math.inf)


def _probability(value, field="score"):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or not 0 <= float(value) <= 1):
        raise AdmissionError(f"{field} must be a finite probability")
    return float(value)


def align_scores(expected_ids, rows):
    """Return (ID, score) tuples in the explicit expected order."""
    expected = list(expected_ids)
    if (any(not isinstance(value, str) or not value for value in expected)
            or len(expected) != len(set(expected))):
        raise AdmissionError("expected IDs must be unique nonempty strings")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise AdmissionError("predictions must be a sequence")
    found = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"pair_id", "score"}:
            raise AdmissionError("prediction schema mismatch")
        pair_id = row["pair_id"]
        if not isinstance(pair_id, str) or not pair_id or pair_id in found:
            raise AdmissionError("prediction IDs must be unique nonempty strings")
        found[pair_id] = _probability(row["score"])
    if set(found) != set(expected):
        raise AdmissionError("prediction IDs have missing or extra values")
    return [(pair_id, found[pair_id]) for pair_id in expected]


def _validate_rows(rows):
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise AdmissionError("metric rows must be a nonempty sequence")
    values, seen = [], set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"pair_id", "label", "score"}:
            raise AdmissionError("metric row schema mismatch")
        pair_id, label = row["pair_id"], row["label"]
        if not isinstance(pair_id, str) or not pair_id or pair_id in seen:
            raise AdmissionError("metric IDs must be unique nonempty strings")
        if type(label) is not int or label not in (0, 1):
            raise AdmissionError("metric labels must be integer 0 or 1")
        seen.add(pair_id)
        values.append({"pair_id": pair_id, "label": label,
                       "score": _probability(row["score"])})
    if not any(row["label"] for row in values):
        raise AdmissionError("metrics require a positive label")
    return values


def average_precision(rows):
    """Non-interpolated AP with each tied score entering as a single group."""
    values = _validate_rows(rows)
    groups = {}
    for row in values:
        groups.setdefault(row["score"], []).append(row["label"])
    positives = sum(row["label"] for row in values)
    seen = seen_rows = total = 0.0
    for score in sorted(groups, reverse=True):
        labels = groups[score]
        seen_rows += len(labels)
        group_positive = sum(labels)
        seen += group_positive
        if group_positive:
            total += (seen / seen_rows) * group_positive
    return total / positives


def grouped_average_precision(rows):
    """Tuple-row adapter for the independently audited AP convention."""
    return average_precision([{"pair_id": pid, "label": label, "score": score}
                              for pid, label, score in rows])


def precision_at_k(rows, k=100):
    values = _validate_rows(rows)
    if type(k) is not int or k <= 0 or len(values) < k:
        raise AdmissionError("Top-k requires a positive integer k and k pairs")
    ordered = sorted(values, key=lambda row: (-row["score"], row["pair_id"]))
    return sum(row["label"] for row in ordered[:k]) / k


def _confusion(rows, threshold):
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    predicted = 0
    for row in rows:
        positive = row["score"] >= threshold
        actual = row["label"] == 1
        predicted += int(positive)
        counts["tp" if positive and actual else "fp" if positive
               else "fn" if actual else "tn"] += 1
    precision = counts["tp"] / predicted if predicted else 0.0
    recall = counts["tp"] / (counts["tp"] + counts["fn"])
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": threshold, "confusion": counts, "precision": precision,
            "recall": recall, "f1": f1}


def threshold_selection(rows):
    values = _validate_rows(rows)
    cutoffs = sorted({row["score"] for row in values}, reverse=True)
    cutoffs.append(math.nextafter(cutoffs[0], math.inf))
    scored = [_confusion(values, threshold) for threshold in cutoffs]
    best = max(scored, key=lambda row: (row["f1"], row["precision"], row["threshold"]))
    return {"selected": best, "candidates": len(scored),
            "includes_all_negative_cutoff": cutoffs[-1] > max(row["score"] for row in values)}


def fixed_threshold_metrics(rows, threshold):
    values = _validate_rows(rows)
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold)) or not 0 <= float(threshold) <= MAX_CUTOFF):
        raise AdmissionError("locked cutoff is outside the probability range")
    if len(values) < 100:
        raise AdmissionError("Top-100 evaluation requires at least 100 pairs")
    ordered = sorted(values, key=lambda row: (-row["score"], row["pair_id"]))
    top = ordered[:100]
    boundary = top[-1]["score"]
    return {"count": len(values), "positive_count": sum(row["label"] for row in values),
            "ap": average_precision(values), "p_at_100": sum(row["label"] for row in top) / 100,
            "top_100_pair_ids": [row["pair_id"] for row in top],
            "top_100_boundary_score": boundary,
            "top_100_boundary_tie_count": sum(row["score"] == boundary for row in ordered),
            **_confusion(values, float(threshold))}


def join_labels(labels, predictions):
    """Join explicit label rows to label-free predictions by pair ID."""
    if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
        raise AdmissionError("labels must be a sequence")
    ids, by_id = [], {}
    for row in labels:
        if not isinstance(row, Mapping) or set(row) != {"pair_id", "label"}:
            raise AdmissionError("label schema mismatch")
        pid, label = row["pair_id"], row["label"]
        if (not isinstance(pid, str) or not pid or pid in by_id
                or type(label) is not int or label not in (0, 1)):
            raise AdmissionError("label values or IDs are invalid")
        ids.append(pid)
        by_id[pid] = label
    return _validate_rows([{"pair_id": pid, "label": by_id[pid], "score": score}
                           for pid, score in align_scores(ids, predictions)])


def mean_scores(expected_ids, seed_predictions):
    """The final method averages probabilities from all three fixed seeds."""
    if not isinstance(seed_predictions, Mapping) or set(seed_predictions) != set(SEEDS):
        raise AdmissionError("all three fixed seed predictions are required")
    ids = list(expected_ids)
    by_seed = {seed: dict(align_scores(ids, seed_predictions[seed])) for seed in SEEDS}
    return [{"pair_id": pid, "score": sum(by_seed[seed][pid] for seed in SEEDS) / len(SEEDS)}
            for pid in ids]


def _ranking(labels, predictions):
    rows = join_labels(labels, predictions)
    return {"ap": average_precision(rows), "p_at_100": precision_at_k(rows),
            "top_100_pair_ids": [row["pair_id"] for row in
                                 sorted(rows, key=lambda row: (-row["score"], row["pair_id"]))[:100]]}


def select_common_epoch(dev_labels, references, cplus):
    """Maximize ensemble dev AP, then P@100, then prefer the earlier epoch."""
    if (not isinstance(references, Mapping) or set(references) != set(REFERENCES)
            or not isinstance(cplus, Mapping) or set(cplus) != set(SEEDS)):
        raise AdmissionError("development method inventory mismatch")
    ref_metrics = {name: _ranking(dev_labels, references[name]) for name in REFERENCES}
    by_seed = {}
    for seed in SEEDS:
        if not isinstance(cplus[seed], Mapping) or set(cplus[seed]) != set(EPOCHS):
            raise AdmissionError("all six epochs per seed are required")
        by_seed[seed] = {epoch: _ranking(dev_labels, cplus[seed][epoch]) for epoch in EPOCHS}
    ids = [row["pair_id"] for row in dev_labels]
    candidates = []
    for epoch in EPOCHS:
        ensemble = mean_scores(ids, {seed: cplus[seed][epoch] for seed in SEEDS})
        candidates.append({"epoch": epoch, **_ranking(dev_labels, ensemble)})
    selected = max(candidates, key=lambda row: (row["ap"], row["p_at_100"], -row["epoch"]))
    return {"selected_epoch": selected["epoch"], "references": ref_metrics,
            "cplus_by_seed": by_seed, "six_common_epoch_candidates": candidates}


select_calibration_threshold = threshold_selection
score_at_threshold = fixed_threshold_metrics
select_for_comparison = select_common_epoch
