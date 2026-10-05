"""Fit-only title IDF evidence used by the final 42-feature representation."""
from __future__ import annotations
from collections import Counter
import math
import re
from typing import Mapping, Sequence
from .features import FEATURE_NAMES, pair_features
from .normalization import normalize_text
from .errors import AdmissionError

EXTRA_FEATURE_NAMES = (
    "title_word_idf_cosine", "title_word_idf_containment", "title_char3_idf_cosine",
    "title_compact_char3_idf_cosine", "title_alphanumeric_jaccard",
    "title_number_jaccard", "title_numbers_both_present", "title_compact_exact",
)


LEXICAL_FEATURE_NAMES = (*FEATURE_NAMES, *EXTRA_FEATURE_NAMES)


_WORDS = re.compile(r"[^\W_]+", re.UNICODE)


_NUMBERS = re.compile(r"\d+", re.UNICODE)


def _grams(text: str) -> set[str]:
    if not text:
        return set()
    return {text[i:i+3] for i in range(len(text)-2)} if len(text) >= 3 else {text}


def title_parts(record: Mapping) -> dict:
    value = record["normalized"].get("title")
    if value is not None and not isinstance(value, str):
        raise ValueError("Title must be string or None")
    title = normalize_text(value) or ""
    sequence = _WORDS.findall(title)
    words = set(sequence)
    compact = "".join(sequence)
    return {"word": words, "char3": _grams(" ".join(sequence)), "compact_char3": _grams(compact),
            "mixed": {w for w in words if any(c.isalpha() for c in w) and any(c.isdigit() for c in w)},
            "numbers": set(_NUMBERS.findall(title)), "compact": compact}


class TitleEvidence:
    def __init__(self, document_count: int, document_frequencies: dict[str, dict[str, int]]):
        if type(document_count) is not int or document_count <= 0:
            raise ValueError("Positive fit document count required")
        if set(document_frequencies) != {"word", "char3", "compact_char3"}:
            raise ValueError("Unexpected document frequency channels")
        if any(type(n) is not int or not 1 <= n <= document_count for d in document_frequencies.values() for n in d.values()):
            raise ValueError("Invalid document frequency")
        self.document_count = document_count
        self.document_frequencies = {k: dict(v) for k, v in document_frequencies.items()}

    @classmethod
    def fit(cls, records: Sequence[Mapping]) -> "TitleEvidence":
        if not records or any(r.get("split") != "fit" for r in records):
            raise ValueError("IDF may only fit on the fit role")
        ids = [r.get("offer_id") for r in records]
        if any(not x for x in ids) or len(ids) != len(set(ids)):
            raise ValueError("Unique fit offer IDs required")
        frequencies = {k: Counter() for k in ("word", "char3", "compact_char3")}
        for record in records:
            parts = title_parts(record)
            for k in frequencies:
                frequencies[k].update(parts[k])
        return cls(len(records), {k: dict(v) for k, v in frequencies.items()})

    def _similarities(self, left: set[str], right: set[str], channel: str) -> tuple[float, float]:
        if not left or not right:
            return 0., 0.
        df = self.document_frequencies[channel]
        def energy(tokens):
            return math.fsum((1. + math.log((1. + self.document_count) / (1. + df.get(t, 0)))) ** 2
                             for t in sorted(tokens))
        a, b, shared = energy(left), energy(right), energy(left & right)
        return min(1., shared / math.sqrt(a*b)), min(1., shared / min(a, b))

    def pair(self, left: Mapping, right: Mapping) -> list[float]:
        a, b = title_parts(left), title_parts(right)
        word_cos, containment = self._similarities(a["word"], b["word"], "word")
        def jaccard(x, y):
            return len(x & y) / len(x | y) if x and y else 0.
        extra = [word_cos, containment,
                 self._similarities(a["char3"], b["char3"], "char3")[0],
                 self._similarities(a["compact_char3"], b["compact_char3"], "compact_char3")[0],
                 jaccard(a["mixed"], b["mixed"]), jaccard(a["numbers"], b["numbers"]),
                 float(bool(a["numbers"] and b["numbers"])), float(bool(a["compact"]) and a["compact"] == b["compact"])]
        base = pair_features(left, right)
        vector = [base[name] for name in FEATURE_NAMES] + extra
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in vector):
            raise ValueError("Feature outside [0,1]")
        return vector

    def payload(self) -> dict:
        return {"document_count": self.document_count, "document_frequencies": self.document_frequencies,
                "fit_role": "fit", "feature_names": list(LEXICAL_FEATURE_NAMES)}

    @classmethod
    def from_payload(cls, state):
        """Restore a JSON-serializable, explicitly fit-only IDF state."""
        if (not isinstance(state, Mapping)
                or set(state) != {"document_count", "document_frequencies", "fit_role", "feature_names"}
                or state["fit_role"] != "fit"
                or state["feature_names"] != list(LEXICAL_FEATURE_NAMES)):
            raise AdmissionError("IDF state schema or fit-only role differs")
        frequencies = state["document_frequencies"]
        if (not isinstance(frequencies, Mapping)
                or any(not isinstance(values, Mapping) for values in frequencies.values())
                or any(not isinstance(token, str) for values in frequencies.values() for token in values)):
            raise AdmissionError("IDF frequencies must be string-keyed mappings")
        try:
            return cls(state["document_count"], dict(frequencies))
        except (ValueError, TypeError) as exc:
            raise AdmissionError("invalid IDF state") from exc
