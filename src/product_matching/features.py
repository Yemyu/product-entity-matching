"""The final model's unchanged 14 + 8 + 12 comparison features.

This module performs no file access, model fitting, or label-dependent feature
construction. Title IDF is supplied by a separately fitted fit-only state.
"""
from __future__ import annotations
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
import math
import re
import unicodedata
from .errors import AdmissionError
from .normalization import normalize_text
from .data import digest_object, compact_native, normalize_native

LOCAL_FEATURE_NAMES = (
    "code_left_missing_or_right_missing", "code_both_present", "code_exact_overlap",
    "code_set_jaccard", "code_best_edit_similarity",
    "code_same_letters_different_digits", "code_same_digits_different_letters",
    "code_one_edit_difference",
)

FEATURE_NAMES = (
    "title_token_jaccard",
    "title_char_jaccard",
    "description_token_jaccard",
    "description_both_present",
    "brand_equal",
    "brand_both_present",
    "brand_missing_any",
    "numeric_token_jaccard",
    "numeric_tokens_both_present",
    "title_length_ratio",
    "price_both_parsed",
    "price_currency_equal",
    "price_comparable",
    "price_similarity",
)


SOURCE_FIELDS = ("brand", "title", "description", "price", "priceCurrency")


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


def _tokens(value: object) -> set[str]:
    text = normalize_text(value)
    return set(_TOKEN_RE.findall(text)) if text else set()


def _numeric_tokens(value: object) -> set[str]:
    text = normalize_text(value)
    return set(_NUMBER_RE.findall(text)) if text else set()


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _char_jaccard(left: object, right: object) -> float:
    left_text = normalize_text(left)
    right_text = normalize_text(right)
    if not left_text or not right_text:
        return 0.0
    left_chars = set(left_text.replace(" ", ""))
    right_chars = set(right_text.replace(" ", ""))
    return _jaccard(left_chars, right_chars)


def _length_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return min(len(left), len(right)) / max(len(left), len(right))


def _price_similarity(left: dict, right: dict) -> tuple[float, float, float, float]:
    left_price = left["normalized"].get("price")
    right_price = right["normalized"].get("price")
    left_currency = left["normalized"].get("priceCurrency")
    right_currency = right["normalized"].get("priceCurrency")
    currency_equal = float(bool(left_currency and right_currency and left_currency == right_currency))
    parsed = float(left["quality"].get("price") == "parsed" and right["quality"].get("price") == "parsed")
    standard = all(r["quality"].get("priceCurrency") == "standard_code" for r in (left, right))
    comparable = float(parsed and currency_equal and standard)
    if not comparable:
        return parsed, currency_equal, 0.0, 0.0
    try:
        left_value = Decimal(str(left_price))
        right_value = Decimal(str(right_price))
    except (InvalidOperation, TypeError):
        return parsed, currency_equal, 0.0, 0.0
    if not left_value.is_finite() or not right_value.is_finite() or left_value < 0 or right_value < 0:
        return parsed, currency_equal, 0.0, 0.0
    # log1p keeps zero prices finite and makes the feature less dominated by
    # large absolute prices. It remains a same-currency comparison only.
    if not all(math.isfinite(float(v)) for v in (left_value, right_value)):
        return parsed, currency_equal, 0.0, 0.0
    distance = abs(math.log1p(float(left_value)) - math.log1p(float(right_value)))
    return parsed, currency_equal, 1.0, 1.0 / (1.0 + distance)


def pair_features(left: dict, right: dict) -> dict[str, float]:
    """Create features from two record views; no IDs or labels are accepted."""

    left_n, right_n = left["normalized"], right["normalized"]
    title_left, title_right = _tokens(left_n.get("title")), _tokens(right_n.get("title"))
    description_left = _tokens(left_n.get("description"))
    description_right = _tokens(right_n.get("description"))
    numeric_left = _numeric_tokens(left_n.get("title")) | _numeric_tokens(left_n.get("description"))
    numeric_right = _numeric_tokens(right_n.get("title")) | _numeric_tokens(right_n.get("description"))
    brand_left, brand_right = left_n.get("brand"), right_n.get("brand")
    brand_both = float(bool(brand_left and brand_right))
    brand_equal = float(bool(brand_both and brand_left == brand_right))
    price_parsed, currency_equal, price_comparable, price_similarity = _price_similarity(left, right)
    return {
        "title_token_jaccard": _jaccard(title_left, title_right),
        "title_char_jaccard": _char_jaccard(left_n.get("title"), right_n.get("title")),
        "description_token_jaccard": _jaccard(description_left, description_right),
        "description_both_present": float(bool(description_left and description_right)),
        "brand_equal": brand_equal,
        "brand_both_present": brand_both,
        "brand_missing_any": float(not brand_both),
        "numeric_token_jaccard": _jaccard(numeric_left, numeric_right),
        "numeric_tokens_both_present": float(bool(numeric_left and numeric_right)),
        "title_length_ratio": _length_ratio(title_left, title_right),
        "price_both_parsed": price_parsed,
        "price_currency_equal": currency_equal,
        "price_comparable": price_comparable,
        "price_similarity": price_similarity,
    }


CODE_RE = re.compile(r"[a-z0-9]+(?:[-./][a-z0-9]+)*")


def _string(value: object, name: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value):
        raise AdmissionError(f"{name} must be a non-empty string")
    return value


def levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, 1):
        current = [i]
        for j, right_char in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def code_tokens(title: str) -> tuple[str, ...]:
    _string(title, "title", empty=True)
    tokens: set[str] = set()
    for raw in CODE_RE.findall(unicodedata.normalize("NFKC", title).casefold()):
        token = re.sub(r"[-./]", "", raw)
        if (
            3 <= len(token) <= 32
            and re.search(r"[a-z]", token)
            and re.search(r"[0-9]", token)
        ):
            tokens.add(token)
    return tuple(sorted(tokens))


def _letters_digits(token: str) -> tuple[str, str]:
    return (
        "".join(char for char in token if char.isalpha()),
        "".join(char for char in token if char.isdigit()),
    )


def local_code_features(left_title: str, right_title: str) -> dict[str, float]:
    """Return the eight fixed local features, symmetric in A and B."""

    left, right = set(code_tokens(left_title)), set(code_tokens(right_title))
    union, intersection = left | right, left & right
    all_pairs = [(a, b) for a in left for b in right]
    unequal_pairs = [(a, b) for a, b in all_pairs if a != b]
    best_edit = max(
        (1.0 - levenshtein(a, b) / max(len(a), len(b)) for a, b in all_pairs),
        default=0.0,
    )
    same_letters_different_digits = any(
        _letters_digits(a)[0] == _letters_digits(b)[0]
        and _letters_digits(a)[1] != _letters_digits(b)[1]
        for a, b in unequal_pairs
    )
    same_digits_different_letters = any(
        _letters_digits(a)[1] == _letters_digits(b)[1]
        and _letters_digits(a)[0] != _letters_digits(b)[0]
        for a, b in unequal_pairs
    )
    one_edit = any(
        len(a) >= 4 and len(b) >= 4 and levenshtein(a, b) == 1
        for a, b in unequal_pairs
    )
    return {
        "code_left_missing_or_right_missing": float(not left or not right),
        "code_both_present": float(bool(left and right)),
        "code_exact_overlap": float(bool(intersection)),
        "code_set_jaccard": len(intersection) / len(union) if union else 0.0,
        "code_best_edit_similarity": best_edit,
        "code_same_letters_different_digits": float(same_letters_different_digits),
        "code_same_digits_different_letters": float(same_digits_different_letters),
        "code_one_edit_difference": float(one_edit),
    }


NATIVE_FEATURE_NAMES = (
    "native_model_missing_any", "native_model_both_present",
    "native_model_compact_exact", "native_model_edit_similarity",
    "native_model_char3_jaccard", "native_model_same_letters_different_digits",
    "native_model_same_digits_different_letters", "cross_native_title_any",
    "cross_native_title_both", "native_category_both_present",
    "native_category_token_jaccard", "native_category_exact",
)


_TOKENS = re.compile(r"[^\W_]+", re.UNICODE)


def _ngrams(text: str) -> set[str]:
    if not text:
        return set()
    return {text[i:i + 3] for i in range(len(text) - 2)} if len(text) >= 3 else {text}


def native_features(
    left: Mapping, right: Mapping, left_title: str | None, right_title: str | None,
) -> dict[str, float]:
    if set(left) != {"modelno", "category"} or set(right) != {"modelno", "category"}:
        raise AdmissionError("v10 native field schema invalid")
    lm, rm = compact_native(left["modelno"]), compact_native(right["modelno"])
    lb, rb = bool(lm), bool(rm)
    present = lb and rb
    same = lm == rm and present
    ls, rs = _ngrams(lm), _ngrams(rm)
    la, ra = "".join(ch for ch in lm if ch.isalpha()), "".join(ch for ch in rm if ch.isalpha())
    ld, rd = "".join(ch for ch in lm if ch.isdigit()), "".join(ch for ch in rm if ch.isdigit())
    left_codes = set(code_tokens(left_title or ""))
    right_codes = set(code_tokens(right_title or ""))
    cross_left = lb and lm in right_codes
    cross_right = rb and rm in left_codes
    lc, rc = normalize_native(left["category"]), normalize_native(right["category"])
    ct_left = set(_TOKENS.findall(lc or ""))
    ct_right = set(_TOKENS.findall(rc or ""))
    result = {
        "native_model_missing_any": float(not present),
        "native_model_both_present": float(present),
        "native_model_compact_exact": float(same),
        "native_model_edit_similarity": (
            max(0.0, 1.0 - levenshtein(lm, rm) / max(len(lm), len(rm))) if present else 0.0),
        "native_model_char3_jaccard": _jaccard(ls, rs),
        "native_model_same_letters_different_digits": float(bool(
            present and lm != rm and la and la == ra and ld and rd and ld != rd)),
        "native_model_same_digits_different_letters": float(bool(
            present and lm != rm and ld and ld == rd and la and ra and la != ra)),
        "cross_native_title_any": float(cross_left or cross_right),
        "cross_native_title_both": float(cross_left and cross_right),
        "native_category_both_present": float(bool(lc and rc)),
        "native_category_token_jaccard": _jaccard(ct_left, ct_right),
        "native_category_exact": float(bool(lc and rc and lc == rc)),
    }
    if set(result) != set(NATIVE_FEATURE_NAMES) or any(
        not math.isfinite(v) or not 0.0 <= v <= 1.0 for v in result.values()
    ):
        raise AdmissionError("v10 native features invalid")
    return result


def lexical42(
    left_business: Mapping, right_business: Mapping,
    left_native: Mapping, right_native: Mapping, fit_idf: TitleEvidence,
) -> list[float]:
    old22 = fit_idf.pair(left_business, right_business)
    local = local_code_features(left_business["normalized"].get("title") or "",
                                right_business["normalized"].get("title") or "")
    native = native_features(left_native, right_native,
                             left_business["normalized"].get("title"),
                             right_business["normalized"].get("title"))
    result = old22 + [local[name] for name in LOCAL_FEATURE_NAMES] + [
        native[name] for name in NATIVE_FEATURE_NAMES]
    if len(result) != 42:
        raise AdmissionError("v10 lexical width is not 42")
    return result


LEXICAL_EXTRA_NAMES = (
    "title_word_idf_cosine", "title_word_idf_containment", "title_char3_idf_cosine",
    "title_compact_char3_idf_cosine", "title_alphanumeric_jaccard",
    "title_number_jaccard", "title_numbers_both_present", "title_compact_exact",
)
FEATURE_NAMES_42 = (*FEATURE_NAMES, *LEXICAL_EXTRA_NAMES, *LOCAL_FEATURE_NAMES, *NATIVE_FEATURE_NAMES)


def fit_title_evidence(fit_rows):
    """Fit IDF on deduplicated business records from explicitly labelled fit rows.

    Rows contain pair_id, left, right, label, and optionally both
    left_native/right_native. IDs and labels never enter the feature vector.
    """
    from .lexical import TitleEvidence
    if not isinstance(fit_rows, list) or not fit_rows:
        raise AdmissionError("fit IDF requires nonempty fit rows")
    by_hash = {}
    ids = set()
    for row in fit_rows:
        if not isinstance(row, Mapping) or set(row) not in (
            {"pair_id", "left", "right", "label"},
            {"pair_id", "left", "right", "left_native", "right_native", "label"},
        ):
            raise AdmissionError("fit IDF row schema invalid")
        if not isinstance(row["pair_id"], str) or not row["pair_id"] or row["pair_id"] in ids:
            raise AdmissionError("fit IDF pair IDs invalid or repeated")
        ids.add(row["pair_id"])
        if type(row["label"]) is not int or row["label"] not in (0, 1):
            raise AdmissionError("fit label must be integer 0 or 1")
        for position in ("left", "right"):
            business = row[position]
            _validate_business(business)
            by_hash.setdefault(digest_object(business), business)
    records = [{**by_hash[key], "offer_id": key, "split": "fit"} for key in sorted(by_hash)]
    return TitleEvidence.fit(records)


def _validate_business(business):
    if (not isinstance(business, Mapping) or set(business) != {"normalized", "quality"}
            or not isinstance(business["normalized"], Mapping)
            or set(business["normalized"]) != set(SOURCE_FIELDS)
            or not isinstance(business["quality"], Mapping)
            or set(business["quality"]) != {"price", "priceCurrency"}):
        raise AdmissionError("business view schema invalid")
    if any(value is not None and not isinstance(value, str)
           for value in business["normalized"].values()):
        raise AdmissionError("normalized business fields must be text or missing")


def prepare_rows(rows, *, role, fit_idf):
    """Add fixed x42 features to explicit business/native pair rows.

    All roles require pair_id,left,right,left_native,right_native. Only fit
    accepts/requires label. Non-fit rows must not contain labels. The caller
    supplies an existing fit-only TitleEvidence; this function never refits it.
    """
    if role not in {"fit", "dev", "calibration", "evaluation"} or not isinstance(rows, list) or not rows:
        raise AdmissionError("role missing or rows empty")
    expected = {"pair_id", "left", "right", "left_native", "right_native"} | ({"label"} if role == "fit" else set())
    seen = set()
    result = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != expected:
            raise AdmissionError("pair role has missing or extra fields")
        pid = row["pair_id"]
        if not isinstance(pid, str) or not pid or pid in seen:
            raise AdmissionError("pair IDs invalid or repeated")
        seen.add(pid)
        for side in ("left", "right"):
            _validate_business(row[side])
            native = row[side + "_native"]
            if not isinstance(native, Mapping) or set(native) != {"modelno", "category"}:
                raise AdmissionError("native view schema invalid")
            for value in native.values():
                normalize_native(value)
        if role == "fit" and (type(row["label"]) is not int or row["label"] not in (0, 1)):
            raise AdmissionError("fit label must be integer 0 or 1")
        x42 = lexical42(row["left"], row["right"], row["left_native"], row["right_native"], fit_idf)
        result.append({**row, "x42": x42})
    return result
