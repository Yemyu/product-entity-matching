"""Pure adapters for explicitly supplied product fields; no dataset file access."""
from __future__ import annotations
from collections.abc import Mapping
import hashlib
import json
import unicodedata
from .errors import AdmissionError
from .normalization import normalize_currency, normalize_text, parse_price

_BUSINESS_FIELDS = ("brand", "title", "description", "price", "priceCurrency")
_MISSING = {"", "na", "n/a", "none", "null", "unknown", "not applicable", "-", "--"}


def normalize_native(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise AdmissionError("native field must be text or missing")
    text = " ".join(unicodedata.normalize("NFKC", value).casefold().split())
    return None if text in _MISSING else text


def compact_native(value: str | None) -> str:
    return "".join(ch for ch in (normalize_native(value) or "") if ch.isalnum())


def digest_object(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def business_record(row: Mapping[str, str]) -> dict[str, object]:
    """Build the unchanged normalized business view from explicit raw fields."""
    if not isinstance(row, Mapping):
        raise AdmissionError("business raw record must be a mapping")
    raw = {field: row.get(field) or None for field in _BUSINESS_FIELDS}
    title = raw["title"]
    if not isinstance(title, str):
        raise AdmissionError("source title must be non-empty")
    price = parse_price(raw["price"])
    currency, currency_status = normalize_currency(raw["priceCurrency"])
    return {
        "normalized": {
            "brand": normalize_text(raw["brand"]),
            "title": normalize_text(title),
            "description": normalize_text(raw["description"]),
            "price": price.normalized,
            "priceCurrency": currency,
        },
        "quality": {"price": price.status, "priceCurrency": currency_status},
    }



def native_record(raw):
    """Normalize explicit modelno/category fields; never infer missing values."""
    if not isinstance(raw, Mapping):
        raise AdmissionError("native raw record must be a mapping")
    return {name: normalize_native(raw.get(name)) for name in ("modelno", "category")}


def prepare_rows(rows, *, role, fit_idf):
    """Convenience alias of features.prepare_rows; never fits on these rows."""
    from .features import prepare_rows as prepare
    return prepare(rows, role=role, fit_idf=fit_idf)
