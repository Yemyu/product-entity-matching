"""Conservative, auditable normalization for product offer fields.

The original values are always kept by the pipeline. Normalized values are a
comparison view only; a missing or ambiguous value is never silently guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re
import unicodedata


_PLAIN_DECIMAL = re.compile(
    r"^[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$"
)
_THREE_LETTER_CURRENCY = re.compile(r"^[A-Z]{3}$")
# The source contains a few non-ISO aliases (for example ``LEI``). Keep the
# observed value, but only call a known ISO-4217 code standard.
_KNOWN_CURRENCIES = {
    "AED", "ALL", "ARS", "AUD", "BAM", "BHD", "BMD", "BGN", "BRL", "CAD",
    "CHF", "CNY", "COP", "CZK", "DKK", "EGP", "EUR", "GBP", "GHS", "HKD",
    "HRK", "HUF", "IDR", "INR", "ISK", "JPY", "KES", "KRW", "KWD", "LBP",
    "MDL", "MXN", "MYR", "NGN", "NOK", "NZD", "OMR", "PHP", "PKR", "PLN",
    "QAR", "RON", "RUB", "SAR", "SEK", "SGD", "THB", "TRY", "TZS", "UAH",
    "USD", "VND", "ZAR",
}


def normalize_text(value: object) -> str | None:
    """Return a comparison-safe text view while retaining the raw value elsewhere."""

    if value is None:
        return None
    text = unicodedata.normalize("NFKC", str(value))
    # Format characters include the invisible direction marker observed in a
    # price string. Do not remove digits, punctuation, or letters.
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    text = " ".join(text.split()).casefold()
    return text or None


def normalize_currency(value: object) -> tuple[str | None, str]:
    """Normalize currency spelling and classify whether it is a safe ISO-like code."""

    normalized = normalize_text(value)
    if normalized is None:
        return None, "missing"
    upper = normalized.upper()
    if _THREE_LETTER_CURRENCY.fullmatch(upper) and upper in _KNOWN_CURRENCIES:
        return upper, "standard_code"
    return upper, "nonstandard_code"


@dataclass(frozen=True)
class PriceResult:
    normalized: str | None
    status: str
    reason: str | None


def _decimal_string(value: Decimal) -> str:
    if value == 0:
        return "0"
    text = format(value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def parse_price(value: object) -> PriceResult:
    """Parse only unadorned dot-decimal/scientific values.

    Currency symbols, comma separators, and mixed regional formats are marked
    ambiguous instead of being guessed. This keeps the raw value available for
    a later, separately tested locale parser.
    """

    if value is None or str(value).strip() == "":
        return PriceResult(None, "missing", "empty_value")
    text = unicodedata.normalize("NFKC", str(value))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf").strip()
    if _PLAIN_DECIMAL.fullmatch(text) is None:
        if any(token in text for token in [",", "$", "€", "£", "¥", "₹"]):
            return PriceResult(None, "ambiguous_format", "symbol_or_separator")
        return PriceResult(None, "invalid_format", "not_plain_decimal")
    try:
        decimal_value = Decimal(text)
    except InvalidOperation:
        return PriceResult(None, "invalid_format", "decimal_parse_failed")
    if not decimal_value.is_finite():
        return PriceResult(None, "invalid_format", "non_finite")
    if decimal_value < 0:
        return PriceResult(None, "invalid_format", "negative_value")
    return PriceResult(_decimal_string(decimal_value), "parsed", None)


def field_quality(raw: object, normalized: str | None) -> str:
    """Classify a text field without manufacturing content."""

    if raw is None:
        return "missing"
    if normalized is None:
        return "blank"
    return "present"
