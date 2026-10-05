"""The fixed field budgets and symmetric RoBERTa pair serialization."""
from __future__ import annotations
from collections.abc import Mapping
import math
from .errors import AdmissionError

FIELD_LIMITS = {"brand": 16, "model": 32, "category": 16, "title": 62}


def _field_tokens(tokenizer, field: str, value: str | None, limit: int, *, first: bool) -> tuple[list[int], bool]:
    prefix = "" if first else "\n"
    text = f"{prefix}{field}: {value if value else 'missing'}"
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not isinstance(tokens, list) or any(type(i) is not int or i < 0 for i in tokens):
        raise AdmissionError("tokenizer output invalid")
    if len(tokens) <= limit:
        return tokens, False
    if field == "title":
        return tokens[:40] + tokens[-22:], True
    return tokens[:limit], True


def record_tokens(tokenizer, business: Mapping, native: Mapping) -> tuple[list[int], dict[str, bool]]:
    if set(native) != {"modelno", "category"}:
        raise AdmissionError("final native serialization schema invalid")
    normalized = business["normalized"]
    values = {"brand": normalized.get("brand"), "model": native["modelno"],
              "category": native["category"], "title": normalized.get("title")}
    ids: list[int] = []
    cut: dict[str, bool] = {}
    for index, (field, limit) in enumerate(FIELD_LIMITS.items()):
        tokens, truncated = _field_tokens(tokenizer, field, values[field], limit,
                                          first=index == 0)
        ids.extend(tokens)
        cut[field] = truncated
    if len(ids) > 126:
        raise AdmissionError("final record token budget exceeded")
    return ids, cut


def pair_tokens(tokenizer, left: list[int], right: list[int]) -> tuple[list[int], list[int]]:
    if any(type(i) is not int or i < 0 for i in left + right):
        raise AdmissionError("final record tokens invalid")
    build = getattr(tokenizer, "build_inputs_with_special_tokens", None)
    if callable(build):
        forward = build(left, right)
        reverse = build(right, left)
    else:
        # Transformers 5.17's slow RobertaTokenizer no longer exposes the
        # sequence-ID builder. Preserve its locked RoBERTa pair template.
        count_special = getattr(tokenizer, "num_special_tokens_to_add", None)
        cls_id = getattr(tokenizer, "cls_token_id", None)
        sep_id = getattr(tokenizer, "sep_token_id", None)
        if (not callable(count_special) or count_special(pair=True) != 4
                or getattr(tokenizer, "cls_token", None) != "<s>"
                or getattr(tokenizer, "sep_token", None) != "</s>"
                or type(cls_id) is not int or type(sep_id) is not int):
            raise AdmissionError("final RoBERTa pair-token API is unsupported")
        forward = [cls_id, *left, sep_id, sep_id, *right, sep_id]
        reverse = [cls_id, *right, sep_id, sep_id, *left, sep_id]
    if (len(forward) != len(left) + len(right) + 4 or len(reverse) != len(forward)
            or len(forward) > 256):
        raise AdmissionError("RoBERTa pair special-token budget invalid")
    return forward, reverse


_ROLES = {"fit", "dev", "calibration", "evaluation"}


_REFERENCES = {"B22", "L30", "S33", "L42"}


_CORE = {"pair_id", "left", "right", "left_native", "right_native", "x42"}


def _admit(rows: object, role: str) -> list[dict]:
    if role not in _ROLES or not isinstance(rows, list) or not rows:
        raise AdmissionError("final worker role missing or unknown")
    expected = _CORE | ({"label"} if role == "fit" else set())
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != expected:
            raise AdmissionError("final worker role has extra or missing fields")
        pid = row["pair_id"]
        if not isinstance(pid, str) or not pid or pid in seen:
            raise AdmissionError("final worker role IDs invalid or repeated")
        seen.add(pid)
        if (not isinstance(row["left"], Mapping)
                or not isinstance(row["right"], Mapping)
                or not isinstance(row["left"].get("normalized"), Mapping)
                or not isinstance(row["right"].get("normalized"), Mapping)
                or not isinstance(row["left_native"], Mapping)
                or not isinstance(row["right_native"], Mapping)
                or set(row["left_native"]) != {"modelno", "category"}
                or set(row["right_native"]) != {"modelno", "category"}):
            raise AdmissionError("final worker business/native fields invalid")
        x42 = row["x42"]
        if (not isinstance(x42, list) or len(x42) != 42
                or any(type(x) not in (float, int) or not math.isfinite(x)
                       or not 0 <= x <= 1 for x in x42)):
            raise AdmissionError("final worker lexical values invalid")
        if role == "fit" and (type(row["label"]) is not int
                              or row["label"] not in (0, 1)):
            raise AdmissionError("final worker fit labels invalid")
    return rows


def reference_rows(rows: object, *, role: str, method: str,
                   semantic_features: object | None = None) -> list[dict]:
    """Shape worker rows; S33 provenance must be verified by the calling host."""
    admitted = _admit(rows, role)
    if method not in _REFERENCES:
        raise AdmissionError("final reference method invalid")
    if method == "S33":
        if (not isinstance(semantic_features, list)
                or [x["pair_id"] for x in semantic_features
                    if isinstance(x, dict) and "pair_id" in x] !=
                   [row["pair_id"] for row in admitted]
                or len(semantic_features) != len(admitted)):
            raise AdmissionError("final S33 semantic role ID mismatch")
        for evidence in semantic_features:
            if (not isinstance(evidence, dict)
                    or set(evidence) != {"pair_id", "semantic3"}
                    or not isinstance(evidence["semantic3"], list)
                    or len(evidence["semantic3"]) != 3
                    or any(type(x) not in (float, int) or not math.isfinite(x)
                           for x in evidence["semantic3"])
                    or any(not -1 <= x <= 1 for x in evidence["semantic3"][:2])
                    or evidence["semantic3"][2] not in (0, 1)):
                raise AdmissionError("final S33 semantic role values invalid")
    elif semantic_features is not None:
        raise AdmissionError("final nonsemantic reference received semantic fields")
    result = []
    for index, row in enumerate(admitted):
        value = {"pair_id": row["pair_id"], "x42": row["x42"]}
        if method == "S33":
            value["semantic3"] = semantic_features[index]["semantic3"]
        if role == "fit":
            value["label"] = row["label"]
        result.append(value)
    return result


def neural_rows(rows: object, *, role: str, tokenizer: object) -> tuple[list[dict], dict]:
    admitted = _admit(rows, role)
    if (getattr(tokenizer, "pad_token_id", None) != 1
            or getattr(tokenizer, "num_special_tokens_to_add", lambda **kw: None)(pair=True) != 4):
        raise AdmissionError("final tokenizer identity or pair template invalid")
    output = []
    cuts = {field: 0 for field in ("brand", "model", "category", "title")}
    for row in admitted:
        left, left_cut = record_tokens(tokenizer, row["left"], row["left_native"])
        right, right_cut = record_tokens(tokenizer, row["right"], row["right_native"])
        ab, ba = pair_tokens(tokenizer, left, right)
        for flag in (left_cut, right_cut):
            for field, truncated in flag.items():
                cuts[field] += int(truncated)
        value = {"pair_id": row["pair_id"], "ab": ab, "ba": ba,
                 "x42": row["x42"]}
        if role == "fit":
            value["label"] = row["label"]
        output.append(value)
    return output, {"role": role, "pairs": len(output), "truncated_record_fields": cuts}



def encode_rows(rows, *, role, tokenizer):
    """Alias of neural_rows: encode admitted feature rows and count truncation."""
    return neural_rows(rows, role=role, tokenizer=tokenizer)
