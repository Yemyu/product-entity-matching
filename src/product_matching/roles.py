"""Join explicit frozen pair views and native sidecars without choosing a split.

Membership is supplied by the caller. This is not a new split algorithm or a
claim about historical entity leakage. Non-fit views must be label-free before
they reach this module; unknown fields are rejected, not silently stripped.
"""
from collections.abc import Mapping
from .errors import ContractError
from .data import native_record
from .features import _validate_business

ROLES = ("fit", "dev", "calibration", "evaluation")
NATIVE_FIELDS = {"pair_id", "left_modelno", "right_modelno",
                 "left_category", "right_category"}


def project_role(views, sidecars, kept_ids, *, role):
    if role not in ROLES or not isinstance(views, list) or not views:
        raise ContractError("A known role and nonempty pair views are required")
    if (not isinstance(kept_ids, list) or not kept_ids
            or any(not isinstance(pid, str) or not pid for pid in kept_ids)
            or len(set(kept_ids)) != len(kept_ids)):
        raise ContractError("Frozen membership IDs must be nonempty and unique")
    expected = {"pair_id", "left", "right"} | ({"label"} if role == "fit" else set())
    by_id = {}
    for row in views:
        if not isinstance(row, Mapping) or set(row) != expected:
            raise ContractError("View fields differ; non-fit labels are forbidden")
        pid = row["pair_id"]
        if not isinstance(pid, str) or not pid or pid in by_id:
            raise ContractError("View pair IDs must be nonempty and unique")
        for side in ("left", "right"):
            _validate_business(row[side])
        if role == "fit" and (type(row["label"]) is not int or row["label"] not in (0, 1)):
            raise ContractError("Fit labels must be integer zero or one")
        by_id[pid] = row
    kept = set(kept_ids)
    if [pid for pid in by_id if pid in kept] != kept_ids:
        raise ContractError("Membership must be the exact ordered view subsequence")
    if (not isinstance(sidecars, list) or len(sidecars) != len(kept_ids)
            or any(not isinstance(row, Mapping) or set(row) != NATIVE_FIELDS for row in sidecars)
            or [row["pair_id"] for row in sidecars] != kept_ids):
        raise ContractError("Native sidecars must match frozen IDs and order")
    result = []
    for pid, fields in zip(kept_ids, sidecars):
        row = by_id[pid]
        result.append({**row,
            "left_native": native_record({"modelno": fields["left_modelno"],
                                           "category": fields["left_category"]}),
            "right_native": native_record({"modelno": fields["right_modelno"],
                                            "category": fields["right_category"]})})
    return result


def project_roles(views, sidecars, kept_ids):
    """Project all four roles and reject overlapping pair IDs.

Pair disjointness alone does not establish entity-disjointness.
"""
    if any(not isinstance(value, Mapping) or set(value) != set(ROLES)
           for value in (views, sidecars, kept_ids)):
        raise ContractError("Exactly four roles must be supplied")
    result, seen = {}, set()
    for role in ROLES:
        result[role] = project_role(views[role], sidecars[role], kept_ids[role], role=role)
        ids = set(kept_ids[role])
        if seen & ids:
            raise ContractError("Pair IDs overlap across roles")
        seen |= ids
    return result
