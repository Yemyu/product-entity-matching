"""Rebuild frozen role inputs from a locally supplied, pinned source archive.

The recipe supplies membership and any original native-field masking. This
module chooses no split, fits no model and certifies no historical exposure.
Product rows and labels are written only to the caller's fresh local output.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

from .data import business_record, native_record
from .errors import ContractError
from .roles import ROLES

SCHEMA = "product-matching-role-reconstruction-v1"
MEMBERS = ("tableA.csv", "tableB.csv", "train.csv", "valid.csv", "test.csv")
PAIR_MEMBERS = MEMBERS[2:]
TABLE_COLUMNS = ["id", "title", "category", "brand", "modelno", "price"]
PAIR_COLUMNS = ["ltable_id", "rtable_id", "label"]
MAX_BYTES = 100 * 1024 * 1024


def canonical_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _fields(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ContractError(name + " fields differ")


def _digest(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ContractError("A lowercase SHA256 digest is required")


def _identity(value, *, fields, name):
    _fields(value, fields, name)
    if type(value["bytes"]) is not int or not 0 < value["bytes"] <= MAX_BYTES:
        raise ContractError(name + " byte count is invalid")
    _digest(value["sha256"])


def _validate_recipe(recipe):
    _fields(recipe, {"schema", "source", "roles"}, "Recipe")
    if recipe["schema"] != SCHEMA:
        raise ContractError("Reconstruction recipe schema differs")
    source = recipe["source"]
    _fields(source, {"archive", "members", "table_columns", "pair_columns"}, "Source")
    _identity(source["archive"], fields={"bytes", "sha256"}, name="Archive")
    _fields(source["members"], MEMBERS, "Source members")
    if source["table_columns"] != TABLE_COLUMNS or source["pair_columns"] != PAIR_COLUMNS:
        raise ContractError("Fixed source column mapping differs")
    parents = set()
    for name in MEMBERS:
        item = source["members"][name]
        _identity(item, fields={"path", "bytes", "sha256"}, name=name)
        value = item["path"]
        if not isinstance(value, str) or "\\" in value or ":" in value or "\0" in value:
            raise ContractError("Source member path is unsafe")
        path = PurePosixPath(value)
        if (str(path) != value or len(path.parts) != 2 or path.name != name
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", path.parts[0]) is None):
            raise ContractError("Source members require one ordinary parent directory")
        parents.add(path.parts[0])
    if len(parents) != 1 or sum(item["bytes"] for item in source["members"].values()) > MAX_BYTES:
        raise ContractError("Source parent or total expanded size differs")
    _fields(recipe["roles"], ROLES, "Roles")
    memberships = set()
    for role in ROLES:
        spec = recipe["roles"][role]
        _fields(spec, {"entries", "count", "expected_business_sha256",
                       "expected_pair_ids_sha256"}, role)
        if (type(spec["count"]) is not int or spec["count"] <= 0
                or not isinstance(spec["entries"], list)
                or len(spec["entries"]) != spec["count"]):
            raise ContractError("Role count and membership differ")
        _digest(spec["expected_business_sha256"])
        _digest(spec["expected_pair_ids_sha256"])
        ids = []
        for entry in spec["entries"]:
            _fields(entry, {"member", "record", "left_missing", "right_missing"}, "Membership")
            if (entry["member"] not in PAIR_MEMBERS or type(entry["record"]) is not int
                    or entry["record"] < 2):
                raise ContractError("Membership needs a source CSV record ordinal")
            identity = (entry["member"], entry["record"])
            if identity in memberships:
                raise ContractError("Role memberships repeat or overlap")
            memberships.add(identity)
            for side in ("left", "right"):
                missing = entry[side + "_missing"]
                if (not isinstance(missing, list)
                        or any(name not in ("modelno", "category") for name in missing)
                        or len(set(missing)) != len(missing)):
                    raise ContractError("Native masks may only set fixed fields to missing")
            ids.append(f"walmart-amazon:structured:{entry['member']}:{entry['record']}")
        if _sha(canonical_bytes(ids)) != spec["expected_pair_ids_sha256"]:
            raise ContractError("Frozen membership order/hash differs")
    return next(iter(parents))


def _hash_stream(stream):
    digest = hashlib.sha256()
    stream.seek(0)
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    stream.seek(0)
    return digest.hexdigest()


def _read_archive(archive, source, parent):
    if archive.is_symlink() or not archive.is_file():
        raise ContractError("Source archive must be a regular, unlinked file")
    try:
        with archive.open("rb") as stream:
            info = os.fstat(stream.fileno())
            if stat.S_IFMT(info.st_mode) != stat.S_IFREG or info.st_size != source["archive"]["bytes"]:
                raise ContractError("Source archive byte count/type differs")
            if _hash_stream(stream) != source["archive"]["sha256"]:
                raise ContractError("Source archive hash differs")
            with zipfile.ZipFile(stream) as bundle:
                expected = {item["path"] for item in source["members"].values()}
                directory = parent + "/"
                seen = set()
                expanded = 0
                for member in bundle.infolist():
                    name = member.filename
                    if name in seen:
                        raise ContractError("Archive repeats a member")
                    seen.add(name)
                    mode = stat.S_IFMT(member.external_attr >> 16)
                    if member.flag_bits & 1:
                        raise ContractError("Encrypted source members are forbidden")
                    if member.is_dir():
                        if name != directory or mode not in (0, stat.S_IFDIR) or member.file_size != 0:
                            raise ContractError("Archive directory entry is unsafe")
                    elif name not in expected or mode not in (0, stat.S_IFREG):
                        raise ContractError("Archive has an unexpected or unsafe member")
                    expanded += member.file_size
                    if expanded > MAX_BYTES:
                        raise ContractError("Expanded archive exceeds 100 MiB")
                if seen - {directory} != expected:
                    raise ContractError("Archive CSV inventory differs")
                payloads = {}
                for name in MEMBERS:
                    identity = source["members"][name]
                    member = bundle.getinfo(identity["path"])
                    if member.file_size != identity["bytes"]:
                        raise ContractError("Source CSV byte count differs: " + name)
                    with bundle.open(member) as part:
                        payload = part.read(identity["bytes"] + 1)
                    if len(payload) != identity["bytes"] or _sha(payload) != identity["sha256"]:
                        raise ContractError("Source CSV hash differs: " + name)
                    payloads[name] = payload
            if _hash_stream(stream) != source["archive"]["sha256"]:
                raise ContractError("Source archive changed while reading")
            return payloads
    except (OSError, zipfile.BadZipFile, RuntimeError, NotImplementedError) as exc:
        raise ContractError("Source archive could not be validated") from exc


def _csv_rows(payload, columns, member):
    try:
        reader = csv.reader(io.StringIO(payload.decode("utf-8-sig"), newline=""), strict=True)
        if next(reader) != columns:
            raise ContractError("Source CSV header differs: " + member)
        rows = []
        for values in reader:
            if len(values) != len(columns):
                raise ContractError("Source CSV row width differs: " + member)
            rows.append(dict(zip(columns, values)))
        return rows
    except (UnicodeDecodeError, csv.Error, StopIteration) as exc:
        raise ContractError("Invalid UTF-8 source CSV: " + member) from exc


def _decode(payloads):
    tables = {}
    for name in MEMBERS[:2]:
        table = {}
        for row in _csv_rows(payloads[name], TABLE_COLUMNS, name):
            identity = row["id"]
            if not identity.strip() or identity in table:
                raise ContractError("Source table IDs are missing or repeated")
            table[identity] = row
        tables[name] = table
    pairs = {}
    endpoint_pairs = set()
    for name in PAIR_MEMBERS:
        rows = _csv_rows(payloads[name], PAIR_COLUMNS, name)
        for row in rows:
            endpoints = (row["ltable_id"], row["rtable_id"])
            if (endpoints[0] not in tables["tableA.csv"]
                    or endpoints[1] not in tables["tableB.csv"]):
                raise ContractError("Source pair references a missing endpoint")
            if row["label"] not in ("0", "1"):
                raise ContractError("Source pair label must be exactly zero or one")
            if endpoints in endpoint_pairs:
                raise ContractError("Source endpoint pairs repeat")
            endpoint_pairs.add(endpoints)
        pairs[name] = rows
    return tables, pairs


def reconstruct_roles(archive_path, recipe, output_path):
    """Validate every input/hash before creating a fresh output directory.

    Record ordinals count parsed CSV records, with the header numbered one.
    Native missing masks preserve the original conservative projection. All
    labels, including fit labels, also get a separate local private file.
    """
    parent = _validate_recipe(recipe)
    archive, output = Path(archive_path), Path(output_path)
    if output.exists() or output.is_symlink():
        raise ContractError("Reconstruction output already exists")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ContractError("Reconstruction output needs an existing ordinary parent")
    payloads = _read_archive(archive, recipe["source"], parent)
    tables, pairs = _decode(payloads)
    outputs, summaries = {}, {}
    for role in ROLES:
        spec = recipe["roles"][role]
        rows, labels, ids = [], [], []
        for entry in spec["entries"]:
            candidates = pairs[entry["member"]]
            index = entry["record"] - 2
            if index >= len(candidates):
                raise ContractError("Membership record is absent from source CSV")
            pair = candidates[index]
            pid = f"walmart-amazon:structured:{entry['member']}:{entry['record']}"
            item = {"pair_id": pid}
            for side, table, endpoint in (("left", "tableA.csv", "ltable_id"),
                                          ("right", "tableB.csv", "rtable_id")):
                raw = tables[table][pair[endpoint]]
                item[side] = business_record(raw)
                native = native_record(raw)
                for name in entry[side + "_missing"]:
                    native[name] = None
                item[side + "_native"] = native
            label = int(pair["label"])
            if role == "fit":
                item["label"] = label
            rows.append(item)
            labels.append({"pair_id": pid, "label": label})
            ids.append(pid)
        business_payload = canonical_bytes(rows)
        if _sha(business_payload) != spec["expected_business_sha256"]:
            raise ContractError("Frozen business bytes differ: " + role)
        outputs[f"roles/{role}-business.json"] = business_payload
        outputs[f"private/{role}-labels.json"] = canonical_bytes(labels)
        summaries[role] = {"count": len(rows), "ordered_pair_ids_sha256": _sha(canonical_bytes(ids))}
    receipt = {"schema": "product-matching-role-reconstruction-receipt-v1",
               "status": "reconstructed_frozen_inputs", "recipe_canonical_sha256": _sha(canonical_bytes(recipe)),
               "source": recipe["source"], "roles": summaries,
               "outputs": {name: {"bytes": len(payload), "sha256": _sha(payload)}
                           for name, payload in outputs.items()},
               "no_new_split": True, "no_training": True, "no_model_inference": True,
               "historical_exposure_certified": False}
    try:
        output.mkdir(mode=0o700, exist_ok=False)
        (output / "roles").mkdir(mode=0o700)
        (output / "private").mkdir(mode=0o700)
        for name, payload in outputs.items():
            with (output / name).open("xb") as target:
                target.write(payload)
            (output / name).chmod(0o600)
        with (output / "RECONSTRUCTION_RECEIPT.json").open("xb") as target:
            target.write(canonical_bytes(receipt))
        (output / "RECONSTRUCTION_RECEIPT.json").chmod(0o600)
    except OSError as exc:
        raise ContractError("Reconstruction write failed; no overwrite or retry was performed") from exc
    return receipt
