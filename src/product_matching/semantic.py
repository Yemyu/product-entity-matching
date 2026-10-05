"""Fixed local MiniLM features for the S33 reference, never a model fit.

The numerical preprocessing, pooling, dimensions and CUDA float32 execution
match the research reference. Only an explicit local snapshot can be loaded;
no package import or constructor downloads a model.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from .normalization import normalize_text
from .errors import AdmissionError

MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
PREPROCESSING_VERSION = "pem-v09-title-mask-v1"
DIMENSION = 384
BATCH_SIZE = 16
MAX_LENGTH = 128
CODE_RE = re.compile(r"[a-z0-9]+(?:[-./][a-z0-9]+)*")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()

def mask_code_spans(title: str | None) -> str:
    """Remove only the original spans of M0 candidate code tokens."""
    normalized = normalize_text(title) or ""
    if not normalized:
        return ""
    cursor = 0
    parts = []
    for match in CODE_RE.finditer(normalized):
        raw = match.group()
        token = re.sub(r"[-./]", "", raw)
        if (3 <= len(token) <= 32 and re.search(r"[a-z]", token)
                and re.search(r"[0-9]", token)):
            parts.append(normalized[cursor:match.start()])
            parts.append(" ")
            cursor = match.end()
    parts.append(normalized[cursor:])
    return " ".join("".join(parts).split())


def _title(record: object) -> str:
    if not isinstance(record, Mapping) or set(record) != {"normalized", "quality"}:
        raise AdmissionError("semantic input requires a canonical business record")
    normalized, quality = record["normalized"], record["quality"]
    if (not isinstance(normalized, Mapping) or set(normalized) !=
            {"brand", "title", "description", "price", "priceCurrency"}
            or not isinstance(quality, Mapping)
            or set(quality) != {"price", "priceCurrency"}):
        raise AdmissionError("semantic input business schema mismatch")
    title = normalized["title"]
    if title is not None and not isinstance(title, str):
        raise AdmissionError("semantic title must be text or missing")
    return normalize_text(title) or ""


def _unit_vectors(vectors: object, expected_count: int) -> list[list[float]]:
    if not isinstance(vectors, Sequence) or isinstance(vectors, (str, bytes)) or len(vectors) != expected_count:
        raise AdmissionError("MiniLM returned an invalid vector count")
    checked = []
    for vector in vectors:
        if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)) or len(vector) != DIMENSION:
            raise AdmissionError("MiniLM vectors must have 384 dimensions")
        values = []
        for raw in vector:
            if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
                raise AdmissionError("MiniLM vectors must be finite numbers")
            values.append(float(raw))
        norm = math.sqrt(math.fsum(value * value for value in values))
        if not 0.9999 <= norm <= 1.0001:
            raise AdmissionError("MiniLM vectors must be L2-normalized")
        checked.append(values)
    return checked


def _cosine(left: str, right: str, vectors: Mapping[str, list[float]]) -> float:
    if not left or not right:
        return 0.0
    value = math.fsum(a * b for a, b in zip(vectors[left], vectors[right]))
    return max(-1.0, min(1.0, value))


def _mean_pool_only(pooling: object) -> bool:
    mode = getattr(pooling, "pooling_mode", None)
    if mode is not None:
        return mode == "mean"
    return bool(getattr(pooling, "pooling_mode_mean_tokens", False)) and not any(
        getattr(pooling, name, False) for name in (
            "pooling_mode_cls_token", "pooling_mode_max_tokens",
            "pooling_mode_mean_sqrt_len_tokens", "pooling_mode_weightedmean_tokens",
            "pooling_mode_lasttoken",
        )
    )


class FixedMiniLMEncoder:
    """Pinned local MiniLM on CUDA; no fit, backward, or network fallback."""

    revision = MODEL_REVISION
    dimension = DIMENSION

    def __init__(self, model_dir: str | Path):
        model_path = Path(model_dir).resolve()
        if not model_path.is_dir() or model_path.name != MODEL_REVISION:
            raise AdmissionError("fixed MiniLM revision directory is unavailable")
        if os.environ.get("PYTHONHASHSEED") != "42":
            raise AdmissionError("PYTHONHASHSEED must be fixed before the MiniLM process starts")
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            if os.environ.get(name) != "1":
                raise AdmissionError(f"{name} must be one thread before MiniLM starts")
        workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if workspace != ":4096:8":
            raise AdmissionError("CUBLAS_WORKSPACE_CONFIG differs from the frozen setting")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise AdmissionError("the pinned GPU environment lacks MiniLM dependencies") from exc
        if not torch.cuda.is_available() or torch.version.cuda != "12.6":
            raise AdmissionError("frozen MiniLM CUDA runtime is unavailable")
        torch.set_num_threads(1)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            model = SentenceTransformer(
                str(model_path), device="cuda", local_files_only=True,
                trust_remote_code=False,
            )
        except Exception as exc:
            raise AdmissionError(f"fixed MiniLM could not load offline: {exc}") from exc
        model.max_seq_length = MAX_LENGTH
        model.to(device="cuda", dtype=torch.float32)
        model.eval()
        parameters = list(model.parameters())
        if not parameters:
            raise AdmissionError("fixed MiniLM has no parameters")
        for parameter in parameters:
            parameter.requires_grad_(False)
            if parameter.dtype != torch.float32 or parameter.device.type != "cuda":
                raise AdmissionError("fixed MiniLM parameters must remain CUDA float32")
        pooling = [module for module in model.modules() if module.__class__.__name__ == "Pooling"]
        if len(pooling) != 1 or not _mean_pool_only(pooling[0]):
            raise AdmissionError("fixed MiniLM must use attention-mask mean pooling only")
        if model.get_sentence_embedding_dimension() != DIMENSION:
            raise AdmissionError("fixed MiniLM embedding dimension differs from M0")
        self._model = model
        self._torch = torch
        self.last_truncated_count = 0

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not isinstance(texts, Sequence) or isinstance(texts, (str, bytes)) or any(
            not isinstance(text, str) or not text for text in texts
        ):
            raise AdmissionError("MiniLM requires non-empty title texts")
        self.last_truncated_count = 0
        tokenizer = self._model.tokenizer
        for text in texts:
            tokenized = tokenizer(text, add_special_tokens=True, truncation=False, return_length=True)
            length = tokenized["length"]
            if isinstance(length, list):
                length = length[0]
            self.last_truncated_count += int(length > MAX_LENGTH)
        with self._torch.inference_mode():
            vectors = self._model.encode(
                list(texts), batch_size=BATCH_SIZE, show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=True,
                precision="float32",
            )
        if str(getattr(vectors, "dtype", "")) != "float32":
            raise AdmissionError("MiniLM output precision differs from M0")
        return _unit_vectors(vectors.tolist(), len(texts))


def semantic3_with_vectors_for_role(rows: list[Mapping], *, role: str,
                                    encoder: object) -> tuple[list[dict], dict, list[dict]]:
    if role not in {"fit", "dev", "calibration", "evaluation"}:
        raise AdmissionError("unknown v10 semantic role")
    if (getattr(encoder, "revision", None) != MODEL_REVISION
            or getattr(encoder, "dimension", None) != DIMENSION):
        raise AdmissionError("v10 MiniLM identity differs from frozen reference")
    if not isinstance(rows, list) or not rows:
        raise AdmissionError("v10 semantic rows empty")
    expected = {"pair_id", "left", "right"} | ({"label"} if role == "fit" else set())
    prepared, texts, ids = [], set(), set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != expected:
            raise AdmissionError("v10 semantic role fields invalid; protected labels forbidden")
        pid = row["pair_id"]
        if not isinstance(pid, str) or not pid or pid in ids:
            raise AdmissionError("v10 semantic pair ID invalid or repeated")
        ids.add(pid)
        if role == "fit" and (type(row["label"]) is not int or row["label"] not in (0, 1)):
            raise AdmissionError("v10 semantic fit label invalid")
        left, right = _title(row["left"]), _title(row["right"])
        masked_left, masked_right = mask_code_spans(left), mask_code_spans(right)
        prepared.append((pid, left, right, masked_left, masked_right))
        texts.update(text for text in (left, right, masked_left, masked_right) if text)
    ordered = sorted(texts)
    vectors = _unit_vectors(encoder.encode(ordered), len(ordered)) if ordered else []
    by_text = dict(zip(ordered, vectors))

    def cosine(left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        value = math.fsum(a * b for a, b in zip(by_text[left], by_text[right]))
        return max(-1.0, min(1.0, value))

    result = [{"pair_id": pid, "semantic3": [
        cosine(left, right), cosine(masked_left, masked_right),
        float(not masked_left or not masked_right),
    ]} for pid, left, right, masked_left, masked_right in prepared]
    vector_evidence = [{"text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "vector": vector}
                       for text, vector in zip(ordered, vectors)]
    evidence = hashlib.sha256(canonical(vector_evidence)).hexdigest()
    receipt = {"role": role, "pairs": len(result), "unique_encoded_texts": len(ordered),
               "model_revision": MODEL_REVISION,
               "preprocessing_version": PREPROCESSING_VERSION,
               "truncated_texts": getattr(encoder, "last_truncated_count", None),
               "role_cache_sha256": evidence}
    return result, receipt, vector_evidence


def semantic3_for_role(rows: list[Mapping], *, role: str,
                       encoder: object) -> tuple[list[dict], dict]:
    features, receipt, _ = semantic3_with_vectors_for_role(
        rows, role=role, encoder=encoder)
    return features, receipt


