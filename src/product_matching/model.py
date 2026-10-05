"""v10 neural core. Imports heavyweight packages only inside GPU entry points.

No function in this module downloads a model. The caller must supply a sealed
local RoBERTa snapshot and keep labels out of prediction processes.
"""

from __future__ import annotations

import math
import random
import hashlib
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from .errors import AdmissionError

SEEDS = (42, 43, 44)
EPOCHS = 6
MICRO_BATCH = 2
EFFECTIVE_BATCH = 16


def window_loss_scale(n_pairs: int) -> float:
    if type(n_pairs) is not int or not 1 <= n_pairs <= EFFECTIVE_BATCH:
        raise AdmissionError("v10 effective window size invalid")
    return 1.0 / (2 * n_pairs)


def training_steps(n_fit: int) -> int:
    if type(n_fit) is not int or n_fit < 1:
        raise AdmissionError("v10 fit size invalid")
    return EPOCHS * math.ceil(n_fit / EFFECTIVE_BATCH)


def learning_rate_multiplier(step: int, total_steps: int) -> float:
    if (type(step) is not int or type(total_steps) is not int
            or total_steps < 1 or not 0 <= step < total_steps):
        raise AdmissionError("v10 scheduler step invalid")
    warmup = math.ceil(0.1 * total_steps)
    return ((step + 1) / warmup if step < warmup
            else (total_steps - step) / (total_steps - warmup))


def epoch_order(n_fit: int, seed: int, epoch: int) -> list[int]:
    if seed not in SEEDS or type(epoch) is not int or not 1 <= epoch <= EPOCHS:
        raise AdmissionError("v10 seed or epoch invalid")
    order = list(range(n_fit))
    random.Random(seed + 1009 * epoch).shuffle(order)
    return order


def make_model(local_snapshot: str, *, zero_lexical: bool, seed: int):
    """Build C+ or an independently trainable C0 with identical dimensions."""
    if not local_snapshot or seed not in SEEDS or type(zero_lexical) is not bool:
        raise AdmissionError("v10 model initialization arguments invalid")
    import torch
    from transformers import RobertaModel

    torch.manual_seed(seed)
    encoder = RobertaModel.from_pretrained(
        local_snapshot, local_files_only=True, trust_remote_code=False,
        use_safetensors=True, add_pooling_layer=False,
    )
    if encoder.config.hidden_size != 768:
        raise AdmissionError("v10 backbone width differs from locked 768")

    class ProductPair(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.first = torch.nn.Linear(768, 64)
            self.second = torch.nn.Linear(106, 64)
            self.dropout = torch.nn.Dropout(0.1)
            self.final = torch.nn.Linear(64, 1)
            self.zero_lexical = zero_lexical

        def forward(self, input_ids, attention_mask, lexical):
            if lexical.ndim != 2 or lexical.shape[1] != 42:
                raise ValueError("v10 lexical tensor must have 42 columns")
            if input_ids.device.type == "cuda" and self.training:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    hidden = self.encoder(
                        input_ids=input_ids, attention_mask=attention_mask,
                    ).last_hidden_state[:, 0, :]
            else:
                hidden = self.encoder(
                    input_ids=input_ids, attention_mask=attention_mask,
                ).last_hidden_state[:, 0, :]
            hidden = hidden.float()
            lexical = lexical.float()
            if self.zero_lexical:
                lexical = torch.zeros_like(lexical)
            u = torch.nn.functional.gelu(self.first(hidden))
            v = torch.nn.functional.gelu(self.second(torch.cat((u, lexical), dim=1)))
            return self.final(self.dropout(v)).squeeze(-1)

    return ProductPair()


def state_digest(model) -> str:
    """Stable tensor-byte summary for paired initializations and replay records."""
    import torch

    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def save_weights(model, path: Path) -> str:
    """Atomic weights-only checkpoint; no optimizer or private label payload."""
    from safetensors.torch import save_file

    destination = Path(path)
    if (destination.suffix != ".safetensors" or destination.is_symlink()
            or destination.exists()):
        raise AdmissionError("v10 checkpoint path invalid")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tensors = {name: value.detach().contiguous().cpu()
               for name, value in model.state_dict().items()}
    fd, temporary = tempfile.mkstemp(prefix=".v10-", suffix=".safetensors",
                                      dir=destination.parent)
    os.close(fd)
    try:
        save_file(tensors, temporary)
        size = os.path.getsize(temporary)
        if not 0 < size <= 1024 ** 3:
            raise AdmissionError("v10 checkpoint exceeds 1 GiB")
        with open(temporary, "r+b") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        return digest
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_weights(model, path: Path, expected_sha256: str) -> None:
    """For a fresh predict-only process; caller checks model/tokenizer identity."""
    from safetensors.torch import load_file

    source = Path(path)
    if (source.is_symlink() or source.suffix != ".safetensors"
            or not source.is_file() or source.stat().st_size > 1024 ** 3
            or len(expected_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in expected_sha256)):
        raise AdmissionError("v10 checkpoint input invalid")
    with source.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected_sha256:
        raise AdmissionError("v10 checkpoint SHA mismatch")
    model.load_state_dict(load_file(str(source), device="cpu"), strict=True)


def make_optimizer(model):
    import torch

    groups = []
    for prefix, module, lr in (("encoder", model.encoder, 2e-5),
                               ("head", model, 2e-4)):
        for decay in (True, False):
            parameters = []
            for name, parameter in module.named_parameters():
                if prefix == "head" and name.startswith("encoder."):
                    continue
                if not parameter.requires_grad:
                    continue
                no_decay = name.endswith("bias") or "LayerNorm.weight" in name
                if decay != (not no_decay):
                    continue
                parameters.append(parameter)
            if parameters:
                groups.append({"params": parameters, "lr": lr,
                               "base_lr": lr, "weight_decay": 0.01 if decay else 0.0})
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8, fused=True)


def _batch_tensors(rows: Sequence[Mapping], direction: str, device):
    import torch

    if direction not in {"ab", "ba"}:
        raise AdmissionError("v10 direction invalid")
    sequences = [row[direction] for row in rows]
    if any(not isinstance(seq, list) or not 1 <= len(seq) <= 256 for seq in sequences):
        raise AdmissionError("v10 pair tokens invalid")
    width = max(map(len, sequences))
    # RoBERTa's locked pad_token_id is 1 in both train and replay.
    ids = torch.full((len(rows), width), 1, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for index, seq in enumerate(sequences):
        ids[index, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        mask[index, :len(seq)] = 1
    features = torch.tensor([row["x42"] for row in rows], dtype=torch.float32, device=device)
    if features.shape != (len(rows), 42) or not torch.isfinite(features).all().item():
        raise AdmissionError("v10 lexical values invalid")
    return ids, mask, features


def train_epoch(model, optimizer, rows: Sequence[Mapping], *, seed: int,
                epoch: int, global_step: int, on_window=None) -> int:
    """Sequential AB/BA BCE, normalized by actual window pairs including tail."""
    import torch

    if not torch.cuda.is_available() or next(model.parameters()).device.type != "cuda":
        raise AdmissionError("v10 training requires the admitted CUDA device")
    n_fit = len(rows)
    if n_fit < 1 or global_step != (epoch - 1) * math.ceil(n_fit / EFFECTIVE_BATCH):
        raise AdmissionError("v10 epoch or step prefix invalid")
    total_steps = training_steps(n_fit)
    model.train()
    order = epoch_order(n_fit, seed, epoch)
    for start in range(0, n_fit, EFFECTIVE_BATCH):
        torch.cuda.synchronize()
        window_started = time.monotonic()
        window = [rows[index] for index in order[start:start + EFFECTIVE_BATCH]]
        n_window = len(window)
        optimizer.zero_grad(set_to_none=True)
        for micro_start in range(0, n_window, MICRO_BATCH):
            micro = window[micro_start:micro_start + MICRO_BATCH]
            labels = [row["label"] for row in micro]
            if any(type(label) is not int or label not in (0, 1) for label in labels):
                raise AdmissionError("v10 training labels invalid")
            target = torch.tensor(labels, dtype=torch.float32, device="cuda")
            for direction in ("ab", "ba"):
                ids, mask, features = _batch_tensors(micro, direction, "cuda")
                logits = model(ids, mask, features).float()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, target, reduction="sum") * window_loss_scale(n_window)
                if not torch.isfinite(loss).item():
                    raise AdmissionError("v10 nonfinite training loss")
                loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0,
                                              error_if_nonfinite=True)
        if not torch.isfinite(norm).item():
            raise AdmissionError("v10 nonfinite gradient norm")
        multiplier = learning_rate_multiplier(global_step, total_steps)
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * multiplier
        optimizer.step()
        global_step += 1
        torch.cuda.synchronize()
        if torch.cuda.max_memory_reserved() > int(5.5 * 1024 ** 3):
            raise AdmissionError("v10 GPU reserved memory limit reached")
        if on_window is not None:
            on_window({"epoch": epoch, "update": global_step, "window_pairs": n_window,
                       "seconds": time.monotonic() - window_started,
                       "reserved_peak_bytes": torch.cuda.max_memory_reserved()})
    if global_step != epoch * math.ceil(n_fit / EFFECTIVE_BATCH):
        raise AdmissionError("v10 optimizer step count mismatch")
    if any(not torch.isfinite(parameter).all().item() for parameter in model.parameters()):
        raise AdmissionError("v10 nonfinite trained parameter")
    return global_step


def predict_rows(model, rows: Sequence[Mapping], *, pad_token_id: int, on_batch=None) -> list[dict]:
    """GPU FP32, batch two, sigmoid(mean direction logits); no labels read."""
    import torch

    if not torch.cuda.is_available() or next(model.parameters()).device.type != "cuda":
        raise AdmissionError("v10 prediction requires the admitted CUDA device")
    if type(pad_token_id) is not int or pad_token_id != 1:
        raise AdmissionError("v10 RoBERTa pad token differs from locked value 1")
    if any("label" in row for row in rows):
        raise AdmissionError("v10 predict-only process received a label field")
    expected_ids = [row["pair_id"] for row in rows]
    if len(set(expected_ids)) != len(expected_ids):
        raise AdmissionError("v10 duplicate prediction IDs")
    model.eval()
    result = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        for start in range(0, len(rows), MICRO_BATCH):
            torch.cuda.synchronize()
            batch_started = time.monotonic()
            micro = rows[start:start + MICRO_BATCH]
            logits = []
            for direction in ("ab", "ba"):
                ids, mask, features = _batch_tensors(micro, direction, "cuda")
                ids[mask == 0] = pad_token_id
                logits.append(model(ids, mask, features).float())
            probabilities = torch.sigmoid((logits[0] + logits[1]) / 2)
            if not torch.isfinite(probabilities).all().item():
                raise AdmissionError("v10 nonfinite prediction")
            result.extend({"pair_id": row["pair_id"], "score": float(score)}
                          for row, score in zip(micro, probabilities.tolist()))
            torch.cuda.synchronize()
            if torch.cuda.max_memory_reserved() > int(5.5 * 1024 ** 3):
                raise AdmissionError("v10 GPU reserved memory limit reached during prediction")
            if on_batch is not None:
                on_batch({"pairs": len(micro), "seconds": time.monotonic() - batch_started,
                          "reserved_peak_bytes": torch.cuda.max_memory_reserved()})
    return result
