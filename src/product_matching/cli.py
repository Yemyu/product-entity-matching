"""Explicit file-based commands for the final matching experiment.

Importing this module does not load a model, download files or start training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from .errors import ContractError

MODEL_ID = "FacebookAI/roberta-base"
MODEL_REVISION = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
SEEDS = (42, 43, 44)
JSON_LIMIT = 128 * 1024 ** 2
FINAL_CHECKPOINTS = (
    "52eaebccc61b74ce50268c11af9c65b50b7b71c354a0b60817870e902048cdd9",
    "926d69653daabb941dab6eb213376ea4a2f850267c3b770e586e8ffaef414fd5",
    "d68636be2327bacdb5d3557e553488efd1ab8fa4326064bab96c81a8250cef35",
)
ABLATION_CHECKPOINTS = (
    "c50d1aeda92048b73dd28e6ab1f0f527e9183d0e32419ba3d9a75bc7cf77adf9",
    "cc02a3dc6a071aaa473564788c0f68c1bb9f7a3bcb35093b74bf4399eca3e7f2",
    "8e225727d772ed4e472eef4ed36431ae0b0fbf735c65751773d770866a9b0223",
)


def read_json_with_digest(path):
    source = Path(path)
    if source.is_symlink() or not source.is_file() or source.stat().st_size > JSON_LIMIT:
        raise ContractError("Input must be a regular JSON file under 128 MiB")
    def invalid(value):
        raise ContractError(f"Nonfinite JSON value: {value}")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ContractError("Duplicate JSON field")
            result[key] = value
        return result
    raw = source.read_bytes()
    if len(raw) > JSON_LIMIT:
        raise ContractError("JSON input grew beyond its size limit")
    value = json.loads(raw.decode("utf-8"), parse_constant=invalid, object_pairs_hook=unique)
    return value, hashlib.sha256(raw).hexdigest()


def read_json(path):
    return read_json_with_digest(path)[0]


def write_json(path, value):
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise ContractError("Output already exists; choose a new output path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(raw)


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fresh_directory(path):
    output = Path(path)
    if output.exists() or output.is_symlink():
        raise ContractError("Output directory already exists; choose a new path")
    output.mkdir(parents=True)
    return output


def validate_neural_rows(rows, *, training):
    if not isinstance(rows, list) or not rows:
        raise ContractError("A nonempty prepared row list is required")
    fields = {"pair_id", "ab", "ba", "x42"} | ({"label"} if training else set())
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != fields:
            raise ContractError("Prepared fields differ; prediction rows must not contain labels")
        pid = row["pair_id"]
        if not isinstance(pid, str) or not pid or pid in seen:
            raise ContractError("Pair IDs must be nonempty and unique")
        seen.add(pid)
        for direction in ("ab", "ba"):
            tokens = row[direction]
            if (not isinstance(tokens, list) or not 1 <= len(tokens) <= 256
                    or any(type(t) is not int or not 0 <= t < 50265 for t in tokens)):
                raise ContractError("Invalid RoBERTa token sequence")
        features = row["x42"]
        if (not isinstance(features, list) or len(features) != 42
                or any(type(x) not in (int, float) or not math.isfinite(x)
                       or not 0 <= x <= 1 for x in features)):
            raise ContractError("Expected 42 finite features in [0,1]")
        if training and (type(row["label"]) is not int or row["label"] not in (0, 1)):
            raise ContractError("Training labels must be integer zero or one")
    if training and {row["label"] for row in rows} != {0, 1}:
        raise ContractError("Training input must contain both classes")
    return rows


def snapshot_manifest(snapshot):
    root = Path(snapshot).resolve()
    if not root.is_dir():
        raise ContractError("Supply a local RoBERTa snapshot directory")
    names = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
             "special_tokens_map.json", "vocab.json", "merges.txt", "added_tokens.json")
    members = []
    for name in names:
        path = root / name
        if path.is_symlink():
            raise ContractError("Snapshot members must not be symlinks")
        if path.is_file():
            members.append({"path": name, "bytes": path.stat().st_size, "sha256": file_sha(path)})
    if not {"config.json", "model.safetensors", "tokenizer.json"} <= {x["path"] for x in members}:
        raise ContractError("Snapshot requires config, safetensors and fast tokenizer files")
    config = read_json(root / "config.json")
    if (config.get("model_type") != "roberta" or config.get("hidden_size") != 768
            or config.get("num_hidden_layers") != 12 or config.get("vocab_size") != 50265):
        raise ContractError("Snapshot architecture differs from the fixed model")
    return {"schema": "product-matching-snapshot-v1", "model_id": MODEL_ID,
            "revision": MODEL_REVISION, "files": sorted(members, key=lambda x: x["path"])}


def verify_snapshot(snapshot, manifest, *, require_fixed=False):
    expected = read_json(manifest)
    actual = snapshot_manifest(snapshot)
    if expected != actual:
        raise ContractError("Snapshot files differ from the supplied integrity manifest")
    if require_fixed:
        from .snapshot import PINNED_SNAPSHOT
        if actual != PINNED_SNAPSHOT:
            raise ContractError("Snapshot differs from the archived fixed model revision")
    return actual


def prepare(args):
    from .features import fit_title_evidence, prepare_rows
    from .lexical import TitleEvidence
    from .serialization import encode_rows
    # Validation and label separation precede importing the tokenizer runtime.
    rows = read_json(args.rows)
    if args.role == "fit":
        if args.idf_state is not None:
            raise ContractError("Fit must create its own IDF state")
        idf = fit_title_evidence(rows)
    else:
        if args.idf_state is None:
            raise ContractError("Non-fit preparation requires the saved fit-only IDF state")
        idf = TitleEvidence.from_payload(read_json(args.idf_state))
    prepared = prepare_rows(rows, role=args.role, fit_idf=idf)
    verify_snapshot(args.snapshot, args.snapshot_manifest, require_fixed=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.snapshot, local_files_only=True, trust_remote_code=False)
    neural, truncation = encode_rows(prepared, role=args.role, tokenizer=tokenizer)
    validate_neural_rows(neural, training=args.role == "fit")
    output = fresh_directory(args.output)
    write_json(output / "features.json", prepared)
    write_json(output / "neural.json", neural)
    write_json(output / "truncation.json", truncation)
    if args.role == "fit":
        write_json(output / "idf.json", idf.payload())


def project_role(args):
    from .roles import project_role as project
    rows = project(read_json(args.views), read_json(args.sidecars),
                   read_json(args.kept_ids), role=args.role)
    write_json(args.output, rows)


def reconstruct_roles(args):
    from .source import reconstruct_roles as reconstruct
    receipt = reconstruct(args.source_archive, read_json(args.recipe), args.output)
    print(json.dumps({"status": receipt["status"], "output": args.output,
                      "training_runs": 0, "model_inference_runs": 0}))


def _configured_gpu(seed):
    from .runtime import configure
    import torch
    # Set deterministic state before any query that might initialize CUDA.
    runtime = configure(torch, seed)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ContractError("This fixed experiment requires a CUDA GPU supporting BF16")
    return torch, runtime


def _gpu_environment(seed):
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(seed)
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    package_parent = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = package_parent + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def _ensure_process_environment(seed, argv):
    if (os.environ.get("PYTHONHASHSEED") != str(seed)
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"):
        os.execve(sys.executable, [sys.executable, "-m", "product_matching.cli", *argv],
                  _gpu_environment(seed))


def train(args):
    from . import model
    rows = validate_neural_rows(read_json(args.fit), training=True)
    snapshot = verify_snapshot(args.snapshot, args.snapshot_manifest, require_fixed=True)
    if Path(args.output).exists():
        raise ContractError("Training output already exists")
    torch, runtime = _configured_gpu(args.seed)
    output = fresh_directory(args.output)
    started = time.monotonic()
    arm_zero = args.arm == "c-zero"
    epochs = 4 if arm_zero else 6
    receipt = {"schema": "product-matching-training-v1", "status": "running",
               "arm": args.arm, "seed": args.seed, "epochs": epochs,
               "scheduler_epochs": 6, "fit_sha256": file_sha(args.fit),
               "snapshot": snapshot, "runtime": runtime, "checkpoints": []}
    try:
        network = model.make_model(args.snapshot, zero_lexical=arm_zero, seed=args.seed).cuda()
        optimizer = model.make_optimizer(network)
        step = 0
        def progress(record):
            if time.monotonic() - started > 3660:
                raise ContractError("Per-seed runtime exceeded 3660 seconds; no automatic retry")
            with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        for epoch in range(1, epochs + 1):
            step = model.train_epoch(network, optimizer, rows, seed=args.seed,
                                     epoch=epoch, global_step=step, on_window=progress)
            path = output / f"epoch-{epoch}.safetensors"
            digest = model.save_weights(network, path)
            receipt["checkpoints"].append({"epoch": epoch, "path": path.name, "sha256": digest})
        receipt.update(status="completed", applied_updates=step)
    except BaseException as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        receipt["seconds"] = time.monotonic() - started
        write_json(output / "run.json", receipt)


def predict_one(args):
    from . import model
    rows = validate_neural_rows(read_json(args.rows), training=False)
    snapshot = verify_snapshot(args.snapshot, args.snapshot_manifest, require_fixed=True)
    if Path(args.output).exists():
        raise ContractError("Prediction output already exists")
    torch, runtime = _configured_gpu(args.seed)
    network = model.make_model(args.snapshot, zero_lexical=args.arm == "c-zero", seed=args.seed)
    model.load_weights(network, Path(args.checkpoint), args.sha256)
    network.cuda()
    scores = model.predict_rows(network, rows, pad_token_id=1)
    write_json(args.output, {"schema": "product-matching-prediction-v1", "seed": args.seed,
               "arm": args.arm, "epoch": args.epoch,
               "rows_sha256": file_sha(args.rows), "checkpoint_sha256": args.sha256,
               "snapshot": snapshot, "runtime": runtime, "scores": scores})


def score_rows(value):
    if isinstance(value, dict) and isinstance(value.get("scores"), list):
        return value["scores"]
    if isinstance(value, list):
        return value
    raise ContractError("Expected score rows or a prediction document")


def ensemble(args):
    from .metrics import mean_scores
    documents = [read_json(path) for path in args.scores]
    if any(not isinstance(d, dict) for d in documents):
        raise ContractError("Ensemble inputs must identify their seeds")
    by_seed = {}
    arms = {doc.get("arm") for doc in documents}
    if len(arms) != 1 or not arms <= {None, "c-plus", "c-zero"}:
        raise ContractError("Do not mix neural arms in an ensemble")
    for doc in documents:
        seed = doc.get("seed")
        if type(seed) is not int or seed not in SEEDS or seed in by_seed:
            raise ContractError("Ensemble requires each fixed seed exactly once")
        by_seed[seed] = score_rows(doc)
    expected_ids = [r["pair_id"] for r in by_seed[42]]
    scores = mean_scores(expected_ids, by_seed)
    write_json(args.output, {"schema": "product-matching-ensemble-v1", "seeds": list(SEEDS),
                            "arm": next(iter(arms)),
                            "aggregation": "equal_probability_mean", "scores": scores})


def validate_predictions(args):
    """Check transferred C+/L42 scores against explicit IDs before any label join."""
    from .metrics import align_scores
    pair_ids, pair_ids_sha = read_json_with_digest(args.pair_ids)
    if (not isinstance(pair_ids, list) or not pair_ids
            or any(not isinstance(pid, str) or not pid for pid in pair_ids)
            or len(pair_ids) != len(set(pair_ids))):
        raise ContractError("Pair IDs must be a nonempty list of unique nonempty strings")
    candidate, candidate_sha = read_json_with_digest(args.candidate)
    if (not isinstance(candidate, dict)
            or set(candidate) != {"schema", "seeds", "arm", "aggregation", "scores"}
            or candidate["schema"] != "product-matching-ensemble-v1"
            or not isinstance(candidate["seeds"], list)
            or any(type(seed) is not int for seed in candidate["seeds"])
            or candidate["seeds"] != list(SEEDS)
            or candidate["arm"] != "c-plus"
            or candidate["aggregation"] != "equal_probability_mean"):
        raise ContractError("Candidate must be the fixed three-seed C+ ensemble document")
    reference, reference_sha = read_json_with_digest(args.reference)
    if (not isinstance(reference, dict)
            or set(reference) != {"schema", "method", "role", "rows_sha256", "scores"}
            or reference["schema"] != "product-matching-reference-prediction-v1"
            or reference["method"] != "L42" or reference["role"] != "evaluation"
            or not isinstance(reference["rows_sha256"], str)
            or len(reference["rows_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in reference["rows_sha256"])):
        raise ContractError("Reference must be an evaluation L42 prediction document")
    align_scores(pair_ids, candidate["scores"])
    align_scores(pair_ids, reference["scores"])
    for path, digest in ((args.pair_ids, pair_ids_sha), (args.candidate, candidate_sha),
                         (args.reference, reference_sha)):
        source = Path(path)
        if source.is_symlink() or not source.is_file() or file_sha(source) != digest:
            raise ContractError("Prediction check input changed after it was read")
    write_json(args.output, {"schema": "product-matching-prediction-check-v1", "status": "passed",
                            "count": len(pair_ids), "pair_ids_sha256": pair_ids_sha,
                            "candidate_sha256": candidate_sha, "reference_sha256": reference_sha,
                            "label_values_read": False, "metrics_computed": False})


def checkpoint_identity(args):
    frozen = FINAL_CHECKPOINTS if args.arm == "c-plus" else ABLATION_CHECKPOINTS
    if tuple(args.sha256) == frozen:
        if args.epoch != 4:
            raise ContractError("Archived final checkpoints are epoch 4")
        return True
    metadata_paths = args.checkpoint_metadata
    if metadata_paths is None:
        raise ContractError("Recreated weights require the three training run.json files")
    fits = set()
    for seed, digest, path in zip(SEEDS, args.sha256, metadata_paths):
        document = read_json(path)
        if (not isinstance(document, dict)
                or document.get("schema") != "product-matching-training-v1"
                or document.get("status") != "completed" or document.get("seed") != seed
                or document.get("arm") != args.arm):
            raise ContractError("Training checkpoint metadata arm/seed/status differs")
        matching = [c for c in document.get("checkpoints", [])
                    if isinstance(c, dict) and c.get("epoch") == args.epoch and c.get("sha256") == digest]
        if len(matching) != 1:
            raise ContractError("Checkpoint digest does not identify the requested epoch")
        fit = document.get("fit_sha256")
        if not isinstance(fit, str) or len(fit) != 64 or any(c not in "0123456789abcdef" for c in fit):
            raise ContractError("Training fit identity is absent")
        fits.add(fit)
    if len(fits) != 1:
        raise ContractError("Ensemble checkpoints were trained on different fit inputs")
    return False


def predict(args):
    from .metrics import align_scores
    document, rows_sha = read_json_with_digest(args.rows)
    rows = validate_neural_rows(document, training=False)
    expected_ids = [row["pair_id"] for row in rows]
    archived = checkpoint_identity(args)
    verify_snapshot(args.snapshot, args.snapshot_manifest, require_fixed=True)
    for checkpoint, digest in zip(args.checkpoints, args.sha256):
        path = Path(checkpoint)
        if path.is_symlink() or not path.is_file() or file_sha(path) != digest:
            raise ContractError("Checkpoint integrity mismatch")
    output = fresh_directory(args.output)
    inputs = []
    for seed, checkpoint, digest in zip(SEEDS, args.checkpoints, args.sha256):
        if file_sha(args.rows) != rows_sha:
            raise ContractError("Prediction input changed after it was frozen")
        target = output / f"seed-{seed}.json"
        command = [sys.executable, "-m", "product_matching.cli", "_predict-one",
                   "--rows", args.rows, "--snapshot", args.snapshot,
                   "--snapshot-manifest", args.snapshot_manifest,
                   "--seed", str(seed), "--checkpoint", checkpoint,
                   "--sha256", digest, "--arm", args.arm, "--epoch", str(args.epoch),
                   "--output", str(target)]
        # Fresh process per seed preserves the original inference environment.
        completed = subprocess.run(command, env=_gpu_environment(seed), check=False)
        if completed.returncode:
            write_json(output / "run.json", {"status": "failed", "seed": seed,
                       "exit_code": completed.returncode, "completed_seeds": [x[0] for x in inputs]})
            raise ContractError("Prediction worker failed; no automatic retry")
        document = read_json(target)
        if (not isinstance(document, dict) or document.get("schema") != "product-matching-prediction-v1"
                or document.get("seed") != seed or document.get("rows_sha256") != rows_sha
                or document.get("arm") != args.arm or document.get("epoch") != args.epoch
                or document.get("checkpoint_sha256") != digest):
            raise ContractError("Prediction worker identity differs")
        align_scores(expected_ids, document.get("scores"))
        inputs.append((seed, str(target)))
    ensemble(argparse.Namespace(scores=[path for _, path in inputs], output=output / "scores.json"))
    write_json(output / "run.json", {"status": "completed", "pairs": len(rows),
                                    "seeds": list(SEEDS), "archived_result_weights": archived,
                                    "arm": args.arm, "selected_epoch": args.epoch,
                                    "checkpoint_sha256": list(args.sha256)})


def select_epoch(args):
    from .metrics import select_common_epoch
    predictions = read_json(args.predictions)
    if not isinstance(predictions, dict) or set(predictions) != {"references", "cplus"}:
        raise ContractError("Development predictions need references and cplus")
    if set(predictions["cplus"]) != {str(seed) for seed in SEEDS}:
        raise ContractError("Development predictions require seeds 42,43,44")
    by_seed = {}
    for seed in SEEDS:
        epochs = predictions["cplus"][str(seed)]
        if not isinstance(epochs, dict) or set(epochs) != {str(e) for e in range(1, 7)}:
            raise ContractError("Development selection requires every epoch 1..6")
        by_seed[seed] = {e: score_rows(epochs[str(e)]) for e in range(1, 7)}
    result = select_common_epoch(read_json(args.labels),
                                {k: score_rows(v) for k, v in predictions["references"].items()},
                                by_seed)
    write_json(args.output, {"schema": "product-matching-epoch-selection-v1", "role": "dev",
               "predictions_sha256": file_sha(args.predictions),
               "labels_sha256": file_sha(args.labels), **result})


def semantic_features(args):
    from .semantic import FixedMiniLMEncoder, semantic3_for_role
    from .serialization import _admit
    rows = _admit(read_json(args.rows), args.role)
    keys = ("pair_id", "left", "right", "label") if args.role == "fit" else ("pair_id", "left", "right")
    projection = [{key: row[key] for key in keys} for row in rows]
    encoder = FixedMiniLMEncoder(args.snapshot)
    features, summary = semantic3_for_role(projection, role=args.role, encoder=encoder)
    write_json(args.output, {"schema": "product-matching-semantic-features-v1", "role": args.role,
                            "rows_sha256": file_sha(args.rows), "features": features, "summary": summary})


def reference_inputs(args, role):
    from .serialization import reference_rows
    extra = None
    if args.semantic_features is not None:
        document = read_json(args.semantic_features)
        if (not isinstance(document, dict) or document.get("role") != role
                or document.get("rows_sha256") != file_sha(args.rows)):
            raise ContractError("Semantic feature source/role differs from the reference input")
        extra = document.get("features")
    return reference_rows(read_json(args.rows), role=role, method=args.method, semantic_features=extra)


def reference_fit(args):
    from .references import fit_reference, save_reference
    rows = reference_inputs(args, "fit")
    if Path(args.output).exists():
        raise ContractError("Reference output already exists")
    bundle = fit_reference(args.method, rows)
    output = fresh_directory(args.output)
    metadata = save_reference(output / "reference.pkl", bundle)
    write_json(output / "run.json", {"status": "completed", "method": args.method,
                                    "fit_sha256": file_sha(args.rows), "checkpoint": metadata})


def reference_predict(args):
    from .references import load_reference, predict_reference
    rows = reference_inputs(args, args.role)
    bundle = load_reference(args.checkpoint, expected_sha256=args.sha256)
    if bundle.get("method") != args.method:
        raise ContractError("Reference method differs from checkpoint")
    scores = predict_reference(bundle, rows)
    write_json(args.output, {"schema": "product-matching-reference-prediction-v1",
                            "method": args.method, "role": args.role,
                            "rows_sha256": file_sha(args.rows), "scores": scores})


def metric_rows(scores, labels):
    from .metrics import align_scores
    if not isinstance(labels, list) or not labels:
        raise ContractError("Labels must be a nonempty list of pair_id/label records")
    expected_ids = []
    values = {}
    for row in labels:
        if (not isinstance(row, dict) or set(row) != {"pair_id", "label"}
                or not isinstance(row["pair_id"], str) or not row["pair_id"]
                or row["pair_id"] in values or type(row["label"]) is not int
                or row["label"] not in (0, 1)):
            raise ContractError("Invalid or duplicate label record")
        expected_ids.append(row["pair_id"])
        values[row["pair_id"]] = row["label"]
    if set(values.values()) != {0, 1}:
        raise ContractError("Metrics require both positive and negative labels")
    return [{"pair_id": pid, "score": score, "label": values[pid]}
            for pid, score in align_scores(expected_ids, score_rows(scores))]


def calibrate(args):
    from .metrics import threshold_selection
    scores, scores_sha = read_json_with_digest(args.scores)
    labels, labels_sha = read_json_with_digest(args.labels)
    rows = metric_rows(scores, labels)
    result = threshold_selection(rows)
    write_json(args.output, {"schema": "product-matching-threshold-v1", "role": "calibration",
                            "scores_sha256": scores_sha, "labels_sha256": labels_sha, **result})


def evaluate(args):
    from .metrics import fixed_threshold_metrics
    scores, scores_sha = read_json_with_digest(args.scores)
    labels, labels_sha = read_json_with_digest(args.labels)
    threshold, threshold_sha = read_json_with_digest(args.threshold)
    rows = metric_rows(scores, labels)
    if not isinstance(threshold, dict) or threshold.get("role") != "calibration":
        raise ContractError("Use a previously saved calibration threshold")
    selected = threshold.get("selected", {})
    result = fixed_threshold_metrics(rows, selected.get("threshold"))
    write_json(args.output, {"schema": "product-matching-evaluation-v1", "role": "evaluation",
               "scores_sha256": scores_sha, "labels_sha256": labels_sha,
               "threshold_sha256": threshold_sha, **result})


def parser():
    p = argparse.ArgumentParser(description="Final product-pair matching and reproducible experiments")
    commands = p.add_subparsers(dest="command", required=True)
    snapshot = commands.add_parser("snapshot-manifest", help="Record a supplied local snapshot's integrity")
    snapshot.add_argument("--snapshot", required=True)
    snapshot.add_argument("--output", required=True)
    snapshot.set_defaults(handler=lambda a: write_json(a.output, snapshot_manifest(a.snapshot)))
    projected = commands.add_parser("project-role", help="Join frozen role views, native sidecars and ordered IDs")
    for name in ("views", "sidecars", "kept-ids", "output"):
        projected.add_argument("--" + name, required=True)
    projected.add_argument("--role", choices=("fit", "dev", "calibration", "evaluation"), required=True)
    projected.set_defaults(handler=project_role)
    reconstruction = commands.add_parser("reconstruct-roles", help="Rebuild frozen roles from a pinned local source ZIP")
    for name in ("source-archive", "recipe", "output"):
        reconstruction.add_argument("--" + name, required=True)
    reconstruction.set_defaults(handler=reconstruct_roles)
    prepared = commands.add_parser("prepare", help="Create fixed features and tokenized model inputs")
    prepared.add_argument("--rows", required=True)
    prepared.add_argument("--role", choices=("fit", "dev", "calibration", "evaluation"), required=True)
    prepared.add_argument("--idf-state")
    prepared.add_argument("--output", required=True)
    for name in ("snapshot", "snapshot-manifest"):
        prepared.add_argument("--" + name, required=True)
    prepared.set_defaults(handler=prepare)
    training = commands.add_parser("train", help="Explicitly train one fixed seed on a CUDA GPU")
    training.add_argument("--fit", required=True)
    training.add_argument("--seed", type=int, choices=SEEDS, required=True)
    training.add_argument("--arm", choices=("c-plus", "c-zero"), default="c-plus")
    training.add_argument("--output", required=True)
    for name in ("snapshot", "snapshot-manifest"):
        training.add_argument("--" + name, required=True)
    training.set_defaults(handler=train)
    for command, arm, description in (("predict", "c-plus", "Predict with the final C+ model"),
            ("predict-ablation", "c-zero", "Reproduce the C0 experimental comparison")):
        prediction = commands.add_parser(command, help=description)
        prediction.add_argument("--rows", required=True)
        prediction.add_argument("--checkpoints", nargs=3, required=True, metavar="SEED_ORDER_42_43_44")
        prediction.add_argument("--sha256", nargs=3, required=True)
        prediction.add_argument("--checkpoint-metadata", nargs=3)
        prediction.add_argument("--epoch", type=int, choices=range(1, 7), default=4)
        prediction.add_argument("--output", required=True)
        for name in ("snapshot", "snapshot-manifest"):
            prediction.add_argument("--" + name, required=True)
        prediction.set_defaults(handler=predict, arm=arm)
    worker = commands.add_parser("_predict-one", help=argparse.SUPPRESS)
    for name in ("rows", "snapshot", "snapshot-manifest", "checkpoint", "sha256", "output"):
        worker.add_argument("--" + name, required=True)
    worker.add_argument("--seed", type=int, choices=SEEDS, required=True)
    worker.add_argument("--arm", choices=("c-plus", "c-zero"), required=True)
    worker.add_argument("--epoch", type=int, choices=range(1, 7), required=True)
    worker.set_defaults(handler=predict_one)
    averaging = commands.add_parser("ensemble", help="Average fixed-seed prediction files without a GPU")
    averaging.add_argument("--scores", nargs=3, required=True)
    averaging.add_argument("--output", required=True)
    averaging.set_defaults(handler=ensemble)
    check = commands.add_parser("validate-predictions", help="Check label-free C+/L42 scores against frozen pair IDs")
    for name in ("pair-ids", "candidate", "reference", "output"):
        check.add_argument("--" + name, required=True)
    check.set_defaults(handler=validate_predictions)
    for name, handler in (("calibrate", calibrate), ("evaluate", evaluate)):
        item = commands.add_parser(name, help="Compute explicit calibration/evaluation metrics")
        item.add_argument("--scores", required=True)
        item.add_argument("--labels", required=True)
        item.add_argument("--output", required=True)
        if name == "evaluate":
            item.add_argument("--threshold", required=True)
        item.set_defaults(handler=handler)
    selection = commands.add_parser("select-epoch", help="Select a common epoch from development predictions")
    for name in ("predictions", "labels", "output"):
        selection.add_argument("--" + name, required=True)
    selection.set_defaults(handler=select_epoch)
    semantic = commands.add_parser("semantic-features", help="Compute the fixed S33 features on a CUDA GPU")
    for name in ("rows", "snapshot", "output"):
        semantic.add_argument("--" + name, required=True)
    semantic.add_argument("--role", choices=("fit", "dev", "calibration", "evaluation"), required=True)
    semantic.set_defaults(handler=semantic_features)
    for name, handler in (("reference-fit", reference_fit), ("reference-predict", reference_predict)):
        reference = commands.add_parser(name, help="Fit/predict a fixed experimental reference")
        reference.add_argument("--method", choices=("B22", "L30", "S33", "L42"), required=True)
        reference.add_argument("--rows", required=True)
        reference.add_argument("--semantic-features")
        reference.add_argument("--output", required=True)
        if name == "reference-predict":
            reference.add_argument("--role", choices=("dev", "calibration", "evaluation"), required=True)
            reference.add_argument("--checkpoint", required=True)
            reference.add_argument("--sha256", required=True)
        reference.set_defaults(handler=handler)
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = parser()
    args = p.parse_args(argv)
    try:
        if args.command in {"train", "_predict-one"}:
            _ensure_process_environment(args.seed, argv)
        if args.command == "semantic-features":
            names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            if (os.environ.get("PYTHONHASHSEED") != "42"
                    or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
                    or any(os.environ.get(name) != "1" for name in names)):
                env = _gpu_environment(42)
                env.update({name: "1" for name in names})
                os.execve(sys.executable, [sys.executable, "-m", "product_matching.cli", *argv], env)
        args.handler(args)
    except (ContractError, OSError, ValueError, ImportError) as exc:
        p.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
