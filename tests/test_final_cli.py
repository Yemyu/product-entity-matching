"""Synthetic end-to-end CLI checks; no real labels, weights or GPU required."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from product_matching import cli, model
from product_matching.errors import ContractError


class FinalCLITests(unittest.TestCase):
    def test_help_has_no_heavy_dependencies_or_training_side_effects(self):
        script = Path(__file__).resolve().parents[1] / "scripts/matching.py"
        completed = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("prepare", completed.stdout)
        self.assertIn("predict", completed.stdout)
        self.assertIn("validate-predictions", completed.stdout)
        self.assertNotIn("D:" + "/", completed.stdout)

    def test_json_rejects_nonfinite_duplicate_fields_and_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            for body in ('{"x":NaN}', '{"x":1,"x":2}'):
                path.write_text(body)
                with self.assertRaises(ContractError): cli.read_json(path)
            with self.assertRaises(ContractError): cli.write_json(path, {})

    def test_neural_inputs_separate_labels_and_validate_tokens_features(self):
        row = {"pair_id": "synthetic:1", "ab": [0, 3, 2], "ba": [0, 4, 2], "x42": [0.5] * 42}
        self.assertEqual(cli.validate_neural_rows([row], training=False), [row])
        for invalid in ({**row, "label": 1}, {**row, "ab": [True]},
                        {**row, "x42": [float("nan")] * 42}):
            with self.assertRaises(ContractError): cli.validate_neural_rows([invalid], training=False)
        fit = [{**row, "label": 0}, {**row, "pair_id": "synthetic:2", "label": 1}]
        self.assertEqual(cli.validate_neural_rows(fit, training=True), fit)
        with self.assertRaises(ContractError): cli.validate_neural_rows([fit[0]], training=True)

    def test_snapshot_integrity_checks_supplied_files_without_loading_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot"; snapshot.mkdir()
            (snapshot / "config.json").write_text(json.dumps({"model_type": "roberta", "hidden_size": 768,
                "num_hidden_layers": 12, "vocab_size": 50265}))
            (snapshot / "model.safetensors").write_bytes(b"synthetic placeholder, not weights")
            (snapshot / "tokenizer.json").write_text('{}')
            manifest = root / "manifest.json"
            cli.write_json(manifest, cli.snapshot_manifest(snapshot))
            cli.verify_snapshot(snapshot, manifest)
            (snapshot / "tokenizer.json").write_text('{"changed":true}')
            with self.assertRaises(ContractError): cli.verify_snapshot(snapshot, manifest)

    def test_ensemble_calibration_evaluation_complete_synthetic_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); files = []
            labels = [{"pair_id": f"synthetic:{i:03}", "label": int(i % 3 == 0)} for i in range(120)]
            for seed in (42, 43, 44):
                scores = [{"pair_id": x["pair_id"], "score": 0.9 if x["label"] else 0.1} for x in labels]
                path = root / f"seed-{seed}.json"
                cli.write_json(path, {"seed": seed, "scores": list(reversed(scores)) if seed == 43 else scores})
                files.append(str(path))
            label_file = root / "labels.json"; cli.write_json(label_file, labels)
            ensemble = root / "ensemble.json"
            cli.main(["ensemble", "--scores", *files, "--output", str(ensemble)])
            threshold = root / "threshold.json"
            cli.main(["calibrate", "--scores", str(ensemble), "--labels", str(label_file), "--output", str(threshold)])
            result = root / "result.json"
            cli.main(["evaluate", "--scores", str(ensemble), "--labels", str(label_file),
                      "--threshold", str(threshold), "--output", str(result)])
            actual = cli.read_json(result)
            self.assertEqual(actual["ap"], 1.0)
            self.assertEqual(actual["f1"], 1.0)
            self.assertEqual(actual["confusion"], {"tp": 40, "fp": 0, "fn": 0, "tn": 80})
            self.assertEqual(actual["p_at_100"], 0.4)

    def test_ensemble_rejects_duplicate_seed_missing_ids_and_mixed_arms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); paths = []
            for seed in (42, 43, 44):
                path = root / f"{seed}.json"; paths.append(str(path))
                cli.write_json(path, {"seed": seed, "scores": [{"pair_id": "s:1", "score": .5}]})
            with self.assertRaises(ContractError):
                cli.ensemble(argparse.Namespace(scores=[paths[0], paths[0], paths[2]], output=root / "bad.json"))
            doc = json.loads(Path(paths[2]).read_text()); doc['scores'][0]['pair_id'] = 's:2'
            Path(paths[2]).write_text(json.dumps(doc))
            with self.assertRaises(ContractError):
                cli.ensemble(argparse.Namespace(scores=paths, output=root / "bad.json"))
            doc['scores'][0]['pair_id'] = 's:1'; doc['arm'] = 'c-zero'
            Path(paths[2]).write_text(json.dumps(doc))
            with self.assertRaises(ContractError):
                cli.ensemble(argparse.Namespace(scores=paths, output=root / "bad.json"))
            self.assertFalse((root / "bad.json").exists())

    def _prediction_check_inputs(self, root):
        ids = ["synthetic:a", "synthetic:b", "synthetic:c"]
        scores = [{"pair_id": pid, "score": score} for pid, score in zip(ids, (0.0, .5, 1.0))]
        candidate = {"schema": "product-matching-ensemble-v1", "seeds": [42, 43, 44],
                     "arm": "c-plus", "aggregation": "equal_probability_mean", "scores": scores[::-1]}
        reference = {"schema": "product-matching-reference-prediction-v1", "method": "L42",
                     "role": "evaluation", "rows_sha256": "a" * 64, "scores": scores}
        paths = {"pair_ids": root / "ids.json", "candidate": root / "candidate.json",
                 "reference": root / "reference.json", "output": root / "check.json"}
        for name, document in (("pair_ids", ids), ("candidate", candidate), ("reference", reference)):
            cli.write_json(paths[name], document)
        return argparse.Namespace(**paths), {"pair_ids": ids, "candidate": candidate, "reference": reference}

    def test_prediction_check_parser_accepts_reordered_scores_without_labels_or_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, _ = self._prediction_check_inputs(root)
            private_labels = root / "private-labels.json"
            private_labels.write_text("this file must never be opened")
            real_open, real_read = Path.open, Path.read_bytes
            def guarded_open(path, *positional, **keywords):
                self.assertNotEqual(path, private_labels, "Prediction check opened a label file")
                return real_open(path, *positional, **keywords)
            def guarded_read(path):
                self.assertNotEqual(path, private_labels, "Prediction check read a label file")
                return real_read(path)
            real_import = __import__
            def guarded_import(name, *positional, **keywords):
                self.assertNotIn(name.split(".")[0], {"torch", "transformers", "numpy", "sklearn"})
                return real_import(name, *positional, **keywords)
            with patch.object(Path, "open", guarded_open), patch.object(Path, "read_bytes", guarded_read), \
                 patch("builtins.__import__", guarded_import), \
                 patch.object(cli, "metric_rows", side_effect=AssertionError("Labels must not be joined")):
                self.assertEqual(cli.main(["validate-predictions", "--pair-ids", str(args.pair_ids),
                    "--candidate", str(args.candidate), "--reference", str(args.reference),
                    "--output", str(args.output)]), 0)
            self.assertEqual(cli.read_json(args.output), {
                "schema": "product-matching-prediction-check-v1", "status": "passed", "count": 3,
                "pair_ids_sha256": hashlib.sha256(args.pair_ids.read_bytes()).hexdigest(),
                "candidate_sha256": hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
                "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                "label_values_read": False, "metrics_computed": False})

    def test_prediction_check_rejects_invalid_frozen_ids(self):
        for invalid in ([], "synthetic:a", [""], [False], ["synthetic:a", "synthetic:a"],
                        ["synthetic:a", []]):
            with self.subTest(ids=invalid), tempfile.TemporaryDirectory() as directory:
                args, _ = self._prediction_check_inputs(Path(directory))
                args.pair_ids.write_text(json.dumps(invalid))
                with self.assertRaises(ContractError): cli.validate_predictions(args)
                self.assertFalse(args.output.exists())

    def test_prediction_check_rejects_invalid_score_rows_for_either_method(self):
        scores = [{"pair_id": "synthetic:a", "score": .5},
                  {"pair_id": "synthetic:b", "score": .5},
                  {"pair_id": "synthetic:c", "score": .5}]
        invalid_rows = [scores[:-1], scores + [{"pair_id": "synthetic:d", "score": .5}],
                        scores + [scores[0]], [{**scores[0], "pair_id": ""}, *scores[1:]],
                        [{**scores[0], "label": 0}, *scores[1:]], "scores"]
        invalid_rows.extend([{**scores[0], "score": value}, *scores[1:]]
                            for value in (True, float("nan"), float("inf"), -.1, 1.1, ".5"))
        for method in ("candidate", "reference"):
            for invalid in invalid_rows:
                with self.subTest(method=method, scores=invalid), tempfile.TemporaryDirectory() as directory:
                    args, documents = self._prediction_check_inputs(Path(directory))
                    document = {**documents[method], "scores": invalid}
                    getattr(args, method).write_text(json.dumps(document))
                    with self.assertRaises(ContractError): cli.validate_predictions(args)
                    self.assertFalse(args.output.exists())

    def test_prediction_check_rejects_wrong_or_labelled_containers(self):
        changes = {
            "candidate": ({"schema": "wrong"}, {"seeds": [42, 43]}, {"seeds": [43, 42, 44]},
                          {"seeds": [42.0, 43, 44]}, {"arm": "c-zero"}, {"aggregation": "logit_mean"},
                          {"label": 1}),
            "reference": ({"schema": "wrong"}, {"method": "B22"}, {"role": "dev"},
                          {"rows_sha256": "A" * 64}, {"rows_sha256": "a" * 63},
                          {"rows_sha256": True}, {"label": 1})}
        for method, patches in changes.items():
            for change in (*patches, None, "list"):
                with self.subTest(method=method, change=change), tempfile.TemporaryDirectory() as directory:
                    args, documents = self._prediction_check_inputs(Path(directory))
                    document = ({**documents[method], **change} if isinstance(change, dict)
                                else None if change is None else documents[method]["scores"])
                    getattr(args, method).write_text(json.dumps(document))
                    with self.assertRaises(ContractError): cli.validate_predictions(args)
                    self.assertFalse(args.output.exists())
            with self.subTest(method=method, missing="scores"), tempfile.TemporaryDirectory() as directory:
                args, documents = self._prediction_check_inputs(Path(directory))
                getattr(args, method).write_text(json.dumps({k: v for k, v in documents[method].items() if k != "scores"}))
                with self.assertRaises(ContractError): cli.validate_predictions(args)
                self.assertFalse(args.output.exists())

    def test_prediction_check_rejects_inputs_changed_after_read(self):
        for name in ("pair_ids", "candidate", "reference"):
            with self.subTest(changed=name), tempfile.TemporaryDirectory() as directory:
                args, _ = self._prediction_check_inputs(Path(directory))
                real_reader = cli.read_json_with_digest
                def changed_reader(path):
                    result = real_reader(path)
                    if path == getattr(args, name):
                        path.write_bytes(path.read_bytes() + b" ")
                    return result
                with patch.object(cli, "read_json_with_digest", side_effect=changed_reader):
                    with self.assertRaisesRegex(ContractError, "changed after it was read"):
                        cli.validate_predictions(args)
                self.assertFalse(args.output.exists())

    def test_prediction_check_never_overwrites_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self._prediction_check_inputs(Path(directory))
            original = b"existing evidence\n"
            args.output.write_bytes(original)
            with self.assertRaises(ContractError): cli.validate_predictions(args)
            self.assertEqual(args.output.read_bytes(), original)

    def test_checkpoint_identity_cannot_label_arbitrary_weights_as_archived_E4(self):
        args = argparse.Namespace(arm="c-plus", epoch=4, sha256=list(cli.FINAL_CHECKPOINTS), checkpoint_metadata=None)
        self.assertTrue(cli.checkpoint_identity(args))
        args.epoch = 3
        with self.assertRaises(ContractError): cli.checkpoint_identity(args)
        args.epoch = 4; args.sha256 = ['a' * 64] * 3
        with self.assertRaises(ContractError): cli.checkpoint_identity(args)
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for seed in (42, 43, 44):
                path = Path(directory) / f"{seed}.json"; paths.append(str(path))
                cli.write_json(path, {"schema": "product-matching-training-v1", "status": "completed",
                    "arm": "c-plus", "seed": seed, "fit_sha256": 'f' * 64,
                    "checkpoints": [{"epoch": 4, "sha256": 'a' * 64}]})
            args.checkpoint_metadata = paths
            self.assertFalse(cli.checkpoint_identity(args))
            args.arm = "c-zero"
            with self.assertRaises(ContractError): cli.checkpoint_identity(args)

    def test_fixed_neural_schedule_tail_and_seed_order_are_preserved(self):
        self.assertEqual(model.training_steps(3738), 1404)
        self.assertEqual(model.window_loss_scale(10), .05)
        self.assertEqual(model.window_loss_scale(16), 1 / 32)
        self.assertEqual(model.epoch_order(25, 42, 1), model.epoch_order(25, 42, 1))
        self.assertNotEqual(model.epoch_order(25, 42, 1), model.epoch_order(25, 43, 1))
        self.assertEqual(model.learning_rate_multiplier(0, 1404), 1 / 141)
        with self.assertRaises(ContractError): model.epoch_order(25, 45, 1)

    def test_exit_zero_wrong_worker_ids_cannot_complete_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows_path = root / "rows.json"
            rows = [{"pair_id": f"synthetic:{i}", "ab": [0, 2], "ba": [0, 2], "x42": [.5] * 42}
                    for i in range(2)]
            cli.write_json(rows_path, rows)
            checkpoints = []
            for i in range(3):
                path = root / f"{i}.safetensors"; path.write_bytes(b"synthetic placeholder")
                checkpoints.append(str(path))
            args = argparse.Namespace(rows=str(rows_path), snapshot="synthetic-snapshot",
                snapshot_manifest="synthetic-manifest", output=str(root / "prediction"),
                arm="c-plus", epoch=4, checkpoints=checkpoints,
                sha256=[cli.file_sha(path) for path in checkpoints], checkpoint_metadata=None)
            def worker(command, **kwargs):
                def arg(name): return command[command.index("--" + name) + 1]
                cli.write_json(arg("output"), {"schema": "product-matching-prediction-v1",
                    "seed": int(arg("seed")), "arm": "c-plus", "epoch": 4,
                    "rows_sha256": cli.file_sha(rows_path), "checkpoint_sha256": arg("sha256"),
                    "scores": [{"pair_id": "wrong-id", "score": .5}]})
                return argparse.Namespace(returncode=0)
            with patch.object(cli, "verify_snapshot", return_value={}), \
                 patch.object(cli, "checkpoint_identity", return_value=False), \
                 patch.object(cli.subprocess, "run", side_effect=worker):
                with self.assertRaises(ContractError): cli.predict(args)
            self.assertFalse((root / "prediction" / "scores.json").exists())
            self.assertFalse((root / "prediction" / "run.json").exists())

    def test_C0_worker_actually_disables_lexical_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); rows_path = root / "rows.json"
            cli.write_json(rows_path, [{"pair_id": "synthetic:1", "ab": [0, 2], "ba": [0, 2], "x42": [.5] * 42}])
            network = MagicMock()
            args = argparse.Namespace(rows=str(rows_path), snapshot="synthetic-snapshot",
                snapshot_manifest="synthetic-manifest", output=str(root / "out.json"),
                seed=42, arm="c-zero", epoch=4, checkpoint="synthetic-weights", sha256='a'*64)
            with patch.object(cli, "verify_snapshot", return_value={}), \
                 patch.object(cli, "_configured_gpu", return_value=(None, {})), \
                 patch.object(model, "make_model", return_value=network) as make, \
                 patch.object(model, "load_weights"), \
                 patch.object(model, "predict_rows", return_value=[{"pair_id": "synthetic:1", "score": .5}]):
                cli.predict_one(args)
            make.assert_called_once_with("synthetic-snapshot", zero_lexical=True, seed=42)
            self.assertEqual(cli.read_json(root / "out.json")["arm"], "c-zero")

    def test_C0_four_epochs_keep_six_epoch_schedule_core(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); fit = root / "fit.json"
            cli.write_json(fit, [{"pair_id": f"synthetic:{i}", "ab": [0, 2], "ba": [0, 2],
                                 "x42": [.5] * 42, "label": i} for i in (0, 1)])
            network = MagicMock()
            args = argparse.Namespace(fit=str(fit), snapshot="synthetic-snapshot", snapshot_manifest="synthetic-manifest",
                                      output=str(root / "train"), arm="c-zero", seed=42)
            def epoch(network, optimizer, rows, **kwargs): return kwargs["epoch"]
            with patch.object(cli, "verify_snapshot", return_value={}), \
                 patch.object(cli, "_configured_gpu", return_value=(None, {})), \
                 patch.object(model, "make_model", return_value=network) as make, \
                 patch.object(model, "make_optimizer", return_value=object()), \
                 patch.object(model, "train_epoch", side_effect=epoch) as fit_epoch, \
                 patch.object(model, "save_weights", return_value='a' * 64):
                cli.train(args)
            self.assertEqual(fit_epoch.call_count, 4)
            self.assertTrue(make.call_args.kwargs["zero_lexical"])
            receipt = cli.read_json(root / "train/run.json")
            self.assertEqual(receipt["epochs"], 4)
            self.assertEqual(receipt["scheduler_epochs"], 6)
            self.assertEqual([c["epoch"] for c in receipt["checkpoints"]], [1, 2, 3, 4])

    def test_reference_cli_roundtrip_uses_explicit_rows_and_hash(self):
        from product_matching.data import business_record, native_record
        from product_matching.features import fit_title_evidence, prepare_rows
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for i in range(4):
                rows.append({"pair_id": f"synthetic:{i}",
                    "left": business_record({"title": "Demo ZX-10", "brand": "Demo"}),
                    "right": business_record({"title": "Demo ZX-10" if i % 2 else "Other YY-99", "brand": "Demo"}),
                    "left_native": native_record({}), "right_native": native_record({}), "label": i % 2})
            prepared = prepare_rows(rows, role="fit", fit_idf=fit_title_evidence(rows))
            fit = root / "fit.json"; cli.write_json(fit, prepared)
            target = root / "reference"
            cli.main(["reference-fit", "--method", "B22", "--rows", str(fit), "--output", str(target)])
            predicted = root / "dev.json"; cli.write_json(predicted, [{k:v for k,v in row.items() if k!='label'} for row in prepared])
            digest = cli.read_json(target / "run.json")["checkpoint"]["sha256"]
            out = root / "scores.json"
            cli.main(["reference-predict", "--method", "B22", "--rows", str(predicted), "--role", "dev",
                      "--checkpoint", str(target / "reference.pkl"), "--sha256", digest, "--output", str(out)])
            self.assertEqual(len(cli.read_json(out)["scores"]), 4)


if __name__ == "__main__":
    unittest.main()
