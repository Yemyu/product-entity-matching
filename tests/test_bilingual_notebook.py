"""Keep both walkthroughs executable without turning translated text into model code."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATHS = {
    "en": ROOT / "notebooks/product-matching-walkthrough.ipynb",
    "zh-CN": ROOT / "notebooks/product-matching-walkthrough.zh-CN.ipynb",
}
SCRIPT = ROOT / "scripts/run_notebook.py"


class BilingualNotebookTests(unittest.TestCase):
    def test_same_computation_and_stable_cell_ids(self):
        documents = {language: json.loads(path.read_text(encoding="utf-8")) for language, path in PATHS.items()}
        ids = [[cell["id"] for cell in document["cells"]] for document in documents.values()]
        self.assertEqual(ids[0], ids[1])
        self.assertEqual(len(ids[0]), len(set(ids[0])))
        computations = {}
        for language, document in documents.items():
            self.assertEqual(document["metadata"]["walkthrough_language"], language)
            code = [cell for cell in document["cells"] if cell["cell_type"] == "code"]
            self.assertEqual(len(code), 8)
            trees = [ast.parse("".join(cell["source"])) for cell in code]
            assignments = [node for node in trees[0].body if isinstance(node, ast.Assign)
                           and any(isinstance(target, ast.Name) and target.id == "LANGUAGE" for target in node.targets)]
            self.assertEqual(len(assignments), 1)
            self.assertEqual(ast.literal_eval(assignments[0].value), language)
            assignments[0].value = ast.Constant(value="TRANSLATION")
            computations[language] = [ast.dump(tree, include_attributes=False) for tree in trees]
            other = PATHS["zh-CN" if language == "en" else "en"].name
            self.assertIn(f"]({other})", "".join(document["cells"][0]["source"]))
        self.assertEqual(computations["en"], computations["zh-CN"])

    def test_both_run_from_another_directory_without_model_dependencies_or_writes(self):
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in PATHS.values()}
        probe = '''
import runpy, sys
class BlockModels:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "transformers", "sklearn", "sentence_transformers", "numpy", "safetensors"}:
            raise AssertionError("A lightweight notebook tried to import " + fullname)
sys.meta_path.insert(0, BlockModels())
namespace = runpy.run_path(sys.argv[1])
assert namespace["execute"](write=False) == 16
print("both notebooks passed")
'''
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-I", "-S", "-c", probe, str(SCRIPT)], cwd=directory,
                                    capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("both notebooks passed", result.stdout)
        self.assertEqual(before, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in PATHS.values()})

    def test_cli_selects_either_translation(self):
        with tempfile.TemporaryDirectory() as directory:
            for language in PATHS:
                with self.subTest(language=language):
                    result = subprocess.run([sys.executable, "-I", "-S", str(SCRIPT), "--language", language],
                                            cwd=directory, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    report = json.loads(result.stdout)
                    self.assertEqual(report["languages"], [language])
                    self.assertEqual(report["executed_code_cells"], 8)
                    self.assertFalse(report["saved_outputs_written"])
                    self.assertFalse(report["model_run"])

    def test_stale_saved_output_is_rejected_without_rewriting(self):
        spec = importlib.util.spec_from_file_location("bilingual_notebook_runner", SCRIPT)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as directory:
            notebook = Path(directory) / PATHS["en"].name
            document = json.loads(PATHS["en"].read_text(encoding="utf-8"))
            cell = next(cell for cell in document["cells"] if cell["cell_type"] == "code")
            cell["outputs"] = []
            notebook.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
            runner.NOTEBOOKS = {"en": notebook}
            before = notebook.read_bytes()
            cwd = Path.cwd()
            with self.assertRaisesRegex(AssertionError, "Saved output differs"):
                runner.execute(write=False, language="en")
            self.assertEqual(notebook.read_bytes(), before)
            self.assertEqual(Path.cwd(), cwd)
            self.assertEqual(runner.execute(write=True, language="en"), 8)
            self.assertEqual(runner.execute(write=False, language="en"), 8)
            self.assertEqual(Path.cwd(), cwd)


if __name__ == "__main__":
    unittest.main()
