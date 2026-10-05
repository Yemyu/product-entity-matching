"""Execute both walkthroughs and check saved output using only the standard library."""

import argparse
import contextlib
import io
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    "en": ROOT / "notebooks/product-matching-walkthrough.ipynb",
    "zh-CN": ROOT / "notebooks/product-matching-walkthrough.zh-CN.ipynb",
}
NOTEBOOK = NOTEBOOKS["en"]


def execute(write=False, language="all"):
    """Check saved output by default; replace it only when ``write=True``.

    The historical ``execute(write=False)`` call remains supported and now
    checks both translations. Return the total number of executed code cells.
    """
    if language not in (*NOTEBOOKS, "all"):
        raise ValueError(f"Unsupported notebook language: {language}")
    languages = tuple(NOTEBOOKS) if language == "all" else (language,)
    documents = []
    total = 0
    initial = Path.cwd()
    try:
        os.chdir(ROOT)
        for selected in languages:
            notebook = NOTEBOOKS[selected]
            document = json.loads(notebook.read_text(encoding="utf-8"))
            if document.get("nbformat") != 4:
                raise ValueError(f"Unsupported notebook format: {notebook.name}")
            namespace = {"__name__": "__main__"}
            count = 0
            for cell in document["cells"]:
                if cell["cell_type"] != "code":
                    continue
                count += 1
                stdout = io.StringIO()
                source = "".join(cell["source"])
                if any(line.lstrip().startswith(("%", "!")) for line in source.splitlines()):
                    raise ValueError(f"Non-Python command: {notebook.name}, cell {count}")
                with contextlib.redirect_stdout(stdout):
                    exec(compile(source, f"{notebook.name}:cell{count}", "exec"), namespace)
                output = stdout.getvalue()
                expected = [
                    {"output_type": "stream", "name": "stdout", "text": output.splitlines(keepends=True)}
                ] if output else []
                if write:
                    cell["execution_count"] = count
                    cell["outputs"] = expected
                elif cell.get("execution_count") != count or cell.get("outputs") != expected:
                    raise AssertionError(f"Saved output differs: {notebook.name}, code cell {count}")
            documents.append((notebook, document))
            total += count
    finally:
        os.chdir(initial)
    if write:
        for notebook, document in documents:
            notebook.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=(*NOTEBOOKS, "all"), default="all")
    parser.add_argument("--write-outputs", action="store_true", help="Replace saved outputs after successful execution.")
    args = parser.parse_args()
    count = execute(write=args.write_outputs, language=args.language)
    print(json.dumps({
        "status": "passed",
        "languages": list(NOTEBOOKS) if args.language == "all" else [args.language],
        "executed_code_cells": count,
        "model_run": False,
        "saved_outputs_written": args.write_outputs,
    }))
