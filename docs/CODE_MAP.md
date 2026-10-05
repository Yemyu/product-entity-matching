# Code map

**English** | [简体中文](zh-CN/CODE_MAP.md)

## Package and entry point

`scripts/matching.py` calls `product_matching.cli`. The 16 modules in `src/product_matching/` implement the final method and its experimental controls. Importing the package or requesting CLI help uses no neural runtime and downloads no model. Heavy libraries are imported only by the commands that need them.

| Module | Responsibility |
| --- | --- |
| [`__init__.py`](../src/product_matching/__init__.py) | Package entry and exports |
| [`data.py`](../src/product_matching/data.py) | Explicit business/native adapters and content identity |
| [`normalization.py`](../src/product_matching/normalization.py) | Text, price and currency handling with missing/ambiguous states |
| [`source.py`](../src/product_matching/source.py) | Reconstruct frozen role inputs from a pinned local source ZIP |
| [`roles.py`](../src/product_matching/roles.py) | Join frozen membership, views and sidecars; no new splitting |
| [`lexical.py`](../src/product_matching/lexical.py) | Fit-only title IDF and its saved state |
| [`features.py`](../src/product_matching/features.py) | Construct and validate the fixed 42 comparisons |
| [`serialization.py`](../src/product_matching/serialization.py) | Field token budgets, AB/BA inputs and reference row formats |
| [`model.py`](../src/product_matching/model.py) | Fixed RoBERTa fusion head, training and prediction |
| [`runtime.py`](../src/product_matching/runtime.py) | Determinism and explicit model execution settings |
| [`snapshot.py`](../src/product_matching/snapshot.py) | Pinned local RoBERTa file identity |
| [`references.py`](../src/product_matching/references.py) | B22/L30/S33/L42 fitting, prediction and artifact loading |
| [`semantic.py`](../src/product_matching/semantic.py) | Fixed MiniLM features used by S33 |
| [`metrics.py`](../src/product_matching/metrics.py) | ID alignment, AP ties, common epoch, cutoff and confusion matrix |
| [`cli.py`](../src/product_matching/cli.py) | Explicit file commands, metadata checks and label-free prediction validation |
| [`errors.py`](../src/product_matching/errors.py) | Input and execution contract exceptions |

Prediction starts a fresh process for each seed. `_predict-one` is the internal worker command; readers use `predict` or `predict-ablation`. The reference loader maps one pinned legacy B22 pickle class name to the current class without importing the old package. Pickle loading still requires a trusted local artifact and its recorded hash.

## Public checks and materials

| Path | Purpose |
| --- | --- |
| `scripts/public_smoke.py` | Lightweight import/input/feature/metric checks |
| `scripts/run_notebook.py` | Execute the bilingual plain-Python Notebook cells and verify saved outputs |
| `scripts/verify_public_release.py` | Verify file inventory, hashes, reported arithmetic and local links |
| `scripts/run_tests.py` and `tests/` | Standard-library synthetic contract tests |
| `scripts/build_public_manifest.py` | Regenerate the public inventory after deliberate file edits |
| `results/fixed-results.json` | Saved six-method aggregate facts |
| `reproducibility/` | Fixed model, experiment, snapshot and feature-order descriptions |
| `docs/` and `docs/zh-CN/` | Seven corresponding English and Chinese technical guides |
| `notebooks/` | English and Chinese executable explanations |
| `showcase/` | Static project pages in both languages |

The package does not depend on a private project checkout or a remote training controller. Product rows, labels and trained artifacts are supplied explicitly for commands that need them. Public synthetic checks are separate from those model commands.

Continue with [CORE_USAGE](CORE_USAGE.md) for file interfaces, [MODEL_CARD](MODEL_CARD.md) for the architecture and [REPRODUCIBILITY](REPRODUCIBILITY.md) for verification scope.
