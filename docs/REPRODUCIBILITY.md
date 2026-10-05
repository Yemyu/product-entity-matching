# Reproducibility

**English** | [简体中文](zh-CN/REPRODUCIBILITY.md)

## Reproduction status

| Level | Evidence | Requirement or limit |
| --- | --- | --- |
| Public checks and Notebook | Synthetic interface/feature/metric checks and saved aggregate arithmetic | Python 3.11+ and the standard library; no product data or model download |
| Existing-weight replay | 20 commands, 23 independently checked result groups, zero differences from frozen outputs | Original role inputs, snapshots and trusted weights supplied locally; no new training |
| Raw-source role reconstruction | Four business input files match the frozen bytes and member order | Original ZIP and the bundled fixed-membership recipe; no model execution |
| Full experiment from scratch | Explicit training and prediction commands are provided | Fresh end-to-end training has not been validated |

The replay covered real token/feature/IDF preparation, MiniLM features, four reference predictions and final three-seed C+ prediction. Matching outputs verify this code entry against existing artifacts. They do not establish independent new Walmart–Amazon sample performance or reproduce the entire original experiment from raw sources.

## Lightweight checks

Create the Python 3.11+ project environment in the [README quickstart](../README.md#quickstart), then run from the repository root. On Windows, replace `python` below with the project’s `.\.venv\Scripts\python.exe`:

```bash
python scripts/public_smoke.py
python scripts/run_notebook.py
python scripts/verify_public_release.py
python scripts/run_tests.py
```

These checks require no package installation, GPU, dataset or model snapshot. The Notebook's real numbers come from [fixed-results.json](../results/fixed-results.json); its invented Acme records and demonstration probabilities are synthetic.

`run_notebook.py` executes both [English](../notebooks/product-matching-walkthrough.ipynb) and [Chinese](../notebooks/product-matching-walkthrough.zh-CN.ipynb) plain-Python cells and verifies saved outputs. The notebooks may also run in an existing Python 3 Jupyter kernel. Standard-library execution does not replace a separate Jupyter visual check.

`verify_public_release.py` checks inventory coverage, hashes, published arithmetic, local links and public file boundaries. It does not execute GPU commands or inspect private data. After deliberate source or material edits, regenerate the inventory with `python scripts/build_public_manifest.py` before running the verifier again. The inventory excludes itself, local environments, caches and Git internals.

## Observed GPU environment

The completed existing-artifact replay reused this environment:

| Component | Observed value |
| --- | --- |
| Platform | Windows 11 |
| Python | 3.11.16 |
| GPU | RTX 3060 laptop |
| PyTorch / CUDA | 2.8.0+cu126 / 12.6 |
| transformers | 5.17.0 |
| sentence-transformers | 6.0.1 |
| scikit-learn | 1.9.1 |
| numpy | 2.4.6 |
| safetensors | 0.8.0 |

These observations are recorded in [EXPERIMENT_LOCK.json](../reproducibility/EXPERIMENT_LOCK.json). [requirements-train.lock.txt](../requirements-train.lock.txt) pins direct packages and the CUDA 12.6 wheel channel. It is not a complete lock of transitive dependencies, wheel hashes or device drivers. Other platforms and GPU combinations have not been validated.

## Project environment for model commands

If reproducing the Windows GPU commands, use a project environment and verify its interpreter. The following documents the installation path; it was not tested as a fresh-environment installation during the material preparation:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-train.lock.txt
.venv/Scripts/python -m pip check
```

The existing environment was reused for replay. The CUDA wheel channel follows [PyTorch's official previous-version instructions](https://pytorch.org/get-started/previous-versions/) for PyTorch 2.8.0/CUDA 12.6. Compatible drivers and device support are external prerequisites. This repository does not install or change them. Keep source files, data and experiment outputs outside `.venv`.

## Local files and identities

The public checkout distributes no product rows, labels, model snapshots or trained weights. It includes source-position metadata for reconstructing the frozen roles. Provide locally obtained inputs under the relevant third-party terms. [CORE_USAGE](CORE_USAGE.md) gives preparation, explicit training, prediction, reference, selection and metric commands.

| Public description | What it identifies |
| --- | --- |
| [FINAL_MODEL.json](../reproducibility/FINAL_MODEL.json) | Final architecture/settings, seeds, selected epoch and original checkpoint hashes |
| [ROBERTA_SNAPSHOT.json](../reproducibility/ROBERTA_SNAPSHOT.json) | Exact backbone/tokenizer file sizes and hashes |
| [ROLE_RECONSTRUCTION.json](../reproducibility/ROLE_RECONSTRUCTION.json) | Fixed source positions, native missing masks and expected role-input hashes |
| [FEATURES.json](../reproducibility/FEATURES.json) | Fixed 42-feature order and groups |
| [EXPERIMENT_LOCK.json](../reproducibility/EXPERIMENT_LOCK.json) | Data roles, selection/calibration assignments and observed environment |

Preparation and model commands verify the local snapshot against the fixed file identity. Original selected weights use the three checkpoint hashes in seed order 42, 43, 44. Newly trained weights require each training `run.json`, matching arm/seed/epoch/hash and a common fit identity. New output paths are required; failures are not silently retried.

## Selection and reconstruction scope

Title IDF is fitted only on fit records. Development predictions from all three seeds select one common epoch; calibration labels select the cutoff before evaluation scoring. Evaluation labels are separate from model inputs and are not used for tuning. Those current interface rules do not resolve the fixed pool's incomplete historical exposure record.

The original frozen role inputs can now be rebuilt from the pinned source ZIP through `reconstruct-roles`. All four business-file hashes and membership orders matched. The bundled recipe restores the original native missing masks, including 23 model-number and 14 category occurrences. Source acquisition is manual. This does not recover the incomplete historical exposure record. Fresh retraining has not been validated and may produce different artifacts; a new random split would create a different experiment. See [DATA_CARD](DATA_CARD.md) and [LIMITATIONS](LIMITATIONS.md).
