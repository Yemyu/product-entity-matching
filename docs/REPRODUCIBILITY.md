# Reproducibility

**🌐 English** | [🇨🇳 简体中文](zh-CN/REPRODUCIBILITY.md)

## Reproduction status

| Level | Evidence | Requirement or limit |
| --- | --- | --- |
| Public checks and Notebook | Synthetic interface/feature/metric checks and saved aggregate arithmetic | Python 3.11+ and the standard library; no product data or model download |
| Existing-weight replay | 20 commands, 23 independently checked result groups, zero differences from retained outputs | Original inputs, snapshots and trusted weights supplied locally; no new training |
| Raw-source role reconstruction | Four business input files match the retained bytes and member order | Original ZIP and the bundled fixed-membership recipe; no model execution |
| Full experiment from scratch | Explicit training and prediction commands are provided | Fresh end-to-end training has not been validated |

Replay covered token/feature/IDF preparation, MiniLM features, four reference predictions and final three-seed C+ prediction. It verified the public code against existing artifacts. It did not retrain the model or evaluate independent new Walmart–Amazon samples. The fixed pool's historical exposure record remains incomplete.

## Lightweight checks

Create the `.venv` environment using the [README quickstart](../README.md#quickstart). From the repository root, run:

```bash
.venv/bin/python scripts/public_smoke.py
.venv/bin/python scripts/run_notebook.py
.venv/bin/python scripts/verify_public_release.py
.venv/bin/python scripts/run_tests.py
```

On Windows, replace `.venv/bin/python` with `.\.venv\Scripts\python.exe`. These checks need no extra packages, GPU, dataset or model snapshot. The Notebook's reported results come from [fixed-results.json](../results/fixed-results.json); Acme records and demonstration probabilities are synthetic.

`run_notebook.py` executes both [English](../notebooks/product-matching-walkthrough.ipynb) and [Chinese](../notebooks/product-matching-walkthrough.zh-CN.ipynb) plain-Python cells and compares saved outputs. The notebooks can also run in a Python 3.11+ Jupyter kernel.

`verify_public_release.py` checks inventory coverage, hashes, reported arithmetic and local links. After editing public files, regenerate the inventory with `.venv/bin/python scripts/build_public_manifest.py` before running the verifier; on Windows use the same explicit project interpreter. The inventory excludes itself, local environments, caches and Git internals.

## Local files and work directory

Keep `.venv` for lightweight checks and `.venv-gpu` for model dependencies. Store product inputs and run outputs beside the checkout:

```text
product-entity-matching/         public checkout; .venv/ and .venv-gpu/
product-matching-work/
  inputs/
    walmart_amazon_exp_data.zip
    roberta/                    files listed in ROBERTA_SNAPSHOT.json
    snapshot.json               generated local snapshot manifest
    minilm/<revision>/          optional; required for S33
  data/                         created by reconstruct-roles
    roles/                      fit/dev/calibration/evaluation-business.json
    private/                    separate *-labels.json files
  prepared/                     features, tokens and fit IDF
  runs/                         training checkpoints and metadata
  predictions/                  reference, seed and ensemble predictions
  results/                      selection, calibration and evaluation reports
```

Run the following directory setup from the repository root. Leave `data/` absent so reconstruction can create a new output directory.

**macOS / Linux — directory setup and standard-library commands**

```bash
mkdir -p ../product-matching-work/inputs/roberta ../product-matching-work/prepared ../product-matching-work/runs ../product-matching-work/predictions ../product-matching-work/results
```

**Windows · PowerShell**

```powershell
New-Item -ItemType Directory -Force -Path "../product-matching-work/inputs/roberta", "../product-matching-work/prepared", "../product-matching-work/runs", "../product-matching-work/predictions", "../product-matching-work/results" | Out-Null
```

Model commands were checked only in the Windows CUDA environment below. The macOS/Linux directory commands do not imply validated model execution on those platforms. Each model command needs a new output file or directory; use another work directory for a repeat run rather than overwriting retained results.

| File | Source or condition |
| --- | --- |
| Original Walmart–Amazon ZIP | Obtain the [DeepMatcher source archive](https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Walmart-Amazon/walmart_amazon_exp_data.zip) separately and place it under `inputs/` |
| RoBERTa backbone/tokenizer | Obtain the official [FacebookAI/roberta-base revision](https://huggingface.co/FacebookAI/roberta-base/tree/e2da8e2f811d1448a5b465c236feacd80ffbac7b); place the exact files identified by [ROBERTA_SNAPSHOT.json](../reproducibility/ROBERTA_SNAPSHOT.json) in `inputs/roberta/` |
| Original trained C+ weights | **No public download entry.** Exact replay requires already holding the three trusted checkpoints identified by [FINAL_MODEL.json](../reproducibility/FINAL_MODEL.json) |
| New trained weights | Created by explicit local training; prediction also requires each seed's `run.json`. Fresh end-to-end retraining has not been validated |
| MiniLM for S33 | Obtain the official [paraphrase-multilingual-MiniLM-L12-v2 revision](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/tree/e8f8c211226b894fcb81acc59f3b34ba3efd5f42); the local folder's final name must be `e8f8c211226b894fcb81acc59f3b34ba3efd5f42` |

The repository and CLI do not download these files. Follow their third-party terms when obtaining and using them. Public aggregate results can be read without any of these local artifacts.

## Observed GPU environment

Existing-artifact replay reused this environment:

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

[EXPERIMENT_LOCK.json](../reproducibility/EXPERIMENT_LOCK.json) records these observations. [requirements-train.lock.txt](../requirements-train.lock.txt) pins direct packages and the CUDA 12.6 wheel channel; it does not completely lock transitive dependencies, wheel hashes or drivers. Other platforms and GPU combinations have not been validated.

## Windows project environment for model commands

These are setup instructions for a compatible Windows CUDA machine. A fresh installation of this environment has not been validated; replay reused an existing environment. From the repository root:

```powershell
py -3.11 --version
py -3.11 -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe --version
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements-train.lock.txt
.\.venv-gpu\Scripts\python.exe -m pip check
```

If `.venv-gpu` already exists, check its interpreter and reuse it rather than creating it again. The CUDA wheel channel follows [PyTorch's official previous-version instructions](https://pytorch.org/get-started/previous-versions/) for PyTorch 2.8.0/CUDA 12.6. A compatible driver and supported GPU are external prerequisites. Keep source files and the sibling work directory outside both virtual environments.

The [command guide](CORE_USAGE.md) uses `.\.venv-gpu\Scripts\python.exe` explicitly for model commands, so activating the environment is optional.

## Configuration and selection rules

| Public description | What it identifies |
| --- | --- |
| [FINAL_MODEL.json](../reproducibility/FINAL_MODEL.json) | Final settings, seeds, selected epoch and original checkpoint hashes |
| [ROBERTA_SNAPSHOT.json](../reproducibility/ROBERTA_SNAPSHOT.json) | Exact backbone/tokenizer file sizes and hashes |
| [ROLE_RECONSTRUCTION.json](../reproducibility/ROLE_RECONSTRUCTION.json) | Ordered source positions, native missing masks and expected role-input hashes |
| [FEATURES.json](../reproducibility/FEATURES.json) | Fixed 42-feature order and groups |
| [EXPERIMENT_LOCK.json](../reproducibility/EXPERIMENT_LOCK.json) | Data roles, selection/calibration assignments and observed environment |

Title IDF is fitted only on fit records. Development predictions select one common epoch for all three seeds; calibration labels select the cutoff before evaluation scoring. Evaluation labels are separate from model inputs and are not used for tuning. These interface rules do not reconstruct the fixed pool's historical exposure record.

`reconstruct-roles` restores four fixed business files from the pinned ZIP, including the original 23 model-number and 14 category missing masks. All file hashes and membership orders matched retained inputs. A new random split would create a different experiment. New training may produce different weights and metrics. See the [data card](DATA_CARD.md) and [limitations](LIMITATIONS.md).
