# Command and input guide

**English** | [简体中文](zh-CN/CORE_USAGE.md)

## Start here

Follow the [README quickstart](../README.md#quickstart) to create a Python 3.11+ project environment, run standard-library checks and preview the website. This guide then explains how product fields become model inputs and how preparation, training and evaluation connect. See the [reproduction guide](REPRODUCIBILITY.md) for GPU dependencies and required local files. C+ names the final RoBERTa model with 42 comparison features in this project.

## Model command prerequisites

The package exposes field preparation, fixed references, C+ prediction, calibration and evaluation through one entry point. Help and public checks use the standard library. Token preparation and model commands additionally need the pinned dependencies and local artifacts described in [REPRODUCIBILITY](REPRODUCIBILITY.md).

```bash
python scripts/matching.py --help
python scripts/matching.py prepare --help
python scripts/matching.py predict --help
```

All examples run from the repository root. Paths such as `inputs/fit-business.json` are conventions for files you supply, not bundled datasets. Create new output paths: commands fail on existing outputs, rather than silently replacing them or retrying. Importing the package starts no training and downloads no model.

## Pair records and frozen roles

A business pair contains `pair_id`, `left`, `right`, `left_native` and `right_native`. Each business side has `normalized` and `quality` dictionaries; each native side has `modelno` and `category`. `data.business_record(raw)` and `data.native_record(raw)` produce these views from explicit fields. Only `fit` pair inputs contain a binary integer `label`.

Development, calibration and evaluation labels are separate `{pair_id, label}` lists. They are joined by ID only for selection, calibration or evaluation. IDs and target labels never enter the feature vector. The [data card](DATA_CARD.md) explains all fields and role counts.

If you already hold the original frozen views, native sidecars and ordered membership IDs, `project-role` joins them:

```bash
python scripts/matching.py project-role --views inputs/fit-views.json --sidecars inputs/fit-native.json --kept-ids inputs/fit-ids.json --role fit --output inputs/fit-business.json
```

It does not infer membership or create a random split. To start from the original ZIP, use the fixed-membership reconstruction below.

## Reconstruct roles from the original ZIP

Obtain the ZIP from the [DeepMatcher source](https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Walmart-Amazon/walmart_amazon_exp_data.zip) under its source terms. The command downloads nothing and checks the ZIP and five CSV members against their sizes, SHA256 digests and exact columns.

```bash
python scripts/matching.py reconstruct-roles --source-archive inputs/walmart_amazon_exp_data.zip --recipe reproducibility/ROLE_RECONSTRUCTION.json --output ../product-matching-inputs
```

Use a nonexistent output directory with an existing parent. `roles/` contains four `*-business.json` inputs; only fit contains labels. `private/` contains four separate `*-labels.json` files for the existing selection and metric commands. `RECONSTRUCTION_RECEIPT.json` is written last. A failed check produces no successful receipt and existing outputs are never overwritten.

[ROLE_RECONSTRUCTION.json](../reproducibility/ROLE_RECONSTRUCTION.json) contains ordered source CSV record positions, original native missing-field masks and input/output digests, with no product values, labels or weights. Positions count parsed CSV records, starting with the header at one; a quoted multiline value is still one record. The recipe restores membership rather than choosing a new split or certifying historical non-exposure.

The command was checked on the original ZIP: all four business inputs, with 3738/199/209/555 pairs, matched the frozen files byte for byte. It preserved 23 model-number and 14 category missing masks. Point subsequent `--rows` arguments to the matching files under `../product-matching-inputs/roles/`; labels remain separate arguments for selection, calibration and evaluation. This step runs no training or model inference.

## Feature and token preparation

Supply a RoBERTa snapshot whose file hashes match [ROBERTA_SNAPSHOT.json](../reproducibility/ROBERTA_SNAPSHOT.json). `snapshot-manifest` records the supplied files. Preparation and GPU commands compare the contents both with that manifest and with the pinned snapshot; a matching revision name alone is insufficient.

```bash
python scripts/matching.py snapshot-manifest --snapshot inputs/roberta --output inputs/snapshot.json
python scripts/matching.py prepare --rows inputs/fit-business.json --role fit --snapshot inputs/roberta --snapshot-manifest inputs/snapshot.json --output prepared/fit
python scripts/matching.py prepare --rows inputs/dev-business.json --role dev --idf-state prepared/fit/idf.json --snapshot inputs/roberta --snapshot-manifest inputs/snapshot.json --output prepared/dev
```

Fit preparation creates the title IDF state. Every subsequent role requires `--idf-state` from that fit directory. For calibration and evaluation, use the same command pattern with the corresponding role and label-free pair file.

The output directory contains `features.json`, `neural.json`, `truncation.json` and, for fit, `idf.json`. The feature file includes `x42`; the neural file contains `pair_id`, `ab`, `ba` and `x42`, plus `label` only for fit. AB/BA serialization uses the field budgets and 256-token limit documented in the [model card](MODEL_CARD.md).

## Fixed-seed training

The fixed C+ experiment trains seeds 42, 43 and 44 in separate processes. Each run keeps six epoch checkpoints so development data can select one common epoch.

```bash
python scripts/matching.py train --fit prepared/fit/neural.json --snapshot inputs/roberta --snapshot-manifest inputs/snapshot.json --seed 42 --output runs/seed42
```

Repeat with `--seed 43 --output runs/seed43` and `--seed 44 --output runs/seed44`. Training preserves the fixed BF16 encoder/FP32 head and parameter settings. It records applied updates and checkpoint hashes in `run.json`. Actual CUDA and nonfinite failures stop the command; there is no automatic overwrite or retry. Host RAM/commit thresholds do not terminate training.

C0 is trained independently with `--arm c-zero` for four training epochs with the original six-epoch scheduler. Its prediction command is `predict-ablation`, which zeros the feature vector as in training. It is an experimental control rather than a separate application workflow. No fresh end-to-end training reproduction is claimed by this guide.

## Prediction and common-epoch selection

For newly trained weights, supply three checkpoints in seed order 42, 43, 44, their hashes and their three `run.json` metadata files. Replace `SHA42`, `SHA43` and `SHA44` with the recorded checkpoint digests.

```bash
python scripts/matching.py predict --rows prepared/dev/neural.json --snapshot inputs/roberta --snapshot-manifest inputs/snapshot.json --checkpoints runs/seed42/epoch-4.safetensors runs/seed43/epoch-4.safetensors runs/seed44/epoch-4.safetensors --sha256 SHA42 SHA43 SHA44 --checkpoint-metadata runs/seed42/run.json runs/seed43/run.json runs/seed44/run.json --output predictions/dev-e4
```

For the original selected weights, use the three digests in [FINAL_MODEL.json](../reproducibility/FINAL_MODEL.json); original artifacts do not require new training metadata. New weights are checked for arm, seed, selected epoch, hash and common fit identity.

Prediction launches a fresh worker per seed, validates worker identity and the exact pair IDs, then averages probabilities. `_predict-one` is internal. Use `--epoch` to identify each of the six development checkpoints during a reproduction; the saved final result uses common epoch 4.

`select-epoch` takes a JSON document with `references` and `cplus` sections. `references` holds all four reference predictions; `cplus` holds all six epoch predictions for each of seeds 42, 43 and 44. Values are score rows or prediction documents. It maximizes ensemble development AP, then P@100, then prefers the earlier epoch:

```bash
python scripts/matching.py select-epoch --predictions predictions/dev-all-epochs.json --labels inputs/dev-labels.json --output results/selected-epoch.json
```

Evaluation labels must not be used to select an epoch, seed or model setting.

## Check prediction files before reading labels

Save the ordered pair IDs from the label-free evaluation input as a JSON list before prediction. After receiving the C+ ensemble and L42 prediction files, check both against that list:

```bash
python scripts/matching.py validate-predictions --pair-ids inputs/evaluation-pair-ids.json --candidate predictions/eval/scores.json --reference predictions/l42-eval.json --output results/prediction-check.json
```

The command rejects missing, extra or duplicate IDs, nonfinite or out-of-range scores, labels and incompatible document fields. Score order may differ; correspondence is checked by ID. Its receipt records the three input-file hashes and pair count. It reads no target-label file, computes no metric and loads no model. This is a file-consistency check: it does not verify checkpoint provenance or establish that the sample is independent. The same input files must be used for subsequent evaluation; compare their hashes with this receipt.

## Calibration and evaluation

Predict on calibration pairs, then select the cutoff using only calibration labels. Predict on evaluation pairs and reuse the saved cutoff:

```bash
python scripts/matching.py calibrate --scores predictions/cal/scores.json --labels inputs/cal-labels.json --output results/cal-threshold.json
python scripts/matching.py evaluate --scores predictions/eval/scores.json --labels inputs/eval-labels.json --threshold results/cal-threshold.json --output results/evaluation.json
```

Calibration maximizes F1, then precision, then chooses the higher cutoff. Evaluation reports AP, P@100 and an integer confusion matrix. AP groups tied scores; Top100 breaks ties by `pair_id`. P@100 requires at least 100 pairs. These commands calculate metrics; they do not establish independent new Walmart–Amazon sample performance.

## Tabular references and S33

`reference-fit` and `reference-predict` use the prepared feature rows. This B22 example also illustrates the file flow for L30/L42:

```bash
python scripts/matching.py reference-fit --method B22 --rows prepared/fit/features.json --output runs/b22
python scripts/matching.py reference-predict --method B22 --rows prepared/dev/features.json --role dev --checkpoint runs/b22/reference.pkl --sha256 REFERENCE_SHA --output predictions/b22-dev.json
```

`REFERENCE_SHA` is the digest of your trusted local reference file. Original reference artifacts are supported at their pinned hashes. The B22 loader maps one legacy class name without importing the old package; a new fit produces a different artifact. Loading an untrusted pickle is unsafe.

S33 first needs `semantic-features` with the corresponding role, prepared rows and fixed local MiniLM snapshot. Pass its output with `--semantic-features` to both reference commands. The CLI checks role/pair identity and the fixed semantic input configuration. Heavy packages load only when these commands request them.

Existing-weight replay through this entry completed 20 commands and 23 independent result checks with zero differences. [REPRODUCIBILITY](REPRODUCIBILITY.md) records the observed environment and separates that replay from fresh training and raw-source reconstruction.
