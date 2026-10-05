# Command and input guide

**🌐 English** | [🇨🇳 简体中文](zh-CN/CORE_USAGE.md)

## Start here

Use the [README quickstart](../README.md#quickstart) for standard-library checks. Then follow the [reproduction guide](REPRODUCIBILITY.md) to create the sibling work directories, provide the original ZIP and pinned snapshots, and set up `.venv-gpu` on a compatible Windows CUDA machine. All examples below run from the repository root. Model commands use PowerShell and the explicit GPU-environment interpreter:

```powershell
$matchingWork = "../product-matching-work"
$matchingPython = ".\.venv-gpu\Scripts\python.exe"
& $matchingPython scripts/matching.py --help
```

Keep these two variables in the same PowerShell session for the later commands. Reconstruction and the two small JSON-building scripts use the lightweight `.venv` interpreter. GPU execution has been checked only on Windows 11 with an RTX 3060 laptop; other platform/GPU combinations are unvalidated.

C+ is the final RoBERTa model with 42 comparison features. The steps below describe new local training and scoring. These command examples have not been executed as a complete run in a new environment; fresh end-to-end retraining has not been validated. The original selected weights have no public download entry. Readers who already hold them can use the original-weight option described under prediction. All outputs must be new paths; commands reject existing outputs.

## Pair records and role reconstruction

A business pair contains `pair_id`, `left`, `right`, `left_native` and `right_native`. Each business side has `normalized` and `quality` dictionaries; each native side has `modelno` and `category`. Only `fit` inputs contain an integer `label` of 0 or 1. Other labels are separate `{pair_id, label}` lists and are joined by ID only for selection, calibration or evaluation. IDs and labels are excluded from the feature vector. See the [data card](DATA_CARD.md).

Place the original ZIP at `../product-matching-work/inputs/walmart_amazon_exp_data.zip`. The following command checks the archive and its five CSV files against the fixed recipe. `../product-matching-work/data` must not exist, and its parent must exist:

**macOS / Linux**

```bash
.venv/bin/python scripts/matching.py reconstruct-roles --source-archive ../product-matching-work/inputs/walmart_amazon_exp_data.zip --recipe reproducibility/ROLE_RECONSTRUCTION.json --output ../product-matching-work/data
```

**Windows · PowerShell**

```powershell
.\.venv\Scripts\python.exe scripts/matching.py reconstruct-roles --source-archive ../product-matching-work/inputs/walmart_amazon_exp_data.zip --recipe reproducibility/ROLE_RECONSTRUCTION.json --output ../product-matching-work/data
```

| Role | Business input under `../product-matching-work/data/` | Label file under the same directory |
| --- | --- | --- |
| fit | `roles/fit-business.json` | `private/fit-labels.json` |
| development | `roles/dev-business.json` | `private/dev-labels.json` |
| calibration | `roles/calibration-business.json` | `private/calibration-labels.json` |
| evaluation | `roles/evaluation-business.json` | `private/evaluation-labels.json` |

The reconstruction matched the retained 3,738/199/209/555 business inputs byte for byte, including 23 model-number and 14 category missing masks. [ROLE_RECONSTRUCTION.json](../reproducibility/ROLE_RECONSTRUCTION.json) contains ordered source positions, missing masks and input/output digests. CSV positions count parsed records, with the header at one; a quoted multiline field remains one record. The command restores the original membership and writes `RECONSTRUCTION_RECEIPT.json` last. It does not choose a new split or recover the incomplete historical exposure record.

If you already have the original views, native sidecars and ordered membership IDs, `project-role --views … --sidecars … --kept-ids … --role fit --output …` is an alternative input route. Keep those supplied files and the new output under the sibling work directory.

## Prepare features and tokens

Put the exact backbone/tokenizer files identified by [ROBERTA_SNAPSHOT.json](../reproducibility/ROBERTA_SNAPSHOT.json) in `$matchingWork/inputs/roberta`. A matching model/revision name alone is insufficient; preparation verifies file hashes against the fixed snapshot as well as the local manifest.

```powershell
& $matchingPython scripts/matching.py snapshot-manifest --snapshot "$matchingWork/inputs/roberta" --output "$matchingWork/inputs/snapshot.json"
& $matchingPython scripts/matching.py prepare --rows "$matchingWork/data/roles/fit-business.json" --role fit --snapshot "$matchingWork/inputs/roberta" --snapshot-manifest "$matchingWork/inputs/snapshot.json" --output "$matchingWork/prepared/fit"
foreach ($matchingRole in "dev", "calibration", "evaluation") {
    & $matchingPython scripts/matching.py prepare --rows "$matchingWork/data/roles/$matchingRole-business.json" --role $matchingRole --idf-state "$matchingWork/prepared/fit/idf.json" --snapshot "$matchingWork/inputs/roberta" --snapshot-manifest "$matchingWork/inputs/snapshot.json" --output "$matchingWork/prepared/$matchingRole"
    if ($LASTEXITCODE -ne 0) { throw "Preparation failed: $matchingRole" }
}
```

Fit preparation creates `prepared/fit/idf.json`; development, calibration and evaluation reuse that state. Each role directory contains `features.json`, `neural.json` and `truncation.json`. Neural rows contain `pair_id`, `ab`, `ba`, `x42` and, only for fit, `label`. The [model card](MODEL_CARD.md) specifies field budgets and the 256-token limit.

Before evaluation prediction, save its ordered IDs from the label-free business input. The JSON format is a list of unique nonempty strings, for example `["pair-id-1", "pair-id-2"]`. Save the following code as `../product-matching-work/save-evaluation-ids.py`, then run it from the repository root:

```python
import json
from pathlib import Path

matching_work = Path("../product-matching-work")
rows = json.loads((matching_work / "data/roles/evaluation-business.json").read_text(encoding="utf-8"))
pair_ids = [row["pair_id"] for row in rows]
with (matching_work / "predictions/evaluation-pair-ids.json").open("x", encoding="utf-8") as stream:
    json.dump(pair_ids, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
```

```powershell
.\.venv\Scripts\python.exe ../product-matching-work/save-evaluation-ids.py
```

## Fit tabular references

B22, L30 and L42 use the prepared feature rows. These commands create each reference and its development prediction:

```powershell
foreach ($matchingMethod in "B22", "L30", "L42") {
    $matchingName = $matchingMethod.ToLowerInvariant()
    & $matchingPython scripts/matching.py reference-fit --method $matchingMethod --rows "$matchingWork/prepared/fit/features.json" --output "$matchingWork/runs/$matchingName"
    if ($LASTEXITCODE -ne 0) { throw "Reference fit failed: $matchingMethod" }
    $matchingRefHash = (Get-Content -Raw "$matchingWork/runs/$matchingName/run.json" | ConvertFrom-Json).checkpoint.sha256
    & $matchingPython scripts/matching.py reference-predict --method $matchingMethod --rows "$matchingWork/prepared/dev/features.json" --role dev --checkpoint "$matchingWork/runs/$matchingName/reference.pkl" --sha256 $matchingRefHash --output "$matchingWork/predictions/$matchingName-dev.json"
    if ($LASTEXITCODE -ne 0) { throw "Reference prediction failed: $matchingMethod" }
}
```

The checkpoint hash comes from the corresponding `run.json`. Use only trusted local pickle files; a hash verifies file identity, not the trustworthiness of its source.

S33 additionally needs the fixed local MiniLM revision described in the [reproduction guide](REPRODUCIBILITY.md). Its final folder name is the revision hash. Prepare semantic features for each role used by S33, then pass the matching file to both reference commands:

```powershell
$matchingMiniLM = "$matchingWork/inputs/minilm/e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
foreach ($matchingRole in "fit", "dev") {
    & $matchingPython scripts/matching.py semantic-features --rows "$matchingWork/prepared/$matchingRole/features.json" --role $matchingRole --snapshot $matchingMiniLM --output "$matchingWork/prepared/$matchingRole/semantic.json"
    if ($LASTEXITCODE -ne 0) { throw "Semantic preparation failed: $matchingRole" }
}
& $matchingPython scripts/matching.py reference-fit --method S33 --rows "$matchingWork/prepared/fit/features.json" --semantic-features "$matchingWork/prepared/fit/semantic.json" --output "$matchingWork/runs/s33"
$matchingRefHash = (Get-Content -Raw "$matchingWork/runs/s33/run.json" | ConvertFrom-Json).checkpoint.sha256
& $matchingPython scripts/matching.py reference-predict --method S33 --rows "$matchingWork/prepared/dev/features.json" --role dev --semantic-features "$matchingWork/prepared/dev/semantic.json" --checkpoint "$matchingWork/runs/s33/reference.pkl" --sha256 $matchingRefHash --output "$matchingWork/predictions/s33-dev.json"
```

The four development reference predictions are required by `select-epoch`. S33 semantic-feature preparation runs on CUDA; it is not part of the lightweight checks.

## Train the three fixed seeds

C+ trains seeds 42, 43 and 44 independently. Each run keeps six epoch checkpoints and their hashes in `run.json`:

```powershell
foreach ($matchingSeed in 42, 43, 44) {
    & $matchingPython scripts/matching.py train --fit "$matchingWork/prepared/fit/neural.json" --snapshot "$matchingWork/inputs/roberta" --snapshot-manifest "$matchingWork/inputs/snapshot.json" --seed $matchingSeed --output "$matchingWork/runs/seed$matchingSeed"
    if ($LASTEXITCODE -ne 0) { throw "Training failed: seed $matchingSeed" }
}
```

The fixed configuration uses BF16 encoder autocast, FP32 parameters/head, and a six-epoch scheduler. Each seed has a 3,660-second runtime limit; CUDA and nonfinite failures stop the command. Failed runs are not automatically retried. C0 uses `--arm c-zero` in separate output directories, trains for four epochs with the original six-epoch scheduler, and uses `predict-ablation` with zeroed features.

## Development predictions and common-epoch selection

For newly trained weights, prediction requires three checkpoints in seed order 42, 43, 44, their hashes, and the three training `run.json` files. The following reads the hashes for every epoch and writes `predictions/dev-e1/` through `dev-e6/`. Each directory includes the three `seed-*.json` predictions and an ensemble `scores.json`:

```powershell
$matchingMetadata = @(42, 43, 44 | ForEach-Object { "$matchingWork/runs/seed$_/run.json" })
foreach ($matchingEpoch in 1..6) {
    $matchingCheckpoints = @(42, 43, 44 | ForEach-Object { "$matchingWork/runs/seed$_/epoch-$matchingEpoch.safetensors" })
    $matchingHashes = @(42, 43, 44 | ForEach-Object {
        $matchingRun = Get-Content -Raw "$matchingWork/runs/seed$_/run.json" | ConvertFrom-Json
        ($matchingRun.checkpoints | Where-Object { $_.epoch -eq $matchingEpoch }).sha256
    })
    & $matchingPython scripts/matching.py predict --rows "$matchingWork/prepared/dev/neural.json" --snapshot "$matchingWork/inputs/roberta" --snapshot-manifest "$matchingWork/inputs/snapshot.json" --checkpoints $matchingCheckpoints --sha256 $matchingHashes --checkpoint-metadata $matchingMetadata --epoch $matchingEpoch --output "$matchingWork/predictions/dev-e$matchingEpoch"
    if ($LASTEXITCODE -ne 0) { throw "Development prediction failed: epoch $matchingEpoch" }
}
```

`dev-all-epochs.json` has exactly two top-level keys. `references` contains `B22`, `L30`, `S33` and `L42`; `cplus` contains string keys `"42"`, `"43"`, `"44"`, each with string epoch keys `"1"` through `"6"`. Each value is the actual prediction document or a list of `{pair_id, score}` rows, not a file-path string. Every prediction must cover the development IDs.

Save this standard-library code as `../product-matching-work/build-development-input.py`. It reads the files created above and embeds their contents in the required structure:

```python
import json
from pathlib import Path

predictions = Path("../product-matching-work/predictions")

def read_prediction(path):
    return json.loads(path.read_text(encoding="utf-8"))

document = {
    "references": {
        method: read_prediction(predictions / f"{method.lower()}-dev.json")
        for method in ("B22", "L30", "S33", "L42")
    },
    "cplus": {
        str(seed): {
            str(epoch): read_prediction(predictions / f"dev-e{epoch}/seed-{seed}.json")
            for epoch in range(1, 7)
        }
        for seed in (42, 43, 44)
    },
}
with (predictions / "dev-all-epochs.json").open("x", encoding="utf-8") as stream:
    json.dump(document, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
```

```powershell
.\.venv\Scripts\python.exe ../product-matching-work/build-development-input.py
& $matchingPython scripts/matching.py select-epoch --predictions "$matchingWork/predictions/dev-all-epochs.json" --labels "$matchingWork/data/private/dev-labels.json" --output "$matchingWork/results/selected-epoch.json"
```

Selection maximizes ensemble development AP, then P@100, then prefers the earlier epoch. The retained original result selected common epoch 4; a new run must use its own development selection. Evaluation labels must not select an epoch, seed or model setting.

**Original-weight option:** if you already hold the original selected checkpoints, use the three hashes in [FINAL_MODEL.json](../reproducibility/FINAL_MODEL.json), set `--epoch 4`, and omit `--checkpoint-metadata`. Place the trusted weights under `$matchingWork/runs/` and pass their actual paths in seed order. The original final weights do not supply the six-epoch development predictions needed to repeat epoch selection.

## Calibration and evaluation predictions

For new weights, use the common epoch just selected. The following recreates the checkpoint/hash arrays for that epoch and predicts the two label-free roles:

```powershell
$matchingEpoch = (Get-Content -Raw "$matchingWork/results/selected-epoch.json" | ConvertFrom-Json).selected_epoch
$matchingCheckpoints = @(42, 43, 44 | ForEach-Object { "$matchingWork/runs/seed$_/epoch-$matchingEpoch.safetensors" })
$matchingHashes = @(42, 43, 44 | ForEach-Object {
    $matchingRun = Get-Content -Raw "$matchingWork/runs/seed$_/run.json" | ConvertFrom-Json
    ($matchingRun.checkpoints | Where-Object { $_.epoch -eq $matchingEpoch }).sha256
})
foreach ($matchingRole in "calibration", "evaluation") {
    & $matchingPython scripts/matching.py predict --rows "$matchingWork/prepared/$matchingRole/neural.json" --snapshot "$matchingWork/inputs/roberta" --snapshot-manifest "$matchingWork/inputs/snapshot.json" --checkpoints $matchingCheckpoints --sha256 $matchingHashes --checkpoint-metadata $matchingMetadata --epoch $matchingEpoch --output "$matchingWork/predictions/$matchingRole"
    if ($LASTEXITCODE -ne 0) { throw "Prediction failed: $matchingRole" }
}
$matchingRefHash = (Get-Content -Raw "$matchingWork/runs/l42/run.json" | ConvertFrom-Json).checkpoint.sha256
& $matchingPython scripts/matching.py reference-predict --method L42 --rows "$matchingWork/prepared/evaluation/features.json" --role evaluation --checkpoint "$matchingWork/runs/l42/reference.pkl" --sha256 $matchingRefHash --output "$matchingWork/predictions/l42-evaluation.json"
& $matchingPython scripts/matching.py validate-predictions --pair-ids "$matchingWork/predictions/evaluation-pair-ids.json" --candidate "$matchingWork/predictions/evaluation/scores.json" --reference "$matchingWork/predictions/l42-evaluation.json" --output "$matchingWork/results/prediction-check.json"
```

`validate-predictions` checks the complete C+ ensemble and evaluation L42 documents against the saved IDs. It rejects missing, extra or duplicate IDs, nonfinite/out-of-range scores, labels and incompatible document fields. Score order may differ because matching is by ID. Its receipt records input hashes and pair count; use these same prediction files for evaluation and compare their hashes with the receipt. This checks file consistency, not historical sample independence or checkpoint provenance.

## Calibrate and evaluate

Select the threshold from calibration scores and labels, then apply it unchanged to evaluation:

```powershell
& $matchingPython scripts/matching.py calibrate --scores "$matchingWork/predictions/calibration/scores.json" --labels "$matchingWork/data/private/calibration-labels.json" --output "$matchingWork/results/cal-threshold.json"
& $matchingPython scripts/matching.py evaluate --scores "$matchingWork/predictions/evaluation/scores.json" --labels "$matchingWork/data/private/evaluation-labels.json" --threshold "$matchingWork/results/cal-threshold.json" --output "$matchingWork/results/evaluation.json"
```

Calibration maximizes F1, then precision, then chooses the higher cutoff. Evaluation reports AP, P@100 and an integer confusion matrix. AP groups tied scores; P@100 breaks ties by `pair_id` and requires at least 100 pairs. The reports describe the supplied pair pool. See [limitations](LIMITATIONS.md) for the interpretation of the retained results.
