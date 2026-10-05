# 命令与输入指南

[🌐 English](../CORE_USAGE.md) | **🇨🇳 简体中文**

## 从这里开始

先按 [README 快速开始](../../README.zh-CN.md#快速开始)完成标准库检查，再按[复现指南](REPRODUCIBILITY.md)创建仓库旁工作目录、提供原 ZIP 与固定快照，并在兼容 Windows CUDA 机器上设置 `.venv-gpu`。下方示例均从仓库根目录运行；模型命令采用 PowerShell，明确使用 GPU 环境解释器：

```powershell
$matchingWork = "../product-matching-work"
$matchingPython = ".\.venv-gpu\Scripts\python.exe"
& $matchingPython scripts/matching.py --help
```

后续命令在同一个 PowerShell 会话中使用这两个变量。重建和两个小型 JSON 生成脚本使用轻量 `.venv` 解释器。GPU 运行仅在 Windows 11 RTX 3060 笔记本上核对过，其他平台/GPU 组合尚未验证。

C+ 指最终采用的 RoBERTa 与 42 项比较特征融合模型。以下步骤说明新的本地训练与评分。这些命令示例尚未在新环境中完整执行，完整重训尚未验证；原选定权重没有公开下载入口，已经持有权重的读者可使用预测部分说明的原权重选项。所有输出必须使用新路径，命令拒绝已有输出。

## 商品对记录与角色重建

业务商品对包含 `pair_id`、`left`、`right`、`left_native`、`right_native`。每侧业务记录有 `normalized`、`quality` 字典，每侧原生记录有 `modelno`、`category`。只有 `fit` 输入包含整数 0 或 1 的 `label`；其他标签单独放在 `{pair_id, label}` 列表中，仅在轮数选择、校准或评价时按 ID 连接。ID 和标签不进入特征向量。详见[数据卡](DATA_CARD.md)。

将原 ZIP 放到 `../product-matching-work/inputs/walmart_amazon_exp_data.zip`。以下命令按固定配方核对压缩包及其五个 CSV 文件。`../product-matching-work/data` 必须不存在，父目录须已存在：

**macOS / Linux**

```bash
.venv/bin/python scripts/matching.py reconstruct-roles --source-archive ../product-matching-work/inputs/walmart_amazon_exp_data.zip --recipe reproducibility/ROLE_RECONSTRUCTION.json --output ../product-matching-work/data
```

**Windows · PowerShell**

```powershell
.\.venv\Scripts\python.exe scripts/matching.py reconstruct-roles --source-archive ../product-matching-work/inputs/walmart_amazon_exp_data.zip --recipe reproducibility/ROLE_RECONSTRUCTION.json --output ../product-matching-work/data
```

| 角色 | `../product-matching-work/data/` 下的业务输入 | 同一目录下的标签文件 |
| --- | --- | --- |
| 训练 | `roles/fit-business.json` | `private/fit-labels.json` |
| 开发 | `roles/dev-business.json` | `private/dev-labels.json` |
| 校准 | `roles/calibration-business.json` | `private/calibration-labels.json` |
| 评价 | `roles/evaluation-business.json` | `private/evaluation-labels.json` |

重建已使保留的 3,738/199/209/555 对业务输入逐字节一致，包含原有 23 处型号与 14 处类别置空标记。[ROLE_RECONSTRUCTION.json](../../reproducibility/ROLE_RECONSTRUCTION.json)保存有序源位置、置空标记与输入输出摘要。CSV 位置按解析后的记录计数，表头为第 1 条；带换行的引号字段仍算一条记录。命令恢复原成员，最后写入 `RECONSTRUCTION_RECEIPT.json`；不选择新划分，也不补全历史接触记录。

若已经持有原视图、原生字段附表与有序成员 ID，也可用 `project-role --views … --sidecars … --kept-ids … --role fit --output …` 连接输入。提供的文件与新输出仍放在仓库旁工作目录。

## 准备特征与 token

将 [ROBERTA_SNAPSHOT.json](../../reproducibility/ROBERTA_SNAPSHOT.json)标识的准确骨干/分词器文件放到 `$matchingWork/inputs/roberta`。模型与版本名称一致仍不充分；准备命令同时按固定快照和本地清单核对文件哈希。

```powershell
& $matchingPython scripts/matching.py snapshot-manifest --snapshot "$matchingWork/inputs/roberta" --output "$matchingWork/inputs/snapshot.json"
& $matchingPython scripts/matching.py prepare --rows "$matchingWork/data/roles/fit-business.json" --role fit --snapshot "$matchingWork/inputs/roberta" --snapshot-manifest "$matchingWork/inputs/snapshot.json" --output "$matchingWork/prepared/fit"
foreach ($matchingRole in "dev", "calibration", "evaluation") {
    & $matchingPython scripts/matching.py prepare --rows "$matchingWork/data/roles/$matchingRole-business.json" --role $matchingRole --idf-state "$matchingWork/prepared/fit/idf.json" --snapshot "$matchingWork/inputs/roberta" --snapshot-manifest "$matchingWork/inputs/snapshot.json" --output "$matchingWork/prepared/$matchingRole"
    if ($LASTEXITCODE -ne 0) { throw "Preparation failed: $matchingRole" }
}
```

训练准备生成 `prepared/fit/idf.json`，开发、校准和评价均复用该状态。每个角色目录包含 `features.json`、`neural.json`、`truncation.json`。神经输入行包含 `pair_id`、`ab`、`ba`、`x42`，只有训练角色包含 `label`。[模型卡](MODEL_CARD.md)说明字段预算与 256-token 限制。

评价预测前，从无标签业务输入保存有序 ID。JSON 格式为唯一非空字符串列表，例如 `["pair-id-1", "pair-id-2"]`。将以下代码保存为 `../product-matching-work/save-evaluation-ids.py`，然后在仓库根目录运行：

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

## 拟合表格对照

B22、L30、L42 使用已准备的特征行。以下命令生成各个对照及其开发预测：

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

检查点哈希从对应的 `run.json` 读取。仅加载可信本地 pickle 文件；哈希核对文件身份，不证明来源可信。

S33 还需要[复现指南](REPRODUCIBILITY.md)中的固定本地 MiniLM 版本，末级文件夹名称为版本哈希。为 S33 使用的每个角色准备语义特征，再将匹配文件传入两个对照命令：

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

`select-epoch` 需要全部四份开发对照预测。S33 语义特征准备使用 CUDA，不属于轻量检查。

## 训练三个固定种子

C+ 分别训练种子 42、43、44，每次保留六轮检查点，在 `run.json` 中记录检查点哈希：

```powershell
foreach ($matchingSeed in 42, 43, 44) {
    & $matchingPython scripts/matching.py train --fit "$matchingWork/prepared/fit/neural.json" --snapshot "$matchingWork/inputs/roberta" --snapshot-manifest "$matchingWork/inputs/snapshot.json" --seed $matchingSeed --output "$matchingWork/runs/seed$matchingSeed"
    if ($LASTEXITCODE -ne 0) { throw "Training failed: seed $matchingSeed" }
}
```

固定配置使用 BF16 编码器 autocast、FP32 参数/分类头及六轮调度器。每个种子的运行上限为 3,660 秒，CUDA 与非有限数错误会停止命令，失败后不自动重试。C0 在独立输出目录中使用 `--arm c-zero`，训练四轮并保留原六轮调度器；使用 `predict-ablation` 预测，特征同样置零。

## 开发预测与共同轮数选择

新训练权重的预测需要按种子 42、43、44 顺序提供三个检查点、哈希与三份训练 `run.json`。以下代码读取每一轮对应哈希，生成 `predictions/dev-e1/` 至 `dev-e6/`，每个目录包含三个 `seed-*.json` 预测与集成 `scores.json`：

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

`dev-all-epochs.json` 只有两个顶层键。`references` 包含 `B22`、`L30`、`S33`、`L42`；`cplus` 包含字符串键 `"42"`、`"43"`、`"44"`，每个种子再包含字符串轮次键 `"1"` 至 `"6"`。各项值是实际预测文档或 `{pair_id, score}` 行列表，不能只填写文件路径字符串；每份预测必须覆盖开发 ID。

将以下标准库代码保存为 `../product-matching-work/build-development-input.py`。它读取上方步骤生成的文件，将内容填入规定结构：

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

轮数选择先最大化开发集成 AP，再比较 P@100，最后选择较早轮次。保留的原结果选择共同第 4 轮；新运行应使用自身的开发选择。不得用评价标签选择轮数、种子或模型设置。

**原权重选项：** 若已经持有原选定检查点，使用 [FINAL_MODEL.json](../../reproducibility/FINAL_MODEL.json)中的三个哈希，设置 `--epoch 4` 并省略 `--checkpoint-metadata`。可信权重放在 `$matchingWork/runs/`，按种子顺序传入实际路径。原最终权重本身不提供重复轮数选择所需的六轮开发预测。

## 校准与评价预测

新权重采用刚选定的共同轮数。以下代码重新读取该轮检查点与哈希，预测两个无标签角色：

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

`validate-predictions` 将完整 C+ 集成文档与评价 L42 文档核对到保存的 ID，拒绝缺失、多余或重复 ID、非有限/越界分数、标签及不兼容字段。分数行可以不同顺序，因为程序按 ID 对齐。回执记录输入哈希与商品对数；评价时使用同一批预测文件，并将哈希与回执核对。这是文件一致性检查，不证明历史样本独立或核验检查点来源。

## 校准与评价指标

仅用校准分数与标签选择阈值，再原样应用于评价：

```powershell
& $matchingPython scripts/matching.py calibrate --scores "$matchingWork/predictions/calibration/scores.json" --labels "$matchingWork/data/private/calibration-labels.json" --output "$matchingWork/results/cal-threshold.json"
& $matchingPython scripts/matching.py evaluate --scores "$matchingWork/predictions/evaluation/scores.json" --labels "$matchingWork/data/private/evaluation-labels.json" --threshold "$matchingWork/results/cal-threshold.json" --output "$matchingWork/results/evaluation.json"
```

校准依次最大化 F1、精确率，再选择更高阈值。评价报告 AP、P@100 与整数混淆矩阵。AP 按并列分数组处理；P@100 的同分按 `pair_id` 排序，需要至少 100 对。报告描述提供的商品池，保留结果的解释范围见[局限](LIMITATIONS.md)。
