# 命令与输入指南

[English](../CORE_USAGE.md) | **简体中文**

## 从哪里开始

先按 [README 快速开始](../../README.zh-CN.md#快速开始)建立 Python 3.11+ 项目环境，运行标准库检查并预览网页。下文说明如何从商品字段构建模型输入，以及如何准备、训练和评价；GPU 依赖与本地文件要求见[复现指南](REPRODUCIBILITY.md)。C+ 是本项目最终采用的 RoBERTa 与 42 项比较特征融合模型的简称。

## 模型命令前提

程序包通过同一个入口提供字段准备、固定对照、C+ 预测、校准与评价。帮助与公开检查只使用标准库；token 准备和模型命令还需要[复现指南](REPRODUCIBILITY.md)中的固定依赖与本地产物。

```bash
python scripts/matching.py --help
python scripts/matching.py prepare --help
python scripts/matching.py predict --help
```

以下示例均从仓库根目录执行。`inputs/fit-business.json` 等路径是本地文件约定，不是随仓库分发的数据集。请使用新输出路径：命令遇到已有输出会失败，不会静默替换或自动重试。导入程序包不启动训练，也不下载模型。

## 商品对记录与固定角色

业务商品对包含 `pair_id`、`left`、`right`、`left_native`、`right_native`。每侧业务记录有 `normalized`、`quality` 字典，每侧原生记录有 `modelno`、`category`。`data.business_record(raw)` 与 `data.native_record(raw)` 将明确字段转成这些视图。只有 `fit` 输入包含整数二元 `label`。

开发、校准与评价标签是单独的 `{pair_id, label}` 列表，只在选轮数、校准或评价时按 ID 连接。ID 和目标标签不进入特征向量。[数据卡](DATA_CARD.md)说明字段与角色数量。

若已有原固定视图、原生字段附表和有序成员 ID，可由 `project-role` 连接：

```bash
python scripts/matching.py project-role --views inputs/fit-views.json --sidecars inputs/fit-native.json --kept-ids inputs/fit-ids.json --role fit --output inputs/fit-business.json
```

该命令不推断成员，也不生成随机划分。若从原 ZIP 开始，使用下面的固定成员重建入口。

## 从原始 ZIP 重建角色输入

从 [DeepMatcher 原始来源](https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Walmart-Amazon/walmart_amazon_exp_data.zip)自行获取 ZIP，并遵守来源使用条件。程序不下载数据；会检查 ZIP 及五个 CSV 的尺寸、SHA256 和列名。

```bash
python scripts/matching.py reconstruct-roles --source-archive inputs/walmart_amazon_exp_data.zip --recipe reproducibility/ROLE_RECONSTRUCTION.json --output ../product-matching-inputs
```

输出目录必须不存在，父目录须已存在。`roles/` 下生成四个 `*-business.json`，只有 fit 输入带标签；`private/` 下生成四个独立 `*-labels.json`，用于已有选择与评价命令。最后写入 `RECONSTRUCTION_RECEIPT.json`；校验失败不创建成功回执，不覆盖旧目录。

[ROLE_RECONSTRUCTION.json](../../reproducibility/ROLE_RECONSTRUCTION.json)仅保存有序源 CSV 记录位置、原型号/类别置空标记及输入输出哈希，没有商品值、标签或权重。记录序号按 CSV 解析后的记录计数，表头为第 1 条，带换行的引号文本仍算一条记录。它恢复原成员，不重新决定划分，也不证明历史样本未被接触。

该命令已在原 ZIP 上验收：3738／199／209／555 四份业务输入逐字节一致，保留23处型号和14处类别置空。将后续准备命令的 `--rows` 指向对应的 `../product-matching-inputs/roles/` 文件即可；标签仍单独传入选择、校准或评价命令。本步骤没有训练或推断模型。

## 特征与 token 准备

提供文件哈希与 [ROBERTA_SNAPSHOT.json](../../reproducibility/ROBERTA_SNAPSHOT.json) 相符的 RoBERTa 快照。`snapshot-manifest` 记录所提供的文件；准备和 GPU 命令同时核对该清单与固定快照，不能只凭版本名判断相同。

```bash
python scripts/matching.py snapshot-manifest --snapshot inputs/roberta --output inputs/snapshot.json
python scripts/matching.py prepare --rows inputs/fit-business.json --role fit --snapshot inputs/roberta --snapshot-manifest inputs/snapshot.json --output prepared/fit
python scripts/matching.py prepare --rows inputs/dev-business.json --role dev --idf-state prepared/fit/idf.json --snapshot inputs/roberta --snapshot-manifest inputs/snapshot.json --output prepared/dev
```

训练角色的准备步骤创建标题 IDF。其他角色均需通过 `--idf-state` 复用该训练目录的状态。准备校准和评价数据时，沿用命令形式，替换对应角色及不含标签的商品对文件。

输出目录包含 `features.json`、`neural.json`、`truncation.json`；训练角色还包含 `idf.json`。特征文件包含 `x42`；神经输入文件包含 `pair_id`、`ab`、`ba`、`x42`，只有训练角色包含 `label`。AB/BA 序列使用[模型卡](MODEL_CARD.md)中的字段预算和 256-token 限制。

## 固定种子训练

固定 C+ 实验在独立进程中训练种子 42、43、44。每次运行保留六个轮次检查点，由开发数据选择共同轮数。

```bash
python scripts/matching.py train --fit prepared/fit/neural.json --snapshot inputs/roberta --snapshot-manifest inputs/snapshot.json --seed 42 --output runs/seed42
```

分别改用 `--seed 43 --output runs/seed43` 和 `--seed 44 --output runs/seed44` 运行另外两个种子。训练保留固定的 BF16 编码器、FP32 分类头及参数设置，并在 `run.json` 中记录实际更新与检查点哈希。CUDA 或非有限数错误会停止命令，不自动覆盖或重试；主机 RAM/commit 阈值不用于终止训练。

C0 通过 `--arm c-zero` 独立训练四轮，保留原六轮调度器；预测使用 `predict-ablation`，像训练一样将特征向量置零。它是实验对照，不是另一个应用产品。本指南不声称已经完成从头训练的完整复现。

## 预测与共同轮数选择

新训练权重需按种子 42、43、44 的顺序提供三个检查点、对应哈希及三份 `run.json`。将 `SHA42`、`SHA43`、`SHA44` 替换为记录的检查点摘要。

```bash
python scripts/matching.py predict --rows prepared/dev/neural.json --snapshot inputs/roberta --snapshot-manifest inputs/snapshot.json --checkpoints runs/seed42/epoch-4.safetensors runs/seed43/epoch-4.safetensors runs/seed44/epoch-4.safetensors --sha256 SHA42 SHA43 SHA44 --checkpoint-metadata runs/seed42/run.json runs/seed43/run.json runs/seed44/run.json --output predictions/dev-e4
```

使用原选定权重时，提供 [FINAL_MODEL.json](../../reproducibility/FINAL_MODEL.json) 中的三个摘要，不需新训练元数据。新权重会核对实验分支、种子、选定轮数、哈希与共同训练输入标识。

预测为每个种子启动独立工作进程，验证进程标识及完整商品对 ID，再平均概率。`_predict-one` 是内部命令。复现时用 `--epoch` 指定六个开发检查点之一；保存的最终结果使用共同第 4 轮。

`select-epoch` 接收包含 `references`、`cplus` 两部分的 JSON。前者提供四个对照的预测；后者提供种子 42、43、44 各六轮预测，值为分数行或预测文档。命令先最大化开发集成 AP，其次比较 P@100，最后选择较早轮次：

```bash
python scripts/matching.py select-epoch --predictions predictions/dev-all-epochs.json --labels inputs/dev-labels.json --output results/selected-epoch.json
```

不得使用评价标签选择轮数、种子或模型设置。

## 连接标签前核对预测文件

预测前，从无标签评价输入保存有序的商品对 ID，格式为 JSON 字符串列表。收到 C+ 集成和 L42 预测文件后，先与这份列表核对：

```bash
python scripts/matching.py validate-predictions --pair-ids inputs/evaluation-pair-ids.json --candidate predictions/eval/scores.json --reference predictions/l42-eval.json --output results/prediction-check.json
```

命令拒绝缺失、多余或重复的 ID、非有限或越界分数、标签以及不符合约定的文件字段。预测行顺序可以不同，程序按 ID 核对对应关系。回执保存三份输入文件的哈希与商品对数量，不读取目标标签文件、不计算指标，也不加载模型。这项检查只验证文件一致性，不核验权重来源或证明样本独立。随后评价应使用相同的预测文件，并将哈希与回执核对。

## 校准与评价

先预测校准商品对，仅用校准标签选择阈值；再预测评价商品对，复用已保存的阈值：

```bash
python scripts/matching.py calibrate --scores predictions/cal/scores.json --labels inputs/cal-labels.json --output results/cal-threshold.json
python scripts/matching.py evaluate --scores predictions/eval/scores.json --labels inputs/eval-labels.json --threshold results/cal-threshold.json --output results/evaluation.json
```

校准依次最大化 F1、精确率，再选择更高阈值。评价报告 AP、P@100 和整数混淆矩阵。AP 按并列分数组处理；Top100 的并列分数按 `pair_id` 排序。P@100 需要至少 100 对。这些命令计算指标，不证明Walmart–Amazon 独立新样本表现。

## 表格对照与 S33

`reference-fit` 和 `reference-predict` 使用已准备的特征行。B22 示例也展示了 L30/L42 的文件流向：

```bash
python scripts/matching.py reference-fit --method B22 --rows prepared/fit/features.json --output runs/b22
python scripts/matching.py reference-predict --method B22 --rows prepared/dev/features.json --role dev --checkpoint runs/b22/reference.pkl --sha256 REFERENCE_SHA --output predictions/b22-dev.json
```

`REFERENCE_SHA` 是可信本地对照文件的摘要。原对照产物在固定哈希下受支持；B22 加载器只映射一个旧类名，不导入旧程序包。新拟合会生成不同产物。不能加载不可信的 pickle 文件。

S33 需先用对应角色、准备行和固定本地 MiniLM 快照运行 `semantic-features`，再通过 `--semantic-features` 将输出交给两个对照命令。CLI 核对角色/商品对标识及固定语义输入配置；大型依赖只在这些命令请求时加载。

该入口的现有权重回放已完成 20 条命令、23 组独立核对，差值为 0。[复现指南](REPRODUCIBILITY.md)记录观察到的环境，并将回放、从头重训和原始数据重建分开说明。
