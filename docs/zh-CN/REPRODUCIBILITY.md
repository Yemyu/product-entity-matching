# 复现说明

[🌐 English](../REPRODUCIBILITY.md) | **🇨🇳 简体中文**

## 复现状态

| 层次 | 证据 | 条件或限制 |
| --- | --- | --- |
| 公开检查与 Notebook | 合成输入的接口/特征/指标检查及保存聚合结果的算术 | Python 3.11+ 与标准库；无需商品数据或模型下载 |
| 现有权重回放 | 20 条命令、23 组独立核对，与保留产物的差值为 0 | 本地提供原输入、快照与可信权重；没有重新训练 |
| 原始来源角色重建 | 四份业务输入的字节与成员顺序和保留文件一致 | 原 ZIP 与仓库内固定成员配方；不运行模型 |
| 从头重建整个实验 | 提供显式训练与预测命令 | 从头重新训练整套实验尚未验证 |

回放覆盖 token/特征/IDF 准备、MiniLM 特征、四个对照的预测和最终三个种子的 C+ 预测，核对了公开代码与已有产物的一致性；没有重新训练模型或评价 Walmart–Amazon 独立新样本。固定池的历史接触记录仍不完整。

## 轻量检查

按 [README 快速开始](../../README.zh-CN.md#快速开始)创建 `.venv` 环境，在仓库根目录运行：

```bash
.venv/bin/python scripts/public_smoke.py
.venv/bin/python scripts/run_notebook.py
.venv/bin/python scripts/verify_public_release.py
.venv/bin/python scripts/run_tests.py
```

Windows 将 `.venv/bin/python` 替换为 `.\.venv\Scripts\python.exe`。这些检查无需额外程序包、GPU、数据集或模型快照。Notebook 的报告结果来自 [fixed-results.json](../../results/fixed-results.json)；Acme 记录和演示概率为合成示例。

`run_notebook.py` 执行[英文](../../notebooks/product-matching-walkthrough.ipynb)与[中文](../../notebooks/product-matching-walkthrough.zh-CN.ipynb)的普通 Python 单元并比对保存输出。也可在 Python 3.11+ Jupyter 内核中运行 Notebook。

`verify_public_release.py` 核对文件清单覆盖、哈希、报告算术与本地链接。修改公开文件后，先用 `.venv/bin/python scripts/build_public_manifest.py` 重新生成清单，再运行校验器；Windows 同样使用明确的项目解释器。清单排除自身、本地环境、缓存与 Git 内部文件。

## 本地文件与工作目录

轻量检查使用 `.venv`，模型依赖使用 `.venv-gpu`。商品输入与运行输出放在仓库旁：

```text
product-entity-matching/         公开仓库；.venv/ 与 .venv-gpu/
product-matching-work/
  inputs/
    walmart_amazon_exp_data.zip
    roberta/                    ROBERTA_SNAPSHOT.json 列出的文件
    snapshot.json               生成的本地快照清单
    minilm/<revision>/          可选；S33 必需
  data/                         由 reconstruct-roles 创建
    roles/                      fit/dev/calibration/evaluation-business.json
    private/                    独立的 *-labels.json
  prepared/                     特征、token 与训练 IDF
  runs/                         训练检查点与元数据
  predictions/                  对照、种子与集成预测
  results/                      轮数选择、校准与评价报告
```

在仓库根目录运行以下目录准备命令。保留 `data/` 为不存在的路径，使重建命令可以创建新输出目录。

**macOS / Linux：目录准备与标准库命令**

```bash
mkdir -p ../product-matching-work/inputs/roberta ../product-matching-work/prepared ../product-matching-work/runs ../product-matching-work/predictions ../product-matching-work/results
```

**Windows · PowerShell**

```powershell
New-Item -ItemType Directory -Force -Path "../product-matching-work/inputs/roberta", "../product-matching-work/prepared", "../product-matching-work/runs", "../product-matching-work/predictions", "../product-matching-work/results" | Out-Null
```

模型命令仅在下述 Windows CUDA 环境中核对过。macOS/Linux 的目录命令不代表这些平台上的模型运行已经验证。每个模型命令需要新的输出文件或目录；重复运行时使用另一个工作目录，避免覆盖保留的结果。

| 文件 | 来源或条件 |
| --- | --- |
| 原始 Walmart–Amazon ZIP | 自行获取 [DeepMatcher 源压缩包](https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Walmart-Amazon/walmart_amazon_exp_data.zip)，放到 `inputs/` |
| RoBERTa 骨干与分词器 | 自行获取官方 [FacebookAI/roberta-base 固定版本](https://huggingface.co/FacebookAI/roberta-base/tree/e2da8e2f811d1448a5b465c236feacd80ffbac7b)，将 [ROBERTA_SNAPSHOT.json](../../reproducibility/ROBERTA_SNAPSHOT.json)标识的准确文件放到 `inputs/roberta/` |
| 原 C+ 训练权重 | **没有公开下载入口。** 精确回放需已经持有 [FINAL_MODEL.json](../../reproducibility/FINAL_MODEL.json)标识的三个可信检查点 |
| 新训练权重 | 由显式本地训练生成，预测还需要每个种子的 `run.json`；完整重训尚未验证 |
| S33 的 MiniLM | 自行获取官方 [paraphrase-multilingual-MiniLM-L12-v2 固定版本](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/tree/e8f8c211226b894fcb81acc59f3b34ba3efd5f42)，本地文件夹的末级名称必须为 `e8f8c211226b894fcb81acc59f3b34ba3efd5f42` |

仓库与 CLI 不下载这些文件。获取与使用时遵守第三方条款。阅读公开聚合结果无需这些本地产物。

## 观察到的 GPU 环境

现有产物回放复用了以下环境：

| 组件 | 观察值 |
| --- | --- |
| 平台 | Windows 11 |
| Python | 3.11.16 |
| GPU | RTX 3060 笔记本 |
| PyTorch / CUDA | 2.8.0+cu126 / 12.6 |
| transformers | 5.17.0 |
| sentence-transformers | 6.0.1 |
| scikit-learn | 1.9.1 |
| numpy | 2.4.6 |
| safetensors | 0.8.0 |

[EXPERIMENT_LOCK.json](../../reproducibility/EXPERIMENT_LOCK.json)记录这些观察。[requirements-train.lock.txt](../../requirements-train.lock.txt)固定直接依赖及 CUDA 12.6 wheel 源，没有完整锁定间接依赖、wheel 哈希或驱动。其他平台与 GPU 组合尚未验证。

## Windows 模型命令的项目环境

以下是兼容 Windows CUDA 机器的环境设置说明。全新环境安装尚未验证，回放复用了已有环境。在仓库根目录运行：

```powershell
py -3.11 --version
py -3.11 -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe --version
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements-train.lock.txt
.\.venv-gpu\Scripts\python.exe -m pip check
```

若 `.venv-gpu` 已存在，先核对解释器并复用该环境，无需重复创建。CUDA wheel 源依据 [PyTorch 官方历史版本说明](https://pytorch.org/get-started/previous-versions/)中的 PyTorch 2.8.0/CUDA 12.6 组合。兼容驱动与受支持 GPU 是外部条件。源码与仓库旁工作目录均放在两个虚拟环境之外。

[命令指南](CORE_USAGE.md)中的模型命令明确使用 `.\.venv-gpu\Scripts\python.exe`，无需先激活环境。

## 配置与选择规则

| 公开说明 | 标识内容 |
| --- | --- |
| [FINAL_MODEL.json](../../reproducibility/FINAL_MODEL.json) | 最终设置、种子、选定轮数与原检查点哈希 |
| [ROBERTA_SNAPSHOT.json](../../reproducibility/ROBERTA_SNAPSHOT.json) | 骨干与分词器文件的准确大小和哈希 |
| [ROLE_RECONSTRUCTION.json](../../reproducibility/ROLE_RECONSTRUCTION.json) | 有序源位置、原生字段置空标记与角色输入哈希 |
| [FEATURES.json](../../reproducibility/FEATURES.json) | 固定 42 项特征顺序与分组 |
| [EXPERIMENT_LOCK.json](../../reproducibility/EXPERIMENT_LOCK.json) | 数据角色、轮数/阈值选择分工与观察环境 |

标题 IDF 只在训练记录上拟合。开发预测为三个种子选择共同轮数；校准标签在评价评分之前确定阈值。评价标签与模型输入分离，不用于调参。这些接口规则不能补全固定池的历史接触记录。

`reconstruct-roles` 从固定 ZIP 恢复四份业务文件，保留原有 23 处型号与 14 处类别置空标记；文件哈希和成员顺序均与保留输入一致。新的随机划分会得到另一个实验，新训练也可能产生不同权重与指标。详见[数据卡](DATA_CARD.md)和[局限](LIMITATIONS.md)。
