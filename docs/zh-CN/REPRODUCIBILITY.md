# 复现说明

[English](../REPRODUCIBILITY.md) | **简体中文**

## 复现状态

| 层次 | 证据 | 条件或限制 |
| --- | --- | --- |
| 公开检查与 Notebook | 合成输入的接口/特征/指标检查及保存聚合结果的算术 | Python 3.11+ 与标准库；无需商品数据或模型下载 |
| 现有权重回放 | 20 条命令、23 组独立核对，与固定产物的差值为 0 | 本地提供原角色输入、快照及可信权重；没有重新训练 |
| 原始来源角色重建 | 四份业务输入的字节及成员顺序与冻结文件一致 | 原 ZIP 与仓库内固定成员配方；不运行模型 |
| 从头重建整个实验 | 提供显式训练与预测命令 | 新的完整训练尚未验收 |

回放覆盖真实 token/特征/IDF 准备、MiniLM 特征、四个对照的预测，以及最终三个种子的 C+ 预测。输出一致验证了代码入口能够回放已有产物，不证明Walmart–Amazon 独立新样本表现，也不等于从原始来源复现整个实验。

## 轻量检查

先按 [README 快速开始](../../README.zh-CN.md#快速开始)建立 Python 3.11+ 项目环境，再从仓库根目录运行。Windows 使用项目的 `.\.venv\Scripts\python.exe` 替换下方的 `python`：

```bash
python scripts/public_smoke.py
python scripts/run_notebook.py
python scripts/verify_public_release.py
python scripts/run_tests.py
```

这些检查无需安装额外程序包、GPU、数据集或模型快照。Notebook 的真实数值来自 [fixed-results.json](../../results/fixed-results.json)；编写的 Acme 记录和演示概率是合成示例。

`run_notebook.py` 执行[英文](../../notebooks/product-matching-walkthrough.ipynb)与[中文](../../notebooks/product-matching-walkthrough.zh-CN.ipynb)的普通 Python 单元，并核对保存输出。也可以在现有 Python 3 Jupyter 内核中运行 Notebook。标准库执行不能替代单独的 Jupyter 视觉检查。

`verify_public_release.py` 核对文件清单覆盖、哈希、报告算术、本地链接和公开文件边界，不执行 GPU 命令或检查私有数据。有意修改源码或材料后，先用 `python scripts/build_public_manifest.py` 重新生成清单，再运行校验器。清单排除自身、本地环境、缓存及 Git 内部文件。

## 观察到的 GPU 环境

已完成的现有产物回放复用了以下环境：

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

这些观察记录在 [EXPERIMENT_LOCK.json](../../reproducibility/EXPERIMENT_LOCK.json) 中。[requirements-train.lock.txt](../../requirements-train.lock.txt)固定直接依赖及 CUDA 12.6 wheel 源，没有完整锁定间接依赖、wheel 哈希或设备驱动。其他平台与 GPU 组合尚未验证。

## 模型命令的项目环境

若复现 Windows GPU 命令，请使用项目环境并核对解释器。以下命令仅说明安装路径；公开材料准备期间没有从新环境执行过这套安装：

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-train.lock.txt
.venv/Scripts/python -m pip check
```

回放复用了已有环境。CUDA wheel 源依据 [PyTorch 官方历史版本说明](https://pytorch.org/get-started/previous-versions/)中的 PyTorch 2.8.0/CUDA 12.6 组合。兼容驱动与设备支持是外部条件，本仓库不安装或修改它们。源码、数据和实验输出应放在 `.venv` 之外。

## 本地文件与产物标识

公开仓库不分发商品行、标签、模型快照或训练权重；包含用于重建固定角色的源记录位置元信息。输入需按第三方条款在本地获取。[命令指南](CORE_USAGE.md)提供准备、显式训练、预测、对照、轮数选择和指标命令。

| 公开说明 | 标识内容 |
| --- | --- |
| [FINAL_MODEL.json](../../reproducibility/FINAL_MODEL.json) | 最终结构/设置、种子、选定轮数与原检查点哈希 |
| [ROBERTA_SNAPSHOT.json](../../reproducibility/ROBERTA_SNAPSHOT.json) | 骨干/分词器文件的准确大小与哈希 |
| [ROLE_RECONSTRUCTION.json](../../reproducibility/ROLE_RECONSTRUCTION.json) | 固定源位置、原生字段置空标记与角色输入哈希 |
| [FEATURES.json](../../reproducibility/FEATURES.json) | 固定 42 项特征的顺序和分组 |
| [EXPERIMENT_LOCK.json](../../reproducibility/EXPERIMENT_LOCK.json) | 数据角色、轮数/阈值选择分工与观察环境 |

准备和模型命令将本地快照核对到固定文件标识。原选定权重按种子 42、43、44 顺序使用三个检查点哈希。新训练权重需提供各自的 `run.json`，核对实验分支/种子/轮数/哈希及共同训练输入标识。命令要求新输出路径，失败后不静默重试。

## 选择规则与重建范围

标题 IDF 只在训练记录上拟合。三个种子的开发预测选择共同轮数；校准标签在评价评分之前确定阈值。评价标签与模型输入分离，不用于调参。当前接口规则不能补全固定池的历史接触记录。

`reconstruct-roles` 已从固定原 ZIP 恢复四份角色业务输入，文件哈希及有序成员全部一致。配方保留原有23处型号与14处类别置空行为；源 ZIP 由读者自行获取。这不能恢复缺失的历史接触记录。从头训练仍未验收，也可能产生不同权重；新的随机划分会得到另一个实验。详见[数据卡](DATA_CARD.md)与[局限](LIMITATIONS.md)。
