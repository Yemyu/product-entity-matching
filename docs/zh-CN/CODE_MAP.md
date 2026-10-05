# 代码地图

[English](../CODE_MAP.md) | **简体中文**

## 程序包与入口

`scripts/matching.py` 调用 `product_matching.cli`。`src/product_matching/` 的 16 个模块实现最终方法与实验对照。导入程序包或请求 CLI 帮助无需神经网络运行库，也不下载模型；大型依赖只在需要它们的命令内部导入。

| 模块 | 职责 |
| --- | --- |
| [`__init__.py`](../../src/product_matching/__init__.py) | 程序包入口与导出 |
| [`data.py`](../../src/product_matching/data.py) | 明确的业务/原生字段适配与内容标识 |
| [`normalization.py`](../../src/product_matching/normalization.py) | 文本、价格、币种处理及缺失/歧义状态 |
| [`source.py`](../../src/product_matching/source.py) | 从固定本地源 ZIP 重建原角色输入 |
| [`roles.py`](../../src/product_matching/roles.py) | 连接固定成员、视图与附表，不生成新划分 |
| [`lexical.py`](../../src/product_matching/lexical.py) | 仅从训练数据拟合的标题 IDF 及其保存状态 |
| [`features.py`](../../src/product_matching/features.py) | 构造并校验固定 42 项比较 |
| [`serialization.py`](../../src/product_matching/serialization.py) | 字段 token 预算、AB/BA 输入与对照行格式 |
| [`model.py`](../../src/product_matching/model.py) | 固定 RoBERTa 融合头、训练与预测 |
| [`runtime.py`](../../src/product_matching/runtime.py) | 确定性与显式模型执行设置 |
| [`snapshot.py`](../../src/product_matching/snapshot.py) | 固定本地 RoBERTa 文件标识 |
| [`references.py`](../../src/product_matching/references.py) | B22/L30/S33/L42 的拟合、预测与产物加载 |
| [`semantic.py`](../../src/product_matching/semantic.py) | S33 使用的固定 MiniLM 特征 |
| [`metrics.py`](../../src/product_matching/metrics.py) | ID 对齐、AP 并列分数、共同轮数、阈值及混淆矩阵 |
| [`cli.py`](../../src/product_matching/cli.py) | 显式文件命令、元数据核对与无标签预测检查 |
| [`errors.py`](../../src/product_matching/errors.py) | 输入和执行约定异常 |

预测为每个种子启动新进程。`_predict-one` 是内部工作命令；读者使用 `predict` 或 `predict-ablation`。对照加载器将一个固定旧 B22 pickle 类名映射到当前类，不导入旧程序包。加载 pickle 仍需使用可信本地产物及已记录的哈希。

## 公开检查与材料

| 路径 | 用途 |
| --- | --- |
| `scripts/public_smoke.py` | 轻量导入、输入、特征与指标检查 |
| `scripts/run_notebook.py` | 执行双语 Notebook 的普通 Python 单元并核对保存输出 |
| `scripts/verify_public_release.py` | 核对文件清单、哈希、报告算术及本地链接 |
| `scripts/run_tests.py` 与 `tests/` | 使用标准库的合成输入约定测试 |
| `scripts/build_public_manifest.py` | 有意修改文件后重新生成公开清单 |
| `results/fixed-results.json` | 保存的六方法聚合事实 |
| `reproducibility/` | 固定模型、实验、快照与特征顺序说明 |
| `docs/` 与 `docs/zh-CN/` | 七份相互对应的英文和中文技术指南 |
| `notebooks/` | 英文与中文的可执行说明 |
| `showcase/` | 两种语言的静态项目页 |

需要商品记录、标签或训练产物的命令读取显式提供的本地文件。[命令指南](CORE_USAGE.md)将这些输入与运行输出统一放在仓库旁的 `../product-matching-work/`。

[命令指南](CORE_USAGE.md)说明文件接口，[模型卡](MODEL_CARD.md)说明结构，[复现指南](REPRODUCIBILITY.md)说明验证范围。
