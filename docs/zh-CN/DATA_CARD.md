# 数据卡

[English](../DATA_CARD.md) | **简体中文**

## 来源、数据用途与范围

实验使用一组自行固定的 Walmart–Amazon 商品对。角色名称是本项目分配的数据用途，不能证明原始来源中的官方测试文件从未被接触。

| 角色 | 商品对数 | 用途 |
| --- | ---: | --- |
| `fit` | 3,738 | 拟合标题 IDF、表格对照和神经模型 |
| `dev` | 199 | 为三个种子的集成选择共同轮数 |
| `calibration` | 209 | 选择 F1 阈值 |
| `evaluation` | 555 | 报告保存的固定池结果；其中匹配 190 对、不匹配 365 对 |

555 对评价池用于回顾性比较，历史接触记录不完整，尚未建立Walmart–Amazon 独立新样本或实体不重叠的测试划分。接口检查将评价标签排除在预测输入之外，但不能重建此前的数据使用历史。

## 字段与归一化

业务字段为 `title`、`brand`、`description`、`price`、`priceCurrency`；原生字段为 `modelno`、`category`。`data.business_record(raw)` 和 `data.native_record(raw)` 生成特征流程接收的明确视图。

文本处理包含 Unicode 归一化、大小写折叠与空白整理，缺失值保持缺失。纯非负十进制数可解析为价格；存在歧义的币种符号或分隔符保留为尚未解决的质量状态。只有价格均解析成功、币种代码均已知且相同，才启用价格相似度。缺失或不可比较的价格有独立指示，不会被当作观测到的零价格。

IDF（逆文档频率）给训练标题中较少见的词更高权重，以减少常见词对相似度的主导作用。标题 IDF 从 `fit` 中按内容去重的业务记录拟合，保存后用于其他角色。ID 与标签分别用于对齐及相应的拟合步骤，不进入特征向量。标题中的型号样式代码与原生型号分开比较，包括字母/数字不一致及跨字段匹配。

## 特征字典

以下顺序由 [FEATURES.json](../../reproducibility/FEATURES.json) 固定。42 个值均为有限数，范围为 [0, 1]。输入缺失时相似度为零，并在已定义的位置用存在指示区分。四组依次为业务比较（1–14）、仅从训练数据拟合的标题证据（15–22）、标题型号代码比较（23–30）、原生型号/类别比较（31–42）。标题证据组包含 IDF 加权比较和直接比较。

| 位置 | 标识 | 含义 |
| --- | --- | --- |
| 1 | `title_token_jaccard` | 标题 token 集合的 Jaccard 重合度 |
| 2 | `title_char_jaccard` | 标题去空格后字符集合的 Jaccard 重合度 |
| 3 | `description_token_jaccard` | 描述 token 集合的 Jaccard 重合度 |
| 4 | `description_both_present` | 两条描述都包含 token |
| 5 | `brand_equal` | 品牌均存在且相等 |
| 6 | `brand_both_present` | 品牌均存在 |
| 7 | `brand_missing_any` | 至少一条记录缺少品牌 |
| 8 | `numeric_token_jaccard` | 标题与描述中数字 token 的 Jaccard 重合度 |
| 9 | `numeric_tokens_both_present` | 两条记录都包含数字 token |
| 10 | `title_length_ratio` | 标题去重 token 数的较小值与较大值之比 |
| 11 | `price_both_parsed` | 两个价格均解析成功 |
| 12 | `price_currency_equal` | 归一化后的币种均存在且相同 |
| 13 | `price_comparable` | 价格已解析且币种为相同的已知代码 |
| 14 | `price_similarity` | 可比较时为 1 / (1 + log1p 价格的绝对差) |
| 15 | `title_word_idf_cosine` | 训练标题词 IDF 加权的余弦相似度 |
| 16 | `title_word_idf_containment` | 共同词的 IDF 加权能量除以较小标题词能量 |
| 17 | `title_char3_idf_cosine` | 标题字符三元组的训练 IDF 加权余弦相似度 |
| 18 | `title_compact_char3_idf_cosine` | 紧凑标题字符三元组的训练 IDF 加权余弦相似度 |
| 19 | `title_alphanumeric_jaccard` | 标题字母数字混合 token 的 Jaccard 重合度 |
| 20 | `title_number_jaccard` | 标题数字 token 的 Jaccard 重合度 |
| 21 | `title_numbers_both_present` | 两个标题都包含数字 token |
| 22 | `title_compact_exact` | 非空紧凑标题相等 |
| 23 | `code_left_missing_or_right_missing` | 至少一个标题没有型号样式的代码 token |
| 24 | `code_both_present` | 两个标题都包含型号样式的代码 token |
| 25 | `code_exact_overlap` | 至少一个提取的型号代码相同 |
| 26 | `code_set_jaccard` | 提取代码集合的 Jaccard 重合度 |
| 27 | `code_best_edit_similarity` | 代码配对中最高的归一化编辑相似度 |
| 28 | `code_same_letters_different_digits` | 存在字母相同、数字不同的代码对 |
| 29 | `code_same_digits_different_letters` | 存在数字相同、字母不同的代码对 |
| 30 | `code_one_edit_difference` | 存在长度均不少于四、编辑距离为一的不同代码对 |
| 31 | `native_model_missing_any` | 至少一个原生型号字段缺失 |
| 32 | `native_model_both_present` | 两个原生型号字段均存在 |
| 33 | `native_model_compact_exact` | 非空紧凑原生型号相等 |
| 34 | `native_model_edit_similarity` | 原生型号的归一化编辑相似度 |
| 35 | `native_model_char3_jaccard` | 原生型号字符三元组的 Jaccard 重合度 |
| 36 | `native_model_same_letters_different_digits` | 原生型号字母相同而数字不同 |
| 37 | `native_model_same_digits_different_letters` | 原生型号数字相同而字母不同 |
| 38 | `cross_native_title_any` | 任一原生型号出现在另一侧标题的代码集合中 |
| 39 | `cross_native_title_both` | 两个原生型号均出现在另一侧标题的代码集合中 |
| 40 | `native_category_both_present` | 两个原生类别字段均存在 |
| 41 | `native_category_token_jaccard` | 类别 token 集合的 Jaccard 重合度 |
| 42 | `native_category_exact` | 归一化类别均存在且相等 |

[特征代码](../../src/product_matching/features.py)和[标题证据代码](../../src/product_matching/lexical.py)规定准确的 token 切分与算术；字典解释这些运算，不另造中文 API 名称。

## 模型输入与标签

一条商品对记录包含 `pair_id`、`left`、`right`、`left_native`、`right_native`。只有 `fit` 记录包含整数 0 或 1 的 `label`。开发、校准与评价标签放在单独的 `{pair_id, label}` 列表中，由对应命令按 ID 与分数连接。

`project-role` 连接显式提供的固定视图、原生字段附表和有序成员 ID，不生成新划分。`prepare` 生成 42 项特征及 AB/BA token 输入；非训练角色必须使用已保存的训练 IDF。[命令指南](CORE_USAGE.md)列出文件约定和命令。

## 数据获取与重建

公开仓库未包含原始商品记录、标签、成员列表、模型快照或训练权重。Notebook 与网页中的 Acme 记录是合成教学示例，不是观测数据或模型预测。第三方数据需按其自身条款获取和使用。

读者自行获取原 ZIP 后，可用 `reconstruct-roles` 与仓库内源位置配方恢复固定角色输入；四份业务文件已与本地冻结输入逐字节核对。命令保留原字段置空标记，非训练标签单独存放，不生成新划分，也不恢复历史接触记录。改用随机划分会得到不同实验。详见[复现指南](REPRODUCIBILITY.md)和[局限](LIMITATIONS.md)。
