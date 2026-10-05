# Project brief

**English** | [简体中文](zh-CN/PROJECT_CARD.md)

## Task and delivered work

The input is a pair of retailer listings; the output is a match score between 0 and 1. Comparing that score with a calibration cutoff gives a match or non-match decision. Differences in title spelling, model numbers, missing fields and prices make exact text equality insufficient.

The repository contains field adapters and conservative normalization, fit-only title IDF, 42 comparison features, four tabular references, and the final RoBERTa feature-fusion model, named C+ in this project. The model handles both listing orders and averages probabilities from seeds 42, 43 and 44. Development data selects one common epoch; calibration data selects the cutoff used for evaluation.

## Fixed-pool results and the C0 comparison

On 555 fixed Walmart–Amazon pairs, the C+ ensemble reached **AP 0.9660, F1 0.9016 and P@100 1.00**. Against L42, the best tabular reference on this pool, AP increased by **1.63 percentage points** and F1 by **4.96 percentage points**.

C0 is independently trained with all 42 feature values zeroed during training and inference. Its AP is 0.9656, close to C+. The aggregate comparison supports a higher F1 at the separately calibrated cutoffs, but does not establish a substantial ranking gain from the feature block. The six-method table and confusion matrix are in the [model card](MODEL_CARD.md).

The pair pool is retrospective and its history of use is not fully documented. Independent new-sample performance has not been tested. These numbers describe the fixed pool; they do not establish statistical superiority, deployment readiness or business benefit.

## Public checks and existing-weight replay

The [Notebook](../notebooks/product-matching-walkthrough.ipynb) uses saved aggregate results and synthetic product records. It runs without downloading models. The public checks cover input rules, feature construction and metric arithmetic.

Existing-weight replay ran 20 commands and independently checked 23 result groups against frozen outputs, with zero differences. The replay covered token/feature preparation, MiniLM features, reference predictions and the final three-seed prediction. It did not retrain the model. A separate source reconstruction reproduced all four frozen business inputs byte for byte. Fresh training has not been reproduced.

## Technical documentation

- [Project pages](../showcase/en/index.html) explain the task, method, results and reproduction.
- [Data card](DATA_CARD.md) explains roles, fields and all 42 features.
- [Model card](MODEL_CARD.md) records the architecture, configuration and comparisons.
- [CLI guide](CORE_USAGE.md), [reproduction guide](REPRODUCIBILITY.md) and [code map](CODE_MAP.md) connect those facts to the implementation.
- [Limitations](LIMITATIONS.md) defines what the evidence can support.

The matching package uses [MIT](../LICENSE). The derived project pages use [CC BY-SA 4.0](../showcase/LICENSE), with their source and modifications recorded in [NOTICE](../showcase/NOTICE.md). Product data, model snapshots and trained weights are not redistributed.
