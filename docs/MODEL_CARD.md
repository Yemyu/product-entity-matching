# Model card

**🌐 English** | [🇨🇳 简体中文](zh-CN/MODEL_CARD.md)

## Purpose and inputs

C+ names the final RoBERTa model with 42 comparison features in this project. It scores whether two product listings refer to the same product. It is an offline research model. Inputs include normalized business fields, native model/category fields and the saved fit-only title IDF state. The text encoder receives brand, native model number, category and title; description and price evidence enter through the comparison features.

## Text and feature fusion

The backbone is `FacebookAI/roberta-base`, revision `e2da8e2f811d1448a5b465c236feacd80ffbac7b`. The first-token representation has 768 dimensions. A 64-unit GELU projection is concatenated with the 42 comparisons, followed by another 64-unit GELU layer, dropout 0.1 and a scalar logit.

Each record has fixed token budgets: brand 16, model 32, category 16 and title 62, including field markers. A long title retains its first 40 and last 22 tokens. Each AB or BA pair is at most 256 tokens including RoBERTa special tokens. Missing text fields are serialized explicitly.

For each seed, `p_seed = sigmoid((logit_AB + logit_BA) / 2)`. The final score is the equal mean of probabilities from seeds 42, 43 and 44. Averaging directions happens before sigmoid; averaging seeds happens after it. Development selected the common epoch 4; evaluation labels were excluded from seed and epoch selection.

A and B denote the two listings; AB and BA are the two input orders. A token is a text unit processed by the encoder, such as a word or part of a word. A logit is the raw score before sigmoid maps it between 0 and 1. Training seeds control random initialization and data order; this model averages probabilities from three fixed seeds.

## Fixed training configuration

| Setting | Value |
| --- | --- |
| C+ training and scheduler | 6 epochs |
| Selected checkpoint | Epoch 4 for all three seeds |
| Micro-batch / effective batch | 2 / 16 pairs |
| Encoder / head learning rate | 2e-5 / 2e-4 |
| Warmup | 10% |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| Encoder training | BF16 autocast; FP32 parameters |
| Head and inference | FP32 |

[FINAL_MODEL.json](../reproducibility/FINAL_MODEL.json) records the settings and three selected checkpoint hashes. [ROBERTA_SNAPSHOT.json](../reproducibility/ROBERTA_SNAPSHOT.json) records the local backbone file hashes. The weights and snapshots themselves are not distributed. The original trained weights have no public download entry. For the backbone source and local file layout, see the [reproduction guide](REPRODUCIBILITY.md).

## References and ablation

| Method | Description |
| --- | --- |
| B22 | Logistic regression on 22 comparisons, with 800 fixed updates |
| L30 | Histogram gradient-boosted trees on 30 comparisons |
| S33 | The L30 representation plus three fixed MiniLM semantic/missingness features |
| L42 | Histogram gradient-boosted trees on all 42 comparisons |
| C+ | RoBERTa with the 42-feature fusion head, averaged over three seeds |
| C0 | Independently trained neural control with the 42-feature vector zeroed during training and inference |

C0 was trained for four epochs while retaining the six-epoch scheduler and is compared at epoch 4. It is an experimental control.

## Saved results on the fixed pool

All six methods were evaluated on the same retrospective pool of 555 pairs, including 190 matches. The pool's historical exposure record is incomplete, and independent new Walmart–Amazon sample performance is untested.

| Method | AP | F1 | P@100 |
| --- | ---: | ---: | ---: |
| B22 | 0.795590 | 0.743719 | 0.81 |
| L30 | 0.844899 | 0.762115 | 0.90 |
| S33 | 0.846702 | 0.767123 | 0.89 |
| L42 | 0.949700 | 0.852041 | 1.00 |
| C+ | 0.965998 | 0.901639 | 1.00 |
| C0 | 0.965616 | 0.875648 | 1.00 |

C+ exceeds L42 by AP **1.63 percentage points** and F1 **4.96 percentage points**. Against C0, its AP is only about **0.04 percentage points** higher, while F1 is about **2.60 percentage points** higher at the separately calibrated cutoffs. Those aggregate results do not establish a substantial feature-driven AP gain or statistical superiority.

The C+ cutoff is **0.3947772259513537**, chosen on calibration data. Its evaluation confusion matrix uses actual labels as rows and predictions as columns:

| Actual / predicted | Nonmatch | Match |
| --- | ---: | ---: |
| Nonmatch | TN 354 | FP 11 |
| Match | FN 25 | TP 165 |

Precision is **0.937500** and recall **0.868421**. AP measures positive-pair ranking; F1 depends on the cutoff; P@100 describes only the first 100 ranked pairs. Full precision, recall, thresholds and counts for each method are in [fixed-results.json](../results/fixed-results.json).

## Verification and use limits

Existing-weight replay checked 23 result groups from 20 commands against retained outputs, with zero differences. The [reproduction guide](REPRODUCIBILITY.md) separates this evidence from public synthetic checks and unvalidated fresh training. Snapshot/checkpoint hashes identify artifacts; they do not establish unseen-data evaluation.

This 555-pair comparison does not measure latency, production throughput or independent Walmart–Amazon generalization. A high score alone does not establish that a catalog merge is correct. See the [data card](DATA_CARD.md) and [limitations](LIMITATIONS.md). The [MIT code license](../LICENSE) does not replace third-party model or data terms.
