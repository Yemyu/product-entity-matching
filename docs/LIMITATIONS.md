# Interpretation and limitations

**English** | [简体中文](zh-CN/LIMITATIONS.md)

## Evaluation scope

The reported metrics describe one fixed pool of 555 Walmart–Amazon pairs. This is a retrospective comparison: the historical exposure record is incomplete. The public code's role rules do not prove that the pool was untouched throughout development. An independent fresh Walmart–Amazon sample or entity-disjoint evaluation has not been established.

AP 0.9660 is a ranking metric, not 96.6% accuracy. F1 0.9016 is measured at the calibration cutoff. P@100 1.00 means that the first 100 ranked pairs were matches on this pool; it does not describe all pairs or future listings.

## Reference and ablation comparison

C+ has higher AP and F1 than L42 on this pool. C0 is independently trained with the 42-feature vector zeroed during training and inference; its AP is 0.9656, close to C+. The aggregate numbers alone do not establish a substantial ranking contribution from the 42 features or statistical superiority. C0 used four training epochs with the original six-epoch scheduler and is compared at epoch 4.

Three training seeds define the final ensemble. They are not three independent test populations. No single-seed uncertainty estimate or confidence interval is published. The saved aggregate data cannot support new PR curves, threshold sweeps or error-case frequencies that were not retained.

## Reproduction scope

Public checks and the Notebook use synthetic fixtures and saved aggregate results. They verify interface behavior and arithmetic, not a new run on product data.

Existing-weight replay ran 20 commands and checked 23 result groups, with zero differences from retained outputs. This verifies that the public code entry can prepare inputs and replay original artifacts. It does not reproduce fresh training. A separate raw-source reconstruction now matches all four fixed business files byte for byte; this restores the original membership rather than establishing an independent test. Exact model replay still requires local snapshots and trusted original weights, which have no public download entry. Fresh end-to-end retraining has not been validated.

## Environment and intended use

The observed model environment was Windows 11, Python 3.11.16, an RTX 3060 laptop and PyTorch 2.8.0+cu126/CUDA 12.6. Other platforms and GPU/dependency combinations have not been validated. The direct dependency pins do not lock every transitive wheel or driver.

Model-number suffix conflicts, spelling variants and missing fields can still produce incorrect scores. No deployment latency, production throughput, calibration drift or business benefit has been measured. A matching score is evidence to review; it is insufficient by itself to justify an automatic catalog merge. This project makes no claim of a benchmark-leading method or production readiness.

See the [model card](MODEL_CARD.md) for measured comparisons, [data card](DATA_CARD.md) for input rules, and [reproduction guide](REPRODUCIBILITY.md) for what has and has not been run.
