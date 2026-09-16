# Ministral 8B QLoRA oversampling: 10-fold results

## Completion status

The full 10-fold cross-validation experiment completed successfully on 15 September 2026. Every fold passed the frozen-data audit, trained for two epochs, completed validation, and produced held-out test predictions. No failed-fold artifact was found.

## Experiment identity

- Experiment ID: `mistral_8b_qlora_oversampling`
- Model: `mistralai/Ministral-3-8B-Instruct-2512-BF16`
- Frozen model revision: `f6fae9795746f63c9be8344932f01275f3c63734`
- Method: 4-bit QLoRA
- Evaluation: 10-fold cross-validation with a frozen inner validation split in every fold
- Training balance: random oversampling with replacement on inner training data only
- Labels: `trace` and `no_trace`
- Hardware: NVIDIA GeForce RTX 4090, 24 GB

## Historical protocol provenance

The experiment follows the corrected Qwen3.5-9B EXP-06 oversampling implementation. It does not use the undersampling experiment.

- Corrected split/config and oversampling implementation: commit `c3ae63b`
- Training implementation: commit `198fc2f`
- Forced-choice evaluation implementation: commit `8ee202f`
- Qwen config generator: `scripts/create_qwen35_inner_val_fold_config.py`
- Qwen oversampling generator: `scripts/create_oversampled_inner_train_fold.py`

The corrected committed EXP-06 configuration specifies a learning rate of `0.00005`; this is the value used here. It supersedes the earlier summary-table value of `0.0001`.

## Frozen data protocol

For every fold, the existing Qwen EXP-06 files were consumed directly; no fold was recreated. The source `inner_train.jsonl` was oversampled by drawing minority `trace` records with replacement until both classes contained 915 records, followed by the historical seeded shuffle. The fold seed is `42 + fold_number`.

- Oversampled training set per fold: 1,830 records (915 `trace`, 915 `no_trace`)
- Validation and test data: unchanged and not resampled
- Exact historical oversampling reproduction: passed for all 10 folds
- Train/validation/test ID non-overlap: passed for all 10 folds
- Mistral test file versus frozen Qwen test file: exact match for all 10 folds
- Fold 01 source inner train: 1,528 records (613 `trace`, 915 `no_trace`)
- Fold 01 oversampled train: 1,830 records, 1,386 unique IDs, 444 duplicated rows
- Fold 01 validation: 170 records (68 `trace`, 102 `no_trace`)
- Fold 01 test: 189 records (76 `trace`, 113 `no_trace`)

Duplicate rows in the oversampled training data are expected because sampling with replacement is the treatment being evaluated.

## Prompt, loss, and training configuration

The prompt and label format is the exact plain-text Pattern-1 protocol used by Qwen EXP-06:

```text
{instruction}

Higher-level requirement:
{higher_level_requirement}

Lower-level requirement:
{lower_level_requirement}

Answer: {trace|no_trace}
```

No chat template is applied. Loss covers the full prompt and label, matching the historical implementation.

| Setting | Value |
| --- | ---: |
| Learning rate | `5e-5` |
| Microbatch size | `1` |
| Gradient accumulation | `8` |
| Effective batch size | `8` |
| Epochs | `2` |
| Warmup steps | `20` |
| Maximum gradient norm | `1.0` |
| Weight decay | `0.01` |
| Maximum sequence length | `1024` |
| Precision | BF16 compute |
| Optimizer | AdamW |
| Scheduler | Linear with warmup |
| DataLoader shuffle | `false` |
| Validation | Every 50 optimizer steps on the first 100 validation records |
| Checkpointing | Every 100 optimizer steps; retain at most 3 |
| Logging | Every 10 optimizer steps |
| Seed | `42` |

The historical EXP-06 partial-accumulation behavior was preserved. Each epoch has 1,830 microbatches: 228 complete optimizer steps and six discarded trailing microbatches. Each fold therefore completed 456 optimizer steps across two epochs, while the historical scheduler formula used 457 total steps.

## QLoRA configuration and preflight

Quantization used 4-bit NF4, BF16 compute, and double quantization. LoRA used rank 16, alpha 32, dropout 0.1, and no bias.

The target regex was selected from the Ministral architecture rather than copied blindly from Qwen:

```text
.*language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)
```

This attached adapters only to the text decoder's attention projections.

| Preflight item | Result |
| --- | ---: |
| Status | Passed; no errors |
| Adapter modules | 136 |
| Trainable parameters | 14,483,456 |
| Total parameters reported | 5,010,835,456 |
| Trainable proportion | 0.289043% |
| Vision/projector adapters | 0 |
| Longest-sequence optimizer step | Completed |
| Raw preflight loss | 2.5179994 |
| GPU process peak allocated | 9,502.04 MiB |
| GPU process peak reserved | 10,334.00 MiB |
| GPU free after optimizer step | 13,254.06 MiB |

Environment: Python 3.12.3, PyTorch 2.11.0+cu130, Transformers 5.8.0, PEFT 0.19.1, bitsandbytes 0.49.2, Datasets 4.8.5, Accelerate 1.13.0, CUDA runtime 13.0.

## Per-fold held-out test results

Invalid generated labels are excluded from classification metrics, exactly as specified by the evaluation protocol.

| Fold | Accuracy | Trace P | Trace R | Trace F1 | No-trace P | No-trace R | No-trace F1 | Valid | Invalid |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 01 | 0.6685 | 0.5546 | 0.8919 | 0.6839 | 0.8769 | 0.5182 | 0.6514 | 184 | 5 |
| 02 | 0.7128 | 0.6833 | 0.5395 | 0.6029 | 0.7266 | 0.8304 | 0.7750 | 188 | 1 |
| 03 | 0.7021 | 0.5833 | 0.9211 | 0.7143 | 0.9118 | 0.5536 | 0.6889 | 188 | 1 |
| 04 | 0.7268 | 0.6250 | 0.8108 | 0.7059 | 0.8391 | 0.6697 | 0.7449 | 183 | 6 |
| 05 | 0.7021 | 0.5909 | 0.8553 | 0.6989 | 0.8590 | 0.5982 | 0.7053 | 188 | 1 |
| 06 | 0.6862 | 0.5669 | 0.9474 | 0.7094 | 0.9344 | 0.5089 | 0.6590 | 188 | 1 |
| 07 | 0.7880 | 0.7108 | 0.7973 | 0.7516 | 0.8515 | 0.7818 | 0.8152 | 184 | 5 |
| 08 | 0.7333 | 0.6154 | 0.9600 | 0.7500 | 0.9524 | 0.5714 | 0.7143 | 180 | 8 |
| 09 | 0.7043 | 0.5862 | 0.9067 | 0.7120 | 0.9000 | 0.5676 | 0.6961 | 186 | 2 |
| 10 | 0.7326 | 0.6087 | 0.9333 | 0.7368 | 0.9306 | 0.5982 | 0.7283 | 187 | 1 |

Best accuracy was Fold 07 at 0.7880; lowest accuracy was Fold 01 at 0.6685.

## Aggregate results

The following are unweighted means across the 10 folds. The reported standard deviation is the population standard deviation used by the experiment summary.

| Metric | Mean | Standard deviation |
| --- | ---: | ---: |
| Accuracy | 0.7157 | 0.0326 |
| Trace precision | 0.6125 | 0.0498 |
| Trace recall | 0.8563 | 0.1242 |
| Trace F1 | 0.7066 | 0.0425 |
| No-trace precision | 0.8782 | 0.0655 |
| No-trace recall | 0.6198 | 0.1086 |
| No-trace F1 | 0.7178 | 0.0505 |

Across all folds there were 1,887 held-out records, 1,856 valid predictions, and 31 invalid predictions. The invalid prediction rate was 1.6428%.

### Pooled valid-prediction metrics

| Metric | Value |
| --- | ---: |
| Accuracy | 0.7155 |
| Trace precision | 0.6049 |
| Trace recall | 0.8562 |
| Trace F1 | 0.7089 |
| No-trace precision | 0.8638 |
| No-trace recall | 0.6199 |
| No-trace F1 | 0.7218 |
| Macro precision | 0.7344 |
| Macro recall | 0.7381 |
| Macro F1 | 0.7154 |

The pooled confusion matrix uses rows as true labels and columns as predicted labels:

| True / predicted | `trace` | `no_trace` |
| --- | ---: | ---: |
| `trace` | 643 | 108 |
| `no_trace` | 420 | 685 |

## Comparison with original-imbalance training

The comparison below uses the corresponding 10-fold Ministral QLoRA original-imbalance experiment and reports changes in fold-mean metrics as oversampling minus original imbalance.

| Metric | Original imbalance | Oversampling | Change |
| --- | ---: | ---: | ---: |
| Accuracy | 0.7404 | 0.7157 | -0.0247 |
| Trace precision | 0.6859 | 0.6125 | -0.0734 |
| Trace recall | 0.7007 | 0.8563 | +0.1556 |
| Trace F1 | 0.6709 | 0.7066 | +0.0356 |
| No-trace precision | 0.8117 | 0.8782 | +0.0665 |
| No-trace recall | 0.7684 | 0.6198 | -0.1486 |
| No-trace F1 | 0.7766 | 0.7178 | -0.0588 |

Oversampling made the model substantially more willing to predict `trace`. It improved trace recall and trace F1, but reduced trace precision, overall accuracy, no-trace recall, and no-trace F1. The practical trade-off is therefore higher sensitivity to trace links at the cost of more false-positive trace predictions.

## Notes and limitations

- Metrics exclude invalid generated labels, so the valid/invalid counts must accompany reported classification scores.
- Fold 02 had substantially lower trace recall than the other oversampled folds, contributing to the relatively high cross-fold variance for trace recall.
- Generation emitted a non-fatal warning because the model configuration contains `max_length=262144` while evaluation explicitly sets `max_new_tokens=5`. Transformers used `max_new_tokens=5`, as intended.
- Large adapters and periodic checkpoints remain server-side under `mistral_traceability/outputs/mistral_8b_qlora_oversampling/` and are intentionally excluded from version control.

## Reproducibility artifacts

- Full configuration: `mistral_traceability/configs/config_mistral_8b_qlora_oversampling_10fold.yaml`
- Fold 01 preflight configuration: `mistral_traceability/configs/config_mistral_8b_qlora_oversampling_fold_01.yaml`
- Frozen-fold manifest: `mistral_traceability/data/manifests/mistral_8b_qlora_oversampling_folds.json`
- Configuration rationale: `mistral_traceability/docs/mistral_8b_qlora_oversampling_config.md`
- Preflight implementation: `mistral_traceability/scripts/preflight_mistral_8b_qlora_oversampling.py`
- 10-fold runner: `mistral_traceability/scripts/run_mistral_8b_qlora_oversampling_10fold_cv.py`
- Data audit: `mistral_traceability/results/mistral_8b_qlora_oversampling/data_audit_fold_01_10.json`
- Fold metrics, predictions, and training histories: `mistral_traceability/results/mistral_8b_qlora_oversampling/fold_01/` through `fold_10/`
- Machine-readable summary: `mistral_traceability/results/mistral_8b_qlora_oversampling/mistral_8b_qlora_oversampling_10fold_summary.json`
- CSV summary: `mistral_traceability/results/mistral_8b_qlora_oversampling/mistral_8b_qlora_oversampling_10fold_summary.csv`
- Preflight report: `mistral_traceability/results/mistral_8b_qlora_oversampling/preflight/fold_01/preflight_report.json`
- Logs: `mistral_traceability/logs/mistral_8b_qlora_oversampling/`
