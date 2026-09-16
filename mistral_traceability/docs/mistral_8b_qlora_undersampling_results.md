# Ministral 8B QLoRA undersampling: 10-fold results

## Completion status

The full 10-fold cross-validation experiment completed successfully on 16 September 2026. Every fold passed the frozen-data audit, trained for two epochs, completed validation, and produced held-out test predictions. No failed-fold artifact was found.

## Experiment identity

- Experiment ID: `mistral_8b_qlora_undersampling`
- Model: `mistralai/Ministral-3-8B-Instruct-2512-BF16`
- Frozen model revision: `f6fae9795746f63c9be8344932f01275f3c63734`
- Method: 4-bit QLoRA
- Evaluation: 10-fold cross-validation with a frozen inner validation split in every fold
- Training balance: random undersampling without replacement on inner training data only
- Labels: `trace` and `no_trace`
- Hardware: NVIDIA GeForce RTX 4090, 24 GB

## Historical protocol provenance

The experiment follows the Qwen3.5-9B EXP-07 undersampling implementation. It does not use the oversampling or original-imbalance treatment.

- Undersampling experiment and frozen data: commit `55c82a5`
- Training implementation: commit `198fc2f`
- Evaluation implementation: commit `a99a99f`
- Qwen config generator: `scripts/create_qwen35_inner_val_undersampling_fold_config.py`
- Qwen undersampling generator: `scripts/create_undersampled_inner_train_fold.py`

The committed EXP-07 configuration specifies a learning rate of `0.00005`.

## Frozen data protocol

For every fold, the existing Qwen EXP-07 files were consumed directly; no fold was recreated. Majority-class `no_trace` records were sampled without replacement until the class count matched the minority `trace` count, followed by the historical seeded shuffle. The fold seed is `42 + fold_number`.

- Folds 01-07: 1,226 training records (613 `trace`, 613 `no_trace`)
- Folds 08-10: 1,228 training records (614 `trace`, 614 `no_trace`)
- Validation and test data: unchanged and not resampled
- Exact historical undersampling reproduction: passed for all 10 folds
- Train/validation/test ID non-overlap: passed for all 10 folds
- Mistral test file versus frozen Qwen test file: exact match for all 10 folds
- Duplicate training IDs: none

Fold 01 used a source inner train set of 1,528 records (613 `trace`, 915 `no_trace`), an undersampled training set of 1,226 records, an unchanged validation set of 170 records, and a held-out test set of 189 records.

## Prompt, loss, and training configuration

The prompt and label format is the exact plain-text Pattern-1 protocol used by Qwen EXP-07:

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

The historical implementation discards an incomplete gradient-accumulation group at each epoch boundary. Every fold completed 153 optimizer steps per epoch and 306 across two epochs. The historical scheduler formula configured 306 steps for Folds 01-07 and 307 for Folds 08-10.

## QLoRA configuration and preflight

Quantization used 4-bit NF4, BF16 compute, and double quantization. LoRA used rank 16, alpha 32, dropout 0.1, and no bias.

The target regex was selected from the Ministral architecture:

```text
.*language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)
```

This attached adapters only to the text decoder's attention projections.

| Preflight item | Result |
| --- | ---: |
| Status | Passed; no errors |
| Fold 01 training records | 1,226 (613/613) |
| Duplicate training rows | 0 |
| Adapter modules | 136 |
| Trainable parameters | 14,483,456 |
| Total parameters reported | 5,010,835,456 |
| Trainable proportion | 0.289043% |
| Vision/projector adapters | 0 |
| Longest-sequence optimizer step | Completed |
| Raw preflight loss | 2.4305863 |
| GPU process peak allocated | 9,502.04 MiB |
| GPU process peak reserved | 10,334.00 MiB |
| GPU free after optimizer step | 13,254.06 MiB |

Environment: Python 3.12.3, PyTorch 2.11.0+cu130, Transformers 5.8.0, PEFT 0.19.1, bitsandbytes 0.49.2, Datasets 4.8.5, Accelerate 1.13.0, CUDA runtime 13.0.

## Per-fold held-out test results

Invalid generated labels are excluded from classification metrics, matching the evaluation protocol.

| Fold | Accuracy | Trace P | Trace R | Trace F1 | No-trace P | No-trace R | No-trace F1 | Valid | Invalid |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 01 | 0.7135 | 0.5965 | 0.9067 | 0.7196 | 0.9014 | 0.5818 | 0.7072 | 185 | 4 |
| 02 | 0.7074 | 0.5943 | 0.8400 | 0.6961 | 0.8537 | 0.6195 | 0.7179 | 188 | 1 |
| 03 | 0.6508 | 0.5368 | 0.9605 | 0.6887 | 0.9434 | 0.4425 | 0.6024 | 189 | 0 |
| 04 | 0.6774 | 0.5630 | 0.8933 | 0.6907 | 0.8806 | 0.5315 | 0.6629 | 186 | 3 |
| 05 | 0.7005 | 0.6020 | 0.7763 | 0.6782 | 0.8090 | 0.6486 | 0.7200 | 187 | 2 |
| 06 | 0.6596 | 0.5577 | 0.7632 | 0.6444 | 0.7857 | 0.5893 | 0.6735 | 188 | 1 |
| 07 | 0.6649 | 0.5469 | 0.9459 | 0.6931 | 0.9298 | 0.4775 | 0.6310 | 185 | 4 |
| 08 | 0.7765 | 0.6782 | 0.8310 | 0.7468 | 0.8696 | 0.7407 | 0.8000 | 179 | 9 |
| 09 | 0.7059 | 0.5820 | 0.9467 | 0.7208 | 0.9385 | 0.5446 | 0.6893 | 187 | 1 |
| 10 | 0.7713 | 0.6509 | 0.9200 | 0.7624 | 0.9268 | 0.6726 | 0.7795 | 188 | 0 |

Best accuracy was Fold 08 at 0.7765; lowest accuracy was Fold 03 at 0.6508.

## Aggregate results

The following are unweighted means across the 10 folds. Standard deviations are sample standard deviations, matching `statistics.stdev` in the experiment runner.

| Metric | Mean | Sample standard deviation |
| --- | ---: | ---: |
| Accuracy | 0.7028 | 0.0434 |
| Trace precision | 0.5908 | 0.0450 |
| Trace recall | 0.8784 | 0.0716 |
| Trace F1 | 0.7041 | 0.0343 |
| No-trace precision | 0.8838 | 0.0549 |
| No-trace recall | 0.5849 | 0.0904 |
| No-trace F1 | 0.6984 | 0.0611 |

Across all folds there were 1,887 held-out records, 1,862 valid predictions, and 25 invalid predictions. The invalid prediction rate was 1.3249%.

### Pooled valid-prediction metrics

| Metric | Value |
| --- | ---: |
| Accuracy | 0.7025 |
| Trace precision | 0.5866 |
| Trace recall | 0.8783 |
| Trace F1 | 0.7034 |
| No-trace precision | 0.8774 |
| No-trace recall | 0.5844 |
| No-trace F1 | 0.7015 |
| Macro precision | 0.7320 |
| Macro recall | 0.7314 |
| Macro F1 | 0.7025 |

The pooled confusion matrix uses rows as true labels and columns as predicted labels:

| True / predicted | `trace` | `no_trace` |
| --- | ---: | ---: |
| `trace` | 657 | 91 |
| `no_trace` | 463 | 651 |

## Comparison of the three training distributions

All values below are unweighted fold means from the corresponding Ministral 10-fold experiments.

| Metric | Original imbalance | Oversampling | Undersampling |
| --- | ---: | ---: | ---: |
| Accuracy | 0.7404 | 0.7157 | 0.7028 |
| Trace precision | 0.6859 | 0.6125 | 0.5908 |
| Trace recall | 0.7007 | 0.8563 | 0.8784 |
| Trace F1 | 0.6709 | 0.7066 | 0.7041 |
| No-trace precision | 0.8117 | 0.8782 | 0.8838 |
| No-trace recall | 0.7684 | 0.6198 | 0.5849 |
| No-trace F1 | 0.7766 | 0.7178 | 0.6984 |

Relative to original-imbalance training, undersampling changed trace recall by +0.1777 and trace F1 by +0.0332, while accuracy changed by -0.0376 and no-trace recall by -0.1835. Relative to oversampling, undersampling added +0.0220 trace recall but changed trace F1 by -0.0025, accuracy by -0.0129, and no-trace recall by -0.0349.

Undersampling produced the highest trace recall of the three treatments, but it was also the most aggressive about predicting `trace`. Oversampling achieved almost the same trace F1 with higher overall accuracy and higher no-trace recall. Original-imbalance training retained the strongest overall accuracy and no-trace performance but had substantially lower trace recall.

## Notes and limitations

- Metrics exclude invalid generated labels, so valid/invalid counts must accompany reported scores.
- Fold 08 generated nine invalid labels, the highest count among the folds.
- Generation emitted a non-fatal warning because the model configuration contains `max_length=262144` while evaluation explicitly sets `max_new_tokens=5`. Transformers used `max_new_tokens=5`, as intended.
- Large adapters and periodic checkpoints remain server-side under `mistral_traceability/outputs/mistral_8b_qlora_undersampling/` and are intentionally excluded from version control.

## Reproducibility artifacts

- Full configuration: `mistral_traceability/configs/config_mistral_8b_qlora_undersampling_10fold.yaml`
- Fold 01 preflight configuration: `mistral_traceability/configs/config_mistral_8b_qlora_undersampling_fold_01.yaml`
- Frozen-fold manifest: `mistral_traceability/data/manifests/mistral_8b_qlora_undersampling_folds.json`
- Configuration rationale: `mistral_traceability/docs/mistral_8b_qlora_undersampling_config.md`
- Preflight implementation: `mistral_traceability/scripts/preflight_mistral_8b_qlora_undersampling.py`
- 10-fold runner: `mistral_traceability/scripts/run_mistral_8b_qlora_undersampling_10fold_cv.py`
- Data audit: `mistral_traceability/results/mistral_8b_qlora_undersampling/data_audit_fold_01_10.json`
- Fold metrics, predictions, and training histories: `mistral_traceability/results/mistral_8b_qlora_undersampling/fold_01/` through `fold_10/`
- Machine-readable summary: `mistral_traceability/results/mistral_8b_qlora_undersampling/mistral_8b_qlora_undersampling_10fold_summary.json`
- CSV summary: `mistral_traceability/results/mistral_8b_qlora_undersampling/mistral_8b_qlora_undersampling_10fold_summary.csv`
- Preflight report: `mistral_traceability/results/mistral_8b_qlora_undersampling/preflight/fold_01/preflight_report.json`
- Logs: `mistral_traceability/logs/mistral_8b_qlora_undersampling/`
