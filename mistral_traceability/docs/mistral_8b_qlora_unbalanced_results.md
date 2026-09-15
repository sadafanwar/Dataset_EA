# Ministral 3 8B QLoRA Original-Imbalance 10-Fold CV Results

## Status

The full 10-fold cross-validation experiment completed successfully on the university server. Each fold started from the same frozen base-model revision and trained an independent QLoRA adapter. No oversampling or undersampling was used.

## Reproducibility identity

- Model: `mistralai/Ministral-3-8B-Instruct-2512-BF16`
- Model revision: `f6fae9795746f63c9be8344932f01275f3c63734`
- Method: QLoRA, NF4 4-bit training with bfloat16 compute and double quantization
- Distribution: original imbalanced inner-training distribution
- Cross-validation: 10 frozen outer test folds with their frozen inner train/validation splits
- Historical protocol source: EXP-08 Qwen3.5-9B original-imbalance runner at commit `6ab46f20d351a1a4a4bb616875ad04cd8205fa4a`
- Prompt/loss: exact EXP-08 Pattern 1 text format, no chat template, full prompt-and-label causal-LM loss
- Final classification: greedy decoding, at most 5 new tokens
- Invalid-prediction policy: exclude invalid generations from classification metrics, matching EXP-08
- Aggregate summary created: `2026-09-15T17:31:22.933710+00:00`
- Approximate server wall-clock time: 1 hour 49 minutes

## Frozen data

Folds 01–07 each used 1,528 training records, 170 validation records, and 189 held-out test records. Folds 08–10 each used 1,529 training records, 170 validation records, and 188 held-out test records.

Training distribution:

- Folds 01–07: 613 `trace`, 915 `no_trace`
- Folds 08–10: 614 `trace`, 915 `no_trace`
- Validation: 68 `trace`, 102 `no_trace` in every fold
- Test, folds 01–07: 76 `trace`, 113 `no_trace`
- Test, folds 08–10: 75 `trace`, 113 `no_trace`

All ten Mistral test folds matched their corresponding frozen Qwen test folds. The audit found no example-ID overlap between training, validation, and test partitions.

## Hyperparameters

| Setting | Value |
|---|---:|
| Epochs | 2 |
| Learning rate | 0.00005 |
| Microbatch size | 1 |
| Gradient accumulation | 8 |
| Effective batch size | 8 |
| Warmup steps | 20 |
| Max gradient norm | 1.0 |
| Weight decay | 0.01 |
| Maximum sequence length | 1024 |
| Optimizer | AdamW |
| Scheduler | Linear with warmup |
| DataLoader shuffle | False |
| Validation interval | 50 optimizer steps |
| Validation subset | First 100 records |
| Checkpoint interval | 100 optimizer steps |
| Checkpoint retention | 3 |
| Seed | 42 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.1 |

The architecture-specific LoRA regex targeted only `q_proj`, `k_proj`, `v_proj`, and `o_proj` within the 34 text-decoder self-attention layers. This produced 136 adapter modules and 14,483,456 trainable parameters (0.289043% of 5,010,835,456 parameters), with no vision-tower or projector adapters.

## Fold 01 preflight

The corrected Fold 01 preflight passed before full training. It verified data provenance, prompt and hyperparameter settings, adapter placement, trainable-parameter count, and a longest-sequence forward/backward/optimizer step.

- Longest training example: 655 tokens; no truncation at the 1,024-token limit
- Peak process GPU allocation: 9,502.04 MiB
- Peak process GPU reservation: 10,334.00 MiB
- GPU free after optimizer step: 13,254.06 MiB
- Saved adapter/checkpoint: no
- Final preflight errors: none

The first preflight attempt detected platform-specific CRLF/LF byte hashes. The validation was corrected to use canonical LF hashes without modifying any dataset. The Mistral tokenizer was loaded with `fix_mistral_regex=True`, and gradient checkpointing used `use_reentrant=False`.

## Per-fold classification results

Metrics below follow the EXP-08 policy and are calculated over valid label generations.

| Fold | Accuracy | Trace P | Trace R | Trace F1 | No-trace P | No-trace R | No-trace F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 01 | 0.7500 | 0.6373 | 0.8784 | 0.7386 | 0.8902 | 0.6636 | 0.7604 |
| 02 | 0.6968 | 0.5812 | 0.8947 | 0.7047 | 0.8873 | 0.5625 | 0.6885 |
| 03 | 0.7953 | 0.7606 | 0.7500 | 0.7552 | 0.8200 | 0.8283 | 0.8241 |
| 04 | 0.7606 | 0.7778 | 0.5600 | 0.6512 | 0.7537 | 0.8938 | 0.8178 |
| 05 | 0.6772 | 0.6316 | 0.4737 | 0.5414 | 0.6970 | 0.8142 | 0.7510 |
| 06 | 0.7571 | 0.7818 | 0.5811 | 0.6667 | 0.7459 | 0.8835 | 0.8089 |
| 07 | 0.7754 | 0.6848 | 0.8289 | 0.7500 | 0.8632 | 0.7387 | 0.7961 |
| 08 | 0.7701 | 0.6404 | 0.9733 | 0.7725 | 0.9726 | 0.6339 | 0.7676 |
| 09 | 0.6522 | 0.6774 | 0.2800 | 0.3962 | 0.6471 | 0.9083 | 0.7557 |
| 10 | 0.7688 | 0.6860 | 0.7867 | 0.7329 | 0.8400 | 0.7568 | 0.7962 |

## Aggregate results

| Metric | Mean | Std. dev. | Min | Max |
|---|---:|---:|---:|---:|
| Accuracy | 0.7404 | 0.0476 | 0.6522 | 0.7953 |
| Trace precision | 0.6859 | 0.0680 | 0.5812 | 0.7818 |
| Trace recall | 0.7007 | 0.2192 | 0.2800 | 0.9733 |
| Trace F1 | 0.6709 | 0.1183 | 0.3962 | 0.7725 |
| No-trace precision | 0.8117 | 0.0996 | 0.6471 | 0.9726 |
| No-trace recall | 0.7684 | 0.1186 | 0.5625 | 0.9083 |
| No-trace F1 | 0.7766 | 0.0408 | 0.6885 | 0.8241 |

Best fold accuracy was 0.7953 on Fold 03; lowest was 0.6522 on Fold 09. Trace recall varied substantially across folds, while no-trace F1 was more stable.

### Pooled valid-prediction metrics

These metrics are calculated once from the summed confusion matrix rather than averaging the ten fold-level scores.

| Metric | Pooled value |
|---|---:|
| Accuracy | 0.7398 |
| Trace precision | 0.6727 |
| Trace recall | 0.7005 |
| Trace F1 | 0.6863 |
| No-trace precision | 0.7891 |
| No-trace recall | 0.7667 |
| No-trace F1 | 0.7777 |
| Macro precision | 0.7309 |
| Macro recall | 0.7336 |
| Macro F1 | 0.7320 |

## Summed confusion matrix

Rows are true labels and columns are predictions in the order `trace`, `no_trace`.

| True / predicted | Trace | No-trace |
|---|---:|---:|
| Trace | 524 | 224 |
| No-trace | 255 | 838 |

The summed matrix contains 1,841 valid predictions. The frozen test folds contain 1,887 examples in total, so 46 generations (2.44%) were invalid and excluded from classification metrics under the historical EXP-08 rule. The valid-prediction micro accuracy derived from the matrix is 0.7398. The reported primary accuracy is the unweighted mean of the ten fold accuracies: 0.7404.

## Interpretation

Original-imbalance QLoRA produced a mean trace F1 of 0.6709 and mean no-trace F1 of 0.7766. The larger variation in trace recall, especially Fold 09 (0.2800) versus Fold 08 (0.9733), indicates sensitivity to the particular training/test partition. Results should therefore be reported as the full 10-fold distribution rather than a single-fold score.

## Runtime notes

- The final corrected preflight and full 10-fold run completed without fatal errors.
- A bitsandbytes/PyTorch future warning appeared during 4-bit loading but did not interrupt training.
- Transformers warned that both a model-level `max_length` and experiment-level `max_new_tokens=5` were present during generation. Transformers explicitly applied `max_new_tokens=5`; this did not stop evaluation.
- Model adapters/checkpoints remain server-side under `mistral_traceability/outputs/mistral_8b_qlora_unbalanced/` and should not be pushed to GitHub because of their size.
