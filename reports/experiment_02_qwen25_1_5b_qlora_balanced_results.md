# Experiment 02: Qwen2.5-1.5B QLoRA Fine-Tuning with Balanced Training Data

## Goal

The goal of this experiment was to improve the poor trace detection observed in Experiment 01.

Experiment 01 used Qwen2.5-0.5B with QLoRA on the imbalanced training set. The fine-tuned model became biased toward `no_trace`.

Experiment 02 changed two things:

1. Used a larger model: `Qwen/Qwen2.5-1.5B-Instruct`
2. Used a balanced oversampled training set

The same held-out test set was used for the baseline, Experiment 01, and Experiment 02.

---

## Dataset Setup

### Original training split

| Label | Count |
|---|---:|
| trace | 525 |
| no_trace | 803 |

### Balanced oversampled training split used in Experiment 02

| Label | Count |
|---|---:|
| trace | 803 |
| no_trace | 803 |

Total balanced training examples: **1606**

Validation and test sets were unchanged.

| Split | File |
|---|---|
| Original train | `data/traceability/train.jsonl` |
| Balanced train | `data/traceability/train_balanced_oversampled.jsonl` |
| Validation | `data/traceability/validation.jsonl` |
| Test | `data/traceability/test.jsonl` |

The test set was not used during fine-tuning.

---

## Experiment Setup

| Stage | Model | Fine-tuning | Training data |
|---|---|---|---|
| Baseline | Qwen2.5-0.5B-Instruct | No | None |
| Experiment 01 | Qwen2.5-0.5B-Instruct | QLoRA | Original imbalanced train set |
| Experiment 02 | Qwen2.5-1.5B-Instruct | QLoRA | Balanced oversampled train set |

### Experiment 02 training configuration

| Item | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Fine-tuning method | QLoRA |
| Epochs | 2 |
| Batch size | 4 |
| Gradient accumulation steps | 2 |
| Effective batch size | 8 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.1 |
| Quantization | NF4 4-bit |
| Hardware | Google Colab Tesla T4 GPU |

---

## Training Result for Experiment 02

Training completed successfully.

| Stage | Validation Perplexity |
|---|---:|
| Before training | 37.75 |
| After training | 9.51 |

Best model was saved to:

`outputs/traceability_qwen25_1_5b_qlora_balanced/best_model`

The model artifact was backed up to Google Drive as:

`traceability_qwen25_1_5b_qlora_balanced_outputs.zip`

---

## Overall Test Set Comparison

The same held-out test set, `data/traceability/test.jsonl`, was used for all three evaluations.

| Stage | Model | Training data | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 |
|---|---|---|---:|---:|---:|---:|
| Baseline | Base Qwen2.5-0.5B-Instruct | No fine-tuning | 0.6617 | 0.6608 | 0.6617 | 0.6612 |
| Experiment 01 | Fine-tuned Qwen2.5-0.5B + QLoRA | Original imbalanced | 0.6208 | 0.7123 | 0.6208 | 0.5344 |
| Experiment 02 | Fine-tuned Qwen2.5-1.5B + QLoRA | Balanced oversampled | 0.7732 | 0.8157 | 0.7732 | 0.7569 |

---

## Trace Class Comparison

Trace detection is the most important part of this task because missed trace links make the traceability matrix incomplete.

| Stage | Correct Trace | Missed Trace | Trace Precision | Trace Recall | Trace F1 |
|---|---:|---:|---:|---:|---:|
| Baseline | 70 | 47 | 0.6140 | 0.5983 | 0.6061 |
| Experiment 01 | 18 | 99 | 0.8571 | 0.1538 | 0.2609 |
| Experiment 02 | 60 | 57 | 0.9375 | 0.5128 | 0.6630 |

---

## No-Trace Class Comparison

| Stage | Correct No-Trace | False Trace Predictions | No-Trace Precision | No-Trace Recall | No-Trace F1 |
|---|---:|---:|---:|---:|---:|
| Baseline | 108 | 44 | 0.6968 | 0.7105 | 0.7036 |
| Experiment 01 | 149 | 3 | 0.6008 | 0.9803 | 0.7450 |
| Experiment 02 | 148 | 4 | 0.7220 | 0.9737 | 0.8291 |

---

## Confusion Matrices

Labels: `['trace', 'no_trace']`

### Baseline: Base Qwen2.5-0.5B-Instruct

| Actual \ Predicted | trace | no_trace |
|---|---:|---:|
| trace | 70 | 47 |
| no_trace | 44 | 108 |

### Experiment 01: Fine-tuned Qwen2.5-0.5B + QLoRA

| Actual \ Predicted | trace | no_trace |
|---|---:|---:|
| trace | 18 | 99 |
| no_trace | 3 | 149 |

### Experiment 02: Fine-tuned Qwen2.5-1.5B + QLoRA with balanced training data

| Actual \ Predicted | trace | no_trace |
|---|---:|---:|
| trace | 60 | 57 |
| no_trace | 4 | 148 |

---

## Key Observations

Experiment 01 performed worse than the baseline. Although it improved `no_trace` recall, it missed most actual trace links.

Experiment 01 detected only **18 out of 117** trace examples.

Experiment 02 improved substantially. It detected **60 out of 117** trace examples and achieved the best overall performance.

| Metric | Baseline | Experiment 01 | Experiment 02 |
|---|---:|---:|---:|
| Accuracy | 0.6617 | 0.6208 | 0.7732 |
| Weighted F1 | 0.6612 | 0.5344 | 0.7569 |
| Trace Precision | 0.6140 | 0.8571 | 0.9375 |
| Trace Recall | 0.5983 | 0.1538 | 0.5128 |
| Trace F1 | 0.6061 | 0.2609 | 0.6630 |

Experiment 02 still has slightly lower trace recall than the baseline, but it has much higher trace precision and the best trace F1.

The model is now conservative when predicting `trace`, but when it predicts `trace`, it is usually correct.

---

## Conclusion

Experiment 02 improved the fine-tuning result.

Using a larger Qwen model and balanced oversampled training data improved:

- overall accuracy
- weighted F1
- trace precision
- trace recall compared with Experiment 01
- trace F1
- no-trace F1

The poor result in Experiment 01 was likely caused by a combination of model size limitations and training class imbalance.

The best result so far is Experiment 02:

| Best current model | Accuracy | Weighted F1 | Trace F1 |
|---|---:|---:|---:|
| Fine-tuned Qwen2.5-1.5B + QLoRA with balanced training data | 0.7732 | 0.7569 | 0.6630 |

---

## Recommended Next Steps

The next experiment should try:

1. Keeping the balanced training data.
2. Testing Qwen2.5-3B or Qwen2.5-7B on the university server.
3. Improving trace recall further.
4. Computing loss only on the answer label rather than the full prompt.
5. Keeping the same prompt format for training and evaluation.