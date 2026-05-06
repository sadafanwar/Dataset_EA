# Experiment 02: Qwen2.5-1.5B QLoRA Fine-Tuning with Balanced Training Data

## Goal

The goal of this experiment was to fix the poor trace detection observed in Experiment 01.

Experiment 01 used Qwen2.5-0.5B with QLoRA on the imbalanced training set. The fine-tuned model became biased toward `no_trace`.

Experiment 02 changed two things:

1. Used a larger model: `Qwen/Qwen2.5-1.5B-Instruct`
2. Used a balanced oversampled training set

---

## Dataset Setup

### Original training split

| Label | Count |
|---|---:|
| trace | 525 |
| no_trace | 803 |

### Balanced oversampled training split

| Label | Count |
|---|---:|
| trace | 803 |
| no_trace | 803 |

Total balanced training examples: **1606**

Validation and test sets were unchanged.

| Split | File |
|---|---|
| Train | `data/traceability/train_balanced_oversampled.jsonl` |
| Validation | `data/traceability/validation.jsonl` |
| Test | `data/traceability/test.jsonl` |

The test set was not used during fine-tuning.

---

## Model and Method

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

## Training Result

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

## Test Set Results

| Metric | Value |
|---|---:|
| Accuracy | 0.7732 |
| Weighted Precision | 0.8157 |
| Weighted Recall | 0.7732 |
| Weighted F1 | 0.7569 |

---

## Per-Class Results

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| trace | 0.9375 | 0.5128 | 0.6630 |
| no_trace | 0.7220 | 0.9737 | 0.8291 |

---

## Confusion Matrix

Labels:

`['trace', 'no_trace']`

Confusion matrix:

| Actual \ Predicted | trace | no_trace |
|---|---:|---:|
| trace | 60 | 57 |
| no_trace | 4 | 148 |

Interpretation:

| Item | Count |
|---|---:|
| Correct trace predictions | 60 |
| Missed trace links | 57 |
| False trace predictions | 4 |
| Correct no-trace predictions | 148 |

---

## Comparison with Experiment 01

| Experiment | Model | Training Data | Accuracy | Weighted F1 | Trace Precision | Trace Recall | Trace F1 |
|---|---|---|---:|---:|---:|---:|---:|
| Experiment 01 | Fine-tuned Qwen2.5-0.5B + QLoRA | Imbalanced | 0.6208 | 0.5344 | 0.8571 | 0.1538 | 0.2609 |
| Experiment 02 | Fine-tuned Qwen2.5-1.5B + QLoRA | Balanced oversampled | 0.7732 | 0.7569 | 0.9375 | 0.5128 | 0.6630 |

---

## Key Observation

Experiment 02 substantially improved performance compared with Experiment 01.

The most important improvement was trace detection:

| Metric | Experiment 01 | Experiment 02 |
|---|---:|---:|
| Trace recall | 0.1538 | 0.5128 |
| Trace F1 | 0.2609 | 0.6630 |

Experiment 02 still misses some trace links, but it no longer collapses strongly toward `no_trace`.

The model is now conservative when predicting `trace`, but when it predicts `trace`, it is usually correct.

---

## Conclusion

The second experiment improved the fine-tuning result.

Using a larger Qwen model and balanced oversampled training data improved:

- overall accuracy
- weighted F1
- trace precision
- trace recall
- trace F1

This suggests that the poor result in Experiment 01 was likely caused by a combination of model size limitations and training class imbalance.

---

## Recommended Next Steps

The next experiment should try:

1. Keeping the balanced training data.
2. Testing Qwen2.5-3B or Qwen2.5-7B on the university server.
3. Improving trace recall further.
4. Computing loss only on the answer label rather than the full prompt.
5. Keeping the same prompt format for training and evaluation.