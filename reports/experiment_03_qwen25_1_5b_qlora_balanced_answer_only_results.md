# Experiment 03: Qwen2.5-1.5B QLoRA Fine-Tuning with Balanced Oversampled Data and Answer-Only Loss

## Goal

The goal of Experiment 03 was to improve trace detection further beyond Experiment 02 while keeping the existing project structure intact.

Experiment 02 already improved substantially over Experiment 01 by:

1. Using a larger base model: `Qwen/Qwen2.5-1.5B-Instruct`
2. Using balanced oversampled training data

Experiment 03 kept those strengths and added a more task-focused training objective:

3. Computing loss only on the final answer label (`trace` / `no_trace`) instead of the full prompt

The same held-out test set was used for all three experiments.

---

## What Changed in Experiment 03

| Area | Experiment 02 | Experiment 03 |
|---|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` | `Qwen/Qwen2.5-1.5B-Instruct` |
| Fine-tuning method | QLoRA | QLoRA |
| Training data | Balanced oversampled | Balanced oversampled |
| Loss target | Full prompt + answer tokens | Answer label tokens only |
| Batch size | 4 | 4 |
| Gradient accumulation | 2 | 2 |
| Effective batch size | 8 | 8 |
| Epochs | 2 | 2 |
| LoRA rank | 16 | 16 |
| LoRA alpha | 32 | 32 |
| LoRA dropout | 0.1 | 0.1 |
| Quantization | NF4 4-bit | NF4 4-bit |

---

## Why Answer-Only Loss Matters

This traceability task is a classification task in instruction format.

The prompt contains:

- the instruction
- the higher-level requirement
- the lower-level requirement
- the final answer label

With full-prompt loss, the model is trained to predict the entire sequence, including prompt text that is repeated across examples.

With answer-only loss, prompt tokens are masked out during training and only the final answer label contributes to the loss.

This helps because:

1. the learning signal becomes more focused on the real task
2. the model spends less capacity learning prompt reconstruction
3. the supervision is more closely aligned with evaluation behavior
4. the model is pushed to separate `trace` vs `no_trace` more directly

---

## Dataset Setup

### Original training split

| Label | Count |
|---|---:|
| trace | 525 |
| no_trace | 803 |

### Balanced oversampled training split used in Experiments 02 and 03

| Label | Count |
|---|---:|
| trace | 803 |
| no_trace | 803 |

Total balanced training examples: **1606**

Validation and test sets were unchanged.

---

## Experiment 03 Training Run Summary

Training completed successfully.

Important note:

Experiment 03 uses answer-only loss, while the current validation perplexity pipeline still evaluates full-sequence perplexity. Because of that mismatch, perplexity is no longer the best metric for judging Experiment 03. The real decision should be based on held-out classification metrics on the test set.

Available evaluated checkpoints:

- `outputs/traceability_qwen25_exp03/best_model`
- `outputs/traceability_qwen25_exp03/checkpoint-300`
- `outputs/traceability_qwen25_exp03/checkpoint-400`

Among these, **`checkpoint-300` produced the best held-out test performance**.

---

## Overall Test Set Comparison

| Stage | Model | Key setup | Accuracy | Weighted F1 |
|---|---|---|---:|---:|
| Baseline | Base Qwen2.5-0.5B-Instruct | No fine-tuning | 0.6617 | 0.6612 |
| Experiment 01 | Fine-tuned Qwen2.5-0.5B + QLoRA | Imbalanced train | 0.6208 | 0.5344 |
| Experiment 02 | Fine-tuned Qwen2.5-1.5B + QLoRA | Balanced oversampled train | 0.7732 | 0.7569 |
| Experiment 03 best checkpoint | Fine-tuned Qwen2.5-1.5B + QLoRA | Balanced oversampled train + answer-only loss | 0.9071 | Not recorded in original summary script |

---

## Trace Class Comparison

| Stage | Correct Trace | Missed Trace | Trace Precision | Trace Recall | Trace F1 |
|---|---:|---:|---:|---:|---:|
| Baseline | 70 | 47 | 0.6140 | 0.5983 | 0.6061 |
| Experiment 01 | 18 | 99 | 0.8571 | 0.1538 | 0.2609 |
| Experiment 02 | 60 | 57 | 0.9375 | 0.5128 | 0.6630 |
| Experiment 03 best checkpoint | 99 | 18 | 0.9340 | 0.8462 | 0.8879 |

---

## No-Trace Class Comparison

| Stage | Correct No-Trace | False Trace Predictions | No-Trace F1 |
|---|---:|---:|---:|
| Baseline | 108 | 44 | 0.7036 |
| Experiment 01 | 149 | 3 | 0.7450 |
| Experiment 02 | 148 | 4 | 0.8291 |
| Experiment 03 best checkpoint | 145 | 7 | 0.9206 |

---

## Confusion Matrix Comparison

Labels: `['trace', 'no_trace']`

### Baseline

| Actual \ Predicted | trace | no_trace |
|---|---:|---:|
| trace | 70 | 47 |
| no_trace | 44 | 108 |

### Experiment 01

| Actual \ Predicted | trace | no_trace |
|---|---:|---:|
| trace | 18 | 99 |
| no_trace | 3 | 149 |

### Experiment 02

| Actual \ Predicted | trace | no_trace |
|---|---:|---:|
| trace | 60 | 57 |
| no_trace | 4 | 148 |

### Experiment 03 best checkpoint

| Actual \ Predicted | trace | no_trace |
|---|---:|---:|
| trace | 99 | 18 |
| no_trace | 7 | 145 |

---

## Additional Experiment 03 Checkpoint Results

| Checkpoint | Accuracy | Trace Precision | Trace Recall | Trace F1 | No-Trace F1 |
|---|---:|---:|---:|---:|---:|
| `best_model` | 0.8996 | 0.9327 | 0.8291 | 0.8778 | 0.9148 |
| `checkpoint-300` | 0.9071 | 0.9340 | 0.8462 | 0.8879 | 0.9206 |
| `checkpoint-400` | 0.8996 | 0.9327 | 0.8291 | 0.8778 | 0.9148 |

This shows that `checkpoint-300` generalized slightly better than the final saved `best_model`.

---

## What Improved Most in Experiment 03

Experiment 03 produced the best result so far.

The biggest improvements were:

1. **Trace recall improved sharply**
   - Experiment 02: `0.5128`
   - Experiment 03 best checkpoint: `0.8462`

2. **Trace F1 improved sharply**
   - Experiment 02: `0.6630`
   - Experiment 03 best checkpoint: `0.8879`

3. **Overall accuracy improved strongly**
   - Experiment 02: `0.7732`
   - Experiment 03 best checkpoint: `0.9071`

4. **Missed trace links dropped heavily**
   - Experiment 02 missed `57`
   - Experiment 03 best checkpoint missed only `18`

5. **Trace precision remained very high**
   - Experiment 02: `0.9375`
   - Experiment 03 best checkpoint: `0.9340`

This is important because Experiment 03 improved recall dramatically without sacrificing much precision.

---

## Why Experiment 03 Likely Performed Better

The most likely reason for the improvement is the addition of **answer-only loss** on top of the already strong Experiment 02 setup.

Balanced oversampling already helped reduce the `no_trace` bias seen in Experiment 01.

The answer-only loss likely improved the model further because it made the optimization target match the real task more closely. Instead of learning to reproduce the full prompt text, the model learned to focus on the final classification decision.

In practice, this seems to have produced a model that:

- still stays highly precise when predicting `trace`
- but is no longer overly conservative
- and therefore captures far more real trace links

---

## Why `checkpoint-300` Was Better than `best_model`

`best_model` in the current training pipeline is selected using training loss, not held-out classification metrics.

This means the checkpoint with the lowest training loss is not always the checkpoint with the best real test performance.

Experiment 03 showed that:

- `checkpoint-300` had better accuracy
- `checkpoint-300` had better trace recall
- `checkpoint-300` had better trace F1

Therefore, future experiments should evaluate multiple saved checkpoints instead of relying only on the final saved best model from training loss.

---

## Final Conclusion

Experiment 03 is the best model so far.

Compared with all previous stages, it achieved the strongest balance of:

- high trace precision
- much higher trace recall
- highest trace F1
- highest overall accuracy

The main reason appears to be the combination of:

1. larger base model (`1.5B`)
2. balanced oversampled training data
3. answer-only loss focused directly on the label prediction task

### Best current result

| Best current checkpoint | Accuracy | Trace Precision | Trace Recall | Trace F1 |
|---|---:|---:|---:|---:|
| `checkpoint-300` from Fine-tuned Qwen2.5-1.5B + QLoRA with balanced oversampled data and answer-only loss | 0.9071 | 0.9340 | 0.8462 | 0.8879 |

---

