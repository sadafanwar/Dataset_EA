# Experiment 01: Qwen2.5-0.5B QLoRA Fine-Tuning Results

## Goal

The goal of this experiment was to compare the traceability classification performance of the base Qwen model before fine-tuning and after QLoRA fine-tuning.

The task is binary trace link detection:

- `trace`
- `no_trace`

The same held-out test set was used before and after fine-tuning.

---

## Dataset

The original cross-project traceability dataset was converted into Pattern 1 JSONL format:

```json
{
  "instruction": "Determine whether the lower-level requirement traces to the higher-level requirement.",
  "input": {
    "higher_level_requirement": "...",
    "lower_level_requirement": "..."
  },
  "output": "trace"
}
```

---

## Dataset Split

| Split | Total Examples | Trace | No-trace |
|---|---:|---:|---:|
| Train | 1328 | 525 | 803 |
| Validation | 290 | 115 | 175 |
| Test | 269 | 117 | 152 |

The test set was not used during fine-tuning.

---

## Model and Method

| Item | Value |
|---|---|
| Base model | Qwen/Qwen2.5-0.5B-Instruct |
| Fine-tuning method | QLoRA |
| Epochs | 2 |
| Batch size | 8 |
| Learning rate | 0.0001 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Hardware | Google Colab Tesla T4 GPU |

---

## Training Result

The QLoRA fine-tuning run completed successfully.

Validation perplexity improved during training:

| Stage | Perplexity |
|---|---:|
| Before fine-tuning | 50.43 |
| After fine-tuning | 11.12 |

However, lower perplexity did not translate into better traceability classification performance.

---

## Test Set Correctness Results

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 |
|---|---:|---:|---:|---:|
| Base Qwen2.5-0.5B-Instruct | 0.6617 | 0.6608 | 0.6617 | 0.6612 |
| Fine-tuned Qwen2.5-0.5B + QLoRA | 0.6208 | 0.7123 | 0.6208 | 0.5344 |

---

## Confusion Matrix

Labels:

```text
['trace', 'no_trace']
```

### Base Qwen

```text
[[ 70  47]
 [ 44 108]]
```

Interpretation:

- Correct trace predictions: 70
- Missed trace links: 47
- False trace predictions: 44
- Correct no-trace predictions: 108

### Fine-tuned Qwen + QLoRA

```text
[[ 18  99]
 [  3 149]]
```

Interpretation:

- Correct trace predictions: 18
- Missed trace links: 99
- False trace predictions: 3
- Correct no-trace predictions: 149

---

## Trace Class Performance

| Model | Correct Trace | Missed Trace | Trace Recall |
|---|---:|---:|---:|
| Base Qwen | 70 | 47 | 0.598 |
| Fine-tuned Qwen + QLoRA | 18 | 99 | 0.154 |

Trace recall is important because the main goal of requirement traceability is to find real trace links.

The fine-tuned model has much lower trace recall, which is problematic because missing real trace links makes the traceability matrix incomplete.

---

## Key Observation

The fine-tuned model became biased toward the `no_trace` class.

Although the fine-tuned model improved no-trace detection, it missed many actual trace links.

It detected only 18 out of 117 actual trace examples.

This means fine-tuning did not improve the main traceability objective.

---

## Conclusion

The first QLoRA fine-tuning run was technically successful, but it did not improve classification correctness.

The base Qwen model performed better on the held-out test set than the fine-tuned model.

The fine-tuned model learned a strong bias toward `no_trace`, which reduced trace recall and overall F1.

---

## Recommended Next Steps

The next experiment should improve the setup by:

1. Handling class imbalance using balanced sampling or oversampling trace examples.
2. Computing loss only on the answer label instead of the full prompt.
3. Reporting trace precision, trace recall, and trace F1 separately.
4. Trying Qwen2.5-1.5B if GPU memory allows.
5. Aligning the training and evaluation prompt format more tightly.
6. Optionally adding rationale or explanation labels for better traceability reasoning.
