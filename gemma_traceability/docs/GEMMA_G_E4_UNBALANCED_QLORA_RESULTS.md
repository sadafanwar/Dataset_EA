# Gemma G-E4 — QLoRA Original-Imbalance 10-Fold CV

**Experiment:** Gemma 2 9B QLoRA original imbalanced 10-fold CV
**Model:** `google/gemma-2-9b-it`
**Model revision:** `11c9b309abf73637e4b6f9a3fa1e92e615547819`
**Method:** QLoRA
**Balancing:** `none_original_imbalanced_inner_train`

## Protocol

- 10-fold cross-validation using the frozen Qwen reference folds
- Original/natural class distribution; no over- or undersampling
- QLoRA with 4-bit NF4 quantization
- LoRA targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- 168 LoRA target modules
- 17,891,328 trainable parameters
- 2 epochs
- Learning rate: 5e-5
- Batch size: 1
- Gradient accumulation: 8
- Effective batch size: 8
- Maximum sequence length: 1024
- Seed: 42

## Aggregate Results

| Metric | Mean ± SD |
|---|---:|
| Accuracy | 0.9195 ± 0.0342 |
| Trace Precision | 0.9238 ± 0.0371 |
| Trace Recall | 0.8719 ± 0.0750 |
| Trace F1 | 0.8956 ± 0.0473 |
| No-trace Precision | 0.9191 ± 0.0443 |
| No-trace Recall | 0.9513 ± 0.0251 |
| No-trace F1 | 0.9343 ± 0.0268 |
| Derived Macro F1 | 0.9150 |

## Prediction Coverage

- Total test records: **1887**
- Valid predictions: **1887**
- Invalid predictions: **0**
- Valid coverage: **100.00%**

## Per-Fold Results

| Fold | Accuracy | Trace P | Trace R | Trace F1 | No-trace P | No-trace R | No-trace F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| fold_01 | 0.9312 | 0.9200 | 0.9079 | 0.9139 | 0.9386 | 0.9469 | 0.9427 |
| fold_02 | 0.9630 | 0.9367 | 0.9737 | 0.9548 | 0.9818 | 0.9558 | 0.9686 |
| fold_03 | 0.8624 | 0.8906 | 0.7500 | 0.8143 | 0.8480 | 0.9381 | 0.8908 |
| fold_04 | 0.9259 | 0.8780 | 0.9474 | 0.9114 | 0.9626 | 0.9115 | 0.9364 |
| fold_05 | 0.9048 | 0.9394 | 0.8158 | 0.8732 | 0.8862 | 0.9646 | 0.9237 |
| fold_06 | 0.9418 | 0.9452 | 0.9079 | 0.9262 | 0.9397 | 0.9646 | 0.9520 |
| fold_07 | 0.8889 | 0.9365 | 0.7763 | 0.8489 | 0.8651 | 0.9646 | 0.9121 |
| fold_08 | 0.9468 | 0.9452 | 0.9200 | 0.9324 | 0.9478 | 0.9646 | 0.9561 |
| fold_09 | 0.9521 | 0.9853 | 0.8933 | 0.9371 | 0.9333 | 0.9912 | 0.9614 |
| fold_10 | 0.8777 | 0.8611 | 0.8267 | 0.8435 | 0.8879 | 0.9115 | 0.8996 |

## Summed Confusion Matrix

Labels: `[trace, no_trace]`

```text
[660, 97]
[55, 1075]
```

## Reporting Note

Metrics are reported as mean ± standard deviation across the ten frozen outer test folds. Macro F1 shown above is derived from the mean Trace F1 and mean No-trace F1.
