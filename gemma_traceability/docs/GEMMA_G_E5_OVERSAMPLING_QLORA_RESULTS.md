# Gemma G-E5 — QLoRA Random Oversampling 10-Fold CV

**Experiment:** Gemma 2 9B QLoRA random oversampling 10-fold CV
**Model:** `google/gemma-2-9b-it`
**Model revision:** `11c9b309abf73637e4b6f9a3fa1e92e615547819`
**Balancing:** `random_oversampling_with_replacement`

## Aggregate Results

| Metric | Mean ± SD |
|---|---:|
| Accuracy | 0.9184 ± 0.0258 |
| Trace Precision | 0.9039 ± 0.0399 |
| Trace Recall | 0.8931 ± 0.0489 |
| Trace F1 | 0.8976 ± 0.0334 |
| No-trace Precision | 0.9297 ± 0.0304 |
| No-trace Recall | 0.9354 ± 0.0286 |
| No-trace F1 | 0.9321 ± 0.0211 |
| Derived Macro F1 | 0.9148 |

## Prediction Coverage

- Total valid predictions: **1887 / 1887**
- Invalid predictions: **0**

## G-E4 vs G-E5

| Metric | G-E4 Original Imbalance | G-E5 Oversampling |
|---|---:|---:|
| Accuracy | 0.9195 | 0.9184 |
| Trace F1 | 0.8956 | 0.8976 |
| No-trace F1 | 0.9343 | 0.9321 |
| Derived Macro F1 | 0.9150 | 0.9148 |

## Summed Confusion Matrix

Labels: `[trace, no_trace]`

```text
[676, 81]
[73, 1057]
```

## Note

Oversampling uses the frozen historical random-oversampling folds with replacement. Validation and outer test folds remain unchanged.