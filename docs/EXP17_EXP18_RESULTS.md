# EXP-17 and EXP-18 Experimental Results

## 1. Experiment overview

| Experiment | Name | Instruction strategy |
|---|---|---|
| EXP-17 | Qwen3.5 Label + Rationale, Single-Template Instruction Tuning | One fixed instruction template |
| EXP-18 | Qwen3.5 Label + Rationale, Multi-Template Instruction Tuning | Five semantically equivalent instruction templates |

The dataset, ten cross-validation folds, model family, training setup, evaluation protocol, and hyperparameters were kept aligned. The intended experimental variable was the instruction-template strategy.

## 2. Primary ten-fold classification results

Values are reported as mean ± sample standard deviation across ten folds.

| Metric | EXP-17 | EXP-18 | Difference |
|---|---:|---:|---:|
| Accuracy | 0.8993 ± 0.0206 | 0.9009 ± 0.0196 | +0.0016 |
| Macro precision | 0.8978 ± 0.0222 | 0.9002 ± 0.0202 | +0.0024 |
| Macro recall | 0.8924 ± 0.0219 | 0.8931 ± 0.0219 | +0.0007 |
| Macro F1 | 0.8945 ± 0.0217 | 0.8960 ± 0.0208 | +0.0015 |
| Weighted F1 | 0.8990 ± 0.0207 | 0.9004 ± 0.0198 | +0.0014 |
| Trace precision | 0.8886 ± 0.0333 | 0.8953 ± 0.0277 | +0.0067 |
| Trace recall | 0.8573 ± 0.0346 | 0.8534 ± 0.0385 | -0.0039 |
| Trace F1 | 0.8722 ± 0.0266 | 0.8734 ± 0.0258 | +0.0012 |
| No-trace precision | 0.9069 ± 0.0208 | 0.9052 ± 0.0231 | -0.0017 |
| No-trace recall | 0.9274 ± 0.0232 | 0.9327 ± 0.0192 | +0.0053 |
| No-trace F1 | 0.9169 ± 0.0170 | 0.9186 ± 0.0159 | +0.0017 |
| Format-valid rate | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 |

## 3. Primary interpretation

EXP-18 achieved small improvements in most aggregate metrics while retaining a 100% format-valid rate.

The largest positive change was in trace precision. Trace recall decreased slightly, indicating a modest precision-recall trade-off rather than a large overall performance gain.

A defensible conclusion is:

> Multi-template instruction tuning preserved the classification performance of the single-template baseline and produced small improvements in several aggregate and class-level metrics.

The differences are small and should not be described as statistically significant without paired fold-level statistical testing.

## 4. EXP-18 five-template robustness evaluation

EXP-18 was evaluated using all five semantically equivalent instruction templates for every test record in every fold.

### Fold-level robustness results

| Fold | Unanimous rate | Pairwise agreement | Majority-vote accuracy |
|---|---:|---:|---:|
| Fold 01 | 0.9630 | 0.9820 | 0.8783 |
| Fold 02 | 0.9259 | 0.9651 | 0.9048 |
| Fold 03 | 0.9471 | 0.9746 | 0.8730 |
| Fold 04 | 0.9259 | 0.9640 | 0.9101 |
| Fold 05 | 0.9788 | 0.9894 | 0.9101 |
| Fold 06 | 0.9524 | 0.9746 | 0.9206 |
| Fold 07 | 0.9259 | 0.9672 | 0.8836 |
| Fold 08 | 0.9521 | 0.9777 | 0.9149 |
| Fold 09 | 0.9521 | 0.9755 | 0.8936 |
| Fold 10 | 0.9362 | 0.9713 | 0.8936 |

### Aggregate robustness results

| Metric | Mean ± sample standard deviation |
|---|---:|
| Unanimous prediction rate | 0.9459 ± 0.0176 |
| Pairwise template agreement | 0.9741 ± 0.0078 |
| Majority-vote accuracy | 0.8983 ± 0.0163 |

## 5. Robustness interpretation

EXP-18 produced the same label across all five templates for approximately 94.59% of test records.

Pairwise agreement between template predictions was 97.41%, with low variation across folds.

Majority-vote accuracy was 89.83%, close to the primary EXP-18 accuracy of 90.09%.

These results show that EXP-18 is internally stable across the evaluated instruction formulations. They do not prove that EXP-18 is more robust than EXP-17 because EXP-17 has not yet been evaluated under the same five-template robustness protocol.

A defensible conclusion is:

> EXP-18 maintained classification performance while showing high prediction consistency across five semantically equivalent instruction templates.

## 6. Main findings

1. EXP-18 slightly improved most aggregate metrics.
2. EXP-17 retained slightly higher trace recall and no-trace precision.
3. Both experiments achieved a 100% format-valid rate.
4. EXP-18 achieved 94.59% unanimous predictions across templates.
5. EXP-18 achieved 97.41% pairwise template agreement.
6. Classification differences between EXP-17 and EXP-18 remain small.
7. Paired statistical testing is required before claiming superiority.

## 7. Paper-ready result statement

> Under an identical ten-fold evaluation protocol, the multi-template EXP-18 configuration preserved the predictive performance of the single-template EXP-17 baseline and produced small improvements in most aggregate metrics. Accuracy increased from 0.8993 to 0.9009 and macro F1 increased from 0.8945 to 0.8960, while both configurations retained a 1.0000 format-valid rate. The secondary robustness evaluation of EXP-18 showed a mean unanimous prediction rate of 0.9459 and a mean pairwise template-agreement rate of 0.9741 across the ten folds. Majority-vote accuracy was 0.8983, remaining close to the primary EXP-18 accuracy. These findings indicate that multi-template instruction tuning maintained classification performance and yielded high consistency across semantically equivalent prompt formulations.

## 8. Pending analyses

- Paired statistical testing for EXP-17 versus EXP-18
- Effect sizes and confidence intervals
- Per-template classification analysis
- Analysis of non-unanimous records
- Qualitative error analysis
- Equivalent five-template robustness testing for EXP-17
