# Google Gemma 2 9B Instruct — No-Fine-Tuning Traceability Results

## Experiment status

The complete no-fine-tuning prompting experiment finished successfully on
2026-09-17. The run produced 2,300 predictions across five evaluation
settings. All predictions were valid and no prompts were truncated.

No training occurred. The model was loaded for inference only, all parameter
`requires_grad` flags were explicitly disabled, the audited trainable-parameter
count was zero, and no optimizer, backward pass, adapter, or updated checkpoint
was created.

## Frozen model and environment

| Item | Value |
| --- | --- |
| Model | `google/gemma-2-9b-it` |
| Frozen revision | `11c9b309abf73637e4b6f9a3fa1e92e615547819` |
| Architecture | `Gemma2ForCausalLM` |
| Tokenizer | `GemmaTokenizer` |
| Inference representation | NF4 4-bit, BF16 compute, double quantization |
| Prompt wrapper | Checkpoint-provided Gemma chat template |
| Trainable parameters | 0 |
| GPU | NVIDIA GeForce RTX 4090 |
| PyTorch | 2.11.0+cu130 |
| Transformers | 5.8.0 |
| BitsAndBytes | 0.49.2 |
| CUDA runtime | 13.0 |

The full run took 758.99 seconds (approximately 12 minutes 39 seconds). Peak
process GPU memory was 6,913.07 MiB allocated and 7,206 MiB reserved.

## Protocol

The experiment reused the corrected historical Qwen prompting design:

- Exact task wording and label definitions.
- Deterministic greedy generation with `do_sample=false`.
- Maximum 8 generated tokens.
- Strict parsing to `trace`, `no_trace`, or `invalid`.
- Invalid predictions excluded from classification metrics; none occurred.
- Labels and confusion matrices use the order `[trace, no_trace]`.
- The fixed test sets contain 50 trace and 50 no-trace examples.
- Demonstration examples do not overlap the associated test set.
- Within each multi-run, the 5-shot and 10-shot demonstrations are disjoint.

The model-specific adaptation was wrapping the unchanged task content in
Gemma's own chat template. The effective prompt limit was 8,184 tokens, which
reserves 8 tokens inside Gemma's 8,192-token context window.

## Simple prompting results

Each simple setting used one run on the same 100-example simple test set.

| Setting | Runs | Accuracy | Trace P | Trace R | Trace F1 | No-trace P | No-trace R | No-trace F1 | Invalid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero-shot | 1 | 0.8800 | 0.9318 | 0.8200 | 0.8723 | 0.8393 | 0.9400 | 0.8868 | 0 |
| 5-shot | 1 | 0.8200 | 0.9444 | 0.6800 | 0.7907 | 0.7500 | 0.9600 | 0.8421 | 0 |
| 10-shot | 1 | 0.8200 | 0.9444 | 0.6800 | 0.7907 | 0.7500 | 0.9600 | 0.8421 | 0 |

Confusion matrices:

- Zero-shot: `[[41, 9], [3, 47]]`
- 5-shot: `[[34, 16], [2, 48]]`
- 10-shot: `[[34, 16], [2, 48]]`

Zero-shot was the strongest simple setting. The selected single 5-shot and
10-shot demonstration sets increased no-trace recall but reduced trace recall,
producing the same aggregate metrics and confusion matrix.

## Multi-run prompting results

The multi-run experiment used a separate fixed 100-example test set. Ten
different demonstration selections were evaluated for each shot count. Values
below are mean ± sample standard deviation over the ten runs.

| Setting | Runs | Accuracy | Trace P | Trace R | Trace F1 | No-trace P | No-trace R | No-trace F1 | Invalid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5-shot multi-run | 10 | 0.8670 ± 0.0189 | 0.9702 ± 0.0252 | 0.7580 ± 0.0382 | 0.8503 ± 0.0235 | 0.8021 ± 0.0251 | 0.9760 ± 0.0207 | 0.8802 ± 0.0158 | 0 |
| 10-shot multi-run | 10 | 0.8720 ± 0.0220 | 0.9734 ± 0.0238 | 0.7660 ± 0.0517 | 0.8560 ± 0.0304 | 0.8083 ± 0.0310 | 0.9780 ± 0.0199 | 0.8846 ± 0.0168 | 0 |

Summed confusion matrices over 1,000 predictions per setting:

- 5-shot multi-run: `[[379, 121], [12, 488]]`
- 10-shot multi-run: `[[383, 117], [11, 489]]`

Per-run accuracies:

| Run/seed | 5-shot | 10-shot |
| ---: | ---: | ---: |
| 1 | 0.86 | 0.88 |
| 2 | 0.87 | 0.88 |
| 3 | 0.89 | 0.87 |
| 4 | 0.84 | 0.82 |
| 5 | 0.90 | 0.85 |
| 6 | 0.88 | 0.87 |
| 7 | 0.85 | 0.89 |
| 8 | 0.87 | 0.89 |
| 9 | 0.86 | 0.88 |
| 10 | 0.85 | 0.89 |

The 10-shot multi-run setting improved mean accuracy by 0.005 and mean trace
F1 by approximately 0.0056 relative to 5-shot multi-run. The gain is small
compared with the observed variation across demonstration selections.

## Interpretation

Gemma performed strongly without task-specific training. Across multi-run
settings it showed very high trace precision (about 0.97) and no-trace recall
(about 0.98), while trace recall was lower (about 0.76). This indicates a
conservative trace-link decision pattern: predicted trace links were usually
correct, but some true trace links were classified as no-trace.

The simple zero-shot result exceeded the two single few-shot results. This does
not establish that demonstrations are generally harmful, because the multi-run
results demonstrate sensitivity to which examples are selected, and the simple
and multi-run protocols use different fixed test sets.

## Integrity checks

- Frozen data audit: passed.
- Exact historical file names and server-side byte copies: passed.
- Prompt/test overlap checks: passed for the simple setting and all ten
  multi-runs.
- Full-run status: completed.
- Prediction files: 23.
- Rows per prediction file: 100.
- Total predictions: 2,300.
- Invalid predictions: 0.
- Truncated prompts: 0.
- Explicitly frozen parameters: true.
- Trainable parameter count: 0.
- Model outputs or checkpoints committed: none.

## Result locations

- Combined table: `gemma_traceability/results/gemma_2_9b_it_no_finetuning/comparison_summary.csv`
- Simple summaries: `gemma_traceability/results/gemma_2_9b_it_no_finetuning/simple/`
- Multi-run summaries: `gemma_traceability/results/gemma_2_9b_it_no_finetuning/multi_run/`
- Full run manifest: `gemma_traceability/results/gemma_2_9b_it_no_finetuning/run_manifest_all.json`
- Preflight reports: `gemma_traceability/results/gemma_2_9b_it_no_finetuning/preflight/`
- Console log: `gemma_traceability/logs/gemma_2_9b_it_no_finetuning/full_no_finetuning_console.log`

## Limitations

- The reported results are from 4-bit inference rather than full-precision
  inference.
- The simple settings each use one fixed demonstration selection.
- The multi-run means summarize demonstration-selection variability, not
  cross-validation over test folds.
- Simple and multi-run scores should not be compared as if they used the same
  test records.

