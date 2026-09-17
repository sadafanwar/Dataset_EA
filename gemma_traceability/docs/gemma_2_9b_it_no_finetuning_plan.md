# Gemma 2 9B Instruct — No-Fine-Tuning Traceability Plan

## Objective

Evaluate the frozen `google/gemma-2-9b-it` base checkpoint on requirements
traceability classification without parameter updates.

## Frozen model

- Model ID: `google/gemma-2-9b-it`
- Revision: `11c9b309abf73637e4b6f9a3fa1e92e615547819`
- Server cache: `/data/caches/huggingface/hub`
- Inference representation: NF4 4-bit with BF16 computation and double
  quantization, matching the historical Qwen prompting implementation where
  architecture-compatible.
- Fine-tuning: none.
- All parameter `requires_grad` flags are explicitly disabled after model load;
  the audited trainable-parameter count must be zero.

## Evaluation design

The data and experiment layout reproduce the corrected historical Qwen
prompting protocol.

| Setting | Demonstrations | Runs | Test records |
| --- | ---: | ---: | ---: |
| Simple zero-shot | 0 | 1 | 100 |
| Simple 5-shot | 5 (3 trace, 2 no-trace) | 1 | 100 |
| Simple 10-shot | 10 (5 trace, 5 no-trace) | 1 | 100 |
| Multi-run 5-shot | 5 per run | 10 | 100 per run |
| Multi-run 10-shot | 10 per run | 10 | 100 per run |

The simple and multi-run protocols retain their respective historical fixed
test sets. Every copied JSONL file must match its Qwen source file byte for
byte. Prompt examples must not overlap the associated test set; within each
multi-run, the 5-shot and 10-shot examples must also remain disjoint.

## Prompt and generation protocol

- The English task content and label definitions are identical to the
  corrected Qwen prompting implementation.
- Allowed labels are `trace` and `no_trace`.
- The task content is wrapped using Gemma's checkpoint-provided chat template.
  This is the model-specific compatibility adaptation; the classification
  content itself is unchanged.
- Deterministic greedy generation: `do_sample=false`.
- Maximum generated tokens: 8.
- Batch size: 1.
- Seed: 42 for the simple evaluation.
- Multi-run demonstration seeds: 1 through 10.
- Prompt limit: at most 8192 tokens, reduced automatically to reserve the 8
  generation tokens if the checkpoint context limit requires it.

## Prediction parsing and metrics

The strict corrected parser accepts a label only when the response begins with
`trace`, `no_trace`, or `no trace`. Other outputs are recorded as invalid.
Invalid outputs are retained in prediction files but excluded from the binary
classification metrics, matching the historical protocol.

Reported values:

- Accuracy
- Trace precision, recall, and F1
- No-trace precision, recall, and F1
- Confusion matrix in `[trace, no_trace]` order
- Valid and invalid prediction counts
- Multi-run mean and sample standard deviation

## Required preflight

Before full evaluation, the preflight must:

1. Verify all 27 frozen data files against the historical Qwen files.
2. Verify all required non-overlap conditions.
3. Load the exact cached model revision in 4-bit mode.
4. Record package versions, model architecture, parameter count, and GPU
   memory.
5. Generate one zero-shot, one 5-shot, and one 10-shot prediction.
6. Write a structured report below
   `gemma_traceability/results/gemma_2_9b_it_no_finetuning/preflight/`.

The full evaluation must not start until this preflight has been reviewed.
