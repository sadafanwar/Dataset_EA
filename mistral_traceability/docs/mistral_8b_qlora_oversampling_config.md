# Ministral 8B QLoRA oversampling configuration

## Experiment identity

- Experiment ID: `mistral_8b_qlora_oversampling`
- Model: `mistralai/Ministral-3-8B-Instruct-2512-BF16`
- Frozen model revision: `f6fae9795746f63c9be8344932f01275f3c63734`
- Evaluation design: independent 10-fold cross-validation with an inner validation split in every fold
- Balancing: random oversampling with replacement on inner training data only

## Qwen EXP-06 provenance

The corrected Qwen oversampling experiment, not the undersampling experiment, is the authority:

- Corrected split/config and oversampling implementation: commit `c3ae63b`
- Training implementation: commit `198fc2f`
- Forced-choice evaluation implementation: commit `8ee202f`
- Config generator: `scripts/create_qwen35_inner_val_fold_config.py`
- Oversampling generator: `scripts/create_oversampled_inner_train_fold.py`

The corrected committed EXP-06 configuration specifies a learning rate of `0.00005`. This value takes precedence over the earlier summary-table value of `0.0001`.

## Frozen data protocol

For each fold, the existing `inner_train.jsonl` is the source. The existing `train_oversampled.jsonl` is used directly and is never regenerated.

- The minority `trace` class is sampled with replacement to the majority-class count.
- The seed is `42 + fold_number`, giving Fold 01 seed `43` through Fold 10 seed `52`.
- The resampled records and unchanged majority records are shuffled using the same Python random generator.
- Every training fold contains 1,830 rows: 915 `trace` and 915 `no_trace`.
- Duplicate training rows are expected because replacement sampling is the treatment.
- Validation and test sets are not oversampled.
- All file counts, label counts, unique-ID counts, duplicate counts, normalized SHA-256 hashes, exact resampling reproduction, and cross-split non-overlap are checked before model loading.
- The Mistral test file for each fold must exactly match the corresponding frozen Qwen test fold.

## Prompt and loss

The format is the exact Pattern-1 plain-text format used by the Qwen experiment:

```text
{instruction}

Higher-level requirement:
{higher_level_requirement}

Lower-level requirement:
{lower_level_requirement}

Answer: {trace|no_trace}
```

- No chat template is applied.
- Loss covers the full prompt and final label, matching the historical training implementation.
- Labels are `trace` and `no_trace`.

## Training values

| Setting | Value | Reason |
| --- | ---: | --- |
| Learning rate | `0.00005` | Corrected EXP-06 config |
| Microbatch size | `1` | Corrected EXP-06 config |
| Gradient accumulation | `8` | Effective batch size 8 |
| Epochs | `2` | Corrected EXP-06 config |
| Warmup steps | `20` | Corrected EXP-06 config |
| Maximum gradient norm | `1.0` | Corrected EXP-06 config |
| Weight decay | `0.01` | Corrected EXP-06 config |
| Precision | BF16 compute | Corrected EXP-06 config and RTX 4090 compatibility |
| Maximum sequence length | `1024` | Corrected EXP-06 config |
| DataLoader shuffle | `false` | Historical loader default; oversampled files are already shuffled |
| Optimizer | AdamW | Historical implementation |
| Scheduler | Linear with warmup | Historical implementation |
| Validation interval | Every 50 optimizer steps | First 100 validation records |
| Checkpoint interval | Every 100 optimizer steps | Retain at most 3 periodic checkpoints |
| Logging interval | Every 10 optimizer steps | Historical configuration |

EXP-06 discarded an incomplete gradient-accumulation group at the end of each epoch. With 1,830 microbatches and accumulation 8, this produces 228 completed optimizer steps per epoch and discards the last 6 partial microbatches. Across two epochs, 456 optimizer steps complete while the historical scheduler formula configures 457 total steps. This behavior is intentionally preserved for comparability.

## QLoRA and architecture-specific settings

- Quantization: 4-bit NF4, BF16 compute, double quantization enabled.
- LoRA rank: 16.
- LoRA alpha: 32.
- LoRA dropout: 0.1.
- Bias: none.
- Task type: causal language modeling.
- Mistral target regex: `.*language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)`.
- Expected adapters: 136 modules and 14,483,456 trainable parameters.

The Mistral target regex adapts only text-decoder attention projections. Vision and projector modules are excluded. `fix_mistral_regex=true` is required by the tokenizer, and gradient checkpointing uses `use_reentrant=false` for compatibility with the installed PEFT/PyTorch stack.

## Evaluation and preflight gate

- Training-time monitoring computes perplexity on the first 100 unchanged validation records.
- The best adapter is selected by lowest epoch training loss, matching EXP-06.
- Final held-out test prediction is greedy with `max_new_tokens=5`.
- Invalid generated labels are reported and excluded from classification metrics, matching the existing evaluation protocol.
- The full 10-fold runner refuses to start without a successful Fold 01 preflight report whose hyperparameters match the full configuration.
- The preflight loads the frozen model, applies the Mistral-specific adapters, runs one longest-sequence optimizer step, and records dataset counts, hyperparameters, trainable parameters, GPU memory, and errors.

Large adapters and checkpoints remain server-side under `mistral_traceability/outputs/mistral_8b_qlora_oversampling/`. Reproducibility files, reports, metrics, predictions, and logs remain under `mistral_traceability/`.
