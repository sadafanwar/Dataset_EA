# Ministral 8B QLoRA undersampling configuration

## Experiment identity

- Experiment ID: `mistral_8b_qlora_undersampling`
- Model: `mistralai/Ministral-3-8B-Instruct-2512-BF16`
- Frozen model revision: `f6fae9795746f63c9be8344932f01275f3c63734`
- Evaluation design: independent 10-fold cross-validation with an inner validation split in every fold
- Balancing: random undersampling without replacement on inner training data only

## Qwen EXP-07 provenance

The Qwen undersampling experiment is the authority. Oversampling and original-imbalance implementations are not used as the data treatment.

- Undersampling experiment and frozen outputs: commit `55c82a5`
- Training implementation: commit `198fc2f`
- Evaluation implementation: commit `a99a99f`
- Config generator: `scripts/create_qwen35_inner_val_undersampling_fold_config.py`
- Undersampling generator: `scripts/create_undersampled_inner_train_fold.py`

The committed EXP-07 configuration specifies a learning rate of `0.00005`.

## Frozen data protocol

For each fold, the existing `inner_train.jsonl` is the source and the existing `train_undersampled.jsonl` is used directly. No data file is regenerated.

- The majority `no_trace` class is sampled without replacement to the minority-class count.
- The seed is `42 + fold_number`, giving Fold 01 seed `43` through Fold 10 seed `52`.
- The retained majority records and unchanged minority records are shuffled with the same Python random generator.
- Folds 01-07 contain 1,226 training rows: 613 `trace` and 613 `no_trace`.
- Folds 08-10 contain 1,228 training rows: 614 `trace` and 614 `no_trace`.
- Undersampled training files contain no duplicate example IDs.
- Validation and test sets are unchanged and are never undersampled.
- Counts, labels, unique IDs, duplicate counts, normalized SHA-256 hashes, exact historical sampling reproduction, and cross-split non-overlap are checked before model loading.
- Every Mistral test fold must exactly match the corresponding frozen Qwen test fold.

## Prompt and loss

The format is the exact Pattern-1 plain-text format used by Qwen EXP-07:

```text
{instruction}

Higher-level requirement:
{higher_level_requirement}

Lower-level requirement:
{lower_level_requirement}

Answer: {trace|no_trace}
```

- No chat template is applied.
- Loss covers the full prompt and final label.
- Labels are `trace` and `no_trace`.

## Training values

| Setting | Value |
| --- | ---: |
| Learning rate | `0.00005` |
| Microbatch size | `1` |
| Gradient accumulation | `8` |
| Epochs | `2` |
| Warmup steps | `20` |
| Maximum gradient norm | `1.0` |
| Weight decay | `0.01` |
| Precision | BF16 compute |
| Maximum sequence length | `1024` |
| DataLoader shuffle | `false` |
| Optimizer | AdamW |
| Scheduler | Linear with warmup |
| Validation interval | Every 50 optimizer steps on the first 100 validation records |
| Checkpoint interval | Every 100 optimizer steps; retain at most 3 |
| Logging interval | Every 10 optimizer steps |

The historical implementation discards an incomplete gradient-accumulation group at each epoch boundary. Every fold therefore completes 153 optimizer steps per epoch and 306 across two epochs. The scheduler formula is `(microbatches * epochs) // 8`: 306 steps for Folds 01-07 and 307 for Folds 08-10.

## QLoRA and architecture-specific settings

- Quantization: 4-bit NF4, BF16 compute, double quantization enabled.
- LoRA rank: 16.
- LoRA alpha: 32.
- LoRA dropout: 0.1.
- Bias: none.
- Task type: causal language modeling.
- Ministral target regex: `.*language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)`.
- Expected adapters: 136 modules and 14,483,456 trainable parameters.

The target regex adapts only the Ministral text decoder's attention projections. Vision and projector modules are excluded. `fix_mistral_regex=true` and gradient checkpointing with `use_reentrant=false` are the architecture/runtime compatibility changes.

## Evaluation and preflight gate

- Training-time monitoring computes perplexity on the first 100 unchanged validation records.
- The best adapter is selected by lowest epoch training loss, matching EXP-07.
- Final held-out test prediction uses greedy decoding with `max_new_tokens=5`.
- Invalid generated labels are reported and excluded from classification metrics.
- Full 10-fold training refuses to start without a successful Fold 01 preflight report whose hyperparameters match the full configuration.
- The preflight loads the frozen model, attaches text-only adapters, runs one longest-sequence optimizer step, and records dataset counts, hyperparameters, trainable parameters, GPU memory, and errors.

Large adapters and checkpoints remain server-side under `mistral_traceability/outputs/mistral_8b_qlora_undersampling/`. Reproducibility files, reports, metrics, predictions, and logs remain under `mistral_traceability/`.
