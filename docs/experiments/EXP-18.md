# EXP-18: Qwen3.5 Label + Rationale, Multi-Template, 10-Fold CV

## Objective
EXP-18 tests whether five semantically equivalent instruction templates improve cross-level requirement traceability classification and rationale generation compared with EXP-17's single-template condition.

## Controlled comparison
Both experiments keep the same Qwen/Qwen3.5-9B model, QLoRA setup, 4-bit NF4 quantization, LoRA r=16/alpha=32/dropout=0.1, two epochs, batch size 1, gradient accumulation 8, learning rate 5e-5, max length 1024, seed 42, source records, outer folds, inner validation splits, target format, and deterministic generation.

## Dataset and templates
Source: `data/traceability_dataset/traceability_pairs_with_rationale.jsonl`

- Total: 1,887
- Trace: 757
- No trace: 1,130

EXP-18 reuses the frozen EXP-17 folds. Five templates are defined in `configs/exp18/templates_exp18.yaml`. Each `pair_id` receives one stable template using `SHA256(pair_id + NUL + seed) modulo 5`. The dataset is not expanded. `template_01` exactly matches EXP-17.

## Training and evaluation
Training scripts:
- `scripts/exp18/train_exp18.py`
- `scripts/exp18/run_exp18_fold.py`
- `scripts/exp18/run_exp18_all_folds.py`

Prompt and padding tokens are masked; loss is computed only on label, rationale, and EOS tokens.

Primary evaluation uses `scripts/exp18/evaluate_exp18.py` with `template_01` for direct paired comparison with EXP-17.

## Robustness evaluation
Secondary robustness evaluation uses `scripts/exp18/evaluate_exp18_robustness.py` across all five templates.

## Aggregation
`scripts/analysis/aggregate_exp17_exp18.py` produces fold-level metrics, mean and standard deviation, class-wise results, paired EXP-18 minus EXP-17 differences, descriptive paired effects, and robustness summaries.

Final command:
```bash
python scripts/analysis/aggregate_exp17_exp18.py --include-robustness
```

## Reproducibility rules
Do not modify frozen folds, use `--force`, use diagnostic limits for final runs, run folds in parallel on the single GPU, or treat robustness majority voting as the primary result. Preserve configurations, logs, adapters, predictions, metrics, and tracking files.

## Status
| Component | Status |
|---|---|
| Core code and orchestrator | Ready |
| Manifest and local validation | Ready |
| Technical documentation | Ready |
| Server deployment and diagnostic | Pending |
| Final 10-fold execution | Pending |
| Final aggregation | Pending results |
