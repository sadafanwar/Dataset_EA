# Few-shot Qwen Multi-run Prompting Experiment

## Experiment

Experiment name: fewshot_qwen35_multi_run  
Model: Qwen/Qwen3.5-9B  
Fine-tuning: No  
LoRA / QLoRA adapter: No  
Mode: prompt-based classification with generation-based prediction  

## Source dataset

`transformed_data/cross_dataset_pattern1.jsonl`

## Data folders

- `data/fewshot_qwen35_multi_run/`
- `results/fewshot_qwen35_multi_run/`

## Test set

Fixed test set:

- file: `data/fewshot_qwen35_multi_run/test_100.jsonl`
- seed: 42
- size: 100 examples
- balance: 50 trace + 50 no_trace

The same fixed test set is used for zero-shot, all 5-shot runs, and all 10-shot runs.

## Prompt-example runs

Run seeds:

`[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`

### Zero-shot

- run once
- no examples in prompt

### Five-shot

- 10 runs
- each run uses different prompt examples
- each run has 3 trace + 2 no_trace examples

### Ten-shot

- 10 runs
- each run uses different prompt examples
- each run has 5 trace + 5 no_trace examples

Within each run, 5-shot and 10-shot examples are disjoint.
Prompt examples do not overlap with the fixed test set.

## Prediction settings

- generation-based prediction
- max_new_tokens: 8
- do_sample: False
- temperature: deterministic generation; sampling disabled

## Output files

Per-run predictions:

- `zero_shot_predictions.csv`
- `five_shot_run_01_predictions.csv` ... `five_shot_run_10_predictions.csv`
- `ten_shot_run_01_predictions.csv` ... `ten_shot_run_10_predictions.csv`

Summaries:

- `fewshot_run_summary.csv`
- `fewshot_run_summary.json`
- `fewshot_aggregate_summary.csv`
- `fewshot_aggregate_summary.json`

## Purpose

This experiment checks whether few-shot traceability performance is stable when the prompt examples change while the test set stays fixed.
