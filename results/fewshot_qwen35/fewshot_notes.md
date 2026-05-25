# Few-shot Qwen Traceability Experiment

## Experiment

Model: Qwen/Qwen3.5-9B  
Fine-tuning: No  
LoRA / QLoRA adapter: No  
Mode: few-shot prompting with generation-based prediction  

## Source dataset

`transformed_data/cross_dataset_pattern1.jsonl`

## Created data files

- `data/fewshot_qwen35/example_bank.jsonl`
- `data/fewshot_qwen35/test_100.jsonl`
- `data/fewshot_qwen35/five_shot_examples.jsonl`
- `data/fewshot_qwen35/ten_shot_examples.jsonl`

## Setup

Seed: 42  
Test set: 100 balanced examples, 50 trace + 50 no_trace  
5-shot: 3 trace + 2 no_trace examples  
10-shot: 5 trace + 5 no_trace examples  

## Prediction

Generation-based prediction.

Generation settings:

- max_new_tokens: 8
- do_sample: False
- temperature: not sampled / deterministic generation

## Labels

- `trace`
- `no_trace`

## Notes

This experiment does not train or fine-tune the model. The examples are included only inside the prompt.
