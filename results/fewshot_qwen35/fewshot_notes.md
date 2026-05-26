# Few-shot Qwen Traceability Experiment

## Experiment

Model: Qwen/Qwen3.5-9B  
Fine-tuning: No  
LoRA / QLoRA adapter: No  
Mode: prompt-based classification with generation-based prediction  

## Source dataset

`transformed_data/cross_dataset_pattern1.jsonl`

## Created data files

- `data/fewshot_qwen35/zero_shot_examples.jsonl`
- `data/fewshot_qwen35/five_shot_examples.jsonl`
- `data/fewshot_qwen35/ten_shot_examples.jsonl`
- `data/fewshot_qwen35/test_100.jsonl`

No `example_bank.jsonl` is used.

## Setup

Seed: 42  
Test set: 100 balanced examples, 50 trace + 50 no_trace  

Prompt settings:

- 0-shot: instructions only, no examples
- 5-shot: instructions + 5 examples
  - 3 trace
  - 2 no_trace
- 10-shot: instructions + 10 examples
  - 5 trace
  - 5 no_trace

The 5-shot and 10-shot examples are disjoint.
Prompt examples do not overlap with the test set.

## Prediction settings

- max_new_tokens: 8
- do_sample: False
- temperature: deterministic generation; sampling disabled

## Labels

- `trace`
- `no_trace`

This experiment does not train or fine-tune the model.
The examples are included only inside the prompt.
