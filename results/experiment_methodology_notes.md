# Traceability Experiment Methodology Notes

## Task
Binary requirements traceability classification:
- Input: higher-level requirement + lower-level requirement
- Output: trace / no_trace

## Dataset
Cleaned dataset:
- Total examples: 1887
- trace: 757
- no_trace: 1130

## Evaluation setup
- Stratified 10-fold setup.
- For fine-tuning experiments:
  - One fold is held out as the final test fold.
  - Remaining nine folds form the training pool.
  - Training pool is split into inner_train and validation.
  - Training uses inner_train or balanced version of inner_train.
  - Validation is used for training-time monitoring/perplexity.
  - Test fold is used only after training for final classification evaluation.

## Leakage prevention
- Test fold is not used during training.
- Test fold is not used for validation/perplexity.
- Oversampling/undersampling is applied only to inner_train.
- Validation and test sets remain untouched.
- Overlap checks are performed between train, validation, and test.

## Experiments completed
1. Base Qwen3.5-9B no fine-tuning
2. Qwen3.5-9B QLoRA with oversampling
3. Qwen3.5-9B QLoRA with undersampling

## Balancing methods
- Oversampling: random oversampling with replacement of the minority trace class.
- Undersampling: random undersampling without replacement of the majority no_trace class.

## Main finding
The base model performs poorly on trace detection. QLoRA fine-tuning significantly improves trace recall and trace F1.
