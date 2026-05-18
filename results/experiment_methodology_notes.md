# Traceability Experiment Methodology Notes

## Task
Binary requirements traceability classification.

Input:
- Higher-level requirement
- Lower-level requirement

Output:
- trace
- no_trace

## Dataset
Cleaned dataset:
- Total examples: 1887
- trace: 757
- no_trace: 1130

## Evaluation setup
A stratified 10-fold evaluation setup was used.

For fine-tuned experiments:
1. One fold was held out as the final test set.
2. The remaining nine folds formed the training pool.
3. The training pool was split into inner_train and validation.
4. The model was trained only on inner_train or a balanced version of inner_train.
5. Validation was used for training-time monitoring/perplexity.
6. The test fold was used only after training for final classification evaluation.

## Leakage prevention
- Test fold was not used during training.
- Test fold was not used for validation or perplexity monitoring.
- Oversampling/undersampling was applied only to inner_train.
- Validation and test sets were not balanced or modified.
- Overlap checks were used between train, validation, and test splits.

## Completed experiments

### E0: Base Qwen3.5-9B no fine-tuning
The base model was evaluated directly on the same 10 held-out test folds.
No training was performed.

### E1: QLoRA with oversampling
Random oversampling with replacement was applied to the minority trace class in inner_train.

### E2: QLoRA with undersampling
Random undersampling without replacement was applied to the majority no_trace class in inner_train.

### E3: QLoRA with imbalanced training
The original imbalanced inner_train.jsonl was used directly.
No oversampling or undersampling was applied.

## Main finding
The base model performed poorly on trace detection. QLoRA fine-tuning significantly improved traceability classification. Among the fine-tuned settings, undersampling achieved the best trace recall and trace F1, while imbalanced training achieved high trace precision but lower trace recall.
