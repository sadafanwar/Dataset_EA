# Mistral Requirements Traceability Experiment Plan

## Objective

Evaluate whether findings previously observed with Qwen3.5-9B generalize to another open-weight LLM family.

## Task

Binary requirements traceability classification.

Labels:
- trace
- no_trace

## Research Questions

### RQ1
How does fine-tuning affect the performance of open-weight LLMs for requirements traceability?

Comparison:
M-E1 vs M-E4

### RQ2
How does few-shot prompting affect the performance of open-weight LLMs for requirements traceability?

Comparison:
M-E1 vs M-E2 vs M-E3

### RQ3
How do class imbalance and balancing strategy affect the performance of fine-tuned open-weight LLMs for requirements traceability?

Comparison:
M-E4 vs M-E5 vs M-E6

## Experiments

- M-E1: Zero-shot baseline
- M-E2: 5-shot prompting
- M-E3: 10-shot prompting
- M-E4: QLoRA using original class distribution
- M-E5: QLoRA with oversampling
- M-E6: QLoRA with undersampling

## Experimental Controls

The following should remain fixed wherever applicable:

- Same cleaned traceability corpus
- Same label mapping
- Same frozen outer folds
- Same evaluation examples
- Same prompt semantics
- Same output parser
- Same evaluation metrics
- Same random seeds where applicable

## Primary Metrics

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Weighted F1
- Trace Precision
- Trace Recall
- Trace F1
- No-trace Precision
- No-trace Recall
- No-trace F1
- Format-valid rate

## Reproducibility Requirements

For every experiment record:

- model ID
- model revision
- tokenizer revision
- dataset hash
- fold manifest
- configuration
- random seed
- Git commit
- Python version
- PyTorch version
- Transformers version
- PEFT version
- bitsandbytes version
- CUDA version
- GPU model
- start/end time
- runtime
- raw predictions
- fold-level metrics
- aggregate metrics
- errors/anomalies
