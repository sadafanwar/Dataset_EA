# Fold 10 Recovery Note

During the Qwen3.5-9B QLoRA imbalanced 10-fold CV experiment, fold_10 evaluation initially failed due to CUDA out-of-memory.

The fold_10 model checkpoint already existed, so training was not rerun. After GPU memory became available, fold_10 evaluation was rerun successfully using the saved best_model.

Recovered files:
- results/qwen35_9b_qlora_unbalanced_10fold_cv_inner_val/fold_10/fold_10_metrics.json
- results/qwen35_9b_qlora_unbalanced_10fold_cv_inner_val/fold_10/fold_10_predictions.csv

Final fold_10 metrics:
- Accuracy: 0.9096
- Trace precision: 0.9265
- Trace recall: 0.8400
- Trace F1: 0.8811
- No-trace precision: 0.9000
- No-trace recall: 0.9558
- No-trace F1: 0.9270

The final summary and master experiment registry were rebuilt after successful fold_10 evaluation.
