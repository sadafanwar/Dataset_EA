#!/usr/bin/env python3

import json
import csv
from pathlib import Path
from collections import Counter


def read_jsonl_count(path):
    if not path or not Path(path).exists():
        return {
            "size": "",
            "trace": "",
            "no_trace": "",
        }

    counts = Counter()
    size = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                label = row.get("output", "")
                counts[label] += 1
                size += 1

    return {
        "size": size,
        "trace": counts.get("trace", 0),
        "no_trace": counts.get("no_trace", 0),
    }


EXPERIMENTS = [
    {
        "experiment_id": "E0",
        "experiment_name": "Base Qwen3.5-9B no fine-tuning",
        "model": "Qwen/Qwen3.5-9B",
        "fine_tuning": "No",
        "method": "zero-shot / no fine-tuning",
        "balancing": "none",
        "results_dir": "results/qwen35_9b_base_10fold_cv_no_ft",
        "summary_file": "results/qwen35_9b_base_10fold_cv_no_ft/qwen35_9b_base_10fold_cv_no_ft_summary.json",
        "config_template": "",
        "output_template": "",
        "train_template": "",
        "validation_template": "",
        "notes": "Base model evaluated directly on the same 10 held-out test folds. No training was performed.",
        "comment": "Weak baseline. Mostly predicts no_trace and misses most trace links.",
    },
    {
        "experiment_id": "E1",
        "experiment_name": "Qwen3.5-9B QLoRA with oversampling",
        "model": "Qwen/Qwen3.5-9B",
        "fine_tuning": "Yes",
        "method": "QLoRA",
        "balancing": "random oversampling with replacement",
        "results_dir": "results/qwen35_9b_qlora_oversampling_10fold_cv_inner_val",
        "summary_file": "results/qwen35_9b_qlora_oversampling_10fold_cv_inner_val/qwen35_9b_qlora_10fold_cv_summary.json",
        "config_template": "src/integration/training_scripts/ft/config_qwen35_9b_10fold_cv_inner_val_fold_{fold}.yaml",
        "output_template": "outputs/qwen35_9b_qlora_oversampling_10fold_cv_inner_val/fold_{fold}",
        "train_template": "data/qwen35_oversampling_10fold_cv/inner_splits/fold_{fold}/train_oversampled.jsonl",
        "validation_template": "data/qwen35_oversampling_10fold_cv/inner_splits/fold_{fold}/validation.jsonl",
        "notes": "Minority trace class was duplicated only in inner_train. Validation and test remained untouched.",
        "comment": "Strong result. Best no-trace F1 and strong overall accuracy.",
    },
    {
        "experiment_id": "E2",
        "experiment_name": "Qwen3.5-9B QLoRA with undersampling",
        "model": "Qwen/Qwen3.5-9B",
        "fine_tuning": "Yes",
        "method": "QLoRA",
        "balancing": "random undersampling without replacement",
        "results_dir": "results/qwen35_9b_qlora_undersampling_10fold_cv_inner_val",
        "summary_file": "results/qwen35_9b_qlora_undersampling_10fold_cv_inner_val/qwen35_9b_qlora_undersampling_10fold_cv_summary.json",
        "config_template": "src/integration/training_scripts/ft/config_qwen35_9b_10fold_cv_inner_val_undersampling_fold_{fold}.yaml",
        "output_template": "outputs/qwen35_9b_qlora_undersampling_10fold_cv_inner_val/fold_{fold}",
        "train_template": "data/qwen35_oversampling_10fold_cv/inner_splits/fold_{fold}/train_undersampled.jsonl",
        "validation_template": "data/qwen35_oversampling_10fold_cv/inner_splits/fold_{fold}/validation.jsonl",
        "notes": "Majority no_trace class was reduced only in inner_train. Validation and test remained untouched.",
        "comment": "Best trace recall and best trace F1. Most useful when missing trace links is costly.",
    },
    {
        "experiment_id": "E3",
        "experiment_name": "Qwen3.5-9B QLoRA with imbalanced training",
        "model": "Qwen/Qwen3.5-9B",
        "fine_tuning": "Yes",
        "method": "QLoRA",
        "balancing": "none / original imbalanced inner_train",
        "results_dir": "results/qwen35_9b_qlora_unbalanced_10fold_cv_inner_val",
        "summary_file": "results/qwen35_9b_qlora_unbalanced_10fold_cv_inner_val/qwen35_9b_qlora_unbalanced_10fold_cv_summary.json",
        "config_template": "src/integration/training_scripts/ft/config_qwen35_9b_10fold_cv_inner_val_unbalanced_fold_{fold}.yaml",
        "output_template": "outputs/qwen35_9b_qlora_unbalanced_10fold_cv_inner_val/fold_{fold}",
        "train_template": "data/qwen35_oversampling_10fold_cv/inner_splits/fold_{fold}/inner_train.jsonl",
        "validation_template": "data/qwen35_oversampling_10fold_cv/inner_splits/fold_{fold}/validation.jsonl",
        "notes": "Original imbalanced inner_train.jsonl was used without oversampling or undersampling. Validation and test remained untouched.",
        "comment": "High trace precision, but lower trace recall. More conservative and misses more true trace links.",
    },
]


def build_registry():
    out_path = Path("results/experiment_registry.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for exp in EXPERIMENTS:
        results_dir = Path(exp["results_dir"])

        for i in range(1, 11):
            fold = f"{i:02d}"
            fold_name = f"fold_{fold}"

            metrics_file = results_dir / fold_name / f"{fold_name}_metrics.json"
            predictions_file = results_dir / fold_name / f"{fold_name}_predictions.csv"
            test_file = Path(f"data/qwen35_oversampling_10fold_cv/folds/{fold_name}/test.jsonl")

            if not metrics_file.exists():
                print(f"Skipping missing metrics: {metrics_file}")
                continue

            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics = json.load(f)

            train_file = exp["train_template"].format(fold=fold) if exp["train_template"] else ""
            validation_file = exp["validation_template"].format(fold=fold) if exp["validation_template"] else ""
            config_file = exp["config_template"].format(fold=fold) if exp["config_template"] else ""
            output_dir = exp["output_template"].format(fold=fold) if exp["output_template"] else ""

            train_counts = read_jsonl_count(train_file)
            validation_counts = read_jsonl_count(validation_file)
            test_counts = read_jsonl_count(test_file)

            rows.append({
                "experiment_id": exp["experiment_id"],
                "experiment_name": exp["experiment_name"],
                "model": exp["model"],
                "fine_tuning": exp["fine_tuning"],
                "method": exp["method"],
                "balancing": exp["balancing"],
                "fold": fold_name,

                "train_file": train_file,
                "validation_file": validation_file,
                "test_file": str(test_file),

                "train_size": train_counts["size"],
                "train_trace": train_counts["trace"],
                "train_no_trace": train_counts["no_trace"],

                "validation_size": validation_counts["size"],
                "validation_trace": validation_counts["trace"],
                "validation_no_trace": validation_counts["no_trace"],

                "test_size": test_counts["size"],
                "test_trace": test_counts["trace"],
                "test_no_trace": test_counts["no_trace"],

                "accuracy": metrics.get("accuracy", ""),
                "trace_precision": metrics.get("trace_precision", ""),
                "trace_recall": metrics.get("trace_recall", ""),
                "trace_f1": metrics.get("trace_f1", ""),
                "no_trace_precision": metrics.get("no_trace_precision", ""),
                "no_trace_recall": metrics.get("no_trace_recall", ""),
                "no_trace_f1": metrics.get("no_trace_f1", ""),
                "confusion_matrix": json.dumps(metrics.get("confusion_matrix", "")),
                "valid_predictions": metrics.get("valid_predictions", ""),
                "invalid_predictions": metrics.get("invalid_predictions", ""),

                "metrics_file": str(metrics_file),
                "predictions_file": str(predictions_file),
                "config_file": config_file,
                "output_dir": output_dir,

                "notes": exp["notes"],
            })

    fieldnames = [
        "experiment_id",
        "experiment_name",
        "model",
        "fine_tuning",
        "method",
        "balancing",
        "fold",
        "train_file",
        "validation_file",
        "test_file",
        "train_size",
        "train_trace",
        "train_no_trace",
        "validation_size",
        "validation_trace",
        "validation_no_trace",
        "test_size",
        "test_trace",
        "test_no_trace",
        "accuracy",
        "trace_precision",
        "trace_recall",
        "trace_f1",
        "no_trace_precision",
        "no_trace_recall",
        "no_trace_f1",
        "confusion_matrix",
        "valid_predictions",
        "invalid_predictions",
        "metrics_file",
        "predictions_file",
        "config_file",
        "output_dir",
        "notes",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created fold-level registry: {out_path}")
    print(f"Rows written: {len(rows)}")


def build_summary_table():
    out_path = Path("results/experiment_summary_table.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for exp in EXPERIMENTS:
        summary_file = Path(exp["summary_file"])

        if not summary_file.exists():
            print(f"Skipping missing summary: {summary_file}")
            continue

        with summary_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        summary = data["summary"]

        rows.append({
            "experiment_id": exp["experiment_id"],
            "experiment_name": exp["experiment_name"],
            "model": exp["model"],
            "fine_tuning": exp["fine_tuning"],
            "method": exp["method"],
            "balancing": exp["balancing"],
            "accuracy": round(summary["accuracy"]["mean"], 4),
            "trace_precision": round(summary["trace_precision"]["mean"], 4),
            "trace_recall": round(summary["trace_recall"]["mean"], 4),
            "trace_f1": round(summary["trace_f1"]["mean"], 4),
            "no_trace_precision": round(summary["no_trace_precision"]["mean"], 4),
            "no_trace_recall": round(summary["no_trace_recall"]["mean"], 4),
            "no_trace_f1": round(summary["no_trace_f1"]["mean"], 4),
            "comment": exp["comment"],
            "summary_file": str(summary_file),
        })

    fieldnames = [
        "experiment_id",
        "experiment_name",
        "model",
        "fine_tuning",
        "method",
        "balancing",
        "accuracy",
        "trace_precision",
        "trace_recall",
        "trace_f1",
        "no_trace_precision",
        "no_trace_recall",
        "no_trace_f1",
        "comment",
        "summary_file",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created experiment-level summary: {out_path}")
    print(f"Rows written: {len(rows)}")


def build_methodology_notes():
    out_path = Path("results/experiment_methodology_notes.md")

    text = """# Traceability Experiment Methodology Notes

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
"""

    out_path.write_text(text, encoding="utf-8")
    print(f"Created methodology notes: {out_path}")


def main():
    build_registry()
    build_summary_table()
    build_methodology_notes()


if __name__ == "__main__":
    main()
