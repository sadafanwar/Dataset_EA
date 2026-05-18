#!/usr/bin/env python3

import json
import csv
from pathlib import Path
from collections import Counter


def read_jsonl_count(path):
    if path is None or not Path(path).exists():
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
                r = json.loads(line)
                label = r.get("output", "")
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
        "config_template": "",
        "output_template": "",
        "train_template": "",
        "validation_template": "",
        "notes": "Base model evaluated directly on the same 10 held-out test folds. No training was performed.",
    },
    {
        "experiment_id": "E1",
        "experiment_name": "Qwen3.5-9B QLoRA with oversampling",
        "model": "Qwen/Qwen3.5-9B",
        "fine_tuning": "Yes",
        "method": "QLoRA",
        "balancing": "random oversampling with replacement",
        "results_dir": "results/qwen35_9b_qlora_oversampling_10fold_cv_inner_val",
        "config_template": "src/integration/training_scripts/ft/config_qwen35_9b_10fold_cv_inner_val_fold_{fold}.yaml",
        "output_template": "outputs/qwen35_9b_qlora_oversampling_10fold_cv_inner_val/fold_{fold}",
        "train_template": "data/qwen35_oversampling_10fold_cv/inner_splits/fold_{fold}/train_oversampled.jsonl",
        "validation_template": "data/qwen35_oversampling_10fold_cv/inner_splits/fold_{fold}/validation.jsonl",
        "notes": "Minority trace class was duplicated only in inner_train. Validation and test remained untouched.",
    },
    {
        "experiment_id": "E2",
        "experiment_name": "Qwen3.5-9B QLoRA with undersampling",
        "model": "Qwen/Qwen3.5-9B",
        "fine_tuning": "Yes",
        "method": "QLoRA",
        "balancing": "random undersampling without replacement",
        "results_dir": "results/qwen35_9b_qlora_undersampling_10fold_cv_inner_val",
        "config_template": "src/integration/training_scripts/ft/config_qwen35_9b_10fold_cv_inner_val_undersampling_fold_{fold}.yaml",
        "output_template": "outputs/qwen35_9b_qlora_undersampling_10fold_cv_inner_val/fold_{fold}",
        "train_template": "data/qwen35_oversampling_10fold_cv/inner_splits/fold_{fold}/train_undersampled.jsonl",
        "validation_template": "data/qwen35_oversampling_10fold_cv/inner_splits/fold_{fold}/validation.jsonl",
        "notes": "Majority no_trace class was reduced only in inner_train. Validation and test remained untouched.",
    },
]


def main():
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
                m = json.load(f)

            train_file = exp["train_template"].format(fold=fold) if exp["train_template"] else ""
            validation_file = exp["validation_template"].format(fold=fold) if exp["validation_template"] else ""
            config_file = exp["config_template"].format(fold=fold) if exp["config_template"] else ""
            output_dir = exp["output_template"].format(fold=fold) if exp["output_template"] else ""

            train_counts = read_jsonl_count(train_file) if train_file else {"size": "", "trace": "", "no_trace": ""}
            val_counts = read_jsonl_count(validation_file) if validation_file else {"size": "", "trace": "", "no_trace": ""}
            test_counts = read_jsonl_count(test_file)

            row = {
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
                "validation_size": val_counts["size"],
                "validation_trace": val_counts["trace"],
                "validation_no_trace": val_counts["no_trace"],
                "test_size": test_counts["size"],
                "test_trace": test_counts["trace"],
                "test_no_trace": test_counts["no_trace"],
                "accuracy": m.get("accuracy", ""),
                "trace_precision": m.get("trace_precision", ""),
                "trace_recall": m.get("trace_recall", ""),
                "trace_f1": m.get("trace_f1", ""),
                "no_trace_precision": m.get("no_trace_precision", ""),
                "no_trace_recall": m.get("no_trace_recall", ""),
                "no_trace_f1": m.get("no_trace_f1", ""),
                "confusion_matrix": json.dumps(m.get("confusion_matrix", "")),
                "valid_predictions": m.get("valid_predictions", ""),
                "invalid_predictions": m.get("invalid_predictions", ""),
                "metrics_file": str(metrics_file),
                "predictions_file": str(predictions_file),
                "config_file": config_file,
                "output_dir": output_dir,
                "notes": exp["notes"],
            }

            rows.append(row)

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

    print(f"Created registry: {out_path}")
    print(f"Rows written: {len(rows)}")


if __name__ == "__main__":
    main()
