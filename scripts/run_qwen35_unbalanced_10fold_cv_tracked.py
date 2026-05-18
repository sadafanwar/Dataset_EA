#!/usr/bin/env python3

import argparse
import csv
import json
import statistics as stats
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter


EXPERIMENT_NAME = "qwen35_9b_qlora_unbalanced_10fold_cv_inner_val"
RESULT_BASE = Path("results") / EXPERIMENT_NAME
OUTPUT_BASE = Path("outputs") / EXPERIMENT_NAME
DATA_BASE = Path("data/qwen35_oversampling_10fold_cv")

MODEL_NAME = "Qwen/Qwen3.5-9B"
METHOD = "QLoRA"
BALANCING = "none_original_imbalanced_inner_train"


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def parse_time(t):
    return datetime.fromisoformat(t)


def duration_minutes(start_iso, end_iso):
    start = parse_time(start_iso)
    end = parse_time(end_iso)
    return round((end - start).total_seconds() / 60, 3)


def fold_name(fold):
    return f"fold_{fold:02d}"


def run(cmd):
    print("\n" + "=" * 100)
    print("RUNNING:")
    print(" ".join(cmd))
    print("=" * 100)
    subprocess.run(cmd, check=True)


def read_jsonl_counts(path):
    path = Path(path)
    counts = Counter()
    total = 0

    if not path.exists():
        return {
            "total": "",
            "trace": "",
            "no_trace": "",
        }

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                label = str(row.get("output", "")).strip()
                counts[label] += 1
                total += 1

    return {
        "total": total,
        "trace": counts.get("trace", 0),
        "no_trace": counts.get("no_trace", 0),
    }


def ensure_registry_header(path):
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        return

    fieldnames = registry_fields()

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def registry_fields():
    return [
        "experiment_name",
        "fold",
        "status",
        "start_time_utc",
        "end_time_utc",
        "duration_minutes",
        "model",
        "method",
        "balancing",
        "train_file",
        "validation_file",
        "test_file",
        "train_total",
        "train_trace",
        "train_no_trace",
        "validation_total",
        "validation_trace",
        "validation_no_trace",
        "test_total",
        "test_trace",
        "test_no_trace",
        "config_file",
        "output_dir",
        "best_model_dir",
        "metrics_file",
        "predictions_file",
        "accuracy",
        "trace_precision",
        "trace_recall",
        "trace_f1",
        "no_trace_precision",
        "no_trace_recall",
        "no_trace_f1",
        "valid_predictions",
        "invalid_predictions",
        "confusion_matrix",
        "notes",
    ]


def append_registry(row):
    registry_path = RESULT_BASE / "run_registry.csv"
    ensure_registry_header(registry_path)

    fieldnames = registry_fields()

    with registry_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def append_progress(fold, status, metrics_path, start_time, end_time):
    progress_path = RESULT_BASE / "progress.csv"
    RESULT_BASE.mkdir(parents=True, exist_ok=True)

    if not progress_path.exists():
        with progress_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["fold", "status", "start_time_utc", "end_time_utc", "duration_minutes", "metrics_path"])

    with progress_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            fold,
            status,
            start_time,
            end_time,
            duration_minutes(start_time, end_time),
            metrics_path,
        ])


def create_summary():
    metric_keys = [
        "accuracy",
        "trace_precision",
        "trace_recall",
        "trace_f1",
        "no_trace_precision",
        "no_trace_recall",
        "no_trace_f1",
    ]

    rows = []
    summed_cm = [[0, 0], [0, 0]]

    for i in range(1, 11):
        name = fold_name(i)
        path = RESULT_BASE / name / f"{name}_metrics.json"

        if not path.exists():
            print(f"Summary not created yet. Missing: {path}")
            return

        with path.open("r", encoding="utf-8") as f:
            m = json.load(f)

        row = {"fold": name}
        for k in metric_keys:
            row[k] = float(m[k])
        rows.append(row)

        cm = m.get("confusion_matrix")
        if cm:
            summed_cm[0][0] += cm[0][0]
            summed_cm[0][1] += cm[0][1]
            summed_cm[1][0] += cm[1][0]
            summed_cm[1][1] += cm[1][1]

    summary = {}
    for k in metric_keys:
        vals = [r[k] for r in rows]
        summary[k] = {
            "mean": stats.mean(vals),
            "std": stats.stdev(vals),
            "min": min(vals),
            "max": max(vals),
        }

    output = {
        "experiment": "Qwen3.5-9B QLoRA 10-fold CV with inner validation and no class balancing",
        "model": MODEL_NAME,
        "method": METHOD,
        "balancing": BALANCING,
        "folds": rows,
        "summary": summary,
        "summed_confusion_matrix_labels": ["trace", "no_trace"],
        "summed_confusion_matrix": summed_cm,
        "created_at_utc": now_iso(),
    }

    RESULT_BASE.mkdir(parents=True, exist_ok=True)

    json_path = RESULT_BASE / "qwen35_9b_qlora_unbalanced_10fold_cv_summary.json"
    csv_path = RESULT_BASE / "qwen35_9b_qlora_unbalanced_10fold_cv_summary.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std", "min", "max"])
        for k in metric_keys:
            writer.writerow([k, summary[k]["mean"], summary[k]["std"], summary[k]["min"], summary[k]["max"]])

    print("\nPer-fold results")
    print("-" * 110)
    print(f"{'Fold':<8} {'Acc':>8} {'Tr_P':>8} {'Tr_R':>8} {'Tr_F1':>8} {'NoTr_P':>8} {'NoTr_R':>8} {'NoTr_F1':>8}")

    for r in rows:
        print(
            f"{r['fold']:<8} "
            f"{r['accuracy']:>8.4f} "
            f"{r['trace_precision']:>8.4f} "
            f"{r['trace_recall']:>8.4f} "
            f"{r['trace_f1']:>8.4f} "
            f"{r['no_trace_precision']:>8.4f} "
            f"{r['no_trace_recall']:>8.4f} "
            f"{r['no_trace_f1']:>8.4f}"
        )

    print("\n10-fold mean ± std")
    print("-" * 70)
    for k in metric_keys:
        print(f"{k:<20}: {summary[k]['mean']:.4f} ± {summary[k]['std']:.4f}")

    print("\nSummed confusion matrix labels: ['trace', 'no_trace']")
    print(summed_cm)

    print(f"\nSaved summary JSON to: {json_path}")
    print(f"Saved summary CSV to: {csv_path}")


def run_fold(fold):
    name = fold_name(fold)

    train_file = DATA_BASE / "inner_splits" / name / "inner_train.jsonl"
    validation_file = DATA_BASE / "inner_splits" / name / "validation.jsonl"
    test_file = DATA_BASE / "folds" / name / "test.jsonl"

    config_file = Path(
        f"src/integration/training_scripts/ft/"
        f"config_qwen35_9b_10fold_cv_inner_val_unbalanced_{name}.yaml"
    )

    output_dir = OUTPUT_BASE / name
    best_model_dir = output_dir / "best_model"
    result_dir = RESULT_BASE / name
    metrics_file = result_dir / f"{name}_metrics.json"
    predictions_file = result_dir / f"{name}_predictions.csv"

    if metrics_file.exists():
        print(f"{name} already completed. Metrics found: {metrics_file}")
        print("Skipping this fold. Use --force to rerun from CLI.")
        return "skipped"

    print("\n" + "#" * 100)
    print(f"STARTING {name}")
    print("#" * 100)

    start_time = now_iso()
    status = "started"

    try:
        run(["python", "scripts/create_inner_train_validation_split.py", "--fold", str(fold)])

        run(["python", "scripts/create_qwen35_inner_val_unbalanced_fold_config.py", "--fold", str(fold)])

        output_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

        train_counts = read_jsonl_counts(train_file)
        val_counts = read_jsonl_counts(validation_file)
        test_counts = read_jsonl_counts(test_file)

        print("\nDataset counts")
        print("Train:", train_counts)
        print("Validation:", val_counts)
        print("Test:", test_counts)

        run([
            "python",
            "src/integration/training_scripts/ft/run_ft.py",
            "--config",
            str(config_file),
            "--method",
            "qlora",
        ])

        if not best_model_dir.exists():
            raise FileNotFoundError(f"best_model not found after training: {best_model_dir}")

        run([
            "python",
            "scripts/evaluate_qwen35_fold_classification.py",
            "--model_dir",
            str(best_model_dir),
            "--test_file",
            str(test_file),
            "--output_dir",
            str(result_dir),
        ])

        if not metrics_file.exists():
            raise FileNotFoundError(f"Metrics file not found after evaluation: {metrics_file}")

        with metrics_file.open("r", encoding="utf-8") as f:
            m = json.load(f)

        end_time = now_iso()
        status = "completed"

        row = {
            "experiment_name": EXPERIMENT_NAME,
            "fold": name,
            "status": status,
            "start_time_utc": start_time,
            "end_time_utc": end_time,
            "duration_minutes": duration_minutes(start_time, end_time),
            "model": MODEL_NAME,
            "method": METHOD,
            "balancing": BALANCING,
            "train_file": str(train_file),
            "validation_file": str(validation_file),
            "test_file": str(test_file),
            "train_total": train_counts["total"],
            "train_trace": train_counts["trace"],
            "train_no_trace": train_counts["no_trace"],
            "validation_total": val_counts["total"],
            "validation_trace": val_counts["trace"],
            "validation_no_trace": val_counts["no_trace"],
            "test_total": test_counts["total"],
            "test_trace": test_counts["trace"],
            "test_no_trace": test_counts["no_trace"],
            "config_file": str(config_file),
            "output_dir": str(output_dir),
            "best_model_dir": str(best_model_dir),
            "metrics_file": str(metrics_file),
            "predictions_file": str(predictions_file),
            "accuracy": m.get("accuracy", ""),
            "trace_precision": m.get("trace_precision", ""),
            "trace_recall": m.get("trace_recall", ""),
            "trace_f1": m.get("trace_f1", ""),
            "no_trace_precision": m.get("no_trace_precision", ""),
            "no_trace_recall": m.get("no_trace_recall", ""),
            "no_trace_f1": m.get("no_trace_f1", ""),
            "valid_predictions": m.get("valid_predictions", ""),
            "invalid_predictions": m.get("invalid_predictions", ""),
            "confusion_matrix": json.dumps(m.get("confusion_matrix", "")),
            "notes": "Original imbalanced inner_train.jsonl used without over/undersampling. Test fold used only for final evaluation.",
        }

        append_registry(row)
        append_progress(name, status, str(metrics_file), start_time, end_time)

        print(f"\n{name} completed and tracked.")
        return "completed"

    except Exception as e:
        end_time = now_iso()
        status = "failed"

        row = {
            "experiment_name": EXPERIMENT_NAME,
            "fold": name,
            "status": status,
            "start_time_utc": start_time,
            "end_time_utc": end_time,
            "duration_minutes": duration_minutes(start_time, end_time),
            "model": MODEL_NAME,
            "method": METHOD,
            "balancing": BALANCING,
            "train_file": str(train_file),
            "validation_file": str(validation_file),
            "test_file": str(test_file),
            "train_total": "",
            "train_trace": "",
            "train_no_trace": "",
            "validation_total": "",
            "validation_trace": "",
            "validation_no_trace": "",
            "test_total": "",
            "test_trace": "",
            "test_no_trace": "",
            "config_file": str(config_file),
            "output_dir": str(output_dir),
            "best_model_dir": str(best_model_dir),
            "metrics_file": str(metrics_file),
            "predictions_file": str(predictions_file),
            "accuracy": "",
            "trace_precision": "",
            "trace_recall": "",
            "trace_f1": "",
            "no_trace_precision": "",
            "no_trace_recall": "",
            "no_trace_f1": "",
            "valid_predictions": "",
            "invalid_predictions": "",
            "confusion_matrix": "",
            "notes": f"FAILED: {repr(e)}",
        }

        append_registry(row)
        append_progress(name, status, str(metrics_file), start_time, end_time)

        print(f"\n{name} failed and tracked.")
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_fold", type=int, default=1)
    parser.add_argument("--end_fold", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    RESULT_BASE.mkdir(parents=True, exist_ok=True)

    if args.force:
        print("WARNING: --force is set, but this runner currently skips only based on metrics existence.")
        print("If you need a clean rerun, manually remove the relevant result/output fold first.")

    for fold in range(args.start_fold, args.end_fold + 1):
        run_fold(fold)

    if args.start_fold == 1 and args.end_fold == 10:
        create_summary()


if __name__ == "__main__":
    main()
