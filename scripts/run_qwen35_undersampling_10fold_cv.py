#!/usr/bin/env python3

import argparse
import csv
import json
import statistics as stats
import subprocess
from pathlib import Path


RESULT_BASE = Path("results/qwen35_9b_qlora_undersampling_10fold_cv_inner_val")
OUTPUT_BASE = Path("outputs/qwen35_9b_qlora_undersampling_10fold_cv_inner_val")
DATA_BASE = Path("data/qwen35_oversampling_10fold_cv")


def run(cmd):
    print("\n" + "=" * 100)
    print("RUNNING:")
    print(" ".join(cmd))
    print("=" * 100)
    subprocess.run(cmd, check=True)


def fold_name(fold):
    return f"fold_{fold:02d}"


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
        "folds": rows,
        "summary": summary,
        "summed_confusion_matrix_labels": ["trace", "no_trace"],
        "summed_confusion_matrix": summed_cm,
    }

    RESULT_BASE.mkdir(parents=True, exist_ok=True)

    json_path = RESULT_BASE / "qwen35_9b_qlora_undersampling_10fold_cv_summary.json"
    csv_path = RESULT_BASE / "qwen35_9b_qlora_undersampling_10fold_cv_summary.csv"

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_fold", type=int, default=1)
    parser.add_argument("--end_fold", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    RESULT_BASE.mkdir(parents=True, exist_ok=True)

    progress_path = RESULT_BASE / "progress.csv"
    if not progress_path.exists():
        with progress_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["fold", "status", "metrics_path"])

    for fold in range(args.start_fold, args.end_fold + 1):
        name = fold_name(fold)

        print("\n" + "#" * 100)
        print(f"STARTING {name}")
        print("#" * 100)

        metrics_path = RESULT_BASE / name / f"{name}_metrics.json"

        if metrics_path.exists() and not args.force:
            print(f"{name} already completed. Metrics found: {metrics_path}")
            print("Skipping this fold. Use --force to rerun.")
            continue

        run(["python", "scripts/create_inner_train_validation_split.py", "--fold", str(fold)])

        run(["python", "scripts/create_undersampled_inner_train_fold.py", "--fold", str(fold)])

        run(["python", "scripts/create_qwen35_inner_val_undersampling_fold_config.py", "--fold", str(fold)])

        config_path = Path(
            f"src/integration/training_scripts/ft/"
            f"config_qwen35_9b_10fold_cv_inner_val_undersampling_{name}.yaml"
        )

        output_dir = OUTPUT_BASE / name
        output_dir.mkdir(parents=True, exist_ok=True)

        run([
            "python",
            "src/integration/training_scripts/ft/run_ft.py",
            "--config",
            str(config_path),
            "--method",
            "qlora",
        ])

        best_model = output_dir / "best_model"
        if not best_model.exists():
            raise FileNotFoundError(f"best_model not found after training: {best_model}")

        result_dir = RESULT_BASE / name
        result_dir.mkdir(parents=True, exist_ok=True)

        test_file = DATA_BASE / "folds" / name / "test.jsonl"

        run([
            "python",
            "scripts/evaluate_qwen35_fold_classification.py",
            "--model_dir",
            str(best_model),
            "--test_file",
            str(test_file),
            "--output_dir",
            str(result_dir),
        ])

        if not metrics_path.exists():
            raise FileNotFoundError(f"Metrics file not found after evaluation: {metrics_path}")

        with progress_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([name, "completed", str(metrics_path)])

        print(f"\n{name} completed and saved.")

    create_summary()


if __name__ == "__main__":
    main()
