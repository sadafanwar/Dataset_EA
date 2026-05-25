#!/usr/bin/env python3

import csv
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


EXP = "qwen35_9b_qlora_train_size_undersampling_10fold_cv_inner_val_v2"
BASE = Path("results") / EXP
SIZES = [100, 200, 300, 500]

# margin = no_trace_loss - trace_loss
# current default rule is: margin > 0 => trace
# More negative threshold predicts trace more often.
THRESHOLDS = [x / 100 for x in range(-300, 101, 5)]  # -3.00 to +1.00


def load_size_predictions(size):
    dfs = []

    for i in range(1, 11):
        fold = f"fold_{i:02d}"
        path = BASE / f"size_{size}" / fold / f"{fold}_predictions.csv"

        if not path.exists():
            raise FileNotFoundError(f"Missing predictions file: {path}")

        df = pd.read_csv(path)

        required = {"gold_label", "margin_no_trace_minus_trace"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")

        df["fold"] = fold
        df["train_size"] = size
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def evaluate_threshold(df, threshold):
    y_true = df["gold_label"].tolist()
    margins = df["margin_no_trace_minus_trace"].tolist()

    y_pred = [
        "trace" if margin > threshold else "no_trace"
        for margin in margins
    ]

    acc = accuracy_score(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=["trace", "no_trace"],
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=["trace", "no_trace"],
    )

    pred_trace = sum(1 for p in y_pred if p == "trace")
    pred_no_trace = sum(1 for p in y_pred if p == "no_trace")

    return {
        "threshold": threshold,
        "accuracy": acc,
        "trace_precision": precision[0],
        "trace_recall": recall[0],
        "trace_f1": f1[0],
        "trace_support": int(support[0]),
        "no_trace_precision": precision[1],
        "no_trace_recall": recall[1],
        "no_trace_f1": f1[1],
        "no_trace_support": int(support[1]),
        "pred_trace": pred_trace,
        "pred_no_trace": pred_no_trace,
        "confusion_matrix": cm.tolist(),
    }


def main():
    all_rows = []
    best_rows = []

    for size in SIZES:
        print(f"\nRunning threshold sweep for train_size={size}")
        df = load_size_predictions(size)

        size_rows = []

        for threshold in THRESHOLDS:
            row = evaluate_threshold(df, threshold)
            row["train_size"] = size
            size_rows.append(row)
            all_rows.append(row)

        # Best by trace_f1. If tie, prefer higher trace_recall, then higher accuracy.
        best = sorted(
            size_rows,
            key=lambda r: (r["trace_f1"], r["trace_recall"], r["accuracy"]),
            reverse=True,
        )[0]

        best_rows.append(best)

        print(
            f"Best size={size}: "
            f"threshold={best['threshold']:.2f}, "
            f"accuracy={best['accuracy']:.4f}, "
            f"trace_p={best['trace_precision']:.4f}, "
            f"trace_r={best['trace_recall']:.4f}, "
            f"trace_f1={best['trace_f1']:.4f}, "
            f"no_trace_f1={best['no_trace_f1']:.4f}, "
            f"pred_trace={best['pred_trace']}, "
            f"pred_no_trace={best['pred_no_trace']}, "
            f"cm={best['confusion_matrix']}"
        )

    out_all = BASE / "threshold_sweep_all_100_200_300_500_v5.csv"
    out_best = BASE / "threshold_sweep_best_by_trace_f1_v5.csv"
    out_json = BASE / "threshold_sweep_best_by_trace_f1_v5.json"

    fieldnames = [
        "train_size",
        "threshold",
        "accuracy",
        "trace_precision",
        "trace_recall",
        "trace_f1",
        "trace_support",
        "no_trace_precision",
        "no_trace_recall",
        "no_trace_f1",
        "no_trace_support",
        "pred_trace",
        "pred_no_trace",
        "confusion_matrix",
    ]

    with out_all.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            r = dict(row)
            r["confusion_matrix"] = json.dumps(r["confusion_matrix"])
            writer.writerow(r)

    with out_best.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in best_rows:
            r = dict(row)
            r["confusion_matrix"] = json.dumps(r["confusion_matrix"])
            writer.writerow(r)

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(best_rows, f, indent=2)

    print("\nSaved:")
    print(out_all)
    print(out_best)
    print(out_json)

    print("\nFinal best thresholds by Trace F1")
    print("-" * 120)
    for r in best_rows:
        print(
            f"size={r['train_size']} | "
            f"threshold={r['threshold']:.2f} | "
            f"acc={r['accuracy']:.4f} | "
            f"trace_p={r['trace_precision']:.4f} | "
            f"trace_r={r['trace_recall']:.4f} | "
            f"trace_f1={r['trace_f1']:.4f} | "
            f"no_trace_f1={r['no_trace_f1']:.4f} | "
            f"pred_trace={r['pred_trace']} | "
            f"pred_no_trace={r['pred_no_trace']} | "
            f"cm={r['confusion_matrix']}"
        )


if __name__ == "__main__":
    main()
