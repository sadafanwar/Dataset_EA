#!/usr/bin/env python3
"""
Evaluate base Qwen3.5-9B model without fine-tuning on the same 10 held-out test folds.

This is a zero-shot / no-fine-tuning baseline:
- no LoRA
- no QLoRA training
- no oversampling
- no undersampling
- same base model evaluated on fold_01/test.jsonl ... fold_10/test.jsonl
"""

import argparse
import json
import csv
from pathlib import Path
import statistics as stats

import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def load_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_prompt(record):
    instruction = record.get(
        "instruction",
        "Determine whether the lower-level requirement traces to the higher-level requirement. Answer only with trace or no_trace.",
    )

    input_obj = record.get("input", {})
    higher_req = input_obj.get("higher_level_requirement", "")
    lower_req = input_obj.get("lower_level_requirement", "")

    prompt = (
        f"{instruction}\n\n"
        f"Higher-level requirement:\n{higher_req}\n\n"
        f"Lower-level requirement:\n{lower_req}\n\n"
        f"Answer only with one label: trace or no_trace.\n"
        f"Answer:"
    )

    return prompt


def normalize_prediction(raw_text: str):
    text = raw_text.strip().lower()
    text = text.replace("\n", " ").replace(".", " ").replace(",", " ").strip()
    tokens = text.split()
    first_token = tokens[0] if tokens else ""

    # Check negative/no_trace cases first because no_trace contains trace.
    if (
        "no_trace" in text
        or "no trace" in text
        or "not trace" in text
        or first_token == "no"
    ):
        return "no_trace"

    if (
        first_token == "yes"
        or first_token == "trace"
        or text.startswith("trace")
    ):
        return "trace"

    return "invalid"


def evaluate_fold(model, tokenizer, fold: int, data_base: Path, output_base: Path, max_new_tokens: int, max_length: int):
    fold_name = f"fold_{fold:02d}"
    test_file = data_base / "folds" / fold_name / "test.jsonl"
    output_dir = output_base / fold_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if not test_file.exists():
        raise FileNotFoundError(f"Missing test file: {test_file}")

    records = load_jsonl(test_file)

    y_true = []
    y_pred = []
    rows = []

    print(f"\nEvaluating {fold_name}")
    print("Test file:", test_file)
    print("Records:", len(records))

    for record in tqdm(records, desc=f"Evaluating {fold_name}"):
        true_label = record["output"]
        prompt = build_prompt(record)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_label_text = tokenizer.decode(
            generated[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        pred_label = normalize_prediction(generated_label_text)

        y_true.append(true_label)
        y_pred.append(pred_label)

        rows.append(
            {
                "example_id": record.get("example_id", ""),
                "true_label": true_label,
                "pred_label": pred_label,
                "raw_prediction": generated_label_text.strip(),
                "higher_level_requirement": record.get("input", {}).get("higher_level_requirement", ""),
                "lower_level_requirement": record.get("input", {}).get("lower_level_requirement", ""),
            }
        )

    labels = ["trace", "no_trace"]

    valid_indices = [i for i, pred in enumerate(y_pred) if pred in labels]
    invalid_indices = [i for i, pred in enumerate(y_pred) if pred not in labels]

    y_true_valid = [y_true[i] for i in valid_indices]
    y_pred_valid = [y_pred[i] for i in valid_indices]

    if len(y_true_valid) == 0:
        raise ValueError(f"No valid predictions for {fold_name}.")

    accuracy = accuracy_score(y_true_valid, y_pred_valid)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_valid,
        y_pred_valid,
        labels=labels,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true_valid,
        y_pred_valid,
        labels=labels,
    )

    report = classification_report(
        y_true_valid,
        y_pred_valid,
        labels=labels,
        zero_division=0,
        output_dict=True,
    )

    metrics = {
        "model": "Qwen/Qwen3.5-9B",
        "experiment": "base_no_finetuning_zero_shot",
        "fold": fold_name,
        "test_file": str(test_file),
        "total_records": len(records),
        "valid_predictions": len(valid_indices),
        "invalid_predictions": len(invalid_indices),
        "accuracy": float(accuracy),
        "labels_order": labels,
        "trace_precision": float(precision[0]),
        "trace_recall": float(recall[0]),
        "trace_f1": float(f1[0]),
        "trace_support": int(support[0]),
        "no_trace_precision": float(precision[1]),
        "no_trace_recall": float(recall[1]),
        "no_trace_f1": float(f1[1]),
        "no_trace_support": int(support[1]),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    predictions_file = output_dir / f"{fold_name}_predictions.csv"
    metrics_file = output_dir / f"{fold_name}_metrics.json"

    pd.DataFrame(rows).to_csv(predictions_file, index=False)

    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{fold_name} complete.")
    print("Accuracy:", round(accuracy, 4))
    print("Trace F1:", round(f1[0], 4))
    print("No-trace F1:", round(f1[1], 4))
    print("Invalid predictions:", len(invalid_indices))
    print("Confusion matrix:")
    print(cm)

    return metrics


def create_summary(output_base: Path):
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
        fold_name = f"fold_{i:02d}"
        metrics_file = output_base / fold_name / f"{fold_name}_metrics.json"

        if not metrics_file.exists():
            raise FileNotFoundError(f"Missing metrics file: {metrics_file}")

        with metrics_file.open("r", encoding="utf-8") as f:
            m = json.load(f)

        row = {"fold": fold_name}
        for k in metric_keys:
            row[k] = float(m[k])
        rows.append(row)

        cm = m["confusion_matrix"]
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
        "experiment": "Qwen3.5-9B base model, no fine-tuning, evaluated on same 10 held-out test folds",
        "folds": rows,
        "summary": summary,
        "summed_confusion_matrix_labels": ["trace", "no_trace"],
        "summed_confusion_matrix": summed_cm,
    }

    json_path = output_base / "qwen35_9b_base_10fold_cv_no_ft_summary.json"
    csv_path = output_base / "qwen35_9b_base_10fold_cv_no_ft_summary.csv"

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
    parser.add_argument("--model_path", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--start_fold", type=int, default=1)
    parser.add_argument("--end_fold", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--output_dir", default="results/qwen35_9b_base_10fold_cv_no_ft")
    args = parser.parse_args()

    data_base = Path("data/qwen35_oversampling_10fold_cv")
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer:", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading base model with 4-bit quantization for inference:", args.model_path)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )

    model.eval()

    for fold in range(args.start_fold, args.end_fold + 1):
        evaluate_fold(
            model=model,
            tokenizer=tokenizer,
            fold=fold,
            data_base=data_base,
            output_base=output_base,
            max_new_tokens=args.max_new_tokens,
            max_length=args.max_length,
        )

    if args.start_fold == 1 and args.end_fold == 10:
        create_summary(output_base)


if __name__ == "__main__":
    main()
