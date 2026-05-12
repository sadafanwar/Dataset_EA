#!/usr/bin/env python3
"""
Evaluate Qwen3.5 QLoRA model on one 10-fold CV test fold.

Task-specific evaluation for traceability classification:
- labels: trace / no_trace
- metrics: accuracy, precision, recall, F1, confusion matrix
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from peft import AutoPeftModelForCausalLM
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tqdm import tqdm
from transformers import AutoTokenizer


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
        "Determine whether the lower-level requirement traces to the higher-level requirement.",
    )

    input_obj = record.get("input", {})
    higher_req = input_obj.get("higher_level_requirement", "")
    lower_req = input_obj.get("lower_level_requirement", "")

    prompt = (
        f"{instruction}\n\n"
        f"Higher-level requirement:\n{higher_req}\n\n"
        f"Lower-level requirement:\n{lower_req}\n\n"
        f"Answer:"
    )

    return prompt


def normalize_prediction(raw_text: str):
    text = raw_text.strip().lower()
    text = text.replace("\n", " ").replace(".", " ").replace(",", " ").strip()
    tokens = text.split()
    first_token = tokens[0] if tokens else ""

    # Check negative/no_trace cases first because "no_trace" contains "trace".
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_dir",
        required=True,
        help="Path to trained PEFT/QLoRA model directory.",
    )
    parser.add_argument(
        "--test_file",
        required=True,
        help="Path to held-out test JSONL file.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where metrics and predictions will be saved.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=5,
        help="Maximum generated tokens for label prediction.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum input token length.",
    )

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    test_file = Path(args.test_file)
    output_dir = Path(args.output_dir)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Model directory:", model_dir)
    print("Test file:", test_file)
    print("Output directory:", output_dir)

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading PEFT/QLoRA model...")
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_dir,
        device_map={"": 0},
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )

    model.eval()

    records = load_jsonl(test_file)
    print(f"\nLoaded test records: {len(records)}")

    y_true = []
    y_pred = []
    rows = []

    for record in tqdm(records, desc="Evaluating"):
        true_label = record["output"]
        prompt = build_prompt(record)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        )

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
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
                "higher_level_requirement": record.get("input", {}).get(
                    "higher_level_requirement", ""
                ),
                "lower_level_requirement": record.get("input", {}).get(
                    "lower_level_requirement", ""
                ),
            }
        )

    labels = ["trace", "no_trace"]

    valid_indices = [i for i, pred in enumerate(y_pred) if pred in labels]
    invalid_indices = [i for i, pred in enumerate(y_pred) if pred not in labels]

    y_true_valid = [y_true[i] for i in valid_indices]
    y_pred_valid = [y_pred[i] for i in valid_indices]

    if len(y_true_valid) == 0:
        raise ValueError("No valid predictions were produced.")

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
        "model_dir": str(model_dir),
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

    fold_name = output_dir.name

    predictions_file = output_dir / f"{fold_name}_predictions.csv"
    metrics_file = output_dir / f"{fold_name}_metrics.json"

    pd.DataFrame(rows).to_csv(predictions_file, index=False)

    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nEvaluation complete.")
    print("Total records:", len(records))
    print("Valid predictions:", len(valid_indices))
    print("Invalid predictions:", len(invalid_indices))
    print()
    print("Accuracy:", round(accuracy, 4))
    print("Trace precision:", round(precision[0], 4))
    print("Trace recall:", round(recall[0], 4))
    print("Trace F1:", round(f1[0], 4))
    print("No-trace precision:", round(precision[1], 4))
    print("No-trace recall:", round(recall[1], 4))
    print("No-trace F1:", round(f1[1], 4))
    print()
    print("Confusion matrix labels:", labels)
    print(cm)
    print()
    print("Saved predictions:", predictions_file)
    print("Saved metrics:", metrics_file)


if __name__ == "__main__":
    main()
