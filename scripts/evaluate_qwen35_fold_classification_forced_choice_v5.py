#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from peft import AutoPeftModelForCausalLM
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from transformers import AutoTokenizer


LABELS = ["trace", "no_trace"]


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_example_id(row, index):
    return row.get("example_id") or row.get("id") or f"row_{index:06d}"


def get_requirement_pair(row):
    inp = row.get("input", {})

    higher = inp.get("higher_level_requirement", "") if isinstance(inp, dict) else ""
    lower = inp.get("lower_level_requirement", "") if isinstance(inp, dict) else str(inp)

    return str(higher).strip(), str(lower).strip()


def build_training_matched_prompt(row):
    """
    Must match run_ft.py training format exactly:

    {instruction}

    Higher-level requirement:
    {higher_req}

    Lower-level requirement:
    {lower_req}

    Answer:
    """
    instruction = str(row.get("instruction", "")).strip()
    higher_req, lower_req = get_requirement_pair(row)

    prompt = (
        f"{instruction}\n\n"
        f"Higher-level requirement:\n{higher_req}\n\n"
        f"Lower-level requirement:\n{lower_req}\n\n"
        f"Answer:"
    )

    return prompt


def load_model_and_tokenizer(model_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoPeftModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    model.eval()
    return model, tokenizer


def candidate_loss(model, tokenizer, prompt: str, label: str, max_length: int):
    """
    Score candidate label using the same prompt style used in training.
    Lower loss = preferred label.

    Training text was:
    prompt + " " + output

    So candidate continuation is also:
    prompt + " " + label
    """
    device = next(model.parameters()).device

    # Reserve candidate space so candidate is never truncated away.
    max_candidate_tokens = 8
    max_prompt_tokens = max_length - max_candidate_tokens

    prompt_text = prompt.rstrip() + " "
    candidate_text = label

    prompt_ids = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_tokens,
        add_special_tokens=True,
    )["input_ids"]

    candidate_ids = tokenizer(
        candidate_text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"]

    if candidate_ids.numel() == 0:
        return float("inf")

    full_ids = torch.cat([prompt_ids, candidate_ids], dim=1).to(device)
    prompt_len = prompt_ids.shape[1]
    candidate_len = candidate_ids.shape[1]

    with torch.no_grad():
        outputs = model(input_ids=full_ids)
        logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = full_ids[:, 1:].contiguous()

    start = prompt_len - 1
    end = start + candidate_len

    cand_logits = shift_logits[:, start:end, :]
    cand_labels = shift_labels[:, start:end]

    if cand_labels.numel() == 0:
        return float("inf")

    loss = F.cross_entropy(
        cand_logits.reshape(-1, cand_logits.size(-1)),
        cand_labels.reshape(-1),
        reduction="mean",
    )

    return float(loss.item())


def predict_forced_choice(model, tokenizer, prompt: str, max_length: int):
    scores = {
        "trace": candidate_loss(model, tokenizer, prompt, "trace", max_length),
        "no_trace": candidate_loss(model, tokenizer, prompt, "no_trace", max_length),
    }

    pred_label = min(scores, key=scores.get)
    return pred_label, scores


def compute_metrics(y_true, y_pred):
    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / total if total else 0.0

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    return {
        "total_records": int(total),
        "valid_predictions": int(total),
        "invalid_predictions": 0,
        "accuracy": float(accuracy),

        "trace_precision": float(precision[0]),
        "trace_recall": float(recall[0]),
        "trace_f1": float(f1[0]),
        "trace_support": int(support[0]),

        "no_trace_precision": float(precision[1]),
        "no_trace_recall": float(recall[1]),
        "no_trace_f1": float(f1[1]),
        "no_trace_support": int(support[1]),

        "confusion_matrix_labels": LABELS,
        "confusion_matrix": cm.tolist(),

        "evaluation_method": "forced_choice_label_scoring_training_matched_prompt",
        "note": (
            "Forced-choice evaluation using the exact prompt prefix used during fine-tuning: "
            "instruction + higher-level requirement + lower-level requirement + Answer:. "
            "Candidate labels trace and no_trace are scored by average negative log-likelihood."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_length", type=int, default=1024)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    test_file = Path(args.test_file)
    output_dir = Path(args.output_dir)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model dir: {model_dir}")
    print(f"Test file: {test_file}")
    print(f"Output dir: {output_dir}")
    print(f"Max length: {args.max_length}")

    model, tokenizer = load_model_and_tokenizer(model_dir)
    rows_in = load_jsonl(test_file)

    print(f"Loaded test records: {len(rows_in)}")

    rows_out = []
    y_true = []
    y_pred = []

    for idx, row in enumerate(rows_in):
        example_id = get_example_id(row, idx)
        gold_label = str(row.get("output", "")).strip()

        if gold_label not in LABELS:
            raise ValueError(f"Invalid gold label in {example_id}: {gold_label}")

        prompt = build_training_matched_prompt(row)
        pred_label, scores = predict_forced_choice(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_length=args.max_length,
        )

        higher, lower = get_requirement_pair(row)

        rows_out.append({
            "example_id": example_id,
            "gold_label": gold_label,
            "pred_label": pred_label,
            "trace_loss": scores["trace"],
            "no_trace_loss": scores["no_trace"],
            "margin_no_trace_minus_trace": scores["no_trace"] - scores["trace"],
            "higher_level_requirement": higher,
            "lower_level_requirement": lower,
        })

        y_true.append(gold_label)
        y_pred.append(pred_label)

        if (idx + 1) % 25 == 0 or (idx + 1) == len(rows_in):
            print(f"Evaluated {idx + 1}/{len(rows_in)}")

    metrics = compute_metrics(y_true, y_pred)

    fold_name = output_dir.name
    predictions_file = output_dir / f"{fold_name}_predictions.csv"
    metrics_file = output_dir / f"{fold_name}_metrics.json"

    pd.DataFrame(rows_out).to_csv(predictions_file, index=False)

    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nEvaluation complete.")
    print("Total records:", metrics["total_records"])
    print("Valid predictions:", metrics["valid_predictions"])
    print("Invalid predictions:", metrics["invalid_predictions"])
    print()
    print("Accuracy:", round(metrics["accuracy"], 4))
    print("Trace precision:", round(metrics["trace_precision"], 4))
    print("Trace recall:", round(metrics["trace_recall"], 4))
    print("Trace F1:", round(metrics["trace_f1"], 4))
    print("No-trace precision:", round(metrics["no_trace_precision"], 4))
    print("No-trace recall:", round(metrics["no_trace_recall"], 4))
    print("No-trace F1:", round(metrics["no_trace_f1"], 4))
    print()
    print("Confusion matrix labels:", metrics["confusion_matrix_labels"])
    print(metrics["confusion_matrix"])
    print()
    print("Saved predictions:", predictions_file)
    print("Saved metrics:", metrics_file)


if __name__ == "__main__":
    main()
