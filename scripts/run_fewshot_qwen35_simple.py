#!/usr/bin/env python3

import argparse
import csv
import json
import random
import re
from pathlib import Path
from collections import Counter

import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen3.5-9B"
SEED = 42

PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_FILE = PROJECT_DIR / "transformed_data" / "cross_dataset_pattern1.jsonl"

DATA_DIR = PROJECT_DIR / "data" / "fewshot_qwen35"
RESULTS_DIR = PROJECT_DIR / "results" / "fewshot_qwen35"

ZERO_SHOT_FILE = DATA_DIR / "zero_shot_examples.jsonl"
FIVE_SHOT_FILE = DATA_DIR / "five_shot_examples.jsonl"
TEN_SHOT_FILE = DATA_DIR / "ten_shot_examples.jsonl"
TEST_FILE = DATA_DIR / "test_100.jsonl"

ZERO_PRED_FILE = RESULTS_DIR / "zero_shot_predictions.csv"
FIVE_PRED_FILE = RESULTS_DIR / "five_shot_predictions.csv"
TEN_PRED_FILE = RESULTS_DIR / "ten_shot_predictions.csv"

SUMMARY_CSV = RESULTS_DIR / "fewshot_summary.csv"
SUMMARY_JSON = RESULTS_DIR / "fewshot_summary.json"
NOTES_FILE = RESULTS_DIR / "fewshot_notes.md"


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_line_id"] = line_no
            rows.append(row)
    return rows


def save_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            clean = {k: v for k, v in row.items() if not k.startswith("_")}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def get_label(row):
    label = str(row.get("output", "")).strip()
    if label not in {"trace", "no_trace"}:
        raise ValueError(f"Invalid label at line {row.get('_line_id')}: {label}")
    return label


def get_requirements(row):
    input_obj = row.get("input", {})
    high = str(input_obj.get("higher_level_requirement", "")).strip()
    low = str(input_obj.get("lower_level_requirement", "")).strip()

    if not high or not low:
        raise ValueError(f"Missing requirement fields at line {row.get('_line_id')}")

    return high, low


def prepare_data():
    random.seed(SEED)

    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Source dataset not found: {SOURCE_FILE}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(SOURCE_FILE)
    trace_rows = [r for r in rows if get_label(r) == "trace"]
    no_trace_rows = [r for r in rows if get_label(r) == "no_trace"]

    print("Loaded source rows:", len(rows))
    print("Source label counts:", Counter(get_label(r) for r in rows))

    # Required rows:
    # trace: 3 for 5-shot + 5 for 10-shot + 50 for test = 58
    # no_trace: 2 for 5-shot + 5 for 10-shot + 50 for test = 57
    if len(trace_rows) < 58 or len(no_trace_rows) < 57:
        raise RuntimeError("Not enough rows for corrected few-shot design.")

    random.shuffle(trace_rows)
    random.shuffle(no_trace_rows)

    zero_examples = []

    # Corrected design:
    # 0-shot: no examples
    # 5-shot and 10-shot examples are separate/disjoint
    # test_100 is separate from both prompt-example sets
    five_examples = trace_rows[0:3] + no_trace_rows[0:2]
    ten_examples = trace_rows[3:8] + no_trace_rows[2:7]
    test_rows = trace_rows[8:58] + no_trace_rows[7:57]

    random.shuffle(five_examples)
    random.shuffle(ten_examples)
    random.shuffle(test_rows)

    five_ids = {r["_line_id"] for r in five_examples}
    ten_ids = {r["_line_id"] for r in ten_examples}
    test_ids = {r["_line_id"] for r in test_rows}

    assert len(five_examples) == 5
    assert len(ten_examples) == 10
    assert len(test_rows) == 100

    assert not (five_ids & ten_ids), "Overlap between five-shot and ten-shot examples"
    assert not (five_ids & test_ids), "Overlap between five-shot and test set"
    assert not (ten_ids & test_ids), "Overlap between ten-shot and test set"

    save_jsonl(zero_examples, ZERO_SHOT_FILE)
    save_jsonl(five_examples, FIVE_SHOT_FILE)
    save_jsonl(ten_examples, TEN_SHOT_FILE)
    save_jsonl(test_rows, TEST_FILE)

    print("Saved:", ZERO_SHOT_FILE)
    print("Saved:", FIVE_SHOT_FILE)
    print("Saved:", TEN_SHOT_FILE)
    print("Saved:", TEST_FILE)

    print("zero_shot labels:", Counter(get_label(r) for r in zero_examples))
    print("five_shot labels:", Counter(get_label(r) for r in five_examples))
    print("ten_shot labels:", Counter(get_label(r) for r in ten_examples))
    print("test_100 labels:", Counter(get_label(r) for r in test_rows))

    print("Overlap checks passed:")
    print("five ∩ ten:", len(five_ids & ten_ids))
    print("five ∩ test:", len(five_ids & test_ids))
    print("ten ∩ test:", len(ten_ids & test_ids))


def build_prompt(demo_examples, test_row):
    prompt = (
        "You are performing requirements traceability classification.\n\n"
        "Allowed labels:\n"
        "trace\n"
        "no_trace\n\n"
        "Label meaning:\n"
        "trace = the lower-level requirement semantically supports, refines, implements, or is linked to the higher-level requirement.\n"
        "no_trace = the lower-level requirement has no meaningful traceability relation with the higher-level requirement.\n"
    )

    if demo_examples:
        prompt += "\nExamples:\n"
        for i, row in enumerate(demo_examples, start=1):
            high, low = get_requirements(row)
            label = get_label(row)
            prompt += (
                f"\nExample {i}:\n"
                f"Higher-level requirement:\n{high}\n\n"
                f"Lower-level requirement:\n{low}\n\n"
                f"Answer:\n{label}\n"
            )

    high, low = get_requirements(test_row)

    prompt += (
        "\nNow classify the following requirement pair.\n\n"
        f"Higher-level requirement:\n{high}\n\n"
        f"Lower-level requirement:\n{low}\n\n"
        "Answer only one label: trace or no_trace.\n"
        "Answer:\n"
    )

    return prompt


def normalize_prediction(raw_text):
    text = raw_text.strip().lower()
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"[*`\"']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Check no_trace first because no_trace contains the word trace.
    if "no_trace" in text or "no trace" in text or text.startswith("no"):
        return "no_trace"

    if text == "trace" or text.startswith("trace") or "answer: trace" in text:
        return "trace"

    return "invalid"


def load_model():
    print("Loading tokenizer:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model:", MODEL_NAME)

    try:
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
        print("Model loaded in 4-bit inference mode.")

    except Exception as e:
        print("4-bit loading failed. Falling back to auto dtype.")
        print("Reason:", repr(e))

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    return tokenizer, model


@torch.no_grad()
def generate_answer(prompt, tokenizer, model):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=8192,
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=8,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def run_setting(method, demo_file, pred_file, tokenizer, model, max_records=None):
    demo_examples = load_jsonl(demo_file)
    test_rows = load_jsonl(TEST_FILE)

    if max_records is not None:
        test_rows = test_rows[:max_records]

    print(f"\nRunning setting: {method}")
    print("Demo examples:", len(demo_examples))
    print("Test records:", len(test_rows))

    rows_out = []

    for i, row in enumerate(test_rows, start=1):
        prompt = build_prompt(demo_examples, row)
        raw_pred = generate_answer(prompt, tokenizer, model)
        pred_label = normalize_prediction(raw_pred)
        gold_label = get_label(row)
        high, low = get_requirements(row)

        rows_out.append({
            "example_id": row.get("example_id") or f"line_{row.get('_line_id', i)}",
            "gold_label": gold_label,
            "pred_label": pred_label,
            "raw_prediction": raw_pred.strip(),
            "shot_setting": method,
            "higher_level_requirement": high,
            "lower_level_requirement": low,
        })

        if i % 10 == 0 or i == len(test_rows):
            print(f"{method}: evaluated {i}/{len(test_rows)}")

    pred_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_out).to_csv(pred_file, index=False)
    print("Saved predictions:", pred_file)

    return rows_out


def compute_metrics(pred_rows, method):
    valid_rows = [r for r in pred_rows if r["pred_label"] in {"trace", "no_trace"}]
    invalid = len(pred_rows) - len(valid_rows)

    if not valid_rows:
        raise RuntimeError(f"No valid predictions for {method}")

    y_true = [r["gold_label"] for r in valid_rows]
    y_pred = [r["pred_label"] for r in valid_rows]

    labels = ["trace", "no_trace"]

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "method": method,
        "accuracy": float(acc),
        "trace_precision": float(precision[0]),
        "trace_recall": float(recall[0]),
        "trace_f1": float(f1[0]),
        "no_trace_precision": float(precision[1]),
        "no_trace_recall": float(recall[1]),
        "no_trace_f1": float(f1[1]),
        "valid_predictions": int(len(valid_rows)),
        "invalid_predictions": int(invalid),
        "total_records": int(len(pred_rows)),
        "confusion_matrix": cm.tolist(),
    }


def save_summary(summary_rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "method",
        "accuracy",
        "trace_precision",
        "trace_recall",
        "trace_f1",
        "no_trace_precision",
        "no_trace_recall",
        "no_trace_f1",
        "valid_predictions",
        "invalid_predictions",
        "total_records",
        "confusion_matrix",
    ]

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            r = dict(row)
            r["confusion_matrix"] = json.dumps(r["confusion_matrix"])
            writer.writerow(r)

    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    print("Saved summary:", SUMMARY_CSV)
    print("Saved summary:", SUMMARY_JSON)


def save_notes():
    text = f"""# Few-shot Qwen Traceability Experiment

## Experiment

Model: {MODEL_NAME}  
Fine-tuning: No  
LoRA / QLoRA adapter: No  
Mode: prompt-based classification with generation-based prediction  

## Source dataset

`transformed_data/cross_dataset_pattern1.jsonl`

## Created data files

- `data/fewshot_qwen35/zero_shot_examples.jsonl`
- `data/fewshot_qwen35/five_shot_examples.jsonl`
- `data/fewshot_qwen35/ten_shot_examples.jsonl`
- `data/fewshot_qwen35/test_100.jsonl`

No `example_bank.jsonl` is used.

## Setup

Seed: {SEED}  
Test set: 100 balanced examples, 50 trace + 50 no_trace  

Prompt settings:

- 0-shot: instructions only, no examples
- 5-shot: instructions + 5 examples
  - 3 trace
  - 2 no_trace
- 10-shot: instructions + 10 examples
  - 5 trace
  - 5 no_trace

The 5-shot and 10-shot examples are disjoint.
Prompt examples do not overlap with the test set.

## Prediction settings

- max_new_tokens: 8
- do_sample: False
- temperature: deterministic generation; sampling disabled

## Labels

- `trace`
- `no_trace`

This experiment does not train or fine-tune the model.
The examples are included only inside the prompt.
"""

    NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
    NOTES_FILE.write_text(text, encoding="utf-8")
    print("Saved notes:", NOTES_FILE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    prepare_data()

    if args.prepare_only:
        print("Prepare-only mode complete. No model loaded.")
        return

    tokenizer, model = load_model()

    zero_preds = run_setting(
        "zero_shot",
        ZERO_SHOT_FILE,
        ZERO_PRED_FILE,
        tokenizer,
        model,
        args.max_records,
    )
    five_preds = run_setting(
        "five_shot",
        FIVE_SHOT_FILE,
        FIVE_PRED_FILE,
        tokenizer,
        model,
        args.max_records,
    )
    ten_preds = run_setting(
        "ten_shot",
        TEN_SHOT_FILE,
        TEN_PRED_FILE,
        tokenizer,
        model,
        args.max_records,
    )

    summary_rows = [
        compute_metrics(zero_preds, "zero_shot"),
        compute_metrics(five_preds, "five_shot"),
        compute_metrics(ten_preds, "ten_shot"),
    ]

    save_summary(summary_rows)
    save_notes()

    print("\nFinal summary")
    print("-" * 100)
    for row in summary_rows:
        print(
            f"{row['method']} | "
            f"acc={row['accuracy']:.4f} | "
            f"trace_p={row['trace_precision']:.4f} | "
            f"trace_r={row['trace_recall']:.4f} | "
            f"trace_f1={row['trace_f1']:.4f} | "
            f"no_trace_f1={row['no_trace_f1']:.4f} | "
            f"valid={row['valid_predictions']} | "
            f"invalid={row['invalid_predictions']} | "
            f"cm={row['confusion_matrix']}"
        )


if __name__ == "__main__":
    main()
