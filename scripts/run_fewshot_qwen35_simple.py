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

EXAMPLE_BANK_FILE = DATA_DIR / "example_bank.jsonl"
TEST_FILE = DATA_DIR / "test_100.jsonl"
FIVE_SHOT_FILE = DATA_DIR / "five_shot_examples.jsonl"
TEN_SHOT_FILE = DATA_DIR / "ten_shot_examples.jsonl"

FIVE_SHOT_PRED_FILE = RESULTS_DIR / "five_shot_predictions.csv"
TEN_SHOT_PRED_FILE = RESULTS_DIR / "ten_shot_predictions.csv"
SUMMARY_CSV = RESULTS_DIR / "fewshot_summary.csv"
SUMMARY_JSON = RESULTS_DIR / "fewshot_summary.json"
NOTES_FILE = RESULTS_DIR / "fewshot_notes.md"


def load_jsonl(path: Path):
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


def save_jsonl(rows, path: Path):
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
    high = input_obj.get("higher_level_requirement", "")
    low = input_obj.get("lower_level_requirement", "")

    if not high or not low:
        raise ValueError(f"Missing requirement fields at line {row.get('_line_id')}")

    return high.strip(), low.strip()


def prepare_data():
    random.seed(SEED)

    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Source dataset not found: {SOURCE_FILE}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(SOURCE_FILE)
    print(f"Loaded source rows: {len(rows)}")

    trace_rows = [r for r in rows if get_label(r) == "trace"]
    no_trace_rows = [r for r in rows if get_label(r) == "no_trace"]

    print("Source label counts:", Counter(get_label(r) for r in rows))

    if len(trace_rows) < 60 or len(no_trace_rows) < 60:
        raise RuntimeError("Not enough trace/no_trace rows to create few-shot files and balanced test_100.")

    random.shuffle(trace_rows)
    random.shuffle(no_trace_rows)

    # Select ten-shot first, then five-shot as a subset of ten-shot.
    ten_trace = trace_rows[:5]
    ten_no_trace = no_trace_rows[:5]

    five_trace = ten_trace[:3]
    five_no_trace = ten_no_trace[:2]

    ten_examples = ten_trace + ten_no_trace
    five_examples = five_trace + five_no_trace

    used_ids = {r["_line_id"] for r in ten_examples}

    remaining_trace = [r for r in trace_rows if r["_line_id"] not in used_ids]
    remaining_no_trace = [r for r in no_trace_rows if r["_line_id"] not in used_ids]

    test_rows = remaining_trace[:50] + remaining_no_trace[:50]
    random.shuffle(test_rows)

    test_ids = {r["_line_id"] for r in test_rows}

    # Example bank = all examples allowed for prompt/demo selection, excluding test questions.
    example_bank = [r for r in rows if r["_line_id"] not in test_ids]

    # Safety overlap checks
    five_ids = {r["_line_id"] for r in five_examples}
    ten_ids = {r["_line_id"] for r in ten_examples}

    assert not (five_ids & test_ids), "Overlap between five-shot examples and test set"
    assert not (ten_ids & test_ids), "Overlap between ten-shot examples and test set"

    save_jsonl(example_bank, EXAMPLE_BANK_FILE)
    save_jsonl(test_rows, TEST_FILE)
    save_jsonl(five_examples, FIVE_SHOT_FILE)
    save_jsonl(ten_examples, TEN_SHOT_FILE)

    print("Saved:", EXAMPLE_BANK_FILE)
    print("Saved:", TEST_FILE)
    print("Saved:", FIVE_SHOT_FILE)
    print("Saved:", TEN_SHOT_FILE)

    print("five_shot labels:", Counter(get_label(r) for r in five_examples))
    print("ten_shot labels:", Counter(get_label(r) for r in ten_examples))
    print("test_100 labels:", Counter(get_label(r) for r in test_rows))


def build_prompt(demo_examples, test_row):
    parts = []

    parts.append(
        "You are performing requirements traceability classification.\n\n"
        "Allowed labels:\n"
        "trace\n"
        "no_trace\n\n"
        "Label meaning:\n"
        "trace = the lower-level requirement semantically supports, refines, implements, or is linked to the higher-level requirement.\n"
        "no_trace = the lower-level requirement has no meaningful traceability relation with the higher-level requirement.\n\n"
        "Examples:\n"
    )

    for idx, row in enumerate(demo_examples, start=1):
        high, low = get_requirements(row)
        label = get_label(row)

        parts.append(
            f"\nExample {idx}:\n"
            f"Higher-level requirement:\n{high}\n\n"
            f"Lower-level requirement:\n{low}\n\n"
            f"Answer:\n{label}\n"
        )

    test_high, test_low = get_requirements(test_row)

    parts.append(
        "\nNow classify the following requirement pair.\n\n"
        f"Higher-level requirement:\n{test_high}\n\n"
        f"Lower-level requirement:\n{test_low}\n\n"
        "Answer only one label: trace or no_trace.\n"
        "Answer:\n"
    )

    return "".join(parts)


def normalize_prediction(raw_text: str):
    text = raw_text.strip().lower()
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"[*`\"']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Important: check no_trace first because it contains trace.
    if "no_trace" in text or "no trace" in text or text.startswith("no"):
        return "no_trace"

    if text.startswith("trace") or "answer: trace" in text or text == "trace":
        return "trace"

    return "invalid"


def load_model():
    print("Loading tokenizer:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model:", MODEL_NAME)

    # 4-bit loading is for memory saving only. This is not LoRA and not fine-tuning.
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
        print("4-bit loading failed, falling back to auto dtype.")
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
def generate_answer(prompt, tokenizer, model, max_new_tokens=8):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)

    # Move tensors to model device if possible
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return raw


def run_setting(setting_name, demo_file, pred_file, tokenizer, model, max_records=None):
    demo_examples = load_jsonl(demo_file)
    test_rows = load_jsonl(TEST_FILE)

    if max_records is not None:
        test_rows = test_rows[:max_records]

    pred_rows = []

    print(f"\nRunning setting: {setting_name}")
    print(f"Demo examples: {len(demo_examples)}")
    print(f"Test records: {len(test_rows)}")

    for idx, row in enumerate(test_rows, start=1):
        prompt = build_prompt(demo_examples, row)
        raw_pred = generate_answer(prompt, tokenizer, model)
        pred_label = normalize_prediction(raw_pred)
        gold_label = get_label(row)
        high, low = get_requirements(row)

        example_id = row.get("example_id") or f"line_{row.get('_line_id', idx)}"

        pred_rows.append({
            "example_id": example_id,
            "gold_label": gold_label,
            "pred_label": pred_label,
            "raw_prediction": raw_pred.strip(),
            "shot_setting": setting_name,
            "higher_level_requirement": high,
            "lower_level_requirement": low,
        })

        if idx % 10 == 0 or idx == len(test_rows):
            print(f"{setting_name}: evaluated {idx}/{len(test_rows)}")

    pred_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pred_rows).to_csv(pred_file, index=False)
    print("Saved predictions:", pred_file)

    return pred_rows


def compute_metrics(pred_rows, method):
    total = len(pred_rows)
    valid_rows = [r for r in pred_rows if r["pred_label"] in {"trace", "no_trace"}]
    invalid = total - len(valid_rows)

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
        "total_records": int(total),
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
Mode: few-shot prompting with generation-based prediction  

## Source dataset

`transformed_data/cross_dataset_pattern1.jsonl`

## Created data files

- `data/fewshot_qwen35/example_bank.jsonl`
- `data/fewshot_qwen35/test_100.jsonl`
- `data/fewshot_qwen35/five_shot_examples.jsonl`
- `data/fewshot_qwen35/ten_shot_examples.jsonl`

## Setup

Seed: {SEED}  
Test set: 100 balanced examples, 50 trace + 50 no_trace  
5-shot: 3 trace + 2 no_trace examples  
10-shot: 5 trace + 5 no_trace examples  

## Prediction

Generation-based prediction.

Generation settings:

- max_new_tokens: 8
- do_sample: False
- temperature: not sampled / deterministic generation

## Labels

- `trace`
- `no_trace`

## Notes

This experiment does not train or fine-tune the model. The examples are included only inside the prompt.
"""

    NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
    NOTES_FILE.write_text(text, encoding="utf-8")
    print("Saved notes:", NOTES_FILE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-records", type=int, default=None, help="Optional smaller debug run.")
    args = parser.parse_args()

    prepare_data()

    if args.prepare_only:
        print("Prepare-only mode complete. No model loaded.")
        return

    tokenizer, model = load_model()

    five_preds = run_setting(
        "five_shot",
        FIVE_SHOT_FILE,
        FIVE_SHOT_PRED_FILE,
        tokenizer,
        model,
        max_records=args.max_records,
    )

    ten_preds = run_setting(
        "ten_shot",
        TEN_SHOT_FILE,
        TEN_SHOT_PRED_FILE,
        tokenizer,
        model,
        max_records=args.max_records,
    )

    summary_rows = [
        compute_metrics(five_preds, "five_shot"),
        compute_metrics(ten_preds, "ten_shot"),
    ]

    save_summary(summary_rows)
    save_notes()

    print("\nFinal summary")
    print("-" * 80)
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
