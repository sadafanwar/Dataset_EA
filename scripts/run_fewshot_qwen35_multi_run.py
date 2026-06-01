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


# -----------------------------
# Experiment configuration
# -----------------------------

MODEL_NAME = "Qwen/Qwen3.5-9B"

TEST_SEED = 42
RUN_SEEDS = list(range(1, 11))  # 10 multi-runs: seeds 1 to 10

PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_FILE = PROJECT_DIR / "transformed_data" / "cross_dataset_pattern1.jsonl"

DATA_DIR = PROJECT_DIR / "data" / "fewshot_qwen35_multi_run"
RESULTS_DIR = PROJECT_DIR / "results" / "fewshot_qwen35_multi_run"

TEST_FILE = DATA_DIR / "test_100.jsonl"
ZERO_SHOT_FILE = DATA_DIR / "zero_shot_examples.jsonl"
AVAILABLE_PROMPT_FILE = DATA_DIR / "available_prompt_examples.jsonl"

ZERO_PRED_FILE = RESULTS_DIR / "zero_shot_predictions.csv"

RUN_SUMMARY_CSV = RESULTS_DIR / "fewshot_run_summary.csv"
RUN_SUMMARY_JSON = RESULTS_DIR / "fewshot_run_summary.json"
AGGREGATE_SUMMARY_CSV = RESULTS_DIR / "fewshot_aggregate_summary.csv"
AGGREGATE_SUMMARY_JSON = RESULTS_DIR / "fewshot_aggregate_summary.json"
NOTES_FILE = RESULTS_DIR / "fewshot_notes.md"


# -----------------------------
# Basic JSONL helpers
# -----------------------------

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


def row_key(row):
    return row["_line_id"]


def count_labels(rows):
    return Counter(get_label(r) for r in rows)


def run_label(run_idx):
    return f"run_{run_idx:02d}"


# -----------------------------
# Data preparation
# -----------------------------

def sample_balanced_rows(trace_rows, no_trace_rows, n_trace, n_no_trace, seed):
    rng = random.Random(seed)
    selected_trace = rng.sample(trace_rows, n_trace)
    selected_no_trace = rng.sample(no_trace_rows, n_no_trace)
    selected = selected_trace + selected_no_trace
    rng.shuffle(selected)
    return selected


def prepare_data():
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Source dataset not found: {SOURCE_FILE}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(SOURCE_FILE)

    trace_rows = [r for r in rows if get_label(r) == "trace"]
    no_trace_rows = [r for r in rows if get_label(r) == "no_trace"]

    print("Loaded source rows:", len(rows))
    print("Source label counts:", count_labels(rows))

    if len(trace_rows) < 60 or len(no_trace_rows) < 60:
        raise RuntimeError("Not enough trace/no_trace examples for test set and prompt runs.")

    # 1. Fixed test set created once using TEST_SEED.
    test_rows = sample_balanced_rows(
        trace_rows=trace_rows,
        no_trace_rows=no_trace_rows,
        n_trace=50,
        n_no_trace=50,
        seed=TEST_SEED,
    )

    test_ids = {row_key(r) for r in test_rows}

    # 2. Prompt pool excludes test rows.
    available_rows = [r for r in rows if row_key(r) not in test_ids]
    available_trace = [r for r in available_rows if get_label(r) == "trace"]
    available_no_trace = [r for r in available_rows if get_label(r) == "no_trace"]

    save_jsonl(test_rows, TEST_FILE)
    save_jsonl([], ZERO_SHOT_FILE)
    save_jsonl(available_rows, AVAILABLE_PROMPT_FILE)

    print("Saved fixed test set:", TEST_FILE)
    print("Saved zero-shot examples:", ZERO_SHOT_FILE)
    print("Saved available prompt examples:", AVAILABLE_PROMPT_FILE)
    print("test_100 labels:", count_labels(test_rows))
    print("available prompt labels:", count_labels(available_rows))

    # 3. For each run, create separate 5-shot and 10-shot files.
    for run_idx, seed in enumerate(RUN_SEEDS, start=1):
        label = run_label(run_idx)

        rng = random.Random(seed)

        # Select 5-shot first.
        five_trace = rng.sample(available_trace, 3)
        five_no_trace = rng.sample(available_no_trace, 2)
        five_examples = five_trace + five_no_trace

        five_ids = {row_key(r) for r in five_examples}

        # Select 10-shot from remaining pool so 5-shot and 10-shot are disjoint in same run.
        remaining_trace = [r for r in available_trace if row_key(r) not in five_ids]
        remaining_no_trace = [r for r in available_no_trace if row_key(r) not in five_ids]

        ten_trace = rng.sample(remaining_trace, 5)
        ten_no_trace = rng.sample(remaining_no_trace, 5)
        ten_examples = ten_trace + ten_no_trace

        rng.shuffle(five_examples)
        rng.shuffle(ten_examples)

        ten_ids = {row_key(r) for r in ten_examples}

        assert len(five_examples) == 5
        assert len(ten_examples) == 10
        assert count_labels(five_examples)["trace"] == 3
        assert count_labels(five_examples)["no_trace"] == 2
        assert count_labels(ten_examples)["trace"] == 5
        assert count_labels(ten_examples)["no_trace"] == 5

        assert not (five_ids & test_ids), f"{label}: five-shot overlaps test set"
        assert not (ten_ids & test_ids), f"{label}: ten-shot overlaps test set"
        assert not (five_ids & ten_ids), f"{label}: five-shot overlaps ten-shot"

        five_file = DATA_DIR / f"five_shot_{label}_examples.jsonl"
        ten_file = DATA_DIR / f"ten_shot_{label}_examples.jsonl"

        save_jsonl(five_examples, five_file)
        save_jsonl(ten_examples, ten_file)

        print(
            f"{label}: saved examples | "
            f"five={count_labels(five_examples)} | "
            f"ten={count_labels(ten_examples)} | "
            f"seed={seed}"
        )

    print("Data preparation complete.")
    print("Fixed test set is reused for zero-shot, all 5-shot runs, and all 10-shot runs.")


# -----------------------------
# Prompting and prediction
# -----------------------------

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
    """
    Strictly normalize model output to one of:
    - trace
    - no_trace
    - invalid

    This avoids the old bug where outputs like:
    "not clearly trace"
    were incorrectly mapped to no_trace because they start with "no".
    """
    text = raw_text.strip().lower()
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"[*`\"']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Remove common leading prefixes only.
    text = re.sub(r"^(answer|label|prediction)\s*:\s*", "", text).strip()

    # Exact label outputs.
    if text in {"trace", "trace."}:
        return "trace"

    if text in {"no_trace", "no_trace.", "no trace", "no trace."}:
        return "no_trace"

    # First-token label outputs, e.g. "trace." or "trace because..."
    tokens = re.split(r"[\s,.;:!?]+", text)
    tokens = [t for t in tokens if t]

    if tokens and tokens[0] == "trace":
        return "trace"

    # Accept "no trace" only when the first two tokens are exactly no + trace.
    # Do NOT accept "not ..." as no_trace.
    if len(tokens) >= 2 and tokens[0] == "no" and tokens[1] == "trace":
        return "no_trace"

    if tokens and tokens[0] == "no_trace":
        return "no_trace"

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
    max_prompt_tokens = 8192

    # Measure true prompt length before truncation.
    length_check = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=False,
    )

    prompt_tokens = int(length_check["input_ids"].shape[1])
    prompt_truncated = prompt_tokens > max_prompt_tokens

    if prompt_truncated:
        print(
            f"WARNING: prompt length {prompt_tokens} exceeds "
            f"max_length={max_prompt_tokens}; prompt will be truncated."
        )

    # Actual model input uses truncation to avoid exceeding context limit.
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_tokens,
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
    raw = tokenizer.decode(new_ids, skip_special_tokens=True)

    return raw, prompt_tokens, prompt_truncated


def run_setting(method, run_idx, seed, demo_file, pred_file, tokenizer, model, max_records=None):
    demo_examples = load_jsonl(demo_file)
    test_rows = load_jsonl(TEST_FILE)

    if max_records is not None:
        test_rows = test_rows[:max_records]

    print(f"\nRunning setting: {method} | run={run_idx} | seed={seed}")
    print("Demo examples:", len(demo_examples))
    print("Test records:", len(test_rows))

    rows_out = []

    for i, row in enumerate(test_rows, start=1):
        prompt = build_prompt(demo_examples, row)
        raw_pred, prompt_tokens, prompt_truncated = generate_answer(prompt, tokenizer, model)
        pred_label = normalize_prediction(raw_pred)
        gold_label = get_label(row)
        high, low = get_requirements(row)

        rows_out.append({
            "example_id": row.get("example_id") or f"line_{row.get('_line_id', i)}",
            "gold_label": gold_label,
            "pred_label": pred_label,
            "raw_prediction": raw_pred.strip(),
            "shot_setting": method,
            "run": run_idx,
            "seed": seed,
            "prompt_tokens": prompt_tokens,
            "prompt_truncated": prompt_truncated,
            "higher_level_requirement": high,
            "lower_level_requirement": low,
        })

        if i % 10 == 0 or i == len(test_rows):
            print(f"{method} run={run_idx}: evaluated {i}/{len(test_rows)}")

    pred_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_out).to_csv(pred_file, index=False)
    print("Saved predictions:", pred_file)

    return rows_out


# -----------------------------
# Metrics and summaries
# -----------------------------

def compute_metrics(pred_rows, method, run_idx, seed):
    valid_rows = [r for r in pred_rows if r["pred_label"] in {"trace", "no_trace"}]
    invalid = len(pred_rows) - len(valid_rows)

    if not valid_rows:
        raise RuntimeError(f"No valid predictions for {method} run={run_idx}")

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
        "run": int(run_idx),
        "seed": int(seed),
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


def save_run_summary(summary_rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "method",
        "run",
        "seed",
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

    with RUN_SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            r = dict(row)
            r["confusion_matrix"] = json.dumps(r["confusion_matrix"])
            writer.writerow(r)

    with RUN_SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    print("Saved run summary:", RUN_SUMMARY_CSV)
    print("Saved run summary:", RUN_SUMMARY_JSON)


def make_aggregate_summary(summary_rows):
    metric_keys = [
        "accuracy",
        "trace_precision",
        "trace_recall",
        "trace_f1",
        "no_trace_precision",
        "no_trace_recall",
        "no_trace_f1",
        "invalid_predictions",
    ]

    aggregate_rows = []

    for method in ["zero_shot", "five_shot", "ten_shot"]:
        method_rows = [r for r in summary_rows if r["method"] == method]
        if not method_rows:
            continue

        out = {
            "method": method,
            "num_runs": len(method_rows),
        }

        for key in metric_keys:
            vals = [float(r[key]) for r in method_rows]
            out[f"{key}_mean"] = sum(vals) / len(vals)

            if len(vals) > 1:
                mean = out[f"{key}_mean"]
                variance = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
                out[f"{key}_std"] = variance ** 0.5
            else:
                out[f"{key}_std"] = 0.0

        aggregate_rows.append(out)

    return aggregate_rows


def save_aggregate_summary(aggregate_rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "method",
        "num_runs",
        "accuracy_mean",
        "accuracy_std",
        "trace_precision_mean",
        "trace_precision_std",
        "trace_recall_mean",
        "trace_recall_std",
        "trace_f1_mean",
        "trace_f1_std",
        "no_trace_precision_mean",
        "no_trace_precision_std",
        "no_trace_recall_mean",
        "no_trace_recall_std",
        "no_trace_f1_mean",
        "no_trace_f1_std",
        "invalid_predictions_mean",
        "invalid_predictions_std",
    ]

    with AGGREGATE_SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregate_rows:
            writer.writerow(row)

    with AGGREGATE_SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(aggregate_rows, f, indent=2)

    print("Saved aggregate summary:", AGGREGATE_SUMMARY_CSV)
    print("Saved aggregate summary:", AGGREGATE_SUMMARY_JSON)


def save_notes():
    text = f"""# Few-shot Qwen Multi-run Prompting Experiment

## Experiment

Experiment name: fewshot_qwen35_multi_run  
Model: {MODEL_NAME}  
Fine-tuning: No  
LoRA / QLoRA adapter: No  
Mode: prompt-based classification with generation-based prediction  

## Source dataset

`transformed_data/cross_dataset_pattern1.jsonl`

## Data folders

- `data/fewshot_qwen35_multi_run/`
- `results/fewshot_qwen35_multi_run/`

## Test set

Fixed test set:

- file: `data/fewshot_qwen35_multi_run/test_100.jsonl`
- seed: {TEST_SEED}
- size: 100 examples
- balance: 50 trace + 50 no_trace

The same fixed test set is used for zero-shot, all 5-shot runs, and all 10-shot runs.

## Prompt-example runs

Run seeds:

`{RUN_SEEDS}`

### Zero-shot

- run once
- no examples in prompt

### Five-shot

- 10 runs
- each run uses different prompt examples
- each run has 3 trace + 2 no_trace examples

### Ten-shot

- 10 runs
- each run uses different prompt examples
- each run has 5 trace + 5 no_trace examples

Within each run, 5-shot and 10-shot examples are disjoint.
Prompt examples do not overlap with the fixed test set.

## Prediction settings

- generation-based prediction
- max_new_tokens: 8
- do_sample: False
- temperature: deterministic generation; sampling disabled

## Output files

Per-run predictions:

- `zero_shot_predictions.csv`
- `five_shot_run_01_predictions.csv` ... `five_shot_run_10_predictions.csv`
- `ten_shot_run_01_predictions.csv` ... `ten_shot_run_10_predictions.csv`

Summaries:

- `fewshot_run_summary.csv`
- `fewshot_run_summary.json`
- `fewshot_aggregate_summary.csv`
- `fewshot_aggregate_summary.json`

## Purpose

This experiment checks whether few-shot traceability performance is stable when the prompt examples change while the test set stays fixed.
"""

    NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
    NOTES_FILE.write_text(text, encoding="utf-8")
    print("Saved notes:", NOTES_FILE)


# -----------------------------
# Main execution
# -----------------------------

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

    summary_rows = []

    # Zero-shot is run once.
    zero_preds = run_setting(
        method="zero_shot",
        run_idx=0,
        seed=TEST_SEED,
        demo_file=ZERO_SHOT_FILE,
        pred_file=ZERO_PRED_FILE,
        tokenizer=tokenizer,
        model=model,
        max_records=args.max_records,
    )
    summary_rows.append(compute_metrics(zero_preds, "zero_shot", 0, TEST_SEED))

    # Five-shot and ten-shot multi-runs.
    for run_idx, seed in enumerate(RUN_SEEDS, start=1):
        label = run_label(run_idx)

        five_demo_file = DATA_DIR / f"five_shot_{label}_examples.jsonl"
        ten_demo_file = DATA_DIR / f"ten_shot_{label}_examples.jsonl"

        five_pred_file = RESULTS_DIR / f"five_shot_{label}_predictions.csv"
        ten_pred_file = RESULTS_DIR / f"ten_shot_{label}_predictions.csv"

        five_preds = run_setting(
            method="five_shot",
            run_idx=run_idx,
            seed=seed,
            demo_file=five_demo_file,
            pred_file=five_pred_file,
            tokenizer=tokenizer,
            model=model,
            max_records=args.max_records,
        )
        summary_rows.append(compute_metrics(five_preds, "five_shot", run_idx, seed))

        ten_preds = run_setting(
            method="ten_shot",
            run_idx=run_idx,
            seed=seed,
            demo_file=ten_demo_file,
            pred_file=ten_pred_file,
            tokenizer=tokenizer,
            model=model,
            max_records=args.max_records,
        )
        summary_rows.append(compute_metrics(ten_preds, "ten_shot", run_idx, seed))

    save_run_summary(summary_rows)

    aggregate_rows = make_aggregate_summary(summary_rows)
    save_aggregate_summary(aggregate_rows)
    save_notes()

    print("\nAggregate summary")
    print("-" * 100)
    for row in aggregate_rows:
        print(
            f"{row['method']} | "
            f"runs={row['num_runs']} | "
            f"acc={row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f} | "
            f"trace_f1={row['trace_f1_mean']:.4f} ± {row['trace_f1_std']:.4f} | "
            f"trace_recall={row['trace_recall_mean']:.4f} ± {row['trace_recall_std']:.4f} | "
            f"no_trace_f1={row['no_trace_f1_mean']:.4f} ± {row['no_trace_f1_std']:.4f}"
        )


if __name__ == "__main__":
    main()
