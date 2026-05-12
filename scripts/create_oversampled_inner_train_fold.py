#!/usr/bin/env python3
"""
Oversample only the inner_train split for a CV fold.

Correct methodology:
- inner_train is oversampled
- validation is not oversampled
- test is not oversampled
"""

import argparse
import json
import random
from pathlib import Path
from collections import Counter


ROOT = Path(__file__).resolve().parents[1]

BASE_DIR = ROOT / "data" / "qwen35_oversampling_10fold_cv"
INNER_SPLITS_DIR = BASE_DIR / "inner_splits"
FOLDS_DIR = BASE_DIR / "folds"

SEED = 42


def load_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def count_labels(records):
    return Counter(r["output"] for r in records)


def check_no_overlap(name_a, records_a, name_b, records_b):
    ids_a = {r["example_id"] for r in records_a}
    ids_b = {r["example_id"] for r in records_b}
    overlap = ids_a.intersection(ids_b)

    if overlap:
        raise ValueError(
            f"Leakage detected between {name_a} and {name_b}. "
            f"Overlap size: {len(overlap)}"
        )

    print(f"No overlap between {name_a} and {name_b}")


def oversample(records, seed):
    random.seed(seed)

    trace_records = [r for r in records if r["output"] == "trace"]
    no_trace_records = [r for r in records if r["output"] == "no_trace"]

    if not trace_records:
        raise ValueError("No trace records found.")
    if not no_trace_records:
        raise ValueError("No no_trace records found.")

    target_count = max(len(trace_records), len(no_trace_records))

    if len(trace_records) < target_count:
        trace_balanced = random.choices(trace_records, k=target_count)
    else:
        trace_balanced = trace_records

    if len(no_trace_records) < target_count:
        no_trace_balanced = random.choices(no_trace_records, k=target_count)
    else:
        no_trace_balanced = no_trace_records

    balanced = trace_balanced + no_trace_balanced
    random.shuffle(balanced)

    return balanced


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()

    fold_name = f"fold_{args.fold:02d}"

    inner_train_file = INNER_SPLITS_DIR / fold_name / "inner_train.jsonl"
    validation_file = INNER_SPLITS_DIR / fold_name / "validation.jsonl"
    test_file = FOLDS_DIR / fold_name / "test.jsonl"

    if not inner_train_file.exists():
        raise FileNotFoundError(f"Missing inner train file: {inner_train_file}")
    if not validation_file.exists():
        raise FileNotFoundError(f"Missing validation file: {validation_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Missing test file: {test_file}")

    inner_train = load_jsonl(inner_train_file)
    validation = load_jsonl(validation_file)
    test_records = load_jsonl(test_file)

    print(f"Creating oversampled inner train file for {fold_name}")
    print()

    print("Before oversampling")
    print("Inner train total:", len(inner_train))
    print("Inner train counts:", count_labels(inner_train))
    print("Validation total:", len(validation))
    print("Validation counts:", count_labels(validation))
    print("Test total:", len(test_records))
    print("Test counts:", count_labels(test_records))

    oversampled_train = oversample(inner_train, seed=SEED + args.fold)

    print()
    print("After oversampling")
    print("Oversampled train total:", len(oversampled_train))
    print("Oversampled train counts:", count_labels(oversampled_train))
    print()

    check_no_overlap(f"{fold_name}_oversampled_train", oversampled_train, f"{fold_name}_validation", validation)
    check_no_overlap(f"{fold_name}_oversampled_train", oversampled_train, f"{fold_name}_test", test_records)
    check_no_overlap(f"{fold_name}_validation", validation, f"{fold_name}_test", test_records)

    output_file = INNER_SPLITS_DIR / fold_name / "train_oversampled.jsonl"
    save_jsonl(output_file, oversampled_train)

    print()
    print("Created:")
    print(output_file.relative_to(ROOT))


if __name__ == "__main__":
    main()
