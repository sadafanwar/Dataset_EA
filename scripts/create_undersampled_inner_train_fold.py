#!/usr/bin/env python3

import argparse
import json
import random
from pathlib import Path
from collections import Counter, defaultdict


def read_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def get_label(record):
    for key in ["output", "label", "true_label", "target"]:
        if key in record:
            value = str(record[key]).strip().lower()
            if value in {"trace", "no_trace"}:
                return value
    raise ValueError(f"Could not find trace/no_trace label in record keys: {list(record.keys())}")


def get_record_id(record):
    for key in ["example_id", "id", "Requirement_ID", "pair_id"]:
        if key in record:
            return str(record[key])
    return json.dumps(record, sort_keys=True, ensure_ascii=False)


def assert_no_overlap(name_a, records_a, name_b, records_b):
    ids_a = {get_record_id(r) for r in records_a}
    ids_b = {get_record_id(r) for r in records_b}
    overlap = ids_a & ids_b

    if overlap:
        examples = list(overlap)[:10]
        raise RuntimeError(
            f"Overlap found between {name_a} and {name_b}: {len(overlap)} examples. "
            f"Examples: {examples}"
        )

    print(f"No overlap between {name_a} and {name_b}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fold_name = f"fold_{args.fold:02d}"

    base = Path("data/qwen35_oversampling_10fold_cv")
    inner_dir = base / "inner_splits" / fold_name
    fold_dir = base / "folds" / fold_name

    inner_train_path = inner_dir / "inner_train.jsonl"
    validation_path = inner_dir / "validation.jsonl"
    test_path = fold_dir / "test.jsonl"
    output_path = inner_dir / "train_undersampled.jsonl"

    if not inner_train_path.exists():
        raise FileNotFoundError(f"Missing inner train file: {inner_train_path}")
    if not validation_path.exists():
        raise FileNotFoundError(f"Missing validation file: {validation_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test file: {test_path}")

    rng = random.Random(args.seed + args.fold)

    inner_train = read_jsonl(inner_train_path)
    validation = read_jsonl(validation_path)
    test = read_jsonl(test_path)

    grouped = defaultdict(list)
    for r in inner_train:
        grouped[get_label(r)].append(r)

    before_counts = Counter(get_label(r) for r in inner_train)

    if set(grouped.keys()) != {"trace", "no_trace"}:
        raise ValueError(f"Expected labels trace/no_trace, found: {set(grouped.keys())}")

    minority_count = min(len(grouped["trace"]), len(grouped["no_trace"]))

    trace_records = grouped["trace"]
    no_trace_records = grouped["no_trace"]

    if len(trace_records) > minority_count:
        trace_records = rng.sample(trace_records, minority_count)

    if len(no_trace_records) > minority_count:
        no_trace_records = rng.sample(no_trace_records, minority_count)

    undersampled = trace_records + no_trace_records
    rng.shuffle(undersampled)

    after_counts = Counter(get_label(r) for r in undersampled)

    print(f"\n{fold_name}")
    print("\nBefore undersampling")
    print(f"Inner train total: {len(inner_train)}")
    print(f"Inner train counts: {before_counts}")
    print(f"Validation total: {len(validation)}")
    print(f"Validation counts: {Counter(get_label(r) for r in validation)}")
    print(f"Test total: {len(test)}")
    print(f"Test counts: {Counter(get_label(r) for r in test)}")

    print("\nAfter undersampling")
    print(f"Undersampled train total: {len(undersampled)}")
    print(f"Undersampled train counts: {after_counts}")

    assert after_counts["trace"] == after_counts["no_trace"], "Classes are not balanced after undersampling."

    print()
    assert_no_overlap(f"{fold_name}_undersampled_train", undersampled, f"{fold_name}_validation", validation)
    assert_no_overlap(f"{fold_name}_undersampled_train", undersampled, f"{fold_name}_test", test)
    assert_no_overlap(f"{fold_name}_validation", validation, f"{fold_name}_test", test)

    write_jsonl(output_path, undersampled)

    print("\nCreated:")
    print(output_path)


if __name__ == "__main__":
    main()
