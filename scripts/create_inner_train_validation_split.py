#!/usr/bin/env python3
"""
Create inner train/validation split from each CV fold's train_pool.

Correct methodology:
- test.jsonl remains untouched
- train_pool.jsonl is split into inner_train.jsonl and validation.jsonl
- validation is used only for training-time monitoring
- test is used only after training for final classification evaluation
"""

import argparse
import json
from pathlib import Path
from collections import Counter

from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]

BASE_DIR = ROOT / "data" / "qwen35_oversampling_10fold_cv"
FOLDS_DIR = BASE_DIR / "folds"
INNER_SPLITS_DIR = BASE_DIR / "inner_splits"

SEED = 42
VALIDATION_SIZE = 0.10


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


def create_inner_split_for_fold(fold: int):
    fold_name = f"fold_{fold:02d}"

    train_pool_file = FOLDS_DIR / fold_name / "train_pool.jsonl"
    test_file = FOLDS_DIR / fold_name / "test.jsonl"

    if not train_pool_file.exists():
        raise FileNotFoundError(f"Missing train pool file: {train_pool_file}")

    if not test_file.exists():
        raise FileNotFoundError(f"Missing test file: {test_file}")

    train_pool = load_jsonl(train_pool_file)
    test_records = load_jsonl(test_file)

    labels = [r["output"] for r in train_pool]

    inner_train, validation = train_test_split(
        train_pool,
        test_size=VALIDATION_SIZE,
        random_state=SEED + fold,
        shuffle=True,
        stratify=labels,
    )

    print(f"\n{fold_name}")
    print("Train pool total:", len(train_pool))
    print("Train pool counts:", count_labels(train_pool))
    print("Inner train total:", len(inner_train))
    print("Inner train counts:", count_labels(inner_train))
    print("Validation total:", len(validation))
    print("Validation counts:", count_labels(validation))
    print("Test total:", len(test_records))
    print("Test counts:", count_labels(test_records))
    print()

    check_no_overlap(f"{fold_name}_inner_train", inner_train, f"{fold_name}_validation", validation)
    check_no_overlap(f"{fold_name}_inner_train", inner_train, f"{fold_name}_test", test_records)
    check_no_overlap(f"{fold_name}_validation", validation, f"{fold_name}_test", test_records)

    output_dir = INNER_SPLITS_DIR / fold_name
    save_jsonl(output_dir / "inner_train.jsonl", inner_train)
    save_jsonl(output_dir / "validation.jsonl", validation)

    print("Created:")
    print((output_dir / "inner_train.jsonl").relative_to(ROOT))
    print((output_dir / "validation.jsonl").relative_to(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        help="Fold number, e.g. 1. Use 1 to redo Fold 01 correctly.",
    )
    args = parser.parse_args()

    create_inner_split_for_fold(args.fold)


if __name__ == "__main__":
    main()
