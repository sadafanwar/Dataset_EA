import json
import random
from pathlib import Path
from collections import Counter, defaultdict


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "traceability" / "cross_dataset_pattern1.jsonl"

OUTPUT_DIR = BASE_DIR / "data" / "traceability"
TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
VALIDATION_FILE = OUTPUT_DIR / "validation.jsonl"
TEST_FILE = OUTPUT_DIR / "test.jsonl"

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def save_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def count_labels(records):
    return Counter(record["output"] for record in records)


def get_high_level_requirements(records):
    return set(
        record["input"]["higher_level_requirement"]
        for record in records
    )


def main():
    records = load_jsonl(INPUT_FILE)

    print("Total records loaded:", len(records))
    print("Overall label counts:", dict(count_labels(records)))

    # Group by higher-level requirement to reduce leakage
    groups = defaultdict(list)

    for record in records:
        high_req = record["input"]["higher_level_requirement"]
        groups[high_req].append(record)

    group_items = list(groups.items())

    print("Unique higher-level requirements:", len(group_items))

    random.seed(RANDOM_SEED)
    random.shuffle(group_items)

    total_records = len(records)

    target_train = int(total_records * TRAIN_RATIO)
    target_validation = int(total_records * VALIDATION_RATIO)

    train_records = []
    validation_records = []
    test_records = []

    for high_req, group_records in group_items:
        if len(train_records) < target_train:
            train_records.extend(group_records)
        elif len(validation_records) < target_validation:
            validation_records.extend(group_records)
        else:
            test_records.extend(group_records)

    save_jsonl(TRAIN_FILE, train_records)
    save_jsonl(VALIDATION_FILE, validation_records)
    save_jsonl(TEST_FILE, test_records)

    print("\nSplit completed.")

    print("\nTrain:")
    print("Total:", len(train_records))
    print("Labels:", dict(count_labels(train_records)))

    print("\nValidation:")
    print("Total:", len(validation_records))
    print("Labels:", dict(count_labels(validation_records)))

    print("\nTest:")
    print("Total:", len(test_records))
    print("Labels:", dict(count_labels(test_records)))

    # Leakage check
    train_highs = get_high_level_requirements(train_records)
    validation_highs = get_high_level_requirements(validation_records)
    test_highs = get_high_level_requirements(test_records)

    train_validation_overlap = train_highs.intersection(validation_highs)
    train_test_overlap = train_highs.intersection(test_highs)
    validation_test_overlap = validation_highs.intersection(test_highs)

    print("\nLeakage check by higher-level requirement:")
    print("Train/Validation overlap:", len(train_validation_overlap))
    print("Train/Test overlap:", len(train_test_overlap))
    print("Validation/Test overlap:", len(validation_test_overlap))

    if (
        len(train_validation_overlap) == 0
        and len(train_test_overlap) == 0
        and len(validation_test_overlap) == 0
    ):
        print("Status: PASS - No higher-level requirement overlap across splits.")
    else:
        print("Status: WARNING - Overlap found.")

    print("\nFiles saved:")
    print(TRAIN_FILE)
    print(VALIDATION_FILE)
    print(TEST_FILE)


if __name__ == "__main__":
    main()