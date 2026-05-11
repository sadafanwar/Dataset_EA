import json
import random
import argparse
from pathlib import Path
from collections import Counter


ROOT = Path(__file__).resolve().parents[1]

BASE_DIR = ROOT / "data" / "qwen35_oversampling_10fold_cv"
FOLDS_DIR = BASE_DIR / "folds"
OVERSAMPLED_DIR = BASE_DIR / "oversampled_folds"

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


def check_no_overlap(train_records, test_records):
    train_ids = {r["example_id"] for r in train_records}
    test_ids = {r["example_id"] for r in test_records}

    overlap = train_ids.intersection(test_ids)

    if overlap:
        raise ValueError(
            f"Leakage detected: oversampled train overlaps test fold. "
            f"Overlap size: {len(overlap)}"
        )

    print("Leakage check passed.")
    print("No overlap between oversampled train and test fold.")


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
        balanced_trace = random.choices(trace_records, k=target_count)
    else:
        balanced_trace = trace_records

    if len(no_trace_records) < target_count:
        balanced_no_trace = random.choices(no_trace_records, k=target_count)
    else:
        balanced_no_trace = no_trace_records

    balanced_records = balanced_trace + balanced_no_trace
    random.shuffle(balanced_records)

    return balanced_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, help="Fold number, e.g. 1")
    args = parser.parse_args()

    fold_name = f"fold_{args.fold:02d}"

    train_pool_file = FOLDS_DIR / fold_name / "train_pool.jsonl"
    test_file = FOLDS_DIR / fold_name / "test.jsonl"

    if not train_pool_file.exists():
        raise FileNotFoundError(f"Missing train pool file: {train_pool_file}")

    if not test_file.exists():
        raise FileNotFoundError(f"Missing test file: {test_file}")

    train_pool = load_jsonl(train_pool_file)
    test_records = load_jsonl(test_file)

    print(f"Creating oversampled train file for {fold_name}")
    print()

    print("Before oversampling")
    print("Train total:", len(train_pool))
    print("Train counts:", count_labels(train_pool))
    print("Test total:", len(test_records))
    print("Test counts:", count_labels(test_records))

    oversampled_train = oversample(train_pool, seed=SEED + args.fold)

    print()
    print("After oversampling")
    print("Train total:", len(oversampled_train))
    print("Train counts:", count_labels(oversampled_train))

    print()
    check_no_overlap(oversampled_train, test_records)

    output_file = OVERSAMPLED_DIR / fold_name / "train_oversampled.jsonl"
    save_jsonl(output_file, oversampled_train)

    print()
    print("Created:")
    print(output_file.relative_to(ROOT))


if __name__ == "__main__":
    main()
