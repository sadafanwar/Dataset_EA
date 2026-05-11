import json
from pathlib import Path
from collections import Counter

from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]

SOURCE_FILE = ROOT / "transformed_data" / "cross_dataset_pattern1.jsonl"

OUTPUT_DIR = ROOT / "data" / "qwen35_oversampling_10fold_cv"
FULL_WITH_IDS_FILE = OUTPUT_DIR / "full_dataset_with_ids.jsonl"
FOLDS_DIR = OUTPUT_DIR / "folds"

SEED = 42
N_SPLITS = 10


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


def add_example_ids(records):
    records_with_ids = []
    for i, record in enumerate(records):
        new_record = dict(record)
        new_record["example_id"] = f"ex_{i:06d}"
        records_with_ids.append(new_record)
    return records_with_ids


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


def main():
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Missing source file: {SOURCE_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FOLDS_DIR.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(SOURCE_FILE)
    records = add_example_ids(records)
    labels = [r["output"] for r in records]

    print("Full dataset")
    print("Total:", len(records))
    print("Counts:", count_labels(records))
    print()

    save_jsonl(FULL_WITH_IDS_FILE, records)

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )

    all_test_ids = []

    for fold_index, (train_idx, test_idx) in enumerate(
        skf.split(records, labels),
        start=1,
    ):
        fold_name = f"fold_{fold_index:02d}"
        fold_dir = FOLDS_DIR / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_records = [records[i] for i in train_idx]
        test_records = [records[i] for i in test_idx]

        check_no_overlap(
            f"{fold_name}_train",
            train_records,
            f"{fold_name}_test",
            test_records,
        )

        save_jsonl(fold_dir / "train_pool.jsonl", train_records)
        save_jsonl(fold_dir / "test.jsonl", test_records)

        all_test_ids.extend([r["example_id"] for r in test_records])

        print()
        print(f"{fold_name}")
        print("Train pool total:", len(train_records))
        print("Train pool counts:", count_labels(train_records))
        print("Test total:", len(test_records))
        print("Test counts:", count_labels(test_records))

    test_id_counts = Counter(all_test_ids)
    repeated_test_ids = [
        example_id
        for example_id, count in test_id_counts.items()
        if count != 1
    ]

    if repeated_test_ids:
        raise ValueError(
            "Some examples do not appear exactly once as CV test examples."
        )

    if len(all_test_ids) != len(records):
        raise ValueError(
            "CV test coverage does not match full dataset size."
        )

    print()
    print("10-fold CV coverage check passed.")
    print("Every example appears exactly once as a test example.")
    print()
    print("Created:")
    print(FULL_WITH_IDS_FILE.relative_to(ROOT))
    print(FOLDS_DIR.relative_to(ROOT))


if __name__ == "__main__":
    main()
