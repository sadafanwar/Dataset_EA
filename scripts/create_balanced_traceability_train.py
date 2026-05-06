import json
import random
from collections import Counter, defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "traceability" / "train.jsonl"
OUTPUT_FILE = BASE_DIR / "data" / "traceability" / "train_balanced_oversampled.jsonl"
RANDOM_SEED = 42


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    random.seed(RANDOM_SEED)
    records = load_jsonl(INPUT_FILE)

    label_to_records = defaultdict(list)
    for record in records:
        label_to_records[record["output"]].append(record)

    label_counts = Counter({label: len(items) for label, items in label_to_records.items()})
    target_count = max(label_counts.values())

    balanced_records = []
    for label, items in label_to_records.items():
        if len(items) < target_count:
            extra = random.choices(items, k=target_count - len(items))
            items = items + extra
        balanced_records.extend(items)

    random.shuffle(balanced_records)
    save_jsonl(OUTPUT_FILE, balanced_records)

    balanced_counts = Counter(record["output"] for record in balanced_records)

    print("Input file:", INPUT_FILE)
    print("Output file:", OUTPUT_FILE)
    print("Original label counts:", dict(label_counts))
    print("Balanced label counts:", dict(balanced_counts))
    print("Total balanced records:", len(balanced_records))


if __name__ == "__main__":
    main()
