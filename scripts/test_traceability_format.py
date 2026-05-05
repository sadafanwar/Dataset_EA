import json
from pathlib import Path
from collections import Counter


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "traceability"

FILES = {
    "train": DATA_DIR / "train.jsonl",
    "validation": DATA_DIR / "validation.jsonl",
    "test": DATA_DIR / "test.jsonl",
}


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at {path}, line {line_number}: {e}")

    return records


def build_training_text(record):
    instruction = record["instruction"]
    input_obj = record["input"]
    output = record["output"]

    higher_req = input_obj["higher_level_requirement"]
    lower_req = input_obj["lower_level_requirement"]

    prompt = (
        f"{instruction}\n\n"
        f"Higher-level requirement:\n{higher_req}\n\n"
        f"Lower-level requirement:\n{lower_req}\n\n"
        f"Answer:"
    )

    return f"{prompt} {output}"


def validate_record(record, index, split_name):
    required_top_keys = {"instruction", "input", "output"}

    missing_keys = required_top_keys - set(record.keys())
    if missing_keys:
        raise ValueError(
            f"{split_name} example {index} is missing keys: {missing_keys}"
        )

    if not isinstance(record["input"], dict):
        raise ValueError(f"{split_name} example {index}: input must be an object")

    required_input_keys = {
        "higher_level_requirement",
        "lower_level_requirement",
    }

    missing_input_keys = required_input_keys - set(record["input"].keys())
    if missing_input_keys:
        raise ValueError(
            f"{split_name} example {index} input missing keys: {missing_input_keys}"
        )

    if record["output"] not in {"trace", "no_trace"}:
        raise ValueError(
            f"{split_name} example {index}: invalid output label: {record['output']}"
        )

    training_text = build_training_text(record)

    if not training_text.strip():
        raise ValueError(f"{split_name} example {index}: empty training text")

    return training_text


def main():
    print("Traceability dataset format dry-run")
    print("=" * 60)

    for split_name, file_path in FILES.items():
        print(f"\nChecking {split_name}: {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")

        records = load_jsonl(file_path)

        label_counts = Counter(record["output"] for record in records)

        print("Total records:", len(records))
        print("Label counts:", dict(label_counts))

        for index, record in enumerate(records, start=1):
            validate_record(record, index, split_name)

        print("Status: PASS")

    # Show one formatted training example
    train_records = load_jsonl(FILES["train"])
    example_text = build_training_text(train_records[0])

    print("\n" + "=" * 60)
    print("Formatted training example preview")
    print("=" * 60)
    print(example_text[:1500])

    print("\n" + "=" * 60)
    print("Dry-run completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()