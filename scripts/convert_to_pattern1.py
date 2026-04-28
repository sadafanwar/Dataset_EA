import pandas as pd
import json
from pathlib import Path


# -----------------------------
# 1. File paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "original_data" / "Cross_dataset.csv"

OUTPUT_JSON = BASE_DIR / "transformed_data" / "cross_dataset_pattern1.json"
OUTPUT_JSONL = BASE_DIR / "transformed_data" / "cross_dataset_pattern1.jsonl"


# -----------------------------
# 2. Helper functions
# -----------------------------

def clean_text(text):
    """
    Clean requirement text by removing extra spaces and line breaks.
    """
    return " ".join(str(text).split())


def map_label(label):
    """
    Convert original dataset labels to Pattern 1 output labels.
    """
    label = int(label)

    if label == 1:
        return "trace"
    elif label == 0:
        return "no_trace"
    else:
        raise ValueError(f"Unexpected label found: {label}")


# -----------------------------
# 3. Load dataset
# -----------------------------

df = pd.read_csv(INPUT_FILE)

print("Original dataset loaded.")
print("Rows:", len(df))
print("Columns:", list(df.columns))


# -----------------------------
# 4. Basic validation
# -----------------------------

required_columns = ["high_text", "low_text", "label"]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

print("Required columns are present.")


# -----------------------------
# 5. Clean dataset
# -----------------------------

df = df.dropna(subset=["high_text", "low_text", "label"])
df = df.drop_duplicates()

print("After removing empty and duplicate rows:", len(df))


# -----------------------------
# 6. Convert rows to Pattern 1 format
# -----------------------------

records = []

instruction = "Determine whether the lower-level requirement traces to the higher-level requirement."

for index, row in df.iterrows():
    high_req = clean_text(row["high_text"])
    low_req = clean_text(row["low_text"])
    output_label = map_label(row["label"])

    record = {
        "instruction": instruction,
        "input": {
            "higher_level_requirement": high_req,
            "lower_level_requirement": low_req
        },
        "output": output_label
    }

    records.append(record)


# -----------------------------
# 7. Save JSON file
# -----------------------------

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print("JSON file saved:", OUTPUT_JSON)


# -----------------------------
# 8. Save JSONL file
# -----------------------------

with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("JSONL file saved:", OUTPUT_JSONL)


# -----------------------------
# 9. Final summary
# -----------------------------

trace_count = sum(1 for r in records if r["output"] == "trace")
no_trace_count = sum(1 for r in records if r["output"] == "no_trace")

print("\nConversion completed.")
print("Total converted examples:", len(records))
print("Trace examples:", trace_count)
print("No-trace examples:", no_trace_count)