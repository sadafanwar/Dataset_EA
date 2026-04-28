import csv
from pathlib import Path
from collections import defaultdict

# -----------------------------
# 1. File paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "original_data" / "Cross_dataset.csv"
OUTPUT_FILE = BASE_DIR / "transformed_data" / "duplicate_rows_with_line_numbers.csv"


# -----------------------------
# 2. Read CSV without pandas
# -----------------------------

print("Reading CSV file...", flush=True)

rows = []

with open(INPUT_FILE, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)

    print("Columns found:", reader.fieldnames, flush=True)

    required_columns = ["high_text", "low_text", "label"]

    for col in required_columns:
        if col not in reader.fieldnames:
            raise ValueError(f"Missing required column: {col}")

    for row_number, row in enumerate(reader, start=2):
        high_text = " ".join(row["high_text"].split())
        low_text = " ".join(row["low_text"].split())
        label = str(row["label"]).strip()

        rows.append({
            "csv_row_number": row_number,
            "high_text": high_text,
            "low_text": low_text,
            "label": label
        })

print("Total rows read:", len(rows), flush=True)


# -----------------------------
# 3. Group rows by exact content
# -----------------------------

groups = defaultdict(list)

for row in rows:
    key = (
        row["high_text"],
        row["low_text"],
        row["label"]
    )
    groups[key].append(row)


# -----------------------------
# 4. Find duplicates
# -----------------------------

duplicate_rows = []
duplicate_group_id = 1
removable_duplicates = 0

print("\nDuplicate groups:", flush=True)

for key, group_rows in groups.items():
    if len(group_rows) > 1:
        line_numbers = [r["csv_row_number"] for r in group_rows]
        label = group_rows[0]["label"]

        print(
            f"Group {duplicate_group_id} | Label: {label} | CSV rows: {line_numbers}",
            flush=True
        )

        removable_duplicates += len(group_rows) - 1

        for r in group_rows:
            duplicate_rows.append({
                "duplicate_group_id": duplicate_group_id,
                "csv_row_number": r["csv_row_number"],
                "label": r["label"],
                "high_text": r["high_text"],
                "low_text": r["low_text"]
            })

        duplicate_group_id += 1


# -----------------------------
# 5. Save duplicate rows
# -----------------------------

with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
    fieldnames = [
        "duplicate_group_id",
        "csv_row_number",
        "label",
        "high_text",
        "low_text"
    ]

    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(duplicate_rows)


# -----------------------------
# 6. Final summary
# -----------------------------

print("\nSummary:", flush=True)
print("Original rows:", len(rows), flush=True)
print("Duplicate rows shown:", len(duplicate_rows), flush=True)
print("Duplicate rows that would be removed:", removable_duplicates, flush=True)

print("\nSaved duplicate rows with line numbers to:", flush=True)
print(OUTPUT_FILE, flush=True)