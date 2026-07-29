import csv
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.model_selection import StratifiedKFold, train_test_split


REPO_ROOT = Path(__file__).resolve().parents[2]

SOURCE_DATASET = (
    REPO_ROOT
    / "data"
    / "traceability_dataset"
    / "traceability_pairs_with_rationale.jsonl"
)

OUTPUT_ROOT = (
    REPO_ROOT
    / "data"
    / "exp17_qwen35_9b_rationale_single_template"
)

N_OUTER_FOLDS = 10
VALIDATION_FRACTION = 0.10
RANDOM_SEED = 42

ID_FIELD = "pair_id"
LABEL_FIELD = "label"
RATIONALE_FIELD = "rationale"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {error}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object at line {line_number}"
                )

            records.append(record)

    return records


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_source(records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError("Source dataset is empty")

    required_fields = {
        ID_FIELD,
        LABEL_FIELD,
        RATIONALE_FIELD,
    }

    seen_ids: set[str] = set()

    for index, record in enumerate(records, start=1):
        missing = required_fields - set(record)

        if missing:
            raise ValueError(
                f"Record {index} is missing fields: {sorted(missing)}"
            )

        pair_id = str(record[ID_FIELD]).strip()

        if not pair_id:
            raise ValueError(
                f"Record {index} has an empty {ID_FIELD}"
            )

        if pair_id in seen_ids:
            raise ValueError(
                f"Duplicate {ID_FIELD}: {pair_id}"
            )

        seen_ids.add(pair_id)

        label = str(record[LABEL_FIELD]).strip()

        if not label:
            raise ValueError(
                f"Record {index} has an empty {LABEL_FIELD}"
            )

        rationale = str(record[RATIONALE_FIELD]).strip()

        if not rationale:
            raise ValueError(
                f"Record {index} has an empty {RATIONALE_FIELD}"
            )


def get_ids(records: list[dict[str, Any]]) -> set[str]:
    return {
        str(record[ID_FIELD]).strip()
        for record in records
    }


def check_disjoint(
    fold_name: str,
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
) -> None:
    train_ids = get_ids(train_records)
    validation_ids = get_ids(validation_records)
    test_ids = get_ids(test_records)

    overlaps = {
        "train-validation": train_ids & validation_ids,
        "train-test": train_ids & test_ids,
        "validation-test": validation_ids & test_ids,
    }

    for split_pair, overlap in overlaps.items():
        if overlap:
            raise ValueError(
                f"{fold_name}: leakage in {split_pair}: "
                f"{sorted(overlap)[:5]}"
            )


def count_labels(
    records: list[dict[str, Any]],
) -> Counter[str]:
    return Counter(
        str(record[LABEL_FIELD])
        for record in records
    )


def main() -> int:
    print("=== Prepare new EXP-17 rationale folds ===")
    print(f"Source dataset : {SOURCE_DATASET}")
    print(f"Output root    : {OUTPUT_ROOT}")
    print(f"Outer folds    : {N_OUTER_FOLDS}")
    print(f"Validation     : {VALIDATION_FRACTION}")
    print(f"Random seed    : {RANDOM_SEED}")

    if not SOURCE_DATASET.is_file():
        print(f"ERROR: Dataset not found: {SOURCE_DATASET}")
        return 1

    records = read_jsonl(SOURCE_DATASET)
    validate_source(records)

    labels = [
        str(record[LABEL_FIELD])
        for record in records
    ]

    print(f"Total records  : {len(records)}")
    print(f"Label counts   : {dict(Counter(labels))}")

    temporary_root = OUTPUT_ROOT.with_name(
        OUTPUT_ROOT.name + "_temporary"
    )

    if temporary_root.exists():
        shutil.rmtree(temporary_root)

    temporary_root.mkdir(parents=True)

    outer_cv = StratifiedKFold(
        n_splits=N_OUTER_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    summary_rows: list[dict[str, Any]] = []
    all_test_ids: list[str] = []

    try:
        for fold_number, (
            train_pool_indices,
            test_indices,
        ) in enumerate(
            outer_cv.split(records, labels),
            start=1,
        ):
            fold_name = f"fold_{fold_number:02d}"

            train_pool = [
                records[index]
                for index in train_pool_indices
            ]

            test_records = [
                records[index]
                for index in test_indices
            ]

            train_pool_labels = [
                str(record[LABEL_FIELD])
                for record in train_pool
            ]

            inner_train, validation_records = train_test_split(
                train_pool,
                test_size=VALIDATION_FRACTION,
                random_state=RANDOM_SEED + fold_number,
                shuffle=True,
                stratify=train_pool_labels,
            )

            random.Random(
                RANDOM_SEED + fold_number
            ).shuffle(inner_train)

            check_disjoint(
                fold_name,
                inner_train,
                validation_records,
                test_records,
            )

            fold_root = (
                temporary_root
                / "folds"
                / fold_name
            )

            write_jsonl(
                fold_root / "train.jsonl",
                inner_train,
            )

            write_jsonl(
                fold_root / "validation.jsonl",
                validation_records,
            )

            write_jsonl(
                fold_root / "test.jsonl",
                test_records,
            )

            train_counts = count_labels(inner_train)
            validation_counts = count_labels(
                validation_records
            )
            test_counts = count_labels(test_records)

            all_test_ids.extend(
                str(record[ID_FIELD])
                for record in test_records
            )

            summary_rows.append(
                {
                    "fold": fold_name,
                    "train_records": len(inner_train),
                    "validation_records": len(
                        validation_records
                    ),
                    "test_records": len(test_records),
                    "train_trace": train_counts.get(
                        "trace",
                        0,
                    ),
                    "train_no_trace": train_counts.get(
                        "no_trace",
                        0,
                    ),
                    "validation_trace": validation_counts.get(
                        "trace",
                        0,
                    ),
                    "validation_no_trace": validation_counts.get(
                        "no_trace",
                        0,
                    ),
                    "test_trace": test_counts.get(
                        "trace",
                        0,
                    ),
                    "test_no_trace": test_counts.get(
                        "no_trace",
                        0,
                    ),
                    "status": "validated",
                }
            )

            print(
                f"{fold_name}: "
                f"train={len(inner_train)}, "
                f"validation={len(validation_records)}, "
                f"test={len(test_records)}"
            )

        source_ids = get_ids(records)

        if len(all_test_ids) != len(records):
            raise ValueError(
                "Outer test coverage count does not match dataset size"
            )

        if len(set(all_test_ids)) != len(records):
            raise ValueError(
                "Some records occur in more than one outer test fold"
            )

        if set(all_test_ids) != source_ids:
            raise ValueError(
                "Outer test folds do not cover the complete dataset"
            )

        summary_path = temporary_root / "fold_summary.csv"

        with summary_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(summary_rows[0]),
            )
            writer.writeheader()
            writer.writerows(summary_rows)

        manifest = {
            "experiment_id": "EXP-17",
            "source_dataset": str(
                SOURCE_DATASET.relative_to(REPO_ROOT)
            ),
            "total_records": len(records),
            "outer_folds": N_OUTER_FOLDS,
            "validation_fraction": VALIDATION_FRACTION,
            "random_seed": RANDOM_SEED,
            "stratified_outer_cv": True,
            "stratified_inner_validation": True,
            "balancing": "none_original_imbalanced",
            "id_field": ID_FIELD,
            "label_field": LABEL_FIELD,
            "rationale_field": RATIONALE_FIELD,
            "label_counts": dict(Counter(labels)),
            "outer_test_coverage_verified": True,
            "split_overlap_verified": True,
        }

        with (
            temporary_root / "split_manifest.json"
        ).open("w", encoding="utf-8") as file:
            json.dump(
                manifest,
                file,
                indent=2,
                ensure_ascii=False,
            )
            file.write("\n")

        if OUTPUT_ROOT.exists():
            shutil.rmtree(OUTPUT_ROOT)

        temporary_root.rename(OUTPUT_ROOT)

    except Exception:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        raise

    print()
    print("EXP-17 fold preparation PASSED")
    print("New folds were generated from the rationale dataset.")
    print("Old fold memberships were not used.")
    print(f"Summary: {OUTPUT_ROOT / 'fold_summary.csv'}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print()
        print(f"ERROR: {error}")
        sys.exit(1)
