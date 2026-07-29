#!/usr/bin/env python3
"""Controlled per-fold runner and tracker for EXP-17."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]

TRAIN_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "exp17"
    / "train_exp17.py"
)

CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "exp17"
    / "config_qwen35_9b_exp17.yaml"
)

FOLDS_ROOT = (
    REPO_ROOT
    / "data"
    / "exp17_qwen35_9b_rationale_single_template"
    / "folds"
)

RESULT_ROOT = (
    REPO_ROOT
    / "results"
    / "exp17_qwen35_9b_rationale_single_template"
)

PROGRESS_FILE = (
    RESULT_ROOT
    / "progress"
    / "progress.csv"
)

REGISTRY_FILE = (
    RESULT_ROOT
    / "registry"
    / "run_registry.csv"
)


PROGRESS_HEADERS = [
    "fold",
    "status",
    "stage",
    "start_time_utc",
    "end_time_utc",
    "duration_minutes",
    "last_updated_utc",
    "message",
]


REGISTRY_HEADERS = [
    "experiment_id",
    "experiment_name",
    "fold",
    "status",
    "start_time_utc",
    "end_time_utc",
    "duration_minutes",
    "model",
    "method",
    "balancing",
    "training_target",
    "template_strategy",
    "train_file",
    "validation_file",
    "test_file",
    "config_file",
    "resolved_config_file",
    "output_dir",
    "best_model_dir",
    "metrics_file",
    "predictions_file",
    "train_total",
    "train_trace",
    "train_no_trace",
    "validation_total",
    "validation_trace",
    "validation_no_trace",
    "test_total",
    "test_trace",
    "test_no_trace",
    "accuracy",
    "macro_f1",
    "trace_precision",
    "trace_recall",
    "trace_f1",
    "no_trace_precision",
    "no_trace_recall",
    "no_trace_f1",
    "valid_predictions",
    "invalid_predictions",
    "format_valid_rate",
    "rationale_present_rate",
    "notes",
]


COMPLETED_STATUSES = {
    "training_completed",
    "evaluation_completed",
    "completed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run and track one EXP-17 "
            "cross-validation fold."
        )
    )

    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        choices=range(1, 11),
        metavar="1-10",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run train_exp17.py in dry-run mode "
            "without changing tracking CSV files."
        ),
    )

    parser.add_argument(
        "--preview-records",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help=(
            "Diagnostic-only training limit. "
            "Do not use for final experiments."
        ),
    )

    parser.add_argument(
        "--method",
        choices=("lora", "qlora", "dora"),
        default=None,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Allow rerunning a fold that already "
            "has a completed status."
        ),
    )

    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def duration_minutes(
    start_time: str,
    end_time: str,
) -> str:
    start = datetime.fromisoformat(
        start_time
    )

    end = datetime.fromisoformat(
        end_time
    )

    minutes = (
        end - start
    ).total_seconds() / 60.0

    return f"{minutes:.4f}"


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(
        REPO_ROOT.resolve()
    ).as_posix()


def load_yaml(
    path: Path,
) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as handle:
        value = yaml.safe_load(handle)

    if not isinstance(value, dict):
        raise ValueError(
            f"YAML root must be an object: {path}"
        )

    return value


def read_csv_rows(
    path: Path,
    expected_headers: list[str],
) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Tracking file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        actual_headers = reader.fieldnames

        if actual_headers != expected_headers:
            raise ValueError(
                f"Unexpected CSV headers in {path}.\n"
                f"Expected: {expected_headers}\n"
                f"Actual:   {actual_headers}"
            )

        return [
            {
                header: row.get(header, "")
                for header in expected_headers
            }
            for row in reader
        ]


def atomic_write_csv(
    path: Path,
    headers: list[str],
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            delete=False,
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as handle:
            temporary_path = Path(
                handle.name
            )

            writer = csv.DictWriter(
                handle,
                fieldnames=headers,
                extrasaction="ignore",
            )

            writer.writeheader()

            for row in rows:
                writer.writerow(
                    {
                        header: row.get(
                            header,
                            "",
                        )
                        for header in headers
                    }
                )

        os.replace(
            temporary_path,
            path,
        )

    finally:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink()


def update_progress(
    *,
    fold_name: str,
    status: str,
    stage: str,
    start_time_utc: str,
    end_time_utc: str = "",
    message: str = "",
) -> None:
    rows = read_csv_rows(
        PROGRESS_FILE,
        PROGRESS_HEADERS,
    )

    row = {
        "fold": fold_name,
        "status": status,
        "stage": stage,
        "start_time_utc": start_time_utc,
        "end_time_utc": end_time_utc,
        "duration_minutes": (
            duration_minutes(
                start_time_utc,
                end_time_utc,
            )
            if end_time_utc
            else ""
        ),
        "last_updated_utc": utc_now(),
        "message": message,
    }

    replaced = False
    updated_rows = []

    for existing_row in rows:
        if (
            existing_row["fold"]
            == fold_name
        ):
            updated_rows.append(row)
            replaced = True
        else:
            updated_rows.append(
                existing_row
            )

    if not replaced:
        updated_rows.append(row)

    updated_rows.sort(
        key=lambda item: item["fold"]
    )

    atomic_write_csv(
        PROGRESS_FILE,
        PROGRESS_HEADERS,
        updated_rows,
    )


def current_progress_status(
    fold_name: str,
) -> str:
    rows = read_csv_rows(
        PROGRESS_FILE,
        PROGRESS_HEADERS,
    )

    for row in rows:
        if row["fold"] == fold_name:
            return row["status"]

    return ""


def upsert_registry(
    row: dict[str, Any],
) -> None:
    rows = read_csv_rows(
        REGISTRY_FILE,
        REGISTRY_HEADERS,
    )

    key = (
        str(row["experiment_id"]),
        str(row["fold"]),
        str(row["start_time_utc"]),
    )

    replaced = False
    updated_rows = []

    for existing_row in rows:
        existing_key = (
            existing_row["experiment_id"],
            existing_row["fold"],
            existing_row["start_time_utc"],
        )

        if existing_key == key:
            updated_rows.append(row)
            replaced = True
        else:
            updated_rows.append(
                existing_row
            )

    if not replaced:
        updated_rows.append(row)

    atomic_write_csv(
        REGISTRY_FILE,
        REGISTRY_HEADERS,
        updated_rows,
    )


def count_split(
    path: Path,
) -> dict[str, int]:
    counts = {
        "total": 0,
        "trace": 0,
        "no_trace": 0,
    }

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} "
                    f"at line {line_number}: {error}"
                ) from error

            label = record.get("label")

            if label not in {
                "trace",
                "no_trace",
            }:
                raise ValueError(
                    f"Invalid label in {path} "
                    f"at line {line_number}: "
                    f"{label!r}"
                )

            counts["total"] += 1
            counts[label] += 1

    return counts


def read_json_if_present(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        return {}

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as handle:
        value = json.load(handle)

    return (
        value
        if isinstance(value, dict)
        else {}
    )


def build_command(
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--fold",
        str(args.fold),
    ]

    if args.dry_run:
        command.extend(
            [
                "--dry-run",
                "--preview-records",
                str(args.preview_records),
            ]
        )

    if args.method:
        command.extend(
            [
                "--method",
                args.method,
            ]
        )

    if args.max_train_samples is not None:
        command.extend(
            [
                "--max-train-samples",
                str(args.max_train_samples),
            ]
        )

    return command


def main() -> int:
    args = parse_args()

    fold_name = (
        f"fold_{args.fold:02d}"
    )

    command = build_command(args)

    print()
    print("=" * 70)
    print("EXP-17 CONTROLLED FOLD RUNNER")
    print("=" * 70)
    print(f"Fold       : {fold_name}")
    print(
        "Mode       : "
        + (
            "dry run"
            if args.dry_run
            else "training"
        )
    )
    print(
        "Command    : "
        + " ".join(command)
    )

    if args.dry_run:
        print(
            "Tracking CSV files will not "
            "be modified."
        )

        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
        )

        return int(
            result.returncode
        )

    current_status = (
        current_progress_status(
            fold_name
        )
    )

    if (
        current_status
        in COMPLETED_STATUSES
        and not args.force
    ):
        raise RuntimeError(
            f"{fold_name} already has status "
            f"{current_status!r}. Use --force "
            "only for an intentional rerun."
        )

    config = load_yaml(
        CONFIG_PATH
    )

    experiment = config.get(
        "experiment",
        {},
    )

    fold_data_dir = (
        FOLDS_ROOT / fold_name
    )

    train_file = (
        fold_data_dir / "train.jsonl"
    )

    validation_file = (
        fold_data_dir
        / "validation.jsonl"
    )

    test_file = (
        fold_data_dir / "test.jsonl"
    )

    for required_file in (
        train_file,
        validation_file,
        test_file,
    ):
        if not required_file.is_file():
            raise FileNotFoundError(
                f"Required split missing: "
                f"{required_file}"
            )

    train_counts = count_split(
        train_file
    )

    validation_counts = count_split(
        validation_file
    )

    test_counts = count_split(
        test_file
    )

    fold_output_dir = (
        RESULT_ROOT / fold_name
    )

    if (
        fold_output_dir.joinpath(
            "best_model"
        ).exists()
        and not args.force
    ):
        raise RuntimeError(
            f"A best_model directory already "
            f"exists for {fold_name}. "
            "Use --force only for an "
            "intentional rerun."
        )

    fold_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    resolved_config_file = (
        fold_output_dir
        / "resolved_config.yaml"
    )

    shutil.copy2(
        CONFIG_PATH,
        resolved_config_file,
    )

    start_time = utc_now()

    registry_row = {
        header: ""
        for header in REGISTRY_HEADERS
    }

    registry_row.update(
        {
            "experiment_id": experiment.get(
                "id",
                "EXP-17",
            ),
            "experiment_name": experiment.get(
                "name",
                "Qwen3.5 Label + Rationale",
            ),
            "fold": fold_name,
            "status": "running",
            "start_time_utc": start_time,
            "model": experiment.get(
                "model",
                config.get(
                    "model_path",
                    "",
                ),
            ),
            "method": experiment.get(
                "method",
                config.get(
                    "method",
                    "",
                ),
            ),
            "balancing": experiment.get(
                "balancing",
                "",
            ),
            "training_target": experiment.get(
                "training_target",
                "",
            ),
            "template_strategy": experiment.get(
                "template_strategy",
                "",
            ),
            "train_file": relative_path(
                train_file
            ),
            "validation_file": relative_path(
                validation_file
            ),
            "test_file": relative_path(
                test_file
            ),
            "config_file": relative_path(
                CONFIG_PATH
            ),
            "resolved_config_file": relative_path(
                resolved_config_file
            ),
            "output_dir": relative_path(
                fold_output_dir
            ),
            "train_total": train_counts[
                "total"
            ],
            "train_trace": train_counts[
                "trace"
            ],
            "train_no_trace": train_counts[
                "no_trace"
            ],
            "validation_total": validation_counts[
                "total"
            ],
            "validation_trace": validation_counts[
                "trace"
            ],
            "validation_no_trace": validation_counts[
                "no_trace"
            ],
            "test_total": test_counts[
                "total"
            ],
            "test_trace": test_counts[
                "trace"
            ],
            "test_no_trace": test_counts[
                "no_trace"
            ],
            "notes": (
                "Training started through "
                "controlled EXP-17 fold runner."
            ),
        }
    )

    update_progress(
        fold_name=fold_name,
        status="running",
        stage="training",
        start_time_utc=start_time,
        message="QLoRA training started.",
    )

    upsert_registry(
        registry_row
    )

    print()
    print(
        f"Tracking status: "
        f"{fold_name} = running"
    )

    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
        )

        end_time = utc_now()

        if result.returncode != 0:
            failure_message = (
                "Training process failed with "
                f"exit code {result.returncode}."
            )

            update_progress(
                fold_name=fold_name,
                status="failed",
                stage="training",
                start_time_utc=start_time,
                end_time_utc=end_time,
                message=failure_message,
            )

            registry_row.update(
                {
                    "status": "failed",
                    "end_time_utc": end_time,
                    "duration_minutes": (
                        duration_minutes(
                            start_time,
                            end_time,
                        )
                    ),
                    "notes": failure_message,
                }
            )

            upsert_registry(
                registry_row
            )

            return int(
                result.returncode
            )

        run_metadata = read_json_if_present(
            fold_output_dir
            / "run_metadata.json"
        )

        history = read_json_if_present(
            fold_output_dir
            / "training_history.json"
        )

        best_model_dir = (
            fold_output_dir
            / "best_model"
        )

        note_parts = [
            "Training completed successfully."
        ]

        best_perplexity = (
            run_metadata.get(
                "best_validation_perplexity"
            )
            or history.get(
                "best_validation_perplexity"
            )
        )

        if best_perplexity is not None:
            note_parts.append(
                "best_validation_perplexity="
                f"{best_perplexity}"
            )

        best_loss = (
            run_metadata.get(
                "best_validation_loss"
            )
            or history.get(
                "best_validation_loss"
            )
        )

        if best_loss is not None:
            note_parts.append(
                "best_validation_loss="
                f"{best_loss}"
            )

        completion_message = "; ".join(
            note_parts
        )

        update_progress(
            fold_name=fold_name,
            status="training_completed",
            stage="training",
            start_time_utc=start_time,
            end_time_utc=end_time,
            message=completion_message,
        )

        registry_row.update(
            {
                "status": "training_completed",
                "end_time_utc": end_time,
                "duration_minutes": (
                    duration_minutes(
                        start_time,
                        end_time,
                    )
                ),
                "best_model_dir": (
                    relative_path(
                        best_model_dir
                    )
                    if best_model_dir.exists()
                    else ""
                ),
                "notes": completion_message,
            }
        )

        upsert_registry(
            registry_row
        )

        print()
        print("=" * 70)
        print("EXP-17 FOLD TRAINING COMPLETED")
        print(f"Fold   : {fold_name}")
        print(
            f"Output : {fold_output_dir}"
        )
        print("=" * 70)

        return 0

    except KeyboardInterrupt:
        end_time = utc_now()

        message = (
            "Training interrupted by user."
        )

        update_progress(
            fold_name=fold_name,
            status="interrupted",
            stage="training",
            start_time_utc=start_time,
            end_time_utc=end_time,
            message=message,
        )

        registry_row.update(
            {
                "status": "interrupted",
                "end_time_utc": end_time,
                "duration_minutes": (
                    duration_minutes(
                        start_time,
                        end_time,
                    )
                ),
                "notes": message,
            }
        )

        upsert_registry(
            registry_row
        )

        return 130

    except Exception as error:
        end_time = utc_now()

        message = (
            f"Runner failure: "
            f"{type(error).__name__}: {error}"
        )

        update_progress(
            fold_name=fold_name,
            status="failed",
            stage="runner",
            start_time_utc=start_time,
            end_time_utc=end_time,
            message=message,
        )

        registry_row.update(
            {
                "status": "failed",
                "end_time_utc": end_time,
                "duration_minutes": (
                    duration_minutes(
                        start_time,
                        end_time,
                    )
                ),
                "notes": message,
            }
        )

        upsert_registry(
            registry_row
        )

        raise


if __name__ == "__main__":
    raise SystemExit(main())
