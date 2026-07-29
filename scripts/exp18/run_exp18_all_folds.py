#!/usr/bin/env python3
"""Sequential and resume-safe orchestrator for EXP-18 folds."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

RESULT_ROOT = (
    REPO_ROOT
    / "results"
    / "exp18_qwen35_9b_rationale_multi_template"
)

PROGRESS_FILE = (
    RESULT_ROOT
    / "progress"
    / "progress.csv"
)

FOLDS_ROOT = (
    REPO_ROOT
    / "data"
    / "exp17_qwen35_9b_rationale_single_template"
    / "folds"
)

TRAIN_RUNNER = (
    REPO_ROOT
    / "scripts"
    / "exp18"
    / "run_exp18_fold.py"
)

PRIMARY_EVALUATOR = (
    REPO_ROOT
    / "scripts"
    / "exp18"
    / "evaluate_exp18.py"
)

ROBUSTNESS_EVALUATOR = (
    REPO_ROOT
    / "scripts"
    / "exp18"
    / "evaluate_exp18_robustness.py"
)

PRIMARY_FILES = (
    "test_predictions.jsonl",
    "classification_metrics.json",
    "rationale_metrics.json",
    "confusion_matrix.csv",
    "fold_evaluation_summary.json",
)

ROBUSTNESS_FILES = (
    "robustness_predictions.jsonl",
    "robustness_per_template_metrics.json",
    "robustness_consistency_metrics.json",
    "robustness_record_consistency.jsonl",
    "robustness_summary.json",
)

MINIMUM_FREE_BYTES = 10 * 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run EXP-18 folds sequentially with "
            "resume-safe artifact validation."
        )
    )

    parser.add_argument(
        "--start-fold",
        type=int,
        default=1,
        choices=range(1, 11),
    )

    parser.add_argument(
        "--end-fold",
        type=int,
        default=10,
        choices=range(1, 11),
    )

    parser.add_argument(
        "--include-robustness",
        action="store_true",
        help=(
            "Run the secondary all-five-template "
            "robustness evaluation after primary evaluation."
        ),
    )

    parser.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "Inspect planned actions without loading "
            "a model or starting any experiment."
        ),
    )

    return parser.parse_args()


def fail(message: str) -> None:
    raise RuntimeError(message)


def read_progress_status(
    fold_name: str,
) -> str:
    if not PROGRESS_FILE.is_file():
        fail(
            f"Progress file not found: {PROGRESS_FILE}"
        )

    with PROGRESS_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            if row.get("fold") == fold_name:
                return str(
                    row.get("status", "")
                ).strip()

    return ""


def all_files_exist(
    directory: Path,
    filenames: tuple[str, ...],
) -> bool:
    return all(
        (directory / filename).is_file()
        for filename in filenames
    )


def any_file_exists(
    directory: Path,
    filenames: tuple[str, ...],
) -> bool:
    return any(
        (directory / filename).exists()
        for filename in filenames
    )


def directory_has_content(
    directory: Path,
) -> bool:
    return (
        directory.is_dir()
        and any(directory.iterdir())
    )


def count_jsonl_rows(
    path: Path,
) -> int:
    if not path.is_file():
        fail(f"JSONL file not found: {path}")

    return sum(
        1
        for line in path.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip()
    )


def check_storage() -> None:
    free_bytes = shutil.disk_usage(
        Path.home()
    ).free

    free_gib = free_bytes / 1024**3

    print(
        f"Available storage: {free_gib:.2f} GiB",
        flush=True,
    )

    if free_bytes < MINIMUM_FREE_BYTES:
        fail(
            "Less than 10 GiB free storage remains. "
            "Stopped before starting the next stage."
        )


def determine_plan(
    *,
    fold_number: int,
    include_robustness: bool,
) -> dict[str, Any]:
    fold_name = f"fold_{fold_number:02d}"

    fold_directory = (
        RESULT_ROOT
        / fold_name
    )

    best_model_directory = (
        fold_directory
        / "best_model"
    )

    robustness_directory = (
        fold_directory
        / "robustness_evaluation"
    )

    status = read_progress_status(
        fold_name
    )

    primary_complete = all_files_exist(
        fold_directory,
        PRIMARY_FILES,
    )

    primary_partial = (
        any_file_exists(
            fold_directory,
            PRIMARY_FILES,
        )
        and not primary_complete
    )

    robustness_complete = all_files_exist(
        robustness_directory,
        ROBUSTNESS_FILES,
    )

    robustness_partial = (
        any_file_exists(
            robustness_directory,
            ROBUSTNESS_FILES,
        )
        and not robustness_complete
    )

    if primary_partial:
        fail(
            f"{fold_name}: partial primary "
            "evaluation outputs were found."
        )

    if robustness_partial:
        fail(
            f"{fold_name}: partial robustness "
            "outputs were found."
        )

    if robustness_complete and not primary_complete:
        fail(
            f"{fold_name}: robustness outputs exist "
            "without complete primary evaluation outputs."
        )

    needs_training = False
    needs_primary = False
    needs_robustness = False

    if status == "evaluation_completed":
        if not primary_complete:
            fail(
                f"{fold_name}: tracking status is "
                "evaluation_completed but primary "
                "outputs are incomplete."
            )

        needs_robustness = (
            include_robustness
            and not robustness_complete
        )

    elif primary_complete:
        fail(
            f"{fold_name}: primary outputs exist but "
            f"tracking status is {status!r}."
        )

    elif best_model_directory.is_dir():
        if status not in {
            "training_completed",
            "evaluation_failed",
            "evaluating",
        }:
            fail(
                f"{fold_name}: best_model exists but "
                f"tracking status is {status!r}."
            )

        needs_primary = True
        needs_robustness = include_robustness

    elif directory_has_content(
        fold_directory
    ):
        fail(
            f"{fold_name}: partial training artifacts "
            "exist without a best_model directory."
        )

    elif status in {"", "failed"}:
        needs_training = True
        needs_primary = True
        needs_robustness = include_robustness

    else:
        fail(
            f"{fold_name}: cannot start safely with "
            f"tracking status {status!r}."
        )

    stages = []

    if needs_training:
        stages.append("train")

    if needs_primary:
        stages.append("primary_evaluation")

    if needs_robustness:
        stages.append("robustness_evaluation")

    action = (
        "+".join(stages)
        if stages
        else "skip"
    )

    return {
        "fold_number": fold_number,
        "fold_name": fold_name,
        "fold_directory": fold_directory,
        "best_model_directory": (
            best_model_directory
        ),
        "robustness_directory": (
            robustness_directory
        ),
        "status": status or "not_started",
        "needs_training": needs_training,
        "needs_primary": needs_primary,
        "needs_robustness": needs_robustness,
        "action": action,
    }


def validate_primary_outputs(
    plan: dict[str, Any],
) -> None:
    fold_directory = plan["fold_directory"]

    if not all_files_exist(
        fold_directory,
        PRIMARY_FILES,
    ):
        fail(
            f"{plan['fold_name']}: primary evaluator "
            "returned success but output files "
            "are incomplete."
        )

    final_status = read_progress_status(
        plan["fold_name"]
    )

    if final_status != "evaluation_completed":
        fail(
            f"{plan['fold_name']}: final primary "
            f"tracking status is {final_status!r}."
        )


def validate_robustness_outputs(
    plan: dict[str, Any],
) -> None:
    robustness_directory = plan[
        "robustness_directory"
    ]

    if not all_files_exist(
        robustness_directory,
        ROBUSTNESS_FILES,
    ):
        fail(
            f"{plan['fold_name']}: robustness "
            "evaluator returned success but "
            "output files are incomplete."
        )

    fold_number = int(
        plan["fold_number"]
    )

    test_file = (
        FOLDS_ROOT
        / f"fold_{fold_number:02d}"
        / "test.jsonl"
    )

    expected_predictions = (
        count_jsonl_rows(test_file)
        * 5
    )

    actual_predictions = count_jsonl_rows(
        robustness_directory
        / "robustness_predictions.jsonl"
    )

    if actual_predictions != expected_predictions:
        fail(
            f"{plan['fold_name']}: expected "
            f"{expected_predictions} robustness "
            f"predictions, found {actual_predictions}."
        )


def run_stage(
    command: list[str],
) -> None:
    environment = os.environ.copy()

    environment["HF_HOME"] = str(
        Path.home()
        / "hf_cache"
    )

    environment["HF_HUB_CACHE"] = str(
        Path.home()
        / "hf_cache"
        / "hub"
    )

    environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment[
        "TOKENIZERS_PARALLELISM"
    ] = "false"

    environment["PYTHONUNBUFFERED"] = "1"

    print(
        "Command: " + " ".join(command),
        flush=True,
    )

    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        check=False,
    )

    if result.returncode != 0:
        fail(
            "Stage failed with exit code "
            f"{result.returncode}: "
            + " ".join(command)
        )


def validate_setup() -> None:
    required_files = (
        TRAIN_RUNNER,
        PRIMARY_EVALUATOR,
        ROBUSTNESS_EVALUATOR,
        PROGRESS_FILE,
    )

    for path in required_files:
        if not path.is_file():
            fail(
                f"Required file is missing: {path}"
            )


def main() -> int:
    args = parse_args()

    if args.start_fold > args.end_fold:
        fail(
            "--start-fold cannot be greater "
            "than --end-fold."
        )

    validate_setup()

    plans = [
        determine_plan(
            fold_number=fold_number,
            include_robustness=(
                args.include_robustness
            ),
        )
        for fold_number in range(
            args.start_fold,
            args.end_fold + 1,
        )
    ]

    print()
    print("=" * 78)
    print("EXP-18 AUTOMATED FOLD ORCHESTRATOR")
    print("=" * 78)
    print(
        "Fold range        : "
        f"{args.start_fold}-{args.end_fold}"
    )
    print(
        "Robustness mode   : "
        + (
            "enabled"
            if args.include_robustness
            else "disabled"
        )
    )
    print(
        "Execution mode    : "
        + (
            "preflight"
            if args.preflight
            else "run"
        )
    )
    print()

    for plan in plans:
        print(
            f"{plan['fold_name']} | "
            f"status={plan['status']} | "
            f"action={plan['action']}"
        )

    if args.preflight:
        print()
        print("=" * 78)
        print(
            "EXP-18 ORCHESTRATOR PREFLIGHT PASSED"
        )
        print("No model was loaded.")
        print(
            "No training or evaluation was started."
        )
        print(
            "No tracking or result file was modified."
        )
        print("=" * 78)

        return 0

    for plan in plans:
        fold_number = int(
            plan["fold_number"]
        )

        fold_name = str(
            plan["fold_name"]
        )

        print()
        print("=" * 78)
        print(
            f"{fold_name}: {plan['action']}"
        )
        print("=" * 78)

        if plan["action"] == "skip":
            print(
                f"{fold_name}: already complete; "
                "skipping."
            )
            continue

        if plan["needs_training"]:
            check_storage()

            run_stage(
                [
                    sys.executable,
                    str(TRAIN_RUNNER),
                    "--fold",
                    str(fold_number),
                ]
            )

            if not plan[
                "best_model_directory"
            ].is_dir():
                fail(
                    f"{fold_name}: training returned "
                    "success but best_model is missing."
                )

            print(
                f"{fold_name}: training completed.",
                flush=True,
            )

        if plan["needs_primary"]:
            check_storage()

            run_stage(
                [
                    sys.executable,
                    str(PRIMARY_EVALUATOR),
                    "--fold",
                    str(fold_number),
                ]
            )

            validate_primary_outputs(
                plan
            )

            print(
                f"{fold_name}: primary evaluation "
                "completed.",
                flush=True,
            )

        if plan["needs_robustness"]:
            check_storage()

            run_stage(
                [
                    sys.executable,
                    str(ROBUSTNESS_EVALUATOR),
                    "--fold",
                    str(fold_number),
                ]
            )

            validate_robustness_outputs(
                plan
            )

            print(
                f"{fold_name}: robustness "
                "evaluation completed.",
                flush=True,
            )

    print()
    print("=" * 78)
    print(
        "EXP-18 REQUESTED FOLDS COMPLETED"
    )
    print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
