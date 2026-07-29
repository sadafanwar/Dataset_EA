#!/usr/bin/env python3
"""Aggregate and compare EXP-17 and EXP-18 across ten folds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]

EXP17_ROOT = (
    REPO_ROOT
    / "results"
    / "exp17_qwen35_9b_rationale_single_template"
)

EXP18_ROOT = (
    REPO_ROOT
    / "results"
    / "exp18_qwen35_9b_rationale_multi_template"
)

DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "exp17_exp18_comparison"
)

EXPECTED_FOLDS = tuple(
    f"fold_{number:02d}"
    for number in range(1, 11)
)

PRIMARY_METRICS_FILE = "classification_metrics.json"
RATIONALE_METRICS_FILE = "rationale_metrics.json"
TRAINING_HISTORY_FILE = "training_history.json"

ROBUSTNESS_DIRECTORY = "robustness_evaluation"
ROBUSTNESS_SUMMARY_FILE = "robustness_summary.json"
ROBUSTNESS_CONSISTENCY_FILE = (
    "robustness_consistency_metrics.json"
)
ROBUSTNESS_TEMPLATE_FILE = (
    "robustness_per_template_metrics.json"
)

CORE_METRICS = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "format_valid_rate",
)

CLASS_METRICS = (
    "precision",
    "recall",
    "f1",
)

VALID_LABELS = (
    "trace",
    "no_trace",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate ten-fold EXP-17 and EXP-18 results "
            "and create paired comparison tables."
        )
    )

    parser.add_argument(
        "--exp17-root",
        type=Path,
        default=EXP17_ROOT,
    )

    parser.add_argument(
        "--exp18-root",
        type=Path,
        default=EXP18_ROOT,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "Aggregate available matching folds. "
            "Final publication analysis must not use this flag."
        ),
    )

    parser.add_argument(
        "--include-robustness",
        action="store_true",
        help=(
            "Also aggregate EXP-18 five-template "
            "robustness outputs."
        ),
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
        help=(
            "Run synthetic metric tests without "
            "reading experiment result folders."
        ),
    )

    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required JSON file is missing: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected JSON object: {path}"
        )

    return payload


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    headers: Iterable[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    header_list = list(headers)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=header_list,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    header: row.get(header, "")
                    for header in header_list
                }
            )


def safe_float(value: Any) -> float:
    if value is None or value == "":
        return math.nan

    return float(value)


def finite_values(
    values: Iterable[float],
) -> list[float]:
    return [
        value
        for value in values
        if math.isfinite(value)
    ]


def descriptive_summary(
    values: Iterable[float],
) -> dict[str, Any]:
    cleaned = finite_values(values)

    if not cleaned:
        return {
            "count": 0,
            "mean": None,
            "standard_deviation": None,
            "minimum": None,
            "maximum": None,
            "median": None,
        }

    return {
        "count": len(cleaned),
        "mean": statistics.mean(cleaned),
        "standard_deviation": (
            statistics.stdev(cleaned)
            if len(cleaned) > 1
            else 0.0
        ),
        "minimum": min(cleaned),
        "maximum": max(cleaned),
        "median": statistics.median(cleaned),
    }


def paired_effect_summary(
    exp17_values: list[float],
    exp18_values: list[float],
) -> dict[str, Any]:
    if len(exp17_values) != len(exp18_values):
        raise ValueError(
            "Paired metric vectors have different lengths."
        )

    differences = [
        exp18 - exp17
        for exp17, exp18 in zip(
            exp17_values,
            exp18_values,
            strict=True,
        )
    ]

    cleaned = finite_values(differences)

    positive = sum(
        difference > 0
        for difference in cleaned
    )

    negative = sum(
        difference < 0
        for difference in cleaned
    )

    ties = sum(
        difference == 0
        for difference in cleaned
    )

    mean_difference = (
        statistics.mean(cleaned)
        if cleaned
        else None
    )

    standard_deviation = (
        statistics.stdev(cleaned)
        if len(cleaned) > 1
        else (
            0.0
            if cleaned
            else None
        )
    )

    cohen_dz = None

    if (
        mean_difference is not None
        and standard_deviation is not None
        and standard_deviation > 0
    ):
        cohen_dz = (
            mean_difference
            / standard_deviation
        )

    return {
        "paired_fold_count": len(cleaned),
        "mean_difference_exp18_minus_exp17": (
            mean_difference
        ),
        "standard_deviation_of_difference": (
            standard_deviation
        ),
        "median_difference": (
            statistics.median(cleaned)
            if cleaned
            else None
        ),
        "minimum_difference": (
            min(cleaned)
            if cleaned
            else None
        ),
        "maximum_difference": (
            max(cleaned)
            if cleaned
            else None
        ),
        "exp18_better_folds": positive,
        "exp17_better_folds": negative,
        "tied_folds": ties,
        "cohen_dz": cohen_dz,
        "interpretation_note": (
            "Cohen's dz is descriptive. Formal inference "
            "should be selected and reported separately "
            "after confirming assumptions and analysis policy."
        ),
    }


def flatten_fold_metrics(
    *,
    experiment_id: str,
    fold_name: str,
    fold_directory: Path,
) -> dict[str, Any]:
    classification = load_json(
        fold_directory
        / PRIMARY_METRICS_FILE
    )

    rationale = load_json(
        fold_directory
        / RATIONALE_METRICS_FILE
    )

    training = load_json(
        fold_directory
        / TRAINING_HISTORY_FILE
    )

    row: dict[str, Any] = {
        "experiment_id": experiment_id,
        "fold": fold_name,
        "test_records": classification.get(
            "total_test_records"
        ),
        "valid_predictions": classification.get(
            "valid_predictions"
        ),
        "invalid_predictions": classification.get(
            "invalid_predictions"
        ),
        "rationale_present_rate": rationale.get(
            "rationale_present_rate"
        ),
        "rationale_lexical_token_f1": rationale.get(
            "lexical_token_f1_mean"
        ),
        "rationale_rouge_l_f1": rationale.get(
            "rouge_l_f1_mean"
        ),
        "best_validation_loss": training.get(
            "best_validation_loss"
        ),
        "best_validation_perplexity": training.get(
            "best_validation_perplexity"
        ),
    }

    for metric in CORE_METRICS:
        row[metric] = classification.get(
            metric
        )

    per_class = classification.get(
        "per_class",
        {},
    )

    for label in VALID_LABELS:
        label_metrics = per_class.get(
            label,
            {},
        )

        for metric in CLASS_METRICS:
            row[
                f"{label}_{metric}"
            ] = label_metrics.get(metric)

        row[
            f"{label}_support"
        ] = label_metrics.get("support")

    return row


def discover_complete_folds(
    root: Path,
) -> set[str]:
    completed = set()

    for fold_name in EXPECTED_FOLDS:
        fold_directory = root / fold_name

        required = (
            fold_directory
            / PRIMARY_METRICS_FILE,
            fold_directory
            / RATIONALE_METRICS_FILE,
            fold_directory
            / TRAINING_HISTORY_FILE,
        )

        if all(path.is_file() for path in required):
            completed.add(fold_name)

    return completed


def aggregate_experiment(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "fold_count": len(rows),
        "folds": [
            row["fold"]
            for row in rows
        ],
        "metrics": {},
    }

    aggregate_metrics = list(
        CORE_METRICS
    ) + [
        "trace_precision",
        "trace_recall",
        "trace_f1",
        "no_trace_precision",
        "no_trace_recall",
        "no_trace_f1",
        "rationale_present_rate",
        "rationale_lexical_token_f1",
        "rationale_rouge_l_f1",
        "best_validation_loss",
        "best_validation_perplexity",
    ]

    for metric in aggregate_metrics:
        summary["metrics"][metric] = (
            descriptive_summary(
                safe_float(row.get(metric))
                for row in rows
            )
        )

    return summary


def build_paired_rows(
    exp17_rows: list[dict[str, Any]],
    exp18_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    exp17_by_fold = {
        row["fold"]: row
        for row in exp17_rows
    }

    exp18_by_fold = {
        row["fold"]: row
        for row in exp18_rows
    }

    common_folds = sorted(
        set(exp17_by_fold)
        & set(exp18_by_fold)
    )

    metrics = list(
        CORE_METRICS
    ) + [
        "trace_f1",
        "no_trace_f1",
        "rationale_present_rate",
        "rationale_lexical_token_f1",
        "rationale_rouge_l_f1",
    ]

    paired_rows = []

    for fold_name in common_folds:
        exp17 = exp17_by_fold[fold_name]
        exp18 = exp18_by_fold[fold_name]

        row: dict[str, Any] = {
            "fold": fold_name,
        }

        for metric in metrics:
            exp17_value = safe_float(
                exp17.get(metric)
            )

            exp18_value = safe_float(
                exp18.get(metric)
            )

            row[
                f"exp17_{metric}"
            ] = exp17_value

            row[
                f"exp18_{metric}"
            ] = exp18_value

            row[
                f"difference_{metric}"
            ] = (
                exp18_value
                - exp17_value
            )

        paired_rows.append(row)

    return paired_rows


def aggregate_paired_comparison(
    paired_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = list(
        CORE_METRICS
    ) + [
        "trace_f1",
        "no_trace_f1",
        "rationale_present_rate",
        "rationale_lexical_token_f1",
        "rationale_rouge_l_f1",
    ]

    output: dict[str, Any] = {
        "paired_fold_count": len(
            paired_rows
        ),
        "metrics": {},
    }

    for metric in metrics:
        exp17_values = [
            safe_float(
                row[f"exp17_{metric}"]
            )
            for row in paired_rows
        ]

        exp18_values = [
            safe_float(
                row[f"exp18_{metric}"]
            )
            for row in paired_rows
        ]

        output["metrics"][metric] = {
            "exp17": descriptive_summary(
                exp17_values
            ),
            "exp18": descriptive_summary(
                exp18_values
            ),
            "paired_effect": (
                paired_effect_summary(
                    exp17_values,
                    exp18_values,
                )
            ),
        }

    return output


def aggregate_robustness(
    exp18_root: Path,
    folds: list[str],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    rows = []

    for fold_name in folds:
        robustness_directory = (
            exp18_root
            / fold_name
            / ROBUSTNESS_DIRECTORY
        )

        consistency = load_json(
            robustness_directory
            / ROBUSTNESS_CONSISTENCY_FILE
        )

        summary = load_json(
            robustness_directory
            / ROBUSTNESS_SUMMARY_FILE
        )

        per_template = load_json(
            robustness_directory
            / ROBUSTNESS_TEMPLATE_FILE
        )

        row: dict[str, Any] = {
            "fold": fold_name,
            "unanimous_valid_rate": (
                consistency.get(
                    "unanimous_valid_rate"
                )
            ),
            "pairwise_agreement_rate": (
                consistency.get(
                    "pairwise_agreement_rate"
                )
            ),
            "majority_vote_accuracy": (
                consistency.get(
                    "majority_vote_accuracy"
                )
            ),
            "all_templates_valid_rate": (
                consistency.get(
                    "all_templates_valid_rate"
                )
            ),
            "template_accuracy_mean": (
                summary.get(
                    "template_accuracy",
                    {},
                ).get("mean")
            ),
            "template_accuracy_minimum": (
                summary.get(
                    "template_accuracy",
                    {},
                ).get("minimum")
            ),
            "template_accuracy_maximum": (
                summary.get(
                    "template_accuracy",
                    {},
                ).get("maximum")
            ),
            "template_macro_f1_mean": (
                summary.get(
                    "template_macro_f1",
                    {},
                ).get("mean")
            ),
            "template_macro_f1_minimum": (
                summary.get(
                    "template_macro_f1",
                    {},
                ).get("minimum")
            ),
            "template_macro_f1_maximum": (
                summary.get(
                    "template_macro_f1",
                    {},
                ).get("maximum")
            ),
        }

        for template_id, values in sorted(
            per_template.items()
        ):
            classification = values.get(
                "classification",
                {},
            )

            row[
                f"{template_id}_accuracy"
            ] = classification.get(
                "accuracy"
            )

            row[
                f"{template_id}_macro_f1"
            ] = classification.get(
                "macro_f1"
            )

        rows.append(row)

    metric_names = sorted(
        {
            key
            for row in rows
            for key in row
            if key != "fold"
        }
    )

    summary = {
        "fold_count": len(rows),
        "folds": [
            row["fold"]
            for row in rows
        ],
        "metrics": {
            metric: descriptive_summary(
                safe_float(row.get(metric))
                for row in rows
            )
            for metric in metric_names
        },
        "interpretation_note": (
            "Robustness results are secondary. "
            "The primary EXP-17 versus EXP-18 comparison "
            "must use template_01 evaluation metrics."
        ),
    }

    return summary, rows


def run_self_test() -> None:
    values = [
        0.80,
        0.82,
        0.84,
    ]

    summary = descriptive_summary(
        values
    )

    if summary["count"] != 3:
        raise RuntimeError(
            "Descriptive count self-test failed."
        )

    if not math.isclose(
        summary["mean"],
        0.82,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Descriptive mean self-test failed."
        )

    paired = paired_effect_summary(
        [0.80, 0.82, 0.84],
        [0.81, 0.81, 0.86],
    )

    if paired["paired_fold_count"] != 3:
        raise RuntimeError(
            "Paired count self-test failed."
        )

    if paired["exp18_better_folds"] != 2:
        raise RuntimeError(
            "Paired positive-count self-test failed."
        )

    if paired["exp17_better_folds"] != 1:
        raise RuntimeError(
            "Paired negative-count self-test failed."
        )

    print(
        "PASS: descriptive-statistics self-test."
    )

    print(
        "PASS: paired-effect self-test."
    )

    print(
        "PASS: aggregation module self-test completed."
    )


def main() -> int:
    args = parse_args()

    if args.self_test:
        run_self_test()
        return 0

    exp17_complete = discover_complete_folds(
        args.exp17_root
    )

    exp18_complete = discover_complete_folds(
        args.exp18_root
    )

    common_folds = sorted(
        exp17_complete
        & exp18_complete
    )

    if not args.allow_incomplete:
        expected = set(EXPECTED_FOLDS)

        if exp17_complete != expected:
            missing = sorted(
                expected - exp17_complete
            )

            raise RuntimeError(
                "EXP-17 is incomplete. Missing folds: "
                + ", ".join(missing)
            )

        if exp18_complete != expected:
            missing = sorted(
                expected - exp18_complete
            )

            raise RuntimeError(
                "EXP-18 is incomplete. Missing folds: "
                + ", ".join(missing)
            )

        common_folds = list(
            EXPECTED_FOLDS
        )

    if not common_folds:
        raise RuntimeError(
            "No matching completed EXP-17 and EXP-18 folds."
        )

    exp17_rows = [
        flatten_fold_metrics(
            experiment_id="EXP-17",
            fold_name=fold_name,
            fold_directory=(
                args.exp17_root
                / fold_name
            ),
        )
        for fold_name in common_folds
    ]

    exp18_rows = [
        flatten_fold_metrics(
            experiment_id="EXP-18",
            fold_name=fold_name,
            fold_directory=(
                args.exp18_root
                / fold_name
            ),
        )
        for fold_name in common_folds
    ]

    paired_rows = build_paired_rows(
        exp17_rows,
        exp18_rows,
    )

    output_root = args.output_root

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_headers = sorted(
        {
            key
            for row in (
                exp17_rows
                + exp18_rows
            )
            for key in row
        }
    )

    paired_headers = sorted(
        {
            key
            for row in paired_rows
            for key in row
        }
    )

    write_csv(
        output_root
        / "fold_level_metrics.csv",
        exp17_rows + exp18_rows,
        fold_headers,
    )

    write_csv(
        output_root
        / "paired_fold_comparison.csv",
        paired_rows,
        paired_headers,
    )

    exp17_summary = aggregate_experiment(
        exp17_rows
    )

    exp18_summary = aggregate_experiment(
        exp18_rows
    )

    paired_summary = (
        aggregate_paired_comparison(
            paired_rows
        )
    )

    write_json(
        output_root
        / "exp17_summary.json",
        exp17_summary,
    )

    write_json(
        output_root
        / "exp18_summary.json",
        exp18_summary,
    )

    write_json(
        output_root
        / "paired_comparison_summary.json",
        paired_summary,
    )

    manifest: dict[str, Any] = {
        "experiments": [
            "EXP-17",
            "EXP-18",
        ],
        "comparison_protocol": (
            "paired identical-fold comparison using "
            "EXP-18 template_01 primary evaluation"
        ),
        "folds": common_folds,
        "fold_count": len(common_folds),
        "allow_incomplete": bool(
            args.allow_incomplete
        ),
        "robustness_included": bool(
            args.include_robustness
        ),
        "files": {
            "fold_metrics": (
                "fold_level_metrics.csv"
            ),
            "paired_folds": (
                "paired_fold_comparison.csv"
            ),
            "exp17_summary": (
                "exp17_summary.json"
            ),
            "exp18_summary": (
                "exp18_summary.json"
            ),
            "paired_summary": (
                "paired_comparison_summary.json"
            ),
        },
    }

    if args.include_robustness:
        robustness_summary, robustness_rows = (
            aggregate_robustness(
                args.exp18_root,
                common_folds,
            )
        )

        robustness_headers = sorted(
            {
                key
                for row in robustness_rows
                for key in row
            }
        )

        write_csv(
            output_root
            / "exp18_robustness_fold_metrics.csv",
            robustness_rows,
            robustness_headers,
        )

        write_json(
            output_root
            / "exp18_robustness_summary.json",
            robustness_summary,
        )

        manifest["files"].update(
            {
                "robustness_fold_metrics": (
                    "exp18_robustness_fold_metrics.csv"
                ),
                "robustness_summary": (
                    "exp18_robustness_summary.json"
                ),
            }
        )

    write_json(
        output_root
        / "comparison_manifest.json",
        manifest,
    )

    print()
    print("=" * 72)
    print(
        "EXP-17 VERSUS EXP-18 AGGREGATION COMPLETED"
    )
    print("=" * 72)
    print(
        f"Paired folds: {len(common_folds)}"
    )
    print(f"Output: {output_root}")
    print(
        "Primary comparison: EXP-18 template_01"
    )
    print(
        "Robustness included: "
        f"{args.include_robustness}"
    )
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
