#!/usr/bin/env python3
"""Evaluate EXP-18 prompt robustness across all five templates."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts.exp18.exp18_data import (  # noqa: E402
    build_prompt,
    load_jsonl,
    load_template_catalog,
    validate_raw_record,
)

from scripts.exp18.evaluate_exp18 import (  # noqa: E402
    compute_classification_metrics,
    compute_rationale_metrics,
    generate_prediction,
    load_trained_model,
    parse_generation,
    save_json,
    save_jsonl,
    utc_now,
)


EXPERIMENT_ID = "EXP-18"

EVALUATION_PROTOCOL = (
    "all_five_template_robustness"
)

CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "exp18"
    / "config_qwen35_9b_exp18.yaml"
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
    / "exp18_qwen35_9b_rationale_multi_template"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one trained EXP-18 fold with "
            "all five frozen instruction templates."
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
            "Validate templates, prompts, parser, "
            "and robustness metrics without loading a model."
        ),
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help=(
            "Diagnostic-only record limit. Final robustness "
            "evaluation must use the complete test split."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Allow intentional replacement of existing "
            "robustness output files."
        ),
    )

    return parser.parse_args()


def load_yaml(
    path: Path,
) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise ValueError(
            f"YAML root must be an object: {path}"
        )

    return payload


def compute_consistency_metrics(
    predictions: list[dict[str, Any]],
    template_ids: tuple[str, ...],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    """
    Measure prediction stability across the five templates.

    Primary EXP-17 versus EXP-18 comparisons must still use
    template_01 results, not majority-vote robustness results.
    """

    grouped: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in predictions:
        grouped[row["pair_id"]].append(row)

    record_rows: list[
        dict[str, Any]
    ] = []

    total_pairwise_valid = 0
    pairwise_agreements = 0

    all_templates_valid_count = 0
    unanimous_valid_count = 0
    majority_correct_count = 0
    majority_available_count = 0

    for pair_id, rows in sorted(
        grouped.items()
    ):
        if len(rows) != len(template_ids):
            raise ValueError(
                f"{pair_id}: expected "
                f"{len(template_ids)} template predictions, "
                f"found {len(rows)}."
            )

        by_template = {
            row["template_id"]: row
            for row in rows
        }

        if set(by_template) != set(template_ids):
            raise ValueError(
                f"{pair_id}: template coverage is incomplete."
            )

        ordered_rows = [
            by_template[template_id]
            for template_id in template_ids
        ]

        valid_labels = [
            row["predicted_label"]
            for row in ordered_rows
            if row["label_valid"]
        ]

        all_templates_valid = (
            len(valid_labels)
            == len(template_ids)
        )

        if all_templates_valid:
            all_templates_valid_count += 1

        unanimous_valid = (
            all_templates_valid
            and len(set(valid_labels)) == 1
        )

        if unanimous_valid:
            unanimous_valid_count += 1

        for first, second in combinations(
            ordered_rows,
            2,
        ):
            if (
                first["label_valid"]
                and second["label_valid"]
            ):
                total_pairwise_valid += 1

                if (
                    first["predicted_label"]
                    == second["predicted_label"]
                ):
                    pairwise_agreements += 1

        majority_label = None

        if valid_labels:
            counts = Counter(valid_labels)

            majority_label = counts.most_common(
                1
            )[0][0]

            majority_available_count += 1

            if (
                majority_label
                == ordered_rows[0]["gold_label"]
            ):
                majority_correct_count += 1

        record_rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "evaluation_protocol": (
                    EVALUATION_PROTOCOL
                ),
                "pair_id": pair_id,
                "gold_label": (
                    ordered_rows[0]["gold_label"]
                ),
                "valid_template_predictions": (
                    len(valid_labels)
                ),
                "all_templates_valid": (
                    all_templates_valid
                ),
                "unique_valid_labels": sorted(
                    set(valid_labels)
                ),
                "unanimous_valid_prediction": (
                    unanimous_valid
                ),
                "majority_label": majority_label,
                "majority_correct": (
                    majority_label
                    == ordered_rows[0]["gold_label"]
                    if majority_label is not None
                    else False
                ),
                "template_predictions": {
                    row["template_id"]: (
                        row["predicted_label"]
                    )
                    for row in ordered_rows
                },
            }
        )

    total_records = len(record_rows)

    metrics = {
        "experiment_id": EXPERIMENT_ID,
        "evaluation_protocol": (
            EVALUATION_PROTOCOL
        ),
        "template_count": len(template_ids),
        "template_ids": list(template_ids),
        "total_test_records": total_records,
        "total_generated_predictions": (
            len(predictions)
        ),
        "all_templates_valid_records": (
            all_templates_valid_count
        ),
        "all_templates_valid_rate": (
            all_templates_valid_count
            / total_records
            if total_records
            else 0.0
        ),
        "unanimous_valid_records": (
            unanimous_valid_count
        ),
        "unanimous_valid_rate": (
            unanimous_valid_count
            / total_records
            if total_records
            else 0.0
        ),
        "non_unanimous_or_invalid_rate": (
            1.0
            - (
                unanimous_valid_count
                / total_records
            )
            if total_records
            else 0.0
        ),
        "pairwise_valid_comparisons": (
            total_pairwise_valid
        ),
        "pairwise_label_agreements": (
            pairwise_agreements
        ),
        "pairwise_agreement_rate": (
            pairwise_agreements
            / total_pairwise_valid
            if total_pairwise_valid
            else 0.0
        ),
        "majority_vote_available_records": (
            majority_available_count
        ),
        "majority_vote_accuracy": (
            majority_correct_count
            / total_records
            if total_records
            else 0.0
        ),
        "interpretation_note": (
            "Robustness metrics measure sensitivity to "
            "instruction paraphrasing. Primary comparison "
            "with EXP-17 must use template_01 results."
        ),
    }

    return metrics, record_rows


def run_dry_test(
    test_records: list[dict[str, Any]],
    template_ids: tuple[str, ...],
) -> None:
    if not test_records:
        raise ValueError(
            "Test split is empty."
        )

    for raw_record in test_records[:5]:
        for template_id in template_ids:
            prompt = build_prompt(
                raw_record,
                include_answer=False,
                template_id=template_id,
            )

            full_text = build_prompt(
                raw_record,
                include_answer=True,
                template_id=template_id,
            )

            if not prompt.endswith("Answer:"):
                raise ValueError(
                    f"{template_id}: inference prompt "
                    "does not end with Answer:."
                )

            if not full_text.startswith(
                prompt + "\nLabel:"
            ):
                raise ValueError(
                    f"{template_id}: gold-answer "
                    "separation failed."
                )

    synthetic_predictions = []

    for template_id, label in zip(
        template_ids,
        (
            "trace",
            "trace",
            "trace",
            "no_trace",
            "trace",
        ),
        strict=True,
    ):
        parsed = parse_generation(
            f"Label: {label}\n"
            "Rationale: Synthetic robustness output."
        )

        synthetic_predictions.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "pair_id": "SYNTHETIC_PAIR",
                "template_id": template_id,
                "gold_label": "trace",
                **parsed,
            }
        )

    metrics, rows = compute_consistency_metrics(
        synthetic_predictions,
        template_ids,
    )

    if metrics["total_test_records"] != 1:
        raise ValueError(
            "Synthetic consistency-record count failed."
        )

    if metrics[
        "total_generated_predictions"
    ] != len(template_ids):
        raise ValueError(
            "Synthetic prediction count failed."
        )

    if len(rows) != 1:
        raise ValueError(
            "Synthetic record summary failed."
        )


def main() -> int:
    args = parse_args()

    fold_name = f"fold_{args.fold:02d}"

    fold_data_dir = (
        FOLDS_ROOT / fold_name
    )

    test_file = (
        fold_data_dir / "test.jsonl"
    )

    fold_output_dir = (
        RESULT_ROOT / fold_name
    )

    best_model_dir = (
        fold_output_dir / "best_model"
    )

    diagnostic = (
        args.max_test_samples
        is not None
    )

    output_dir = (
        fold_output_dir
        / (
            "robustness_diagnostic"
            if diagnostic
            else "robustness_evaluation"
        )
    )

    config = load_yaml(
        CONFIG_PATH
    )

    catalog = load_template_catalog()

    template_ids = tuple(
        catalog["template_ids"]
    )

    if len(template_ids) != 5:
        raise ValueError(
            "EXP-18 robustness evaluation requires "
            "exactly five templates."
        )

    test_records = load_jsonl(
        test_file
    )

    if diagnostic:
        if args.max_test_samples < 1:
            raise ValueError(
                "--max-test-samples must be at least 1."
            )

        test_records = test_records[
            :args.max_test_samples
        ]

    print()
    print("=" * 72)
    print("EXP-18 FIVE-TEMPLATE ROBUSTNESS EVALUATOR")
    print("=" * 72)
    print(f"Fold              : {fold_name}")
    print(f"Test records      : {len(test_records)}")
    print(f"Templates         : {len(template_ids)}")
    print(
        "Expected generations: "
        f"{len(test_records) * len(template_ids)}"
    )
    print(
        "Mode              : "
        + (
            "dry run"
            if args.dry_run
            else (
                "diagnostic"
                if diagnostic
                else "final robustness evaluation"
            )
        )
    )
    print(f"Output folder     : {output_dir}")

    if args.dry_run:
        run_dry_test(
            test_records,
            template_ids,
        )

        print()
        print("=" * 72)
        print(
            "EXP-18 ROBUSTNESS EVALUATOR DRY RUN PASSED"
        )
        print(
            "All five templates produced answer-free prompts."
        )
        print(
            "Consistency metric calculations passed."
        )
        print("Model weights were not loaded.")
        print("No robustness files were created.")
        print("=" * 72)

        return 0

    predictions_file = (
        output_dir
        / "robustness_predictions.jsonl"
    )

    per_template_file = (
        output_dir
        / "robustness_per_template_metrics.json"
    )

    consistency_file = (
        output_dir
        / "robustness_consistency_metrics.json"
    )

    record_consistency_file = (
        output_dir
        / "robustness_record_consistency.jsonl"
    )

    summary_file = (
        output_dir
        / "robustness_summary.json"
    )

    output_files = (
        predictions_file,
        per_template_file,
        consistency_file,
        record_consistency_file,
        summary_file,
    )

    if (
        any(path.exists() for path in output_files)
        and not args.force
    ):
        raise RuntimeError(
            "Robustness outputs already exist. "
            "Use --force only for an intentional rerun."
        )

    if not best_model_dir.is_dir():
        raise FileNotFoundError(
            f"Trained best model not found: "
            f"{best_model_dir}"
        )

    evaluation_start = utc_now()

    print()
    print(
        "Loading base model and trained "
        "EXP-18 QLoRA adapter..."
    )

    model, tokenizer = load_trained_model(
        config=config,
        best_model_dir=best_model_dir,
    )

    max_length = int(
        config["data"]["max_length"]
    )

    predictions: list[
        dict[str, Any]
    ] = []

    for record_index, raw_record in enumerate(
        test_records,
        start=1,
    ):
        record = validate_raw_record(
            raw_record
        )

        for template_id in template_ids:
            prompt = build_prompt(
                raw_record,
                include_answer=False,
                template_id=template_id,
            )

            (
                generated_text,
                prompt_tokens,
                generated_tokens,
            ) = generate_prediction(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_length=max_length,
                max_new_tokens=args.max_new_tokens,
            )

            parsed = parse_generation(
                generated_text
            )

            predictions.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "evaluation_protocol": (
                        EVALUATION_PROTOCOL
                    ),
                    "fold": fold_name,
                    "test_index": record_index,
                    "pair_id": record["pair_id"],
                    "template_id": template_id,
                    "gold_label": record["label"],
                    "predicted_label": parsed[
                        "predicted_label"
                    ],
                    "gold_rationale": record[
                        "rationale"
                    ],
                    "generated_rationale": parsed[
                        "generated_rationale"
                    ],
                    "label_valid": parsed[
                        "label_valid"
                    ],
                    "rationale_present": parsed[
                        "rationale_present"
                    ],
                    "format_valid": parsed[
                        "format_valid"
                    ],
                    "raw_generation": generated_text,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": (
                        generated_tokens
                    ),
                }
            )

            print(
                f"[{record_index}/{len(test_records)}] "
                f"{record['pair_id']} | "
                f"{template_id} -> "
                f"{parsed['predicted_label']}"
            )

    per_template_metrics = {}

    for template_id in template_ids:
        template_rows = [
            row
            for row in predictions
            if row["template_id"] == template_id
        ]

        classification, _ = (
            compute_classification_metrics(
                template_rows
            )
        )

        rationale = (
            compute_rationale_metrics(
                template_rows
            )
        )

        per_template_metrics[
            template_id
        ] = {
            "classification": classification,
            "rationale": rationale,
        }

    consistency_metrics, record_rows = (
        compute_consistency_metrics(
            predictions,
            template_ids,
        )
    )

    template_accuracies = [
        per_template_metrics[
            template_id
        ]["classification"]["accuracy"]
        for template_id in template_ids
    ]

    template_macro_f1 = [
        per_template_metrics[
            template_id
        ]["classification"]["macro_f1"]
        for template_id in template_ids
    ]

    evaluation_end = utc_now()

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "fold": fold_name,
        "diagnostic": diagnostic,
        "evaluation_protocol": (
            EVALUATION_PROTOCOL
        ),
        "primary_comparison_template": (
            "template_01"
        ),
        "template_ids": list(
            template_ids
        ),
        "evaluated_test_records": len(
            test_records
        ),
        "generated_predictions": len(
            predictions
        ),
        "started_at_utc": evaluation_start,
        "completed_at_utc": evaluation_end,
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": (
                args.max_new_tokens
            ),
        },
        "template_accuracy": {
            "mean": mean(
                template_accuracies
            ),
            "minimum": min(
                template_accuracies
            ),
            "maximum": max(
                template_accuracies
            ),
        },
        "template_macro_f1": {
            "mean": mean(
                template_macro_f1
            ),
            "minimum": min(
                template_macro_f1
            ),
            "maximum": max(
                template_macro_f1
            ),
        },
        "consistency": consistency_metrics,
        "interpretation_note": (
            "This robustness evaluation is secondary. "
            "The primary EXP-17 versus EXP-18 comparison "
            "must use template_01 evaluation metrics."
        ),
    }

    save_jsonl(
        predictions_file,
        predictions,
    )

    save_json(
        per_template_file,
        per_template_metrics,
    )

    save_json(
        consistency_file,
        consistency_metrics,
    )

    save_jsonl(
        record_consistency_file,
        record_rows,
    )

    save_json(
        summary_file,
        summary,
    )

    print()
    print("=" * 72)
    print(
        "EXP-18 ROBUSTNESS EVALUATION COMPLETED"
    )
    print(f"Fold            : {fold_name}")
    print(
        "Predictions     : "
        f"{len(predictions)}"
    )
    print(
        "Unanimous rate  : "
        f"{consistency_metrics['unanimous_valid_rate']:.4f}"
    )
    print(
        "Pair agreement  : "
        f"{consistency_metrics['pairwise_agreement_rate']:.4f}"
    )
    print(
        "Majority accuracy: "
        f"{consistency_metrics['majority_vote_accuracy']:.4f}"
    )
    print(f"Outputs         : {output_dir}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
