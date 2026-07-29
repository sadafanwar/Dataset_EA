#!/usr/bin/env python3
"""Evaluate one trained EXP-17 fold on its unseen test split."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts.exp17.exp17_data import (  # noqa: E402
    build_prompt,
    load_jsonl,
    validate_raw_record,
)

from scripts.exp17.run_exp17_fold import (  # noqa: E402
    REGISTRY_FILE,
    REGISTRY_HEADERS,
    atomic_write_csv,
    read_csv_rows,
    update_progress,
)


EXPERIMENT_ID = "EXP-17"

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

VALID_LABELS = (
    "trace",
    "no_trace",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one trained EXP-17 fold "
            "on its unseen test split."
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
            "Validate test data, inference prompts, "
            "output parser, and metrics without loading "
            "the trained model."
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
            "Diagnostic-only test limit. Final evaluation "
            "must use the complete test split."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow replacement of existing final evaluation files.",
    )

    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


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


def relative_path(
    path: Path,
) -> str:
    return path.resolve().relative_to(
        REPO_ROOT.resolve()
    ).as_posix()


def save_json(
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


def save_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
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
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def save_confusion_matrix(
    path: Path,
    matrix: dict[str, dict[str, int]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    columns = [
        "gold_label",
        "predicted_trace",
        "predicted_no_trace",
        "invalid_prediction",
        "support",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
        )

        writer.writeheader()

        for gold_label in VALID_LABELS:
            row = matrix[gold_label]

            writer.writerow(
                {
                    "gold_label": gold_label,
                    "predicted_trace": row["trace"],
                    "predicted_no_trace": row["no_trace"],
                    "invalid_prediction": row["invalid"],
                    "support": sum(row.values()),
                }
            )


def normalize_label(
    value: str,
) -> str | None:
    normalized = re.sub(
        r"[\s-]+",
        "_",
        value.strip().lower(),
    )

    if normalized in VALID_LABELS:
        return normalized

    return None


def parse_generation(
    generated_text: str,
) -> dict[str, Any]:
    cleaned = generated_text.strip()

    label_match = re.search(
        r"(?im)"
        r"(?:^|\n)\s*"
        r"(?:\*\*)?label(?:\*\*)?"
        r"\s*:\s*"
        r"(no[\s_-]*trace|trace)\b",
        cleaned,
    )

    predicted_label = (
        normalize_label(
            label_match.group(1)
        )
        if label_match
        else None
    )

    rationale_match = re.search(
        r"(?is)"
        r"(?:^|\n)\s*"
        r"(?:\*\*)?rationale(?:\*\*)?"
        r"\s*:\s*(.+?)\s*$",
        cleaned,
    )

    generated_rationale = (
        rationale_match.group(1).strip()
        if rationale_match
        else ""
    )

    label_valid = (
        predicted_label in VALID_LABELS
    )

    rationale_present = bool(
        generated_rationale
    )

    return {
        "predicted_label": predicted_label,
        "generated_rationale": generated_rationale,
        "label_valid": label_valid,
        "rationale_present": rationale_present,
        "format_valid": (
            label_valid
            and rationale_present
        ),
    }


def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def class_f1(
    precision: float,
    recall: float,
) -> float:
    return safe_divide(
        2 * precision * recall,
        precision + recall,
    )


def compute_classification_metrics(
    predictions: list[dict[str, Any]],
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, int]],
]:
    if not predictions:
        raise ValueError(
            "Cannot calculate metrics from zero predictions."
        )

    total = len(predictions)

    valid_predictions = sum(
        bool(row["label_valid"])
        for row in predictions
    )

    correct_predictions = sum(
        (
            row["label_valid"]
            and row["predicted_label"]
            == row["gold_label"]
        )
        for row in predictions
    )

    confusion = {
        label: {
            "trace": 0,
            "no_trace": 0,
            "invalid": 0,
        }
        for label in VALID_LABELS
    }

    per_class: dict[str, dict[str, Any]] = {}

    for row in predictions:
        gold = row["gold_label"]
        predicted = row["predicted_label"]

        if predicted in VALID_LABELS:
            confusion[gold][predicted] += 1
        else:
            confusion[gold]["invalid"] += 1

    for label in VALID_LABELS:
        true_positive = sum(
            1
            for row in predictions
            if (
                row["gold_label"] == label
                and row["predicted_label"] == label
            )
        )

        false_positive = sum(
            1
            for row in predictions
            if (
                row["gold_label"] != label
                and row["predicted_label"] == label
            )
        )

        false_negative = sum(
            1
            for row in predictions
            if (
                row["gold_label"] == label
                and row["predicted_label"] != label
            )
        )

        support = sum(
            1
            for row in predictions
            if row["gold_label"] == label
        )

        precision = safe_divide(
            true_positive,
            true_positive + false_positive,
        )

        recall = safe_divide(
            true_positive,
            true_positive + false_negative,
        )

        f1_value = class_f1(
            precision,
            recall,
        )

        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1_value,
            "support": support,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
        }

    macro_f1 = mean(
        per_class[label]["f1"]
        for label in VALID_LABELS
    )

    macro_precision = mean(
        per_class[label]["precision"]
        for label in VALID_LABELS
    )

    macro_recall = mean(
        per_class[label]["recall"]
        for label in VALID_LABELS
    )

    weighted_f1 = safe_divide(
        sum(
            per_class[label]["f1"]
            * per_class[label]["support"]
            for label in VALID_LABELS
        ),
        total,
    )

    format_valid = sum(
        bool(row["format_valid"])
        for row in predictions
    )

    metrics = {
        "total_test_records": total,
        "correct_predictions": correct_predictions,
        "valid_predictions": valid_predictions,
        "invalid_predictions": (
            total - valid_predictions
        ),
        "accuracy": safe_divide(
            correct_predictions,
            total,
        ),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "format_valid_predictions": format_valid,
        "format_valid_rate": safe_divide(
            format_valid,
            total,
        ),
        "per_class": per_class,
        "invalid_output_policy": (
            "Invalid labels count as incorrect predictions "
            "and as false negatives for the gold class."
        ),
        "confusion_matrix_columns": [
            "trace",
            "no_trace",
            "invalid",
        ],
    }

    return metrics, confusion


def word_tokens(
    text: str,
) -> list[str]:
    return re.findall(
        r"[a-z0-9]+",
        text.lower(),
    )


def lexical_f1(
    reference: str,
    candidate: str,
) -> float:
    reference_tokens = word_tokens(
        reference
    )

    candidate_tokens = word_tokens(
        candidate
    )

    if not reference_tokens or not candidate_tokens:
        return 0.0

    reference_counts: dict[str, int] = {}
    candidate_counts: dict[str, int] = {}

    for token in reference_tokens:
        reference_counts[token] = (
            reference_counts.get(token, 0)
            + 1
        )

    for token in candidate_tokens:
        candidate_counts[token] = (
            candidate_counts.get(token, 0)
            + 1
        )

    overlap = sum(
        min(
            reference_counts.get(token, 0),
            candidate_counts.get(token, 0),
        )
        for token in set(
            reference_counts
        )
        | set(candidate_counts)
    )

    precision = safe_divide(
        overlap,
        len(candidate_tokens),
    )

    recall = safe_divide(
        overlap,
        len(reference_tokens),
    )

    return class_f1(
        precision,
        recall,
    )


def lcs_length(
    first: list[str],
    second: list[str],
) -> int:
    previous = [
        0
    ] * (len(second) + 1)

    for first_token in first:
        current = [
            0
        ] * (len(second) + 1)

        for index, second_token in enumerate(
            second,
            start=1,
        ):
            if first_token == second_token:
                current[index] = (
                    previous[index - 1]
                    + 1
                )
            else:
                current[index] = max(
                    current[index - 1],
                    previous[index],
                )

        previous = current

    return previous[-1]


def rouge_l_f1(
    reference: str,
    candidate: str,
) -> float:
    reference_tokens = word_tokens(
        reference
    )

    candidate_tokens = word_tokens(
        candidate
    )

    if not reference_tokens or not candidate_tokens:
        return 0.0

    longest_common_subsequence = lcs_length(
        reference_tokens,
        candidate_tokens,
    )

    precision = safe_divide(
        longest_common_subsequence,
        len(candidate_tokens),
    )

    recall = safe_divide(
        longest_common_subsequence,
        len(reference_tokens),
    )

    return class_f1(
        precision,
        recall,
    )


def compute_rationale_metrics(
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(predictions)

    present_rows = [
        row
        for row in predictions
        if row["rationale_present"]
    ]

    word_counts = [
        len(
            word_tokens(
                row["generated_rationale"]
            )
        )
        for row in present_rows
    ]

    lexical_scores = [
        lexical_f1(
            row["gold_rationale"],
            row["generated_rationale"],
        )
        for row in present_rows
    ]

    rouge_scores = [
        rouge_l_f1(
            row["gold_rationale"],
            row["generated_rationale"],
        )
        for row in present_rows
    ]

    return {
        "total_test_records": total,
        "rationales_present": len(
            present_rows
        ),
        "rationales_missing": (
            total - len(present_rows)
        ),
        "rationale_present_rate": safe_divide(
            len(present_rows),
            total,
        ),
        "generated_rationale_word_count": {
            "mean": (
                mean(word_counts)
                if word_counts
                else 0.0
            ),
            "median": (
                median(word_counts)
                if word_counts
                else 0.0
            ),
            "minimum": (
                min(word_counts)
                if word_counts
                else 0
            ),
            "maximum": (
                max(word_counts)
                if word_counts
                else 0
            ),
        },
        "lexical_token_f1_mean": (
            mean(lexical_scores)
            if lexical_scores
            else 0.0
        ),
        "rouge_l_f1_mean": (
            mean(rouge_scores)
            if rouge_scores
            else 0.0
        ),
        "interpretation_note": (
            "Lexical overlap metrics do not establish "
            "rationale correctness or faithfulness. "
            "Human or semantic evaluation is required "
            "for final qualitative claims."
        ),
    }


def get_model_input_device(
    model: Any,
) -> Any:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device

    raise RuntimeError(
        "Could not determine model input device."
    )


def load_trained_model(
    *,
    config: dict[str, Any],
    best_model_dir: Path,
) -> tuple[Any, Any]:
    import torch

    from peft import PeftModel

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    model_path = config["model_path"]

    tokenizer = AutoTokenizer.from_pretrained(
        best_model_dir,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "left"

    quantization = config["quantization"]

    compute_dtype_name = quantization.get(
        "compute_dtype",
        "bfloat16",
    )

    compute_dtype = getattr(
        torch,
        compute_dtype_name,
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quantization.get(
            "type",
            "nf4",
        ),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=bool(
            quantization.get(
                "double_quant",
                True,
            )
        ),
    )

    model_kwargs = {
        "quantization_config": quantization_config,
        "device_map": "auto",
        "trust_remote_code": True,
    }

    try:
        base_model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=compute_dtype,
                **model_kwargs,
            )
        )
    except TypeError:
        base_model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=compute_dtype,
                **model_kwargs,
            )
        )

    model = PeftModel.from_pretrained(
        base_model,
        best_model_dir,
    )

    model.eval()

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    return model, tokenizer


def generate_prediction(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_length: int,
    max_new_tokens: int,
) -> tuple[str, int, int]:
    import torch

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    input_device = get_model_input_device(
        model
    )

    inputs = {
        key: value.to(input_device)
        for key, value in inputs.items()
    }

    input_length = int(
        inputs["input_ids"].shape[1]
    )

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    continuation_ids = generated_ids[
        0,
        input_length:,
    ]

    generated_text = tokenizer.decode(
        continuation_ids,
        skip_special_tokens=True,
    ).strip()

    return (
        generated_text,
        input_length,
        int(continuation_ids.shape[0]),
    )


def update_registry_after_evaluation(
    *,
    fold_name: str,
    classification_metrics: dict[str, Any],
    rationale_metrics: dict[str, Any],
    metrics_file: Path,
    predictions_file: Path,
) -> None:
    rows = read_csv_rows(
        REGISTRY_FILE,
        REGISTRY_HEADERS,
    )

    candidates = [
        (
            index,
            row,
        )
        for index, row in enumerate(rows)
        if (
            row["experiment_id"]
            == EXPERIMENT_ID
            and row["fold"]
            == fold_name
        )
    ]

    if not candidates:
        raise RuntimeError(
            f"No training registry row found for {fold_name}."
        )

    row_index, row = max(
        candidates,
        key=lambda item: item[1][
            "start_time_utc"
        ],
    )

    trace_metrics = (
        classification_metrics[
            "per_class"
        ]["trace"]
    )

    no_trace_metrics = (
        classification_metrics[
            "per_class"
        ]["no_trace"]
    )

    row.update(
        {
            "status": "evaluation_completed",
            "metrics_file": relative_path(
                metrics_file
            ),
            "predictions_file": relative_path(
                predictions_file
            ),
            "accuracy": (
                classification_metrics[
                    "accuracy"
                ]
            ),
            "macro_f1": (
                classification_metrics[
                    "macro_f1"
                ]
            ),
            "trace_precision": (
                trace_metrics["precision"]
            ),
            "trace_recall": (
                trace_metrics["recall"]
            ),
            "trace_f1": trace_metrics["f1"],
            "no_trace_precision": (
                no_trace_metrics["precision"]
            ),
            "no_trace_recall": (
                no_trace_metrics["recall"]
            ),
            "no_trace_f1": (
                no_trace_metrics["f1"]
            ),
            "valid_predictions": (
                classification_metrics[
                    "valid_predictions"
                ]
            ),
            "invalid_predictions": (
                classification_metrics[
                    "invalid_predictions"
                ]
            ),
            "format_valid_rate": (
                classification_metrics[
                    "format_valid_rate"
                ]
            ),
            "rationale_present_rate": (
                rationale_metrics[
                    "rationale_present_rate"
                ]
            ),
            "notes": (
                row.get("notes", "")
                + " Test evaluation completed."
            ).strip(),
        }
    )

    rows[row_index] = row

    atomic_write_csv(
        REGISTRY_FILE,
        REGISTRY_HEADERS,
        rows,
    )


def run_dry_test(
    test_records: list[dict[str, Any]],
) -> None:
    if not test_records:
        raise ValueError(
            "Test split is empty."
        )

    for raw_record in test_records[:5]:
        prompt = build_prompt(
            raw_record,
            include_answer=False,
        )

        full_text = build_prompt(
            raw_record,
            include_answer=True,
        )

        if not full_text.startswith(
            prompt + "\nLabel:"
        ):
            raise ValueError(
                "Inference prompt separation failed."
            )

    synthetic_rows = [
        {
            "gold_label": "trace",
            **parse_generation(
                "Label: trace\n"
                "Rationale: The requirements describe "
                "the same behavior."
            ),
        },
        {
            "gold_label": "no_trace",
            **parse_generation(
                "Label: no_trace\n"
                "Rationale: The requirements concern "
                "different functions."
            ),
        },
        {
            "gold_label": "trace",
            **parse_generation(
                "Label: no_trace\n"
                "Rationale: Synthetic incorrect output."
            ),
        },
        {
            "gold_label": "no_trace",
            **parse_generation(
                "Unable to determine the relationship."
            ),
        },
    ]

    metrics, confusion = (
        compute_classification_metrics(
            synthetic_rows
        )
    )

    if metrics["total_test_records"] != 4:
        raise ValueError(
            "Synthetic metrics test failed."
        )

    if metrics["valid_predictions"] != 3:
        raise ValueError(
            "Output-parser valid count is incorrect."
        )

    if metrics["invalid_predictions"] != 1:
        raise ValueError(
            "Output-parser invalid count is incorrect."
        )

    if sum(
        sum(row.values())
        for row in confusion.values()
    ) != 4:
        raise ValueError(
            "Confusion matrix test failed."
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

    config = load_yaml(
        CONFIG_PATH
    )

    test_records = load_jsonl(
        test_file
    )

    diagnostic = (
        args.max_test_samples
        is not None
    )

    if diagnostic:
        if args.max_test_samples < 1:
            raise ValueError(
                "--max-test-samples must be at least 1."
            )

        test_records = test_records[
            :args.max_test_samples
        ]

        evaluation_dir = (
            fold_output_dir
            / "diagnostic_evaluation"
        )
    else:
        evaluation_dir = fold_output_dir

    print()
    print("=" * 70)
    print("EXP-17 TEST EVALUATOR")
    print("=" * 70)
    print(f"Fold          : {fold_name}")
    print(f"Test records  : {len(test_records)}")
    print(
        "Mode          : "
        + (
            "dry run"
            if args.dry_run
            else (
                "diagnostic evaluation"
                if diagnostic
                else "final evaluation"
            )
        )
    )
    print(f"Test file     : {test_file}")
    print(f"Output folder : {evaluation_dir}")

    if args.dry_run:
        run_dry_test(
            test_records
        )

        print()
        print("=" * 70)
        print("EXP-17 EVALUATOR DRY RUN PASSED")
        print(
            "Inference prompts contain no gold answer."
        )
        print(
            "Output parser and metric calculations passed."
        )
        print("Model weights were not loaded.")
        print("No evaluation files were created.")
        print("=" * 70)

        return 0

    predictions_file = (
        evaluation_dir
        / "test_predictions.jsonl"
    )

    metrics_file = (
        evaluation_dir
        / "classification_metrics.json"
    )

    rationale_file = (
        evaluation_dir
        / "rationale_metrics.json"
    )

    confusion_file = (
        evaluation_dir
        / "confusion_matrix.csv"
    )

    summary_file = (
        evaluation_dir
        / "fold_evaluation_summary.json"
    )

    output_files = (
        predictions_file,
        metrics_file,
        rationale_file,
        confusion_file,
        summary_file,
    )

    if (
        any(path.exists() for path in output_files)
        and not args.force
    ):
        raise RuntimeError(
            "Evaluation outputs already exist. "
            "Use --force only for an intentional rerun."
        )

    if not best_model_dir.is_dir():
        raise FileNotFoundError(
            f"Trained best model not found: "
            f"{best_model_dir}"
        )

    evaluation_start = utc_now()

    if not diagnostic:
        update_progress(
            fold_name=fold_name,
            status="evaluating",
            stage="evaluation",
            start_time_utc=evaluation_start,
            message="Unseen test-set evaluation started.",
        )

    try:
        print()
        print("Loading base model and trained QLoRA adapter...")

        model, tokenizer = load_trained_model(
            config=config,
            best_model_dir=best_model_dir,
        )

        predictions: list[
            dict[str, Any]
        ] = []

        max_length = int(
            config["data"]["max_length"]
        )

        for index, raw_record in enumerate(
            test_records,
            start=1,
        ):
            record = validate_raw_record(
                raw_record
            )

            prompt = build_prompt(
                raw_record,
                include_answer=False,
            )

            generated_text, prompt_tokens, generated_tokens = (
                generate_prediction(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_length=max_length,
                    max_new_tokens=args.max_new_tokens,
                )
            )

            parsed = parse_generation(
                generated_text
            )

            prediction = {
                "experiment_id": EXPERIMENT_ID,
                "fold": fold_name,
                "test_index": index,
                "pair_id": record["pair_id"],
                "higher_level_requirement": (
                    record[
                        "higher_level_requirement"
                    ]
                ),
                "lower_level_requirement": (
                    record[
                        "lower_level_requirement"
                    ]
                ),
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
                "generated_tokens": generated_tokens,
            }

            predictions.append(
                prediction
            )

            print(
                f"[{index}/{len(test_records)}] "
                f"{record['pair_id']} -> "
                f"{parsed['predicted_label']}"
            )

        classification_metrics, confusion = (
            compute_classification_metrics(
                predictions
            )
        )

        rationale_metrics = (
            compute_rationale_metrics(
                predictions
            )
        )

        evaluation_end = utc_now()

        summary = {
            "experiment_id": EXPERIMENT_ID,
            "fold": fold_name,
            "diagnostic": diagnostic,
            "model": config["model_path"],
            "best_model_dir": relative_path(
                best_model_dir
            ),
            "test_file": relative_path(
                test_file
            ),
            "evaluated_records": len(
                predictions
            ),
            "started_at_utc": (
                evaluation_start
            ),
            "completed_at_utc": (
                evaluation_end
            ),
            "generation": {
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": (
                    args.max_new_tokens
                ),
            },
            "classification": (
                classification_metrics
            ),
            "rationale": (
                rationale_metrics
            ),
        }

        save_jsonl(
            predictions_file,
            predictions,
        )

        save_json(
            metrics_file,
            classification_metrics,
        )

        save_json(
            rationale_file,
            rationale_metrics,
        )

        save_confusion_matrix(
            confusion_file,
            confusion,
        )

        save_json(
            summary_file,
            summary,
        )

        if not diagnostic:
            update_registry_after_evaluation(
                fold_name=fold_name,
                classification_metrics=(
                    classification_metrics
                ),
                rationale_metrics=(
                    rationale_metrics
                ),
                metrics_file=metrics_file,
                predictions_file=(
                    predictions_file
                ),
            )

            update_progress(
                fold_name=fold_name,
                status="evaluation_completed",
                stage="evaluation",
                start_time_utc=evaluation_start,
                end_time_utc=evaluation_end,
                message=(
                    "Unseen test-set evaluation "
                    "completed successfully."
                ),
            )

        print()
        print("=" * 70)
        print("EXP-17 FOLD EVALUATION COMPLETED")
        print(f"Fold      : {fold_name}")
        print(
            "Accuracy  : "
            f"{classification_metrics['accuracy']:.4f}"
        )
        print(
            "Macro-F1  : "
            f"{classification_metrics['macro_f1']:.4f}"
        )
        print(
            "Valid rate: "
            f"{classification_metrics['format_valid_rate']:.4f}"
        )
        print(f"Outputs   : {evaluation_dir}")
        print("=" * 70)

        return 0

    except Exception as error:
        if not diagnostic:
            failure_time = utc_now()

            update_progress(
                fold_name=fold_name,
                status="evaluation_failed",
                stage="evaluation",
                start_time_utc=evaluation_start,
                end_time_utc=failure_time,
                message=(
                    f"{type(error).__name__}: "
                    f"{error}"
                ),
            )

        raise


if __name__ == "__main__":
    raise SystemExit(main())
