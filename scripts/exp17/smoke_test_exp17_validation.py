#!/usr/bin/env python3
"""Validate EXP-17 target-masked validation DataLoader."""

from __future__ import annotations

import sys
from pathlib import Path

from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts.exp17.exp17_data import load_jsonl  # noqa: E402

from scripts.exp17.exp17_training_dataset import (  # noqa: E402
    create_validation_dataloader,
)

from scripts.exp17.train_exp17 import (  # noqa: E402
    CONFIG_PATH,
    get_fold_paths,
    load_yaml,
)


EXPECTED_VALIDATION_RECORDS = 170


def main() -> int:
    print()
    print("=" * 70)
    print("EXP-17 TARGET-MASKED VALIDATION SMOKE TEST")
    print("=" * 70)

    config = load_yaml(CONFIG_PATH)

    evaluation_config = config["evaluation"]

    if evaluation_config["eval_split"] != "validation":
        raise ValueError(
            "eval_split must be 'validation'."
        )

    if int(
        evaluation_config["eval_samples"]
    ) != EXPECTED_VALIDATION_RECORDS:
        raise ValueError(
            "eval_samples must equal 170."
        )

    fold_paths = get_fold_paths(1)

    validation_records = load_jsonl(
        fold_paths["validation"]
    )

    if (
        len(validation_records)
        != EXPECTED_VALIDATION_RECORDS
    ):
        raise ValueError(
            "Fold 01 validation split contains "
            f"{len(validation_records)} records; "
            "expected 170."
        )

    model_path = config["model_path"]

    print(f"Tokenizer          : {model_path}")
    print(
        "Validation records : "
        f"{len(validation_records)}"
    )
    print(
        "Maximum length     : "
        f"{config['data']['max_length']}"
    )
    print(
        "Evaluation split   : "
        f"{evaluation_config['eval_split']}"
    )

    print()
    print("Loading tokenizer only...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    validation_dataloader = (
        create_validation_dataloader(
            records=validation_records,
            tokenizer=tokenizer,
            config=config,
        )
    )

    dataset = validation_dataloader.dataset
    stats = dataset.stats

    if len(dataset) != EXPECTED_VALIDATION_RECORDS:
        raise ValueError(
            "Validation DataLoader dataset size "
            f"is {len(dataset)}; expected 170."
        )

    if stats["truncated_records"] != 0:
        raise ValueError(
            "Validation records were truncated."
        )

    if stats["truncated_targets"] != 0:
        raise ValueError(
            "Validation targets were truncated."
        )

    total_records = 0
    total_batches = 0
    total_masked_tokens = 0
    total_supervised_tokens = 0

    first_batch = None

    for batch in validation_dataloader:
        if first_batch is None:
            first_batch = batch

        required_keys = {
            "input_ids",
            "attention_mask",
            "labels",
        }

        missing = required_keys - set(batch)

        if missing:
            raise ValueError(
                "Validation batch is missing fields: "
                + ", ".join(sorted(missing))
            )

        input_ids = batch["input_ids"]
        attention_mask = batch[
            "attention_mask"
        ]
        labels = batch["labels"]

        if not (
            input_ids.shape
            == attention_mask.shape
            == labels.shape
        ):
            raise ValueError(
                "Validation batch tensor shapes differ."
            )

        masked_tokens = int(
            (labels == -100).sum().item()
        )

        supervised_tokens = int(
            (labels != -100).sum().item()
        )

        if masked_tokens == 0:
            raise ValueError(
                "A validation batch has no "
                "masked prompt tokens."
            )

        if supervised_tokens == 0:
            raise ValueError(
                "A validation batch has no "
                "supervised target tokens."
            )

        total_records += int(
            labels.shape[0]
        )

        total_batches += 1

        total_masked_tokens += (
            masked_tokens
        )

        total_supervised_tokens += (
            supervised_tokens
        )

    if total_records != EXPECTED_VALIDATION_RECORDS:
        raise ValueError(
            f"Processed {total_records} validation "
            "records; expected 170."
        )

    if first_batch is None:
        raise ValueError(
            "Validation DataLoader produced no batches."
        )

    decoded_first = tokenizer.decode(
        first_batch["input_ids"][0],
        skip_special_tokens=True,
    )

    if "Label:" not in decoded_first:
        raise ValueError(
            "Decoded validation record lacks Label."
        )

    if "Rationale:" not in decoded_first:
        raise ValueError(
            "Decoded validation record lacks Rationale."
        )

    print()
    print("Validation DataLoader results:")
    print(f"Dataset records     : {len(dataset)}")
    print(f"Processed records   : {total_records}")
    print(f"Validation batches  : {total_batches}")
    print(
        f"Masked prompt tokens: "
        f"{total_masked_tokens}"
    )
    print(
        f"Supervised targets  : "
        f"{total_supervised_tokens}"
    )
    print(
        f"Truncated records   : "
        f"{stats['truncated_records']}"
    )
    print(
        f"Truncated targets   : "
        f"{stats['truncated_targets']}"
    )

    print()
    print("=" * 70)
    print(
        "EXP-17 VALIDATION SMOKE TEST PASSED"
    )
    print(
        "All 170 validation records were processed."
    )
    print(
        "Prompt and padding tokens are excluded "
        "from validation loss."
    )
    print(
        "Only label and rationale tokens "
        "are supervised."
    )
    print("Model weights were not loaded.")
    print("Training was not started.")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
