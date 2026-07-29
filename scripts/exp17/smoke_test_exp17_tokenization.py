#!/usr/bin/env python3
"""Validate EXP-17 target-masked tokenization and DataLoader."""

from __future__ import annotations

import sys
from pathlib import Path

from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts.exp17.exp17_data import load_jsonl  # noqa: E402

from scripts.exp17.exp17_training_dataset import (  # noqa: E402
    create_train_dataloader,
)

from scripts.exp17.train_exp17 import (  # noqa: E402
    CONFIG_PATH,
    get_fold_paths,
    load_yaml,
)


SMOKE_TEST_RECORDS = 8


def main() -> int:
    print()
    print("=" * 70)
    print("EXP-17 TARGET-MASKED TOKENIZATION SMOKE TEST")
    print("=" * 70)

    config = load_yaml(CONFIG_PATH)
    model_path = config["model_path"]

    fold_paths = get_fold_paths(1)

    raw_records = load_jsonl(
        fold_paths["train"]
    )

    smoke_records = raw_records[
        :SMOKE_TEST_RECORDS
    ]

    print(f"Model/tokenizer ID : {model_path}")
    print(f"Input records      : {len(smoke_records)}")
    print("Formatted in RAM   : yes")
    print("Dataset files made : no")

    print()
    print("Loading tokenizer only...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    train_dataloader = create_train_dataloader(
        records=smoke_records,
        tokenizer=tokenizer,
        config=config,
    )

    batch = next(iter(train_dataloader))

    required_keys = {
        "input_ids",
        "attention_mask",
        "labels",
    }

    missing_keys = required_keys - set(batch)

    if missing_keys:
        raise ValueError(
            "Batch is missing fields: "
            + ", ".join(sorted(missing_keys))
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
            "Batch tensor shapes do not match."
        )

    masked_prompt_tokens = int(
        (labels == -100).sum().item()
    )

    supervised_target_tokens = int(
        (labels != -100).sum().item()
    )

    if masked_prompt_tokens < 1:
        raise ValueError(
            "Prompt tokens were not masked."
        )

    if supervised_target_tokens < 1:
        raise ValueError(
            "No target tokens are supervised."
        )

    if (
        supervised_target_tokens
        == labels.numel()
    ):
        raise ValueError(
            "All tokens are supervised; "
            "prompt masking failed."
        )

    dataset_stats = (
        train_dataloader.dataset.stats
    )

    print()
    print("First DataLoader batch:")
    print(f"Batch keys        : {sorted(batch.keys())}")
    print(f"input_ids shape   : {tuple(input_ids.shape)}")
    print(f"labels shape      : {tuple(labels.shape)}")
    print(f"Masked prompt     : {masked_prompt_tokens}")
    print(f"Supervised target : {supervised_target_tokens}")

    print()
    print("Dataset statistics:")

    for key, value in dataset_stats.items():
        print(f"{key:<25}: {value}")

    decoded_text = tokenizer.decode(
        input_ids[0],
        skip_special_tokens=True,
    )

    print()
    print("Decoded first example preview:")
    print("-" * 70)
    print(decoded_text[:1000])
    print("-" * 70)

    if "Label:" not in decoded_text:
        raise ValueError(
            "Decoded example lacks Label."
        )

    if "Rationale:" not in decoded_text:
        raise ValueError(
            "Decoded example lacks Rationale."
        )

    print()
    print("=" * 70)
    print(
        "EXP-17 TARGET-MASKING SMOKE TEST PASSED"
    )
    print("Prompt tokens are excluded from loss.")
    print(
        "Only label and rationale tokens "
        "are supervised."
    )
    print("Model weights were not loaded.")
    print("Training was not started.")
    print(
        "No additional dataset files were created."
    )
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
