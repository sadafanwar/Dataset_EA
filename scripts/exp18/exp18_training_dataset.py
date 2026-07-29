"""EXP-18 causal-LM dataset with multi-template prompt masking."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from scripts.exp17.exp17_data import (
    validate_raw_record,
)
from scripts.exp18.exp18_data import (
    DEFAULT_TEMPLATE_SEED,
    build_prompt,
)


class SupervisedCausalDataset(Dataset):
    """
    Tokenize EXP-18 records with deterministic templates.

    Labels use:
    - -100 for BOS, instruction, HLR, LLR, and Answer:
    - token IDs for the gold label and rationale
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer: Any,
        max_length: int,
        *,
        template_seed: int = DEFAULT_TEMPLATE_SEED,
    ) -> None:
        if not records:
            raise ValueError(
                "Cannot create a dataset from zero records."
            )

        if max_length < 2:
            raise ValueError(
                "max_length must be at least 2."
            )

        self.items: list[
            dict[str, list[int]]
        ] = []

        self.template_seed = int(template_seed)

        truncated_records = 0
        truncated_targets = 0
        total_prompt_tokens = 0
        total_supervised_tokens = 0

        for record in records:
            item = validate_raw_record(record)

            prompt = build_prompt(
                record,
                include_answer=False,
                seed=self.template_seed,
            )

            target = (
                f"Label: {item['label']}\n"
                f"Rationale: {item['rationale']}"
            )

            prefix_ids: list[int] = []

            if tokenizer.bos_token_id is not None:
                prefix_ids.append(
                    tokenizer.bos_token_id
                )

            prompt_ids = tokenizer.encode(
                prompt,
                add_special_tokens=False,
            )

            target_ids = tokenizer.encode(
                "\n" + target,
                add_special_tokens=False,
            )

            if tokenizer.eos_token_id is not None:
                target_ids.append(
                    tokenizer.eos_token_id
                )

            input_ids = (
                prefix_ids
                + prompt_ids
                + target_ids
            )

            labels = (
                [-100]
                * (
                    len(prefix_ids)
                    + len(prompt_ids)
                )
                + target_ids
            )

            original_target_tokens = len(target_ids)

            if len(input_ids) > max_length:
                truncated_records += 1
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]

            supervised_tokens = sum(
                token_id != -100
                for token_id in labels
            )

            if supervised_tokens == 0:
                raise ValueError(
                    f"{item['pair_id']}: max_length="
                    f"{max_length} removed the complete "
                    "gold label and rationale."
                )

            if (
                supervised_tokens
                < original_target_tokens
            ):
                truncated_targets += 1

            total_prompt_tokens += sum(
                token_id == -100
                for token_id in labels
            )

            total_supervised_tokens += (
                supervised_tokens
            )

            self.items.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": (
                        [1] * len(input_ids)
                    ),
                    "labels": labels,
                }
            )

        self.stats = {
            "records": len(self.items),
            "template_seed": self.template_seed,
            "truncated_records": truncated_records,
            "truncated_targets": truncated_targets,
            "masked_prompt_tokens": (
                total_prompt_tokens
            ),
            "supervised_target_tokens": (
                total_supervised_tokens
            ),
        }

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, list[int]]:
        return self.items[index]


class SupervisedCausalCollator:
    """Right-pad inputs while retaining -100 label masks."""

    def __init__(
        self,
        tokenizer: Any,
    ) -> None:
        if tokenizer.pad_token_id is None:
            raise ValueError(
                "Tokenizer must define pad_token_id."
            )

        self.pad_token_id = (
            tokenizer.pad_token_id
        )

    def __call__(
        self,
        features: list[
            dict[str, list[int]]
        ],
    ) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError(
                "Cannot collate an empty batch."
            )

        batch_max_length = max(
            len(feature["input_ids"])
            for feature in features
        )

        input_ids_batch = []
        attention_mask_batch = []
        labels_batch = []

        for feature in features:
            padding_length = (
                batch_max_length
                - len(feature["input_ids"])
            )

            input_ids_batch.append(
                feature["input_ids"]
                + [self.pad_token_id]
                * padding_length
            )

            attention_mask_batch.append(
                feature["attention_mask"]
                + [0] * padding_length
            )

            labels_batch.append(
                feature["labels"]
                + [-100] * padding_length
            )

        return {
            "input_ids": torch.tensor(
                input_ids_batch,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                attention_mask_batch,
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                labels_batch,
                dtype=torch.long,
            ),
        }


def _prepare_tokenizer_padding(
    tokenizer: Any,
) -> None:
    """Ensure deterministic right padding."""

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                "Tokenizer defines neither PAD "
                "nor EOS token."
            )

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

        tokenizer.pad_token_id = (
            tokenizer.eos_token_id
        )

    tokenizer.padding_side = "right"


def _template_seed(
    config: dict[str, Any],
) -> int:
    """Read the frozen assignment seed from config."""

    advanced_seed = int(
        config.get(
            "advanced",
            {},
        ).get(
            "seed",
            DEFAULT_TEMPLATE_SEED,
        )
    )

    assignment = (
        config.get(
            "multi_template",
            {},
        ).get(
            "assignment",
            {},
        )
    )

    assignment_seed = int(
        assignment.get(
            "seed",
            advanced_seed,
        )
    )

    if assignment_seed != advanced_seed:
        raise ValueError(
            "EXP-18 assignment seed and advanced "
            "seed must remain identical."
        )

    return assignment_seed


def create_train_dataloader(
    records: list[dict[str, Any]],
    tokenizer: Any,
    config: dict[str, Any],
) -> DataLoader:
    """Build the shuffled EXP-18 target-masked DataLoader."""

    _prepare_tokenizer_padding(tokenizer)

    data_config = config["data"]
    training_config = config["training"]
    seed = _template_seed(config)

    dataset = SupervisedCausalDataset(
        records=records,
        tokenizer=tokenizer,
        max_length=int(
            data_config.get(
                "max_length",
                1024,
            )
        ),
        template_seed=seed,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=int(
            training_config["batch_size"]
        ),
        shuffle=True,
        collate_fn=(
            SupervisedCausalCollator(
                tokenizer
            )
        ),
        num_workers=int(
            data_config.get(
                "num_workers",
                0,
            )
        ),
        generator=generator,
    )


def create_validation_dataloader(
    records: list[dict[str, Any]],
    tokenizer: Any,
    config: dict[str, Any],
) -> DataLoader:
    """
    Build the complete deterministic validation DataLoader.

    Validation uses the same stable pair-to-template assignment,
    no shuffling, and target-only loss masking.
    """

    _prepare_tokenizer_padding(tokenizer)

    data_config = config["data"]
    evaluation_config = config.get(
        "evaluation",
        {},
    )
    seed = _template_seed(config)

    dataset = SupervisedCausalDataset(
        records=records,
        tokenizer=tokenizer,
        max_length=int(
            data_config.get(
                "max_length",
                1024,
            )
        ),
        template_seed=seed,
    )

    return DataLoader(
        dataset,
        batch_size=int(
            evaluation_config.get(
                "batch_size",
                1,
            )
        ),
        shuffle=False,
        collate_fn=(
            SupervisedCausalCollator(
                tokenizer
            )
        ),
        num_workers=int(
            data_config.get(
                "num_workers",
                0,
            )
        ),
    )
