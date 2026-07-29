#!/usr/bin/env python3
"""
EXP-17: Qwen3.5 Label + Rationale, Single-Template, 10-Fold CV.

Responsibilities:
- load one frozen EXP-17 fold;
- validate train/validation/test separation;
- format raw rationale records in memory;
- tokenize supervised training text;
- initialize the existing QLoRA implementation;
- execute a corrected gradient-accumulation loop;
- evaluate on the validation split;
- save fold-specific checkpoints and training history.

The generic run_ft.py file remains unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
INTEGRATION_ROOT = SRC_ROOT / "integration"

for import_path in (
    REPO_ROOT,
    SRC_ROOT,
    INTEGRATION_ROOT,
):
    import_path_text = str(import_path)

    if import_path_text not in sys.path:
        sys.path.insert(0, import_path_text)


from scripts.exp17.exp17_data import (  # noqa: E402
    build_inference_prompts,
    build_training_texts,
    load_jsonl,
    validate_raw_record,
)

from scripts.exp17.exp17_training_dataset import (  # noqa: E402
    create_train_dataloader,
    create_validation_dataloader,
)

from hivemind_toolkit.ft import FTManager  # noqa: E402

from integration.training_scripts.ft.run_ft import (  # noqa: E402
    create_ft_config,
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

OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "exp17_qwen35_9b_rationale_single_template"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one EXP-17 outer cross-validation fold."
        )
    )

    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        choices=range(1, 11),
        metavar="1-10",
        help="Outer cross-validation fold number.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate fold data, formatting, configuration, "
            "and import paths without loading the model."
        ),
    )

    parser.add_argument(
        "--preview-records",
        type=int,
        default=1,
        help="Number of records to preview during dry run.",
    )

    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=("lora", "qlora", "dora"),
        help=(
            "Override the fine-tuning method configured in YAML."
        ),
    )

    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help=(
            "Optional diagnostic limit. Do not use for final "
            "10-fold experiment runs."
        ),
    )

    return parser.parse_args()


def set_all_seeds(seed: int) -> None:
    """Set reproducibility seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration must contain a YAML object: {path}"
        )

    return config


def validate_config(
    config: dict[str, Any],
) -> None:
    required_sections = {
        "model_path",
        "training",
        "data",
        "evaluation",
        "checkpointing",
    }

    missing = sorted(
        key
        for key in required_sections
        if key not in config
    )

    if missing:
        raise ValueError(
            "Configuration is missing required sections: "
            + ", ".join(missing)
        )

    for section in (
        "training",
        "data",
        "evaluation",
        "checkpointing",
    ):
        if not isinstance(config[section], dict):
            raise ValueError(
                f"Configuration section '{section}' "
                "must be a YAML object."
            )

    training = config["training"]

    required_training_fields = (
        "batch_size",
        "gradient_accumulation_steps",
        "num_epochs",
        "learning_rate",
    )

    for field in required_training_fields:
        if training.get(field) is None:
            raise ValueError(
                f"training.{field} is not configured."
            )

    if int(training["batch_size"]) < 1:
        raise ValueError(
            "training.batch_size must be at least 1."
        )

    if int(
        training["gradient_accumulation_steps"]
    ) < 1:
        raise ValueError(
            "training.gradient_accumulation_steps "
            "must be at least 1."
        )

    if int(training["num_epochs"]) < 1:
        raise ValueError(
            "training.num_epochs must be at least 1."
        )


def get_fold_paths(
    fold_number: int,
) -> dict[str, Path]:
    fold_name = f"fold_{fold_number:02d}"
    fold_root = FOLDS_ROOT / fold_name

    paths = {
        "fold_root": fold_root,
        "train": fold_root / "train.jsonl",
        "validation": fold_root / "validation.jsonl",
        "test": fold_root / "test.jsonl",
    }

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        split_path = paths[split_name]

        if not split_path.is_file():
            raise FileNotFoundError(
                f"{split_name} file not found: "
                f"{split_path}"
            )

    return paths


def normalize_records(
    records: list[dict[str, Any]],
) -> list[dict[str, str]]:
    return [
        validate_raw_record(record)
        for record in records
    ]


def extract_unique_pair_ids(
    records: list[dict[str, str]],
    split_name: str,
) -> set[str]:
    pair_ids = [
        record["pair_id"]
        for record in records
    ]

    counts = Counter(pair_ids)

    duplicate_ids = sorted(
        pair_id
        for pair_id, count in counts.items()
        if count > 1
    )

    if duplicate_ids:
        preview = ", ".join(duplicate_ids[:5])

        raise ValueError(
            f"Duplicate pair_id values in {split_name}: "
            f"{preview}"
        )

    return set(pair_ids)


def verify_no_split_leakage(
    train_ids: set[str],
    validation_ids: set[str],
    test_ids: set[str],
) -> None:
    overlaps = {
        "train/validation": (
            train_ids & validation_ids
        ),
        "train/test": train_ids & test_ids,
        "validation/test": (
            validation_ids & test_ids
        ),
    }

    problems = {
        name: values
        for name, values in overlaps.items()
        if values
    }

    if not problems:
        return

    messages = []

    for name, values in problems.items():
        preview = ", ".join(
            sorted(values)[:5]
        )

        messages.append(
            f"{name}: {len(values)} overlap(s), "
            f"examples: {preview}"
        )

    raise ValueError(
        "Data leakage detected: "
        + "; ".join(messages)
    )


def print_distribution(
    split_name: str,
    records: list[dict[str, str]],
) -> None:
    counts = Counter(
        record["label"]
        for record in records
    )

    total = len(records)

    trace_count = counts.get("trace", 0)
    no_trace_count = counts.get(
        "no_trace",
        0,
    )

    trace_percent = (
        trace_count / total * 100
        if total
        else 0.0
    )

    no_trace_percent = (
        no_trace_count / total * 100
        if total
        else 0.0
    )

    print(
        f"{split_name:<12}: "
        f"{total:>4} records | "
        f"trace={trace_count:>4} "
        f"({trace_percent:5.2f}%) | "
        f"no_trace={no_trace_count:>4} "
        f"({no_trace_percent:5.2f}%)"
    )


def prepare_fold_data(
    fold_number: int,
    max_train_samples: int | None,
) -> dict[str, Any]:
    paths = get_fold_paths(fold_number)

    raw_train = load_jsonl(paths["train"])
    raw_validation = load_jsonl(
        paths["validation"]
    )
    raw_test = load_jsonl(paths["test"])

    if max_train_samples is not None:
        if max_train_samples < 1:
            raise ValueError(
                "--max-train-samples must be at least 1."
            )

        raw_train = raw_train[
            :max_train_samples
        ]

    train_records = normalize_records(raw_train)
    validation_records = normalize_records(
        raw_validation
    )
    test_records = normalize_records(raw_test)

    train_ids = extract_unique_pair_ids(
        train_records,
        "train",
    )

    validation_ids = extract_unique_pair_ids(
        validation_records,
        "validation",
    )

    test_ids = extract_unique_pair_ids(
        test_records,
        "test",
    )

    verify_no_split_leakage(
        train_ids,
        validation_ids,
        test_ids,
    )

    train_texts = build_training_texts(
        raw_train
    )

    validation_texts = build_training_texts(
        raw_validation
    )

    test_prompts = build_inference_prompts(
        raw_test
    )

    if len(train_texts) != len(train_records):
        raise AssertionError(
            "Training text count mismatch."
        )

    if len(validation_texts) != len(
        validation_records
    ):
        raise AssertionError(
            "Validation text count mismatch."
        )

    if len(test_prompts) != len(test_records):
        raise AssertionError(
            "Test prompt count mismatch."
        )

    return {
        "paths": paths,
        "raw_train": raw_train,
        "raw_validation": raw_validation,
        "raw_test": raw_test,
        "train_records": train_records,
        "validation_records": (
            validation_records
        ),
        "test_records": test_records,
        "train_texts": train_texts,
        "validation_texts": validation_texts,
        "test_prompts": test_prompts,
    }



def evaluate_target_masked(
    *,
    model: Any,
    validation_dataloader: DataLoader,
) -> dict[str, Any]:
    """
    Evaluate complete validation data using target-only loss.

    Prompt and padding labels remain -100. Only gold label and
    rationale tokens contribute to validation loss.
    """

    if len(validation_dataloader) == 0:
        raise ValueError(
            "Validation DataLoader contains no batches."
        )

    was_training = bool(model.training)
    model.eval()

    try:
        device = next(
            parameter.device
            for parameter in model.parameters()
            if parameter.device.type != "meta"
        )
    except StopIteration as error:
        raise RuntimeError(
            "Could not determine model device."
        ) from error

    total_weighted_loss = 0.0
    total_supervised_tokens = 0
    evaluated_records = 0
    evaluated_batches = 0

    try:
        with torch.no_grad():
            for batch in tqdm(
                validation_dataloader,
                desc="EXP-17 validation",
            ):
                moved_batch = {
                    key: value.to(device)
                    for key, value in batch.items()
                }

                labels = moved_batch["labels"]

                # Causal-LM loss internally shifts labels by one
                # position, so count supervised shifted tokens.
                supervised_tokens = int(
                    (
                        labels[:, 1:] != -100
                    ).sum().item()
                )

                if supervised_tokens == 0:
                    continue

                outputs = model(
                    **moved_batch
                )

                if outputs.loss is None:
                    raise RuntimeError(
                        "Model did not return validation loss."
                    )

                total_weighted_loss += (
                    float(outputs.loss.item())
                    * supervised_tokens
                )

                total_supervised_tokens += (
                    supervised_tokens
                )

                evaluated_records += int(
                    labels.shape[0]
                )

                evaluated_batches += 1

    finally:
        if was_training:
            model.train()

    if total_supervised_tokens == 0:
        raise RuntimeError(
            "No supervised validation tokens were evaluated."
        )

    average_loss = (
        total_weighted_loss
        / total_supervised_tokens
    )

    # Avoid numerical overflow in pathological diagnostic runs.
    perplexity = float(
        torch.exp(
            torch.tensor(
                min(average_loss, 80.0),
                dtype=torch.float64,
            )
        ).item()
    )

    return {
        "loss": float(average_loss),
        "perplexity": perplexity,
        "n_samples": evaluated_records,
        "n_batches": evaluated_batches,
        "n_supervised_tokens": (
            total_supervised_tokens
        ),
        "loss_scope": (
            "label_and_rationale_tokens_only"
        ),
    }


def corrected_training_loop(
    manager: FTManager,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    output_dir: Path,
) -> dict[str, Any]:
    """
    Execute EXP-17 training with correct remainder handling.

    Unlike the generic loop, the final partial gradient-
    accumulation group is applied instead of discarded.
    """

    if manager.trainer is None:
        raise RuntimeError(
            "Trainer has not been initialized."
        )

    if manager.model is None:
        raise RuntimeError(
            "Model has not been loaded."
        )

    config = manager.config
    trainer = manager.trainer

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("Preparing QLoRA model...")

    manager.model = trainer.prepare_model()
    trainer.model = manager.model

    if hasattr(
        manager.model,
        "enable_input_require_grads",
    ):
        manager.model.enable_input_require_grads()

    trainable_parameters = [
        parameter
        for parameter in manager.model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        raise RuntimeError(
            "No trainable parameters were found "
            "after QLoRA preparation."
        )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    accumulation_steps = int(
        config.gradient_accumulation_steps
    )

    batches_per_epoch = len(train_dataloader)

    optimizer_steps_per_epoch = math.ceil(
        batches_per_epoch
        / accumulation_steps
    )

    total_optimizer_steps = (
        optimizer_steps_per_epoch
        * int(config.num_epochs)
    )

    if config.max_steps is not None:
        total_optimizer_steps = min(
            total_optimizer_steps,
            int(config.max_steps),
        )

    if total_optimizer_steps < 1:
        raise ValueError(
            "Calculated total optimizer steps "
            "must be at least 1."
        )

    warmup_steps = min(
        int(config.warmup_steps),
        max(total_optimizer_steps - 1, 0),
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    history: dict[str, Any] = {
        "train_loss": [],
        "validation_perplexity": [],
        "validation_loss": [],
        "learning_rate": [],
        "global_step": [],
    }

    global_step = 0
    best_validation_perplexity = float("inf")
    best_validation_loss = float("inf")
    stop_training = False

    optimizer.zero_grad(set_to_none=True)

    for epoch_index in range(
        int(config.num_epochs)
    ):
        if stop_training:
            break

        manager.model.train()

        epoch_loss_sum = 0.0
        epoch_batch_count = 0
        pending_accumulation_batches = 0

        progress_bar = tqdm(
            total=optimizer_steps_per_epoch,
            desc=(
                f"EXP-17 Epoch "
                f"{epoch_index + 1}/"
                f"{config.num_epochs}"
            ),
            unit="optimizer-step",
        )

        for batch_index, batch in enumerate(
            train_dataloader
        ):
            loss, metrics = trainer.train_step(
                batch
            )

            raw_loss = float(
                loss.detach().item()
            )

            scaled_loss = (
                loss / accumulation_steps
            )

            scaled_loss.backward()

            epoch_loss_sum += raw_loss
            epoch_batch_count += 1
            pending_accumulation_batches += 1

            is_regular_boundary = (
                pending_accumulation_batches
                == accumulation_steps
            )

            is_last_batch = (
                batch_index + 1
                == batches_per_epoch
            )

            should_update = (
                is_regular_boundary
                or is_last_batch
            )

            if not should_update:
                continue

            if (
                is_last_batch
                and pending_accumulation_batches
                < accumulation_steps
            ):
                correction_factor = (
                    accumulation_steps
                    / pending_accumulation_batches
                )

                for parameter in trainable_parameters:
                    if parameter.grad is not None:
                        parameter.grad.mul_(
                            correction_factor
                        )

            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                config.max_grad_norm,
            )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(
                set_to_none=True
            )

            global_step += 1
            pending_accumulation_batches = 0

            trainer.global_step = global_step
            trainer.epoch = epoch_index

            current_learning_rate = (
                scheduler.get_last_lr()[0]
            )

            average_loss = (
                epoch_loss_sum
                / epoch_batch_count
            )

            progress_bar.set_postfix(
                {
                    "loss": (
                        f"{average_loss:.4f}"
                    ),
                    "lr": (
                        f"{current_learning_rate:.2e}"
                    ),
                    "step": (
                        f"{global_step}/"
                        f"{total_optimizer_steps}"
                    ),
                }
            )

            progress_bar.update(1)

            if (
                global_step
                % int(config.logging_steps)
                == 0
            ):
                history["train_loss"].append(
                    average_loss
                )

                history["learning_rate"].append(
                    current_learning_rate
                )

                history["global_step"].append(
                    global_step
                )

            if (
                int(config.save_steps) > 0
                and global_step
                % int(config.save_steps)
                == 0
            ):
                checkpoint_dir = (
                    output_dir
                    / f"checkpoint-{global_step}"
                )

                trainer.save_checkpoint(
                    str(checkpoint_dir)
                )

            if (
                config.max_steps is not None
                and global_step
                >= int(config.max_steps)
            ):
                stop_training = True
                break

        progress_bar.close()

        if epoch_batch_count == 0:
            raise RuntimeError(
                "No training batches were processed."
            )

        average_epoch_loss = (
            epoch_loss_sum
            / epoch_batch_count
        )

        print()
        print(
            f"Epoch {epoch_index + 1} "
            f"average training loss: "
            f"{average_epoch_loss:.6f}"
        )

        print(
            "Running validation perplexity..."
        )

        validation_results = evaluate_target_masked(
            model=manager.model,
            validation_dataloader=validation_dataloader,
        )

        validation_perplexity = float(
            validation_results["perplexity"]
        )

        validation_loss = float(
            validation_results["loss"]
        )

        history[
            "validation_perplexity"
        ].append(validation_perplexity)

        history[
            "validation_loss"
        ].append(validation_loss)

        print(
            "Validation perplexity: "
            f"{validation_perplexity:.6f}"
        )

        print(
            "Validation loss: "
            f"{validation_loss:.6f}"
        )

        trainer.global_step = global_step
        trainer.epoch = epoch_index + 1
        trainer.best_loss = min(
            getattr(
                trainer,
                "best_loss",
                float("inf"),
            ),
            validation_loss,
        )

        epoch_checkpoint = (
            output_dir
            / f"epoch-{epoch_index + 1}"
        )

        trainer.save_checkpoint(
            str(epoch_checkpoint)
        )

        if (
            validation_perplexity
            < best_validation_perplexity
        ):
            best_validation_perplexity = (
                validation_perplexity
            )
            best_validation_loss = validation_loss

            best_model_dir = (
                output_dir / "best_model"
            )

            trainer.save_checkpoint(
                str(best_model_dir)
            )

            print(
                "Best validation model updated."
            )

    history["completed_optimizer_steps"] = (
        global_step
    )

    history[
        "best_validation_perplexity"
    ] = best_validation_perplexity

    history[
        "best_validation_loss"
    ] = best_validation_loss

    return history


def save_json(
    path: Path,
    payload: Any,
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


def preview_records(
    train_texts: list[str],
    test_prompts: list[str],
    number_to_preview: int,
) -> None:
    preview_count = min(
        max(number_to_preview, 0),
        len(train_texts),
        len(test_prompts),
    )

    for index in range(preview_count):
        print()
        print("=" * 70)
        print(
            f"TRAINING TEXT PREVIEW {index + 1}"
        )
        print("=" * 70)
        print(train_texts[index])

        print()
        print("=" * 70)
        print(
            f"INFERENCE PROMPT PREVIEW {index + 1}"
        )
        print("=" * 70)
        print(test_prompts[index])


def main() -> int:
    args = parse_args()

    fold_name = f"fold_{args.fold:02d}"

    print()
    print("=" * 70)
    print(
        f"{EXPERIMENT_ID} — "
        f"SINGLE-TEMPLATE — "
        f"{fold_name}"
    )
    print("=" * 70)

    config = load_yaml(CONFIG_PATH)
    validate_config(config)

    seed = int(
        config.get(
            "advanced",
            {},
        ).get("seed", 42)
    )

    set_all_seeds(seed)

    fold_data = prepare_fold_data(
        fold_number=args.fold,
        max_train_samples=(
            args.max_train_samples
        ),
    )

    print()
    print("Class distribution:")

    print_distribution(
        "Train",
        fold_data["train_records"],
    )

    print_distribution(
        "Validation",
        fold_data["validation_records"],
    )

    print_distribution(
        "Test",
        fold_data["test_records"],
    )

    print()
    print("Runtime formatting:")

    print(
        "Training texts   : "
        f"{len(fold_data['train_texts'])}"
    )

    print(
        "Validation texts : "
        f"{len(fold_data['validation_texts'])}"
    )

    print(
        "Test prompts     : "
        f"{len(fold_data['test_prompts'])}"
    )

    preview_records(
        fold_data["train_texts"],
        fold_data["test_prompts"],
        args.preview_records,
    )

    fold_output_dir = (
        OUTPUT_ROOT / fold_name
    )

    method = (
        args.method
        or config.get("method", "qlora")
    )

    run_metadata = {
        "experiment_id": EXPERIMENT_ID,
        "fold": args.fold,
        "fold_name": fold_name,
        "method": method,
        "model_path": config["model_path"],
        "seed": seed,
        "config_path": str(CONFIG_PATH),
        "train_path": str(
            fold_data["paths"]["train"]
        ),
        "validation_path": str(
            fold_data["paths"]["validation"]
        ),
        "test_path": str(
            fold_data["paths"]["test"]
        ),
        "train_records": len(
            fold_data["train_records"]
        ),
        "validation_records": len(
            fold_data["validation_records"]
        ),
        "test_records": len(
            fold_data["test_records"]
        ),
        "formatting": "runtime_in_memory",
        "template": "single_template",
        "started_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    if args.dry_run:
        print()
        print(
            json.dumps(
                run_metadata,
                indent=2,
                ensure_ascii=False,
            )
        )

        print()
        print("=" * 70)
        print(
            "EXP-17 TRAINING DRY RUN PASSED"
        )
        print("Model was not loaded.")
        print("Training was not started.")
        print(
            "No additional formatted dataset files were created."
        )
        print("=" * 70)

        return 0

    config.setdefault(
        "checkpointing",
        {},
    )

    config["checkpointing"][
        "output_dir"
    ] = str(fold_output_dir)

    ft_config = create_ft_config(config)

    manager = FTManager(ft_config)

    print()
    print("Loading model and tokenizer...")

    manager.load_model()

    if manager.tokenizer is None:
        raise RuntimeError(
            "Tokenizer was not loaded."
        )

    train_dataloader = create_train_dataloader(
        records=fold_data["raw_train"],
        tokenizer=manager.tokenizer,
        config=config,
    )

    validation_dataloader = create_validation_dataloader(
        records=fold_data["raw_validation"],
        tokenizer=manager.tokenizer,
        config=config,
    )

    print()
    print(
        "Training batches per epoch: "
        f"{len(train_dataloader)}"
    )

    print(
        "Validation batches per epoch: "
        f"{len(validation_dataloader)}"
    )

    print(
        "Validation records per epoch: "
        f"{len(validation_dataloader.dataset)}"
    )

    manager.initialize_trainer(
        method=method
    )

    save_json(
        fold_output_dir
        / "run_metadata.json",
        run_metadata,
    )

    history = corrected_training_loop(
        manager=manager,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        output_dir=fold_output_dir,
    )

    run_metadata["completed_at_utc"] = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    run_metadata["status"] = "completed"

    run_metadata[
        "completed_optimizer_steps"
    ] = history[
        "completed_optimizer_steps"
    ]

    run_metadata[
        "best_validation_perplexity"
    ] = history[
        "best_validation_perplexity"
    ]

    run_metadata[
        "best_validation_loss"
    ] = history[
        "best_validation_loss"
    ]

    save_json(
        fold_output_dir
        / "training_history.json",
        history,
    )

    save_json(
        fold_output_dir
        / "run_metadata.json",
        run_metadata,
    )

    print()
    print("=" * 70)
    print(
        f"{EXPERIMENT_ID} {fold_name} "
        "TRAINING COMPLETED"
    )
    print(
        f"Outputs: {fold_output_dir}"
    )
    print("=" * 70)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except KeyboardInterrupt:
        print()
        print("Training interrupted by user.")
        raise SystemExit(130)

    except Exception as error:
        print()
        print(f"ERROR: {error}")
        raise
