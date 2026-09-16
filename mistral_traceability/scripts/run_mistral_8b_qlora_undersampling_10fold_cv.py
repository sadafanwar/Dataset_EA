#!/usr/bin/env python3
"""Ministral QLoRA 10-fold traceability experiment with frozen EXP-07 undersampling."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import os
import platform
import random
import shutil
import statistics
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Mistral3ForConditionalGeneration,
    get_linear_schedule_with_warmup,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / (
    "mistral_traceability/configs/config_mistral_8b_qlora_undersampling_10fold.yaml"
)
LABELS = ["trace", "no_trace"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fold_name(fold: int) -> str:
    return f"fold_{fold:02d}"


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def relative_repo_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")


def canonical_jsonl_sha256(path: Path) -> str:
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def raw_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc
    return records


def build_prompt(record: dict[str, Any]) -> str:
    instruction = record.get(
        "instruction",
        "Determine whether the lower-level requirement traces to the higher-level requirement.",
    )
    input_obj = record.get("input", {})
    higher_req = input_obj.get("higher_level_requirement", "")
    lower_req = input_obj.get("lower_level_requirement", "")
    return (
        f"{instruction}\n\n"
        f"Higher-level requirement:\n{higher_req}\n\n"
        f"Lower-level requirement:\n{lower_req}\n\n"
        f"Answer:"
    )


def build_training_text(record: dict[str, Any]) -> str:
    return f"{build_prompt(record)} {record['output']}"


def normalize_prediction(raw_text: str) -> str:
    text = raw_text.strip().lower()
    text = text.replace("\n", " ").replace(".", " ").replace(",", " ").strip()
    tokens = text.split()
    first_token = tokens[0] if tokens else ""
    if (
        "no_trace" in text
        or "no trace" in text
        or "not trace" in text
        or first_token == "no"
    ):
        return "no_trace"
    if first_token == "yes" or first_token == "trace" or text.startswith("trace"):
        return "trace"
    return "invalid"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config["data"]["balancing"] != "random_undersampling_without_replacement":
        raise ValueError("This experiment must use random undersampling without replacement")
    if config["data"]["recreate_splits"]:
        raise ValueError("Frozen folds must never be regenerated")
    undersampling = config["data"]["undersampling"]
    if undersampling != {
        "method": "random_sample_without_replacement",
        "majority_class": "no_trace",
        "target_rule": "match_inner_train_minority_count",
        "seed_base": 42,
        "seed_rule": "seed_base_plus_fold_number",
        "shuffle_after_resampling": True,
        "use_frozen_files": True,
        "regenerate": False,
    }:
        raise ValueError("Undersampling protocol differs from historical EXP-07")
    if config["training"]["dataloader_shuffle"]:
        raise ValueError("Historical EXP-07 used DataLoader shuffle=False")
    if config["prompt"]["chat_template"]:
        raise ValueError("Historical EXP-07 did not use a chat template")
    if config["prompt"]["loss_scope"] != "full_prompt_and_label":
        raise ValueError("Historical EXP-07 trained on full prompt-and-label loss")
    if config["model"]["class_name"] != "Mistral3ForConditionalGeneration":
        raise ValueError("Unexpected model class")
    training = config["training"]
    if training["effective_batch_size"] != (
        training["batch_size"] * training["gradient_accumulation_steps"]
    ):
        raise ValueError("Effective batch size does not match batch and accumulation settings")
    if training["max_steps"] is not None:
        raise ValueError("Historical EXP-07 used two complete epochs, not max_steps")
    if training["optimizer"] != "AdamW" or training["scheduler"] != "linear_with_warmup":
        raise ValueError("Optimizer or scheduler differs from historical EXP-07")
    if training["partial_accumulation_policy"] != (
        "discard_incomplete_accumulation_at_each_epoch_historical_exp07"
    ):
        raise ValueError("Partial gradient accumulation policy differs from EXP-07")
    if not config["validation"]["compute_perplexity"]:
        raise ValueError("Historical EXP-07 monitored validation perplexity")
    if config["evaluation"]["labels_order"] != LABELS:
        raise ValueError("Evaluation label order must match historical EXP-07")
    if config["evaluation"]["do_sample"]:
        raise ValueError("Final classification evaluation must use greedy decoding")
    if config["evaluation"]["invalid_prediction_policy"] != (
        "exclude_from_classification_metrics"
    ):
        raise ValueError("Invalid-prediction policy differs from historical EXP-07")
    if config["checkpointing"]["best_model_criterion"] != (
        "lowest_epoch_training_loss_historical_exp07"
    ):
        raise ValueError("Best-model selection differs from historical EXP-07")
    return config


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("mistral_8b_qlora_undersampling_10fold")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gpu_memory() -> dict[str, float]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    mib = 1024**2
    return {
        "free_mib": round(free_bytes / mib, 2),
        "total_mib": round(total_bytes / mib, 2),
        "global_used_mib": round((total_bytes - free_bytes) / mib, 2),
        "process_allocated_mib": round(torch.cuda.memory_allocated(0) / mib, 2),
        "process_reserved_mib": round(torch.cuda.memory_reserved(0) / mib, 2),
        "process_peak_allocated_mib": round(torch.cuda.max_memory_allocated(0) / mib, 2),
        "process_peak_reserved_mib": round(torch.cuda.max_memory_reserved(0) / mib, 2),
    }


def split_paths(config: dict[str, Any], fold: int) -> dict[str, Path]:
    name = fold_name(fold)
    qwen_root = resolve_repo_path(config["data"]["qwen_data_root"])
    mistral_test_root = resolve_repo_path(config["data"]["mistral_test_root"])
    return {
        "source_train": qwen_root / "inner_splits" / name / "inner_train.jsonl",
        "train": qwen_root / "inner_splits" / name / "train_undersampled.jsonl",
        "validation": qwen_root / "inner_splits" / name / "validation.jsonl",
        "qwen_test": qwen_root / "folds" / name / "test.jsonl",
        "test": mistral_test_root / name / "test.jsonl",
    }


def audit_dataset(path: Path, expected: dict[str, Any], split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {split} file: {path}")
    records = load_jsonl(path)
    counts = Counter(record.get("output") for record in records)
    example_ids = [str(record.get("example_id", "")) for record in records]
    if any(not example_id for example_id in example_ids):
        raise ValueError(f"Every {split} record must have a non-empty example_id")
    observed = {
        "path": relative_repo_path(path),
        "records": len(records),
        "trace": counts.get("trace", 0),
        "no_trace": counts.get("no_trace", 0),
        "other_labels": {
            str(label): count for label, count in counts.items() if label not in LABELS
        },
        "unique_example_ids": len(set(example_ids)),
        "duplicate_rows": len(example_ids) - len(set(example_ids)),
        "sha256": canonical_jsonl_sha256(path),
        "raw_sha256": raw_sha256(path),
    }
    for key in (
        "records",
        "trace",
        "no_trace",
        "unique_example_ids",
        "duplicate_rows",
        "sha256",
    ):
        if observed[key] != expected[key]:
            raise ValueError(
                f"{split} provenance mismatch for {key}: "
                f"expected {expected[key]!r}, observed {observed[key]!r}"
            )
    if observed["other_labels"]:
        raise ValueError(f"Unexpected labels in {split}: {observed['other_labels']}")
    return records, observed


def record_ids(
    records: list[dict[str, Any]], *, allow_duplicates: bool = False
) -> set[str]:
    ids = [str(record.get("example_id", "")) for record in records]
    if any(not value for value in ids):
        raise ValueError("Every record must have a non-empty example_id")
    if not allow_duplicates and len(ids) != len(set(ids)):
        raise ValueError("Duplicate example_id values found within a split")
    return set(ids)


def reproduce_historical_undersampling(
    source_records: list[dict[str, Any]], seed: int
) -> list[dict[str, Any]]:
    """Reproduce scripts/create_undersampled_inner_train_fold.py at 55c82a5."""
    rng = random.Random(seed)
    trace_records = [record for record in source_records if record["output"] == "trace"]
    no_trace_records = [
        record for record in source_records if record["output"] == "no_trace"
    ]
    target_count = min(len(trace_records), len(no_trace_records))
    trace_balanced = (
        rng.sample(trace_records, target_count)
        if len(trace_records) > target_count
        else trace_records
    )
    no_trace_balanced = (
        rng.sample(no_trace_records, target_count)
        if len(no_trace_records) > target_count
        else no_trace_records
    )
    balanced = trace_balanced + no_trace_balanced
    rng.shuffle(balanced)
    return balanced


def audit_fold(config: dict[str, Any], manifest: dict[str, Any], fold: int) -> dict[str, Any]:
    name = fold_name(fold)
    expected = manifest["folds"][name]
    paths = split_paths(config, fold)
    source_train, source_train_audit = audit_dataset(
        paths["source_train"], expected["source_train"], "source_train"
    )
    train, train_audit = audit_dataset(paths["train"], expected["train"], "train")
    validation, validation_audit = audit_dataset(
        paths["validation"], expected["validation"], "validation"
    )
    qwen_test, qwen_test_audit = audit_dataset(paths["qwen_test"], expected["test"], "qwen_test")
    test, test_audit = audit_dataset(paths["test"], expected["test"], "mistral_test")
    if test_audit["sha256"] != qwen_test_audit["sha256"]:
        raise ValueError(f"{name} Mistral test fold is not identical to the frozen Qwen test fold")
    expected_seed = config["data"]["undersampling"]["seed_base"] + fold
    if expected["undersampling_seed"] != expected_seed:
        raise ValueError(f"{name} undersampling seed differs from seed_base + fold")
    if reproduce_historical_undersampling(source_train, expected_seed) != train:
        raise ValueError(
            f"{name} frozen train file does not exactly reproduce historical EXP-07 undersampling"
        )
    train_ids = record_ids(train)
    validation_ids = record_ids(validation)
    test_ids = record_ids(test)
    if train_ids & validation_ids:
        raise ValueError(f"{name} train/validation leakage detected")
    if train_ids & test_ids:
        raise ValueError(f"{name} train/test leakage detected")
    if validation_ids & test_ids:
        raise ValueError(f"{name} validation/test leakage detected")
    return {
        "source_train": source_train_audit,
        "train": train_audit,
        "validation": validation_audit,
        "test": test_audit,
        "qwen_reference_test": qwen_test_audit,
        "mistral_test_matches_qwen_test": True,
        "exact_historical_undersampling": True,
        "undersampling_seed": expected_seed,
        "no_example_id_overlap": True,
    }


def require_passed_preflight(config: dict[str, Any]) -> dict[str, Any]:
    result_root = resolve_repo_path(config["paths"]["result_root"])
    report_path = result_root / "preflight" / "fold_01" / "preflight_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Required Fold 01 preflight report not found: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "passed" or report.get("errors"):
        raise RuntimeError("Fold 01 preflight has not passed cleanly")
    hyperparameters = report.get("hyperparameters", {})
    training = config["training"]
    lora = config["lora"]
    expected = {
        "learning_rate": training["learning_rate"],
        "batch_size": training["batch_size"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "effective_batch_size": training["effective_batch_size"],
        "num_epochs": training["num_epochs"],
        "warmup_steps": training["warmup_steps"],
        "max_grad_norm": training["max_grad_norm"],
        "weight_decay": training["weight_decay"],
        "max_length": training["max_length"],
        "bf16": training["bf16"],
        "gradient_checkpointing": training["gradient_checkpointing"],
        "gradient_checkpointing_use_reentrant": training[
            "gradient_checkpointing_use_reentrant"
        ],
        "seed": training["seed"],
        "dataloader_shuffle": training["dataloader_shuffle"],
        "partial_accumulation_policy": training["partial_accumulation_policy"],
        "balancing": config["data"]["balancing"],
        "undersampling_fold_seed": config["data"]["undersampling"]["seed_base"] + 1,
        "eval_steps": config["validation"]["eval_steps"],
        "eval_samples": config["validation"]["eval_samples"],
        "save_steps": config["checkpointing"]["save_steps"],
        "save_total_limit": config["checkpointing"]["save_total_limit"],
        "logging_steps": config["logging"]["logging_steps"],
        "lora_r": lora["r"],
        "lora_alpha": lora["alpha"],
        "lora_dropout": lora["dropout"],
        "lora_target_modules_regex": lora["target_modules_regex"],
        "fix_mistral_regex": config["model"]["fix_mistral_regex"],
    }
    mismatches = {
        key: {"preflight": hyperparameters.get(key), "training_config": value}
        for key, value in expected.items()
        if hyperparameters.get(key) != value
    }
    if hyperparameters.get("quantization") != config["quantization"]:
        mismatches["quantization"] = {
            "preflight": hyperparameters.get("quantization"),
            "training_config": config["quantization"],
        }
    if mismatches:
        raise RuntimeError(f"Training config does not match passed preflight: {mismatches}")
    return report


class TokenizedDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], tokenizer: Any, max_length: int):
        self.features = [
            tokenizer(
                build_training_text(record),
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None,
            )
            for record in records
        ]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.features[index]


def load_tokenizer(config: dict[str, Any], source: str | Path | None = None) -> Any:
    model_config = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        source or model_config["id"],
        revision=model_config["revision"] if source is None else None,
        trust_remote_code=True,
        local_files_only=model_config["local_files_only"],
        fix_mistral_regex=model_config["fix_mistral_regex"],
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def quantization_config(config: dict[str, Any]) -> BitsAndBytesConfig:
    quantization = config["quantization"]
    if quantization["compute_dtype"] != "bfloat16":
        raise ValueError("Only bfloat16 compute is approved for this experiment")
    return BitsAndBytesConfig(
        load_in_4bit=quantization["load_in_4bit"],
        bnb_4bit_quant_type=quantization["quant_type"],
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=quantization["double_quant"],
    )


def load_quantized_base(config: dict[str, Any]) -> Any:
    model_config = config["model"]
    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
        quantization_config=quantization_config(config),
        device_map={"": 0},
        dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=model_config["local_files_only"],
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    return model


def load_bf16_base_for_evaluation(config: dict[str, Any]) -> Any:
    if config["evaluation"]["base_model_precision"] != "bfloat16":
        raise ValueError("Historical EXP-07 final evaluation used a bfloat16 base model")
    model_config = config["model"]
    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
        device_map={"": 0},
        dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=model_config["local_files_only"],
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = True
    return model


def attach_lora(config: dict[str, Any], model: Any) -> tuple[Any, dict[str, Any]]:
    training = config["training"]
    lora = config["lora"]
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=training["gradient_checkpointing"],
        gradient_checkpointing_kwargs={
            "use_reentrant": training["gradient_checkpointing_use_reentrant"]
        },
    )
    adapter_config = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        bias=lora["bias"],
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora["target_modules_regex"],
    )
    model = get_peft_model(model, adapter_config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    adapter_modules = [name for name, module in model.named_modules() if hasattr(module, "lora_A")]
    invalid_modules = [
        name
        for name in adapter_modules
        if ".language_model.layers." not in name or ".self_attn." not in name
    ]
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    if invalid_modules:
        raise ValueError(f"LoRA attached outside text self-attention: {invalid_modules}")
    if len(adapter_modules) != lora["expected_adapter_modules"]:
        raise ValueError(
            f"Expected {lora['expected_adapter_modules']} adapters, observed {len(adapter_modules)}"
        )
    if trainable != lora["expected_trainable_parameters"]:
        raise ValueError(
            f"Expected {lora['expected_trainable_parameters']} trainable parameters, observed {trainable}"
        )
    return model, {
        "adapter_module_count": len(adapter_modules),
        "trainable_parameters": trainable,
        "all_parameters": total,
        "trainable_percent": 100.0 * trainable / total,
        "vision_or_projector_adapter_modules": invalid_modules,
    }


@torch.no_grad()
def evaluate_perplexity(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    max_length: int,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    evaluated = 0
    for text in tqdm(texts, desc="Validation perplexity", leave=False):
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        outputs = model(**encoded, labels=encoded["input_ids"])
        token_count = encoded["input_ids"].numel()
        total_loss += outputs.loss.item() * token_count
        total_tokens += token_count
        evaluated += 1
    if was_training:
        model.train()
    if total_tokens == 0:
        raise ValueError("No valid validation tokens")
    average_loss = total_loss / total_tokens
    return {
        "loss": average_loss,
        "perplexity": float(np.exp(average_loss)),
        "n_samples": evaluated,
        "n_tokens": total_tokens,
    }


def save_adapter_checkpoint(
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    state: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(state, output_dir / "trainer_state.pt")
    (output_dir / "experiment_metadata.json").write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )


def cleanup_checkpoints(output_dir: Path, output_root: Path, limit: int, logger: logging.Logger) -> None:
    resolved_output = output_dir.resolve()
    resolved_root = output_root.resolve()
    if not resolved_output.is_relative_to(resolved_root):
        raise RuntimeError(f"Refusing checkpoint cleanup outside {resolved_root}")
    checkpoints = [
        path
        for path in output_dir.iterdir()
        if path.is_dir() and path.name.startswith("checkpoint-")
    ]
    checkpoints.sort(key=lambda path: int(path.name.split("-")[1]))
    while len(checkpoints) > limit:
        old = checkpoints.pop(0)
        shutil.rmtree(old)
        logger.info("Removed old checkpoint due to save_total_limit: %s", old)


def train_one_fold(
    config: dict[str, Any],
    fold: int,
    audit: dict[str, Any],
    logger: logging.Logger,
) -> tuple[Path, dict[str, Any]]:
    name = fold_name(fold)
    paths = split_paths(config, fold)
    training = config["training"]
    validation_config = config["validation"]
    checkpointing = config["checkpointing"]
    output_root = resolve_repo_path(config["paths"]["output_root"])
    result_root = resolve_repo_path(config["paths"]["result_root"])
    output_dir = output_root / name
    result_dir = result_root / name
    best_model_dir = output_dir / "best_model"
    metrics_file = result_dir / f"{name}_metrics.json"
    if metrics_file.exists():
        logger.info("%s already completed; metrics exist at %s", name, metrics_file)
        return best_model_dir, {"status": "skipped_completed"}
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Partial output already exists for {name}: {output_dir}. "
            "Inspect it before deciding how to recover; automatic deletion is disabled."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    train_records = load_jsonl(paths["train"])
    validation_records = load_jsonl(paths["validation"])
    validation_texts = [
        build_training_text(record)
        for record in validation_records[: validation_config["eval_samples"]]
    ]
    set_seed(training["seed"])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    memory = {"before_model_load": gpu_memory()}
    tokenizer = load_tokenizer(config)
    tokenized = TokenizedDataset(train_records, tokenizer, training["max_length"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(
        tokenized,
        batch_size=training["batch_size"],
        shuffle=training["dataloader_shuffle"],
        collate_fn=collator,
        num_workers=training["num_workers"],
        pin_memory=training["num_workers"] > 0,
    )
    expected_microbatches = training["expected_train_microbatches_by_fold"][name]
    if len(dataloader) != expected_microbatches:
        raise RuntimeError(
            f"{name} expected {expected_microbatches} "
            f"training microbatches, observed {len(dataloader)}"
        )
    model = load_quantized_base(config)
    memory["after_quantized_model_load"] = gpu_memory()
    baseline_validation = evaluate_perplexity(
        model,
        tokenizer,
        validation_texts,
        training["max_length"],
    )
    logger.info("%s baseline validation: %s", name, baseline_validation)
    model, lora_audit = attach_lora(config, model)
    memory["after_lora_attach"] = gpu_memory()
    logger.info("%s LoRA audit: %s", name, lora_audit)

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    total_optimizer_steps = (
        len(dataloader) * training["num_epochs"]
    ) // training["gradient_accumulation_steps"]
    expected_scheduler_steps = training["scheduler_total_steps_by_fold"][name]
    if total_optimizer_steps != expected_scheduler_steps:
        raise RuntimeError("Scheduler step count differs from historical EXP-07")
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training["warmup_steps"],
        num_training_steps=total_optimizer_steps,
    )
    history: dict[str, Any] = {
        "train_loss": [],
        "eval_perplexity": [],
        "learning_rate": [],
        "epochs": [],
    }
    global_step = 0
    best_loss = float("inf")
    model.train()

    for epoch in range(training["num_epochs"]):
        epoch_loss = 0.0
        optimizer.zero_grad()
        expected_steps = len(dataloader) // training["gradient_accumulation_steps"]
        if expected_steps != training["expected_optimizer_steps_per_epoch"]:
            raise RuntimeError("Optimizer steps per epoch differ from historical EXP-07")
        progress = tqdm(total=expected_steps, desc=f"{name} epoch {epoch + 1}", unit="step")
        microbatches = 0
        for step, batch in enumerate(dataloader):
            microbatches += 1
            batch = {key: value.to(model.device) for key, value in batch.items()}
            outputs = model(**batch)
            raw_loss = outputs.loss
            (raw_loss / training["gradient_accumulation_steps"]).backward()
            epoch_loss += raw_loss.item()
            if (step + 1) % training["gradient_accumulation_steps"] != 0:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), training["max_grad_norm"]
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            if global_step % 10 == 0:
                torch.cuda.empty_cache()
            average_loss = epoch_loss / (step + 1)
            progress.set_postfix(
                loss=f"{average_loss:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
                global_step=f"{global_step}/{total_optimizer_steps}",
            )
            progress.update(1)
            if global_step % config["logging"]["logging_steps"] == 0:
                history["train_loss"].append(
                    {"step": global_step, "loss": average_loss, "gradient_norm": float(grad_norm)}
                )
                history["learning_rate"].append(
                    {"step": global_step, "learning_rate": scheduler.get_last_lr()[0]}
                )
            if global_step % validation_config["eval_steps"] == 0:
                result = evaluate_perplexity(
                    model,
                    tokenizer,
                    validation_texts,
                    training["max_length"],
                )
                result["step"] = global_step
                history["eval_perplexity"].append(result)
                logger.info("%s step %d validation: %s", name, global_step, result)
            if global_step % checkpointing["save_steps"] == 0:
                checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                state = {
                    "experiment_id": config["experiment_id"],
                    "fold": name,
                    "global_step": global_step,
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "model_id": config["model"]["id"],
                    "model_revision": config["model"]["revision"],
                    "saved_at_utc": utc_now(),
                }
                save_adapter_checkpoint(model, tokenizer, checkpoint_dir, state)
                cleanup_checkpoints(
                    output_dir,
                    output_root,
                    checkpointing["save_total_limit"],
                    logger,
                )
        progress.close()
        average_epoch_loss = epoch_loss / microbatches
        discarded_microbatches = microbatches % training["gradient_accumulation_steps"]
        epoch_record = {
            "epoch": epoch + 1,
            "average_training_loss": average_epoch_loss,
            "microbatches": microbatches,
            "optimizer_steps": expected_steps,
            "discarded_partial_accumulation_microbatches": discarded_microbatches,
        }
        history["epochs"].append(epoch_record)
        logger.info("%s epoch result: %s", name, epoch_record)
        if average_epoch_loss < best_loss:
            best_loss = average_epoch_loss
            state = {
                "experiment_id": config["experiment_id"],
                "fold": name,
                "global_step": global_step,
                "epoch": epoch,
                "best_loss": best_loss,
                "selection_criterion": checkpointing["best_model_criterion"],
                "model_id": config["model"]["id"],
                "model_revision": config["model"]["revision"],
                "saved_at_utc": utc_now(),
            }
            save_adapter_checkpoint(model, tokenizer, best_model_dir, state)
            logger.info("%s best model saved with training loss %.6f", name, best_loss)

    final_validation = evaluate_perplexity(
        model,
        tokenizer,
        validation_texts,
        training["max_length"],
    )
    validation_perplexity_improvement_percent = (
        (baseline_validation["perplexity"] - final_validation["perplexity"])
        / baseline_validation["perplexity"]
        * 100.0
    )
    final_model_dir = output_dir / "final_model"
    final_state = {
        "experiment_id": config["experiment_id"],
        "fold": name,
        "global_step": global_step,
        "epoch": training["num_epochs"] - 1,
        "best_loss": best_loss,
        "model_id": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "saved_at_utc": utc_now(),
    }
    save_adapter_checkpoint(model, tokenizer, final_model_dir, final_state)
    logger.info("%s final validation: %s", name, final_validation)
    logger.info(
        "%s validation perplexity improvement: %.4f%%",
        name,
        validation_perplexity_improvement_percent,
    )
    torch.cuda.synchronize()
    memory["after_training"] = gpu_memory()
    training_report = {
        "status": "completed",
        "fold": name,
        "dataset_audit": audit,
        "lora": lora_audit,
        "total_optimizer_steps": total_optimizer_steps,
        "completed_optimizer_steps": global_step,
        "best_epoch_training_loss": best_loss,
        "best_model_criterion": checkpointing["best_model_criterion"],
        "baseline_validation": baseline_validation,
        "final_validation": final_validation,
        "validation_perplexity_improvement_percent": validation_perplexity_improvement_percent,
        "final_model_dir": relative_repo_path(final_model_dir),
        "gpu_memory_mib": memory,
        "history": history,
    }
    if global_step != training["expected_completed_optimizer_steps_per_fold"]:
        raise RuntimeError("Completed optimizer steps differ from historical EXP-07")
    (result_dir / f"{name}_training.json").write_text(
        json.dumps(training_report, indent=2), encoding="utf-8"
    )
    del outputs, raw_loss, batch, grad_norm, model, optimizer, scheduler
    del dataloader, tokenized, collator, tokenizer, progress
    gc.collect()
    torch.cuda.empty_cache()
    return best_model_dir, training_report


def evaluate_best_model(
    config: dict[str, Any],
    fold: int,
    best_model_dir: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    name = fold_name(fold)
    result_dir = resolve_repo_path(config["paths"]["result_root"]) / name
    test_file = split_paths(config, fold)["test"]
    tokenizer = load_tokenizer(config, best_model_dir)
    base_model = load_bf16_base_for_evaluation(config)
    model = PeftModel.from_pretrained(base_model, best_model_dir, is_trainable=False)
    model.config.use_cache = True
    model.eval()
    records = load_jsonl(test_file)
    y_true: list[str] = []
    y_pred: list[str] = []
    rows: list[dict[str, Any]] = []
    for record in tqdm(records, desc=f"{name} test evaluation"):
        prompt = build_prompt(record)
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=config["evaluation"]["max_length"],
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=config["evaluation"]["max_new_tokens"],
                do_sample=config["evaluation"]["do_sample"],
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated_text = tokenizer.decode(
            generated[0][encoded["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        prediction = normalize_prediction(generated_text)
        true_label = record["output"]
        y_true.append(true_label)
        y_pred.append(prediction)
        rows.append(
            {
                "example_id": record.get("example_id", ""),
                "true_label": true_label,
                "pred_label": prediction,
                "raw_prediction": generated_text.strip(),
                "higher_level_requirement": record.get("input", {}).get(
                    "higher_level_requirement", ""
                ),
                "lower_level_requirement": record.get("input", {}).get(
                    "lower_level_requirement", ""
                ),
            }
        )
    valid_indices = [index for index, value in enumerate(y_pred) if value in LABELS]
    invalid_indices = [index for index, value in enumerate(y_pred) if value not in LABELS]
    if not valid_indices:
        raise ValueError("No valid test predictions were produced")
    y_true_valid = [y_true[index] for index in valid_indices]
    y_pred_valid = [y_pred[index] for index in valid_indices]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_valid,
        y_pred_valid,
        labels=LABELS,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true_valid, y_pred_valid, labels=LABELS)
    metrics = {
        "model_dir": relative_repo_path(best_model_dir),
        "test_file": relative_repo_path(test_file),
        "total_records": len(records),
        "valid_predictions": len(valid_indices),
        "invalid_predictions": len(invalid_indices),
        "accuracy": float(accuracy_score(y_true_valid, y_pred_valid)),
        "labels_order": LABELS,
        "trace_precision": float(precision[0]),
        "trace_recall": float(recall[0]),
        "trace_f1": float(f1[0]),
        "trace_support": int(support[0]),
        "no_trace_precision": float(precision[1]),
        "no_trace_recall": float(recall[1]),
        "no_trace_f1": float(f1[1]),
        "no_trace_support": int(support[1]),
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            y_true_valid,
            y_pred_valid,
            labels=LABELS,
            zero_division=0,
            output_dict=True,
        ),
        "invalid_prediction_policy": config["evaluation"]["invalid_prediction_policy"],
        "completed_at_utc": utc_now(),
    }
    predictions_file = result_dir / f"{name}_predictions.csv"
    with predictions_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    metrics_file = result_dir / f"{name}_metrics.json"
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("%s test metrics: %s", name, metrics)
    del model, base_model
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def registry_fields() -> list[str]:
    return [
        "experiment_name",
        "fold",
        "status",
        "start_time_utc",
        "end_time_utc",
        "duration_minutes",
        "model",
        "model_revision",
        "method",
        "balancing",
        "train_file",
        "validation_file",
        "test_file",
        "train_total",
        "train_trace",
        "train_no_trace",
        "validation_total",
        "validation_trace",
        "validation_no_trace",
        "test_total",
        "test_trace",
        "test_no_trace",
        "output_dir",
        "best_model_dir",
        "metrics_file",
        "predictions_file",
        "accuracy",
        "trace_precision",
        "trace_recall",
        "trace_f1",
        "no_trace_precision",
        "no_trace_recall",
        "no_trace_f1",
        "valid_predictions",
        "invalid_predictions",
        "confusion_matrix",
        "notes",
    ]


def append_csv(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def track_fold(
    config: dict[str, Any],
    fold: int,
    audit: dict[str, Any],
    status: str,
    start: str,
    end: str,
    metrics: dict[str, Any] | None,
    notes: str,
) -> None:
    name = fold_name(fold)
    result_root = resolve_repo_path(config["paths"]["result_root"])
    output_root = resolve_repo_path(config["paths"]["output_root"])
    paths = split_paths(config, fold)
    duration = (
        datetime.fromisoformat(end) - datetime.fromisoformat(start)
    ).total_seconds() / 60
    metrics = metrics or {}
    row = {
        "experiment_name": config["experiment_name"],
        "fold": name,
        "status": status,
        "start_time_utc": start,
        "end_time_utc": end,
        "duration_minutes": round(duration, 3),
        "model": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "method": "QLoRA",
        "balancing": config["data"]["balancing"],
        "train_file": relative_repo_path(paths["train"]),
        "validation_file": relative_repo_path(paths["validation"]),
        "test_file": relative_repo_path(paths["test"]),
        "train_total": audit["train"]["records"],
        "train_trace": audit["train"]["trace"],
        "train_no_trace": audit["train"]["no_trace"],
        "validation_total": audit["validation"]["records"],
        "validation_trace": audit["validation"]["trace"],
        "validation_no_trace": audit["validation"]["no_trace"],
        "test_total": audit["test"]["records"],
        "test_trace": audit["test"]["trace"],
        "test_no_trace": audit["test"]["no_trace"],
        "output_dir": relative_repo_path(output_root / name),
        "best_model_dir": relative_repo_path(output_root / name / "best_model"),
        "metrics_file": relative_repo_path(result_root / name / f"{name}_metrics.json"),
        "predictions_file": relative_repo_path(result_root / name / f"{name}_predictions.csv"),
        "accuracy": metrics.get("accuracy", ""),
        "trace_precision": metrics.get("trace_precision", ""),
        "trace_recall": metrics.get("trace_recall", ""),
        "trace_f1": metrics.get("trace_f1", ""),
        "no_trace_precision": metrics.get("no_trace_precision", ""),
        "no_trace_recall": metrics.get("no_trace_recall", ""),
        "no_trace_f1": metrics.get("no_trace_f1", ""),
        "valid_predictions": metrics.get("valid_predictions", ""),
        "invalid_predictions": metrics.get("invalid_predictions", ""),
        "confusion_matrix": json.dumps(metrics.get("confusion_matrix", "")),
        "notes": notes,
    }
    append_csv(result_root / "run_registry.csv", registry_fields(), row)
    append_csv(
        result_root / "progress.csv",
        ["fold", "status", "start_time_utc", "end_time_utc", "duration_minutes", "metrics_path"],
        {
            "fold": name,
            "status": status,
            "start_time_utc": start,
            "end_time_utc": end,
            "duration_minutes": round(duration, 3),
            "metrics_path": row["metrics_file"],
        },
    )


def create_summary(config: dict[str, Any]) -> None:
    result_root = resolve_repo_path(config["paths"]["result_root"])
    metric_keys = [
        "accuracy",
        "trace_precision",
        "trace_recall",
        "trace_f1",
        "no_trace_precision",
        "no_trace_recall",
        "no_trace_f1",
    ]
    rows = []
    summed_matrix = [[0, 0], [0, 0]]
    for fold in range(1, 11):
        name = fold_name(fold)
        metrics_path = result_root / name / f"{name}_metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Cannot create 10-fold summary; missing {metrics_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        row = {"fold": name, **{key: float(metrics[key]) for key in metric_keys}}
        rows.append(row)
        matrix = metrics["confusion_matrix"]
        for i in range(2):
            for j in range(2):
                summed_matrix[i][j] += matrix[i][j]
    summary = {
        key: {
            "mean": statistics.mean(row[key] for row in rows),
            "std": statistics.stdev(row[key] for row in rows),
            "min": min(row[key] for row in rows),
            "max": max(row[key] for row in rows),
        }
        for key in metric_keys
    }
    result = {
        "experiment": config["experiment_name"],
        "model": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "method": "QLoRA",
        "balancing": config["data"]["balancing"],
        "folds": rows,
        "summary": summary,
        "summed_confusion_matrix_labels": LABELS,
        "summed_confusion_matrix": summed_matrix,
        "created_at_utc": utc_now(),
    }
    json_path = result_root / "mistral_8b_qlora_undersampling_10fold_summary.json"
    csv_path = result_root / "mistral_8b_qlora_undersampling_10fold_summary.csv"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "mean", "std", "min", "max"])
        for key in metric_keys:
            writer.writerow(
                [key, summary[key]["mean"], summary[key]["std"], summary[key]["min"], summary[key]["max"]]
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--start-fold", type=int, default=1)
    parser.add_argument("--end-fold", type=int, default=10)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.start_fold <= args.end_fold <= 10:
        parser.error("Fold range must satisfy 1 <= start-fold <= end-fold <= 10")
    config_path = resolve_repo_path(args.config)
    config = load_config(config_path)
    manifest_path = resolve_repo_path(config["data"]["manifest_file"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["balancing"] != config["data"]["balancing"]:
        raise ValueError("Manifest balancing policy does not match config")
    if manifest["source_implementation_commit"] != config["protocol_provenance"][
        "qwen_reference_commit"
    ]:
        raise ValueError("Manifest and config disagree on the historical Qwen implementation")
    if manifest["hash_policy"] != "sha256_after_normalizing_text_newlines_to_lf":
        raise ValueError("Unexpected fold-manifest hash policy")
    log_root = resolve_repo_path(config["logging"]["log_root"])
    action = "audit" if args.audit_only else "train"
    log_path = log_root / f"{action}_fold_{args.start_fold:02d}_{args.end_fold:02d}.log"
    logger = setup_logging(log_path)
    selected = list(range(args.start_fold, args.end_fold + 1))
    audits: dict[str, Any] = {}
    for fold in selected:
        audits[fold_name(fold)] = audit_fold(config, manifest, fold)
        logger.info("%s frozen data audit passed: %s", fold_name(fold), audits[fold_name(fold)])
    result_root = resolve_repo_path(config["paths"]["result_root"])
    result_root.mkdir(parents=True, exist_ok=True)
    audit_path = result_root / f"data_audit_fold_{args.start_fold:02d}_{args.end_fold:02d}.json"
    audit_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "folds": audits,
                "balancing": config["data"]["balancing"],
                "completed_at_utc": utc_now(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if args.audit_only:
        logger.info("Audit-only run passed; no model was loaded and no training occurred")
        return 0
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU with bfloat16 support is required")
    require_passed_preflight(config)
    run_manifest = {
        "experiment_id": config["experiment_id"],
        "config_file": relative_repo_path(config_path),
        "config_sha256": raw_sha256(config_path),
        "data_manifest_file": relative_repo_path(manifest_path),
        "data_manifest_sha256": raw_sha256(manifest_path),
        "started_at_utc": utc_now(),
        "fold_range": [args.start_fold, args.end_fold],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "hf_hub_cache": os.environ.get("HF_HUB_CACHE"),
        },
    }
    run_manifest_path = result_root / (
        f"run_manifest_fold_{args.start_fold:02d}_{args.end_fold:02d}.json"
    )
    run_manifest_path.write_text(
        json.dumps(run_manifest, indent=2), encoding="utf-8"
    )
    for fold in selected:
        name = fold_name(fold)
        start = utc_now()
        try:
            best_model_dir, training_report = train_one_fold(
                config, fold, audits[name], logger
            )
            if training_report.get("status") == "skipped_completed":
                continue
            metrics = evaluate_best_model(config, fold, best_model_dir, logger)
            end = utc_now()
            track_fold(
                config,
                fold,
                audits[name],
                "completed",
                start,
                end,
                metrics,
                "Frozen EXP-07 train_undersampled.jsonl; validation/test unchanged; frozen test used only after training.",
            )
        except KeyboardInterrupt:
            end = utc_now()
            logger.warning("%s interrupted by user", name)
            track_fold(
                config,
                fold,
                audits[name],
                "interrupted",
                start,
                end,
                None,
                "INTERRUPTED by user; partial outputs preserved and automatic deletion is disabled.",
            )
            return 130
        except Exception as exc:
            end = utc_now()
            logger.exception("%s failed", name)
            track_fold(
                config,
                fold,
                audits[name],
                "failed",
                start,
                end,
                None,
                f"FAILED: {type(exc).__name__}: {exc}",
            )
            failure_path = result_root / name / f"{name}_failure.json"
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure_path.write_text(
                json.dumps(
                    {
                        "fold": name,
                        "status": "failed",
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                        "failed_at_utc": end,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            return 1
    all_metrics_exist = all(
        (
            result_root
            / fold_name(fold)
            / f"{fold_name(fold)}_metrics.json"
        ).is_file()
        for fold in range(1, 11)
    )
    if all_metrics_exist:
        create_summary(config)
        logger.info("10-fold aggregate summary created")
    logger.info("Requested fold range completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
