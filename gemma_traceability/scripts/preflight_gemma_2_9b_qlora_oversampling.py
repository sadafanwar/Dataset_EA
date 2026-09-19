#!/usr/bin/env python3
"""Fold 01 preflight for Gemma QLoRA with frozen EXP-06 oversampling."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import random
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import accelerate
import bitsandbytes
import datasets
import numpy as np
import peft
import torch
import transformers
import yaml
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    Gemma2ForCausalLM,
    get_linear_schedule_with_warmup,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / (
    "gemma_traceability/configs/config_gemma_2_9b_qlora_oversampling_fold_01.yaml"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_jsonl_sha256(path: Path) -> str:
    """Hash JSONL with LF newlines so Windows and Linux checkouts agree."""
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


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


def build_training_text(record: dict[str, Any]) -> str:
    instruction = record.get(
        "instruction",
        "Determine whether the lower-level requirement traces to the higher-level requirement.",
    )
    input_obj = record.get("input", {})
    higher_req = input_obj.get("higher_level_requirement", "")
    lower_req = input_obj.get("lower_level_requirement", "")
    prompt = (
        f"{instruction}\n\n"
        f"Higher-level requirement:\n{higher_req}\n\n"
        f"Lower-level requirement:\n{lower_req}\n\n"
        f"Answer:"
    )
    return f"{prompt} {record['output']}"


def dataset_audit(name: str, path: Path, expected: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name} dataset: {path}")
    records = load_jsonl(path)
    labels = Counter(record.get("output") for record in records)
    example_ids = [str(record.get("example_id", "")) for record in records]
    if any(not example_id for example_id in example_ids):
        raise ValueError(f"Every {name} record must have a non-empty example_id")
    observed = {
        "path": str(path.relative_to(REPO_ROOT)),
        "records": len(records),
        "trace": labels.get("trace", 0),
        "no_trace": labels.get("no_trace", 0),
        "other_labels": {str(k): v for k, v in labels.items() if k not in {"trace", "no_trace"}},
        "unique_example_ids": len(set(example_ids)),
        "duplicate_rows": len(example_ids) - len(set(example_ids)),
        "sha256": canonical_jsonl_sha256(path),
        "raw_sha256": sha256_file(path),
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
                f"{name} provenance mismatch for {key}: expected {expected[key]!r}, "
                f"observed {observed[key]!r}"
            )
    if observed["other_labels"]:
        raise ValueError(f"Unexpected labels in {name}: {observed['other_labels']}")
    return records, observed


def reproduce_historical_oversampling(
    source_records: list[dict[str, Any]], seed: int
) -> list[dict[str, Any]]:
    """Reproduce scripts/create_oversampled_inner_train_fold.py at c3ae63b."""
    random.seed(seed)
    trace_records = [record for record in source_records if record["output"] == "trace"]
    no_trace_records = [
        record for record in source_records if record["output"] == "no_trace"
    ]
    target_count = max(len(trace_records), len(no_trace_records))
    trace_balanced = (
        random.choices(trace_records, k=target_count)
        if len(trace_records) < target_count
        else trace_records
    )
    no_trace_balanced = (
        random.choices(no_trace_records, k=target_count)
        if len(no_trace_records) < target_count
        else no_trace_records
    )
    balanced = trace_balanced + no_trace_balanced
    random.shuffle(balanced)
    return balanced


def gpu_memory() -> dict[str, Any]:
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


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("gemma_2_9b_qlora_oversampling_preflight")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    config_path = resolve_repo_path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    log_path = resolve_repo_path(config["logging"]["log_file"])
    report_path = resolve_repo_path(config["logging"]["report_file"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_path)

    report: dict[str, Any] = {
        "experiment_id": config["experiment_id"],
        "fold": config["fold"],
        "status": "running",
        "started_at_utc": utc_now(),
        "config_file": str(config_path.relative_to(REPO_ROOT)),
        "errors": [],
    }

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("GPU does not support bfloat16")

        training = config["training"]
        quantization = config["quantization"]
        lora = config["lora"]
        data = config["data"]
        expected = data["expected"]
        model_config = config["model"]

        if data["balancing"] != "random_oversampling_with_replacement":
            raise ValueError("The oversampling experiment must use random replacement oversampling")
        if Path(data["train_file"]).name != "train_oversampled.jsonl":
            raise ValueError("The oversampling train_file must be frozen train_oversampled.jsonl")
        if Path(data["source_inner_train_file"]).name != "inner_train.jsonl":
            raise ValueError("The oversampling source must be frozen inner_train.jsonl")
        if "undersampled" in Path(data["train_file"]).name:
            raise ValueError("Undersampling data is forbidden in this experiment")

        source_records, source_audit = dataset_audit(
            "source_inner_train",
            resolve_repo_path(data["source_inner_train_file"]),
            expected["source_inner_train"],
        )
        train_records, train_audit = dataset_audit(
            "train", resolve_repo_path(data["train_file"]), expected["train"]
        )
        _, validation_audit = dataset_audit(
            "validation", resolve_repo_path(data["validation_file"]), expected["validation"]
        )
        _, test_audit = dataset_audit(
            "test", resolve_repo_path(data["test_file"]), expected["test"]
        )
        _, qwen_test_audit = dataset_audit(
            "qwen_reference_test",
            resolve_repo_path(data["qwen_reference_test_file"]),
            expected["test"],
        )
        if test_audit["sha256"] != qwen_test_audit["sha256"]:
            raise ValueError("Gemma and Qwen Fold 01 test files are not byte-identical")
        reproduced_train = reproduce_historical_oversampling(
            source_records, data["oversampling"]["fold_seed"]
        )
        if reproduced_train != train_records:
            raise ValueError(
                "Frozen Fold 01 train file does not exactly reproduce historical EXP-06 oversampling"
            )
        train_ids = {record["example_id"] for record in train_records}
        validation_ids = {
            record["example_id"]
            for record in load_jsonl(resolve_repo_path(data["validation_file"]))
        }
        test_ids = {
            record["example_id"]
            for record in load_jsonl(resolve_repo_path(data["test_file"]))
        }
        if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
            raise ValueError("Cross-split example_id leakage detected")

        report["datasets"] = {
            "source_inner_train": source_audit,
            "train": train_audit,
            "validation": validation_audit,
            "test": test_audit,
            "qwen_reference_test": qwen_test_audit,
            "gemma_test_matches_qwen_test": True,
            "exact_historical_oversampling": True,
            "no_cross_split_overlap": True,
        }
        report["hyperparameters"] = {
            "learning_rate": training["learning_rate"],
            "batch_size": training["batch_size"],
            "gradient_accumulation_steps": training["gradient_accumulation_steps"],
            "effective_batch_size": training["batch_size"] * training["gradient_accumulation_steps"],
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
            "balancing": data["balancing"],
            "oversampling_fold_seed": data["oversampling"]["fold_seed"],
            "eval_steps": config["evaluation"]["eval_steps"],
            "eval_samples": config["evaluation"]["eval_samples"],
            "save_steps": config["checkpointing"]["save_steps"],
            "save_total_limit": config["checkpointing"]["save_total_limit"],
            "logging_steps": config["logging"]["logging_steps"],
            "lora_r": lora["r"],
            "lora_alpha": lora["alpha"],
            "lora_dropout": lora["dropout"],
            "lora_target_modules_regex": lora["target_modules_regex"],
            "quantization": {
                "load_in_4bit": quantization["load_in_4bit"],
                "quant_type": quantization["quant_type"],
                "compute_dtype": quantization["compute_dtype"],
                "double_quant": quantization["double_quant"],
            },
            "prompt_protocol": config["prompt"]["protocol"],
            "loss_scope": config["prompt"]["loss_scope"],
        }

        set_seed(training["seed"])
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        report["environment"] = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "bitsandbytes": bitsandbytes.__version__,
            "datasets": datasets.__version__,
            "accelerate": accelerate.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "hf_hub_cache": os.environ.get("HF_HUB_CACHE"),
        }
        report["gpu_memory_mib"] = {"before_model_load": gpu_memory()}

        logger.info("Dataset provenance checks passed")
        logger.info("Train counts: %s", train_audit)
        logger.info("Validation counts: %s", validation_audit)
        logger.info("Test counts: %s", test_audit)
        logger.info("Hyperparameters: %s", report["hyperparameters"])
        logger.info("GPU before load: %s", report["gpu_memory_mib"]["before_model_load"])

        tokenizer = AutoTokenizer.from_pretrained(
            model_config["id"],
            revision=model_config["revision"],
            trust_remote_code=True,
            local_files_only=model_config["local_files_only"],
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quantization["load_in_4bit"],
            bnb_4bit_quant_type=quantization["quant_type"],
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=quantization["double_quant"],
        )
        model = Gemma2ForCausalLM.from_pretrained(
            model_config["id"],
            revision=model_config["revision"],
            quantization_config=bnb_config,
            device_map={"": 0},
            dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=model_config["local_files_only"],
            low_cpu_mem_usage=True,
        )
        model.config.use_cache = False
        report["gpu_memory_mib"]["after_quantized_model_load"] = gpu_memory()

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training["gradient_checkpointing"],
            gradient_checkpointing_kwargs={
                "use_reentrant": training["gradient_checkpointing_use_reentrant"]
            },
        )
        lora_config = LoraConfig(
            r=lora["r"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            bias=lora["bias"],
            task_type=TaskType.CAUSAL_LM,
            target_modules=lora["target_modules_regex"],
        )
        model = get_peft_model(model, lora_config)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        adapter_modules = [
            name for name, module in model.named_modules() if hasattr(module, "lora_A")
        ]
        trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        all_parameters = sum(parameter.numel() for parameter in model.parameters())
        invalid_adapter_modules = [
            name
            for name in adapter_modules
            if ".model.layers." not in name or ".self_attn." not in name
        ]
        if invalid_adapter_modules:
            raise ValueError(f"LoRA attached outside the text decoder: {invalid_adapter_modules}")
        if len(adapter_modules) != lora["expected_adapter_modules"]:
            raise ValueError(
                f"Expected {lora['expected_adapter_modules']} LoRA modules, observed {len(adapter_modules)}"
            )
        if trainable_parameters != lora["expected_trainable_parameters"]:
            raise ValueError(
                f"Expected {lora['expected_trainable_parameters']} trainable parameters, "
                f"observed {trainable_parameters}"
            )

        report["lora"] = {
            "adapter_module_count": len(adapter_modules),
            "adapter_modules": adapter_modules,
            "trainable_parameters": trainable_parameters,
            "all_parameters": all_parameters,
            "trainable_percent": 100.0 * trainable_parameters / all_parameters,
            "vision_or_projector_adapter_modules": invalid_adapter_modules,
        }
        report["gpu_memory_mib"]["after_lora_attach"] = gpu_memory()
        logger.info("LoRA module count: %d", len(adapter_modules))
        logger.info(
            "Trainable parameters: %d / %d (%.6f%%)",
            trainable_parameters,
            all_parameters,
            report["lora"]["trainable_percent"],
        )

        longest_index = -1
        longest_untruncated_tokens = -1
        longest_text = ""
        for index, record in enumerate(train_records):
            text = build_training_text(record)
            token_count = len(tokenizer(text, truncation=False, padding=False)["input_ids"])
            if token_count > longest_untruncated_tokens:
                longest_index = index
                longest_untruncated_tokens = token_count
                longest_text = text

        encoded = tokenizer(
            longest_text,
            return_tensors="pt",
            truncation=True,
            max_length=training["max_length"],
            padding=False,
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        labels = encoded["input_ids"].clone()
        report["preflight_batch"] = {
            "record_index": longest_index,
            "example_id": train_records[longest_index].get("example_id", ""),
            "untruncated_tokens": longest_untruncated_tokens,
            "training_tokens_after_truncation": int(encoded["input_ids"].shape[1]),
            "microbatch_size": int(encoded["input_ids"].shape[0]),
        }

        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=training["learning_rate"],
            weight_decay=training["weight_decay"],
        )
        total_optimizer_steps = (
            len(train_records) * training["num_epochs"]
        ) // training["gradient_accumulation_steps"]
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training["warmup_steps"],
            num_training_steps=total_optimizer_steps,
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**encoded, labels=labels)
        loss = outputs.loss / training["gradient_accumulation_steps"]
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), training["max_grad_norm"]
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        report["optimizer_step"] = {
            "completed": True,
            "raw_loss": float(outputs.loss.detach().cpu()),
            "scaled_loss": float(loss.detach().cpu()),
            "gradient_norm_before_clipping": float(grad_norm.detach().cpu()),
            "scheduler_total_steps": total_optimizer_steps,
            "learning_rate_after_step": scheduler.get_last_lr()[0],
            "adapter_saved": False,
        }
        report["gpu_memory_mib"]["after_optimizer_step"] = gpu_memory()
        report["status"] = "passed"
        logger.info("One longest-sequence optimizer step completed")
        logger.info("Optimizer result: %s", report["optimizer_step"])
        logger.info("GPU after step: %s", report["gpu_memory_mib"]["after_optimizer_step"])
    except Exception as exc:
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        report["status"] = "failed"
        report["errors"].append(error)
        logger.exception("Gemma QLoRA oversampling preflight failed")
    finally:
        report["finished_at_utc"] = utc_now()
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        logger.info("Preflight report: %s", report_path)
        logger.info("Preflight status: %s", report["status"])

    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
