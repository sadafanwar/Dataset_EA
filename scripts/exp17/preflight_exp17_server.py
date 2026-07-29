#!/usr/bin/env python3
"""
EXP-17 server preflight.

Default mode:
- checks repository structure, folds, configuration, and token audit;
- does not require a GPU;
- safe to run on the local preparation machine.

--server mode:
- additionally validates Python, CUDA, GPU, QLoRA packages,
  Hugging Face access, and server disk space;
- does not load model weights;
- does not start training.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]

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

TOKEN_AUDIT_PATH = (
    REPO_ROOT
    / "results"
    / "exp17_qwen35_9b_rationale_single_template"
    / "dataset_token_audit.json"
)

REQUIRED_PACKAGES = (
    "torch",
    "transformers",
    "datasets",
    "peft",
    "accelerate",
    "bitsandbytes",
    "tqdm",
    "PyYAML",
    "prefect",
    "mlflow",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate EXP-17 before server training."
    )

    parser.add_argument(
        "--server",
        action="store_true",
        help=(
            "Enable strict server checks including CUDA, GPU, "
            "bitsandbytes, and Python compatibility."
        ),
    )

    return parser.parse_args()


def count_jsonl_records(path: Path) -> int:
    count = 0

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line "
                    f"{line_number}: {error}"
                ) from error

            if not isinstance(item, dict):
                raise ValueError(
                    f"Expected JSON object in {path} "
                    f"at line {line_number}"
                )

            count += 1

    return count


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = yaml.safe_load(handle)

    if not isinstance(value, dict):
        raise ValueError(
            f"YAML root must be an object: {path}"
        )

    return value


def package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def print_check(
    status: str,
    name: str,
    detail: str,
) -> None:
    print(f"[{status:<4}] {name:<28} {detail}")


def verify_structure(
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    print()
    print("Repository and experiment structure")
    print("-" * 70)

    if CONFIG_PATH.is_file():
        print_check("PASS", "Configuration", str(CONFIG_PATH))
    else:
        print_check("FAIL", "Configuration", "file missing")
        errors.append(
            f"Configuration file missing: {CONFIG_PATH}"
        )
        return {}

    config = load_yaml(CONFIG_PATH)

    model_path = config.get("model_path")
    method = config.get("method", "qlora")
    max_length = (
        config.get("data", {}).get("max_length")
    )
    quantization_enabled = (
        config.get("quantization", {}).get(
            "enabled",
            False,
        )
    )

    print_check(
        "PASS" if model_path else "FAIL",
        "Model ID",
        str(model_path),
    )

    if not model_path:
        errors.append("model_path is not configured")

    print_check(
        "PASS" if method == "qlora" else "WARN",
        "Fine-tuning method",
        str(method),
    )

    if method != "qlora":
        warnings.append(
            f"Expected method=qlora but found {method}"
        )

    print_check(
        "PASS" if quantization_enabled else "FAIL",
        "4-bit quantization",
        f"enabled={quantization_enabled}",
    )

    if not quantization_enabled:
        errors.append(
            "QLoRA requires quantization.enabled=true"
        )

    print_check(
        "PASS",
        "Maximum token length",
        str(max_length),
    )

    print()
    print("Fold files")
    print("-" * 70)

    fold_summary: dict[str, dict[str, int]] = {}

    for fold_number in range(1, 11):
        fold_name = f"fold_{fold_number:02d}"
        fold_path = FOLDS_ROOT / fold_name
        fold_summary[fold_name] = {}

        fold_ok = True

        for split_name in (
            "train",
            "validation",
            "test",
        ):
            split_path = (
                fold_path / f"{split_name}.jsonl"
            )

            if not split_path.is_file():
                fold_ok = False
                errors.append(
                    f"Missing fold file: {split_path}"
                )
                continue

            count = count_jsonl_records(split_path)
            fold_summary[fold_name][split_name] = count

        if fold_ok:
            detail = (
                f"train={fold_summary[fold_name]['train']}, "
                f"validation="
                f"{fold_summary[fold_name]['validation']}, "
                f"test={fold_summary[fold_name]['test']}"
            )

            print_check("PASS", fold_name, detail)
        else:
            print_check(
                "FAIL",
                fold_name,
                "one or more split files missing",
            )

    print()
    print("Token audit")
    print("-" * 70)

    if not TOKEN_AUDIT_PATH.is_file():
        print_check(
            "FAIL",
            "Token audit report",
            "file missing",
        )
        errors.append(
            f"Token audit missing: {TOKEN_AUDIT_PATH}"
        )
    else:
        with TOKEN_AUDIT_PATH.open(
            "r",
            encoding="utf-8-sig",
        ) as handle:
            audit = json.load(handle)

        truncation = audit.get("truncation", {})

        affected = int(
            truncation.get(
                "records_exceeding_max_length",
                -1,
            )
        )

        targets_truncated = int(
            truncation.get(
                "targets_partially_truncated",
                -1,
            )
        )

        targets_removed = int(
            truncation.get(
                "targets_fully_removed",
                -1,
            )
        )

        audit_ok = (
            affected == 0
            and targets_truncated == 0
            and targets_removed == 0
        )

        print_check(
            "PASS" if audit_ok else "FAIL",
            "Token audit report",
            (
                f"affected={affected}, "
                f"targets_truncated={targets_truncated}, "
                f"targets_removed={targets_removed}"
            ),
        )

        if not audit_ok:
            errors.append(
                "Token audit reports truncated records or targets"
            )

    return {
        "config": config,
        "fold_summary": fold_summary,
    }


def verify_server_environment(
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    print()
    print("Server runtime environment")
    print("-" * 70)

    python_version = (
        f"{sys.version_info.major}."
        f"{sys.version_info.minor}."
        f"{sys.version_info.micro}"
    )

    supported_python = (
        sys.version_info.major == 3
        and 10 <= sys.version_info.minor <= 12
    )

    print_check(
        "PASS" if supported_python else "FAIL",
        "Python",
        f"{python_version} ({sys.executable})",
    )

    if not supported_python:
        errors.append(
            "Use Python 3.10, 3.11, or 3.12 "
            "for the server environment"
        )

    operating_system = platform.system()

    print_check(
        "PASS" if operating_system == "Linux" else "WARN",
        "Operating system",
        f"{operating_system} {platform.release()}",
    )

    if operating_system != "Linux":
        warnings.append(
            "Linux is recommended for bitsandbytes QLoRA training"
        )

    print()
    print("Required Python packages")
    print("-" * 70)

    for package_name in REQUIRED_PACKAGES:
        version = package_version(package_name)

        if version == "not installed":
            print_check(
                "FAIL",
                package_name,
                version,
            )
            errors.append(
                f"Required package missing: {package_name}"
            )
        else:
            print_check(
                "PASS",
                package_name,
                version,
            )

    print()
    print("Import validation")
    print("-" * 70)

    import_names = {
        "torch": "torch",
        "transformers": "transformers",
        "datasets": "datasets",
        "peft": "peft",
        "accelerate": "accelerate",
        "bitsandbytes": "bitsandbytes",
    }

    imported: dict[str, Any] = {}

    for display_name, import_name in import_names.items():
        try:
            imported[display_name] = (
                importlib.import_module(import_name)
            )

            print_check(
                "PASS",
                f"import {display_name}",
                "successful",
            )
        except Exception as error:
            print_check(
                "FAIL",
                f"import {display_name}",
                str(error),
            )

            errors.append(
                f"Cannot import {display_name}: {error}"
            )

    torch = imported.get("torch")

    if torch is None:
        return {}

    print()
    print("CUDA and GPU")
    print("-" * 70)

    cuda_available = bool(
        torch.cuda.is_available()
    )

    print_check(
        "PASS" if cuda_available else "FAIL",
        "CUDA available",
        str(cuda_available),
    )

    if not cuda_available:
        errors.append(
            "CUDA GPU is not available to PyTorch"
        )
        return {}

    device_count = torch.cuda.device_count()

    print_check(
        "PASS",
        "CUDA device count",
        str(device_count),
    )

    gpu_summary = []

    for device_index in range(device_count):
        properties = torch.cuda.get_device_properties(
            device_index
        )

        memory_gb = (
            properties.total_memory
            / 1024**3
        )

        capability = torch.cuda.get_device_capability(
            device_index
        )

        gpu_detail = (
            f"{properties.name}; "
            f"{memory_gb:.2f} GiB; "
            f"compute capability "
            f"{capability[0]}.{capability[1]}"
        )

        print_check(
            "PASS",
            f"GPU {device_index}",
            gpu_detail,
        )

        gpu_summary.append(
            {
                "index": device_index,
                "name": properties.name,
                "memory_gib": round(
                    memory_gb,
                    2,
                ),
                "compute_capability": (
                    f"{capability[0]}."
                    f"{capability[1]}"
                ),
            }
        )

    cuda_version = torch.version.cuda

    print_check(
        "PASS" if cuda_version else "FAIL",
        "PyTorch CUDA build",
        str(cuda_version),
    )

    if not cuda_version:
        errors.append(
            "Installed PyTorch is not a CUDA build"
        )

    bf16_supported = bool(
        torch.cuda.is_bf16_supported()
    )

    print_check(
        "PASS" if bf16_supported else "WARN",
        "BF16 support",
        str(bf16_supported),
    )

    if not bf16_supported:
        warnings.append(
            "GPU does not report BF16 support; "
            "training precision may need adjustment"
        )

    disk_usage = shutil.disk_usage(REPO_ROOT)

    free_disk_gib = (
        disk_usage.free / 1024**3
    )

    print_check(
        "PASS",
        "Free repository disk",
        f"{free_disk_gib:.2f} GiB",
    )

    hf_token_present = bool(
        os.environ.get("HF_TOKEN")
        or os.environ.get(
            "HUGGING_FACE_HUB_TOKEN"
        )
    )

    print_check(
        "PASS" if hf_token_present else "WARN",
        "Hugging Face token",
        (
            "configured"
            if hf_token_present
            else "not configured"
        ),
    )

    if not hf_token_present:
        warnings.append(
            "HF_TOKEN is not configured; "
            "Hub downloads may be rate-limited"
        )

    return {
        "python": python_version,
        "operating_system": operating_system,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "bf16_supported": bf16_supported,
        "gpus": gpu_summary,
        "free_disk_gib": round(
            free_disk_gib,
            2,
        ),
        "hf_token_present": hf_token_present,
    }


def main() -> int:
    args = parse_args()

    print()
    print("=" * 70)
    print("EXP-17 CONTROLLED PREFLIGHT")
    print("=" * 70)

    print(
        "Mode: "
        + (
            "strict server"
            if args.server
            else "local structure-only"
        )
    )

    print("Model weights will not be loaded.")
    print("Training will not be started.")

    errors: list[str] = []
    warnings: list[str] = []

    structure = verify_structure(
        errors,
        warnings,
    )

    server_environment: dict[str, Any] = {}

    if args.server:
        server_environment = (
            verify_server_environment(
                errors,
                warnings,
            )
        )

    print()
    print("=" * 70)
    print("PREFLIGHT SUMMARY")
    print("=" * 70)

    print(f"Errors   : {len(errors)}")
    print(f"Warnings : {len(warnings)}")

    if warnings:
        print()
        print("Warnings:")

        for warning in warnings:
            print(f"- {warning}")

    if errors:
        print()
        print("Errors:")

        for error in errors:
            print(f"- {error}")

        print()
        print("EXP-17 PREFLIGHT FAILED")
        print("Training must not be started.")

        return 1

    print()
    print("EXP-17 PREFLIGHT PASSED")

    if args.server:
        print(
            "Server environment is ready for "
            "a controlled one-fold training test."
        )
    else:
        print(
            "Local repository structure is ready."
        )
        print(
            "Run this script with --server "
            "on the training server."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
