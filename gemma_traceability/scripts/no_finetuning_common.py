#!/usr/bin/env python3

import csv
import hashlib
import json
import os
import random
import re
import statistics
import time
from collections import Counter
from pathlib import Path

import yaml


LABELS = ["trace", "no_trace"]
METRIC_KEYS = [
    "accuracy",
    "trace_precision",
    "trace_recall",
    "trace_f1",
    "no_trace_precision",
    "no_trace_recall",
    "no_trace_f1",
]


def project_root():
    return Path(__file__).resolve().parents[2]


def load_config(path):
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(path)
    return config


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else project_root() / path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_line_id"] = line_number
            rows.append(row)
    return rows


def get_label(row):
    label = str(row.get("output", "")).strip()
    if label not in LABELS:
        raise ValueError(f"Invalid label at line {row.get('_line_id')}: {label!r}")
    return label


def get_requirements(row):
    input_object = row.get("input", {})
    high = str(input_object.get("higher_level_requirement", "")).strip()
    low = str(input_object.get("lower_level_requirement", "")).strip()
    if not high or not low:
        raise ValueError(f"Missing requirement text at line {row.get('_line_id')}")
    return high, low


def row_key(row):
    if row.get("example_id"):
        return f"id:{row['example_id']}"
    payload = {"input": row.get("input"), "output": row.get("output")}
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def file_audit(path):
    rows = load_jsonl(path)
    counts = Counter(get_label(row) for row in rows)
    keys = [row_key(row) for row in rows]
    return {
        "path": str(Path(path).relative_to(project_root())).replace("\\", "/"),
        "records": len(rows),
        "trace": counts["trace"],
        "no_trace": counts["no_trace"],
        "unique_examples": len(set(keys)),
        "duplicate_rows": len(keys) - len(set(keys)),
        "sha256": sha256_file(path),
    }


def _compare_tree(copy_dir, reference_dir):
    copy_files = sorted(path.name for path in Path(copy_dir).glob("*.jsonl"))
    reference_files = sorted(path.name for path in Path(reference_dir).glob("*.jsonl"))
    report = {
        "copy_dir": str(Path(copy_dir).relative_to(project_root())).replace("\\", "/"),
        "reference_dir": str(Path(reference_dir).relative_to(project_root())).replace("\\", "/"),
        "same_file_names": copy_files == reference_files,
        "files": {},
    }
    for name in sorted(set(copy_files) | set(reference_files)):
        copy_path = Path(copy_dir) / name
        reference_path = Path(reference_dir) / name
        item = {
            "copy_exists": copy_path.exists(),
            "reference_exists": reference_path.exists(),
        }
        if copy_path.exists():
            item["copy"] = file_audit(copy_path)
        if reference_path.exists():
            item["reference_sha256"] = sha256_file(reference_path)
        item["exact_copy"] = (
            copy_path.exists()
            and reference_path.exists()
            and item["copy"]["sha256"] == item["reference_sha256"]
        )
        report["files"][name] = item
    report["all_exact_copies"] = report["same_file_names"] and all(
        item["exact_copy"] for item in report["files"].values()
    )
    return report


def audit_datasets(config, strict=True):
    data = config["data"]
    simple_dir = resolve_path(data["simple_dir"])
    simple_reference = resolve_path(data["simple_reference_dir"])
    multi_dir = resolve_path(data["multi_run_dir"])
    multi_reference = resolve_path(data["multi_run_reference_dir"])

    report = {
        "simple": _compare_tree(simple_dir, simple_reference),
        "multi_run": _compare_tree(multi_dir, multi_reference),
        "overlap_checks": {},
        "errors": [],
    }

    simple_test = {row_key(row) for row in load_jsonl(simple_dir / "test_100.jsonl")}
    simple_five = {row_key(row) for row in load_jsonl(simple_dir / "five_shot_examples.jsonl")}
    simple_ten = {row_key(row) for row in load_jsonl(simple_dir / "ten_shot_examples.jsonl")}
    report["overlap_checks"]["simple"] = {
        "five_vs_test": len(simple_five & simple_test),
        "ten_vs_test": len(simple_ten & simple_test),
        "five_vs_ten": len(simple_five & simple_ten),
    }

    multi_test = {row_key(row) for row in load_jsonl(multi_dir / "test_100.jsonl")}
    for run_index in data["run_seeds"]:
        label = f"run_{int(run_index):02d}"
        five = {
            row_key(row)
            for row in load_jsonl(multi_dir / f"five_shot_{label}_examples.jsonl")
        }
        ten = {
            row_key(row)
            for row in load_jsonl(multi_dir / f"ten_shot_{label}_examples.jsonl")
        }
        report["overlap_checks"][label] = {
            "five_vs_test": len(five & multi_test),
            "ten_vs_test": len(ten & multi_test),
            "five_vs_ten": len(five & ten),
        }

    if not report["simple"]["all_exact_copies"]:
        report["errors"].append("Simple prompting files are not exact historical copies")
    if not report["multi_run"]["all_exact_copies"]:
        report["errors"].append("Multi-run prompting files are not exact historical copies")
    if any(
        value != 0
        for checks in report["overlap_checks"].values()
        for value in checks.values()
    ):
        report["errors"].append("Prompt/test overlap detected")

    report["status"] = "passed" if not report["errors"] else "failed"
    if strict and report["errors"]:
        raise RuntimeError("; ".join(report["errors"]))
    return report


def build_task_prompt(demo_examples, test_row):
    prompt = (
        "You are performing requirements traceability classification.\n\n"
        "Allowed labels:\n"
        "trace\n"
        "no_trace\n\n"
        "Label meaning:\n"
        "trace = the lower-level requirement semantically supports, refines, implements, or is linked to the higher-level requirement.\n"
        "no_trace = the lower-level requirement has no meaningful traceability relation with the higher-level requirement.\n"
    )
    if demo_examples:
        prompt += "\nExamples:\n"
        for index, row in enumerate(demo_examples, start=1):
            high, low = get_requirements(row)
            prompt += (
                f"\nExample {index}:\n"
                f"Higher-level requirement:\n{high}\n\n"
                f"Lower-level requirement:\n{low}\n\n"
                f"Answer:\n{get_label(row)}\n"
            )
    high, low = get_requirements(test_row)
    prompt += (
        "\nNow classify the following requirement pair.\n\n"
        f"Higher-level requirement:\n{high}\n\n"
        f"Lower-level requirement:\n{low}\n\n"
        "Answer only one label: trace or no_trace.\n"
        "Answer:\n"
    )
    return prompt


def normalize_prediction(raw_text):
    text = raw_text.strip().lower().replace("\n", " ").replace("\t", " ")
    text = re.sub(r"[*`\"']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(answer|label|prediction)\s*:\s*", "", text).strip()
    if text in {"trace", "trace."}:
        return "trace"
    if text in {"no_trace", "no_trace.", "no trace", "no trace."}:
        return "no_trace"
    tokens = [token for token in re.split(r"[\s,.;:!?]+", text) if token]
    if tokens and tokens[0] == "trace":
        return "trace"
    if tokens and tokens[0] == "no_trace":
        return "no_trace"
    if len(tokens) >= 2 and tokens[0] == "no" and tokens[1] == "trace":
        return "no_trace"
    return "invalid"


def set_reproducibility(seed):
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gpu_memory_mib():
    import torch

    if not torch.cuda.is_available():
        return {"cuda_available": False}
    free, total = torch.cuda.mem_get_info()
    divisor = 1024**2
    return {
        "cuda_available": True,
        "device": torch.cuda.get_device_name(0),
        "free_mib": round(free / divisor, 2),
        "total_mib": round(total / divisor, 2),
        "global_used_mib": round((total - free) / divisor, 2),
        "process_allocated_mib": round(torch.cuda.memory_allocated() / divisor, 2),
        "process_reserved_mib": round(torch.cuda.memory_reserved() / divisor, 2),
        "process_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / divisor, 2),
        "process_peak_reserved_mib": round(torch.cuda.max_memory_reserved() / divisor, 2),
    }


def load_model(config):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_config = config["model"]
    model_id = model_config["model_id"]
    revision = model_config["revision"]
    cache_dir = os.environ.get("HF_HUB_CACHE", model_config["cache_dir"])

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=bool(model_config["local_files_only"]),
        trust_remote_code=False,
    )
    if not tokenizer.chat_template:
        raise RuntimeError("Gemma tokenizer has no chat template")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quantization = BitsAndBytesConfig(
        load_in_4bit=bool(model_config["load_in_4bit"]),
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=bool(model_config["double_quant"]),
        bnb_4bit_quant_type=model_config["quant_type"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=bool(model_config["local_files_only"]),
        trust_remote_code=False,
        quantization_config=quantization,
        dtype=torch.bfloat16,
        device_map=model_config["device_map"],
        low_cpu_mem_usage=True,
    )
    trainable_before_freeze = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.requires_grad_(False)
    model.eval()
    trainable_after_freeze = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    details = {
        "model_id": model_id,
        "revision": revision,
        "model_class": model.__class__.__name__,
        "model_type": getattr(model.config, "model_type", None),
        "architectures": getattr(model.config, "architectures", None),
        "max_position_embeddings": getattr(model.config, "max_position_embeddings", None),
        "loaded_parameter_elements": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters_before_explicit_freeze": trainable_before_freeze,
        "trainable_parameters": trainable_after_freeze,
        "all_parameters_explicitly_frozen": trainable_after_freeze == 0,
        "quantized_4bit": bool(getattr(model, "is_loaded_in_4bit", False)),
        "tokenizer_class": tokenizer.__class__.__name__,
        "chat_template_used": True,
        "cache_dir": cache_dir,
    }
    return tokenizer, model, details


def render_prompt(tokenizer, task_prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": task_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_prediction(task_prompt, tokenizer, model, config):
    import torch

    inference = config["inference"]
    rendered = render_prompt(tokenizer, task_prompt)
    raw_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    original_prompt_tokens = len(raw_ids)
    max_new_tokens = int(inference["max_new_tokens"])
    configured_limit = int(inference["max_prompt_tokens"])
    model_limit = getattr(model.config, "max_position_embeddings", None)
    effective_limit = configured_limit
    if model_limit:
        effective_limit = min(effective_limit, max(1, int(model_limit) - max_new_tokens))
    truncated = original_prompt_tokens > effective_limit

    inputs = tokenizer(
        rendered,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=effective_limit,
    )
    device = next(model.parameters()).device
    inputs = {name: value.to(device) for name, value in inputs.items()}
    started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - started
    new_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    raw = tokenizer.decode(new_ids, skip_special_tokens=True)
    return {
        "raw_prediction": raw.strip(),
        "pred_label": normalize_prediction(raw),
        "prompt_tokens": original_prompt_tokens,
        "effective_prompt_limit": effective_limit,
        "prompt_truncated": truncated,
        "generated_tokens": int(new_ids.shape[0]),
        "generation_seconds": elapsed,
    }


def compute_metrics(prediction_rows, method, run, seed):
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    valid_rows = [row for row in prediction_rows if row["pred_label"] in LABELS]
    invalid_count = len(prediction_rows) - len(valid_rows)
    if not valid_rows:
        raise RuntimeError(f"No valid predictions for {method} run={run}")
    truth = [row["gold_label"] for row in valid_rows]
    predicted = [row["pred_label"] for row in valid_rows]
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=LABELS, zero_division=0
    )
    matrix = confusion_matrix(truth, predicted, labels=LABELS)
    return {
        "method": method,
        "run": int(run),
        "seed": int(seed),
        "accuracy": float(accuracy_score(truth, predicted)),
        "trace_precision": float(precision[0]),
        "trace_recall": float(recall[0]),
        "trace_f1": float(f1[0]),
        "trace_support": int(support[0]),
        "no_trace_precision": float(precision[1]),
        "no_trace_recall": float(recall[1]),
        "no_trace_f1": float(f1[1]),
        "no_trace_support": int(support[1]),
        "valid_predictions": len(valid_rows),
        "invalid_predictions": invalid_count,
        "total_records": len(prediction_rows),
        "confusion_matrix": matrix.tolist(),
    }


def aggregate_metrics(rows, method):
    selected = [row for row in rows if row["method"] == method]
    if not selected:
        raise ValueError(f"No metric rows for {method}")
    output = {"method": method, "runs": len(selected)}
    for key in METRIC_KEYS:
        values = [float(row[key]) for row in selected]
        output[f"{key}_mean"] = statistics.mean(values)
        output[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    output["total_valid_predictions"] = sum(row["valid_predictions"] for row in selected)
    output["total_invalid_predictions"] = sum(row["invalid_predictions"] for row in selected)
    output["total_records"] = sum(row["total_records"] for row in selected)
    output["confusion_matrix_sum"] = [
        [sum(row["confusion_matrix"][i][j] for row in selected) for j in range(2)]
        for i in range(2)
    ]
    return output


def save_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def save_csv(rows, path, fieldnames=None):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows supplied for {path}")
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
