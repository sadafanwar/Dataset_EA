#!/usr/bin/env python3

import argparse
import platform
import traceback
from datetime import datetime, timezone

from no_finetuning_common import (
    audit_datasets,
    build_task_prompt,
    generate_prediction,
    gpu_memory_mib,
    load_config,
    load_jsonl,
    load_model,
    resolve_path,
    save_json,
    set_reproducibility,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preflight Google Gemma 2 9B Instruct no-fine-tuning prompting."
    )
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    results_dir = resolve_path(config["outputs"]["results_dir"])
    report_path = results_dir / "preflight" / "preflight_report.json"
    report = {
        "experiment": config["experiment"],
        "model": config["model"],
        "inference": config["inference"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
        "errors": [],
        "warnings": [],
    }

    try:
        import accelerate
        import bitsandbytes
        import datasets
        import peft
        import torch
        import transformers

        report["environment"] = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "accelerate": accelerate.__version__,
            "bitsandbytes": bitsandbytes.__version__,
            "datasets": datasets.__version__,
            "peft": peft.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        }
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required")

        print("Auditing frozen prompting data...", flush=True)
        report["data_audit"] = audit_datasets(config, strict=False)
        if report["data_audit"]["status"] != "passed":
            raise RuntimeError("Frozen data audit failed")

        set_reproducibility(int(config["inference"]["seed"]))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        report["gpu_before_load_mib"] = gpu_memory_mib()

        print("Loading Gemma tokenizer and 4-bit base model...", flush=True)
        tokenizer, model, model_details = load_model(config)
        report["model_loaded"] = model_details
        report["gpu_after_load_mib"] = gpu_memory_mib()

        simple_dir = resolve_path(config["data"]["simple_dir"])
        test_row = load_jsonl(simple_dir / "test_100.jsonl")[0]
        settings = [
            ("zero_shot", []),
            ("five_shot", load_jsonl(simple_dir / "five_shot_examples.jsonl")),
            ("ten_shot", load_jsonl(simple_dir / "ten_shot_examples.jsonl")),
        ]
        generations = []
        for method, examples in settings:
            print(f"Running one-example {method} preflight...", flush=True)
            prediction = generate_prediction(
                build_task_prompt(examples, test_row), tokenizer, model, config
            )
            prediction["method"] = method
            prediction["demo_examples"] = len(examples)
            prediction["gold_label"] = str(test_row["output"])
            generations.append(prediction)
            print(
                f"{method}: gold={prediction['gold_label']} "
                f"raw={prediction['raw_prediction']!r} parsed={prediction['pred_label']}",
                flush=True,
            )
            if prediction["pred_label"] == "invalid":
                report["warnings"].append(f"{method} preflight prediction was invalid")

        report["generations"] = generations
        report["gpu_after_generation_mib"] = gpu_memory_mib()
        report["status"] = "passed"
    except Exception as exc:
        report["errors"].append({
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        })
        print(report["errors"][-1]["traceback"], flush=True)
    finally:
        report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(report, report_path)
        print(f"Preflight report: {report_path}", flush=True)
        print(f"Preflight status: {report['status']}", flush=True)

    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
